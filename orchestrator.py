"""Claude-powered analysis orchestrator for BrainNet."""

from __future__ import annotations

import json
import os
from typing import Any, Generator

from brainnet.session_store import SessionStore
from brainnet.tools import TOOL_DEFINITIONS, execute_tool

SYSTEM_PROMPT = """You are BrainNet's analysis orchestrator, a scientific fMRI analysis assistant.

Given a user's analysis goal:
1. Load and preprocess the requested data
2. Propose 2-3 distinct analysis strategies that address the goal
3. Execute each strategy using your tools
4. Compare results using the compare_strategies tool
5. Summarize findings and recommend the best approach

Rules:
- Always preprocess data before running analysis
- Maximum 3 strategies per comparison round (user can request more)
- Report data quality issues honestly — do not proceed if QC fails
- Frame all findings as candidate observations, not confirmed discoveries
- Acknowledge limitations in the comparison
- If a tool fails, explain why and try alternative parameters
- If a required dependency is missing, tell the user what to install
"""

TOKEN_BUDGET = 50_000
TOKEN_WARNING_THRESHOLD = 0.8


def sse_format(event: str, data: dict[str, Any]) -> str:
    """Format an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _get_client():
    """Lazily import and create the Anthropic client."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "The anthropic package is required for the chat feature. "
            "Install it with: pip install -e .[llm]"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is required for the chat feature."
        )
    return anthropic.Anthropic(api_key=api_key)


def _create_stream(client, messages, system, tools, max_tokens):
    """Create a streaming message — extracted for test mocking."""
    return client.messages.stream(
        model="claude-sonnet-4-20250514",
        system=system,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
    )


def _summarize_old_tool_results(session: SessionStore) -> None:
    """Trim older tool_result content when approaching token budget."""
    if session.estimate_tokens() < int(TOKEN_BUDGET * TOKEN_WARNING_THRESHOLD):
        return
    for msg in session.messages[:-6]:  # keep last 6 messages intact
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, str) and len(content) > 200:
                        block["content"] = content[:200] + "... [truncated]"


def run_orchestrator(
    session: SessionStore, user_message: str
) -> Generator[str, None, None]:
    """Synchronous generator that streams SSE events for one user turn."""
    session.messages.append({"role": "user", "content": user_message})
    _summarize_old_tool_results(session)

    token_estimate = session.estimate_tokens()
    if token_estimate > int(TOKEN_BUDGET * TOKEN_WARNING_THRESHOLD):
        yield sse_format("text_delta", {
            "text": f"\n\n\u26a0\ufe0f Session using ~{token_estimate} tokens "
            f"(budget: {TOKEN_BUDGET}). Consider starting a new session.\n\n"
        })

    client = _get_client()

    while True:
        with _create_stream(
            client, session.messages, SYSTEM_PROMPT, TOOL_DEFINITIONS, 4096
        ) as stream:
            for event in stream:
                if hasattr(event, "delta"):
                    if getattr(event.delta, "type", None) == "text_delta":
                        yield sse_format("text_delta", {"text": event.delta.text})

            final = stream.get_final_message()

        # Record assistant message
        session.messages.append({
            "role": "assistant",
            "content": final.content,
        })

        # Process any tool calls
        tool_uses = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            yield sse_format("done", {})
            break

        tool_results = []
        for tool_block in tool_uses:
            yield sse_format("tool_start", {
                "tool": tool_block.name,
                "params": tool_block.input,
            })
            result = execute_tool(session, tool_block.name, tool_block.input)
            yield sse_format("tool_result", {
                "tool": tool_block.name,
                "summary": result,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

        session.messages.append({"role": "user", "content": tool_results})

        if final.stop_reason == "end_turn":
            yield sse_format("done", {})
            break
