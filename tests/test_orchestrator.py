from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from brainnet.orchestrator import (
    SYSTEM_PROMPT,
    run_orchestrator,
    sse_format,
)
from brainnet.session_store import SessionStore


def test_sse_format():
    line = sse_format("text_delta", {"text": "hello"})
    assert line.startswith("event: text_delta\n")
    assert '"text": "hello"' in line
    assert line.endswith("\n\n")


def test_system_prompt_contains_safety_rules():
    assert "candidate observations" in SYSTEM_PROMPT
    assert "Maximum 3 strategies" in SYSTEM_PROMPT


def test_run_orchestrator_streams_text():
    session = SessionStore(session_id="test")

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    text_event = MagicMock()
    text_event.type = "content_block_delta"
    text_event.delta = MagicMock()
    text_event.delta.type = "text_delta"
    text_event.delta.text = "Hello!"

    mock_stream.__iter__ = MagicMock(return_value=iter([text_event]))
    mock_stream.get_final_message.return_value = MagicMock(stop_reason="end_turn")

    with patch("brainnet.orchestrator._get_client", return_value=MagicMock()), \
         patch("brainnet.orchestrator._create_stream", return_value=mock_stream):
        events = list(run_orchestrator(session, "test message"))

    assert len(events) >= 1
    assert any("text_delta" in e for e in events)
    assert any("done" in e for e in events)
    assert session.messages[0]["role"] == "user"
    assert session.messages[0]["content"] == "test message"


def test_run_orchestrator_handles_tool_use():
    """Test the tool-calling path where Claude requests a tool and gets results."""
    session = SessionStore(session_id="test")

    # First response: Claude wants to call a tool
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "inspect_qc"
    tool_block.input = {"roi_timeseries_id": "nonexistent"}
    tool_block.id = "tool_123"

    first_msg = MagicMock()
    first_msg.content = [tool_block]
    first_msg.stop_reason = "tool_use"

    first_stream = MagicMock()
    first_stream.__enter__ = MagicMock(return_value=first_stream)
    first_stream.__exit__ = MagicMock(return_value=False)
    first_stream.__iter__ = MagicMock(return_value=iter([]))
    first_stream.get_final_message.return_value = first_msg

    # Second response: Claude sends text after getting tool result
    text_event = MagicMock()
    text_event.type = "content_block_delta"
    text_event.delta = MagicMock()
    text_event.delta.type = "text_delta"
    text_event.delta.text = "The QC data was not found."

    text_block = MagicMock()
    text_block.type = "text"

    second_msg = MagicMock()
    second_msg.content = [text_block]
    second_msg.stop_reason = "end_turn"

    second_stream = MagicMock()
    second_stream.__enter__ = MagicMock(return_value=second_stream)
    second_stream.__exit__ = MagicMock(return_value=False)
    second_stream.__iter__ = MagicMock(return_value=iter([text_event]))
    second_stream.get_final_message.return_value = second_msg

    call_count = 0

    def mock_create_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return first_stream if call_count == 1 else second_stream

    with patch("brainnet.orchestrator._get_client", return_value=MagicMock()), \
         patch("brainnet.orchestrator._create_stream", side_effect=mock_create_stream):
        events = list(run_orchestrator(session, "check QC"))

    # Should have tool_start, tool_result, text_delta, and done events
    event_types = [e.split("\n")[0] for e in events]
    assert any("tool_start" in e for e in event_types)
    assert any("tool_result" in e for e in event_types)
    assert any("done" in e for e in events)

    # Session should have user message, assistant (tool_use), user (tool_result), assistant (text)
    assert len(session.messages) >= 3
    assert session.messages[0]["role"] == "user"
