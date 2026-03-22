"""ChatGPT conversation summarizer.

This module extracts, filters, and summarizes ChatGPT conversations.
It supports two data sources:

1. **Export JSON** — The ``conversations.json`` file obtained from
   ChatGPT Settings → Data Controls → Export data.
2. **Unofficial API** — Fetches conversations directly from
   ``chatgpt.com/backend-api`` using a session token from the browser.

The summarizer uses the OpenAI API (or a local fallback) to extract
key points from each conversation and produce a structured daily digest.

Examples
--------
Summarize today's conversations from an export file::

    python -m brainnet.chatgpt_summarizer --file conversations.json

Fetch and summarize via session token::

    python -m brainnet.chatgpt_summarizer --token <session_token>

Generate a summary for a specific date::

    python -m brainnet.chatgpt_summarizer --file conversations.json --date 2026-03-21
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional dependencies --------------------------------------------------
try:  # pragma: no cover - optional
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # type: ignore

try:  # pragma: no cover - optional
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


# -- configuration --------------------------------------------------------


@dataclass
class SummarizerConfig:
    """Configuration for the ChatGPT conversation summarizer.

    Parameters
    ----------
    openai_api_key : str | None
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var or
        ``~/.openai_api_key`` file.
    model : str
        OpenAI model used for summarization.
    max_conversations : int
        Maximum number of conversations to process per run.
    min_message_count : int
        Skip conversations with fewer messages than this threshold.
    language : str
        Output language for summaries.
    output_dir : str
        Directory to save summary reports.
    """

    openai_api_key: str | None = None
    model: str = "gpt-4o-mini"
    max_conversations: int = 50
    min_message_count: int = 2
    language: str = "zh"
    output_dir: str = "./chatgpt_summaries"

    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_conversations < 1:
            raise ValueError("max_conversations must be >= 1")
        if self.min_message_count < 1:
            raise ValueError("min_message_count must be >= 1")

    def resolve_api_key(self) -> str | None:
        """Resolve the OpenAI API key from config, env, or file."""
        if self.openai_api_key:
            return self.openai_api_key
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
        cfg = os.path.expanduser("~/.openai_api_key")
        if os.path.exists(cfg):
            try:
                with open(cfg, "r", encoding="utf-8") as fh:
                    return fh.read().strip()
            except OSError:
                return None
        return None


# -- data model -----------------------------------------------------------


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime | None = None


@dataclass
class Conversation:
    """A parsed ChatGPT conversation."""

    id: str
    title: str
    create_time: datetime
    update_time: datetime
    messages: list[Message] = field(default_factory=list)
    model_slug: str = ""

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "user"]

    @property
    def assistant_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "assistant"]

    def to_text(self, max_chars: int = 8000) -> str:
        """Render conversation as plain text, truncated to *max_chars*."""
        lines: list[str] = [f"## {self.title}\n"]
        for msg in self.messages:
            if msg.role in ("user", "assistant"):
                label = "User" if msg.role == "user" else "Assistant"
                lines.append(f"**{label}:** {msg.content}\n")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text


@dataclass
class ConversationSummary:
    """Summary of a single conversation."""

    conversation_id: str
    title: str
    key_points: list[str]
    topics: list[str]
    importance: str  # "high", "medium", "low"
    one_line: str


@dataclass
class DailySummary:
    """Aggregated daily summary."""

    date: str
    total_conversations: int
    summaries: list[ConversationSummary]
    overview: str
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "total_conversations": self.total_conversations,
            "overview": self.overview,
            "generated_at": self.generated_at,
            "conversations": [
                {
                    "id": s.conversation_id,
                    "title": s.title,
                    "importance": s.importance,
                    "one_line": s.one_line,
                    "key_points": s.key_points,
                    "topics": s.topics,
                }
                for s in self.summaries
            ],
        }

    def to_markdown(self) -> str:
        """Render the daily summary as Markdown."""
        lines = [
            f"# ChatGPT 每日对话总结 — {self.date}\n",
            f"**对话数量：** {self.total_conversations}\n",
            f"## 总览\n\n{self.overview}\n",
            "---\n",
        ]
        for i, s in enumerate(self.summaries, 1):
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                s.importance, "⚪"
            )
            lines.append(f"### {i}. {s.title}  {icon} {s.importance}\n")
            lines.append(f"> {s.one_line}\n")
            if s.key_points:
                lines.append("**要点：**\n")
                for pt in s.key_points:
                    lines.append(f"- {pt}")
                lines.append("")
            if s.topics:
                lines.append(f"**话题标签：** {', '.join(s.topics)}\n")
            lines.append("---\n")
        lines.append(f"\n_生成时间: {self.generated_at}_\n")
        return "\n".join(lines)


# -- parser ---------------------------------------------------------------


class ExportParser:
    """Parse a ChatGPT data export ``conversations.json`` file.

    Parameters
    ----------
    path : str | Path
        Path to the ``conversations.json`` file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def parse(self) -> list[Conversation]:
        """Parse all conversations from the export file.

        Returns
        -------
        list[Conversation]
            Parsed conversations sorted by creation time (newest first).
        """
        with open(self.path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        conversations: list[Conversation] = []
        for item in raw:
            conv = self._parse_conversation(item)
            if conv is not None:
                conversations.append(conv)
        conversations.sort(key=lambda c: c.create_time, reverse=True)
        logger.info("Parsed %d conversations from %s", len(conversations), self.path)
        return conversations

    def _parse_conversation(self, item: dict[str, Any]) -> Conversation | None:
        """Parse a single conversation dict from the export."""
        try:
            create_ts = item.get("create_time", 0)
            update_ts = item.get("update_time", 0)
            conv = Conversation(
                id=item.get("id", ""),
                title=item.get("title", "Untitled"),
                create_time=datetime.fromtimestamp(create_ts, tz=timezone.utc),
                update_time=datetime.fromtimestamp(update_ts, tz=timezone.utc),
            )
            # Extract messages from the mapping structure
            mapping = item.get("mapping", {})
            messages = self._extract_messages(mapping)
            conv.messages = messages
            return conv
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to parse conversation: %s", exc)
            return None

    @staticmethod
    def _extract_messages(mapping: dict[str, Any]) -> list[Message]:
        """Extract ordered messages from the ChatGPT mapping structure."""
        messages: list[Message] = []
        for node in mapping.values():
            msg_data = node.get("message")
            if msg_data is None:
                continue
            role = msg_data.get("author", {}).get("role", "")
            if role not in ("user", "assistant"):
                continue
            content_parts = msg_data.get("content", {}).get("parts", [])
            text_parts = [str(p) for p in content_parts if isinstance(p, str)]
            content = "\n".join(text_parts).strip()
            if not content:
                continue
            ts = msg_data.get("create_time")
            timestamp = (
                datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
            )
            messages.append(Message(role=role, content=content, timestamp=timestamp))
        # Sort by timestamp if available
        messages.sort(key=lambda m: m.timestamp or datetime.min.replace(tzinfo=timezone.utc))
        return messages


# -- API fetcher ----------------------------------------------------------


class APIFetcher:
    """Fetch conversations from the ChatGPT unofficial backend API.

    Parameters
    ----------
    session_token : str
        The ``__Secure-next-auth.session-token`` cookie value from
        ``chatgpt.com``.  Obtain this from your browser's developer tools.
    base_url : str
        Base URL for the ChatGPT backend API.

    Notes
    -----
    This uses an unofficial API that may change without notice.  Use at
    your own risk and in accordance with OpenAI's terms of service.
    """

    DEFAULT_BASE_URL = "https://chatgpt.com/backend-api"

    def __init__(
        self,
        session_token: str,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        if requests is None:
            raise RuntimeError(
                "The 'requests' library is required for API fetching. "
                "Install it with: pip install requests"
            )
        self.session_token = session_token
        self.base_url = base_url
        self._access_token: str | None = None

    def _get_access_token(self) -> str:
        """Exchange the session token for a short-lived access token."""
        if self._access_token:
            return self._access_token
        resp = requests.get(
            "https://chatgpt.com/api/auth/session",
            cookies={"__Secure-next-auth.session-token": self.session_token},
            timeout=30,
        )
        resp.raise_for_status()
        self._access_token = resp.json()["accessToken"]
        return self._access_token

    def _headers(self) -> dict[str, str]:
        token = self._get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def fetch_conversation_list(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Fetch a page of conversation metadata.

        Parameters
        ----------
        limit : int
            Number of conversations to fetch.
        offset : int
            Pagination offset.

        Returns
        -------
        list[dict]
            List of conversation metadata dicts.
        """
        resp = requests.get(
            f"{self.base_url}/conversations",
            params={"limit": limit, "offset": offset},
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("items", [])

    def fetch_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Fetch the full content of a single conversation.

        Parameters
        ----------
        conversation_id : str
            The UUID of the conversation.

        Returns
        -------
        dict
            Full conversation data including all messages.
        """
        resp = requests.get(
            f"{self.base_url}/conversation/{conversation_id}",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_recent(self, limit: int = 50) -> list[Conversation]:
        """Fetch recent conversations and parse them.

        Parameters
        ----------
        limit : int
            Maximum number of conversations to fetch.

        Returns
        -------
        list[Conversation]
            Parsed conversations.
        """
        items = self.fetch_conversation_list(limit=limit)
        conversations: list[Conversation] = []
        for item in items:
            try:
                full = self.fetch_conversation(item["id"])
                conv = self._parse_api_response(full)
                if conv:
                    conversations.append(conv)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to fetch conversation %s: %s", item.get("id"), exc
                )
        logger.info("Fetched %d conversations via API", len(conversations))
        return conversations

    @staticmethod
    def _parse_api_response(data: dict[str, Any]) -> Conversation | None:
        """Parse a conversation from the API response format."""
        try:
            create_ts = data.get("create_time", 0)
            update_ts = data.get("update_time", 0)
            conv = Conversation(
                id=data.get("id", ""),
                title=data.get("title", "Untitled"),
                create_time=datetime.fromtimestamp(create_ts, tz=timezone.utc),
                update_time=datetime.fromtimestamp(update_ts, tz=timezone.utc),
            )
            mapping = data.get("mapping", {})
            conv.messages = ExportParser._extract_messages(mapping)
            return conv
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to parse API conversation: %s", exc)
            return None


# -- summarizer -----------------------------------------------------------


class ConversationSummarizer:
    """Summarize ChatGPT conversations using the OpenAI API.

    Parameters
    ----------
    config : SummarizerConfig
        Summarizer configuration.
    """

    SINGLE_PROMPT_TEMPLATE = """\
你是一个对话分析助手。请分析以下 ChatGPT 对话，并提取关键信息。

对话内容：
{conversation_text}

请以 JSON 格式返回，包含以下字段：
- "key_points": 列出 3-5 个关键要点（数组）
- "topics": 列出相关话题标签（数组，2-4个）
- "importance": 重要程度，"high"/"medium"/"low"
- "one_line": 一句话总结

只返回 JSON，不要包含其他内容。"""

    DAILY_PROMPT_TEMPLATE = """\
你是一个每日总结助手。请根据以下对话摘要，生成一份简洁的每日总结。

今日对话摘要：
{summaries_text}

请生成一段 200 字以内的总览，涵盖今日主要话题和重点发现。
用{language}输出。只输出总览文本，不要包含其他内容。"""

    def __init__(self, config: SummarizerConfig | None = None) -> None:
        self.config = config or SummarizerConfig()
        self.config.validate()

    def summarize_conversation(self, conv: Conversation) -> ConversationSummary:
        """Summarize a single conversation.

        Parameters
        ----------
        conv : Conversation
            The conversation to summarize.

        Returns
        -------
        ConversationSummary
            Extracted summary with key points and topics.
        """
        api_key = self.config.resolve_api_key()
        if api_key and openai is not None:
            return self._llm_summarize(conv, api_key)
        return self._rule_based_summarize(conv)

    def summarize_daily(
        self,
        conversations: list[Conversation],
        target_date: str | None = None,
    ) -> DailySummary:
        """Generate a daily summary for the given conversations.

        Parameters
        ----------
        conversations : list[Conversation]
            Conversations to summarize (should be pre-filtered by date).
        target_date : str | None
            Date string (YYYY-MM-DD).  Defaults to today.

        Returns
        -------
        DailySummary
            The aggregated daily summary.
        """
        if target_date is None:
            target_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        # Filter by min message count
        filtered = [
            c for c in conversations
            if c.message_count >= self.config.min_message_count
        ]
        # Limit
        filtered = filtered[: self.config.max_conversations]

        logger.info(
            "Summarizing %d conversations for %s", len(filtered), target_date
        )

        # Summarize each conversation
        summaries: list[ConversationSummary] = []
        for conv in filtered:
            summary = self.summarize_conversation(conv)
            summaries.append(summary)

        # Generate overview
        overview = self._generate_overview(summaries)

        return DailySummary(
            date=target_date,
            total_conversations=len(summaries),
            summaries=summaries,
            overview=overview,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    def _llm_summarize(
        self, conv: Conversation, api_key: str
    ) -> ConversationSummary:
        """Summarize using the OpenAI API."""
        client = openai.OpenAI(api_key=api_key)
        prompt = self.SINGLE_PROMPT_TEMPLATE.format(
            conversation_text=conv.to_text(max_chars=6000)
        )
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            content = response.choices[0].message.content or ""
            return self._parse_llm_response(conv, content)
        except Exception as exc:
            logger.warning("LLM summarization failed: %s; falling back", exc)
            return self._rule_based_summarize(conv)

    def _parse_llm_response(
        self, conv: Conversation, content: str
    ) -> ConversationSummary:
        """Parse the JSON response from the LLM."""
        # Extract JSON from possible markdown code block
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return ConversationSummary(
                    conversation_id=conv.id,
                    title=conv.title,
                    key_points=data.get("key_points", []),
                    topics=data.get("topics", []),
                    importance=data.get("importance", "medium"),
                    one_line=data.get("one_line", conv.title),
                )
            except json.JSONDecodeError:
                pass
        return self._rule_based_summarize(conv)

    def _rule_based_summarize(self, conv: Conversation) -> ConversationSummary:
        """Fallback: generate a basic summary without LLM."""
        user_msgs = conv.user_messages
        # Extract first question as one-line summary
        one_line = user_msgs[0].content[:100] if user_msgs else conv.title
        # Key points: first sentence of each user message (up to 5)
        key_points: list[str] = []
        for msg in user_msgs[:5]:
            first_line = msg.content.split("\n")[0][:120]
            key_points.append(first_line)
        # Simple topic extraction from title
        topics = [conv.title] if conv.title else []
        # Importance heuristic based on message count
        if conv.message_count >= 20:
            importance = "high"
        elif conv.message_count >= 8:
            importance = "medium"
        else:
            importance = "low"
        return ConversationSummary(
            conversation_id=conv.id,
            title=conv.title,
            key_points=key_points,
            topics=topics,
            importance=importance,
            one_line=one_line,
        )

    def _generate_overview(self, summaries: list[ConversationSummary]) -> str:
        """Generate a daily overview from individual summaries."""
        api_key = self.config.resolve_api_key()
        if api_key and openai is not None:
            return self._llm_overview(summaries, api_key)
        return self._rule_based_overview(summaries)

    def _llm_overview(
        self, summaries: list[ConversationSummary], api_key: str
    ) -> str:
        """Generate overview using LLM."""
        summary_lines = []
        for s in summaries:
            summary_lines.append(
                f"- [{s.importance}] {s.title}: {s.one_line}"
            )
        summaries_text = "\n".join(summary_lines)
        language = "中文" if self.config.language == "zh" else "English"
        prompt = self.DAILY_PROMPT_TEMPLATE.format(
            summaries_text=summaries_text,
            language=language,
        )
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("LLM overview failed: %s", exc)
            return self._rule_based_overview(summaries)

    @staticmethod
    def _rule_based_overview(summaries: list[ConversationSummary]) -> str:
        """Fallback overview without LLM."""
        high = [s for s in summaries if s.importance == "high"]
        total = len(summaries)
        topics = []
        for s in summaries:
            topics.extend(s.topics)
        unique_topics = list(dict.fromkeys(topics))[:10]
        overview = f"今日共 {total} 段对话"
        if high:
            overview += f"，其中 {len(high)} 段为高重要性"
        if unique_topics:
            overview += f"。涉及话题：{'、'.join(unique_topics)}"
        overview += "。"
        return overview


# -- date filter ----------------------------------------------------------


def filter_by_date(
    conversations: list[Conversation],
    target_date: str | None = None,
) -> list[Conversation]:
    """Filter conversations by date.

    Parameters
    ----------
    conversations : list[Conversation]
        All conversations.
    target_date : str | None
        Target date in ``YYYY-MM-DD`` format.  Defaults to today (UTC).

    Returns
    -------
    list[Conversation]
        Conversations created or updated on the target date.
    """
    if target_date is None:
        target_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    target = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    next_day = target + timedelta(days=1)
    return [
        c
        for c in conversations
        if target <= c.update_time < next_day or target <= c.create_time < next_day
    ]


# -- output ---------------------------------------------------------------


def save_summary(summary: DailySummary, output_dir: str = "./chatgpt_summaries") -> Path:
    """Save a daily summary as both JSON and Markdown.

    Parameters
    ----------
    summary : DailySummary
        The summary to save.
    output_dir : str
        Output directory path.

    Returns
    -------
    Path
        Path to the saved Markdown file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / f"summary_{summary.date}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, ensure_ascii=False, indent=2)
    logger.info("Saved JSON summary to %s", json_path)

    # Markdown
    md_path = out / f"summary_{summary.date}.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(summary.to_markdown())
    logger.info("Saved Markdown summary to %s", md_path)

    return md_path


# -- scheduler ------------------------------------------------------------


class AutoScheduler:
    """Automated scheduler that runs the summarizer on a daily schedule.

    Supports two modes:

    1. **Daemon mode** — Runs as a long-lived process, executing the
       summarizer at a configured time each day.
    2. **Crontab setup** — Generates and installs a crontab entry so
       the OS handles scheduling.

    Parameters
    ----------
    config : SummarizerConfig
        Summarizer configuration.
    source_file : str | None
        Path to ChatGPT export file (re-read each run for fresh data).
    session_token : str | None
        Session token for API-based fetching.
    run_hour : int
        Hour of day (0-23) to run the summarizer in daemon mode.
    run_minute : int
        Minute (0-59) to run the summarizer in daemon mode.
    """

    def __init__(
        self,
        config: SummarizerConfig,
        source_file: str | None = None,
        session_token: str | None = None,
        run_hour: int = 23,
        run_minute: int = 30,
    ) -> None:
        if source_file is None and session_token is None:
            raise ValueError("Either source_file or session_token is required")
        self.config = config
        self.source_file = source_file
        self.session_token = session_token
        self.run_hour = run_hour
        self.run_minute = run_minute
        self._running = False

    def _load_conversations(self) -> list[Conversation]:
        """Load conversations from the configured source."""
        if self.source_file:
            return ExportParser(self.source_file).parse()
        if self.session_token:
            fetcher = APIFetcher(session_token=self.session_token)
            return fetcher.fetch_recent(limit=self.config.max_conversations)
        return []

    def run_once(self, target_date: str | None = None) -> DailySummary | None:
        """Execute a single summarization run.

        Parameters
        ----------
        target_date : str | None
            Date to summarize.  Defaults to today.

        Returns
        -------
        DailySummary | None
            The generated summary, or ``None`` if no conversations found.
        """
        conversations = self._load_conversations()
        if not conversations:
            logger.warning("No conversations loaded")
            return None

        daily_convs = filter_by_date(conversations, target_date)
        target = target_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if not daily_convs:
            logger.info("No conversations for %s", target)
            return None

        summarizer = ConversationSummarizer(self.config)
        daily = summarizer.summarize_daily(daily_convs, target_date=target)
        save_summary(daily, self.config.output_dir)
        logger.info("Summary saved for %s (%d conversations)", target, daily.total_conversations)
        return daily

    def run_daemon(self) -> None:
        """Run as a daemon, executing daily at the configured time.

        This method blocks indefinitely.  Use Ctrl+C or send SIGTERM to stop.
        """
        import signal
        import time

        self._running = True

        def _handle_signal(signum: int, frame: Any) -> None:
            logger.info("Received signal %d, shutting down...", signum)
            self._running = False

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        logger.info(
            "Daemon started. Will run daily at %02d:%02d",
            self.run_hour,
            self.run_minute,
        )

        while self._running:
            now = datetime.now()
            # Calculate next run time
            next_run = now.replace(
                hour=self.run_hour,
                minute=self.run_minute,
                second=0,
                microsecond=0,
            )
            if next_run <= now:
                next_run += timedelta(days=1)

            wait_seconds = (next_run - now).total_seconds()
            logger.info("Next run at %s (in %.0f seconds)", next_run, wait_seconds)

            # Wait in small increments so we can respond to signals
            while wait_seconds > 0 and self._running:
                time.sleep(min(wait_seconds, 60))
                wait_seconds -= 60

            if not self._running:
                break

            logger.info("Running scheduled summarization...")
            try:
                self.run_once()
            except Exception as exc:
                logger.error("Scheduled run failed: %s", exc)

    def install_crontab(self) -> str:
        """Generate and install a crontab entry for daily execution.

        Returns
        -------
        str
            The crontab line that was installed.
        """
        import shutil
        import subprocess
        import sys

        python = sys.executable
        script = Path(__file__).resolve()

        # Build the command
        parts = [python, str(script)]
        if self.source_file:
            parts.extend(["--file", str(Path(self.source_file).resolve())])
        elif self.session_token:
            parts.extend(["--token", self.session_token])
        parts.extend(["--output", str(Path(self.config.output_dir).resolve())])
        parts.extend(["--model", self.config.model])
        parts.extend(["--lang", self.config.language])
        cmd = " ".join(parts)

        cron_line = f"{self.run_minute} {self.run_hour} * * * {cmd}"

        if shutil.which("crontab") is None:
            logger.warning("crontab not found; printing entry for manual setup")
            print(f"Add this to your crontab:\n{cron_line}")
            return cron_line

        # Read existing crontab
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True,
            text=True,
        )
        existing = result.stdout if result.returncode == 0 else ""

        # Check if already installed
        marker = "# chatgpt-summarizer"
        if marker in existing:
            # Replace existing entry
            lines = existing.splitlines()
            lines = [l for l in lines if marker not in l and "chatgpt_summarizer" not in l]
            existing = "\n".join(lines) + "\n"

        new_crontab = existing.rstrip("\n") + f"\n{cron_line}  {marker}\n"
        subprocess.run(
            ["crontab", "-"],
            input=new_crontab,
            text=True,
            check=True,
        )
        logger.info("Crontab entry installed: %s", cron_line)
        return cron_line

    def generate_systemd_service(self, output_path: str | None = None) -> str:
        """Generate a systemd user service file for daemon mode.

        Parameters
        ----------
        output_path : str | None
            Path to write the service file.  If ``None``, prints to stdout.

        Returns
        -------
        str
            The service file content.
        """
        import sys

        python = sys.executable
        script = Path(__file__).resolve()

        parts = [python, str(script), "--daemon"]
        if self.source_file:
            parts.extend(["--file", str(Path(self.source_file).resolve())])
        elif self.session_token:
            parts.extend(["--token", "$SESSION_TOKEN"])
        parts.extend(["--output", str(Path(self.config.output_dir).resolve())])
        parts.extend(["--time", f"{self.run_hour:02d}:{self.run_minute:02d}"])
        exec_cmd = " ".join(parts)

        service = f"""\
[Unit]
Description=ChatGPT Conversation Daily Summarizer
After=network-online.target

[Service]
Type=simple
ExecStart={exec_cmd}
Restart=on-failure
RestartSec=60
Environment=OPENAI_API_KEY=

[Install]
WantedBy=default.target
"""
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(service)
            logger.info("Service file written to %s", output_path)
            print(f"Install with:\n  cp {output_path} ~/.config/systemd/user/")
            print("  systemctl --user daemon-reload")
            print("  systemctl --user enable --now chatgpt-summarizer")
        else:
            print(service)
        return service


# -- CLI ------------------------------------------------------------------


def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ChatGPT Conversation Daily Summarizer",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--file", "-f",
        help="Path to ChatGPT export conversations.json",
    )
    src.add_argument(
        "--token", "-t",
        help="ChatGPT session token for API fetching",
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="Target date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--output", "-o",
        default="./chatgpt_summaries",
        help="Output directory (default: ./chatgpt_summaries)",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="OpenAI model for summarization (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--lang",
        default="zh",
        choices=["zh", "en"],
        help="Output language (default: zh)",
    )
    parser.add_argument(
        "--max", type=int,
        default=50,
        help="Max conversations to process (default: 50)",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Summarize all dates, not just the target date",
    )

    # Automation options
    auto = parser.add_argument_group("automation")
    auto.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a daemon, executing daily at the scheduled time",
    )
    auto.add_argument(
        "--install-cron",
        action="store_true",
        help="Install a crontab entry for daily execution",
    )
    auto.add_argument(
        "--generate-service",
        metavar="PATH",
        nargs="?",
        const="",
        help="Generate a systemd service file (optionally specify output path)",
    )
    auto.add_argument(
        "--time",
        default="23:30",
        help="Scheduled run time HH:MM (default: 23:30)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse scheduled time
    hour, minute = (int(x) for x in args.time.split(":"))

    config = SummarizerConfig(
        model=args.model,
        max_conversations=args.max,
        language=args.lang,
        output_dir=args.output,
    )

    # Handle automation commands
    if args.daemon or args.install_cron or args.generate_service is not None:
        scheduler = AutoScheduler(
            config=config,
            source_file=args.file,
            session_token=args.token,
            run_hour=hour,
            run_minute=minute,
        )
        if args.install_cron:
            scheduler.install_crontab()
            return
        if args.generate_service is not None:
            path = args.generate_service or None
            scheduler.generate_systemd_service(output_path=path)
            return
        if args.daemon:
            scheduler.run_daemon()
            return

    # One-shot mode
    if args.file:
        logger.info("Parsing export file: %s", args.file)
        conversations = ExportParser(args.file).parse()
    else:
        logger.info("Fetching conversations via API...")
        fetcher = APIFetcher(session_token=args.token)
        conversations = fetcher.fetch_recent(limit=args.max)

    if not conversations:
        logger.warning("No conversations found.")
        return

    logger.info("Total conversations loaded: %d", len(conversations))

    if args.all_dates:
        from collections import defaultdict

        by_date: dict[str, list[Conversation]] = defaultdict(list)
        for c in conversations:
            date_key = c.update_time.strftime("%Y-%m-%d")
            by_date[date_key].append(c)
        for date_key in sorted(by_date.keys(), reverse=True):
            summarizer = ConversationSummarizer(config)
            daily = summarizer.summarize_daily(by_date[date_key], target_date=date_key)
            md_path = save_summary(daily, args.output)
            print(f"[{date_key}] {daily.total_conversations} conversations → {md_path}")
    else:
        daily_convs = filter_by_date(conversations, args.date)
        target = args.date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if not daily_convs:
            logger.warning("No conversations found for %s", target)
            return
        logger.info("Conversations for %s: %d", target, len(daily_convs))

        summarizer = ConversationSummarizer(config)
        daily = summarizer.summarize_daily(daily_convs, target_date=target)
        md_path = save_summary(daily, args.output)

        print(daily.to_markdown())
        print(f"\nSaved to: {md_path}")


if __name__ == "__main__":
    main()
