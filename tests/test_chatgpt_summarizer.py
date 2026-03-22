"""Tests for the ChatGPT conversation summarizer module."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from brainnet.chatgpt_summarizer import (
    AutoScheduler,
    Conversation,
    ConversationSummarizer,
    DailySummary,
    ExportParser,
    Message,
    SummarizerConfig,
    filter_by_date,
    save_summary,
)


# -- fixtures -------------------------------------------------------------


def _make_conversation(
    conv_id: str = "conv-1",
    title: str = "Test Conversation",
    create_ts: float | None = None,
    update_ts: float | None = None,
    n_messages: int = 4,
) -> Conversation:
    """Helper to create a test conversation."""
    now = time.time()
    create_ts = create_ts or now - 3600
    update_ts = update_ts or now
    messages = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(
            Message(
                role=role,
                content=f"Test message {i} from {role}",
                timestamp=datetime.fromtimestamp(
                    create_ts + i * 60, tz=timezone.utc
                ),
            )
        )
    return Conversation(
        id=conv_id,
        title=title,
        create_time=datetime.fromtimestamp(create_ts, tz=timezone.utc),
        update_time=datetime.fromtimestamp(update_ts, tz=timezone.utc),
        messages=messages,
    )


def _make_export_json(tmp_path: Path, conversations: list[dict]) -> Path:
    """Write a fake conversations.json export file."""
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(conversations), encoding="utf-8")
    return path


# -- SummarizerConfig tests -----------------------------------------------


class TestSummarizerConfig:
    def test_defaults(self) -> None:
        cfg = SummarizerConfig()
        assert cfg.model == "gpt-4o-mini"
        assert cfg.max_conversations == 50
        assert cfg.min_message_count == 2
        assert cfg.language == "zh"

    def test_validate_ok(self) -> None:
        SummarizerConfig().validate()

    def test_validate_bad_max(self) -> None:
        with pytest.raises(ValueError, match="max_conversations"):
            SummarizerConfig(max_conversations=0).validate()

    def test_validate_bad_min(self) -> None:
        with pytest.raises(ValueError, match="min_message_count"):
            SummarizerConfig(min_message_count=0).validate()

    def test_resolve_api_key_from_config(self) -> None:
        cfg = SummarizerConfig(openai_api_key="sk-test")
        assert cfg.resolve_api_key() == "sk-test"

    def test_resolve_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        cfg = SummarizerConfig()
        assert cfg.resolve_api_key() == "sk-env"

    def test_resolve_api_key_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = SummarizerConfig()
        # Will be None unless ~/.openai_api_key exists
        key = cfg.resolve_api_key()
        assert key is None or isinstance(key, str)


# -- Message / Conversation model tests -----------------------------------


class TestDataModels:
    def test_conversation_properties(self) -> None:
        conv = _make_conversation(n_messages=6)
        assert conv.message_count == 6
        assert len(conv.user_messages) == 3
        assert len(conv.assistant_messages) == 3

    def test_conversation_to_text(self) -> None:
        conv = _make_conversation(n_messages=4)
        text = conv.to_text()
        assert "Test Conversation" in text
        assert "User" in text
        assert "Assistant" in text

    def test_conversation_to_text_truncation(self) -> None:
        conv = _make_conversation(n_messages=4)
        text = conv.to_text(max_chars=50)
        assert len(text) <= 70  # some slack for truncation marker
        assert "truncated" in text


# -- ExportParser tests ---------------------------------------------------


class TestExportParser:
    def test_parse_empty(self, tmp_path: Path) -> None:
        path = _make_export_json(tmp_path, [])
        parser = ExportParser(path)
        assert parser.parse() == []

    def test_parse_basic(self, tmp_path: Path) -> None:
        now = time.time()
        data = [
            {
                "id": "conv-1",
                "title": "Test Chat",
                "create_time": now - 3600,
                "update_time": now,
                "mapping": {
                    "node-1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Hello, how are you?"]},
                            "create_time": now - 3600,
                        }
                    },
                    "node-2": {
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["I'm fine, thanks!"]},
                            "create_time": now - 3500,
                        }
                    },
                    "node-3": {
                        "message": None,  # system node, should be skipped
                    },
                },
            }
        ]
        path = _make_export_json(tmp_path, data)
        convs = ExportParser(path).parse()
        assert len(convs) == 1
        assert convs[0].title == "Test Chat"
        assert len(convs[0].messages) == 2
        assert convs[0].messages[0].role == "user"
        assert convs[0].messages[1].role == "assistant"

    def test_parse_skips_empty_content(self, tmp_path: Path) -> None:
        now = time.time()
        data = [
            {
                "id": "conv-2",
                "title": "Empty",
                "create_time": now,
                "update_time": now,
                "mapping": {
                    "node-1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": [""]},
                            "create_time": now,
                        }
                    },
                },
            }
        ]
        path = _make_export_json(tmp_path, data)
        convs = ExportParser(path).parse()
        assert len(convs) == 1
        assert convs[0].message_count == 0  # empty content skipped

    def test_parse_multiple_sorted(self, tmp_path: Path) -> None:
        now = time.time()
        data = [
            {
                "id": "older",
                "title": "Older",
                "create_time": now - 7200,
                "update_time": now - 7200,
                "mapping": {},
            },
            {
                "id": "newer",
                "title": "Newer",
                "create_time": now,
                "update_time": now,
                "mapping": {},
            },
        ]
        path = _make_export_json(tmp_path, data)
        convs = ExportParser(path).parse()
        assert convs[0].id == "newer"  # newest first


# -- filter_by_date tests -------------------------------------------------


class TestFilterByDate:
    def test_filter_today(self) -> None:
        now = time.time()
        today_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        c1 = _make_conversation("c1", create_ts=now - 100, update_ts=now)
        c2 = _make_conversation("c2", create_ts=now - 86400 * 2, update_ts=now - 86400 * 2)
        result = filter_by_date([c1, c2], today_str)
        assert len(result) == 1
        assert result[0].id == "c1"

    def test_filter_specific_date(self) -> None:
        # Create conversation on 2025-01-15
        ts = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc).timestamp()
        c = _make_conversation("c1", create_ts=ts, update_ts=ts)
        assert len(filter_by_date([c], "2025-01-15")) == 1
        assert len(filter_by_date([c], "2025-01-16")) == 0


# -- ConversationSummarizer tests -----------------------------------------


class TestConversationSummarizer:
    def test_rule_based_summary(self) -> None:
        conv = _make_conversation(n_messages=4)
        summarizer = ConversationSummarizer()
        summary = summarizer.summarize_conversation(conv)
        assert summary.conversation_id == "conv-1"
        assert summary.title == "Test Conversation"
        assert len(summary.key_points) > 0
        assert summary.importance in ("high", "medium", "low")

    def test_importance_high(self) -> None:
        conv = _make_conversation(n_messages=24)
        summarizer = ConversationSummarizer()
        summary = summarizer.summarize_conversation(conv)
        assert summary.importance == "high"

    def test_importance_medium(self) -> None:
        conv = _make_conversation(n_messages=10)
        summarizer = ConversationSummarizer()
        summary = summarizer.summarize_conversation(conv)
        assert summary.importance == "medium"

    def test_importance_low(self) -> None:
        conv = _make_conversation(n_messages=4)
        summarizer = ConversationSummarizer()
        summary = summarizer.summarize_conversation(conv)
        assert summary.importance == "low"

    def test_summarize_daily(self) -> None:
        convs = [_make_conversation(f"c{i}", n_messages=4) for i in range(3)]
        summarizer = ConversationSummarizer()
        daily = summarizer.summarize_daily(convs, target_date="2025-03-22")
        assert daily.date == "2025-03-22"
        assert daily.total_conversations == 3
        assert len(daily.summaries) == 3
        assert daily.overview != ""

    def test_min_message_filter(self) -> None:
        config = SummarizerConfig(min_message_count=5)
        convs = [
            _make_conversation("short", n_messages=2),
            _make_conversation("long", n_messages=6),
        ]
        summarizer = ConversationSummarizer(config)
        daily = summarizer.summarize_daily(convs, target_date="2025-03-22")
        assert daily.total_conversations == 1

    def test_max_conversations_limit(self) -> None:
        config = SummarizerConfig(max_conversations=2)
        convs = [_make_conversation(f"c{i}", n_messages=4) for i in range(5)]
        summarizer = ConversationSummarizer(config)
        daily = summarizer.summarize_daily(convs, target_date="2025-03-22")
        assert daily.total_conversations == 2

    def test_parse_llm_response_valid(self) -> None:
        conv = _make_conversation()
        summarizer = ConversationSummarizer()
        content = json.dumps({
            "key_points": ["point 1", "point 2"],
            "topics": ["python", "testing"],
            "importance": "high",
            "one_line": "A test conversation",
        })
        result = summarizer._parse_llm_response(conv, content)
        assert result.importance == "high"
        assert len(result.key_points) == 2

    def test_parse_llm_response_markdown(self) -> None:
        conv = _make_conversation()
        summarizer = ConversationSummarizer()
        content = '```json\n{"key_points": ["p1"], "topics": ["t1"], "importance": "low", "one_line": "test"}\n```'
        result = summarizer._parse_llm_response(conv, content)
        assert result.importance == "low"

    def test_parse_llm_response_invalid(self) -> None:
        conv = _make_conversation()
        summarizer = ConversationSummarizer()
        result = summarizer._parse_llm_response(conv, "not json at all")
        # Falls back to rule-based
        assert result.conversation_id == conv.id


# -- DailySummary tests ---------------------------------------------------


class TestDailySummary:
    def _make_daily(self) -> DailySummary:
        from brainnet.chatgpt_summarizer import ConversationSummary

        summaries = [
            ConversationSummary(
                conversation_id="c1",
                title="Chat about Python",
                key_points=["Discussed async", "Reviewed decorators"],
                topics=["python", "async"],
                importance="high",
                one_line="Deep dive into Python async patterns",
            ),
            ConversationSummary(
                conversation_id="c2",
                title="Quick question",
                key_points=["Asked about git"],
                topics=["git"],
                importance="low",
                one_line="Git rebase question",
            ),
        ]
        return DailySummary(
            date="2025-03-22",
            total_conversations=2,
            summaries=summaries,
            overview="Today focused on Python and git.",
            generated_at="2025-03-22T23:30:00+00:00",
        )

    def test_to_dict(self) -> None:
        daily = self._make_daily()
        d = daily.to_dict()
        assert d["date"] == "2025-03-22"
        assert len(d["conversations"]) == 2
        assert d["conversations"][0]["importance"] == "high"

    def test_to_markdown(self) -> None:
        daily = self._make_daily()
        md = daily.to_markdown()
        assert "每日对话总结" in md
        assert "Python" in md
        assert "2025-03-22" in md

    def test_save_summary(self, tmp_path: Path) -> None:
        daily = self._make_daily()
        md_path = save_summary(daily, str(tmp_path))
        assert md_path.exists()
        assert (tmp_path / "summary_2025-03-22.json").exists()

        # Verify JSON content
        with open(tmp_path / "summary_2025-03-22.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_conversations"] == 2


# -- AutoScheduler tests --------------------------------------------------


class TestAutoScheduler:
    def test_requires_source(self) -> None:
        with pytest.raises(ValueError, match="source_file or session_token"):
            AutoScheduler(config=SummarizerConfig())

    def test_run_once_with_file(self, tmp_path: Path) -> None:
        now = time.time()
        today_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        data = [
            {
                "id": "conv-1",
                "title": "Test",
                "create_time": now - 100,
                "update_time": now,
                "mapping": {
                    "n1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["hello"]},
                            "create_time": now - 100,
                        }
                    },
                    "n2": {
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["hi there"]},
                            "create_time": now - 50,
                        }
                    },
                },
            }
        ]
        path = _make_export_json(tmp_path, data)
        out_dir = tmp_path / "output"

        config = SummarizerConfig(output_dir=str(out_dir))
        scheduler = AutoScheduler(config=config, source_file=str(path))
        result = scheduler.run_once(target_date=today_str)

        assert result is not None
        assert result.total_conversations == 1
        assert (out_dir / f"summary_{today_str}.json").exists()

    def test_run_once_no_conversations(self, tmp_path: Path) -> None:
        path = _make_export_json(tmp_path, [])
        config = SummarizerConfig(output_dir=str(tmp_path / "out"))
        scheduler = AutoScheduler(config=config, source_file=str(path))
        result = scheduler.run_once()
        assert result is None
