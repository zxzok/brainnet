"""In-memory session state for the BrainNet orchestrator."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RoiData:
    """Preprocessed ROI time series and QC metrics."""

    timeseries: np.ndarray
    labels: list[str]
    qc: dict[str, Any]
    source_path: str


@dataclass
class StrategyResult:
    """Output of one analysis strategy."""

    strategy_type: str
    method: str
    params: dict[str, Any]
    artifacts: Any  # AnalysisArtifacts or None
    metrics_summary: dict[str, Any]


@dataclass
class SessionStore:
    """State for a single orchestrator chat session."""

    session_id: str
    roi_data: dict[str, RoiData] = field(default_factory=dict)
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)

    def store_roi_data(self, data: RoiData) -> str:
        rid = f"roi_{uuid.uuid4().hex[:8]}"
        self.roi_data[rid] = data
        return rid

    def get_roi_data(self, roi_id: str) -> RoiData | None:
        return self.roi_data.get(roi_id)

    def store_strategy(self, result: StrategyResult) -> str:
        sid = f"strategy_{uuid.uuid4().hex[:8]}"
        self.strategies[sid] = result
        return sid

    def get_strategy(self, strategy_id: str) -> StrategyResult | None:
        return self.strategies.get(strategy_id)

    def estimate_tokens(self) -> int:
        text = str(self.messages)
        return len(text) // 4

    def touch(self) -> None:
        self.last_active = time.time()


class SessionManager:
    """Manage multiple concurrent sessions with limits and eviction."""

    def __init__(self, *, max_sessions: int = 10, timeout_seconds: int = 1800) -> None:
        self._sessions: dict[str, SessionStore] = {}
        self._max_sessions = max_sessions
        self._timeout_seconds = timeout_seconds

    def create_session(self) -> SessionStore:
        self.evict_expired()
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(
                f"Maximum {self._max_sessions} sessions reached. "
                "Please wait for an existing session to expire."
            )
        sid = uuid.uuid4().hex
        session = SessionStore(session_id=sid)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> SessionStore | None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active > self._timeout_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
