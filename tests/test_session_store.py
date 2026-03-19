from __future__ import annotations

import time

import numpy as np
import pytest

from brainnet.session_store import RoiData, SessionManager, SessionStore, StrategyResult


def test_create_session():
    mgr = SessionManager(max_sessions=10, timeout_seconds=1800)
    session = mgr.create_session()
    assert session.session_id
    assert session.roi_data == {}
    assert session.strategies == {}
    assert session.messages == []


def test_get_session():
    mgr = SessionManager(max_sessions=10, timeout_seconds=1800)
    session = mgr.create_session()
    retrieved = mgr.get_session(session.session_id)
    assert retrieved is session


def test_get_nonexistent_session():
    mgr = SessionManager(max_sessions=10, timeout_seconds=1800)
    assert mgr.get_session("nonexistent") is None


def test_max_sessions_enforced():
    mgr = SessionManager(max_sessions=2, timeout_seconds=1800)
    mgr.create_session()
    mgr.create_session()
    with pytest.raises(RuntimeError, match="Maximum.*sessions"):
        mgr.create_session()


def test_expired_sessions_evicted():
    mgr = SessionManager(max_sessions=10, timeout_seconds=0)
    session = mgr.create_session()
    sid = session.session_id
    time.sleep(0.05)
    mgr.evict_expired()
    assert mgr.get_session(sid) is None


def test_store_and_retrieve_roi_data():
    session = SessionStore(session_id="test")
    roi = RoiData(
        timeseries=np.ones((10, 3)),
        labels=["A", "B", "C"],
        qc={"tsnr": 50.0},
        source_path="/tmp/test.nii.gz",
    )
    rid = session.store_roi_data(roi)
    assert rid.startswith("roi_")
    assert session.get_roi_data(rid) is roi


def test_store_and_retrieve_strategy():
    session = SessionStore(session_id="test")
    strategy = StrategyResult(
        strategy_type="static",
        method="pearson",
        params={"method": "pearson"},
        artifacts=None,
        metrics_summary={"modularity": 0.5},
    )
    sid = session.store_strategy(strategy)
    assert sid.startswith("strategy_")
    assert session.get_strategy(sid) is strategy


def test_estimate_tokens():
    session = SessionStore(session_id="test")
    session.messages = [{"role": "user", "content": "hello world"}]
    tokens = session.estimate_tokens()
    assert tokens > 0
