from __future__ import annotations

import pytest

from brainnet.strategy_compare import compare_strategies, ComparisonResult
from brainnet.session_store import StrategyResult


def _make_static_strategy(modularity: float = 0.5, clustering: float = 0.3) -> StrategyResult:
    return StrategyResult(
        strategy_type="static",
        method="pearson",
        params={"method": "pearson"},
        artifacts=None,
        metrics_summary={
            "modularity": modularity,
            "clustering_coefficient": clustering,
            "small_world": 1.2,
            "n_connections": 50,
        },
    )


def _make_dynamic_strategy(
    method: str = "kmeans",
    n_states: int = 4,
    switching_rate: float = 0.15,
    occupancy_entropy: float = 1.3,
) -> StrategyResult:
    return StrategyResult(
        strategy_type="dynamic",
        method=method,
        params={"method": method, "n_states": n_states, "window_length": 30, "step": 10},
        artifacts=None,
        metrics_summary={
            "n_states": n_states,
            "switching_rate": switching_rate,
            "occupancy_entropy": occupancy_entropy,
            "temporal_complexity": 0.7,
        },
    )


def test_compare_two_dynamic_strategies():
    strategies = {
        "s1": _make_dynamic_strategy(n_states=4, switching_rate=0.15),
        "s2": _make_dynamic_strategy(n_states=3, switching_rate=0.10),
    }
    result = compare_strategies(strategies)
    assert isinstance(result, ComparisonResult)
    assert result.confidence in ("high", "moderate", "low")
    assert len(result.limitations) >= 1
    assert result.comparison_table  # not empty


def test_compare_static_vs_dynamic():
    strategies = {
        "s1": _make_static_strategy(),
        "s2": _make_dynamic_strategy(),
    }
    result = compare_strategies(strategies)
    assert result.confidence in ("high", "moderate", "low")
    assert len(result.ranking_axes) == 3
    assert "information_richness" in result.ranking_axes
    assert "stability" in result.ranking_axes
    assert "interpretability" in result.ranking_axes


def test_compare_requires_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        compare_strategies({"s1": _make_static_strategy()})


def test_limitations_always_populated():
    strategies = {
        "s1": _make_static_strategy(),
        "s2": _make_dynamic_strategy(),
    }
    result = compare_strategies(strategies)
    assert any("single subject" in lim.lower() for lim in result.limitations)
