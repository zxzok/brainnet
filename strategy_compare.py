"""Cross-strategy metric comparison and ranking for BrainNet orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brainnet.session_store import StrategyResult


@dataclass
class ComparisonResult:
    """Output of a multi-strategy comparison."""

    comparison_table: dict[str, dict[str, Any]]
    metric_differences: dict[str, Any]
    ranking_axes: dict[str, dict[str, float]]  # axis -> {strategy_id: score}
    recommended_strategy: str
    confidence: str  # "high", "moderate", "low"
    reasoning_hints: list[str]
    limitations: list[str]


def _score_information_richness(strategy: StrategyResult) -> float:
    """More distinct non-zero metrics = higher score."""
    return float(len([v for v in strategy.metrics_summary.values() if v]))


def _score_stability(strategy: StrategyResult) -> float:
    """Fewer parameters = more stable (heuristic for prototype)."""
    return 1.0 / max(len(strategy.params), 1)


def _score_interpretability(strategy: StrategyResult) -> float:
    """Static methods are simpler to interpret than dynamic."""
    base = 1.0 if strategy.strategy_type == "static" else 0.5
    return base / max(len(strategy.params), 1)


def _build_comparison_table(
    strategies: dict[str, StrategyResult],
) -> dict[str, dict[str, Any]]:
    """Build a per-strategy table of all available metrics."""
    return {sid: dict(s.metrics_summary) for sid, s in strategies.items()}


def _compute_metric_differences(
    strategies: dict[str, StrategyResult],
) -> dict[str, Any]:
    """Compute pairwise differences for shared numeric metrics."""
    all_keys: set[str] = set()
    for s in strategies.values():
        all_keys.update(s.metrics_summary.keys())

    diffs: dict[str, Any] = {}
    ids = list(strategies.keys())
    for key in sorted(all_keys):
        vals = {}
        for sid in ids:
            v = strategies[sid].metrics_summary.get(key)
            if isinstance(v, (int, float)):
                vals[sid] = v
        if len(vals) >= 2:
            values = list(vals.values())
            diffs[key] = {
                "values": vals,
                "range": max(values) - min(values),
            }
    return diffs


def _determine_confidence(ranking_axes: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Return (confidence, recommended_strategy)."""
    # Count how many axes each strategy wins
    wins: dict[str, int] = {}
    for axis_scores in ranking_axes.values():
        if not axis_scores:
            continue
        best = max(axis_scores, key=lambda k: axis_scores[k])
        wins[best] = wins.get(best, 0) + 1

    if not wins:
        return "low", ""

    best_strategy = max(wins, key=lambda k: wins[k])
    best_count = wins[best_strategy]

    if best_count >= 3:
        confidence = "high"
    elif best_count >= 2:
        confidence = "moderate"
    else:
        confidence = "low"

    return confidence, best_strategy


def compare_strategies(strategies: dict[str, StrategyResult]) -> ComparisonResult:
    """Compare multiple analysis strategies and produce a ranking."""
    if len(strategies) < 2:
        raise ValueError("compare_strategies requires at least 2 strategies")

    comparison_table = _build_comparison_table(strategies)
    metric_differences = _compute_metric_differences(strategies)

    ranking_axes = {
        "information_richness": {
            sid: _score_information_richness(s) for sid, s in strategies.items()
        },
        "stability": {sid: _score_stability(s) for sid, s in strategies.items()},
        "interpretability": {sid: _score_interpretability(s) for sid, s in strategies.items()},
    }

    confidence, recommended = _determine_confidence(ranking_axes)

    reasoning_hints = []
    for axis, scores in ranking_axes.items():
        best = max(scores, key=lambda k: scores[k])
        reasoning_hints.append(f"{axis}: {best} scores highest")

    limitations = [
        "Comparison based on single subject — results may not generalize.",
        "Ranking heuristics are simplified for prototype; not a formal statistical test.",
        "Parameter sensitivity not assessed — different parameters may change ranking.",
    ]

    types_present = {s.strategy_type for s in strategies.values()}
    if len(types_present) > 1:
        limitations.append(
            "Cross-type comparison (static vs dynamic) uses proxy metrics "
            "that may not capture all differences."
        )

    return ComparisonResult(
        comparison_table=comparison_table,
        metric_differences=metric_differences,
        ranking_axes=ranking_axes,
        recommended_strategy=recommended,
        confidence=confidence,
        reasoning_hints=reasoning_hints,
        limitations=limitations,
    )
