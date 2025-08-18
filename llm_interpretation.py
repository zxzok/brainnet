"""Utilities for generating natural language interpretations using LLMs.

This module provides a helper function :func:`summarize_analysis` which
uses either the OpenAI API or HuggingFace transformers to produce a short
text summary of analysis results. The function degrades gracefully if no
API key or model is available and will fall back to a simple
rule-based summary.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Optional dependencies
try:  # pragma: no cover - optional
    import openai  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    openai = None  # type: ignore

try:  # pragma: no cover - optional
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    pipeline = None  # type: ignore


def _load_api_key() -> str | None:
    """Load an API key from environment or ``~/.openai_api_key``."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    cfg = os.path.expanduser("~/.openai_api_key")
    if os.path.exists(cfg):  # pragma: no cover - filesystem dependent
        try:
            with open(cfg, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return None
    return None


def _basic_summary(conn_matrix: Any, graph_metrics: Any, dyn_model: Any) -> str:
    """Generate a very small textual summary without LLMs."""
    n_rois = getattr(conn_matrix, "matrix", None)
    n_rois = n_rois.shape[0] if hasattr(n_rois, "shape") else "unknown"
    mean_conn = getattr(conn_matrix, "matrix", None)
    mean_conn = float(mean_conn.mean()) if hasattr(mean_conn, "mean") else 0.0
    n_states = getattr(dyn_model, "n_states", None)
    glob = getattr(graph_metrics, "global_metrics", {})
    summary = (
        f"Connectivity across {n_rois} regions shows an average correlation of {mean_conn:.2f}. "
        f"The model identified {n_states} dynamic states. Global metrics: {glob}."
    )
    return summary


def summarize_analysis(conn_matrix: Any, graph_metrics: Any, dyn_model: Any) -> str:
    """Return a natural language interpretation of the analysis results.

    Parameters
    ----------
    conn_matrix, graph_metrics, dyn_model
        Objects produced by the analysis pipeline.  Only a few attributes are
        used to construct the prompt so any object with the expected
        attributes will work.

    Returns
    -------
    str
        A short textual interpretation.  If no LLM backend is available the
        summary will fall back to a deterministic rule‑based message.
    """
    prompt_data = {
        "n_regions": getattr(conn_matrix, "matrix", None).shape[0]
        if getattr(conn_matrix, "matrix", None) is not None
        else None,
        "mean_connectivity": float(getattr(conn_matrix, "matrix").mean())
        if getattr(conn_matrix, "matrix", None) is not None
        else None,
        "global_metrics": getattr(graph_metrics, "global_metrics", {}),
        "n_states": getattr(dyn_model, "n_states", None),
    }
    prompt = (
        "Provide a concise clinical interpretation for the following brain "
        "connectivity analysis results:\n" + json.dumps(prompt_data)
    )

    api_key = _load_api_key()
    if openai and api_key:  # pragma: no cover - network dependent
        try:
            try:
                client = openai.OpenAI(api_key=api_key)  # type: ignore[attr-defined]
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()
            except AttributeError:
                openai.api_key = api_key
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    if pipeline:  # pragma: no cover - heavy download, best effort
        try:
            summarizer = pipeline("text-generation")
            generated = summarizer(prompt, max_length=60, num_return_sequences=1)
            if generated:
                return generated[0]["generated_text"].strip()
        except Exception:
            pass

    # Fallback deterministic summary
    return _basic_summary(conn_matrix, graph_metrics, dyn_model)

