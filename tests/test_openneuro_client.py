"""Tests for :mod:`openneuro_client` utilities."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict

import pytest

import openneuro_client


def _fake_response(has_next: bool) -> Dict[str, Any]:
    """Construct a fake GraphQL response."""
    return {
        "datasets": {
            "pageInfo": {"hasNextPage": has_next},
            "edges": [
                {
                    "node": {
                        "id": "ds1",
                        "name": "DS1",
                        "latestSnapshot": {"description": {"Name": "First"}},
                    }
                },
                {
                    "node": {
                        "id": "ds2",
                        "name": "DS2",
                        "latestSnapshot": {"description": {"Name": "Second"}},
                    }
                },
            ],
        }
    }


def test_list_datasets_pagination(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_query(query: str, variables: Dict[str, Any]):
        captured.update({"query": query, "variables": variables})
        return _fake_response(True)

    monkeypatch.setattr(openneuro_client, "_graphql_query", fake_query)

    result = openneuro_client.list_datasets(page=2, per_page=2)

    assert result["has_next"] is True
    assert len(result["datasets"]) == 2
    expected_after = base64.b64encode(json.dumps({"offset": 2}).encode()).decode()
    assert captured["variables"] == {"first": 2, "after": expected_after}


def test_list_datasets_search(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_query(query: str, variables: Dict[str, Any]):
        captured.update({"query": query, "variables": variables})
        return {
            "search": {
                "pageInfo": {"hasNextPage": False},
                "edges": [
                    {
                        "node": {
                            "id": "ds-search",
                            "name": "Search",
                            "latestSnapshot": {"description": {"Name": None}},
                        }
                    }
                ],
            }
        }

    monkeypatch.setattr(openneuro_client, "_graphql_query", fake_query)

    result = openneuro_client.list_datasets(search="brain", page=1, per_page=1)

    assert result == {
        "datasets": [
            {"id": "ds-search", "name": "Search", "description": None}
        ],
        "has_next": False,
    }
    assert captured["variables"] == {"first": 1, "after": None, "q": "brain"}

