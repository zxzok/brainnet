"""Minimal client for querying OpenNeuro datasets."""

from __future__ import annotations

import requests
from typing import Dict, List, Optional

GRAPHQL_URL = "https://openneuro.org/crn/graphql"


def list_datasets(
    search: str = "",
    page: int = 1,
    page_size: int = 10,
) -> Dict[str, object]:
    """List public datasets available on OpenNeuro.

    Parameters
    ----------
    search:
        Optional case-insensitive substring to filter dataset IDs or names.
    page:
        Page number (1-indexed).
    page_size:
        Number of results per page.

    Returns
    -------
    dict
        Dictionary with keys ``datasets`` (list of dicts with ``id`` and ``name``),
        ``total`` (number of matched datasets) and ``page``.
    """

    # Retrieve a reasonably large list and filter locally. This keeps the
    # implementation simple and avoids the need to track GraphQL cursors.
    first = max(page_size * page, 100)
    query = (
        "query($first:Int){ datasets(first:$first){ edges{ node{ id name } } } }"
    )
    resp = requests.post(GRAPHQL_URL, json={"query": query, "variables": {"first": first}})
    resp.raise_for_status()
    edges = resp.json()["data"]["datasets"]["edges"]
    items = [
        {"id": edge["node"]["id"], "name": edge["node"].get("name", "")}
        for edge in edges
    ]
    if search:
        s = search.lower()
        items = [
            ds for ds in items if s in ds["id"].lower() or s in ds["name"].lower()
        ]
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "datasets": items[start:end],
        "total": total,
        "page": page,
        "has_next": end < total,
    }
