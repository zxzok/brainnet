
"""Utility functions for interacting with the OpenNeuro platform."""

from __future__ import annotations

import logging
from pathlib import Path
import base64
import json
from typing import Any, Dict, List, Optional

import requests
from openneuro import download as _openneuro_download

LOGGER = logging.getLogger(__name__)
GRAPHQL_ENDPOINT = "https://openneuro.org/crn/graphql"


def _graphql_query(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a GraphQL query against the OpenNeuro API."""
    response = requests.post(
        GRAPHQL_ENDPOINT, json={"query": query, "variables": variables or {}}, timeout=30
    )
    response.raise_for_status()
    payload: Dict[str, Any] = response.json()
    if "errors" in payload:
        if not payload.get("data"):
            raise RuntimeError(f"GraphQL query failed: {payload['errors']}")
        LOGGER.warning(  # pragma: no cover - network edge case
            "GraphQL reported errors: %s", payload["errors"][0].get("message")
        )
    return payload.get("data", {})


def list_datasets(
    search: Optional[str] = None, page: int = 1, per_page: int = 50
) -> Dict[str, Any]:
    """Retrieve a single page of datasets from OpenNeuro.

    Parameters
    ----------
    search : str | None
        Optional text to filter dataset names and descriptions.
    page : int
        Page number (1-indexed).
    per_page : int
        Number of datasets to return per page.

    Returns
    -------
    dict
        Dictionary with keys ``datasets`` and ``has_next``. ``datasets`` is a list
        of dictionaries containing ``id``, ``name``, ``description`` and summary
        fields such as ``modalities``, ``tasks``, ``sessions``, ``subjects``,
        ``size`` and ``total_files``.
    """
    offset = max(page - 1, 0) * per_page
    after = None
    if offset:
        after = base64.b64encode(json.dumps({"offset": offset}).encode()).decode()

    query_all = (
        "query($first:Int!,$after:String){"
        "datasets(first:$first,after:$after,filterBy:{all:true})"
        "{pageInfo{endCursor hasNextPage}"
        " edges{node{id name latestSnapshot{description{Name} summary{modalities sessions subjects tasks size totalFiles}}}}}"
        "}"
    )
    query_search = (
        "query($q:String!,$first:Int!,$after:String){"
        "search(q:$q,first:$first,after:$after)"
        "{pageInfo{endCursor hasNextPage}"
        " edges{node{id name latestSnapshot{description{Name} summary{modalities sessions subjects tasks size totalFiles}}}}}"
        "}"
    )

    variables: Dict[str, Any] = {"first": per_page, "after": after}
    if search:
        variables["q"] = search
        data = _graphql_query(query_search, variables).get("search", {})
    else:
        data = _graphql_query(query_all, variables).get("datasets", {})

    datasets: List[Dict[str, Optional[str]]] = []
    for edge in data.get("edges", []):
        node = edge.get("node") if edge else None
        if not node:
            continue
        latest = node.get("latestSnapshot", {})
        desc_obj = latest.get("description") or {}
        summary = latest.get("summary") or {}
        sessions = summary.get("sessions")
        subjects = summary.get("subjects")
        datasets.append(
            {
                "id": node.get("id"),
                "name": node.get("name"),
                "description": desc_obj.get("Name"),
                "modalities": summary.get("modalities", []),
                "tasks": summary.get("tasks", []),
                "sessions": len(sessions) if isinstance(sessions, list) else sessions,
                "subjects": len(subjects) if isinstance(subjects, list) else subjects,
                "size": summary.get("size"),
                "total_files": summary.get("totalFiles"),
            }
        )

    page_info = data.get("pageInfo", {})
    return {"datasets": datasets, "has_next": bool(page_info.get("hasNextPage"))}


def get_dataset_metadata(dataset_id: str) -> Dict[str, Any]:
    """Fetch metadata for a single dataset.

    Parameters
    ----------
    dataset_id : str
        Accession number of the dataset (e.g., ``"ds000001"``).

    Returns
    -------
    dict
        Dictionary with keys ``id``, ``name``, ``tag``, ``description`` and ``summary``.
        ``summary`` is a nested dictionary with keys ``modalities``, ``sessions``,
        ``subjects``, ``tasks``, ``size`` and ``totalFiles``.
    """
    query = (
        "query($id:ID!){"  # noqa: E501
        "dataset(id:$id){id name latestSnapshot{tag description{Name} summary{modalities sessions subjects tasks size totalFiles}}}"
        "}"
    )
    result = _graphql_query(query, {"id": dataset_id})["dataset"]
    snapshot = result.get("latestSnapshot", {})
    description = snapshot.get("description", {})
    summary = snapshot.get("summary", {})
    return {
        "id": result.get("id"),
        "name": result.get("name"),
        "tag": snapshot.get("tag"),
        "description": description.get("Name"),
        "summary": summary,
    }


def download_dataset(dataset_id: str, dest_dir: Path | str) -> Path:
    """Download a dataset from OpenNeuro with caching and resume support.

    Parameters
    ----------
    dataset_id : str
        Dataset accession number.
    dest_dir : str or Path
        Directory in which to store the dataset. A subdirectory named after
        ``dataset_id`` will be created.

    Returns
    -------
    pathlib.Path
        Path to the local dataset directory.

    Notes
    -----
    Utilizes :func:`openneuro.download` which skips files that already exist
    and resumes interrupted downloads using HTTP range requests.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / dataset_id
    try:
        _openneuro_download(dataset=dataset_id, target_dir=target)
    except Exception as exc:  # pragma: no cover - network issues
        LOGGER.error("Failed to download dataset %s: %s", dataset_id, exc)
        raise
    return target

