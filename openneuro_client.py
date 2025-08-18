
"""Utility functions for interacting with the OpenNeuro platform."""

from __future__ import annotations

import logging
from pathlib import Path
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


def list_datasets(search_term: Optional[str] = None) -> List[Dict[str, Optional[str]]]:
    """Retrieve available datasets from OpenNeuro.

    Parameters
    ----------
    search_term : str | None
        Optional text to filter dataset names and descriptions.

    Returns
    -------
    list of dict
        Each dictionary contains ``id``, ``name`` and ``description`` keys.
    """
    datasets: List[Dict[str, Optional[str]]] = []
    cursor: Optional[str] = None
    query_all = (
        "query($after:String){"  # noqa: E501
        "datasets(first:50,after:$after,filterBy:{all:true})"
        "{pageInfo{endCursor hasNextPage}"
        " edges{node{id name latestSnapshot{description{Name}}}}}"
        "}"
    )
    query_search = (
        "query($q:String!,$after:String){"  # noqa: E501
        "search(q:$q,first:50,after:$after)"
        "{pageInfo{endCursor hasNextPage}"
        " edges{node{id name latestSnapshot{description{Name}}}}}"
        "}"
    )
    while True:
        variables: Dict[str, Any] = {"after": cursor}
        if search_term:
            variables["q"] = search_term
            data = _graphql_query(query_search, variables)["search"]
        else:
            data = _graphql_query(query_all, variables)["datasets"]
        for edge in data.get("edges", []):
            if not edge:
                continue
            node = edge.get("node")
            if not node:
                continue
            desc_obj = node.get("latestSnapshot", {}).get("description") or {}
            datasets.append(
                {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "description": desc_obj.get("Name"),
                }
            )
        page = data.get("pageInfo", {})
        if page.get("hasNextPage"):
            cursor = page.get("endCursor")
        else:
            break
    return datasets


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

