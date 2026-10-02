"""Canonical TRACE data paths."""

from __future__ import annotations

import os
from pathlib import Path


def configured_path(name: str, default: str | Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default)


def dataset_root(default: str | Path = "data/datasets") -> Path:
    return configured_path("TRACE_DATASET_ROOT", default)


def embedding_root(default: str | Path = "data/embeddings") -> Path:
    return configured_path("TRACE_EMBEDDING_ROOT", default)


def concept_root(default: str | Path | None = None) -> Path | None:
    value = os.environ.get("TRACE_CONCEPT_ROOT")
    if value:
        return Path(value).expanduser()
    return Path(default) if default is not None else None
