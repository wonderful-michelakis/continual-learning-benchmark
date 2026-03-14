"""Stable hashing utilities for manifests, deduplication, and reproducibility checks."""

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_hash(obj: Any) -> str:
    """Compute a deterministic SHA-256 hash of a JSON-serializable object.

    Keys are sorted recursively to ensure stability regardless of insertion order.
    """
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(text: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instance_uid(
    suite: str,
    family: str,
    stage: int,
    split: str,
    index: int,
    seed: int,
) -> str:
    """Generate a unique, stable instance ID.

    Format: {suite}_{family}_stage{stage:02d}_{split}_{index:06d}_{seed_hash}

    The seed hash suffix ensures uniqueness even if indices collide across rebuilds
    with different configs.
    """
    seed_suffix = hashlib.sha256(str(seed).encode()).hexdigest()[:8]
    return f"{suite}_{family}_stage{stage:02d}_{split}_{index:06d}_{seed_suffix}"
