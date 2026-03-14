"""Load and validate stream definitions from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from continual_benchmark.core.schemas import StreamDefinition
from continual_benchmark.utils.paths import STREAM_DEFINITIONS_DIR


def load_stream(stream_id: str, search_dir: Path | None = None) -> StreamDefinition:
    """Load a stream definition by ID.

    Searches for {stream_id}.yaml in the search directory
    (defaults to the built-in definitions directory).
    """
    search_dir = search_dir or STREAM_DEFINITIONS_DIR
    path = search_dir / f"{stream_id}.yaml"
    if not path.exists():
        available = list_available_streams(search_dir)
        raise FileNotFoundError(
            f"Stream definition '{stream_id}' not found at {path}. "
            f"Available streams: {', '.join(available) or 'none'}"
        )
    return load_stream_from_file(path)


def load_stream_from_file(path: Path) -> StreamDefinition:
    """Load a stream definition from a specific YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return StreamDefinition.model_validate(data)


def list_available_streams(search_dir: Path | None = None) -> list[str]:
    """List available stream definition IDs."""
    search_dir = search_dir or STREAM_DEFINITIONS_DIR
    if not search_dir.exists():
        return []
    return sorted(p.stem for p in search_dir.glob("*.yaml"))
