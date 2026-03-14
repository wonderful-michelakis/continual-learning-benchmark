"""IO utilities for reading and writing JSONL, JSON, and manifest files."""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from continual_benchmark.core.schemas import Instance, Manifest, Prediction

T = TypeVar("T", bound=BaseModel)


def write_jsonl(instances: list[Instance], path: Path) -> None:
    """Write a list of instances to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(inst.model_dump_json() + "\n")


def read_jsonl(path: Path, model_class: type[T] = Instance) -> list[T]:
    """Read a JSONL file into a list of Pydantic model instances."""
    items: list[T] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(model_class.model_validate_json(line))
    return items


def read_instances(path: Path) -> list[Instance]:
    """Read instances from a JSONL file."""
    return read_jsonl(path, Instance)


def read_predictions(path: Path) -> list[Prediction]:
    """Read predictions from a JSONL file."""
    return read_jsonl(path, Prediction)


def write_json(data: Any, path: Path, indent: int = 2) -> None:
    """Write a JSON-serializable object or Pydantic model to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=indent)
    else:
        text = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Write a manifest to JSON."""
    write_json(manifest, path)


def read_manifest(path: Path) -> Manifest:
    """Read a manifest from JSON."""
    data = read_json(path)
    return Manifest.model_validate(data)
