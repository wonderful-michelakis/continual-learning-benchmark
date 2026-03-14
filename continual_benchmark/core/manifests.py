"""Manifest generation and validation utilities."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from continual_benchmark.core.hashing import file_hash
from continual_benchmark.core.schemas import (
    Manifest,
    StageManifest,
    StreamDefinition,
)


def get_git_commit() -> str | None:
    """Try to get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def build_manifest(
    stream_def: StreamDefinition,
    global_seed: int,
    generator_version: str,
    output_dir: Path,
) -> Manifest:
    """Build a manifest for a generated suite.

    Scans the output directory for generated JSONL files and computes
    per-stage metadata including file hashes and instance counts.
    """
    stage_manifests: list[StageManifest] = []
    all_families: set[str] = set()

    for stage_def in stream_def.stages:
        stage_dir = output_dir / f"stage_{stage_def.stage:03d}"
        split_counts: dict[str, int] = {}
        split_hashes: dict[str, str] = {}

        for split_name in ["train", "dev", "public_test", "private_test"]:
            jsonl_path = stage_dir / f"{split_name}.jsonl"
            if jsonl_path.exists():
                with open(jsonl_path) as f:
                    count = sum(1 for line in f if line.strip())
                split_counts[split_name] = count
                split_hashes[split_name] = file_hash(jsonl_path)

        all_families.add(stage_def.family.value)

        stage_manifests.append(StageManifest(
            stage=stage_def.stage,
            spec_ids=[stage_def.spec_id],
            family=stage_def.family,
            split_counts=split_counts,
            split_hashes=split_hashes,
            difficulty=stage_def.difficulty,
            drift=stage_def.drift,
        ))

    return Manifest(
        suite=stream_def.suite,
        stream_id=stream_def.stream_id,
        generator_version=generator_version,
        global_seed=global_seed,
        stage_count=len(stream_def.stages),
        families=sorted(all_families),
        stages=stage_manifests,
        created_at=datetime.now(tz=timezone.utc),
        git_commit=get_git_commit(),
    )
