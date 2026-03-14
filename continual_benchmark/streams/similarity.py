"""Similarity metadata utilities for stream analysis.

Provides lightweight tools to summarize task similarity structure
using hand-authored metadata rather than learned embeddings.
"""

from __future__ import annotations

from continual_benchmark.core.schemas import Spec, StreamDefinition


def compute_similarity_matrix(specs: list[Spec]) -> list[list[float]]:
    """Compute a lightweight similarity matrix from spec metadata.

    Similarity is estimated from:
    - Same family: base similarity of 0.5
    - Same cluster: additional 0.3
    - Drift parent relationship: 0.8
    - Same family + same difficulty level: additional 0.1
    """
    n = len(specs)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = _pairwise_similarity(specs[i], specs[j])
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix


def _pairwise_similarity(a: Spec, b: Spec) -> float:
    """Estimate similarity between two specs."""
    sim = 0.0

    # Same family
    if a.family == b.family:
        sim += 0.5
        # Same difficulty level
        if a.difficulty.level == b.difficulty.level:
            sim += 0.1

    # Same cluster
    if a.cluster_id and b.cluster_id and a.cluster_id == b.cluster_id:
        sim += 0.3

    # Drift relationship
    if a.drift.drift_parent == b.spec_id or b.drift.drift_parent == a.spec_id:
        sim = max(sim, 0.8)

    return min(sim, 1.0)


def summarize_stream_structure(stream_def: StreamDefinition) -> dict:
    """Produce a summary of the stream's similarity/interference structure."""
    family_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    drift_stages: list[int] = []

    for stage_def in stream_def.stages:
        family = stage_def.family.value
        family_counts[family] = family_counts.get(family, 0) + 1

        if stage_def.cluster_id:
            cluster_counts[stage_def.cluster_id] = (
                cluster_counts.get(stage_def.cluster_id, 0) + 1
            )

        if stage_def.drift.drift_type.value != "none":
            drift_stages.append(stage_def.stage)

    return {
        "total_stages": len(stream_def.stages),
        "family_distribution": family_counts,
        "cluster_distribution": cluster_counts,
        "drift_stages": drift_stages,
        "drift_fraction": len(drift_stages) / max(len(stream_def.stages), 1),
    }
