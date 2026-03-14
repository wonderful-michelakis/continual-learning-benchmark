"""Structured transformation verifier."""

from __future__ import annotations

from typing import Any

from continual_benchmark.tasks.structured_transform.canonicalize import compare_json_outputs


def verify_transform(
    prediction: str, target: str, metadata: dict[str, Any]
) -> tuple[bool, float]:
    """Verify a structured transformation prediction.

    Args:
        prediction: The model's JSON output.
        target: The reference JSON output.
        metadata: Instance metadata.

    Returns:
        (correct, score) where score is 1.0 if correct, 0.0 otherwise.
    """
    return compare_json_outputs(prediction, target)
