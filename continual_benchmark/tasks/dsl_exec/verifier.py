"""DSL execution verifier: compares predictions against reference targets."""

from __future__ import annotations

from typing import Any

from continual_benchmark.tasks.dsl_exec.canonicalize import compare_dsl_outputs


def verify_dsl(prediction: str, target: str, metadata: dict[str, Any]) -> tuple[bool, float]:
    """Verify a DSL execution prediction.

    Args:
        prediction: The model's output.
        target: The reference answer (JSON with "result" key).
        metadata: Instance metadata (unused for DSL verification).

    Returns:
        (correct, score) where score is 1.0 if correct, 0.0 otherwise.
    """
    return compare_dsl_outputs(prediction, target)
