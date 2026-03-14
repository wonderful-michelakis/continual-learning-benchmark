"""Canonicalization utilities for DSL execution outputs.

Handles JSON parsing, numeric normalization, and output comparison.
"""

from __future__ import annotations

import json
import math
from typing import Any


def canonicalize_dsl_output(output: str) -> str:
    """Canonicalize a DSL execution output for comparison.

    Parses JSON, normalizes numeric values, and re-serializes with sorted keys.
    """
    try:
        parsed = json.loads(output.strip())
    except json.JSONDecodeError:
        # Try to extract just a number
        try:
            val = float(output.strip())
            parsed = {"result": _normalize_number(val)}
        except ValueError:
            return output.strip()

    normalized = _normalize_value(parsed)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _normalize_number(val: float) -> int | float:
    """Normalize a numeric value: integers stay as int, floats are rounded."""
    if math.isnan(val) or math.isinf(val):
        return val
    if val == int(val) and not math.isinf(val):
        return int(val)
    return round(val, 10)


def _normalize_value(val: Any) -> Any:
    """Recursively normalize a JSON value."""
    if isinstance(val, dict):
        return {k: _normalize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, list):
        return [_normalize_value(v) for v in val]
    if isinstance(val, float):
        return _normalize_number(val)
    return val


def compare_dsl_outputs(prediction: str, target: str) -> tuple[bool, float]:
    """Compare a prediction against a target after canonicalization.

    Returns:
        (correct, score) — score is 1.0 if correct, 0.0 otherwise.
    """
    canon_pred = canonicalize_dsl_output(prediction)
    canon_target = canonicalize_dsl_output(target)
    correct = canon_pred == canon_target
    return correct, 1.0 if correct else 0.0
