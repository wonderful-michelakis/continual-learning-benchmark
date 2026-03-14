"""Canonicalization for structured transform outputs.

Recursively normalizes JSON structures for comparison:
sorted keys, normalized numbers, stable serialization.
"""

from __future__ import annotations

import json
import math
from typing import Any


def canonicalize_json_output(output: str) -> str:
    """Canonicalize a JSON output string.

    Parses the JSON, recursively sorts keys, normalizes numbers,
    and re-serializes to a canonical form.
    """
    try:
        parsed = json.loads(output.strip())
    except json.JSONDecodeError:
        return output.strip()

    normalized = _normalize_value(parsed)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_value(val: Any) -> Any:
    """Recursively normalize a value for canonical comparison."""
    if isinstance(val, dict):
        return {str(k): _normalize_value(v) for k, v in sorted(val.items(), key=lambda x: str(x[0]))}
    if isinstance(val, list):
        return [_normalize_value(v) for v in val]
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return val
        if val == int(val):
            return int(val)
        return round(val, 10)
    return val


def compare_json_outputs(prediction: str, target: str) -> tuple[bool, float]:
    """Compare two JSON outputs after canonicalization.

    Returns:
        (correct, score) where score is 1.0 if correct, 0.0 otherwise.
    """
    canon_pred = canonicalize_json_output(prediction)
    canon_target = canonicalize_json_output(target)
    correct = canon_pred == canon_target
    return correct, 1.0 if correct else 0.0
