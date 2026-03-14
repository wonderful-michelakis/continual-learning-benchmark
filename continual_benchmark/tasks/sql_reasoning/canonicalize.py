"""Canonicalization for SQL reasoning outputs.

Handles table result comparison with optional row-order insensitivity.
"""

from __future__ import annotations

import json
import math
from typing import Any


def canonicalize_table_output(output: str, order_matters: bool = False) -> str:
    """Canonicalize a SQL query result for comparison.

    Handles both scalar results and table results (list of dicts or list of values).
    """
    try:
        parsed = json.loads(output.strip())
    except json.JSONDecodeError:
        return output.strip()

    normalized = _normalize_result(parsed, order_matters)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_result(val: Any, order_matters: bool) -> Any:
    """Normalize a query result value."""
    if isinstance(val, dict):
        return {str(k): _normalize_scalar(v) for k, v in sorted(val.items())}
    if isinstance(val, list):
        normalized = [_normalize_item(item) for item in val]
        if not order_matters:
            # Sort rows by their canonical string representation
            normalized.sort(key=lambda x: json.dumps(x, sort_keys=True))
        return normalized
    return _normalize_scalar(val)


def _normalize_item(item: Any) -> Any:
    """Normalize a single result item (could be a row dict or a scalar)."""
    if isinstance(item, dict):
        return {str(k): _normalize_scalar(v) for k, v in sorted(item.items())}
    return _normalize_scalar(item)


def _normalize_scalar(val: Any) -> Any:
    """Normalize a scalar value."""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return val
        if val == int(val):
            return int(val)
        return round(val, 10)
    if val is None:
        return None
    return val


def compare_table_outputs(
    prediction: str,
    target: str,
    order_matters: bool = False,
) -> tuple[bool, float]:
    """Compare two SQL result outputs after canonicalization."""
    canon_pred = canonicalize_table_output(prediction, order_matters)
    canon_target = canonicalize_table_output(target, order_matters)
    correct = canon_pred == canon_target
    return correct, 1.0 if correct else 0.0
