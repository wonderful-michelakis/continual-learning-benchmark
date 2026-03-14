"""SQL reasoning verifier."""

from __future__ import annotations

from typing import Any

from continual_benchmark.tasks.sql_reasoning.canonicalize import compare_table_outputs


def verify_sql(prediction: str, target: str, metadata: dict[str, Any]) -> tuple[bool, float]:
    """Verify a SQL reasoning prediction.

    Compares the predicted result against the reference query result.
    Row order is not significant unless the query implies ordering.
    """
    order_matters = metadata.get("order_matters", False)
    return compare_table_outputs(prediction, target, order_matters=order_matters)
