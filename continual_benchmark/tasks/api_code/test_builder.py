"""Generate test code for API code verification tasks.

Creates deterministic test cases for verifying that user-submitted code
correctly uses the toy API.
"""

from __future__ import annotations

import random
from typing import Any


def build_test_code(
    function_name: str,
    test_cases: list[dict[str, Any]],
) -> str:
    """Generate test code that validates a user function.

    Each test case should have:
    - "args": list of arguments to pass
    - "expected": expected return value
    - "description": human-readable test description

    The generated code uses _passed/_total/_errors variables
    set up by the sandbox runner.
    """
    lines: list[str] = []

    for i, tc in enumerate(test_cases):
        args_str = ", ".join(repr(a) for a in tc["args"])
        expected = repr(tc["expected"])
        desc = tc.get("description", f"test_{i}")

        lines.append("_total += 1")
        lines.append("try:")
        lines.append(f"    _result = {function_name}({args_str})")
        lines.append(f"    if _result == {expected}:")
        lines.append("        _passed += 1")
        lines.append("    else:")
        lines.append(f'        _errors.append("{desc}: expected {expected}, got " + repr(_result))')
        lines.append("except Exception as _e:")
        lines.append(f'    _errors.append("{desc}: raised " + str(_e))')
        lines.append("")

    return "\n".join(lines)


def generate_test_cases(
    rng: random.Random,
    api_version: str,
    function_name: str,
    function_spec: dict[str, Any],
    count: int = 5,
) -> list[dict[str, Any]]:
    """Generate deterministic test cases for a function."""
    test_cases: list[dict[str, Any]] = []

    for i in range(count):
        tc = _generate_single_test(rng, api_version, function_name, function_spec, i)
        if tc:
            test_cases.append(tc)

    return test_cases


def _generate_single_test(
    rng: random.Random,
    api_version: str,
    function_name: str,
    function_spec: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    """Generate a single test case."""
    spec_type = function_spec.get("type", "summarize")

    generators = {
        "summarize": _generate_summarize_test,
        "process_batch": _generate_process_batch_test,
        "transform_data": _generate_transform_data_test,
        "smooth_and_rank": _generate_smooth_and_rank_test,
        "cumulative_normalize": _generate_cumulative_normalize_test,
        "paired_topk_smooth": _generate_paired_topk_smooth_test,
        "normalize_and_accumulate": _generate_normalize_and_accumulate_test,
        "group_and_count": _generate_group_and_count_test,
        "filter_group_sort": _generate_filter_group_sort_test,
        "blend_and_summarize": _generate_blend_and_summarize_test,
    }

    gen = generators.get(spec_type)
    if gen:
        return gen(rng, api_version, index)
    return None


# ---------------------------------------------------------------------------
# Helpers for computing expected values
# ---------------------------------------------------------------------------

def _normalize(xs):
    total = sum(xs)
    if total == 0:
        return [0.0] * len(xs)
    return [x / total for x in xs]


def _pairwise_sum(xs, doubled=False):
    result = []
    for i in range(0, len(xs) - 1, 2):
        val = xs[i] + xs[i + 1]
        result.append(2 * val if doubled else val)
    if len(xs) % 2 == 1:
        result.append(2 * xs[-1] if doubled else xs[-1])
    return result


def _top_k(xs, k):
    return sorted(xs, reverse=True)[:k]


def _running_mean(xs, window):
    result = []
    for i in range(len(xs)):
        start = max(0, i - window + 1)
        chunk = xs[start:i + 1]
        result.append(round(sum(chunk) / len(chunk), 6))
    return result


def _cumulative_sum(xs):
    result = []
    total = 0
    for x in xs:
        total += x
        result.append(total)
    return result


# ---------------------------------------------------------------------------
# Test generators per task type
# ---------------------------------------------------------------------------

def _generate_summarize_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    n = rng.randint(3, 8)
    xs = [rng.randint(1, 20) for _ in range(n)]
    k = rng.randint(1, max(1, n // 2))

    normalized = _normalize(xs)
    doubled = api_version == "2.0"
    paired = _pairwise_sum(normalized, doubled=doubled)
    result = [round(v, 6) for v in _top_k(paired, k)]

    return {
        "args": [xs, k],
        "expected": result,
        "description": f"summarize_test_{index}",
    }


def _generate_process_batch_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    n = rng.randint(2, 6)
    items = [{"value": rng.randint(1, 100), "label": f"item_{i}"} for i in range(n)]
    threshold = rng.randint(20, 80)

    filtered = [item for item in items if item["value"] >= threshold]
    filtered.sort(key=lambda x: x["value"], reverse=True)

    if api_version == "2.0":
        for i, item in enumerate(filtered):
            item["rank"] = i + 1

    return {
        "args": [items, threshold],
        "expected": filtered,
        "description": f"process_batch_test_{index}",
    }


def _generate_transform_data_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    keys = [f"key_{i}" for i in range(rng.randint(2, 5))]
    data = {k: rng.randint(0, 100) for k in keys}
    operation = rng.choice(["double", "negate", "square"])

    if operation == "double":
        expected = {k: v * 2 for k, v in data.items()}
    elif operation == "negate":
        expected = {k: -v for k, v in data.items()}
    else:
        expected = {k: v * v for k, v in data.items()}

    return {
        "args": [data, operation],
        "expected": expected,
        "description": f"transform_data_test_{index}",
    }


def _generate_smooth_and_rank_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    n = rng.randint(4, 8)
    xs = [rng.randint(1, 50) for _ in range(n)]
    window = rng.randint(2, min(4, n))
    k = rng.randint(1, max(1, n // 2))

    smoothed = _running_mean(xs, window)
    result = [round(v, 6) for v in _top_k(smoothed, k)]

    return {
        "args": [xs, window, k],
        "expected": result,
        "description": f"smooth_and_rank_test_{index}",
    }


def _generate_cumulative_normalize_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    n = rng.randint(3, 7)
    xs = [rng.randint(1, 20) for _ in range(n)]

    cumsum = _cumulative_sum(xs)
    result = _normalize(cumsum)

    return {
        "args": [xs],
        "expected": result,
        "description": f"cumulative_normalize_test_{index}",
    }


def _generate_paired_topk_smooth_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    n = rng.randint(4, 8)
    xs = [rng.randint(1, 30) for _ in range(n)]
    window = rng.randint(2, 3)
    doubled = api_version == "2.0"
    paired = _pairwise_sum(xs, doubled=doubled)
    k = rng.randint(1, max(1, len(paired) // 2))

    smoothed = _running_mean(paired, window)
    result = [round(v, 6) for v in _top_k(smoothed, k)]

    return {
        "args": [xs, k, window],
        "expected": result,
        "description": f"paired_topk_smooth_test_{index}",
    }


def _generate_normalize_and_accumulate_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    n = rng.randint(3, 7)
    xs = [rng.randint(1, 20) for _ in range(n)]

    normalized = _normalize(xs)
    cumsum = _cumulative_sum(normalized)
    result = [round(v, 6) for v in cumsum]

    return {
        "args": [xs],
        "expected": result,
        "description": f"normalize_and_accumulate_test_{index}",
    }


def _generate_group_and_count_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    categories = ["alpha", "beta", "gamma", "delta"]
    n = rng.randint(3, 8)
    items = [
        {"value": rng.randint(1, 100), "category": rng.choice(categories)}
        for _ in range(n)
    ]

    groups: dict[str, int] = {}
    for item in items:
        k = item["category"]
        groups[k] = groups.get(k, 0) + 1

    return {
        "args": [items, "category"],
        "expected": groups,
        "description": f"group_and_count_test_{index}",
    }


def _generate_filter_group_sort_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    categories = ["alpha", "beta", "gamma"]
    n = rng.randint(4, 8)
    items = [
        {"value": rng.randint(1, 100), "category": rng.choice(categories), "label": f"item_{i}"}
        for i in range(n)
    ]
    threshold = rng.randint(20, 60)

    filtered = [item for item in items if item["value"] >= threshold]
    groups: dict[str, list] = {}
    for item in filtered:
        k = item["category"]
        groups.setdefault(k, []).append(item)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x["value"], reverse=True)

    return {
        "args": [items, threshold, "category"],
        "expected": groups,
        "description": f"filter_group_sort_test_{index}",
    }


def _generate_blend_and_summarize_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    n = rng.randint(3, 6)
    xs = [rng.randint(1, 20) for _ in range(n)]
    ys = [rng.randint(1, 20) for _ in range(n)]
    k = rng.randint(1, max(1, n // 2))

    nx = _normalize(xs)
    ny = _normalize(ys)
    blended = [x + y for x, y in zip(nx, ny)]
    result = [round(v, 6) for v in _top_k(blended, k)]

    return {
        "args": [xs, ys, k],
        "expected": result,
        "description": f"blend_and_summarize_test_{index}",
    }
