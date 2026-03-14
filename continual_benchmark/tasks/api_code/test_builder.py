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

        lines.append(f"_total += 1")
        lines.append(f"try:")
        lines.append(f"    _result = {function_name}({args_str})")
        lines.append(f"    if _result == {expected}:")
        lines.append(f"        _passed += 1")
        lines.append(f"    else:")
        lines.append(f'        _errors.append("{desc}: expected {expected}, got " + repr(_result))')
        lines.append(f"except Exception as _e:")
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
    """Generate deterministic test cases for a function.

    Args:
        rng: Seeded random generator.
        api_version: API version (affects expected behavior).
        function_name: Name of the function to test.
        function_spec: Specification of expected behavior.
        count: Number of test cases to generate.

    Returns:
        List of test case dicts with args, expected, description.
    """
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

    if spec_type == "summarize":
        return _generate_summarize_test(rng, api_version, index)
    elif spec_type == "process_batch":
        return _generate_process_batch_test(rng, api_version, index)
    elif spec_type == "transform_data":
        return _generate_transform_data_test(rng, api_version, index)

    return None


def _generate_summarize_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    """Generate test for the 'summarize' function."""
    n = rng.randint(3, 8)
    xs = [rng.randint(1, 20) for _ in range(n)]
    k = rng.randint(1, max(1, n // 2))

    # Compute expected result based on API version
    # normalize -> pairwise_sum -> top_k
    total = sum(xs)
    if total == 0:
        normalized = [0.0] * len(xs)
    else:
        normalized = [x / total for x in xs]

    # pairwise_sum
    paired = []
    for i in range(0, len(normalized) - 1, 2):
        paired.append(normalized[i] + normalized[i + 1])
    if len(normalized) % 2 == 1:
        paired.append(normalized[-1])

    # Apply API version drift
    if api_version == "2.0":
        # In v2.0, pairwise_sum also multiplies by 2
        paired = [p * 2 for p in paired]

    # top_k
    result = sorted(paired, reverse=True)[:k]
    result = [round(v, 6) for v in result]

    return {
        "args": [xs, k],
        "expected": result,
        "description": f"summarize_test_{index}",
    }


def _generate_process_batch_test(
    rng: random.Random, api_version: str, index: int
) -> dict[str, Any]:
    """Generate test for the 'process_batch' function."""
    n = rng.randint(2, 6)
    items = [{"value": rng.randint(1, 100), "label": f"item_{i}"} for i in range(n)]
    threshold = rng.randint(20, 80)

    # Filter items above threshold, then sort by value descending
    filtered = [item for item in items if item["value"] >= threshold]
    filtered.sort(key=lambda x: x["value"], reverse=True)

    if api_version == "2.0":
        # In v2.0, also add a "rank" field
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
    """Generate test for the 'transform_data' function."""
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
