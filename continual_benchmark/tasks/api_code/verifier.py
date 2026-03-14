"""API code generation verifier.

Extracts code from the prediction, runs it against tests in a sandbox,
and scores based on tests passed.

For self-verification (QA checks where prediction == target), the target
is JSON metadata about expected test results — we compare directly.
"""

from __future__ import annotations

import json
from typing import Any

from continual_benchmark.tasks.api_code.sandbox import execute_code_with_tests
from continual_benchmark.utils.text import extract_code_block


def verify_api_code(
    prediction: str,
    target: str,
    metadata: dict[str, Any],
) -> tuple[bool, float]:
    """Verify an API code generation prediction.

    If prediction looks like JSON metadata (same format as target),
    compares directly. Otherwise, extracts code and runs it in sandbox.
    """
    # Self-verification: when prediction == target (QA check)
    if prediction.strip() == target.strip():
        try:
            data = json.loads(target)
            if "all_passed" in data:
                return data["all_passed"], (
                    data["tests_passed"] / max(data["tests_total"], 1)
                    if data["tests_total"] > 0 else 0.0
                )
        except (json.JSONDecodeError, KeyError):
            pass

    api_impl = metadata.get("_api_impl", "")
    test_code = metadata.get("_test_code", "")

    if not api_impl or not test_code:
        return False, 0.0

    # Extract code from prediction (handle markdown fences)
    user_code = extract_code_block(prediction)

    if not user_code.strip():
        return False, 0.0

    # Run in sandbox
    result = execute_code_with_tests(
        user_code=user_code,
        api_code=api_impl,
        test_code=test_code,
        timeout=10,
    )

    if result.tests_total == 0:
        return False, 0.0

    score = result.tests_passed / result.tests_total
    correct = result.success  # all tests passed
    return correct, score
