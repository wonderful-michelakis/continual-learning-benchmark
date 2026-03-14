"""Sandboxed code execution for API code verification.

Uses subprocess with resource limits and AST pre-scanning
to safely execute untrusted code.
"""

from __future__ import annotations

import ast
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

# Patterns that are blocked in submitted code
BLOCKED_PATTERNS = {
    "__import__", "importlib", "exec(", "eval(",
    "open(", "subprocess", "os.system", "os.popen",
    "shutil", "socket", "requests", "urllib",
    "ctypes", "pickle", "marshal",
}

# Maximum execution time in seconds
DEFAULT_TIMEOUT = 10
# Maximum memory in bytes (256 MB)
DEFAULT_MEMORY_LIMIT = 256 * 1024 * 1024


class SandboxError(Exception):
    """Raised when sandbox security checks fail."""


class ExecutionResult:
    """Result of a sandboxed execution."""

    def __init__(
        self,
        success: bool,
        output: str = "",
        error: str = "",
        tests_passed: int = 0,
        tests_total: int = 0,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.tests_passed = tests_passed
        self.tests_total = tests_total


def pre_scan_code(code: str) -> list[str]:
    """AST-based pre-scan to reject obviously dangerous patterns.

    Returns a list of security warnings (empty if code passes).
    """
    warnings: list[str] = []

    # Check for blocked string patterns
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            warnings.append(f"Blocked pattern found: {pattern}")

    # AST analysis
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        warnings.append(f"Syntax error: {e}")
        return warnings

    for node in ast.walk(tree):
        # Block import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ""
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = node.names[0].name if node.names else ""
            if module not in ("math", "collections", "itertools", "functools", "typing"):
                warnings.append(f"Blocked import: {module}")

        # Block attribute access to dangerous builtins
        if isinstance(node, ast.Attribute):
            if node.attr in ("__subclasses__", "__bases__", "__globals__",
                             "__code__", "__builtins__"):
                warnings.append(f"Blocked attribute access: {node.attr}")

    return warnings


def execute_code_with_tests(
    user_code: str,
    api_code: str,
    test_code: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> ExecutionResult:
    """Execute user code against tests in a sandboxed subprocess.

    Args:
        user_code: The model's submitted code.
        api_code: The toy API implementation.
        test_code: Test code that tests the user's function.
        timeout: Maximum execution time in seconds.

    Returns:
        ExecutionResult with pass/fail details.
    """
    # Pre-scan
    warnings = pre_scan_code(user_code)
    if warnings:
        return ExecutionResult(
            success=False,
            error=f"Security check failed: {'; '.join(warnings)}",
        )

    # Build the test script by joining parts at top-level indentation
    parts = [
        "import json",
        "import sys",
        "",
        "# === API Implementation ===",
        api_code.strip(),
        "",
        "# === User Code ===",
        user_code.strip(),
        "",
        "# === Tests ===",
        "_passed = 0",
        "_total = 0",
        "_errors = []",
        "",
        test_code.strip(),
        "",
        'print(json.dumps({"passed": _passed, "total": _total, "errors": _errors}))',
    ]
    script = "\n".join(parts) + "\n"

    # Write to temp file and execute in subprocess
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "", "HOME": "/tmp"},
        )

        if result.returncode != 0:
            return ExecutionResult(
                success=False,
                error=result.stderr[:1000],
                output=result.stdout[:1000],
            )

        # Parse test results
        try:
            import json
            test_results = json.loads(result.stdout.strip().split("\n")[-1])
            passed = test_results.get("passed", 0)
            total = test_results.get("total", 0)
            return ExecutionResult(
                success=passed == total and total > 0,
                output=result.stdout,
                tests_passed=passed,
                tests_total=total,
            )
        except (json.JSONDecodeError, IndexError):
            return ExecutionResult(
                success=False,
                error="Could not parse test results",
                output=result.stdout[:500],
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            error=f"Execution timed out after {timeout}s",
        )
    finally:
        Path(script_path).unlink(missing_ok=True)
