"""Tests for the API code generation task family."""

import random

from continual_benchmark.core.constants import FamilyName
from continual_benchmark.core.schemas import DifficultyConfig, Spec
from continual_benchmark.tasks.api_code.sandbox import ExecutionResult, pre_scan_code
from continual_benchmark.tasks.api_code.test_builder import build_test_code


class TestSandbox:
    def test_pre_scan_clean_code(self):
        code = "def foo(x):\n    return x + 1"
        warnings = pre_scan_code(code)
        assert len(warnings) == 0

    def test_pre_scan_blocked_import(self):
        code = "import os\ndef foo(): os.system('ls')"
        warnings = pre_scan_code(code)
        assert len(warnings) > 0

    def test_pre_scan_blocked_pattern(self):
        code = "x = __import__('os')"
        warnings = pre_scan_code(code)
        assert len(warnings) > 0


class TestTestBuilder:
    def test_build_test_code(self):
        test_cases = [
            {"args": [[1, 2, 3], 2], "expected": [0.5, 0.5], "description": "test_0"},
        ]
        code = build_test_code("summarize", test_cases)
        assert "_total += 1" in code
        assert "summarize" in code


class TestGeneration:
    def test_generate_instance(self):
        from continual_benchmark.core.registry import get_family
        import continual_benchmark.tasks  # noqa: F401

        spec = Spec(
            spec_id="api_code:v1:stage10",
            family=FamilyName.API_CODE,
            stage=10,
            difficulty=DifficultyConfig(level=2, function_count=1, test_count=3),
            generator_config={"api_version": "1.0"},
        )
        family = get_family("api_code")
        rng = random.Random(42)
        inst = family.generate_instance(spec, "test", 0, "train", 42, rng)

        assert inst.prompt
        assert inst.target
        assert inst.metadata.get("_api_impl")
        assert inst.metadata.get("_test_code")
