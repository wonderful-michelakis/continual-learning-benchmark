"""Concrete TaskFamily implementations for all task families.

Each family registers itself with the task registry on import.
"""

from __future__ import annotations

import random
from typing import Any

from continual_benchmark.core.constants import FamilyName
from continual_benchmark.core.registry import TaskFamily, register_family
from continual_benchmark.core.schemas import Instance, Spec


@register_family(FamilyName.DSL_EXEC.value)
class DSLExecFamily(TaskFamily):
    """DSL Execution: execute programs in a small domain-specific language."""

    @property
    def name(self) -> str:
        return FamilyName.DSL_EXEC.value

    def generate_instance(
        self, spec: Spec, stream_id: str, index: int, split: str,
        seed: int, rng: random.Random,
    ) -> Instance:
        from continual_benchmark.tasks.dsl_exec.generator import generate_dsl_instance
        return generate_dsl_instance(spec, stream_id, index, split, seed, rng)

    def verify(self, prediction: str, target: str, metadata: dict[str, Any]) -> tuple[bool, float]:
        from continual_benchmark.tasks.dsl_exec.verifier import verify_dsl
        return verify_dsl(prediction, target, metadata)

    def canonicalize(self, output: str) -> str:
        from continual_benchmark.tasks.dsl_exec.canonicalize import canonicalize_dsl_output
        return canonicalize_dsl_output(output)


@register_family(FamilyName.STRUCTURED_TRANSFORM.value)
class StructuredTransformFamily(TaskFamily):
    """Structured Transformation: transform JSON according to stage-specific rules."""

    @property
    def name(self) -> str:
        return FamilyName.STRUCTURED_TRANSFORM.value

    def generate_instance(
        self, spec: Spec, stream_id: str, index: int, split: str,
        seed: int, rng: random.Random,
    ) -> Instance:
        from continual_benchmark.tasks.structured_transform.generator import (
            generate_transform_instance,
        )
        return generate_transform_instance(spec, stream_id, index, split, seed, rng)

    def verify(self, prediction: str, target: str, metadata: dict[str, Any]) -> tuple[bool, float]:
        from continual_benchmark.tasks.structured_transform.verifier import verify_transform
        return verify_transform(prediction, target, metadata)

    def canonicalize(self, output: str) -> str:
        from continual_benchmark.tasks.structured_transform.canonicalize import (
            canonicalize_json_output,
        )
        return canonicalize_json_output(output)


@register_family(FamilyName.SQL_REASONING.value)
class SQLReasoningFamily(TaskFamily):
    """SQL Reasoning: answer questions about synthetic relational databases."""

    @property
    def name(self) -> str:
        return FamilyName.SQL_REASONING.value

    def generate_instance(
        self, spec: Spec, stream_id: str, index: int, split: str,
        seed: int, rng: random.Random,
    ) -> Instance:
        from continual_benchmark.tasks.sql_reasoning.generator import generate_sql_instance
        return generate_sql_instance(spec, stream_id, index, split, seed, rng)

    def verify(self, prediction: str, target: str, metadata: dict[str, Any]) -> tuple[bool, float]:
        from continual_benchmark.tasks.sql_reasoning.verifier import verify_sql
        return verify_sql(prediction, target, metadata)

    def canonicalize(self, output: str) -> str:
        from continual_benchmark.tasks.sql_reasoning.canonicalize import canonicalize_table_output
        return canonicalize_table_output(output)


@register_family(FamilyName.API_CODE.value)
class APICodeFamily(TaskFamily):
    """API Code Generation: write functions using a stage-specific toy API."""

    @property
    def name(self) -> str:
        return FamilyName.API_CODE.value

    def generate_instance(
        self, spec: Spec, stream_id: str, index: int, split: str,
        seed: int, rng: random.Random,
    ) -> Instance:
        from continual_benchmark.tasks.api_code.generator import generate_api_code_instance
        return generate_api_code_instance(spec, stream_id, index, split, seed, rng)

    def verify(self, prediction: str, target: str, metadata: dict[str, Any]) -> tuple[bool, float]:
        from continual_benchmark.tasks.api_code.verifier import verify_api_code
        return verify_api_code(prediction, target, metadata)

    def canonicalize(self, output: str) -> str:
        from continual_benchmark.utils.text import extract_code_block
        return extract_code_block(output)
