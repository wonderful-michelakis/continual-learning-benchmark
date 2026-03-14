"""Task family registry.

Each task family implements the TaskFamily interface and registers itself
via the @register_family decorator. The registry is populated explicitly
in continual_benchmark/tasks/__init__.py.
"""

from __future__ import annotations

import abc
import random
from typing import Any

from continual_benchmark.core.schemas import Instance, Spec


class TaskFamily(abc.ABC):
    """Base class for all task families.

    A task family knows how to:
    1. Generate instances from a spec + seed
    2. Verify a prediction against a gold target
    3. Canonicalize outputs for comparison
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The family's registered name (must match FamilyName enum)."""

    @abc.abstractmethod
    def generate_instance(
        self,
        spec: Spec,
        stream_id: str,
        index: int,
        split: str,
        seed: int,
        rng: random.Random,
    ) -> Instance:
        """Generate a single benchmark instance.

        Args:
            spec: The stage spec defining the task distribution.
            stream_id: The stream this instance belongs to.
            index: Instance index within the split.
            split: Which split (train/dev/public_test/private_test).
            seed: Deterministic seed for this instance.
            rng: Seeded Random instance for generation.

        Returns:
            A fully populated Instance.
        """

    @abc.abstractmethod
    def verify(self, prediction: str, target: str, metadata: dict[str, Any]) -> tuple[bool, float]:
        """Verify a prediction against the gold target.

        Args:
            prediction: The model's output string.
            target: The reference answer string.
            metadata: Instance metadata (may contain family-specific info).

        Returns:
            (correct, score) where correct is bool and score is 0.0-1.0.
        """

    @abc.abstractmethod
    def canonicalize(self, output: str) -> str:
        """Canonicalize a model output for comparison.

        Handles harmless formatting variation while preserving semantic content.
        """

    def validate_instance(self, instance: Instance) -> list[str]:
        """Optional: run family-specific validation checks on a generated instance.

        Returns a list of error messages (empty if valid).
        """
        return []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_FAMILY_REGISTRY: dict[str, type[TaskFamily]] = {}


def register_family(name: str):
    """Decorator to register a TaskFamily implementation.

    Usage:
        @register_family("dsl_exec")
        class DSLExecFamily(TaskFamily):
            ...
    """
    def decorator(cls: type[TaskFamily]) -> type[TaskFamily]:
        if name in TASK_FAMILY_REGISTRY:
            raise ValueError(f"Task family '{name}' is already registered")
        TASK_FAMILY_REGISTRY[name] = cls
        return cls
    return decorator


def get_family(name: str) -> TaskFamily:
    """Instantiate and return a registered task family by name."""
    if name not in TASK_FAMILY_REGISTRY:
        available = ", ".join(sorted(TASK_FAMILY_REGISTRY.keys()))
        raise KeyError(
            f"Unknown task family '{name}'. Available families: {available}"
        )
    return TASK_FAMILY_REGISTRY[name]()


def list_families() -> list[str]:
    """Return sorted list of registered family names."""
    return sorted(TASK_FAMILY_REGISTRY.keys())
