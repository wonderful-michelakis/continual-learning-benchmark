"""Task families registry.

Imports all task family implementations to trigger registration.
"""

from continual_benchmark.tasks.families import (
    APICodeFamily,
    DSLExecFamily,
    SQLReasoningFamily,
    StructuredTransformFamily,
)

__all__ = [
    "DSLExecFamily",
    "StructuredTransformFamily",
    "SQLReasoningFamily",
    "APICodeFamily",
]
