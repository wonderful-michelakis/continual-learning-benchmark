"""Pydantic schemas for benchmark data model.

Core types: Spec, Instance, Manifest, StreamDefinition, StageDefinition,
EvalResult, PerformanceMatrix, MetricReport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from continual_benchmark.core.constants import DriftType, FamilyName, Split


# ---------------------------------------------------------------------------
# Spec: describes a task distribution for one stage
# ---------------------------------------------------------------------------

class DifficultyConfig(BaseModel):
    """Family-agnostic difficulty parameters. Families interpret relevant fields."""

    level: int = Field(default=3, ge=1, le=10, description="Overall difficulty 1-10")
    # DSL-specific
    program_length: int | None = None
    nesting_depth: int | None = None
    operator_count: int | None = None
    # Transform-specific
    schema_depth: int | None = None
    rule_count: int | None = None
    # SQL-specific
    num_tables: int | None = None
    join_depth: int | None = None
    aggregation_complexity: int | None = None
    # API code-specific
    function_count: int | None = None
    test_count: int | None = None
    api_surface_size: int | None = None


class DriftConfig(BaseModel):
    """Configuration for concept drift at a stage."""

    drift_type: DriftType = DriftType.NONE
    drift_parent: str | None = Field(
        default=None,
        description="Spec ID of the parent stage this drifts from",
    )
    drift_description: str | None = None
    drift_intensity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How much drift: 0 = none, 1 = maximal",
    )


class SplitSizes(BaseModel):
    """Number of instances per split."""

    train: int = 1000
    dev: int = 200
    public_test: int = 200
    private_test: int = 200


class Spec(BaseModel):
    """A task specification for a single stage.

    Each spec defines a task distribution: which family, what rules/params,
    how to generate instances, and how to verify them.
    """

    spec_id: str = Field(description="Unique ID, e.g. 'dsl_exec:v1:stage03'")
    suite: str = Field(default="v1")
    family: FamilyName
    family_version: str = Field(default="1.0")
    stage: int = Field(ge=1)
    description: str = Field(default="")
    instruction: str = Field(default="", description="Task instruction shown to models")

    difficulty: DifficultyConfig = Field(default_factory=DifficultyConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    split_sizes: SplitSizes = Field(default_factory=SplitSizes)

    # Family-specific generator config (opaque to core, interpreted by family)
    generator_config: dict[str, Any] = Field(default_factory=dict)

    # Similarity metadata
    cluster_id: str | None = None
    expected_similarity: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Expected similarity to previous specs in same cluster",
    )
    expected_interference: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Expected interference with other active specs",
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Instance: a single benchmark example
# ---------------------------------------------------------------------------

class Instance(BaseModel):
    """A single generated benchmark instance."""

    uid: str = Field(description="Globally unique stable ID")
    suite: str
    stream_id: str
    family: FamilyName
    spec_id: str
    stage: int
    split: Split
    prompt: str
    target: str = Field(description="Reference answer, typically JSON string")
    seed: int

    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manifest: reproducibility metadata for a built suite
# ---------------------------------------------------------------------------

class StageManifest(BaseModel):
    """Per-stage metadata in a built suite."""

    stage: int
    spec_ids: list[str]
    family: FamilyName
    split_counts: dict[str, int] = Field(default_factory=dict)
    split_hashes: dict[str, str] = Field(default_factory=dict)
    difficulty: DifficultyConfig = Field(default_factory=DifficultyConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)


class Manifest(BaseModel):
    """Top-level manifest for a fully built benchmark suite."""

    suite: str
    stream_id: str
    generator_version: str
    global_seed: int
    stage_count: int
    families: list[str]
    stages: list[StageManifest]
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    git_commit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------

class StageDefinition(BaseModel):
    """A single stage in a stream definition."""

    stage: int = Field(ge=1)
    spec_id: str
    family: FamilyName
    difficulty: DifficultyConfig = Field(default_factory=DifficultyConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    split_sizes: SplitSizes = Field(default_factory=SplitSizes)
    generator_config: dict[str, Any] = Field(default_factory=dict)
    cluster_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamDefinition(BaseModel):
    """Full stream definition: an ordered sequence of stages."""

    stream_id: str
    suite: str = "v1"
    description: str = ""
    stages: list[StageDefinition]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

class ExampleResult(BaseModel):
    """Evaluation result for a single instance."""

    uid: str
    correct: bool
    score: float = Field(ge=0.0, le=1.0)
    predicted: str = ""
    expected: str = ""
    error: str | None = None


class StageEvalResult(BaseModel):
    """Aggregated evaluation for one stage's test set."""

    stage: int
    family: FamilyName
    spec_id: str
    split: Split
    accuracy: float
    total: int
    correct_count: int
    results: list[ExampleResult] = Field(default_factory=list)


class PerformanceMatrix(BaseModel):
    """The continual learning performance matrix.

    matrix[i][j] = score on task j after training through stage i.
    Rows = training stages, columns = evaluation tasks.
    """

    training_stages: list[int]
    eval_stages: list[int]
    matrix: list[list[float | None]]
    families: dict[int, str] = Field(
        default_factory=dict,
        description="Map from stage number to family name",
    )


class MetricReport(BaseModel):
    """Computed continual learning metrics."""

    average_accuracy: list[float] = Field(
        description="Average accuracy after each training stage",
    )
    forgetting: dict[int, float] = Field(
        default_factory=dict,
        description="Per-task forgetting: max_prev - current",
    )
    average_forgetting: float = 0.0
    backward_transfer: dict[int, float] = Field(
        default_factory=dict,
        description="Per-task backward transfer",
    )
    average_backward_transfer: float = 0.0
    forward_transfer: dict[int, float] = Field(
        default_factory=dict,
        description="Per-task forward transfer (if measurable)",
    )
    average_forward_transfer: float = 0.0

    # Breakdowns
    family_breakdown: dict[str, dict[str, float]] = Field(default_factory=dict)
    stream_summary: dict[str, Any] = Field(default_factory=dict)

    performance_matrix: PerformanceMatrix | None = None


# ---------------------------------------------------------------------------
# Prediction format
# ---------------------------------------------------------------------------

class Prediction(BaseModel):
    """A single model prediction."""

    uid: str
    prediction: str
    model_name: str | None = None
    stage_trained_through: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# QA report
# ---------------------------------------------------------------------------

class QACheck(BaseModel):
    """Result of a single QA check."""

    name: str
    passed: bool
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class QAReport(BaseModel):
    """Full QA report for a built suite."""

    suite: str
    stream_id: str
    total_checks: int
    passed: int
    failed: int
    warnings: int
    checks: list[QACheck] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
