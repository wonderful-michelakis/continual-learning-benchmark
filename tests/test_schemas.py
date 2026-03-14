"""Tests for core schemas."""

import json

from continual_benchmark.core.constants import DriftType, FamilyName, Split
from continual_benchmark.core.schemas import (
    DifficultyConfig,
    DriftConfig,
    ExampleResult,
    Instance,
    Manifest,
    MetricReport,
    PerformanceMatrix,
    Prediction,
    QACheck,
    QAReport,
    Spec,
    SplitSizes,
    StageDefinition,
    StreamDefinition,
)


class TestSpec:
    def test_create_minimal(self):
        spec = Spec(
            spec_id="dsl_exec:v1:stage01",
            family=FamilyName.DSL_EXEC,
            stage=1,
        )
        assert spec.spec_id == "dsl_exec:v1:stage01"
        assert spec.family == FamilyName.DSL_EXEC
        assert spec.difficulty.level == 3  # default

    def test_create_with_drift(self):
        spec = Spec(
            spec_id="dsl_exec:v1:stage03",
            family=FamilyName.DSL_EXEC,
            stage=3,
            drift=DriftConfig(
                drift_type=DriftType.SEMANTIC,
                drift_parent="dsl_exec:v1:stage01",
                drift_intensity=0.5,
            ),
        )
        assert spec.drift.drift_type == DriftType.SEMANTIC
        assert spec.drift.drift_intensity == 0.5

    def test_serialization_roundtrip(self):
        spec = Spec(
            spec_id="test:v1:s1",
            family=FamilyName.SQL_REASONING,
            stage=1,
            difficulty=DifficultyConfig(level=5, num_tables=3),
        )
        data = json.loads(spec.model_dump_json())
        restored = Spec.model_validate(data)
        assert restored.spec_id == spec.spec_id
        assert restored.difficulty.num_tables == 3


class TestInstance:
    def test_create(self):
        inst = Instance(
            uid="test_uid",
            suite="v1",
            stream_id="v1_clustered",
            family=FamilyName.DSL_EXEC,
            spec_id="dsl_exec:v1:stage01",
            stage=1,
            split=Split.TRAIN,
            prompt="test prompt",
            target='{"result": 42}',
            seed=42,
        )
        assert inst.uid == "test_uid"
        assert inst.family == FamilyName.DSL_EXEC


class TestStreamDefinition:
    def test_create(self):
        stream = StreamDefinition(
            stream_id="test_stream",
            stages=[
                StageDefinition(
                    stage=1,
                    spec_id="dsl_exec:v1:stage01",
                    family=FamilyName.DSL_EXEC,
                ),
            ],
        )
        assert len(stream.stages) == 1
        assert stream.stages[0].family == FamilyName.DSL_EXEC


class TestPerformanceMatrix:
    def test_create(self):
        pm = PerformanceMatrix(
            training_stages=[1, 2],
            eval_stages=[1, 2],
            matrix=[[0.9, None], [0.85, 0.92]],
        )
        assert pm.matrix[0][0] == 0.9
        assert pm.matrix[0][1] is None
        assert pm.matrix[1][1] == 0.92
