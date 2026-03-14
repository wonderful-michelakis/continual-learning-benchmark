"""Materialize stream definitions into Spec objects for generation."""

from __future__ import annotations

from continual_benchmark.core.schemas import Spec, StreamDefinition


def materialize_specs(stream_def: StreamDefinition) -> list[Spec]:
    """Convert a StreamDefinition into a list of Spec objects, one per stage.

    Each StageDefinition in the stream is converted to a Spec that the
    task family generator can use to produce instances.
    """
    specs: list[Spec] = []
    for stage_def in stream_def.stages:
        spec = Spec(
            spec_id=stage_def.spec_id,
            suite=stream_def.suite,
            family=stage_def.family,
            stage=stage_def.stage,
            description=stage_def.metadata.get("description", ""),
            difficulty=stage_def.difficulty,
            drift=stage_def.drift,
            split_sizes=stage_def.split_sizes,
            generator_config=stage_def.generator_config,
            cluster_id=stage_def.cluster_id,
            metadata=stage_def.metadata,
        )
        specs.append(spec)
    return specs
