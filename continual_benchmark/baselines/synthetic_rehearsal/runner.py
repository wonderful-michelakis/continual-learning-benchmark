"""Synthetic rehearsal baseline scaffold.

A framework for pseudo-rehearsal: generate synthetic examples from task
specs or prior summaries, combine with current stage training data.

This is a credible placeholder — the actual generation of synthetic examples
would typically involve a model. The framework makes it easy to plug in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from continual_benchmark.baselines.sequential_ft.runner import DummyTrainer, Trainer
from continual_benchmark.core.io import read_instances, read_json, write_json
from continual_benchmark.core.schemas import Instance
from continual_benchmark.utils.logging import logger


class SyntheticGenerator(Protocol):
    """Protocol for synthetic example generation.

    Implement this to plug in your synthetic rehearsal strategy.
    """

    def generate_synthetic(
        self,
        spec_summary: dict[str, Any],
        count: int,
    ) -> list[Instance]:
        """Generate synthetic examples for a past stage.

        Args:
            spec_summary: Summary of the past stage spec (from stage summary.json).
            count: Number of synthetic examples to generate.

        Returns:
            List of synthetic instances.
        """
        ...


class DummySyntheticGenerator:
    """No-op synthetic generator for testing."""

    def generate_synthetic(
        self, spec_summary: dict[str, Any], count: int,
    ) -> list[Instance]:
        logger.info(
            f"[DummySyntheticGenerator] Would generate {count} synthetic "
            f"examples for {spec_summary.get('spec_id', 'unknown')}"
        )
        return []


@dataclass
class SyntheticRehearsalConfig:
    """Configuration for synthetic rehearsal baseline."""

    suite_path: Path
    output_dir: Path
    synthetic_per_stage: int = 50
    train_split: str = "train"
    eval_split: str = "public_test"


def run_synthetic_rehearsal(
    config: SyntheticRehearsalConfig,
    trainer: Trainer | None = None,
    generator: SyntheticGenerator | None = None,
) -> list[dict[str, Any]]:
    """Run the synthetic rehearsal baseline.

    For each stage:
    1. Load current stage data
    2. Generate synthetic examples for all past stages
    3. Combine current + synthetic data (tracked separately)
    4. Train
    5. Evaluate on all seen stages
    """
    if trainer is None:
        trainer = DummyTrainer()
    if generator is None:
        generator = DummySyntheticGenerator()

    manifest = read_json(config.suite_path / "manifest.json")
    stage_count = manifest["stage_count"]

    past_summaries: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for current_stage in range(1, stage_count + 1):
        logger.info(f"=== Stage {current_stage}/{stage_count} (synthetic rehearsal) ===")

        # Load current stage data
        train_path = config.suite_path / f"stage_{current_stage:03d}" / f"{config.train_split}.jsonl"
        if not train_path.exists():
            continue
        current_data = read_instances(train_path)

        # Generate synthetic examples for past stages
        synthetic_data: list[Instance] = []
        for summary in past_summaries:
            synth = generator.generate_synthetic(summary, config.synthetic_per_stage)
            synthetic_data.extend(synth)

        combined = current_data + synthetic_data
        logger.info(
            f"Training on {len(current_data)} real + {len(synthetic_data)} synthetic "
            f"= {len(combined)} total"
        )

        # Train
        trainer.train(combined, current_stage)

        # Record this stage's summary for future synthetic generation
        summary_path = config.suite_path / f"stage_{current_stage:03d}" / "summary.json"
        if summary_path.exists():
            past_summaries.append(read_json(summary_path))

        # Evaluate
        stage_results: dict[int, float] = {}
        for eval_stage in range(1, current_stage + 1):
            eval_path = (
                config.suite_path / f"stage_{eval_stage:03d}" / f"{config.eval_split}.jsonl"
            )
            if not eval_path.exists():
                continue
            eval_instances = read_instances(eval_path)
            predictions = trainer.predict(eval_instances)
            correct = sum(
                1 for inst, pred in zip(eval_instances, predictions)
                if pred.get("prediction", "") == inst.target
            )
            stage_results[eval_stage] = correct / max(len(eval_instances), 1)

        avg = sum(stage_results.values()) / max(len(stage_results), 1) if stage_results else 0.0
        results.append({
            "training_stage": current_stage,
            "eval_results": stage_results,
            "average_accuracy": avg,
            "synthetic_count": len(synthetic_data),
            "real_count": len(current_data),
        })

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(results, config.output_dir / "synthetic_rehearsal_results.json")
    return results
