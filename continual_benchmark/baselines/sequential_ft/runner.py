"""Sequential fine-tuning baseline.

A canonical stage-by-stage training skeleton. The actual training logic
is stubbed — users plug in their own trainer (HuggingFace, LoRA, etc.).

This baseline implements Track A (no-data replay): after each stage,
only the current stage's data is used for training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from continual_benchmark.core.io import read_instances, read_json, write_json
from continual_benchmark.core.schemas import Instance, StageEvalResult
from continual_benchmark.eval.score import score_predictions
from continual_benchmark.utils.logging import logger


class Trainer(Protocol):
    """Protocol for a model trainer.

    Implement this to plug in your training logic.
    """

    def train(self, instances: list[Instance], stage: int) -> None:
        """Train on the given instances for a stage."""
        ...

    def predict(self, instances: list[Instance]) -> list[dict[str, str]]:
        """Generate predictions for instances.

        Returns list of dicts with at least {"uid": ..., "prediction": ...}.
        """
        ...

    def save_checkpoint(self, path: Path, stage: int) -> None:
        """Save a model checkpoint."""
        ...

    def load_checkpoint(self, path: Path) -> None:
        """Load a model checkpoint."""
        ...


class DummyTrainer:
    """A no-op trainer for testing the pipeline."""

    def train(self, instances: list[Instance], stage: int) -> None:
        logger.info(f"[DummyTrainer] Would train on {len(instances)} instances for stage {stage}")

    def predict(self, instances: list[Instance]) -> list[dict[str, str]]:
        return [{"uid": inst.uid, "prediction": inst.target} for inst in instances]

    def save_checkpoint(self, path: Path, stage: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / f"checkpoint_stage_{stage:03d}.json").write_text("{}")

    def load_checkpoint(self, path: Path) -> None:
        pass


@dataclass
class SequentialFTConfig:
    """Configuration for sequential fine-tuning baseline."""

    suite_path: Path
    output_dir: Path
    train_split: str = "train"
    eval_split: str = "public_test"
    checkpoint_dir: Path | None = None


def run_sequential_ft(
    config: SequentialFTConfig,
    trainer: Trainer | None = None,
) -> list[dict[str, Any]]:
    """Run the sequential fine-tuning baseline.

    For each stage:
    1. Load training data
    2. Train on current stage
    3. Evaluate on all seen stages
    4. Save checkpoint
    5. Generate report

    Args:
        config: Baseline configuration.
        trainer: A Trainer implementation. Defaults to DummyTrainer.

    Returns:
        Per-stage evaluation summaries.
    """
    if trainer is None:
        trainer = DummyTrainer()

    manifest = read_json(config.suite_path / "manifest.json")
    stage_count = manifest["stage_count"]
    checkpoint_dir = config.checkpoint_dir or config.output_dir / "checkpoints"

    results_by_stage: list[dict[str, Any]] = []

    for current_stage in range(1, stage_count + 1):
        logger.info(f"=== Stage {current_stage}/{stage_count} ===")

        # 1. Load training data
        train_path = config.suite_path / f"stage_{current_stage:03d}" / f"{config.train_split}.jsonl"
        if not train_path.exists():
            logger.warning(f"No training data for stage {current_stage}")
            continue
        train_instances = read_instances(train_path)

        # 2. Train
        logger.info(f"Training on {len(train_instances)} instances")
        trainer.train(train_instances, current_stage)

        # 3. Save checkpoint
        trainer.save_checkpoint(checkpoint_dir, current_stage)

        # 4. Evaluate on all seen stages
        stage_results: dict[int, float] = {}
        for eval_stage in range(1, current_stage + 1):
            eval_path = (
                config.suite_path / f"stage_{eval_stage:03d}" / f"{config.eval_split}.jsonl"
            )
            if not eval_path.exists():
                continue

            eval_instances = read_instances(eval_path)
            predictions = trainer.predict(eval_instances)

            # Score
            correct = sum(
                1 for inst, pred in zip(eval_instances, predictions)
                if pred.get("prediction", "") == inst.target
            )
            accuracy = correct / max(len(eval_instances), 1)
            stage_results[eval_stage] = accuracy

        # 5. Record results
        avg_accuracy = (
            sum(stage_results.values()) / max(len(stage_results), 1)
            if stage_results else 0.0
        )
        results_by_stage.append({
            "training_stage": current_stage,
            "eval_results": stage_results,
            "average_accuracy": avg_accuracy,
        })

        logger.info(
            f"Stage {current_stage}: avg accuracy = {avg_accuracy:.4f} "
            f"over {len(stage_results)} tasks"
        )

    # Write results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_by_stage, config.output_dir / "sequential_ft_results.json")

    return results_by_stage
