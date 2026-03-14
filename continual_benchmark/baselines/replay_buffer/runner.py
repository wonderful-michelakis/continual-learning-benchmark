"""Replay buffer baseline runner.

Implements Track B (limited-memory replay): combines current stage data
with replay data from previous stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from continual_benchmark.baselines.replay_buffer.buffer import (
    FIFOBuffer,
    ReplayBuffer,
    ReservoirBuffer,
)
from continual_benchmark.baselines.sequential_ft.runner import DummyTrainer, Trainer
from continual_benchmark.core.io import read_instances, read_json, write_json
from continual_benchmark.utils.logging import logger


@dataclass
class ReplayBaselineConfig:
    """Configuration for replay buffer baseline."""

    suite_path: Path
    output_dir: Path
    buffer_type: str = "reservoir"  # "fifo", "reservoir", "token_budget"
    buffer_size: int = 500
    train_split: str = "train"
    eval_split: str = "public_test"
    seed: int = 42


def run_replay_baseline(
    config: ReplayBaselineConfig,
    trainer: Trainer | None = None,
) -> list[dict[str, Any]]:
    """Run the replay buffer baseline.

    For each stage:
    1. Load current stage data
    2. Combine with replay buffer
    3. Train on combined data
    4. Update replay buffer with current stage data
    5. Evaluate on all seen stages
    """
    if trainer is None:
        trainer = DummyTrainer()

    # Create buffer
    if config.buffer_type == "fifo":
        buffer: ReplayBuffer = FIFOBuffer(max_examples=config.buffer_size)
    else:
        buffer = ReservoirBuffer(max_examples=config.buffer_size, seed=config.seed)

    manifest = read_json(config.suite_path / "manifest.json")
    stage_count = manifest["stage_count"]

    results: list[dict[str, Any]] = []

    for current_stage in range(1, stage_count + 1):
        logger.info(f"=== Stage {current_stage}/{stage_count} (replay) ===")

        # Load current stage data
        train_path = config.suite_path / f"stage_{current_stage:03d}" / f"{config.train_split}.jsonl"
        if not train_path.exists():
            continue
        current_data = read_instances(train_path)

        # Combine with replay data
        replay_data = buffer.get_replay_data()
        combined = current_data + replay_data

        logger.info(
            f"Training on {len(current_data)} current + {len(replay_data)} replay "
            f"= {len(combined)} total"
        )

        # Train
        trainer.train(combined, current_stage)

        # Update buffer
        buffer.add_stage(current_data)

        # Evaluate on all seen stages
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
            "buffer_stats": buffer.stats(),
        })

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(results, config.output_dir / "replay_baseline_results.json")
    return results
