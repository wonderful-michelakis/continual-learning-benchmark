"""Score predictions against gold data.

Loads gold instances, matches predictions by UID, routes each example
to the correct verifier, and computes per-example and per-stage scores.
"""

from __future__ import annotations

from pathlib import Path

from continual_benchmark.core.io import read_instances, read_json, read_predictions
from continual_benchmark.core.registry import get_family
from continual_benchmark.core.schemas import (
    ExampleResult,
    Prediction,
    StageEvalResult,
)
from continual_benchmark.utils.logging import logger
from continual_benchmark.utils.paths import split_path

import continual_benchmark.tasks  # noqa: F401


def score_predictions(
    gold_dir: Path,
    pred_path: Path,
    split: str = "public_test",
    verbose: bool = False,
) -> list[StageEvalResult]:
    """Score predictions against gold data.

    Args:
        gold_dir: Path to the built suite (gold data directory).
        pred_path: Path to predictions JSONL file.
        split: Which split to score against.
        verbose: If True, log per-example details.

    Returns:
        List of StageEvalResult, one per stage.
    """
    # Load manifest
    manifest = read_json(gold_dir / "manifest.json")
    stage_count = manifest["stage_count"]

    # Load predictions and index by UID
    predictions = read_predictions(pred_path)
    pred_by_uid: dict[str, Prediction] = {p.uid: p for p in predictions}
    logger.info(f"Loaded {len(predictions)} predictions")

    results: list[StageEvalResult] = []

    for stage_info in manifest["stages"]:
        stage_num = stage_info["stage"]
        family_name = stage_info["family"]

        # Load gold instances for this stage
        jsonl_path = split_path(gold_dir, stage_num, split)
        if not jsonl_path.exists():
            logger.warning(f"No gold data for stage {stage_num}, split {split}")
            continue

        gold_instances = read_instances(jsonl_path)
        if not gold_instances:
            continue

        family = get_family(family_name)

        # Score each instance
        example_results: list[ExampleResult] = []
        correct_count = 0

        for inst in gold_instances:
            pred = pred_by_uid.get(inst.uid)
            if pred is None:
                example_results.append(ExampleResult(
                    uid=inst.uid,
                    correct=False,
                    score=0.0,
                    expected=inst.target,
                    error="No prediction found",
                ))
                continue

            try:
                correct, score = family.verify(
                    pred.prediction, inst.target, inst.metadata,
                )
            except Exception as e:
                correct, score = False, 0.0
                logger.warning(f"Verification error for {inst.uid}: {e}")
                example_results.append(ExampleResult(
                    uid=inst.uid,
                    correct=False,
                    score=0.0,
                    predicted=pred.prediction,
                    expected=inst.target,
                    error=str(e),
                ))
                continue

            if correct:
                correct_count += 1

            example_results.append(ExampleResult(
                uid=inst.uid,
                correct=correct,
                score=score,
                predicted=pred.prediction,
                expected=inst.target,
            ))

        accuracy = correct_count / max(len(gold_instances), 1)
        spec_ids = stage_info.get("spec_ids", [])

        results.append(StageEvalResult(
            stage=stage_num,
            family=family_name,
            spec_id=spec_ids[0] if spec_ids else "",
            split=split,
            accuracy=accuracy,
            total=len(gold_instances),
            correct_count=correct_count,
            results=example_results,
        ))

        logger.info(
            f"Stage {stage_num} ({family_name}): "
            f"{accuracy:.1%} ({correct_count}/{len(gold_instances)})"
        )

    return results
