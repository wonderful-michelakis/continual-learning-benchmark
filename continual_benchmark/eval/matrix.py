"""Build the continual learning performance matrix.

The performance matrix has:
- rows = training stages (which stage the model was trained through)
- columns = evaluation tasks/stages
- cell[i][j] = score on task j after training through stage i
"""

from __future__ import annotations

from continual_benchmark.core.schemas import PerformanceMatrix, StageEvalResult


def build_performance_matrix(
    eval_results_by_training_stage: dict[int, list[StageEvalResult]],
    all_eval_stages: list[int] | None = None,
) -> PerformanceMatrix:
    """Build a performance matrix from evaluation results.

    Args:
        eval_results_by_training_stage: Map from training_stage -> list of
            StageEvalResult (one per evaluated task).
        all_eval_stages: Optional explicit list of eval stages (columns).
            If not provided, inferred from the results.

    Returns:
        PerformanceMatrix with the full continual learning matrix.
    """
    training_stages = sorted(eval_results_by_training_stage.keys())

    # Determine eval stages (columns)
    if all_eval_stages is None:
        eval_stage_set: set[int] = set()
        for results in eval_results_by_training_stage.values():
            for r in results:
                eval_stage_set.add(r.stage)
        all_eval_stages = sorted(eval_stage_set)

    eval_stage_to_idx = {s: i for i, s in enumerate(all_eval_stages)}

    # Build the matrix
    matrix: list[list[float | None]] = []
    families: dict[int, str] = {}

    for train_stage in training_stages:
        row = [None] * len(all_eval_stages)
        results = eval_results_by_training_stage[train_stage]

        for r in results:
            if r.stage in eval_stage_to_idx:
                col = eval_stage_to_idx[r.stage]
                row[col] = r.accuracy
                families[r.stage] = r.family

        matrix.append(row)

    return PerformanceMatrix(
        training_stages=training_stages,
        eval_stages=all_eval_stages,
        matrix=matrix,
        families=families,
    )
