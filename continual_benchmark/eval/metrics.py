"""Continual learning metrics computed from the performance matrix.

Implements:
- Average performance after each stage
- Forgetting (per-task and average)
- Backward transfer (per-task and average)
- Forward transfer (per-task and average)
- Family-wise breakdowns
"""

from __future__ import annotations

from continual_benchmark.core.schemas import MetricReport, PerformanceMatrix


def compute_cl_metrics(perf_matrix: PerformanceMatrix) -> MetricReport:
    """Compute all continual learning metrics from a performance matrix.

    Args:
        perf_matrix: The performance matrix where matrix[i][j] = score
            on task j after training through stage i.

    Returns:
        MetricReport with all computed metrics.
    """
    matrix = perf_matrix.matrix
    training_stages = perf_matrix.training_stages
    eval_stages = perf_matrix.eval_stages
    n_train = len(training_stages)
    n_eval = len(eval_stages)

    # --- Average performance after each training stage ---
    avg_accuracy: list[float] = []
    for i in range(n_train):
        # Average over seen tasks (tasks with stage <= current training stage)
        seen_scores = []
        for j in range(n_eval):
            if eval_stages[j] <= training_stages[i] and matrix[i][j] is not None:
                seen_scores.append(matrix[i][j])
        avg = sum(seen_scores) / max(len(seen_scores), 1) if seen_scores else 0.0
        avg_accuracy.append(avg)

    # --- Forgetting ---
    # For task j: forgetting = max_prev_performance - current_performance
    forgetting: dict[int, float] = {}
    for j in range(n_eval):
        task_stage = eval_stages[j]
        max_prev = None
        current = None

        for i in range(n_train):
            val = matrix[i][j]
            if val is None:
                continue
            # Track max performance up to current stage
            if max_prev is None or (val is not None and val > max_prev):
                max_prev = val
            current = val

        if max_prev is not None and current is not None:
            forgetting[task_stage] = max(0.0, max_prev - current)

    avg_forgetting = (
        sum(forgetting.values()) / max(len(forgetting), 1)
        if forgetting else 0.0
    )

    # --- Backward transfer ---
    # BWT for task j: performance on j after all training - performance on j
    # right after learning j
    backward_transfer: dict[int, float] = {}
    for j in range(n_eval):
        task_stage = eval_stages[j]
        # Find the row where training_stage == task_stage (just learned j)
        just_learned_row = None
        for i, ts in enumerate(training_stages):
            if ts == task_stage:
                just_learned_row = i
                break

        if just_learned_row is not None and matrix[just_learned_row][j] is not None:
            perf_after_learning = matrix[just_learned_row][j]
            # Final performance (last row)
            final_perf = matrix[-1][j]
            if final_perf is not None:
                backward_transfer[task_stage] = final_perf - perf_after_learning

    avg_bwt = (
        sum(backward_transfer.values()) / max(len(backward_transfer), 1)
        if backward_transfer else 0.0
    )

    # --- Forward transfer ---
    # FWT for task j: performance on j before learning j (zero-shot)
    # minus a baseline (0 for tasks not yet seen)
    forward_transfer: dict[int, float] = {}
    for j in range(n_eval):
        task_stage = eval_stages[j]
        # Find the row just before task_stage
        prev_row = None
        for i, ts in enumerate(training_stages):
            if ts >= task_stage:
                break
            prev_row = i

        if prev_row is not None and matrix[prev_row][j] is not None:
            forward_transfer[task_stage] = matrix[prev_row][j]

    avg_fwt = (
        sum(forward_transfer.values()) / max(len(forward_transfer), 1)
        if forward_transfer else 0.0
    )

    # --- Family-wise breakdown ---
    family_breakdown: dict[str, dict[str, float]] = {}
    families = perf_matrix.families

    family_forgetting: dict[str, list[float]] = {}
    family_accuracy: dict[str, list[float]] = {}

    for j in range(n_eval):
        task_stage = eval_stages[j]
        family = families.get(task_stage, "unknown")

        # Final accuracy
        final = matrix[-1][j] if matrix[-1][j] is not None else 0.0
        family_accuracy.setdefault(family, []).append(final)

        # Forgetting
        if task_stage in forgetting:
            family_forgetting.setdefault(family, []).append(forgetting[task_stage])

    for family in set(list(family_accuracy.keys()) + list(family_forgetting.keys())):
        breakdown: dict[str, float] = {}
        if family in family_accuracy:
            accs = family_accuracy[family]
            breakdown["average_accuracy"] = sum(accs) / max(len(accs), 1)
        if family in family_forgetting:
            fgts = family_forgetting[family]
            breakdown["average_forgetting"] = sum(fgts) / max(len(fgts), 1)
        family_breakdown[family] = breakdown

    return MetricReport(
        average_accuracy=avg_accuracy,
        forgetting=forgetting,
        average_forgetting=avg_forgetting,
        backward_transfer=backward_transfer,
        average_backward_transfer=avg_bwt,
        forward_transfer=forward_transfer,
        average_forward_transfer=avg_fwt,
        family_breakdown=family_breakdown,
        performance_matrix=perf_matrix,
    )
