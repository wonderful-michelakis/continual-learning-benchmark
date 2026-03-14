# Metrics

## Performance Matrix

The core data structure is a matrix `M` where:
- Rows = training stages (which stage the model was trained through)
- Columns = evaluation tasks/stages
- `M[i][j]` = score on task `j` after training through stage `i`

## Average Accuracy

After training stage `t`: average score over all seen tasks `j ≤ t`.

## Forgetting

For task `j`: `forgetting_j = max(M[1..T][j]) - M[T][j]`

The maximum previous performance minus the current (final) performance. Measures how much a task degraded over time. Higher = worse.

**Average forgetting** = mean of per-task forgetting values.

## Backward Transfer

For task `j`: `BWT_j = M[T][j] - M[j][j]`

Final performance on task `j` minus performance right after learning task `j`. Negative values indicate degradation.

## Forward Transfer

For task `j`: `FWT_j = M[j-1][j]` (if measurable)

Performance on task `j` before it was learned (zero-shot). Indicates how much prior learning helps on new tasks.

## Family-wise Breakdowns

All metrics are also computed per task family (DSL, Transform, SQL, API Code) to identify which families are most affected by forgetting.

## Report Formats

Reports are generated in:
- JSON (`metrics.json`)
- CSV (`metrics.csv`, `stage_results.csv`)
- Markdown (`report.md`)
