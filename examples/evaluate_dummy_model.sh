#!/usr/bin/env bash
# Demonstrate the full evaluation pipeline using the dummy (copy-target) baseline.
#
# Usage:
#   bash examples/evaluate_dummy_model.sh [SUITE_DIR]
#
# If SUITE_DIR is not provided, builds a fresh suite first.

set -euo pipefail

SUITE_DIR="${1:-./artifacts/v1_clustered}"

# Build if not present
if [ ! -d "$SUITE_DIR" ]; then
    echo "=== Suite not found — building v1_clustered ==="
    uv run continual-benchmark build-suite \
        --suite v1 \
        --stream v1_clustered \
        --seed 42 \
        --out "$SUITE_DIR"
fi

PRED_DIR="./artifacts/dummy_predictions"
METRICS_DIR="./artifacts/dummy_metrics"
mkdir -p "$PRED_DIR" "$METRICS_DIR"

echo "=== Running dummy baseline (copies gold targets as predictions) ==="
uv run python -c "
from continual_benchmark.baselines.sequential_ft.runner import run_sequential_ft, DummyTrainer
from pathlib import Path

suite_dir = Path('$SUITE_DIR')
pred_dir = Path('$PRED_DIR')

results = run_sequential_ft(
    suite_dir=suite_dir,
    output_dir=pred_dir,
    trainer=DummyTrainer(),
)
print(f'Completed {len(results)} stages')
"

echo ""
echo "=== Scoring predictions ==="
# Score each stage's predictions
for pred_file in "$PRED_DIR"/stage_*/predictions.jsonl; do
    stage_num=$(basename "$(dirname "$pred_file")" | sed 's/stage_//')
    echo "Scoring stage $stage_num..."
    uv run continual-benchmark score \
        --gold "$SUITE_DIR" \
        --pred "$pred_file" \
        --out "$METRICS_DIR/stage_${stage_num}_scores.json" 2>/dev/null || true
done

echo ""
echo "=== Done ==="
echo "Predictions: $PRED_DIR"
echo "Metrics:     $METRICS_DIR"
