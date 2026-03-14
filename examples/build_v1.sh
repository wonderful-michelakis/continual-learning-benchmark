#!/usr/bin/env bash
# Build the v1_clustered benchmark suite (12 stages, ~2400 instances).
#
# Usage:
#   bash examples/build_v1.sh [OUTPUT_DIR]
#
# The output directory defaults to ./artifacts/v1_clustered.

set -euo pipefail

OUT_DIR="${1:-./artifacts/v1_clustered}"

echo "=== Building v1_clustered suite ==="
python -m continual_benchmark.cli build-suite \
    --suite v1 \
    --stream v1_clustered \
    --seed 42 \
    --out "$OUT_DIR"

echo ""
echo "=== Running QA checks ==="
python -m continual_benchmark.cli qa-suite --path "$OUT_DIR"

echo ""
echo "=== Inspecting stage 1 ==="
python -m continual_benchmark.cli inspect-stage --path "$OUT_DIR" --stage 1

echo ""
echo "Suite built at: $OUT_DIR"
echo "To list available streams: python -m continual_benchmark.cli list-streams"
