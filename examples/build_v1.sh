#!/usr/bin/env bash
# Build the v1_clustered benchmark suite (12 stages, ~19,200 instances).
#
# Usage:
#   bash examples/build_v1.sh [OUTPUT_DIR]
#
# The output directory defaults to ./artifacts/v1_clustered.

set -euo pipefail

OUT_DIR="${1:-./artifacts/v1_clustered}"

echo "=== Building v1_clustered suite ==="
uv run continual-benchmark build-suite \
    --suite v1 \
    --stream v1_clustered \
    --seed 42 \
    --out "$OUT_DIR"

echo ""
echo "=== Running QA checks ==="
uv run continual-benchmark qa-suite --path "$OUT_DIR"

echo ""
echo "=== Inspecting stage 1 ==="
uv run continual-benchmark inspect-stage --path "$OUT_DIR" --stage 1

echo ""
echo "Suite built at: $OUT_DIR"
echo "To list available streams: uv run continual-benchmark list-streams"
