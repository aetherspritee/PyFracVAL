#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?Usage: profile_analyze.sh <run_dir> [output_json]}"
OUT_JSON="${2:-${RUN_DIR}/analysis.json}"

uv run python benchmarks/analyze_dask_profiling_results.py \
	"${RUN_DIR}/remote/unified_N128_rep*.json" \
	"${RUN_DIR}/remote/unified_N256_rep*.json" \
	"${RUN_DIR}/remote/unified_N512_rep*.json" \
	--output-json "${OUT_JSON}"

echo "Analysis saved to: ${OUT_JSON}"
