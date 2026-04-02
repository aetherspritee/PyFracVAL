#!/usr/bin/env bash
set -euo pipefail

SCHEDULER_ADDRESS="${1:-tcp://marvin.bv.e-technik.tu-dortmund.de:8786}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="dask_profile_pilot_${TS}"
OUT_BASE="benchmark_results/profiles/${RUN_ID}"

mkdir -p "${OUT_BASE}/remote"

COMMON_ARGS=(
	--n-aggregates 48
	--warmup-tasks 4
	--local-workers 4
)

echo "Run ID: ${RUN_ID}"
echo "Scheduler: ${SCHEDULER_ADDRESS}"

uv run python benchmarks/unified_local_remote_benchmark.py \
	--n 128 \
	--scheduler-address "${SCHEDULER_ADDRESS}" \
	--profile-dir "${OUT_BASE}" \
	"${COMMON_ARGS[@]}" \
	--output-json "${OUT_BASE}/remote/unified_N128_rep1.json"

uv run python benchmarks/unified_local_remote_benchmark.py \
	--n 256 \
	--scheduler-address "${SCHEDULER_ADDRESS}" \
	--profile-dir "${OUT_BASE}" \
	"${COMMON_ARGS[@]}" \
	--output-json "${OUT_BASE}/remote/unified_N256_rep1.json"

echo "Pilot profiling runs complete: ${OUT_BASE}"
