#!/usr/bin/env bash
set -euo pipefail

SCHEDULER_ADDRESS="${1:-tcp://marvin.bv.e-technik.tu-dortmund.de:8786}"
REPEATS="${2:-5}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="dask_profile_5reps_${TS}"
OUT_BASE="benchmark_results/profiles/${RUN_ID}"
MAX_RETRIES="${MAX_RETRIES:-3}"

mkdir -p "${OUT_BASE}/remote"
FAIL_LOG="${OUT_BASE}/failures.log"

COMMON_ARGS=(
	--n-aggregates 48
	--warmup-tasks 4
	--local-workers 4
	--scheduler-address "${SCHEDULER_ADDRESS}"
)

echo "Run ID: ${RUN_ID}"
echo "Scheduler: ${SCHEDULER_ADDRESS}"
echo "Repeats: ${REPEATS}"
echo "Max retries per run: ${MAX_RETRIES}"

run_one() {
	local n="$1"
	local rep="$2"
	local out_json="${OUT_BASE}/remote/unified_N${n}_rep${rep}.json"
	local attempt

	if [ -s "${out_json}" ]; then
		echo "Skipping N=${n}, rep=${rep} (already exists)"
		return 0
	fi

	for attempt in $(seq 1 "${MAX_RETRIES}"); do
		echo "Running N=${n}, rep=${rep}, attempt=${attempt}/${MAX_RETRIES}"
		if uv run python benchmarks/unified_local_remote_benchmark.py \
			--n "${n}" \
			--profile-dir "${OUT_BASE}" \
			"${COMMON_ARGS[@]}" \
			--output-json "${out_json}"; then
			return 0
		fi
		echo "Attempt failed for N=${n}, rep=${rep}, attempt=${attempt}" >&2
	done

	echo "FAILED N=${n} rep=${rep} after ${MAX_RETRIES} attempts" | tee -a "${FAIL_LOG}" >&2
	return 1
}

for N in 128 256 512; do
	for REP in $(seq 1 "${REPEATS}"); do
		run_one "${N}" "${REP}" || true
	done
done

echo "Batch profiling runs complete: ${OUT_BASE}"
