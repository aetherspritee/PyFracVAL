# Benchmarks

Config-first benchmark workflows for PyFracVAL.

## Single Entrypoint

Use `benchmarks/run.py` for benchmark execution.

```bash
# Unified local/remote benchmark
uv run python benchmarks/run.py unified --config configs/unified_local_smoke.toml

# Stability sweep
uv run python benchmarks/run.py stability --config configs/stability_n_sweep.toml

# Sticking benchmark (quick)
uv run python benchmarks/run.py sticking --suite stable --trials 3
```

## Recommended Config Presets

- `configs/unified_local_smoke.toml` - fast local smoke validation
- `configs/unified_marvin_profile.toml` - local/remote profiling against Marvin
- `configs/release_validation.toml` - larger-N release checks
- `configs/stability_n_sweep.toml` - stability sweep grid

## Unified Benchmark Notes

The unified runner is config-first:
- `--config` points to orchestrator TOML
- if omitted, it auto-discovers `benchmark_orchestrator.toml` or `configs/benchmark_orchestrator.toml`
- CLI flags act as overrides

Execution behavior:
- default: `execution_mode = "sequential"`
- optional: `execution_mode = "parallel"` (prints reproducibility warning)

## Analysis Tools

Analysis scripts remain separate and can be called directly or through the entrypoint:

```bash
# Dask profile analysis
uv run python benchmarks/run.py analyze dask-profiles   "benchmark_results/profiles/<run>/remote/unified_N128_rep*.json"   "benchmark_results/profiles/<run>/remote/unified_N256_rep*.json"   "benchmark_results/profiles/<run>/remote/unified_N512_rep*.json"   --output-json benchmark_results/profiles/<run>/analysis.json

# Stability analysis
uv run python benchmarks/run.py analyze stability --help
```

## Shell Helpers

Existing helper scripts in `benchmarks/scripts/` now call the unified entrypoint:
- `profile_pilot.sh`
- `profile_batch_5reps.sh`
- `profile_analyze.sh`

## Scope Change

Legacy one-off benchmark runner scripts were removed in favor of the unified entrypoint and config presets.
