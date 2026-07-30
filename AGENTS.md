# AGENTS.md

This file documents repository conventions for humans and coding agents.
`CLAUDE.md` and `GEMINI.md` are symlinks to this file — edit `AGENTS.md` only.

## Opencode

Currently Opencode has a problem with the `write`, `edit`, and `apply_patch` tools.
Refrain from using them and instead use the `bash` tool
with an appropriate command to replace them!

## Environment / Running

**CRITICAL EXECUTION RULE:** This project uses `devenv` (Nix) and `uv`. You must NEVER call binaries directly. Doing so could invoke the global system environment and causes fatal errors.

1. **For Python Scripts & Dependencies (`uv`):**
   - Always wrap the command using: `devenv shell -- uv run <command>`
   - Examples: `devenv shell -- uv run pytest`, `devenv shell -- uv run pyfracval -n 256 --df 1.8`
2. **For non-Python system tools:**
   - Always wrap the command using ONLY `devenv shell --`:
   - Example: `devenv shell -- sphinx-build ...`

Common commands:

- Run all tests: `devenv shell -- uv run pytest`
- Run a single test: `devenv shell -- uv run pytest tests/test_geometry.py::test_name -v`
- Run the CLI: `devenv shell -- uv run pyfracval --help` (entry point: `pyfracval.cli:cli`)
- Build docs: `devenv tasks run docs:build` (strict/linkcheck variants exist — see `tasks` in `devenv.nix`)
- Formatting: `ruff format` runs as a git pre-commit hook (configured in `devenv.nix`)
- Profiling: `devenv shell -- uv run py-spy record --format speedscope -o profile.speedscope.json -- pyfracval -n 512 --df 1.6 --kf 1.1 --rp-gstd 1.2`

Numba JIT caches persist in devenv state (`NUMBA_CACHE_DIR`); the first run after
kernel changes recompiles and is slow — that is expected.

## Architecture (Big Picture)

PyFracVAL generates 3D fractal-like aggregates with tunable fractal dimension
(Df) and prefactor (kf) — a Python port of the FracVAL Fortran algorithm
(Morán et al. 2019). The approach is hierarchical: PCA builds small
subclusters, CCA merges them pairwise into the final aggregate.

### Simulation pipeline

`pyfracval/main_runner.py:run_simulation()` orchestrates one aggregate:

1. Validate parameters (`schemas.SimulationParameters`), seed RNG.
2. `particle_generation.py` — lognormal primary-particle radii.
3. `feasibility.py` — predictive feasibility-boundary check (is this Df/kf/σ regime geometrically achievable?).
4. `pca_subclusters.Subclusterer` — split particles into subclusters, build each via PCA (`pca_agg.py`).
5. `cca.CCAggregator` — pair and stick subclusters level-by-level until one aggregate remains.
6. `fractal.validate_fractal_structure` + `quality.py` — verify Df/kf/Rg of the result.
7. `schemas.Metadata` — save data + metadata to `RESULTS/`.

Entry points that call into this: `cli.py` (Click), `app.py` (Streamlit),
`batch_runner.py`, and `dask_runner.py` (distributed sweeps).

### CCA package (`pyfracval/cca/`)

`CCAggregator` (`cca/aggregator.py`) is composed from four mixins, each owning
one concern:

- `pairing.py` — pair generation, Gamma_pc calculation
- `candidates.py` — candidate pair selection, scoring, telemetry
- `sticking.py` — rigid-body sticking, rotation, overlap checks
- `fallbacks.py` — gamma expansion, pair prechecks, sticking fallbacks

Plus `matching.py` (cluster matching) and `rescue.py` (drop/rescue of
frustrated clusters). Failure statistics are recorded through the structured
event log (`pyfracval/event_log.py`); `benchmarks/analyze_event_log.py`
consumes it.

### Performance-critical kernels

Hot paths are Numba JIT kernels: `cca_kernels.py`, `pca_kernels.py`, and the
overlap dispatch in `overlap.py` (`PARALLEL_OVERLAP_THRESHOLD = 200` switches
serial/parallel). Their early-exit/bounding-sphere tricks were benchmarked
against JAX/GPU and won decisively at all realistic sizes
(`docs/source/gpu_acceleration.md`) — do not move them to GPU without new data.

### Configuration flow

- `pyfracval/config.py` — Pydantic v2 models; `OrchestratorConfig` tree with
  `load_config_dict()` supporting `.toml`/`.yaml`/`.json` (presets in `configs/`).
- `OrchestratorAlgorithmConfig` carries all algorithm knobs (snake_case fields,
  e.g. `cca_sticking_method`, `densify_*`) and is passed explicitly into
  `CCAggregator`/`run_simulation`. There are no module-level config constants
  anymore (the old uppercase constants and `config_adapter.py` are gone).
- `pyfracval/environments.py` — `get_env_config()` for `PYFRACVAL_*` and
  thread-control env vars.

### Backward-compat shims (do not add code here)

- `pyfracval/utils.py` → re-exports from `geometry.py`, `fractal.py`,
  `overlap.py`, `cca_kernels.py`, `pca_kernels.py`
- `pyfracval/cca_agg.py` → re-exports `CCAggregator` from `pyfracval.cca`

Prefer the specific module in new imports:

```python
from pyfracval.geometry import rodrigues_rotation, FLOATING_POINT_ERROR
from pyfracval.fractal import calculate_rg, validate_fractal_structure
from pyfracval.overlap import calculate_max_overlap_cca_auto
from pyfracval.cca import CCAggregator
```

### Experimental package (`pyfracval/experimental/`)

Archived CCA sticking experiments (FFT docking, soft relaxation, candidate
policies, retry modes) that did **not** beat the vanilla Fibonacci baseline —
see `docs/source/experiments.md` for the data. Opt-in only via config flags
(`cca_sticking_method`, `cca_soft_relaxation_enabled`), default off. Don't
resurrect these without new benchmark evidence.

### Known hard regime

High Df + low kf + wide polydispersity (e.g. Df=2.25, kf=0.95, rp_gstd=1.9)
collapses CCA sticking success (~17-20%) via geometric frustration.
Backtracking pairing reaches the hard regime directly; densification
(`densify.py`, `densify_*` config) exists as a reshaping mitigation but
currently fails structural validation (see `TODO.md`). Check
`docs/source/experiments.md` before attempting "fixes" here — many obvious
ideas were already tried and measured to not help.

## Tests

- Run tests with: `devenv shell -- uv run pytest`
- Test coverage should include:
  - PCA (Particle-Cluster Aggregation) edge cases
  - CCA (Cluster-Cluster Aggregation) pairing logic
  - Sticking process convergence
  - Overlap calculations
- Focus on testing specific functions/modules rather than full simulation runs.

## Benchmarks

- Benchmarks live in `benchmarks/`; use the single config-first entrypoint
  `benchmarks/run.py` (see `benchmarks/README.md` for presets):
  - `devenv shell -- uv run python benchmarks/run.py unified --config configs/unified_local_smoke.toml`
  - `devenv shell -- uv run python benchmarks/run.py stability --config configs/stability_n_sweep.toml`
  - `devenv shell -- uv run python benchmarks/run.py sticking --suite stable --trials 3`
- Benchmarks must be reproducible:
  - Use deterministic seeding for RNG
  - Set thread-related env vars for NumPy/SciPy reproducibility:
    - `OMP_NUM_THREADS=1`
    - `MKL_NUM_THREADS=1`
    - `OPENBLAS_NUM_THREADS=1`
    - `NUMEXPR_NUM_THREADS=1`
- Benchmark results are saved to `benchmark_results/` directory.
- Generated aggregate data files (`.dat`) are saved to `benchmark_results/aggregates/`.

## Repo Hygiene (Generated Files)

- **DO commit:** Benchmark summary JSON files (`benchmark_results/*_summary.json`)
- **DO NOT commit:** Individual aggregate data files (`benchmark_results/aggregates/*.dat`)
- **DO NOT commit:** Temporary test outputs in `/tmp/`
- Keep the `docs/` folder for reference materials (original Fortran code, papers)
- `papers/` contains reference papers in markdown format - keep these.

## Coding Conventions

- Keep changes minimal and focused on the requested task.
- Prefer `pathlib.Path` for filesystem paths; prefer `match`/`case` over
  dict-of-lambdas for dispatch (see `load_config_dict` for the house style).
- Use Pydantic v2 patterns when touching Pydantic models (see `pyfracval/config.py`).
- NumPy-style docstrings.
- Avoid adding new dependencies unless strictly needed.
- Use logging appropriately:
  - `logger.debug()` for detailed tracing
  - `logger.info()` for important events (swaps, iterations)
  - `logger.warning()` for retry attempts
  - `logger.error()` for failures
- Document significant algorithmic changes / experiment write-ups as markdown
  under `docs/source/` (e.g. `experiments.md`, `correlation_validation.md`).
- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:` …) —
  python-semantic-release derives versions and the changelog from them.
  `CHANGELOG.md` uses an insertion marker; never regenerate it wholesale.

## Issue Tracking

We use a plain [`TODO.md`](TODO.md) in the repo root instead of an external
tool — no daemon, no database, no extra dependency. Open items live under
`## Open`, add new ones as you find them, check items off or move them to
`## Done` as they land. If an item needs more than a couple of sentences
(design rationale, benchmark data, etc.), link out to `PLAN.md` or a
`docs/source/*.md` page rather than growing `TODO.md` into a wiki.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **Update `TODO.md`** - Add anything that needs follow-up, check off / move finished items to "Done"
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
4. **Clean up** - Clear stashes, prune remote branches
5. **Verify** - All changes committed AND pushed
6. **Hand off** - Provide context for next session

**CRITICAL RULES:**

- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
