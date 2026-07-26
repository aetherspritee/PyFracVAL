# AGENTS.md

This file documents repository conventions for humans and coding agents.

## Opencode

Currently Opencode has a problem with the `write`, `edit`, and `apply_patch` tool.
Refrain from using them and instead use the `bash` tool
with appropriate an appropriate command to replace them!

## Environment / Running

**CRITICAL EXECUTION RULE:** This project uses `devenv` (Nix) and `uv`. You must NEVER call binaries directly. Doing so could invoke the global system environment and causes fatal errors.

1. **For Python Scripts & Dependencies (`uv`):**
   - Always wrap the command using: `devenv shell -- uv run <command>`
   - Examples: `devenv shell -- uv run pytest`, `devenv shell -- uv run python pyfracval/main_runner.py`
2. **For non-Python system tools:**
   - Always wrap the command using ONLY `devenv shell --`:
   - Example: `devenv shell -- sphinx-build ...`

- Use `devenv` for local development environment (see `devenv.nix`).

## Tests

- Run tests with: `devenv shell -- uv run pytest`
- Test coverage should include:
  - PCA (Particle-Cluster Aggregation) edge cases
  - CCA (Cluster-Cluster Aggregation) pairing logic
  - Sticking process convergence
  - Overlap calculations
- Focus on testing specific functions/modules rather than full simulation runs.

## Benchmarks

- Benchmarks are located in `benchmarks/` directory.
- Run sticking benchmarks: `devenv shell -- uv run python benchmarks/sticking_benchmark.py`
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
- Prefer `pathlib.Path` for filesystem paths.
- Use Pydantic v2 patterns when touching Pydantic models (see `pyfracval/config.py`).
- Avoid adding new dependencies unless strictly needed.
- Follow existing code structure:
  - `pyfracval/pca_agg.py` - PCA implementation
  - `pyfracval/cca_agg.py` - CCA implementation
  - `pyfracval/utils.py` - Utility functions (overlap, geometry)
  - `pyfracval/config.py` - Configuration and validation
- Use logging appropriately:
  - `logger.debug()` for detailed tracing
  - `logger.info()` for important events (swaps, iterations)
  - `logger.warning()` for retry attempts
  - `logger.error()` for failures
- Document significant algorithmic changes in markdown files in repo root.

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

## Module Structure (Updated 2026-04-10)

The codebase was refactored from monolithic files into domain-specific modules.
`utils.py` and `cca_agg.py` remain as backward-compatible re-export shims.

### Core Algorithm Modules
- `pyfracval/pca_agg.py` - PCA (Particle-Cluster Aggregation) implementation
- `pyfracval/cca_agg.py` - CCA (Cluster-Cluster Aggregation) orchestrator (delegates to sub-modules)
- `pyfracval/pca_subclusters.py` - PCA subcluster generation with parallel support

### Extracted from `utils.py` (backward-compatible shim still exists)
- `pyfracval/geometry.py` - Rodrigues rotation, sphere intersection, `FLOATING_POINT_ERROR`
- `pyfracval/fractal.py` - Fractal metrics (`calculate_rg`, `gamma_calculation`, `validate_fractal_structure`)
- `pyfracval/overlap.py` - Overlap calculation dispatch (`calculate_max_overlap_*_auto`, `PARALLEL_OVERLAP_THRESHOLD`)
- `pyfracval/cca_kernels.py` - CCA-specific JIT kernels (`_cca_reintento_kernel`, `batch_rotate_cluster_cca`, `_GOLDEN_RATIO`, `_TWO_PI`)
- `pyfracval/pca_kernels.py` - PCA-specific JIT kernels (`batch_calculate_positions_pca`, `batch_check_overlaps_pca`)

### Configuration & Environment
- `pyfracval/config.py` - Pydantic models + legacy constants (deprecated in favor of config adapter)
- `pyfracval/config_adapter.py` - `get_config()` returns `OrchestratorAlgorithmConfig`; `getattr_config()` drop-in for `getattr(config, "X", default)`
- `pyfracval/environments.py` - `get_env_config()` for `PYFRACVAL_*` and thread control env vars
- `pyfracval/schemas.py` - Pydantic schemas for simulation results

### I/O & Runner
- `pyfracval/main_runner.py` - Main simulation entry point
- `pyfracval/cli.py` - Click CLI interface
- `pyfracval/app.py` - Streamlit web app
- `pyfracval/dask_runner.py` - Dask distributed execution
- `pyfracval/batch_runner.py` - Batch simulation runner

### Specialized Modules
- `pyfracval/densify.py` - Aggregate densification
- `pyfracval/fft_docking.py` - FFT-based docking for CCA sticking
- `pyfracval/soft_relaxation.py` - Soft potential relaxation fallback
- `pyfracval/particle_generation.py` - Lognormal particle radius generation
- `pyfracval/visualization.py` - PyVista-based 3D visualization
- `pyfracval/logs.py` - Custom logging setup

### Import Conventions
```python
# Preferred (new modules):
from pyfracval.geometry import rodrigues_rotation, FLOATING_POINT_ERROR
from pyfracval.fractal import calculate_rg, validate_fractal_structure
from pyfracval.overlap import calculate_max_overlap_cca_auto
from pyfracval.config_adapter import get_config

# Still works (backward compatible, deprecated):
from pyfracval.utils import calculate_rg, rodrigues_rotation
from pyfracval.config import CCA_STICKING_METHOD  # emits DeprecationWarning
```

### Key Constants
- `FLOATING_POINT_ERROR = 1e-9` — defined in `geometry.py`, re-exported from `utils.py`
- `_GOLDEN_RATIO`, `_TWO_PI` — defined in `cca_kernels.py`, used by JIT kernels
- `PARALLEL_OVERLAP_THRESHOLD = 200` — defined in `overlap.py`
- Legacy uppercase constants in `config.py` are deprecated; use `get_config()` from `config_adapter.py`
