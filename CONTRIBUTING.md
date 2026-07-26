# Contributing to PyFracVAL

Thanks for your interest in contributing! This document covers the practical
bits of working in this repo. For the broader picture of ongoing cleanup work,
see [`PLAN.md`](PLAN.md); for what's already been tried on the CCA sticking
algorithm and what actually worked, see
[`docs/source/experiments.md`](docs/source/experiments.md).

## Environment

This project uses [`devenv`](https://devenv.sh/) (Nix) and
[`uv`](https://github.com/astral-sh/uv). Always run Python commands through
`devenv shell -- uv run <command>` rather than calling `python`/`pytest`
directly — this ensures you get the pinned interpreter and dependencies
rather than whatever's on your system `PATH`.

```bash
devenv shell -- uv run pytest              # run tests
devenv shell -- uv run python pyfracval/main_runner.py
devenv shell -- pyfracval --help           # the installed CLI entry point
```

## Tests

```bash
devenv shell -- uv run pytest
```

Test coverage should include:

- PCA (Particle-Cluster Aggregation) edge cases
- CCA (Cluster-Cluster Aggregation) pairing logic
- Sticking process convergence
- Overlap calculations

Focus on testing specific functions/modules rather than full simulation runs
(those are slow and better suited to the `benchmarks/` harness).

## Benchmarks

Benchmarks live in `benchmarks/` and write output to `benchmark_results/`.

```bash
devenv shell -- uv run python benchmarks/sticking_benchmark.py
```

Benchmarks must be reproducible: use deterministic RNG seeding, and set
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1` for NumPy/SciPy reproducibility.

**Repo hygiene for benchmark output:**

- Commit: benchmark summary JSON/CSV/PNG/HTML analysis outputs
  (`benchmark_results/*_summary.json` and similar).
- Do not commit: individual per-run aggregate coordinate dumps
  (`benchmark_results/**/aggregates/*.dat`) — these are large, regenerable,
  and already excluded via `.gitignore`.

## Coding conventions

- Keep changes minimal and focused on the requested task.
- Prefer `pathlib.Path` for filesystem paths.
- Use Pydantic v2 patterns when touching Pydantic models (see
  `pyfracval/config.py`).
- Avoid adding new dependencies unless strictly needed.
- Follow the existing module structure — see the "Module Structure" section
  in [`AGENTS.md`](AGENTS.md) for the full current map, or `PLAN.md` §2 for a
  narrative walkthrough with the original Fortran → Python lineage.
- Use logging appropriately: `logger.debug()` for detailed tracing,
  `logger.info()` for important events (swaps, iterations),
  `logger.warning()` for retry attempts, `logger.error()` for failures.
- Document significant algorithmic changes in markdown files in the repo
  root or `docs/source/`.

## Pull requests

1. Fork the project and create a feature branch.
2. Install development dependencies: `devenv shell` (handles this
   automatically via `uv sync`).
3. Make your changes, keeping commits focused.
4. Run tests and linters: `devenv shell -- uv run pytest`,
   `devenv shell -- uv run ruff check .`, `devenv shell -- uv run ruff format .`.
   Pre-commit hooks (`isort`, `ruff-format`) run automatically on commit.
5. Open a pull request describing what changed and why.
