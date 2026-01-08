# AGENTS.md

This file documents repository conventions for humans and coding agents.

## Environment / Running

- Use `uv` for running Python code.
  - Examples: `uv run pytest`, `uv run python pyfracval/main_runner.py`
- The main entry point is `pyfracval/main_runner.py` with `run_simulation()` function.
- Use `devenv` for local development environment (see `devenv.nix`).

## Tests

- Run tests with: `uv run pytest`
- Test coverage should include:
  - PCA (Particle-Cluster Aggregation) edge cases
  - CCA (Cluster-Cluster Aggregation) pairing logic
  - Sticking process convergence
  - Overlap calculations
- Focus on testing specific functions/modules rather than full simulation runs.

## Benchmarks

- Benchmarks are located in `benchmarks/` directory.
- Run sticking benchmarks: `uv run python benchmarks/sticking_benchmark.py`
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

We use bd (beads) for issue tracking instead of Markdown TODOs or external tools.

### Quick Reference

```bash
# Find ready work (no blockers)
bd ready --json

# Find ready work including future deferred issues
bd ready --include-deferred --json

# Create new issue
bd create "Issue title" -t bug|feature|task -p 0-4 -d "Description" --json

# Create issue with due date and defer (GH#820)
bd create "Task" --due=+6h              # Due in 6 hours
bd create "Task" --defer=tomorrow       # Hidden from bd ready until tomorrow
bd create "Task" --due="next monday" --defer=+1h  # Both

# Update issue status
bd update <id> --status in_progress --json

# Update issue with due/defer dates
bd update <id> --due=+2d                # Set due date
bd update <id> --defer=""               # Clear defer (show immediately)

# Link discovered work
bd dep add <discovered-id> <parent-id> --type discovered-from

# Complete work
bd close <id> --reason "Done" --json

# Show dependency tree
bd dep tree <id>

# Get issue details
bd show <id> --json

# Query issues by time-based scheduling (GH#820)
bd list --deferred              # Show issues with defer_until set
bd list --defer-before=tomorrow # Deferred before tomorrow
bd list --defer-after=+1w       # Deferred after one week from now
bd list --due-before=+2d        # Due within 2 days
bd list --due-after="next monday" # Due after next Monday
bd list --overdue               # Due date in past (not closed)
```

### Workflow

1. **Check for ready work**: Run `bd ready` to see what's unblocked
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work**: If you find bugs or TODOs, create issues:
   - `bd create "Found bug in auth" -t bug -p 1 --json`
   - Link it: `bd dep add <new-id> <current-id> --type discovered-from`
5. **Complete**: `bd close <id> --reason "Implemented"`
6. **Persist**: Ensure `.beads/issues.jsonl` is updated before committing.
   - With auto-flush enabled, it should update automatically after edits.
   - Otherwise run: `bd export -o .beads/issues.jsonl`
7. **Git**: `.beads/issues.jsonl` is meant to be committed; `.beads/beads.db` is local-only.
   - If `bd init` added `.beads/issues.jsonl` to `.git/info/exclude`, remove that line (or `git add -f .beads/issues.jsonl` once).

### Issue Types

- `bug` - Something broken that needs fixing
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature composed of multiple issues
- `chore` - Maintenance work (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (nice-to-have features, minor bugs)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Dependency Types

- `blocks` - Hard dependency (issue X blocks issue Y)
- `related` - Soft relationship (issues are connected)
- `parent-child` - Epic/subtask relationship
- `discovered-from` - Track issues discovered during work

Only `blocks` dependencies affect the ready work queue.
