# AGENTS.md

This file documents repository conventions for humans and coding agents.

## Environment / Running

**CRITICAL EXECUTION RULE:** This project uses `devenv` (Nix) and `uv`. You must NEVER call binaries directly. Doing so could invoke the global system environment and causes fatal errors.

1. **For Python Scripts & Dependencies (`uv`):**
   - Always wrap the command using: `devenv shell -- uv run <command>`
   - Examples: `devenv shell -- uv run pytest`, `devenv shell -- uv run python pyfracval/main_runner.py`
2. **For Non-Python System Tools (like `bd`):**
   - Always wrap the command using ONLY `devenv shell --`:
   - Example: `devenv shell -- bd ready`

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

We use bd (beads) for issue tracking instead of Markdown TODOs or external tools.
_Always remember to wrap `bd` commands with `devenv shell --`._

### Quick Reference

```bash
# Find ready work (no blockers)
devenv shell -- bd ready --json

# Find ready work including future deferred issues
devenv shell -- bd ready --include-deferred --json

# Create new issue
devenv shell -- bd create "Issue title" -t bug|feature|task -p 0-4 -d "Description" --json

# Create issue with due date and defer (GH#820)
devenv shell -- bd create "Task" --due=+6h              # Due in 6 hours
devenv shell -- bd create "Task" --defer=tomorrow       # Hidden from bd ready until tomorrow
devenv shell -- bd create "Task" --due="next monday" --defer=+1h  # Both

# Update issue status
devenv shell -- bd update <id> --status in_progress --json

# Update issue with due/defer dates
devenv shell -- bd update <id> --due=+2d                # Set due date
devenv shell -- bd update <id> --defer=""               # Clear defer (show immediately)

# Link discovered work
devenv shell -- bd dep add <discovered-id> <parent-id> --type discovered-from

# Complete work
devenv shell -- bd close <id> --reason "Done" --json

# Show dependency tree
devenv shell -- bd dep tree <id>

# Get issue details
devenv shell -- bd show <id> --json

# Query issues by time-based scheduling (GH#820)
devenv shell -- bd list --deferred              # Show issues with defer_until set
devenv shell -- bd list --defer-before=tomorrow # Deferred before tomorrow
devenv shell -- bd list --defer-after=+1w       # Deferred after one week from now
devenv shell -- bd list --due-before=+2d        # Due within 2 days
devenv shell -- bd list --due-after="next monday" # Due after next Monday
devenv shell -- bd list --overdue               # Due date in past (not closed)
```

### Workflow

1. **Check for ready work**: Run `devenv shell -- bd ready` to see what's unblocked
2. **Claim your task**: `devenv shell -- bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work**: If you find bugs or TODOs, create issues:
   - `devenv shell -- bd create "Found bug in auth" -t bug -p 1 --json`
   - Link it: `devenv shell -- bd dep add <new-id> <current-id> --type discovered-from`
5. **Complete**: `devenv shell -- bd close <id> --reason "Implemented"`
6. **Persist**: Ensure `.beads/issues.jsonl` is updated before committing.
   - With auto-flush enabled, it should update automatically after edits.
   - Otherwise run: `devenv shell -- bd export -o .beads/issues.jsonl`
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

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   devenv shell -- bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**

- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
