# TODO

Plain-markdown issue tracking (replaced `bd`/beads on 2026-07-26 — see git
history for the old `.beads/` database if you ever need the historical
record). Convention: group by rough priority, check items off as they land,
move finished items to the bottom "Done" section (or just delete them —
git history has the detail).

## Open

- [ ] **Route confirmed-losing CCA features into `pyfracval/experimental/`** (priority: medium)
  Now that `cca_agg.py` is split into `pyfracval/cca/{pairing,candidates,
  sticking,fallbacks}.py` (see Done, below), the losing features are
  visible as identifiable methods/branches rather than being buried in one
  2,400-line file — but they weren't moved out in that pass, since it was
  already large and risky enough as a pure mechanical split. Still to do,
  as a separate, smaller pass per area (confirmed not to help, see
  `docs/source/experiments.md`):
  - Extra retry rotation modes (`alternate`, `dual_jitter`, `coarse_grid`,
    `coarse_to_fine`) in `cca/sticking.py::_apply_retry_rotation_mode` —
    keep only `single` on the production path.
  - Soft-accept + rigid repair (config flags `cca_soft_accept_*`/
    `cca_repair_*`) — appears to already be dead/unused in the current
    sticking flow; confirm and delete rather than relocate if so.
  - BV/SSA pair-feasibility prefilters in `cca/candidates.py` /
    `cca/fallbacks.py` — confirmed no improvement.
  - Γ-expansion in `cca/fallbacks.py::_perform_cca_sticking_with_expansion`.
  - Non-baseline candidate policies (`cca_candidate_policy` != `baseline`)
    in `cca/candidates.py`.
  Each of these is now an isolated method or clearly-scoped branch instead
  of inline logic mixed into a monolith, which is exactly what makes this
  now tractable as small, independent, well-tested extractions.

- [ ] **Phase 5 follow-up — consolidate `dask_runner.py`, decide `SweepConfig`'s dask pattern** (priority: low)
  The CLI-facing piece landed (see Done, below). Left for a smaller
  follow-up: `dask_runner.py` (worker deployment/wheel-install plumbing)
  and `batch_runner.py` (sequential/parallel dispatch) are still separate
  modules with some overlapping responsibility with `benchmarks/`'s own
  Dask setup code. `SweepConfig.dask` still uses the older
  `DaskSettings.enable: bool` pattern (untouched by this pass) rather than
  the new presence-based `RunConfig.dask: DaskSettings | None` — worth
  deciding whether to unify these or leave `SweepConfig` as its own
  established convention (it's `benchmarks/`-only, different audience than
  the CLI). See `PLAN.md` §9, §10 for the original analysis.

- [ ] **Fix or document the `ext_case=1` / `random_point_sc` bug** (priority: low)
  Discovered during the Phase 4 utils.py migration: `cca/sticking.py`'s
  `_cca_sticking_v1` calls `utils.random_point_sc(...)` in the
  `self.ext_case == 1` branch, but no such function exists anywhere in the
  codebase (confirmed by grep and by a Pyright "not a known attribute"
  warning). This is a **pre-existing bug**, not something introduced by
  the Phase 3/4 refactors — the mechanical extraction just preserved it
  faithfully. It's latent because `ext_case` defaults to `0`; anyone who
  explicitly sets `ext_case=1` will hit an `AttributeError` at runtime.
  Needs either an implementation (based on what the Fortran `RAND_SAMPLE.f90`
  path was meant to do — see `docs/FracVAL/`) or, if `ext_case=1` is
  genuinely unsupported/experimental, a clear error message instead of a
  silent `AttributeError`.

## In progress / needs a decision

- [ ] **Coarse-grid CCA retry mode evaluation** (was `PyFracVAL-bcy`)
  Implemented `coarse_grid` and `coarse_to_fine` retry rotation modes with
  CM→contact spin axes, benchmarked against `single`/`alternate`. Own
  benchmark notes already concluded: all modes reach the same success rate,
  `coarse_grid`/`coarse_to_fine` are *slower* than `single`/`alternate`, and
  in the hard regime (Df=2.25, kf=0.95, σ=1.9) none of the four rotation
  modes show a meaningful difference. This independently confirms the
  broader Phase 2 retrospective finding (`docs/source/experiments.md`) that
  retry-mode search strategy doesn't matter when the real problem is
  geometric frustration. **Decision needed:** close this out and let the
  "route confirmed-losing CCA features" item above move
  `alternate`/`coarse_grid`/`coarse_to_fine`/`dual_jitter` into
  `pyfracval/experimental/`, keeping only `single` on the production path?

## Done (recent)

- [x] **Phase 5 — Unified execution entry points, opt-in Dask via config
      presence.** Author direction (2026-07-26): a simple example runner,
      Dask kept but opt-in by presence, `main_runner.run_simulation()`
      confirmed as the one shared core (CLI/`batch_runner`/most
      `benchmarks/` already all called it, so no core duplication to
      unify there).
      - Added `examples/generate_aggregate.py` — the minimal on-ramp for
        writing a custom runner (construct a `RunConfig`, call
        `run_simulation()`, done). Actually run end-to-end, not just
        written.
      - `RunConfig.dask: DaskSettings | None = None` — presence-based, no
        separate `enable` flag to also remember (per your call: "presence
        alone is enough"). Only `RunConfig`; `SweepConfig`'s existing
        `enable`-flag pattern is untouched (separate, `benchmarks/`-only
        config family — left as a decision for the follow-up item above).
      - CLI's generation loop now branches: `[dask]` table present →
        `batch_runner.generate_aggregates_parallel` (no outer
        `--max-attempts` retry — each Dask task calls `run_simulation`
        once, which already retries internally up to 20x; per your call
        to keep Dask mode simple rather than add resubmission bookkeeping
        for a re-run of failed tasks); absent → the existing sequential
        loop, byte-for-byte behavior unchanged.
      - Verified: full test suite, a real sequential CLI run (unchanged
        output), a real Dask-config CLI run (3/3 aggregates via a local
        2-worker cluster), the example script actually executed
        end-to-end, and a clean docs build.
- [x] **Phase 4 — Migrated internal `utils.py` shim callers to the real
      domain modules.** `cca/pairing.py`, `cca/fallbacks.py`,
      `cca/sticking.py`, `pca_agg.py`, `densify.py`, `main_runner.py` now
      import directly from `fractal.py`/`overlap.py`/`geometry.py`/
      `cca_kernels.py`/`pca_kernels.py` instead of going through
      `from . import utils; utils.X()`. `utils.py` itself is **kept** as
      the backward-compat shim (per the earlier "unsure on external
      importers, don't break it blind" decision — see `PLAN.md` §4)
      — it still re-exports everything, just isn't used internally
      anymore except for `shuffle_array`/`sort_clusters`/`random_point_sc`,
      which genuinely live there (`sort_clusters`) or have no other home
      to migrate to (`random_point_sc` — see the bug entry above).
      `main_runner.py` also switched from the `cca_agg` shim to importing
      `CCAggregator` directly from `pyfracval.cca`. Verified: `ruff check
      --select F821,F401` clean across every touched file, full test
      suite green, a real end-to-end CLI run, and a clean docs build.
- [x] **Phase 3 — Split `cca_agg.py` monolith into `pyfracval/cca/`.**
      2,424-line `CCAggregator` → `cca/pairing.py`, `cca/candidates.py`,
      `cca/sticking.py`, `cca/fallbacks.py`, `cca/aggregator.py` (thin
      orchestrator composing the four via multiple inheritance).
      `cca_agg.py` is now a 12-line backward-compat re-export shim. Used
      `ast`-derived exact line ranges for the extraction rather than manual
      line-counting (this size of file is too easy to get wrong by hand);
      caught and fixed one real bug from that approach (`@staticmethod`
      decorators silently dropped, since `ast` anchors on the `def` line
      not the decorator line) before committing. Verified with a full test
      run, an `F821` undefined-name scan across all 5 files, and three
      real end-to-end CLI runs including a densify+hard-regime run that
      exercises the largest extracted file. See `PLAN.md` §6.
- [x] **Phase 1 — Config unification.** Config files (TOML/YAML/JSON) are
      now the source of truth, with CLI flags overriding only what's
      explicitly passed. `OrchestratorAlgorithmConfig`/`OrchestratorSimulationConfig`
      completed to cover every former legacy constant (20 fields added);
      new `RunConfig` model + `load_config_dict()` multi-format loader;
      `--config PATH` added to the CLI; the config object is now threaded
      as a real constructor parameter into `CCAggregator`/`PCAggregator`/
      `Subclusterer` (replacing 64 `getattr(config, ...)`/module-global
      reads); `main_runner`'s `setattr`/`finally` monkeypatching deleted;
      the ~70 legacy `UPPERCASE` constants and `config_adapter.py` deleted
      entirely once verified nothing read them. See `PLAN.md` §5 and the
      commit history (6 incremental commits, tests green throughout).
- [x] Deleted orphaned dead modules (`cca_pairing.py`, `cca_candidates.py`,
      `cca_sticking.py`, `cca_fallbacks.py`) and broken top-level `main.py`.
- [x] Fixed `benchmark_results/` git hygiene (`.gitignore` +
      unstage/untrack) — turned out to be staged clutter, not history bloat;
      no history rewrite needed.
- [x] Wrote `docs/source/experiments.md` — full retrospective on CCA sticking
      experiments with real benchmark numbers. Densification confirmed as
      the one feature that actually wins.
- [x] Archived `fft_docking.py` and `soft_relaxation.py` into
      `pyfracval/experimental/` (confirmed no improvement over baseline).
- [x] Archived 20 one-off/losing-feature `configs/*.toml` files into
      `configs/archive/`.
- [x] Fixed `pyfracval/__init__.py` (dead `pyproject.toml` parsing at
      import) and a typo in `config.py`'s deprecation-warning set.
- [x] Added `CONTRIBUTING.md`, refreshed README roadmap.
- [x] Removed `bd`/beads from the project (this change).

See `PLAN.md` for the full codebase analysis this list is drawn from.
