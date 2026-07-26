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

- [ ] **Phase 4/5 — Shim cleanup, `main_runner`/`batch_runner`/`dask_runner` consolidation** (priority: low)
  Migrate internal `from . import utils` callers to the real domain modules
  but *keep* the `utils.py` re-export shim (unconfirmed whether anything
  external imports `pyfracval.utils.*` — don't break it blind). Rationalize
  the three runner modules into one clear execution story; decide the fate
  of Dask. See `PLAN.md` §4, §9, §10.

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
