# TODO

Plain-markdown issue tracking (replaced `bd`/beads on 2026-07-26 — see git
history for the old `.beads/` database if you ever need the historical
record). Convention: group by rough priority, check items off as they land,
move finished items to the bottom "Done" section (or just delete them —
git history has the detail).

## Open

- [ ] **Phase 3 — Split `cca_agg.py` monolith into a `cca/` subpackage** (priority: medium)
  2,400-line `CCAggregator` → `cca/pairing.py`, `cca/candidates.py`,
  `cca/sticking.py`, `cca/fallbacks.py`, `cca/aggregator.py` (thin
  orchestrator). Work against the *live* code, not the stale orphaned
  extraction that was already deleted (see `PLAN.md` §3, §6). This is also
  where the CCA-monolith-embedded losing features (extra retry modes,
  soft-accept/repair, BV/SSA filters, Γ-expansion, non-baseline candidate
  policies — confirmed not to help, see `docs/source/experiments.md`) should
  get routed into `pyfracval/experimental/` as part of the same refactor.
  Phase 1 (config unification) landed 2026-07-26, so this is now unblocked
  and ready to start — each of the 39+ former `getattr(config, ...)` sites
  in `cca_agg.py` now reads `self.algorithm_config.*`, making it much
  clearer which methods depend on which config knobs before splitting them
  out.

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
  geometric frustration. **Decision needed:** close this out and let Phase 3
  route `alternate`/`coarse_grid`/`coarse_to_fine`/`dual_jitter` into
  `pyfracval/experimental/`, keeping only `single` on the production path?

## Done (recent)

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
