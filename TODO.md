# TODO

Plain-markdown issue tracking (replaced `bd`/beads on 2026-07-26 — see git
history for the old `.beads/` database if you ever need the historical
record). Convention: group by rough priority, check items off as they land,
move finished items to the bottom "Done" section (or just delete them —
git history has the detail).

## Open

- [ ] Implement a **backtracking** CCA pairing strategy (retry a failed
      pair's cluster with a different partner using the *real* sticking
      outcome, not a precomputed graph). `docs/source/matching_pairing.md`
      found that matching over the cheap upfront gamma-feasibility graph
      does not work (+0.2pp over 4200 trials, statistically noise) because
      that graph is too optimistic — geometrically-feasible does not mean
      actually-sticks. A backtracking approach that reacts to a real stick
      failure instead of trying to predict it upfront is the design the
      evidence now points to.
- [ ] Drop-a-few-particles rescue fallback (Phase 3 of the paper-worthy
      roadmap — see `/home/mar/.claude/plans/scalable-cooking-brooks.md`).
      `docs/source/overlap_failure_census.md`'s real data is a caution,
      not a green light: at N=128 hard regime, failures always happen at
      CCA round 1 between ~12-particle subclusters, with a median 9/24
      (~37.5%) particles implicated — not the "5 out of 512" scale the
      idea was framed around. Needs checking whether late-round merges
      between large, already-built clusters look different before
      building the rescue mechanism around an assumption the N=128 data
      doesn't support.

## Done (recent)

- [x] **Statistical overlap-failure census** (Phase 2 of the roadmap).
      Added `pyfracval/overlap_statistics.py` (`compute_overlap_census`,
      a two-cluster non-early-exit overlap scan modeled on
      `densify.py`'s existing self-overlap kernel) and a new
      `OverlapCensus` schema, wired in as a strictly opt-in
      (`cca_overlap_census_enabled`, default off) hook at
      `fallbacks.py::_perform_cca_sticking`'s give-up point, threaded out
      through `run_simulation`'s `diagnostics` parameter into
      `BenchmarkResult.overlap_census`. Ran `benchmarks/overlap_census_probe.py`
      (40 seeds, hard + easy-control regimes, retry-inclusive metric):
      found every hard-regime failure happens at a fixed 24-particle
      cluster pair (confirms `pairing_frustration.md`'s "always round 1"
      finding quantitatively), with a median of 9 offending particles —
      informs (and complicates) the Phase 3 item above. See
      `docs/source/overlap_failure_census.md`.
- [x] **Implement and benchmark matching-based CCA pairing** (closes the
      previous "Implement a matching-based (or backtracking) CCA pairing
      strategy" item). Added `pyfracval/cca/matching.py` (exact
      maximum-cardinality matching + leaf-weighted variant, both via
      memoized brute-force DP — cheap and exact at CCA's round-pool sizes,
      ~16 max) and a new `cca_pairing_strategy` config flag
      (`"greedy"`/`"matching"`/`"matching_leaf_weighted"`, default
      unchanged). Benchmarked against the exact
      `pairing_frustration_probe.py` regime/seeds and the full
      `hard_regime_boundary_sweep.toml` grid (4200 trials): **no
      measurable improvement** (72.4% → 72.6% overall, differences scatter
      in both directions with no systematic pattern). Root cause
      identified and documented: `pairing_frustration.md`'s 97.4%
      "rescuable" figure was computed against a feasibility graph built
      from real sticking outcomes; `_generate_pairs()` can only afford the
      cheap gamma-feasibility graph, which is necessary-but-not-sufficient
      for actual sticking success, so maximizing cardinality over it
      doesn't reliably find pairs that actually work. See
      `docs/source/matching_pairing.md`. Both strategies remain available
      as opt-in config values but are not promoted to default.
- [x] **Phase 0 benchmarking baseline**: fixed `run_simulation()` to
      report real PCA/CCA/TIMEOUT failure attribution (previously
      hardcoded to "UNKNOWN" everywhere) and benchmarked
      `densify_method="voronoi"` for the first time (worse than `radial`
      on both speed and accuracy). See `docs/source/pipeline_baseline.md`.

- [x] **Phase 5 follow-up — surveyed `dask_runner.py`/`batch_runner.py` vs.
      `benchmarks/`, decided `SweepConfig`'s dask pattern.** Actually
      checked the premise before acting on it: `benchmarks/
      unified_local_remote_benchmark.py` and `benchmarks/stability_sweep.py`
      already import and reuse `dask_runner.get_client()` directly rather
      than reimplementing their own Dask setup — there wasn't real
      duplication to consolidate, just one genuinely dead code path found
      along the way: `dask_runner._install_wheel_bytes` (module-level) was
      an exact unused duplicate of the `_install_wheel_bytes_embedded`
      closure inside `_register_package` (the embedded version is
      deliberately self-contained with its own local imports so cloudpickle
      serialises it by value for `client.run`/`client.run_on_scheduler` —
      see the comment above `_register_package`). Deleted the dead
      module-level copy and its now-unused `importlib`/`sys`/`tempfile`
      imports.
      **Decision:** leave `SweepConfig.dask`'s `enable`-flag pattern as its
      own established convention rather than unifying with `RunConfig`'s
      presence-based one. It's `benchmarks/`-only (not user-facing CLI
      surface), unifying would mean touching every existing sweep config's
      semantics for no functional benefit, and the presence-based pattern
      was specifically the author's call for the CLI-facing `RunConfig`
      (2026-07-26: "make a config entry to configure dask and only use it
      if there is a dask config entry in the file") — not a general mandate
      to replace `enable: bool` everywhere.
      Verified: full test suite green, ruff F821/F401 clean, a real
      `[dask]`-config CLI run (2/2 aggregates via a local 2-worker cluster)
      still succeeds after the dead-code removal.

- [x] **Fix the `ext_case=1` / `random_point_sc` bug** — implemented, not just
      documented. `cca/sticking.py`'s `_cca_sticking_v1` called
      `utils.random_point_sc(...)` in the `ext_case == 1` branch, but that
      function never existed anywhere (pre-existing bug, latent because
      `ext_case` defaults to `0`). The original guess that the missing logic
      lived in `RAND_SAMPLE.f90` was wrong — that file is an unrelated
      Fisher-Yates shuffle. The real routine is `CCA_module.f90`'s
      `Random_point_SC`/`Spherical_cap_angle` subroutines
      (`docs/FracVAL/CCA_module.f90`), now ported to
      `geometry.spherical_cap_angle`/`geometry.random_point_sc`.
      `cca/sticking.py` calls the new `geometry.random_point_sc` instead.
      Verified: new `tests/test_geometry.py` (closed-form check on symmetric
      spheres, point-on-surface and cap-angle-bounds invariants, degenerate
      cases), full test suite green, real `--ext-case 1` CLI runs across 6
      seeds all succeeding (previously a guaranteed `AttributeError`), and a
      hand-crafted geometry directly confirming the `case>0` branch invokes
      the new function correctly.

- [x] **Route confirmed-losing CCA features into `pyfracval/experimental/`**
      (closes the "Coarse-grid CCA retry mode evaluation" decision item too —
      answer was yes, close it out and archive). Five separate small passes,
      each committed and verified independently (full test suite + a real
      end-to-end CLI run exercising the archived path via its config flag):
      - Extra retry rotation modes (`alternate`, `dual_jitter`, `coarse_grid`,
        `coarse_to_fine`) → `pyfracval/experimental/retry_modes.py`.
        `cca/sticking.py::_apply_retry_rotation_mode` now only has `single`
        inline, delegates elsewhere.
      - Soft-accept + rigid repair (`cca_soft_accept_*`/`cca_repair_*`) —
        confirmed genuinely dead (grepped: no reader anywhere outside
        `config.py` and already-archived `configs/archive/*.toml`) and
        **deleted** rather than relocated, per the original item's own call.
      - BV/SSA pair-feasibility prefilters → moved
        `_bounding_volume_precheck` (was in `fallbacks.py`) and
        `_surface_accessible_mask` (was in `candidates.py`) to
        `pyfracval/experimental/pair_prefilters.py`; updated
        `tests/test_cca_features.py` to import them from their new home.
      - Γ-expansion → `pyfracval/experimental/gamma_expansion.py`
        (`run_gamma_expansion(aggregator, ...)`, takes the aggregator
        instance directly since the loop is tightly coupled to its telemetry
        counters and the `_gamma_pc_override` side-channel).
      - Non-baseline candidate policies (`leaf_soft`/`leaf_score`/
        `leaf_hybrid`) → `pyfracval/experimental/candidate_policies.py`.
        Also fixed `config.py`'s `cca_candidate_policy` default, which was
        `"leaf_hybrid"` even though `docs/source/experiments.md` already
        claimed `"baseline"` was the production default — the doc and the
        code disagreed; the doc was right, the config default was fixed to
        match.
      All five follow the existing `fft_docking`/`soft_relaxation` archival
      pattern: implementation moves to `pyfracval/experimental/`, a thin
      opt-in dispatch stays in `cca/` gated by the same config flag as
      before, so nothing reachable via config actually stopped working —
      it just isn't cluttering the production sticking loop anymore.
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
