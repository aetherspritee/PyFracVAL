# TODO

Plain-markdown issue tracking (replaced `bd`/beads on 2026-07-26 — see git
history for the old `.beads/` database if you ever need the historical
record). Convention: group by rough priority, check items off as they land,
move finished items to the bottom "Done" section (or just delete them —
git history has the detail).

## Open

- [ ] **Reconcile the two empirical-Rg definitions.**
      `fractal.compute_empirical_rg` (and `densify.py`'s private copy)
      treat particles as point masses, omitting Eq. 4's per-particle
      `r_g,i²` term; `fractal.compute_empirical_rg_polydisperse` (added
      2026-07-30) implements it. The point-mass form is inconsistent with
      the Γ derivation (Appendix A carries the term), so
      `validate_fractal_structure`'s `rg_error_pct` — the basis of the
      accuracy tables in `docs/source/experiments.md` — is biased low,
      more so at high σ. Not changed unilaterally because it shifts every
      previously-reported accuracy number and densify's convergence
      target. Decide deliberately, then restate the affected tables.
- [ ] **Densification needs a compression scheme that preserves
      structure**, if it is to be useful at all. It currently matches Rg
      by radial compression while producing neither the target fractal
      dimension (f(r) says ~0.5 low) nor valid geometry (overlap
      resolution does not converge). Both are now detected and reported
      honestly rather than silently accepted, so the feature fails loudly
      instead of emitting garbage — but it does not work. Lower priority
      than it was: backtracking reaches the hard regime directly, which
      was densification's whole reason for existing. See
      `docs/source/correlation_validation.md`.
- [ ] **Catalog overlap leak — re-check the original artifact.** The
      densify path is now root-caused and fixed (see Done below), which
      matches the leak's confirmed example being a `densify_retry`
      cluster. Re-run that exact config and confirm the leak is gone
      before closing. `docs/source/catalog_overlap_leak.md`.

## Done (recent)

- [x] **Profiled the pipeline and took up to 1.8x** (`benchmarks/profile_pipeline.py`,
      `docs/source/profiling.md`). Three findings, none in the JIT'd
      overlap kernels: `np.cross` on 3-vectors costs 21us (12.4x slower
      than explicit component arithmetic) and the CCA sticking path made
      ~19.5k norm / ~1.5k cross calls per aggregate; `quality.max_self_overlap`
      built an (N,N,3) array, a self-inflicted regression costing 211ms
      and a 101MB temporary at N=2048 where pdist takes 5ms; and leaf
      masks (O(n^2) per cluster) plus candidate scores were computed on
      every sticking call although nothing reads them under production
      defaults. Then fused the whole ext_case=0 sticking placement into
      one numba kernel (`cca_kernels.cca_sticking_v1_kernel`), pinned
      bit-for-bit against the interpreted reference in
      `tests/test_sticking_kernel.py` by hoisting its two RNG draws out
      as arguments; its self time fell 5.8x. End-to-end 1.18-1.80x,
      growing with N and with regime difficulty. Results unchanged and
      still bit-reproducible. Also recorded a miss: guarding eager
      `logger.trace(f"...")` f-strings is correct but measured no gain,
      and cProfile self-time over-attributed the caller that led there.

- [x] **Measured densification's actual working range** — it converges
      only when the target Df is within ~0.02 of the source, i.e. when it
      does essentially nothing. Residual overlap grows monotonically with
      compression (0.08 at dDf=0.05, 0.52 at dDf=0.5). See the table in
      `docs/source/correlation_validation.md`.

- [x] **f(r) structural validator** (`pyfracval/correlation.py`) — the
      paper's own validation metric, validated against a filled ball
      (recovers Df=3.0+-0.6) before being trusted. Using it produced the
      session's biggest negative result: **densification does not produce
      the requested fractal dimension.** At target Df=2.1 a native
      aggregate measures 2.03 from f(r) while a densified one measures
      1.52, and densified aggregates land closer to their *source* Df
      than their target — while having *better* Rg agreement, which is
      exactly the blind spot Rg-only validation has. Also found every
      densified aggregate carried 43-69% residual particle overlap.
      Root-caused and fixed three separate defects (densify ignored
      `resolve_overlaps`' verdict and returned success on Rg alone; its
      self-overlap check compared the aggregate against itself via the
      two-cluster helper, scoring every particle against itself at
      distance 0; and `main_runner` used the densified result identically
      whether the flag was True or False). This closes the catalog
      overlap leak for the densify path. `experiments.md`'s densification
      conclusion is withdrawn. See `docs/source/correlation_validation.md`.
- [x] **Predictive feasibility criterion** (`pyfracval/feasibility.py`,
      fitted by `benchmarks/fit_feasibility_boundary.py`). Logistic model
      over Df, kf, log sigma, log N with Df*kf and Df*log(sigma)
      interactions, fitted to boundary_sweep_v2's 4200 trials: 97.7%
      agreement with the measured >=50% feasibility call, trial-weighted
      MAE 0.035. `run_simulation` now warns up front when a request sits
      past the boundary, naming the Df that *would* be reliable, and
      discloses when it is extrapolating. Advisory, never blocks. See
      `docs/source/feasibility_criterion.md`.

- [x] **Re-ran the Df/kf/sigma/N boundary sweep against the new defaults**
      (`configs/boundary_sweep_v2.toml`, same grid/seeds as the original).
      Overall 72.4% -> 80.3%; the collapse boundary moved outward at every
      sigma. Sharpest result: at sigma=1.9, Df=2.2/kf=1.0 went from
      degrading to 0.00 at N=1024 to a flat 1.00 across N=64..1024. The
      whole Df=2.1 row at sigma=1.9 is now uniformly 1.00 (was 0.08-0.80
      at high kf). Also found and fixed that `trial_timeout` could never
      fire, because the wall clock was only checked *between* PCA+CCA
      attempts - backtracking makes a single attempt far more expensive in
      infeasible corners. `CCAggregator` now takes a `deadline`. See
      `docs/source/boundary_sweep_v2.md`.
- [x] **Re-evaluated drop-rescue after backtracking — recommend leaving it
      off.** Its original 3x win was measured against the greedy baseline
      that no longer exists. At the new failure frontier the success
      effect is inconsistent (+15pp at one point, -7.5pp at another) while
      mean |Rg error| degrades from 1.2-1.9% to 4.5-10.7% once the budget
      is loose enough to fire. Kept, documented and tested so the idea can
      be answered with a measurement rather than re-litigated. See
      `docs/source/drop_rescue.md`'s new section and
      `benchmarks/drop_rescue_after_backtracking.py`.
- [x] **Per-merge statistics tooling** — `benchmarks/analyze_merge_log.py`
      aggregates the JSONL merge log into failure-mode breakdowns,
      per-round failure rates, search effort, how close failures came, and
      offending-particle counts. Using it immediately exposed three
      defects in its own inputs (all fixed): `min_overlap` was always 0
      because it read the incremental scan's lower bound, offending counts
      could be attributed to the wrong merge via a stale census, and the
      analyzer misaligned two lists. Findings: failures are
      `failed_overlap` not `failed_no_candidates`, they are *not*
      near-misses (median overlap 0.126 against tol_ov 1e-6), and failure
      rate falls monotonically with round (45.8% -> 0%).

- [x] **Backtracking CCA pairing, now the default** (`cca_pairing_strategy
      = "backtracking"`), plus the four supporting items from `NOTE.md`.
      Hard-regime single-shot success went **5.0% → 100%** (40 seeds,
      N=128, Df=2.25, kf=0.95, σ=1.9); the easy regime is unchanged at
      100% with zero backtracking activity, so it costs nothing where the
      first partner already works. Along the way this exposed and fixed a
      **pre-existing overlap-acceptance bug**: the adaptive-tolerance
      path compared an early-terminated overlap scan (which returns the
      first pair above `tol_ov`, not the maximum) against a 10x-larger
      `relaxed_tol`, so placements whose real worst-case overlap was up
      to 0.75 were accepted. 6/174 PCA subclusters were affected before
      the fix, 0/169 after. Also landed: mass-based CCA Gamma (Moran
      Eq. 6) with optional per-particle densities, `cca_gamma_measured_rg`,
      `pyfracval/merge_log.py` (opt-in per-merge JSONL),
      `pyfracval/quality.py` (unconditional per-aggregate quality record
      wired into `run_simulation`), and an N-aware drop-rescue budget
      (`cca_drop_rescue_max_particles=0` disables the absolute cap).
      See `docs/source/backtracking_pairing.md`.

- [x] **Drop-a-few-particles rescue fallback** (Phase 3 of the roadmap,
      closes the item that used to be here). Added `pyfracval/cca/rescue.py`
      (`select_drop_candidates` with a combined absolute+relative budget,
      `retry_sticking_with_drops` which reuses the exact already-censused
      failing geometry rather than re-running a full search) wired into
      `aggregator.py::_run_iteration` as a third fallback tier, gated on
      `cca_drop_rescue_enabled` (auto-enables the Phase 2 census). No
      backfill — `AggregateProperties.n_particles_dropped` reports any
      shortfall. Also fixed two latent bugs this feature exposed:
      `_run_iteration` never trimmed `coords_next`/`radii_next` to the
      actual fill count (harmless until particles could be dropped
      mid-round), and `_identify_monomers` sized its scratch array to
      `self.N` instead of the current active count, spuriously logging
      every dropped particle as "unassigned" every subsequent round.
      Benchmarked (`benchmarks/drop_rescue_accuracy.py`, single-shot
      hard-regime methodology): the conservative default budget has
      **zero effect** (2.5% → 2.5%, budget too tight for N=128's
      24-particle failing pairs), a relaxed budget triples single-shot
      success (2.5% → 7.5%) with no obvious fractal-accuracy penalty on
      the (small) sample rescued. See `docs/source/drop_rescue.md`.

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
