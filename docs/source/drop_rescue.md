# Drop-a-Few-Particles Rescue

The idea: when a CCA sticking failure is localized - only a handful of
particles are actually overlapping, not a fundamental incompatibility -
drop those particles and keep going rather than discarding the whole
attempt. [overlap_failure_census.md](overlap_failure_census.md) already
found that at N=128 hard regime, this premise mostly doesn't hold (median
9/24 particles implicated, not "a few"). This page implements the
mechanism anyway, and reports what actually happens when it's used, at
both the conservative default budget and a more permissive one.

## Scoping decision: no backfill

A rescued merge produces fewer than the requested N particles - dropped
particles are not regenerated elsewhere. `AggregateProperties.n_particles_dropped`
records the shortfall in the saved output metadata so it's never silent.
This was an explicit decision, not an oversight: backfilling would mean
re-entering the PCA/CCA loop for the dropped particles, a substantially
larger feature than detect-and-drop.

## Method

`pyfracval/cca/rescue.py`:

- `select_drop_candidates(census, max_drop_particles, max_drop_fraction)`
  decides whether a failure is within budget, using
  [overlap_failure_census.md](overlap_failure_census.md)'s
  `OverlapCensus` data. Budget per side is
  `min(max_drop_particles, ceil(max_drop_fraction * cluster_size))` - an
  absolute safety cap and a relative one, so a small cluster can't lose a
  large fraction of itself just because the fixed count allows it.
- `retry_sticking_with_drops(...)` does *not* re-run the full
  candidate/rotation search on a reduced cluster pair (which would need
  plumbing a raw-coordinates entry point through the whole
  `_perform_cca_sticking` machinery). Instead it takes the *exact*
  geometry the overlap census was computed against - the placement that
  was already tried and already failed - removes the identified
  offending particles from it, and checks whether that specific placement
  is now overlap-free. If some other pair is still too close, the rescue
  fails; no second search is attempted.

Wired into `aggregator.py::_run_iteration` as a third fallback tier after
soft relaxation, gated on `cca_drop_rescue_enabled` (default `False`,
auto-enables `cca_overlap_census_enabled` when set).

One correctness fix this feature required: `_run_iteration` pre-allocates
`coords_next`/`radii_next` sized to the particle count entering the
round, then fills them as pairs are processed. Every particle from every
cluster previously always carried forward unchanged, so the array was
always filled exactly full - once particles can be dropped, that's no
longer true, and the arrays now need to be trimmed to the actual fill
count before being carried into the next round (previously-latent, now
fixed: `_run_iteration`'s output is now trimmed, a no-op whenever nothing
was dropped). `_identify_monomers` had a related but purely cosmetic
issue - it sized its scratch array to `self.N` (the originally-requested
total) instead of the currently-active particle count, spuriously
logging every dropped particle's index as "unassigned" every round after
a drop. Both fixed together.

## Results

### Success rate and fractal accuracy

`benchmarks/drop_rescue_accuracy.py`: same hard-regime single-shot
methodology as `pairing_frustration_probe.py` (N=128, Df=2.25, kf=0.95,
σ=1.9, 40 seeds), comparing baseline (no rescue) against drop-rescue at
the config defaults and at a more permissive budget.

| Config | Success rate | Rescued successes | Avg particles dropped | Avg Rg error | Rg within 5% |
|---|---:|---:|---:|---:|---:|
| Baseline (no rescue) | 2.5% (1/40) | - | - | -0.89% | 1/1 |
| Drop-rescue, default budget (max 5 particles, 2% per side) | 2.5% (1/40) | 0 | 0.0 | -0.89% | 1/1 |
| Drop-rescue, relaxed budget (max 5 particles, 25% per side) | 7.5% (3/40) | 2 | 4.0 | +0.46% | 3/3 |

The default budget has **zero measurable effect**: at a cluster-pair size
of 24 (the fixed size every hard-regime failure happens at, per
[overlap_failure_census.md](overlap_failure_census.md)), a 2%-per-side
relative cap allows dropping `ceil(0.02*12) = 1` particle per side -
nowhere near enough given a median of 9/24 particles implicated. This is
the conservative default working as intended, not a bug: it would rather
rescue nothing than aggressively restructure a cluster.

The relaxed budget (up to 25% per side, still capped at 5 particles
absolute) triples the single-shot success rate and, on the small sample
of rescued successes obtained, shows no obvious fractal-accuracy penalty
(Rg error actually landed *closer* to zero than the unrescued baseline's
single success, though n=1 vs n=3 is too small to call that a real
difference either way).

Raw output: `benchmark_results/drop_rescue_accuracy.json`.

### Not yet tested: larger N, later rounds

[overlap_failure_census.md](overlap_failure_census.md) already flagged
that its N=128 census data - and by extension the budget conclusions
above - comes entirely from round-1 failures between small (~12-particle)
PCA subclusters, since that's where every N=128 hard-regime failure
happens. Whether a late-round merge between two large, already-built
clusters looks different (a smaller *fraction* of particles implicated,
even if the absolute count is similar) is a real open question this page
has not answered - it would require sampling N large enough, and lucky
enough in its round structure, for a late-round failure to actually
occur, which is a separate, more involved probe than reused here. Left as
follow-up work rather than assumed either way.

## Discussion

Both results point the same direction: the mechanism works exactly as
designed (it does rescue localized failures, and does so without
obviously distorting fractal accuracy in the cases it succeeds on), but
"localized" is doing a lot of work in that sentence - at the one regime
and scale actually measured, most failures are not localized enough for
even a fairly permissive budget to help. This is not a reason to abandon
the feature (a real, reproducible 3x single-shot success-rate improvement
at a relaxed budget is not nothing), but it is a reason not to promote it
beyond opt-in, and not to assume the "5 out of 512" framing that
motivated it generalizes to the regime this project has actually
characterized in depth.

## Limitations

No backfill (see Scoping decision above) - every downstream consumer of
"the aggregate has exactly N particles" needs to check
`n_particles_dropped`. Not yet benchmarked in combination with Phase 1's
matching-based pairing (fewer failures even reaching the sticking stage
changes how often this fallback gets exercised at all) or against the
full `hard_regime_boundary_sweep.toml`/`full_stability_sweep.toml` grids
(this page's validation uses the faster single-shot methodology only,
consistent with how Phase 1's `matching_leaf_weighted` variant was scoped
once its single-shot result was already clear). The relaxed-budget
fractal-accuracy comparison has an unavoidably small sample (n=1 baseline
success, n=3 rescued successes) given how rare hard-regime single-shot
successes are - a firmer accuracy conclusion would need either a larger
seed count or a less extreme regime.
