# Drop-a-Few-Particles Rescue

When a CCA sticking failure is localized — only a handful of particles
overlapping, rather than a fundamental incompatibility — the failing
particles could in principle be dropped and the merge kept, instead of
discarding the whole attempt.
[overlap_failure_census.md](overlap_failure_census.md) found that in the
N=128 hard regime this premise mostly does not hold (median 9/24
particles implicated). This page describes the mechanism, implemented
regardless to test the premise directly, and reports its measured
behaviour at the conservative default budget and a more permissive one.

The conclusion, re-evaluated after backtracking pairing landed, is that
the feature should remain disabled; the measurements supporting this are
given below.

## Method

`pyfracval/cca/rescue.py`:

- `select_drop_candidates(census, max_drop_particles,
  max_drop_fraction)` decides whether a failure is within budget, using
  [overlap_failure_census.md](overlap_failure_census.md)'s
  `OverlapCensus` data. The budget per side is
  `min(max_drop_particles, ceil(max_drop_fraction * cluster_size))` —
  an absolute cap and a relative one, so a small cluster cannot lose a
  large fraction of itself merely because the fixed count allows it.
- `retry_sticking_with_drops(...)` does not re-run the full
  candidate/rotation search on a reduced cluster pair (which would
  require a raw-coordinates entry point through the whole
  `_perform_cca_sticking` machinery). It takes the exact geometry the
  overlap census was computed against — the placement already tried and
  already failed — removes the identified offending particles, and
  checks whether that placement is now overlap-free. If another pair
  remains too close, the rescue fails; no second search is attempted.

The rescue is wired into `aggregator.py::_run_iteration` as a third
fallback tier after soft relaxation, gated on
`cca_drop_rescue_enabled` (default `False`; enabling it auto-enables
`cca_overlap_census_enabled`).

A rescued merge produces fewer than the requested N particles; dropped
particles are not regenerated.
`AggregateProperties.n_particles_dropped` records the shortfall in the
saved metadata. This scoping was deliberate: backfilling would require
re-entering the PCA/CCA loop for the dropped particles, a substantially
larger feature than detect-and-drop.

## Results

### Against the greedy-pairing baseline

`benchmarks/drop_rescue_accuracy.py`: same hard-regime single-shot
methodology as `pairing_frustration_probe.py` (N=128, Df=2.25, kf=0.95,
σ=1.9, 40 seeds), comparing no rescue against drop-rescue at the config
defaults and at a more permissive budget.

| Config | Success rate | Rescued successes | Avg particles dropped | Avg Rg error | Rg within 5% |
|---|---:|---:|---:|---:|---:|
| Baseline (no rescue) | 2.5% (1/40) | - | - | -0.89% | 1/1 |
| Drop-rescue, default budget (max 5 particles, 2% per side) | 2.5% (1/40) | 0 | 0.0 | -0.89% | 1/1 |
| Drop-rescue, relaxed budget (max 5 particles, 25% per side) | 7.5% (3/40) | 2 | 4.0 | +0.46% | 3/3 |

The default budget has no measurable effect. At the cluster-pair size
of 24 (the fixed size of every hard-regime failure, per
[overlap_failure_census.md](overlap_failure_census.md)), a 2%-per-side
relative cap allows dropping `ceil(0.02*12) = 1` particle per side —
far below the median of 9/24 particles implicated. This is the
conservative default operating as designed rather than a defect: it
rescues nothing rather than aggressively restructure a cluster.

The relaxed budget (up to 25% per side, still capped at 5 particles
absolute) triples the single-shot success rate. On the small sample of
rescued successes, no fractal-accuracy penalty is apparent (Rg error
landed closer to zero than the unrescued baseline's single success,
though n=1 vs. n=3 does not support a conclusion in either direction).

Raw output: `benchmark_results/drop_rescue_accuracy.json`.

### Larger N

[overlap_failure_census.md](overlap_failure_census.md)'s N=512
comparison (added after this page's initial budget validation) shows
the relative offending-particle fraction shrinking with N (20% at N=512
vs. 37.5% at N=128) — a trend in the direction the "5 out of 512"
framing assumed. The absolute count needed (median 20 at N=512) still
exceeds what either budget tested above allows: at N=512's cluster-pair
size of 100, the relaxed budget's 25%-per-side cap is itself capped by
the absolute `max_drop_particles=5` limit
(`min(5, ceil(0.25*100)) = 5`), well under the ~10–15 per side a median
failure would require. The two budget parameters would need different
tuning at different N for the relative cap to engage at all; this was
not evaluated.

All hard-regime failures observed, at both N=128 and N=512, occur at
CCA round 1 (merging PCA subclusters directly, before any cluster has
grown large). Neither sample includes the late-round merge between two
large, already-built clusters that originally motivated the idea; that
case remains untested and would require a probe that waits for or
forces a later-round failure.

### After backtracking pairing (2026-07-30)

The results above were measured against the greedy pairing baseline,
where hard-regime single-shot success was 2.5% and drop-rescue's 7.5%
constituted a threefold improvement. That baseline no longer exists:
[backtracking_pairing.md](backtracking_pairing.md) reaches ~100% at the
same point, so the failures this feature was built to catch largely no
longer occur there. The remaining question is whether it helps at the
new failure frontier — the Df/kf/σ region where backtracking still
fails, per [boundary_sweep_v2.md](boundary_sweep_v2.md).

`benchmarks/drop_rescue_after_backtracking.py`, 40 seeds, same
single-shot methodology, σ=1.9, N=128:

| Point | Config | Success | Aggregates short of N | Particles dropped | mean \|Rg error\| |
|---|---|---:|---:|---:|---:|
| Df=2.3, kf=1.0 | baseline | 55.0% | 0 | 0 | 1.89% |
| | default budget | 57.5% | 1 | 2 | 1.98% |
| | relaxed (25%/side) | 40.0% | 10 | 83 | 3.84% |
| | relative-only (no absolute cap) | 47.5% | 16 | 253 | 10.73% |
| Df=2.4, kf=0.8 | baseline | 45.0% | 0 | 0 | 1.18% |
| | default budget | 42.5% | 1 | 2 | 1.29% |
| | relaxed (25%/side) | 42.5% | 13 | 113 | 4.49% |
| | relative-only (no absolute cap) | 60.0% | 20 | 268 | 6.54% |

Two observations, pointing the same way:

1. The success effect is inconsistent. The relative-only budget gains
   15pp at one frontier point and loses 7.5pp at the other — a pattern
   indistinguishable from noise around zero. A plausible mechanism for
   the losses exists: a rescued merge yields a cluster smaller than the
   hierarchy expects, which shifts every subsequent Γ and can cascade
   into failures later in the same run.
2. The accuracy cost is systematic. Mean \|Rg error\| rises from
   1.2–1.9% to 4.5–10.7% once the budget is loose enough to fire. A
   tunable algorithm exists to hit a prescribed Df/kf; dropping 5–8% of
   the particles misses the target by several times the 5% tolerance
   the rest of the pipeline is held to, in exchange for an unreliable
   change in success rate.

The conservative default budget remains effectively inert (1 rescue,
2 particles, across 40 seeds), consistent with its design.

## Discussion

The mechanism operates as designed — it rescues localized failures
without apparent accuracy cost in the cases it succeeds on — but at
every regime and scale measured, most failures are not localized enough
for even a fairly permissive budget to apply: the merge-log census puts
the median failure at ~35–45% of the cluster pair offending
([event_logging.md](event_logging.md)), not a handful of particles.
After backtracking, the residual failures show no consistent benefit
and a clear accuracy penalty when the budget is loosened enough to
engage.

**Recommendation: leave `cca_drop_rescue_enabled` off.** The mechanism
is retained, documented, and tested because "drop the few offenders" is
an idea that recurs, and a measurement is a more durable answer than
re-deriving the argument. The measurement is that the premise — failures
localized to a handful of particles — does not hold in any regime this
project has characterized.

## Limitations

There is no backfill (see Method): every downstream consumer of "the
aggregate has exactly N particles" must check `n_particles_dropped`.
The feature has not been benchmarked in combination with matching-based
pairing, nor against the full
`hard_regime_boundary_sweep.toml`/`full_stability_sweep.toml` grids;
this page's validation uses the faster single-shot methodology
throughout. The relaxed-budget fractal-accuracy comparison against the
greedy baseline has an unavoidably small sample (n=1 baseline success,
n=3 rescued successes) given how rare hard-regime single-shot successes
were under greedy pairing; a firmer accuracy conclusion would require a
larger seed count or a less extreme regime.

## Implementation notes

Two array-bookkeeping corrections were required by this feature.
`_run_iteration` pre-allocates `coords_next`/`radii_next` sized to the
particle count entering the round and fills them as pairs are
processed; previously every particle always carried forward, so the
arrays were always exactly full. Once particles can be dropped this no
longer holds, and the arrays are now trimmed to the actual fill count
before being carried into the next round (a no-op when nothing is
dropped). Separately, `_identify_monomers` sized a scratch array to
`self.N` (the originally-requested total) instead of the currently
active particle count, spuriously logging every dropped particle's
index as "unassigned" in each round after a drop; this was cosmetic
and is fixed alongside.
