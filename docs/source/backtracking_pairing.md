# Backtracking CCA Pairing

[pairing_frustration.md](pairing_frustration.md) established that ~97%
of hard-regime CCA round failures had a different pairing of the same
cluster pool that would have worked.
[matching_pairing.md](matching_pairing.md) implemented the direct
response — exact maximum-cardinality matching up front — and measured no
improvement (+0.2pp over 4200 trials), because matching optimizes over
the cheap gamma-feasibility graph, which is necessary but not sufficient
for sticking and cannot predict which feasible-looking pairs stick.

Backtracking is the design the evidence pointed to instead: rather than
predicting outcomes, it reacts to the real sticking outcome. This page
describes the mechanism, its measured effect (5% → 100% single-shot
success in the hard regime), and a pre-existing overlap-acceptance
defect the new success rate exposed.

## Mechanism

`pyfracval/cca/aggregator.py::_run_iteration_backtracking`. Per round:

1. Build the same cheap gamma-feasibility graph as before.
2. Repeatedly take the most-constrained cluster (fewest remaining
   feasible partners) and attempt to stick it against its partners in
   increasing order of their own constraint, up to
   `cca_backtracking_max_partners` (default 4).
3. On a sticking failure, mark that edge dead and try the next partner.
   On success, both clusters leave the pool.
4. A cluster that exhausts its partners is passed through to the next
   round unmerged rather than failing the attempt
   (`cca_backtracking_pass_through`, default on) — the same treatment
   the odd cluster in an odd-sized pool has always received, extended to
   clusters that are merely frustrated in the current round.
5. A round that merges nothing still fails, so a stalled pool cannot
   loop indefinitely.

The critical difference from the previous behaviour concerns the rest
of the round. Previously, a single failed pair set `not_able_cca` and
discarded every other successful merge in that round, restarting PCA+CCA
from scratch (up to 20 times). Under backtracking those merges are kept.

The cost is bounded: at most 4 sticking attempts per cluster over a
round pool of ~11, against a baseline that discarded whole attempts.

## Results

`benchmarks/backtracking_pairing_benchmark.py`, 40 seeds, single-shot
methodology (no internal retry, so one seed maps to one outcome — the
same metric used in [pairing_frustration.md](pairing_frustration.md)
and [drop_rescue.md](drop_rescue.md), directly comparable to their 2.5%
figure).

Hard regime (N=128, Df=2.25, kf=0.95, σ=1.9). These runs predate mass
becoming CCA's unconditional Γ form, so the Γ column records what was
varied at the time; the pairing comparison is unaffected.

| Config | Success | Merges rescued by backtracking | avg \|Rg error\| |
|---|---:|---:|---:|
| Greedy first-fit (previous default) | 5.0% (2/40) | - | 1.08% |
| Backtracking | 100.0% (40/40) | 93 | 1.40% |
| Backtracking, mass-based Γ (now unconditional) | 95.0% (38/40) | 98 | 0.52% |
| Backtracking + measured-Rg Γ | 80.0% (32/40) | 140 | 1.59% |
| Backtracking + mass + measured-Rg | 90.0% (36/40) | 149 | 1.67% |

In the easy control regime (N=128, Df=1.8, kf=1.0, σ=1.5) every arm
reaches 100%, so no regression occurs in the safe region. The Γ variants
do improve accuracy there, where success is not at stake: avg
\|Rg error\| falls from 2.01% to 1.11% (measured-Rg) to 0.31% (mass +
measured-Rg).

"Merges rescued" counts merges that succeeded only because a later
partner was tried — the direct measure of what backtracking contributes.
It is zero in the easy regime: when the first choice always works,
backtracking costs nothing and changes nothing.

## Overlap-acceptance defect exposed by the new success rate

Backtracking initially reported 100% success while 9 of 40 aggregates
contained severe residual overlap (worst case 0.61, i.e. two particles
interpenetrating by 61% of their combined radii). Tracing identified a
pre-existing defect that backtracking did not introduce but had stopped
masking: the affected runs previously failed for unrelated reasons and
were discarded before the invalid geometry could be observed.

`overlap.calculate_max_overlap_*_fast` early-terminates: it returns the
first pair whose overlap exceeds the `tolerance` it was handed, not the
maximum. That behaviour is correct and fast for the binary question "is
this placement within `tol_ov`". The adaptive-tolerance path, however,
compared that same return value against `relaxed_tol = 1e-5`, ten times
larger:

```python
cov_max = calculate_max_overlap_pca_auto(..., tolerance=self.tol_ov)  # early exit at 1e-6
...
if intento >= 180 and cov_max <= relaxed_tol:   # compared against 1e-5
    used_adaptive_tol = True                    # accepted!
```

Above `tol_ov` the returned value is only a lower bound. In one observed
case, a placement whose first offending pair overlapped by 2.6e-6 was
accepted as within relaxed tolerance while a different pair overlapped
by 0.43.

Four call sites carried this flaw (two in `pca_agg.py`, two in
`cca/fallbacks.py`). All four now re-evaluate the overlap with the early
exit set at the threshold actually being compared against
(`PCAggregator._true_overlap_at` and its CCA equivalents) before
accepting: either no early exit occurs and the value is the true
maximum, or it exceeds the threshold and the placement is rejected.

Measured in isolation on PCA subcluster generation (12 particles,
hard-regime parameters, seeds 0–399): 6 of 174 successful subclusters
carried internal overlap up to 0.75 before the fix; 0 of 169 after.
End-to-end, all 40 hard-regime aggregates are now clean to machine
precision (max residual overlap ~4e-15, i.e. point contact).

This defect is a plausible root cause of
[catalog_overlap_leak.md](catalog_overlap_leak.md): PCA emitted
overlapping subclusters, and CCA propagated them untouched, since CCA
only checks cluster-against-cluster overlap and never re-validates
within a cluster. The leak's confirmed example, however, is a
monodisperse `densify_retry` cluster, which this mechanism does not
obviously cover; that page's open questions are not all closed by this
fix.

## Choice of Γ form

There is deliberately no "masses vs. counts" configuration flag. Mass
weighting is expressed entirely through the optional per-particle
`densities` argument, and the form of Γ each stage solves is treated as
a property of that stage's geometry rather than a user preference.
Three weightings are distinguishable:

| Weighting | Meaning | Where |
|---|---|---|
| counts (`n₁, n₂, n₃`) | every particle equal regardless of size | PCA |
| mass, uniform density (∝ r³) | weight by volume — `densities=None` | CCA |
| mass, per-particle density (∝ r³ρ) | full heterogeneous case | CCA |

`densities=None` is therefore not the count form: it is volume
weighting, the physically appropriate default. Counts are only reachable
as PCA's internal behaviour, where they are required (below).

The remaining Γ option is `cca_gamma_measured_rg` (default `False`):
feed each cluster's measured Rg (Eq. 4, including the per-particle
gyration term) into the next Γ instead of re-deriving it from the
scaling law, so that deviations introduced by the 1.10 pairing
relaxation factor and the adaptive tolerance cannot accumulate
uncorrected. This improves accuracy in the easy regime (2.01% → 1.11%)
but costs hard-regime success (100% → 80%): measured Rg for a frustrated
cluster runs below the scaling-law value, which raises Γ, which causes
the pairing gate to reject more edges (pass-throughs rise from 175 to
581).

For reference, the mass-vs-count comparison measured before the count
form was retired from CCA (150 hard-regime seeds): counts reached
146/150 success at 1.74% mean \|Rg error\|, masses 138/150 at 1.22%.
Masses trade a few points of success for better fidelity, and are the
only form that can represent heterogeneity.

### PCA retains counts while CCA uses masses

The original Fortran splits the Γ form by stage (`PCA_cca.f90`'s
`Gamma_calculation` takes `n1, n2, n3`; `CCA_module.f90` uses
Σ(4π/3)r³), which initially appeared to be an inconsistency worth
reporting upstream. Measurement indicates otherwise — applying the mass
form to PCA is catastrophic:

| PCA Γ form | Subclusters built (σ=1.9, N=12, 150 seeds) |
|---|---:|
| counts (default) | 93/150 (62.0%) |
| masses | 1/150 (0.7%) |

Eq. 6's mass moments are only consistent with a count-derived
scaling-law Rg when both bodies are aggregates that law describes. In
PCA the second body is a single monomer: its scaling-law Rg is
meaningless at n=1, while its mass can rival the entire growing
cluster's under a wide size distribution. Mixing the two bases produces
Γ values that admit almost no candidates.

The Fortran's split is therefore necessary rather than accidental, and
is hard-coded per stage rather than exposed as a flag that could be set
to a non-working value. It governs only which scalars enter Γ:
everything mass-weighted in PCA's own bookkeeping (`self.mass`,
`self.m1`, the running center of mass) is density-aware regardless, so
supplied densities still shape the subclusters.

## Per-merge statistics

`event_log_path` writes one JSONL record per merge attempt
(`pyfracval/event_log.py`): round, pool size, both cluster sizes, Γ,
`sum_rmax`, candidate pairs available and tried, rotations used, best
overlap reached, outcome, and the number of offending particles.
`attempt_index` distinguishes first-partner from later-partner
successes. The same log carries PCA-failure and per-run records — see
[event_logging.md](event_logging.md). Off by default; nothing is opened
or built when unset.

`benchmarks/analyze_event_log.py` aggregates these records. Over 25
hard-regime trials (412 merge attempts, 24/25 aggregates completed):

```
Outcome breakdown            By CCA round
  stuck              60.2%     round  success  failure  fail rate
  failed_overlap     39.6%       1      122      103      45.8%
  failed_no_candidates 0.2%      2       64       42      39.6%
                                 3       31       16      34.0%
Merges rescued by trying        4       24        3      11.1%
a later partner: 65 (26.2%      5        7        0       0.0%
of all successes)
```

Three observations follow:

1. Failures are overwhelmingly `failed_overlap`, not
   `failed_no_candidates`: candidate pairs exist, and none of them
   work. This is a geometry problem rather than a search-coverage
   problem, consistent with the candidate-ordering experiments in
   [experiments.md](experiments.md) showing no effect.
2. Failures are not near-misses. The best overlap a failing merge
   reaches has median 0.126 — two particles overlapping by an eighth of
   their combined radii — against a `tol_ov` of 1e-6. Finer rotation
   sampling cannot close a gap of that size, which explains why
   broadening the rotation search never helped.
3. Failure rate falls monotonically with round (45.8% → 0%). Round 1,
   merging the fresh PCA subclusters, concentrates the difficulty —
   confirming quantitatively what
   [pairing_frustration.md](pairing_frustration.md) found by other
   means, and explaining why backtracking, which operates within a
   round, is effective.

The Γ feasibility margin `(sum_rmax - gamma)/sum_rmax` overlaps almost
completely between successes (median 0.432) and failures (median
0.462) — direct evidence that the cheap upfront gate cannot predict
sticking, and hence why reacting to real outcomes outperforms
pre-filtering.

## Implementation notes

**Densities.** Per-particle densities are optional
(`np.ndarray | None`; `None` means uniform, and every density-aware
quantity then reduces exactly to its single-material form). They are
accepted by `PCAggregator`, `Subclusterer`, `CCAggregator`, and
`run_simulation(..., densities=...)`. Mass-based Γ exists primarily for
this case: polydispersity alone makes mass a steeper function of
radius, but heterogeneity breaks the function entirely, and no count-
or radius-derived weighting can then place the center of mass or radius
of gyration correctly.

The implementation invariant is that a density follows its particle,
never its array slot. This is not automatic anywhere in the pipeline:
PCA swaps particles between indices, subclustering splits and
reassembles them, CCA reorders and concatenates clusters every round,
and drop-rescue removes some. Concretely:

- PCA swaps `initial_densities` in lockstep with
  `initial_radii`/`initial_mass` and records a placement-ordered
  `self.densities` — callers cannot reconstruct the output order from
  the input order.
- `Subclusterer` splits densities per subcluster and reassembles from
  each PCA run's own output ordering.
- CCA rebuilds densities at the merge boundary in
  `_attempt_pair_merge` rather than threading them through every
  sticking routine; this is sound because every path (rigid, soft
  relaxation, FFT docking, drop-rescue) emits rows as
  `[cluster1..., cluster2...]` in the parents' order. A length mismatch
  raises rather than silently misattributing.
- `run_simulation` shuffles radii and densities under one shared
  permutation.

`tests/test_densities.py` tests this invariant directly.

**Per-aggregate quality record.**
`pyfracval/quality.py::compute_aggregate_quality` runs unconditionally
in `main_runner.run_simulation` before saving, recording
`max_residual_overlap`, `n_overlapping_pairs`, `overlap_ok`,
`measured_rg` and `rg_error_pct` into `AggregateProperties` — one O(N²)
pass against a generation costing seconds to minutes. This is the
structural guard the catalog leak lacked: `success` previously meant
only that PCA+CCA reached the requested particle count, which is how
invalid geometry could reach the catalog labelled successful. Overlaps
are counted above a 1e-12 floor, since point-touching particles sit at
~1e-15 from floating-point round-off and would otherwise flag every
healthy aggregate.

**Regression coverage.** `tests/test_aggregate_quality.py` pins the
three seeds that reproduced the overlap-acceptance defect, plus a
60-seed sweep.
