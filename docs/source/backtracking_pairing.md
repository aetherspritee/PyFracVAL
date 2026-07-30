# Backtracking CCA Pairing (and the overlap-acceptance bug it exposed)

[pairing_frustration.md](pairing_frustration.md) diagnosed that ~97% of
hard-regime CCA round failures had a *different* pairing of the same
cluster pool that would have worked.
[matching_pairing.md](matching_pairing.md) then implemented the obvious
response - choose better pairs up front via exact maximum-cardinality
matching - and measured **no improvement** (+0.2pp over 4200 trials).
The reason it could not work: matching optimizes over the cheap
gamma-feasibility graph, which is necessary-but-not-sufficient for
sticking. It cannot predict which feasible-looking pairs actually stick.

Backtracking is the design that evidence pointed to instead. It does not
try to predict anything: it reacts to the *real* sticking outcome.

## Mechanism

`pyfracval/cca/aggregator.py::_run_iteration_backtracking`. Per round:

1. Build the same cheap gamma-feasibility graph.
2. Repeatedly take the **most-constrained** cluster (fewest remaining
   feasible partners) and attempt to stick it against its partners in
   increasing order of how constrained *they* are, up to
   `cca_backtracking_max_partners` (default 4).
3. On a real sticking failure, mark that edge dead and try the next
   partner. On success, both clusters leave the pool.
4. A cluster that exhausts its partners is **passed through** to the next
   round unmerged rather than failing the attempt
   (`cca_backtracking_pass_through`, default on). This is what the odd
   cluster in an odd-sized pool has always done; it is now also available
   to a cluster that is merely frustrated this round.
5. A round that merges *nothing* still fails, so a stalled pool cannot
   loop forever.

The critical difference from the old behaviour is what happens to the
rest of the round. Previously a single failed pair set `not_able_cca` and
discarded **every other successful merge in that round**, restarting
PCA+CCA from scratch (up to 20 times). Now those merges are kept.

Cost is bounded: at most 4 sticking attempts per cluster over a round
pool of ~11, against a baseline that threw away whole attempts.

## Results

`benchmarks/backtracking_pairing_benchmark.py`, 40 seeds, single-shot
methodology (no internal retry, so one seed maps to one outcome - the
same metric [pairing_frustration.md](pairing_frustration.md) and
[drop_rescue.md](drop_rescue.md) use, and directly comparable to their
2.5% figure).

Hard regime (N=128, Df=2.25, kf=0.95, σ=1.9):

| Config | Success | Merges rescued by backtracking | avg \|Rg error\| |
|---|---:|---:|---:|
| Greedy first-fit (previous default) | 5.0% (2/40) | - | 1.08% |
| **Backtracking (new default)** | **100.0% (40/40)** | 93 | 1.40% |
| Backtracking + mass-based Γ | 95.0% (38/40) | 98 | 0.52% |
| Backtracking + measured-Rg Γ | 80.0% (32/40) | 140 | 1.59% |
| Backtracking + both | 90.0% (36/40) | 149 | 1.67% |

Easy control regime (N=128, Df=1.8, kf=1.0, σ=1.5): every arm is 100%,
so nothing regresses in the safe region. The Γ flags do measurably
improve accuracy there, where success is not at stake: avg \|Rg error\|
falls from 2.01% (default) to 1.11% (measured-Rg) to 0.31% (both).

"Merges rescued" counts merges that succeeded only because a *later*
partner was tried - the direct measure of what backtracking buys. It is
zero in the easy regime: when the first choice always works, backtracking
costs nothing and changes nothing.

## The overlap-acceptance bug this exposed

Backtracking initially reported 100% success while **9 of 40 aggregates
contained severe residual overlap** (worst: 0.61, i.e. two particles
interpenetrating by 61% of their combined radii). Tracing it found a
pre-existing bug that backtracking did not cause - it merely stopped
masking it, because the runs that surfaced it used to fail for unrelated
reasons and be discarded.

`overlap.calculate_max_overlap_*_fast` **early-terminates**: it returns
the first pair whose overlap exceeds the `tolerance` it was handed, not
the maximum. That is correct and fast for the binary question "is this
placement within `tol_ov`". But the adaptive-tolerance path then compared
that same value against `relaxed_tol = 1e-5`, ten times larger:

```python
cov_max = calculate_max_overlap_pca_auto(..., tolerance=self.tol_ov)  # early exit at 1e-6
...
if intento >= 180 and cov_max <= relaxed_tol:   # compared against 1e-5
    used_adaptive_tol = True                    # accepted!
```

Above `tol_ov` the returned value is only a *lower bound*. A placement
whose first offending pair overlapped by 2.6e-6 was accepted as "within
relaxed tolerance" while a different pair overlapped by 0.43.

Four sites had this flaw (two in `pca_agg.py`, two in `cca/fallbacks.py`).
All now re-evaluate the overlap with early exit at the threshold actually
being compared against (`PCAggregator._true_overlap_at` and its CCA
equivalents) before accepting. That comparison is sound: either no early
exit occurs and the value is the true maximum, or it exceeds the
threshold and the placement is rejected anyway.

Measured effect, in isolation on PCA subcluster generation (12 particles,
hard-regime parameters, seeds 0-399): **6 of 174 successful subclusters
carried internal overlap up to 0.75 before the fix; 0 of 169 after.**
End-to-end, all 40 hard-regime aggregates are now clean to machine
precision (max residual overlap ~4e-15, i.e. point contact).

This is very likely the root cause of
[catalog_overlap_leak.md](catalog_overlap_leak.md): PCA emitted
overlapping subclusters, and CCA propagated them untouched, since CCA
only ever checks cluster-against-cluster overlap and never re-validates
*within* a cluster. Note the leak's confirmed example is a monodisperse
`densify_retry` cluster, which this explanation does not obviously cover -
that page's own open questions are not all closed by this fix.

Regression coverage: `tests/test_aggregate_quality.py` pins the three
seeds that reproduced the bug plus a 60-seed sweep.

## The two Γ flags

Both default off; both are faithfulness/accuracy knobs rather than
stability ones, and the table above is the evidence for keeping them
opt-in.

- `cca_gamma_use_mass` - solve Γ with true (∝r³) masses, Moran et al.
  (2019) Eq. 6, instead of substituting particle counts (Filippov et al.
  2000 Eq. 7). The Fortran **CCA** uses masses; the Fortran **PCA** uses
  counts; this port used counts everywhere. Identical for monodisperse
  input, divergent as σ grows. Best measured Rg accuracy of any arm
  (0.52% hard, 1.56% easy), at a small and not-clearly-significant
  success cost (38/40 vs 40/40). PCA deliberately stays on counts so each
  stage matches its own Fortran counterpart.
- `cca_gamma_measured_rg` - feed each cluster's *measured* Rg (Eq. 4,
  including the per-particle gyration term) into the next Γ instead of
  re-deriving it from the scaling law, so deviations introduced by the
  1.10 pairing relaxation factor and the adaptive tolerance cannot
  accumulate uncorrected. Improves accuracy in the easy regime
  (2.01% → 1.11%) but *costs* hard-regime success (100% → 80%): measured
  Rg for a frustrated cluster runs below the scaling-law value, which
  raises Γ, which makes the pairing gate reject more edges (pass-throughs
  rise from 175 to 581).

Because Eq. 6 is an identity in the true masses, the measured-Rg
correction is only exact when `cca_gamma_use_mass` is also set; the two
are best evaluated together.

## Per-merge event log

`cca_merge_log_path` writes one JSONL record per merge attempt
(`pyfracval/merge_log.py`): round, pool size, both cluster sizes, Γ,
`sum_rmax`, candidate pairs available and tried, rotations used, best
overlap reached, outcome, and - when the census is on - how many
particles were offending. `attempt_index` distinguishes "first partner
worked" from "third partner worked". Off by default; nothing is opened or
built when unset.

## Per-aggregate quality record

`pyfracval/quality.py::compute_aggregate_quality` now runs
unconditionally in `main_runner.run_simulation` before saving, recording
`max_residual_overlap`, `n_overlapping_pairs`, `overlap_ok`,
`measured_rg` and `rg_error_pct` into `AggregateProperties`. One O(N²)
pass against a generation costing seconds to minutes.

This is the structural guard the catalog leak needed: `success` used to
mean only "PCA+CCA reached the requested particle count", which is
exactly why invalid geometry could reach the catalog labelled successful.
Overlaps are counted above a 1e-12 floor, since point-touching particles
sit at ~1e-15 from floating-point round-off and counting those would flag
every healthy aggregate.
