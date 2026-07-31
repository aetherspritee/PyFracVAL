# Structural Validation via the Correlation Function f(r)

Accuracy claims in this project had previously rested on the radius of
gyration. Rg is a single number, and an aggregate can match it while
having a substantially different internal structure. FracVAL's own
validation (:cite:`Moran2019FracVAL` §4.2) therefore uses the
density–density correlation function `f(r)`, whose log–log slope should
approach `Df − 3`. `pyfracval/correlation.py` implements this
estimator; this page reports its validation and its application to
natively-built and densified aggregates.

The principal finding concerns densification: densified aggregates do
not reach the requested fractal dimension, and — prior to the fixes
described below — carried severe residual overlap while being reported
as successful.

## Method

The estimator follows the paper's Eqs. 14–15: displace a copy of the
aggregate by `r` in a random direction, compute the volume shared with
the original analytically (the exact two-sphere lens formula, so no
binning or kernel choice enters), average over orientations, and
normalize by the aggregate's own volume. Radii are spaced geometrically
from `r_p/10` to `3.5 Rg`. A k-d tree prunes the pair search to
genuinely overlapping pairs, which makes the paper's `n_or = 300`
affordable at N=1024.

The implementation was validated against cases with known answers
before use (`tests/test_correlation.py`): the lens formula against its
closed form in all three regimes, and a uniformly filled ball, which
must return `Df ≈ 3` and returns 3.0 ± 0.6.

The fit window is constrained on both sides. Below about `2 r_p` the
curve reflects single-particle overlap rather than aggregate structure
(as noted in the paper), and beyond about `Rg` finite size cuts the
power law off. `fit_correlation_slope` uses only the range between and
reports `n_points`, so a window too short to support a fit is visible
rather than silently producing a slope from two samples.

## Results

`benchmarks/correlation_validation.py`, N=512, 6 seeds per arm.

| Arm | Target Df | Df from f(r) | Error | mean \|Rg err\| | Overlap-invalid |
|---|---:|---:|---:|---:|---:|
| native, Df=1.8, σ=1.5 | 1.80 | 1.64 ± 0.05 | −0.16 | 1.16% | 0/6 |
| native, Df=2.1, σ=1.5 | 2.10 | 2.03 ± 0.04 | −0.07 | 1.54% | 0/6 |
| densified → Df=2.1 | 2.10 | 1.52 ± 0.05 | −0.58 | 0.26% | 6/6 |
| native, Df=2.3, σ=1.9 | 2.30 | 2.37 ± 0.00 (n=1) | +0.07 | 5.09% | 0/6 |
| densified → Df=2.3 | 2.30 | 1.83 ± 0.06 | −0.47 | 1.72% | 6/6 |

### Densified aggregates match Rg but not structure

The comparison is controlled: same estimator, same N, same target, same
fit window, so any residual estimator bias cancels between arms. At
Df=2.1 a natively built aggregate measures 2.03 (−0.07); a densified
one measures 1.52 (−0.58). Both densified arms land far closer to their
*source* Df (1.8) than to their target.

The direction of the Rg column is instructive: densified aggregates
show *better* Rg agreement than native ones (0.26% vs. 1.54%). This is
precisely the failure mode Rg-only validation cannot detect.
Densification compresses radially until Rg reaches its target, so Rg
agreement is close to guaranteed and carries almost no information,
while the resulting mass distribution differs substantially from a
Df=2.1 fractal.

### Densified aggregates were physically invalid

Every densified aggregate carried residual overlap of 43–69% —
particles interpenetrating by more than half their combined radii —
while native aggregates measured exactly zero. Three specific defects
were identified, and they constitute the densify-path root cause of
[catalog_overlap_leak.md](catalog_overlap_leak.md):

1. `densify_aggregate` returned `True` as soon as the Rg error was
   within tolerance. It called `resolve_overlaps` and logged the
   verdict, but omitted it from the return value. Radial compression
   creates overlaps faster than the push-apart step removes them, so
   overlap resolution routinely failed without consequence.
2. Its final self-overlap check passed the same array to the
   *two-cluster* CCA overlap helper, which scores every particle
   against itself at distance 0 — an overlap of exactly 1.0 for any
   input. The check could never have been meaningful. (It sat after an
   early return that fired first, which is why the flag came back
   `True` rather than always `False`.)
3. `main_runner` used the densified result identically whether the flag
   was `True` or `False`; the two branches of the `if` were the same
   three lines.

All three are fixed: densification reports success only when overlap
resolution converged, its self-overlap check compares distinct
particles, and a non-converged densification falls back to the
undensified aggregate with an error stating that the requested Df/kf
was not achieved.

## Consequences for densification

[experiments.md](experiments.md) described densification as the one
approach that changed hard-regime outcomes qualitatively (100% success,
~20× faster than rigid search). That conclusion is withdrawn:

- the 100% figure counted aggregates that were geometrically invalid;
- it was scored on Rg agreement, which densification optimizes directly
  and which therefore cannot distinguish a genuine Df=2.1 aggregate
  from a compressed Df=1.8 one.

Densification remains opt-in (`densify_enabled`, default off) and now
fails loudly rather than silently. Making it produce correct structure
would require a compression scheme that preserves the correlation
structure rather than only the second moment — a substantial piece of
work rather than a parameter adjustment.

The practical impact of the withdrawal is limited, because the original
motivation for densification has largely lapsed:
[boundary_sweep_v2.md](boundary_sweep_v2.md) shows backtracking
reaching the hard regime directly, with valid geometry and correct
structure (native Df=2.3 measures 2.37 from f(r)).

## Limitations

The native arms show errors of −0.16, −0.07 and +0.07 — small, not
uniformly signed, and consistent with the finite-size effects the paper
describes (f(r) approaches the ideal slope only for large N; at N=512
the usable fit window spans well under a decade). These are large
enough to matter for quoting an absolute Df from f(r), and far too
small to account for the −0.5 deviation of the densified arms. An
absolute-Df application would require larger N and the paper's full
`n_or = 300`.
