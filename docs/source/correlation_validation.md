# Structural Validation via f(r) — and what it revealed about densification

Every accuracy claim in this project so far has rested on the radius of
gyration. Rg is one number, and an aggregate can match it while having
quite the wrong internal structure. FracVAL's own validation
(:cite:`Moran2019FracVAL` §4.2) therefore uses the density-density
correlation function `f(r)`, whose log-log slope should approach
`Df − 3`. `pyfracval/correlation.py` implements that estimator, and this
page reports what it found.

The headline is not the estimator. It is that **densification does not
produce aggregates at the requested fractal dimension**, and that it was
producing physically invalid geometry that the pipeline reported as
successful.

## The estimator

Paper Eqs. 14–15: displace a copy of the aggregate by `r` in a random
direction, compute the volume it shares with the original analytically
(the exact two-sphere lens formula, so no binning or kernel choice
enters), average over orientations, and normalize by the aggregate's own
volume. Radii are spaced geometrically from `r_p/10` to `3.5 Rg`. A k-d
tree prunes the pair search to genuinely overlapping pairs, which is what
makes the paper's `n_or = 300` affordable at N=1024.

Validated in `tests/test_correlation.py` against cases with known
answers before being trusted: the lens formula against its closed form in
all three regimes, and — the real check — a uniformly filled ball, which
must return `Df ≈ 3`. It returns 3.0 ± 0.6.

The fit window is not free. Below about `2 r_p` the curve reflects
single-particle overlap rather than aggregate structure (the paper says
so explicitly), and beyond about `Rg` finite size cuts the power law off.
`fit_correlation_slope` uses only the range between, and reports
`n_points` so a window too short to fit is visible rather than silently
producing a slope from two samples.

## Results

`benchmarks/correlation_validation.py`, N=512, 6 seeds per arm.

| Arm | Target Df | Df from f(r) | Error | mean \|Rg err\| | Overlap-invalid |
|---|---:|---:|---:|---:|---:|
| native, Df=1.8, σ=1.5 | 1.80 | 1.64 ± 0.05 | −0.16 | 1.16% | 0/6 |
| native, Df=2.1, σ=1.5 | 2.10 | 2.03 ± 0.04 | −0.07 | 1.54% | 0/6 |
| **densified → Df=2.1** | 2.10 | **1.52 ± 0.05** | **−0.58** | 0.26% | **6/6** |
| native, Df=2.3, σ=1.9 | 2.30 | 2.37 ± 0.00 (n=1) | +0.07 | 5.09% | 0/6 |
| **densified → Df=2.3** | 2.30 | **1.83 ± 0.06** | **−0.47** | 1.72% | **6/6** |

### Densified aggregates match Rg but not structure

This is the comparison the page exists for, and it is controlled: same
estimator, same N, same target, same fit window. At Df=2.1 a natively
built aggregate measures 2.03 (−0.07); a densified one measures **1.52**
(−0.58). Whatever residual bias the estimator has cancels between the
two.

Note the direction of the Rg column: the densified aggregates have
*better* Rg agreement than native ones (0.26% vs 1.54%). That is exactly
the failure mode Rg-only validation cannot see. Densification compresses
radially until Rg hits its target — so Rg agreement is close to
guaranteed and carries almost no information — while the mass
distribution it produces is nothing like a Df=2.1 fractal. Both densified
arms land far closer to their *source* Df (1.8) than to their target.

### Densified aggregates were physically invalid

Every densified aggregate carried residual overlap of **43–69%** —
particles interpenetrating by more than half their combined radii — while
native aggregates measured exactly zero. This has a specific cause, and
it is the root of [catalog_overlap_leak.md](catalog_overlap_leak.md):

1. `densify_aggregate` returned `True` as soon as the Rg error was within
   tolerance. It called `resolve_overlaps`, logged its verdict, and then
   **left that verdict out of the return value entirely**. Radial
   compression creates overlaps faster than the push-apart step removes
   them, so the overlap resolution routinely failed and nobody noticed.
2. Its final self-overlap check handed the same array to the
   *two-cluster* CCA overlap helper, which scores every particle against
   itself at distance 0 — an overlap of exactly 1.0 for any input. That
   check could never have been meaningful. (It sat after an early return
   that fired first, which is why the flag came back `True` rather than
   always `False`.)
3. `main_runner` then used the densified result identically whether the
   flag was `True` or `False` — the two branches of that `if` were the
   same three lines.

All three are fixed. Densification now reports success only when the
overlap resolution actually converged, its self-overlap check compares
distinct particles, and a non-converged densification falls back to the
undensified aggregate with a loud error saying the requested Df/kf was
not achieved.

## What this means for densification

[experiments.md](experiments.md) called densification "the only approach
that changes the outcome qualitatively: 100% success, roughly 20x faster
than any rigid search variant." That conclusion must now be withdrawn:

- The 100% counted aggregates that were geometrically invalid.
- It was measured on Rg agreement, which densification optimizes directly
  and which therefore cannot distinguish a real Df=2.1 aggregate from a
  compressed Df=1.8 one.

Densification remains opt-in (`densify_enabled`, default off) and is now
honest about failing. Making it actually work would mean a compression
scheme that preserves the correlation structure rather than only the
second moment — a real piece of work, not a parameter tweak.

The practical consequence is smaller than it sounds, because the reason
densification was attractive has largely gone away:
[boundary_sweep_v2.md](boundary_sweep_v2.md) shows backtracking reaches
the hard regime directly, with valid geometry and correct structure
(native Df=2.3 measures 2.37 from f(r)).

## Caveat on the estimator's own bias

Native arms show errors of −0.16, −0.07 and +0.07 — small, not uniformly
signed, and consistent with the finite-size effects the paper describes
(f(r) approaches the ideal slope only for large N; at N=512 the usable
fit window spans well under a decade). These are large enough to matter
if one wanted to quote an absolute Df from f(r), and far too small to
explain the −0.5 seen for densified aggregates. Quoting absolute Df
would want larger N and the paper's full `n_or = 300`.
