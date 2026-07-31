# Closing the catalog's coverage gaps

The generated catalog is a grid over `(sigma, Df, N)` — 228 cells, five
aggregates each. An earlier batch left 30-odd cells partly or entirely
empty, which is a problem for a paper: a sweep with holes in it forces
every downstream figure to explain its own missing squares.

The holes turned out not to be physics. Two separate causes, both
artefacts of the tooling.

## Cause 1: the pairing gate measured the wrong distance

Before attempting to stick two subclusters, CCA screens the pair on
whether the required centre-of-mass separation Γ is reachable at all.
That gate compared Γ against `R_max1 + R_max2`, where `R_max` is the
distance from a cluster's centre of mass to its furthest particle
*centre*. The candidate test one level down — paper Eq. 10 — uses
`D+ = d + r`, the distance to the furthest particle *surface*.

The two differ by up to one particle radius per cluster. For a
500-particle aggregate that is a rounding error. For a 3-particle one it
is most of the cluster. Measured at sigma=1.0, Df=1.4, N=8:

```
gamma = 5.502    sumRmax = 4.022    sumRmax + radii = 6.022
gate: False      actual candidate test: True
```

So the gate rejected pairs that the sticking search would have placed
successfully, and it rejected them hardest exactly where the catalog was
emptiest: low Df (large Γ) and small N (few particles, so the radius term
dominates). `cluster_surface_reach()` in `cca/pairing.py` now supplies
the same reach the candidate test uses, and both the greedy and matching
pairing paths use it.

One test moved as a result. `test_zero_progress_round_fails_instead_of_
looping` built its unreachable geometry at Df=1.4/kf=1.4, which is now
correctly feasible; it was re-founded on Df=1.05/kf=0.1.

## Cause 2: kf was inherited from a build that no longer exists

`kf` is not a swept axis. It is chosen per cell so the cell is generable
at all. Those choices came from a sweep run against an older
implementation, and were never revisited — so some cells carried a kf
that no longer works, and others a kf that never worked and only appeared
to under the overlap-acceptance defect fixed earlier in this work.

`scripts/select_kf.py` re-measures the choice against the current code
with one rule applied uniformly to every cell:

> among the kf values that reach the required success rate,
> take the one closest to kf = 1.0

Preferring 1.0 keeps the catalog near the physically typical prefactor
and lets the boundary decide the rest, without hard-coding a direction.
The rule matters more than the values: a catalog where each cell's kf was
picked by a different ad-hoc process is hard to describe in a methods
section.

The first full selection left six cells unfilled, all at Df=1.4 and large
N, two of which had selected kf=1.8 — the top of the ladder — at partial
success. That is the signature of a ladder limit rather than a physical
one: `Rg = a (n/kf)^(1/Df)` means raising kf shrinks Rg, and with it the
separation two subclusters must span, so low Df wants a *large*
prefactor. Extending the ladder to 2.7, the top of the range Moran et al.
explore, fills all six:

```
sigma   Df     N | kf=2.0  kf=2.2  kf=2.4  kf=2.7
  1.0  1.4   384 |  0/3     3/3     3/3     3/3
 1.25  1.4   384 |  3/3     3/3     3/3     3/3
 1.25  1.4   512 |  1/3     3/3     3/3     3/3
  1.5  1.4   512 |  3/3     3/3     3/3     3/3
  1.5  1.4   640 |  0/3     3/3     3/3     3/3
  1.5  1.4  1024 |  3/3     3/3     3/3     3/3
```

## Result

```
combos fully satisfied : 228/228
clusters saved         : 1140/1140
rejected for overlap   : 0
elapsed                : 3.4 min
```

First-attempt success went from 878 clusters per 1300 tasks to 1096 per
1100 — 99.6%. Independent from-disk validation (`validate_cluster_catalog.py`, which
re-reads every `.dat` and recomputes its geometry rather than trusting
the generation-time record) reports 0 files over tolerance, 0 short of N,
228 distinct combos, and **no measurable particle contact anywhere in the
catalog** — worst overlap 0.000e+00 across all 1140 files, checked at a
1e-9 bound that is three orders of magnitude stricter than the `tol_ov`
of 1e-6 the runs were configured with.

One file needed replacing to reach that. The original aggregate at
sigma=1.25, Df=1.60, N=128 carried 5.1e-07 of contact on a single pair —
*inside* the configured tolerance, so the generator was correct to accept
it, and the algorithm was honouring its contract rather than failing.
Regenerating that slot on a later seed (attempt 25, outside the range the
campaign used, so the derived seed cannot collide with an earlier one)
produced an aggregate with exactly zero contact, and that is what the
catalog now carries.

Worth stating plainly for the methods section: that one slot was selected
under a stricter criterion than the other 1139, which were accepted on
the generator's normal `tol_ov` gate. The distinction is not visible in
the finished catalog — every file now measures zero — but the selection
process was not uniform across all 1140, and the seed and attempt number
are recorded in both the file header and `cluster_index.csv` for anyone
reconstructing it.

The validator's default was also changed to check each file against its
own header's `tol_ov` rather than a fixed 1e-9, since a run configured at
1e-6 is entitled to place a pair 1e-7 into each other and judging it at
1e-9 tests a promise nobody made. Pass `--tol 1e-9` for the strict bound;
either way the worst overlap seen is printed unconditionally, so the
number cannot be tuned away quietly.

## What this does *not* claim

The boundary is real; it just sits further out than the tooling suggested.
Cells outside this grid — very low Df at very large N in particular — remain
genuinely hard, and the feasibility model in `pyfracval/feasibility.py`
still describes where. What changed is that none of the 228 cells the
catalog is meant to cover are on the far side of it.
