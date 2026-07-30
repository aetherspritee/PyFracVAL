# Catalog Overlap Leak: `success=True` Clusters With Severe Residual Overlap

`cluster_index.csv` marks clusters as `success` based only on whether PCA+CCA
reached the target particle count. It says nothing about whether the *saved
geometry* is actually overlap-free. In practice, some `success=True` clusters
contain severe residual overlaps — this page documents one confirmed,
reproducible case (from `configs/densify_retry`) and separates what's
verified from what's still open, so a follow-up investigation doesn't have
to redo the narrowing.

## The symptom

`cluster_data/densify_retry/sigma_1p00__Df_1p80__N_512/fracval_N512_Df1p80_kf1p00_rpg1p0_rpgstd1p00_agg0_20260430-135703.dat`
(seed 1678904704, `cluster_index.csv` row: `densify_retry,1.0,1.8,512,1.0,0,1678904704,True,...`)
has 512 monodisperse particles (radius exactly 1.0, std 0.0) and, checked
directly against the saved coordinates:

```
worst pre-existing (unscaled) gap: -1.023836250350294
pairs already overlapping (before any downstream gap scaling at all): 272
```

i.e. the closest pair sits at center-to-center distance 0.976 with radii
summing to 2.0 — over a full radius of overlap — in the file as saved, no
downstream processing involved. This is what's actually causing
space-weathering's gap-factor scaling (`pyfracval.gap_scaling`,
`mode="average"`) to fail its own overlap-validation safety net downstream:
a heuristic scale factor derived from the mean radius has no way to know
about (or correct for) a pair that starts out already deep in overlap.
`mode="strict"` (KD-tree, exact) does fix the immediate symptom, but the
geometry it's fixing shouldn't have been saved as `success=True` in the
first place.

Repro:

```python
import numpy as np
from scipy.spatial.distance import pdist

data = np.loadtxt("cluster_data/densify_retry/sigma_1p00__Df_1p80__N_512/"
                   "fracval_N512_Df1p80_kf1p00_rpg1p0_rpgstd1p00_agg0_20260430-135703.dat")
coords, radii = data[:, :3], data[:, 3]
dists = pdist(coords)
i, j = np.triu_indices(len(radii), k=1)
gaps = dists - (radii[i] + radii[j])
print(gaps.min(), int((gaps < 0).sum()))
```

## Confirmed bug: `densify_ok=False` is silently accepted

`pyfracval/main_runner.py:316-338` calls `densify_aggregate()`, which
returns `(coords, radii, success)` — `success=False` means its internal
overlap-resolution loop (`densify.py::resolve_overlaps`, iterative
push-apart, gated by `max_push_iters`/`push_patience`) stalled or ran out of
iterations without clearing every overlap. Both branches of the caller do
the same thing with the result:

```python
if densify_ok:
    logger.info("Densification succeeded, using densified coordinates.")
    final_coords = densified_coords
    ...
else:
    logger.warning("Densification did not fully converge; using best result.")
    final_coords = densified_coords   # identical to the success branch
    ...
```

There is no retry, no rejection, and no propagation of `densify_ok` into
the catalog's `success` column (confirmed: `main_runner.py`'s own
`success_flag` docstring defines it purely in terms of PCA/CCA reaching the
target N — densification happens *after* that flag is already decided and
never feeds back into it). Any cluster whose densification step fails to
converge is written to disk and cataloged exactly like a clean one, with a
log warning as the only trace.

**This is a real bug and should be fixed independently of the rest of this
page** — at minimum, `densify_ok=False` should not silently produce a
catalog `success=True` entry. Whether the right fix is "reject and mark
failed", "retry with a larger `max_push_iters`/looser `push_patience`", or
something else is a design call for whoever picks this up.

## What this bug does *not* explain

Checked directly, because it was the obvious first hypothesis: **densify
never ran on the example cluster above.** `densify_aggregate()` early-returns
unchanged, pre-densify coordinates whenever
`rg_current <= rg_target * (1 + rg_rtol)` — and for this cluster:

```
measured rg_current (from saved coords):        31.6375
rg_target for Df=1.8 (the actual target):        32.0000
rg_target for Df=2.0 (densify_source_df, the generation Df): 22.6274
```

`rg_current` (31.64) is already below `rg_target` for Df=1.8 (32.0, within
`rg_rtol`), so `densify_aggregate` hit its no-op path and returned the raw
PCA+CCA output untouched, `success=True`, no compression or push-apart ever
ran. The severe overlaps in this specific file therefore come from PCA/CCA
sticking itself, not from densification — the bug above is real and worth
fixing, but it is a second, independent problem, not the explanation for
*this* example.

## What's still open

`pyfracval/cca/fallbacks.py`'s rotation-search loop (the default sticking
path — `cca_drop_rescue_enabled` is `False` by default and wasn't set by
the `densify_retry` generation config, so drop-rescue isn't in play here
either) enforces `tol_ov` tightly at every stick: the loop only accepts a
placement when `cov_max <= self.tol_ov`, with a single, narrow escape valve
(`relaxed_tol = 1.0e-5`, i.e. 10x the metadata's `tol_ov: 1.0e-06`, only
after `adaptive_tol_threshold` rotation attempts). `pyfracval/pca_agg.py`'s
coarse-scan/bisection placement search is built the same way. Neither path,
read on its own, looks like it should ever accept a placement anywhere near
a full-radius overlap.

That leaves a real, unresolved gap between "every local per-step check
looks like it enforces ~1e-6–1e-5 tolerance" and "the final saved geometry
has 272 pairs overlapping by up to 100%+ of a radius". Candidate places to
look next (not yet checked):

- Whether CCA's per-stick overlap check (`calculate_max_overlap_cca_auto`,
  and the `use_incremental` active-collision-scanning shortcut in
  particular) validates the *new* candidate pair against the **full**
  existing aggregate, or only a subset that could miss a pair.
- Whether PCA-built subclusters are re-validated for *internal*
  consistency before CCA treats them as trusted, already-clean input, or
  whether CCA only ever checks new cross-cluster pairs and implicitly
  trusts each incoming subcluster's own internal state.
- Whether `use_batch_rotation` (the experimental Phase 3 batch-rotation
  path, `pyfracval/cca/fallbacks.py:426-504`) has a different effective
  tolerance/acceptance path than the sequential one walked above.

## Scope

Not yet measured how many other cataloged clusters carry the same problem.
A quick census (`resolve_overlaps`'s `_find_overlaps`/`_self_overlap_pairs_kernel`
kernel in `densify.py`, or the `pdist` check above, run once per file in
`cluster_index.csv` and cross-referenced against `success`) would answer
that cheaply and should probably be the first thing a follow-up does —
right now this page only establishes that at least one cataloged
`success=True` cluster is badly wrong, not how widespread it is.
