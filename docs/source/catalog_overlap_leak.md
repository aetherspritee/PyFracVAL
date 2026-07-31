# Catalog Overlap Leak: `success=True` Clusters With Severe Residual Overlap

`cluster_index.csv` marks clusters as `success` based only on whether
PCA+CCA reached the target particle count; it says nothing about whether
the saved geometry is overlap-free. Some `success=True` clusters contain
severe residual overlaps. This page documents one confirmed,
reproducible case (from `configs/densify_retry`) and separates what has
been verified from what remains open.

```{note}
**Status (2026-07-30).** Two subsequently identified root causes cover
most of this page's territory: the densification acceptance defects
(root-caused and fixed — see
[correlation_validation.md](correlation_validation.md)) and the
overlap-acceptance defect in the adaptive-tolerance path (fixed — see
[backtracking_pairing.md](backtracking_pairing.md)). A per-aggregate
quality record now guards the catalog path. The remaining open item is
to re-run the original configuration and confirm the leak is gone;
until then this page stands as the record of the original observation.
```

## Symptom

`cluster_data/densify_retry/sigma_1p00__Df_1p80__N_512/fracval_N512_Df1p80_kf1p00_rpg1p0_rpgstd1p00_agg0_20260430-135703.dat`
(seed 1678904704, `cluster_index.csv` row:
`densify_retry,1.0,1.8,512,1.0,0,1678904704,True,...`) contains 512
monodisperse particles (radius exactly 1.0, std 0.0). Checked directly
against the saved coordinates:

```
worst pre-existing (unscaled) gap: -1.023836250350294
pairs already overlapping (before any downstream gap scaling at all): 272
```

The closest pair sits at center-to-center distance 0.976 with radii
summing to 2.0 — more than a full radius of overlap — in the file as
saved, with no downstream processing involved. This is the direct cause
of a downstream failure in gap-factor scaling
(`pyfracval.gap_scaling`, `mode="average"`), whose overlap-validation
safety net rejects the result: a heuristic scale factor derived from
the mean radius cannot correct a pair that starts deep in overlap.
`mode="strict"` (KD-tree, exact) does resolve the immediate symptom,
but the geometry being corrected should not have been saved as
`success=True`.

Reproduction:

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

## Confirmed defect: `densify_ok=False` was silently accepted

`pyfracval/main_runner.py` (at the time of this investigation) called
`densify_aggregate()`, which returns `(coords, radii, success)` —
`success=False` meaning its internal overlap-resolution loop
(`densify.py::resolve_overlaps`, iterative push-apart, gated by
`max_push_iters`/`push_patience`) stalled or ran out of iterations
without clearing every overlap. Both branches of the caller did the
same thing with the result:

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

There was no retry, no rejection, and no propagation of `densify_ok`
into the catalog's `success` column (`main_runner.py`'s own
`success_flag` docstring defines it purely in terms of PCA/CCA reaching
the target N; densification happens after that flag is decided and
never fed back into it). Any cluster whose densification failed to
converge was written to disk and cataloged exactly like a clean one,
with a log warning as the only trace. This defect has since been fixed
(see status note above and
[correlation_validation.md](correlation_validation.md)).

## What this defect does not explain

Checked directly, since it was the obvious first hypothesis:
densification never ran on the example cluster above.
`densify_aggregate()` early-returns unchanged, pre-densify coordinates
whenever `rg_current <= rg_target * (1 + rg_rtol)` — and for this
cluster:

```
measured rg_current (from saved coords):        31.6375
rg_target for Df=1.8 (the actual target):        32.0000
rg_target for Df=2.0 (densify_source_df, the generation Df): 22.6274
```

`rg_current` (31.64) is already below `rg_target` for Df=1.8 (32.0,
within `rg_rtol`), so `densify_aggregate` took its no-op path and
returned the raw PCA+CCA output untouched; no compression or push-apart
ran. The severe overlaps in this specific file therefore originate in
PCA/CCA sticking itself, not in densification. The densification defect
above is real but is a second, independent problem, not the explanation
for this example.

## Originally open questions

At the time of this investigation, the sticking paths appeared to
enforce tolerance correctly when read in isolation:
`pyfracval/cca/fallbacks.py`'s rotation-search loop (the default
sticking path — `cca_drop_rescue_enabled` was not set by the
`densify_retry` generation config) accepts a placement only when
`cov_max <= self.tol_ov`, with a single narrow escape valve
(`relaxed_tol = 1.0e-5`, i.e. 10× the metadata's `tol_ov: 1.0e-06`,
only after `adaptive_tol_threshold` rotation attempts), and
`pyfracval/pca_agg.py`'s coarse-scan/bisection placement search is
built the same way. That left an unresolved gap between "every local
per-step check enforces ~1e-6–1e-5 tolerance" and "the final saved
geometry has 272 pairs overlapping by up to a full radius". Candidate
explanations recorded at the time:

- whether CCA's per-stick overlap check (`calculate_max_overlap_cca_auto`,
  particularly the `use_incremental` active-collision shortcut)
  validates a new candidate pair against the full existing aggregate or
  only a subset;
- whether PCA-built subclusters are re-validated for internal
  consistency before CCA treats them as trusted input, or whether CCA
  only checks new cross-cluster pairs;
- whether `use_batch_rotation` (the experimental batch-rotation path)
  has a different effective acceptance path than the sequential one.

The second candidate was subsequently confirmed as a real mechanism:
the adaptive-tolerance comparison accepted early-exit lower bounds as
if they were maxima, allowing PCA to emit internally overlapping
subclusters that CCA then propagated unchecked. See
[backtracking_pairing.md](backtracking_pairing.md) for the analysis and
fix.

## Scope

The prevalence of the problem across the existing catalog was not
measured. A census (`resolve_overlaps`'s
`_find_overlaps`/`_self_overlap_pairs_kernel` kernel in `densify.py`,
or the `pdist` check above, run once per file in `cluster_index.csv`
and cross-referenced against `success`) would answer this cheaply and
is the natural first step of the remaining follow-up, together with
re-running the original `densify_retry` configuration under the fixed
code. This page establishes that at least one cataloged `success=True`
cluster was badly invalid, not how widespread the condition was.
