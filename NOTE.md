# NOTE.md — Algorithm audit, prior findings, and directions toward a "FracVAL V2" paper

Written 2026-07-30. Purpose: consolidate (a) a faithfulness audit of the
Python rewrite against the original Fortran FracVAL and the Moran et al.
2019 paper, (b) everything already tried and measured in this repo, and
(c) candidate directions — both for improving the tool and for justifying
a publication. Companion to the evidence pages under `docs/source/`
(indexed in `docs/source/knowledge_base.md`); this file is the top-level
narrative those pages plug into.

References: paper `docs/moran2019.pdf` (markdown: `papers/fracval.md`),
Fortran original `docs/FracVAL/*.f90`.

---

## 1. Faithfulness audit: Python vs. Fortran vs. paper

### 1.1 Faithful (verified line-by-line)

- **Pipeline structure.** PCA subclusters (N_sub = 5 / 0.1·N / 50 by the
  paper's Step-2 rule) → hierarchical CCA halving rounds with the odd
  cluster passing through (`CCA_module.f90:47-195` ↔
  `cca/aggregator.py::_run_iteration`).
- **Γ feasibility gate for pairing** (`R_max1 + R_max2 ≥ Γ12`, paper
  Step 3) — `CCA_Generate_pairs` ↔ `cca/pairing.py::_generate_pairs_greedy`
  (modulo the 1.10 relaxation factor, see §1.2).
- **Candidate matrix** (paper Eq. 10 + Appendix C cases 1–3, both
  `Ext_case` variants): `CCA_Random_select_list` ↔
  `cca/candidates.py::_cca_select_candidates`, vectorized but logically
  identical.
- **Sticking geometry** (paper Step 4, stages 1–3): translate CM2 to
  distance Γ along CM1→s1; place s1 via sphere–sphere intersection
  (Appendix D) or spherical-cap sampling (Appendix C); rotate cluster 1
  by Euler–Rodrigues; place s2 on intersection of the contact sphere
  (r_s1+r_s2 around s1) with the d_s2 sphere around CM2; rotate cluster 2.
  `CCA_Sticking_process_v1`/`Random_point_SC`/`Spherical_cap_angle` ↔
  `cca/sticking.py::_cca_sticking_v1`, `geometry.two_sphere_intersection`,
  `geometry.random_point_sc`/`spherical_cap_angle`. (The `ext_case=1` path
  was a genuine porting hole until mid-2026 — now fixed and tested, see
  TODO "Done".)
- **Overlap metric and tolerance semantics.** C_ij = ((R_i+R_j) − d_ij) /
  (R_i+R_j), accept when max ≤ tol_ov — identical.
- **Rg is never measured during CCA.** Both implementations feed the
  *scaling-law* Rg (geometric-mean radius · (n/kf)^(1/Df), Eq. 9) for
  clusters 1, 2 and 3 into the Γ equation — not the geometric Rg of the
  actual coordinates. This is per the paper's Step 3 and is worth stating
  explicitly in any methods section, because it means Γ preservation is
  exact *by construction* only insofar as the sub-clusters themselves obey
  the scaling law.
- **PCA.** Count-based Γ (see §1.2 — the Fortran PCA itself uses counts),
  rg3 ≥ rg1 guard, geometric-mean radius over the **full N-particle
  population** for rg3 (`PCA_cca.f90:50` sums log(R) over all N; Python
  passes `all_radii=self.initial_radii`), monomer Rg = √0.6·r, and the
  "swap monomer k with an untried one when no candidate works"
  reordering (`Search_list` ↔ `pca_agg.py` swap logic) — all faithful.

### 1.2 Deviations found (ranked by potential impact)

1. **Masses vs. particle counts in the CCA Γ equation — the headline
   discrepancy.** The paper's central contribution is Eq. (6):
   `m²Rg² = m(m₁Rg₁² + m₂Rg₂²) + Γ²m₁m₂` with true (∝ r³) masses; Eq. (7)
   with counts is the monodisperse Filippov special case whose direct
   polydisperse use (Skorupski/FLAGE) the paper explicitly criticizes.
   The Fortran is split:
   - Fortran **CCA** uses true masses (`CCA_module.f90:301-302, 468-469`,
     m from `CCA_AGG_properties2` = Σ(4π/3)r³) → Eq. (6). ✔ paper
   - Fortran **PCA** uses counts (`PCA_cca.f90:221-222`,
     `Gamma_calculation(rg1,rg2,rg3,n1,n2,n3,…)`) → Eq. (7). ✘ paper
     (an inconsistency in the *original* code, worth mentioning to Moran)
   - Python `fractal.gamma_calculation(heuristic=True)` (the default, and
     no call site overrides it: `cca/pairing.py:26`, `pca_agg.py:223`)
     replaces m₁,m₂,m₃ with n₁,n₂,n₃ → **Eq. (7) everywhere, including
     CCA**. So the Python CCA is faithful to the Fortran *PCA*'s
     heuristic but not to the Fortran *CCA*, and not to the paper.

   Consequence: for polydisperse primary particles (σ_g > 1) the rewrite
   is, strictly speaking, running the Filippov/FLAGE variant that the
   FracVAL paper was written to supersede — per-aggregate Df/kf
   preservation may silently degrade toward ensemble-only preservation.
   For monodisperse input the two are identical, so all monodisperse
   validation stays valid. **Action:** expose `heuristic` as a config
   flag, A/B the hard-regime boundary sweep with mass-based Γ, and check
   per-aggregate Rg error vs. σ_g. Note the boundary maps in
   `docs/source/hard_regime_boundary_sweep.md` show the collapse boundary
   moving with σ — a mass-correct Γ changes Γ most exactly where σ is
   large, i.e. exactly in the region where we collapse. It may not fix
   frustration (it could even shrink feasibility), but right now our
   polydisperse numbers characterize the wrong equation.

2. **Pairing relaxation factor.** `_generate_pairs` accepts pairs with
   Γ < 1.10·(R_max1+R_max2); the Fortran is strict (`Γ < R_max1+R_max2`).
   Trades exactness for success rate; logged when triggered, but is a
   methods deviation to disclose.

3. **Adaptive overlap tolerance.** After 180 rotations Python accepts
   overlap ≤ 1e-5 regardless of tol_ov; Fortran never relaxes. Same
   category as (2): pragmatic, small, but must be disclosed (and is a
   candidate suspect for the open catalog-overlap-leak bug, though that
   leak's confirmed example is far larger than 1e-5).

4. **Rotation-angle sequence.** Fortran draws 359 *uniform-random*
   angles θ per (s1,s2); Python sweeps a deterministic golden-angle
   (Fibonacci) sequence — strictly better coverage of the circle for the
   same budget, benchmarked as equivalent-or-better. Intentional
   improvement; keep, document.

5. **Retry-state semantics.** On candidate exhaustion the Fortran
   *carries the already-rotated cluster coordinates into the next
   attempt* (COR1/COR2 are re-packed from the mutated X1/X2 —
   `CCA_module.f90:579-590`), so successive attempts random-walk the
   orientation space. Python restarts every candidate pair from the
   pristine input pose and walks a flat shuffled list of all (i,j)
   feasible pairs, instead of Fortran's "random row s1, then cycle s2
   within the row, zero the row, pick a new s1" nesting. Different
   sampling distribution over poses/pairs, same search space. Python's is
   cleaner and reproducible; keep, document.

6. **Initial-contact-point refinement.** After sampling the intersection
   point x, Fortran displaces x by r_s1 *toward CM1* and rotates s1's
   radial direction onto it (`CCA_module.f90:913-942`); Python displaces
   from the *particle's current center* toward x
   (`cca/sticking.py:149-163`). Since only the direction from CM1 enters
   the rotation, constraints (Γ distance, point-touch, via later stages)
   are unaffected — but the orientation *distribution* differs slightly.
   Harmless in practice; note for completeness.

### 1.3 Bugs in the original Fortran (good "V2 paper" material)

- **Last-candidate exclusion bias.** Every random selection uses
  `1 + INT((size-1)*rand)` with rand ∈ [0,1)
  (`CCA_module.f90:744,785`, `PCA_cca.f90:304,349`): the *last* eligible
  candidate in the packed list has probability 0 of being chosen whenever
  more than one exists, and the rest are uniform over size−1. A genuine
  sampling bug (should be `1 + INT(size*rand)`). Python uses
  `rng.integers(len)` — correct.
- **acos clamp bug in the retry rotation.** `CCA_module.f90:1319-1323`
  clamps *both* out-of-domain sides of the dot-product ratio to
  `acos(1.0) = 0`: a numerically anti-parallel pair (ratio < −1) yields a
  0 rotation instead of π. Python clips to [−1, 1] correctly.
- **Division-by-C in the intersection-plane basis.** `i_vec = (1, 1,
  −(A+B)/C)/‖·‖` (`CCA_module.f90:1079-1082`, `PCA_cca.f90:407-410`)
  blows up when the plane normal has C ≈ 0. Python's kernel constructs
  the basis robustly.
- **PCA/CCA Γ inconsistency** in the original (counts vs. masses, §1.2) —
  arguably a bug in FracVAL itself given the paper's Eq. (6).

These are exactly the kind of "we audited, found, and fixed" items that
justify a V2 code paper independent of any algorithmic novelty.

---

## 2. What has already been tried (map of the evidence)

Full details live in `docs/source/`; one line each here. Hard regime
throughout: Df=2.25, kf=0.95, σ_g=1.9, N=128 unless noted.

| Idea | Where | Outcome |
|---|---|---|
| Leaf-monomer candidate preference (`leaf_soft/score/hybrid`) | `experiments.md`, `experimental/candidate_policies.py` | No measurable difference |
| Extra rotations: rotate both clusters, jitter, coarse grids | `experiments.md`, `experimental/retry_modes.py` | Identical success, identical timing |
| Pair prefilters (bounding volume, surface accessibility) | `experiments.md`, `experimental/pair_prefilters.py` | No improvement |
| Γ-expansion (relax Γ upward on failure, ≤3 steps) | `experiments.md`, `experimental/gamma_expansion.py` | No improvement |
| FFT rigid-body docking (64³/128³) | `experiments.md`, `experimental/fft_docking.py` | No improvement |
| Soft potential relaxation | `experiments.md`, `experimental/soft_relaxation.py` | Indistinguishable from baseline |
| **Densification** (generate at easy Df/kf, densify to target) | `experiments.md`, `densify.py` | **100% success, ~20× faster, better Rg accuracy** |
| Matching-based pairing over the cheap Γ-feasibility graph | `matching_pairing.md`, `cca/matching.py` | +0.2pp over 4200 trials = noise; graph too optimistic |
| Overlap-failure census (count/identify offending particles) | `overlap_failure_census.md`, `overlap_statistics.py` | Implemented; median 9/24 particles implicated at N=128, 20/100 at N=512 |
| Drop-a-few-particles rescue | `drop_rescue.md`, `cca/rescue.py` | Default budget: zero effect. Relaxed budget: 2.5%→7.5% single-shot, no Rg penalty on (tiny) rescued sample |
| Df/kf/σ/N boundary map (4200 trials) | `hard_regime_boundary_sweep.md` | Collapse at Df≈2.3 (σ=1) → 2.2 (σ=1.5) → 2.0–2.1 (σ=1.9); sharp Df×kf interaction; N sharpens the boundary |
| Numba JIT kernels, incremental active-set overlap | `cca_kernels.py`, `sticking.py` | Production path; performance only |

The failure-mechanism diagnosis (`pairing_frustration.md`) is the load-
bearing result: hard-regime failures are ~97% *pairing* failures, not
search failures — the feasibility graph over the real sticking outcomes
is dense (median ~70% of pairs work), every failure happens at CCA round
1, and greedy first-fit simply picks a bad partner and aborts the round.
Search-space enhancements can't help because no orientation exists *for
that pair at that Γ*; a different partner usually exists.

Answers to the specific "things we remember trying" from the project
history:

- *"Leaf monomers as sticking candidates — verify what the original
  does."* Verified: the original picks **uniformly at random among all
  geometrically feasible monomers** (the Eq.-10 shell test), interior
  ones included — there is no leaf logic in FracVAL. Our leaf-preference
  policies were tried and don't move the needle (above), consistent with
  the diagnosis that candidate choice isn't the binding constraint.
- *"Rotating both aggregates about the CM–candidate axis."* Implemented
  as the `alternate`/`dual_jitter` retry modes; no effect. The original
  rotates only aggregate 2 about the intersection circle (s2 around s1).
- *"Log not just success/failure but how many particles intersect."*
  Exists: `cca_overlap_census_enabled` + telemetry counters. Gap: it's
  opt-in and surfaced per-failure, not as a per-merge event log (§4).
- *"If only ~5 particles intersect, drop them."* Exists
  (`cca_drop_rescue_enabled`); works, but the premise ("just a few")
  mostly doesn't hold at N=128, and the budget needs N-aware scaling
  (`min(max_particles, ceil(frac·n))` bottlenecks on the absolute cap at
  large N). See `drop_rescue.md`.

---

## 3. Directions (ranked by evidence × effort)

### 3.1 Backtracking pairing — the evidence-backed fix (tool + paper)

The one lever every measurement points at and nothing has yet pulled:
when pair (i,j) fails to stick, **retry cluster i with its next feasible
partner inside the same round** instead of aborting the whole attempt.
Distinct from the failed matching experiment because it reacts to *real*
sticking outcomes rather than predicting them from the too-optimistic Γ
graph. Round pools are small (~11), failures concentrate at round 1, and
already-stuck pairs in the round can be kept, so worst case is bounded by
a few extra sticking attempts — compare against the current behavior of
discarding *all* successful merges in the round and restarting PCA from
scratch (up to 20×). Expected effect, from the census: most of the ~97%
"rescuable" failures convert to successes; the boundary maps in
`hard_regime_boundary_sweep.md` are the before/after benchmark. Already
tracked in TODO; this is the highest-value implementation item.

### 3.2 Mass-based Γ restoration (faithfulness + polydisperse correctness)

Flip `heuristic=False` behind a config flag (default TBD after
measurement), rerun the σ=1.5/1.9 slices of the boundary sweep and a
per-aggregate Rg-error check. Cheap (the code path already exists), and
required before any publication claims "implements FracVAL": right now
the polydisperse Γ is Filippov's, not Moran's (§1.2). Also scientifically
interesting on its own — nobody has published how Eq. (6) vs. Eq. (7)
changes the *feasible (Df, kf, σ) region*.

### 3.3 Densification as the headline V2 method (paper core candidate)

It is already the only qualitative winner (100% success where rigid CCA
collapses, 20× faster, better Rg accuracy). What's missing to make it
publishable rather than a code feature:

- **Validate f(r), not just Rg.** The FracVAL paper's own validation
  metric is the density–density correlation function (Eq. 14–15) and its
  Df−3 slope. An aggregate can match Rg while having the wrong internal
  structure. Implement the paper's f(r) estimator (aggregate-copy
  displacement + analytic sphere-intersection volumes) and compare
  densified vs. natively-generated aggregates in the regime where both
  exist (e.g. Df=2.1–2.2), plus anisotropy (gyration-tensor eigenvalue
  ratios) and coordination-number distribution.
- If densified aggregates pass f(r), the paper story writes itself:
  *"rigid tunable CCA has a hard feasibility boundary (we map it, §3.5);
  beyond it we generate at feasible parameters and densify, preserving
  the scaling law and correlation structure"* — a genuine extension of
  the tunable-algorithm family beyond its known Df≈2.2–2.5 ceiling into
  Df→3 with polydisperse PPs.

### 3.4 Γ error-budget compensation ("closed-loop" scaling-law control) — new idea

The archived Γ-expansion relaxed Γ unilaterally and lost exactness. The
untried variant: allow Γ_k → Γ_k(1+ε_k) when a merge is frustrated, but
**track the induced Rg error and compensate at subsequent merges** by
solving Eq. (6) for the Γ that restores the cumulative target
(equivalently: feed the *measured* Rg of the just-built cluster into the
next round's Γ instead of the scaling-law Rg, so errors cannot
accumulate — note §1.1: today Rg is never measured, so per-merge errors
from the relaxation factor and adaptive tolerance already accumulate
uncorrected). Two levels:
  (a) *Measured-Rg feedback* alone is nearly free and corrects the
      existing 1.10-factor deviations — worth doing regardless.
  (b) *Deliberate slack + compensation* turns hard-frustrated merges
      into soluble ones while keeping the *final* aggregate exactly on
      the scaling law. Cheap to implement inside `_calculate_cca_gamma`;
      validate with per-aggregate Rg error + f(r).

### 3.5 Analytical/empirical feasibility criterion (paper-grade novelty)

Moran 2019 only says aggregates are generable "as long as the pair of Df
and kf falls in the valid range" — the range is uncharacterized in the
literature. We already own the empirical map (4200-trial boundary
sweep). The step to novelty: a *predictive* criterion — e.g. compare the
required contact distance Γ(n₁,n₂,σ) against the available surface shell
(R_max distributions, mean feasible-candidate count from the Eq.-10
shell), yielding a closed-form or semi-empirical boundary
Df_max(kf, σ, N). Even a fitted scaling law with a geometric
interpretation, validated against the sweep, would be a citable
contribution and would let the tool *warn before running* instead of
failing after 20 retries.

### 3.6 Smaller items

- **N-aware drop-rescue budget** (make the relative cap actually engage
  at N≥512), plus optional backfill; re-benchmark.
- **Late-round failure probe** (TODO item): force/wait for a failure
  between two large clusters to test whether drop-rescue's original
  motivation holds there.
- **Asymmetric merges as a frustration escape:** when a round-1 pool is
  frustrated, allow unequal-size pairings (Γ feasibility depends on
  n₁/n₂ balance). Paper's Appendix B shows final f(r) is insensitive to
  sub-cluster details, which is the license to relax strict halving.
  Untried; medium effort.
- **Fix the catalog overlap leak** (open TODO): success-flagged clusters
  with severe overlaps would undermine any published dataset. Must be
  closed before paper-grade data generation. Suspects to rule out:
  adaptive tolerance path (§1.2.3), drop-rescue accounting,
  densify_ok=False handling (confirmed real, separate bug).

---

## 4. Instrumentation for paper statistics

Mostly built; the gap is persistence and uniformity, not collection.

Exists today: per-class candidate attempt/success counters, retry-mode
stats, active/full overlap-check telemetry, opt-in overlap census at
failure (offending-particle counts/indices/depths), drop-rescue and
fallback counters, failure-stage attribution (PCA/CCA/timeout) in
`run_simulation`.

Missing for "build statistics on what is happening":

1. **A per-merge JSONL event log** (opt-in flag): one record per sticking
   attempt with round, pool size, (n₁, n₂), Γ, R_max sum, feasible-pair
   count, candidate attempts used, rotations used, min overlap achieved,
   outcome ∈ {stuck, stuck_relaxed_tol, failed_no_candidates,
   failed_overlap, rescued_drop(n), …}, and on failure the census
   summary. This turns every production run into sweep data and is what a
   paper's statistics section would be built from.
2. **Always-on cheap failure summary** (not just when census flag set):
   at minimum offending-particle count and worst overlap at give-up —
   the census is already O(n₁·n₂) once per failure; cost is negligible
   against a failed 360-rotation search.
3. **Per-aggregate final-quality record**: measured Rg vs. scaling-law
   Rg error, max residual overlap (this doubles as the catalog-leak
   guard), n_particles_dropped — written into the saved metadata for
   *every* aggregate, success or not.

---

## 5. Suggested order of attack

1. Backtracking pairing (§3.1) + merge event log (§4.1) — one PR, since
   backtracking needs the event log's bookkeeping anyway. Benchmark
   against the boundary-sweep grid.
2. Mass-based Γ A/B (§3.2) — small, decides which equation all further
   polydisperse results are reported under.
3. Measured-Rg feedback (§3.4a) + final-quality record (§4.3) — closes
   the catalog-leak class of problems structurally.
4. f(r) validator + densification validation (§3.3) — the paper's
   experimental backbone.
5. Feasibility-boundary modeling (§3.5) — analysis on data we largely
   already have.

Items 1–3 improve the tool regardless of publication; 3.3 + 3.5 + the
audit findings (§1) are the paper.
