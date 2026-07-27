#!/usr/bin/env python3
"""Probe: when a hard-regime CCA run fails, was the failure rescuable by a
different pairing choice, or was the cluster pool universally frustrated?

Background
----------
docs/source/experiments.md established that every *search-strategy* fix
tried (rotation modes, candidate scoring, prefilters, gamma-expansion, FFT
docking, soft relaxation) makes no difference to hard-regime success rates
(~17-20%). All of those operate *within* a fixed pairing: given a pair of
clusters and a scaling-law-fixed contact distance gamma_pc, they search for
a collision-free orientation. None of them touch *which* clusters get
paired in the first place - pyfracval/cca/pairing.py's `_generate_pairs` is
a greedy first-fit over arbitrary index order, and when the chosen pair
fails to stick, `aggregator.py::_run_iteration` aborts the *entire* attempt
(pyfracval/cca/aggregator.py:321-326) rather than trying a different
partner for the surviving cluster.

This probe answers: at the exact round where a real attempt failed, does
*any* pairing of that round's cluster pool admit a fully collision-free
matching? Concretely: build a feasibility graph over the round's clusters
(edge (i,j) = sticking i and j succeeds within the normal per-pair budget,
tested with N_RETRIES_PER_PAIR independent attempts to avoid blaming
pairing for what's really just bad luck), then check for a perfect
matching (every cluster paired, or one left over if the pool is odd - CCA
already allows a single cluster to pass through unmerged for a round).

- Matching often exists but greedy first-fit missed it -> pairing choice
  (or backtracking on merge failure) is a real, cheap lever worth building.
- Matching essentially never exists -> the clusters themselves are too
  compact at the scaling-law-fixed gamma_pc; only densification (reshaping
  at an easier Df/kf) escapes, exactly as experiments.md already found.

Determinism: the census consumes extra draws from the shared RNG (it tries
sticking pairs the real run never would have tried). This must not change
what the *real* run does - after the census, the aggregator's RNG bit-
generator state is restored to what it was right before the census ran, so
downstream retries within the same top-level trial are unaffected. Each
top-level trial here is a *single* PCA+CCA attempt (no internal retries -
main_runner.run_simulation's hidden 20-attempt retry loop is deliberately
bypassed so one seed maps to exactly one outcome, matching what
experiments.md's success-rate figures measure).

Usage:
    devenv shell -- uv run python benchmarks/pairing_frustration_probe.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from pyfracval import particle_generation, utils
from pyfracval.cca import CCAggregator
from pyfracval.config import OrchestratorAlgorithmConfig
from pyfracval.pca_subclusters import Subclusterer
from pyfracval.schemas import SimulationParameters

logging.basicConfig(level=logging.CRITICAL)  # silence per-trial noise
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "benchmark_results"
N_RETRIES_PER_PAIR = 3
N_SEEDS_PER_REGIME = 40
SEED_BASE = 20260727


class _InstrumentedCCAggregator(CCAggregator):
    """Adds a one-shot feasibility census on the round that kills the run.

    Overrides nothing about production sticking behavior - `_run_iteration`
    and `_perform_cca_sticking` are called exactly as the real algorithm
    would call them. The census only runs *after* the real round has
    already failed, and only probes pairs the real round never tried.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.census_result: dict | None = None
        self._census_done = False
        self._last_failed_pair: tuple[int, int] | None = None

    def _perform_cca_sticking_with_expansion(
        self, cluster_idx1, cluster_idx2, cluster_props_cache=None
    ):
        result = super()._perform_cca_sticking_with_expansion(
            cluster_idx1, cluster_idx2, cluster_props_cache
        )
        if result is None:
            self._last_failed_pair = (cluster_idx1, cluster_idx2)
        return result

    def _run_iteration(self) -> bool:
        ok = super()._run_iteration()
        if not ok and not self._census_done:
            self._census_done = True
            rng_state = self._rng.bit_generator.state
            try:
                self.census_result = self._run_census()
                if (
                    self.census_result is not None
                    and self._last_failed_pair is not None
                ):
                    # cluster indices can be numpy.int64 (from np.where(...))
                    # mixed with plain int (from range()) - normalize before
                    # this ends up in a JSON-serialized report.
                    k, other = (int(x) for x in self._last_failed_pair)
                    self.census_result["killing_pair"] = [k, other]
                    self.census_result["killing_pair_degrees"] = [
                        len(self.census_result["_adj"].get(k, [])),
                        len(self.census_result["_adj"].get(other, [])),
                    ]
                    # Did census's own re-test of the *exact same* pair the
                    # real run tried also fail? If so, a rescue requires a
                    # genuinely different partner - not just a retry of the
                    # same pair getting lucky on another rotation draw.
                    self.census_result["killing_pair_edge_confirmed"] = (
                        other in self.census_result["_adj"].get(k, [])
                    )
                    del self.census_result["_adj"]
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Census failed: {exc}")
                self.census_result = {"error": str(exc)}
            finally:
                self._rng.bit_generator.state = rng_state
        return ok

    def _run_census(self) -> dict:
        cluster_ids = [idx for idx in range(self.i_t) if self.i_orden[idx, 2] > 0]
        sizes = {idx: int(self.i_orden[idx, 2]) for idx in cluster_ids}
        n = len(cluster_ids)

        adj: dict[int, set[int]] = {idx: set() for idx in cluster_ids}
        for a_pos in range(n):
            for b_pos in range(a_pos + 1, n):
                i, j = cluster_ids[a_pos], cluster_ids[b_pos]
                found = False
                for _ in range(N_RETRIES_PER_PAIR):
                    result = self._perform_cca_sticking(i, j, None)
                    if result is not None:
                        found = True
                        break
                if found:
                    adj[i].add(j)
                    adj[j].add(i)

        max_matched = _max_matching_size(adj, cluster_ids)
        unmatched_after = n - 2 * max_matched
        # A round is fully resolvable if at most one cluster is left over
        # (CCA already allows a single odd-one-out to pass through a round
        # unmerged - see aggregator.py's odd-cluster handling).
        perfect_possible = unmatched_after <= (n % 2)

        return {
            "n_clusters": n,
            "sizes": sizes,
            "n_edges": sum(len(v) for v in adj.values()) // 2,
            "n_possible_edges": n * (n - 1) // 2,
            "max_matching_pairs": max_matched,
            "unmatched_after_max_matching": unmatched_after,
            "perfect_matching_possible": perfect_possible,
            "_adj": {k: sorted(v) for k, v in adj.items()},
        }


def _max_matching_size(adj: dict[int, set[int]], nodes: list[int]) -> int:
    """Brute-force max matching with memoization (fine for small n - CCA
    round pool sizes are bounded by ~1/n_subcl_percentage, typically <=16,
    regardless of N)."""
    memo: dict[frozenset, int] = {}

    def rec(remaining: frozenset) -> int:
        if len(remaining) <= 1:
            return 0
        if remaining in memo:
            return memo[remaining]
        remaining_list = list(remaining)
        first = remaining_list[0]
        rest = remaining_list[1:]
        best = rec(frozenset(rest))  # leave `first` unmatched
        for other in rest:
            if other in adj[first]:
                candidate_rest = frozenset(x for x in rest if x != other)
                candidate = 1 + rec(candidate_rest)
                if candidate > best:
                    best = candidate
        memo[remaining] = best
        return best

    return rec(frozenset(nodes))


def run_one_trial(
    sim_params: SimulationParameters,
    algorithm_config: OrchestratorAlgorithmConfig,
    rng: np.random.Generator,
) -> dict:
    """Exactly one PCA+CCA attempt - no internal retries."""
    initial_radii = particle_generation.lognormal_pp_radii(
        sim_params.rp_gstd, sim_params.rp_g, sim_params.N, rng=rng
    )
    shuffled_radii = utils.shuffle_array(initial_radii, rng=rng)

    subcluster_runner = Subclusterer(
        initial_radii=shuffled_radii,
        df=sim_params.Df,
        kf=sim_params.kf,
        tol_ov=sim_params.tol_ov,
        n_subcl_percentage=sim_params.n_subcl_percentage,
        rp_g=sim_params.rp_g,
        rp_gstd=sim_params.rp_gstd,
        rng=rng,
        algorithm_config=algorithm_config,
    )
    pca_success = subcluster_runner.run_subclustering()
    if not pca_success or subcluster_runner.not_able_pca:
        return {"outcome": "pca_failed"}

    num_clusters, not_able_pca_flag, pca_coords_radii, pca_i_orden, _ = (
        subcluster_runner.get_results()
    )
    if not_able_pca_flag or pca_coords_radii is None or pca_i_orden is None:
        return {"outcome": "pca_failed"}

    cca_runner = _InstrumentedCCAggregator(
        initial_coords=pca_coords_radii[:, :3],
        initial_radii=pca_coords_radii[:, 3],
        initial_i_orden=pca_i_orden,
        n_total=sim_params.N,
        df=sim_params.Df,
        kf=sim_params.kf,
        tol_ov=sim_params.tol_ov,
        ext_case=sim_params.ext_case,
        rng=rng,
        algorithm_config=algorithm_config,
    )
    cca_result = cca_runner.run_cca()
    if cca_result is None or cca_runner.not_able_cca:
        return {"outcome": "cca_failed", "census": cca_runner.census_result}

    return {"outcome": "success"}


def run_regime(name: str, N: int, Df: float, kf: float, rp_gstd: float) -> dict:
    sim_params = SimulationParameters(
        N=N,
        Df=Df,
        kf=kf,
        rp_g=100.0,
        rp_gstd=rp_gstd,
        tol_ov=1e-6,
        n_subcl_percentage=0.1,
        ext_case=0,
    )
    algorithm_config = OrchestratorAlgorithmConfig()  # all defaults, densify off

    outcomes = {"success": 0, "cca_failed": 0, "pca_failed": 0}
    censuses = []
    t0 = time.time()

    for i in range(N_SEEDS_PER_REGIME):
        rng = np.random.default_rng(SEED_BASE + i)
        trial = run_one_trial(sim_params, algorithm_config, rng)
        outcomes[trial["outcome"]] += 1
        if trial["outcome"] == "cca_failed" and trial.get("census") is not None:
            censuses.append(trial["census"])
        print(
            f"  [{name}] seed {i + 1}/{N_SEEDS_PER_REGIME}: {trial['outcome']}",
            flush=True,
        )

    elapsed = time.time() - t0

    n_cca_failed = outcomes["cca_failed"]
    censuses_valid = [c for c in censuses if "error" not in c]
    n_rescuable = sum(1 for c in censuses_valid if c["perfect_matching_possible"])
    n_frustrated = len(censuses_valid) - n_rescuable
    # Of the rescuable failures: was the killed cluster stranded (degree 0 -
    # no rescue possible for it specifically, but *some other* pairing of
    # the pool still resolves everyone) or did greedy just mispair it
    # (degree >= 1 - it had a working partner available)?
    n_rescuable_but_killed_stranded = sum(
        1
        for c in censuses_valid
        if c["perfect_matching_possible"]
        and min(c.get("killing_pair_degrees", [1, 1])) == 0
    )
    n_rescuable_killed_mispaired = n_rescuable - n_rescuable_but_killed_stranded
    # Of rescuable failures: did census's own re-test of the exact original
    # pair also fail (rescue needs a genuinely different partner) vs.
    # succeed (a same-pair retry alone would also have worked)?
    n_rescue_needs_different_partner = sum(
        1
        for c in censuses_valid
        if c["perfect_matching_possible"]
        and not c.get("killing_pair_edge_confirmed", True)
    )

    summary = {
        "regime": name,
        "params": {"N": N, "Df": Df, "kf": kf, "rp_gstd": rp_gstd},
        "n_seeds": N_SEEDS_PER_REGIME,
        "elapsed_s": elapsed,
        "outcomes": outcomes,
        "success_rate": outcomes["success"] / N_SEEDS_PER_REGIME,
        "n_cca_failed": n_cca_failed,
        "n_census_valid": len(censuses_valid),
        "n_rescuable_by_pairing": n_rescuable,
        "n_universally_frustrated": n_frustrated,
        "n_rescuable_killed_mispaired": n_rescuable_killed_mispaired,
        "n_rescuable_but_killed_stranded": n_rescuable_but_killed_stranded,
        "n_rescue_needs_different_partner": n_rescue_needs_different_partner,
        "rescuable_fraction_of_failures": (
            n_rescuable / len(censuses_valid) if censuses_valid else None
        ),
        "censuses": censuses,
    }
    print(
        f"[{name}] success_rate={summary['success_rate']:.1%}  "
        f"cca_failed={n_cca_failed}  "
        f"rescuable={n_rescuable}/{len(censuses_valid)}  "
        f"({elapsed:.1f}s)"
    )
    return summary


def main() -> None:
    print("=" * 80)
    print("Pairing-frustration probe")
    print("=" * 80)

    regimes = [
        run_regime("hard", N=128, Df=2.25, kf=0.95, rp_gstd=1.9),
        run_regime("easy_control", N=128, Df=1.8, kf=1.0, rp_gstd=1.5),
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pairing_frustration_probe.json"
    out_path.write_text(json.dumps(regimes, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
