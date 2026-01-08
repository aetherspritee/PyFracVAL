"""Divides initial particles into subclusters using PCA."""

import logging
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .pca_agg import PCAggregator

# Ensure access to utils if needed, though likely not directly
# from . import utils

logger = logging.getLogger(__name__)


class Subclusterer(BaseModel):
    """Handles division of particles into subclusters using PCA.

    Takes a full set of initial particle radii, determines appropriate
    subcluster sizes, and runs the `PCAggregator` on each subset of radii
    to generate initial cluster structures. These subclusters are intended
    as input for subsequent Cluster-Cluster Aggregation (CCA).

    Parameters
    ----------
    initial_radii : np.ndarray
        1D array of all initial primary particle radii.
    df : float
        Target fractal dimension (passed to `PCAggregator`).
    kf : float
        Target fractal prefactor (passed to `PCAggregator`).
    tol_ov : float
        Overlap tolerance (passed to `PCAggregator`).
    n_subcl_percentage : float
        Target fraction of N used to determine the approximate
        size of each subcluster. Actual sizes may vary.

    Attributes
    ----------
    N : int
        Total number of particles.
    all_coords : np.ndarray
        Nx3 array storing coordinates of all particles after PCA subclustering.
    all_radii : np.ndarray
        N array storing radii of all particles (should match initial radii order
        if PCA doesn't reorder, but uses radii from PCA output).
    i_orden : np.ndarray | None
        Mx3 array defining the start index, end index (inclusive), and count
        for each generated subcluster within the `all_coords`/`all_radii` arrays.
        None until `run_subclustering` is successful.
    number_clusters : int
        The number of subclusters generated.
    not_able_pca : bool
        Flag indicating if any PCA run for a subcluster failed.
    number_clusters_processed : int
        Index of the last subcluster processed (useful for error reporting).
    """

    initial_radii: np.ndarray
    df: float = Field(..., gt=1.0, lt=3.0)
    kf: float = Field(..., gt=0.0)
    tol_ov: float
    n_subcl_percentage: float = Field(default=0.1, lt=1.0)

    N: int = Field(default=0)
    all_coords: np.ndarray = Field(default=np.zeros(0))
    all_radii: np.ndarray = Field(default=np.zeros(0))
    i_orden: np.ndarray | None = Field(default=None)
    number_clusters: int = Field(default=0)
    not_able_pca: bool = Field(default=False)
    number_clusters_processed: int = Field(default=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context):
        self.N = len(self.initial_radii)
        self.initial_radii = self.initial_radii.copy()  # Use a copy

        self.all_coords = np.zeros((self.N, 3), dtype=float)
        self.all_radii = np.zeros(self.N, dtype=float)

    # def __init__(
    #     self,
    #     initial_radii: np.ndarray,
    #     df: float,  # Target Df for final aggregate
    #     kf: float,  # Target kf for final aggregate
    #     tol_ov: float,
    #     n_subcl_percentage: float,
    #     # Optional overrides for PCA stage Df/kf could be added here
    #     # pca_df_override: float | None = None,
    #     # pca_kf_override: float | None = None,
    # ):
    #     self.N = len(initial_radii)
    #     self.initial_radii = initial_radii.copy()  # Use a copy
    #     self.df = df  # Store target Df
    #     self.kf = kf  # Store target kf
    #     self.tol_ov = tol_ov
    #     self.n_subcl_percentage = n_subcl_percentage
    #     # self.pca_df_override = pca_df_override # Store overrides if used
    #     # self.pca_kf_override = pca_kf_override

    #     self.all_coords = np.zeros((self.N, 3), dtype=float)
    #     self.all_radii = np.zeros(self.N, dtype=float)
    #     self.i_orden: np.ndarray | None = None
    #     self.number_clusters: int = 0
    #     self.not_able_pca: bool = False
    #     self.number_clusters_processed = 0  # Track for error reporting

    def _determine_subcluster_sizes(self) -> np.ndarray:
        """Calculates the size of each subcluster."""
        # --- Heuristic for N_subcl based on N (from Fortran comments/logic) ---
        if self.N < 50:
            # # Fortran uses Nsub=5, leading to many small clusters
            # # Let's use the percentage but ensure min size (e.g., 5?)
            # n_subcl_target = max(5, int(self.n_subcl_percentage * self.N))
            # # Ensure n_subcl is at least 2
            # n_subcl = max(2, n_subcl_target)
            n_subcl = 5
        elif self.N > 500:
            # # Fortran uses Nsub=50
            # n_subcl_target = max(50, int(self.n_subcl_percentage * self.N))
            # n_subcl = n_subcl_target  # Allow larger for large N if percentage dictates
            n_subcl = 50
        else:  # 50 <= N <= 500
            # # Use percentage, but ensure min size (e.g., 5 or 10?)
            # n_subcl_target = max(10, int(self.n_subcl_percentage * self.N))
            # n_subcl = n_subcl_target
            n_subcl = int(self.n_subcl_percentage * self.N)

        # # Ensure n_subcl is not larger than N
        # n_subcl = min(n_subcl, self.N)

        # Calculate number of clusters needed
        self.number_clusters = math.ceil(self.N / n_subcl)
        subcluster_sizes = np.full(self.number_clusters, n_subcl, dtype=int)
        # Adjust the last cluster size if N is not perfectly divisible
        remainder = self.N % n_subcl
        if remainder != 0:
            # remainder = self.N - n_subcl * (self.number_clusters - 1)
            subcluster_sizes[-1] = remainder

        # Sanity check
        if np.sum(subcluster_sizes) != self.N:
            raise ValueError(
                f"Subcluster size calculation error: Sum={np.sum(subcluster_sizes)} != N={self.N}. Sizes={subcluster_sizes}"
            )

        logger.info(
            f"Subclustering N={self.N} into {self.number_clusters} clusters with target size ~{n_subcl}."
        )
        logger.info(f"Actual sizes: {subcluster_sizes}")
        return subcluster_sizes

    def run_subclustering(self) -> bool:
        """Perform the subclustering process.

        Determines subcluster sizes, then iterates through subsets of the
        initial radii, running `PCAggregator` for each subset. Stores the
        resulting coordinates and radii contiguously and updates `i_orden`.

        Returns
        -------
        bool
            True if all subclusters were generated successfully, False otherwise.
            Sets `self.not_able_pca` to True on failure.
        """
        subcluster_sizes = self._determine_subcluster_sizes()

        # Handle the edge case of only 1 cluster (N < n_subcl or N=n_subcl)
        if self.number_clusters == 1 and subcluster_sizes[0] == self.N:
            logger.info(
                "Only one subcluster required (N <= effective n_subcl). Running PCA on all particles."
            )
            # Proceed with the loop below, it will just run once.

        self.i_orden = np.zeros((self.number_clusters, 3), dtype=int)
        self.not_able_pca = False
        current_n_start_idx = 0  # Index in the initial_radii array
        current_fill_idx = 0  # Index in the final all_coords/all_radii

        pca_df = self.df
        pca_kf = self.kf
        # --- Define Df/kf to use *specifically* for PCA ---
        # Use fixed, stable values, e.g., typical DLCA or Filippov mono values
        # pca_df = 1.79
        # pca_kf = 1.40
        # Alternatively, use overrides if they were passed during init:
        # pca_df = self.pca_df_override if self.pca_df_override is not None else 1.79
        # pca_kf = self.pca_kf_override if self.pca_kf_override is not None else 1.40
        # --- Do NOT use self.target_df / self.target_kf here if they cause issues ---
        logger.info(
            f"--- Using fixed parameters for PCA stage: Df={pca_df:.2f}, kf={pca_kf:.2f} ---"
        )

        # TODO: replace with enumerate, its cleaner
        for i in range(self.number_clusters):
            self.number_clusters_processed = i  # Track for error reporting
            num_particles_in_subcluster = subcluster_sizes[i]
            logger.info(
                f"--- Processing Subcluster {i + 1}/{self.number_clusters} (Size: {num_particles_in_subcluster}) ---"
            )

            # TODO: why creash if only one subcluster is 1 (last one?)
            if num_particles_in_subcluster < 2:
                logger.error(
                    f"Subcluster {i + 1} has size {num_particles_in_subcluster}, needs >= 2 for PCA."
                )
                # This should not happen if _determine_subcluster_sizes is correct
                self.not_able_pca = True
                return False

            # Extract radii for this subcluster
            idx_start = current_n_start_idx
            idx_end = current_n_start_idx + num_particles_in_subcluster
            subcluster_radii = self.initial_radii[idx_start:idx_end]

            # Run PCA for this subcluster using the *fixed* pca_df, pca_kf
            pca_runner = PCAggregator(subcluster_radii, pca_df, pca_kf, self.tol_ov)
            subcluster_data = pca_runner.run()  # Returns Nx4 [X,Y,Z,R] or None

            if subcluster_data is None or pca_runner.not_able_pca:
                logger.error(f"PCA failed for subcluster {i + 1}.")
                self.not_able_pca = True
                # No need to return immediately, let main_runner handle retry
                return False  # Signal failure for this attempt

            # Store the results
            num_added = subcluster_data.shape[0]
            # Basic check if PCA returned expected number of particles
            if num_added != num_particles_in_subcluster:
                logger.warning(
                    f"PCA for subcluster {i + 1} returned {num_added} particles, expected {num_particles_in_subcluster}."
                )
                # This might indicate internal PCA issues, but proceed cautiously

            if current_fill_idx + num_added > self.N:
                logger.error(
                    f"Exceeding total particle count N during subclustering "
                    f"(current_fill_idx={current_fill_idx}, num_added={num_added}, N={self.N})."
                )
                self.not_able_pca = True
                return False

            fill_slice = slice(current_fill_idx, current_fill_idx + num_added)
            self.all_coords[fill_slice, :] = subcluster_data[:, :3]
            self.all_radii[fill_slice] = subcluster_data[:, 3]

            # Update i_orden (0-based inclusive indices)
            start_cluster_idx = current_fill_idx
            end_cluster_idx = current_fill_idx + num_added - 1
            self.i_orden[i, :] = [start_cluster_idx, end_cluster_idx, num_added]

            # Update indices for next iteration
            # Advance by expected size
            current_n_start_idx += num_particles_in_subcluster
            current_fill_idx += num_added  # Advance by actual added size

        # Final check after loop
        if current_fill_idx != self.N:
            logger.warning(
                f"Final particle count ({current_fill_idx}) after subclustering does not match N ({self.N}). "
                f"This might indicate inconsistent particle counts returned by PCA runs."
            )
            # Correct i_orden if necessary? Might be complex. Let CCA handle potential mismatch.

        logger.info("PCA Subclustering completed for this attempt.")
        return True  # Success for this attempt

    def get_results(
        self,
    ) -> tuple[int, bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Return the results of the subclustering process.

        Returns
        -------
        tuple[int, bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]
            A tuple containing:
                - number_clusters (int): The intended number of clusters.
                - not_able_pca (bool): Flag indicating if any PCA failed.
                - combined_data (np.ndarray | None): Nx4 array [X, Y, Z, R] of all
                  particles, or None on failure.
                - i_orden (np.ndarray | None): Mx3 array describing subcluster
                  indices, or None on failure.
                - final_radii (np.ndarray | None): N array of radii corresponding
                  to `combined_data`, or None on failure.
        """
        if self.not_able_pca or self.i_orden is None:  # Check i_orden initialization
            return 0, True, None, None, None
        else:
            # Combine coords and radii into the 'Data' format [X, Y, Z, R]
            # Ensure slicing is correct if fill_idx != N
            final_count = self.i_orden[-1, 1] + 1 if self.i_orden.shape[0] > 0 else 0
            if final_count != self.N:
                logger.warning(
                    f"get_results: final count in i_orden ({final_count}) != N ({self.N}). Returning sliced data."
                )

            combined_data = np.hstack(
                (
                    self.all_coords[:final_count],
                    self.all_radii[:final_count].reshape(-1, 1),
                )
            )
            # Return only the valid part of i_orden if fewer clusters were made (shouldn't happen here)
            valid_i_orden = (
                self.i_orden[: self.number_clusters_processed + 1, :]
                if self.number_clusters_processed + 1 < self.number_clusters
                else self.i_orden
            )

            return (
                self.number_clusters,  # Still return the intended number
                False,
                combined_data,
                valid_i_orden,
                self.all_radii[:final_count],  # Return only valid radii
            )
