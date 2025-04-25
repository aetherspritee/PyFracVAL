# pca_subclusters.py
"""
Divides initial particles into subclusters using Particle-Cluster Aggregation (PCA).
"""

import logging
import math

import numpy as np

from .pca_agg import PCAggregator

logger = logging.getLogger(__name__)


class Subclusterer:
    """Handles the division of particles into subclusters using PCA."""

    def __init__(
        self,
        initial_radii: np.ndarray,
        df: float,
        kf: float,
        tol_ov: float,
        n_subcl_percentage: float,
    ):
        self.N = len(initial_radii)
        self.initial_radii = initial_radii.copy()  # Use a copy
        self.df = df
        self.kf = kf
        self.tol_ov = tol_ov
        self.n_subcl_percentage = n_subcl_percentage

        self.all_coords = np.zeros((self.N, 3), dtype=float)
        self.all_radii = np.zeros(self.N, dtype=float)
        self.i_orden: np.ndarray | None = None
        self.number_clusters: int = 0
        self.not_able_pca: bool = False
        self.number_clusters_processed = 0

    def _determine_subcluster_sizes(self) -> np.ndarray:
        """Calculates the size of each subcluster."""
        if self.N < 50:
            n_subcl = 5
        elif self.N > 500:
            n_subcl = 50
        else:
            n_subcl = int(self.n_subcl_percentage * self.N)

        # Ensure n_subcl is at least 2 for PCA and not larger than N
        n_subcl = max(2, min(n_subcl, self.N))

        self.number_clusters = math.ceil(self.N / n_subcl)
        subcluster_sizes = np.full(self.number_clusters, n_subcl, dtype=int)

        # Adjust the last cluster size if N is not perfectly divisible
        remainder = self.N % n_subcl
        if remainder != 0:
            # Distribute remainder or assign to last? Fortran assigns to last.
            actual_size_last = self.N - n_subcl * (self.number_clusters - 1)
            subcluster_sizes[-1] = actual_size_last
        elif self.N % n_subcl == 0 and self.number_clusters * n_subcl == self.N:
            pass  # Sizes are already correct

        # Sanity check
        if np.sum(subcluster_sizes) != self.N:
            logger.warning(
                "Sum of subcluster sizes does not equal N. Adjusting last size."
            )
            subcluster_sizes[-1] = self.N - np.sum(subcluster_sizes[:-1])
            if subcluster_sizes[-1] < 0:
                raise ValueError(
                    "Subcluster size calculation resulted in negative size."
                )

        logger.info(
            f"Subclustering N={self.N} into {self.number_clusters} clusters with target size {n_subcl}."
        )
        logger.info(f"Actual sizes: {subcluster_sizes}")
        return subcluster_sizes

    def run_subclustering(self) -> bool:
        """
        Performs the subclustering process.

        Returns:
            bool: True if successful, False otherwise. Updates instance attributes.
        """
        subcluster_sizes = self._determine_subcluster_sizes()
        self.i_orden = np.zeros((self.number_clusters, 3), dtype=int)
        self.not_able_pca = False
        current_n_start_idx = 0  # Index in the initial_radii array
        current_fill_idx = 0  # Index in the final all_coords/all_radii

        # Option 1: Always use relaxed parameters (e.g., typical monodisperse values)
        # pca_df = 1.79  # Typical DLCA/Filippov mono value
        # pca_kf = 1.40  # Typical DLCA/Filippov mono value
        # logger.info(
        #     f"--- Using relaxed parameters for PCA stage: Df={pca_df}, kf={pca_kf} ---"
        # )

        # Option 2: Use override values if provided during init
        # pca_df = self.pca_df_override if self.pca_df_override is not None else self.df
        # pca_kf = self.pca_kf_override if self.pca_kf_override is not None else self.kf

        # Option 3: Use target Df/kf (Original way, prone to failure)
        pca_df = self.df
        pca_kf = self.kf

        for i in range(self.number_clusters):
            self.number_clusters_processed = i
            num_particles_in_subcluster = subcluster_sizes[i]
            logger.info(
                f"--- Processing Subcluster {i + 1}/{self.number_clusters} (Size: {num_particles_in_subcluster}) ---"
            )

            if num_particles_in_subcluster < 2:
                logger.error(
                    f"Subcluster {i + 1} has size {num_particles_in_subcluster}, needs >= 2 for PCA."
                )
                self.not_able_pca = True
                return False  # Cannot proceed

            # Extract radii for this subcluster
            idx_start = current_n_start_idx
            idx_end = current_n_start_idx + num_particles_in_subcluster
            subcluster_radii = self.initial_radii[idx_start:idx_end]

            # Run PCA for this subcluster
            pca_runner = PCAggregator(subcluster_radii, pca_df, pca_kf, self.tol_ov)
            subcluster_data = pca_runner.run()  # Returns Nx4 [X,Y,Z,R] or None

            if subcluster_data is None or pca_runner.not_able_pca:
                logger.error(f"PCA failed for subcluster {i + 1}.")
                self.not_able_pca = True
                return False  # PCA failed for this subcluster

            # Store the results
            num_added = subcluster_data.shape[0]
            if current_fill_idx + num_added > self.N:
                logger.error(f"Exceeding total particle count N during subclustering.")
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
            current_n_start_idx += num_particles_in_subcluster
            current_fill_idx += num_added

        # Final check
        if current_fill_idx != self.N:
            logger.warning(
                f"Final particle count ({current_fill_idx}) after subclustering does not match N ({self.N})."
            )
            # This might indicate an issue in size calculation or PCA runs returning unexpected sizes.

        logger.info("PCA Subclustering completed.")
        return True  # Success

    def get_results(
        self,
    ) -> tuple[int, bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Returns the results of the subclustering."""
        if self.not_able_pca:
            return 0, True, None, None, None
        else:
            # Combine coords and radii into the 'Data' format expected by CCA_sub
            combined_data = np.hstack((self.all_coords, self.all_radii.reshape(-1, 1)))
            return (
                self.number_clusters,
                False,
                combined_data,
                self.i_orden,
                self.all_radii,
            )
