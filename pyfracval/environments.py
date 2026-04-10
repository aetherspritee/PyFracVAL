"""Environment variables used by PyFracVAL.

This module centralizes the environment variables that influence runtime
behavior in PyFracVAL.

Environment Variables
---------------------
PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS : str
    When set to ``"1"``, disables multiprocessing for PCA subcluster
    generation and forces sequential execution.
PYFRACVAL_INSTALLED_WHEEL : str
    Records the wheel filename installed on Dask workers during remote
    package registration.
PYFRACVAL_EXPECTED_VERSION : str
    Records the package version expected on Dask workers after installation.
OMP_NUM_THREADS : str
    Controls the number of OpenMP threads used by NumPy/SciPy-backed code.
MKL_NUM_THREADS : str
    Controls the number of Intel MKL threads used by NumPy/SciPy-backed code.
OPENBLAS_NUM_THREADS : str
    Controls the number of OpenBLAS threads used by NumPy/SciPy-backed code.
NUMEXPR_NUM_THREADS : str
    Controls the number of NumExpr threads used by NumPy/SciPy-backed code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS: Final[str] = (
    "PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS"
)
PYFRACVAL_INSTALLED_WHEEL: Final[str] = "PYFRACVAL_INSTALLED_WHEEL"
PYFRACVAL_EXPECTED_VERSION: Final[str] = "PYFRACVAL_EXPECTED_VERSION"
OMP_NUM_THREADS: Final[str] = "OMP_NUM_THREADS"
MKL_NUM_THREADS: Final[str] = "MKL_NUM_THREADS"
OPENBLAS_NUM_THREADS: Final[str] = "OPENBLAS_NUM_THREADS"
NUMEXPR_NUM_THREADS: Final[str] = "NUMEXPR_NUM_THREADS"

THREAD_CONTROL_ENV_VARS: Final[tuple[str, ...]] = (
    OMP_NUM_THREADS,
    MKL_NUM_THREADS,
    OPENBLAS_NUM_THREADS,
    NUMEXPR_NUM_THREADS,
)


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Runtime environment configuration for PyFracVAL.

    Parameters
    ----------
    disable_parallel_subclusters : bool
        Whether PCA subcluster generation should run sequentially.
    installed_wheel : str | None
        Wheel filename recorded when a package installation is performed on a
        Dask scheduler or worker.
    expected_version : str | None
        Expected package version recorded during remote installation.
    omp_num_threads : str | None
        Value of ``OMP_NUM_THREADS``.
    mkl_num_threads : str | None
        Value of ``MKL_NUM_THREADS``.
    openblas_num_threads : str | None
        Value of ``OPENBLAS_NUM_THREADS``.
    numexpr_num_threads : str | None
        Value of ``NUMEXPR_NUM_THREADS``.
    """

    disable_parallel_subclusters: bool
    installed_wheel: str | None
    expected_version: str | None
    omp_num_threads: str | None
    mkl_num_threads: str | None
    openblas_num_threads: str | None
    numexpr_num_threads: str | None


def _read_flag(name: str) -> bool:
    return os.getenv(name, "") == "1"


def get_env_config() -> EnvironmentConfig:
    """Return the PyFracVAL runtime environment configuration.

    Returns
    -------
    EnvironmentConfig
        Snapshot of the environment variables used by PyFracVAL.
    """
    return EnvironmentConfig(
        disable_parallel_subclusters=_read_flag(PYFRACVAL_DISABLE_PARALLEL_SUBCLUSTERS),
        installed_wheel=os.getenv(PYFRACVAL_INSTALLED_WHEEL),
        expected_version=os.getenv(PYFRACVAL_EXPECTED_VERSION),
        omp_num_threads=os.getenv(OMP_NUM_THREADS),
        mkl_num_threads=os.getenv(MKL_NUM_THREADS),
        openblas_num_threads=os.getenv(OPENBLAS_NUM_THREADS),
        numexpr_num_threads=os.getenv(NUMEXPR_NUM_THREADS),
    )
