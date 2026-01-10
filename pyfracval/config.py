import numpy as np

# --- Simulation Parameters ---
N: int = 128  # Number of Primary Particles (PP)
# N: int = 1024  # Number of Primary Particles (PP)
DF: float = 2.0  # Target Fractal dimension
KF: float = 1.0  # Target Fractal prefactor
QUANTITY_AGGREGATES: int = 1  # Number of aggregates to generate

# --- Primary Particle Properties ---
RP_GEOMETRIC_MEAN: float = 100.0  # Geometric mean radius of PP
RP_GEOMETRIC_STD: float = 1.50  # Geometric standard deviation of PP radii
# RP_GEOMETRIC_STD: float = 1.25  # Geometric standard deviation of PP radii
# RP_GEOMETRIC_STD: float = 1.00  # Geometric standard deviation of PP radii

# --- Algorithm Tuning Parameters ---
EXT_CASE: int = 0  # CCA Sticking: 0 for standard, 1 for 'extreme cases'
N_SUBCL_PERCENTAGE: float = 0.1  # PCA: Target fraction of N for subcluster size
TOL_OVERLAP: float = 1.0e-6  # Overlap tolerance for sticking

# --- Performance Tuning ---
USE_BATCH_ROTATION: bool = (
    False  # Enable batch rotation (slower for N<1000, keep False)
)
ROTATION_BATCH_SIZE: int = (
    32  # Number of rotation angles to evaluate in parallel (if enabled)
)

# --- Constants ---
PI: float = np.pi

# --- Derived Parameters (can be calculated later if needed) ---
# N_SUBCL: int = ... # Calculated in pca_subclusters.py
