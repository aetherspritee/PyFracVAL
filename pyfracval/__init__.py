"""Core package for PyFracVAL, a fractal aggregate generator."""

from pyfracval.catalog import filter_catalog, load_catalog
from pyfracval.gap_scaling import compute_gap_scale
from pyfracval.schemas import ClusterEntry

# Bumped in-place by python-semantic-release (see [tool.semantic_release] in
# pyproject.toml, version_variable = "pyfracval/__init__.py:__version__").
__version__ = "0.1.0"

__all__ = [
    "ClusterEntry",
    "compute_gap_scale",
    "filter_catalog",
    "load_catalog",
]
