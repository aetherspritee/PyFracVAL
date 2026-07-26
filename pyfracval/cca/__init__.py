"""Cluster-Cluster Aggregation (CCA) - split into focused modules.

``CCAggregator`` in :mod:`pyfracval.cca.aggregator` is composed from four
mixins, each owning one concern:

- :mod:`pyfracval.cca.pairing` - pair generation, Gamma_pc calculation
- :mod:`pyfracval.cca.candidates` - candidate pair selection, scoring, telemetry
- :mod:`pyfracval.cca.sticking` - rigid-body sticking, rotation, overlap checks
- :mod:`pyfracval.cca.fallbacks` - gamma expansion, pair prechecks, sticking fallbacks

``pyfracval.cca_agg`` re-exports ``CCAggregator`` for backward compatibility.
"""

from .aggregator import CCAggregator

__all__ = ["CCAggregator"]
