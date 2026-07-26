"""Backward-compatible re-export shim.

``CCAggregator`` now lives in :mod:`pyfracval.cca` (split across
``pyfracval/cca/{pairing,candidates,sticking,fallbacks,aggregator}.py`` -
see that package's docstring for the breakdown). This module re-exports it
under the old import path so existing code (``from pyfracval.cca_agg import
CCAggregator``) keeps working unchanged.
"""

from .cca import CCAggregator

__all__ = ["CCAggregator"]
