"""Stable public API; scientific defaults are documented in docs/method.md."""
from .config import PrivTDDConfig
from .release.pipeline import estimate_tdd, estimate_tdd_from_reports

__all__ = ["PrivTDDConfig", "estimate_tdd", "estimate_tdd_from_reports"]
