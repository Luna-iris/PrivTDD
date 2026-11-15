"""PrivTDD: private triangle-degree distribution estimation.

The API is imported lazily so inspecting the installed version does not start
Numba compilation or initialize experiment tools.
"""
__version__ = "0.1.0"
__all__ = ["PrivTDDConfig", "estimate_tdd", "estimate_tdd_from_reports"]


def __getattr__(name):
    if name == "PrivTDDConfig":
        from .config import PrivTDDConfig
        return PrivTDDConfig
    if name in ("estimate_tdd", "estimate_tdd_from_reports"):
        from .release import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
