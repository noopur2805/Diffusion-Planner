"""Safety-tier utilities for the production-ready planner.

Currently exposes the :class:`ConformalPredictor` and :func:`compute_delta`
helpers used to wrap the reward model's predictions with a finite-sample,
distribution-free coverage guarantee (split conformal prediction).
"""
from diffusion_planner.safety.conformal import (
    ConformalPredictor,
    compute_delta,
    pdms_from_metrics,
)
from diffusion_planner.safety.fallback import constant_velocity_fallback

__all__ = [
    "ConformalPredictor",
    "compute_delta",
    "pdms_from_metrics",
    "constant_velocity_fallback",
]
