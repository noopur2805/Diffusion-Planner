"""Split-conformal calibration of the dense reward model.

The reward model emits ``(mu, log_var)`` in pre-sigmoid space for every
``(horizon, metric)`` cell. After applying ``sigmoid`` we have a per-cell
probability ``p_hat in [0, 1]``; aggregating these via the PDMS formula
gives a scalar score in the same range. Both are *uncalibrated*: there is
no proof that the true value lies within any specific neighborhood of the
prediction.

This module learns a single scalar ``delta`` (per metric and one for the
PDMS aggregate) from a held-out calibration split such that, for any new
exchangeable sample,

    P(y >= p_hat - delta) >= 1 - alpha

i.e. ``p_hat - delta`` is a finite-sample lower confidence bound on the
true reward / PDMS. The guarantee is distribution-free; it does not
assume the reward model is well-specified. We use the standard split
formulation (Vovk et al. 2005; Angelopoulos & Bates tutorial 2023):

    delta = s_{(k)} with k = ceil((n + 1)(1 - alpha))

where ``s_1, ..., s_n`` are the one-sided non-conformity scores on the
calibration split.

The planner uses ``delta_pdms`` at inference to gate a defensive-fallback
trajectory: if ``p_hat_pdms - delta_pdms < tau_safe`` the planner emits
the fallback instead of the model's prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import yaml


# PDMS aggregation matches ``evaluate.py``: multiplicative gates +
# weighted-sum scored terms, weights summing to 16.
_PDMS_GATES: Sequence[str] = ("nc", "dac", "ddc", "mp")
_PDMS_SCORED_WEIGHTS: Dict[str, float] = {"ttc": 5.0, "ep": 5.0, "comfort": 2.0, "sl": 4.0}
_PDMS_DIVISOR: float = sum(_PDMS_SCORED_WEIGHTS.values())  # = 16.0


def pdms_from_metrics(per_metric: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Aggregate per-metric reward tensors into a scalar PDMS in [0, 1].

    ``per_metric`` maps each name in :data:`reward_labeling.METRIC_ORDER`
    to a tensor of arbitrary leading shape (typically ``(B,)`` after a
    mean over the horizon axis). All tensors must broadcast together.
    """
    gates = per_metric[_PDMS_GATES[0]]
    for k in _PDMS_GATES[1:]:
        gates = gates * per_metric[k]
    scored = sum(w * per_metric[k] for k, w in _PDMS_SCORED_WEIGHTS.items())
    return gates * scored / _PDMS_DIVISOR


def compute_delta(scores: torch.Tensor, alpha: float) -> float:
    """Split-conformal ``delta`` = the ``ceil((n+1)(1-alpha))``-th order
    statistic of ``scores`` (one-sided non-conformity).

    Returns ``+inf`` when ``n`` is too small to certify the requested
    coverage at finite-sample level (i.e. ``ceil((n+1)(1-alpha)) > n``).
    """
    s = scores.detach().flatten().float()
    n = s.numel()
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    sorted_s, _ = torch.sort(s)
    return float(sorted_s[k - 1].item())


@dataclass
class ConformalPredictor:
    """Loaded calibration constants + the inference-time gate.

    Attributes:
        alpha:           target miscoverage (e.g. 0.10 for 90% coverage).
        delta_pdms:      one-sided non-conformity threshold on PDMS.
        delta_per_metric: per-metric thresholds in :data:`metric_names` order.
        metric_names:    names matching ``delta_per_metric`` positions.
        n_cal:           calibration set size used to derive the deltas.
        mode:            ``'absolute'`` or ``'normalized'`` (sigma-scaled).
    """

    alpha: float
    delta_pdms: float
    delta_per_metric: List[float] = field(default_factory=list)
    metric_names: List[str] = field(default_factory=list)
    n_cal: int = 0
    mode: str = "absolute"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConformalPredictor":
        with open(path, "r") as f:
            blob = yaml.safe_load(f)
        return cls(
            alpha=float(blob["alpha"]),
            delta_pdms=float(blob["delta_pdms"]),
            delta_per_metric=[float(x) for x in blob.get("delta_per_metric", [])],
            metric_names=list(blob.get("metric_names", [])),
            n_cal=int(blob.get("n_cal", 0)),
            mode=str(blob.get("mode", "absolute")),
        )

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "alpha": self.alpha,
            "delta_pdms": self.delta_pdms,
            "delta_per_metric": list(self.delta_per_metric),
            "metric_names": list(self.metric_names),
            "n_cal": self.n_cal,
            "mode": self.mode,
        }
        with open(path, "w") as f:
            yaml.safe_dump(blob, f, sort_keys=False)

    def lower_bound_pdms(self, p_hat_pdms: torch.Tensor) -> torch.Tensor:
        """One-sided lower bound: ``p_hat_pdms - delta_pdms`` clipped to [0, 1]."""
        return (p_hat_pdms - self.delta_pdms).clamp_(min=0.0, max=1.0)

    def lower_bound_per_metric(self, p_hat_per_metric: torch.Tensor) -> torch.Tensor:
        """``p_hat - delta`` per metric. ``p_hat_per_metric`` last dim = K."""
        delta = torch.as_tensor(
            self.delta_per_metric, dtype=p_hat_per_metric.dtype, device=p_hat_per_metric.device,
        )
        return (p_hat_per_metric - delta).clamp_(min=0.0, max=1.0)

    def is_unsafe(self, p_hat_pdms: torch.Tensor, tau_safe: float) -> torch.Tensor:
        """``True`` for samples whose lower-bound PDMS falls below ``tau_safe``."""
        return self.lower_bound_pdms(p_hat_pdms) < float(tau_safe)
