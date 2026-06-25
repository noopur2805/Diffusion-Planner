"""
Patience-based early stopping on a scalar epoch metric (lower is better).

Default ``patience=0`` disables the mechanism and ``step`` always returns
``False``, so adding this to a script is a no-op until the user supplies a
positive ``--early_stop_patience``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EarlyStopper:
    patience: int = 0
    min_delta: float = 0.0
    mode: str = "min"

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        self.best: float = math.inf if self.mode == "min" else -math.inf
        self.best_epoch: int = -1
        self.bad_epochs: int = 0
        self.should_stop: bool = False
        self.improved: bool = False

    @property
    def disabled(self) -> bool:
        return self.patience <= 0

    def step(self, value: float, epoch: int) -> bool:
        """Update state with the latest epoch metric. Returns ``should_stop``."""
        if self.disabled:
            self.improved = False
            return False
        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        self.improved = improved
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop
