"""Timestamp synchronization utilities."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class TimestampIndex:
    """Fast nearest-neighbor lookup for sorted timestamps."""

    def __init__(self, timestamps: Sequence[float]):
        self._ts = np.array(timestamps, dtype=np.float64)

    def nearest(self, target: float) -> int:
        """Return index of nearest timestamp."""
        idx = np.searchsorted(self._ts, target)
        if idx == 0:
            return 0
        if idx == len(self._ts):
            return len(self._ts) - 1
        if abs(self._ts[idx - 1] - target) <= abs(self._ts[idx] - target):
            return idx - 1
        return idx

    def nearest_within(self, target: float, max_dt: float) -> Optional[int]:
        """Return index of nearest timestamp within max_dt, or None."""
        idx = self.nearest(target)
        if abs(self._ts[idx] - target) <= max_dt:
            return idx
        return None

    def __len__(self) -> int:
        return len(self._ts)

    def __getitem__(self, idx: int) -> float:
        return float(self._ts[idx])
