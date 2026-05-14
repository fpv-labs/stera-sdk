"""Shared data types used throughout the SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Keypoint:
    """A single 2D or 3D keypoint with optional confidence."""

    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0
    name: Optional[str] = None

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class BBox:
    """Axis-aligned bounding box (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    label: Optional[str] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class Pose3D:
    """A set of named 3D keypoints representing a pose."""

    keypoints: list[Keypoint] = field(default_factory=list)
    timestamp: Optional[float] = None

    def as_array(self) -> np.ndarray:
        return np.array([kp.as_array() for kp in self.keypoints])


@dataclass
class Pose6D:
    """6-DoF pose: rotation (3x3) + translation (3,)."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    timestamp: Optional[float] = None

    def as_matrix(self) -> np.ndarray:
        """Return 4x4 homogeneous transformation matrix."""
        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat

    @classmethod
    def from_matrix(cls, mat: np.ndarray, timestamp: float | None = None) -> Pose6D:
        return cls(
            rotation=mat[:3, :3].copy(),
            translation=mat[:3, 3].copy(),
            timestamp=timestamp,
        )
