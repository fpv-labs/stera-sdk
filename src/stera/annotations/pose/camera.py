"""Camera pose (6-DoF extrinsics + optional intrinsics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from stera.core.types import Pose6D


@dataclass
class CameraPose:
    """Camera extrinsics (6-DoF) with optional intrinsic matrix.

    Parameters
    ----------
    extrinsics : 6-DoF pose (rotation + translation) in world frame.
    intrinsics : optional 3x3 camera intrinsic matrix (K).
    frame_idx  : optional frame index this pose corresponds to.
    """

    extrinsics: Pose6D
    intrinsics: Optional[np.ndarray] = None  # (3, 3)
    frame_idx: Optional[int] = None

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project Nx3 world points to Nx2 image coordinates.

        Requires intrinsics to be set.
        """
        if self.intrinsics is None:
            raise ValueError("Intrinsics required for projection")
        R = self.extrinsics.rotation
        t = self.extrinsics.translation
        cam_pts = (R @ points_3d.T).T + t
        proj = (self.intrinsics @ cam_pts.T).T
        return proj[:, :2] / proj[:, 2:3]

    @classmethod
    def from_dict(cls, d: dict) -> CameraPose:
        ext = Pose6D(
            rotation=np.array(d["rotation"]),
            translation=np.array(d["translation"]),
            timestamp=d.get("timestamp"),
        )
        intrinsics = np.array(d["intrinsics"]) if "intrinsics" in d else None
        return cls(extrinsics=ext, intrinsics=intrinsics, frame_idx=d.get("frame_idx"))

    def to_dict(self) -> dict:
        d = {
            "rotation": self.extrinsics.rotation.tolist(),
            "translation": self.extrinsics.translation.tolist(),
        }
        if self.extrinsics.timestamp is not None:
            d["timestamp"] = self.extrinsics.timestamp
        if self.intrinsics is not None:
            d["intrinsics"] = self.intrinsics.tolist()
        if self.frame_idx is not None:
            d["frame_idx"] = self.frame_idx
        return d
