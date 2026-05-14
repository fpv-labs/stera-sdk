"""Upper body pose annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from stera.core.types import Keypoint, Pose3D

UPPER_BODY_JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

UPPER_BODY_CONNECTIONS = [
    (0, 1), (0, 2),   # nose -> eyes
    (1, 3), (2, 4),   # eyes -> ears
    (5, 6),            # shoulder -> shoulder
    (5, 7), (6, 8),   # shoulders -> elbows
    (7, 9), (8, 10),  # elbows -> wrists
]


@dataclass
class UpperBodyPose:
    """Upper body pose: head + shoulders + arms.

    Parameters
    ----------
    keypoints : list of keypoints matching UPPER_BODY_JOINTS ordering.
    frame_idx : optional frame index.
    confidence : overall detection confidence.
    """

    keypoints: list[Keypoint] = field(default_factory=list)
    frame_idx: Optional[int] = None
    confidence: float = 1.0

    def as_pose3d(self) -> Pose3D:
        return Pose3D(keypoints=self.keypoints)

    @classmethod
    def from_dict(cls, d: dict) -> UpperBodyPose:
        kps = [
            Keypoint(x=kp["x"], y=kp["y"], z=kp.get("z", 0.0),
                     confidence=kp.get("confidence", 1.0),
                     name=UPPER_BODY_JOINTS[i])
            for i, kp in enumerate(d["keypoints"])
        ]
        return cls(
            keypoints=kps,
            frame_idx=d.get("frame_idx"),
            confidence=d.get("confidence", 1.0),
        )
