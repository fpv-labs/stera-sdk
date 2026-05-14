"""Hand pose: wrist + optional per-finger keypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from stera.core.types import Keypoint


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_JOINTS = ["mcp", "pip", "dip", "tip"]


@dataclass
class HandPose:
    """Hand pose: wrist keypoint plus optional per-finger joints.

    Parameters
    ----------
    wrist : wrist keypoint (2D pixel coords or 3D camera-frame metres).
    fingers : dict mapping finger name → ``[MCP, PIP, DIP, tip]`` keypoints.
        Empty when only the wrist was detected (e.g. wrist-only trackers
        like MediaPipe's wrist mode or ArUco markers).
    hand_side : "left" or "right".
    frame_idx : optional frame index.
    confidence : overall hand detection confidence.
    """

    wrist: Keypoint
    fingers: dict[str, list[Keypoint]] = field(default_factory=dict)
    hand_side: str = "right"
    frame_idx: Optional[int] = None
    confidence: float = 1.0

    @property
    def has_fingers(self) -> bool:
        return bool(self.fingers)

    @property
    def all_keypoints(self) -> list[Keypoint]:
        """Flat keypoint list.

        Full hand → 21 keypoints in MANO canonical order: wrist, then
        ``[mcp, pip, dip, tip]`` for thumb/index/middle/ring/pinky.
        Wrist-only → ``[wrist]``.
        """
        if not self.fingers:
            return [self.wrist]
        kps = [self.wrist]
        for name in FINGER_NAMES:
            kps.extend(self.fingers[name])
        return kps
