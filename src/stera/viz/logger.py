"""Central Rerun recording wrapper for stera-sdk."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import rerun as rr

from stera.core.types import BBox, Keypoint


class FPVLogger:
    """Thin wrapper around a Rerun recording for FPV data.

    Parameters
    ----------
    app_id : Rerun application ID (shown in the viewer title).
    recording_id : optional unique recording ID.
    spawn : if True, automatically spawn the Rerun viewer on init.
    """

    def __init__(
        self,
        app_id: str = "stera",
        recording_id: Optional[str] = None,
        spawn: bool = True,
    ):
        rr.init(app_id, recording_id=recording_id, spawn=spawn)

    def set_time(self, frame_idx: int) -> None:
        """Set the current timeline position (frame index)."""
        rr.set_time_sequence("frame", frame_idx)

    def log_image(self, entity: str, image: np.ndarray) -> None:
        """Log an RGB image."""
        rr.log(entity, rr.Image(image))

    def log_points_2d(
        self,
        entity: str,
        keypoints: Sequence[Keypoint],
        radii: float = 4.0,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        """Log 2D keypoints as a point cloud."""
        positions = np.array([[kp.x, kp.y] for kp in keypoints])
        labels = [kp.name for kp in keypoints if kp.name]
        rr.log(
            entity,
            rr.Points2D(
                positions,
                radii=radii,
                colors=colors,
                labels=labels or None,
            ),
        )

    def log_points_3d(
        self,
        entity: str,
        keypoints: Sequence[Keypoint],
        radii: float = 0.02,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        """Log 3D keypoints as a point cloud."""
        positions = np.array([kp.as_array() for kp in keypoints])
        labels = [kp.name for kp in keypoints if kp.name]
        rr.log(
            entity,
            rr.Points3D(
                positions,
                radii=radii,
                colors=colors,
                labels=labels or None,
            ),
        )

    def log_skeleton(
        self,
        entity: str,
        keypoints: Sequence[Keypoint],
        connections: Sequence[tuple[int, int]],
        radii: float = 1.5,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Log a 2D skeleton (keypoints + line segments)."""
        self.log_points_2d(f"{entity}/joints", keypoints, radii=4.0)
        strips = []
        for i, j in connections:
            strips.append([[keypoints[i].x, keypoints[i].y],
                           [keypoints[j].x, keypoints[j].y]])
        rr.log(
            f"{entity}/bones",
            rr.LineStrips2D(strips, radii=radii, colors=color),
        )

    def log_skeleton_3d(
        self,
        entity: str,
        keypoints: Sequence[Keypoint],
        connections: Sequence[tuple[int, int]],
        radii: float = 0.01,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Log a 3D skeleton (keypoints + line segments)."""
        self.log_points_3d(f"{entity}/joints", keypoints, radii=0.02)
        strips = []
        for i, j in connections:
            strips.append([keypoints[i].as_array(), keypoints[j].as_array()])
        rr.log(
            f"{entity}/bones",
            rr.LineStrips3D(strips, radii=radii, colors=color),
        )

    def log_bboxes(
        self,
        entity: str,
        bboxes: Sequence[BBox],
        color: tuple[int, int, int] = (255, 0, 0),
    ) -> None:
        """Log 2D bounding boxes."""
        mins = np.array([[b.x1, b.y1] for b in bboxes])
        sizes = np.array([[b.width, b.height] for b in bboxes])
        labels = [b.label for b in bboxes if b.label]
        rr.log(
            entity,
            rr.Boxes2D(
                mins=mins,
                sizes=sizes,
                colors=color,
                labels=labels or None,
            ),
        )

    def log_text(self, entity: str, text: str) -> None:
        """Log a text annotation."""
        rr.log(entity, rr.TextLog(text))
