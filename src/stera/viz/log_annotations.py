"""High-level helpers to log SDK annotation types to Rerun."""

from __future__ import annotations


import numpy as np
import rerun as rr

from stera.annotations.pose.camera import CameraPose
from stera.annotations.pose.body import UpperBodyPose, UPPER_BODY_CONNECTIONS
from stera.annotations.hands.hand import HandPose

# Hand skeleton connections: wrist(0) -> each finger MCP, then MCP-PIP-DIP-TIP chains.
_HAND_CONNECTIONS = (
    [(0, 1), (1, 2), (2, 3), (3, 4)]        # thumb
    + [(0, 5), (5, 6), (6, 7), (7, 8)]      # index
    + [(0, 9), (9, 10), (10, 11), (11, 12)]  # middle
    + [(0, 13), (13, 14), (14, 15), (15, 16)]  # ring
    + [(0, 17), (17, 18), (18, 19), (19, 20)]  # pinky
)


def log_camera_pose(
    entity: str,
    camera_pose: CameraPose,
    image_size: tuple[int, int] | None = None,
) -> None:
    """Log a camera pose as a Rerun pinhole + transform3D.

    Parameters
    ----------
    entity : Rerun entity path (e.g. "world/camera").
    camera_pose : CameraPose with extrinsics (and optional intrinsics).
    image_size : (width, height) needed when logging a pinhole camera.
    """
    ext = camera_pose.extrinsics
    rr.log(
        entity,
        rr.Transform3D(
            translation=ext.translation,
            mat3x3=ext.rotation,
        ),
    )
    if camera_pose.intrinsics is not None and image_size is not None:
        w, h = image_size
        rr.log(
            f"{entity}/pinhole",
            rr.Pinhole(
                image_from_camera=camera_pose.intrinsics,
                width=w,
                height=h,
            ),
        )


def log_hand_pose(
    entity: str,
    hand: HandPose,
    color: tuple[int, int, int] = (255, 200, 0),
) -> None:
    """Log a full 21-keypoint hand pose as joints + bones."""
    kps = hand.all_keypoints
    positions = np.array([[kp.x, kp.y] for kp in kps])
    labels = [kp.name for kp in kps]

    rr.log(
        f"{entity}/joints",
        rr.Points2D(positions, radii=4.0, colors=color, labels=labels),
    )

    strips = []
    for i, j in _HAND_CONNECTIONS:
        if i < len(kps) and j < len(kps):
            strips.append([positions[i], positions[j]])
    if strips:
        rr.log(
            f"{entity}/bones",
            rr.LineStrips2D(strips, radii=1.5, colors=color),
        )


def log_upper_body(
    entity: str,
    body: UpperBodyPose,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Log an upper body pose as joints + bones."""
    kps = body.keypoints
    positions = np.array([[kp.x, kp.y] for kp in kps])
    labels = [kp.name for kp in kps]

    rr.log(
        f"{entity}/joints",
        rr.Points2D(positions, radii=5.0, colors=color, labels=labels),
    )

    strips = []
    for i, j in UPPER_BODY_CONNECTIONS:
        if i < len(kps) and j < len(kps):
            strips.append([positions[i], positions[j]])
    if strips:
        rr.log(
            f"{entity}/bones",
            rr.LineStrips2D(strips, radii=2.0, colors=color),
        )


