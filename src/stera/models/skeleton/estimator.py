"""Upper-body skeleton estimation from FPV camera pose and wrist positions.

Generates a plausible upper-body skeleton (head, neck, shoulders, elbows, wrists)
from the egocentric camera pose and detected wrist 3D positions, using inverse
kinematics for elbow placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Joint indices

J_HEAD = 0
J_NECK = 1
J_SPINE = 2
J_L_SHOULDER = 3
J_L_ELBOW = 4
J_L_WRIST = 5
J_R_SHOULDER = 6
J_R_ELBOW = 7
J_R_WRIST = 8
J_MOUNT_CAM = 9
NUM_JOINTS = 10

JOINT_NAMES = [
    "head", "neck", "spine",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
    "mount_cam",
]

DEFAULT_EDGES = [
    (J_NECK, J_L_SHOULDER),
    (J_L_SHOULDER, J_L_ELBOW),
    (J_L_ELBOW, J_L_WRIST),
    (J_NECK, J_R_SHOULDER),
    (J_R_SHOULDER, J_R_ELBOW),
    (J_R_ELBOW, J_R_WRIST),
    (J_L_SHOULDER, J_SPINE),
    (J_R_SHOULDER, J_SPINE),
    (J_NECK, J_MOUNT_CAM),
]


@dataclass
class SkeletonConfig:
    """Tunable skeleton parameters."""

    # Neck placement (fixed distance below and behind camera)
    neck_back: float = 0.10       # how far behind cam (horizontal)
    neck_drop: float = 0.20       # how far below cam (vertical)

    shoulder_drop: float = 0.12
    neck_to_head_up: float = 0.10
    neck_to_shoulder: float = 0.18
    torso_drop: float = 0.45

    # Arm IK
    upper_arm_ratio: float = 0.55
    arm_length: float = 0.60

    up_axis: int = -1
    edges: list = field(default_factory=lambda: list(DEFAULT_EDGES))


@dataclass
class SkeletonFrame:
    """Skeleton result for a single frame."""

    joints: np.ndarray         # (NUM_JOINTS, 3), NaN for missing
    visible: np.ndarray        # (NUM_JOINTS,) bool
    edges: list[tuple[int, int]]
    fwd_horiz: np.ndarray | None = None  # horizontal forward direction
    up: np.ndarray | None = None         # world up direction

    def bone_lines(self) -> list[list[list[float]]]:
        """Line segments for visible bones: [[p1, p2], ...]."""
        lines = []
        for a, b in self.edges:
            if self.visible[a] and self.visible[b]:
                lines.append([self.joints[a].tolist(), self.joints[b].tolist()])
        return lines

    def visible_joints(self) -> np.ndarray:
        """(M, 3) array of only visible joint positions."""
        return self.joints[self.visible]


def _solve_elbow_ik(shoulder, wrist, upper_len, forearm_len, down):
    sw = wrist - shoulder
    reach = np.linalg.norm(sw)
    if reach < 1e-6:
        return shoulder + down * upper_len

    total = upper_len + forearm_len
    if reach >= total * 0.99:
        return shoulder + sw / reach * upper_len

    cos_a = np.clip(
        (upper_len**2 + reach**2 - forearm_len**2) / (2 * upper_len * reach),
        -1.0, 1.0,
    )
    angle_a = np.arccos(cos_a)

    fw = sw / reach
    side = np.cross(fw, down)
    side_len = np.linalg.norm(side)
    if side_len < 1e-6:
        side = np.cross(fw, np.array([1.0, 0.0, 0.0]))
        side_len = np.linalg.norm(side)
    side /= side_len
    perp = np.cross(side, fw)

    return shoulder + fw * (upper_len * np.cos(angle_a)) + perp * (upper_len * np.sin(angle_a))


class UpperBodyEstimator:
    """Estimate upper-body skeleton from camera pose + hand wrist positions.

    Works per-frame, with no dependency on JSON files. Feed it frames from
    MCAPReader and HandPose lists from HandTracker.

    The world frame up-axis is auto-detected from camera positions
    (the axis with the smallest range = vertical / head height).

    Usage::

        from stera.models.skeleton import UpperBodyEstimator

        estimator = UpperBodyEstimator(session=session)
        for frame in session.frames():
            hands = hand_tracker.detect_hands(frame)
            skeleton = estimator.estimate(frame, hands=hands)
            # skeleton.joints, skeleton.visible, skeleton.edges
    """

    def __init__(
        self,
        session=None,
        config: SkeletonConfig | None = None,
        R_optical_to_link: np.ndarray | None = None,
    ):
        """``session`` (MCAPReader or anything with ``R_optical_to_link``) auto-fills
        the optical-to-link rotation. Explicit ``R_optical_to_link`` still wins.
        """
        self.config = config or SkeletonConfig()
        self._upper_len = self.config.arm_length * self.config.upper_arm_ratio
        self._forearm_len = self.config.arm_length * (1.0 - self.config.upper_arm_ratio)
        self._up: np.ndarray | None = None
        self._down: np.ndarray | None = None
        if R_optical_to_link is None and session is not None:
            R_optical_to_link = getattr(session, "R_optical_to_link", None)
        self._R_o2l = R_optical_to_link
        # Link-frame column indices for camera forward, up, and right
        self._link_fwd_col: int | None = None
        self._link_fwd_sign: float = 1.0
        self._link_up_col: int | None = None
        self._link_up_sign: float = 1.0
        self._link_right_col: int | None = None
        self._link_right_sign: float = 1.0
        # Buffer for auto-detecting world up from positions
        self._up_buffer: list[np.ndarray] = []

    def reset(self) -> None:
        """Reset state (call between sequences)."""
        pass

    def _init_link_axes(self) -> None:
        """Determine which link-frame columns are camera forward, up, and right."""
        if self._link_fwd_col is not None:
            return
        if self._R_o2l is not None:
            R = self._R_o2l
            # optical: X=right, Y=down, Z=forward
            link_fwd = R @ np.array([0.0, 0.0, 1.0])
            self._link_fwd_col = int(np.argmax(np.abs(link_fwd)))
            self._link_fwd_sign = np.sign(link_fwd[self._link_fwd_col])
            link_up = R @ np.array([0.0, -1.0, 0.0])
            self._link_up_col = int(np.argmax(np.abs(link_up)))
            self._link_up_sign = np.sign(link_up[self._link_up_col])
            link_right = R @ np.array([1.0, 0.0, 0.0])
            self._link_right_col = int(np.argmax(np.abs(link_right)))
            self._link_right_sign = np.sign(link_right[self._link_right_col])
        else:
            self._link_fwd_col = 0
            self._link_fwd_sign = 1.0
            self._link_up_col = 2
            self._link_up_sign = 1.0
            self._link_right_col = 1
            self._link_right_sign = -1.0

    def _ensure_up(self, t_cam: np.ndarray, R_cam: np.ndarray) -> bool:
        """Determine the world up/down vectors. Returns True when ready."""
        if self._up is not None:
            return True

        self._init_link_axes()

        ax = self.config.up_axis
        if ax >= 0:
            cam_up_world = R_cam[:, self._link_up_col] * self._link_up_sign
            sign = np.sign(cam_up_world[ax]) if cam_up_world[ax] != 0 else 1.0
            self._up = np.zeros(3)
            self._up[ax] = sign
            self._down = -self._up
            return True

        # Detect up direction from the camera's local "up" in world frame.
        # The SLAM system outputs Y-up world coordinates, but we verify the
        # sign from the camera orientation: camera-up should have a positive
        # component along the world vertical axis.
        cam_up_world = R_cam[:, self._link_up_col] * self._link_up_sign
        self._up_buffer.append(cam_up_world.copy())

        if len(self._up_buffer) < 10:
            return False

        mean_cam_up = np.array(self._up_buffer).mean(axis=0)

        # Default: Y-up (standard for visual-inertial SLAM)
        ax = 1
        sign = np.sign(mean_cam_up[ax]) if mean_cam_up[ax] != 0 else 1.0

        self._up = np.zeros(3)
        self._up[ax] = sign
        self._down = -self._up
        self._up_buffer = []
        return True

    def estimate(
        self,
        frame,
        hands=None,
    ) -> SkeletonFrame | None:
        """Estimate skeleton for a single frame.

        Parameters
        ----------
        frame : SyncedFrame with camera_pose.
        hands : list of HandPose (from HandTracker).

        Returns None if the frame has no camera pose.
        """
        if frame.camera_pose is None:
            return None

        R_cam = frame.camera_pose.rotation
        t_cam = frame.camera_pose.translation

        if not self._ensure_up(t_cam, R_cam):
            return None

        l_wrist_world, r_wrist_world = self._extract_wrists(R_cam, t_cam, hands=hands)

        joints, visible, fwd_horiz, up_vec = self._build_skeleton(t_cam, R_cam, l_wrist_world, r_wrist_world)

        return SkeletonFrame(
            joints=joints, visible=visible, edges=self.config.edges,
            fwd_horiz=fwd_horiz, up=up_vec,
        )

    def _extract_wrists(self, R_cam, t_cam, hands=None):
        """Extract left/right wrist world positions from HandPose detections."""
        from stera.core.transforms import optical_to_world

        l_wrist = r_wrist = None
        if hands:
            for hp in hands:
                wrist_3d = np.array([hp.wrist.x, hp.wrist.y, hp.wrist.z])
                if wrist_3d[2] == 0.0:
                    continue
                wrist_world = optical_to_world(wrist_3d.reshape(1, 3), R_cam, t_cam, self._R_o2l)[0]
                if hp.hand_side == "left":
                    l_wrist = wrist_world
                else:
                    r_wrist = wrist_world

        return l_wrist, r_wrist

    def _build_skeleton(self, cam_pos, R_cam, l_wrist, r_wrist):
        cfg = self.config
        up = self._up
        down = self._down
        up_ax = int(np.argmax(np.abs(up)))

        joints = np.full((NUM_JOINTS, 3), np.nan)
        visible = np.zeros(NUM_JOINTS, dtype=bool)

        cam_fwd = R_cam[:, self._link_fwd_col] * self._link_fwd_sign
        cam_right = R_cam[:, self._link_right_col] * self._link_right_sign

        # Body forward = camera forward projected to horizontal plane
        fwd_horiz = cam_fwd.copy()
        fwd_horiz[up_ax] = 0
        fwd_norm = np.linalg.norm(fwd_horiz)
        if fwd_norm > 1e-6:
            fwd_horiz /= fwd_norm
        else:
            fallback = np.zeros(3)
            fallback[(up_ax + 1) % 3] = 1.0
            fwd_horiz = fallback

        # Body right = camera right projected to horizontal plane
        right_horiz = cam_right.copy()
        right_horiz[up_ax] = 0
        rn = np.linalg.norm(right_horiz)
        if rn > 1e-6:
            right_horiz /= rn
        else:
            right_horiz = np.cross(fwd_horiz, up)
            right_horiz /= np.linalg.norm(right_horiz) + 1e-9

        # Neck: fixed distance below and behind camera, using HORIZONTAL
        # directions only, so head tilt doesn't change neck position.
        neck = cam_pos.copy()
        neck -= fwd_horiz * cfg.neck_back   # behind camera (horizontal)
        neck += down * cfg.neck_drop         # below camera (vertical)

        # Camera mount
        joints[J_MOUNT_CAM] = cam_pos
        visible[J_MOUNT_CAM] = True

        # Neck
        joints[J_NECK] = neck
        visible[J_NECK] = True

        # Head
        joints[J_HEAD] = neck + up * cfg.neck_to_head_up
        visible[J_HEAD] = True

        # Shoulders
        shoulder_base = neck + down * cfg.shoulder_drop
        # Egocentric (head-mounted) view: camera right == wearer's right.
        l_shoulder = shoulder_base - right_horiz * cfg.neck_to_shoulder
        r_shoulder = shoulder_base + right_horiz * cfg.neck_to_shoulder
        joints[J_L_SHOULDER] = l_shoulder
        joints[J_R_SHOULDER] = r_shoulder
        visible[J_L_SHOULDER] = True
        visible[J_R_SHOULDER] = True

        # Spine
        shoulder_mid = (l_shoulder + r_shoulder) * 0.5
        joints[J_SPINE] = shoulder_mid + down * cfg.torso_drop
        visible[J_SPINE] = True

        # Arms via IK
        if l_wrist is not None:
            joints[J_L_WRIST] = l_wrist
            visible[J_L_WRIST] = True
            joints[J_L_ELBOW] = _solve_elbow_ik(
                l_shoulder, l_wrist, self._upper_len, self._forearm_len, down,
            )
            visible[J_L_ELBOW] = True

        if r_wrist is not None:
            joints[J_R_WRIST] = r_wrist
            visible[J_R_WRIST] = True
            joints[J_R_ELBOW] = _solve_elbow_ik(
                r_shoulder, r_wrist, self._upper_len, self._forearm_len, down,
            )
            visible[J_R_ELBOW] = True

        return joints, visible, fwd_horiz, up

