"""Per-topic decoders: raw ROS msg -> SDK types."""

from __future__ import annotations

import numpy as np
import cv2

from stera.core.types import Pose6D


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def decode_compressed_image(msg) -> tuple[float, np.ndarray]:
    """CompressedImage -> (timestamp, RGB HxWx3 uint8)."""
    ts = _stamp_to_sec(msg.header.stamp)
    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return ts, img


def decode_depth_image(msg) -> tuple[float, np.ndarray]:
    """Image (16UC1 or 32FC1) -> (timestamp, uint16 mm depth HxW)."""
    ts = _stamp_to_sec(msg.header.stamp)
    if msg.encoding == "32FC1":
        depth = (np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width) * 1000).astype(np.uint16)
    else:
        depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
    return ts, depth


def decode_pose_stamped(msg) -> tuple[float, Pose6D]:
    """PoseStamped -> (timestamp, Pose6D)."""
    ts = _stamp_to_sec(msg.header.stamp)
    p = msg.pose.position
    q = msg.pose.orientation
    rot = _quat_to_rot(q.x, q.y, q.z, q.w)
    trans = np.array([p.x, p.y, p.z])
    return ts, Pose6D(rotation=rot, translation=trans, timestamp=ts)


def decode_camera_info(msg) -> dict:
    """CameraInfo -> dict with width, height, K (3x3), D, distortion_model."""
    return {
        "width": msg.width,
        "height": msg.height,
        "K": np.array(msg.k).reshape(3, 3),
        "D": np.array(msg.d),
        "distortion_model": msg.distortion_model,
        "R": np.array(msg.r).reshape(3, 3),
        "P": np.array(msg.p).reshape(3, 4),
    }


def decode_imu(msg) -> tuple[float, dict]:
    """Imu -> (timestamp, dict with acceleration, angular_velocity, orientation)."""
    ts = _stamp_to_sec(msg.header.stamp)
    la = msg.linear_acceleration
    av = msg.angular_velocity
    q = msg.orientation
    return ts, {
        "linear_acceleration": np.array([la.x, la.y, la.z]),
        "angular_velocity": np.array([av.x, av.y, av.z]),
        "orientation": np.array([q.x, q.y, q.z, q.w]),
    }


def decode_tf_message(msg) -> list[tuple[float, str, str, Pose6D]]:
    """TFMessage -> list of (timestamp, parent_frame, child_frame, Pose6D)."""
    results = []
    for t in msg.transforms:
        ts = _stamp_to_sec(t.header.stamp)
        tr = t.transform.translation
        rot = t.transform.rotation
        pose = Pose6D(
            rotation=_quat_to_rot(rot.x, rot.y, rot.z, rot.w),
            translation=np.array([tr.x, tr.y, tr.z]),
            timestamp=ts,
        )
        results.append((ts, t.header.frame_id, t.child_frame_id, pose))
    return results


def decode_tracking_state(msg) -> tuple[float, dict]:
    """TrackingState -> (timestamp, dict with state, reason, state_str, reason_str)."""
    ts = _stamp_to_sec(msg.header.stamp)
    return ts, {
        "state": msg.state,
        "reason": msg.reason,
        "state_str": msg.state_str,
        "reason_str": msg.reason_str,
    }


def decode_point_cloud2(msg) -> tuple[np.ndarray, np.ndarray | None]:
    """PointCloud2 -> (Nx3 float32 xyz, Nx3 uint8 rgb or None)."""
    n_pts = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_pts, msg.point_step)
    xyz = np.zeros((n_pts, 3), dtype=np.float32)
    rgb = None

    for field in msg.fields:
        off = field.offset
        if field.name == "x":
            xyz[:, 0] = np.frombuffer(raw[:, off:off + 4].tobytes(), np.float32)
        elif field.name == "y":
            xyz[:, 1] = np.frombuffer(raw[:, off:off + 4].tobytes(), np.float32)
        elif field.name == "z":
            xyz[:, 2] = np.frombuffer(raw[:, off:off + 4].tobytes(), np.float32)
        elif field.name == "rgb":
            if rgb is None:
                rgb = np.zeros((n_pts, 3), dtype=np.uint8)
            rgb_packed = np.frombuffer(raw[:, off:off + 4].tobytes(), np.uint32)
            rgb[:, 0] = (rgb_packed >> 16) & 0xFF
            rgb[:, 1] = (rgb_packed >> 8) & 0xFF
            rgb[:, 2] = rgb_packed & 0xFF

    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    if rgb is not None:
        rgb = rgb[mask]
    return xyz, rgb


def decode_mesh_marker(msg) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Marker (TRIANGLE_LIST) -> (Nx3 vertices, Mx3 triangle indices, Mx3 colors or None)."""
    pts = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
    n_tris = len(pts) // 3
    indices = np.arange(len(pts), dtype=np.uint32).reshape(n_tris, 3)
    colors = None
    if msg.colors:
        colors = np.array(
            [[c.r, c.g, c.b] for c in msg.colors], dtype=np.float32,
        )
        colors = (colors * 255).clip(0, 255).astype(np.uint8)
    return pts, indices, colors


def decode_path(msg) -> list[tuple[float, Pose6D]]:
    """Path -> list of (timestamp, Pose6D)."""
    results = []
    for pose_stamped in msg.poses:
        ts = _stamp_to_sec(pose_stamped.header.stamp)
        p = pose_stamped.pose.position
        q = pose_stamped.pose.orientation
        rot = _quat_to_rot(q.x, q.y, q.z, q.w)
        results.append((ts, Pose6D(rotation=rot, translation=np.array([p.x, p.y, p.z]), timestamp=ts)))
    return results
