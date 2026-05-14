"""Coordinate frame transforms for FPV cameras."""

from __future__ import annotations

import numpy as np


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix to quaternion (qx, qy, qz, qw)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


# Fixed rotation: camera_optical_frame -> camera_link
OPTICAL_QUAT_XYZW = [-0.7071, 0.0, -0.7071, 0.0]
R_OPTICAL_TO_LINK = quat_to_rot(0.7071, 0.0, 0.7071, 0.0)


def optical_to_world(
    pts: np.ndarray,
    R_world: np.ndarray,
    t_world: np.ndarray,
    R_o2l: np.ndarray | None = None,
) -> np.ndarray:
    """Transform points from camera optical frame to world frame.

    Parameters
    ----------
    pts : (N, 3) points in camera optical frame.
    R_world : (3, 3) camera-to-world rotation.
    t_world : (3,) camera position in world.
    R_o2l : (3, 3) optical-to-link rotation. Defaults to R_OPTICAL_TO_LINK.
    """
    if R_o2l is None:
        R_o2l = R_OPTICAL_TO_LINK
    pts_link = (R_o2l @ pts.T).T
    return (R_world @ pts_link.T).T + t_world


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    max_pts: int = 50_000,
    min_depth: float = 0.3,
    max_depth: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth image + RGB to colored 3D point cloud in camera optical frame.

    Parameters
    ----------
    depth : (H, W) uint16 depth in mm.
    rgb : (H, W, 3) uint8 RGB image (can be different resolution).
    K : (3, 3) depth camera intrinsics.
    max_pts : Subsample to this many points.
    min_depth : Minimum depth in meters.
    max_depth : Maximum depth in meters.

    Returns
    -------
    (N, 3) float32 xyz, (N, 3) uint8 RGB colors.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dh, dw = depth.shape[:2]
    v_grid, u_grid = np.mgrid[0:dh, 0:dw]
    z = depth.astype(np.float32) / 1000.0
    valid = (z > min_depth) & (z < max_depth)

    x = (u_grid - cx) * z / fx
    y = (v_grid - cy) * z / fy
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1)

    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    if len(pts) > max_pts:
        idx = np.random.default_rng(42).choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        u_valid = u_grid[valid][idx]
        v_valid = v_grid[valid][idx]
    else:
        u_valid = u_grid[valid]
        v_valid = v_grid[valid]

    rh, rw = rgb.shape[:2]
    u_rgb = np.clip((u_valid * rw / dw).astype(int), 0, rw - 1)
    v_rgb = np.clip((v_valid * rh / dh).astype(int), 0, rh - 1)
    colors = rgb[v_rgb, u_rgb]

    return pts, colors
