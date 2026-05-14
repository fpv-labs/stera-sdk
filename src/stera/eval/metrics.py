"""All metric computations for a recorded session.

Each compute_* function returns a plain dict (or None when the underlying
stream is missing) so the report layer can render whatever is available
without doing any logic.
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Subsample knobs (kept private — Evaluate exposes no kwargs).
_DEPTH_SAMPLE_TARGET = 200            # depth frames sampled for stats
_DEPTH_HIST_PIXEL_TARGET = 1_000_000  # pixels accumulated into the global histogram


def _pct(numer: float, denom: float) -> float:
    return 100.0 * numer / denom if denom > 0 else 0.0


def _percentiles(arr: np.ndarray, qs=(5, 50, 95)) -> dict:
    if arr.size == 0:
        return {f"p{q}": None for q in qs}
    vals = np.percentile(arr, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, vals)}


def _safe_div(a, b):
    return float(a / b) if b else None


# Recording metadata

def compute_recording(session) -> dict:
    path = Path(session.path)
    try:
        summ = session._reader.summary()
    except Exception:
        summ = {}

    size_bytes = path.stat().st_size if path.exists() else 0
    start = summ.get("start_time", 0.0)
    end = summ.get("end_time", 0.0)
    duration = session.duration
    topics = summ.get("topics", {}) or {}

    # Topics expected vs missing
    ref = set(getattr(session, "REFERENCE_TOPICS", ()))
    present = {t for t, c in topics.items() if c > 0}
    missing = sorted(ref - present)

    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": int(size_bytes),
        "size_mb": size_bytes / (1024 * 1024),
        "duration_s": float(duration),
        "duration_hms": _fmt_hms(duration),
        "start_time": float(start),
        "end_time": float(end),
        "start_iso": _iso(start),
        "end_iso": _iso(end),
        "weekday": _weekday(start),
        "message_count": int(summ.get("message_count", 0)),
        "topic_counts": {str(k): int(v) for k, v in sorted(topics.items())},
        "topics_present_count": len(present),
        "missing_reference_topics": missing,
    }


def _fmt_hms(seconds: float) -> str:
    s = max(int(round(seconds)), 0)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _iso(epoch_s: float) -> str:
    if not epoch_s:
        return ""
    try:
        return datetime.fromtimestamp(epoch_s).isoformat(timespec="seconds")
    except (OSError, ValueError):
        return ""


def _weekday(epoch_s: float) -> str:
    if not epoch_s:
        return ""
    try:
        return datetime.fromtimestamp(epoch_s).strftime("%A")
    except (OSError, ValueError):
        return ""


# RGB stream

def compute_rgb(session) -> dict:
    rgb_ts = np.asarray(session._rgb_ts(), dtype=np.float64)
    n = len(rgb_ts)
    duration = session.duration

    intr = session.rgb_intrinsics
    intr_dict = _intrinsics_dict(intr)

    dts = np.diff(rgb_ts) if n > 1 else np.array([])
    median_dt = float(np.median(dts)) if dts.size else None
    gap_count = int((dts > (2 * median_dt)).sum()) if median_dt else 0

    return {
        "frame_count": int(n),
        "effective_fps": _safe_div(n, duration),
        "median_dt_ms": median_dt * 1000 if median_dt else None,
        "min_dt_ms": float(dts.min() * 1000) if dts.size else None,
        "max_dt_ms": float(dts.max() * 1000) if dts.size else None,
        "dt_std_ms": float(dts.std() * 1000) if dts.size else None,
        "gap_count": gap_count,
        "intrinsics": intr_dict,
        "timestamps": rgb_ts,  # used by plots / sync
    }


def _intrinsics_dict(intr) -> dict | None:
    if intr is None:
        return None
    K = intr.K
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = int(intr.width), int(intr.height)
    fov_x = 2 * math.degrees(math.atan(w / (2 * fx))) if fx else None
    fov_y = 2 * math.degrees(math.atan(h / (2 * fy))) if fy else None
    return {
        "width": w,
        "height": h,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fx_over_fy": fx / fy if fy else None,
        "fov_x_deg": fov_x,
        "fov_y_deg": fov_y,
        "aspect_ratio": w / h if h else None,
        "principal_offset_px": (cx - w / 2, cy - h / 2),
        "principal_offset_pct": (
            100 * (cx - w / 2) / w if w else None,
            100 * (cy - h / 2) / h if h else None,
        ),
        "distortion_model": intr.distortion_model,
        "distortion": [float(x) for x in np.asarray(intr.D).ravel()],
    }


# Depth stream

def compute_depth(session) -> dict | None:
    n_total = session.num_depth_frames
    if n_total == 0:
        return None

    depth_ts = np.asarray(session._depth_ts(), dtype=np.float64)
    duration = session.duration
    intr = session.depth_intrinsics

    dts = np.diff(depth_ts) if len(depth_ts) > 1 else np.array([])
    median_dt = float(np.median(dts)) if dts.size else None

    every_n = max(1, n_total // _DEPTH_SAMPLE_TARGET)
    target_pixels_per_frame = max(
        1, _DEPTH_HIST_PIXEL_TARGET // max(_DEPTH_SAMPLE_TARGET, 1),
    )

    valid_pct_series: list[float] = []
    ts_series: list[float] = []
    mean_depth_series: list[float] = []
    empty_count = 0
    hist_samples: list[np.ndarray] = []
    global_min = math.inf
    global_max = 0.0

    logger.info(
        "Sampling %d/%d depth frames for stats", n_total // every_n + 1, n_total,
    )
    rng = np.random.default_rng(0)
    for i, (ts, depth) in enumerate(session.depth_frames()):
        if i % every_n != 0:
            continue
        if depth is None or depth.size == 0:
            empty_count += 1
            continue
        z = depth.astype(np.float32) / 1000.0
        valid = z > 0
        n_valid = int(valid.sum())
        valid_pct_series.append(_pct(n_valid, depth.size))
        ts_series.append(float(ts))
        if n_valid == 0:
            empty_count += 1
            mean_depth_series.append(float("nan"))
            continue
        z_valid = z[valid]
        mean_depth_series.append(float(z_valid.mean()))
        global_min = min(global_min, float(z_valid.min()))
        global_max = max(global_max, float(z_valid.max()))
        # Subsample pixels for the global histogram.
        if z_valid.size > target_pixels_per_frame:
            idx = rng.choice(z_valid.size, target_pixels_per_frame, replace=False)
            hist_samples.append(z_valid[idx])
        else:
            hist_samples.append(z_valid)

    all_samples = (
        np.concatenate(hist_samples) if hist_samples else np.zeros(0, dtype=np.float32)
    )

    bins = [0, 1, 2, 5, np.inf]
    if all_samples.size:
        hist, _ = np.histogram(all_samples, bins=bins)
        hist_counts = {
            "<1m": int(hist[0]),
            "1-2m": int(hist[1]),
            "2-5m": int(hist[2]),
            ">5m": int(hist[3]),
        }
        hist_pct = {k: _pct(v, all_samples.size) for k, v in hist_counts.items()}
    else:
        hist_counts = hist_pct = {}

    valid_pct_arr = np.asarray(valid_pct_series, dtype=np.float32)
    return {
        "frame_count": int(n_total),
        "effective_fps": _safe_div(n_total, duration),
        "median_dt_ms": median_dt * 1000 if median_dt else None,
        "intrinsics": _intrinsics_dict(intr),
        "sampled_frames": int(valid_pct_arr.size),
        "valid_pct_mean": float(valid_pct_arr.mean()) if valid_pct_arr.size else None,
        "valid_pct_min": float(valid_pct_arr.min()) if valid_pct_arr.size else None,
        "valid_pct_max": float(valid_pct_arr.max()) if valid_pct_arr.size else None,
        "valid_pct_std": float(valid_pct_arr.std()) if valid_pct_arr.size else None,
        "global_min_m": float(global_min) if math.isfinite(global_min) else None,
        "global_max_m": float(global_max) if global_max > 0 else None,
        "depth_percentiles_m": _percentiles(all_samples) if all_samples.size else {},
        "depth_hist_counts": hist_counts,
        "depth_hist_pct": hist_pct,
        "empty_frame_count": int(empty_count),
        # series for plots
        "ts_series": ts_series,
        "valid_pct_series": valid_pct_series,
        "mean_depth_series": mean_depth_series,
        "global_depth_samples": all_samples,
    }


# Camera trajectory

def compute_trajectory(session) -> dict | None:
    pose_pairs = session.all_camera_poses()
    if not pose_pairs:
        return None

    ts = np.array([p[0] for p in pose_pairs], dtype=np.float64)
    pos = np.array([p[1].translation for p in pose_pairs], dtype=np.float64)
    rots = np.stack([p[1].rotation for p in pose_pairs], axis=0)
    n = len(ts)
    duration = session.duration

    deltas = np.diff(pos, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    path_length = float(seg_lens.sum())
    net_disp = float(np.linalg.norm(pos[-1] - pos[0]))

    dts = np.diff(ts)
    speeds = seg_lens / np.maximum(dts, 1e-6) if dts.size else np.array([])
    # Acceleration magnitude between consecutive segments.
    if speeds.size > 1:
        accel = np.diff(speeds) / np.maximum(dts[1:], 1e-6)
        accel_abs = np.abs(accel)
    else:
        accel_abs = np.array([])

    # Heading from camera forward axis (Z in optical, but we project pos deltas
    # onto horizontal plane for a stable compass heading).
    horiz = deltas.copy()
    horiz[:, 1] = 0.0  # Y-up world
    headings = np.degrees(np.arctan2(horiz[:, 0], horiz[:, 2]))
    head_unwrap = np.unwrap(np.radians(headings))
    cumulative_rotation_deg = float(np.degrees(
        np.abs(np.diff(head_unwrap)).sum()
    )) if head_unwrap.size > 1 else 0.0

    # Turn count: heading change > 45° within 1 second window.
    turn_count = 0
    if head_unwrap.size > 1 and dts.size:
        # Cumulative heading change in a rolling 1s window via simple loop.
        turn_count = int((np.abs(np.diff(head_unwrap)) > math.radians(45)).sum())

    # Stationary periods: speed < 0.05 m/s.
    stationary_mask = speeds < 0.05 if speeds.size else np.array([], dtype=bool)
    stationary_duration = float(
        np.sum(dts[stationary_mask]) if stationary_mask.size else 0.0
    )

    # 2D top-down footprint area: convex-hull over xz.
    xz = pos[:, [0, 2]]
    footprint_area = _convex_hull_area(xz)

    # Per-frame angular rates from rotation matrices.
    yaw_rate = pitch_rate = roll_rate = None
    if n > 1 and dts.size:
        eulers = np.array([_rot_to_euler_zyx(R) for R in rots])  # (n, 3): yaw, pitch, roll
        d_eul = np.unwrap(eulers, axis=0)
        d_eul = np.diff(d_eul, axis=0) / np.maximum(dts[:, None], 1e-6)
        yaw_rate = float(np.degrees(np.median(np.abs(d_eul[:, 0]))))
        pitch_rate = float(np.degrees(np.median(np.abs(d_eul[:, 1]))))
        roll_rate = float(np.degrees(np.median(np.abs(d_eul[:, 2]))))

    bbox_min = pos.min(axis=0)
    bbox_max = pos.max(axis=0)
    bbox_ext = bbox_max - bbox_min

    return {
        "pose_count": int(n),
        "effective_rate_hz": _safe_div(n, duration),
        "path_length_m": path_length,
        "net_displacement_m": net_disp,
        "tortuosity": _safe_div(path_length, net_disp),
        "bbox_min": [float(x) for x in bbox_min],
        "bbox_max": [float(x) for x in bbox_max],
        "bbox_extents": [float(x) for x in bbox_ext],
        "bbox_volume_m3": float(np.prod(bbox_ext)),
        "footprint_area_m2": footprint_area,
        "height_min_m": float(bbox_min[1]),
        "height_max_m": float(bbox_max[1]),
        "height_mean_m": float(pos[:, 1].mean()),
        "height_std_m": float(pos[:, 1].std()),
        "speed_mean_mps": float(speeds.mean()) if speeds.size else None,
        "speed_median_mps": float(np.median(speeds)) if speeds.size else None,
        "speed_p95_mps": float(np.percentile(speeds, 95)) if speeds.size else None,
        "speed_max_mps": float(speeds.max()) if speeds.size else None,
        "accel_mean_mps2": float(accel_abs.mean()) if accel_abs.size else None,
        "accel_max_mps2": float(accel_abs.max()) if accel_abs.size else None,
        "yaw_rate_deg_per_s": yaw_rate,
        "pitch_rate_deg_per_s": pitch_rate,
        "roll_rate_deg_per_s": roll_rate,
        "cumulative_rotation_deg": cumulative_rotation_deg,
        "turn_count": turn_count,
        "stationary_duration_s": stationary_duration,
        "stationary_pct": _pct(stationary_duration, duration),
        # series for plots
        "ts_series": ts,
        "positions": pos,
        "speed_ts": ts[1:] if dts.size else np.array([]),
        "speed_series": speeds,
        "headings_deg": headings,
    }


def _convex_hull_area(pts: np.ndarray) -> float:
    """Convex-hull area in the plane (Andrew's monotone chain)."""
    if len(pts) < 3:
        return 0.0
    p = np.unique(pts, axis=0)
    if len(p) < 3:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _rot_to_euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX (yaw, pitch, roll) Euler angles in radians."""
    sy = -R[2, 0]
    sy = max(min(sy, 1.0), -1.0)
    pitch = math.asin(sy)
    if abs(sy) > 0.9999:
        yaw = math.atan2(-R[0, 1], R[1, 1])
        roll = 0.0
    else:
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


# IMU

def compute_imu(session) -> dict | None:
    pairs = session.all_imu_samples()
    if not pairs:
        return None

    ts = np.array([p[0] for p in pairs], dtype=np.float64)
    accel = np.stack([p[1]["linear_acceleration"] for p in pairs]).astype(np.float64)
    gyro = np.stack([p[1]["angular_velocity"] for p in pairs]).astype(np.float64)
    n = len(ts)
    duration = float(ts[-1] - ts[0]) if n > 1 else 1e-3

    dts = np.diff(ts)
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    gravity_dir = accel.mean(axis=0)
    grav_norm = float(np.linalg.norm(gravity_dir))

    motion_mask = accel_mag > (grav_norm * 1.1) if grav_norm else accel_mag > 11.0
    rate = n / duration if duration > 0 else None
    motion_duration = float(np.sum(dts[motion_mask[:-1]])) if dts.size else 0.0

    return {
        "sample_count": int(n),
        "effective_rate_hz": rate,
        "rate_jitter_ms": float(dts.std() * 1000) if dts.size else None,
        "accel_axis_mean": [float(v) for v in accel.mean(0)],
        "accel_axis_std": [float(v) for v in accel.std(0)],
        "accel_axis_min": [float(v) for v in accel.min(0)],
        "accel_axis_max": [float(v) for v in accel.max(0)],
        "accel_mag_mean": float(accel_mag.mean()),
        "accel_mag_std": float(accel_mag.std()),
        "accel_mag_p95": float(np.percentile(accel_mag, 95)),
        "accel_mag_max": float(accel_mag.max()),
        "gyro_axis_mean": [float(v) for v in gyro.mean(0)],
        "gyro_axis_std": [float(v) for v in gyro.std(0)],
        "gyro_axis_min": [float(v) for v in gyro.min(0)],
        "gyro_axis_max": [float(v) for v in gyro.max(0)],
        "gyro_mag_mean": float(gyro_mag.mean()),
        "gyro_mag_std": float(gyro_mag.std()),
        "gyro_mag_p95": float(np.percentile(gyro_mag, 95)),
        "gyro_mag_max": float(gyro_mag.max()),
        "gravity_vector": [float(v) for v in gravity_dir],
        "gravity_magnitude": grav_norm,
        "gravity_deviation": abs(grav_norm - 9.81),
        "jolt_count": int((accel_mag > 20.0).sum()),
        "high_rotation_events": int((gyro_mag > 2.0).sum()),
        "motion_duration_s": motion_duration,
        "still_duration_s": max(duration - motion_duration, 0.0),
        # series for plots
        "ts_series": ts,
        "accel_mag_series": accel_mag,
        "gyro_mag_series": gyro_mag,
    }


# Tracking state

def compute_tracking_state(session) -> dict | None:
    states = list(session.tracking_states())
    if not states:
        return None
    counts = Counter(d.get("state_str", str(d.get("state", "?"))) for _, d in states)
    total = sum(counts.values())
    return {
        "message_count": total,
        "state_counts": dict(counts),
        "state_pct": {k: _pct(v, total) for k, v in counts.items()},
    }


# TF transforms

def compute_tf(session) -> dict | None:
    try:
        tfs = session.tf_transforms()
    except Exception as e:
        logger.warning("compute_tf failed: %s", e)
        return None
    if not tfs:
        return None

    duration = max(session.duration, 1e-3)
    pair_counter: Counter = Counter()
    for _ts, parent, child, _pose in tfs:
        pair_counter[(parent, child)] += 1

    pairs = [
        {
            "parent": parent,
            "child": child,
            "count": count,
            "rate_hz": count / duration,
        }
        for (parent, child), count in pair_counter.most_common()
    ]

    return {
        "message_count": int(sum(pair_counter.values())),
        "unique_pair_count": len(pair_counter),
        "pairs": pairs,
    }


# Trajectory topic vs /camera/pose comparison

def compute_trajectory_topic(session) -> dict | None:
    traj = session.trajectory()
    if not traj:
        return None
    pos = np.array([p[1].translation for p in traj], dtype=np.float64)
    deltas = np.diff(pos, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1) if len(pos) > 1 else np.array([])
    return {
        "pose_count": int(len(traj)),
        "path_length_m": float(seg_lens.sum()),
    }


# Mesh

def compute_mesh(session) -> dict | None:
    try:
        mesh = session.mesh()
    except Exception as e:
        logger.warning("compute_mesh failed: %s", e)
        return None
    if mesh is None:
        return None

    verts, faces, colors = mesh
    if len(verts) == 0 or len(faces) == 0:
        return None

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_ext = bbox_max - bbox_min
    bbox_vol = float(np.prod(bbox_ext))

    # Surface area = sum of triangle areas via cross product.
    tri = verts[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    tri_areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    surface_area = float(tri_areas.sum())

    edges = np.concatenate([
        tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]],
    ], axis=0)
    edge_lens = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)

    color_coverage = None
    if colors is not None and len(colors) == len(verts):
        # "Default" placeholder in the SDK is grey [128,128,128].
        non_default = np.any(colors != 128, axis=1)
        color_coverage = _pct(int(non_default.sum()), len(verts))

    return {
        "vertex_count": int(len(verts)),
        "face_count": int(len(faces)),
        "bbox_extents": [float(x) for x in bbox_ext],
        "bbox_volume_m3": bbox_vol,
        "surface_area_m2": surface_area,
        "edge_length_mean_m": float(edge_lens.mean()),
        "edge_length_p5_m": float(np.percentile(edge_lens, 5)),
        "edge_length_p95_m": float(np.percentile(edge_lens, 95)),
        "color_coverage_pct": color_coverage,
        "verts_per_m2": _safe_div(len(verts), surface_area),
        "faces_per_m2": _safe_div(len(faces), surface_area),
    }


# Point cloud

def compute_point_cloud(session) -> dict | None:
    try:
        xyz, rgb = session.point_cloud(source="auto")
    except Exception as e:
        logger.warning("compute_point_cloud failed: %s", e)
        return None
    if xyz is None or len(xyz) == 0:
        return None

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    ext = bbox_max - bbox_min
    vol = float(np.prod(ext))
    color_coverage = None
    if rgb is not None and len(rgb) == len(xyz):
        nz = np.any(rgb != 0, axis=1)
        color_coverage = _pct(int(nz.sum()), len(rgb))

    return {
        "point_count": int(len(xyz)),
        "bbox_extents": [float(x) for x in ext],
        "bbox_volume_m3": vol,
        "density_pts_per_m3": _safe_div(len(xyz), vol),
        "color_coverage_pct": color_coverage,
    }


# Sync quality

def compute_sync(session, rgb_metrics: dict) -> dict:
    rgb_ts = np.asarray(rgb_metrics.get("timestamps", []), dtype=np.float64)
    depth_ts = np.asarray(session._depth_ts(), dtype=np.float64)
    pose_ts = np.asarray(session._pose_ts(), dtype=np.float64)
    imu_ts = np.asarray(session._imu_ts(), dtype=np.float64)

    out = {}
    out["rgb_vs_depth"] = _nearest_dt(rgb_ts, depth_ts)
    out["rgb_vs_pose"] = _nearest_dt(rgb_ts, pose_ts)
    out["rgb_vs_imu"] = _nearest_dt(rgb_ts, imu_ts)
    return out


def _nearest_dt(a: np.ndarray, b: np.ndarray) -> dict | None:
    if a.size == 0 or b.size == 0:
        return None
    b_sorted = np.sort(b)
    idx = np.searchsorted(b_sorted, a)
    idx_clipped = np.clip(idx, 1, len(b_sorted) - 1)
    left = b_sorted[idx_clipped - 1]
    right = b_sorted[np.minimum(idx_clipped, len(b_sorted) - 1)]
    dt_left = np.abs(a - left)
    dt_right = np.abs(right - a)
    dts = np.minimum(dt_left, dt_right)
    return {
        "median_ms": float(np.median(dts) * 1000),
        "p95_ms": float(np.percentile(dts, 95) * 1000),
        "max_ms": float(dts.max() * 1000),
        "within_50ms_pct": _pct(int((dts <= 0.05).sum()), len(dts)),
        "within_100ms_pct": _pct(int((dts <= 0.10).sum()), len(dts)),
        "dts_ms": dts * 1000,  # for the histogram
    }


# Hand-pose annotations

def compute_hands(session) -> dict | None:
    hand_buffer: dict[int, list] = getattr(session, "hand_poses", None) or {}
    if not hand_buffer:
        return None

    n_rgb = session.num_rgb_frames
    rgb_intr = session.rgb_intrinsics
    rgb_w = rgb_intr.width if rgb_intr else None
    rgb_h = rgb_intr.height if rgb_intr else None

    counts_per_frame = np.zeros(n_rgb, dtype=np.int8)
    left_valid = np.zeros(n_rgb, dtype=bool)
    right_valid = np.zeros(n_rgb, dtype=bool)
    left_conf: list[float] = []
    right_conf: list[float] = []
    left_depths: list[float] = []
    right_depths: list[float] = []
    left_kpts_3d: list[np.ndarray] = []
    right_kpts_3d: list[np.ndarray] = []
    has_3d_flag = False
    has_mano_count = 0
    in_frame_count = 0
    in_frame_total = 0
    backend = ""
    palm_widths: list[float] = []
    grip_closures: list[float] = []

    for fi, hands in hand_buffer.items():
        if fi < 0 or fi >= n_rgb or not hands:
            continue
        counts_per_frame[fi] = min(len(hands), 100)
        for h in hands:
            be = getattr(h, "_backend", None)
            if be and not backend:
                backend = be
            kps = h.all_keypoints
            arr_xyz = np.array([[k.x, k.y, k.z] for k in kps], dtype=np.float32)
            is_3d = bool(np.any(arr_xyz[:, 2] != 0.0))
            if is_3d:
                has_3d_flag = True
            side = "left" if h.hand_side == "left" else "right"
            if side == "left":
                left_valid[fi] = True
                left_conf.append(float(h.confidence))
                if is_3d:
                    left_depths.append(float(arr_xyz[0, 2]))
                    left_kpts_3d.append(arr_xyz)
            else:
                right_valid[fi] = True
                right_conf.append(float(h.confidence))
                if is_3d:
                    right_depths.append(float(arr_xyz[0, 2]))
                    right_kpts_3d.append(arr_xyz)
            if getattr(h, "_mano_vertices", None) is not None:
                has_mano_count += 1
            # 2D keypoint in-frame check.
            kp_2d = getattr(h, "_kpts_2d_rgb", None)
            if kp_2d is not None and rgb_w and rgb_h:
                kp_2d = np.asarray(kp_2d)
                in_frame_count += int(np.sum(
                    (kp_2d[:, 0] >= 0) & (kp_2d[:, 0] < rgb_w) &
                    (kp_2d[:, 1] >= 0) & (kp_2d[:, 1] < rgb_h)
                ))
                in_frame_total += kp_2d.shape[0]
            # Palm width: 5-MCP (idx 5) -> 17-MCP (idx 17) distance.
            if len(arr_xyz) == 21 and is_3d:
                palm_widths.append(float(
                    np.linalg.norm(arr_xyz[5] - arr_xyz[17])
                ))
                # Grip closure: mean tip-to-wrist distance.
                tips = arr_xyz[[4, 8, 12, 16, 20]]
                grip_closures.append(float(
                    np.linalg.norm(tips - arr_xyz[0], axis=1).mean()
                ))

    frames_total = n_rgb
    frames_any = int((counts_per_frame >= 1).sum())
    frames_1plus = frames_any  # ≥1 hand == any-hand
    frames_2_exact = int((counts_per_frame == 2).sum())
    frames_more = int((counts_per_frame > 2).sum())
    both_hand_frames = int((left_valid & right_valid).sum())

    # Wrist trajectory length / speed in camera frame.
    def _wrist_track(kpts: list[np.ndarray]) -> dict:
        if len(kpts) < 2:
            return {"length_m": None, "speed_mean_mps": None, "speed_max_mps": None}
        wrists = np.stack([k[0] for k in kpts])
        seg = np.linalg.norm(np.diff(wrists, axis=0), axis=1)
        # Rough timestamp: assume frame-rate even spacing (we don't have ts).
        fps = max(_safe_div(session.num_rgb_frames, session.duration) or 30.0, 1.0)
        return {
            "length_m": float(seg.sum()),
            "speed_mean_mps": float(seg.mean() * fps),
            "speed_max_mps": float(seg.max() * fps),
        }

    return {
        "backend": backend or "unknown",
        "frames_total": int(frames_total),
        "frames_with_any_hand": frames_any,
        "frames_with_any_hand_pct": _pct(frames_any, frames_total),
        # ≥1 hand (same value as any_hand; kept for naming symmetry)
        "frames_with_1plus_hand": frames_1plus,
        "frames_with_1plus_hand_pct": _pct(frames_1plus, frames_total),
        # exactly 2 hands
        "frames_with_2_hands": frames_2_exact,
        "frames_with_2_hands_pct": _pct(frames_2_exact, frames_total),
        # >2 hands (often a detection error)
        "frames_with_more_hands": frames_more,
        "frames_with_more_hands_pct": _pct(frames_more, frames_total),
        "left_detection_pct": _pct(int(left_valid.sum()), frames_total),
        "right_detection_pct": _pct(int(right_valid.sum()), frames_total),
        "both_hands_pct": _pct(both_hand_frames, frames_total),
        "left_conf_mean": float(np.mean(left_conf)) if left_conf else None,
        "left_conf_p10": float(np.percentile(left_conf, 10)) if left_conf else None,
        "left_conf_p50": float(np.percentile(left_conf, 50)) if left_conf else None,
        "left_conf_p90": float(np.percentile(left_conf, 90)) if left_conf else None,
        "right_conf_mean": float(np.mean(right_conf)) if right_conf else None,
        "right_conf_p10": float(np.percentile(right_conf, 10)) if right_conf else None,
        "right_conf_p50": float(np.percentile(right_conf, 50)) if right_conf else None,
        "right_conf_p90": float(np.percentile(right_conf, 90)) if right_conf else None,
        "has_3d": has_3d_flag,
        "mano_frames_pct": _pct(has_mano_count, frames_any) if frames_any else None,
        "kpts_in_frame_pct": _pct(in_frame_count, in_frame_total) if in_frame_total else None,
        "left_wrist_depth_mean_m": float(np.mean(left_depths)) if left_depths else None,
        "right_wrist_depth_mean_m": float(np.mean(right_depths)) if right_depths else None,
        "palm_width_mean_m": float(np.mean(palm_widths)) if palm_widths else None,
        "grip_closure_mean_m": float(np.mean(grip_closures)) if grip_closures else None,
        "left_wrist_track": _wrist_track(left_kpts_3d),
        "right_wrist_track": _wrist_track(right_kpts_3d),
        # Series for plots
        "left_conf_series": np.asarray(left_conf, dtype=np.float32),
        "right_conf_series": np.asarray(right_conf, dtype=np.float32),
        "left_valid_series": left_valid,
        "right_valid_series": right_valid,
        "counts_per_frame": counts_per_frame,
    }


# Skeleton (optional)

def compute_skeleton(skeleton_frames) -> dict | None:
    if not skeleton_frames:
        return None
    n = len(skeleton_frames)
    visible_total = np.zeros(10, dtype=np.int64)
    elbow_l: list[float] = []
    elbow_r: list[float] = []
    reach_l: list[float] = []
    reach_r: list[float] = []
    head_heights: list[float] = []
    for f in skeleton_frames:
        visible_total += f.visible.astype(np.int64)
        if f.visible[0]:
            head_heights.append(float(f.joints[0, 1]))
        # Elbow angles: vector(shoulder->elbow) vs vector(elbow->wrist).
        for side, sh, el, wr in (
            ("L", 3, 4, 5), ("R", 6, 7, 8),
        ):
            if f.visible[sh] and f.visible[el] and f.visible[wr]:
                v1 = f.joints[sh] - f.joints[el]
                v2 = f.joints[wr] - f.joints[el]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    ang = math.degrees(
                        math.acos(max(min(float(np.dot(v1, v2) / (n1 * n2)), 1.0), -1.0))
                    )
                    (elbow_l if side == "L" else elbow_r).append(ang)
                    (reach_l if side == "L" else reach_r).append(
                        float(np.linalg.norm(f.joints[sh] - f.joints[wr]))
                    )
    joint_names = [
        "head", "neck", "spine",
        "l_shoulder", "l_elbow", "l_wrist",
        "r_shoulder", "r_elbow", "r_wrist", "mount_cam",
    ]
    return {
        "frame_count": int(n),
        "detection_pct": 100.0,
        "joint_visibility_pct": {
            name: _pct(int(visible_total[i]), n)
            for i, name in enumerate(joint_names)
        },
        "elbow_left_mean_deg": float(np.mean(elbow_l)) if elbow_l else None,
        "elbow_right_mean_deg": float(np.mean(elbow_r)) if elbow_r else None,
        "reach_left_mean_m": float(np.mean(reach_l)) if reach_l else None,
        "reach_right_mean_m": float(np.mean(reach_r)) if reach_r else None,
        "head_height_mean_m": float(np.mean(head_heights)) if head_heights else None,
    }


# Cross-stream health

def compute_health(rec, rgb, depth, traj, imu, sync, hands, config) -> dict:
    """0-100 health roll-up, fully configurable via ``EvaluateConfig``.

    Starts at 100 and subtracts penalties for: RGB frame gaps, depth valid %
    below target, missing streams (depth/pose/IMU), sync offsets, and hand
    presence (any / 1-hand / 2-hand) below targets. Each check has its own
    target and weight; set weight to 0 to disable a check.
    """
    score = 100.0
    notes: list[str] = []

    def deduct(target: float, actual: float, weight: float) -> float:
        if weight <= 0 or actual is None:
            return 0.0
        return max(0.0, target - actual) * weight

    # RGB frame gaps
    if rgb and rgb.get("gap_count", 0) > 0 and config.rgb_gap_max_penalty > 0:
        pen = min(config.rgb_gap_max_penalty, float(rgb["gap_count"]))
        score -= pen
        notes.append(f"{rgb['gap_count']} RGB frame gaps")

    # Depth
    if depth is None:
        if config.depth_required and config.depth_missing_penalty > 0:
            score -= config.depth_missing_penalty
            notes.append("No depth stream")
    elif depth.get("valid_pct_mean") is not None:
        v = depth["valid_pct_mean"]
        pen = deduct(config.depth_valid_target, v, config.depth_valid_weight)
        if pen > 0:
            score -= pen
            notes.append(f"Depth valid {v:.0f}%")

    # Pose
    if traj is None and config.pose_missing_penalty > 0:
        score -= config.pose_missing_penalty
        notes.append("No camera pose")

    # IMU
    if imu is None and config.imu_required and config.imu_missing_penalty > 0:
        score -= config.imu_missing_penalty
        notes.append("No IMU")

    # Sync
    if sync:
        for key, label in (
            ("rgb_vs_depth", "RGB↔Depth"),
            ("rgb_vs_pose", "RGB↔Pose"),
        ):
            s = sync.get(key)
            if s is None:
                continue
            pen = deduct(
                config.sync_target, s["within_50ms_pct"], config.sync_weight,
            )
            if pen > 0:
                score -= pen
                notes.append(
                    f"{label} sync {s['within_50ms_pct']:.0f}% within 50ms"
                )

    # Hands
    if hands is None:
        if config.hand_missing_penalty > 0:
            score -= config.hand_missing_penalty
            notes.append("No hand-pose data")
    else:
        for actual_key, target, weight, note_label in (
            ("frames_with_any_hand_pct", config.hand_any_target,
             config.hand_any_weight, "any-hand"),
            ("frames_with_1plus_hand_pct", config.hand_1plus_target,
             config.hand_1plus_weight, "≥1-hand"),
            ("frames_with_2_hands_pct", config.hand_2_target,
             config.hand_2_weight, "exactly-2-hand"),
        ):
            actual = hands.get(actual_key)
            pen = deduct(target, actual, weight)
            if pen > 0:
                score -= pen
                notes.append(f"{note_label} {actual:.0f}% (target {target:.0f}%)")

    return {
        "score": max(0.0, min(100.0, score)),
        "notes": notes,
    }


# Streamed rgb.mp4 from add_rgb_frame

def compute_streamed_rgb(session) -> dict | None:
    writer = getattr(session, "_rgb_writer", None)
    tmp = getattr(session, "_rgb_writer_tmp_path", None)
    mid = getattr(session, "_rgb_mid_frame", None)
    if writer is None and tmp is None and mid is None:
        return None
    intr = session.rgb_intrinsics
    return {
        "active": writer is not None,
        "tmp_path": str(tmp) if tmp else None,
        "tmp_size_bytes": tmp.stat().st_size if tmp and Path(tmp).exists() else 0,
        "width": intr.width if intr else None,
        "height": intr.height if intr else None,
        "fps": _safe_div(session.num_rgb_frames, session.duration),
        "has_thumbnail": mid is not None,
    }


# Top-level entry point

def compute_all(session, skeleton=None, config=None) -> dict:
    """Compute every metric block. Sections with no data return None."""
    from stera.eval.config import EvaluateConfig
    if config is None:
        config = EvaluateConfig()

    rec = compute_recording(session)
    rgb = compute_rgb(session)
    depth = compute_depth(session)
    traj = compute_trajectory(session)
    imu = compute_imu(session)
    tracking = compute_tracking_state(session)
    tf = compute_tf(session)
    traj_topic = compute_trajectory_topic(session)
    mesh = compute_mesh(session)
    pc = compute_point_cloud(session)
    sync = compute_sync(session, rgb)
    hands = compute_hands(session)
    skel = compute_skeleton(skeleton) if skeleton else None
    streamed = compute_streamed_rgb(session)
    health = compute_health(rec, rgb, depth, traj, imu, sync, hands, config)
    thumbnail = getattr(session, "_rgb_mid_frame", None)

    return {
        "recording": rec,
        "rgb": rgb,
        "depth": depth,
        "trajectory": traj,
        "imu": imu,
        "tracking_state": tracking,
        "tf": tf,
        "trajectory_topic": traj_topic,
        "mesh": mesh,
        "point_cloud": pc,
        "sync": sync,
        "hands": hands,
        "skeleton": skel,
        "streamed_rgb": streamed,
        "health": health,
        "thumbnail": thumbnail,
        "config": config,
    }
