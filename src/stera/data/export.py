"""Export an MCAP session to the standard episode layout.

Top-level entry point: ``session.export(out_dir, viz=None)``. Writes
whichever of the following are available and reports saved / skipped:

    episode/
    ├── rgb.mp4
    ├── mesh.ply
    ├── thumbnail.jpg
    ├── visualization.rrd           (only if ``viz`` is given)
    ├── annotation.hdf5
    │   ├── /depth                  (frames/timestamps/valid, gzip level 4)
    │   ├── /cam-pose               (timestamps/translations/rotations)
    │   ├── /imu                    (ts / accel / gyro / orientation_xyzw)
    │   ├── /metadata               (counts / times)
    │   └── /hand-pose              (only when add_hand_pose was called)
    └── calibrations/               (rgb_K/D, depth_K/D, R_optical_to_link, meta.json)
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


_HAND_EXTRA_FIELDS = {
    # field name on HandPose →  (per-frame array shape excluding F)
    "_kpts_2d_rgb":         ("kpts_2d_rgb",         (21, 2),  np.float32),
    "_mano_vertices":       ("mano_vertices",       (778, 3), np.float32),
    "_mano_global_orient":  ("mano_global_orient",  (1, 3, 3), np.float32),
    "_mano_hand_pose":      ("mano_hand_pose",      (15, 3, 3), np.float32),
    "_mano_betas":          ("mano_betas",          (10,),    np.float32),
    "_pred_cam":            ("pred_cam",            (3,),     np.float32),
    "_pred_cam_t":          ("pred_cam_t",          (3,),     np.float32),
    "_cam_t":               ("cam_t",               (3,),     np.float32),
    "_focal_length":        ("focal_length",        (2,),     np.float32),
}


def _hands_to_arrays(hands_by_idx: dict[int, list], n_rgb: int) -> dict:
    """Collapse per-frame hand lists into (F, …) arrays for left/right.

    Always includes the 21-joint keypoints and per-frame validity/confidence.
    Additionally, every private extra a tracker may attach to a HandPose
    (MANO params, vertices, weak-perspective camera, focal length, …) is
    written when present on at least one frame. NaN-filled where missing.
    """
    out: dict = {"has_3d": False, "backend": ""}
    for side in ("left", "right"):
        out[f"{side}_joints"] = np.full((n_rgb, 21, 3), np.nan, dtype=np.float32)
        out[f"{side}_valid"] = np.zeros(n_rgb, dtype=bool)
        out[f"{side}_confidence"] = np.zeros(n_rgb, dtype=np.float32)

    # Lazy buffers for optional extras: only allocate when a tracker
    # actually attached the field (skips the 778x3 vertex grid for
    # backends that don't produce MANO output).
    extra_buffers: dict[str, dict[str, np.ndarray]] = {}

    def _ensure_extra(side: str, attr_name: str, out_name: str, shape, dtype):
        key = f"{side}_{out_name}"
        if key in extra_buffers:
            return extra_buffers[key]
        arr = np.full((n_rgb,) + tuple(shape), np.nan, dtype=dtype)
        extra_buffers[key] = arr
        return arr

    for fi, hands in hands_by_idx.items():
        if fi < 0 or fi >= n_rgb or not hands:
            continue
        for h in hands:
            kps = h.all_keypoints
            arr = np.array([[k.x, k.y, k.z] for k in kps], dtype=np.float32)
            if np.any(arr[:, 2] != 0.0):
                out["has_3d"] = True
            side = "left" if h.hand_side == "left" else "right"
            out[f"{side}_joints"][fi] = arr
            out[f"{side}_valid"][fi] = True
            out[f"{side}_confidence"][fi] = h.confidence
            be = getattr(h, "_backend", None)
            if be and not out["backend"]:
                out["backend"] = be

            for attr, (name, shape, dtype) in _HAND_EXTRA_FIELDS.items():
                v = getattr(h, attr, None)
                if v is None:
                    continue
                buf = _ensure_extra(side, attr, name, shape, dtype)
                vv = np.asarray(v, dtype=dtype)
                if vv.shape == buf[fi].shape:
                    buf[fi] = vv

    out.update(extra_buffers)
    return out


# RGB.mp4 writer

class _RGBMP4Writer:
    def __init__(self, path: Path, width: int, height: int, fps: float,
                 crf: int = 18, preset: str = "medium"):
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", f"{max(fps, 1.0):.6f}",
                "-i", "-",
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, rgb: np.ndarray) -> None:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        self.proc.stdin.write(rgb.tobytes())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=300)


# PLY writer

def _write_ply(path: Path, verts: np.ndarray, faces: np.ndarray,
               colors: Optional[np.ndarray]) -> None:
    verts = verts.astype(np.float32, copy=False)
    faces = faces.astype(np.int32, copy=False)
    has_color = colors is not None
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(verts)}",
        "property float x", "property float y", "property float z",
    ]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if has_color:
            c = colors.astype(np.uint8, copy=False)
            vs = np.empty(len(verts), dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ])
            vs["x"], vs["y"], vs["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
            vs["r"], vs["g"], vs["b"] = c[:, 0], c[:, 1], c[:, 2]
            f.write(vs.tobytes())
        else:
            f.write(verts.tobytes())
        fs = np.empty(len(faces), dtype=[
            ("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4"),
        ])
        fs["n"] = 3
        fs["a"], fs["b"], fs["c"] = faces[:, 0], faces[:, 1], faces[:, 2]
        f.write(fs.tobytes())


# Calibrations

def _write_calibrations(session, cal_dir: Path) -> None:
    cal_dir.mkdir(parents=True, exist_ok=True)
    rgb_i = session.rgb_intrinsics
    d_i = session.depth_intrinsics
    meta: dict = {}
    if rgb_i is not None:
        np.save(cal_dir / "rgb_K.npy", rgb_i.K)
        np.save(cal_dir / "rgb_D.npy", rgb_i.D)
        meta["rgb"] = {
            "width": int(rgb_i.width),
            "height": int(rgb_i.height),
            "distortion_model": rgb_i.distortion_model,
            "files": {"K": "rgb_K.npy", "D": "rgb_D.npy"},
        }
    if d_i is not None:
        np.save(cal_dir / "depth_K.npy", d_i.K)
        np.save(cal_dir / "depth_D.npy", d_i.D)
        meta["depth"] = {
            "width": int(d_i.width),
            "height": int(d_i.height),
            "distortion_model": d_i.distortion_model,
            "files": {"K": "depth_K.npy", "D": "depth_D.npy"},
            "units": "mm (uint16)",
        }
    if session.R_optical_to_link is not None:
        np.save(cal_dir / "R_optical_to_link.npy", session.R_optical_to_link)
        meta["R_optical_to_link"] = "R_optical_to_link.npy"
    with open(cal_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# Main entry point

def write_episode(
    session,
    out_dir,
    visualizer=None,
    skip_rgb_mp4: bool = False,
    skip_thumbnail: bool = False,
    thumbnail_rgb: Optional[np.ndarray] = None,
) -> dict[str, list[str]]:
    """Write a session to the standard episode layout.

    Parameters
    ----------
    session
        An ``MCAPReader`` (or anything with the same interface).
    out_dir
        Target directory; created if missing.
    visualizer
        Optional ``RerunVisualizer`` that has already been used to log frames.
        When given, its backing .rrd is promoted to ``<out_dir>/visualization.rrd``
        via ``visualizer.export``.
    skip_rgb_mp4
        If True, don't write ``rgb.mp4`` here. Use when the caller has
        already streamed its own (e.g. post-processed/blurred) frames to
        rgb.mp4 during the main frame loop using ``RGBMP4Writer``.
    skip_thumbnail
        If True, don't write ``thumbnail.jpg`` at all.
    thumbnail_rgb
        Optional pre-computed thumbnail (RGB ``np.ndarray``). When provided
        and ``skip_thumbnail`` is False, this image is saved as
        ``thumbnail.jpg`` instead of grabbing the mid-frame from the mcap.

    Returns
    -------
    Dict ``{"saved": [...], "skipped": [...]}`` and prints a summary.
    """
    import h5py

    from tqdm import tqdm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    skipped: list[str] = []

    rgb_i = session.rgb_intrinsics
    d_i = session.depth_intrinsics or rgb_i
    n_frames = session.num_rgb_frames
    duration = max(session.duration, 1e-3)
    fps = n_frames / duration

    logger.info(
        "Exporting episode to %s (frames=%d, duration=%.1fs, fps=%.2f)",
        out_dir, n_frames, duration, fps,
    )

    # rgb.mp4 writer
    # If the caller streamed post-processed frames via session.add_rgb_frame,
    # finalize that temp file into the episode dir and skip our own pass.
    session_rgb_writer = getattr(session, "_rgb_writer", None)
    session_rgb_tmp = getattr(session, "_rgb_writer_tmp_path", None)
    if session_rgb_writer is not None and session_rgb_tmp is not None:
        logger.info("Finalizing session-streamed rgb.mp4")
        session_rgb_writer.close()
        shutil.move(str(session_rgb_tmp), str(out_dir / "rgb.mp4"))
        session._rgb_writer = None
        session._rgb_writer_tmp_path = None
        saved.append("rgb.mp4")
        skip_rgb_mp4 = True  # don't double-write below

    rgb_writer: Optional[_RGBMP4Writer] = None
    if skip_rgb_mp4:
        if "rgb.mp4" not in saved:
            skipped.append("rgb.mp4 (skipped: caller wrote it)")
    elif rgb_i is None:
        skipped.append("rgb.mp4 (no RGB intrinsics)")
    elif shutil.which("ffmpeg") is None:
        skipped.append("rgb.mp4 (ffmpeg not on PATH)")
    else:
        logger.info("Opening rgb.mp4 writer (%dx%d @ %.2f fps)",
                    rgb_i.width, rgb_i.height, fps)
        rgb_writer = _RGBMP4Writer(
            out_dir / "rgb.mp4", rgb_i.width, rgb_i.height, fps,
        )

    # annotation.hdf5 + /depth streaming
    logger.info("Opening annotation.hdf5")
    h5f = h5py.File(out_dir / "annotation.hdf5", "w")
    depth_frames_dset = depth_ts_dset = depth_valid_dset = None
    d_h = d_w = 0
    if d_i is not None:
        d_h, d_w = int(d_i.height), int(d_i.width)
        depth_grp = h5f.create_group("depth")
        depth_frames_dset = depth_grp.create_dataset(
            "frames", shape=(n_frames, d_h, d_w), dtype=np.uint16,
            chunks=(1, d_h, d_w), compression="gzip", compression_opts=4,
        )
        depth_ts_dset = depth_grp.create_dataset(
            "timestamps", shape=(n_frames,), dtype=np.float64,
        )
        depth_valid_dset = depth_grp.create_dataset(
            "valid", shape=(n_frames,), dtype=bool,
        )
        depth_grp.attrs["units"] = "mm"
        depth_grp.attrs["height"] = d_h
        depth_grp.attrs["width"] = d_w
    else:
        skipped.append("annotation.hdf5:/depth (no depth intrinsics)")

    # Single pass over frames: rgb.mp4 + /depth + thumbnail
    thumbnail: Optional[np.ndarray] = None
    mid = n_frames // 2
    frame_ts: list[float] = []
    needs_frame_pass = (rgb_writer is not None) or (depth_frames_dset is not None) or not skip_thumbnail
    if needs_frame_pass:
        logger.info(
            "Streaming frames to rgb.mp4 / depth / thumbnail (n=%d)", n_frames,
        )
    for i, frame in enumerate(tqdm(
        session.frames(),
        total=n_frames,
        desc="Export frames",
        unit="fr",
        disable=not needs_frame_pass,
    )):
        if rgb_writer is not None:
            rgb_writer.write(frame.rgb)
        if depth_frames_dset is not None and frame.depth is not None:
            dep = frame.depth
            if dep.shape != (d_h, d_w):
                dep = cv2.resize(dep, (d_w, d_h), interpolation=cv2.INTER_NEAREST)
            if dep.dtype != np.uint16:
                dep = dep.astype(np.uint16)
            depth_frames_dset[i] = dep
            depth_ts_dset[i] = float(frame.timestamp)
            depth_valid_dset[i] = True
        frame_ts.append(float(frame.timestamp))
        if i == mid:
            thumbnail = frame.rgb.copy()

    if rgb_writer is not None:
        logger.info("Finalizing rgb.mp4 (waiting for ffmpeg)")
        rgb_writer.close()
        saved.append("rgb.mp4")
    if depth_frames_dset is not None:
        saved.append("annotation.hdf5:/depth")

    # thumbnail.jpg
    if skip_thumbnail:
        skipped.append("thumbnail.jpg (skipped)")
    else:
        # Priority: explicit kwarg > session-captured mid-frame from
        # add_rgb_frame > export-loop captured thumbnail from raw mcap.
        session_mid = getattr(session, "_rgb_mid_frame", None)
        thumb = (
            thumbnail_rgb
            if thumbnail_rgb is not None
            else (session_mid if session_mid is not None else thumbnail)
        )
        if thumb is not None:
            logger.info("Writing thumbnail.jpg")
            cv2.imwrite(
                str(out_dir / "thumbnail.jpg"),
                cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            saved.append("thumbnail.jpg")
        else:
            skipped.append("thumbnail.jpg (no frames)")

    # mesh.ply
    logger.info("Writing mesh.ply")
    try:
        mesh_data = session.mesh()
    except Exception as e:
        mesh_data = None
        skipped.append(f"mesh.ply ({e!r})")
    if mesh_data is not None:
        _write_ply(out_dir / "mesh.ply", *mesh_data)
        saved.append("mesh.ply")
    elif not any(s.startswith("mesh.ply") for s in skipped):
        skipped.append("mesh.ply (no /map/mesh topic)")

    # calibrations/
    logger.info("Writing calibrations/")
    _write_calibrations(session, out_dir / "calibrations")
    saved.append("calibrations/")

    # /cam-pose
    logger.info("Writing /cam-pose")
    pose_pairs = session.all_camera_poses()
    if pose_pairs:
        pose_ts = np.array([ts for ts, _ in pose_pairs], dtype=np.float64)
        pose_t = np.array([p.translation for _, p in pose_pairs], dtype=np.float32)
        pose_R = np.array([p.rotation for _, p in pose_pairs], dtype=np.float32)
        cp = h5f.create_group("cam-pose")
        cp.create_dataset("timestamps", data=pose_ts)
        cp.create_dataset("translations", data=pose_t)
        cp.create_dataset("rotations", data=pose_R)
        saved.append("annotation.hdf5:/cam-pose")
    else:
        pose_ts = np.zeros(0, dtype=np.float64)
        skipped.append("annotation.hdf5:/cam-pose (no /camera/pose messages)")

    # /imu
    logger.info("Writing /imu")
    imu_pairs = session.all_imu_samples()
    if imu_pairs:
        imu_ts = np.array([ts for ts, _ in imu_pairs], dtype=np.float64)
        imu_la = np.array([d["linear_acceleration"] for _, d in imu_pairs], dtype=np.float32)
        imu_av = np.array([d["angular_velocity"] for _, d in imu_pairs], dtype=np.float32)
        imu_ori = np.array([d["orientation"] for _, d in imu_pairs], dtype=np.float32)
        imu = h5f.create_group("imu")
        imu.create_dataset("timestamps", data=imu_ts)
        imu.create_dataset("linear_acceleration", data=imu_la)
        imu.create_dataset("angular_velocity", data=imu_av)
        imu.create_dataset("orientation_xyzw", data=imu_ori)
        saved.append("annotation.hdf5:/imu")
    else:
        imu_ts = np.zeros(0, dtype=np.float64)
        skipped.append("annotation.hdf5:/imu (no /device/imu messages)")

    # /hand-pose (from session.add_hand_pose accumulated during the loop)
    hand_buffer = getattr(session, "hand_poses", None) or {}
    if hand_buffer:
        logger.info("Writing /hand-pose (frames with hands=%d)", len(hand_buffer))
        arrs = _hands_to_arrays(hand_buffer, n_rgb=n_frames)
        hp = h5f.create_group("hand-pose")
        hp.create_dataset("timestamps", data=np.asarray(frame_ts, dtype=np.float64))
        # Always written: 21-joint keypoints + per-frame validity / confidence.
        hp.create_dataset("left_joints", data=arrs["left_joints"])
        hp.create_dataset("right_joints", data=arrs["right_joints"])
        hp.create_dataset("left_valid", data=arrs["left_valid"])
        hp.create_dataset("right_valid", data=arrs["right_valid"])
        hp.create_dataset("left_confidence", data=arrs["left_confidence"])
        hp.create_dataset("right_confidence", data=arrs["right_confidence"])
        hp.attrs["coord_frame"] = "camera_3d" if arrs["has_3d"] else "image_2d"
        hp.attrs["joint_layout"] = (
            "wrist, 5×[mcp,pip,dip,tip] (thumb,index,middle,ring,pinky)"
        )
        if arrs.get("backend"):
            hp.attrs["backend"] = arrs["backend"]
        # Optional extras: MANO params, vertices, weak-perspective camera,
        # focal length, 2D keypoints. Written when the tracker attached
        # them. Vertices are gzip-compressed (the bulky one).
        compress_keys = {"left_mano_vertices", "right_mano_vertices"}
        for key in (
            "left_kpts_2d_rgb", "right_kpts_2d_rgb",
            "left_mano_vertices", "right_mano_vertices",
            "left_mano_global_orient", "right_mano_global_orient",
            "left_mano_hand_pose", "right_mano_hand_pose",
            "left_mano_betas", "right_mano_betas",
            "left_pred_cam", "right_pred_cam",
            "left_pred_cam_t", "right_pred_cam_t",
            "left_cam_t", "right_cam_t",
            "left_focal_length", "right_focal_length",
        ):
            buf = arrs.get(key)
            if buf is None:
                continue
            kw = (
                {"chunks": (1,) + buf.shape[1:], "compression": "gzip", "compression_opts": 4}
                if key in compress_keys else {}
            )
            hp.create_dataset(key, data=buf, **kw)
        saved.append("annotation.hdf5:/hand-pose")
    else:
        skipped.append("annotation.hdf5:/hand-pose (no session.add_hand_pose calls)")

    # /text-annotations is not written. If you have your own annotation
    # source, append it to annotation.hdf5 after session.export() returns.

    # /metadata
    logger.info("Writing /metadata")
    meta_grp = h5f.create_group("metadata")
    meta_grp.attrs["num_rgb_frames"] = int(n_frames)
    meta_grp.attrs["num_depth_frames"] = int(getattr(session, "num_depth_frames", 0) or 0)
    meta_grp.attrs["num_pose_samples"] = int(len(pose_ts))
    meta_grp.attrs["num_imu_samples"] = int(len(imu_ts))
    meta_grp.attrs["duration_s"] = float(duration)
    try:
        summ = session._reader.summary()
        meta_grp.attrs["start_time"] = float(summ.get("start_time", 0))
        meta_grp.attrs["end_time"] = float(summ.get("end_time", 0))
    except Exception:
        pass
    saved.append("annotation.hdf5:/metadata")

    h5f.close()

    # visualization.rrd (optional)
    if visualizer is not None:
        logger.info("Writing visualization.rrd")
        try:
            visualizer.export(out_dir / "visualization.rrd")
            saved.append("visualization.rrd")
        except Exception as e:
            skipped.append(f"visualization.rrd ({e!r})")

    manifest = {"saved": saved, "skipped": skipped}
    _log_manifest(out_dir, manifest)
    return manifest


def _log_manifest(out_dir: Path, manifest: dict[str, list[str]]) -> None:
    logger.info("Episode written to %s", out_dir)
    logger.info("  saved (%d):", len(manifest["saved"]))
    for s in manifest["saved"]:
        logger.info("    + %s", s)
    if manifest["skipped"]:
        logger.info("  skipped (%d):", len(manifest["skipped"]))
        for s in manifest["skipped"]:
            logger.info("    - %s", s)
