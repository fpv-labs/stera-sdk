"""MCAPReader: high-level structured access to an FPV recording."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from stera.core.types import Pose6D
from stera.data.mcap._reader import MCAPRawReader, TopicConfig
from stera.data.mcap._sync import TimestampIndex
from stera.data.mcap import _decoders as dec

logger = logging.getLogger(__name__)


def _voxel_downsample(
    xyz: np.ndarray, rgb: np.ndarray, voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-grid downsample: keep one point (average) per voxel."""
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    # Pack 3 ints into a single key via structured array
    dt = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    packed = np.empty(len(keys), dtype=dt)
    packed["x"] = keys[:, 0]
    packed["y"] = keys[:, 1]
    packed["z"] = keys[:, 2]
    _, inv, counts = np.unique(packed, return_inverse=True, return_counts=True)
    n_voxels = len(counts)
    sum_xyz = np.zeros((n_voxels, 3), dtype=np.float64)
    sum_rgb = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(sum_xyz, inv, xyz)
    np.add.at(sum_rgb, inv, rgb.astype(np.float64))
    avg_xyz = (sum_xyz / counts[:, None]).astype(np.float32)
    avg_rgb = np.clip(sum_rgb / counts[:, None], 0, 255).astype(np.uint8)
    return avg_xyz, avg_rgb


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    width: int
    height: int
    K: np.ndarray  # 3x3
    D: np.ndarray  # distortion coefficients
    distortion_model: str = "plumb_bob"


@dataclass
class SyncedFrame:
    """A single synchronized frame across multiple streams.

    Intrinsics (``depth_K`` / ``rgb_K``) are populated by the session on yield
    so downstream callers (e.g. HandTracker) don't have to re-plumb them.
    """

    index: int
    timestamp: float
    rgb: np.ndarray  # HxWx3 uint8
    depth: Optional[np.ndarray] = None  # HxW uint16 mm
    camera_pose: Optional[Pose6D] = None
    imu: Optional[dict] = None
    depth_K: Optional[np.ndarray] = None  # 3x3, depth-camera intrinsics
    rgb_K: Optional[np.ndarray] = None    # 3x3, RGB-camera intrinsics


_SENTINEL = object()


class MCAPReader:
    """Structured access to a recorded FPV session from an MCAP file.

    Usage::

        from stera.data import MCAPReader

        session = MCAPReader("recording.mcap")
        print(session.duration, session.num_rgb_frames)

        # Iterate synchronized frames
        for frame in session.frames():
            print(frame.index, frame.rgb.shape, frame.depth.shape)

        # Access intrinsics
        K = session.rgb_intrinsics.K

        # Get all camera poses
        poses = session.all_camera_poses()
    """

    #: Reference topic fingerprint. Only checked when ``check_format=True``
    #: is passed at construction; by default any MCAP is accepted, and missing
    #: topics simply become empty iterators downstream.
    REFERENCE_TOPICS: tuple[str, ...] = (
        "/camera/camera_info",
        "/camera/depth",
        "/camera/pose",
        "/camera/rgb/compressed",
        "/camera/tracking_state",
        "/device/imu",
        "/map/mesh",
        "/map/mesh_cloud",
        "/map/point_cloud",
        "/tf",
        "/trajectory",
    )

    def __init__(
        self,
        path: str | Path,
        topics: TopicConfig | None = None,
        check_format: bool = False,
    ):
        """Open an MCAP recording.

        Parameters
        ----------
        path : Path to the .mcap file.
        topics : Optional ``TopicConfig`` override for topic names.
        check_format : When True, verify the MCAP matches the reference topic
            fingerprint (``REFERENCE_TOPICS``) and reject otherwise. Default
            False: any MCAP is accepted, and missing topics simply become
            empty iterators downstream.
        """
        self._reader = MCAPRawReader(path, topics)
        self._topics = topics or TopicConfig()

        if check_format:
            self._check_format()

        summ = self._reader.summary()
        logger.info(
            "Opened MCAP %s duration=%.1fs rgb=%d depth=%d",
            self._reader.path.name,
            summ.get("duration_sec", 0.0),
            summ.get("topics", {}).get(self._topics.rgb, 0),
            summ.get("topics", {}).get(self._topics.depth, 0),
        )

        # Lazy caches
        self._summary: dict | None = None
        self._rgb_intrinsics: CameraIntrinsics | object = _SENTINEL
        self._depth_intrinsics: CameraIntrinsics | object = _SENTINEL
        self._rgb_timestamps: list[float] | None = None
        self._depth_timestamps: list[float] | None = None
        self._pose_timestamps: list[float] | None = None
        self._imu_timestamps: list[float] | None = None
        self._cached_poses: list[tuple[float, Pose6D]] | None = None
        self._cached_imu: list[tuple[float, dict]] | None = None
        self._tf_cache: list[tuple[float, str, str, Pose6D]] | None = None
        self._optical_to_link: np.ndarray | object = _SENTINEL

        # Annotation buffers (populated by callers during iteration)
        self._hand_poses: dict[int, list] = {}

        # Optional caller-streamed rgb.mp4 (e.g. blurred frames). Lazily
        # created on the first ``add_rgb_frame`` call; finalized by ``export``.
        self._rgb_writer = None
        self._rgb_writer_tmp_path: Path | None = None
        # Mid-frame snapshot captured by ``add_rgb_frame`` and used as the
        # episode thumbnail at export time (so a post-processed thumbnail
        # falls out automatically from the same loop).
        self._rgb_mid_frame: np.ndarray | None = None

    def _check_format(self) -> None:
        """Verify the MCAP matches the reference topic fingerprint."""
        summ = self._reader.summary()
        counts = summ.get("topics", {})
        missing = [t for t in self.REFERENCE_TOPICS if counts.get(t, 0) == 0]
        if missing:
            present = sorted(k for k, v in counts.items() if v > 0)
            raise ValueError(
                f"MCAP at {self._reader.path} does not match the reference "
                f"format: missing/empty topics: {missing}.\n"
                f"  Topics present: {present}\n"
                f"  Pass check_format=False (the default) to skip this check."
            )

    @property
    def path(self) -> Path:
        return self._reader.path

    @property
    def R_optical_to_link(self) -> np.ndarray:
        """Rotation matrix from camera_optical_frame to camera_link, read from TF."""
        if self._optical_to_link is _SENTINEL:
            from stera.core.transforms import R_OPTICAL_TO_LINK
            tfs = self.tf_transforms()
            for _ts, parent, child, pose in tfs:
                if "link" in parent and "optical" in child:
                    # TF gives link -> optical, we want optical -> link = inverse
                    self._optical_to_link = pose.rotation.T
                    break
            if self._optical_to_link is _SENTINEL:
                self._optical_to_link = R_OPTICAL_TO_LINK
        return self._optical_to_link

    def _get_summary(self) -> dict:
        if self._summary is None:
            self._summary = self._reader.summary()
        return self._summary

    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return self._get_summary()["duration_sec"]

    @property
    def num_rgb_frames(self) -> int:
        """Number of RGB frames in the recording."""
        return self._get_summary()["topics"].get(self._topics.rgb, 0)

    @property
    def num_depth_frames(self) -> int:
        return self._get_summary()["topics"].get(self._topics.depth, 0)

    @property
    def rgb_intrinsics(self) -> CameraIntrinsics | None:
        """RGB camera intrinsics (read once, cached)."""
        if self._rgb_intrinsics is _SENTINEL:
            msg = self._reader.read_first(self._topics.rgb_camera_info)
            if msg is not None:
                info = dec.decode_camera_info(msg)
                self._rgb_intrinsics = CameraIntrinsics(
                    width=info["width"], height=info["height"],
                    K=info["K"], D=info["D"],
                    distortion_model=info["distortion_model"],
                )
            else:
                self._rgb_intrinsics = None
        return self._rgb_intrinsics

    @property
    def depth_intrinsics(self) -> CameraIntrinsics | None:
        """Depth camera intrinsics (read once, cached).

        Falls back to RGB intrinsics scaled to depth resolution if no
        dedicated depth camera info topic exists.
        """
        if self._depth_intrinsics is _SENTINEL:
            msg = self._reader.read_first(self._topics.depth_camera_info)
            if msg is not None:
                info = dec.decode_camera_info(msg)
                self._depth_intrinsics = CameraIntrinsics(
                    width=info["width"], height=info["height"],
                    K=info["K"], D=info["D"],
                    distortion_model=info["distortion_model"],
                )
            else:
                # Fallback: scale RGB intrinsics to depth image resolution
                rgb_intr = self.rgb_intrinsics
                if rgb_intr is not None:
                    # Read first depth frame to get its resolution
                    for _ts, depth_img in self.depth_frames():
                        dh, dw = depth_img.shape[:2]
                        sx = dw / rgb_intr.width
                        sy = dh / rgb_intr.height
                        K_scaled = rgb_intr.K.copy()
                        K_scaled[0, :] *= sx  # fx, cx
                        K_scaled[1, :] *= sy  # fy, cy
                        self._depth_intrinsics = CameraIntrinsics(
                            width=dw, height=dh,
                            K=K_scaled, D=rgb_intr.D,
                            distortion_model=rgb_intr.distortion_model,
                        )
                        break
                    else:
                        self._depth_intrinsics = rgb_intr
                else:
                    self._depth_intrinsics = None
        return self._depth_intrinsics

    # Iterators

    def rgb_frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (timestamp, RGB image) for each frame."""
        for _ts, msg in self._reader.iter_topic(self._topics.rgb):
            yield dec.decode_compressed_image(msg)

    def depth_frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (timestamp, depth image uint16 mm) for each frame."""
        for _ts, msg in self._reader.iter_topic(self._topics.depth):
            yield dec.decode_depth_image(msg)

    def camera_poses(self) -> Iterator[tuple[float, Pose6D]]:
        """Yield (timestamp, Pose6D) for each camera pose."""
        for _ts, msg in self._reader.iter_topic(self._topics.camera_pose):
            yield dec.decode_pose_stamped(msg)

    def imu_samples(self) -> Iterator[tuple[float, dict]]:
        """Yield (timestamp, imu_dict) for each IMU sample."""
        for _ts, msg in self._reader.iter_topic(self._topics.imu):
            yield dec.decode_imu(msg)

    def tracking_states(self) -> Iterator[tuple[float, dict]]:
        """Yield (timestamp, tracking_state_dict) for each tracking state."""
        for _ts, msg in self._reader.iter_topic(self._topics.tracking_state):
            yield dec.decode_tracking_state(msg)

    # Bulk accessors (cached)

    def all_camera_poses(self) -> list[tuple[float, Pose6D]]:
        """All camera poses as (timestamp, Pose6D) list."""
        if self._cached_poses is None:
            self._cached_poses = list(self.camera_poses())
        return self._cached_poses

    def all_imu_samples(self) -> list[tuple[float, dict]]:
        """All IMU samples as (timestamp, dict) list."""
        if self._cached_imu is None:
            self._cached_imu = list(self.imu_samples())
        return self._cached_imu

    def tf_transforms(self) -> list[tuple[float, str, str, Pose6D]]:
        """All TF transforms as (timestamp, parent, child, Pose6D) list."""
        if self._tf_cache is None:
            self._tf_cache = []
            for _ts, msg in self._reader.iter_topic(self._topics.tf):
                self._tf_cache.extend(dec.decode_tf_message(msg))
        return self._tf_cache

    def trajectory(self) -> list[tuple[float, Pose6D]]:
        """Trajectory poses from /trajectory topic."""
        msg = self._reader.read_first(self._topics.trajectory)
        if msg is None:
            return []
        return dec.decode_path(msg)

    def point_cloud(
        self, source: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Map point cloud (Nx3 xyz, Nx3 rgb or None).

        Parameters
        ----------
        source
            Which topic to read from:
            ``"auto"`` (default) tries ``/map/mesh_cloud`` first, falls back
            to ``/map/point_cloud``. ``"mesh_cloud"`` reads only
            ``/map/mesh_cloud``. ``"point_cloud"`` reads only
            ``/map/point_cloud``. Returns empty arrays when the requested
            topic is missing.
        """
        if source == "auto":
            msg = self._reader.read_last(self._topics.mesh_cloud)
            if msg is None:
                msg = self._reader.read_first(self._topics.point_cloud)
        elif source == "mesh_cloud":
            msg = self._reader.read_last(self._topics.mesh_cloud)
        elif source == "point_cloud":
            msg = self._reader.read_first(self._topics.point_cloud)
        else:
            raise ValueError(
                f"point_cloud(source={source!r}): expected one of "
                "'auto', 'mesh_cloud', 'point_cloud'."
            )
        if msg is None:
            return np.empty((0, 3), dtype=np.float32), None
        return dec.decode_point_cloud2(msg)

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Triangle mesh from /map/mesh topic.

        Returns (vertices Nx3, triangle_indices Mx3, vertex_colors Nx3 or None),
        or None if no mesh topic exists.
        """
        msg = self._reader.read_first(self._topics.mesh)
        if msg is None:
            return None
        return dec.decode_mesh_marker(msg)

    def dense_point_cloud(
        self,
        every_n: int = 10,
        max_pts_per_frame: int = 5_000,
        cam_exclude_radius: float = 1.0,
        voxel_size: float = 0.02,
        min_depth: float = 0.3,
        max_depth: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a dense coloured point cloud from depth frames + camera poses.

        Parameters
        ----------
        every_n : Use every Nth frame (1 = all frames).
        max_pts_per_frame : Max points to sample per depth frame.
        cam_exclude_radius : Drop points within this distance (m) of the camera.
        voxel_size : Voxel grid size for downsampling (metres). 0 = no downsampling.
        min_depth : Minimum depth in metres.
        max_depth : Maximum depth in metres.

        Returns
        -------
        (N, 3) float32 xyz in world frame, (N, 3) uint8 RGB colours.
        """
        from tqdm import tqdm
        from stera.core.transforms import optical_to_world

        depth_intr = self.depth_intrinsics
        if depth_intr is None:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        K = depth_intr.K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        all_xyz = []
        all_rgb = []

        for frame in tqdm(self.frames(), total=self.num_rgb_frames, desc="Building map", unit="frame"):
            if frame.index % every_n != 0:
                continue
            if frame.depth is None or frame.camera_pose is None:
                continue

            R_w = frame.camera_pose.rotation
            t_w = frame.camera_pose.translation
            depth = frame.depth
            rgb = frame.rgb

            dh, dw = depth.shape[:2]
            v_grid, u_grid = np.mgrid[0:dh, 0:dw]
            z = depth.astype(np.float32) / 1000.0
            valid = (z > min_depth) & (z < max_depth)

            x = (u_grid - cx) * z / fx
            y = (v_grid - cy) * z / fy
            pts_cam = np.stack([x[valid], y[valid], z[valid]], axis=1)

            if len(pts_cam) == 0:
                continue

            # Subsample per frame
            if len(pts_cam) > max_pts_per_frame:
                idx = np.random.default_rng(frame.index).choice(
                    len(pts_cam), max_pts_per_frame, replace=False,
                )
                pts_cam = pts_cam[idx]
                u_valid = u_grid[valid][idx]
                v_valid = v_grid[valid][idx]
            else:
                u_valid = u_grid[valid]
                v_valid = v_grid[valid]

            # Camera-optical -> world
            pts_world = optical_to_world(pts_cam, R_w, t_w, self.R_optical_to_link)

            # Exclude points near camera
            if cam_exclude_radius > 0:
                dists = np.linalg.norm(pts_world - t_w, axis=1)
                keep = dists > cam_exclude_radius
                pts_world = pts_world[keep]
                u_valid = u_valid[keep]
                v_valid = v_valid[keep]

            if len(pts_world) == 0:
                continue

            # Sample colours from RGB
            rh, rw = rgb.shape[:2]
            u_rgb = np.clip((u_valid * rw / dw).astype(int), 0, rw - 1)
            v_rgb = np.clip((v_valid * rh / dh).astype(int), 0, rh - 1)
            colors = rgb[v_rgb, u_rgb]

            all_xyz.append(pts_world)
            all_rgb.append(colors)

        if not all_xyz:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        xyz = np.concatenate(all_xyz).astype(np.float32)
        rgb_arr = np.concatenate(all_rgb).astype(np.uint8)

        # Voxel downsample
        if voxel_size > 0 and len(xyz) > 0:
            xyz, rgb_arr = _voxel_downsample(xyz, rgb_arr, voxel_size)

        return xyz, rgb_arr

    def color_mesh(
        self,
        vertices: np.ndarray,
        every_n: int = 10,
    ) -> np.ndarray:
        """Color mesh vertices by projecting them into RGB camera frames.

        Parameters
        ----------
        vertices : (N, 3) float32 world-frame mesh vertices.
        every_n : Use every Nth frame for color sampling.

        Returns
        -------
        (N, 3) uint8 per-vertex RGB colors.
        """
        from tqdm import tqdm

        rgb_intr = self.rgb_intrinsics
        if rgb_intr is None:
            return np.full((len(vertices), 3), 128, dtype=np.uint8)

        K = rgb_intr.K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R_l2o = self.R_optical_to_link.T

        n_verts = len(vertices)
        color_sum = np.zeros((n_verts, 3), dtype=np.float64)
        color_count = np.zeros(n_verts, dtype=np.int32)

        for frame in tqdm(self.frames(), total=self.num_rgb_frames,
                          desc="Coloring mesh", unit="frame"):
            if frame.index % every_n != 0:
                continue
            if frame.camera_pose is None:
                continue

            R_w = frame.camera_pose.rotation
            t_w = frame.camera_pose.translation
            rgb = frame.rgb
            rh, rw = rgb.shape[:2]

            # World -> optical
            pts_optical = (R_l2o @ (R_w.T @ (vertices - t_w).T)).T
            z = pts_optical[:, 2]

            # Project + bounds check
            u = pts_optical[:, 0] * fx / z + cx
            v = pts_optical[:, 1] * fy / z + cy
            visible = (z > 0.1) & (u >= 0) & (u < rw) & (v >= 0) & (v < rh)

            if not np.any(visible):
                continue

            ui = u[visible].astype(int)
            vi = v[visible].astype(int)
            color_sum[visible] += rgb[vi, ui].astype(np.float64)
            color_count[visible] += 1

        colored = color_count > 0
        colors = np.full((n_verts, 3), 128, dtype=np.uint8)
        colors[colored] = (color_sum[colored] / color_count[colored, None]).clip(0, 255).astype(np.uint8)
        return colors

    # Timestamp indices

    def _rgb_ts(self) -> list[float]:
        if self._rgb_timestamps is None:
            self._rgb_timestamps = []
            for _ts, msg in self._reader.iter_topic(self._topics.rgb):
                self._rgb_timestamps.append(dec._stamp_to_sec(msg.header.stamp))
        return self._rgb_timestamps

    def _depth_ts(self) -> list[float]:
        if self._depth_timestamps is None:
            self._depth_timestamps = []
            for _ts, msg in self._reader.iter_topic(self._topics.depth):
                self._depth_timestamps.append(dec._stamp_to_sec(msg.header.stamp))
        return self._depth_timestamps

    def _pose_ts(self) -> list[float]:
        if self._pose_timestamps is None:
            poses = self.all_camera_poses()
            self._pose_timestamps = [ts for ts, _ in poses]
        return self._pose_timestamps

    def _imu_ts(self) -> list[float]:
        if self._imu_timestamps is None:
            samples = self.all_imu_samples()
            self._imu_timestamps = [ts for ts, _ in samples]
        return self._imu_timestamps

    # Annotation buffers

    def add_hand_pose(self, frame_index: int, hands) -> None:
        """Attach per-frame hand detections to this session.

        Call once per frame during your detection loop; ``session.export()``
        will write the accumulated hands into ``annotation.hdf5:/hand-pose``.
        Overwrites any previously-stored hands for the same frame_index.
        """
        self._hand_poses[int(frame_index)] = list(hands) if hands else []

    @property
    def hand_poses(self) -> dict[int, list]:
        """Hand detections accumulated via ``add_hand_pose``, keyed by frame index."""
        return self._hand_poses

    def add_rgb_frame(self, frame_index: int, rgb) -> None:
        """Stream a (typically post-processed) RGB frame to the output rgb.mp4.

        Use this when your loop modifies frames before logging (e.g. running
        face-blur on each frame). The session lazily opens an internal H.264
        writer on the first call and finalizes it during ``session.export()``.

        Frames must be added in iteration order (the writer is sequential).
        """
        if self._rgb_writer is None:
            rgb_i = self.rgb_intrinsics
            if rgb_i is None:
                logger.warning(
                    "add_rgb_frame: rgb_intrinsics unavailable; frame dropped"
                )
                return
            import shutil
            import tempfile
            from stera.data.export import _RGBMP4Writer

            if shutil.which("ffmpeg") is None:
                logger.warning(
                    "add_rgb_frame: ffmpeg not on PATH; frame dropped"
                )
                return

            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            self._rgb_writer_tmp_path = Path(tmp.name)
            fps = self.num_rgb_frames / max(self.duration, 1e-3)
            self._rgb_writer = _RGBMP4Writer(
                self._rgb_writer_tmp_path,
                rgb_i.width, rgb_i.height, fps,
            )
        self._rgb_writer.write(rgb)

        # Capture the mid-frame snapshot for the episode thumbnail.
        if (
            self._rgb_mid_frame is None
            and int(frame_index) == self.num_rgb_frames // 2
        ):
            self._rgb_mid_frame = rgb.copy()

    # Episode export

    def export(self, out_dir, visualizer=None, **kwargs):
        """Write this session to the standard episode layout.

        Parameters
        ----------
        out_dir : final path folder (created if missing).
        visualizer : optional ``RerunVisualizer`` already populated with frames;
              its backing .rrd is promoted to ``<out_dir>/visualization.rrd``.
        **kwargs : forwarded to ``write_episode`` (``skip_rgb_mp4``,
              ``skip_thumbnail``, ``thumbnail_rgb``).

        If frames were streamed via ``add_rgb_frame`` during the loop, the
        session-managed writer is finalized and its mp4 is moved into
        ``<out_dir>/rgb.mp4`` automatically (no need to pass
        ``skip_rgb_mp4``).

        Returns a ``{"saved": [...], "skipped": [...]}`` manifest and prints it.
        """
        from stera.data.export import write_episode
        return write_episode(self, out_dir, visualizer=visualizer, **kwargs)

    # Synchronized frames

    def frames(self, max_depth_dt: float = 0.1, max_pose_dt: float = 0.1) -> Iterator[SyncedFrame]:
        """Iterate synchronized frames (RGB + nearest depth + pose + IMU).

        Parameters
        ----------
        max_depth_dt : Max time offset for depth matching (seconds).
        max_pose_dt : Max time offset for pose matching (seconds).
        """
        d_intr = self.depth_intrinsics or self.rgb_intrinsics
        r_intr = self.rgb_intrinsics
        depth_K = d_intr.K if d_intr is not None else None
        rgb_K = r_intr.K if r_intr is not None else None

        # Pre-load lightweight data
        depth_ts_list = self._depth_ts()
        pose_data = self.all_camera_poses()
        imu_data = self.all_imu_samples()

        depth_idx = TimestampIndex(depth_ts_list) if depth_ts_list else None
        pose_idx = TimestampIndex([ts for ts, _ in pose_data]) if pose_data else None
        imu_idx = TimestampIndex([ts for ts, _ in imu_data]) if imu_data else None

        # Pre-load all depth frames
        depth_frames = {}
        if depth_idx is not None:
            for i, (ts, depth) in enumerate(self.depth_frames()):
                depth_frames[i] = depth

        # Stream RGB and synchronize
        for frame_i, (rgb_ts, rgb) in enumerate(self.rgb_frames()):
            depth = None
            if depth_idx is not None:
                di = depth_idx.nearest_within(rgb_ts, max_depth_dt)
                if di is not None:
                    depth = depth_frames.get(di)

            pose = None
            if pose_idx is not None:
                pi = pose_idx.nearest_within(rgb_ts, max_pose_dt)
                if pi is not None:
                    pose = pose_data[pi][1]

            imu = None
            if imu_idx is not None:
                ii = imu_idx.nearest_within(rgb_ts, 0.05)
                if ii is not None:
                    imu = imu_data[ii][1]

            yield SyncedFrame(
                index=frame_i,
                timestamp=rgb_ts,
                rgb=rgb,
                depth=depth,
                camera_pose=pose,
                imu=imu,
                depth_K=depth_K,
                rgb_K=rgb_K,
            )
