"""WiLoR hand tracker: wraps the frozen WiLoR installation for inference."""

from __future__ import annotations

import logging
import os
import sys
import queue
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

from stera.core.types import Keypoint
from stera.annotations.hands import HandPose, FINGER_NAMES, FINGER_JOINTS
from stera.models.wilor.config import WiLoRConfig
from stera.models.wilor.constants import FINGER_JOINT_SLICES, ANCHOR_JOINTS

logger = logging.getLogger(__name__)


def _infer_image_is_rotated(rgb: np.ndarray, depth: Optional[np.ndarray]) -> bool:
    """True iff RGB is rotated 90° relative to the depth image orientation.

    The anchoring pipeline requires RGB 2D keypoints to live in the same
    orientation as the depth image so back-projection through the depth
    intrinsics is consistent. When only RGB is available, no rotation is
    applied (2D keypoints are returned in native RGB coords).
    """
    if depth is None:
        return False
    rgb_landscape = rgb.shape[1] > rgb.shape[0]
    depth_landscape = depth.shape[1] > depth.shape[0]
    return rgb_landscape != depth_landscape


class WiLoRHandTracker:
    """Hand tracker using the frozen WiLoR model.

    Usage::

        from stera.models.wilor import WiLoRHandTracker, WiLoRConfig

        tracker = WiLoRHandTracker(WiLoRConfig(wilor_dir="/path/to/WiLoR-fresh"))
        hands = tracker.detect_hands(rgb_frame)
        hands = tracker.detect_hands(rgb_frame, depth_frame, intrinsics_K)
    """

    def __init__(self, config: Optional[WiLoRConfig] = None):
        self.config = config or WiLoRConfig()
        self._model = None
        self._model_cfg = None
        self._detector = None
        self._loaded = False

    def load(self) -> None:
        """Load WiLoR + YOLO models. Called automatically on first inference."""
        if self._loaded:
            return

        wilor_dir = self.config.wilor_dir
        if not wilor_dir:
            raise RuntimeError(
                "WiLoR directory not set. Pass wilor_dir to WiLoRConfig or "
                "model_path to HandTracker, e.g. "
                "HandTracker(model='wilor', model_path='/path/to/WiLoR')."
            )
        if not os.path.isdir(wilor_dir):
            raise FileNotFoundError(
                f"WiLoR directory not found: {wilor_dir}. "
                "Clone https://github.com/rolpotamias/WiLoR and point wilor_dir at it."
            )
        ckpt = os.path.join(wilor_dir, "pretrained_models", "wilor_final.ckpt")
        detector = os.path.join(wilor_dir, "pretrained_models", "detector.pt")
        missing = [p for p in (ckpt, detector) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"WiLoR checkpoints missing: {missing}. "
                "Run WiLoR's download script to populate pretrained_models/."
            )

        logger.info("Loading WiLoR from %s", wilor_dir)

        import torch
        self._torch = torch

        sys.path.insert(0, wilor_dir)
        old_cwd = os.getcwd()
        os.chdir(wilor_dir)

        try:
            from wilor.models import load_wilor
            from ultralytics import YOLO

            self._model, self._model_cfg = load_wilor(
                ckpt,
                os.path.join(wilor_dir, "pretrained_models", "model_config.yaml"),
            )
            self._model = self._model.cuda().eval()
            self._detector = YOLO(detector)
        finally:
            os.chdir(old_cwd)

        self._loaded = True
        logger.info("WiLoR ready (device=cuda)")

    def detect_hands(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        image_is_rotated: Optional[bool] = None,
    ) -> list[HandPose]:
        """Detect hands in a single frame.

        Parameters
        ----------
        rgb : RGB image (H, W, 3). If ``image_is_rotated`` is True, this is the
              rotated image as stored (e.g. 1280x960 landscape from a portrait camera).
        depth : Optional 16-bit depth image in the *unrotated* camera frame (mm).
        intrinsics : Optional 3x3 camera intrinsics matrix K (for the depth image).
        image_is_rotated : If True, 2D keypoints are transformed from the rotated WiLoR
                           frame to the unrotated depth/camera frame. ``None`` (default)
                           auto-detects by comparing RGB and depth orientations.

        Returns
        -------
        List of HandPose objects (0, 1, or 2 hands).
        """
        self.load()

        torch = self._torch
        cfg = self.config

        if image_is_rotated is None:
            image_is_rotated = _infer_image_is_rotated(rgb, depth)

        rot_h, rot_w = rgb.shape[:2]

        # YOLO detection
        detections = self._detector(rgb, conf=cfg.yolo_conf, verbose=False, half=True)[0]
        bboxes = detections.boxes.xyxy.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()

        if len(bboxes) == 0:
            return []

        from wilor.datasets.vitdet_dataset import ViTDetDataset

        WILOR_FOCAL = self._model_cfg.EXTRA.FOCAL_LENGTH
        WILOR_IMG_SIZE = self._model_cfg.MODEL.IMAGE_SIZE

        right_arr = np.array([1 if c == 1 else 0 for c in classes], dtype=np.float32)
        dataset = ViTDetDataset(self._model_cfg, rgb, bboxes, right_arr, rescale_factor=cfg.rescale_factor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=min(len(bboxes), cfg.batch_size), shuffle=False, num_workers=0,
        )

        raw_results = []

        for batch in dataloader:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                out = self._model(batch)

            pred_joints = out["pred_keypoints_3d"].cpu().numpy()
            pred_kpts2d = out["pred_keypoints_2d"].cpu().numpy()
            pred_verts = out["pred_vertices"].cpu().numpy() if cfg.save_mano_vertices else None
            pred_cam = out["pred_cam"].cpu().numpy()
            # Full MANO parameters (rotation matrices + shape coefs).
            # WiLoR returns rotmats; each MANO joint is a 3x3.
            mano_params = out.get("pred_mano_params", {}) or {}
            mano_global = (
                mano_params["global_orient"].cpu().numpy()
                if "global_orient" in mano_params else None
            )
            mano_pose = (
                mano_params["hand_pose"].cpu().numpy()
                if "hand_pose" in mano_params else None
            )
            mano_betas = (
                mano_params["betas"].cpu().numpy()
                if "betas" in mano_params else None
            )
            pred_cam_t = (
                out["pred_cam_t"].cpu().numpy() if "pred_cam_t" in out else None
            )
            focal_length = (
                out["focal_length"].cpu().numpy() if "focal_length" in out else None
            )
            box_centers = batch["box_center"].cpu().numpy()
            box_sizes = batch["box_size"].cpu().numpy()
            is_right = batch["right"].cpu().numpy()

            for hi in range(pred_joints.shape[0]):
                hand_type = "right" if is_right[hi] > 0.5 else "left"
                is_left = is_right[hi] <= 0.5

                # Camera translation from WiLoR's weak-perspective model
                s_cam, tx, ty = pred_cam[hi]
                max_dim = max(rot_w, rot_h)
                scaled_fl = WILOR_FOCAL / WILOR_IMG_SIZE * max_dim
                tz = 2 * scaled_fl / (WILOR_IMG_SIZE * s_cam + 1e-9)
                cam_tx = tx + 2 * (box_centers[hi, 0] - rot_w / 2) / max_dim * tz
                cam_ty = ty + 2 * (box_centers[hi, 1] - rot_h / 2) / max_dim * tz
                cam_t = np.array([cam_tx, cam_ty, tz])

                # 2D keypoints in rotated image space
                kpts_2d_rotated = np.zeros((21, 2))
                if is_left:
                    kpts_2d_rotated[:, 0] = (-pred_kpts2d[hi, :, 0] + 0.5) * box_sizes[hi] + (
                        box_centers[hi, 0] - box_sizes[hi] / 2
                    )
                else:
                    kpts_2d_rotated[:, 0] = (pred_kpts2d[hi, :, 0] + 0.5) * box_sizes[hi] + (
                        box_centers[hi, 0] - box_sizes[hi] / 2
                    )
                kpts_2d_rotated[:, 1] = (pred_kpts2d[hi, :, 1] + 0.5) * box_sizes[hi] + (
                    box_centers[hi, 1] - box_sizes[hi] / 2
                )

                # Convert rotated coords to unrotated for depth anchoring
                if image_is_rotated:
                    kpts_2d_unrot = np.zeros((21, 2))
                    kpts_2d_unrot[:, 0] = rot_h - 1 - kpts_2d_rotated[:, 1]
                    kpts_2d_unrot[:, 1] = kpts_2d_rotated[:, 0]
                else:
                    kpts_2d_unrot = kpts_2d_rotated.copy()

                # Depth anchoring
                joints_cam_real = None
                wrist_depth = None
                if depth is not None and intrinsics is not None:
                    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                    cx_k, cy_k = intrinsics[0, 2], intrinsics[1, 2]
                    dh, dw = depth.shape[:2]
                    if image_is_rotated:
                        rgb_h, rgb_w = rot_w, rot_h  # unrotated dims
                    else:
                        rgb_h, rgb_w = rot_h, rot_w
                    joints_cam_real, wrist_depth = self._anchor_hand(
                        pred_joints[hi], cam_t, kpts_2d_unrot, depth,
                        fx, fy, cx_k, cy_k, dw, dh, rgb_w, rgb_h,
                        is_left=is_left,
                        sample_radius=cfg.depth_sample_radius,
                        image_is_rotated=image_is_rotated,
                    )

                raw_results.append({
                    "hand_type": hand_type,
                    "joints_cam": pred_joints[hi],
                    "cam_t": cam_t,
                    "joints_cam_real": joints_cam_real,
                    "wrist_depth": wrist_depth,
                    "kpts_2d_rgb": kpts_2d_rotated,
                    "kpts_2d_unrot": kpts_2d_unrot,
                    "confidence": float(confs[hi]),
                    "mano_vertices": pred_verts[hi] if pred_verts is not None else None,
                    # Full MANO regression output (used by export).
                    "mano_global_orient": mano_global[hi] if mano_global is not None else None,
                    "mano_hand_pose":     mano_pose[hi]   if mano_pose   is not None else None,
                    "mano_betas":         mano_betas[hi] if mano_betas is not None else None,
                    "pred_cam":           pred_cam[hi],
                    "pred_cam_t":         pred_cam_t[hi] if pred_cam_t is not None else None,
                    "focal_length":       focal_length[hi] if focal_length is not None else None,
                    "backend":            "wilor",
                })

        # Dedup: keep best detection per hand side (by 2D spread)
        if len(raw_results) > 1:
            best = {}
            for det in raw_results:
                ht = det["hand_type"]
                spread = np.ptp(det["kpts_2d_rgb"], axis=0).sum()
                if ht not in best or spread > best[ht][1]:
                    best[ht] = (det, spread)
            raw_results = [v[0] for v in best.values()]

        # Convert to SDK types
        hands = []
        for det in raw_results:
            joints_3d = det["joints_cam_real"] if det["joints_cam_real"] is not None else None
            if joints_3d is not None:
                if not self._sanity_check(joints_3d, cfg):
                    continue
            hand = self._to_hand_pose(
                det["hand_type"],
                joints_3d,
                det["kpts_2d_unrot"],
                det["confidence"],
            )
            # Stash the full regression output on the HandPose so export
            # can write everything to annotation.hdf5:/hand-pose. None
            # values are tolerated downstream.
            hand._mano_vertices = det.get("mano_vertices")
            hand._mano_global_orient = det.get("mano_global_orient")
            hand._mano_hand_pose = det.get("mano_hand_pose")
            hand._mano_betas = det.get("mano_betas")
            hand._pred_cam = det.get("pred_cam")
            hand._pred_cam_t = det.get("pred_cam_t")
            hand._cam_t = det.get("cam_t")
            hand._focal_length = det.get("focal_length")
            hand._backend = "wilor"
            hands.append(hand)
        return hands

    def detect_sequence(
        self,
        rgb_video: str | np.ndarray,
        depth_dir: Optional[str] = None,
        intrinsics: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        image_is_rotated: Optional[bool] = None,
    ) -> list[list[HandPose]]:
        """Detect hands across a video sequence with interpolation.

        Parameters
        ----------
        rgb_video : Path to an RGB video file.
        depth_dir : Optional directory of 16-bit depth PNGs (sorted alphabetically).
        intrinsics : Optional 3x3 camera intrinsics matrix K.
        timestamps : Optional per-frame timestamps array.
        image_is_rotated : Whether the video frames are stored rotated.

        Returns
        -------
        Per-frame list of HandPose lists.
        """
        self.load()

        torch = self._torch
        cfg = self.config

        from wilor.datasets.vitdet_dataset import ViTDetDataset

        WILOR_FOCAL = self._model_cfg.EXTRA.FOCAL_LENGTH
        WILOR_IMG_SIZE = self._model_cfg.MODEL.IMAGE_SIZE

        # Intrinsics
        fx = fy = cx_k = cy_k = dw = dh = None
        has_depth = depth_dir is not None and intrinsics is not None
        if has_depth:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx_k, cy_k = intrinsics[0, 2], intrinsics[1, 2]

        # Frame counts
        cap = cv2.VideoCapture(rgb_video)
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        depth_files = []
        if depth_dir is not None:
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

        if image_is_rotated is None:
            cap_peek = cv2.VideoCapture(rgb_video)
            ok, rgb0 = cap_peek.read()
            cap_peek.release()
            depth0 = None
            if depth_files:
                depth0 = cv2.imread(
                    os.path.join(depth_dir, depth_files[0]), cv2.IMREAD_UNCHANGED,
                )
            image_is_rotated = _infer_image_is_rotated(rgb0, depth0) if ok else False

        n_frames = n_video
        if depth_files:
            n_frames = min(n_frames, len(depth_files))
        if timestamps is not None:
            n_frames = min(n_frames, len(timestamps))

        detect_frames = list(range(0, n_frames, cfg.detect_every_n))
        if detect_frames[-1] != n_frames - 1:
            detect_frames.append(n_frames - 1)

        logger.info(
            "detect_sequence: frames=%d detect_every_n=%d yolo_conf=%.2f",
            n_frames, cfg.detect_every_n, cfg.yolo_conf,
        )

        # Per-frame raw results
        frame_results: list[list[dict]] = [[] for _ in range(n_frames)]

        # Prefetch worker
        prefetch_q: queue.Queue = queue.Queue(maxsize=6)

        def prefetch_worker(indices):
            cap_w = cv2.VideoCapture(rgb_video)
            for fi in indices:
                cap_w.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, rgb = cap_w.read()
                if not ret:
                    break
                rgb_rot = rgb
                if image_is_rotated:
                    rgb_unrot = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
                else:
                    rgb_unrot = rgb
                depth = None
                if depth_files:
                    depth = cv2.imread(
                        os.path.join(depth_dir, depth_files[fi]), cv2.IMREAD_UNCHANGED
                    )
                prefetch_q.put((fi, rgb_rot, rgb_unrot, depth))
            cap_w.release()
            prefetch_q.put(None)

        pool = ThreadPoolExecutor(max_workers=1)
        pool.submit(prefetch_worker, detect_frames)

        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(detect_frames), desc="  Detect", unit="fr")
        except ImportError:
            pbar = None

        total_det = 0
        depth_anchored = 0

        while True:
            item = prefetch_q.get()
            if item is None:
                break
            fi, rgb_rot, rgb_unrot, depth = item
            rot_h, rot_w = rgb_rot.shape[:2]
            if image_is_rotated:
                rgb_h, rgb_w = rot_w, rot_h
            else:
                rgb_h, rgb_w = rot_h, rot_w

            if has_depth and depth is not None:
                dh_cur, dw_cur = depth.shape[:2]
            else:
                dh_cur, dw_cur = None, None

            detections = self._detector(rgb_rot, conf=cfg.yolo_conf, verbose=False, half=True)[0]
            bboxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            confs = detections.boxes.conf.cpu().numpy()

            if len(bboxes) > 0:
                right_arr = np.array([1 if c == 1 else 0 for c in classes], dtype=np.float32)
                dataset = ViTDetDataset(
                    self._model_cfg, rgb_rot, bboxes, right_arr, rescale_factor=cfg.rescale_factor,
                )
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=min(len(bboxes), cfg.batch_size), shuffle=False, num_workers=0,
                )

                for batch in dataloader:
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                        out = self._model(batch)

                    pred_joints = out["pred_keypoints_3d"].cpu().numpy()
                    pred_kpts2d = out["pred_keypoints_2d"].cpu().numpy()
                    pred_verts = out["pred_vertices"].cpu().numpy() if cfg.save_mano_vertices else None
                    pred_cam = out["pred_cam"].cpu().numpy()
                    box_centers = batch["box_center"].cpu().numpy()
                    box_sizes = batch["box_size"].cpu().numpy()
                    is_right_arr = batch["right"].cpu().numpy()

                    for hi in range(pred_joints.shape[0]):
                        hand_type = "right" if is_right_arr[hi] > 0.5 else "left"
                        is_left = is_right_arr[hi] <= 0.5

                        s_cam, tx, ty = pred_cam[hi]
                        max_dim = max(rot_w, rot_h)
                        scaled_fl = WILOR_FOCAL / WILOR_IMG_SIZE * max_dim
                        tz = 2 * scaled_fl / (WILOR_IMG_SIZE * s_cam + 1e-9)
                        cam_tx = tx + 2 * (box_centers[hi, 0] - rot_w / 2) / max_dim * tz
                        cam_ty = ty + 2 * (box_centers[hi, 1] - rot_h / 2) / max_dim * tz
                        cam_t = np.array([cam_tx, cam_ty, tz])

                        kpts_2d_rotated = np.zeros((21, 2))
                        if is_left:
                            kpts_2d_rotated[:, 0] = (-pred_kpts2d[hi, :, 0] + 0.5) * box_sizes[hi] + (
                                box_centers[hi, 0] - box_sizes[hi] / 2
                            )
                        else:
                            kpts_2d_rotated[:, 0] = (pred_kpts2d[hi, :, 0] + 0.5) * box_sizes[hi] + (
                                box_centers[hi, 0] - box_sizes[hi] / 2
                            )
                        kpts_2d_rotated[:, 1] = (pred_kpts2d[hi, :, 1] + 0.5) * box_sizes[hi] + (
                            box_centers[hi, 1] - box_sizes[hi] / 2
                        )

                        if image_is_rotated:
                            kpts_2d_unrot = np.zeros((21, 2))
                            kpts_2d_unrot[:, 0] = rot_h - 1 - kpts_2d_rotated[:, 1]
                            kpts_2d_unrot[:, 1] = kpts_2d_rotated[:, 0]
                        else:
                            kpts_2d_unrot = kpts_2d_rotated.copy()

                        joints_cam_real = None
                        wrist_depth = None
                        if has_depth and depth is not None:
                            joints_cam_real, wrist_depth = self._anchor_hand(
                                pred_joints[hi], cam_t, kpts_2d_unrot, depth,
                                fx, fy, cx_k, cy_k, dw_cur, dh_cur, rgb_w, rgb_h,
                                is_left=is_left,
                                sample_radius=cfg.depth_sample_radius,
                                image_is_rotated=image_is_rotated,
                            )

                        total_det += 1
                        if joints_cam_real is not None:
                            depth_anchored += 1

                        det_result = {
                            "hand_type": hand_type,
                            "joints_cam": pred_joints[hi],
                            "cam_t": cam_t,
                            "joints_cam_real": joints_cam_real,
                            "wrist_depth": wrist_depth,
                            "kpts_2d_rgb": kpts_2d_rotated,
                            "kpts_2d_unrot": kpts_2d_unrot,
                            "confidence": float(confs[hi]),
                        }
                        if pred_verts is not None:
                            det_result["mano_vertices"] = pred_verts[hi]
                        frame_results[fi].append(det_result)

                # Dedup per frame
                if len(frame_results[fi]) > 1:
                    best = {}
                    for det in frame_results[fi]:
                        ht = det["hand_type"]
                        spread = np.ptp(det["kpts_2d_rgb"], axis=0).sum()
                        if ht not in best or spread > best[ht][1]:
                            best[ht] = (det, spread)
                    frame_results[fi] = [v[0] for v in best.values()]

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(det=total_det, depth=depth_anchored)

        if pbar is not None:
            pbar.close()
        pool.shutdown()

        # Interpolate between detection frames
        if cfg.detect_every_n > 1:
            for idx in range(len(detect_frames) - 1):
                fa, fb = detect_frames[idx], detect_frames[idx + 1]
                if fb - fa <= 1:
                    continue
                ra, rb = frame_results[fa], frame_results[fb]
                if not ra and not rb:
                    continue
                for fi in range(fa + 1, fb):
                    alpha = (fi - fa) / (fb - fa)
                    if ra and rb:
                        frame_results[fi] = self._interpolate_hands(ra, rb, alpha)
                    elif ra:
                        frame_results[fi] = [d.copy() for d in ra]
                    else:
                        frame_results[fi] = [d.copy() for d in rb]

        # Convert to SDK types with depth smoothing
        tracks = {"left": _DepthTrack(cfg.depth_buffer_size), "right": _DepthTrack(cfg.depth_buffer_size)}
        all_hands: list[list[HandPose]] = []

        for fi in range(n_frames):
            frame_hands = []
            for det in frame_results[fi]:
                ht = det["hand_type"]
                track = tracks[ht]
                track.update(det["wrist_depth"])

                if det["joints_cam_real"] is not None:
                    joints_3d = det["joints_cam_real"]
                elif has_depth and track.smoothed_depth is not None:
                    # Fallback: use relative joints + smoothed wrist depth
                    joints_rel = det["joints_cam"] - det["joints_cam"][0:1]
                    if ht == "left":
                        joints_rel[:, 0] *= -1
                    if image_is_rotated:
                        # Rotate 90 deg CCW around Z to correct for rotated WiLoR input
                        joints_rel = np.column_stack([-joints_rel[:, 1], joints_rel[:, 0], joints_rel[:, 2]])
                    kpts = det["kpts_2d_unrot"]
                    sx, sy = dw_cur / rgb_w if dw_cur else 1.0, dh_cur / rgb_h if dh_cur else 1.0
                    wu, wv = kpts[0, 0] * sx, kpts[0, 1] * sy
                    z = track.smoothed_depth
                    wrist_real = np.array([(wu - cx_k) * z / fx, (wv - cy_k) * z / fy, z])
                    joints_3d = joints_rel + wrist_real
                else:
                    continue

                if not self._sanity_check(joints_3d, cfg):
                    continue

                hand = self._to_hand_pose(ht, joints_3d, det["kpts_2d_unrot"], det["confidence"], fi)
                frame_hands.append(hand)
            all_hands.append(frame_hands)

        logger.info(
            "detect_sequence: done. %d frames, %d hand detections",
            n_frames, sum(len(h) for h in all_hands),
        )
        return all_hands

    # Private helpers

    @staticmethod
    def _sample_depth(depth_img, u, v, w, h, radius=7):
        u, v = int(round(u)), int(round(v))
        if u < 0 or u >= w or v < 0 or v >= h:
            return None
        r = radius
        patch = depth_img[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)].astype(np.float32)
        valid = patch[(patch > 50) & (patch < 2000)]
        if len(valid) < max(3, 0.15 * patch.size):
            return None
        return float(np.median(valid)) / 1000.0

    @staticmethod
    def _anchor_hand(joints_cam, cam_t, kpts_2d_rgb, depth_img,
                     fx, fy, cx, cy, dw, dh, rgb_w, rgb_h,
                     is_left=False, sample_radius=7, image_is_rotated=True):
        sx, sy = dw / rgb_w, dh / rgb_h
        z_real = None
        anchor_u = anchor_v = None

        for ai in ANCHOR_JOINTS:
            u_d, v_d = kpts_2d_rgb[ai, 0] * sx, kpts_2d_rgb[ai, 1] * sy
            z = WiLoRHandTracker._sample_depth(depth_img, u_d, v_d, dw, dh, sample_radius)
            if z is not None:
                z_real = z
                anchor_u, anchor_v = u_d, v_d
                break

        if z_real is None:
            zs = [
                WiLoRHandTracker._sample_depth(depth_img, kpts_2d_rgb[j, 0] * sx, kpts_2d_rgb[j, 1] * sy, dw, dh, 5)
                for j in range(21)
            ]
            zs = [z for z in zs if z is not None]
            if len(zs) < 3:
                return None, None
            z_real = np.median(zs)
            anchor_u, anchor_v = kpts_2d_rgb[0, 0] * sx, kpts_2d_rgb[0, 1] * sy

        anchor_real = np.array([(anchor_u - cx) * z_real / fx, (anchor_v - cy) * z_real / fy, z_real])
        joints_rel = joints_cam - joints_cam[0:1]
        if is_left:
            joints_rel[:, 0] *= -1
        if image_is_rotated:
            # Rotate 90 deg CCW around Z to correct for rotated WiLoR input
            joints_rel = np.column_stack([-joints_rel[:, 1], joints_rel[:, 0], joints_rel[:, 2]])
        return joints_rel + anchor_real, z_real

    @staticmethod
    def _interpolate_hands(ra, rb, alpha):
        hands_a = {d["hand_type"]: d for d in ra}
        hands_b = {d["hand_type"]: d for d in rb}
        out = []
        for ht in set(list(hands_a.keys()) + list(hands_b.keys())):
            if ht in hands_a and ht in hands_b:
                da, db = hands_a[ht], hands_b[ht]
                interp = {
                    "hand_type": ht,
                    "joints_cam": (1 - alpha) * da["joints_cam"] + alpha * db["joints_cam"],
                    "cam_t": (1 - alpha) * da["cam_t"] + alpha * db["cam_t"],
                    "kpts_2d_rgb": (1 - alpha) * da["kpts_2d_rgb"] + alpha * db["kpts_2d_rgb"],
                    "kpts_2d_unrot": (1 - alpha) * da["kpts_2d_unrot"] + alpha * db["kpts_2d_unrot"],
                    "wrist_depth": None,
                    "joints_cam_real": None,
                    "confidence": min(da.get("confidence", 1), db.get("confidence", 1)),
                }
                if da["joints_cam_real"] is not None and db["joints_cam_real"] is not None:
                    interp["joints_cam_real"] = (1 - alpha) * da["joints_cam_real"] + alpha * db["joints_cam_real"]
                elif da["joints_cam_real"] is not None:
                    interp["joints_cam_real"] = da["joints_cam_real"]
                elif db["joints_cam_real"] is not None:
                    interp["joints_cam_real"] = db["joints_cam_real"]
                out.append(interp)
            elif ht in hands_a:
                out.append(hands_a[ht].copy())
            else:
                out.append(hands_b[ht].copy())
        return out

    @staticmethod
    def _sanity_check(joints_3d: np.ndarray, cfg: WiLoRConfig) -> bool:
        if np.any(np.abs(joints_3d) > cfg.max_joint_abs):
            return False
        if joints_3d[0, 2] < cfg.min_wrist_depth:
            return False
        palm_span = np.max(np.linalg.norm(joints_3d - joints_3d[0:1], axis=1))
        if palm_span < cfg.min_palm_span or palm_span > cfg.max_palm_span:
            return False
        return True

    @staticmethod
    def _to_hand_pose(
        hand_type: str,
        joints_3d: Optional[np.ndarray],
        kpts_2d: np.ndarray,
        confidence: float,
        frame_idx: Optional[int] = None,
    ) -> HandPose:
        """Convert raw numpy arrays to SDK HandPose."""

        def _kp(idx: int, name: str) -> Keypoint:
            if joints_3d is not None:
                return Keypoint(
                    x=float(joints_3d[idx, 0]),
                    y=float(joints_3d[idx, 1]),
                    z=float(joints_3d[idx, 2]),
                    confidence=confidence,
                    name=name,
                )
            return Keypoint(
                x=float(kpts_2d[idx, 0]),
                y=float(kpts_2d[idx, 1]),
                z=0.0,
                confidence=confidence,
                name=name,
            )

        fingers = {}
        for finger_name in FINGER_NAMES:
            sl = FINGER_JOINT_SLICES[finger_name]
            fingers[finger_name] = [
                _kp(sl.start + j, FINGER_JOINTS[j]) for j in range(4)
            ]

        hp = HandPose(
            wrist=_kp(0, "wrist"),
            fingers=fingers,
            hand_side=hand_type,
            frame_idx=frame_idx,
            confidence=confidence,
        )
        # Stash the 2D pixel coords so the rerun viz can draw the hand
        # overlay on /camera/rgb_overlay without re-projecting from 3D.
        # ``kpts_2d`` is in ORIGINAL RGB pixel space (what frame.rgb stores).
        hp._kpts_2d_rgb = np.asarray(kpts_2d, dtype=np.float32)
        return hp


class _DepthTrack:
    """Rolling median buffer for wrist depth smoothing."""

    def __init__(self, buffer_size: int = 15):
        self._buffer: deque = deque(maxlen=buffer_size)
        self.smoothed_depth: Optional[float] = None

    def update(self, depth: Optional[float]) -> None:
        if depth is not None:
            self._buffer.append(depth)
            self.smoothed_depth = float(np.median(list(self._buffer)))
