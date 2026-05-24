"""HaMeR hand tracker: drop-in alternative to WiLoR.

Uses HaMeR's native "as-is" pipeline:

  1. detectron2 (Cascade Mask R-CNN ViTDet-H, or RegNet-Y) -> human bboxes.
  2. ViTPose+-Huge -> wholebody keypoints; last 21 = right hand, last 42-21 = left.
  3. Build hand bboxes from confident hand keypoints.
  4. HaMeR -> MANO regression on each hand bbox.

Returns the same ``HandPose`` list as ``WiLoRHandTracker``. Per-frame depth
grounding uses the same ``_anchor_hand`` trick as WiLoR.

Point ``HaMeRConfig.hamer_dir`` at a HaMeR clone with ``_DATA/`` extracted.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np

from stera.core.types import Keypoint
from stera.annotations.hands import HandPose, FINGER_NAMES, FINGER_JOINTS
from stera.models.hamer.config import HaMeRConfig

logger = logging.getLogger(__name__)


# MANO joint indices per finger (same convention as WiLoR).
ANCHOR_JOINTS = [0, 9, 5, 13, 17, 1]


class HaMeRHandTracker:
    """HaMeR-based hand tracker.

    Usage::

        from stera.models.hamer import HaMeRHandTracker, HaMeRConfig

        tracker = HaMeRHandTracker(HaMeRConfig(
            hamer_dir="/path/to/hamer",
        ))
        hands = tracker.detect_hands(rgb)
        hands = tracker.detect_hands(rgb, depth=depth, intrinsics=K)
    """

    def __init__(self, config: Optional[HaMeRConfig] = None):
        self.config = config or HaMeRConfig()
        self._model = None
        self._model_cfg = None
        self._body_detector = None      # detectron2 DefaultPredictor_Lazy
        self._pose_model = None         # ViTPose wrapper
        self._loaded = False
        self._torch = None

    def load(self) -> None:
        if self._loaded:
            return

        cfg = self.config
        if not cfg.hamer_dir:
            raise RuntimeError(
                "HaMeRConfig.hamer_dir not set. Point it at a clone of "
                "https://github.com/geopavlakos/hamer."
            )
        if not os.path.isdir(cfg.hamer_dir):
            raise FileNotFoundError(f"HaMeRConfig.hamer_dir not found: {cfg.hamer_dir}")

        ckpt = cfg.checkpoint or os.path.join(
            cfg.hamer_dir, "_DATA", "hamer_ckpts", "checkpoints", "hamer.ckpt",
        )
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"HaMeR checkpoint not found: {ckpt}\n"
                "  Extract hamer_demo_data.tar.gz inside hamer_dir or set "
                "HaMeRConfig.checkpoint explicitly."
            )

        if cfg.hamer_dir not in sys.path:
            sys.path.insert(0, cfg.hamer_dir)

        import torch
        self._torch = torch

        # HaMeR's load_hamer reads model_config.yaml with relative MANO
        # paths resolved via CACHE_DIR_HAMER. ViTPose's config also uses
        # relative paths (from its own dir). chdir into hamer_dir while
        # loading so all of these resolve, then restore.
        import hamer.configs as _hamer_cfg  # type: ignore[import]
        _hamer_cfg.CACHE_DIR_HAMER = os.path.join(cfg.hamer_dir, "_DATA")

        old_cwd = os.getcwd()
        os.chdir(cfg.hamer_dir)
        try:
            from hamer.models import load_hamer  # type: ignore[import]

            logger.info("Loading HaMeR from %s", ckpt)
            self._model, self._model_cfg = load_hamer(ckpt)
            self._model = self._model.cuda().eval()

            # Body detector via detectron2 (HaMeR's native pipeline).
            from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy  # type: ignore[import]
            if cfg.body_detector == "vitdet":
                from detectron2.config import LazyConfig
                from pathlib import Path
                import hamer  # type: ignore[import]
                d2_cfg_path = (
                    Path(hamer.__file__).parent / "configs"
                    / "cascade_mask_rcnn_vitdet_h_75ep.py"
                )
                d2_cfg = LazyConfig.load(str(d2_cfg_path))
                d2_cfg.train.init_checkpoint = (
                    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                    "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
                )
                for i in range(3):
                    d2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = (
                        cfg.body_detector_score_thresh
                    )
                logger.info("Loading detectron2 ViTDet-H body detector "
                            "(downloads ~3 GB on first call)")
                self._body_detector = DefaultPredictor_Lazy(d2_cfg)
            elif cfg.body_detector == "regnety":
                from detectron2 import model_zoo
                d2_cfg = model_zoo.get_config(
                    "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
                    trained=True,
                )
                d2_cfg.model.roi_heads.box_predictor.test_score_thresh = (
                    cfg.body_detector_score_thresh
                )
                d2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
                logger.info("Loading detectron2 RegNet-Y body detector")
                self._body_detector = DefaultPredictor_Lazy(d2_cfg)
            else:
                raise ValueError(
                    f"body_detector={cfg.body_detector!r} unknown; "
                    "expected 'vitdet' or 'regnety'."
                )

            # ViTPose for wholebody keypoints (last 42 = both hands).
            # vitpose_model.py lives at hamer_dir root, not inside hamer/.
            import importlib
            spec = importlib.util.spec_from_file_location(
                "vitpose_model_for_hamer",
                os.path.join(cfg.hamer_dir, "vitpose_model.py"),
            )
            vp_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vp_mod)
            logger.info("Loading ViTPose+-Huge wholebody pose model")
            # The wholebody checkpoint is multi-task (per-task expert heads).
            # mmpose dumps thousands of "unexpected key in source state_dict:
            # backbone.blocks.N.mlp.experts.M.weight, …" warnings into the
            # console when only the wholebody head is used. Silence stdout/
            # stderr + mmcv/mmpose loggers around the load to keep the log
            # readable. The shared backbone weights still load correctly.
            with _silence_state_dict_dump():
                self._pose_model = vp_mod.ViTPoseModel("cuda")
        finally:
            os.chdir(old_cwd)

        self._loaded = True
        logger.info("HaMeR pipeline ready (device=cuda)")

    def detect_hands(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        image_is_rotated: Optional[bool] = None,
    ) -> list[HandPose]:
        """Detect hands in a single RGB frame.

        Mirrors ``WiLoRHandTracker.detect_hands``: same input shapes,
        same ``HandPose`` outputs, same depth-anchored joint positions.
        """
        self.load()

        torch = self._torch
        cfg = self.config

        if image_is_rotated is None:
            image_is_rotated = _infer_image_is_rotated(rgb, depth)

        rot_h, rot_w = rgb.shape[:2]

        # 1. Body detection (detectron2)
        # detectron2 takes BGR; ours is RGB.
        bgr = rgb[:, :, ::-1]
        det_out = self._body_detector(bgr)
        det_inst = det_out["instances"]
        valid = (det_inst.pred_classes == 0) & (
            det_inst.scores > cfg.body_detector_score_thresh
        )
        body_bboxes = det_inst.pred_boxes.tensor[valid].cpu().numpy()
        body_scores = det_inst.scores[valid].cpu().numpy()
        if len(body_bboxes) == 0:
            return []

        # 2. ViTPose wholebody -> hand keypoints -> hand bboxes
        vitposes_out = self._pose_model.predict_pose(
            rgb,   # ViTPose path expects RGB
            [np.concatenate([body_bboxes, body_scores[:, None]], axis=1)],
        )
        bboxes_list: list[list[float]] = []
        is_right_list: list[int] = []
        confs_list: list[float] = []
        for vitposes in vitposes_out:
            kps = vitposes["keypoints"]
            left_kp = kps[-42:-21]
            right_kp = kps[-21:]
            for keyp, side in ((left_kp, 0), (right_kp, 1)):
                valid_kp = keyp[:, 2] > cfg.hand_keypoint_score_thresh
                if int(valid_kp.sum()) < cfg.min_hand_keypoints:
                    continue
                bbox = [
                    float(keyp[valid_kp, 0].min()),
                    float(keyp[valid_kp, 1].min()),
                    float(keyp[valid_kp, 0].max()),
                    float(keyp[valid_kp, 1].max()),
                ]
                bboxes_list.append(bbox)
                is_right_list.append(side)
                # Use mean keypoint score as a proxy hand confidence.
                confs_list.append(float(keyp[valid_kp, 2].mean()))
        if not bboxes_list:
            return []

        bboxes = np.array(bboxes_list, dtype=np.float32)
        classes = np.array(is_right_list, dtype=np.float32)
        confs = np.array(confs_list, dtype=np.float32)

        # HaMeR's ViTDet-style dataset (handles cropping + normalization).
        from hamer.datasets.vitdet_dataset import ViTDetDataset  # type: ignore[import]
        from hamer.utils import recursive_to  # type: ignore[import]
        from hamer.utils.renderer import cam_crop_to_full  # type: ignore[import]

        right_arr = np.array([1 if c == 1 else 0 for c in classes], dtype=np.float32)
        dataset = ViTDetDataset(
            self._model_cfg, rgb, bboxes, right_arr,
            rescale_factor=cfg.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )

        FOCAL = self._model_cfg.EXTRA.FOCAL_LENGTH
        IMG_SIZE = self._model_cfg.MODEL.IMAGE_SIZE

        raw_results: list[dict] = []
        for batch in dataloader:
            batch = recursive_to(batch, "cuda")
            with torch.no_grad():
                out = self._model(batch)

            pred_joints = out["pred_keypoints_3d"].cpu().numpy()
            pred_kpts2d = out["pred_keypoints_2d"].cpu().numpy()
            pred_verts = out["pred_vertices"].cpu().numpy()
            pred_cam = out["pred_cam"]
            # Full MANO regression output (rotation matrices + shape).
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
            box_centers = batch["box_center"].float()
            box_sizes = batch["box_size"].float()
            img_size = batch["img_size"].float()
            is_right_b = batch["right"].cpu().numpy()

            # HaMeR's cam_crop_to_full computes the full-image cam_t in one call.
            scaled_focal_length = FOCAL / IMG_SIZE * float(img_size.max().item())
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_centers, box_sizes, img_size, scaled_focal_length,
            ).detach().cpu().numpy()

            box_centers_np = box_centers.cpu().numpy()
            box_sizes_np = box_sizes.cpu().numpy()

            for hi in range(pred_joints.shape[0]):
                hand_type = "right" if is_right_b[hi] > 0.5 else "left"
                is_left = is_right_b[hi] <= 0.5

                cam_t = pred_cam_t_full[hi]

                # 2D keypoints in image space. HaMeR's pred_kpts2d are
                # normalised to [-0.5, 0.5] in the cropped patch; convert
                # back to full-image pixel coords.
                kpts_2d = np.zeros((21, 2), dtype=np.float32)
                bc = box_centers_np[hi]
                bs = box_sizes_np[hi]
                if is_left:
                    kpts_2d[:, 0] = (-pred_kpts2d[hi, :, 0] + 0.5) * bs + (bc[0] - bs / 2)
                else:
                    kpts_2d[:, 0] = (pred_kpts2d[hi, :, 0] + 0.5) * bs + (bc[0] - bs / 2)
                kpts_2d[:, 1] = (pred_kpts2d[hi, :, 1] + 0.5) * bs + (bc[1] - bs / 2)

                # Convert to unrotated frame for depth anchoring (same as
                # WiLoR's path when the camera image is rotated).
                if image_is_rotated:
                    kpts_2d_unrot = np.zeros((21, 2), dtype=np.float32)
                    kpts_2d_unrot[:, 0] = rot_h - 1 - kpts_2d[:, 1]
                    kpts_2d_unrot[:, 1] = kpts_2d[:, 0]
                else:
                    kpts_2d_unrot = kpts_2d.copy()

                # Depth anchoring → joints in camera-optical frame, metric.
                joints_cam_real = None
                if depth is not None and intrinsics is not None:
                    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                    cx_k, cy_k = intrinsics[0, 2], intrinsics[1, 2]
                    dh, dw = depth.shape[:2]
                    rgb_w, rgb_h = (rot_w, rot_h) if not image_is_rotated else (rot_h, rot_w)
                    joints_cam_real = _anchor_hand(
                        pred_joints[hi], cam_t, kpts_2d_unrot, depth,
                        fx, fy, cx_k, cy_k, dw, dh, rgb_w, rgb_h,
                        is_left=is_left,
                        sample_radius=cfg.depth_sample_radius,
                        image_is_rotated=image_is_rotated,
                    )

                raw_results.append({
                    "hand_type": hand_type,
                    "joints_cam_real": joints_cam_real,
                    "kpts_2d_rgb": kpts_2d,            # in original (rotated) RGB frame
                    "confidence": float(confs[hi]),
                    "mano_vertices": pred_verts[hi] if cfg.save_mano_vertices else None,
                    "mano_global_orient": mano_global[hi] if mano_global is not None else None,
                    "mano_hand_pose":     mano_pose[hi]   if mano_pose   is not None else None,
                    "mano_betas":         mano_betas[hi] if mano_betas is not None else None,
                    "pred_cam":           pred_cam[hi].detach().cpu().numpy(),
                    "pred_cam_t":         cam_t,                            # full-image translation
                    "focal_length":       float(scaled_focal_length),
                })

        # Dedup: keep the highest-spread detection per hand side.
        if len(raw_results) > 1:
            best: dict[str, tuple] = {}
            for det in raw_results:
                ht = det["hand_type"]
                spread = float(np.ptp(det["kpts_2d_rgb"], axis=0).sum())
                if ht not in best or spread > best[ht][1]:
                    best[ht] = (det, spread)
            raw_results = [v[0] for v in best.values()]

        # Convert to HandPose objects.
        hands: list[HandPose] = []
        for det in raw_results:
            joints_3d = det["joints_cam_real"]
            if joints_3d is not None and not _sanity_check(joints_3d, cfg):
                joints_3d = None
            hp = _to_hand_pose(
                det["hand_type"], joints_3d, det["kpts_2d_rgb"], det["confidence"],
            )
            # Stash the full regression output on the HandPose so export
            # can write everything to annotation.hdf5:/hand-pose.
            hp._mano_vertices = det.get("mano_vertices")
            hp._mano_global_orient = det.get("mano_global_orient")
            hp._mano_hand_pose = det.get("mano_hand_pose")
            hp._mano_betas = det.get("mano_betas")
            hp._pred_cam = det.get("pred_cam")
            hp._pred_cam_t = det.get("pred_cam_t")
            hp._cam_t = det.get("pred_cam_t")  # alias
            hp._focal_length = det.get("focal_length")
            hp._backend = "hamer"
            hands.append(hp)

        return hands


# Helpers: depth anchoring + sanity check + HandPose builder.
# Lifted from WiLoR's tracker so HaMeR produces identical-shaped output.


def _silence_state_dict_dump():
    """Context manager that swallows the noisy multi-page state_dict
    mismatch dump mmpose/mmcv prints when loading the wholebody ViTPose+
    checkpoint (multi-task expert heads → "unexpected key" warnings).

    Redirects stdout + stderr to /dev/null and bumps the relevant loggers
    to ERROR for the duration. Restores everything on exit, including in
    the face of exceptions.
    """
    import contextlib
    import logging as _logging

    class _Silencer:
        def __enter__(self_):
            self_._stdout_redirect = contextlib.redirect_stdout(open(os.devnull, "w"))
            self_._stderr_redirect = contextlib.redirect_stderr(open(os.devnull, "w"))
            self_._stdout_redirect.__enter__()
            self_._stderr_redirect.__enter__()
            self_._levels: dict[str, int] = {}
            for name in ("mmcv", "mmpose", "root"):
                lg = _logging.getLogger(name) if name != "root" else _logging.getLogger()
                self_._levels[name] = lg.level
                lg.setLevel(_logging.ERROR)
            return self_

        def __exit__(self_, *exc):
            for name, lvl in self_._levels.items():
                lg = _logging.getLogger(name) if name != "root" else _logging.getLogger()
                lg.setLevel(lvl)
            self_._stderr_redirect.__exit__(*exc)
            self_._stdout_redirect.__exit__(*exc)
            return False  # don't swallow exceptions

    return _Silencer()


def _infer_image_is_rotated(rgb: np.ndarray, depth: Optional[np.ndarray]) -> bool:
    if depth is None:
        return False
    rh, rw = rgb.shape[:2]
    dh, dw = depth.shape[:2]
    return (rw > rh) != (dw > dh)


def _sample_depth(depth_img, u, v, w, h, radius=7):
    u, v = int(round(u)), int(round(v))
    if u < 0 or u >= w or v < 0 or v >= h:
        return None
    r = radius
    patch = depth_img[max(0, v - r):min(h, v + r + 1),
                      max(0, u - r):min(w, u + r + 1)].astype(np.float32)
    valid = patch[(patch > 50) & (patch < 2000)]
    if len(valid) < max(3, 0.15 * patch.size):
        return None
    return float(np.median(valid)) / 1000.0


def _anchor_hand(
    joints_cam, cam_t, kpts_2d_rgb, depth_img,
    fx, fy, cx, cy, dw, dh, rgb_w, rgb_h,
    is_left=False, sample_radius=7, image_is_rotated=True,
):
    sx, sy = dw / rgb_w, dh / rgb_h
    z_real = None

    # ANCHOR_JOINTS is only a priority list for finding a *valid depth value*
    # (the wrist is often occluded or sits on a depth discontinuity). The
    # anchor *pixel* is always the wrist, because joints_rel below is
    # wrist-relative; anchoring on a different joint shifts the whole hand.
    for ai in ANCHOR_JOINTS:
        u_d, v_d = kpts_2d_rgb[ai, 0] * sx, kpts_2d_rgb[ai, 1] * sy
        z = _sample_depth(depth_img, u_d, v_d, dw, dh, sample_radius)
        if z is not None:
            z_real = z
            break

    if z_real is None:
        zs = [
            _sample_depth(depth_img, kpts_2d_rgb[j, 0] * sx, kpts_2d_rgb[j, 1] * sy, dw, dh, 5)
            for j in range(21)
        ]
        zs = [z for z in zs if z is not None]
        if len(zs) < 3:
            return None
        z_real = float(np.median(zs))

    # Back-project the wrist pixel at the resolved depth (see note above).
    anchor_u, anchor_v = kpts_2d_rgb[0, 0] * sx, kpts_2d_rgb[0, 1] * sy
    anchor_real = np.array([
        (anchor_u - cx) * z_real / fx,
        (anchor_v - cy) * z_real / fy,
        z_real,
    ])
    joints_rel = joints_cam - joints_cam[0:1]
    if is_left:
        joints_rel[:, 0] *= -1
    if image_is_rotated:
        # Rotate 90° CCW around Z to bring HaMeR's output into the unrotated frame.
        joints_rel = np.column_stack([-joints_rel[:, 1], joints_rel[:, 0], joints_rel[:, 2]])
    return joints_rel + anchor_real


def _sanity_check(joints_3d: np.ndarray, cfg: HaMeRConfig) -> bool:
    if np.any(np.abs(joints_3d) > cfg.max_joint_abs):
        return False
    if joints_3d[0, 2] < cfg.min_wrist_depth:
        return False
    palm_span = float(np.max(np.linalg.norm(joints_3d - joints_3d[0:1], axis=1)))
    if palm_span < cfg.min_palm_span or palm_span > cfg.max_palm_span:
        return False
    return True


def _to_hand_pose(
    hand_type: str,
    joints_3d: Optional[np.ndarray],
    kpts_2d: np.ndarray,
    confidence: float,
) -> HandPose:
    """Build a HandPose mirroring WiLoRHandTracker._to_hand_pose."""

    def _kp(idx: int, name: str) -> Keypoint:
        if joints_3d is not None:
            return Keypoint(
                x=float(joints_3d[idx, 0]),
                y=float(joints_3d[idx, 1]),
                z=float(joints_3d[idx, 2]),
                confidence=confidence, name=name,
            )
        return Keypoint(
            x=float(kpts_2d[idx, 0]),
            y=float(kpts_2d[idx, 1]),
            z=0.0, confidence=confidence, name=name,
        )

    fingers = {
        # MANO finger order (1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky)
        # matches the slice in WiLoR's mapping.
        name: [_kp(start + j, FINGER_JOINTS[j]) for j in range(4)]
        for name, (start, _end) in {
            "thumb":  (1, 5),
            "index":  (5, 9),
            "middle": (9, 13),
            "ring":   (13, 17),
            "pinky":  (17, 21),
        }.items()
    }

    hp = HandPose(
        wrist=_kp(0, "wrist"),
        fingers=fingers,
        hand_side=hand_type,
        confidence=confidence,
    )
    # Stash 2D pixel coords for the rerun viz overlay.
    hp._kpts_2d_rgb = np.asarray(kpts_2d, dtype=np.float32)
    return hp
