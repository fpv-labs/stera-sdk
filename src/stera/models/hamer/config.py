"""HaMeR hand tracker configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HaMeRConfig:
    """Configuration for the HaMeR hand mesh tracker.

    Parameters
    ----------
    hamer_dir
        Path to a local clone of https://github.com/geopavlakos/hamer
        (the directory containing ``hamer/`` and ``_DATA/``). Prepended to
        ``sys.path`` at load time so ``import hamer`` resolves.
    checkpoint
        Path to ``hamer.ckpt``. If unset, defaults to
        ``<hamer_dir>/_DATA/hamer_ckpts/checkpoints/hamer.ckpt``.
    hand_detector_path
        Path to a YOLO ``detector.pt`` that emits hand bboxes (right=class 1,
        left=class 0). The WiLoR-bundled ``detector.pt`` works as-is. If
        unset, you must provide ``wilor_dir`` (we'll resolve it from there).
    wilor_dir
        Optional fallback for ``hand_detector_path``. If both are unset,
        loading raises.
    yolo_conf
        YOLO detection confidence threshold for hand bboxes.
    rescale_factor
        Bbox padding factor passed to HaMeR's ``ViTDetDataset``.
    batch_size
        Inference batch size when multiple hands are detected.
    depth_buffer_size
        Rolling-median window for the depth-anchored wrist tracker.
    depth_sample_radius
        Pixel patch radius when sampling depth at a 2D joint.
    save_mano_vertices
        Whether to attach ``_mano_vertices`` (778, 3) to each ``HandPose``.
        Default False (keep memory low). Flip on for downstream rendering.
    max_joint_abs, min_wrist_depth, min_palm_span, max_palm_span
        Same sanity-check thresholds as WiLoR.
    """

    hamer_dir: Optional[str] = None
    checkpoint: Optional[str] = None

    # Body detector (HaMeR's "as-is" pipeline). Choices:
    #   "vitdet"   detectron2 cascade_mask_rcnn_vitdet_h. Most accurate,
    #              ~3 GB checkpoint auto-downloaded from FB AI public files.
    #   "regnety"  detectron2 mask_rcnn_regnety_4gf. Faster, smaller.
    body_detector: str = "vitdet"
    body_detector_score_thresh: float = 0.5

    # Minimum number of confident hand keypoints from ViTPose required to
    # accept a hand for HaMeR. Lower = more recall, more false positives.
    min_hand_keypoints: int = 3
    hand_keypoint_score_thresh: float = 0.5

    rescale_factor: float = 2.0
    batch_size: int = 8

    depth_buffer_size: int = 15
    depth_sample_radius: int = 7
    save_mano_vertices: bool = True   # captured per-frame; written by session.export

    # Sanity check thresholds (match WiLoR defaults)
    max_joint_abs: float = 3.0
    min_wrist_depth: float = 0.05
    min_palm_span: float = 0.02
    max_palm_span: float = 0.5
