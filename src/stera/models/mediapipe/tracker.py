"""MediaPipe hand tracker: fast 21-joint hand pose with depth anchoring.

Drop-in alternative to ``WiLoRHandTracker``. Returns the same ``HandPose``
type, so downstream code (viz, skeleton estimator, episode export) doesn't
care which backend produced them.
"""

from __future__ import annotations

import logging
import os
import sys
import urllib.request
from collections import deque
from typing import Optional

import cv2
import numpy as np

from stera.core.types import Keypoint
from stera.annotations.hands import HandPose, FINGER_NAMES, FINGER_JOINTS
from stera.models.mediapipe.config import MediaPipeConfig

logger = logging.getLogger(__name__)

# MediaPipe hand landmark indices per finger (4 joints each: MCP, PIP, DIP, TIP)
_MP_FINGER_SLICES = {
    "thumb":  (1, 5),
    "index":  (5, 9),
    "middle": (9, 13),
    "ring":   (13, 17),
    "pinky":  (17, 21),
}

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
_MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe", "hand_landmarker.task")


def _ensure_model() -> str:
    if os.path.exists(_MODEL_CACHE):
        return _MODEL_CACHE
    os.makedirs(os.path.dirname(_MODEL_CACHE), exist_ok=True)
    logger.info("Downloading MediaPipe HandLandmarker model to %s", _MODEL_CACHE)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
    logger.info("MediaPipe HandLandmarker model cached")
    return _MODEL_CACHE


class MediaPipeHandTracker:
    """Full 21-joint hand tracker using MediaPipe HandLandmarker.

    Drop-in alternative to ``WiLoRHandTracker``. Returns the same
    ``HandPose`` type. Lighter weight than WiLoR, runs on CPU, with higher
    recall on far/small hands; WiLoR is generally tighter on finger joint
    positions in the world frame.

    Usage::

        from stera.models.mediapipe import MediaPipeHandTracker

        tracker = MediaPipeHandTracker()
        hands = tracker.detect_hands(rgb)
        hands = tracker.detect_hands(rgb, depth=depth, intrinsics=K)
    """

    def __init__(self, config: Optional[MediaPipeConfig] = None):
        self.config = config or MediaPipeConfig()
        self._landmarker = None
        self._mp = None
        self._loaded = False
        self._depth_tracks: dict[str, _DepthTrack] = {}

    def load(self) -> None:
        """Load MediaPipe HandLandmarker model."""
        if self._loaded:
            return

        logger.info("Loading MediaPipe HandLandmarker")

        import mediapipe as mp

        self._mp = mp
        model_path = _ensure_model()
        cfg = self.config

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=cfg.max_num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._depth_tracks = {
            "left": _DepthTrack(cfg.depth_buffer_size),
            "right": _DepthTrack(cfg.depth_buffer_size),
        }
        # MediaPipe's HandLandmarker.__del__ runs at interpreter shutdown,
        # after its thread executor is destroyed. Install a one-shot
        # ``sys.unraisablehook`` that swallows the resulting noise.
        _install_mediapipe_unraisable_filter()
        self._loaded = True
        logger.info("MediaPipe HandLandmarker ready (max_num_hands=%d)", cfg.max_num_hands)

    def close(self) -> None:
        """Manually release the MediaPipe landmarker (rarely needed; the
        unraisable filter swallows shutdown noise)."""
        lm = self._landmarker
        self._landmarker = None
        if lm is None:
            return
        try:
            lm.close()
        except Exception:
            pass

    def detect_hands(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> list[HandPose]:
        """Detect full 21-joint hands using MediaPipe world landmarks + depth anchoring.

        Returns HandPose objects compatible with WiLoR output.
        """
        self.load()

        mp = self._mp
        h, w = rgb.shape[:2]

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.hand_landmarks or not results.handedness:
            return []

        hands: list[HandPose] = []

        for hand_lm, hand_wl, handedness in zip(
            results.hand_landmarks, results.hand_world_landmarks, results.handedness
        ):
            # FPV camera observes the wearer's own hands directly (no mirror
            # in front of them). MediaPipe's handedness label is in the
            # camera's perspective; "right" = the user's right hand, matching
            # WiLoR's convention. Use the label as-is.
            mp_label = handedness[0].category_name.lower()
            hand_side = mp_label
            confidence = handedness[0].score

            # 2D landmarks in pixel coords
            landmarks_2d = np.array(
                [[lm.x * w, lm.y * h] for lm in hand_lm], dtype=np.float32
            )

            # World landmarks: 21 joints in meters, relative to hand center
            joints_rel = np.array(
                [[wl.x, wl.y, wl.z] for wl in hand_wl], dtype=np.float32
            )
            # Make relative to wrist (joint 0)
            joints_rel = joints_rel - joints_rel[0:1]

            # Depth-anchor the wrist to get absolute 3D in camera frame
            wrist_3d = None
            if depth is not None and intrinsics is not None:
                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                dh, dw_px = depth.shape[:2]
                sx, sy = dw_px / w, dh / h

                z_real = None
                for ai in [0, 9, 5, 13, 17]:
                    u_d = landmarks_2d[ai, 0] * sx
                    v_d = landmarks_2d[ai, 1] * sy
                    z = _sample_depth(depth, u_d, v_d, dw_px, dh, self.config.depth_sample_radius)
                    if z is not None:
                        z_real = z
                        break

                if z_real is None:
                    zs = [
                        _sample_depth(depth, landmarks_2d[j, 0] * sx, landmarks_2d[j, 1] * sy, dw_px, dh, 5)
                        for j in range(21)
                    ]
                    zs = [z for z in zs if z is not None]
                    if len(zs) >= 3:
                        z_real = float(np.median(zs))

                if z_real is not None:
                    wu = landmarks_2d[0, 0] * sx
                    wv = landmarks_2d[0, 1] * sy
                    wrist_3d = np.array([
                        (wu - cx) * z_real / fx,
                        (wv - cy) * z_real / fy,
                        z_real,
                    ])

                # Depth smoothing
                track = self._depth_tracks[hand_side]
                track.update(z_real)
                if wrist_3d is None and track.smoothed_depth is not None:
                    z = track.smoothed_depth
                    wu = landmarks_2d[0, 0] * sx
                    wv = landmarks_2d[0, 1] * sy
                    wrist_3d = np.array([(wu - cx) * z / fx, (wv - cy) * z / fy, z])

            # Sanity check
            if wrist_3d is not None:
                if np.any(np.abs(wrist_3d) > self.config.max_joint_abs) or wrist_3d[2] < self.config.min_wrist_depth:
                    wrist_3d = None

            # Build absolute 3D joints (camera optical frame)
            if wrist_3d is not None:
                joints_3d = joints_rel + wrist_3d
            else:
                joints_3d = None

            # Build HandPose
            hand = self._to_hand_pose(hand_side, joints_3d, landmarks_2d, confidence)
            hands.append(hand)

        # Dedup: keep best per hand side
        if len(hands) > 1:
            best: dict[str, HandPose] = {}
            for hp in hands:
                if hp.hand_side not in best or hp.confidence > best[hp.hand_side].confidence:
                    best[hp.hand_side] = hp
            hands = list(best.values())

        return hands

    @staticmethod
    def _to_hand_pose(hand_side, joints_3d, kpts_2d, confidence):
        """Convert numpy arrays to HandPose."""
        def _kp(idx, name):
            if joints_3d is not None:
                return Keypoint(
                    x=float(joints_3d[idx, 0]), y=float(joints_3d[idx, 1]),
                    z=float(joints_3d[idx, 2]), confidence=confidence, name=name,
                )
            return Keypoint(
                x=float(kpts_2d[idx, 0]), y=float(kpts_2d[idx, 1]),
                z=0.0, confidence=confidence, name=name,
            )

        fingers = {}
        for finger_name in FINGER_NAMES:
            start, _end = _MP_FINGER_SLICES[finger_name]
            fingers[finger_name] = [_kp(start + j, FINGER_JOINTS[j]) for j in range(4)]

        hp = HandPose(
            wrist=_kp(0, "wrist"),
            fingers=fingers,
            hand_side=hand_side,
            confidence=confidence,
        )
        # Stash 2D pixel coords (original RGB frame) so the rerun viz can
        # draw the hand overlay on /camera/rgb_overlay without re-projecting.
        hp._kpts_2d_rgb = np.asarray(kpts_2d, dtype=np.float32)
        hp._backend = "mediapipe"
        return hp

    def reset_tracks(self) -> None:
        """Reset depth smoothing buffers (call between sequences)."""
        for track in self._depth_tracks.values():
            track.reset()


_UNRAISABLE_FILTER_INSTALLED = False


def _install_mediapipe_unraisable_filter() -> None:
    """Install a ``sys.unraisablehook`` that swallows MediaPipe's
    shutdown-time __del__ noise. Idempotent."""
    global _UNRAISABLE_FILTER_INSTALLED
    if _UNRAISABLE_FILTER_INSTALLED:
        return
    _UNRAISABLE_FILTER_INSTALLED = True

    prev_hook = sys.unraisablehook

    def _filter(unraisable):
        # The unraisable.object is the bound __del__ for HandLandmarker
        # (or similar MediaPipe Task instances). Match by the exception's
        # source file path; anything inside the mediapipe site-package
        # tracebacks gets silenced at shutdown.
        tb = unraisable.exc_traceback
        while tb is not None:
            fname = tb.tb_frame.f_code.co_filename
            if "/mediapipe/" in fname:
                return  # swallow
            tb = tb.tb_next
        prev_hook(unraisable)

    sys.unraisablehook = _filter


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


class _DepthTrack:
    def __init__(self, buffer_size=15):
        self._buffer: deque = deque(maxlen=buffer_size)
        self.smoothed_depth: float | None = None

    def update(self, depth: float | None):
        if depth is not None:
            self._buffer.append(depth)
            self.smoothed_depth = float(np.median(list(self._buffer)))

    def reset(self):
        self._buffer.clear()
        self.smoothed_depth = None
