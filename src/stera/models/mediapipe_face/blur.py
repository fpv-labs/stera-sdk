"""MediaPipe face detector + elliptical blur: drop-in alternative to EgoBlur.

Uses MediaPipe's modern ``Tasks`` API (BlazeFace short / full range
detectors). The detector runs on CPU but is fast enough for FPV scale,
and MediaPipe is already installed if you set up MediaPipe hands.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import List, Optional

import numpy as np

from stera.utils.face_blur import apply_elliptical_blur
from stera.models.mediapipe_face.config import MediaPipeFaceConfig

logger = logging.getLogger(__name__)

# Auto-downloaded face detector model assets (cached at ~/.cache/mediapipe/).
_MODEL_URLS = {
    # Short-range works well for selfies / close-up faces (≤2 m).
    0: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    # Full-range trades short-range accuracy for general-purpose 5+ m reach.
    1: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
}


def _ensure_model(model_selection: int) -> str:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
    os.makedirs(cache_dir, exist_ok=True)
    fname = "blaze_face_short_range.tflite"
    path = os.path.join(cache_dir, fname)
    if os.path.exists(path):
        return path
    url = _MODEL_URLS.get(model_selection, _MODEL_URLS[0])
    logger.info("Downloading MediaPipe face detector model to %s", path)
    urllib.request.urlretrieve(url, path)
    return path


class MediaPipeFaceBlurrer:
    """MediaPipe BlazeFace-based face blur.

    Usage::

        from stera.models.mediapipe_face import MediaPipeFaceBlurrer

        blurrer = MediaPipeFaceBlurrer()
        blurred = blurrer.blur(rgb_or_frame)
    """

    def __init__(self, config: Optional[MediaPipeFaceConfig] = None):
        self.config = config or MediaPipeFaceConfig()
        self._detector = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        logger.info(
            "Loading MediaPipe FaceDetector (model_selection=%d, conf=%.2f)",
            self.config.model_selection, self.config.min_detection_confidence,
        )
        # Reuse the unraisable-hook filter installed by the MediaPipe hand
        # tracker so this module is silent at interpreter shutdown too.
        # Defined in stera.models.mediapipe.tracker; importing it has no
        # side effects beyond the function definition.
        from stera.models.mediapipe.tracker import (
            _install_mediapipe_unraisable_filter,
        )
        _install_mediapipe_unraisable_filter()

        import mediapipe as mp  # type: ignore[import]
        self._mp = mp
        model_path = _ensure_model(self.config.model_selection)
        base = mp.tasks.BaseOptions(model_asset_path=model_path)
        opts = mp.tasks.vision.FaceDetectorOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=self.config.min_detection_confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(opts)
        self._loaded = True

    def detect_boxes(self, rgb_frames: List[np.ndarray]) -> List[np.ndarray]:
        """Per-frame face detection. Returns one ``(N, 4)`` array of
        ``[x1, y1, x2, y2]`` boxes per input (pixel coords)."""
        self.load()
        mp = self._mp
        out: List[np.ndarray] = []
        for rgb in rgb_frames:
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._detector.detect(mp_img)
            if not res.detections:
                out.append(np.empty((0, 4), dtype=np.float32))
                continue
            boxes = []
            for det in res.detections:
                bb = det.bounding_box   # in PIXEL coords
                x1 = float(bb.origin_x)
                y1 = float(bb.origin_y)
                x2 = x1 + float(bb.width)
                y2 = y1 + float(bb.height)
                boxes.append([x1, y1, x2, y2])
            out.append(np.asarray(boxes, dtype=np.float32))
        return out

    def apply_blur(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        return apply_elliptical_blur(
            image, boxes, scale_factor_detections=self.config.scale_factor_detections,
        )

    def blur(self, rgb: np.ndarray) -> np.ndarray:
        boxes = self.detect_boxes([rgb])[0]
        return self.apply_blur(rgb, boxes)

    def blur_batch(self, rgb_frames: List[np.ndarray]) -> List[np.ndarray]:
        boxes_list = self.detect_boxes(rgb_frames)
        return [self.apply_blur(im, bx) for im, bx in zip(rgb_frames, boxes_list)]
