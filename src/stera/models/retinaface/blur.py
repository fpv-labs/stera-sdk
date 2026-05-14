"""RetinaFace face detector + elliptical blur: drop-in alternative to EgoBlur.

Wraps the pure-PyTorch ``batch-face`` RetinaFace implementation. Faster
than EgoBlur on CPU, comparable on GPU, with auto-downloaded weights
(no manual setup needed).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from stera.utils.face_blur import apply_elliptical_blur
from stera.models.retinaface.config import RetinaFaceConfig

logger = logging.getLogger(__name__)


class RetinaFaceBlurrer:
    """RetinaFace-based face blur (via the ``batch-face`` package).

    Usage::

        from stera.models.retinaface import RetinaFaceBlurrer

        blurrer = RetinaFaceBlurrer()
        blurred = blurrer.blur(rgb_or_frame)
        blurred_list = blurrer.blur_batch([rgb1, rgb2, rgb3])
    """

    def __init__(self, config: Optional[RetinaFaceConfig] = None):
        self.config = config or RetinaFaceConfig()
        self._detector = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        cfg = self.config
        try:
            from batch_face import RetinaFace  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                "batch-face is not installed. Run: pip install batch-face"
            ) from e
        logger.info(
            "Loading RetinaFace (%s) on gpu=%d", cfg.network, cfg.gpu_id,
        )
        self._detector = RetinaFace(
            gpu_id=cfg.gpu_id, network=cfg.network,
        )
        self._loaded = True

    def detect_boxes(self, rgb_frames: List[np.ndarray]) -> List[np.ndarray]:
        """Per-frame face detection. Returns one ``(N, 4)`` array of
        ``[x1, y1, x2, y2]`` boxes per input (pixel coords)."""
        self.load()
        if not rgb_frames:
            return []
        cfg = self.config
        out: List[np.ndarray] = [None] * len(rgb_frames)  # type: ignore[list-item]
        bs = max(1, int(cfg.batch_size))
        for start in range(0, len(rgb_frames), bs):
            batch = rgb_frames[start:start + bs]
            results = self._detector(batch)
            # batch-face returns list-of-list for list input, list of tuples
            # for single-image input. Normalise.
            if results and not isinstance(results[0], list):
                results = [results]
            for off, faces in enumerate(results):
                boxes = []
                for face in faces:
                    bbox, _landmarks, score = face
                    if score < cfg.score_thresh:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in bbox)
                    boxes.append([x1, y1, x2, y2])
                out[start + off] = (
                    np.asarray(boxes, dtype=np.float32) if boxes
                    else np.empty((0, 4), dtype=np.float32)
                )
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
