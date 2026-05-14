"""Shared utilities for the face-blur backends (EgoBlur / MediaPipe / RetinaFace).

The blur function is the same for all three so the *visual* output of
``FaceBlurrer.blur(...)`` looks identical regardless of which detector
located the face. Backends differ only in how they produce face bboxes.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_elliptical_blur(
    image: np.ndarray,
    boxes: np.ndarray,
    scale_factor_detections: float = 1.15,
) -> np.ndarray:
    """Elliptical large-kernel blur over each bbox on a copy of ``image``.

    Mirrors EgoBlur's gen2 demo ``visualize_blur``: for each bbox, run
    ``cv2.blur`` with a ``(H//2, W//2)`` kernel over the bbox interior,
    then composite the blurred patch back over the original via an
    elliptical mask. ``scale_factor_detections > 1`` inflates the bbox
    around its centre before blurring (safer coverage on near-misses).

    Parameters
    ----------
    image
        ``(H, W, 3)`` ``uint8`` RGB image. Returned unchanged when
        ``boxes`` is empty.
    boxes
        ``(N, 4)`` float ``[x1, y1, x2, y2]`` pixel coords.
    scale_factor_detections
        Multiplier on each bbox before blurring (defaults to 1.15).
    """
    if boxes is None or len(boxes) == 0:
        return image
    h, w = image.shape[:2]
    image_fg = image.copy()
    mask = np.zeros((h, w, 1), dtype=np.uint8)
    ksize = (max(1, h // 2), max(1, w // 2))
    s = float(scale_factor_detections)

    for box in np.asarray(boxes, dtype=np.float32):
        x1, y1, x2, y2 = box.tolist()
        if s != 1.0:
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            bw = (x2 - x1) * s
            bh = (y2 - y1) * s
            x1, y1 = cx - bw * 0.5, cy - bh * 0.5
            x2, y2 = cx + bw * 0.5, cy + bh * 0.5
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(w, int(round(x2)))
        y2 = min(h, int(round(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
        cv2.ellipse(
            mask,
            (((x1 + x2) // 2, (y1 + y2) // 2), (x2 - x1, y2 - y1), 0),
            255, -1,
        )
    inv_mask = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(image, image, mask=inv_mask)
    fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
    return cv2.add(bg, fg)
