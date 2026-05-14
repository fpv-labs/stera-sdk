"""EgoBlur face-blur wrapper.

Thin helper around Meta's vendored EgoBlur gen2 pipeline: batched GPU
inference + NMS + score filter (ported from benchmark_face_fast.py) and the
demo's elliptical cv2.blur compositing (ported from demo_ego_blur_gen2.py).

The gen2 source tree and ``.jit`` model file are not distributed with this
SDK; the caller is expected to point ``EgoBlurConfig.code_dir`` at a local
checkout of the gen2 code and ``EgoBlurConfig.model_path`` at the JIT model.
Nothing is downloaded at runtime.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from stera.models.egoblur.config import EgoBlurConfig

logger = logging.getLogger(__name__)


class EgoBlurFace:
    """Run EgoBlur face detection + blurring on RGB frames.

    Usage::

        from stera.models.egoblur import EgoBlurFace, EgoBlurConfig

        blurrer = EgoBlurFace(EgoBlurConfig(
            model_path="/opt/egoblur/ego_blur_face_gen2.jit",
            code_dir="/opt/egoblur",
        ))
        blurred = blurrer.blur_batch([rgb1, rgb2, rgb3])
    """

    def __init__(self, config: Optional[EgoBlurConfig] = None):
        self.config = config or EgoBlurConfig()
        self.model = None
        self.use_fp16 = False
        self.device = None
        self._loaded = False
        self._resize_cache: dict[tuple[int, int], tuple[int, int]] = {}

    def load(self) -> None:
        """Load the EgoBlur JIT model. Called automatically on first inference."""
        if self._loaded:
            return

        cfg = self.config
        # Resolve JIT + code paths. Prefer explicit overrides; otherwise
        # derive from ``egoblur_dir``.
        if not cfg.egoblur_dir and not cfg.model_path:
            raise RuntimeError(
                "EgoBlurConfig.egoblur_dir not set. Pass it via "
                "FaceBlurrer(model='egoblur', model_path='/path/to/EgoBlur')."
            )
        if cfg.egoblur_dir and not os.path.isdir(cfg.egoblur_dir):
            raise FileNotFoundError(
                f"EgoBlur directory not found: {cfg.egoblur_dir}"
            )
        jit_path = cfg.model_path or os.path.join(
            cfg.egoblur_dir, "ego_blur_face_gen2.jit",
        )
        code_dir = cfg.code_dir or cfg.egoblur_dir or os.path.dirname(jit_path)
        if not os.path.isfile(jit_path):
            raise FileNotFoundError(
                f"EgoBlur JIT model not found: {jit_path}\n"
                "  Drop ``ego_blur_face_gen2.jit`` into egoblur_dir or set "
                "EgoBlurConfig.model_path explicitly."
            )
        if not os.path.isdir(code_dir):
            raise FileNotFoundError(
                f"EgoBlur code dir not found: {code_dir}. "
                "Point egoblur_dir/code_dir at the directory containing gen2/script/..."
            )

        logger.info("Loading EgoBlur model from %s", jit_path)

        import torch
        self._torch = torch

        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        from gen2.script.constants import RESIZE_MIN_GEN2, RESIZE_MAX_GEN2
        from gen2.script.detectron2.export.torchscript_patch import patch_instances
        from gen2.script.detectron2.structures import Boxes, Instances  # noqa: F401
        from gen2.script.detectron2.utils import (
            ResizeShortestEdge,
            convert_scripted_instances,
            detector_postprocess,
        )
        from gen2.script.predictor import ClassID, PATCH_INSTANCES_FIELDS

        self._resize_min = RESIZE_MIN_GEN2
        self._resize_max = RESIZE_MAX_GEN2
        self._patch_instances = patch_instances
        self._patch_fields = PATCH_INSTANCES_FIELDS
        self._ResizeShortestEdge = ResizeShortestEdge
        self._convert_scripted_instances = convert_scripted_instances
        self._detector_postprocess = detector_postprocess
        self._face_class = int(ClassID.FACE.value)

        self.device = torch.device(cfg.device)

        # TorchScript models are always in eval mode at load; calling .eval()
        # fails for this build because `training` is a frozen jit constant.
        model = torch.jit.load(jit_path, map_location="cpu")
        self.use_fp16 = False
        if self.device.type == "cuda" and cfg.use_fp16:
            try:
                model.half()
                self.use_fp16 = True
            except Exception as e:
                logger.warning("EgoBlur FP16 half() failed (%s); falling back to FP32", e)
        model.to(self.device)
        self.model = model

        self._loaded = True
        logger.info("EgoBlur ready (device=%s, fp16=%s)", self.device, self.use_fp16)

    def _resize_dims(self, h: int, w: int) -> Tuple[int, int]:
        key = (h, w)
        if key not in self._resize_cache:
            self._resize_cache[key] = self._ResizeShortestEdge.get_output_shape(
                h, w, self._resize_min, self._resize_max,
            )
        return self._resize_cache[key]

    def detect_boxes(self, rgb_frames: List[np.ndarray]) -> List[np.ndarray]:
        """Batched face detection.

        Returns one ``(N, 4)`` float32 array of ``[x1, y1, x2, y2]`` boxes
        per input frame (pixel coords in the ORIGINAL resolution).
        """
        self.load()

        torch = self._torch
        import torch.nn.functional as F
        import torchvision.ops as tvops

        if not rgb_frames:
            return []
        h, w = rgb_frames[0].shape[:2]
        th, tw = self._resize_dims(h, w)
        dtype = torch.float16 if self.use_fp16 else torch.float32

        out: List[np.ndarray] = []
        cfg = self.config
        with self._patch_instances(fields=self._patch_fields):
            for start in range(0, len(rgb_frames), cfg.batch_size):
                batch = rgb_frames[start:start + cfg.batch_size]
                # The gen2 demo feeds BGR into the detector (cv2.imread
                # returns BGR). Our frames are RGB; swap channels.
                bgr_chw = np.stack([
                    np.ascontiguousarray(np.transpose(f[:, :, ::-1], (2, 0, 1)))
                    for f in batch
                ])
                t = torch.from_numpy(bgr_chw).to(self.device).to(dtype)
                if t.shape[2] != th or t.shape[3] != tw:
                    t = F.interpolate(
                        t, size=(th, tw), mode="bilinear", align_corners=False,
                    )
                with torch.no_grad():
                    preds = self.model.inference(
                        [{"image": t[i]} for i in range(t.shape[0])],
                        do_postprocess=False,
                    )
                for pred in preds:
                    inst = self._convert_scripted_instances(pred)
                    inst = self._detector_postprocess(inst, h, w)
                    boxes = inst.pred_boxes.tensor
                    scores = inst.scores
                    if inst.has("pred_classes"):
                        mask = inst.pred_classes == self._face_class
                        boxes = boxes[mask]
                        scores = scores[mask]
                    if boxes.numel() == 0:
                        out.append(np.empty((0, 4), dtype=np.float32))
                        continue
                    keep = tvops.nms(boxes, scores, cfg.iou_thresh)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    score_mask = scores > cfg.score_thresh
                    out.append(
                        boxes[score_mask].cpu().numpy().astype(np.float32)
                    )
        return out

    def apply_blur(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Elliptical large-kernel blur over each bbox. Shared with the
        MediaPipe / RetinaFace backends so visual output is identical
        across detectors."""
        from stera.utils.face_blur import apply_elliptical_blur
        return apply_elliptical_blur(
            image, boxes,
            scale_factor_detections=self.config.scale_factor_detections,
        )

    def blur(self, rgb: np.ndarray) -> np.ndarray:
        """Detect + blur a single RGB frame. Returns a new array."""
        boxes = self.detect_boxes([rgb])[0]
        return self.apply_blur(rgb, boxes)

    def blur_batch(self, rgb_frames: List[np.ndarray]) -> List[np.ndarray]:
        """Detect + blur a list of RGB frames. Returns a new list aligned with input."""
        boxes_list = self.detect_boxes(rgb_frames)
        return [self.apply_blur(img, boxes)
                for img, boxes in zip(rgb_frames, boxes_list)]
