"""EgoBlur face blurrer configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EgoBlurConfig:
    """Configuration for EgoBlur face detection + blur.

    Parameters
    ----------
    egoblur_dir : Path to a local clone of EgoBlur (the directory that
        contains ``gen2/script/...`` and ``ego_blur_face_gen2.jit``).
        Both the JIT model and the vendored gen2 source code are derived
        from this single path. Pass via
        ``FaceBlurrer(model="egoblur", model_path=egoblur_dir)``.
    model_path : Optional override for the JIT path. Defaults to
        ``<egoblur_dir>/ego_blur_face_gen2.jit``.
    code_dir : Optional override for the gen2 code directory. Defaults to
        ``egoblur_dir`` (we prepend it to ``sys.path`` so
        ``import gen2.script.…`` resolves).
    device : Torch device for inference. Typically ``"cuda"``.
    score_thresh : Minimum face-detection score to keep.
    iou_thresh : NMS IoU threshold.
    use_fp16 : Run the JIT model in FP16 on CUDA when supported.
    batch_size : Internal sub-batch size for the JIT model.
    scale_factor_detections : Multiplier applied to each detected bbox
        before blurring (>1.0 inflates the box for safer coverage).
    """

    egoblur_dir: Optional[str] = None
    model_path: Optional[str] = None
    code_dir: Optional[str] = None
    device: str = "cuda"
    score_thresh: float = 0.8
    iou_thresh: float = 0.5
    use_fp16: bool = True
    batch_size: int = 8
    scale_factor_detections: float = 1.15
