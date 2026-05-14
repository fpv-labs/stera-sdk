"""RetinaFace face detector configuration."""

from dataclasses import dataclass


@dataclass
class RetinaFaceConfig:
    """Configuration for the RetinaFace backend (via ``batch-face``).

    Parameters
    ----------
    network
        ``"mobilenet"`` (~2 MB, fast) or ``"resnet50"`` (~109 MB, more
        accurate). The model file is auto-downloaded on first call to
        ``~/.cache/torch/hub/checkpoints/``.
    gpu_id
        CUDA device ordinal to load the model on. ``-1`` forces CPU.
    score_thresh
        Minimum face-detection score to keep.
    batch_size
        Internal sub-batch size when calling ``blur_batch`` /
        ``detect_boxes`` with many frames at once.
    scale_factor_detections
        Multiplier applied to each bbox before blurring (>1 inflates).
    """

    network: str = "mobilenet"
    gpu_id: int = 0
    score_thresh: float = 0.8
    batch_size: int = 8
    scale_factor_detections: float = 1.15
