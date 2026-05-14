"""MediaPipe face detector configuration."""

from dataclasses import dataclass


@dataclass
class MediaPipeFaceConfig:
    """Configuration for the MediaPipe face-detection backend.

    Notes
    -----
    No path needed: MediaPipe downloads its blaze face detector model
    asset on first call (cached at ``~/.cache/mediapipe/``).
    """

    # Detector knobs
    min_detection_confidence: float = 0.5
    # 0 = short-range (close subjects), 1 = full-range (general purpose).
    model_selection: int = 1

    # Blur compositing
    scale_factor_detections: float = 1.15
