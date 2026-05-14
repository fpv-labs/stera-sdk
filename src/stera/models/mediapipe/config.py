"""MediaPipe wrist tracker configuration."""

from dataclasses import dataclass


@dataclass
class MediaPipeConfig:
    """Configuration for MediaPipe wrist tracking.

    Parameters
    ----------
    max_num_hands : Maximum hands to detect per frame.
    min_detection_confidence : YOLO-like detection threshold. Lower = fewer misses.
    min_presence_confidence : Hand presence threshold. Lower = stickier tracking.
    min_tracking_confidence : Tracking threshold between frames.
    depth_sample_radius : Pixel radius for depth sampling patch.
    depth_buffer_size : Rolling median buffer for depth smoothing.
    """

    max_num_hands: int = 2
    min_detection_confidence: float = 0.3
    min_presence_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    depth_sample_radius: int = 7
    depth_buffer_size: int = 15

    # Sanity checks
    max_joint_abs: float = 3.0
    min_wrist_depth: float = 0.05
