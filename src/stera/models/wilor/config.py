"""WiLoR tracker configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WiLoRConfig:
    """Configuration for WiLoR hand tracking.

    Parameters
    ----------
    wilor_dir : Path to the frozen WiLoR installation directory. Must be set
        before the tracker loads. Pass via ``WiLoRConfig(wilor_dir=...)`` or
        ``HandTracker(model="wilor", model_path=...)``.
    yolo_conf : YOLO detection confidence threshold.
    rescale_factor : Bounding box rescale factor for ViTDetDataset.
    batch_size : Batch size for WiLoR inference.
    detect_every_n : Run detection every N frames (1 = every frame).
    depth_buffer_size : Rolling median buffer size for depth smoothing.
    depth_sample_radius : Pixel radius for depth sampling patch.
    save_mano_vertices : Whether to extract MANO mesh vertices (778 per hand).
    """

    wilor_dir: Optional[str] = None
    yolo_conf: float = 0.4
    rescale_factor: float = 2.0
    batch_size: int = 16
    detect_every_n: int = 1
    depth_buffer_size: int = 15
    depth_sample_radius: int = 7
    save_mano_vertices: bool = True   # captured per-frame; written by session.export

    # Sanity check thresholds
    max_joint_abs: float = 3.0
    min_wrist_depth: float = 0.05
    min_palm_span: float = 0.02
    max_palm_span: float = 0.5
