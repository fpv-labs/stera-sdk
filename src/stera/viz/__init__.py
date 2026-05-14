"""Rerun-based visualization for frames and poses."""

from stera.viz.logger import FPVLogger
from stera.viz.log_annotations import (
    log_camera_pose,
    log_hand_pose,
    log_upper_body,
)
from stera.viz.session import Visualizer, RerunVisualizer

__all__ = [
    "FPVLogger",
    "Visualizer",
    "RerunVisualizer",   # deprecated alias
    "log_camera_pose",
    "log_hand_pose",
    "log_upper_body",
]
