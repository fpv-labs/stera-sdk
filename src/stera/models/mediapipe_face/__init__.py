"""MediaPipe face detector backend for FaceBlurrer."""

from stera.models.mediapipe_face.config import MediaPipeFaceConfig
from stera.models.mediapipe_face.blur import MediaPipeFaceBlurrer

__all__ = ["MediaPipeFaceBlurrer", "MediaPipeFaceConfig"]
