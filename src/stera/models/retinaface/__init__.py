"""RetinaFace face detector backend for FaceBlurrer.

Wraps the ``batch-face`` PyTorch RetinaFace implementation
(https://github.com/elliottzheng/batch-face). The model file is
auto-downloaded on first call and cached under
``~/.cache/torch/hub/checkpoints/``.
"""

from stera.models.retinaface.config import RetinaFaceConfig
from stera.models.retinaface.blur import RetinaFaceBlurrer

__all__ = ["RetinaFaceBlurrer", "RetinaFaceConfig"]
