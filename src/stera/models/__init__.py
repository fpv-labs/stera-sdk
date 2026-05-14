"""Model wrappers for inference."""

from stera.models.wilor import WiLoRHandTracker, WiLoRConfig
from stera.models.skeleton import UpperBodyEstimator, SkeletonConfig, SkeletonFrame
from stera.models.egoblur import EgoBlurFace, EgoBlurConfig


class HandTracker:
    """Unified hand tracker that delegates to a model backend.

    Usage::

        from stera.models import HandTracker

        tracker = HandTracker(model="wilor", model_path="/path/to/WiLoR")
    """

    SUPPORTED_MODELS = {"wilor", "mediapipe", "hamer"}

    def __init__(self, model: str = "wilor", model_path: str | None = None, **kwargs):
        self.model_name = model.lower()
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown hand tracker model '{model}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_MODELS))}"
            )

        if self.model_name == "wilor":
            config = WiLoRConfig()
            if model_path is not None:
                config.wilor_dir = model_path
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            self._backend = WiLoRHandTracker(config)

        elif self.model_name == "mediapipe":
            # MediaPipe downloads its asset to ~/.cache/mediapipe; no
            # model_path needed.
            from stera.models.mediapipe import (
                MediaPipeHandTracker,
                MediaPipeConfig,
            )
            config = MediaPipeConfig()
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            self._backend = MediaPipeHandTracker(config)

        elif self.model_name == "hamer":
            # model_path: HaMeR repo dir. Uses detectron2 + ViTPose for
            # body/hand bboxes, then HaMeR MANO regression.
            from stera.models.hamer import HaMeRHandTracker, HaMeRConfig
            config = HaMeRConfig()
            if model_path is not None:
                config.hamer_dir = model_path
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            self._backend = HaMeRHandTracker(config)

        self._backend.load()

    def load(self):
        self._backend.load()

    def detect_hands(self, rgb_or_frame, depth=None, intrinsics=None):
        """Run hand detection.

        Accepts either a ``SyncedFrame`` (pulls rgb/depth/depth_K from it) or
        an ``(rgb, depth=None, intrinsics=None)`` triple.
        """
        if hasattr(rgb_or_frame, "rgb"):
            frame = rgb_or_frame
            rgb = frame.rgb
            if depth is None:
                depth = frame.depth
            if intrinsics is None:
                intrinsics = getattr(frame, "depth_K", None)
                if intrinsics is None:
                    intrinsics = getattr(frame, "rgb_K", None)
        else:
            rgb = rgb_or_frame
        return self._backend.detect_hands(rgb, depth=depth, intrinsics=intrinsics)

    def __repr__(self):
        return f"HandTracker(model='{self.model_name}')"


class FaceBlurrer:
    """Unified face blurrer that delegates to a model backend.

    All backends produce identical elliptical-blur output; they differ
    only in the face detector.

    ``model="egoblur"``     Meta's gen2 EgoBlur (TorchScript). Requires a
                            local clone of the EgoBlur repo:
                            ``FaceBlurrer(model="egoblur",
                                          model_path="/path/to/EgoBlur")``.

    ``model="mediapipe"``   MediaPipe BlazeFace. CPU, zero setup.

    ``model="retinaface"``  RetinaFace (via ``batch-face``). GPU,
                            auto-downloads weights on first call.
                            Pass ``network="resnet50"`` for the larger
                            model.

    Methods are the same regardless of backend::

        blurred_rgb  = blurrer.blur(frame_or_rgb)
        blurred_list = blurrer.blur_batch([rgb1, rgb2])
        boxes_list   = blurrer.detect_boxes([rgb1, rgb2])
    """

    SUPPORTED_MODELS = {"egoblur", "mediapipe", "retinaface"}

    def __init__(self, model: str = "egoblur", model_path: str | None = None, **kwargs):
        self.model_name = model.lower()
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown face blurrer model '{model}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_MODELS))}"
            )

        if self.model_name == "egoblur":
            # model_path: EgoBlur repo dir containing ego_blur_face_gen2.jit
            # and the gen2/ source. Override jit or code dir via kwargs.
            config = EgoBlurConfig()
            if model_path is not None:
                config.egoblur_dir = model_path
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            self._backend = EgoBlurFace(config)

        elif self.model_name == "mediapipe":
            from stera.models.mediapipe_face import (
                MediaPipeFaceBlurrer, MediaPipeFaceConfig,
            )
            config = MediaPipeFaceConfig()
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            self._backend = MediaPipeFaceBlurrer(config)

        elif self.model_name == "retinaface":
            from stera.models.retinaface import (
                RetinaFaceBlurrer, RetinaFaceConfig,
            )
            config = RetinaFaceConfig()
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            self._backend = RetinaFaceBlurrer(config)

        self._backend.load()

    def load(self):
        self._backend.load()

    def blur(self, rgb_or_frame) -> "np.ndarray":
        """Detect + blur a single frame. Returns a new RGB array.

        Accepts either a ``SyncedFrame`` (pulls ``frame.rgb``) or a raw RGB
        ``(H, W, 3)`` ndarray.
        """
        rgb = rgb_or_frame.rgb if hasattr(rgb_or_frame, "rgb") else rgb_or_frame
        return self._backend.blur(rgb)

    def blur_batch(self, frames_or_rgbs) -> list:
        """Detect + blur a list of frames or RGB arrays.

        Accepts a list of ``SyncedFrame`` objects or a list of RGB arrays.
        Returns a list of blurred RGB arrays aligned with the input.
        """
        rgbs = [f.rgb if hasattr(f, "rgb") else f for f in frames_or_rgbs]
        return self._backend.blur_batch(rgbs)

    def detect_boxes(self, frames_or_rgbs) -> list:
        """Run face detection only (no blurring). Returns one (N, 4) array
        of [x1, y1, x2, y2] boxes per input."""
        rgbs = [f.rgb if hasattr(f, "rgb") else f for f in frames_or_rgbs]
        return self._backend.detect_boxes(rgbs)

    def __repr__(self):
        return f"FaceBlurrer(model='{self.model_name}')"


__all__ = [
    "HandTracker", "WiLoRHandTracker", "WiLoRConfig",
    "FaceBlurrer", "EgoBlurFace", "EgoBlurConfig",
    "UpperBodyEstimator", "SkeletonConfig", "SkeletonFrame",
]

try:
    from stera.models.mediapipe import MediaPipeHandTracker, MediaPipeConfig
    __all__ += ["MediaPipeHandTracker", "MediaPipeConfig"]
except ImportError:
    pass

try:
    from stera.models.mediapipe_face import (
        MediaPipeFaceBlurrer, MediaPipeFaceConfig,
    )
    __all__ += ["MediaPipeFaceBlurrer", "MediaPipeFaceConfig"]
except ImportError:
    pass

try:
    from stera.models.retinaface import RetinaFaceBlurrer, RetinaFaceConfig
    __all__ += ["RetinaFaceBlurrer", "RetinaFaceConfig"]
except ImportError:
    pass

try:
    from stera.models.hamer import HaMeRHandTracker, HaMeRConfig
    __all__ += ["HaMeRHandTracker", "HaMeRConfig"]
except ImportError:
    pass

