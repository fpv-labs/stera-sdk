"""stera-sdk: first-person video data loading, visualization, and annotation."""

from __future__ import annotations

import logging

__version__ = "0.0.1"

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int | str = "INFO", fmt: str = _LOG_FORMAT) -> None:
    """Enable console logging for the ``stera`` package.

    Call once from your script to see SDK progress messages. Opt-in so
    library callers keep control of their own logging.
    """
    logger = logging.getLogger("stera")
    if not any(getattr(h, "_stera", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))
        handler._stera = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# Silence "No handler" warnings when setup_logging() isn't called.
logging.getLogger("stera").addHandler(logging.NullHandler())


try:
    from stera.models import HandTracker, HandTrackerConfig
except ImportError:
    pass

try:
    from stera.models import FaceBlurrer, EgoBlurConfig
except ImportError:
    pass

try:
    from stera.data.mcap import MCAPReader
except ImportError:
    pass

try:
    from stera.eval import Evaluate, EvaluateConfig
except ImportError:
    pass
