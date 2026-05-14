"""HaMeR (Hand Mesh Recovery) backend for HandTracker.

See https://github.com/geopavlakos/hamer.

This module is a thin Python adapter that drives HaMeR's MANO regression
over our ``SyncedFrame`` data. The HaMeR source tree, model checkpoint,
and MANO files live OUTSIDE the SDK. Point ``HaMeRConfig.hamer_dir`` at
a local clone with the demo data extracted.
"""

from stera.models.hamer.config import HaMeRConfig
from stera.models.hamer.tracker import HaMeRHandTracker

__all__ = ["HaMeRConfig", "HaMeRHandTracker"]
