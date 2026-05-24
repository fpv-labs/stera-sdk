"""Regression test for depth anchoring (issue #2: "depth anchoring logic is wrong").

The hand trackers return a *wrist-relative* skeleton (``joints_cam - joints_cam[0]``)
that is anchored into the camera frame by back-projecting a 2D pixel at a sampled
depth. ``ANCHOR_JOINTS`` is only a priority list for finding a *valid depth value*
(the wrist is often occluded / at a depth discontinuity). The anchor *pixel* must
always be the wrist, otherwise the whole hand is translated by the wrist->joint
offset (~palm length, several cm).

These tests exercise the real ``_anchor_hand`` helpers in both backends.
"""
from __future__ import annotations

import numpy as np
import pytest

from stera.models.hamer.tracker import _anchor_hand as _hamer_anchor
from stera.models.wilor.tracker import WiLoRHandTracker

# --- synthetic camera / hand setup ---------------------------------------
FX = FY = 600.0
CX, CY = 320.0, 240.0
DW, DH = 640, 480
WRIST_PX = (100.0, 100.0)
J9_PX = (160.0, 80.0)        # middle-finger MCP, well away from the wrist
Z_MM = 500                   # depth sample, in millimetres
Z_M = Z_MM / 1000.0

# Both backends expose the same signature; HaMeR returns the joints array,
# WiLoR returns (joints, wrist_depth).
IMPLS = [
    pytest.param(WiLoRHandTracker._anchor_hand, id="wilor"),
    pytest.param(_hamer_anchor, id="hamer"),
]


def _backproject(px, z=Z_M):
    u, v = px
    return np.array([(u - CX) * z / FX, (v - CY) * z / FY, z])


def _fill(depth, px, value, half=10):
    u, v = int(px[0]), int(px[1])
    depth[v - half:v + half + 1, u - half:u + half + 1] = value


def _kpts():
    k = np.full((21, 2), 300.0, dtype=np.float32)  # default: land on invalid depth
    k[0] = WRIST_PX
    k[9] = J9_PX
    return k


def _joints_cam():
    j = np.zeros((21, 3), dtype=np.float32)
    j[0] = (0.50, 0.50, 0.60)     # wrist
    j[9] = (0.53, 0.42, 0.61)     # middle-MCP, ~8.6 cm from the wrist
    for i in range(21):
        if i not in (0, 9):       # keep the skeleton non-degenerate
            j[i] = (0.50 + 0.005 * i, 0.50 - 0.004 * i, 0.60 + 0.001 * i)
    return j


def _call(impl, depth):
    out = impl(
        _joints_cam(), np.zeros(3), _kpts(), depth,
        FX, FY, CX, CY, DW, DH, DW, DH,
        is_left=False, sample_radius=7, image_is_rotated=False,
    )
    joints = out[0] if isinstance(out, tuple) else out
    return np.asarray(joints)


@pytest.mark.parametrize("impl", IMPLS)
def test_anchor_uses_wrist_pixel_when_wrist_depth_missing(impl):
    """Wrist depth invalid, palm joint valid -> wrist still anchored at the wrist pixel."""
    depth = np.zeros((DH, DW), dtype=np.uint16)
    _fill(depth, J9_PX, Z_MM)          # only the middle-MCP has valid depth

    joints = _call(impl, depth)

    np.testing.assert_allclose(
        joints[0], _backproject(WRIST_PX), atol=1e-6,
        err_msg="wrist must anchor on the wrist pixel, not the depth-sampled joint",
    )
    # Guard against the pre-fix behaviour (anchoring on the middle-MCP pixel).
    assert not np.allclose(joints[0], _backproject(J9_PX), atol=1e-3)


@pytest.mark.parametrize("impl", IMPLS)
def test_anchor_unchanged_when_wrist_depth_present(impl):
    """Common case (wrist depth valid) must be unaffected by the fix."""
    depth = np.zeros((DH, DW), dtype=np.uint16)
    _fill(depth, WRIST_PX, Z_MM)
    _fill(depth, J9_PX, Z_MM)

    joints = _call(impl, depth)
    np.testing.assert_allclose(joints[0], _backproject(WRIST_PX), atol=1e-6)
