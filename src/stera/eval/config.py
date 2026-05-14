"""Configurable thresholds and penalty weights for ``Evaluate``.

Two kinds of knobs:

1. **Color thresholds** — pairs ``(good_min, ok_min)``. A metric ≥ ``good_min``
   renders green; ≥ ``ok_min`` renders amber; below renders red.

2. **Health-score deductions** — each check subtracts from a starting 100:
   ``penalty = max(0, target - actual_pct) * weight``. Set ``weight=0`` to
   disable a check entirely.

Hand metrics report three buckets:

* ``hand_1plus_*`` → frames with **at least one** hand (= any-hand)
* ``hand_2_*``    → frames with **exactly two** hands
* ``frames_with_more_hands*`` → frames with **strictly more than two** hands
  (metric only, no score deduction by default — usually a detection error
  indicator).

Usage::

    from stera import Evaluate, EvaluateConfig

    cfg = EvaluateConfig(
        sync_target=95.0,         # stricter sync target
        hand_2_weight=0.0,        # don't penalize low exactly-2-hands %
        depth_required=False,     # don't fail the score when depth is missing
    )
    Evaluate(session, config=cfg).show()
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluateConfig:
    # --- color thresholds (good_min, ok_min). higher = better ---
    health_thresholds: tuple[float, float] = (80.0, 60.0)
    depth_valid_thresholds: tuple[float, float] = (80.0, 50.0)
    sync_thresholds: tuple[float, float] = (90.0, 70.0)
    hand_any_thresholds: tuple[float, float] = (70.0, 30.0)
    hand_1plus_thresholds: tuple[float, float] = (40.0, 15.0)
    hand_2_thresholds: tuple[float, float] = (30.0, 10.0)

    # IMU gravity vector magnitude is colored good when |Δ from 9.81| ≤ this
    imu_gravity_max_dev: float = 0.5  # m/s²

    # --- stream-required flags ---
    # When True (default), a missing stream subtracts its _missing_penalty.
    # Set to False to skip the penalty entirely, regardless of the penalty value.
    depth_required: bool = True
    imu_required: bool = True

    # --- health-score deductions (subtracted from 100) ---
    # RGB frame gaps: each gap subtracts 1, clipped at this maximum
    rgb_gap_max_penalty: float = 10.0

    # Depth: penalty = max(0, target - valid_pct_mean) * weight
    depth_valid_target: float = 80.0
    depth_valid_weight: float = 0.3
    depth_missing_penalty: float = 10.0

    # Pose / IMU missing: flat penalties
    pose_missing_penalty: float = 15.0
    imu_missing_penalty: float = 5.0

    # Sync: penalty per pair = max(0, target - within_50ms_pct) * weight
    sync_target: float = 90.0
    sync_weight: float = 0.1

    # Hand presence: penalty = max(0, target - actual_pct) * weight.
    hand_any_target: float = 50.0
    hand_any_weight: float = 0.15
    # ≥1 hand
    hand_1plus_target: float = 30.0
    hand_1plus_weight: float = 0.05
    # exactly 2 hands
    hand_2_target: float = 20.0
    hand_2_weight: float = 0.10
    hand_missing_penalty: float = 0.0  # if hand-pose buffer is empty


DEFAULT_CONFIG = EvaluateConfig()
