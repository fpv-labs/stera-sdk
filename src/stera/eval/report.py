"""Render the precomputed metrics + Plotly figures to a single HTML file.

Editorial-style report: sticky side nav, typographic hierarchy, generous
whitespace, no nested boxes. Detailed reference tables tucked behind a single
"Technical reference" disclosure at the bottom.

The Plotly.js source is inlined once in the head so the file opens offline.
"""

from __future__ import annotations

import base64
import html
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------- formatting helpers ----------------------------

def _fmt(value, spec: str = ".3g") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NaN"
        return format(float(value), spec)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v, spec) for v in value) + "]"
    return html.escape(str(value))


def _kv(rows: Iterable[tuple[str, str]]) -> str:
    body = "\n".join(
        f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{body}</table>"


def _stat(label: str, value: str, hint: str | None = None,
          tone: str | None = None) -> str:
    """One inline KPI: huge value, tiny label."""
    cls = f" stat--{tone}" if tone else ""
    hint_html = f"<div class='stat-hint'>{html.escape(hint)}</div>" if hint else ""
    return (
        f"<div class='stat{cls}'>"
        f"<div class='stat-label'>{html.escape(label)}</div>"
        f"<div class='stat-value'>{value}</div>"
        f"{hint_html}"
        "</div>"
    )


def _stat_row(stats: list[str], cols: int = 4) -> str:
    return (
        f"<div class='stat-row stat-row--{cols}'>"
        f"{''.join(stats)}"
        "</div>"
    )


def _tone(value, thresholds: tuple[float, float]) -> str | None:
    """Return 'good' / 'warn' / 'bad' for a percentage (higher = better)."""
    if value is None:
        return None
    good, ok = thresholds
    if value >= good:
        return "good"
    if value >= ok:
        return "warn"
    return "bad"


def _thumb_b64(thumb) -> str | None:
    if thumb is None:
        return None
    try:
        import cv2
        bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as e:
        logger.warning("thumbnail encode failed: %s", e)
        return None


# ----------------------------- top-level entry -------------------------------

def write_report(path: Path, metrics: dict, session=None) -> None:
    from stera.eval.plots import build_figures
    import plotly.io as pio
    from plotly.offline import get_plotlyjs

    figures = build_figures(metrics)
    plotlyjs = get_plotlyjs()
    figs = {
        name: pio.to_html(
            fig, include_plotlyjs=False, full_html=False,
            div_id=f"fig_{name}",
            config={"displaylogo": False, "responsive": True},
        )
        for name, fig in figures.items()
    }

    sections: list[tuple[str, str, str]] = []
    sections.append(("summary", "Summary", _section_summary(metrics)))
    sections.append(("trajectory", "Trajectory", _section_trajectory(metrics, figs)))
    sections.append(("imu", "IMU", _section_imu(metrics, figs)))
    sections.append(("depth", "Depth", _section_depth(metrics, figs)))
    sections.append(("hands", "Hands", _section_hands(metrics, figs)))
    sections.append(("sync", "Sync", _section_sync(metrics, figs)))
    sections.append(("map", "3D map", _section_map(metrics)))
    sections.append(("coverage", "Coverage", _section_coverage(figs)))
    sections.append(("technical", "Technical reference",
                     _section_technical(metrics)))

    Path(path).write_text(
        _build_html(plotlyjs, sections, metrics), encoding="utf-8",
    )


# --------------------------------- Summary -----------------------------------

def _section_summary(metrics: dict) -> str:
    rec = metrics["recording"]
    rgb = metrics["rgb"]
    traj = metrics["trajectory"] or {}
    hands = metrics["hands"] or {}
    imu = metrics["imu"] or {}
    depth = metrics["depth"] or {}
    health = metrics["health"]
    cfg = metrics.get("config")

    thumb_b64 = _thumb_b64(metrics.get("thumbnail"))
    thumb_html = (
        f"<img class='thumb' src='data:image/jpeg;base64,{thumb_b64}' alt=''/>"
        if thumb_b64 else "<div class='thumb thumb--empty'>no thumbnail</div>"
    )

    score = health["score"]
    tone = _tone(score, cfg.health_thresholds) if cfg else (
        "good" if score >= 80 else ("warn" if score >= 60 else "bad")
    )
    label = {"good": "Good", "warn": "Watch", "bad": "Issues"}.get(tone, "—")

    notes_html = "".join(
        f"<li>{html.escape(n)}</li>" for n in health["notes"]
    ) or "<li class='good'>all checks pass</li>"

    distance = traj.get("path_length_m")
    distance_str = f"{distance:.1f} m" if distance is not None else "—"
    hand_pct = hands.get("frames_with_any_hand_pct")
    hand_str = f"{hand_pct:.1f}%" if hand_pct is not None else "—"
    imu_rate = imu.get("effective_rate_hz")
    imu_str = f"{imu_rate:.0f} Hz" if imu_rate else "—"
    depth_valid = depth.get("valid_pct_mean")
    depth_str = f"{depth_valid:.1f}%" if depth_valid is not None else "—"

    stats = [
        _stat("Duration", rec["duration_hms"]),
        _stat("Frames", f"{rgb['frame_count']:,}"),
        _stat("FPS", _fmt(rgb.get("effective_fps"), ".1f")),
        _stat("File size", f"{rec['size_mb']:.0f} MB"),
        _stat("Distance", distance_str, hint="camera path"),
        _stat("Hands present", hand_str, hint="of all frames"),
        _stat("IMU rate", imu_str),
        _stat("Depth valid", depth_str, hint="mean per frame"),
    ]

    return f"""
<div class='summary'>
  <div class='summary-left'>
    {thumb_html}
    <p class='filename'>{html.escape(rec['filename'])}</p>
    <p class='filesub'>{html.escape(rec.get('start_iso') or '')}</p>
  </div>
  <div class='summary-right'>
    <div class='health'>
      <div class='health-num stat--{tone}'>
        <span class='health-score'>{score:.0f}</span>
        <span class='health-suffix'>/ 100</span>
      </div>
      <div class='health-meta'>
        <div class='health-label stat--{tone}'>{label}</div>
        <ul class='health-notes'>{notes_html}</ul>
      </div>
    </div>
    {_stat_row(stats, cols=4)}
  </div>
</div>
"""


# ------------------------------- Trajectory ----------------------------------

def _section_trajectory(metrics: dict, figs: dict) -> str:
    t = metrics["trajectory"]
    if t is None:
        return _empty("No camera-pose data in this session.")

    stats = [
        _stat("Path length", f"{t.get('path_length_m', 0):.2f}",
              hint="metres"),
        _stat("Net displacement",
              f"{t.get('net_displacement_m', 0):.2f}", hint="metres"),
        _stat("Footprint",
              f"{t.get('footprint_area_m2', 0):.2f}", hint="m² (xz hull)"),
        _stat("Speed mean",
              f"{t.get('speed_mean_mps') or 0:.2f}", hint="m/s"),
        _stat("Speed max",
              f"{t.get('speed_max_mps') or 0:.2f}", hint="m/s"),
        _stat("Turns", _fmt(t.get('turn_count')),
              hint="> 45° steps"),
        _stat("Stationary",
              f"{t.get('stationary_pct', 0):.1f}%",
              hint="time < 0.05 m/s"),
        _stat("Cum. rotation",
              f"{t.get('cumulative_rotation_deg', 0):.0f}°"),
    ]

    rows = [
        ("Pose samples", _fmt(t["pose_count"])),
        ("Effective rate", _fmt(t.get("effective_rate_hz"), ".2f") + " Hz"),
        ("Tortuosity", _fmt(t.get("tortuosity"), ".3f")),
        ("BBox extents (x,y,z)",
         _fmt(t.get("bbox_extents"), ".2f") + " m"),
        ("BBox volume", _fmt(t.get("bbox_volume_m3"), ".3f") + " m³"),
        ("Height min / max / mean",
         f"{_fmt(t.get('height_min_m'), '.2f')} / "
         f"{_fmt(t.get('height_max_m'), '.2f')} / "
         f"{_fmt(t.get('height_mean_m'), '.2f')} m"),
        ("Speed median / P95",
         f"{_fmt(t.get('speed_median_mps'), '.3f')} / "
         f"{_fmt(t.get('speed_p95_mps'), '.3f')} m/s"),
        ("Accel mean / max",
         f"{_fmt(t.get('accel_mean_mps2'), '.3f')} / "
         f"{_fmt(t.get('accel_max_mps2'), '.3f')} m/s²"),
        ("Yaw / Pitch / Roll rate (median)",
         f"{_fmt(t.get('yaw_rate_deg_per_s'), '.2f')} / "
         f"{_fmt(t.get('pitch_rate_deg_per_s'), '.2f')} / "
         f"{_fmt(t.get('roll_rate_deg_per_s'), '.2f')} °/s"),
        ("Stationary duration",
         _fmt(t.get("stationary_duration_s"), ".1f") + " s"),
    ]

    return f"""
{_stat_row(stats, cols=4)}
<div class='chart chart--lg'>
  <div class='chart-cap'>Top-down path (xz plane)</div>
  {figs['trajectory_topdown']}
</div>
<div class='chart'>
  <div class='chart-cap'>Height (Y) over time</div>
  {figs['height_time']}
</div>
<div class='chart-grid chart-grid--2'>
  <div class='chart'>
    <div class='chart-cap'>Speed over time</div>
    {figs['speed_time']}
  </div>
  <div class='chart'>
    <div class='chart-cap'>Speed distribution</div>
    {figs['speed_hist']}
  </div>
</div>
{_more("All trajectory metrics", _kv(rows))}
"""


# ----------------------------------- IMU -------------------------------------

def _section_imu(metrics: dict, figs: dict) -> str:
    imu = metrics["imu"]
    if imu is None:
        return _empty("No IMU samples.")
    cfg = metrics.get("config")

    grav_dev = imu.get("gravity_deviation", 0)
    grav_max = cfg.imu_gravity_max_dev if cfg else 0.5
    grav_tone = "good" if grav_dev < grav_max else "warn"

    stats = [
        _stat("Samples", _fmt(imu['sample_count'])),
        _stat("Rate", f"{imu.get('effective_rate_hz') or 0:.0f}", hint="Hz"),
        _stat("|gravity|",
              f"{imu.get('gravity_magnitude') or 0:.2f}",
              hint=f"Δ {grav_dev:.2f} from 9.81",
              tone=grav_tone),
        _stat("|accel| max",
              f"{imu.get('accel_mag_max') or 0:.1f}",
              hint="m/s²"),
        _stat("|gyro| max",
              f"{imu.get('gyro_mag_max') or 0:.2f}",
              hint="rad/s"),
        _stat("Jolts", _fmt(imu.get('jolt_count')),
              hint="|a| > 20 m/s²"),
        _stat("Motion",
              f"{imu.get('motion_duration_s') or 0:.0f}", hint="seconds"),
        _stat("Still",
              f"{imu.get('still_duration_s') or 0:.0f}", hint="seconds"),
    ]

    rows = [
        ("Rate jitter (std)",
         _fmt(imu.get("rate_jitter_ms"), ".2f") + " ms"),
        ("Accel axis mean (x,y,z)",
         _fmt(imu.get("accel_axis_mean"), ".3f")),
        ("Accel axis std (x,y,z)",
         _fmt(imu.get("accel_axis_std"), ".3f")),
        ("Accel axis min / max",
         _fmt(imu.get("accel_axis_min"), ".3f") + " · " +
         _fmt(imu.get("accel_axis_max"), ".3f")),
        ("|accel| mean / std / P95",
         f"{_fmt(imu.get('accel_mag_mean'), '.3f')} / "
         f"{_fmt(imu.get('accel_mag_std'), '.3f')} / "
         f"{_fmt(imu.get('accel_mag_p95'), '.3f')} m/s²"),
        ("Gyro axis mean (x,y,z)",
         _fmt(imu.get("gyro_axis_mean"), ".3f")),
        ("Gyro axis std (x,y,z)",
         _fmt(imu.get("gyro_axis_std"), ".3f")),
        ("Gyro axis min / max",
         _fmt(imu.get("gyro_axis_min"), ".3f") + " · " +
         _fmt(imu.get("gyro_axis_max"), ".3f")),
        ("|gyro| mean / std / P95",
         f"{_fmt(imu.get('gyro_mag_mean'), '.3f')} / "
         f"{_fmt(imu.get('gyro_mag_std'), '.3f')} / "
         f"{_fmt(imu.get('gyro_mag_p95'), '.3f')} rad/s"),
        ("Gravity vector",
         _fmt(imu.get("gravity_vector"), ".3f")),
        ("High-rotation events (>2 rad/s)",
         _fmt(imu.get("high_rotation_events"))),
    ]

    return f"""
{_stat_row(stats, cols=4)}
<div class='chart chart--lg'>
  <div class='chart-cap'>Acceleration &amp; angular-velocity magnitudes</div>
  {figs['imu_mag']}
</div>
{_more("All IMU metrics", _kv(rows))}
"""


# ---------------------------------- Depth ------------------------------------

def _section_depth(metrics: dict, figs: dict) -> str:
    depth = metrics["depth"]
    if depth is None:
        return _empty("No depth stream.")
    cfg = metrics.get("config")
    valid_mean = depth.get("valid_pct_mean") or 0
    valid_tone = _tone(
        valid_mean, cfg.depth_valid_thresholds if cfg else (80.0, 50.0),
    )
    pcts = depth.get("depth_percentiles_m") or {}

    stats = [
        _stat("Frames", _fmt(depth['frame_count'])),
        _stat("FPS", _fmt(depth.get("effective_fps"), ".1f")),
        _stat("Valid mean", f"{valid_mean:.1f}%",
              hint="per-frame", tone=valid_tone),
        _stat("Empty frames", _fmt(depth.get('empty_frame_count'))),
        _stat("Min", f"{depth.get('global_min_m') or 0:.2f}", hint="metres"),
        _stat("Median", f"{(pcts.get('p50') or 0):.2f}", hint="metres"),
        _stat("P95", f"{(pcts.get('p95') or 0):.2f}", hint="metres"),
        _stat("Max", f"{depth.get('global_max_m') or 0:.2f}", hint="metres"),
    ]

    rows = [
        ("Sampled frames", _fmt(depth.get("sampled_frames"))),
        ("Median dt", _fmt(depth.get("median_dt_ms"), ".2f") + " ms"),
        ("Valid pixels min / max / std",
         f"{_fmt(depth.get('valid_pct_min'), '.2f')} / "
         f"{_fmt(depth.get('valid_pct_max'), '.2f')} / "
         f"{_fmt(depth.get('valid_pct_std'), '.2f')} %"),
        ("P5 / P50 / P95 depth",
         f"{_fmt(pcts.get('p5'), '.2f')} / "
         f"{_fmt(pcts.get('p50'), '.2f')} / "
         f"{_fmt(pcts.get('p95'), '.2f')} m"),
    ]
    bins = depth.get("depth_hist_pct") or {}
    for b, v in bins.items():
        rows.append((f"Range %  {b}", _fmt(v, '.2f') + " %"))

    return f"""
{_stat_row(stats, cols=4)}
<div class='chart-grid chart-grid--2'>
  <div class='chart'>
    <div class='chart-cap'>Valid-pixel % per frame</div>
    {figs['depth_valid']}
  </div>
  <div class='chart'>
    <div class='chart-cap'>Global depth distribution</div>
    {figs['depth_hist']}
  </div>
</div>
{_more("All depth metrics", _kv(rows))}
"""


# ---------------------------------- Hands ------------------------------------

def _section_hands(metrics: dict, figs: dict) -> str:
    h = metrics["hands"]
    if h is None:
        return _empty(
            "No hand-pose data buffered. Call "
            "<code>session.add_hand_pose(frame.index, hands)</code> "
            "during your detection loop, then re-run Evaluate."
        )
    cfg = metrics.get("config")
    any_pct = h.get("frames_with_any_hand_pct") or 0
    one_plus_pct = h.get("frames_with_1plus_hand_pct") or 0
    two_exact_pct = h.get("frames_with_2_hands_pct") or 0
    more_pct = h.get("frames_with_more_hands_pct") or 0

    tone_any  = _tone(any_pct, cfg.hand_any_thresholds if cfg else (70.0, 30.0))
    tone_1plus = _tone(one_plus_pct,
                       cfg.hand_1plus_thresholds if cfg else (40.0, 15.0))
    tone_2 = _tone(two_exact_pct,
                   cfg.hand_2_thresholds if cfg else (30.0, 10.0))

    stats = [
        _stat("Backend", html.escape(h.get('backend', '—'))),
        _stat("Any hand", f"{any_pct:.1f}%", tone=tone_any,
              hint="of all frames"),
        _stat("≥1 hand", f"{one_plus_pct:.1f}%", tone=tone_1plus,
              hint="at least one detection"),
        _stat("2 hands", f"{two_exact_pct:.1f}%", tone=tone_2,
              hint="exactly two detections"),
        _stat(">2 hands", f"{more_pct:.1f}%",
              hint="strictly more than two"),
        _stat("Left", f"{h.get('left_detection_pct', 0):.1f}%"),
        _stat("Right", f"{h.get('right_detection_pct', 0):.1f}%"),
        _stat("Both", f"{h.get('both_hands_pct', 0):.1f}%"),
    ]

    rows = [
        ("Frames total", _fmt(h["frames_total"])),
        ("Frames with any hand",
         f"{_fmt(h['frames_with_any_hand'])} ({h['frames_with_any_hand_pct']:.2f}%)"),
        ("≥1 / =2 / >2 hand frames",
         f"{_fmt(h['frames_with_1plus_hand'])} / "
         f"{_fmt(h['frames_with_2_hands'])} / "
         f"{_fmt(h['frames_with_more_hands'])}"),
        ("Left conf mean / P10/P50/P90",
         f"{_fmt(h.get('left_conf_mean'), '.3f')} / "
         f"{_fmt(h.get('left_conf_p10'), '.3f')} / "
         f"{_fmt(h.get('left_conf_p50'), '.3f')} / "
         f"{_fmt(h.get('left_conf_p90'), '.3f')}"),
        ("Right conf mean / P10/P50/P90",
         f"{_fmt(h.get('right_conf_mean'), '.3f')} / "
         f"{_fmt(h.get('right_conf_p10'), '.3f')} / "
         f"{_fmt(h.get('right_conf_p50'), '.3f')} / "
         f"{_fmt(h.get('right_conf_p90'), '.3f')}"),
        ("3D keypoints", _fmt(h.get("has_3d"))),
        ("MANO frames",
         _fmt(h.get("mano_frames_pct"), '.2f') + " %"
         if h.get("mano_frames_pct") is not None else "n/a"),
        ("2D kpts inside RGB frame",
         _fmt(h.get("kpts_in_frame_pct"), '.2f') + " %"
         if h.get("kpts_in_frame_pct") is not None else "n/a"),
        ("Left wrist mean depth",
         _fmt(h.get("left_wrist_depth_mean_m"), '.3f') + " m"),
        ("Right wrist mean depth",
         _fmt(h.get("right_wrist_depth_mean_m"), '.3f') + " m"),
        ("Palm width mean",
         _fmt(h.get("palm_width_mean_m"), '.3f') + " m"),
        ("Grip closure mean",
         _fmt(h.get("grip_closure_mean_m"), '.3f') + " m"),
    ]

    def _wrist_row(label: str, t: dict) -> tuple[str, str]:
        if not t:
            return label, "—"
        return label, (
            f"length {_fmt(t.get('length_m'), '.2f')} m · "
            f"mean {_fmt(t.get('speed_mean_mps'), '.3f')} m/s · "
            f"max {_fmt(t.get('speed_max_mps'), '.3f')} m/s"
        )
    rows.append(_wrist_row("Left wrist motion (camera)", h.get("left_wrist_track", {})))
    rows.append(_wrist_row("Right wrist motion (camera)", h.get("right_wrist_track", {})))

    return f"""
{_stat_row(stats, cols=4)}
<div class='chart-grid chart-grid--2'>
  <div class='chart'>
    <div class='chart-cap'>Hands per frame</div>
    {figs['hand_pie']}
  </div>
  <div class='chart'>
    <div class='chart-cap'>Detection confidence</div>
    {figs['hand_confidence']}
  </div>
</div>
<div class='chart'>
  <div class='chart-cap'>Detection per frame (bars = detected)</div>
  {figs['hand_timeline']}
</div>
{_more("All hand metrics", _kv(rows))}
"""


# ----------------------------------- Sync ------------------------------------

def _section_sync(metrics: dict, figs: dict) -> str:
    sync = metrics["sync"]
    cfg = metrics.get("config")
    sync_thr = cfg.sync_thresholds if cfg else (90.0, 70.0)

    def _stat_for(label, s):
        if s is None:
            return _stat(label, "—")
        pct = s["within_50ms_pct"]
        return _stat(
            label, f"{pct:.1f}%", hint="within 50 ms", tone=_tone(pct, sync_thr),
        )

    stats = [
        _stat_for("RGB ↔ Depth", sync.get("rgb_vs_depth")),
        _stat_for("RGB ↔ Pose", sync.get("rgb_vs_pose")),
        _stat_for("RGB ↔ IMU", sync.get("rgb_vs_imu")),
    ]

    rows: list[tuple[str, str]] = []
    for key, label in (
        ("rgb_vs_depth", "RGB ↔ Depth"),
        ("rgb_vs_pose", "RGB ↔ Pose"),
        ("rgb_vs_imu",  "RGB ↔ IMU"),
    ):
        s = sync.get(key)
        if s is None:
            rows.append((label, "no data"))
            continue
        rows.append((label,
                     f"median {_fmt(s['median_ms'], '.1f')} ms · "
                     f"P95 {_fmt(s['p95_ms'], '.1f')} ms · "
                     f"max {_fmt(s['max_ms'], '.1f')} ms · "
                     f"≤50 ms: {_fmt(s['within_50ms_pct'], '.1f')}% · "
                     f"≤100 ms: {_fmt(s['within_100ms_pct'], '.1f')}%"))

    return f"""
{_stat_row(stats, cols=3)}
<div class='chart-grid chart-grid--2'>
  <div class='chart'>
    <div class='chart-cap'>RGB ↔ Depth offset</div>
    {figs['sync_rgb_depth']}
  </div>
  <div class='chart'>
    <div class='chart-cap'>RGB ↔ Pose offset</div>
    {figs['sync_rgb_pose']}
  </div>
</div>
{_more("All sync metrics", _kv(rows))}
"""


# ---------------------------------- 3D map -----------------------------------

def _section_map(metrics: dict) -> str:
    mesh = metrics["mesh"]
    pc = metrics["point_cloud"]
    if mesh is None and pc is None:
        return _empty("No mesh or point-cloud topic.")

    stats: list[str] = []
    if mesh is not None:
        stats += [
            _stat("Vertices", _fmt(mesh['vertex_count'])),
            _stat("Faces", _fmt(mesh['face_count'])),
            _stat("Surface",
                  f"{mesh.get('surface_area_m2') or 0:.1f}", hint="m²"),
            _stat("BBox vol",
                  f"{mesh.get('bbox_volume_m3') or 0:.1f}", hint="m³"),
        ]
    if pc is not None:
        stats += [
            _stat("Points", _fmt(pc['point_count'])),
            _stat("Density",
                  f"{(pc.get('density_pts_per_m3') or 0):.0f}", hint="pts/m³"),
        ]

    rows: list[tuple[str, str]] = []
    if mesh is not None:
        rows += [
            ("Mesh vertices", _fmt(mesh["vertex_count"])),
            ("Mesh faces", _fmt(mesh["face_count"])),
            ("BBox extents",
             _fmt(mesh.get("bbox_extents"), '.2f') + " m"),
            ("BBox volume",
             _fmt(mesh.get("bbox_volume_m3"), '.3f') + " m³"),
            ("Surface area",
             _fmt(mesh.get("surface_area_m2"), '.3f') + " m²"),
            ("Edge length mean",
             _fmt(mesh.get("edge_length_mean_m"), '.4f') + " m"),
            ("Edge length P5 / P95",
             f"{_fmt(mesh.get('edge_length_p5_m'), '.4f')} / "
             f"{_fmt(mesh.get('edge_length_p95_m'), '.4f')} m"),
            ("Color coverage",
             _fmt(mesh.get("color_coverage_pct"), '.2f') + " %"
             if mesh.get("color_coverage_pct") is not None else "n/a"),
        ]
    if pc is not None:
        rows += [
            ("Point-cloud points", _fmt(pc["point_count"])),
            ("PC BBox extents",
             _fmt(pc.get("bbox_extents"), '.2f') + " m"),
            ("PC density",
             _fmt(pc.get("density_pts_per_m3"), '.1f') + " pts/m³"),
            ("PC color coverage",
             _fmt(pc.get("color_coverage_pct"), '.2f') + " %"
             if pc.get("color_coverage_pct") is not None else "n/a"),
        ]

    return f"""
{_stat_row(stats, cols=4)}
<p class='lead'>
  3D map metrics summarise the SLAM artefacts that ship with the recording
  (<code>/map/mesh</code> and <code>/map/point_cloud</code>).
</p>
{_more("All map metrics", _kv(rows))}
"""


# --------------------------------- Coverage ----------------------------------

def _section_coverage(figs: dict) -> str:
    return f"""
<p class='lead'>Which streams are live at each second of the recording.</p>
<div class='chart chart--lg'>
  {figs['coverage']}
</div>
"""


# --------------------------- Technical reference -----------------------------

def _section_technical(metrics: dict) -> str:
    blocks: list[str] = []
    blocks.append(_block_recording(metrics))
    blocks.append(_block_rgb(metrics))
    blocks.append(_block_intrinsics(metrics))
    blocks.append(_block_tf(metrics))
    blocks.append(_block_tracking(metrics))
    blocks.append(_block_skeleton(metrics))
    blocks.append(_block_streamed_rgb(metrics))
    blocks.append(_block_eval_config(metrics))
    body = "\n".join(b for b in blocks if b)
    return (
        "<p class='lead'>"
        "Reference tables — file metadata, topic counts, intrinsics, "
        "transforms, and other static facts about the recording."
        "</p>"
        f"<div class='tech-grid'>{body}</div>"
    )


def _block_recording(metrics: dict) -> str:
    rec = metrics["recording"]
    rows = [
        ("Path", html.escape(rec["path"])),
        ("Filename", html.escape(rec["filename"])),
        ("File size", f"{rec['size_mb']:.2f} MB"),
        ("Duration", f"{rec['duration_s']:.2f} s ({rec['duration_hms']})"),
        ("Start (ISO)", html.escape(rec.get("start_iso") or "—")),
        ("End (ISO)", html.escape(rec.get("end_iso") or "—")),
        ("Weekday", html.escape(rec.get("weekday") or "—")),
        ("Total messages", _fmt(rec["message_count"])),
        ("Topics present", _fmt(rec["topics_present_count"])),
        ("Missing reference topics",
         html.escape(", ".join(rec["missing_reference_topics"]) or "none")),
    ]
    topics = rec["topic_counts"]
    topic_rows = "".join(
        f"<tr><td>{html.escape(t)}</td><td class='num'>{c:,}</td></tr>"
        for t, c in topics.items()
    )
    return (
        "<div class='tech-block'>"
        "<h4>Recording</h4>"
        f"{_kv(rows)}"
        f"<h5>Topic message counts</h5>"
        f"<table class='topics'><thead><tr><th>topic</th><th>count</th></tr>"
        f"</thead><tbody>{topic_rows}</tbody></table>"
        "</div>"
    )


def _block_rgb(metrics: dict) -> str:
    rgb = metrics["rgb"]
    rows = [
        ("Frames", _fmt(rgb["frame_count"])),
        ("Effective FPS", _fmt(rgb.get("effective_fps"), ".2f")),
        ("Median dt", _fmt(rgb.get("median_dt_ms"), ".2f") + " ms"),
        ("Min / max dt",
         _fmt(rgb.get("min_dt_ms"), ".2f") + " / " +
         _fmt(rgb.get("max_dt_ms"), ".2f") + " ms"),
        ("dt std", _fmt(rgb.get("dt_std_ms"), ".2f") + " ms"),
        ("Frame gaps (>2× median)", _fmt(rgb.get("gap_count"))),
    ]
    return f"<div class='tech-block'><h4>RGB stream</h4>{_kv(rows)}</div>"


def _block_intrinsics(metrics: dict) -> str:
    rgb_intr = (metrics["rgb"] or {}).get("intrinsics")
    depth_intr = (metrics["depth"] or {}).get("intrinsics") if metrics["depth"] else None
    parts = []
    if rgb_intr:
        parts.append("<h5>RGB</h5>" + _intrinsics_table(rgb_intr))
    if depth_intr:
        parts.append("<h5>Depth</h5>" + _intrinsics_table(depth_intr))
    if not parts:
        return ""
    return "<div class='tech-block'><h4>Camera intrinsics</h4>" + "".join(parts) + "</div>"


def _intrinsics_table(intr: dict) -> str:
    rows = [
        ("Resolution", f"{intr['width']} × {intr['height']}"),
        ("fx, fy", f"{intr['fx']:.2f}, {intr['fy']:.2f}"),
        ("cx, cy", f"{intr['cx']:.2f}, {intr['cy']:.2f}"),
        ("fx / fy", _fmt(intr.get("fx_over_fy"), ".4f")),
        ("FOV (h × v)",
         f"{_fmt(intr.get('fov_x_deg'), '.1f')}° × "
         f"{_fmt(intr.get('fov_y_deg'), '.1f')}°"),
        ("Aspect ratio", _fmt(intr.get("aspect_ratio"), ".4f")),
        ("Principal offset px",
         f"{_fmt(intr['principal_offset_px'][0], '.2f')}, "
         f"{_fmt(intr['principal_offset_px'][1], '.2f')}"),
        ("Principal offset %",
         f"{_fmt(intr['principal_offset_pct'][0], '.2f')}%, "
         f"{_fmt(intr['principal_offset_pct'][1], '.2f')}%"),
        ("Distortion model", html.escape(intr.get("distortion_model") or "—")),
        ("Distortion coeffs", _fmt(intr.get("distortion"), ".4f")),
    ]
    return _kv(rows)


def _block_tf(metrics: dict) -> str:
    tf = metrics["tf"]
    if tf is None:
        return ""
    rows = [
        ("Total messages", _fmt(tf["message_count"])),
        ("Unique pairs", _fmt(tf["unique_pair_count"])),
    ]
    pair_rows = "".join(
        f"<tr><td>{html.escape(p['parent'])} → {html.escape(p['child'])}</td>"
        f"<td class='num'>{p['count']:,}</td>"
        f"<td class='num'>{p['rate_hz']:.2f} Hz</td></tr>"
        for p in tf["pairs"]
    )
    pair_table = (
        "<table class='topics'><thead><tr><th>pair</th><th>count</th>"
        f"<th>rate</th></tr></thead><tbody>{pair_rows}</tbody></table>"
    )
    return f"<div class='tech-block'><h4>TF transforms</h4>{_kv(rows)}{pair_table}</div>"


def _block_tracking(metrics: dict) -> str:
    t = metrics["tracking_state"]
    if t is None:
        return ""
    rows = [("Total messages", _fmt(t["message_count"]))]
    for state, pct in t["state_pct"].items():
        rows.append((html.escape(str(state)),
                     f"{_fmt(pct, '.1f')}% "
                     f"({_fmt(t['state_counts'].get(state))})"))
    return f"<div class='tech-block'><h4>Tracking state</h4>{_kv(rows)}</div>"


def _block_skeleton(metrics: dict) -> str:
    s = metrics["skeleton"]
    if s is None:
        return ""
    rows = [
        ("Frames", _fmt(s["frame_count"])),
        ("Left elbow mean",
         _fmt(s.get("elbow_left_mean_deg"), ".1f") + "°"),
        ("Right elbow mean",
         _fmt(s.get("elbow_right_mean_deg"), ".1f") + "°"),
        ("Left arm reach mean",
         _fmt(s.get("reach_left_mean_m"), ".3f") + " m"),
        ("Right arm reach mean",
         _fmt(s.get("reach_right_mean_m"), ".3f") + " m"),
        ("Head height mean",
         _fmt(s.get("head_height_mean_m"), ".3f") + " m"),
    ]
    vis = s.get("joint_visibility_pct") or {}
    vis_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{_fmt(v, '.1f')} %</td></tr>"
        for k, v in vis.items()
    )
    vis_table = f"<table class='topics'>{vis_rows}</table>"
    return (
        "<div class='tech-block'><h4>Skeleton</h4>"
        f"{_kv(rows)}<h5>Joint visibility</h5>{vis_table}</div>"
    )


def _block_eval_config(metrics: dict) -> str:
    cfg = metrics.get("config")
    if cfg is None:
        return ""

    def _thr(t: tuple[float, float]) -> str:
        return f"good ≥ {t[0]:g} · ok ≥ {t[1]:g}"

    def _pw(target: float, weight: float) -> str:
        if weight <= 0:
            return f"disabled (weight 0)"
        return f"target {target:g}%, weight {weight:g}"

    def _req(flag: bool, penalty: float) -> str:
        if not flag:
            return "optional (no penalty when missing)"
        return f"-{penalty:g} when missing"

    rows = [
        ("Health color", _thr(cfg.health_thresholds)),
        ("Depth valid color", _thr(cfg.depth_valid_thresholds)),
        ("Sync color", _thr(cfg.sync_thresholds)),
        ("Any-hand color", _thr(cfg.hand_any_thresholds)),
        ("≥1-hand color", _thr(cfg.hand_1plus_thresholds)),
        ("Exactly-2-hand color", _thr(cfg.hand_2_thresholds)),
        ("IMU gravity Δ max", f"{cfg.imu_gravity_max_dev:g} m/s²"),
        ("RGB gap cap", f"{cfg.rgb_gap_max_penalty:g}"),
        ("Depth penalty",
         _pw(cfg.depth_valid_target, cfg.depth_valid_weight)),
        ("Depth required",
         _req(cfg.depth_required, cfg.depth_missing_penalty)),
        ("Pose missing", f"-{cfg.pose_missing_penalty:g}"),
        ("IMU required",
         _req(cfg.imu_required, cfg.imu_missing_penalty)),
        ("Sync penalty (per pair)",
         _pw(cfg.sync_target, cfg.sync_weight)),
        ("Any-hand penalty",
         _pw(cfg.hand_any_target, cfg.hand_any_weight)),
        ("≥1-hand penalty",
         _pw(cfg.hand_1plus_target, cfg.hand_1plus_weight)),
        ("Exactly-2-hand penalty",
         _pw(cfg.hand_2_target, cfg.hand_2_weight)),
        ("Hand-missing penalty", f"-{cfg.hand_missing_penalty:g}"),
    ]
    return (
        "<div class='tech-block'>"
        "<h4>Health score config</h4>"
        "<p class='lead' style='margin-bottom:8px;font-size:12px'>"
        "Override via <code>EvaluateConfig(...)</code>; set any weight to 0 "
        "to disable that check."
        "</p>"
        f"{_kv(rows)}"
        "</div>"
    )


def _block_streamed_rgb(metrics: dict) -> str:
    s = metrics["streamed_rgb"]
    if s is None:
        return ""
    rows = [
        ("Active writer", _fmt(s.get("active"))),
        ("Temp path", html.escape(s.get("tmp_path") or "—")),
        ("Temp size",
         f"{s.get('tmp_size_bytes', 0) / (1024 * 1024):.2f} MB"),
        ("Resolution",
         f"{s.get('width')} × {s.get('height')}" if s.get("width") else "—"),
        ("FPS", _fmt(s.get("fps"), ".2f")),
        ("Mid-frame thumbnail captured", _fmt(s.get("has_thumbnail"))),
    ]
    return f"<div class='tech-block'><h4>Streamed rgb.mp4</h4>{_kv(rows)}</div>"


# ------------------------------- shared bits ---------------------------------

def _empty(message: str) -> str:
    return f"<p class='empty'>{message}</p>"


def _more(label: str, body: str) -> str:
    return (
        "<details class='more'>"
        f"<summary>{html.escape(label)} <span class='arrow'>→</span></summary>"
        f"<div class='more-body'>{body}</div>"
        "</details>"
    )


# ----------------------------------- CSS -------------------------------------

_CSS = r"""
:root {
  color-scheme: dark;
  --bg: #0d1117;
  --surface: #161b22;
  --surface-2: #1c232c;
  --line: rgba(240,246,252,0.06);
  --line-strong: rgba(240,246,252,0.10);

  --text: #c9d1d9;
  --text-strong: #f0f6fc;
  --text-dim: #8b949e;
  --text-dim-2: #6e7681;

  --accent: #58a6ff;
  --good: #3fb950;
  --warn: #d29922;
  --bad:  #f85149;

  --sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Inter",
          "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, "Cascadia Code", monospace;

  --shell-max: 1400px;
  --nav-w: 200px;
  --gap: 28px;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); }
body {
  font-family: var(--sans);
  color: var(--text);
  font-size: 14px;
  line-height: 1.55;
  letter-spacing: -0.005em;
  font-feature-settings: "ss01", "cv11";
  -webkit-font-smoothing: antialiased;
}

/* Header */
.shell {
  max-width: var(--shell-max); margin: 0 auto;
  display: grid;
  grid-template-columns: var(--nav-w) 1fr;
  gap: 56px;
  padding: 40px 36px 80px;
}
.banner {
  grid-column: 1 / -1;
  padding-bottom: 24px;
  margin-bottom: 12px;
  border-bottom: 1px solid var(--line);
  display: flex; align-items: baseline; flex-wrap: wrap; gap: 14px;
}
.brand {
  font-family: var(--mono);
  font-size: 11px; font-weight: 600;
  color: var(--accent);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.banner h1 {
  font-size: 22px; font-weight: 600; margin: 0;
  color: var(--text-strong); letter-spacing: -0.01em;
}
.crumb { color: var(--text-dim); font-size: 13px; }

/* Sidebar nav (sticky) */
nav.toc {
  position: sticky; top: 32px;
  align-self: start;
  display: flex; flex-direction: column; gap: 2px;
  font-size: 13px;
}
nav.toc a {
  color: var(--text-dim);
  text-decoration: none;
  padding: 7px 12px;
  border-radius: 6px;
  display: block;
  border-left: 2px solid transparent;
  margin-left: -14px; padding-left: 14px;
  transition: color 0.12s ease;
}
nav.toc a:hover { color: var(--text-strong); }
nav.toc a.active {
  color: var(--text-strong);
  border-left-color: var(--accent);
}

/* Main column */
main { min-width: 0; }
section.block {
  padding: 56px 0;
  border-bottom: 1px solid var(--line);
}
section.block:first-child { padding-top: 12px; }
section.block:last-child { border-bottom: none; padding-bottom: 24px; }

.block-head { margin-bottom: 24px; }
.block-head h2 {
  font-size: 26px; font-weight: 600; margin: 0;
  color: var(--text-strong); letter-spacing: -0.02em;
}
.block-head .block-sub {
  color: var(--text-dim); font-size: 14px; margin-top: 4px;
}
p.lead {
  color: var(--text-dim); font-size: 14px;
  max-width: 60ch; margin: 0 0 24px;
}

/* Summary */
.summary {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 48px;
  align-items: start;
}
.summary-left .thumb {
  width: 100%; aspect-ratio: 4/3; object-fit: cover;
  border-radius: 8px; display: block;
  background: var(--surface);
}
.summary-left .thumb--empty {
  display: flex; align-items: center; justify-content: center;
  color: var(--text-dim-2); font-size: 12px;
}
.summary-left .filename {
  font-family: var(--mono); font-size: 11px;
  color: var(--text-dim); word-break: break-all;
  margin: 14px 0 4px; line-height: 1.5;
}
.summary-left .filesub { color: var(--text-dim-2); font-size: 12px; margin: 0; }

.health {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 28px;
  align-items: center;
  padding-bottom: 28px;
  margin-bottom: 28px;
  border-bottom: 1px solid var(--line);
}
.health-num { display: flex; align-items: baseline; gap: 4px; }
.health-score {
  font-size: 72px; font-weight: 700; line-height: 1;
  letter-spacing: -0.05em;
  font-feature-settings: "tnum", "lnum";
}
.health-suffix { font-size: 14px; color: var(--text-dim); font-weight: 500; }
.health-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em;
  font-weight: 600; margin-bottom: 6px;
}
.health-notes {
  margin: 0; padding-left: 18px;
  color: var(--text); font-size: 13px;
  list-style: disc; list-style-position: outside;
}
.health-notes li { margin: 2px 0; }
.health-notes li.good { color: var(--good); list-style: none; margin-left: -18px; }
.health-notes li.good::before {
  content: "✓ "; font-weight: 600; margin-right: 4px;
}

/* Stat row */
.stat-row {
  display: grid; gap: 24px 32px;
  margin-bottom: 32px;
}
.stat-row--3 { grid-template-columns: repeat(3, 1fr); }
.stat-row--4 { grid-template-columns: repeat(4, 1fr); }
.stat {
  min-width: 0;
}
.stat-label {
  font-size: 11px; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.08em;
  font-weight: 500;
  margin-bottom: 4px;
}
.stat-value {
  font-size: 26px; font-weight: 600;
  color: var(--text-strong); letter-spacing: -0.02em;
  line-height: 1.1;
  font-feature-settings: "tnum", "lnum";
}
.stat-hint {
  font-size: 11px; color: var(--text-dim-2);
  margin-top: 4px;
}
.stat--good .stat-value, .stat--good.health-label, .stat--good .health-score { color: var(--good); }
.stat--warn .stat-value, .stat--warn.health-label, .stat--warn .health-score { color: var(--warn); }
.stat--bad  .stat-value, .stat--bad.health-label,  .stat--bad  .health-score { color: var(--bad); }

/* Charts */
.chart {
  margin: 24px 0;
}
.chart--lg { margin: 28px 0; }
.chart-cap {
  font-size: 12px;
  color: var(--text-dim);
  margin: 0 0 4px;
  letter-spacing: -0.005em;
  font-weight: 500;
}
.chart-grid {
  display: grid; gap: 24px;
  margin: 24px 0;
}
.chart-grid--2 { grid-template-columns: 1fr 1fr; }
.chart .js-plotly-plot, .chart-grid .js-plotly-plot {
  background: var(--surface);
  border-radius: 6px;
}

/* Disclosures */
details.more { margin-top: 16px; }
details.more > summary {
  cursor: pointer;
  list-style: none;
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 13px; font-weight: 500;
  color: var(--accent);
  padding: 6px 0;
  user-select: none;
}
details.more > summary::-webkit-details-marker { display: none; }
details.more > summary:hover { color: var(--text-strong); }
details.more > summary .arrow { transition: transform 0.15s ease; display: inline-block; }
details.more[open] > summary .arrow { transform: rotate(90deg); }
details.more .more-body {
  margin-top: 14px;
  padding: 16px 20px;
  background: var(--surface);
  border-radius: 8px;
  border: 1px solid var(--line);
}

/* Tables */
table.kv {
  width: 100%; border-collapse: collapse;
}
table.kv th, table.kv td {
  text-align: left; padding: 8px 0;
  border-bottom: 1px solid var(--line);
  vertical-align: top; font-size: 13px;
}
table.kv th {
  font-weight: 400; color: var(--text-dim);
  width: 40%; padding-right: 24px;
}
table.kv td {
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
table.kv tr:last-child th, table.kv tr:last-child td { border-bottom: none; }

table.topics {
  width: 100%; border-collapse: collapse; margin-top: 8px;
  font-size: 12px;
  font-family: var(--mono);
}
table.topics th, table.topics td {
  text-align: left; padding: 6px 0;
  border-bottom: 1px solid var(--line);
}
table.topics thead th {
  color: var(--text-dim-2); font-weight: 500;
  text-transform: uppercase; font-size: 10.5px;
  letter-spacing: 0.06em;
  font-family: var(--sans);
}
table.topics td.num, table.topics th:not(:first-child) { text-align: right; }
table.topics td { color: var(--text); font-variant-numeric: tabular-nums; }

/* Technical reference grid */
.tech-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 36px 48px;
}
.tech-block { min-width: 0; }
.tech-block h4 {
  font-size: 15px; font-weight: 600;
  color: var(--text-strong); margin: 0 0 12px;
  letter-spacing: -0.005em;
}
.tech-block h5 {
  font-size: 11px; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.08em;
  font-weight: 500;
  margin: 18px 0 6px;
}

.empty {
  color: var(--text-dim); font-style: italic;
  font-size: 14px; margin: 8px 0;
}
.empty code {
  font-family: var(--mono); background: var(--surface);
  padding: 1px 5px; border-radius: 3px; color: var(--text);
}

footer {
  grid-column: 1 / -1;
  padding-top: 24px; margin-top: 8px;
  border-top: 1px solid var(--line);
  color: var(--text-dim-2); font-size: 12px;
  text-align: center;
}

/* Responsive: collapse sidebar on small screens */
@media (max-width: 960px) {
  .shell {
    grid-template-columns: 1fr; gap: 32px;
    padding: 24px 20px 60px;
  }
  nav.toc {
    position: static;
    flex-direction: row; flex-wrap: wrap;
    border-bottom: 1px solid var(--line); padding-bottom: 16px;
  }
  nav.toc a { border-left: none; margin-left: 0; padding-left: 12px; }
  nav.toc a.active { border-bottom: 2px solid var(--accent); }
  .summary { grid-template-columns: 1fr; gap: 28px; }
  .health { grid-template-columns: 1fr; gap: 12px; }
  .stat-row--3, .stat-row--4 { grid-template-columns: repeat(2, 1fr); }
  .chart-grid--2 { grid-template-columns: 1fr; }
  .tech-grid { grid-template-columns: 1fr; gap: 28px; }
  .block-head h2 { font-size: 22px; }
  .health-score { font-size: 56px; }
}
"""


# Tiny client-side script to highlight the active section in the sticky nav.
_SCRIPT = r"""
(function () {
  const links = Array.from(document.querySelectorAll('nav.toc a'));
  const sections = links
    .map(a => document.getElementById(a.getAttribute('href').slice(1)))
    .filter(Boolean);
  if (!sections.length) return;
  function setActive() {
    const y = window.scrollY + 120;
    let active = sections[0];
    for (const s of sections) {
      if (s.offsetTop <= y) active = s;
    }
    links.forEach(a => a.classList.toggle(
      'active', a.getAttribute('href') === '#' + active.id
    ));
  }
  setActive();
  window.addEventListener('scroll', setActive, { passive: true });
})();
"""


def _build_html(plotlyjs: str, sections: list[tuple[str, str, str]],
                metrics: dict) -> str:
    rec = metrics["recording"]
    rgb = metrics["rgb"]
    title = f"Stera evaluation — {html.escape(rec['filename'])}"

    toc = "\n".join(
        f"<a href='#{sid}'>{html.escape(label)}</a>"
        for sid, label, _ in sections
    )

    blocks = []
    for sid, label, body in sections:
        head = (
            f"<div class='block-head'><h2>{html.escape(label)}</h2></div>"
            if sid != "summary" else ""
        )
        blocks.append(
            f"<section class='block' id='{sid}'>{head}{body}</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
<script type="text/javascript">{plotlyjs}</script>
</head>
<body>
<div class="shell">
  <header class="banner">
    <span class="brand">stera · evaluation</span>
    <h1>{html.escape(rec['filename'])}</h1>
    <span class="crumb">{rec['duration_hms']} · {rgb['frame_count']:,} frames · {rec['size_mb']:.0f} MB</span>
  </header>
  <nav class="toc">{toc}</nav>
  <main>
    {''.join(blocks)}
  </main>
  <footer>Generated by stera-sdk · Evaluate</footer>
</div>
<script type="text/javascript">{_SCRIPT}</script>
</body>
</html>
"""
