"""Interactive Plotly figures rendered from precomputed metrics.

All figures are built ready to be embedded via ``fig.to_html(include_plotlyjs=False)``.
The single Plotly.js source is inlined once in the HTML head by ``report.py``.

Titles are kept outside the figure (provided by the surrounding HTML caption)
so the chart canvas is as clean as possible.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _plotly():
    import plotly.graph_objects as go
    return go


# Palette
PRIMARY = "#58a6ff"   # GitHub blue
SUCCESS = "#3fb950"   # green
WARN    = "#d29922"   # amber
DANGER  = "#f85149"   # red
LEFT    = "#ff7b72"   # left hand
RIGHT   = "#3fb950"   # right hand
MUTED   = "#6e7681"

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.04)",
    zeroline=False,
    linecolor="rgba(255,255,255,0.08)",
    tickcolor="rgba(255,255,255,0.08)",
    tickfont=dict(color="#8b949e", size=10.5),
    title_font=dict(color="#8b949e", size=11),
)
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=56, r=20, t=12, b=44),
    font=dict(
        family='ui-sans-serif, -apple-system, BlinkMacSystemFont, '
               '"Inter", "Segoe UI", sans-serif',
        size=12,
        color="#c9d1d9",
    ),
    height=320,
    hoverlabel=dict(
        bgcolor="#161b22",
        bordercolor="rgba(255,255,255,0.12)",
        font=dict(family='ui-monospace, "SF Mono", Menlo, monospace', size=12),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#c9d1d9"),
        orientation="h",
        y=1.08, x=0,
    ),
)


def _empty_fig():
    go = _plotly()
    fig = go.Figure()
    fig.add_annotation(
        text="no data", showarrow=False,
        font=dict(size=13, color=MUTED),
    )
    fig.update_layout(**_LAYOUT)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _apply(fig, xtitle: str = "", ytitle: str = "", **extra):
    layout = {**_LAYOUT, **extra}  # extras can override _LAYOUT keys
    fig.update_layout(
        xaxis=dict(title=xtitle, **_AXIS_STYLE),
        yaxis=dict(title=ytitle, **_AXIS_STYLE),
        **layout,
    )
    return fig


# Trajectory

def fig_trajectory_topdown(traj: dict | None):
    go = _plotly()
    if traj is None:
        return _empty_fig()
    pos = np.asarray(traj["positions"])
    x, z = pos[:, 0], pos[:, 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=z, mode="lines",
        line=dict(width=2, color=PRIMARY), name="path",
        hovertemplate="x=%{x:.2f} m<br>z=%{y:.2f} m<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[x[0]], y=[z[0]], mode="markers", name="start",
        marker=dict(size=11, color=SUCCESS, symbol="circle",
                    line=dict(width=2, color="rgba(255,255,255,0.4)")),
    ))
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[z[-1]], mode="markers", name="end",
        marker=dict(size=11, color=DANGER, symbol="x-thin",
                    line=dict(width=3, color=DANGER)),
    ))
    fig.update_layout(
        xaxis=dict(title="x (m)", **_AXIS_STYLE),
        yaxis=dict(title="z (m)", scaleanchor="x", scaleratio=1, **_AXIS_STYLE),
        **{**_LAYOUT, "height": 420},
    )
    return fig


def fig_height_time(traj: dict | None):
    go = _plotly()
    if traj is None:
        return _empty_fig()
    ts = np.asarray(traj["ts_series"])
    y = np.asarray(traj["positions"])[:, 1]
    t = ts - ts[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines",
        fill="tozeroy",
        line=dict(width=1.5, color=WARN),
        fillcolor="rgba(210, 153, 34, 0.08)",
        hovertemplate="t=%{x:.1f} s<br>y=%{y:.2f} m<extra></extra>",
    ))
    return _apply(fig, "time (s)", "height (m)")


def fig_speed(traj: dict | None):
    go = _plotly()
    if traj is None or len(traj.get("speed_series", [])) == 0:
        return _empty_fig()
    speeds = np.asarray(traj["speed_series"])
    ts = np.asarray(traj["speed_ts"])
    t = ts - ts[0] if len(ts) else np.arange(len(speeds))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=speeds, mode="lines",
        line=dict(width=1.2, color=SUCCESS),
        hovertemplate="t=%{x:.1f} s<br>%{y:.2f} m/s<extra></extra>",
    ))
    return _apply(fig, "time (s)", "speed (m/s)")


def fig_speed_hist(traj: dict | None):
    go = _plotly()
    if traj is None or len(traj.get("speed_series", [])) == 0:
        return _empty_fig()
    speeds = np.asarray(traj["speed_series"])
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=speeds, nbinsx=40,
        marker=dict(color=SUCCESS, line=dict(width=0)),
        hovertemplate="%{x:.2f} m/s × %{y}<extra></extra>",
    ))
    return _apply(fig, "speed (m/s)", "frames")


# IMU

def fig_imu_mag(imu: dict | None):
    go = _plotly()
    if imu is None:
        return _empty_fig()
    ts = np.asarray(imu["ts_series"])
    t = ts - ts[0]
    accel = np.asarray(imu["accel_mag_series"])
    gyro = np.asarray(imu["gyro_mag_series"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=accel, mode="lines", name="|accel| (m/s²)",
        line=dict(width=1, color=PRIMARY),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=gyro, mode="lines", name="|gyro| (rad/s)",
        line=dict(width=1, color=WARN), yaxis="y2",
    ))
    fig.update_layout(
        xaxis=dict(title="time (s)", **_AXIS_STYLE),
        yaxis=dict(title="|accel| (m/s²)", **_AXIS_STYLE),
        yaxis2=dict(
            title="|gyro| (rad/s)", overlaying="y", side="right",
            **_AXIS_STYLE,
        ),
        **{**_LAYOUT, "height": 360},
    )
    return fig


# Depth

def fig_depth_valid_pct(depth: dict | None):
    go = _plotly()
    if depth is None or not depth.get("ts_series"):
        return _empty_fig()
    ts = np.asarray(depth["ts_series"])
    valid = np.asarray(depth["valid_pct_series"])
    t = ts - ts[0] if len(ts) else np.arange(len(valid))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=valid, mode="lines",
        line=dict(width=1.5, color=SUCCESS),
        fill="tozeroy", fillcolor="rgba(63, 185, 80, 0.08)",
        hovertemplate="t=%{x:.1f} s<br>%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="time (s)", **_AXIS_STYLE),
        yaxis=dict(title="valid pixels (%)", range=[0, 100], **_AXIS_STYLE),
        **_LAYOUT,
    )
    return fig


def fig_depth_histogram(depth: dict | None):
    go = _plotly()
    if depth is None or "global_depth_samples" not in depth:
        return _empty_fig()
    samples = np.asarray(depth["global_depth_samples"])
    if samples.size == 0:
        return _empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=60,
        marker=dict(color=PRIMARY, line=dict(width=0)),
        hovertemplate="%{x:.2f} m × %{y}<extra></extra>",
    ))
    return _apply(fig, "depth (m)", "pixels (sampled)")


# Sync

def fig_sync_hist(sync: dict, key: str, label: str):
    go = _plotly()
    s = sync.get(key) if sync else None
    if s is None or "dts_ms" not in s:
        return _empty_fig()
    dts = np.asarray(s["dts_ms"])
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=dts, nbinsx=50,
        marker=dict(color=WARN, line=dict(width=0)),
        hovertemplate="%{x:.1f} ms × %{y}<extra></extra>",
    ))
    fig.add_vline(x=50, line_dash="dash", line_color=MUTED,
                  annotation_text="50 ms",
                  annotation_font_color=MUTED,
                  annotation_font_size=10)
    return _apply(fig, "|dt| (ms)", "frames")


# Hands

def fig_hand_detection_timeline(hands: dict | None):
    go = _plotly()
    if hands is None:
        return _empty_fig()
    left = np.asarray(hands["left_valid_series"], dtype=np.float32)
    right = np.asarray(hands["right_valid_series"], dtype=np.float32)
    idx = np.arange(len(left))
    # Draw as filled bands per side
    fig = go.Figure()
    # Right hand on top row, left on bottom (visual convention)
    fig.add_trace(go.Bar(
        x=idx, y=right,
        name="right",
        marker=dict(color=RIGHT, line=dict(width=0)),
        offsetgroup="r", base=1.2,
        hovertemplate="frame %{x}<extra>right</extra>",
    ))
    fig.add_trace(go.Bar(
        x=idx, y=left,
        name="left",
        marker=dict(color=LEFT, line=dict(width=0)),
        offsetgroup="l",
        hovertemplate="frame %{x}<extra>left</extra>",
    ))
    fig.update_layout(
        bargap=0,
        xaxis=dict(title="frame index", **_AXIS_STYLE),
        yaxis=dict(
            tickvals=[0.5, 1.7], ticktext=["left", "right"],
            range=[-0.1, 2.3], **_AXIS_STYLE,
        ),
        **{**_LAYOUT, "height": 220},
        showlegend=False,
    )
    return fig


def fig_hand_confidence(hands: dict | None):
    go = _plotly()
    if hands is None:
        return _empty_fig()
    fig = go.Figure()
    left = np.asarray(hands["left_conf_series"])
    right = np.asarray(hands["right_conf_series"])
    if left.size:
        fig.add_trace(go.Histogram(
            x=left, nbinsx=30, name="left",
            marker=dict(color=LEFT, line=dict(width=0)),
            opacity=0.75,
        ))
    if right.size:
        fig.add_trace(go.Histogram(
            x=right, nbinsx=30, name="right",
            marker=dict(color=RIGHT, line=dict(width=0)),
            opacity=0.75,
        ))
    return _apply(fig, "confidence", "detections", barmode="overlay")


def fig_hand_count_pie(hands: dict | None):
    """Pie of disjoint buckets: exactly 0 / 1 / 2 / >2 hands per frame."""
    go = _plotly()
    if hands is None:
        return _empty_fig()
    counts = np.asarray(hands.get("counts_per_frame", []))
    none = int((counts == 0).sum())
    one_exact = int((counts == 1).sum())
    two_exact = int((counts == 2).sum())
    more = int((counts > 2).sum())
    fig = go.Figure(go.Pie(
        labels=["no hands", "exactly 1", "exactly 2", ">2"],
        values=[none, one_exact, two_exact, more],
        hole=0.65,
        marker=dict(
            colors=["#30363d", WARN, SUCCESS, DANGER],
            line=dict(width=0),
        ),
        textinfo="label+percent",
        textfont=dict(size=11, color="#f0f6fc"),
    ))
    fig.update_layout(
        showlegend=False,
        **{**_LAYOUT, "margin": dict(l=10, r=10, t=10, b=10)},
    )
    return fig


# Coverage timeline

def fig_coverage_timeline(metrics: dict):
    go = _plotly()
    streams: list[tuple[str, np.ndarray | None, str]] = []
    rgb = metrics.get("rgb") or {}
    streams.append(("RGB", np.asarray(rgb.get("timestamps", [])), PRIMARY))
    streams.append(("Depth", _safe_arr(metrics.get("depth"), "ts_series"), SUCCESS))
    streams.append(("Pose", _safe_arr(metrics.get("trajectory"), "ts_series"), WARN))
    streams.append(("IMU", _safe_arr(metrics.get("imu"), "ts_series"), LEFT))

    fig = go.Figure()
    ref_ts: Optional[np.ndarray] = None
    labels: list[str] = []
    for label, ts, color in streams:
        if ts is None or len(ts) == 0:
            continue
        ts = np.asarray(ts)
        if ref_ts is None:
            ref_ts = ts
        labels.append(label)
        rel = ts - ref_ts[0]
        fig.add_trace(go.Scatter(
            x=rel, y=[label] * len(rel), mode="markers",
            marker=dict(size=4, color=color, symbol="line-ns-open",
                        line=dict(width=1, color=color)),
            name=label,
            hovertemplate=f"{label} @ %{{x:.2f}} s<extra></extra>",
        ))
    return _apply(
        fig, "time (s)", "",
        showlegend=False,
        height=220,
    )


def _safe_arr(block: dict | None, key: str):
    if not block:
        return None
    return block.get(key)


# Entry point

def build_figures(metrics: dict) -> dict:
    """Return ``{name: plotly.Figure}`` for every plot we render."""
    return {
        "trajectory_topdown": fig_trajectory_topdown(metrics.get("trajectory")),
        "height_time": fig_height_time(metrics.get("trajectory")),
        "speed_time": fig_speed(metrics.get("trajectory")),
        "speed_hist": fig_speed_hist(metrics.get("trajectory")),
        "imu_mag": fig_imu_mag(metrics.get("imu")),
        "depth_valid": fig_depth_valid_pct(metrics.get("depth")),
        "depth_hist": fig_depth_histogram(metrics.get("depth")),
        "sync_rgb_depth": fig_sync_hist(
            metrics.get("sync", {}), "rgb_vs_depth", "RGB ↔ Depth",
        ),
        "sync_rgb_pose": fig_sync_hist(
            metrics.get("sync", {}), "rgb_vs_pose", "RGB ↔ Pose",
        ),
        "hand_timeline": fig_hand_detection_timeline(metrics.get("hands")),
        "hand_confidence": fig_hand_confidence(metrics.get("hands")),
        "hand_pie": fig_hand_count_pie(metrics.get("hands")),
        "coverage": fig_coverage_timeline(metrics),
    }
