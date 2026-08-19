"""Plotly time-series and compass-diagnostic figures."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .schema import (CHANNELS_BY_KEY, POWER_CHANNELS, PROPULSION_COLORS,
                     SAIL_FAULT_OK, SAIL_STATE_COLORS, TACK_COLORS, TACK_LABELS)

# The app follows the browser's light/dark preference, so chart chrome uses
# translucent greys that read correctly on either background. Font colour is
# left unset so Streamlit's own Plotly theme supplies it.
GRID = "rgba(128,128,128,0.22)"
MUTED = "rgba(128,128,128,0.95)"
ZERO = "rgba(128,128,128,0.75)"
SERIES = ["#3d8fd6", "#eb6834", "#1baf7a", "#9457c9", "#d64b8a", "#0f9bb5"]

TARGET_POINTS = 2000


def _layout(fig, height, title=None):
    # Passing title=None explicitly renders as the string "undefined" in some
    # plotly.js builds, so the key is omitted entirely when there is no title.
    fig.update_layout(
        height=height,
        **({"title": title} if title else {}),
        margin=dict(l=56, r=16, t=40 if title else 16, b=36),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        hovermode="x unified", showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickcolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickcolor=MUTED)
    return fig


def _times(values):
    """Time axis as naive UTC datetime64.

    A tz-aware pandas column yields an object array of Timestamps, which are not
    JSON-serialisable in figure output. Everything here is UTC already, so
    dropping the tz loses nothing and keeps the array a real datetime64.
    """
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]")


def _envelope(t, y, target=TARGET_POINTS):
    """Reduce a long series to min/max bands so no spike is hidden.

    Plotting every one of 375000 samples is both slow and no more informative
    than the band they occupy.
    """
    n = len(t)
    if n <= target * 2:
        return t, y, None, None

    edges = np.linspace(0, n, target + 1).astype(int)
    tc, lo, hi, mean = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        finite = seg[np.isfinite(seg)]
        tc.append(t[(a + b) // 2])
        if finite.size:
            lo.append(finite.min()); hi.append(finite.max()); mean.append(finite.mean())
        else:
            lo.append(np.nan); hi.append(np.nan); mean.append(np.nan)
    return np.array(tc), np.array(mean), np.array(lo), np.array(hi)


def _break_wraps(t, y, threshold=180.0):
    """Insert gaps where a circular series crosses 0/360.

    Without this the trace draws a vertical streak across the whole panel every
    time the heading passes north.
    """
    y = y.astype(float).copy()
    jump = np.abs(np.diff(y)) > threshold
    if not jump.any():
        return t, y
    idx = np.flatnonzero(jump) + 1
    return np.insert(t, idx, np.datetime64("NaT")), np.insert(y, idx, np.nan)


def timeseries(logs, keys, height_per=150):
    """Stacked panels with a shared time axis, one row per channel."""
    keys = [k for k in keys if k]
    if not keys or not logs:
        return None

    titles = [CHANNELS_BY_KEY[k].label if k in CHANNELS_BY_KEY else k for k in keys]
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True,
                        vertical_spacing=0.045, subplot_titles=titles)

    for row, key in enumerate(keys, start=1):
        ch = CHANNELS_BY_KEY.get(key)
        for i, lg in enumerate(logs):
            df = lg.df_view if hasattr(lg, "df_view") else lg.df
            if key not in df.columns:
                continue
            sub = df[["t", key]].dropna()
            if sub.empty:
                continue

            t = _times(sub["t"])
            y = sub[key].to_numpy(dtype=float)
            color = SERIES[i % len(SERIES)]

            if ch is not None and (ch.circular or ch.signed_circular):
                td, yd, _, _ = _envelope(t, y)
                td, yd = _break_wraps(td, yd)
                fig.add_trace(go.Scattergl(x=td, y=yd, mode="lines", name=lg.id,
                                           line=dict(color=color, width=1.2),
                                           connectgaps=False,
                                           hovertemplate="%{y:.1f}<extra></extra>"),
                              row=row, col=1)
            else:
                _add_band_line(fig, row, t, y, color, lg.id)

        if ch is not None:
            fig.update_yaxes(title_text=ch.unit, row=row, col=1)
            if ch.circular:
                fig.update_yaxes(range=[0, 360], dtick=90, row=row, col=1)
            elif ch.signed_circular:
                fig.update_yaxes(range=[-180, 180], dtick=90, row=row, col=1)
            if ch.y_range:
                fig.update_yaxes(range=list(ch.y_range), row=row, col=1)
            if ch.zero_line:
                # _layout turns the axis zeroline off, but for a signed quantity
                # the sign is the reading: which side of the line, which way the
                # rudder is over.
                fig.add_hline(y=0, line=dict(color=ZERO, width=1), row=row, col=1)

    for ann in fig.layout.annotations:
        ann.font.size = 12
    return _layout(fig, height_per * len(keys))


def _add_band_line(fig, row, t, y, color, name, *, hovertemplate="%{y:.3g}<extra></extra>",
                   width=1.4, dash=None, showlegend=True):
    """Mean line over a min/max band, so a decimated series still shows spikes."""
    td, ym, lo, hi = _envelope(t, y)
    if lo is not None:
        fig.add_trace(go.Scattergl(
            x=np.concatenate([td, td[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself", mode="none",
            fillcolor=_rgba(color, 0.18), hoverinfo="skip",
            showlegend=False), row=row, col=1)
    fig.add_trace(go.Scattergl(x=td, y=ym, mode="lines", name=name,
                               line=dict(color=color, width=width, dash=dash),
                               showlegend=showlegend,
                               hovertemplate=hovertemplate), row=row, col=1)


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


POWER_COLORS = {"ch1": "#3d8fd6", "ch2": "#eb6834", "ch3": "#1baf7a"}
TOTAL_COLOR = "#9457c9"
# _rgba parses hex, so a fallback has to be hex too, not one of the rgba greys.
OTHER_CHANNEL_COLOR = "#898781"


def power_timeseries(log, channels=POWER_CHANNELS, height=430):
    """Watts and amps per monitor channel, with the three-channel total.

    Drawn from raw rows rather than fixes: draw changes continuously while the
    GPS is still repeating one position, so the deduplicated frame would sample
    the load at whatever instant the receiver happened to update.
    """
    df = log.df_view if hasattr(log, "df_view") else log.df
    live = [ch for ch in channels
            if ch.power in df.columns and df[ch.power].notna().any()]
    if not live or len(df) < 2:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=["Power", "Current"])
    t = _times(df["t"])
    total = np.zeros(len(df))

    for ch in live:
        color = POWER_COLORS.get(ch.key, OTHER_CHANNEL_COLOR)
        w = df[ch.power].to_numpy(dtype=float)
        total += np.nan_to_num(w)
        _add_band_line(fig, 1, t, w, color, ch.label,
                       hovertemplate="%{y:.2f} W<extra>" + ch.label + "</extra>")
        if ch.current in df.columns:
            _add_band_line(fig, 2, t, df[ch.current].to_numpy(dtype=float), color,
                           ch.label, showlegend=False,
                           hovertemplate="%{y:.3f} A<extra>" + ch.label + "</extra>")

    if len(live) > 1:
        _add_band_line(fig, 1, t, total, TOTAL_COLOR, "Total", dash="dot",
                       hovertemplate="%{y:.2f} W<extra>total</extra>")

    fig.update_yaxes(title_text="W", row=1, col=1)
    fig.update_yaxes(title_text="A", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 12

    fig = _layout(fig, height)
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.04,
                                  xanchor="left", x=0))
    return fig


# ------------------------------------------------------------- sail and nav

BOOL_TRUE_COLORS = {"beating": "#3d8fd6", "motor_assist": "#eb6834",
                    "sailing": "#1baf7a", "settled": "#9457c9", "tacked": "#d64b8a"}
BOOL_FALSE_COLOR = "rgba(128,128,128,0.28)"
FAULT_COLORS = {SAIL_FAULT_OK: "#1baf7a"}
FAULT_OTHER = "#d03b3b"


def _lane_text(df, column):
    """A column as display strings, "" wherever it was not logged."""
    if column not in df.columns:
        return np.array([""] * len(df), dtype=object)
    col = df[column]
    if column == "tack":
        v = col.to_numpy(dtype=float)
        return np.array([TACK_LABELS.get(int(np.sign(x)), "") if np.isfinite(x) else ""
                         for x in v], dtype=object)
    return col.astype(object).where(col.notna(), "").astype(str).to_numpy()


def _value_runs(t, values):
    """Contiguous runs of one value, as (value, start time, end time).

    Blank stretches -- a column the firmware only fills in AUTO -- are left out
    rather than drawn, so the lane shows a gap where the state was not defined.
    """
    values = np.asarray(values, dtype=object)
    if values.size == 0:
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change - 1, [values.size - 1]])
    return [(values[a], t[a], t[b]) for a, b in zip(starts, ends) if values[a] != ""]


def _lane_color(column, value):
    if column == "propulsion":
        return PROPULSION_COLORS.get(value, OTHER_CHANNEL_COLOR)
    if column == "sail_state":
        return SAIL_STATE_COLORS.get(value, OTHER_CHANNEL_COLOR)
    if column == "sail_fault":
        return FAULT_COLORS.get(value, FAULT_OTHER)
    if column == "tack":
        return {v: TACK_COLORS[k] for k, v in TACK_LABELS.items()}.get(value, OTHER_CHANNEL_COLOR)
    return BOOL_TRUE_COLORS.get(column, "#3d8fd6") if value == "True" else BOOL_FALSE_COLOR


SAIL_LANES = [("propulsion", "Propulsion"), ("sail_state", "Sail state"),
              ("sail_fault", "Sail fault"), ("tack", "Tack"),
              ("beating", "Beating"), ("motor_assist", "Motor assist"),
              ("sailing", "Sailing"), ("settled", "Settled")]


def sail_state_lanes(log, height_per=30):
    """What the sail logic was doing, as one ribbon per state column.

    A stacked time series cannot show this: these are categories, and what
    matters is when each one held and how they line up against each other.
    """
    df = log.df_view if hasattr(log, "df_view") else log.df
    if df.empty:
        return None

    lanes = [(col, label) for col, label in SAIL_LANES
             if col in df.columns and (_lane_text(df, col) != "").any()]
    if not lanes:
        return None

    t = _times(df["t"])
    fig = go.Figure()
    labels = []

    for i, (column, label) in enumerate(lanes):
        labels.append(label)
        runs = _value_runs(t, _lane_text(df, column))

        # One trace per distinct value rather than per run: a beating flag that
        # flickers produces hundreds of runs, and hundreds of traces is what
        # makes a Plotly figure slow to serialise and slow to draw.
        by_value = {}
        for value, a, b in runs:
            by_value.setdefault(value, []).append((a, b))

        for value, spans in by_value.items():
            x, y = [], []
            for a, b in spans:
                x += [a, b, np.datetime64("NaT")]
                y += [i, i, None]
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", connectgaps=False,
                line=dict(color=_lane_color(column, value), width=14),
                hovertemplate=f"{label}: {value}<extra></extra>"))

    _add_reason_hover(fig, df, t, lanes)

    fig.update_yaxes(tickmode="array", tickvals=list(range(len(lanes))),
                     ticktext=labels, range=[-0.6, len(lanes) - 0.4], showgrid=False)
    fig.update_xaxes(range=[t[0], t[-1]])
    fig = _layout(fig, 60 + height_per * len(lanes), "Sail logic state")
    # Unified hover would fire all eight lanes at once; each lane is a separate
    # statement and is read on its own.
    fig.update_layout(hovermode="closest")
    return fig


def _add_reason_hover(fig, df, t, lanes):
    """Attach the sail logic's own explanation to the lane it explains.

    sail_reason is free text that changes every second or so -- far too much to
    draw -- but it is the only place the firmware says *why* it chose a heading,
    so it rides along as invisible hover points at each change.
    """
    if "sail_reason" not in df.columns:
        return
    reason = _lane_text(df, "sail_reason")
    if not (reason != "").any():
        return

    row = next((i for i, (col, _) in enumerate(lanes) if col == "sail_state"), 0)
    changed = np.concatenate([[True], reason[1:] != reason[:-1]])
    keep = changed & (reason != "")
    fig.add_trace(go.Scatter(
        x=t[keep], y=np.full(keep.sum(), row), mode="markers",
        marker=dict(size=16, opacity=0), customdata=reason[keep],
        hovertemplate="%{customdata}<extra>reason</extra>"))


def navigation_timeseries(log, height=640):
    """The autopilot's own view: where it wants to go and how far off it is."""
    df = log.df_view if hasattr(log, "df_view") else log.df
    wanted = ["des_hdg_deg", "brg_deg", "err_deg", "xte_m", "dist_m"]
    if df.empty or not any(c in df.columns and df[c].notna().any() for c in wanted):
        return None

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        subplot_titles=["Heading: actual vs desired vs bearing to waypoint",
                        "Bearing error (bearing − heading, what the rudder answers)",
                        "Cross-track error (+ is right of the line)",
                        "Distance to waypoint"],
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]])

    t = _times(df["t"])

    for column, name, color, dash in (("heading", "actual", "#3d8fd6", None),
                                      ("des_hdg_deg", "desired", "#eb6834", "dash"),
                                      ("brg_deg", "to waypoint", "#9457c9", "dot")):
        if column not in df.columns or not df[column].notna().any():
            continue
        td, yd, _, _ = _envelope(t, df[column].to_numpy(dtype=float))
        td, yd = _break_wraps(td, yd)
        fig.add_trace(go.Scattergl(
            x=td, y=yd, mode="lines", name=name, connectgaps=False,
            line=dict(color=color, width=1.4, dash=dash),
            hovertemplate="%{y:.1f}&deg;<extra>" + name + "</extra>"), row=1, col=1)

    if "err_deg" in df.columns:
        td, yd, _, _ = _envelope(t, df["err_deg"].to_numpy(dtype=float))
        td, yd = _break_wraps(td, yd)
        fig.add_trace(go.Scattergl(
            x=td, y=yd, mode="lines", showlegend=False, connectgaps=False,
            line=dict(color="#d64b8a", width=1.2),
            hovertemplate="%{y:.1f}&deg;<extra>bearing error</extra>"), row=2, col=1)

    if "xte_m" in df.columns:
        _add_band_line(fig, 3, t, df["xte_m"].to_numpy(dtype=float), "#0f9bb5",
                       "cross-track", showlegend=False,
                       hovertemplate="%{y:.1f} m<extra>off the line</extra>")

    if "dist_m" in df.columns:
        _add_band_line(fig, 4, t, df["dist_m"].to_numpy(dtype=float), "#1baf7a",
                       "distance", showlegend=False,
                       hovertemplate="%{y:.1f} m<extra>to waypoint</extra>")
    if "wp_left" in df.columns and df["wp_left"].notna().any():
        sub = df[["t", "wp_left"]].dropna()
        fig.add_trace(go.Scatter(
            x=_times(sub["t"]), y=sub["wp_left"].to_numpy(dtype=float),
            mode="lines", name="waypoints left", line_shape="hv",
            line=dict(color=MUTED, width=1.2, dash="dot"),
            hovertemplate="%{y:.0f} left<extra></extra>"),
            row=4, col=1, secondary_y=True)
        fig.update_yaxes(title_text="left", showgrid=False, row=4, col=1,
                         secondary_y=True)

    fig.update_yaxes(title_text="°", range=[0, 360], dtick=90, row=1, col=1)
    fig.update_yaxes(title_text="°", range=[-180, 180], dtick=90, row=2, col=1)
    fig.update_yaxes(title_text="m", row=3, col=1)
    fig.update_yaxes(title_text="m", row=4, col=1, secondary_y=False)
    for row in (2, 3):
        fig.add_hline(y=0, line=dict(color=ZERO, width=1), row=row, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 12

    fig = _layout(fig, height)
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.03,
                                  xanchor="left", x=0))
    return fig


def polar_speed(fix, bins=24, min_points=10):
    """Boat speed against true wind angle -- the boat's own polar diagram.

    Every angle the boat actually sailed, with the median speed at each,
    measured rather than predicted.
    """
    if "twa" not in fix.columns or "speed" not in fix.columns:
        return None
    d = fix[["twa", "speed"]].dropna()
    if "moving" in fix.columns:
        d = d[fix.loc[d.index, "moving"].astype(bool)]
    if len(d) < min_points:
        return None

    twa = d["twa"].to_numpy(dtype=float)
    speed = d["speed"].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=speed, theta=twa, mode="markers", name="fixes",
        marker=dict(size=5, color=_rgba("#3d8fd6", 0.45)),
        hovertemplate="TWA %{theta:.0f}&deg;, %{r:.2f} m/s<extra></extra>"))

    edges = np.linspace(-180, 180, bins + 1)
    idx = np.clip(np.digitize(twa, edges) - 1, 0, bins - 1)
    centers, medians = [], []
    for b in range(bins):
        sel = speed[idx == b]
        if sel.size >= 3:
            centers.append((edges[b] + edges[b + 1]) / 2)
            medians.append(float(np.median(sel)))
    if len(centers) >= 3:
        fig.add_trace(go.Scatterpolar(
            r=medians, theta=centers, mode="lines+markers", name="median",
            line=dict(color="#eb6834", width=2),
            hovertemplate="TWA %{theta:.0f}&deg;: median %{r:.2f} m/s<extra></extra>"))

    fig.update_layout(
        height=380, margin=dict(l=40, r=40, t=48, b=32),
        title="Speed by true wind angle (0° = head to wind)",
        font=dict(size=12), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(gridcolor=GRID, ticksuffix=" m/s", angle=90),
            angularaxis=dict(direction="clockwise", rotation=90,
                             gridcolor=GRID, color=MUTED, dtick=45)))
    return fig


def residual_timeseries(fix, log_id=""):
    """Heading minus course over time. A usable compass hugs the zero line."""
    d = fix[["t", "resid", "speed"]].dropna(subset=["resid"])
    if d.empty:
        return None

    speed = d["speed"].to_numpy(dtype=float)
    hi = np.nanpercentile(speed, 95) if np.isfinite(speed).any() else 1.0
    opacity = 0.25 + 0.75 * np.clip(speed / (hi or 1.0), 0, 1)

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=ZERO, width=1.5))
    fig.add_trace(go.Scattergl(
        x=_times(d["t"]), y=d["resid"], mode="markers",
        marker=dict(size=5, color=[_rgba("#3d8fd6", o) for o in opacity]),
        hovertemplate="%{y:.1f}&deg;<extra></extra>", name="residual"))

    fig.update_yaxes(range=[-180, 180], dtick=90, title_text="heading - course (°)")
    return _layout(fig, 300, f"Compass residual{' — ' + log_id if log_id else ''}"
                             " (darker = faster, more trustworthy)")


def residual_rose(fix, bins=36):
    """Circular histogram of the residual. A peak at 0 means the compass tracks."""
    r = fix["resid"].dropna().to_numpy(dtype=float)
    if r.size < 10:
        return None

    edges = np.linspace(-180, 180, bins + 1)
    counts, _ = np.histogram(r, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    fig = go.Figure(go.Barpolar(
        r=counts, theta=centers, width=360 / bins * 0.95,
        marker=dict(color=counts, colorscale=[[0, "#a9cdf2"], [1, "#1d5fa8"]],
                    line=dict(width=0)),
        hovertemplate="%{theta:.0f}&deg;: %{r} fixes<extra></extra>"))

    fig.update_layout(
        height=340, margin=dict(l=40, r=40, t=48, b=32),
        title="Residual distribution (0° = compass agrees with course)",
        font=dict(size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(showticklabels=False, gridcolor=GRID, ticks=""),
            angularaxis=dict(direction="clockwise", rotation=90,
                             gridcolor=GRID, color=MUTED, dtick=45)))
    return fig


def speed_histogram(fix, threshold=0.5):
    """Speed distribution, with the jitter floor marked.

    Makes it immediately obvious that most 'motion' in these logs is receiver
    noise rather than travel.
    """
    s = fix["speed"].dropna().to_numpy(dtype=float)
    if s.size < 10:
        return None

    hi = float(np.nanpercentile(s, 99.5))
    fig = go.Figure(go.Histogram(x=s[s <= hi], nbinsx=60,
                                 marker=dict(color="#3d8fd6", line=dict(width=0)),
                                 hovertemplate="%{x:.2f} m/s: %{y}<extra></extra>"))
    fig.add_vline(x=threshold, line=dict(color="#d03b3b", width=2, dash="dash"),
                  annotation_text=f"moving threshold {threshold:g} m/s",
                  annotation_position="top right",
                  annotation_font=dict(color="#d03b3b", size=11))
    fig.update_xaxes(title_text="speed (m/s)")
    fig.update_yaxes(title_text="fixes")
    return _layout(fig, 260, "Speed distribution")


def timebar(logs, sel0, sel1, min_speed=0.5, height_per=40):
    """Overview of the whole recording with the selected window highlighted.

    One lane per log, showing a speed profile and where the boat was actually
    moving. Without this the range slider is a blind control: these logs are
    mostly idle, so knowing *where* the interesting minutes are is the whole
    problem.
    """
    if not logs:
        return None

    # Shapes are serialised separately from trace data, and pandas Timestamps
    # are not JSON-serialisable there -- hand them native datetimes.
    def native(ts):
        return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

    t_lo = native(min(lg.t_start for lg in logs))
    t_hi = native(max(lg.t_end for lg in logs))
    sel0, sel1 = native(sel0), native(sel1)

    fig = make_subplots(rows=len(logs), cols=1, shared_xaxes=True,
                        vertical_spacing=0.12 / max(len(logs), 1))

    for row, lg in enumerate(logs, start=1):
        color = SERIES[(row - 1) % len(SERIES)]
        fix = lg.fix

        # Full extent of the log, so gaps between recordings stay visible.
        fig.add_trace(go.Scatter(
            x=[native(lg.t_start), native(lg.t_end)], y=[0, 0],
            mode="lines", hoverinfo="skip",
            line=dict(color="rgba(128,128,128,0.28)", width=10)), row=row, col=1)

        if "speed" in fix.columns and len(fix):
            d = fix[["t", "speed"]].dropna()
            if len(d):
                t, y = _times(d["t"]), d["speed"].to_numpy(dtype=float)
                td, ym, _, hi = _envelope(t, y, target=600)
                peak = hi if hi is not None else ym
                cap = float(np.nanpercentile(y, 99)) or 1.0
                norm = np.clip(peak / cap, 0, 1)
                fig.add_trace(go.Scatter(
                    x=td, y=norm, mode="lines", fill="tozeroy",
                    line=dict(color=color, width=1),
                    fillcolor=_rgba(color, 0.35),
                    hovertemplate="%{x|%H:%M:%S}<extra></extra>"), row=row, col=1)

        # Mark the stretches that cleared the jitter threshold.
        if "moving" in fix.columns and fix["moving"].any():
            for a, b in _true_spans(_times(fix["t"]), fix["moving"].to_numpy()):
                fig.add_trace(go.Scatter(
                    x=[native(a), native(b)], y=[-0.22, -0.22], mode="lines",
                    line=dict(color="#0ca30c", width=6),
                    hovertemplate="moving<extra></extra>"), row=row, col=1)

        fig.update_yaxes(range=[-0.4, 1.12], showticklabels=False,
                         showgrid=False, row=row, col=1)

    # Dim everything outside the selection.
    for row in range(1, len(logs) + 1):
        if sel0 > t_lo:
            fig.add_vrect(x0=t_lo, x1=sel0, row=row, col=1, line_width=0,
                          fillcolor="rgba(128,128,128,0.40)", layer="above")
        if sel1 < t_hi:
            fig.add_vrect(x0=sel1, x1=t_hi, row=row, col=1, line_width=0,
                          fillcolor="rgba(128,128,128,0.40)", layer="above")
        for edge in (sel0, sel1):
            fig.add_vline(x=edge, row=row, col=1,
                          line=dict(color="#d03b3b", width=2))

    fig.update_xaxes(range=[t_lo, t_hi])
    fig = _layout(fig, max(120, height_per * len(logs) + 46))

    # Lane labels as horizontal annotations. A y-axis title would be rotated
    # vertically, and with several short lanes those collide with each other.
    for row, lg in enumerate(logs, start=1):
        axis = fig.layout[f"yaxis{row}" if row > 1 else "yaxis"]
        lo, hi = axis.domain
        fig.add_annotation(
            xref="paper", yref="paper", x=-0.008, y=(lo + hi) / 2,
            xanchor="right", yanchor="middle", showarrow=False,
            text=lg.id.replace("boat_log_", ""),
            font=dict(size=10, color=MUTED))

    fig.update_layout(hovermode="closest", margin=dict(l=132, r=16, t=8, b=28))
    return fig


def _true_spans(t, flag):
    """Contiguous [start, end] ranges where a boolean series is true."""
    flag = np.asarray(flag, dtype=bool)
    if not flag.any():
        return []
    edges = np.diff(flag.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if flag[0]:
        starts.insert(0, 0)
    if flag[-1]:
        ends.append(len(flag) - 1)
    return [(t[a], t[b]) for a, b in zip(starts, ends) if b > a]


def aliveness_strip(qa, t0, t1, keys):
    """Timeline ribbon showing when each channel carried real signal."""
    rows = [(k, qa.channel(k)) for k in keys if qa.channel(k) is not None]
    if not rows:
        return None

    status_color = {"ALIVE": "#0ca30c", "PARTIAL": "#fab219",
                    "FROZEN": "#ec835a", "CONSTANT": "#d03b3b", "ABSENT": "#d5d3cc"}

    fig = go.Figure()
    labels = []
    for i, (key, rep) in enumerate(rows):
        labels.append(CHANNELS_BY_KEY[key].label if key in CHANNELS_BY_KEY else key)
        fig.add_trace(go.Scatter(
            x=[t0, t1], y=[i, i], mode="lines", hoverinfo="skip",
            line=dict(color="rgba(128,128,128,0.25)", width=14)))
        spans = rep.alive_windows or ([(t0, t1)] if rep.verdict == "ALIVE" else [])
        for a, b in spans:
            fig.add_trace(go.Scatter(
                x=[a, b], y=[i, i], mode="lines",
                line=dict(color=status_color.get(rep.verdict, MUTED), width=14),
                hovertemplate=f"{labels[-1]}: {rep.verdict}<extra></extra>"))

    fig.update_yaxes(tickmode="array", tickvals=list(range(len(rows))),
                     ticktext=labels, range=[-0.6, len(rows) - 0.4],
                     showgrid=False)
    fig.update_xaxes(range=[t0, t1])
    return _layout(fig, 42 + 30 * len(rows), "When was each channel alive?")
