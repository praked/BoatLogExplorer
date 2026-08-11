"""Plotly time-series and compass-diagnostic figures."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .schema import CHANNELS_BY_KEY, POWER_CHANNELS

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

            if ch is not None and ch.circular:
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
