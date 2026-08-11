"""Boat Log Explorer — load boat CSV logs and visualise them on a map.

Run with:  streamlit run app.py
"""

import io
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from boatviz import charts, derive, export, mapview
from boatviz.ingest import (STATUS_EMPTY, STATUS_ERROR, STATUS_FRAGMENT,
                            STATUS_NO_GPS, STATUS_OK, discover_logs, load_log)
from boatviz.qa import ABSENT, ALIVE, CONSTANT, FROZEN, PARTIAL, assess
from boatviz.schema import CHANNELS, MODE_LABELS, POWER_CHANNELS

st.set_page_config(page_title="Boat Log Explorer", page_icon="⛵",
                   layout="wide", initial_sidebar_state="expanded")

# First folder that actually exists wins, so the same code serves a local
# checkout with real logs and the hosted build that ships only samples.
DEFAULT_FOLDER = next(
    (c for c in ("logs_4Aug2026", "sample_logs", "logs") if Path(c).is_dir()),
    "sample_logs")

STATUS_STYLE = {
    ALIVE:    ("●", "#0ca30c"),
    PARTIAL:  ("◐", "#fab219"),
    FROZEN:   ("◌", "#ec835a"),
    CONSTANT: ("✕", "#d03b3b"),
    ABSENT:   ("–", "#898781"),
}

# Colours inherit from Streamlit's active theme so the app stays legible in
# both light and dark; only the status accents are fixed, and they are chosen
# to hold contrast either way.
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 100%; }
  .chip { display:inline-block; padding:1px 7px; border-radius:999px;
          font-size:11px; font-weight:600; letter-spacing:.02em; }
  .qa-row { display:flex; justify-content:space-between; align-items:baseline;
            gap:8px; font-size:12px; padding:2px 0; }
  .qa-row .name { color:inherit; }
  .qa-row .val  { opacity:.65; font-variant-numeric:tabular-nums;
                  text-align:right; }
  div[data-testid="stMetricValue"] { font-size:1.25rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------- data loading

def _finish(log, speed_window, d_min, v_max, min_speed):
    if log.status in (STATUS_OK, STATUS_FRAGMENT, STATUS_NO_GPS):
        derive.enrich(log, speed_window_s=speed_window, d_min_m=d_min,
                      v_max_ms=v_max, min_speed_ms=min_speed)
        log.qa = assess(log)
    return log


@st.cache_data(show_spinner=False)
def _load_path(path_str, mtime, size, speed_window, d_min, v_max, min_speed):
    """Parse and analyse one log. Keyed on file identity plus derive settings.

    Without this cache Streamlit would re-parse every log on each widget change.
    """
    return _finish(load_log(path_str), speed_window, d_min, v_max, min_speed)


@st.cache_data(show_spinner=False)
def _load_bytes(name, data, speed_window, d_min, v_max, min_speed):
    """Same, for an uploaded file. Keyed on the bytes themselves."""
    log = load_log(io.BytesIO(data), log_id=Path(name).stem, size=len(data))
    log.path = Path(name)
    return _finish(log, speed_window, d_min, v_max, min_speed)


def load_all(sources, speed_window, d_min, v_max, min_speed):
    logs, bar = [], st.sidebar.progress(0.0, "Loading logs…")
    for i, src in enumerate(sources):
        bar.progress(i / max(len(sources), 1), f"Loading {getattr(src, 'name', src)}…")
        if hasattr(src, "read"):
            logs.append(_load_bytes(src.name, src.getvalue(),
                                    speed_window, d_min, v_max, min_speed))
        else:
            p = Path(src)
            stat = p.stat()
            logs.append(_load_path(str(p), stat.st_mtime, stat.st_size,
                                   speed_window, d_min, v_max, min_speed))
    bar.empty()
    return logs


# ------------------------------------------------------------------- sidebar

st.sidebar.title("⛵ Boat Log Explorer")

# A folder path only works when the app runs on the same machine as the logs.
# Hosted deployments need upload, so both are offered and upload wins when used.
uploaded = st.sidebar.file_uploader(
    "Upload logs", type="csv", accept_multiple_files=True,
    help="Drop one or more boat_log_*.csv files here")

folder = st.sidebar.text_input(
    "…or read a local folder", DEFAULT_FOLDER,
    help="Only works when the app runs on the machine holding the logs")

sources = list(uploaded) if uploaded else discover_logs(folder)

if not sources:
    st.title("Boat Log Explorer")
    st.info("**Upload one or more `boat_log_*.csv` files** using the sidebar "
            "to begin — or, if you are running this app on the machine that "
            "holds the logs, point the folder box at their directory.")
    st.caption("Every CSV is checked on load, so files that turn out to be "
               "empty, truncated, or recorded with a dead sensor are reported "
               "rather than silently drawn as an empty map.")
    st.stop()

with st.sidebar.expander("Analysis settings", expanded=False):
    min_speed = st.slider("Moving threshold (m/s)", 0.0, 3.0, 0.5, 0.1,
                          help="Below this, GPS jitter dominates and course "
                               "over ground is treated as undefined.")
    d_min = st.slider("Min displacement for course (m)", 1.0, 20.0, 3.0, 1.0)
    speed_window = st.slider("Speed window (s)", 1.0, 30.0, 5.0, 1.0)
    v_max = st.slider("Reject fixes faster than (m/s)", 2.0, 30.0, 8.0, 1.0)

logs = load_all(sources, speed_window, d_min, v_max, min_speed)
usable = [lg for lg in logs if lg.has_track]

st.sidebar.subheader("Logs")
if usable:
    default = [usable[0].id] if len(usable) > 3 else [lg.id for lg in usable]
    chosen = st.sidebar.multiselect(
        f"{len(usable)} of {len(logs)} files have a track",
        [lg.id for lg in usable], default=default,
        format_func=lambda i: f"{i.replace('boat_log_', '')}")
else:
    chosen = []
    st.sidebar.warning("No log in this folder contains a usable GPS track.")

selected = [lg for lg in usable if lg.id in chosen]

skipped = [lg for lg in logs if not lg.has_track]
if skipped:
    with st.sidebar.expander(f"{len(skipped)} file(s) with no track", expanded=False):
        for lg in skipped:
            reason = {STATUS_EMPTY: "empty file", STATUS_ERROR: lg.message,
                      STATUS_NO_GPS: "no GPS fixes",
                      STATUS_FRAGMENT: f"fragment, {lg.n_rows} rows"
                      }.get(lg.status, lg.message or lg.status)
            st.markdown(f"<div class='qa-row'><span class='name'>"
                        f"{lg.id.replace('boat_log_', '')}</span>"
                        f"<span class='val'>{reason}</span></div>",
                        unsafe_allow_html=True)


# ------------------------------------------------------- time + view filtering

def _error_shade(err):
    """Green where the compass agrees with course, red at or past random guessing."""
    if not np.isfinite(err):
        return ""
    frac = min(max(err / 90.0, 0.0), 1.0)
    r, g, b = (int(round(a + (c - a) * frac))
               for a, c in ((12, 208), (163, 59), (12, 59)))
    return f"background-color: rgba({r},{g},{b},0.16)"


def _as_utc(value):
    """Streamlit's slider may hand back naive or tz-aware datetimes."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _recorded(frame, col):
    """True when a categorical column exists and holds a value in this frame.

    The nav and power columns arrived in a later firmware revision, so every
    feature built on them has to cope with logs that predate it.
    """
    return col in frame.columns and bool((frame[col].astype(str) != "").any())


def _has_power(log):
    return any(ch.power in log.df.columns and log.df[ch.power].notna().any()
               for ch in POWER_CHANNELS)


def apply_time_filter(logs_, t0, t1):
    for lg in logs_:
        fix = lg.fix
        lg.view = fix[(fix["t"] >= t0) & (fix["t"] <= t1)]
        lg.df_view = lg.df[(lg.df["t"] >= t0) & (lg.df["t"] <= t1)]
    return logs_


if not selected:
    st.title("Boat Log Explorer")
    st.info("Select at least one log in the sidebar.")
    if skipped:
        st.caption(f"{len(skipped)} file(s) in this folder have no usable track — "
                   "see the sidebar for why.")
    st.stop()

t_lo = min(lg.t_start for lg in selected)
t_hi = max(lg.t_end for lg in selected)

st.title("Boat Log Explorer")


# ----------------------------------------------------------------- time range

def _fmt_span(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _slider_step(span_s):
    """Roughly 500 stops across the range, snapped to a readable unit.

    Streamlit's default step for a datetime slider is 15 minutes, which on a
    two-hour log leaves about seven usable positions.
    """
    for unit in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if span_s / unit <= 500:
            return timedelta(seconds=unit)
    return timedelta(seconds=3600)


span_total = (t_hi - t_lo).total_seconds()
sel_key = "time_sel"

st.markdown("#### Time range")

# Streamlit forbids writing a widget's session-state key once that widget
# exists, so the preset buttons must run before the slider is created. These
# slots reserve the display order while letting execution order differ.
slider_slot = st.container()
preset_slot = st.container()
readout_slot = st.empty()


def _set_range(a, b):
    """Store a preset selection, clamped to the axis.

    A preset can overshoot: "Moving only" pads by 30 s, and motion often starts
    or ends within 30 s of the log's own bounds. The slider rejects a stored
    value outside [min_value, max_value] on the following run.
    """
    a, b = _as_utc(a), _as_utc(b)
    st.session_state[sel_key] = (min(max(a, t_lo), t_hi).to_pydatetime(),
                                 min(max(b, t_lo), t_hi).to_pydatetime())


# Changing the log selection changes the time axis, and a selection carried
# over from the previous axis would sit outside the new bounds.
span_sig = (t_lo.isoformat(), t_hi.isoformat())
if st.session_state.get("_time_axis") != span_sig:
    st.session_state["_time_axis"] = span_sig
    st.session_state.pop(sel_key, None)


# Jumping straight to the moving stretches matters because these logs are
# mostly idle -- finding the few real minutes by hand is the tedious part.
moving_times = [lg.fix.loc[lg.fix["moving"], "t"] for lg in selected
                if "moving" in lg.fix.columns and lg.fix["moving"].any()]
all_moving = pd.concat(moving_times) if moving_times else None

with preset_slot:
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Whole log", width="stretch"):
        _set_range(t_lo, t_hi)
    if b2.button("Moving only", width="stretch", disabled=all_moving is None,
                 help="Zoom to the stretch containing all motion above the "
                      "threshold — usually the only part worth looking at"):
        pad = pd.Timedelta(seconds=30)
        _set_range(all_moving.min() - pad, all_moving.max() + pad)
    if b3.button("First 10 min", width="stretch"):
        _set_range(t_lo, min(t_hi, t_lo + pd.Timedelta(minutes=10)))
    if b4.button("Last 10 min", width="stretch"):
        _set_range(max(t_lo, t_hi - pd.Timedelta(minutes=10)), t_hi)

if span_total > 1:
    # value= is only the initial default; once the key holds a selection,
    # passing it again would be ignored and Streamlit warns about it.
    extra = ({} if sel_key in st.session_state
             else {"value": (t_lo.to_pydatetime(), t_hi.to_pydatetime())})
    with slider_slot:
        picked = st.slider(
            "Drag either end to select a section of the log",
            min_value=t_lo.to_pydatetime(), max_value=t_hi.to_pydatetime(),
            step=_slider_step(span_total),
            format="MMM D, HH:mm:ss" if span_total < 7200 else "MMM D, HH:mm",
            key=sel_key, **extra)
    t0, t1 = (_as_utc(p) for p in picked)
else:
    t0, t1 = t_lo, t_hi

selected = apply_time_filter(selected, t0, t1)
sel_span = (t1 - t0).total_seconds()

readout_slot.caption(
    f"**{t0.strftime('%b %d %H:%M:%S')} → {t1.strftime('%H:%M:%S')} UTC** · "
    f"{_fmt_span(sel_span)} of {_fmt_span(span_total)} "
    f"({sel_span / span_total * 100:.0f}% of the recording)")

bar = charts.timebar(selected, t0, t1, min_speed)
if bar:
    st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
    st.caption("Filled area is speed; green marks underneath are where the boat "
               "was moving. Everything shaded grey is outside your selection.")

st.divider()


# --------------------------------------------------------------------- header

n_fix = sum(len(lg.view) for lg in selected)
n_moving = sum(int(lg.view["moving"].sum()) for lg in selected if "moving" in lg.view)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Logs selected", len(selected))
c2.metric("GPS fixes in window", f"{n_fix:,}")
c3.metric("Fixes while moving", f"{n_moving:,}",
          f"{n_moving / n_fix * 100:.1f}%" if n_fix else None)
c4.metric("Window", _fmt_span(sel_span))

if n_fix and n_moving / n_fix < 0.02:
    st.warning(
        f"**The boat was essentially stationary for this window.** Only "
        f"{n_moving:,} of {n_fix:,} fixes ({n_moving / max(n_fix,1) * 100:.1f}%) "
        f"exceed {min_speed:g} m/s. What looks like a track is mostly GPS "
        f"receiver jitter — a stationary receiver still wanders a few metres per "
        f"minute. Turn on **Only while moving** to see just the real motion.")


# ------------------------------------------------------------------- QA cards

def qa_card(lg):
    bb = lg.bbox()
    qa = lg.qa
    st.markdown(f"**{lg.id.replace('boat_log_', '')}**")
    a, b, c = st.columns(3)
    a.caption(f"{lg.n_rows:,} rows · {lg.n_fixes:,} fixes")
    b.caption(f"{qa.dup_frac * 100:.0f}% duplicate fixes")
    c.caption(f"{bb[4]:.0f} × {bb[5]:.0f} m · {lg.duration_s / 3600:.1f} h")

    verdict_color = "#0ca30c" if qa.gps_verdict == "MOVING" else "#ec835a"
    st.markdown(
        f"<div class='qa-row'><span class='name'>GPS</span>"
        f"<span class='chip' style='background:{verdict_color}1a;color:{verdict_color}'>"
        f"{qa.gps_verdict}</span></div>", unsafe_allow_html=True)

    for ch in CHANNELS:
        rep = qa.channel(ch.key)
        if rep is None:
            continue
        icon, color = STATUS_STYLE.get(rep.verdict, ("–", "#898781"))
        dim = "" if rep.verdict in (ALIVE, PARTIAL) else "opacity:.55;"
        st.markdown(
            f"<div class='qa-row' style='{dim}'>"
            f"<span class='name'><span style='color:{color}'>{icon}</span> {ch.label}</span>"
            f"<span class='val'>{rep.summary}</span></div>",
            unsafe_allow_html=True)

    if qa.n_outliers:
        st.caption(f"{qa.n_outliers} implausible fix(es) rejected (> {v_max:g} m/s).")


with st.sidebar.expander("Sensor health", expanded=True):
    for lg in selected:
        qa_card(lg)
        st.divider()


# ----------------------------------------------------------------------- tabs

tab_map, tab_series, tab_power, tab_compass, tab_data = st.tabs(
    ["Map", "Time series", "Battery", "Compass check", "Data"])


with tab_map:
    ctrl, view = st.columns([1, 4])

    with ctrl:
        basemap = st.selectbox("Basemap", list(mapview.BASEMAPS), index=0)

        # Mode is offered only when a log actually carries it: on older logs the
        # option would silently colour the whole track one "not recorded" grey.
        color_choices = ["log", "time", "speed", "heading", "awa"]
        if any(_recorded(lg.fix, "mode") for lg in selected):
            color_choices.insert(1, "mode")
        color_by = st.selectbox(
            "Colour by", color_choices, index=0,
            format_func=lambda c: "control mode" if c == "mode" else c,
            help="Control mode splits the track by what was steering the boat — "
                 "RC, autonomous, or hybrid." if "mode" in color_choices else None)
        st.divider()

        show_track = st.checkbox("Track line", True)
        show_points = st.checkbox("GPS points", True)
        moving_only = st.checkbox("Only while moving", False,
                                  help=f"Hide fixes below {min_speed:g} m/s")
        st.divider()

        st.caption("**Direction arrows**")
        any_hdg = any(lg.qa.is_alive("heading") for lg in selected)
        any_awa = any(lg.qa.is_alive("awa_deg") for lg in selected)

        arrows = []
        if st.checkbox("Heading", any_hdg,
                       help=None if any_hdg else "Compass is frozen in every selected log"):
            arrows.append("heading")
        if st.checkbox("Course over ground", False):
            arrows.append("cog")
        if st.checkbox("Wind from", False,
                       help=None if any_awa else "Wind vane is stuck in every selected log"):
            arrows.append("wind_from")

        arrow_budget = st.slider("Arrow count", 0, 600, 150, 25)
        arrow_len = st.slider("Arrow length (m)", 2.0, 30.0, 6.0, 1.0)
        point_budget = st.select_slider(
            "Point budget", [250, 500, 1500, 3000, 6000], 1500,
            help="At or below 600 points each marker gets a full detail popup; "
                 "above that, detail collapses into the hover tooltip.")
        st.divider()

        convention = st.radio(
            "Heading convention",
            ["CW (clockwise, N=0 E=90)", "ACW (anticlockwise, N=0 W=90)"],
            index=0,
            help="Clockwise is the default because it matches the data: "
                 "see the Compass check tab for the evidence.")
        wind_sign = 1.0 if st.radio(
            "Wind angle sign", ["heading + AWA", "heading − AWA"], index=0,
            help="AWA is logged 0–360 increasing to starboard.") == "heading + AWA" else -1.0

    with view:
        cursor = None
        m = mapview.build_map(
            selected, basemap=basemap, color_by=color_by,
            show_track=show_track, show_points=show_points,
            point_budget=point_budget, arrows=arrows,
            arrow_budget=arrow_budget, arrow_len_m=arrow_len,
            convention=convention, wind_sign=wind_sign,
            moving_only=moving_only, cursor=cursor)
        # Rendered as plain HTML rather than via streamlit-folium: none of that
        # component's click-return values are used here, and dropping it keeps
        # the app to packages that also run in a browser-hosted build.
        components.html(m.get_root().render(), height=680, scrolling=False)

        if color_by == "mode":
            present = sorted({v for lg in selected
                              for v in lg.view.get("mode", pd.Series(dtype=str))
                                              .astype(str).unique() if v})
            st.caption(" · ".join(
                f"**{name}** — {MODE_LABELS[name]}" if name in MODE_LABELS
                else f"**{name}** — not a mode this app knows; check "
                     "`utils/modes.py` in the firmware"
                for name in present) or "No control mode recorded in this window.")

        if moving_only and n_moving == 0:
            st.info("No fixes in this window exceed the moving threshold, so the "
                    "map is empty. Lower the threshold or widen the time window.")


with tab_series:
    alive_keys, frozen_keys = [], []
    for ch in CHANNELS:
        reps = [lg.qa.channel(ch.key) for lg in selected]
        if any(r and r.verdict in (ALIVE, PARTIAL) for r in reps):
            alive_keys.append(ch.key)
        else:
            frozen_keys.append(ch.key)

    st.caption("Channels that are frozen or constant in every selected log are "
               "off by default. `sys_*` are Raspberry Pi health readings — CPU "
               "temperature and core voltage — not water, air or battery.")

    keys = st.multiselect("Channels", [c.key for c in CHANNELS], default=alive_keys,
                          format_func=lambda k: next(
                              (c.label for c in CHANNELS if c.key == k), k))

    if keys:
        fig = charts.timeseries(selected, keys)
        if fig:
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Select at least one channel.")

    st.divider()
    for lg in selected:
        strip = charts.aliveness_strip(lg.qa, lg.t_start, lg.t_end,
                                       [c.key for c in CHANNELS])
        if strip:
            st.caption(f"**{lg.id.replace('boat_log_', '')}**")
            # Streamlit derives a chart's identity from its contents, so two
            # logs that happen to produce identical figures collide. Every
            # chart drawn once per log therefore carries the log id as its key.
            st.plotly_chart(strip, width='stretch', key=f"strip_{lg.id}")


with tab_power:
    powered = [lg for lg in selected if _has_power(lg)]

    if not powered:
        st.info("**No power monitor in these logs.** The `ch1`–`ch3` columns "
                "arrived with the INA3221 in a later firmware revision; logs "
                "recorded before it have nothing to show here.")
    else:
        st.caption(
            "Three INA3221 channels, logged as bus power in **watts** and "
            "current in **amps** exactly as the driver reports them. Energy "
            "covers the selected time window only, integrated over the raw rows "
            "rather than GPS fixes. Which load sits on which channel is a wiring "
            "choice the firmware does not record, so they stay numbered.")

        for lg in powered:
            st.subheader(lg.id.replace("boat_log_", ""))
            rows = derive.power_summary(lg.df_view)
            live = [r for r in rows if r["present"]]

            if not live:
                st.warning("No power samples inside the selected time window.")
                continue

            total_wh = sum(r["energy_wh"] for r in live)
            mean_w = sum(r["mean_w"] for r in live)
            peak_total = sum(r["peak_w"] for r in live)

            a, b, c, d = st.columns(4)
            a.metric("Energy used", f"{total_wh:.2f} Wh")
            b.metric("Mean draw", f"{mean_w:.2f} W")
            c.metric("Peak, channels summed", f"{peak_total:.2f} W",
                     help="Sum of each channel's own peak, so it is an upper "
                          "bound — the peaks need not have coincided.")
            d.metric("Window", _fmt_span(sel_span))

            table = pd.DataFrame([{
                "Channel": r["label"],
                "Samples": r["n"],
                "Mean (W)": r["mean_w"],
                "Peak (W)": r["peak_w"],
                "Energy (Wh)": r["energy_wh"],
                "Mean (A)": r["mean_a"],
                "Peak (A)": r["peak_a"],
                "Bus (V)": r["bus_v"],
            } for r in rows])

            st.dataframe(
                table.style.format({
                    "Samples": "{:,}", "Mean (W)": "{:.2f}", "Peak (W)": "{:.2f}",
                    "Energy (Wh)": "{:.3f}", "Mean (A)": "{:.3f}",
                    "Peak (A)": "{:.3f}", "Bus (V)": "{:.2f}"}, na_rep="–"),
                width="stretch", hide_index=True, key=f"power_table_{lg.id}")
            st.caption("**Bus (V)** is not logged — it is power ÷ current, which "
                       "recovers it because the driver computes power as bus "
                       "voltage × current. Only samples drawing more than "
                       f"{derive.CURRENT_NOISE_A * 1000:.0f} mA count, so an "
                       "unloaded channel shows no voltage rather than noise ÷ noise.")

            flat = [r["label"] for r in live
                    if r["peak_w"] == 0.0 and not r["peak_a"] > 0]
            if flat:
                one = len(flat) == 1
                st.warning(
                    f"**{', '.join(flat)} read a flat zero for this whole "
                    f"window.** Either nothing is wired to "
                    f"{'that channel' if one else 'those channels'}, or the "
                    f"shunt is open and the rail is drawing current the monitor "
                    f"cannot see.")

            fig = charts.power_timeseries(lg)
            if fig:
                st.plotly_chart(fig, width="stretch", key=f"power_{lg.id}")
            st.divider()


with tab_compass:
    st.caption(
        "Is the compass trustworthy? The only independent reference is the "
        "direction the boat actually travelled between GPS fixes, so each "
        "candidate convention is scored against course over ground. A random "
        "guess scores 90°.")

    for lg in selected:
        st.subheader(lg.id.replace("boat_log_", ""))
        hdg_alive = lg.qa.is_alive("heading")
        awa_alive = lg.qa.is_alive("awa_deg")

        table, verdict, detail = derive.convention_scoreboard(
            lg.view if len(lg.view) else lg.fix, heading_alive=hdg_alive)

        if verdict == "conclusive":
            st.success(detail)
        elif verdict == "inconclusive":
            st.warning(detail)
        else:
            st.info(detail)

        if len(table):
            show = table.rename(columns={
                "convention": "Convention", "n": "Samples",
                "mean_abs_err": "Mean error (°)",
                "median_abs_err": "Median error (°)",
                "circ_offset": "Mean offset (°)", "R": "Concentration"})
            st.dataframe(
                show.style
                    .format({"Mean error (°)": "{:.1f}", "Median error (°)": "{:.1f}",
                             "Mean offset (°)": "{:+.1f}", "Concentration": "{:.3f}"})
                    .map(_error_shade, subset=["Median error (°)"]),
                width="stretch", hide_index=True)
            st.caption("A random guess scores 90°. Green means the compass "
                       "genuinely tracks the direction of travel.")

            a, b = st.columns(2)
            f1 = charts.residual_timeseries(lg.view if len(lg.view) else lg.fix)
            if f1:
                a.plotly_chart(f1, width='stretch', key=f"resid_{lg.id}")
            f2 = charts.residual_rose(lg.view if len(lg.view) else lg.fix)
            if f2:
                b.plotly_chart(f2, width='stretch', key=f"rose_{lg.id}")

        _, wverdict, wdetail = derive.wind_sign_evidence(
            lg.view if len(lg.view) else lg.fix, hdg_alive, awa_alive)
        st.markdown(f"**Wind sign convention** — {wdetail}")

        hist = charts.speed_histogram(lg.view if len(lg.view) else lg.fix, min_speed)
        if hist:
            st.plotly_chart(hist, width='stretch', key=f"speed_{lg.id}")
        st.divider()

    with st.expander("Why clockwise is the default", expanded=False):
        st.markdown("""
The earlier HTML viewers in this project negate the heading to enforce an
*anticlockwise* convention. Testing against GPS course over ground on the one
log with real motion shows that is backwards: the identity mapping (clockwise,
N=0 E=90 S=180 W=270) lands within about 12° median error, while the negated
form scores worse than random guessing.

The firmware agrees — `find_heading` in `boatv1/sensors/imu_handler.py` converts
its yaw to *"0..360 clockwise"* before logging.

All headings are **magnetic**: no declination correction is applied anywhere in
the firmware. At Lake Constance that is roughly +3°, small next to the residual
error you can see above.
""")


with tab_data:
    pick = st.selectbox("Log", [x.id for x in selected],
                        format_func=lambda i: i.replace("boat_log_", ""))
    lg = next(x for x in selected if x.id == pick)

    view = lg.view
    if st.checkbox("Only while moving", False, key="data_moving") and "moving" in view:
        view = view[view["moving"]]

    st.caption(f"{len(view):,} of {lg.n_fixes:,} fixes in the current selection. "
               f"Table shows the first 2,000; downloads include everything selected.")
    st.dataframe(view.head(2000), width='stretch', height=400)

    a, b, c = st.columns(3)
    a.download_button(
        "Download CSV", view.to_csv(index=False).encode(),
        f"{lg.id}_fixes.csv", "text/csv", width='stretch')
    b.download_button(
        "Download GeoJSON", export.to_geojson(view, lg.id),
        f"{lg.id}.geojson", "application/geo+json", width='stretch')
    c.download_button(
        "Sensor health report",
        "\n\n".join(export.qa_summary(x) for x in selected).encode(),
        "sensor_health.txt", "text/plain", width='stretch')

    with st.expander("What the columns mean", expanded=False):
        st.markdown("""
| column | meaning |
|---|---|
| `row` | index of this fix in the original CSV, for cross-referencing |
| `lat`, `lon` | WGS84 decimal degrees, deduplicated to distinct fixes |
| `dwell` | seconds this fix persisted before the receiver reported a new one |
| `hdg` | filtered magnetic heading, degrees clockwise from north |
| `cog` | course over ground, `NaN` unless the boat actually moved |
| `resid` | `hdg − cog`, the compass error against travel direction |
| `awa` | apparent wind angle, 0 = head-to-wind, increasing to starboard |
| `mode` | control mode the autopilot was in — `MANUAL`, `AUTO`, … — blank on logs older than the nav columns |
| `auto_mode` | which autonomous behaviour was running; only meaningful in `AUTO` |
| `speed` | metres per second over the centred speed window |
| `moving` | speed cleared the jitter threshold |
| `outlier` | rejected as a physically implausible jump |
""")
