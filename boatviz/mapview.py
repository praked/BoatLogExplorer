"""Folium map construction.

Folium rather than pydeck because these logs cover areas only 60-400 m across.
That demands satellite imagery zoomed past its native tile level, which needs
explicit max_native_zoom/max_zoom control and a token-free imagery source --
both of which Leaflet gives directly and pydeck does not.
"""

import html

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
from folium.plugins import Fullscreen, MeasureControl, MousePosition

from .derive import wind_direction
from .geo import offset_latlon, wrap360
from .schema import AUTO_MODE_LABELS

# Native tiles stop at 19; allowing zoom to 22 upscales them instead of going
# blank, which is what lets a 90 m track fill the viewport.
MAX_NATIVE_ZOOM = 19
MAX_ZOOM = 22

BASEMAPS = {
    "Satellite (Esri)": dict(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri"),
    "Street (OSM)": dict(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors"),
    "Light (Carto)": dict(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap, &copy; CARTO"),
    "Dark (Carto)": dict(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap, &copy; CARTO"),
}

LOG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9457c9",
              "#d64b8a", "#0f9bb5", "#c99a1e", "#6b7280"]

ARROW_COLORS = {
    "heading": "#eb6834",
    "cog": "#1baf7a",
    "wind_from": "#9457c9",
    "wind_to": "#c07fe0",
}

# Control mode is categorical, so it gets fixed colours rather than a ramp:
# MANUAL and AUTO must stay distinguishable at a glance, which is the whole
# point of colouring by it. Keyed by the names boatv1/utils/modes.py writes.
MODE_COLORS = {
    "MANUAL": "#2a78d6",
    "SIM_STEERED": "#0f9bb5",
    "HYBRID": "#c99a1e",
    "AUTO": "#1baf7a",
    "RESET": "#d03b3b",
    "": "#898781",          # column absent, or the row predates nav logging
}

# A mode the firmware gained after this app was written still draws, in a colour
# no listed mode uses, so it shows up as new rather than passing for grey.
UNKNOWN_MODE_COLOR = "#9457c9"


def build_map(logs, *, basemap="Satellite (Esri)", color_by="log",
              show_track=True, show_points=True, point_budget=3000,
              arrows=(), arrow_budget=150, arrow_len_m=6.0,
              convention="CW (clockwise, N=0 E=90)", wind_sign=1.0,
              moving_only=False, cursor=None, markers=(), fit_markers=False):
    """Assemble the Folium map for the currently selected logs and filters."""
    tracks = [lg for lg in logs if lg.has_track and len(lg.view) >= 1]
    m = _base_map(tracks, basemap,
                  [(mk.lat, mk.lon) for mk in markers] if fit_markers else ())
    _add_markers(m, markers)
    if not tracks:
        return m

    scale = _color_scale(tracks, color_by)
    modes_drawn = set()

    for i, lg in enumerate(tracks):
        base = LOG_COLORS[i % len(LOG_COLORS)]
        view = lg.view
        if moving_only and "moving" in view.columns:
            view = view[view["moving"]]
        if view.empty:
            continue
        if color_by == "mode":
            modes_drawn.update(_mode_values(view).tolist())

        group = folium.FeatureGroup(name=f"{lg.id}", show=True)

        if show_track:
            _add_track(group, view, color_by, base, scale)
        if show_points:
            _add_points(group, view, lg.id, color_by, base, scale, point_budget)
        for kind in arrows:
            _add_arrows(group, view, kind, arrow_budget, arrow_len_m,
                        convention, wind_sign)
        group.add_to(m)

    if cursor is not None:
        _add_cursor(m, cursor, convention, wind_sign)
    if scale is not None:
        scale.add_to(m)
    if color_by == "mode":
        _add_mode_legend(m, modes_drawn)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _base_map(tracks, basemap, extra_points=()):
    cfg = BASEMAPS.get(basemap, BASEMAPS["Satellite (Esri)"])
    m = folium.Map(location=[47.695, 9.194], zoom_start=17, tiles=None,
                   prefer_canvas=True, control_scale=True)

    folium.TileLayer(tiles=cfg["tiles"], attr=cfg["attr"], name=basemap,
                     max_native_zoom=MAX_NATIVE_ZOOM, max_zoom=MAX_ZOOM,
                     overlay=False, control=False).add_to(m)

    Fullscreen(position="topleft").add_to(m)
    MousePosition(position="bottomleft", separator=", ", num_digits=6,
                  prefix="lat, lon:", empty_string="").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="meters",
                   primary_area_unit="sqmeters").add_to(m)

    bounds = _bounds(tracks, extra_points)
    if bounds:
        m.fit_bounds(bounds, padding=(24, 24))
    return m


def _bounds(tracks, extra_points=()):
    """Extent of the tracks, optionally widened to take in loose points.

    Markers are only included on request. A mistyped coordinate would otherwise
    pull the view out to sea and shrink a 90 m track to a dot, with nothing on
    screen to explain why.
    """
    pts = [(lg.view["lat"].min(), lg.view["lon"].min(),
            lg.view["lat"].max(), lg.view["lon"].max())
           for lg in tracks if len(lg.view)]
    pts += [(lat, lon, lat, lon) for lat, lon in extra_points]
    if not pts:
        return None
    a = np.array(pts)
    return [[a[:, 0].min(), a[:, 1].min()], [a[:, 2].max(), a[:, 3].max()]]


# ---------------------------------------------------------------- colouring

CYCLIC_STEPS = ["#d94a4a", "#d98a2b", "#c9c02e", "#5fb84a", "#2eb5a5",
                "#3d8fd6", "#7a5fd0", "#c454b0", "#d94a4a"]


def _color_scale(tracks, color_by):
    """Return a branca colormap for continuous variables, or None.

    Categorical variables get their own discrete legend instead -- a continuous
    ramp over mode names would imply an ordering the values do not have.
    """
    if color_by in ("log", "heading_source", "mode", "none"):
        return None

    col = {"time": "t", "speed": "speed", "heading": "hdg", "awa": "awa"}.get(color_by)
    if col is None:
        return None

    if color_by in ("heading", "awa"):
        # Direction is circular: a linear ramp would invent a discontinuity
        # between 359 and 0 degrees, so the ramp wraps back to its start.
        return cm.LinearColormap(CYCLIC_STEPS, vmin=0, vmax=360,
                                 caption=f"{color_by} (deg, clockwise from N)")

    vals = pd.concat([lg.view[col] for lg in tracks if col in lg.view]).dropna()
    if vals.empty:
        return None

    if color_by == "time":
        lo, hi = vals.min().value / 1e9, vals.max().value / 1e9
        return cm.LinearColormap(["#cde2fb", "#5ba3e8", "#1d5fa8", "#0d366b"],
                                 vmin=lo, vmax=hi, caption="time (earlier to later)")

    hi = float(np.nanpercentile(vals, 98)) or 1.0
    return cm.LinearColormap(["#e8f2d8", "#9ccb6b", "#3f9a4a", "#0d5c3a"],
                             vmin=0, vmax=hi, caption="speed (m/s)")


def _values_for(view, color_by):
    if color_by == "time":
        return view["t"].astype("int64") / 1e9
    return view.get({"speed": "speed", "heading": "hdg", "awa": "awa"}.get(color_by))


def mode_color(value):
    return MODE_COLORS.get(str(value), UNKNOWN_MODE_COLOR)


def _mode_values(view):
    """Control mode per row as plain strings, "" where it was never logged."""
    if "mode" not in view.columns:
        return np.array([""] * len(view), dtype=object)
    col = view["mode"]
    return col.astype(object).where(col.notna(), "").astype(str).to_numpy()


def _point_colors(view, color_by, base, scale):
    if color_by == "mode":
        return [mode_color(v) for v in _mode_values(view)]
    if scale is None or color_by in ("log", "none"):
        return [base] * len(view)
    vals = _values_for(view, color_by)
    if vals is None:
        return [base] * len(view)
    return [base if not np.isfinite(v) else scale(v) for v in np.asarray(vals, dtype=float)]


def _add_mode_legend(m, modes):
    """Discrete swatch legend, in place of the colormap bar a ramp would add."""
    if not modes:
        return
    known = [k for k in MODE_COLORS if k in modes]
    ordered = known + sorted(v for v in modes if v not in MODE_COLORS)

    rows = "".join(
        "<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
        f"<span style='width:12px;height:12px;border-radius:2px;flex:none;"
        f"background:{mode_color(v)}'></span>"
        f"<span>{html.escape(v) if v else 'not recorded'}</span></div>"
        for v in ordered)

    # Fixed light card rather than theme-aware colours: this sits on top of a
    # satellite or dark basemap, neither of which follows the app's theme.
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:26px;right:12px;z-index:9999;"
        "background:rgba(255,255,255,0.92);color:#1a1a1a;font:12px system-ui;"
        "padding:8px 10px;border:1px solid rgba(0,0,0,0.25);border-radius:4px'>"
        "<div style='font-weight:600;margin-bottom:4px'>Control mode</div>"
        f"{rows}</div>"))


# ---------------------------------------------------------------- markers

# Symbols are inline SVG in a DivIcon rather than Leaflet's image pins or
# AwesomeMarkers: no extra CDN fetch, exact control of colour, and the shape
# scales cleanly when the imagery is zoomed past its native tile level.
MARKER_BOX = 18          # px; the symbol is centred on the coordinate
MARKER_SHAPES = {
    "circle": "<circle cx='9' cy='9' r='6'/>",
    "square": "<rect x='3.5' y='3.5' width='11' height='11' rx='1'/>",
    "diamond": "<path d='M9 2.5 L15.5 9 L9 15.5 L2.5 9 Z'/>",
    "triangle": "<path d='M9 2.5 L15.5 15 L2.5 15 Z'/>",
    "cross": "<path d='M4 4 L14 14 M14 4 L4 14' fill='none' stroke-width='3.4'/>",
    "pin": "<path d='M9 16.5 C9 16.5 3 10.6 3 7 A6 6 0 1 1 15 7 C15 10.6 9 16.5 9 16.5 Z'/>",
}


def _add_markers(m, markers):
    """User-placed reference points, in their own toggleable layer."""
    if not markers:
        return
    group = folium.FeatureGroup(name="Markers", show=True)
    for mk in markers:
        folium.Marker(
            [mk.lat, mk.lon],
            icon=folium.DivIcon(
                icon_size=(MARKER_BOX, MARKER_BOX),
                icon_anchor=(MARKER_BOX // 2, MARKER_BOX // 2),
                # Leaflet's own .leaflet-div-icon paints a white box and a grey
                # border behind whatever it holds, so the class is replaced
                # rather than extended.
                class_name="boat-marker",
                html=_marker_icon_html(mk)),
            tooltip=(f"{html.escape(mk.name)} &middot; {mk.type}<br>"
                     f"{mk.lat:.6f}, {mk.lon:.6f}"),
        ).add_to(group)
    group.add_to(m)


def _marker_icon_html(mk):
    shape = MARKER_SHAPES.get(mk.style.shape, MARKER_SHAPES["circle"])
    # The white stroke is what keeps every colour legible against satellite
    # imagery, which is dark in water and bright on a wake.
    symbol = (f"<svg width='{MARKER_BOX}' height='{MARKER_BOX}' "
              f"viewBox='0 0 {MARKER_BOX} {MARKER_BOX}' "
              f"fill='{mk.style.color}' stroke='#ffffff' stroke-width='1.6' "
              f"stroke-linejoin='round'>{shape}</svg>")

    # The label hangs outside the icon box, so the wrapper must not clip it. The
    # label also ignores the pointer: it can be wide, and swallowing clicks that
    # far from the symbol would block the map under it. The symbol itself stays
    # hoverable, which is what shows the tooltip.
    return (
        f"<div style='position:relative;width:{MARKER_BOX}px;"
        f"height:{MARKER_BOX}px;overflow:visible'>{symbol}"
        f"<span style='position:absolute;left:{MARKER_BOX + 3}px;top:1px;"
        f"white-space:nowrap;pointer-events:none;font:600 11px/1.35 system-ui;"
        f"padding:0 5px;border-radius:3px;background:rgba(255,255,255,0.92);"
        f"color:#1a1a1a;border:1px solid rgba(0,0,0,0.25)'>"
        f"{html.escape(mk.name)}</span></div>")


# ---------------------------------------------------------------- layers

TRACK_VERTEX_CAP = 12000


def _add_track(group, view, color_by, base, scale):
    view = _decimate(view, TRACK_VERTEX_CAP)
    coords = view[["lat", "lon"]].to_numpy()
    if len(coords) < 2:
        return

    codes, color_of, tips = _track_classes(view, color_by, scale)
    if codes is None:
        folium.PolyLine(coords.tolist(), color=base, weight=2.5, opacity=0.85,
                        tooltip="track").add_to(group)
        return

    start = 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            # Runs share their boundary vertex, so consecutive stretches join up
            # instead of leaving a one-segment hole between them -- and a run of
            # a single sample, such as a brief RESET, still draws.
            stop = min(i + 1, len(coords))
            if codes[start] >= 0 and stop - start >= 2:
                folium.PolyLine(coords[start:stop].tolist(),
                                color=color_of(codes[start]), weight=2.5,
                                opacity=0.9, tooltip=tips(codes[start])).add_to(group)
            start = i


def _track_classes(view, color_by, scale):
    """Per-vertex class code, plus colour and tooltip for a code.

    Returns (None, None, None) when the track should be drawn in one flat
    colour. Continuous variables are quantised so the track becomes a handful of
    polylines rather than one per segment; categorical ones use their own values.
    """
    if color_by == "mode":
        values = _mode_values(view)
        index = {v: i for i, v in enumerate(dict.fromkeys(values))}
        names = {i: v for v, i in index.items()}
        codes = np.array([index[v] for v in values], dtype=int)
        return (codes,
                lambda c: mode_color(names[c]),
                lambda c: f"mode {names[c] or 'not recorded'}")

    if scale is None or color_by in ("log", "none"):
        return None, None, None

    vals = _values_for(view, color_by)
    if vals is None:
        return None, None, None

    vals = np.asarray(vals, dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        return None, None, None

    lo, hi = np.nanmin(vals[finite]), np.nanmax(vals[finite])
    span = (hi - lo) or 1.0
    # Non-finite entries are cast from a stand-in rather than from NaN, which
    # has no defined integer value, then marked -1 so they are skipped.
    codes = np.clip(((np.where(finite, vals, lo) - lo) / span * 11).astype(int), 0, 11)
    codes[~finite] = -1
    return codes, lambda c: scale(lo + (c + 0.5) / 12 * span), lambda c: color_by


# Above this many points, per-marker popups dominate the generated HTML (each
# costs roughly 2 KB), so detail collapses into the tooltip instead.
POPUP_LIMIT = 600


def _add_points(group, view, log_id, color_by, base, scale, budget):
    view = _decimate(view, budget)
    colors = _point_colors(view, color_by, base, scale)
    rich = len(view) <= POPUP_LIMIT

    for (_, r), c in zip(view.iterrows(), colors):
        marker = folium.CircleMarker(
            [r["lat"], r["lon"]], radius=2.5, color=c, weight=1,
            fill=True, fill_color=c, fill_opacity=0.85,
            tooltip=_tooltip(r))
        if rich:
            marker.add_child(folium.Popup(_popup_html(r, log_id), max_width=280))
        marker.add_to(group)


def _fmt(v, unit="", digits=1, dash="--"):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return dash
    return f"{v:.{digits}f}{unit}"


def _cell(r, name):
    """A row's value as a string, empty for missing or NaN.

    `or ""` alone will not do it: NaN is truthy, so a missing cell would render
    as the literal text "nan".
    """
    v = r.get(name)
    return "" if v is None or pd.isna(v) else str(v)


def _mode_text(r):
    """Control mode, qualified by the AUTO sub-mode when there is one."""
    mode = _cell(r, "mode")
    if not mode:
        return ""
    auto = _cell(r, "auto_mode")
    return f"{mode} · {AUTO_MODE_LABELS.get(auto, auto)}" if auto else mode


def _tooltip(r):
    parts = [f"row {int(r['row'])}", pd.Timestamp(r["t"]).strftime("%H:%M:%S")]
    mode = _mode_text(r)
    if mode:
        parts.append(mode)
    if np.isfinite(r.get("hdg", np.nan)):
        parts.append(f"hdg {_fmt(r['hdg'], '°', 0)}")
    if np.isfinite(r.get("speed", np.nan)):
        parts.append(f"{_fmt(r['speed'], ' m/s', 2)}")
    return " · ".join(parts)


def _popup_html(r, log_id):
    def row(label, value):
        shade = " style='color:#898781'" if value == "--" else ""
        return f"<tr><td>{label}</td><td{shade}><b>{value}</b></td></tr>"

    return (
        f"<div style='font:12px system-ui'><b>{log_id}</b><br>"
        f"<span style='color:#898781'>row {int(r['row'])} &middot; "
        f"{pd.Timestamp(r['t']).strftime('%Y-%m-%d %H:%M:%S')} UTC</span>"
        f"<table style='margin-top:6px;border-spacing:6px 2px'>"
        + row("lat, lon", f"{r['lat']:.6f}, {r['lon']:.6f}")
        + row("mode", html.escape(_mode_text(r)) or "--")
        + row("heading", _fmt(r.get("hdg"), "&deg;"))
        + row("course", _fmt(r.get("cog"), "&deg;"))
        + row("hdg - cog", _fmt(r.get("resid"), "&deg;"))
        + row("AWA", _fmt(r.get("awa"), "&deg;"))
        + row("speed", _fmt(r.get("speed"), " m/s", 2))
        + "</table></div>")


def _add_arrows(group, view, kind, budget, length_m, convention, wind_sign):
    """Draw direction arrows, spaced by distance travelled rather than by index.

    Index spacing would clump every arrow where the boat sat still, which is
    most of these logs.
    """
    angles = _arrow_angles(view, kind, convention, wind_sign)
    if angles is None:
        return

    sel = _space_by_distance(view, angles, budget)
    if sel.empty:
        return

    color = ARROW_COLORS.get(kind, "#eb6834")
    lat, lon = sel["lat"].to_numpy(), sel["lon"].to_numpy()
    th = sel["_angle"].to_numpy()

    tip_lat, tip_lon = offset_latlon(lat, lon, th, length_m)
    bl_lat, bl_lon = offset_latlon(tip_lat, tip_lon, th + 150.0, length_m * 0.35)
    br_lat, br_lon = offset_latlon(tip_lat, tip_lon, th - 150.0, length_m * 0.35)

    label = kind.replace("_", " ")
    for i in range(len(sel)):
        folium.PolyLine([[lat[i], lon[i]], [tip_lat[i], tip_lon[i]]],
                        color=color, weight=2.5, opacity=0.9,
                        tooltip=f"{label} {th[i]:.0f}&deg;").add_to(group)
        folium.PolyLine([[bl_lat[i], bl_lon[i]], [tip_lat[i], tip_lon[i]],
                         [br_lat[i], br_lon[i]]],
                        color=color, weight=2, opacity=0.9).add_to(group)


def _arrow_angles(view, kind, convention, wind_sign):
    """Angle each arrow should point, as a clockwise compass bearing."""
    acw = convention.startswith("ACW")

    if kind == "heading":
        a = view["hdg"].to_numpy(dtype=float)
        return wrap360(360.0 - a) if acw else wrap360(a)
    if kind == "cog":
        return view["cog"].to_numpy(dtype=float) if "cog" in view else None
    if kind in ("wind_from", "wind_to"):
        h = view["hdg"].to_numpy(dtype=float)
        if acw:
            h = wrap360(360.0 - h)
        w = wind_direction(h, view["awa"].to_numpy(dtype=float), wind_sign)
        return wrap360(w + 180.0) if kind == "wind_to" else w
    return None


def _space_by_distance(view, angles, budget):
    v = view.copy()
    v["_angle"] = angles
    v = v[np.isfinite(v["_angle"])]
    if v.empty or budget <= 0:
        return v.iloc[0:0]
    if len(v) <= budget:
        return v

    lat, lon = v["lat"].to_numpy(), v["lon"].to_numpy()
    m_lat = 111132.0
    m_lon = 111320.0 * np.cos(np.radians(lat.mean()))
    step = np.hypot(np.diff(lat) * m_lat, np.diff(lon) * m_lon)
    cumulative = np.concatenate([[0.0], np.cumsum(step)])

    total = cumulative[-1]
    if total <= 0:
        return v.iloc[:: max(1, len(v) // budget)]
    targets = np.linspace(0, total, budget)
    idx = np.unique(np.searchsorted(cumulative, targets))
    return v.iloc[np.clip(idx, 0, len(v) - 1)]


def _decimate(view, budget):
    if budget <= 0 or len(view) <= budget:
        return view
    return view.iloc[np.linspace(0, len(view) - 1, budget).astype(int)]


def _add_cursor(m, cursor, convention, wind_sign):
    """Boat marker at the current time, with heading and wind indicators."""
    lat, lon = cursor["lat"], cursor["lon"]
    folium.CircleMarker([lat, lon], radius=7, color="#ffffff", weight=2,
                        fill=True, fill_color="#d03b3b", fill_opacity=1.0,
                        tooltip="current position").add_to(m)

    hdg = cursor.get("hdg")
    if hdg is not None and np.isfinite(hdg):
        th = 360.0 - hdg if convention.startswith("ACW") else hdg
        tip = offset_latlon(lat, lon, th, 12.0)
        folium.PolyLine([[lat, lon], [float(tip[0]), float(tip[1])]],
                        color="#d03b3b", weight=3.5, opacity=1.0,
                        tooltip=f"heading {hdg:.0f}&deg;").add_to(m)

        awa = cursor.get("awa")
        if awa is not None and np.isfinite(awa):
            wf = float(wind_direction(th, awa, wind_sign))
            tip = offset_latlon(lat, lon, wf, 10.0)
            folium.PolyLine([[lat, lon], [float(tip[0]), float(tip[1])]],
                            color="#9457c9", weight=3, opacity=0.95, dash_array="4",
                            tooltip=f"wind from {wf:.0f}&deg;").add_to(m)
