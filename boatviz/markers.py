"""Named reference points the user places on the map by hand.

A track passing 4 m from a buoy and one passing 40 m from it look identical
without something to measure against, so the app lets you type the buoy in.

Nothing here imports Streamlit: the app owns the sheet widget, this module owns
what comes out of it -- validation, the CSV round-trip, and how near the boat
actually got.
"""

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import haversine_m


@dataclass(frozen=True)
class MarkerType:
    label: str
    shape: str
    color: str


# Colours are drawn from the palette the map already uses, so a marker never
# reads as a different visual system from the tracks under it. Shape carries the
# same distinction independently, which keeps the types apart for anyone who
# cannot separate the hues.
MARKER_TYPES = {t.label: t for t in (
    MarkerType("Waypoint", "diamond", "#2a78d6"),
    MarkerType("Buoy", "circle", "#eb6834"),
    MarkerType("Hazard", "cross", "#d03b3b"),
    MarkerType("Start", "triangle", "#1baf7a"),
    MarkerType("Finish", "square", "#9457c9"),
    MarkerType("Note", "pin", "#6b7280"),
)}

DEFAULT_TYPE = "Waypoint"

# Matches point_to_point_tolerance_m in every boat_config_*.json, so the ring
# drawn here is the same circle the autopilot retires a waypoint inside. Station
# keeping uses 25 m, which is why this is per-marker rather than one setting.
DEFAULT_RADIUS_M = 10.0

# Types are matched case-insensitively and stored canonically: a file written by
# hand or by another tool says "buoy", and silently demoting that to a waypoint
# would lose information the user did supply.
TYPE_LOOKUP = {label.lower(): label for label in MARKER_TYPES}

COLUMNS = ["name", "lat", "lon", "type", "wp_radius"]

# Spreadsheets and chartplotters disagree about what the coordinate columns are
# called, and the file is meant to be editable outside this app.
COLUMN_ALIASES = {
    "name": ("name", "label", "id"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude", "lng", "long"),
    "type": ("type", "kind", "symbol"),
    "wp_radius": ("wp_radius", "radius", "wp_radius_m", "radius_m", "tolerance_m"),
}


@dataclass(frozen=True)
class Marker:
    name: str
    lat: float
    lon: float
    type: str = DEFAULT_TYPE
    wp_radius: float = DEFAULT_RADIUS_M

    @property
    def style(self) -> MarkerType:
        return MARKER_TYPES.get(self.type, MARKER_TYPES[DEFAULT_TYPE])


def empty_frame() -> pd.DataFrame:
    """Correctly typed empty sheet. Dtypes matter: they decide what the editor
    offers in each cell, and an all-object frame gives text boxes for latitude."""
    return pd.DataFrame({"name": pd.Series(dtype="object"),
                         "lat": pd.Series(dtype="float64"),
                         "lon": pd.Series(dtype="float64"),
                         "type": pd.Series(dtype="object"),
                         "wp_radius": pd.Series(dtype="float64")})


def valid_rows(frame):
    """(markers, complaints) from a sheet.

    Wholly blank rows are ignored rather than reported: the editor keeps one for
    typing into, and a row is incomplete for a few keystrokes on the way to being
    finished. Only a row carrying something but not enough is worth a complaint.
    """
    markers, problems = [], []
    if frame is None or not len(frame):
        return markers, problems

    for n, (_, r) in enumerate(frame.iterrows(), start=1):
        name = _text(r.get("name"))
        lat, lon = _number(r.get("lat")), _number(r.get("lon"))

        if not name and lat is None and lon is None:
            continue
        if not name:
            problems.append(f"Row {n} has coordinates but no name.")
            continue
        if lat is None or lon is None:
            problems.append(f"Row {n} (**{name}**) needs both a latitude and a "
                            f"longitude.")
            continue
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            problems.append(f"Row {n} (**{name}**): {lat}, {lon} is not a "
                            f"position on Earth.")
            continue

        # A blank radius means "the usual one", not "no ring" -- zero is how you
        # ask for no ring, and a negative number is a mistake worth naming.
        radius = _number(r.get("wp_radius"))
        if radius is None:
            radius = DEFAULT_RADIUS_M
        elif radius < 0:
            problems.append(f"Row {n} (**{name}**): a radius cannot be "
                            f"negative. Using {DEFAULT_RADIUS_M:g} m.")
            radius = DEFAULT_RADIUS_M

        markers.append(Marker(name, lat, lon, _type(r.get("type")), radius))
    return markers, problems


def parse_csv(data) -> tuple[pd.DataFrame, list[str]]:
    """Read a markers file back into a sheet. Returns (frame, problems).

    Rows with unusable coordinates are kept, not dropped -- they come back as
    empty cells so they can be fixed in place, and valid_rows says what is wrong
    with each. Only structural trouble is reported here.
    """
    try:
        source = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
        raw = pd.read_csv(source)
    except Exception as exc:                                       # noqa: BLE001
        return empty_frame(), [f"Could not read that file: {exc}"]

    lower = {str(c).strip().lower(): c for c in raw.columns}
    found = {role: next((lower[a] for a in aliases if a in lower), None)
             for role, aliases in COLUMN_ALIASES.items()}

    missing = [role for role in ("name", "lat", "lon") if found[role] is None]
    if missing:
        return empty_frame(), [
            f"That file has no {', '.join(missing)} column. Expected columns "
            f"{', '.join(COLUMNS)} in any order."]

    frame = empty_frame()
    frame["name"] = raw[found["name"]].astype(object)
    frame["lat"] = pd.to_numeric(raw[found["lat"]], errors="coerce")
    frame["lon"] = pd.to_numeric(raw[found["lon"]], errors="coerce")
    frame["wp_radius"] = (
        pd.to_numeric(raw[found["wp_radius"]], errors="coerce")
        .fillna(DEFAULT_RADIUS_M) if found["wp_radius"] is not None
        else DEFAULT_RADIUS_M)

    problems = []
    if found["type"] is None:
        frame["type"] = DEFAULT_TYPE
    else:
        given = raw[found["type"]].map(_text)
        frame["type"] = given.map(_type)
        unknown = sorted({g for g in given if g and g.lower() not in TYPE_LOOKUP})
        if unknown:
            problems.append(
                f"Unrecognised type(s) {', '.join(unknown)} were set to "
                f"{DEFAULT_TYPE}. Known types: {', '.join(MARKER_TYPES)}.")

    return frame.reset_index(drop=True), problems


def to_csv(markers) -> bytes:
    frame = pd.DataFrame([(m.name, m.lat, m.lon, m.type, m.wp_radius)
                          for m in markers], columns=COLUMNS)
    return frame.to_csv(index=False).encode()


def closest_approach(markers, logs) -> pd.DataFrame:
    """How near the boat got to each marker, and when.

    Measured against `view`, the fixes inside the selected time window, so the
    answer follows the time range rather than the whole recording.
    """
    rows = []
    for mk in markers:
        best = None
        for lg in logs:
            view = getattr(lg, "view", None)
            if view is None or not len(view):
                continue
            d = haversine_m(mk.lat, mk.lon,
                            view["lat"].to_numpy(), view["lon"].to_numpy())
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (float(d[i]), view["t"].iloc[i], lg.id)

        rows.append({
            "Marker": mk.name,
            "Type": mk.type,
            "Closest approach (m)": best[0] if best else np.nan,
            "Radius (m)": mk.wp_radius,
            # The ring is an arrival test, so the table says whether it passed.
            "Inside radius": ("–" if not best or mk.wp_radius <= 0
                              else "yes" if best[0] <= mk.wp_radius else "no"),
            "When (UTC)": pd.Timestamp(best[1]).strftime("%b %d %H:%M:%S") if best else "–",
            "Log": best[2].replace("boat_log_", "") if best else "–",
        })
    return pd.DataFrame(rows)


def _text(v) -> str:
    return "" if v is None or pd.isna(v) else str(v).strip()


def _number(v):
    if v is None or pd.isna(v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _type(v) -> str:
    return TYPE_LOOKUP.get(_text(v).lower(), DEFAULT_TYPE)
