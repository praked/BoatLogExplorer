"""Load boat CSV logs into a compact in-memory form.

Ingest does three things: parse, reject invalid GPS, and collapse the ~89% of
GPS rows that merely repeat the previous fix (the logger samples at ~10 Hz
against a ~1 Hz receiver).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import haversine_m
from .schema import detect_columns

# Latitude and longitude must stay float64. At 47 deg N a float32 mantissa
# quantizes position to ~0.6 m, which silently merges genuinely distinct GPS
# fixes during deduplication.
GPS_COLUMNS = ["gps_lat", "gps_lon"]
SCALAR_COLUMNS = ["awa_deg", "heading", "rotation_vector", "geomag_rotation_vector",
                  "gyro_z_dps", "sys_temp_c", "sys_volts_v"]
NUMERIC_COLUMNS = GPS_COLUMNS + SCALAR_COLUMNS

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_FRAGMENT = "fragment"
STATUS_NO_GPS = "no_gps"
STATUS_ERROR = "error"


@dataclass
class Log:
    id: str
    path: Path
    size: int
    status: str
    message: str = ""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    fix: pd.DataFrame = field(default_factory=pd.DataFrame)
    qa: object = None

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_fixes(self) -> int:
        return len(self.fix)

    @property
    def has_track(self) -> bool:
        return self.status == STATUS_OK and len(self.fix) >= 2

    @property
    def t_start(self):
        return self.df["t"].iloc[0] if len(self.df) else None

    @property
    def t_end(self):
        return self.df["t"].iloc[-1] if len(self.df) else None

    @property
    def duration_s(self) -> float:
        if len(self.df) < 2:
            return 0.0
        return (self.df["t"].iloc[-1] - self.df["t"].iloc[0]).total_seconds()

    def bbox(self):
        """(min_lat, max_lat, min_lon, max_lon, width_m, height_m)."""
        if not len(self.fix):
            return None
        la, lo = self.fix["lat"].to_numpy(), self.fix["lon"].to_numpy()
        mnla, mxla, mnlo, mxlo = la.min(), la.max(), lo.min(), lo.max()
        h = haversine_m(mnla, mnlo, mxla, mnlo)
        w = haversine_m(mnla, mnlo, mnla, mxlo)
        return mnla, mxla, mnlo, mxlo, float(w), float(h)


def discover_logs(folder) -> list[Path]:
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.csv") if not p.name.startswith("."))


def load_log(source, log_id: str | None = None, size: int | None = None) -> Log:
    """Parse one log.

    `source` is a filesystem path or any file-like object with a name, so the
    same code path serves both a local folder and a browser upload.
    """
    if hasattr(source, "read"):
        name = getattr(source, "name", "uploaded.csv")
        path = Path(name)
        lid = log_id or path.stem
        size = size if size is not None else getattr(source, "size", None)
        if size is None:
            pos = source.tell()
            source.seek(0, 2)
            size = source.tell()
            source.seek(pos)
    else:
        path = Path(source)
        lid = log_id or path.stem
        size = path.stat().st_size if path.exists() else 0

    if not size:
        return Log(lid, path, size or 0, STATUS_EMPTY, "empty file (0 bytes)")

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        raw = pd.read_csv(source, engine="c", low_memory=False)
    except Exception as exc:                                   # noqa: BLE001
        return Log(lid, path, size, STATUS_ERROR, f"parse failed: {exc}")

    if raw.empty:
        return Log(lid, path, size, STATUS_EMPTY, "no data rows")

    cols = detect_columns(raw.columns)
    if "ts_iso" not in cols:
        return Log(lid, path, size, STATUS_ERROR, "no timestamp column found")

    df = pd.DataFrame(index=range(len(raw)))
    df["row"] = np.arange(len(raw), dtype=np.int64)
    df["t"] = pd.to_datetime(raw[cols["ts_iso"]], format="ISO8601", utc=True, errors="coerce")

    for name in NUMERIC_COLUMNS:
        src = cols.get(name)
        dtype = "float64" if name in GPS_COLUMNS else "float32"
        df[name] = (pd.to_numeric(raw[src], errors="coerce").astype(dtype)
                    if src is not None else np.array(np.nan, dtype=dtype))

    # Deliberately category rather than the nullable "string" dtype: this frame
    # gets pickled by Streamlit's cache, and StringDtype does not survive that
    # round-trip on older pandas builds (notably the one bundled with Pyodide,
    # which raises NotImplementedError on unpickling). heading_source holds only
    # a handful of distinct values, so category is also far smaller.
    src = cols.get("heading_source")
    values = (raw[src].astype(object).where(raw[src].notna(), "")
              if src is not None else pd.Series("", index=raw.index))
    df["heading_source"] = pd.Series(values.to_numpy(), index=df.index).astype("category")

    df = df[df["t"].notna()].reset_index(drop=True)
    if df.empty:
        return Log(lid, path, size, STATUS_ERROR, "no parseable timestamps")

    monotonic_breaks = int((df["t"].diff() < pd.Timedelta(0)).sum())
    fix = _build_fixes(df)

    if len(df) < 2:
        return Log(lid, path, size, STATUS_FRAGMENT, f"{len(df)} row", df=df, fix=fix)

    log = Log(lid, path, size, STATUS_OK, df=df, fix=fix)
    log.monotonic_breaks = monotonic_breaks
    log.n_gps_rows = int(_valid_gps_mask(df).sum())

    if len(fix) < 2:
        log.status = STATUS_NO_GPS
        log.message = "no usable GPS fixes"
    elif len(df) < 800:
        log.status = STATUS_FRAGMENT
        log.message = f"{len(df)} rows"
    return log


def _valid_gps_mask(df: pd.DataFrame) -> pd.Series:
    lat, lon = df["gps_lat"], df["gps_lon"]
    return (lat.notna() & lon.notna()
            & ~((lat == 0) & (lon == 0))
            & lat.between(-90, 90) & lon.between(-180, 180))


def _build_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate consecutive identical GPS fixes, keeping the first occurrence.

    The first occurrence is closest to the true fix time; later repeats are just
    the logger re-reading a stale value. dwell records how long each fix persisted.
    """
    valid = _valid_gps_mask(df)
    g = df.loc[valid, ["row", "t", "gps_lat", "gps_lon", "heading", "awa_deg",
                       "geomag_rotation_vector", "gyro_z_dps", "heading_source"]]
    if g.empty:
        return pd.DataFrame(columns=["row", "t", "lat", "lon", "hdg", "awa",
                                     "geomag", "gyro", "source", "dwell"])

    changed = (g["gps_lat"] != g["gps_lat"].shift()) | (g["gps_lon"] != g["gps_lon"].shift())
    fix = g[changed].copy()
    fix = fix.rename(columns={"gps_lat": "lat", "gps_lon": "lon", "heading": "hdg",
                              "awa_deg": "awa", "geomag_rotation_vector": "geomag",
                              "gyro_z_dps": "gyro", "heading_source": "source"})
    fix["lat"] = fix["lat"].astype("float64")
    fix["lon"] = fix["lon"].astype("float64")

    t = fix["t"].to_numpy()
    dwell = np.diff(t).astype("timedelta64[ms]").astype(np.float64) / 1000.0
    fix["dwell"] = np.append(dwell, np.nan).astype("float32")
    return fix.reset_index(drop=True)
