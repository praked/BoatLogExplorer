"""Speed, course over ground, and heading residuals.

Everything here runs on deduplicated fixes, never on raw rows: at ~10 Hz logging
against a ~1 Hz receiver, consecutive raw rows are usually the same fix and any
speed computed from them is meaningless.
"""

import numpy as np
import pandas as pd

from .geo import ang_diff, bearing_deg, circ_mean_R, haversine_m, wrap180, wrap360

DEFAULT_SPEED_WINDOW_S = 5.0
DEFAULT_D_MIN_M = 3.0
DEFAULT_COG_MAX_WINDOW_S = 60.0
DEFAULT_V_MAX_MS = 8.0

# Below this speed the receiver's own jitter dominates: a stationary boat still
# wanders several metres per minute, which is enough to satisfy any displacement
# threshold while carrying no directional information at all. Course over ground
# is only defined above it.
DEFAULT_MIN_SPEED_MS = 0.5

CONVENTIONS = {
    "COG": lambda c: c,
    "90 - COG": lambda c: 90.0 - c,
    "180 - COG": lambda c: 180.0 - c,
    "-COG": lambda c: -c,
    "COG + 90": lambda c: c + 90.0,
    "COG + 180": lambda c: c + 180.0,
}


def _epoch_seconds(series):
    """Seconds since the epoch, without the tz-stripping warning numpy emits
    when a tz-aware column is cast to datetime64 directly."""
    s = series.dt.tz_convert("UTC").dt.tz_localize(None) \
        if isinstance(series.dtype, pd.DatetimeTZDtype) else series
    return s.to_numpy(dtype="datetime64[ms]").astype(np.float64) / 1000.0


def enrich(log, speed_window_s=DEFAULT_SPEED_WINDOW_S, d_min_m=DEFAULT_D_MIN_M,
           cog_max_window_s=DEFAULT_COG_MAX_WINDOW_S, v_max_ms=DEFAULT_V_MAX_MS,
           min_speed_ms=DEFAULT_MIN_SPEED_MS):
    """Add speed, outlier, moving, cog and residual columns to log.fix in place."""
    fix = log.fix
    if len(fix) < 2:
        for c in ("speed", "cog", "resid"):
            fix[c] = np.nan
        for c in ("outlier", "moving"):
            fix[c] = False
        return log

    lat = fix["lat"].to_numpy()
    lon = fix["lon"].to_numpy()
    ts = _epoch_seconds(fix["t"])

    fix["outlier"] = _flag_outliers(lat, lon, ts, v_max_ms)
    keep = ~fix["outlier"].to_numpy()

    speed = _windowed_speed(lat, lon, ts, keep, speed_window_s)
    fix["speed"] = speed
    fix["moving"] = np.nan_to_num(speed, nan=0.0) >= min_speed_ms

    cog = _adaptive_cog(lat, lon, ts, keep, d_min_m, cog_max_window_s)
    cog[~fix["moving"].to_numpy()] = np.nan
    fix["cog"] = cog
    fix["resid"] = wrap180(fix["hdg"].to_numpy(dtype=float) - cog)
    return log


def _flag_outliers(lat, lon, ts, v_max):
    """Flag fixes reachable only by an implausible jump.

    A single bad fix produces a large jump in and a large jump out while leaving
    the neighbours close together, so both conditions are required.
    """
    n = len(lat)
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out

    d = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.diff(ts)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 0, d / dt, np.inf)

    fast_in, fast_out = v[:-1] > v_max, v[1:] > v_max
    span = haversine_m(lat[:-2], lon[:-2], lat[2:], lon[2:])
    out[1:-1] = fast_in & fast_out & (span < d[:-1] * 0.5)

    # An isolated leading or trailing spike has only one neighbour to test.
    if v[0] > v_max and not out[1]:
        out[0] = True
    if v[-1] > v_max and not out[-2]:
        out[-1] = True
    return out


def _windowed_speed(lat, lon, ts, keep, window_s):
    """Speed over a centred time window, so GPS jitter averages out."""
    n = len(lat)
    speed = np.full(n, np.nan)
    idx = np.flatnonzero(keep)
    if idx.size < 2:
        return speed

    t = ts[idx]
    half = window_s / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right") - 1

    span = t[hi] - t[lo]
    ok = span >= min(3.0, window_s * 0.6)
    if not ok.any():
        return speed

    a, b = idx[lo[ok]], idx[hi[ok]]
    dist = haversine_m(lat[a], lon[a], lat[b], lon[b])
    speed[idx[ok]] = dist / span[ok]
    return speed


def _adaptive_cog(lat, lon, ts, keep, d_min, max_window_s):
    """Course over ground, measured over however long it takes to move d_min metres.

    A fixed time window is useless here: 5 seconds at the observed 0.14 m/s
    jitter spans 0.7 m, which is pure noise. Instead the window grows until the
    boat has actually gone somewhere, and stays NaN if it never does.
    """
    n = len(lat)
    cog = np.full(n, np.nan)
    idx = np.flatnonzero(keep)
    if idx.size < 2:
        return cog

    la, lo_, t = lat[idx], lon[idx], ts[idx]
    m = idx.size
    j = 0
    for i in range(m):
        if j <= i:
            j = i + 1
        while j < m:
            if t[j] - t[i] > max_window_s:
                break
            if haversine_m(la[i], lo_[i], la[j], lo_[j]) >= d_min:
                break
            j += 1
        if j >= m or t[j] - t[i] > max_window_s:
            continue
        if haversine_m(la[i], lo_[i], la[j], lo_[j]) < d_min:
            continue
        cog[idx[i]] = bearing_deg(la[i], lo_[i], la[j], lo_[j])
    return cog


MIN_COG_SAMPLES = 50
CONCLUSIVE_R = 0.35          # concentration the winning candidate must reach
CONCLUSIVE_MARGIN_DEG = 15.0  # median-error gap the winner must open over second place


def convention_scoreboard(fix, heading_col="hdg", heading_alive=True, min_samples=MIN_COG_SAMPLES):
    """Score candidate heading conventions against GPS course over ground.

    Returns (table, verdict, detail). The table is ranked by median absolute
    error; a compass that agrees with the direction of travel scores far below
    the 90-degree random baseline.

    Two things can make the comparison meaningless, and both are reported rather
    than papered over: a frozen compass has no directional content to test, and
    a boat that never moved has no course to test it against.
    """
    if not heading_alive:
        return pd.DataFrame(), "unusable", (
            "The compass is frozen in this log, so there is no heading variation "
            "to compare against course. Nothing can be concluded here.")

    if "cog" not in fix.columns:
        return pd.DataFrame(), "unusable", "No course-over-ground was derived."

    d = fix[[heading_col, "cog"]].dropna()
    n = len(d)
    if n < min_samples:
        return pd.DataFrame(), "insufficient", (
            f"Only {n} fixes had both a live heading and real motion "
            f"({min_samples} needed). The boat barely moved, so the compass "
            "cannot be assessed.")

    h, c = d[heading_col].to_numpy(dtype=float), d["cog"].to_numpy(dtype=float)
    rows = []
    for name, fn in CONVENTIONS.items():
        err = np.abs(ang_diff(fn(c), h))
        offset, R = circ_mean_R(ang_diff(h, fn(c)))
        rows.append({
            "convention": name,
            "n": n,
            "mean_abs_err": float(err.mean()),
            "median_abs_err": float(np.median(err)),
            "circ_offset": float(wrap180(offset)),
            "R": R,
        })
    table = pd.DataFrame(rows).sort_values("median_abs_err").reset_index(drop=True)

    best, second = table.iloc[0], table.iloc[1]
    margin = second["median_abs_err"] - best["median_abs_err"]
    if best["R"] >= CONCLUSIVE_R and margin >= CONCLUSIVE_MARGIN_DEG:
        verdict, detail = "conclusive", (
            f"**{best['convention']}** wins with a median error of "
            f"{best['median_abs_err']:.1f}° over {n} samples, beating the next "
            f"candidate by {margin:.0f}°. Against a 90° random baseline this is "
            f"a real agreement between compass and course.")
    else:
        verdict, detail = "inconclusive", (
            f"No candidate separates cleanly (best median "
            f"{best['median_abs_err']:.1f}°, concentration R={best['R']:.2f}, "
            f"margin {margin:.0f}°). The apparent motion is likely receiver "
            f"jitter rather than travel — treat this ranking as noise.")
    return table, verdict, detail


def wind_direction(heading, awa, sign=1.0):
    """Global direction the apparent wind blows FROM, clockwise from North."""
    return wrap360(np.asarray(heading, dtype=float) + sign * np.asarray(awa, dtype=float))


def wind_sign_evidence(fix, heading_alive=True, awa_alive=True):
    """Compare the two AWA sign conventions by how well each yields a steady wind.

    Over a short window the true wind direction should be roughly constant, so
    the correct sign produces a more concentrated set of derived directions.
    This only means anything if both inputs vary: if the compass and the vane
    are both stuck, every convention trivially yields R=1 and proves nothing.
    """
    if not (heading_alive and awa_alive):
        return {}, "unusable", (
            "Both a live compass and a live wind vane are needed to test the "
            "sign convention. At least one is frozen in this log.")

    d = fix[["hdg", "awa"]].dropna()
    if len(d) < 50:
        return {}, "insufficient", f"Only {len(d)} samples with both channels."

    h, a = d["hdg"].to_numpy(dtype=float), d["awa"].to_numpy(dtype=float)
    out = {}
    for label, sign in (("heading + AWA", 1.0), ("heading - AWA", -1.0)):
        mean, R = circ_mean_R(wind_direction(h, a, sign))
        out[label] = {"mean_dir": mean, "R": R, "n": len(d)}

    plus, minus = out["heading + AWA"]["R"], out["heading - AWA"]["R"]
    better = "heading + AWA" if plus >= minus else "heading - AWA"
    if max(plus, minus) < 0.30:
        verdict, detail = "weak", (
            f"{better} is the better of the two (R={max(plus, minus):.3f} vs "
            f"{min(plus, minus):.3f}), but neither produces a convincingly steady "
            "wind direction. Treat wind overlays as indicative only.")
    else:
        verdict, detail = "supported", (
            f"{better} produces a markedly steadier wind direction "
            f"(R={max(plus, minus):.3f} vs {min(plus, minus):.3f}).")
    return out, verdict, detail
