"""Speed, course over ground, and heading residuals.

Everything here runs on deduplicated fixes, never on raw rows: at ~10 Hz logging
against a ~1 Hz receiver, consecutive raw rows are usually the same fix and any
speed computed from them is meaningless.
"""

import numpy as np
import pandas as pd

from .geo import ang_diff, bearing_deg, circ_mean_R, haversine_m, wrap180, wrap360
from .schema import POWER_CHANNELS, SAIL_FAULT_OK

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


# ----------------------------------------------------------------------- power

# Power runs on raw rows, not on deduplicated fixes: consumption is a property
# of the electrical system and keeps changing while the GPS repeats a fix.

# Longer than this between rows means the logger stopped, not that the load held
# steady, so the energy integral drops the gap rather than charging the whole
# interval to the last sample. Rows land ~9 Hz apart, so this is generous.
MAX_POWER_GAP_S = 5.0

# The shunt reads a few milliamps of noise with nothing connected. Below this a
# channel is treated as unloaded, which keeps implied bus voltage -- power over
# current -- from being computed by dividing noise by noise.
CURRENT_NOISE_A = 0.005


def power_summary(df, channels=POWER_CHANNELS):
    """Per-channel consumption over the rows handed in.

    One dict per channel, always all of them and in order, so a channel that is
    wired but idle stays visible instead of vanishing from the table. `present`
    distinguishes "logged nothing" from "logged zero" -- the first means this
    firmware or run had no monitor, the second means nothing drew current.
    """
    out = []
    ts = _epoch_seconds(df["t"]) if len(df) else np.zeros(0)

    for ch in channels:
        row = {"key": ch.key, "label": ch.label, "n": 0, "present": False,
               "mean_w": np.nan, "peak_w": np.nan, "energy_wh": 0.0,
               "mean_a": np.nan, "peak_a": np.nan, "bus_v": np.nan}

        w = _column(df, ch.power)
        a = _column(df, ch.current)
        finite = np.isfinite(w)
        row["n"] = int(finite.sum())
        row["present"] = bool(row["n"])

        if row["present"]:
            row["mean_w"] = float(np.nanmean(w))
            row["peak_w"] = float(np.nanmax(w))
            row["energy_wh"] = _energy_wh(ts, w)
        if np.isfinite(a).any():
            row["mean_a"] = float(np.nanmean(a))
            row["peak_a"] = float(np.nanmax(a))
            # power = bus_voltage * current in the driver, so the division
            # recovers a voltage the logger never wrote down.
            loaded = np.isfinite(w) & (a > CURRENT_NOISE_A)
            if loaded.any():
                row["bus_v"] = float(np.median(w[loaded] / a[loaded]))

        out.append(row)
    return out


def _column(df, name):
    if name not in df.columns:
        return np.full(len(df), np.nan)
    return df[name].to_numpy(dtype=float)


# ------------------------------------------------------- sail and navigation

# Like power, these read raw rows rather than fixes: the sail logic re-decides
# every cycle, so collapsing to GPS fixes would throw away most of its state.

def _text(df, name):
    """A category column as plain strings, "" where absent or null."""
    if name not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    col = df[name]
    return col.astype(object).where(col.notna(), "").astype(str)


def _time_frac(df, mask):
    """Fraction of elapsed time the mask holds, weighting each row by its own
    interval so a logging gap cannot inflate whichever state preceded it."""
    if len(df) < 2:
        return float(mask.mean()) if len(df) else 0.0
    dt = np.clip(np.diff(_epoch_seconds(df["t"])), 0.0, MAX_POWER_GAP_S)
    total = dt.sum()
    if total <= 0:
        return 0.0
    return float(dt[np.asarray(mask)[:-1]].sum() / total)


def sail_summary(df) -> dict:
    """What the sail logic was doing over the rows handed in."""
    out = {"n": len(df), "present": False, "frac_sail": np.nan, "frac_motor": np.nan,
           "frac_beating": np.nan, "frac_motor_assist": np.nan,
           "n_tacks": 0, "states": {}, "faults": []}
    if df.empty:
        return out

    propulsion = _text(df, "propulsion")
    state = _text(df, "sail_state")
    out["present"] = bool((propulsion != "").any() or (state != "").any())
    if not out["present"]:
        return out

    out["frac_sail"] = _time_frac(df, propulsion == "sail")
    out["frac_motor"] = _time_frac(df, propulsion == "motor")
    out["frac_beating"] = _time_frac(df, _text(df, "beating") == "True")
    out["frac_motor_assist"] = _time_frac(df, _text(df, "motor_assist") == "True")

    seen = state[state != ""]
    out["states"] = seen.value_counts().to_dict()

    # A tack is a sign change while the value is actually being logged; the
    # blank stretches between AUTO runs would otherwise each read as two tacks.
    tack = _column(df, "tack")
    ok = np.isfinite(tack)
    if ok.any():
        s = np.sign(tack[ok])
        out["n_tacks"] = int((np.diff(s) != 0).sum())

    fault = _text(df, "sail_fault")
    bad = (fault != "") & (fault != SAIL_FAULT_OK)
    if bad.any():
        t = df["t"].to_numpy()
        for value, i0, i1 in _runs(fault.to_numpy(), bad.to_numpy()):
            out["faults"].append((value, t[i0], t[i1]))
    return out


def nav_summary(df) -> dict:
    """How well the autopilot held its line over the rows handed in."""
    out = {"n": 0, "present": False, "wp_start": np.nan, "wp_end": np.nan,
           "mean_abs_err": np.nan, "rms_xte": np.nan, "max_abs_xte": np.nan,
           "dist_end": np.nan}
    if df.empty:
        return out

    des = _column(df, "des_hdg_deg")
    ok = np.isfinite(des)
    out["n"] = int(ok.sum())
    out["present"] = bool(out["n"])
    if not out["present"]:
        return out

    err = _column(df, "err_deg")
    err = err[np.isfinite(err)]
    if err.size:
        out["mean_abs_err"] = float(np.mean(np.abs(err)))

    xte = _column(df, "xte_m")
    xte = xte[np.isfinite(xte)]
    if xte.size:
        out["rms_xte"] = float(np.sqrt(np.mean(xte ** 2)))
        out["max_abs_xte"] = float(np.max(np.abs(xte)))

    wp = _column(df, "wp_left")
    wp = wp[np.isfinite(wp)]
    if wp.size:
        out["wp_start"], out["wp_end"] = float(wp[0]), float(wp[-1])

    dist = _column(df, "dist_m")
    dist = dist[np.isfinite(dist)]
    if dist.size:
        out["dist_end"] = float(dist[-1])
    return out


def _runs(values, mask):
    """Contiguous runs where mask holds, as (value, first index, last index)."""
    spans, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((values[start], start, i - 1))
            start = None
    if start is not None:
        spans.append((values[start], start, len(mask) - 1))
    return spans


def _energy_wh(ts, w):
    """Trapezoidal integral of watts over seconds, in watt-hours."""
    if len(w) < 2:
        return 0.0
    dt = np.clip(np.diff(ts), 0.0, MAX_POWER_GAP_S)
    return float(np.nansum((w[:-1] + w[1:]) / 2.0 * dt) / 3600.0)
