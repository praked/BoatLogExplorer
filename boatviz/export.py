"""Export filtered selections to GeoJSON."""

import json

import numpy as np
import pandas as pd

POINT_FIELDS = ["row", "hdg", "cog", "resid", "awa", "speed", "dwell", "source",
                "mode", "auto_mode",
                "aws", "twd", "twa", "tws",
                "des_hdg", "brg", "err", "xte", "dist", "wp_left",
                "rudder", "sail", "thr", "tack", "fix_q",
                "sail_state", "sail_fault", "sail_reason", "propulsion",
                "beating", "motor_assist"]


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 6)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return str(v)


def to_geojson(fix, log_id="", include_points=True) -> str:
    """FeatureCollection with the track as a LineString plus optional points."""
    features = []

    if len(fix) >= 2:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": fix[["lon", "lat"]].to_numpy().tolist()},
            "properties": {
                "log": log_id, "kind": "track", "fixes": int(len(fix)),
                "start": str(fix["t"].iloc[0]), "end": str(fix["t"].iloc[-1]),
            },
        })

    if include_points:
        for _, r in fix.iterrows():
            props = {"log": log_id, "kind": "fix",
                     "t_iso": str(pd.Timestamp(r["t"]))}
            props.update({f: _clean(r.get(f)) for f in POINT_FIELDS if f in fix.columns})
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [float(r["lon"]), float(r["lat"])]},
                "properties": props,
            })

    return json.dumps({"type": "FeatureCollection", "features": features})


def qa_summary(log) -> str:
    """Plain-text sensor-health report, suitable for pasting into notes."""
    qa = log.qa
    bb = log.bbox()
    lines = [
        f"{log.id}",
        f"  file      {log.path}  ({log.size / 1e6:.1f} MB)",
        f"  rows      {log.n_rows:,}   GPS rows {getattr(log, 'n_gps_rows', 0):,}",
        f"  fixes     {log.n_fixes:,}  ({qa.dup_frac * 100:.1f}% duplicates removed)",
        f"  span      {log.t_start} -> {log.t_end}  ({log.duration_s / 3600:.2f} h)",
    ]
    if bb:
        lines.append(f"  area      {bb[4]:.0f} x {bb[5]:.0f} m")
    lines.append(f"  GPS       {qa.gps_verdict}  ({qa.moving_frac * 100:.1f}% of fixes moving)")
    if qa.n_outliers:
        lines.append(f"  rejected  {qa.n_outliers} implausible fix(es)")
    if qa.monotonic_breaks:
        lines.append(f"  warning   {qa.monotonic_breaks} non-monotonic timestamps")
    lines.append("  channels:")
    for key, rep in qa.channels.items():
        lines.append(f"    {rep.verdict:9s} {rep.label:28s} {rep.summary}")
    return "\n".join(lines)
