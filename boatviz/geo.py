"""Geodesy and circular statistics."""

import numpy as np

EARTH_R = 6371008.8


def wrap360(deg):
    return np.mod(deg, 360.0)


def wrap180(deg):
    return np.mod(deg + 180.0, 360.0) - 180.0


def ang_diff(a, b):
    """Smallest signed difference a - b, in (-180, 180]."""
    return wrap180(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def circ_mean_R(deg):
    """Circular mean direction and concentration R in [0, 1].

    R near 1 means tightly clustered; R near 0 means uniformly spread.
    """
    d = np.asarray(deg, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, 0.0
    r = np.radians(d)
    s, c = np.sin(r).sum(), np.cos(r).sum()
    R = float(np.hypot(s, c) / d.size)
    return float(np.degrees(np.arctan2(s, c)) % 360.0), R


def circ_spread_deg(deg):
    """Circular standard deviation in degrees, derived from the concentration.

    This is the frozen-channel detector. Counting distinct values does not work:
    a frozen heading channel dithers in the 4th decimal and shows ~77000 distinct
    values across a 0.3 deg range.
    """
    _, R = circ_mean_R(deg)
    R = min(max(R, 1e-12), 1.0)
    return float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(R)))))


def meters_per_deg(lat_deg):
    """Metres per degree of latitude and longitude at a given latitude."""
    lat = np.radians(lat_deg)
    return 111132.92 - 559.82 * np.cos(2 * lat), 111412.84 * np.cos(lat)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing, degrees clockwise from North."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return wrap360(np.degrees(np.arctan2(y, x)))


def offset_latlon(lat, lon, bearing_deg_cw, dist_m):
    """Point reached by travelling dist_m along a clockwise compass bearing.

    Uses a local flat-earth approximation, which is exact enough at the
    50-400 m scales these logs cover.
    """
    m_lat, m_lon = meters_per_deg(np.asarray(lat, dtype=float))
    th = np.radians(np.asarray(bearing_deg_cw, dtype=float))
    return lat + np.cos(th) * dist_m / m_lat, lon + np.sin(th) * dist_m / np.maximum(m_lon, 1e-9)
