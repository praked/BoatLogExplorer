"""Channel metadata and column detection for boat CSV logs.

The canonical schema written by boatv1/utils/data_logger.py is:

    ts_iso, gps_lat, gps_lon, gps_ts, awa_deg, heading, rotation_vector,
    geomag_rotation_vector, gyro_z_dps, heading_source, sys_temp_c, sys_volts_v

A later revision of that logger widened the row with an INA3221 power monitor
(ch1..ch3) and the autopilot's own state (mode, auto_mode, and the waypoint
fields). Those columns are optional everywhere: logs from either revision load.
"""

from dataclasses import dataclass

CANONICAL_COLUMNS = [
    "ts_iso", "gps_lat", "gps_lon", "gps_ts", "awa_deg", "heading",
    "rotation_vector", "geomag_rotation_vector", "gyro_z_dps",
    "heading_source", "sys_temp_c", "sys_volts_v",
    "mode", "auto_mode",
    "ch1_pwr", "ch1_cur", "ch2_pwr", "ch2_cur", "ch3_pwr", "ch3_cur",
]


@dataclass(frozen=True)
class Channel:
    key: str
    label: str
    unit: str
    circular: bool
    freeze_tau: float   # spread below this over a window means the channel is frozen
    group: str
    note: str = ""


# freeze_tau values are calibrated against measured spreads in the known-frozen
# logs: heading p2p 0.311 deg, geomag_rotation_vector p2p 0.620 deg, and
# rotation_vector / awa_deg / gyro_z_dps fully constant.
CHANNELS = [
    Channel("heading", "Heading (filtered)", "°", True, 1.0, "navigation",
            "Magnetic, clockwise from North. No declination applied."),
    Channel("geomag_rotation_vector", "Heading (geomag, raw)", "°", True, 1.0, "navigation",
            "Raw per-sample yaw, not a quaternion despite the name."),
    Channel("rotation_vector", "Heading (rotation vector, raw)", "°", True, 1.0, "navigation",
            "Raw per-sample yaw. Frozen in most 2026 logs."),
    Channel("awa_deg", "Apparent wind angle", "°", True, 0.5, "wind",
            "0 = head-to-wind, increases to starboard. Includes a +15° vane mount offset."),
    Channel("gyro_z_dps", "Yaw rate", "°/s", False, 0.05, "navigation",
            "Sign unverified — the firmware contradicts itself. 96% exact zeros in 2026 logs."),
    Channel("sys_temp_c", "Pi CPU temperature", "°C", False, 0.2, "system",
            "Raspberry Pi CPU die temperature — not water or air temperature."),
    Channel("sys_volts_v", "Pi core voltage", "V", False, 0.005, "system",
            "Raspberry Pi core voltage from vcgencmd — not battery voltage."),
]

CHANNELS_BY_KEY = {c.key: c for c in CHANNELS}
HEADING_KEYS = ["heading", "geomag_rotation_vector", "rotation_vector"]


# ------------------------------------------------------------------ power rail

@dataclass(frozen=True)
class PowerChannel:
    key: str
    label: str
    power: str      # column holding bus power, watts
    current: str    # column holding current, amps


# The firmware logs ChannelReading.power_w and .current_a straight from
# sensors/ina3221_sensor.py, so these really are watts and amps -- no scaling.
# What each channel feeds is a wiring decision the firmware does not record, so
# they stay numbered rather than being given invented names. Bus voltage is not
# logged, but power / current recovers it: the driver computes power as
# bus_voltage_v * current_a.
POWER_CHANNELS = [
    PowerChannel("ch1", "Channel 1", "ch1_pwr", "ch1_cur"),
    PowerChannel("ch2", "Channel 2", "ch2_pwr", "ch2_cur"),
    PowerChannel("ch3", "Channel 3", "ch3_pwr", "ch3_cur"),
]

POWER_COLUMNS = [c for ch in POWER_CHANNELS for c in (ch.power, ch.current)]


# --------------------------------------------------------------- control modes

# Names as boatv1/utils/modes.py writes them through mode_name(). An unlisted
# value still plots -- it just falls through to the "other" colour rather than
# being dropped, which is what makes a firmware change visible instead of silent.
MODE_LABELS = {
    "MANUAL": "RC sticks drive the actuators",
    "SIM_STEERED": "rudder/sail follow des_heading from MOOS",
    "HYBRID": "rudder from RC, sail automatic by wind",
    "AUTO": "autonomous: station-keeping, point-to-point or rendezvous",
    "RESET": "surfaces centred, motor stopped",
}

# boat/main.py AutoModes -- only meaningful while mode is AUTO.
AUTO_MODE_LABELS = {
    "SK": "station-keeping",
    "AutoP2P": "auto point-to-point",
    "RDV": "rendezvous",
}

CATEGORY_COLUMNS = ["heading_source", "mode", "auto_mode"]


def detect_columns(columns) -> dict:
    """Map canonical roles onto the actual column names present in a file.

    Falls back to fuzzy matching so logs from other firmware revisions still load.
    """
    lower = {str(c).strip().lower(): c for c in columns}
    found = {}

    for name in CANONICAL_COLUMNS:
        if name in lower:
            found[name] = lower[name]

    if "gps_lat" not in found:
        for lc, orig in lower.items():
            if "lat" in lc:
                found["gps_lat"] = orig
                break
    if "gps_lon" not in found:
        for lc, orig in lower.items():
            if "lon" in lc or "lng" in lc:
                found["gps_lon"] = orig
                break
    if "ts_iso" not in found:
        for lc, orig in lower.items():
            if "time" in lc or lc.startswith("ts"):
                found["ts_iso"] = orig
                break

    return found
