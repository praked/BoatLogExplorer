"""Channel metadata and column detection for boat CSV logs.

The canonical schema written by boatv1/utils/data_logger.py is:

    ts_iso, gps_lat, gps_lon, gps_ts, awa_deg, heading, rotation_vector,
    geomag_rotation_vector, gyro_z_dps, heading_source, sys_temp_c, sys_volts_v

A later revision of that logger widened the row with an INA3221 power monitor
(ch1..ch3) and the autopilot's own state (mode, auto_mode, and the waypoint
fields). The 2026-08 revision widened it again with true wind, the waypoint
targets the autopilot is steering to, the actuator commands it sends, and the
sail logic's own state. Those columns are optional everywhere: logs from any
revision load, and a column a log does not carry is materialized empty rather
than skipped, so every consumer sees the same frame shape.
"""

from dataclasses import dataclass

CANONICAL_COLUMNS = [
    "ts_iso", "gps_lat", "gps_lon", "gps_ts", "awa_deg", "heading",
    "rotation_vector", "geomag_rotation_vector", "gyro_z_dps",
    "heading_source", "sys_temp_c", "sys_volts_v",
    "mode", "auto_mode",
    "ch1_pwr", "ch1_cur", "ch2_pwr", "ch2_cur", "ch3_pwr", "ch3_cur",
    # 2026-08: wind, waypoint targets, actuator commands, sail logic state
    "wind_spd", "awa_raw_deg", "twd_deg", "twa_deg", "tws_mps",
    "wp_left", "dist_m", "brg_deg", "err_deg", "xte_m", "des_hdg_deg",
    "rudder_deg", "sail_deg", "thr_us",
    "sail_state", "sail_fault", "sail_reason", "propulsion",
    "beating", "tacked", "tack", "sailing", "settled", "motor_assist",
    "fix_q", "session",
]


@dataclass(frozen=True)
class Channel:
    key: str
    label: str
    unit: str
    circular: bool      # compass-style, 0-360 with a wrap at North
    freeze_tau: float   # spread below this over a window means the channel is frozen
    group: str
    note: str = ""
    # Data in [-180, 180] that wraps at +/-180 rather than at 0/360. Plots on a
    # fixed +/-180 axis and still needs the wrap-gap treatment.
    signed_circular: bool = False
    zero_line: bool = False          # signed quantity: draw a y=0 reference
    y_range: tuple | None = None     # fixed y-axis range, for bounded commands
    # Freeze detection asks "is this sensor still reading?". That question is
    # meaningless for a value the autopilot computes or commands -- a rudder held
    # steady is good steering, not a fault -- so those channels skip it. It also
    # keeps the 60 s resample in qa off two thirds of the channel list.
    freeze_check: bool = True
    default_on: bool = True          # preselected in the Time series multiselect


# freeze_tau values are calibrated against measured spreads in the known-frozen
# logs: heading p2p 0.311 deg, geomag_rotation_vector p2p 0.620 deg, and
# rotation_vector / awa_deg / gyro_z_dps fully constant.
#
# Order matters: the Time series multiselect, the sensor-health card and the
# aliveness strip all iterate this list, so it runs group by group.
CHANNELS = [
    Channel("heading", "Heading (filtered)", "°", True, 1.0, "navigation",
            "Magnetic, clockwise from North. No declination applied."),
    Channel("geomag_rotation_vector", "Heading (geomag, raw)", "°", True, 1.0, "navigation",
            "Raw per-sample yaw, not a quaternion despite the name."),
    Channel("rotation_vector", "Heading (rotation vector, raw)", "°", True, 1.0, "navigation",
            "Raw per-sample yaw. Frozen in most 2026 logs."),
    Channel("gyro_z_dps", "Yaw rate", "°/s", False, 0.05, "navigation",
            "Sign unverified — the firmware contradicts itself. 96% exact zeros in 2026 logs."),

    Channel("awa_deg", "Apparent wind angle", "°", True, 0.5, "wind",
            "Boat-relative, after the firmware's mast-rotation compensation. "
            "0 = head-to-wind, increases to starboard."),
    Channel("awa_raw_deg", "Apparent wind angle (raw)", "°", True, 0.5, "wind",
            "What the vane actually reported, before mast-rotation compensation. "
            "On a rotating mast the difference is the whole sail angle.",
            default_on=False),
    Channel("wind_spd", "Apparent wind speed", "m/s", False, 0.05, "wind",
            "Anemometer, boat-relative. The 2026-08 firmware is the first to log it."),
    Channel("twd_deg", "True wind direction", "°", True, 0.0, "wind",
            "Where the true wind blows FROM, compass degrees. Sailing straight "
            "upwind means heading equals this.",
            freeze_check=False, default_on=False),
    Channel("twa_deg", "True wind angle", "°", False, 0.0, "wind",
            "True wind relative to the bow, signed. Positive = wind from "
            "starboard, which is the port tack.",
            signed_circular=True, zero_line=True, y_range=(-180, 180),
            freeze_check=False, default_on=False),
    Channel("tws_mps", "True wind speed", "m/s", False, 0.0, "wind",
            "Apparent wind with the boat's own motion taken out.",
            freeze_check=False, default_on=False),

    Channel("des_hdg_deg", "Desired heading", "°", True, 0.0, "autopilot",
            "The heading the controller asked for, against the 'heading' it "
            "achieved. Differs from the bearing whenever the boat is beating.",
            freeze_check=False, default_on=False),
    Channel("brg_deg", "Bearing to waypoint", "°", True, 0.0, "autopilot",
            "Straight-line compass bearing from the boat to the active waypoint.",
            freeze_check=False, default_on=False),
    Channel("err_deg", "Bearing error", "°", False, 0.0, "autopilot",
            "Bearing minus heading, signed — what the rudder is answering. Note "
            "it is measured against the bearing, not the desired heading, so it "
            "is large by design while beating.",
            signed_circular=True, zero_line=True, freeze_check=False, default_on=False),
    Channel("xte_m", "Cross-track error", "m", False, 0.0, "autopilot",
            "Metres off the line from the previous waypoint to the active one, "
            "positive to the right of it. Reaching the corridor edge is what "
            "triggers a tack.",
            zero_line=True, freeze_check=False, default_on=False),
    Channel("dist_m", "Distance to waypoint", "m", False, 0.0, "autopilot",
            "Resets upward each time a waypoint is reached and the next one arms.",
            freeze_check=False, default_on=False),
    Channel("wp_left", "Waypoints remaining", "", False, 0.0, "autopilot",
            "Still queued. A false arrival shows up here as an early drop.",
            freeze_check=False, default_on=False),

    Channel("rudder_deg", "Rudder command", "°", False, 0.0, "actuators",
            "Commanded angle, not a measured one — there is no rudder feedback sensor.",
            zero_line=True, y_range=(-45, 45), freeze_check=False, default_on=False),
    Channel("sail_deg", "Sail command", "°", False, 0.0, "actuators",
            "Commanded sheet position: 0 is close-hauled, larger is eased.",
            zero_line=True, y_range=(-95, 95), freeze_check=False, default_on=False),
    Channel("thr_us", "Throttle", "µs", False, 0.0, "actuators",
            "Commanded servo pulse width, so a dry run still records the intent. "
            "1500 µs is neutral. Only logged outside AUTO.",
            y_range=(950, 2050), freeze_check=False, default_on=False),

    Channel("sys_temp_c", "Pi CPU temperature", "°C", False, 0.2, "system",
            "Raspberry Pi CPU die temperature — not water or air temperature."),
    Channel("sys_volts_v", "Pi core voltage", "V", False, 0.005, "system",
            "Raspberry Pi core voltage from vcgencmd — not battery voltage."),
]

CHANNELS_BY_KEY = {c.key: c for c in CHANNELS}
HEADING_KEYS = ["heading", "geomag_rotation_vector", "rotation_vector"]

GROUP_LABELS = {
    "navigation": "Heading",
    "wind": "Wind",
    "autopilot": "Autopilot",
    "actuators": "Actuators",
    "system": "System",
}
GROUP_ORDER = list(GROUP_LABELS)


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


# ---------------------------------------------------------------- sail logic

# The names sail_func/tack.py writes. Only SAILING appears in the 2026-08 log,
# but the tack machine passes through the others on every tack.
SAIL_STATE_LABELS = {
    "SAILING": "settled on a tack",
    "INITIATING": "starting a tack",
    "THROUGH_WIND": "swinging through the wind",
    "SETTLING": "steadying on the new tack",
    "IN_IRONS": "stalled head to wind",
}
SAIL_STATE_COLORS = {
    "SAILING": "#1baf7a",
    "INITIATING": "#c99a1e",
    "THROUGH_WIND": "#eb6834",
    "SETTLING": "#0f9bb5",
    "IN_IRONS": "#d03b3b",
    "": "#898781",
}

# sail_func/supervisor.py FAULT_ORDER, highest precedence first. The supervisor
# reports one fault per cycle and the first match wins.
SAIL_FAULT_OK = "NOMINAL"
SAIL_FAULT_LABELS = {
    "RC_OVERRIDE": "a human took the sticks",
    "HEADING_LOST": "no trustworthy heading",
    "EXCESSIVE_HEEL": "heeled past the limit",
    "GEOFENCE": "outside the permitted area",
    "IN_IRONS": "stalled head to wind",
    "BECALMED": "not enough wind to sail",
    "WIND_STALE": "wind reading went stale",
    "POSITION_LOST": "no position fix",
    SAIL_FAULT_OK: "healthy",
}

PROPULSION_COLORS = {"sail": "#1baf7a", "motor": "#eb6834", "": "#898781"}

# sail_func/guidance.py: "q  tack side: +1 port tack (wind from starboard,
# TWA > 0), -1 starboard tack."
TACK_LABELS = {1: "port tack", -1: "starboard tack"}
TACK_COLORS = {1: "#2a78d6", -1: "#eb6834", "": "#898781"}

# Logged as the strings "True"/"False", except where a column is populated on
# every row and the CSV parser hands back real booleans instead. Ingest
# normalizes both forms to these strings.
BOOL_COLUMNS = ["beating", "tacked", "sailing", "settled", "motor_assist"]

CATEGORY_COLUMNS = ["heading_source", "mode", "auto_mode",
                    "sail_state", "sail_fault", "sail_reason", "propulsion",
                    "session"]

# Numeric, but only ever a handful of values, so they are coloured and labelled
# like categories rather than ramped.
DISCRETE_NUMERIC = ["tack", "fix_q"]


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
