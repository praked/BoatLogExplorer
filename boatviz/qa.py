"""Per-channel health assessment.

The 2026 logs are mostly a stationary boat with dead sensors, so "is this channel
actually alive?" has to be answered before anything is plotted. Distinct-value
counts cannot answer it: in boat_log_20260804_061853.csv the heading channel has
77370 distinct values across 77371 rows, and a peak-to-peak of 0.311 degrees.
The complementary filter dithers in the 4th decimal forever. Spread is the only
reliable signal.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .geo import circ_spread_deg
from .schema import CHANNELS

ALIVE, PARTIAL, FROZEN, CONSTANT, ABSENT = "ALIVE", "PARTIAL", "FROZEN", "CONSTANT", "ABSENT"

WINDOW = "60s"
PRESENT_MIN = 0.02      # below this fraction of non-null samples the channel is absent
ALIVE_FRAC = 0.10       # fraction of live windows needed for an overall ALIVE verdict
PARTIAL_FRAC = 0.01
STATIONARY_FRAC = 0.02  # fraction of fixes above the speed gate to count as moving


@dataclass
class ChannelReport:
    key: str
    label: str
    unit: str
    verdict: str
    spread: float = 0.0
    vmin: float = np.nan
    vmax: float = np.nan
    const_value: float = np.nan
    present_frac: float = 0.0
    alive_frac: float = 0.0
    alive_windows: list = field(default_factory=list)   # [(start, end)] merged live spans

    circular: bool = False
    freeze_checked: bool = True

    @property
    def summary(self) -> str:
        if self.verdict == ABSENT:
            return "no data"
        if self.verdict == CONSTANT:
            return f"constant @ {self.const_value:g}{self.unit}"
        if not self.freeze_checked:
            # A commanded or computed value; "live x% of the log" would be
            # measuring the autopilot's activity, not a sensor's health.
            return f"logged {self.present_frac * 100:.0f}%, range {self.vmin:.3g}–{self.vmax:.3g}{self.unit}"
        if self.verdict == FROZEN:
            # For a circular channel the min/max range is misleading (a value
            # wobbling across 0 spans 0-360), so report the spread instead.
            span = self.spread if self.circular else self.vmax - self.vmin
            return f"frozen, spread {span:.3f}{self.unit}"
        live = f"live {self.alive_frac * 100:.0f}% of the log"
        if self.circular:
            return f"{live}, spread {self.spread:.0f}{self.unit}"
        return f"{live}, range {self.vmin:.3g}–{self.vmax:.3g}{self.unit}"


@dataclass
class QaReport:
    channels: dict = field(default_factory=dict)
    gps_verdict: str = "NO GPS"
    moving_frac: float = 0.0
    n_outliers: int = 0
    dup_frac: float = 0.0
    monotonic_breaks: int = 0

    def channel(self, key) -> ChannelReport | None:
        return self.channels.get(key)

    def is_alive(self, key) -> bool:
        c = self.channels.get(key)
        return c is not None and c.verdict == ALIVE


def assess_channel(df: pd.DataFrame, ch) -> ChannelReport:
    s = df[ch.key]
    present = float(s.notna().mean()) if len(s) else 0.0
    rep = ChannelReport(ch.key, ch.label, ch.unit, ABSENT,
                        present_frac=present, circular=ch.circular,
                        freeze_checked=ch.freeze_check)

    if present < PRESENT_MIN:
        return rep

    vals = s.dropna()
    rep.vmin, rep.vmax = float(vals.min()), float(vals.max())

    if vals.nunique() == 1:
        rep.verdict = CONSTANT
        rep.const_value = float(vals.iloc[0])
        return rep

    if not ch.freeze_check:
        # It is varying and it is present; there is nothing further to ask of a
        # value the boat computed itself. Skipping the resample below also keeps
        # the per-window Python loop off two thirds of the channel list.
        rep.verdict = ALIVE
        return rep

    spread_fn = circ_spread_deg if ch.circular else _linear_spread
    rep.spread = spread_fn(vals.to_numpy())

    windows = df[["t", ch.key]].dropna().set_index("t")[ch.key].resample(WINDOW)
    per_window = windows.apply(lambda w: spread_fn(w.to_numpy()) if len(w) >= 3 else np.nan)
    per_window = per_window.dropna()

    if per_window.empty:
        rep.verdict = FROZEN if rep.spread < ch.freeze_tau else ALIVE
        return rep

    live = per_window >= ch.freeze_tau
    rep.alive_frac = float(live.mean())
    rep.alive_windows = _merge_spans(per_window.index[live], pd.Timedelta(WINDOW))

    if rep.alive_frac >= ALIVE_FRAC:
        rep.verdict = ALIVE
    elif rep.alive_frac >= PARTIAL_FRAC:
        rep.verdict = PARTIAL
    else:
        rep.verdict = FROZEN
    return rep


def _linear_spread(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    return float(np.percentile(a, 95) - np.percentile(a, 5))


def _merge_spans(starts, width):
    spans = []
    for s in starts:
        if spans and s - spans[-1][1] <= width:
            spans[-1][1] = s + width
        else:
            spans.append([s, s + width])
    return [tuple(s) for s in spans]


def assess(log) -> QaReport:
    rep = QaReport()
    if log.df.empty:
        return rep

    for ch in CHANNELS:
        rep.channels[ch.key] = assess_channel(log.df, ch)

    rep.monotonic_breaks = getattr(log, "monotonic_breaks", 0)
    n_gps = getattr(log, "n_gps_rows", 0)
    if n_gps:
        rep.dup_frac = 1.0 - len(log.fix) / n_gps

    if "speed" in log.fix.columns and len(log.fix):
        spd = log.fix["speed"].to_numpy()
        ok = np.isfinite(spd)
        if ok.any():
            rep.moving_frac = float((spd[ok] > 0.5).mean())
            rep.gps_verdict = "MOVING" if rep.moving_frac >= STATIONARY_FRAC else "STATIONARY"
        rep.n_outliers = int(log.fix.get("outlier", pd.Series(dtype=bool)).sum())
    elif len(log.fix) >= 2:
        rep.gps_verdict = "UNKNOWN"

    return rep
