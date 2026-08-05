# Boat Log Explorer

Sensor data log explorer for the **SailSwarm** project. Loads the boat's CSV
logs, checks which sensors were actually working, and plots the track on a map.

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

It opens at <http://localhost:8501>. Drop CSVs onto the sidebar uploader, or put
a folder path in the box below it to read logs straight off disk — the folder
route avoids uploading and is the one to use for the large recordings.

## Hosting it

**GitHub Pages** (`docs/`, static). GitHub Pages cannot run Python, so the page
uses [stlite](https://github.com/whitphx/stlite) to run Streamlit as WebAssembly
in the visitor's browser. No server, no install, just a link.

```bash
python build_pages.py     # regenerate docs/ after changing any source file
```

Then in the repo's **Settings → Pages**, set the source to `main` / `/docs`. It
publishes to `https://<user>.github.io/BoatLogExplorer/`. That is a *project*
page on a subpath; it does not affect a `<user>.github.io` user page, which is a
separate repository.

Two things to know about the browser build: the first visit downloads roughly
30 MB of Python runtime and takes about 35 seconds before the app appears
(cached afterwards), and everything is parsed in browser memory, so the big
multi-gigabyte-scale logs are better handled by running locally. The two sample
logs load automatically.

**Streamlit Community Cloud** is the better home if you want full speed and the
ability to open the 64 MB recordings. Point <https://share.streamlit.io> at this
repo with `app.py` as the entry point; it installs `requirements.txt` and runs
real CPython. Nothing in the code needs to change.

## Read this before trusting a plot

**Most of the 2026 logs are a stationary boat with dead sensors.** Of the 23 files
in `logs_4Aug2026/`, two are empty, seventeen are short fragments with no GPS, and
only four contain a track. Of those four, only `boat_log_20260803_145351` has a
compass that moved and a wind vane that ever reported anything but 140.0°.

| log | duration | fixes | area | moving | compass | wind vane |
|---|---|---|---|---|---|---|
| `20260803_145351` | 1.9 h | 6 698 | 90 × 63 m | 8.5 % | **working** | working for the last 33 min |
| `20260803_164444` | 11.1 h | 39 523 | 105 × 176 m | 0.8 % | frozen | stuck @ 140° |
| `20260804_061853` | 2.5 h | 8 795 | 370 × 153 m | 0.8 % | frozen | stuck @ 140° |
| `20260804_084601` | 2.2 h | 7 825 | 222 × 129 m | 0.5 % | frozen | stuck @ 140° |

That is why the app leads with sensor health rather than a map. A frozen channel
is switched off by default and marked in the sidebar, so a dead sensor looks dead
instead of looking like data.

**A track is not the same as motion.** The logger samples at ~10 Hz against a ~1 Hz
GPS, so about 89 % of rows repeat the previous fix; those are collapsed on load.
What remains is still dominated by receiver noise — a stationary receiver wanders
at roughly 0.1 m/s indefinitely. Anything below the **moving threshold** (0.5 m/s
by default) is treated as jitter, and course over ground is left undefined there
rather than computed from noise. Use **Only while moving** on the Map tab to see
just the real travel.

## Selecting a section of a log

The time range bar sits at the top of the page. Drag either handle to pick a
section; everything below — map, charts, compass check, exports — follows the
selection. Underneath the slider is an overview of the whole recording: one lane
per selected log, filled area showing speed, and green marks where the boat was
actually moving. Regions outside the selection are shaded grey.

Four presets sit under the slider. **Moving only** is the useful one on these
logs: it jumps straight to the stretch containing all the real motion, which on
`20260803_145351` is 13:38–14:32 — about half the recording, with the rest being
the boat sitting still.

The slider step adapts to the log length, giving roughly 500 stops across the
range (15 seconds on a two-hour log), so short sections can be picked precisely.

## Conventions

**Heading is a normal clockwise compass bearing** — N=0, E=90, S=180, W=270 — and
is drawn directly. The older HTML viewers in this repo negate it to enforce an
"anticlockwise" convention; that is wrong. Scored against GPS course over ground
on the one log with real motion, the identity mapping lands at about 12° median
error while the negated form does worse than random guessing. The **Compass check**
tab re-runs this test on whatever you have loaded, and refuses to answer when the
compass is frozen or the boat never moved.

**All headings are magnetic.** No declination correction exists anywhere in the
firmware. At Lake Constance that is roughly +3°, small next to the residual error.

**Wind is `heading + AWA`**, where AWA is 0 at head-to-wind and increases to
starboard, and already includes the vane's +15° mount offset. The evidence for the
sign is weak, so wind overlays are off by default and both toggles are exposed.

**True wind is not computed.** It needs apparent wind *speed*, which this firmware
never logs. Any number labelled "true wind direction" here would be invented.

**`sys_temp_c` and `sys_volts_v` are Raspberry Pi health readings** — CPU die
temperature and core voltage from `vcgencmd`. They are not water temperature and
not battery voltage.

**Use `ts_iso`, not `gps_ts`.** `ts_iso` is the Pi clock in UTC at microsecond
resolution and is present on every row. `gps_ts` comes only from RMC sentences, so
it is often blank, and it has been observed more than an hour out. Filenames are
local time (UTC+2) while all timestamps in the app are UTC.

## Layout

```
app.py              Streamlit UI — the only module that imports streamlit
boatviz/
  schema.py         channel metadata, column detection
  ingest.py         CSV parsing, GPS validation, fix deduplication
  qa.py             alive / frozen / constant detection per channel
  derive.py         speed, course over ground, residuals, convention scoring
  geo.py            circular statistics and geodesy
  mapview.py        Folium map construction
  charts.py         Plotly figures
  export.py         GeoJSON and sensor-health report
```

Nothing under `boatviz/` imports Streamlit, so every function is usable from a
notebook or a test:

```python
from boatviz import load_log, enrich, assess, qa_summary
log = load_log("logs_4Aug2026/boat_log_20260803_145351.csv")
enrich(log); log.qa = assess(log)
print(qa_summary(log))
```

## Notes on the implementation

Frozen channels are detected by **value spread over a rolling window, not distinct
value counts**. A frozen heading dithers in the fourth decimal forever: in
`boat_log_20260804_061853` it has 77 370 distinct values across 77 371 rows and a
total range of 0.311°. Counting distinct values would call that channel healthy.

Latitude and longitude are held as float64 throughout. float32 quantizes position
to about 0.6 m at this latitude, which silently merges distinct GPS fixes during
deduplication.

Satellite imagery is served at `max_zoom` 22 over `max_native_zoom` 19, so tiles
upscale rather than disappearing when you zoom into a 90 m track.
