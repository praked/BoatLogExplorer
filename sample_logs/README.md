# Sample logs

Small enough to keep in git while preserving what makes the originals
interesting. These two are also what the GitHub Pages build ships, so the set is
deliberately short: every megabyte here is downloaded before the hosted app can
show anything.

| file | from | why it is here |
|---|---|---|
| `boat_log_20260804_084601_sample.csv` | first 25 min of `boat_log_20260804_084601` | a **frozen-sensor** recording, so the sensor-health panel has something to flag |
| `boat_log_20260819_172759.csv` | a full 11 min AutoP2P run | the first log with the **2026-08 schema**: true wind, waypoint targets, actuator commands and the sail logic's own state, over a real beat to windward |

The 2026-08-04 extract keeps the original ~89 % GPS duplicate rate, so
deduplication behaves exactly as it does on the full files.

A third extract, `boat_log_20260803_145351_sample.csv`, was dropped in favour of
the 2026-08-19 run: it was here as the one recording with a working compass and
wind vane, and the newer log demonstrates that as well while also exercising
every column added since. At 4.8 MB it was also the largest of the three, and the
hosted build fetches all of them at start-up.
