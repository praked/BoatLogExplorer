"""Boat log analysis toolkit.

Deliberately free of Streamlit imports so every function here is usable from a
notebook or a test.
"""

from .ingest import Log, discover_logs, load_log
from .qa import assess, QaReport
from .derive import enrich, convention_scoreboard, wind_direction, wind_sign_evidence
from .export import to_geojson, qa_summary
from .markers import Marker, MARKER_TYPES, closest_approach

__all__ = ["Log", "discover_logs", "load_log", "assess", "QaReport",
           "enrich", "convention_scoreboard", "wind_direction", "wind_sign_evidence",
           "to_geojson", "qa_summary",
           "Marker", "MARKER_TYPES", "closest_approach"]
