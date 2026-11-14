"""Measured backend coverage.

Each feature of a target language gets a minimal probe program. The probe is translated with CrossTL and the
result is checked with the language's reference validator, so every number in ``coverage.json`` names the
validator and the pinned feature list it came from. ``python -m tools.coverage run`` writes the report.
"""
