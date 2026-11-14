"""Append host dispatch records, optionally without their long values."""

import hashlib
import json

from crosstl.project.runtime_values import exact_types_within

COMPACT_TRACE_LIMIT = 256
# Longer strings, such as hex input observations, keep only length and digest.
COMPACT_STRING_LIMIT = 4096
# The record fields the upstream runners check after a native run.
SUMMARY_KEYS = ("entry", "target", "threads")
_SCALAR_TYPES = frozenset({type(None), bool, int, float, str})


def compact_trace_values(value):
    """Replace long scalar arrays and long strings with their length and digest.

    Dictionaries and short or nested arrays keep their structure, so a compact
    record still names every entry, launch, artifact and diagnostic. An array's
    digest covers its compact JSON encoding, a string's its UTF-8 bytes.
    """
    if isinstance(value, str):
        if len(value) <= COMPACT_STRING_LIMIT:
            return value
        return {
            "compactedString": {
                "length": len(value),
                "sha256": hashlib.sha256(value.encode()).hexdigest(),
            }
        }
    if isinstance(value, dict):
        return {key: compact_trace_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > COMPACT_TRACE_LIMIT and (
            exact_types_within(value, _SCALAR_TYPES)
            or all(
                item is None or isinstance(item, (bool, int, float, str))
                for item in value
            )
        ):
            encoded = json.dumps(list(value), separators=(",", ":")).encode()
            return {
                "compactedValues": {
                    "length": len(value),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            }
        return [compact_trace_values(item) for item in value]
    return value


def write_trace(host, event):
    """Append ``event``; compact it when the host does not retain trace values."""
    if not getattr(host, "trace_values", True):
        event = compact_trace_values(event)
    with host.trace.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def read_trace_summary(path, keys=SUMMARY_KEYS):
    """Stream ``path`` and keep only ``keys`` of each record that has them.

    A long native run writes gigabytes of records; loading them whole can
    exhaust a runner's memory before the trace is checked.
    """
    summary = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            summary.append({key: record[key] for key in keys if key in record})
    return summary
