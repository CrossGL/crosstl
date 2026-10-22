"""Append host dispatch records, optionally without their long value arrays."""

import hashlib
import json

from crosstl.project.runtime_values import exact_types_within

COMPACT_TRACE_LIMIT = 256
_SCALAR_TYPES = frozenset({type(None), bool, int, float, str})


def compact_trace_values(value):
    """Replace scalar arrays longer than the limit with their length and digest.

    Dictionaries and short or nested arrays keep their structure, so a compact
    record still names every entry, launch, artifact and diagnostic. The digest
    covers the array's compact JSON encoding.
    """
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
