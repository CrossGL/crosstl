"""Append host dispatch records, optionally without their long values."""

import hashlib
import json
import math
import sys
from array import array
from pathlib import Path

from crosstl.project.runtime_values import exact_types_within

COMPACT_TRACE_LIMIT = 256
# Longer strings, such as hex input observations, keep only length and digest.
COMPACT_STRING_LIMIT = 4096
# The record fields the upstream runners check after a native run.
SUMMARY_KEYS = ("entry", "target", "threads")
_SCALAR_TYPES = frozenset({type(None), bool, int, float, str})
# Bool cannot be subclassed; these return exact built-in values for the rest.
_BASE_CONVERSIONS = ((int, int.__int__), (float, float.__float__), (str, str.__str__))


def _little_endian(values):
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _typed_encoding(values):
    """Tagged machine words of a list of one exact type, or ``None``.

    Booleans, int64 integers and finite floats encode as bytes, int64 words
    and float64 words. Each such list has exactly one JSON encoding and one
    word encoding, so two lists share either encoding exactly when they share
    the other; NaN, infinities and wider integers keep their JSON encoding.
    """
    kind = type(values[0])
    if not exact_types_within(values, frozenset({kind})):
        return None
    if kind is bool:
        return b"bool\0" + bytes(values)
    if kind is int:
        try:
            return b"int64\0" + _little_endian(array("q", values))
        except OverflowError:
            return None
    # A finite sum excludes NaN and infinities.
    if kind is float and math.isfinite(sum(values)):
        return b"float64\0" + _little_endian(array("d", values))
    return None


def _plain(item):
    """``item`` as the exact built-in scalar ``json.dumps`` writes for it.

    ``json`` writes an instance of a subclass of ``int``, ``float`` or
    ``str`` through the base type's own conversion, as these do.
    """
    if item is None or type(item) in _SCALAR_TYPES:
        return item
    for kind, convert in _BASE_CONVERSIONS:
        if isinstance(item, kind):
            return convert(item)
    raise TypeError("Trace values must be JSON scalars")


def value_digest(values):
    """SHA-256 of a long scalar array, equal exactly when its JSON is.

    Arrays of one exact type hash their tagged words (``_typed_encoding``),
    which is several times faster than encoding them; others hash their
    compact JSON encoding, which no word encoding can equal.
    """
    if type(values) is not list:
        values = list(values)
    encoded = _typed_encoding(values) if values else None
    if encoded is None and not exact_types_within(values, _SCALAR_TYPES):
        values = [_plain(item) for item in values]
        encoded = _typed_encoding(values)
    if encoded is None:
        encoded = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def integer_digest(np, values, chunk=1 << 22):
    """``value_digest(values.tolist())`` of a NumPy integer array, in chunks.

    Every element of an integer array of at most 64 bits converts to an exact
    ``int`` that fits int64, so the list's digest covers its int64 words.
    """
    values = np.asarray(values).reshape(-1)
    if values.dtype.kind not in "iu" or values.dtype.itemsize > 8 or not values.size:
        raise ValueError("Integer digests need a non-empty integer array")
    if values.dtype == np.uint64 and values.max() > np.iinfo(np.int64).max:
        raise ValueError("Integer digests need values that fit int64")
    digest = hashlib.sha256(b"int64\0")
    for start in range(0, values.size, chunk):
        digest.update(values[start : start + chunk].astype("<i8").tobytes())
    return digest.hexdigest()


def recorded_length(values):
    """The element count of a retained list or of its compact digest."""
    if type(values) is list:
        return len(values)
    compact = values.get("compactedValues") if type(values) is dict else None
    if type(compact) is not dict or type(compact.get("length")) is not int:
        raise ValueError("Trace values are neither a list nor a compact digest")
    return compact["length"]


def matches_integers(np, recorded, expected):
    """Whether a trace field holds exactly the integers of ``expected``.

    ``recorded`` is the retained list of exact ``int`` values, or the
    ``compact_trace_values`` digest a list longer than ``COMPACT_TRACE_LIMIT``
    becomes.
    """
    expected = np.asarray(expected).reshape(-1)
    if type(recorded) is list:
        return (
            len(recorded) == expected.size
            and exact_types_within(recorded, frozenset({int}))
            and recorded == expected.tolist()
        )
    compact = recorded.get("compactedValues") if type(recorded) is dict else None
    return (
        type(compact) is dict
        and set(recorded) == {"compactedValues"}
        and set(compact) == {"length", "sha256"}
        and compact["length"] == expected.size > COMPACT_TRACE_LIMIT
        and compact["sha256"] == integer_digest(np, expected)
    )


def compact_trace_values(value):
    """Replace long scalar arrays and long strings with their length and digest.

    Dictionaries and short or nested arrays keep their structure, so a compact
    record still names every entry, launch, artifact and diagnostic. An array's
    digest is ``value_digest``, a string's covers its UTF-8 bytes.
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
            return {
                "compactedValues": {"length": len(value), "sha256": value_digest(value)}
            }
        return [compact_trace_values(item) for item in value]
    return value


def retains_values(host, entry):
    """Whether ``host`` keeps the long values of ``entry``'s records.

    ``host.trace_values`` is a Boolean for every entry, or a tuple of entry
    prefixes whose records keep their values while all others are compacted.
    """
    retained = getattr(host, "trace_values", True)
    if type(retained) is tuple:
        return isinstance(entry, str) and entry.startswith(retained)
    return bool(retained)


def write_trace(host, event, *, prepared=False):
    """Append ``event``; compact it when the host does not retain its values.

    A ``prepared`` event already replaced the values its audit recomputes by
    their digests and keeps the ones the audit reads, so it is written as is.
    """
    if not prepared and not retains_values(host, event.get("entry")):
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


class TraceStream:
    """Iterate a dispatch trace record by record, counting and hashing it.

    A long native run writes gigabytes of records, which would not fit a
    runner's memory once decoded. ``records``, ``bytes`` and ``summary()``
    describe the lines read so far.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.records = 0
        self.bytes = 0
        self._digest = hashlib.sha256()

    def __iter__(self):
        with self.path.open("rb") as handle:
            for line in handle:
                self._digest.update(line)
                self.bytes += len(line)
                self.records += 1
                yield json.loads(line)

    def summary(self):
        return {
            "records": self.records,
            "bytes": self.bytes,
            "sha256": self._digest.hexdigest(),
        }
