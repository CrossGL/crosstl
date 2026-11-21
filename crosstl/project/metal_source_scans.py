"""Skip-ahead helpers for the character scans of Metal project materialization.

The materialization passes in :mod:`crosstl.project.pipeline` walk Metal source text one character at a time, and
most characters only advance the scan. These helpers find the next character at which a scan can act, so that the
scan jumps over the others and reaches the same positions with the same state, only sooner.

A stop pattern may match more characters than a scan acts on. The scan re-examines the character it lands on, so a
superset changes nothing; a pattern that missed a character the scan acts on would change the result.
"""

from __future__ import annotations

import re

__all__ = [
    "next_identifier_scan_stop",
    "next_statement_scan_stop",
]

# An identifier scan acts on a quote that opens a literal, a slash that may open a comment, and a character that may
# begin an identifier. ``[^\W\d]`` matches the underscore and every character ``str.isalpha`` accepts, because
# ``\w`` covers ``str.isalnum`` and no alphabetic character is a decimal digit.
_IDENTIFIER_SCAN_STOP = re.compile(r"[\"'/]|[^\W\d]")

# A statement scan also acts on the delimiters whose depth it tracks and on the semicolon that ends a statement.
_STATEMENT_SCAN_STOP = re.compile(r"[\"'/;()\[\]<>{}]")


def next_identifier_scan_stop(source: str, start: int, end: int) -> int:
    """The first position in ``source[start:end]`` an identifier scan acts on, or ``end``."""
    match = _IDENTIFIER_SCAN_STOP.search(source, start, end)
    return end if match is None else match.start()


def next_statement_scan_stop(source: str, start: int, end: int) -> int:
    """The first position in ``source[start:end]`` a statement scan acts on, or ``end``."""
    match = _STATEMENT_SCAN_STOP.search(source, start, end)
    return end if match is None else match.start()
