"""Find the comments in which a translation says that CrossTL wrote a placeholder for what it could not translate.

CrossTL must refuse what it cannot translate. Some of its importers and generators still write a comment and a
neutral value instead, such as ``(/* HIP device timer clock64() not directly supported in CrossGL */ 0)``. The
result compiles, so the reference tool alone would count the feature as supported. The words that mark a loss are
the ones ``tests/conformance/fail_closed.py`` looks for in the translator's source, and "not supported".
"""

from __future__ import annotations

import re

# The first line of a wrong result's diagnostic. It is the same for every placeholder, so the wrong results of one
# row that it explains are one cause and one bug issue (:mod:`.causes`); the comment follows on the next line.
PLACEHOLDER = "CrossTL wrote a placeholder for what it could not translate"

# Line and block comments of the C family, and line comments of SPIR-V assembly (;) and Mojo (#).
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|^[ \t]*;[^\n]*|#[^\n]*", re.S | re.M)
_LOSS = re.compile(
    r"^(?://|/\*|;|#)\s*WARNING\b"
    r"|\b(?:not (?:directly )?supported|unsupported|placeholder|using default value)\b",
    re.I,
)


def placeholder(source: str) -> str:
    """The first comment of ``source``, a translation by CrossTL, that marks a loss, on one line; empty when there is
    none."""
    for comment in _COMMENT.finditer(source):
        text = comment.group().strip()
        if _LOSS.search(text):
            return " ".join(text.split())
    return ""


def detail(comment: str, prefix: str = "") -> str:
    """The diagnostic of a result whose translation has the placeholder ``comment``."""
    return f"{prefix}{PLACEHOLDER}\n{comment}"
