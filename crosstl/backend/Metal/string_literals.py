"""Concatenation of adjacent Metal string literals.

C++ joins adjacent string literals into one literal in translation phase 6, so ``"a" "b"`` is the single
literal ``"ab"``. The Metal parser folds such a run into one ordinary literal spelling that means the same
characters:

- a raw literal ``R"d(...)d"`` contributes its body with backslashes, quotes and line breaks escaped;
- an escape at the end of one piece must not absorb characters at the start of the next one (``"\\x4" "1"`` is
  the two characters ``\\x04 1``, not ``\\x41``), so such a trailing hexadecimal or short octal escape is
  rewritten as a three-digit octal escape before the pieces are joined.
"""

import re
from typing import Sequence

from crosstl.translator.refusals import MetalSourceRefusal

_ORDINARY = re.compile(r'"((?:\\.|[^"\\])*)"\Z', re.DOTALL)
_RAW = re.compile(r'R"(?P<delim>[^\s()\\]{0,16})\((?P<body>[\s\S]*?)\)(?P=delim)"\Z')
_ESCAPE = re.compile(r"\\(?:x[0-9A-Fa-f]+|[0-7]{1,3}|.)", re.DOTALL)
_HEX_DIGIT = re.compile(r"[0-9A-Fa-f]")
_OCTAL_DIGIT = re.compile(r"[0-7]")
_MAX_NARROW_CHARACTER = 0o377


def _literal_body(spelling: str) -> str:
    """Return the escaped body of one literal, as it would appear between the quotes of an ordinary literal."""
    raw = _RAW.match(spelling)
    if raw is not None:
        body = raw.group("body")
        return (
            body.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r", "\\r")
            .replace("\n", "\\n")
        )
    ordinary = _ORDINARY.match(spelling)
    if ordinary is None:
        raise MetalSourceRefusal(f"not a Metal string literal: {spelling!r}")
    return ordinary.group(1)


def _seal_trailing_escape(body: str, following: str) -> str:
    """Rewrite an escape that ends ``body`` when the next piece would extend it."""
    if not following:
        return body
    last = None
    for match in _ESCAPE.finditer(body):
        last = match
    if last is None or last.end() != len(body):
        return body
    escape = last.group(0)
    if escape.startswith("\\x") and _HEX_DIGIT.match(following):
        value = int(escape[2:], 16)
    elif escape[1:].isdigit() and len(escape) < 4 and _OCTAL_DIGIT.match(following):
        value = int(escape[1:], 8)
    else:
        return body
    if value > _MAX_NARROW_CHARACTER:
        raise MetalSourceRefusal(
            f"escape '{escape}' does not fit a narrow character and cannot be concatenated"
        )
    return body[: last.start()] + f"\\{value:03o}"


def concatenate_string_literals(spellings: Sequence[str]) -> str:
    """Fold adjacent string literal spellings into one ordinary literal spelling."""
    if not spellings:
        raise MetalSourceRefusal("no string literals to concatenate")
    if len(spellings) == 1:
        return spellings[0]
    bodies = [_literal_body(spelling) for spelling in spellings]
    joined = []
    for index, body in enumerate(bodies):
        following = next((part for part in bodies[index + 1 :] if part), "")
        joined.append(_seal_trailing_escape(body, following))
    return '"' + "".join(joined) + '"'
