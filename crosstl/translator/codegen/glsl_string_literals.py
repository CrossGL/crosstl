"""String literals a GLSL program cannot contain.

GLSL has no string or character-pointer type. The double quote belongs to the character set only for the
``#include`` and ``#line`` directives (GLSL 4.60 and GLSL ES 3.20, section 3.1), and GL_EXT_debug_printf adds a
string literal only as the format argument of ``debugPrintfEXT``. A Metal or C-like source can pass strings to
helpers (``constant char*`` parameters, ``__FILE__``), and emitting such a call yields a shader every GLSL compiler
rejects, so the OpenGL and WebGL emitters scan the rendered program and refuse each other string literal.
"""

from __future__ import annotations

import re

__all__ = ["string_literal_reasons"]

# Comments and string literals in source order, so a ``//`` inside a string is not taken for a comment.
_TOKEN = re.compile(r'(//[^\n]*|/\*.*?\*/)|("(?:\\.|[^"\\\n])*")', re.S)
_PRINTF_FORMAT_PREFIX = re.compile(r"\bdebugPrintfEXT\s*\(\s*$")
_PREPROCESSOR_LINE = re.compile(r"^\s*#")
_SHOWN_CHARACTERS = 40

STRING_LITERAL_REASON = (
    "GLSL has no string type: a string literal may appear only in #include and #line directives "
    "(GLSL 4.60 section 3.1) or as the format argument of debugPrintfEXT (GL_EXT_debug_printf)"
)


def _shown(literal):
    if len(literal) <= _SHOWN_CHARACTERS:
        return literal
    return literal[: _SHOWN_CHARACTERS - 4] + '..."'


def string_literal_reasons(code):
    """Return one refusal reason per distinct string literal of ``code`` that GLSL cannot contain, in order."""
    if not isinstance(code, str) or '"' not in code:
        return []
    reasons = []
    for match in _TOKEN.finditer(code):
        literal = match.group(2)
        if literal is None:
            continue
        line_start = code.rfind("\n", 0, match.start()) + 1
        prefix = code[line_start : match.start()]
        if _PREPROCESSOR_LINE.match(prefix):
            continue
        if _PRINTF_FORMAT_PREFIX.search(
            code[max(0, match.start() - 64) : match.start()]
        ):
            continue
        entry = f"string literal {_shown(literal)}: {STRING_LITERAL_REASON}"
        if entry not in reasons:
            reasons.append(entry)
    return reasons
