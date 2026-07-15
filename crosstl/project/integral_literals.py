"""Parse C-family integral literals used by project metadata."""

from __future__ import annotations

import re

_INTEGER_SUFFIX = r"(?:[uU](?:[lL]{1,2})?|[lL]{1,2}[uU]?)"
_INTEGER_LITERAL = re.compile(
    r"^(?P<body>"
    r"0[xX][0-9A-Fa-f](?:'?[0-9A-Fa-f])*|"
    r"0[bB][01](?:'?[01])*|"
    r"0(?:'?[0-7])*|"
    r"[1-9](?:'?[0-9])*"
    rf")(?P<suffix>{_INTEGER_SUFFIX})?$"
)


class CFamilyIntegralLiteralError(ValueError):
    """A value is not a supported C-family integral literal."""

    def __init__(self, spelling: str, reason: str) -> None:
        self.spelling = spelling
        self.reason = reason
        super().__init__(f"invalid C-family integral literal {spelling!r}: {reason}")


def _invalid_literal_reason(spelling: str) -> str:
    if not spelling:
        return "missing"
    if spelling.startswith("-"):
        return "negative"
    if spelling.startswith("+") or re.search(r"\s", spelling):
        return "unsupported-expression"
    if "''" in spelling or spelling.startswith("'") or spelling.endswith("'"):
        return "invalid-digit-separator"

    lowered = spelling.lower()
    if lowered.startswith("0x"):
        body = re.match(r"0[xX]([0-9A-Za-z']*)", spelling)
        digits = body.group(1) if body is not None else ""
        if not digits or any(
            character not in "0123456789abcdefABCDEF'" for character in digits
        ):
            return "invalid-digit"
    elif lowered.startswith("0b"):
        body = re.match(r"0[bB]([0-9A-Za-z']*)", spelling)
        digits = body.group(1) if body is not None else ""
        if not digits or any(character not in "01'" for character in digits):
            return "invalid-digit"
    elif re.match(r"^0[0-9']", spelling) and any(
        character in "89" for character in spelling
    ):
        return "invalid-octal-digit"

    if re.match(r"^[0-9]", spelling) and re.search(r"[A-Za-z_]", spelling):
        return "invalid-suffix"
    if re.search(r"[()+\-*/%<>&|^~?:,]", spelling):
        return "unsupported-expression"
    return "invalid-literal"


def parse_c_family_integral_literal(spelling: str) -> int:
    """Return the numeric value of one non-negative C-family integer literal."""

    if not isinstance(spelling, str):
        raise TypeError("C-family integral literal spelling must be a string")
    text = spelling.strip()
    match = _INTEGER_LITERAL.fullmatch(text)
    if match is None:
        raise CFamilyIntegralLiteralError(text, _invalid_literal_reason(text))

    body = match.group("body").replace("'", "")
    if body.lower().startswith("0x"):
        return int(body[2:], 16)
    if body.lower().startswith("0b"):
        return int(body[2:], 2)
    if len(body) > 1 and body.startswith("0"):
        return int(body, 8)
    return int(body, 10)
