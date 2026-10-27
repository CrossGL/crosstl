"""Fixed-width integer literal typing shared by source and canonical parsing."""

import re


class IntegerLiteralError(ValueError):
    """An integer spelling has no supported, exact source type."""

    project_diagnostic_code = "project.translate.integer-literal-unsupported"


_LEGACY_OCTAL = re.compile(r"0(?P<digits>[0-9]+)(?P<suffix>[uUlL]*)")


def crossgl_legacy_octal(text):
    """``text`` with a C-family octal integer literal, ``017`` or ``017u``, in CrossGL's ``0o17`` spelling.

    C11 6.4.4.1, C++17 [lex.icon], GLSL 4.60 section 4.1.3, HLSL and Slang read an integer literal with a leading
    zero as octal, while CrossGL reads ``017`` as decimal. Other spellings are returned unchanged. A leading-zero
    integer with an 8 or a 9 is not an integer literal in those languages and raises :class:`IntegerLiteralError`.
    """
    match = _LEGACY_OCTAL.fullmatch(str(text))
    if match is None:
        return text
    digits = match.group("digits")
    if not set(digits) <= set("01234567"):
        raise IntegerLiteralError(f"Invalid octal integer literal '{text}'")
    return f"0o{digits}{match.group('suffix')}"


def integer_literal_parts(value, *, legacy_octal=False):
    """Return digits and the first representable 32/64-bit C++ integer type."""
    spelling = str(value)
    match = re.fullmatch(
        r"(?P<digits>0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|[0-9]+)"
        r"(?P<suffix>[uU](?:ll|LL|[lL])?|(?:ll|LL|[lL])[uU]?)?",
        spelling,
    )
    if match is None:
        raise IntegerLiteralError(f"Invalid integer literal '{spelling}'")
    digits = match.group("digits")
    suffix = (match.group("suffix") or "").lower()
    if (
        legacy_octal
        and len(digits) > 1
        and digits.startswith("0")
        and digits[1].isdigit()
    ):
        if not re.fullmatch(r"0[0-7]+", digits):
            raise IntegerLiteralError(f"Invalid octal integer literal '{spelling}'")
        digits = "0o" + digits[1:]
    nondecimal = digits.lower().startswith(("0x", "0b", "0o"))
    number = int(digits, 0 if nondecimal else 10)
    if "u" in suffix:
        candidates = ("uint64_t",) if "l" in suffix else ("uint", "uint64_t")
    elif "l" in suffix:
        candidates = ("int64_t", "uint64_t") if nondecimal else ("int64_t",)
    elif nondecimal:
        candidates = ("int", "uint", "int64_t", "uint64_t")
    else:
        candidates = ("int", "int64_t")
    limits = {
        "int": 2**31 - 1,
        "uint": 2**32 - 1,
        "int64_t": 2**63 - 1,
        "uint64_t": 2**64 - 1,
    }
    for type_name in candidates:
        if number <= limits[type_name]:
            return digits, type_name
    raise IntegerLiteralError(
        f"Integer literal '{spelling}' exceeds its supported source type range"
    )
