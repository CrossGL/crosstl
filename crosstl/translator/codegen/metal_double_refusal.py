"""Refuse double-precision values in Metal output.

The Metal Shading Language has no double-precision type: "Metal does not support the double, long long, unsigned
long long, and long double data types" (Metal Shading Language Specification, section 2.1, "Scalar Data Types").
The Metal compiler rejects ``double`` with "'double' is not supported in Metal", and ``double2`` to ``double4``
are not usable vector types either. The emitter used to print CrossGL ``double``, ``dvecN`` and ``dmatN`` values
as ``double``, ``doubleN`` or the GLSL spelling unchanged, so the program failed to compile instead of being
refused. Emulating binary64 with pairs of binary32 values would not round like binary64, so no lowering is
faithful; generation records a refusal instead.

The check reads the rendered program, so it covers every place a type or a constructor is printed: declarations,
parameters, struct members, casts and constructor calls.
"""

from __future__ import annotations

import re

__all__ = ["refuse_metal_double_values"]

_DOUBLE_TOKEN = re.compile(
    r"\b(?:long\s+)?(?:double(?:[234](?:x[234])?)?|dvec[234]|dmat[234](?:x[234])?)\b"
)
# Comments and string literals cannot name a type; they are blanked before the scan.
_NON_CODE = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"', re.DOTALL)


def _code_only(source: str) -> str:
    return _NON_CODE.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), source)


def refuse_metal_double_values(output, refuse) -> None:
    """Call ``refuse(reason)`` once if the rendered Metal program ``output`` uses a double-precision type."""
    if not isinstance(output, str):
        return
    code = _code_only(output)
    match = _DOUBLE_TOKEN.search(code)
    if match is None:
        return
    line_number = code.count("\n", 0, match.start()) + 1
    line = output.splitlines()[line_number - 1].strip()
    spellings = sorted(
        {" ".join(token.split()) for token in _DOUBLE_TOKEN.findall(code)}
    )
    refuse(
        "double-precision value: Metal has no double type (Metal Shading Language "
        f"Specification, section 2.1); the program uses {', '.join(spellings)}, "
        f"first at line {line_number}: {line}"
    )
