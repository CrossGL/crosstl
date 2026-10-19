"""Shared helpers for the feature-list parsers."""

from __future__ import annotations

import re

from ..model import Param, Signature, Type

# Vector and matrix spellings shared by GLSL and CrossGL.
_VECTOR = re.compile(r"(b|i|u|d)?vec([2-4])")
_MATRIX = re.compile(r"(d)?mat([2-4])(?:x([2-4]))?")
_GLSL_SCALARS = {
    "bool": "bool",
    "int": "int",
    "uint": "uint",
    "float": "float",
    "double": "double",
}
_VECTOR_PREFIX = {None: "float", "b": "bool", "i": "int", "u": "uint", "d": "double"}


def glsl_type(spelling: str) -> Type:
    """Parse a GLSL type spelling; anything that is not a value type becomes opaque."""
    if spelling in _GLSL_SCALARS:
        return Type(_GLSL_SCALARS[spelling])
    match = _VECTOR.fullmatch(spelling)
    if match:
        return Type(_VECTOR_PREFIX[match.group(1)], (int(match.group(2)),))
    match = _MATRIX.fullmatch(spelling)
    if match:
        columns = int(match.group(2))
        rows = int(match.group(3) or columns)
        return Type("double" if match.group(1) else "float", (columns, rows))
    return Type.other(spelling)


def pick(overloads: list[Signature]) -> Signature:
    """The overload a probe uses: the first one built from value types only, else the first one."""
    for signature in overloads:
        if not signature.unsupported:
            return signature
    return overloads[0]


def signature(result: Type, params: list[tuple[Type, str]]) -> Signature:
    return Signature(
        result, tuple(Param(kind, direction) for kind, direction in params)
    )


def slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def split(text: str) -> list[str]:
    """Split on the commas that are not inside ``<...>`` or ``(...)``."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "<(":
            depth += 1
        elif char in ">)":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts
