"""WebGL 2 built-in functions from ANGLE's built-in function table (``builtin_function_declarations.txt``).

WebGL 2 shaders are GLSL ES 3.00, and ANGLE, the shader translator the major browsers validate them with, lists every
built-in function with the GLSL ES level that introduced it. The list keeps the functions of
GLSL ES 1.00 and 3.00 (``COMMON_BUILTINS`` and ``ESSL3_BUILTINS``) that need no extension. Vertex-only functions are
left out, because a probe runs in a fragment shader: GLSL ES 3.00 has no compute stage. Prototypes use the generic
type names of the GLSL ES specification, made concrete as the GLSL list makes them.
"""

from __future__ import annotations

import json
import re

from ..model import Feature, Signature
from ..objects import UNIFORMS
from .common import pick, signature, slug
from .glsl import concrete_type

LEVELS = frozenset({"COMMON_BUILTINS", "ESSL3_BUILTINS"})
# GLSL ES writes ``genType`` where GLSL 4.60 writes ``genFType``.
_GENERIC = {"genType": "genFType"}
_GROUP_BEGIN = re.compile(r"GROUP BEGIN (\w+)\s*(\{.*\})?$")
_GROUP_END = re.compile(r"GROUP END (\w+)$")
_DEFAULT = re.compile(r"DEFAULT METADATA (\{.*\})$")
_DECLARATION = re.compile(r"(\w+)\s+(\w+)\s*\((.*)\)\s*;$")


def _declarations(text: str):
    """Yield ``(group path, metadata, result, name, parameters)`` for every function the table declares."""
    groups: list[tuple[str, dict]] = []
    default: dict = {}
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        begin = _GROUP_BEGIN.fullmatch(line)
        if begin:
            groups.append((begin.group(1), json.loads(begin.group(2) or "{}")))
            continue
        if _GROUP_END.fullmatch(line):
            groups.pop()
            # Default metadata lasts until the end of a top-level group.
            if not groups:
                default = {}
            continue
        metadata = _DEFAULT.fullmatch(line)
        if metadata:
            default = json.loads(metadata.group(1))
            continue
        declaration = _DECLARATION.fullmatch(line)
        if not declaration:
            raise ValueError(f"unrecognised line in ANGLE's built-in table: {line!r}")
        merged: dict = {}
        for _, group_metadata in groups:
            merged.update(group_metadata)
        merged.update(default)
        path = "/".join(slug(name) for name, _ in groups)
        yield path, merged, *declaration.groups()


def _kept(metadata: dict) -> bool:
    return (
        metadata.get("essl_level") in LEVELS
        and not metadata.get("essl_extension")
        and metadata.get("shader_type") in (None, "FRAGMENT")
    )


def _signature(name: str, result: str, params: str) -> Signature:
    parts = []
    for text in filter(None, (p.strip() for p in params.split(","))):
        words = text.split()
        direction = words.pop(0) if words[0] in ("out", "inout") else "in"
        parts.append((concrete_type(_GENERIC.get(words[-1], words[-1])), direction))
    if name.endswith("Offset"):
        # A texel offset, the last integer vector parameter, must be a constant expression (GLSL ES 3.00,
        # section 8.8).
        for index in range(len(parts) - 1, -1, -1):
            kind, _ = parts[index]
            if kind.scalar == "int" and kind.shape:
                parts[index] = (kind, "const")
                break
    return signature(concrete_type(_GENERIC.get(result, result)), parts)


def overloads(text: str) -> dict[str, tuple[str, list[Signature]]]:
    """Every kept prototype, by function name: ``{name: (group, [signature, ...])}``."""
    found: dict[str, tuple[str, list[Signature]]] = {}
    for group, metadata, result, name, params in _declarations(text):
        if _kept(metadata):
            found.setdefault(name, (group, []))[1].append(
                _signature(name, result, params)
            )
    return found


def parse(text: str) -> list[Feature]:
    return [
        Feature(
            "webgl",
            "function",
            name,
            pick(signatures, UNIFORMS["webgl"]),
            group=group,
            crossgl=name,
            stage="fragment",
        )
        for name, (group, signatures) in sorted(overloads(text).items())
    ]
