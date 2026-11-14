"""GLSL subgroup functions from the GL_KHR_shader_subgroup specification (KhronosGroup/GLSL).

The extension adds section 8.18 of the GLSL specification. Each ``Syntax:`` block lists the prototypes of one
function and is followed by the extension that enables it, such as ``GL_KHR_shader_subgroup_ballot``; that extension
is the feature's group, and a GLSL probe of it enables the extension. The specification also lists, as the expected
mapping of GLSL to SPIR-V, the ``OpGroupNonUniform*`` instruction each function compiles to, which is how the SPIR-V
feature list (:mod:`.spirv`) probes those instructions.

An argument the specification requires to be an integral constant expression, such as the ``id`` of
``subgroupBroadcast`` or the ``clusterSize`` of the clustered operations, is passed as a constant.
"""

from __future__ import annotations

import re

from ..model import Feature, Param, Signature
from ..objects import UNIFORMS
from .common import pick
from .glsl import parse_prototype

_CHAPTER = "Additions to Chapter 8"
_SYNTAX = re.compile(
    r"^[ \t]*Syntax:\n\n(.*?)\n\n[ \t]*Only usable if the extension\s+(\w+)\s+is\s+enabled\.",
    re.M | re.S,
)
# "<id> must be an integral constant expression when targeting SPIR-V 1.4 and below", and "The <clusterSize> as used
# in the subgroupClustered<op>() operations must be: - An integral constant expression."
_CONSTANT = re.compile(
    r"<(\w+)>(?:\s+as\s+used\s+in\s[^.]*?)?\s+must\s+be:?\s+(?:-\s+)?an\s+integral\s+constant\s+expression",
    re.I,
)
# "subgroupAdd(value) -> OpGroupNonUniformIAdd | OpGroupNonUniformFAdd(": the instructions a function compiles to.
_MAPPING = re.compile(
    r"^\s+(subgroup\w+)\([^)]*\)\s*->\s*((?:\s*\|?\s*OpGroupNonUniform\w+)+)\(", re.M
)
# The operand type an instruction's name selects: OpGroupNonUniformIAdd takes integers, FAdd floats, SMin signed and
# UMin unsigned integers, and the bitwise and logical operations integers and booleans.
_OPERAND = re.compile(r"OpGroupNonUniform(?:(I|F|S|U)(?=[A-Z])|(Bitwise|Logical))")
_SCALARS = {
    "I": "int",
    "F": "float",
    "S": "int",
    "U": "uint",
    "Bitwise": "uint",
    "Logical": "bool",
}


def _constants(description: str) -> set[str]:
    return {match.group(1) for match in _CONSTANT.finditer(description)}


def _signature(prototype: str, constants: set[str]) -> Signature | None:
    parsed = parse_prototype(prototype)
    if parsed is None:
        return None
    _, signature = parsed
    names = [
        part.split()[-1]
        for part in prototype[prototype.index("(") + 1 : prototype.rindex(")")].split(
            ","
        )
    ]
    params = tuple(
        Param(param.type, "const") if name in constants else param
        for param, name in zip(signature.params, names)
    )
    return Signature(signature.result, params)


def functions(text: str) -> dict[str, tuple[str, list[Signature]]]:
    """Every function of section 8.18, by name: ``{name: (extension, [signature, ...])}``."""
    chapter = text[text.index(_CHAPTER) :]
    # Constraints the overview states for a family of functions, as for every clustered operation's clusterSize.
    shared = _constants(text[: text.index(_CHAPTER)])
    blocks = list(_SYNTAX.finditer(chapter))
    found: dict[str, tuple[str, list[Signature]]] = {}
    for index, block in enumerate(blocks):
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(chapter)
        constants = shared | _constants(chapter[block.end() : end])
        for line in block.group(1).splitlines():
            prototype = " ".join(line.split())
            signature = _signature(prototype, constants) if prototype else None
            if signature is None:
                continue
            name = re.match(r"\w+\s+(\w+)", prototype).group(1)
            found.setdefault(name, (block.group(2), []))[1].append(signature)
    return found


def instructions(text: str) -> dict[str, tuple[str, str]]:
    """The ``OpGroupNonUniform*`` instruction each function compiles to: ``{instruction: (function, scalar)}``.

    ``scalar`` is the operand type that selects the instruction among those the function compiles to, or empty when
    it compiles to one only. An instruction several functions compile to, such as ``OpGroupNonUniformIAdd`` for the
    reduction, the scans and the clustered sum, is probed with the first function the specification lists for it.
    """
    found: dict[str, tuple[str, str]] = {}
    for function, alternatives in _MAPPING.findall(text):
        names = [name.strip() for name in alternatives.split("|") if name.strip()]
        for name in names:
            operand = _OPERAND.match(name) if len(names) > 1 else None
            scalar = _SCALARS[operand.group(1) or operand.group(2)] if operand else ""
            found.setdefault(name, (function, scalar))
    return found


def parse(text: str) -> list[Feature]:
    features = []
    for name, (extension, signatures) in sorted(functions(text).items()):
        features.append(
            Feature(
                "glsl",
                "function",
                name,
                pick(signatures, UNIFORMS["glsl"]),
                group=extension,
                crossgl=name,
                extension=extension,
            )
        )
    return features
