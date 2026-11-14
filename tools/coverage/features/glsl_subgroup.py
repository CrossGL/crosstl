"""GLSL subgroup functions from the extensions that add them to section 8.18 (KhronosGroup/GLSL).

:data:`SOURCES` are the extensions whose functions compile to SPIR-V's ``OpGroupNonUniform*`` instructions:
GL_KHR_shader_subgroup, GL_KHR_shader_subgroup_rotate, GL_EXT_shader_quad_control and
GL_NV_shader_subgroup_partitioned. Each ``Syntax:`` block lists the prototypes of one function and is followed by the
extensions that enable it, such as ``GL_KHR_shader_subgroup_ballot``. The last extension named is the feature's group,
and a GLSL probe of it enables every one: subgroupQuadAll needs GL_KHR_shader_subgroup_vote besides
GL_EXT_shader_quad_control. Each specification also lists, as the expected mapping of GLSL to SPIR-V, the
``OpGroupNonUniform*`` instruction each function compiles to, which is how the SPIR-V feature list (:mod:`.spirv`)
probes those instructions.

An argument the specification requires to be an integral constant expression, such as the ``id`` of
``subgroupBroadcast`` or the ``clusterSize`` of the clustered operations, is passed as a constant.
"""

from __future__ import annotations

import re

from ..model import Feature, Param, Signature
from ..objects import UNIFORMS
from .common import pick
from .glsl import parse_prototype

# The upstream sources (:mod:`..upstream`) of the extensions, in the order their mappings are read.
SOURCES = (
    "glsl-subgroup",
    "glsl-subgroup-rotate",
    "glsl-quad-control",
    "glsl-subgroup-partitioned",
)
_CHAPTER = "Additions to Chapter 8"
# "Only usable if the extension GL_KHR_shader_subgroup_basic is enabled", or "if the extensions
# GL_KHR_shader_subgroup_vote and GL_EXT_shader_quad_control are enabled".
_SYNTAX = re.compile(
    r"^[ \t]*Syntax:\n\n(.*?)\n\n[ \t]*Only usable if the extensions?\s+(\w+(?:\s+and\s+\w+)*)\s+"
    r"(?:is|are)\s+enabled\.",
    re.M | re.S,
)
# "<id> must be an integral constant expression when targeting SPIR-V 1.4 and below", and "The <clusterSize> as used
# in the subgroupClustered<op>() operations must be: - An integral constant expression."
_CONSTANT = re.compile(
    r"<(\w+)>(?:\s+as\s+used\s+in\s[^.]*?)?\s+must\s+be:?\s+(?:-\s+)?an\s+integral\s+constant\s+expression",
    re.I,
)
# "subgroupAdd(value) -> OpGroupNonUniformIAdd | OpGroupNonUniformFAdd(": the instructions a function compiles to.
# GL_KHR_shader_subgroup_rotate puts the instruction on the next line, after "Opcode:", and
# GL_NV_shader_subgroup_partitioned names the function without its arguments: "subgroupPartitionNV -> ...".
_MAPPING = re.compile(
    r"^[ \t]+(subgroup\w+)(?:\([^)\n]*\))?[ \t]*->\s*(?:Opcode:[ \t]*)?"
    r"(OpGroupNonUniform\w+(?:[ \t]*\|[ \t]*OpGroupNonUniform\w+)*)\b",
    re.M,
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


def functions(*texts: str) -> dict[str, tuple[str, list[Signature]]]:
    """Every function the specifications ``texts`` add to section 8.18, by name: ``{name: (extensions,
    [signature, ...])}``, where ``extensions`` are the names of the extensions that enable it, separated by spaces.
    """
    # Constraints an overview states for a family of functions, as GL_KHR_shader_subgroup's for the clusterSize of
    # every subgroupClustered<op>, which also binds GL_KHR_shader_subgroup_rotate's subgroupClusteredRotate.
    shared = set().union(*(_constants(text[: text.index(_CHAPTER)]) for text in texts))
    found: dict[str, tuple[str, list[Signature]]] = {}
    for text in texts:
        for name, extensions, signature in _functions(text, shared):
            found.setdefault(name, (extensions, []))[1].append(signature)
    return found


def _functions(text: str, shared: set[str]):
    chapter = text[text.index(_CHAPTER) :]
    blocks = list(_SYNTAX.finditer(chapter))
    for index, block in enumerate(blocks):
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(chapter)
        constants = shared | _constants(chapter[block.end() : end])
        extensions = " ".join(re.split(r"\s+and\s+", block.group(2)))
        for line in block.group(1).splitlines():
            prototype = " ".join(line.split())
            signature = _signature(prototype, constants) if prototype else None
            if signature is not None:
                name = re.match(r"\w+\s+(\w+)", prototype).group(1)
                yield name, extensions, signature


def instructions(*texts: str) -> dict[str, tuple[str, str]]:
    """The ``OpGroupNonUniform*`` instruction each function compiles to: ``{instruction: (function, scalar)}``.

    ``scalar`` is the operand type that selects the instruction among those the function compiles to, or empty when
    it compiles to one only. An instruction several functions compile to, such as ``OpGroupNonUniformIAdd`` for the
    reduction, the scans and the clustered sum, is probed with the first function the specifications list for it.
    An instruction keeps the name the specification gives it, which may be an alias the SPIR-V grammar lists, as
    ``OpGroupNonUniformPartitionNV`` is of ``OpGroupNonUniformPartitionEXT``.
    """
    found: dict[str, tuple[str, str]] = {}
    for text in texts:
        for function, alternatives in _MAPPING.findall(text):
            names = [name.strip() for name in alternatives.split("|") if name.strip()]
            for name in names:
                operand = _OPERAND.match(name) if len(names) > 1 else None
                scalar = (
                    _SCALARS[operand.group(1) or operand.group(2)] if operand else ""
                )
                found.setdefault(name, (function, scalar))
    return found


def parse(*texts: str) -> list[Feature]:
    features = []
    for name, (extensions, signatures) in sorted(functions(*texts).items()):
        features.append(
            Feature(
                "glsl",
                "function",
                name,
                pick(signatures, UNIFORMS["glsl"]),
                group=extensions.split()[-1],
                crossgl=name,
                extension=extensions,
            )
        )
    return features
