"""Name the ids of SPIR-V assembly as ``spirv-as`` followed by ``spirv-dis`` would.

``crosstl.translate(..., backend="vulkan")`` formats SPIR-V through SPIRV-Tools when ``spirv-as`` and ``spirv-dis``
are installed, which renumbers the ids and gives them friendly names (``%uint``, ``%v4float``,
``%_ptr_Uniform_uint``, ``%int_0``, ``%Params``). Without SPIRV-Tools the same call returns the generator's numeric
ids. Tests that read SPIR-V assembly translate with ``format_output=False`` and pass the text through
``friendly_spirv`` so that they see the same text in either environment.

The naming follows SPIRV-Tools' ``FriendlyNameMapper`` (source/name_mapper.cpp): ``OpName`` first, then built-in
decorations, types and scalar constants, with ``_0``, ``_1``... appended to repeated names.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import crosstl

__all__ = ["friendly_spirv", "translate_spirv"]

_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')
_VALID = re.compile(r"[^A-Za-z0-9_]")
_INDENT = 15

_BUILTIN_NAMES = {
    "Position": "gl_Position",
    "PointSize": "gl_PointSize",
    "ClipDistance": "gl_ClipDistance",
    "CullDistance": "gl_CullDistance",
    "VertexId": "gl_VertexID",
    "InstanceId": "gl_InstanceID",
    "PrimitiveId": "gl_PrimitiveID",
    "InvocationId": "gl_InvocationID",
    "Layer": "gl_Layer",
    "ViewportIndex": "gl_ViewportIndex",
    "TessLevelOuter": "gl_TessLevelOuter",
    "TessLevelInner": "gl_TessLevelInner",
    "TessCoord": "gl_TessCoord",
    "PatchVertices": "gl_PatchVertices",
    "FragCoord": "gl_FragCoord",
    "PointCoord": "gl_PointCoord",
    "FrontFacing": "gl_FrontFacing",
    "SampleId": "gl_SampleID",
    "SamplePosition": "gl_SamplePosition",
    "SampleMask": "gl_SampleMask",
    "FragDepth": "gl_FragDepth",
    "HelperInvocation": "gl_HelperInvocation",
    "NumWorkgroups": "gl_NumWorkGroups",
    "WorkgroupSize": "gl_WorkGroupSize",
    "WorkgroupId": "gl_WorkGroupID",
    "LocalInvocationId": "gl_LocalInvocationID",
    "GlobalInvocationId": "gl_GlobalInvocationID",
    "LocalInvocationIndex": "gl_LocalInvocationIndex",
    "VertexIndex": "gl_VertexIndex",
    "InstanceIndex": "gl_InstanceIndex",
    "BaseInstance": "gl_BaseInstance",
    "BaseVertex": "gl_BaseVertex",
    "DrawIndex": "gl_DrawID",
    "SubgroupSize": "SubgroupSize",
    "NumSubgroups": "NumSubgroups",
    "SubgroupId": "SubgroupId",
    "SubgroupLocalInvocationId": "SubgroupLocalInvocationId",
}

_INT_ROOTS = {8: "char", 16: "short", 32: "int", 64: "long"}
_FLOAT_ROOTS = {16: "half", 32: "float", 64: "double"}


def _split(line):
    """``(tokens, comment)`` of one assembly line; a ``;`` inside a string is not a comment."""
    tokens = []
    for match in _TOKEN.finditer(line):
        token = match.group(0)
        if token.startswith(";"):
            return tokens, line[match.start() :]
        tokens.append(token)
    return tokens, ""


def _float_text(value, width):
    if width == 64:
        return f"{value:.17g}"
    return f"{struct.unpack('<f', struct.pack('<f', value))[0]:.9g}"


class _Names:
    def __init__(self):
        self.by_id = {}
        self.used = set()

    def save(self, id_, suggested):
        if id_ in self.by_id:
            return
        base = _VALID.sub("_", suggested) if suggested else "_"
        name = base
        index = 0
        while name in self.used:
            name = f"{base}_{index}"
            index += 1
        self.used.add(name)
        self.by_id[id_] = name

    def get(self, id_):
        return self.by_id.get(id_, id_.lstrip("%"))


def _renumber(lines):
    """Number ids by first appearance, as ``spirv-as`` does without ``--preserve-numeric-ids``."""
    numbers = {}
    renumbered = []
    for tokens, comment in lines:
        new_tokens = []
        for token in tokens:
            if token.startswith("%"):
                numbers.setdefault(token, f"%{len(numbers) + 1}")
                token = numbers[token]
            new_tokens.append(token)
        renumbered.append((new_tokens, comment))
    return renumbered


def _name_ids(instructions):
    names = _Names()
    types = {}
    for tokens, _comment in instructions:
        if not tokens:
            continue
        if len(tokens) > 2 and tokens[1] == "=":
            result, opcode, operands = tokens[0], tokens[2], tokens[3:]
        else:
            result, opcode, operands = None, tokens[0], tokens[1:]
        if opcode == "OpName" and len(operands) >= 2:
            names.save(operands[0], re.sub(r"\\(.)", r"\1", operands[1][1:-1]))
        elif opcode == "OpDecorate" and len(operands) >= 3 and operands[1] == "BuiltIn":
            builtin = _BUILTIN_NAMES.get(operands[2])
            if builtin is not None:
                names.save(operands[0], builtin)
        elif result is None:
            continue
        elif opcode in ("OpTypeVoid", "OpTypeBool"):
            names.save(result, opcode[6:].lower())
        elif opcode == "OpTypeInt":
            width, signed = int(operands[0]), operands[1] != "0"
            types[result] = ("int", width)
            root = _INT_ROOTS.get(width)
            prefix = "" if signed else "u"
            if root is None:
                root, prefix = str(width), "i" if signed else "u"
            names.save(result, prefix + root)
        elif opcode == "OpTypeFloat":
            width = int(operands[0])
            types[result] = ("float", width)
            names.save(result, _FLOAT_ROOTS.get(width, f"fp{width}"))
        elif opcode == "OpTypeVector":
            names.save(result, f"v{operands[1]}{names.get(operands[0])}")
        elif opcode == "OpTypeMatrix":
            names.save(result, f"mat{operands[1]}{names.get(operands[0])}")
        elif opcode == "OpTypeArray":
            names.save(
                result, f"_arr_{names.get(operands[0])}_{names.get(operands[1])}"
            )
        elif opcode == "OpTypeRuntimeArray":
            names.save(result, f"_runtimearr_{names.get(operands[0])}")
        elif opcode == "OpTypePointer":
            names.save(result, f"_ptr_{operands[0]}_{names.get(operands[1])}")
        elif opcode == "OpTypeStruct":
            names.save(result, f"_struct_{result[1:]}")
        elif opcode == "OpConstantTrue":
            names.save(result, "true")
        elif opcode == "OpConstantFalse":
            names.save(result, "false")
        elif opcode == "OpConstant" and len(operands) >= 2 and operands[0] in types:
            text = _literal(operands[1], types[operands[0]])
            names.save(result, f"{names.get(operands[0])}_{text.replace('-', 'n')}")
        else:
            names.save(result, result[1:])
    return names, types


def _literal(text, type_info):
    kind, width = type_info
    if kind == "float":
        return _float_text(float(text), width)
    return str(int(text, 0))


def _format(tokens, comment, names, types):
    if not tokens:
        return comment
    named = [names.get(token) if token.startswith("%") else token for token in tokens]
    named = [
        f"%{token}" if original.startswith("%") else token
        for token, original in zip(named, tokens)
    ]
    if len(tokens) > 2 and tokens[1] == "=":
        if (
            tokens[2] in ("OpConstant", "OpSpecConstant")
            and len(tokens) > 4
            and tokens[3] in types
        ):
            named[4] = _literal(tokens[4], types[tokens[3]])
        prefix = f"{named[0]} = "
        return " " * max(0, _INDENT - len(prefix)) + prefix + " ".join(named[2:])
    return " " * _INDENT + " ".join(named)


def friendly_spirv(assembly: str) -> str:
    """``assembly`` with its ids renumbered and named as ``spirv-as`` and ``spirv-dis`` would."""
    lines = [_split(line) for line in assembly.splitlines()]
    header = [comment for tokens, comment in lines if not tokens and comment]
    body = _renumber([(tokens, comment) for tokens, comment in lines if tokens])
    names, types = _name_ids(body)
    formatted = [_format(tokens, comment, names, types) for tokens, comment in body]
    return "\n".join(header + formatted) + "\n"


def translate_spirv(path, **kwargs) -> str:
    """Translate ``path`` to SPIR-V assembly with friendly ids, whether or not SPIRV-Tools is installed."""
    kwargs.setdefault("backend", "vulkan")
    return friendly_spirv(
        crosstl.translate(str(Path(path)), format_output=False, **kwargs)
    )
