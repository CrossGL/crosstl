"""SPIR-V extended instructions from the GLSL.std.450 grammar (KhronosGroup/SPIRV-Headers).

An instruction is reached from CrossGL through the GLSL built-in function of the same operation: ``FAbs`` and
``SAbs`` are ``abs`` on float and int operands. The probe signature is the GLSL specification's overload with the
instruction's operand count and operand type.
"""

from __future__ import annotations

import json
import re

from ..model import Feature, Param, Signature, Type
from . import glsl
from .common import pick

# Instructions whose GLSL name does not follow from the opname by dropping the type prefix.
RENAMED = {
    "Atan2": "atan",
    "InverseSqrt": "inversesqrt",
    "MatrixInverse": "inverse",
    "SmoothStep": "smoothstep",
    "FaceForward": "faceforward",
    "FindILsb": "findLSB",
    "FindSMsb": "findMSB",
    "FindUMsb": "findMSB",
}
# Instructions that are not operations of their own, with the reason.
UNREACHABLE = {
    "IMix": "reserved by the GLSL.std.450 specification, not a valid instruction",
    "ModfStruct": "alternative encoding of Modf, chosen by the compiler",
    "FrexpStruct": "alternative encoding of Frexp, chosen by the compiler",
}
# NaN-aware instructions, which no GLSL built-in lowers to: the GLSL built-in whose signature the probe uses, and the
# CrossGL spelling. CrossGL's fmin and fmax return the operand that is not NaN, as C's and MSL's do, and GLSL.std.450
# defines NClamp as NMin(NMax(x, minVal), maxVal).
NAN_AWARE = {
    "NMin": ("min", "fmin"),
    "NMax": ("max", "fmax"),
    "NClamp": ("clamp", "fmin(fmax({}, {}), {})"),
}
_TYPED = re.compile(r"(F|S|U)([A-Z]\w*)")
_PREFIX_SCALAR = {"F": "float", "S": "int", "U": "uint"}


def _retyped(signature: Signature, scalar: str) -> Signature:
    def convert(kind: Type) -> Type:
        return (
            kind.with_scalar(scalar)
            if not kind.opaque and kind.scalar in ("float", "int", "uint")
            else kind
        )

    return Signature(
        convert(signature.result),
        tuple(Param(convert(p.type), p.direction) for p in signature.params),
    )


def _match(opname: str, operands: int, functions: dict) -> tuple[str, str]:
    """The GLSL function and operand scalar type an instruction corresponds to."""
    if opname in NAN_AWARE:
        return NAN_AWARE[opname][0], "float"
    if opname in RENAMED:
        prefix = (
            "U"
            if opname.startswith("FindU")
            else ("S" if opname.startswith("Find") else "")
        )
        return RENAMED[opname], _PREFIX_SCALAR.get(prefix, "")
    typed = _TYPED.fullmatch(opname)
    if typed:
        name = typed.group(2)[0].lower() + typed.group(2)[1:]
        if name in functions:
            return name, _PREFIX_SCALAR[typed.group(1)]
    return opname[0].lower() + opname[1:], ""


def parse(grammar_text: str, glsl_text: str) -> list[Feature]:
    grammar = json.loads(grammar_text)
    functions = glsl.overloads(glsl_text)
    features = []
    for instruction in grammar["instructions"]:
        opname = instruction["opname"]
        group = "glsl-std-450"
        if opname in UNREACHABLE:
            features.append(
                Feature(
                    "spirv",
                    "function",
                    opname,
                    None,
                    group=group,
                    note=UNREACHABLE[opname],
                )
            )
            continue
        operands = sum(1 for o in instruction["operands"] if o["kind"] == "IdRef")
        name, scalar = _match(opname, operands, functions)
        if name not in functions:
            features.append(
                Feature(
                    "spirv",
                    "function",
                    opname,
                    None,
                    group=group,
                    note=f"no GLSL built-in named {name}",
                )
            )
            continue
        section, signatures = functions[name]
        candidates = [s for s in signatures if len(s.params) == operands]
        if scalar:
            # Prefer an overload already on the instruction's operand type; otherwise retype the float one.
            exact = [
                s for s in candidates if s.params and s.params[0].type.scalar == scalar
            ]
            candidates = exact or [
                _retyped(s, scalar)
                for s in candidates
                if s.params and s.params[0].type.scalar == "float"
            ]
        if not candidates:
            features.append(
                Feature(
                    "spirv",
                    "function",
                    opname,
                    None,
                    group=group,
                    note=f"no {name} overload with {operands} operands",
                )
            )
            continue
        features.append(
            Feature(
                "spirv",
                "function",
                opname,
                pick(candidates),
                group=group,
                crossgl=NAN_AWARE[opname][1] if opname in NAN_AWARE else name,
                stage=glsl.STAGES.get(section, "compute"),
            )
        )
    return sorted(features, key=lambda f: f.name)
