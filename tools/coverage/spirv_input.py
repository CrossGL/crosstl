"""SPIR-V input probes: the SPIR-V glslang compiles a feature's GLSL probe to, disassembled.

SPIR-V has no source language of its own to write a probe in, and glslang is the reference GLSL compiler, so its
output is the SPIR-V a GLSL project would hand to CrossTL. A function probe counts only when the module calls the
feature's own GLSL.std.450 instruction, and the round trip is correct only when its module still calls it. The
NaN-aware instructions have no GLSL built-in; their probe is the one of the instruction with the same operands,
renamed in the assembly.

GLSL.std.450 encodes ``frexp`` and ``modf`` twice, with an out pointer (``Frexp``, ``Modf``) or a struct result
(``FrexpStruct``, ``ModfStruct``). glslang, like DXC, writes the struct forms, so they are probed with the GLSL call,
and a round trip that writes the other form still makes the call. The feature list gives the struct forms no
signature, because the output row cannot tell them apart: CrossGL's ``frexp`` leaves the form to the generator.
"""

from __future__ import annotations

import dataclasses
import re

from . import constructs, probes, validators
from .dialect import ProbeError
from .features.spirv import UNSPELLED
from .model import Feature, Param, Signature, Type

_STAND_INS = {"NMin": "FMin", "NMax": "FMax", "NClamp": "FClamp"}
_VEC3, _IVEC3 = Type("float", (3,)), Type("int", (3,))
# Each struct form, with its GLSL call and the GLSL signature the pointer form's probe uses.
_STRUCT_FORMS = {
    "FrexpStruct": ("frexp", Signature(_VEC3, (Param(_VEC3), Param(_IVEC3, "out")))),
    "ModfStruct": ("modf", Signature(_VEC3, (Param(_VEC3), Param(_VEC3, "out")))),
}
# The instructions a round trip may write for a feature's own.
_ENCODINGS = {
    "Frexp": {"Frexp", "FrexpStruct"},
    "FrexpStruct": {"Frexp", "FrexpStruct"},
    "Modf": {"Modf", "ModfStruct"},
    "ModfStruct": {"Modf", "ModfStruct"},
}
_EXTENDED = re.compile(r"(\bOpExtInst\s+%\w+\s+%\w+\s+)(\w+)\b")


def build(feature: Feature) -> probes.Probe:
    """The SPIR-V input probe of ``feature``; raises :class:`ProbeError` when there is none."""
    if not validators.can_lower("glsl"):
        raise ProbeError("glslangValidator or spirv-dis is not installed")
    instruction = "" if constructs.is_construct(feature) else feature.name
    glsl = dataclasses.replace(feature, language="glsl")
    if instruction in _STRUCT_FORMS:
        spelling, signature = _STRUCT_FORMS[instruction]
        glsl = dataclasses.replace(glsl, signature=signature, spelling=spelling)
    elif instruction:
        if feature.signature is None:
            raise ProbeError(feature.note or "the feature list gives no signature")
        spelling = (
            UNSPELLED[instruction][0] if instruction in UNSPELLED else feature.crossgl
        )
        glsl = dataclasses.replace(glsl, spelling=spelling)
    probe = probes.build(glsl, "glsl")
    assembly, log = validators.lower(
        "glsl", probe.source, probe.entry, feature.stage, raw_ids=False
    )
    if not assembly:
        raise ProbeError(
            f"glslang rejects the GLSL probe: {validators.first_error(log)}"
        )
    if instruction:
        stand_in = _STAND_INS.get(instruction, instruction)
        emitted = sorted({match[2] for match in _EXTENDED.finditer(assembly)})
        if stand_in not in emitted:
            raise ProbeError(
                f"glslang lowers {glsl.spelling} to {', '.join(emitted) or 'no extended instruction'}, "
                f"not {stand_in}"
            )
        if stand_in != instruction:
            assembly = _EXTENDED.sub(
                lambda m: m[1] + (instruction if m[2] == stand_in else m[2]), assembly
            )
    return probes.Probe("spirv", assembly, probe.entry, ".spvasm")


def dropped(feature: Feature, assembly: str) -> str:
    """Why the round-tripped ``assembly`` no longer calls ``feature``'s instruction; empty when it does.

    The reason names nothing but the feature, so the instructions one importer rewrites the same way are one bug.
    """
    if constructs.is_construct(feature):
        return ""
    written = _ENCODINGS.get(feature.name, {feature.name})
    if any(match[2] in written for match in _EXTENDED.finditer(assembly)):
        return ""
    return f"the round trip no longer calls {feature.name}"
