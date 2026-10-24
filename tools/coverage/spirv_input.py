"""SPIR-V input probes: the SPIR-V glslang compiles a feature's GLSL probe to, disassembled.

SPIR-V has no source language of its own to write a probe in, and glslang is the reference GLSL compiler, so its
output is the SPIR-V a GLSL project would hand to CrossTL. A function probe counts only when the module calls the
feature's own GLSL.std.450 instruction, and the round trip is correct only when its module still calls it. The
NaN-aware instructions have no GLSL built-in; their probe is the one of the instruction with the same operands,
renamed in the assembly.

GLSL.std.450 encodes ``frexp`` and ``modf`` twice, with an out pointer (``Frexp``, ``Modf``) or a struct result
(``FrexpStruct``, ``ModfStruct``). glslang, like DXC, writes the struct forms, so they are probed with the GLSL call,
and a round trip that writes the other form still makes the call. The feature list gives the struct forms no
signature, because the output row cannot tell them apart: CrossGL's ``frexp`` leaves the form to the generator. The
pointer forms' probe is the struct form's module with the call rewritten to write the out variable directly, as
other SPIR-V producers do.
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
# Each pointer form, with the struct form glslang writes for the GLSL call.
_POINTER_FORMS = {"Frexp": "FrexpStruct", "Modf": "ModfStruct"}


def _uses(assembly: str, ident: str) -> int:
    return len(re.findall(rf"{re.escape(ident)}\b", assembly))


def _pointer_form(assembly: str, instruction: str) -> str:
    """``assembly`` with its struct-form call rewritten to ``instruction``, the pointer form.

    glslang writes ``%r = OpExtInst %Res %ext FrexpStruct %x``, stores member 1 of ``%r`` into the out variable and
    reads member 0. The rewrite passes the out variable as the pointer operand, gives the call member 0's id, and
    drops the two extracts and the store. Raises :class:`ProbeError` for a module of any other shape.
    """
    struct_form = _POINTER_FORMS[instruction]
    call = re.search(
        rf"^( *)(%\w+) = OpExtInst %\w+ (%\w+) {struct_form} (%\w+)$",
        assembly,
        re.MULTILINE,
    )
    shape = ProbeError(f"glslang's {struct_form} call has an unexpected shape")
    if call is None or assembly.count(f" {struct_form} ") != 1:
        raise shape
    indent, result, ext, operand = call.groups()
    member = {
        match[3]: match
        for match in re.finditer(
            rf"^ *(%\w+) = OpCompositeExtract (%\w+) {result} (\d+)$",
            assembly,
            re.MULTILINE,
        )
    }
    if set(member) != {"0", "1"} or _uses(assembly, result) != 3:
        raise shape
    stored = member["1"][1]
    store = re.search(
        rf"^ *OpStore (%\w+) {re.escape(stored)}$", assembly, re.MULTILINE
    )
    if store is None or _uses(assembly, stored) != 2:
        raise shape
    value, kind = member["0"][1], member["0"][2]
    rewritten = assembly.replace(
        call[0],
        f"{indent}{value} = OpExtInst {kind} {ext} {instruction} {operand} {store[1]}",
    )
    for line in (member["0"][0], member["1"][0], store[0]):
        rewritten = rewritten.replace(line + "\n", "", 1)
    return rewritten


def build(feature: Feature) -> probes.Probe:
    """The SPIR-V input probe of ``feature``; raises :class:`ProbeError` when there is none."""
    if not validators.can_lower("glsl"):
        raise ProbeError("glslangValidator or spirv-dis is not installed")
    instruction = "" if constructs.is_construct(feature) else feature.name
    glsl = dataclasses.replace(feature, language="glsl")
    struct_form = _POINTER_FORMS.get(instruction, instruction)
    if struct_form in _STRUCT_FORMS:
        spelling, signature = _STRUCT_FORMS[struct_form]
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
        stand_in = _STAND_INS.get(instruction, struct_form)
        emitted = sorted({match[2] for match in _EXTENDED.finditer(assembly)})
        if stand_in not in emitted:
            raise ProbeError(
                f"glslang lowers {glsl.spelling} to {', '.join(emitted) or 'no extended instruction'}, "
                f"not {stand_in}"
            )
        if instruction in _POINTER_FORMS:
            assembly = _pointer_form(assembly, instruction)
        elif stand_in != instruction:
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
