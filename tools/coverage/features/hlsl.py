"""HLSL intrinsics from DXC's intrinsic table (``utils/hct/gen_intrin_main.txt``).

The ``Intrinsics`` and ``DxIntrinsics`` namespaces hold the global functions; every ``<Object>Methods`` namespace
holds the methods of one resource or object type. The ``Vk`` namespaces and ``BufferPointer`` are left out: those are
DXC's SPIR-V extensions (``vk::``), not Direct3D HLSL. Declarations follow the grammar described at the top of the
table; generic layouts (``<>``, ``<c>``, ``<r, c>``) are made concrete as two-component vectors and 2x2 matrices, meta
types such as ``float_like`` or ``numeric`` as their 32-bit member, and an object's element type (``$classT``) as the
one :data:`~tools.coverage.objects.OBJECTS` declares it with.
"""

from __future__ import annotations

import re

from .. import crossgl
from ..model import (
    OTHER_WIDTHS,
    VOID,
    Feature,
    Param,
    Signature,
    Type,
    raytracing_stage,
)
from ..objects import OBJECTS, SAMPLERS, Object
from ..validators import HLSL_SHADER_MODEL as SHADER_MODEL
from .common import pick

_VULKAN = "DXC's SPIR-V extensions, not Direct3D HLSL"
LEFT_OUT = dict.fromkeys(
    (
        "VkIntrinsics",
        "VkSubpassInputMethods",
        "VkSubpassInputMSMethods",
        "BufferPointerMethods",
    ),
    _VULKAN,
)
# Namespaces of global functions, and the prefix HLSL calls their functions by.
_FUNCTIONS = {"Intrinsics": "", "DxIntrinsics": "dx::"}
# Methods that take implicit derivatives, which a compute probe of one thread cannot.
_DERIVATIVES = frozenset(
    "Sample SampleBias SampleCmp SampleCmpBias CalculateLevelOfDetail CalculateLevelOfDetailUnclamped".split()
)
SCALAR = {
    "bool": "bool",
    "int": "int",
    "int32_only": "int",
    "any_int": "int",
    "any_int32": "int",
    "any_int16or32": "int",
    "sint16or32_only": "int",
    "uint": "uint",
    "uint_only": "uint",
    "float": "float",
    "float_like": "float",
    "float32_only": "float",
    "any_float": "float",
    "numeric": "float",
    "numeric32_only": "float",
    "any": "float",
    "double": "double",
    "double_only": "double",
    "float16_t": "half",
    "numeric16_only": "half",
    "int16_t": "int16",
    "uint16_t": "uint16",
    "u64": "uint64",
    "any_int64": "int64",
    "int64_only": "int64",
    "p32i8": "int8x4_packed",
    "p32u8": "uint8x4_packed",
}
# Intrinsics that only exist in one kind of shader stage (Direct3D 12 functional specification).
STAGES = {
    "fragment": (
        "clip ddx ddx_coarse ddx_fine ddy ddy_coarse ddy_fine fwidth EvaluateAttributeAtSample "
        "EvaluateAttributeCentroid EvaluateAttributeSnapped GetAttributeAtVertex GetRenderTargetSampleCount "
        "GetRenderTargetSamplePosition"
    ),
    "raytracing": (
        "CallShader TraceRay ReportHit IgnoreHit AcceptHitAndEndSearch DispatchRaysIndex DispatchRaysDimensions "
        "WorldRayOrigin WorldRayDirection ObjectRayOrigin ObjectRayDirection RayTMin RayTCurrent PrimitiveIndex "
        "InstanceID InstanceIndex GeometryIndex HitKind RayFlags ObjectToWorld ObjectToWorld3x4 ObjectToWorld4x3 "
        "WorldToObject WorldToObject3x4 WorldToObject4x3 ClusterID GetRemainingRecursionLevels "
        "TriangleObjectPositions dx::MaybeReorderThread"
    ),
    "mesh": "SetMeshOutputCounts",
    "amplification": "DispatchMesh",
    "hull": (
        "Process2DQuadTessFactorsAvg Process2DQuadTessFactorsMax Process2DQuadTessFactorsMin "
        "ProcessIsolineTessFactors ProcessQuadTessFactorsAvg ProcessQuadTessFactorsMax ProcessQuadTessFactorsMin "
        "ProcessTriTessFactorsAvg ProcessTriTessFactorsMax ProcessTriTessFactorsMin"
    ),
}
_STAGE_OF = {
    name: raytracing_stage(name) if stage == "raytracing" else stage
    for stage, names in STAGES.items()
    for name in names.split()
}
# Parameters that must be immediate values although the table declares them ``in``: GetAttributeAtVertex
# reads one of the primitive's three vertices, so its index is a literal 0, 1 or 2 (HLSL SM 6.1 barycentrics),
# and Barrier's memory-type and semantic flags must be constants (HLSL SM 6.8 barriers).
_IMMEDIATE = {"GetAttributeAtVertex": (1,), "Barrier": (0, 1)}


def _immediate(name: str, signature: Signature | None) -> Signature | None:
    """``signature`` with the parameters ``_IMMEDIATE`` names for ``name`` marked ``const``."""
    positions = _IMMEDIATE.get(name, ())
    if signature is None or not positions:
        return signature
    params = tuple(
        Param(p.type, "const") if index in positions else p
        for index, p in enumerate(signature.params)
    )
    return Signature(signature.result, params)


_DECLARATION = re.compile(
    r"(?P<result>.+?)\s*\[\[(?P<attributes>[^\]]*)\]\]\s*(?P<name>\w+)\s*\((?P<params>.*)\)\s*(?::\s*\w+)?\s*;"
)
_TYPE = re.compile(
    r"(?:\$match<\s*(?P<x>-?\d+)\s*,\s*(?P<y>-?\d+)\s*>\s*)?(?P<base>\$?\w+)\s*(?:<(?P<layout>[^>]*)>)?"
)
_QUALIFIERS = {"in", "out", "inout", "ref", "col_major", "row_major"}


class _Unsupported(ValueError):
    pass


def _layout(text: str | None) -> tuple[int, ...]:
    if text is None:
        return ()
    parts = [p.strip() for p in text.split(",")] if text.strip() else ["n"]
    sizes = [int(p) if p.isdigit() else 2 for p in parts]
    if len(sizes) == 1:
        return () if sizes[0] == 1 else (sizes[0],)
    rows, columns = sizes
    return (columns, rows)


def _own(spelling: str) -> tuple[dict | None, Type | None]:
    """The type a spelling names by itself, and its ``$match`` or ``$typeN`` reference if it has one."""
    typed = re.fullmatch(r"\$type(\d+)", spelling.strip())
    if typed:
        index = int(typed.group(1))
        return {"x": index, "y": index}, None
    if spelling.strip() in ("$classT", "$funcT"):
        return None, Type.other(spelling.strip())
    match = _TYPE.fullmatch(spelling.strip())
    if not match:
        return None, Type.other(spelling.strip() or "unknown")
    base, layout = match.group("base"), match.group("layout")
    reference = (
        {"x": int(match.group("x")), "y": int(match.group("y"))}
        if match.group("x") is not None
        else None
    )
    if base == "void":
        return reference, Type.other("void")
    if base not in SCALAR:
        return reference, Type.other(base)
    shape = _layout(layout)
    if SCALAR[base] not in ("float", "double", "half") and len(shape) == 2:
        shape = (shape[0],)
    return reference, Type(SCALAR[base], shape)


def _resolve(spellings: list[str]) -> list[Type]:
    """Apply ``$match<X, Y>``: the layout comes from type X and the component type from type Y."""
    parsed = [_own(s) for s in spellings]
    own = [kind for _, kind in parsed]
    final: list[Type | None] = list(own)

    def lookup(index: int, position: int) -> Type | None:
        if index < 0 or index >= len(own) or index == position:
            return own[position]
        return final[index] or own[index]

    for position in [*range(1, len(parsed)), 0]:
        reference, kind = parsed[position]
        if reference is None:
            continue
        layout, component = lookup(reference["x"], position), lookup(
            reference["y"], position
        )
        if kind is None:
            final[position] = layout
            continue
        if layout is None or component is None or layout.opaque or component.opaque:
            final[position] = kind
            continue
        final[position] = Type(component.scalar, layout.shape if layout else ())
    return [kind or Type.other("unknown") for kind in final]


def _signature(result: str, params: str) -> Signature:
    if "..." in params:
        raise _Unsupported("variadic")
    spellings, directions = [result], []
    for text in [p.strip() for p in re.split(r",(?![^<]*>)", params) if p.strip()]:
        words = text.split()
        direction = "in"
        while words and words[0] in _QUALIFIERS:
            word = words.pop(0)
            if word in ("in", "out", "inout"):
                direction = word
            elif word == "ref":
                direction = "inout"
        type_text = " ".join(words[:-1])
        spellings.append(type_text)
        directions.append(direction)
    kinds = _resolve(spellings)
    return Signature(
        kinds[0], tuple(Param(k, d) for k, d in zip(kinds[1:], directions))
    )


def _shader_model(attributes: str) -> tuple[int, ...]:
    match = re.search(r"min_sm=(\d+)\.(\d+)", attributes)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _preference(signature: Signature) -> tuple[bool, bool]:
    """Sort key among overloads: value-returning before ``void``, 32-bit before 64-bit or 16-bit; stable otherwise."""
    kinds = [signature.result, *(p.type for p in signature.params)]
    return signature.result == VOID and any(
        p.direction == "out" for p in signature.params
    ), any(k.scalar in OTHER_WIDTHS for k in kinds)


def declarations(text: str):
    """Yield ``(namespace, name, attributes, result, params)`` for every declaration in the table."""
    namespace = ""
    for line in text.splitlines():
        line = line.split("//")[0].strip()
        opened = re.fullmatch(r"namespace\s+(\w+)\s*\{", line)
        if opened:
            namespace = opened.group(1)
            continue
        if line.startswith("}"):
            namespace = ""
            continue
        match = _DECLARATION.fullmatch(line)
        if match and namespace:
            yield (
                namespace,
                match.group("name"),
                match.group("attributes"),
                match.group("result"),
                match.group("params"),
            )


def _concrete(text: str, kind: Object) -> str:
    """``text`` with an object's generic types made concrete: ``$classT`` is its element type, ``$funcT`` (the type a
    templated load returns) ``uint``, and ``$match<0, -1> void<n>`` the element's component type in that layout.
    """
    element = kind.element or Type("uint")
    spelled = element.scalar + (f"<{element.shape[0]}>" if element.shape else "")
    text = re.sub(r"\$match<\s*0\s*,\s*-1\s*>\s*void\b", element.scalar, text)
    text = re.sub(r"\$match<\s*-1\s*,\s*0\s*>\s*void\b", spelled, text)
    text = re.sub(r"\bany_sampler\b", "sampler", text)
    return text.replace("$classT", spelled).replace("$funcT", "uint")


def _probed(signature: Signature) -> bool:
    """Whether a method probe can pass every argument: values, and the samplers it declares."""
    kinds = [signature.result, *(p.type for p in signature.params)]
    return all(not k.opaque or k == VOID or k.opaque in SAMPLERS for k in kinds)


def _overloads(
    declared: list, concrete=lambda text: text
) -> tuple[list[Signature], str]:
    """The overloads the probes' shader model has, and a note when there are none."""
    needed = min(_shader_model(attributes) for attributes, _, _ in declared)
    signatures = []
    for attributes, result, params in declared:
        if _shader_model(attributes) > SHADER_MODEL:
            continue
        try:
            signatures.append(_signature(concrete(result), concrete(params)))
        except _Unsupported:
            continue
    if needed > SHADER_MODEL:
        return [], f"needs shader model {needed[0]}.{needed[1]}"
    return signatures, "" if signatures else "variadic"


def _method(owner: str, name: str, declared: list) -> Feature:
    kind = OBJECTS.get(owner)
    signatures, note = _overloads(
        declared, (lambda text: _concrete(text, kind)) if kind else str
    )
    if kind is None and not note.startswith("needs"):
        signatures, note = [], f"no probe for {owner} objects yet"
    # The plainest overload: no offset, clamp or tiled-resource status.
    ordered = sorted(signatures, key=lambda s: (_preference(s), len(s.params)))
    chosen = next((s for s in ordered if _probed(s)), ordered[0] if ordered else None)
    return Feature(
        "hlsl",
        "method",
        f"{owner}.{name}",
        None if note else chosen,
        group=owner,
        crossgl=name,
        stage="fragment" if name in _DERIVATIVES else kind.stage if kind else "compute",
        note=note,
        spelling=name,
    )


def parse(
    text: str, glsl_names: set[str] | frozenset[str] = frozenset()
) -> list[Feature]:
    functions: dict[str, list] = {}
    methods: dict[tuple[str, str], list] = {}
    for namespace, name, attributes, result, params in declarations(text):
        if namespace in LEFT_OUT or "hidden" in attributes or name.startswith("__"):
            continue
        if namespace in _FUNCTIONS:
            name = _FUNCTIONS[namespace] + name
            functions.setdefault(name, []).append((attributes, result, params))
        else:
            owner = namespace.removesuffix("Methods")
            methods.setdefault((owner, name), []).append((attributes, result, params))
    features = []
    for name, declared in sorted(functions.items()):
        signatures, note = _overloads(declared)
        features.append(
            Feature(
                "hlsl",
                "function",
                name,
                (
                    None
                    if note
                    else _immediate(name, pick(sorted(signatures, key=_preference)))
                ),
                group="intrinsics",
                crossgl=crossgl.spelling("hlsl", name, glsl_names),
                stage=_STAGE_OF.get(name, "compute"),
                note=note,
            )
        )
    features += [
        _method(owner, name, declared)
        for (owner, name), declared in sorted(methods.items())
    ]
    return features
