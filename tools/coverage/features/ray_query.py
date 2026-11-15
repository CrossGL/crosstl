"""spirv_std's ray-tracing types: ``RayQuery``, through whose methods a rust-gpu shader traces an inline ray query
and reads its intersections, and ``AccelerationStructure``, which traces a ray from a ray-tracing shader and is made
from a 64-bit address.

``ray_tracing.rs`` writes each method as one ``asm!`` block that executes one SPIR-V instruction, so this module
reads the instruction from the block, and the intersection operand from the ``OpConstant`` it defines: 0 for the
candidate intersection and 1 for the committed one (SPV_KHR_ray_query). CrossGL spells ray queries as DXR does, so
:data:`QUERIES` gives the DXR method each instruction computes, and the CrossGL probe calls it on a ``RayQuery`` that
:data:`..objects.OBJECTS` declares. A rust-gpu probe makes its query with spirv_std's ``ray_query!`` macro
(:attr:`..kernels.Rust.own_objects`).
"""

from __future__ import annotations

import re

from ..model import VOID, Feature, Param, Signature, Type
from .common import prototype
from .rust import enclosed, header, separated
from .spirv_std import clean

# The DXR method of each instruction, and the HLSL type it returns as a GLSL type: a DXR matrix of 4 rows and 3
# columns is GLSL's mat3x4. An instruction that reads an intersection maps to a (candidate, committed) pair. DXR asks
# whether a candidate procedural primitive is non-opaque, where SPIR-V asks whether it is opaque.
QUERIES = {
    "OpRayQueryProceedKHR": ("Proceed", "bool"),
    "OpRayQueryTerminateKHR": ("Abort", "void"),
    "OpRayQueryConfirmIntersectionKHR": ("CommitNonOpaqueTriangleHit", "void"),
    "OpRayQueryGetRayTMinKHR": ("RayTMin", "float"),
    "OpRayQueryGetRayFlagsKHR": ("RayFlags", "uint"),
    "OpRayQueryGetWorldRayDirectionKHR": ("WorldRayDirection", "vec3"),
    "OpRayQueryGetWorldRayOriginKHR": ("WorldRayOrigin", "vec3"),
    "OpRayQueryGetIntersectionCandidateAABBOpaqueKHR": (
        "!{}.CandidateProceduralPrimitiveNonOpaque()",
        "bool",
    ),
    "OpRayQueryGetIntersectionTypeKHR": (("CandidateType", "CommittedStatus"), "uint"),
    "OpRayQueryGetIntersectionTKHR": (
        ("CandidateTriangleRayT", "CommittedRayT"),
        "float",
    ),
    "OpRayQueryGetIntersectionInstanceCustomIndexKHR": (
        ("CandidateInstanceID", "CommittedInstanceID"),
        "uint",
    ),
    "OpRayQueryGetIntersectionInstanceIdKHR": (
        ("CandidateInstanceIndex", "CommittedInstanceIndex"),
        "uint",
    ),
    "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR": (
        (
            "CandidateInstanceContributionToHitGroupIndex",
            "CommittedInstanceContributionToHitGroupIndex",
        ),
        "uint",
    ),
    "OpRayQueryGetIntersectionGeometryIndexKHR": (
        ("CandidateGeometryIndex", "CommittedGeometryIndex"),
        "uint",
    ),
    "OpRayQueryGetIntersectionPrimitiveIndexKHR": (
        ("CandidatePrimitiveIndex", "CommittedPrimitiveIndex"),
        "uint",
    ),
    "OpRayQueryGetIntersectionBarycentricsKHR": (
        ("CandidateTriangleBarycentrics", "CommittedTriangleBarycentrics"),
        "vec2",
    ),
    "OpRayQueryGetIntersectionFrontFaceKHR": (
        ("CandidateTriangleFrontFace", "CommittedTriangleFrontFace"),
        "bool",
    ),
    "OpRayQueryGetIntersectionObjectRayDirectionKHR": (
        ("CandidateObjectRayDirection", "CommittedObjectRayDirection"),
        "vec3",
    ),
    "OpRayQueryGetIntersectionObjectRayOriginKHR": (
        ("CandidateObjectRayOrigin", "CommittedObjectRayOrigin"),
        "vec3",
    ),
    "OpRayQueryGetIntersectionObjectToWorldKHR": (
        ("CandidateObjectToWorld4x3", "CommittedObjectToWorld4x3"),
        "mat3x4",
    ),
}
# Instructions whose DXR form takes other operands than spirv_std's: DXR initializes a query from a ray description
# where spirv_std takes the ray's origin, range and direction, and CrossGL writes a triangle's vertex positions to an
# array argument, as GLSL's rayQueryGetIntersectionTriangleVertexPositionsEXT does.
_COUNTERPARTS = {
    "OpRayQueryInitializeKHR": (
        "{}.TraceRayInline({}, {}, {}, {})",
        "void TraceRayInline(RayQuery, acceleration_struct, uint, uint, ray_desc)",
    ),
    "OpTraceRayKHR": (
        "TraceRay",
        "void TraceRay(acceleration_struct, uint, uint, uint, uint, uint, ray_desc, inout udt)",
    ),
}
_POSITIONS = "OpRayQueryGetIntersectionTriangleVertexPositionsKHR"
_POSITION_METHODS = (
    "CandidateTriangleVertexPositions",
    "CommittedTriangleVertexPositions",
)
_UNSPELLED = {
    "OpConvertUToAccelerationStructureKHR": (
        "CrossGL declares an acceleration structure as a resource, and makes none from an address"
    ),
}
# The probe type of each type a method takes or returns. The result types no buffer element holds are read as
# :attr:`..kernels.Rust._CONVERTED` gives; a ray's flags are made from a u32 (:attr:`..kernels.Rust._MADE`).
_TYPES = {
    "f32": Type("float"),
    "u32": Type("uint"),
    "i32": Type("int"),
    "u64": Type("uint64"),
    "bool": Type("bool"),
    "Vec2": Type("float", (2,)),
    "Vec3": Type("float", (3,)),
    "UVec2": Type("uint", (2,)),
}
_INSTRUCTION = re.compile(
    r"\b(OpRayQuery\w+KHR|OpTraceRayKHR|OpConvertUToAccelerationStructureKHR)\b"
)
# The intersection operand, as spirv_std.clean leaves ``%intersection = OpConstant %u32 1``.
_INTERSECTION = re.compile(r"\bintersection\s+OpConstant\s+u32\s+(\d)\b")
_METHOD = re.compile(r"^    pub (unsafe )?fn (\w+)", re.M)
_OWNERS = ("AccelerationStructure", "RayQuery")
# The stage of the instructions a compute kernel cannot execute: a closest-hit shader traces rays (SPIR-V
# specification, OpTraceRayKHR), as the HLSL probe of TraceRay does.
_STAGES = {"OpTraceRayKHR": "closest-hit"}


def parse(text: str) -> list[Feature]:
    """The methods of ``RayQuery`` and ``AccelerationStructure`` in ``ray_tracing.rs``."""
    text = clean(text)
    features: list[Feature] = []
    for owner in _OWNERS:
        match = re.search(rf"^impl {owner} \{{", text, re.M)
        if match is None:
            raise ValueError(f"ray_tracing.rs has no impl block for {owner}")
        block, _ = enclosed(text, match.end() - 1)
        for method in _METHOD.finditer(block):
            _, end = enclosed(block, block.index("(", method.end()))
            body, _ = enclosed(block, block.index("{", end))
            features.append(
                _feature(
                    owner,
                    method.group(2),
                    bool(method.group(1)),
                    header(block, method.end()),
                    body,
                )
            )
    return sorted(features, key=lambda f: f.name)


def _type(spelling: str) -> Type:
    spelling = spelling.strip()
    if spelling in _TYPES:
        return _TYPES[spelling]
    return Type.other(spelling.removeprefix("&").strip())


def _signature(owner: str, parsed: tuple[str, str, str]) -> Signature:
    """The probe signature of a method on an ``owner``: the object, then the method's parameters. A ray's payload is
    the ray-tracing shader's own (``&mut T``)."""
    _, parameters, result = parsed
    params: list[Param] = []
    for part in separated(parameters):
        if re.fullmatch(r"&\s*(mut\s+)?self", part):
            params.append(Param(Type.other(owner)))
            continue
        spelling = part.partition(":")[2].strip()
        if re.fullmatch(r"&\s*mut\s+T", spelling):
            params.append(Param(Type.other("udt"), "inout"))
        else:
            params.append(Param(_type(spelling)))
    return Signature(_type(result) if result else VOID, tuple(params))


def _feature(
    owner: str, bare: str, unsafe: bool, parsed: tuple[str, str, str], body: str
) -> Feature:
    name = f"{owner}::{bare}"
    found = _INSTRUCTION.search(body)
    if found is None:
        raise ValueError(f"{name} executes no ray-tracing instruction")
    instruction = found.group(1)
    intersection = _INTERSECTION.search(body)
    signature = _signature(owner, parsed)
    spelling = f"{'unsafe ' if unsafe else ''}.{bare}"
    crossgl, counterpart, note = "", None, ""
    if signature.params and signature.params[0].type.opaque != owner:
        # A method without a receiver, such as from_u64, is called by its path.
        spelling = f"{'unsafe ' if unsafe else ''}spirv_std::ray_tracing::{name}"
    if instruction in _COUNTERPARTS:
        crossgl, text = _COUNTERPARTS[instruction]
        counterpart = prototype(text)[1]
    elif instruction == _POSITIONS:
        crossgl = f"{{}}.{_POSITION_METHODS[int(intersection.group(1))]}({{}})"
        counterpart = Signature(
            VOID,
            (Param(Type.other("RayQuery")), Param(Type("float", (3,), count=3), "out")),
        )
    elif instruction in QUERIES:
        method, returned = QUERIES[instruction]
        if isinstance(method, tuple):
            method = method[int(intersection.group(1))]
        crossgl = method if "{" in method else f"{{}}.{method}()"
        counterpart = prototype(f"{returned} {_bare(method)}(RayQuery)")[1]
    else:
        note = _UNSPELLED[instruction]
    return Feature(
        "rust",
        "function",
        name,
        signature,
        group=owner,
        crossgl=crossgl,
        stage=_STAGES.get(instruction, "compute"),
        note=note,
        spelling=spelling,
        counterpart=counterpart,
    )


def _bare(method: str) -> str:
    """The name of the DXR method a CrossGL spelling calls."""
    return re.search(r"(\w+)\(\)$", method).group(1) if "{" in method else method
