"""GLSL built-in function names that an emitted variable, function or type must not spell.

Built-in functions live in a scope outside the shader's global scope. A program variable named ``texture`` hides
the built-in ``texture`` for the rest of its scope, so the ``texture(texture, uv)`` that GLSL output writes for a
sampled read fails with "can't use function syntax on variable" (GLSL 4.60 section 4.2, Scoping). A user function
named like a built-in overloads it, and an overload whose parameters differ from a built-in's only in their
qualifiers, such as ``float frexp(float, int)`` beside ``frexp(float, out int)``, is an error (section 6.1,
Function Definitions). GLSL ES 3.00 goes further: a global declaration or a user function with a built-in
function's name is a redefinition error (GLSL ES 3.00 section 4.2, Scoping, and section 6.1), whether or not the
program calls the built-in. HLSL, MSL and CrossGL treat these names as ordinary identifiers.

So WebGL output renames every program identifier that spells a built-in function, as it renames GLSL keywords.
Desktop GLSL output renames one only when the program it writes calls that name: a local ``step`` that no
``step(...)`` call meets keeps its name, and so does a uniform the host looks up by name, while a user function is
always renamed because its own definition is such a call. :meth:`GLSLCodeGen._generate_program_with_vulkan_forms`
renders the program again with the called names reserved. Structure members are renamed too, because a stage
interface structure's members become global inputs and outputs.

:data:`GLSL_CORE_BUILTIN_FUNCTIONS` is chapter 8 of the GLSL 4.60 specification, the source that
``tools.coverage.features.glsl`` reads. :data:`GLSL_EXTENSION_BUILTIN_FUNCTIONS` lists the extension functions and
statement keywords that GLSL output can write.
"""

from __future__ import annotations

import re

__all__ = [
    "GLSL_BUILTIN_FUNCTION_NAMES",
    "GLSL_CORE_BUILTIN_FUNCTIONS",
    "GLSL_EXTENSION_BUILTIN_FUNCTIONS",
    "called_builtin_spellings",
    "glsl_builtin_name_reserved",
    "reset_builtin_spellings",
]

GLSL_CORE_BUILTIN_FUNCTIONS = frozenset("""
    EmitStreamVertex EmitVertex EndPrimitive EndStreamPrimitive abs acos acosh all allInvocations
    allInvocationsEqual any anyInvocation asin asinh atan atanh atomicAdd atomicAnd atomicCompSwap atomicCounter
    atomicCounterAdd atomicCounterAnd atomicCounterCompSwap atomicCounterDecrement atomicCounterExchange
    atomicCounterIncrement atomicCounterMax atomicCounterMin atomicCounterOr atomicCounterSubtract atomicCounterXor
    atomicExchange atomicMax atomicMin atomicOr atomicXor barrier bitCount bitfieldExtract bitfieldInsert
    bitfieldReverse ceil clamp cos cosh cross dFdx dFdxCoarse dFdxFine dFdy dFdyCoarse dFdyFine degrees determinant
    distance dot equal exp exp2 faceforward findLSB findMSB floatBitsToInt floatBitsToUint floor fma fract frexp
    fwidth fwidthCoarse fwidthFine greaterThan greaterThanEqual groupMemoryBarrier imageAtomicAdd imageAtomicAnd
    imageAtomicCompSwap imageAtomicExchange imageAtomicMax imageAtomicMin imageAtomicOr imageAtomicXor imageLoad
    imageSamples imageSize imageStore imulExtended intBitsToFloat interpolateAtCentroid interpolateAtOffset
    interpolateAtSample inverse inversesqrt isinf isnan ldexp length lessThan lessThanEqual log log2 matrixCompMult
    max memoryBarrier memoryBarrierAtomicCounter memoryBarrierBuffer memoryBarrierImage memoryBarrierShared min mix
    mod modf normalize not notEqual outerProduct packDouble2x32 packHalf2x16 packSnorm2x16 packSnorm4x8
    packUnorm2x16 packUnorm4x8 pow radians reflect refract round roundEven sign sin sinh smoothstep sqrt step
    subpassLoad tan tanh texelFetch texelFetchOffset texture textureGather textureGatherOffset textureGatherOffsets
    textureGrad textureGradOffset textureLod textureLodOffset textureOffset textureProj textureProjGrad
    textureProjGradOffset textureProjLod textureProjLodOffset textureProjOffset textureQueryLevels textureQueryLod
    textureSamples textureSize transpose trunc uaddCarry uintBitsToFloat umulExtended unpackDouble2x32
    unpackHalf2x16 unpackSnorm2x16 unpackSnorm4x8 unpackUnorm2x16 unpackUnorm4x8 usubBorrow
    noise1 noise2 noise3 noise4
    """.split())

_SUBGROUP_OPERATIONS = (
    "Add",
    "Mul",
    "Min",
    "Max",
    "And",
    "Or",
    "Xor",
)

GLSL_EXTENSION_BUILTIN_FUNCTIONS = frozenset(
    """
    subgroupAll subgroupAllEqual subgroupAny subgroupBallot subgroupBallotBitCount subgroupBallotBitExtract
    subgroupBallotExclusiveBitCount subgroupBallotFindLSB subgroupBallotFindMSB subgroupBallotInclusiveBitCount
    subgroupBarrier subgroupBroadcast subgroupBroadcastFirst subgroupClusteredRotate subgroupElect
    subgroupInverseBallot subgroupMemoryBarrier subgroupMemoryBarrierBuffer subgroupMemoryBarrierImage
    subgroupMemoryBarrierShared subgroupPartitionNV subgroupQuadAll subgroupQuadAny subgroupQuadBroadcast
    subgroupQuadSwapDiagonal subgroupQuadSwapHorizontal subgroupQuadSwapVertical subgroupRotate subgroupShuffle
    subgroupShuffleDown subgroupShuffleUp subgroupShuffleXor
    ballotARB readInvocationARB readFirstInvocationARB anyInvocationARB allInvocationsARB allInvocationsEqualARB
    atomicLoad atomicStore controlBarrier
    beginInvocationInterlockARB endInvocationInterlockARB
    clockARB clock2x32ARB clockRealtimeEXT clockRealtime2x32EXT
    colorAttachmentReadEXT depthAttachmentReadEXT stencilAttachmentReadEXT
    coopMatLoad coopMatStore coopMatMulAdd
    debugPrintfEXT demote helperInvocationEXT terminateInvocation
    doubleBitsToInt64 doubleBitsToUint64 int64BitsToDouble uint64BitsToDouble
    float16BitsToInt16 float16BitsToUint16 int16BitsToFloat16 uint16BitsToFloat16
    packFloat2x16 unpackFloat2x16 packInt2x32 unpackInt2x32 packUint2x32 unpackUint2x32
    packInt2x16 unpackInt2x16 packUint2x16 unpackUint2x16 packInt4x16 unpackInt4x16 packUint4x16 unpackUint4x16
    nonuniformEXT
    SetMeshOutputsEXT EmitMeshTasksEXT
    traceRayEXT reportIntersectionEXT executeCallableEXT ignoreIntersectionEXT terminateRayEXT
    rayQueryInitializeEXT rayQueryProceedEXT rayQueryTerminateEXT rayQueryGenerateIntersectionEXT
    rayQueryConfirmIntersectionEXT rayQueryGetIntersectionTypeEXT rayQueryGetRayTMinEXT rayQueryGetRayFlagsEXT
    rayQueryGetWorldRayOriginEXT rayQueryGetWorldRayDirectionEXT rayQueryGetIntersectionTEXT
    rayQueryGetIntersectionInstanceCustomIndexEXT rayQueryGetIntersectionInstanceIdEXT
    rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT rayQueryGetIntersectionGeometryIndexEXT
    rayQueryGetIntersectionPrimitiveIndexEXT rayQueryGetIntersectionBarycentricsEXT
    rayQueryGetIntersectionFrontFaceEXT rayQueryGetIntersectionCandidateAABBOpaqueEXT
    rayQueryGetIntersectionObjectRayDirectionEXT rayQueryGetIntersectionObjectRayOriginEXT
    rayQueryGetIntersectionObjectToWorldEXT rayQueryGetIntersectionWorldToObjectEXT
    rayQueryGetIntersectionTriangleVertexPositionsEXT
    sparseTexelsResidentARB sparseTextureARB sparseTextureLodARB sparseTextureOffsetARB sparseTexelFetchARB
    sparseTexelFetchOffsetARB sparseTextureLodOffsetARB sparseTextureGradARB sparseTextureGradOffsetARB
    sparseTextureGatherARB sparseTextureGatherOffsetARB sparseTextureGatherOffsetsARB sparseImageLoadARB
    sparseTextureClampARB sparseTextureOffsetClampARB sparseTextureGradClampARB sparseTextureGradOffsetClampARB
    textureClampARB textureOffsetClampARB textureGradClampARB textureGradOffsetClampARB
    """.split()
    + [
        f"subgroup{scan}{operation}"
        for scan in ("", "Inclusive", "Exclusive", "Clustered")
        for operation in _SUBGROUP_OPERATIONS
    ]
    + [
        f"subgroupPartitioned{scan}{operation}NV"
        for scan in ("", "Inclusive", "Exclusive")
        for operation in _SUBGROUP_OPERATIONS
    ]
)

GLSL_BUILTIN_FUNCTION_NAMES = (
    GLSL_CORE_BUILTIN_FUNCTIONS | GLSL_EXTENSION_BUILTIN_FUNCTIONS
)

# A name followed by an argument list, and not a member call such as ``values.length()``.
_CALLED_NAME = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")


def reset_builtin_spellings(codegen):
    """Forget the built-in spellings of an earlier program before ``codegen`` renders a new one."""
    codegen.glsl_builtin_spelled_identifiers = set()
    codegen.glsl_shadowed_builtin_names = frozenset()


def glsl_builtin_name_reserved(codegen, name):
    """Whether ``codegen`` renames a program identifier ``name`` because it spells a GLSL built-in function.

    Output that reserves every built-in name (``GLSL_RESERVES_BUILTIN_FUNCTION_NAMES``, GLSL ES) renames it at once.
    Otherwise the name is recorded, and it is renamed once a rendering of the program has called it.
    """
    if name not in GLSL_BUILTIN_FUNCTION_NAMES:
        return False
    if getattr(codegen, "GLSL_RESERVES_BUILTIN_FUNCTION_NAMES", True):
        return True
    spelled = getattr(codegen, "glsl_builtin_spelled_identifiers", None)
    if spelled is None:
        # Rendered outside a program pass, where no later rendering can rename it: keep the strict rule.
        return True
    spelled.add(name)
    return name in codegen.glsl_shadowed_builtin_names


def called_builtin_spellings(codegen, code):
    """The built-in names that program identifiers spelled and that ``code`` calls, not reserved yet."""
    spelled = getattr(codegen, "glsl_builtin_spelled_identifiers", None) or set()
    pending = spelled - codegen.glsl_shadowed_builtin_names
    if not pending:
        return frozenset()
    return frozenset(
        pending & {match.group(1) for match in _CALLED_NAME.finditer(code)}
    )
