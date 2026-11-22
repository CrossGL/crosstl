"""Semantic feature ids: what a backend feature means, independent of how the backend spells it.

A feature's id is the set of SPIR-V instructions its call lowers to when the language's reference compiler builds its
native probe: HLSL ``mad``, GLSL ``fma`` and the SPIR-V ``Fma`` instruction all become ``GLSL.std.450:Fma``. The
instructions the probe needs around the call are removed by subtracting a baseline probe that reads the same
arguments but makes no call. A feature SPIR-V cannot express, or one no compiler here lowers, has no id; it is listed
in :data:`CURATED` when it is known to equal another feature, and otherwise stays a semantic feature of its own.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter

from . import constructs, lowering, probes, validators
from .features.spirv import is_core
from .model import Feature

# Features with no SPIR-V lowering, keyed ``language/category/name``, and the semantic id they share with features
# that have one. C's and MSL's fmin and fmax return the operand that is not NaN (C17 7.12.12, MSL 3.2 section 6.5),
# as NMin and NMax do; the languages that borrow ids would otherwise take FMin's and FMax's through GLSL's min and max.
CURATED: dict[str, str] = {
    "cuda/function/fmin": "GLSL.std.450:NMin",
    "cuda/function/fminf": "GLSL.std.450:NMin",
    "cuda/function/fmax": "GLSL.std.450:NMax",
    "cuda/function/fmaxf": "GLSL.std.450:NMax",
    "hip/function/fmin": "GLSL.std.450:NMin",
    "hip/function/fminf": "GLSL.std.450:NMin",
    "hip/function/fmax": "GLSL.std.450:NMax",
    "hip/function/fmaxf": "GLSL.std.450:NMax",
    "metal/function/fmin": "GLSL.std.450:NMin",
    "metal/function/fmax": "GLSL.std.450:NMax",
    # HLSL's shader execution reordering call with a coherence hint and its bit count (Shader Model 6.9), which DXC
    # does not lower to SPIR-V ("MaybeReorderThread intrinsic function unimplemented"). It is the instruction that
    # SPV_EXT_shader_invocation_reorder defines with the same operands, as Slang's ReorderThread lowers to.
    "hlsl/function/dx::MaybeReorderThread": "OpReorderThreadWithHintEXT",
}
# Languages whose functions, or some of them, have no id of their own. They take the id of the function of the named
# language that CrossGL spells the same way: Metal ``fabs`` and CUDA ``fabsf`` are CrossGL ``abs``, so they share
# GLSL ``abs``'s id. WGSL ``frexp`` returns a result structure that its probe cannot spell, so it has no id either and
# shares GLSL ``frexp``'s. GLSL ES 3.00 does not compile to SPIR-V, so WebGL's functions share GLSL's ids, and so do
# the functions of GLSL ES 3.20, which are GLSL's under the same names.
BORROWED = {
    "cuda": "glsl",
    "gles": "glsl",
    "hip": "glsl",
    "metal": "glsl",
    "mojo": "glsl",
    "rust": "glsl",
    "webgl": "glsl",
    "wgsl": "glsl",
}
# Metal's SIMD-group and quad-group functions, which CrossGL spells as Metal does, so they cannot borrow by their
# CrossGL spelling. Each takes the id of the GLSL function that SPIRV-Cross writes it for when it compiles that
# function's SPIR-V to MSL (spirv_msl.cpp, SPIRV-Cross vulkan-sdk-1.4.363.0). tests/coverage/test_semantic.py checks
# every pair against SPIRV-Cross when it is installed.
CROSS_COMPILED = {
    "metal": {
        "simd_sum": "subgroupAdd",
        "simd_product": "subgroupMul",
        "simd_min": "subgroupMin",
        "simd_max": "subgroupMax",
        "simd_and": "subgroupAnd",
        "simd_or": "subgroupOr",
        "simd_xor": "subgroupXor",
        "simd_prefix_inclusive_sum": "subgroupInclusiveAdd",
        "simd_prefix_exclusive_sum": "subgroupExclusiveAdd",
        "simd_prefix_inclusive_product": "subgroupInclusiveMul",
        "simd_prefix_exclusive_product": "subgroupExclusiveMul",
        "simd_broadcast": "subgroupBroadcast",
        "simd_broadcast_first": "subgroupBroadcastFirst",
        "simd_is_first": "subgroupElect",
        "simd_shuffle": "subgroupShuffle",
        "simd_shuffle_xor": "subgroupShuffleXor",
        "simd_shuffle_up": "subgroupShuffleUp",
        "simd_shuffle_down": "subgroupShuffleDown",
        "simd_shuffle_rotate_down": "subgroupRotate",
        "quad_broadcast": "subgroupQuadBroadcast",
        "quad_all": "subgroupQuadAll",
        "quad_any": "subgroupQuadAny",
    }
}
# Feature groups whose SPIR-V does not identify them: glam writes Vec3's methods in plain Rust arithmetic, so
# ``Vec3::dot`` lowers to multiplies and adds rather than to OpDot.
UNIDENTIFIED = {("rust", "vec3"): "glam computes it with Rust arithmetic"}
# Instructions that only move or address values, or structure control flow. They never identify a call; the
# unreachable block after a call that never returns, such as Rust's ``kill() -> !``, ends in OpUnreachable, and the
# match by which spirv_std turns a ray query's intersection type into an enumeration lowers to OpSwitch.
_MEMORY = frozenset(
    "OpLoad OpStore OpAccessChain OpInBoundsAccessChain OpPtrAccessChain OpVariable OpCopyObject OpCopyMemory "
    "OpLine OpNoLine OpLabel OpBranch OpBranchConditional OpSwitch OpSelectionMerge OpLoopMerge OpPhi OpUndef "
    "OpReturn OpReturnValue OpUnreachable OpFunction OpFunctionEnd OpFunctionParameter OpFunctionCall".split()
)
# Instructions that also reshape or convert the call's result for the output buffer. They identify a call only when
# it lowers to nothing else, as a bit cast or a select does.
_RESHAPING = frozenset(
    "OpCompositeConstruct OpCompositeExtract OpCompositeInsert OpVectorShuffle OpSelect OpBitcast OpFConvert "
    "OpSConvert OpUConvert OpConvertFToS OpConvertFToU OpConvertSToF OpConvertUToF".split()
)
# Instructions from SPV_KHR_subgroup_vote that SPIR-V 1.3 replaced with core instructions of the same meaning.
_ALIASES = {
    "OpSubgroupAllKHR": "OpGroupNonUniformAll",
    "OpSubgroupAnyKHR": "OpGroupNonUniformAny",
    "OpSubgroupAllEqualKHR": "OpGroupNonUniformAllEqual",
}
# Instructions whose constant operands are part of the operation, so their values join the id: the scope and memory
# semantics of a barrier, the direction of a quad swap.
_CONSTANT_OPERANDS = frozenset(
    {"OpControlBarrier", "OpMemoryBarrier", "OpGroupNonUniformQuadSwap"}
)
# The GroupOperation operand tells a reduction from an inclusive or exclusive scan, and one over the invocations of a
# partition (HLSL's WaveMultiPrefix*, GLSL's subgroupPartitioned*NV) from one over the whole subgroup. SPIRV-Tools
# releases before the EXT names print the partitioned operations with the NV suffix.
_GROUP_OPERATIONS = {
    operation: operation
    for operation in (
        "Reduce",
        "InclusiveScan",
        "ExclusiveScan",
        "ClusteredReduce",
        "PartitionedReduceEXT",
        "PartitionedInclusiveScanEXT",
        "PartitionedExclusiveScanEXT",
    )
} | {
    "PartitionedReduceNV": "PartitionedReduceEXT",
    "PartitionedInclusiveScanNV": "PartitionedInclusiveScanEXT",
    "PartitionedExclusiveScanNV": "PartitionedExclusiveScanEXT",
}
_INSTRUCTION = re.compile(r"^\s*(?:%\w+\s*=\s*)?(Op\w+)(.*)$")
_IMPORT = re.compile(r'^\s*(%\w+)\s*=\s*OpExtInstImport\s+"([^"]+)"')
_CONSTANT = re.compile(r"^\s*(%\w+)\s*=\s*OpConstant\s+%\w+\s+(\S+)")
_BUILTIN = re.compile(r"^\s*OpDecorate\s+(%\w+)\s+BuiltIn\s+(\w+)")
# Instructions that read a variable through the pointer that is their second operand.
_READS = frozenset({"OpLoad", "OpAccessChain", "OpInBoundsAccessChain"})


def canonical(opcode: str) -> str:
    """The instruction ``opcode`` is counted as: the core instruction of a :data:`_ALIASES` entry, otherwise itself."""
    return _ALIASES.get(opcode, opcode)


def instructions(assembly: str, aliases: bool = True) -> Counter:
    """The instructions inside function bodies, with extended instructions named ``set:instruction``. A read of a
    built-in variable also counts as ``BuiltIn:<name>``, since a function such as HLSL ``WorldRayOrigin`` lowers to
    that read alone. With ``aliases``, an instruction is counted as its :func:`canonical` one.
    """
    sets: dict[str, str] = {}
    constants: dict[str, str] = {}
    builtins: dict[str, str] = {}
    found: Counter = Counter()
    inside = False
    for line in assembly.splitlines():
        imported = _IMPORT.match(line)
        if imported:
            sets[imported.group(1)] = imported.group(2)
            continue
        constant = _CONSTANT.match(line)
        if constant:
            constants[constant.group(1)] = constant.group(2)
        builtin = _BUILTIN.match(line)
        if builtin:
            builtins[builtin.group(1)] = builtin.group(2)
        match = _INSTRUCTION.match(line)
        if not match:
            continue
        opcode, operands = match.groups()
        if opcode == "OpFunction":
            inside = True
        if not inside:
            continue
        if opcode == "OpFunctionEnd":
            inside = False
        words = operands.split()
        token = _token(canonical(opcode) if aliases else opcode, words, sets, constants)
        if token:
            found[token] += 1
        if opcode in _READS and len(words) > 1 and words[1] in builtins:
            found[f"BuiltIn:{builtins[words[1]]}"] += 1
    return found


def _token(
    opcode: str, words: list[str], sets: dict[str, str], constants: dict[str, str]
) -> str:
    if opcode == "OpExtInst":
        if len(words) < 3:
            return opcode
        library = sets.get(words[1], words[1])
        return "" if library.startswith("NonSemantic.") else f"{library}:{words[2]}"
    if opcode in _CONSTANT_OPERANDS:
        return f"{opcode}({','.join(constants[w] for w in words if w in constants)})"
    operation = next(
        (_GROUP_OPERATIONS[w] for w in words if w in _GROUP_OPERATIONS), ""
    )
    return f"{opcode}.{operation}" if operation else opcode


def identify(probe: Counter, baseline: Counter) -> str:
    """The id of the call: what ``probe`` has beyond ``baseline``, without plumbing; empty when nothing remains."""
    extra = {op for op, count in (probe - baseline).items() if count > 0}
    tokens = extra - _MEMORY - _RESHAPING or extra - _MEMORY
    return "+".join(sorted(tokens))


def lower(feature: Feature) -> tuple[str, str]:
    """``(semantic id, "")`` for ``feature``, or ``("", reason)`` when it has none."""
    if constructs.is_construct(feature):
        return constructs.semantic_id(feature), ""
    if feature.key in CURATED:
        return CURATED[feature.key], ""
    if is_core(feature):
        # A core instruction's id is the one of the GLSL call that compiles to it.
        return lower(
            dataclasses.replace(
                feature, language="glsl", name=feature.spelling, spelling=""
            )
        )
    if feature.language == "spirv":
        return f"GLSL.std.450:{feature.name}", ""
    if (feature.language, feature.group) in UNIDENTIFIED:
        return "", UNIDENTIFIED[feature.language, feature.group]
    # A probe checked for another target than the usual one has no SPIR-V the usual compiler writes, except an HLSL
    # probe that DXC checks as SPIR-V, as for printf.
    spirv = (feature.language, feature.target) == ("hlsl", validators.HLSL_SPIRV)
    if feature.target and not spirv:
        return "", f"probed for {feature.target}, not SPIR-V"
    if feature.language not in lowering.SPIRV_COMPILERS:
        return "", f"no SPIR-V compiler for {feature.language}"
    if not lowering.can_lower(feature.language):
        tool = lowering.SPIRV_COMPILERS[feature.language][0]
        return "", f"{tool} or spirv-dis is not installed"
    try:
        probe = probes.build(feature, feature.language)
        baseline = probes.build(feature, feature.language, baseline=True)
    except probes.ProbeError as error:
        return "", str(error)
    stage = feature.stage
    lowered, log = lowering.lower(feature.language, probe.source, probe.entry, stage)
    if not lowered:
        return (
            "",
            f"does not lower to SPIR-V: {validators.first_error(log, feature.language)}",
        )
    reference, log = lowering.lower(
        feature.language, baseline.source, baseline.entry, stage
    )
    if not reference:
        return (
            "",
            f"the baseline does not lower to SPIR-V: {validators.first_error(log, feature.language)}",
        )
    semantic = identify(instructions(lowered), instructions(reference))
    return (semantic, "") if semantic else ("", "the call lowers to no instruction")
