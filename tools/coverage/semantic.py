"""Semantic feature ids: what a backend feature means, independent of how the backend spells it.

A feature's id is the set of SPIR-V instructions its call lowers to when the language's reference compiler builds its
native probe: HLSL ``mad``, GLSL ``fma`` and the SPIR-V ``Fma`` instruction all become ``GLSL.std.450:Fma``. The
instructions the probe needs around the call are removed by subtracting a baseline probe that reads the same
arguments but makes no call. A feature SPIR-V cannot express, or one no compiler here lowers, has no id; it is listed
in :data:`CURATED` when it is known to equal another feature, and otherwise stays a semantic feature of its own.
"""

from __future__ import annotations

import re
from collections import Counter

from . import constructs, probes, validators
from .model import Feature

# Features with no SPIR-V lowering, keyed ``language/category/name``, and the semantic id they share with features
# that have one.
CURATED: dict[str, str] = {}
# Languages whose functions, or some of them, have no id of their own. They take the id of the function of the named
# language that CrossGL spells the same way: Metal ``fabs`` and CUDA ``fabsf`` are CrossGL ``abs``, so they share
# GLSL ``abs``'s id. WGSL ``frexp`` returns a result structure that its probe cannot spell, so it has no id either and
# shares GLSL ``frexp``'s. GLSL ES 3.00 does not compile to SPIR-V, so WebGL's functions share GLSL's ids.
BORROWED = {
    "cuda": "glsl",
    "hip": "glsl",
    "metal": "glsl",
    "mojo": "glsl",
    "rust": "glsl",
    "webgl": "glsl",
    "wgsl": "glsl",
}
# Feature groups whose SPIR-V does not identify them: glam writes Vec3's methods in plain Rust arithmetic, so
# ``Vec3::dot`` lowers to multiplies and adds rather than to OpDot.
UNIDENTIFIED = {("rust", "vec3"): "glam computes it with Rust arithmetic"}
# Instructions that only move or address values, or structure control flow. They never identify a call; the
# unreachable block after a call that never returns, such as Rust's ``kill() -> !``, ends in OpUnreachable.
_MEMORY = frozenset(
    "OpLoad OpStore OpAccessChain OpInBoundsAccessChain OpPtrAccessChain OpVariable OpCopyObject OpCopyMemory "
    "OpLine OpNoLine OpLabel OpBranch OpBranchConditional OpSelectionMerge OpLoopMerge OpPhi OpUndef OpReturn "
    "OpReturnValue OpUnreachable OpFunction OpFunctionEnd OpFunctionParameter OpFunctionCall".split()
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
# The GroupOperation operand tells a reduction from an inclusive or exclusive scan.
_GROUP_OPERATIONS = frozenset(
    {"Reduce", "InclusiveScan", "ExclusiveScan", "ClusteredReduce"}
)
_INSTRUCTION = re.compile(r"^\s*(?:%\w+\s*=\s*)?(Op\w+)(.*)$")
_IMPORT = re.compile(r'^\s*(%\w+)\s*=\s*OpExtInstImport\s+"([^"]+)"')
_CONSTANT = re.compile(r"^\s*(%\w+)\s*=\s*OpConstant\s+%\w+\s+(\S+)")
_BUILTIN = re.compile(r"^\s*OpDecorate\s+(%\w+)\s+BuiltIn\s+(\w+)")
# Instructions that read a variable through the pointer that is their second operand.
_READS = frozenset({"OpLoad", "OpAccessChain", "OpInBoundsAccessChain"})


def instructions(assembly: str) -> Counter:
    """The instructions inside function bodies, with extended instructions named ``set:instruction``. A read of a
    built-in variable also counts as ``BuiltIn:<name>``, since a function such as HLSL ``WorldRayOrigin`` lowers to
    that read alone."""
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
        token = _token(opcode, words, sets, constants)
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
    opcode = _ALIASES.get(opcode, opcode)
    if opcode in _CONSTANT_OPERANDS:
        return f"{opcode}({','.join(constants[w] for w in words if w in constants)})"
    operation = next((w for w in words if w in _GROUP_OPERATIONS), "")
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
    if feature.language == "spirv":
        return f"GLSL.std.450:{feature.name}", ""
    if (feature.language, feature.group) in UNIDENTIFIED:
        return "", UNIDENTIFIED[feature.language, feature.group]
    if feature.target:
        return "", f"probed for {feature.target}, not SPIR-V"
    if feature.language not in validators.SPIRV_COMPILERS:
        return "", f"no SPIR-V compiler for {feature.language}"
    if not validators.can_lower(feature.language):
        tool = validators.SPIRV_COMPILERS[feature.language][0]
        return "", f"{tool} or spirv-dis is not installed"
    try:
        probe = probes.build(feature, feature.language)
        baseline = probes.build(feature, feature.language, baseline=True)
    except probes.ProbeError as error:
        return "", str(error)
    stage = feature.stage
    lowered, log = validators.lower(feature.language, probe.source, probe.entry, stage)
    if not lowered:
        return (
            "",
            f"does not lower to SPIR-V: {validators.first_error(log, feature.language)}",
        )
    reference, log = validators.lower(
        feature.language, baseline.source, baseline.entry, stage
    )
    if not reference:
        return (
            "",
            f"the baseline does not lower to SPIR-V: {validators.first_error(log, feature.language)}",
        )
    semantic = identify(instructions(lowered), instructions(reference))
    return (semantic, "") if semantic else ("", "the call lowers to no instruction")
