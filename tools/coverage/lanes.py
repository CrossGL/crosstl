"""Subgroup operations, whose probes run with more than one invocation.

A subgroup operation combines the values of the subgroup's active invocations (GL_KHR_shader_subgroup; SPIR-V 1.6,
section 3.42.24, "Non-Uniform Instructions"). Every other probe runs each invocation in a workgroup of its own, where
it is its subgroup's only active invocation, and a translation that returns the invocation's own value, a single-lane
stand-in, would compute what the native probe computes there. So the probe of a subgroup operation runs as one
workgroup of :data:`.execution.THREADS` invocations, and only the invocations whose index is not a multiple of
:data:`SKIPPED` make the call. Invocation 0 and every third one after it are then inactive, so an operation that
reads every lane, or lane 0, instead of the active ones computes something else too.

An operation that reads the value of a lane it names (a shuffle, a broadcast from a given lane, a quad swap or a
rotation) returns an undefined value when that lane is inactive (GL_KHR_shader_subgroup, ``subgroupShuffle()``), so
its probe keeps every invocation active, and so does the probe of an operation that every invocation of the subgroup
must call: OpenCL C's ``cl_khr_subgroups`` functions and its subgroup pipe reservations ("must be encountered by all
work items in a sub-group", OpenCL C 3.0, section 6.15.16). Barriers return no value to compare, and stay
single-invocation probes.
"""

from __future__ import annotations

import re

from . import spellings
from .model import VOID, Feature

# Every invocation whose index is a multiple of this one skips the call.
SKIPPED = 3
# The names of subgroup operations in each language, in lower case and without underscores: HLSL's and Slang's wave
# and quad intrinsics, CrossGL's, GLSL's and WGSL's subgroup and quad functions, GLSL's invocation votes, Metal's simd
# and quad functions (not its simdgroup matrices), Rust's and OpenCL C's subgroup functions, and SPIR-V's non-uniform
# instructions.
_OPERATION = re.compile(
    r"wave|quad|subgroup|simd(?!group)|allinvocations|anyinvocation|opgroupnonuniform"
)
# CUDA's and HIP's warp shuffles, votes, reductions and matches, and HIP's lane permutes, in lower case.
_WARP = re.compile(
    r"__(?:shfl|ballot|activemask|uni_sync|(?:all|any)(?:_sync)?$|reduce_\w+_sync|match_\w+_sync|hip_ds_\w*permute)"
)
# Groups of operations that every invocation of the subgroup must call, by language.
_UNIFORM = {"opencl": frozenset({"cl_khr_subgroups", "pipe"})}
# Those that read the value of a lane they name, in lower case and without underscores.
_NAMED_LANE = re.compile(
    r"shuffle|shfl|broadcast(?!first)|readlaneat|readacross|quadswap|quadbroadcast(?!first)|quadread|rotate|permute"
)


def _names(feature: Feature) -> list[str]:
    """The lower-case names of what ``feature``'s native and CrossGL probes call."""
    spelled = feature.crossgl or feature.spelling or feature.name
    names = spellings.named(spelled) if spellings.is_expression(spelled) else [spelled]
    return [name.lower() for name in (*names, feature.name)]


def _operation(name: str) -> bool:
    return bool(_WARP.match(name) or _OPERATION.match(name.replace("_", "")))


def cooperative(feature: Feature) -> bool:
    """Whether ``feature`` is a subgroup operation that returns a value, whose probe runs as one workgroup."""
    signature = feature.signature
    if feature.stage != "compute" or signature is None or signature.result == VOID:
        return False
    return any(_operation(name) for name in _names(feature))


def divergent(feature: Feature) -> bool:
    """Whether the probe of the subgroup operation ``feature`` leaves the invocations :data:`SKIPPED` names inactive."""
    if feature.group in _UNIFORM.get(feature.language, ()):
        return False
    return cooperative(feature) and not any(
        _NAMED_LANE.search(name.replace("_", "")) for name in _names(feature)
    )
