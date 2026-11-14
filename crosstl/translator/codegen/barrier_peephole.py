"""Merge adjacent barrier statements in generated Metal and HLSL into one barrier.

CrossGL spells a control barrier that also orders buffer or image memory as ``fence; barrier; fence`` (the Metal,
HLSL and SPIR-V importers produce it, and GLSL sources write ``memoryBarrierShared(); barrier();``). Metal and HLSL
express that synchronization as a single barrier: ``threadgroup_barrier(mem_flags::mem_device |
mem_flags::mem_threadgroup)`` or ``AllMemoryBarrierWithGroupSync()``. Emitting each CrossGL statement separately
synchronizes the threadgroup up to three times instead of once.

Barrier statements that follow one another directly in the same block, with nothing executed between them, order
memory exactly as one barrier over the union of their memory does, and synchronize execution when any of them does:
no access can fall between the barriers, so no access can tell them apart (Metal Shading Language 3.2 section 6.9.1;
HLSL ``GroupMemoryBarrier``, ``DeviceMemoryBarrier`` and ``AllMemoryBarrier`` and their ``WithGroupSync`` forms).
The merge only joins barrier statements on consecutive lines at the same indentation, which the emitters produce
only for statements of one block, and only barriers of one kind (Metal threadgroup barriers, Metal SIMD-group
barriers, or HLSL group barriers).
"""

from __future__ import annotations

import re

__all__ = [
    "HLSL_EXECUTION_BARRIER_PLACEHOLDER",
    "merge_adjacent_hlsl_barriers",
    "merge_adjacent_metal_barriers",
]

_METAL_FLAG_ORDER = ("mem_device", "mem_threadgroup", "mem_texture")
_METAL_BARRIER = re.compile(
    r"^(?P<indent>[ \t]*)(?P<kind>threadgroup_barrier|simdgroup_barrier)"
    r"\((?P<flags>mem_flags::mem_\w+(?:\s*\|\s*mem_flags::mem_\w+)*)\);$"
)
_HLSL_BARRIERS = {
    "GroupMemoryBarrier": (False, frozenset({"groupshared"})),
    "DeviceMemoryBarrier": (False, frozenset({"device"})),
    "AllMemoryBarrier": (False, frozenset({"groupshared", "device"})),
    "GroupMemoryBarrierWithGroupSync": (True, frozenset({"groupshared"})),
    "DeviceMemoryBarrierWithGroupSync": (True, frozenset({"device"})),
    "AllMemoryBarrierWithGroupSync": (True, frozenset({"groupshared", "device"})),
}
_HLSL_NAMES = {value: name for name, value in _HLSL_BARRIERS.items()}
# HLSL has no execution barrier without memory semantics. The emitter writes CrossGL's workgroupExecutionBarrier as
# this placeholder so that the merge can tell it from GroupMemoryBarrierWithGroupSync; on its own it becomes
# GroupMemoryBarrierWithGroupSync, the narrowest HLSL execution barrier.
HLSL_EXECUTION_BARRIER_PLACEHOLDER = "__crossgl_workgroup_execution_barrier"
_HLSL_EXECUTION_BARRIER = (True, frozenset())
_HLSL_NAMES[_HLSL_EXECUTION_BARRIER] = "GroupMemoryBarrierWithGroupSync"
_HLSL_BARRIER = re.compile(
    r"^(?P<indent>[ \t]*)(?P<name>"
    + "|".join([*_HLSL_BARRIERS, HLSL_EXECUTION_BARRIER_PLACEHOLDER])
    + r")\(\);$"
)


def _merge_runs(code: str, parse, render) -> str:
    """Replace each run of consecutive same-indentation barrier lines with one rendered barrier."""
    lines = code.split("\n")
    merged = []
    run_key = None
    run_value = None
    for line in lines:
        parsed = parse(line)
        if parsed is not None and run_key == parsed[0]:
            run_value = (run_value[0] or parsed[1][0], run_value[1] | parsed[1][1])
            merged[-1] = render(parsed[0], run_value)
            continue
        if parsed is not None:
            run_key, run_value = parsed
        else:
            run_key = run_value = None
        merged.append(line)
    return "\n".join(merged)


def _parse_metal(line: str):
    match = _METAL_BARRIER.match(line)
    if match is None:
        return None
    flags = frozenset(
        flag.strip().split("::")[-1] for flag in match.group("flags").split("|")
    )
    if not flags <= set(_METAL_FLAG_ORDER) | {"mem_none"}:
        return None
    return (match.group("indent"), match.group("kind")), (False, flags - {"mem_none"})


def _render_metal(key, value) -> str:
    indent, kind = key
    flags = [flag for flag in _METAL_FLAG_ORDER if flag in value[1]] or ["mem_none"]
    rendered = " | ".join(f"mem_flags::{flag}" for flag in flags)
    return f"{indent}{kind}({rendered});"


def merge_adjacent_metal_barriers(code: str) -> str:
    """Merge consecutive Metal threadgroup (or SIMD-group) barrier statements into one barrier."""
    if "_barrier(mem_flags::" not in code:
        return code
    return _merge_runs(code, _parse_metal, _render_metal)


def _parse_hlsl(line: str):
    match = _HLSL_BARRIER.match(line)
    if match is None:
        return None
    name = match.group("name")
    if name == HLSL_EXECUTION_BARRIER_PLACEHOLDER:
        return (match.group("indent"),), _HLSL_EXECUTION_BARRIER
    return (match.group("indent"),), _HLSL_BARRIERS[name]


def _render_hlsl(key, value) -> str:
    return f"{key[0]}{_HLSL_NAMES[(value[0], frozenset(value[1]))]}();"


def merge_adjacent_hlsl_barriers(code: str) -> str:
    """Merge consecutive HLSL group barrier statements into one barrier."""
    if "MemoryBarrier" not in code and HLSL_EXECUTION_BARRIER_PLACEHOLDER not in code:
        return code
    merged = _merge_runs(code, _parse_hlsl, _render_hlsl)
    return merged.replace(
        HLSL_EXECUTION_BARRIER_PLACEHOLDER, "GroupMemoryBarrierWithGroupSync"
    )
