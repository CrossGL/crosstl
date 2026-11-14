"""Slang lowerings of CrossGL's relative wave shuffles.

CrossGL's ``WaveShuffleDown``, ``WaveShuffleUp`` and ``WaveShuffleXor`` (MSL ``simd_shuffle_down``, ``_up`` and
``_xor``; GLSL ``subgroupShuffleDown``, ``Up`` and ``Xor``) read a value from the lane ``delta`` above, ``delta``
below, or ``lane ^ delta``. ``WaveShuffleAndFillUp(value, fill, delta[, modulo])`` (MSL
``simd_shuffle_and_fill_up``) reads ``value`` from ``delta`` lanes below within each ``modulo``-lane segment, and
``fill`` from the top of the segment for the first ``delta`` lanes (MSL 3.2 section 6.9.2, Table 6.13). Slang has
no relative shuffles, so the generator refused them. That rejected every kernel including MLX's ``utils.h``, whose
64-bit, Boolean and complex overloads are built from them.

Slang's ``WaveReadLaneAt`` takes a per-lane source lane (slangc emits ``OpGroupNonUniformShuffle`` for SPIR-V and
``simd_shuffle`` for Metal), so each shuffle becomes a helper that computes the source lane and reads it once with
every lane active. A source lane outside the SIMD-group reads the calling lane, which is MSL's documented result
(the upper ``delta`` lanes of ``simd_shuffle_down`` keep their value). It also refines the GLSL and SPIR-V
contract, which leaves that value undefined. A ``modulo`` of zero or above the SIMD-group width is undefined in
MSL. The helper never reads an out-of-range lane for it: the source lane is clamped to the calling lane.
"""

from __future__ import annotations

import re

__all__ = [
    "SLANG_WAVE_SHUFFLE_ARITIES",
    "as_type_target",
    "render_slang_wave_shuffle_helper",
    "slang_subgroup_parameter_alias",
    "slang_wave_shuffle_helper_name",
]

# Entry parameters for these built-ins have no Slang system value (HLSL has none either); Slang reads them from
# intrinsics, which slangc lowers to SubgroupLocalInvocationId/SubgroupSize for SPIR-V and to
# [[thread_index_in_simdgroup]]/[[threads_per_simdgroup]] for Metal.
_SUBGROUP_PARAMETER_INTRINSICS = {
    "gl_subgroupinvocationid": "WaveGetLaneIndex()",
    "gl_subgroupsize": "WaveGetLaneCount()",
}
_SUBGROUP_PARAMETER_TYPES = frozenset(
    {"uint", "int", "uint16_t", "int16_t", "ushort", "short"}
)

SLANG_WAVE_SHUFFLE_ARITIES = {
    "WaveShuffleDown": frozenset({2}),
    "WaveShuffleUp": frozenset({2}),
    "WaveShuffleXor": frozenset({2}),
    "WaveShuffleAndFillUp": frozenset({3, 4}),
}

# Whether the computed source lane is inside the SIMD-group, and the source lane.
_RELATIVE = {
    "WaveShuffleDown": ("delta < laneCount - lane", "lane + delta"),
    "WaveShuffleUp": ("delta <= lane", "lane - delta"),
    "WaveShuffleXor": ("(lane ^ delta) < laneCount", "lane ^ delta"),
}
_SUFFIXES = {
    "WaveShuffleDown": "down",
    "WaveShuffleUp": "up",
    "WaveShuffleXor": "xor",
    "WaveShuffleAndFillUp": "and_fill_up",
}


def slang_wave_shuffle_helper_name(operation: str, value_type: str) -> str:
    """The helper that performs ``operation`` on values of the Slang type ``value_type``."""
    type_suffix = re.sub(r"[^A-Za-z0-9_]", "_", value_type)
    return f"__crossgl_wave_shuffle_{_SUFFIXES[operation]}_{type_suffix}"


def render_slang_wave_shuffle_helper(operation: str, value_type: str) -> str:
    """The Slang definition of :func:`slang_wave_shuffle_helper_name` for ``operation`` and ``value_type``."""
    name = slang_wave_shuffle_helper_name(operation, value_type)
    if operation == "WaveShuffleAndFillUp":
        return (
            f"{value_type} {name}({value_type} value, {value_type} fill, uint delta, uint modulo)\n"
            "{\n"
            "    uint lane = WaveGetLaneIndex();\n"
            "    uint laneCount = WaveGetLaneCount();\n"
            "    uint segmentLane = modulo == 0u ? lane : lane % modulo;\n"
            "    uint segmentBase = lane - segmentLane;\n"
            "    bool useValue = segmentLane >= delta;\n"
            "    uint valueSource = useValue ? lane - delta : lane;\n"
            "    uint fillSource = useValue ? lane : segmentBase + modulo + segmentLane - delta;\n"
            "    fillSource = fillSource < laneCount ? fillSource : lane;\n"
            f"    {value_type} shifted = WaveReadLaneAt(value, valueSource);\n"
            f"    {value_type} filled = WaveReadLaneAt(fill, fillSource);\n"
            "    return useValue ? shifted : filled;\n"
            "}\n"
        )
    valid, source = _RELATIVE[operation]
    return (
        f"{value_type} {name}({value_type} value, uint delta)\n"
        "{\n"
        "    uint lane = WaveGetLaneIndex();\n"
        "    uint laneCount = WaveGetLaneCount();\n"
        f"    uint source = ({valid}) ? {source} : lane;\n"
        "    return WaveReadLaneAt(value, source);\n"
        "}\n"
    )


def slang_subgroup_parameter_alias(codegen, parameter, param_type, semantic):
    """The local that binds a lane-index or SIMD-group-width entry parameter, or ``None`` for other parameters.

    A parameter of a non-integer scalar type is refused rather than converted.
    """
    if semantic is None:
        return None
    intrinsic = _SUBGROUP_PARAMETER_INTRINSICS.get(str(semantic).lower())
    if intrinsic is None:
        return None
    name = getattr(parameter, "name", None)
    type_name = str(param_type)
    if type_name not in _SUBGROUP_PARAMETER_TYPES:
        codegen.refuse_lowering(
            f"Slang @{semantic} parameter '{name}' must be an integer scalar, not {type_name}"
        )
        return None
    value = intrinsic if type_name == "uint" else f"{type_name}({intrinsic})"
    return f"{type_name} {name} = {value};"


_AS_TYPE = re.compile(r"^\s*as_type\s*<\s*(?P<type>.+?)\s*>\s*$")


def as_type_target(function_name) -> str | None:
    """The CrossGL type of ``as_type<T>``, the IR's bit reinterpretation, or ``None`` for any other callee.

    Slang spells it ``bit_cast<T>(value)``, which slangc lowers to ``OpBitcast``, MSL ``as_type`` and HLSL
    ``asuint``/``asfloat``, and which it rejects for types of different sizes. The MLX shuffle overloads reinterpret
    64-bit values as ``uint2`` through it.
    """
    if not isinstance(function_name, str):
        return None
    match = _AS_TYPE.match(function_name)
    return match.group("type") if match else None
