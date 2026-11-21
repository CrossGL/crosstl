"""Slang spellings of CrossGL's subgroup built-in variables.

CrossGL reads the subgroup built-ins as GLSL does (GL_KHR_shader_subgroup): ``gl_SubgroupSize``,
``gl_SubgroupInvocationID``, the lane masks ``gl_Subgroup{Eq,Ge,Gt,Le,Lt}Mask``, and, in compute-class stages,
``gl_NumSubgroups`` and ``gl_SubgroupID``. Slang declares none of these names and has no system value for them: it
reads them from intrinsics (Slang ``docs/wave-intrinsics.md``), which slangc writes as the SPIR-V built-ins
``SubgroupSize``, ``SubgroupLocalInvocationId``, ``Subgroup*MaskKHR``, ``NumSubgroups`` and ``SubgroupId``. A
reference the generator wrote verbatim was an undefined identifier for slangc.

``WaveGetNumWaves`` and ``WaveGetWaveIndex`` exist only in compute, mesh and amplification stages, as their GLSL
counterparts do, so another entry point that reads them is refused.
"""

from __future__ import annotations

__all__ = [
    "SLANG_SUBGROUP_LANE_BUILTINS",
    "SLANG_SUBGROUP_WORKGROUP_BUILTINS",
    "slang_subgroup_builtin_candidates",
]

# Built-in -> (CrossGL type, Slang intrinsic) for the values every stage with subgroup operations reads.
SLANG_SUBGROUP_LANE_BUILTINS = {
    "gl_SubgroupSize": ("uint", "WaveGetLaneCount()"),
    "gl_SubgroupInvocationID": ("uint", "WaveGetLaneIndex()"),
    "gl_SubgroupEqMask": ("uvec4", "WaveGetLaneEqMask()"),
    "gl_SubgroupGeMask": ("uvec4", "WaveGetLaneGeMask()"),
    "gl_SubgroupGtMask": ("uvec4", "WaveGetLaneGtMask()"),
    "gl_SubgroupLeMask": ("uvec4", "WaveGetLaneLeMask()"),
    "gl_SubgroupLtMask": ("uvec4", "WaveGetLaneLtMask()"),
}

# The workgroup-relative values, which Slang's capability system admits only in compute-class stages.
SLANG_SUBGROUP_WORKGROUP_BUILTINS = {
    "gl_NumSubgroups": ("uint", "WaveGetNumWaves()"),
    "gl_SubgroupID": ("uint", "WaveGetWaveIndex()"),
}

_COMPUTE_CLASS_STAGES = frozenset({"compute", "mesh", "amplification"})


def slang_subgroup_builtin_candidates(shader_stage):
    """The subgroup built-ins a function of the Slang stage ``shader_stage`` reads through intrinsics.

    ``shader_stage`` is ``None`` for a helper function: slangc checks the stage of each entry point that calls it.
    """
    candidates = dict(SLANG_SUBGROUP_LANE_BUILTINS)
    if shader_stage is None or shader_stage in _COMPUTE_CLASS_STAGES:
        candidates.update(SLANG_SUBGROUP_WORKGROUP_BUILTINS)
    return candidates
