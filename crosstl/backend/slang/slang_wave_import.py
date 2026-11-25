"""CrossGL spellings of the wave and dispatch functions Slang's core module names differently.

Slang v2026.20 (``hlsl.meta.slang``) declares:

- ``WavePrefix{Min,Max,BitAnd,BitOr,BitXor}``, exclusive scans like ``WavePrefixSum`` (they lower to the SPIR-V
  ``ExclusiveScan`` group operations), which CrossGL names ``WavePrefixExclusive*``;
- ``WaveMultiPrefixExclusive{Sum,Product,BitAnd,BitOr,BitXor}``, which are Shader Model 6.5's
  ``WaveMultiPrefix*`` scans, CrossGL's names for them;
- ``WaveShuffle(value, lane)`` and ``WaveBroadcastLaneAt(value, lane)``, the value of another lane, which CrossGL
  reads with ``WaveReadLaneAt``;
- ``WaveGetConvergedMulti()`` and ``WaveGetActiveMulti()``, the mask of the lanes that execute the call, which is
  ``WaveActiveBallot(true)``;
- built-in values read through functions: ``WaveGetLane{Eq,Ge,Gt,Le,Lt}Mask()`` (``gl_Subgroup*Mask``),
  ``WaveGetNumWaves()`` (``gl_NumSubgroups``), ``WaveGetWaveIndex()`` (``gl_SubgroupID``), ``WorkgroupCount()``
  (``gl_NumWorkGroups``) and ``WorkgroupSize()``, an ``int3`` (``ivec3(gl_WorkGroupSize)``).
"""

from __future__ import annotations

__all__ = ["crossgl_slang_wave_call"]

_RENAMES = {
    "WavePrefixMin": "WavePrefixExclusiveMin",
    "WavePrefixMax": "WavePrefixExclusiveMax",
    "WavePrefixBitAnd": "WavePrefixExclusiveBitAnd",
    "WavePrefixBitOr": "WavePrefixExclusiveBitOr",
    "WavePrefixBitXor": "WavePrefixExclusiveBitXor",
    "WaveMultiPrefixExclusiveSum": "WaveMultiPrefixSum",
    "WaveMultiPrefixExclusiveProduct": "WaveMultiPrefixProduct",
    "WaveMultiPrefixExclusiveBitAnd": "WaveMultiPrefixBitAnd",
    "WaveMultiPrefixExclusiveBitOr": "WaveMultiPrefixBitOr",
    "WaveMultiPrefixExclusiveBitXor": "WaveMultiPrefixBitXor",
    "WaveShuffle": "WaveReadLaneAt",
    "WaveBroadcastLaneAt": "WaveReadLaneAt",
}
_ARITIES = {
    name: 2 if "Multi" in name or name in {"WaveShuffle", "WaveBroadcastLaneAt"} else 1
    for name in _RENAMES
}
# Functions without operands and the CrossGL expression of the value each returns.
_VALUES = {
    "WaveGetConvergedMulti": "WaveActiveBallot(true)",
    "WaveGetActiveMulti": "WaveActiveBallot(true)",
    "WaveGetLaneEqMask": "gl_SubgroupEqMask",
    "WaveGetLaneGeMask": "gl_SubgroupGeMask",
    "WaveGetLaneGtMask": "gl_SubgroupGtMask",
    "WaveGetLaneLeMask": "gl_SubgroupLeMask",
    "WaveGetLaneLtMask": "gl_SubgroupLtMask",
    "WaveGetNumWaves": "gl_NumSubgroups",
    "WaveGetWaveIndex": "gl_SubgroupID",
    "WorkgroupCount": "gl_NumWorkGroups",
    "WorkgroupSize": "ivec3(gl_WorkGroupSize)",
}


def crossgl_slang_wave_call(name, rendered_args):
    """CrossGL source of a call to one of these Slang functions, or ``None`` for any other callee.

    A call with the wrong number of operands is left alone, so the CrossGL program refuses the Slang name.
    """
    if name in _VALUES:
        return _VALUES[name] if not rendered_args else None
    if name in _RENAMES and len(rendered_args) == _ARITIES[name]:
        return f"{_RENAMES[name]}({', '.join(rendered_args)})"
    return None
