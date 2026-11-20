"""Subgroup built-in variables and reserved ``gl_`` names in CUDA and HIP output.

CUDA and HIP form the warps (wavefronts) of a block from consecutive linear thread IDs, ``x + y * Dx + z * Dx * Dy``
(CUDA C++ Programming Guide, "SIMT Architecture" and "Thread Hierarchy"; the HIP programming guide follows the same
rule), and ``warpSize`` holds the width. The GLSL subgroup built-ins (``GL_KHR_shader_subgroup``) of a compute
invocation therefore lower to:

- ``gl_SubgroupSize``: ``warpSize``;
- ``gl_SubgroupInvocationID``: the linear thread ID modulo ``warpSize``;
- ``gl_SubgroupID``: the linear thread ID divided by ``warpSize``;
- ``gl_NumSubgroups``: the block size divided by ``warpSize``, rounded up.

HLSL's ``WaveGetLaneIndex()`` is the same lane index. Any other identifier in the reserved ``gl_`` namespace
(GLSL 4.60 section 3.7: identifiers starting with ``gl_`` are reserved for the implementation, so a program never
declares one) that the generator does not map names a built-in variable with no CUDA or HIP equivalent, and is
refused instead of being written as an undeclared identifier.
"""

from __future__ import annotations

__all__ = [
    "THREAD_BLOCK_LINEAR_INDEX",
    "reserved_builtin_reason",
    "subgroup_builtin_expression",
    "warp_lane_index_expression",
]

# The linear thread ID within the block.
THREAD_BLOCK_LINEAR_INDEX = (
    "(threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x)"
)
_BLOCK_SIZE = "(blockDim.x * blockDim.y * blockDim.z)"
_WARP_SIZE = "((unsigned int)warpSize)"
_RESERVED_PREFIX = "gl_"


def warp_lane_index_expression():
    """The lane of the current thread in its warp: the linear thread ID modulo ``warpSize``."""
    return f"({THREAD_BLOCK_LINEAR_INDEX} % {_WARP_SIZE})"


def subgroup_builtin_expression(name, lane_index=None):
    """The CUDA and HIP expression of the GLSL subgroup built-in ``name``, or ``None`` for any other name.

    ``lane_index`` is the generator's spelling of the lane index, if it has one; it must equal
    :func:`warp_lane_index_expression`.
    """
    if name == "gl_SubgroupSize":
        return _WARP_SIZE
    if name == "gl_SubgroupInvocationID":
        return lane_index or warp_lane_index_expression()
    if name == "gl_SubgroupID":
        return f"({THREAD_BLOCK_LINEAR_INDEX} / {_WARP_SIZE})"
    if name == "gl_NumSubgroups":
        return f"(({_BLOCK_SIZE} + {_WARP_SIZE} - 1u) / {_WARP_SIZE})"
    return None


def reserved_builtin_reason(name, label):
    """Why the identifier ``name``, which the generator does not map, cannot be written, or ``None``."""
    if isinstance(name, str) and name.startswith(_RESERVED_PREFIX):
        return f"built-in variable {name}: no {label} equivalent"
    return None
