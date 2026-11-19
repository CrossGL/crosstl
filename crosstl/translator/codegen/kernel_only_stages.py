"""CUDA and HIP declare only compute kernels.

CUDA C++ and HIP have no graphics or ray-tracing pipeline. Their only entry
point is a ``__global__`` function that the host launches over a grid (CUDA C++
Programming Guide, "Function Execution Space Specifiers"; HIP Programming Guide,
"Kernels"). A vertex, fragment, geometry, tessellation, mesh, task or
ray-tracing entry has no CUDA or HIP form: emitted as a ``__device__`` function
it leaves the program without any entry point, and named ``main`` it is not
even accepted by nvcc or hipcc. Such a stage is refused instead.
"""

from __future__ import annotations

from ..stage_utils import SHADER_STAGE_NAMES
from .stage_refusals import StageUnsupportedError

KERNEL_STAGES = frozenset({"compute"})
NON_KERNEL_STAGES = frozenset(SHADER_STAGE_NAMES - KERNEL_STAGES)

KERNEL_ONLY_REASON = (
    "a __global__ kernel launched over a grid is the only entry point, and there "
    "is no graphics or ray-tracing pipeline to run another stage"
)


def refuse_non_kernel_stages(target, label, stages):
    """Raise the stage refusal for the non-kernel ``stages`` of a CUDA or HIP program."""
    if stages:
        raise StageUnsupportedError(target, label, sorted(stages), KERNEL_ONLY_REASON)
