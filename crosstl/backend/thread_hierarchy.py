"""Thread-hierarchy built-ins of CUDA and HIP kernels and the CrossGL they import as.

CUDA and HIP launch a kernel over ``gridDim`` blocks of ``blockDim`` threads, and the host picks both at launch
time. ``__launch_bounds__`` and ``amdgpu_flat_work_group_size`` only bound the block size. An importer therefore
marks every kernel ``@ host_workgroup_size``, naming the size when its ``threadgroup_size`` source option gives
it, and the CrossGL parser keeps the mark only on kernels whose results depend on that size (see
:mod:`crosstl.translator.host_workgroup_size`).

``blockIdx.c * blockDim.c + threadIdx.c`` is the thread's index in the grid. It imports as
``gl_GlobalInvocationID.c``, which has that value, the same ``unsigned int`` type, and no dependence on the block
size.
"""

from __future__ import annotations

from typing import Any

from ..translator.host_workgroup_size import (
    host_workgroup_size_attribute_text,
    normalize_threadgroup_size_option,
)

__all__ = [
    "global_invocation_axis",
    "kernel_workgroup_size_attribute",
    "normalize_threadgroup_size_option",
]

_AXES = ("x", "y", "z")


def _builtin_axis(node: Any) -> tuple[str, str] | None:
    name = getattr(node, "builtin_name", None)
    axis = getattr(node, "component", None)
    if not isinstance(name, str) or axis not in _AXES:
        return None
    return name, axis


def _block_offset_axis(node: Any) -> str | None:
    """The axis of ``blockIdx.c * blockDim.c`` in either operand order, or ``None``."""
    if getattr(node, "op", None) != "*":
        return None
    left = _builtin_axis(getattr(node, "left", None))
    right = _builtin_axis(getattr(node, "right", None))
    if not left or not right or left[1] != right[1]:
        return None
    return left[1] if {left[0], right[0]} == {"blockIdx", "blockDim"} else None


def global_invocation_axis(node: Any) -> str | None:
    """The axis of ``blockIdx.c * blockDim.c + threadIdx.c``, in any operand order, or ``None``."""
    if getattr(node, "op", None) != "+":
        return None
    operands = (getattr(node, "left", None), getattr(node, "right", None))
    for thread, block in (operands, operands[::-1]):
        thread_axis = _builtin_axis(thread)
        if thread_axis and thread_axis[0] == "threadIdx":
            if _block_offset_axis(block) == thread_axis[1]:
                return thread_axis[1]
    return None


def kernel_workgroup_size_attribute(threadgroup_size: Any) -> str:
    """The CrossGL attribute line of a kernel whose host picks its block size."""
    size = normalize_threadgroup_size_option(threadgroup_size)
    return f"@{host_workgroup_size_attribute_text(size)}"
