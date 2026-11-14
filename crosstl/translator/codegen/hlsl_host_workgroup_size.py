"""HLSL thread-group size of a compute, mesh or amplification entry whose size the source's host chooses.

A Metal kernel whose result depends on its threadgroup size, but whose source gives none, is sized by the host
when it dispatches it (:mod:`crosstl.translator.host_workgroup_size`). HLSL fixes the size in the shader with
``[numthreads(X, Y, Z)]``, and Direct3D has no specialization constants to set it when the pipeline is created.
The size only has to be fixed when the shader is compiled, though: the host compiles the generated source with the
three :data:`~.host_workgroup_size_ids.HOST_WORKGROUP_SIZE_MACROS` defined (``dxc -D CROSSGL_WORKGROUP_SIZE_X=64``,
``D3DCompile`` defines), the way Direct3D programs select any other shader variant, and the source stops at an
``#error`` that names them otherwise. The macros are the ones the generated GLSL reads when no SPIR-V
specialization step runs, so a host passes the same three numbers it dispatched the source kernel with to either
target, and ``gl_WorkGroupSize`` reads them.

The size is unknown when the shader is generated, so a proof that bounds invocation ids by it has no size to use
(:func:`hlsl_entry_proof_local_size`), and lowerings that need a concrete size, such as software subgroups,
refuse the entry.
"""

from __future__ import annotations

from typing import Any

from .host_workgroup_size_ids import HOST_WORKGROUP_SIZE_MACROS, is_host_sized_workgroup
from .stage_utils import compute_local_size

__all__ = [
    "hlsl_entry_local_size",
    "hlsl_entry_proof_local_size",
    "hlsl_host_numthreads",
    "hlsl_host_workgroup_size_guard",
]


def hlsl_entry_local_size(config: Any) -> tuple:
    """The three ``numthreads`` operands of an entry: the host's macros when it chooses the size."""
    if is_host_sized_workgroup(config):
        return HOST_WORKGROUP_SIZE_MACROS
    return tuple(compute_local_size(config))


def hlsl_entry_proof_local_size(config: Any) -> tuple | None:
    """The thread-group size an invocation-id proof may assume, or ``None`` when the host sets it."""
    if is_host_sized_workgroup(config):
        return None
    return tuple(compute_local_size(config))


def hlsl_host_workgroup_size_guard() -> str:
    """Preprocessor lines that stop compilation unless the host defines every size macro."""
    undefined = " || ".join(
        f"!defined({macro})" for macro in HOST_WORKGROUP_SIZE_MACROS
    )
    family = HOST_WORKGROUP_SIZE_MACROS[0][:-1] + "{X,Y,Z}"
    return (
        f"#if {undefined}\n"
        f'#error "the host chooses this thread-group size: define {family}"\n'
        "#endif\n"
    )


def hlsl_host_numthreads() -> str:
    """The guarded ``numthreads`` attribute of an entry whose size the host chooses."""
    return (
        f"{hlsl_host_workgroup_size_guard()}"
        f"[numthreads({', '.join(HOST_WORKGROUP_SIZE_MACROS)})]\n"
    )
