"""GLSL workgroup layout of a compute, mesh or task entry whose size the source's host chooses.

GLSL fixes the workgroup size in the shader. Compiled to SPIR-V (GL_ARB_gl_spirv, GL_KHR_vulkan_glsl), it takes
the size from three specialization constants that the host sets when it creates the pipeline, through
``layout(local_size_{x,y,z}_id = N) in;`` (GLSL 4.60 section 4.4.1.4). Their ids are those of
:mod:`.host_workgroup_size_ids`, so they match the generated SPIR-V's. The ``local_size`` defaults of one are the
placeholder the qualifier requires.

A driver that compiles the source has no specialization step, so the layout reads the size from the
:data:`~.host_workgroup_size_ids.HOST_WORKGROUP_SIZE_MACROS`, which the host defines after ``#version`` as it does
to select any other shader variant, and the source stops at an ``#error`` that names them otherwise. A host that
dispatched the source kernel with threadgroups of ``(x, y, z)`` passes the same three numbers, and
``gl_WorkGroupSize`` reads them.

The size is unknown when the shader is generated, so a proof that bounds invocation ids by it has no size to use
(:func:`glsl_workgroup_proof_local_size`), and lowerings that need a size, such as software subgroups, refuse the
entry.
"""

from __future__ import annotations

from typing import Any

from .host_workgroup_size_ids import (
    HOST_WORKGROUP_SIZE_MACROS,
    host_workgroup_size_ids,
    is_host_sized_workgroup,
)
from .specialized_workgroup_size import has_specialized_workgroup_size
from .stage_utils import compute_local_size

__all__ = [
    "glsl_host_workgroup_layout",
    "glsl_program_host_workgroup_size_ids",
    "glsl_workgroup_proof_local_size",
]


def glsl_host_workgroup_layout(ids, spirv_guard: str) -> str:
    """The input layout of a host-sized entry, specialized as ``ids`` on the SPIR-V path."""
    specialized = ", ".join(
        f"local_size_{axis}_id = {spec_id}" for axis, spec_id in zip("xyz", ids)
    )
    x, y, z = HOST_WORKGROUP_SIZE_MACROS
    defined = " && ".join(f"defined({macro})" for macro in HOST_WORKGROUP_SIZE_MACROS)
    return (
        f"{spirv_guard}"
        f"layout(local_size_x = 1, local_size_y = 1, local_size_z = 1, {specialized}) in;\n"
        f"#elif {defined}\n"
        f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;\n"
        "#else\n"
        f'#error "the host chooses this workgroup size: define {x}, {y} and {z} '
        'or compile this shader to SPIR-V and specialize it"\n'
        "#endif\n"
    )


def glsl_program_host_workgroup_size_ids(generator) -> tuple[int, int, int]:
    """The host-size ids of the program being generated, or of the whole program an entry was cut from."""
    program = getattr(generator, "opengl_descriptor_set_ast", None)
    cached = getattr(generator, "_glsl_host_workgroup_size_ids", None)
    if cached is not None and cached[0] is program:
        return cached[1]
    ids = host_workgroup_size_ids(program)
    generator._glsl_host_workgroup_size_ids = (program, ids)
    return ids


def glsl_workgroup_proof_local_size(config: Any):
    """The workgroup size an invocation-id proof may assume, or ``None`` when the pipeline or host sets it.

    A specialized size's ``local_size`` values are only the defaults that a pipeline may replace.
    """
    if is_host_sized_workgroup(config) or has_specialized_workgroup_size(config):
        return None
    return compute_local_size(config)
