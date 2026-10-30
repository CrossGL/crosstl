"""Subgroup-size, subgroup-invocation-id, num-subgroups, subgroup-id and workgroup-size probes: the compute built-ins
that say how a workgroup is split into subgroups and how large it is.

Each probe writes one built-in to its output as a float. GLSL reads ``gl_SubgroupSize``, ``gl_SubgroupInvocationID``,
``gl_NumSubgroups`` and ``gl_SubgroupID`` (GL_KHR_shader_subgroup_basic) and ``gl_WorkGroupSize``, and CrossGL the
same names. HLSL has ``WaveGetLaneCount()`` and ``WaveGetLaneIndex()``; Slang adds ``WaveGetNumWaves()``,
``WaveGetWaveIndex()`` and ``WorkgroupSize()``. Metal uses the ``[[threads_per_simdgroup]]``,
``[[thread_index_in_simdgroup]]``, ``[[simdgroups_per_threadgroup]]``, ``[[simdgroup_index_in_threadgroup]]`` and
``[[threads_per_threadgroup]]`` attributes, WGSL the ``subgroup_size``, ``subgroup_invocation_id``, ``num_subgroups``
and ``subgroup_id`` built-ins, and Rust the attributes of the same names. OpenCL C has ``get_sub_group_size()``,
``get_sub_group_local_id()``, ``get_num_sub_groups()``, ``get_sub_group_id()`` and ``get_local_size()``.

CUDA and Mojo have the warp size (``warpSize``, ``WARP_SIZE``) and the block size (``blockDim``, ``block_dim``), and
Mojo the lane and warp indices (``lane_id()``, ``warp_id()``) too. CUDA kernels compute the lane and warp indices, and
both compute the number of warps, from the flattened thread index the way GLSL defines them; HIP has ``__lane_id()``.
HLSL has no built-in for the number or index of waves in a group or for the group's size, WGSL and Rust none for the
workgroup's size (rust-gpu leaves out the ``WorkgroupSize`` constant), and WebGL 2 has no compute stage.

naga rejects ``enable subgroups;`` ("not yet implemented in Naga") and accepts the built-ins without it, so the WGSL
probes have no ``enable`` directive.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "subgroup-size",
    "subgroup-invocation-id",
    "num-subgroups",
    "subgroup-id",
    "workgroup-size",
)

_BASIC = "#extension GL_KHR_shader_subgroup_basic : require"
GLSL = {
    "subgroup-size": Template("outp[i] = float(gl_SubgroupSize);", prelude=_BASIC),
    "subgroup-invocation-id": Template(
        "outp[i] = float(gl_SubgroupInvocationID);", prelude=_BASIC
    ),
    "num-subgroups": Template("outp[i] = float(gl_NumSubgroups);", prelude=_BASIC),
    "subgroup-id": Template("outp[i] = float(gl_SubgroupID);", prelude=_BASIC),
    "workgroup-size": Template("outp[i] = float(gl_WorkGroupSize.x);"),
}
CROSSGL = {name: Template(template.body) for name, template in GLSL.items()}
WEBGL = dict.fromkeys(NAMES, ABSENT)

HLSL = {
    "subgroup-size": Template("outp[i] = float(WaveGetLaneCount());"),
    "subgroup-invocation-id": Template("outp[i] = float(WaveGetLaneIndex());"),
    "num-subgroups": ABSENT,
    "subgroup-id": ABSENT,
    "workgroup-size": ABSENT,
}
SLANG = {
    **HLSL,
    "num-subgroups": Template("outp[i] = float(WaveGetNumWaves());"),
    "subgroup-id": Template("outp[i] = float(WaveGetWaveIndex());"),
    "workgroup-size": Template("outp[i] = float(WorkgroupSize().x);"),
}


def _parameter(spelling: str, attribute: str, kind: str = "uint") -> Template:
    return Template(
        f"outp[i] = float({spelling});", params=f"{kind} {spelling} {attribute}"
    )


METAL = {
    "subgroup-size": _parameter("width", "[[threads_per_simdgroup]]"),
    "subgroup-invocation-id": _parameter("lane", "[[thread_index_in_simdgroup]]"),
    "num-subgroups": _parameter("groups", "[[simdgroups_per_threadgroup]]"),
    "subgroup-id": _parameter("group", "[[simdgroup_index_in_threadgroup]]"),
    "workgroup-size": Template(
        "outp[i] = float(size.x);", params="uint3 size [[threads_per_threadgroup]]"
    ),
}
WGSL = {
    **{
        name: Template(
            f"outp[i] = f32({spelling});", params=f"@builtin({builtin}) {spelling}: u32"
        )
        for name, builtin, spelling in (
            ("subgroup-size", "subgroup_size", "width"),
            ("subgroup-invocation-id", "subgroup_invocation_id", "lane"),
            ("num-subgroups", "num_subgroups", "groups"),
            ("subgroup-id", "subgroup_id", "group"),
        )
    },
    "workgroup-size": ABSENT,
}
RUST = {
    **{
        name: Template(
            f"outp[i] = {spelling} as f32;",
            params=f"#[spirv({builtin})] {spelling}: u32",
        )
        for name, builtin, spelling in (
            ("subgroup-size", "subgroup_size", "width"),
            ("subgroup-invocation-id", "subgroup_local_invocation_id", "lane"),
            ("num-subgroups", "num_subgroups", "groups"),
            ("subgroup-id", "subgroup_id", "group"),
        )
    },
    "workgroup-size": ABSENT,
}

_CUDA_INDEX = (
    "uint index = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);"
)
_CUDA_SIZE = "uint size = blockDim.x * blockDim.y * blockDim.z;"
CUDA = {
    "subgroup-size": Template("outp[i] = float(warpSize);"),
    "subgroup-invocation-id": Template(
        f"{_CUDA_INDEX}\noutp[i] = float(index % warpSize);"
    ),
    "num-subgroups": Template(
        f"{_CUDA_SIZE}\noutp[i] = float((size + warpSize - 1) / warpSize);"
    ),
    "subgroup-id": Template(f"{_CUDA_INDEX}\noutp[i] = float(index / warpSize);"),
    "workgroup-size": Template("outp[i] = float(blockDim.x);"),
}
HIP = {**CUDA, "subgroup-invocation-id": Template("outp[i] = float(__lane_id());")}
OPENCL = {
    "subgroup-size": Template("outp[i] = (float)get_sub_group_size();"),
    "subgroup-invocation-id": Template("outp[i] = (float)get_sub_group_local_id();"),
    "num-subgroups": Template("outp[i] = (float)get_num_sub_groups();"),
    "subgroup-id": Template("outp[i] = (float)get_sub_group_id();"),
    "workgroup-size": Template("outp[i] = (float)get_local_size(0);"),
}

_MOJO_GPU = "from max.gpu import {}"
MOJO = {
    "subgroup-size": Template(
        "outp[i] = Float32(WARP_SIZE)", prelude=_MOJO_GPU.format("WARP_SIZE")
    ),
    "subgroup-invocation-id": Template(
        "outp[i] = Float32(lane_id())", prelude=_MOJO_GPU.format("lane_id")
    ),
    "num-subgroups": Template(
        "var size = block_dim.x * block_dim.y * block_dim.z\n"
        "outp[i] = Float32((size + WARP_SIZE - 1) // WARP_SIZE)",
        prelude=_MOJO_GPU.format("WARP_SIZE, block_dim"),
    ),
    "subgroup-id": Template(
        "outp[i] = Float32(warp_id())", prelude=_MOJO_GPU.format("warp_id")
    ),
    "workgroup-size": Template(
        "outp[i] = Float32(block_dim.x)", prelude=_MOJO_GPU.format("block_dim")
    ),
}
