"""Task-stage probes: an array in the payload a task stage hands its mesh workgroups, and the IDs a task invocation reads.

Each probe is the task probe of :mod:`.graphics`, a stage that writes a payload and launches one mesh workgroup, with
one part changed: its payload holds an array, it writes the ID of its workgroup into the payload, or it runs two
invocations, each writing the element of a payload array its local invocation index names. GLSL declares the payload
``taskPayloadSharedEXT`` and reads ``gl_WorkGroupID`` and ``gl_LocalInvocationIndex``. HLSL and Slang declare the
payload ``groupshared``, pass it to ``DispatchMesh`` and read the IDs through ``SV_GroupID`` and ``SV_GroupIndex``.
Metal's task stage is an object function: it writes the payload through an ``object_data`` reference, reads the IDs as
``[[threadgroup_position_in_grid]]`` and ``[[thread_index_in_threadgroup]]`` and sizes the mesh grid through
``mesh_grid_properties``, and its threadgroup size is set when the pipeline is dispatched rather than in the source.
rust-gpu takes the payload and the IDs as attributed parameters. CrossGL writes them as its HLSL importer does, with the
payload in task-payload storage as in the task stage probe of :mod:`..crossgl`.

WGSL, GLSL ES 3.00 and the kernel languages have no task stage.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("task-payload-array", "task-workgroup-id", "task-local-invocation-index")
_SCALAR = "uint value"
_ARRAY = "uint values[2]"


def _glsl(body: str, member: str = _SCALAR, threads: int = 1) -> Template:
    return Template(
        program="#version 460\n#extension GL_EXT_mesh_shader : require\n"
        f"layout(local_size_x = {threads}) in;\nstruct Payload {{\n    {member};\n}};\n"
        f"taskPayloadSharedEXT Payload payload;\nvoid main() {{\n{body}    EmitMeshTasksEXT(1, 1, 1);\n}}\n"
    )


GLSL = {
    "task-payload-array": _glsl(
        "    payload.values[0] = 1u;\n    payload.values[1] = 2u;\n", _ARRAY
    ),
    "task-workgroup-id": _glsl("    payload.value = gl_WorkGroupID.x;\n"),
    "task-local-invocation-index": _glsl(
        "    payload.values[gl_LocalInvocationIndex] = 1u;\n", _ARRAY, 2
    ),
}
WEBGL = dict.fromkeys(NAMES, ABSENT)
WGSL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)


def _hlsl_program(
    stage: str, body: str, member: str = _SCALAR, threads: int = 1, params: str = ""
) -> Template:
    return Template(
        program=f"struct Payload {{\n    {member};\n}};\n\ngroupshared Payload payload;\n\n"
        f"{stage}[numthreads({threads}, 1, 1)]\nvoid main({params}) {{\n{body}"
        "    DispatchMesh(1, 1, 1, payload);\n}\n",
        entry="main",
    )


def _hlsl(stage: str = "") -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage with ``stage`` ahead of the entry point's attributes."""
    return {
        "task-payload-array": _hlsl_program(
            stage, "    payload.values[0] = 1;\n    payload.values[1] = 2;\n", _ARRAY
        ),
        "task-workgroup-id": _hlsl_program(
            stage,
            "    payload.value = group.x;\n",
            params="uint3 group : SV_GroupID",
        ),
        "task-local-invocation-index": _hlsl_program(
            stage,
            "    payload.values[index] = 1;\n",
            _ARRAY,
            2,
            "uint index : SV_GroupIndex",
        ),
    }


HLSL = _hlsl()
SLANG = _hlsl('[shader("amplification")]\n')


def _metal(body: str, member: str = _SCALAR, params: str = "") -> Template:
    return Template(
        program="#include <metal_stdlib>\nusing namespace metal;\n\n"
        f"struct Payload {{\n    {member};\n}};\n\n"
        "[[object]] void probe(object_data Payload& payload [[payload]], mesh_grid_properties grid"
        f"{params}) {{\n{body}    grid.set_threadgroups_per_grid(uint3(1, 1, 1));\n}}\n"
    )


METAL = {
    "task-payload-array": _metal(
        "    payload.values[0] = 1;\n    payload.values[1] = 2;\n", _ARRAY
    ),
    "task-workgroup-id": _metal(
        "    payload.value = group.x;\n",
        params=",\n                      uint3 group [[threadgroup_position_in_grid]]",
    ),
    "task-local-invocation-index": _metal(
        "    payload.values[index] = 1;\n",
        _ARRAY,
        ",\n                      uint index [[thread_index_in_threadgroup]]",
    ),
}


def _rust(body: str, member: str, threads: int = 1, params: str = "") -> Template:
    return Template(
        program="#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
        f"pub struct Payload {{\n    pub {member},\n}}\n\n"
        f"#[spirv(task_ext(threads({threads})))]\n"
        f"pub fn main({params}#[spirv(task_payload_workgroup_ext)] payload: &mut Payload) {{\n"
        f"{body}    unsafe {{\n        spirv_std::arch::emit_mesh_tasks_ext_payload(1, 1, 1, payload);\n    }}\n}}\n",
        entry="main",
    )


RUST = {
    "task-payload-array": _rust(
        "    payload.values[0] = 1;\n    payload.values[1] = 2;\n", "values: [u32; 2]"
    ),
    "task-workgroup-id": _rust(
        "    payload.value = group.x;\n",
        "value: u32",
        params="#[spirv(workgroup_id)] group: UVec3, ",
    ),
    "task-local-invocation-index": _rust(
        "    payload.values[index as usize] = 1;\n",
        "values: [u32; 2]",
        2,
        "#[spirv(local_invocation_index)] index: u32, ",
    ),
}


def _crossgl(
    body: str, member: str = _SCALAR, threads: int = 1, params: str = ""
) -> Template:
    return Template(
        program=f"shader Probe {{\n    struct Payload {{\n        {member};\n    }}\n"
        "    groupshared Payload payload @ taskPayloadSharedEXT;\n    task {\n"
        f"        void main({params}) @numthreads({threads}, 1, 1) {{\n{body}"
        "            DispatchMesh(1u, 1u, 1u, payload);\n        }\n    }\n}\n"
    )


CROSSGL = {
    "task-payload-array": _crossgl(
        "            payload.values[0] = 1u;\n            payload.values[1] = 2u;\n",
        _ARRAY,
    ),
    "task-workgroup-id": _crossgl(
        "            payload.value = group.x;\n",
        params="uvec3 group @ gl_WorkGroupID",
    ),
    "task-local-invocation-index": _crossgl(
        "            payload.values[index] = 1u;\n",
        _ARRAY,
        2,
        "uint index @ gl_LocalInvocationIndex",
    ),
}
