"""Mesh and task stages in the rust-gpu output (:mod:`crosstl.translator.codegen.rust_gpu_mesh`).

CrossGL's HLSL form of a mesh stage (output arrays of vertex and primitive structures, an index array, a payload
parameter) becomes one ``#[spirv(mesh_ext(...))]`` entry point with an output array per member, and a task stage a
``#[spirv(task_ext(...))]`` entry point whose module-scope payload is task-payload storage. Each crate compiles with
rust-gpu for Vulkan 1.2 with SPV_EXT_mesh_shader, as the coverage harness compiles mesh probes.
"""

from __future__ import annotations

import subprocess

import pytest

from crosstl.translator.codegen.rust_refusals import RustLoweringRefused
from tools.coverage import rustgpu

from .test_codegen.test_rust_codegen import generate_code, parse_code, tokenize_code


def _rust(source):
    return generate_code(parse_code(tokenize_code(source)))


def _assert_mesh_crate_compiles(code, tmp_path):
    command = rustgpu.command(rustgpu.MESH_TARGET)
    if not command:
        pytest.skip(
            "rust-gpu is not built; set CROSSTL_COVERAGE_RUST_GPU to a built checkout"
        )
    (tmp_path / "generated.rs").write_text(code, encoding="utf-8")
    result = subprocess.run(
        [*command, "generated.rs", "--out-dir", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n\n" + code


_VERTEX = (
    "struct Vertex {\n    vec4 position @ gl_Position;\n    vec2 uv @ TEXCOORD0;\n}"
)


def _mesh(parameters, body, structs=_VERTEX, topology="triangle", extra=""):
    return f"""
        shader Probe {{
            {structs}
            {extra}
            mesh {{
                void main({parameters}) @numthreads(2, 1, 1) @outputtopology({topology}) {{
                    {body}
                }}
            }}
        }}
        """


def test_mesh_outputs_become_per_member_arrays(tmp_path):
    code = _rust(
        _mesh(
            "@ vertices out Vertex verts[3], @ indices out uvec3 tris[1], uint li @ SV_GroupIndex",
            "SetMeshOutputCounts(3, 1);\n"
            "verts[li].position = vec4(float(li), 0.0, 0.0, 1.0);\n"
            "verts[li].uv = vec2(0.5, 0.5);\n"
            "tris[0] = uvec3(0, 1, 2);",
        )
    )
    assert (
        "#[spirv(mesh_ext(threads(2, 1, 1), output_vertices = 3, output_primitives_ext = 1, "
        "output_triangles_ext))]" in code
    ), code
    assert "#[spirv(position)] verts_position: &mut [Vec4; 3]" in code
    # TEXCOORD0 takes location 5 on every target (stage_io_semantic_locations).
    assert "#[spirv(location = 5)] verts_uv: &mut [Vec2; 3]" in code
    assert "#[spirv(primitive_triangle_indices_ext)] tris: &mut [UVec3; 1]" in code
    assert "#[spirv(local_invocation_index)] li: u32" in code
    assert "spirv_std::arch::set_mesh_outputs_ext(3, 1)" in code
    assert "verts_uv[li as usize] = " in code
    _assert_mesh_crate_compiles(code, tmp_path)


def test_per_primitive_outputs_and_built_ins(tmp_path):
    structs = (
        _VERTEX + "\nstruct Primitive {\n    vec4 color @ COLOR0;\n"
        "    uint id @ SV_PrimitiveID;\n    bool culled @ SV_CullPrimitive;\n}"
    )
    code = _rust(
        _mesh(
            "@ vertices out Vertex verts[3], @ indices out uvec3 tris[1], @ primitives out Primitive prims[1]",
            "SetMeshOutputCounts(3, 1);\n"
            "verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);\n"
            "tris[0] = uvec3(0, 1, 2);\n"
            "prims[0].color = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "prims[0].id = 7u;\n"
            "prims[0].culled = false;",
            structs=structs,
        )
    )
    assert (
        "#[spirv(location = 13, per_primitive_ext)] prims_color: &mut [Vec4; 1]" in code
    )
    # Vulkan requires a signed PrimitiveId; the HLSL uint converts on write.
    assert "#[spirv(primitive_id, per_primitive_ext)] prims_id: &mut [i32; 1]" in code
    assert (
        "#[spirv(cull_primitive_ext, per_primitive_ext)] prims_culled: &mut [bool; 1]"
        in code
    )
    _assert_mesh_crate_compiles(code, tmp_path)


@pytest.mark.parametrize(
    "topology, index_type, mode, builtin",
    [
        ("line", "uvec2", "output_lines_ext", "primitive_line_indices_ext"),
        ("point", "uint", "output_points", "primitive_point_indices_ext"),
    ],
)
def test_line_and_point_topologies(topology, index_type, mode, builtin, tmp_path):
    value = "uvec2(0, 1)" if index_type == "uvec2" else "0u"
    code = _rust(
        _mesh(
            f"@ vertices out Vertex verts[2], @ indices out {index_type} prims[1]",
            f"SetMeshOutputCounts(2, 1);\nverts[0].position = vec4(0.0);\nprims[0] = {value};",
            topology=topology,
        )
    )
    assert f"output_primitives_ext = 1, {mode}))]" in code, code
    assert f"#[spirv({builtin})] prims:" in code
    _assert_mesh_crate_compiles(code, tmp_path)


_TASK = """
    groupshared Payload payload @ taskPayloadSharedEXT;
    task {
        void main() @numthreads(1, 1, 1) {
            payload.values[0] = 1u;
            payload.values[1] = 2u;
            DispatchMesh(1u, 1u, 1u, payload);
        }
    }
"""
_PAYLOAD_MESH = """
    mesh {
        void main(@ vertices out Vertex verts[3], @ indices out uvec3 tris[1], @ mesh_payload in Payload payload)
            @numthreads(1, 1, 1) @outputtopology(triangle) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = vec4(float(payload.values[1]), 0.0, 0.0, 1.0);
            tris[0] = uvec3(0, 1, 2);
        }
    }
"""
_PAYLOAD_STRUCTS = """
    struct Vertex {
        vec4 position @ gl_Position;
    }
    struct Payload {
        uint values[2];
    }
"""


def _program(*stages):
    return "shader Probe {" + _PAYLOAD_STRUCTS + "".join(stages) + "}"


def test_task_payload_is_task_payload_storage(tmp_path):
    code = _rust(_program(_TASK))
    assert "#[spirv(task_ext(threads(1, 1, 1)))]" in code
    assert "#[spirv(task_payload_workgroup_ext)] payload: &mut Payload" in code, code
    assert "(*payload).values[0] = 1;" in code
    assert "spirv_std::arch::emit_mesh_tasks_ext_payload(1, 1, 1, payload)" in code
    _assert_mesh_crate_compiles(code, tmp_path)


def test_mesh_reads_the_task_payload(tmp_path):
    code = _rust(_program(_PAYLOAD_MESH))
    assert "#[spirv(task_payload_workgroup_ext)] payload: &mut Payload" in code, code
    assert "((*payload).values[1] as f32)" in code
    _assert_mesh_crate_compiles(code, tmp_path)


def test_mesh_entry_beside_another_entry_point_is_refused():
    with pytest.raises(
        RustLoweringRefused, match="only entry point of its rust-gpu crate"
    ):
        _rust(_program(_TASK, _PAYLOAD_MESH))


def test_glsl_emit_mesh_tasks_without_payload(tmp_path):
    code = _rust("""
        shader Probe {
            task {
                buffer uint* counts;
                void main() @numthreads(1, 1, 1) {
                    EmitMeshTasksEXT(counts[0], 1u, 1u);
                }
            }
        }
        """)
    assert (
        "spirv_std::arch::emit_mesh_tasks_ext(counts[0 as usize], 1, 1)"
        in code.replace("(0) as usize", "0 as usize")
        or "spirv_std::arch::emit_mesh_tasks_ext(" in code
    ), code
    _assert_mesh_crate_compiles(code, tmp_path)


@pytest.mark.parametrize(
    "parameters, body, message",
    [
        (
            "@ vertices out Vertex verts[3]",
            "SetMeshOutputCounts(3, 0);",
            "needs a vertices and an indices output parameter",
        ),
        (
            "@ vertices out Vertex verts[3], @ indices out uvec2 tris[1]",
            "SetMeshOutputCounts(3, 1);",
            "are uvec2; triangle outputs take uvec3 indices",
        ),
        (
            "@ vertices out Vertex verts[3], @ indices out uvec3 tris[1], float scale",
            "SetMeshOutputCounts(3, 1);",
            "is neither a mesh output, a task payload nor a compute built-in",
        ),
    ],
)
def test_mesh_entry_points_rust_gpu_cannot_declare_are_refused(
    parameters, body, message
):
    with pytest.raises(RustLoweringRefused, match=message):
        _rust(_mesh(parameters, body))


def test_vertex_struct_array_member_is_refused():
    structs = "struct Vertex {\n    vec4 position @ gl_Position;\n    float clip[2] @ SV_ClipDistance;\n}"
    with pytest.raises(RustLoweringRefused, match="rust-gpu mesh outputs are scalars"):
        _rust(
            _mesh(
                "@ vertices out Vertex verts[3], @ indices out uvec3 tris[1]",
                "SetMeshOutputCounts(3, 1);",
                structs=structs,
            )
        )


@pytest.mark.parametrize(
    "semantic, builtin",
    [
        ("SV_RenderTargetArrayIndex", "layer"),
        ("SV_ViewportArrayIndex", "viewport_index"),
        ("gl_Layer", "layer"),
        ("gl_ViewportIndex", "viewport_index"),
        ("gl_PrimitiveID", "primitive_id"),
    ],
)
def test_unsigned_primitive_built_ins_are_signed_records(semantic, builtin, tmp_path):
    # Vulkan requires a signed 32-bit Layer, ViewportIndex and PrimitiveId; the uint member converts on each access.
    structs = _VERTEX + f"\nstruct Primitive {{\n    uint value @ {semantic};\n}}"
    code = _rust(
        _mesh(
            "@ vertices out Vertex verts[3], @ indices out uvec3 tris[1], @ primitives out Primitive prims[1]",
            "SetMeshOutputCounts(3, 1);\n"
            "verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);\n"
            "tris[0] = uvec3(0, 1, 2);\n"
            "prims[0].value = 1u;\n"
            "prims[0].value = prims[0].value + 1u;",
            structs=structs,
        )
    )
    assert f"#[spirv({builtin}, per_primitive_ext)] prims_value: &mut [i32; 1]" in code
    assert "prims_value[0] = (((prims_value[0] as u32) + 1) as i32);" in code
    _assert_mesh_crate_compiles(code, tmp_path)
