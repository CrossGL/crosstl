"""GLSL-style mesh and task stages reach HLSL and Metal through entry parameters.

A GLSL mesh shader writes module-scope output arrays (``gl_MeshVerticesEXT``, ``out`` arrays and blocks,
``perprimitiveEXT`` outputs, ``gl_PrimitiveTriangleIndicesEXT``) and reads its task payload from a
``taskPayloadSharedEXT`` variable; a GLSL task shader writes that variable and launches mesh groups with
``EmitMeshTasksEXT(x, y, z)``. HLSL and MSL take the outputs and the payload as entry parameters, and an HLSL
amplification shader passes a ``groupshared`` payload to ``DispatchMesh``. The DirectX generator refused the GLSL
form and the Metal generator printed the GLSL identifiers verbatim. In the other direction, the GLSL and SPIR-V
generators refused an HLSL amplification shader's module-scope ``groupshared`` payload.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.diagnostics import CODE_PREFIX
from crosstl.translator.semantics import GlslMeshStageUnsupportedError

MESH_GLSL = """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 32) out;
struct Task { uint base; uint count; };
taskPayloadSharedEXT Task payload;
layout(location = 0) out VertexData { vec3 normal; flat uint id; } vdata[];
layout(location = 2) perprimitiveEXT out vec4 primColor[];
layout(location = 3) out vec2 uv[];
void main() {
    uint i = gl_LocalInvocationIndex;
    SetMeshOutputsEXT(64, 32);
    gl_MeshVerticesEXT[i].gl_Position = vec4(float(payload.base + i), 0.0, 0.0, 1.0);
    gl_MeshVerticesEXT[i + 32u].gl_Position = vec4(float(gl_WorkGroupID.x), 1.0, 0.0, 1.0);
    vdata[i].normal = vec3(0.0, 0.0, 1.0);
    vdata[i].id = i;
    vdata[i + 32u].normal = vec3(0.0, 1.0, 0.0);
    vdata[i + 32u].id = i + 32u;
    uv[i] = vec2(0.0);
    uv[i + 32u] = vec2(1.0, 0.0);
    primColor[i] = vec4(1.0);
    gl_MeshPrimitivesEXT[i].gl_PrimitiveID = int(i) + 1;
    gl_PrimitiveTriangleIndicesEXT[i] = uvec3(i, i + 32u, (i + 1u) % 32u);
}
"""

FRAGMENT_GLSL = """#version 460
#extension GL_EXT_mesh_shader : require
layout(location = 0) in VertexData { vec3 normal; flat uint id; } vdata;
layout(location = 2) perprimitiveEXT in vec4 primColor;
layout(location = 0) out vec4 color;
void main() { color = vec4(vdata.normal, float(vdata.id)) * primColor; }
"""

TASK_GLSL = """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 32) in;
struct Task { uint base; uint count; };
taskPayloadSharedEXT Task payload;
void main() {
    if (gl_LocalInvocationIndex == 0u) {
        payload.base = gl_WorkGroupID.x * gl_WorkGroupSize.x;
        payload.count = 4u;
    }
    barrier();
    EmitMeshTasksEXT(payload.count, 1, 1);
}
"""

TASK_HLSL = """struct Payload { uint base; };
groupshared Payload s_payload;
[numthreads(4, 1, 1)]
void ASMain(uint3 gtid : SV_GroupThreadID) {
    if (gtid.x == 0) {
        s_payload.base = 7;
    }
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(3, 1, 1, s_payload);
}
"""

# GLSL mesh and task identifiers, none of which HLSL or MSL declares.
_GLSL_NAMES = re.compile(
    r"\b(gl_\w+|taskPayloadSharedEXT|EmitMeshTasksEXT|SetMeshOutputsEXT)\b"
)


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    # Unformatted, so the assertions do not depend on clang-format being installed.
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _struct_semantics(code, struct_name):
    body = re.search(rf"struct {struct_name}\s*\{{(.*?)\}};", code, re.S).group(1)
    return dict(re.findall(r"(\w+)\s*:\s*(\w+)\s*;", body))


def _dxc(tmp_path, code, profile, entry):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / f"out_{profile}.hlsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", profile, "-E", entry, str(path), "-Fo", str(tmp_path / "o")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _metal(tmp_path, code, name):
    if shutil.which("xcrun") is None:
        return
    if (
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "--find", "metal"],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    ):
        return
    path = tmp_path / f"{name}.metal"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(path),
            "-o",
            str(tmp_path / f"{name}.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_glsl_mesh_outputs_become_hlsl_output_parameters(tmp_path):
    code = _translate(tmp_path, MESH_GLSL, "shader.mesh", "directx")
    flat = _flat(code)
    assert "[numthreads(32, 1, 1)]" in flat
    assert '[outputtopology("triangle")]' in flat
    assert re.search(
        r"void MSMain\(in payload Task payload, out vertices MeshVertex meshVertices\[64\], "
        r"out primitives MeshPrimitive meshPrimitives\[32\], out indices uint3 meshIndices\[32\], "
        r"uint localInvocationIndex : SV_GroupIndex, uint3 workGroupID : SV_GroupID\)",
        flat,
    )
    assert _struct_semantics(code, "MeshVertex") == {
        "position": "SV_POSITION",
        "normal": "TEXCOORD0",
        "id": "TEXCOORD1",
        "uv": "TEXCOORD3",
    }
    assert "nointerpolation uint id" in flat
    assert _struct_semantics(code, "MeshPrimitive") == {
        "primitiveID": "SV_PrimitiveID",
        "primColor": "TEXCOORD2",
    }
    assert "meshVertices[i].normal = float3(0.0, 0.0, 1.0);" in flat
    assert "meshVertices[(i + 32u)].uv = float2(1.0, 0.0);" in flat
    # HLSL declares SV_PrimitiveID uint, GLSL gl_PrimitiveID int.
    assert re.search(r"meshPrimitives\[i\]\.primitiveID = uint\(.*int\(i\)", flat)
    assert "meshIndices[i] = uint3(i, (i + 32u), ((i + 1u) % 32u));" in flat
    assert not _GLSL_NAMES.search(code)
    _dxc(tmp_path, code, "ms_6_5", "MSMain")


def test_hlsl_mesh_outputs_take_component_writes(tmp_path):
    code = _flat(
        _translate(
            tmp_path,
            _mesh(
                "for (uint k = 0u; k < 3u; k++) { gl_MeshVerticesEXT[k].gl_Position = vec4(0.0); }\n"
                "    color[1].y = 1.0; color[2].xz = vec2(0.5);\n"
                "    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);"
            ),
            "a.mesh",
            "directx",
        )
    )
    assert "meshVertices[1].color.y = 1.0;" in code
    assert "meshVertices[2].color.xz = float2(0.5, 0.5);" in code
    _dxc(tmp_path, code, "ms_6_5", "MSMain")


def test_mesh_varyings_link_with_the_imported_fragment_stage(tmp_path):
    mesh = _translate(tmp_path, MESH_GLSL, "shader.mesh", "directx")
    fragment = _translate(tmp_path, FRAGMENT_GLSL, "shader.frag", "directx")
    assert re.search(r"VertexData vdata\s*:\s*TEXCOORD0;", fragment)
    assert re.search(r"float4 primColor\s*:\s*TEXCOORD2;", fragment)
    assert _struct_semantics(mesh, "MeshVertex")["normal"] == "TEXCOORD0"
    assert _struct_semantics(mesh, "MeshPrimitive")["primColor"] == "TEXCOORD2"
    _dxc(tmp_path, fragment, "ps_6_5", "PSMain")


def test_glsl_mesh_outputs_become_a_metal_mesh_object(tmp_path):
    code = _translate(tmp_path, MESH_GLSL, "shader.mesh", "metal")
    flat = _flat(code)
    assert "[[mesh, max_total_threads_per_threadgroup(32)]]" in flat
    assert "const object_data Task& payload [[payload]]" in flat
    assert "uint localInvocationIndex [[thread_index_in_threadgroup]]" in flat
    assert "mesh<MeshVertex, MeshPrimitive, 64, 32, topology::triangle>" in flat
    assert "float3 normal [[user(locn0)]];" in flat
    assert "uint id [[user(locn1)]] [[flat]];" in flat
    assert "float2 uv [[user(locn3)]];" in flat
    assert "uint primitiveID [[primitive_id]];" in flat
    assert "float4 primColor [[user(locn2)]];" in flat
    assert "set_primitive(i," in flat
    assert not _GLSL_NAMES.search(code)
    _metal(tmp_path, code, "mesh")


def test_glsl_task_payload_becomes_a_dispatch_mesh_argument(tmp_path):
    hlsl = _flat(_translate(tmp_path, TASK_GLSL, "shader.task", "directx"))
    assert "groupshared Task payload;" in hlsl
    assert re.search(
        r"\[numthreads\(32, 1, 1\)\]\s*\[shader\(\"amplification\"\)\]\s*void ASMain\("
        r"uint localInvocationIndex : SV_GroupIndex, uint3 workGroupID : SV_GroupID\)",
        hlsl,
    )
    assert "payload.base = (workGroupID.x * uint3(32, 1, 1).x);" in hlsl
    assert "DispatchMesh(payload.count, 1, 1, payload);" in hlsl
    assert not _GLSL_NAMES.search(hlsl)
    _dxc(tmp_path, hlsl, "as_6_5", "ASMain")

    metal = _flat(_translate(tmp_path, TASK_GLSL, "shader.task", "metal"))
    assert "object_data Task& _crossglMeshPayload [[payload]]" in metal
    assert "_crossglMeshPayload = payload;" in metal
    assert "set_threadgroups_per_grid(uint3(payload.count, 1, 1));" in metal
    assert not _GLSL_NAMES.search(metal)
    _metal(tmp_path, metal, "task")


def test_hlsl_groupshared_task_payload_reaches_glsl_and_spirv(tmp_path):
    glsl = _translate(tmp_path, TASK_HLSL, "shader.hlsl", "opengl")
    assert "taskPayloadSharedEXT Payload s_payload;" in glsl
    assert "EmitMeshTasksEXT(3, 1, 1);" in glsl
    assert "shared Payload" not in glsl.replace("taskPayloadSharedEXT", "")
    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        path = tmp_path / "out.task"
        path.write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            [
                glslang,
                "--target-env",
                "vulkan1.3",
                str(path),
                "-o",
                str(tmp_path / "out.spv"),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    spirv = _translate(tmp_path, TASK_HLSL, "shader.hlsl", "vulkan")
    payload = re.search(r"(%\w+) = OpVariable %\w+ TaskPayloadWorkgroupEXT", spirv)
    assert payload is not None
    assert re.search(rf"OpEmitMeshTasksEXT %\w+ %\w+ %\w+ {payload.group(1)}\b", spirv)


def test_glsl_and_spirv_targets_keep_the_glsl_mesh_form(tmp_path):
    glsl = _translate(tmp_path, MESH_GLSL, "shader.mesh", "opengl")
    assert "gl_MeshVerticesEXT[i].gl_Position" in glsl
    assert "taskPayloadSharedEXT Task payload;" in glsl
    assert "meshVertices" not in glsl


def _mesh(body, layouts=None, declarations=""):
    layouts = layouts or (
        "layout(local_size_x = 1) in;\n"
        "layout(triangles, max_vertices = 3, max_primitives = 1) out;"
    )
    return (
        "#version 460\n#extension GL_EXT_mesh_shader : require\n"
        f"{layouts}\nlayout(location = 0) out vec3 color[];\n{declarations}\n"
        f"void main() {{\n    SetMeshOutputsEXT(3, 1);\n    {body}\n}}\n"
    )


@pytest.mark.parametrize(
    "source, name, reason",
    [
        (
            _mesh("gl_MeshVerticesEXT[0].gl_PointSize = 1.0;"),
            "shader.mesh",
            "output-builtin",
        ),
        (
            _mesh("gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT = true;"),
            "shader.mesh",
            "output-builtin",
        ),
        (_mesh("vec3 copy[3] = color;"), "shader.mesh", "output-array-use"),
        (_mesh("vec3 c = color[0];"), "shader.mesh", "output-read"),
        (_mesh("color[0] += vec3(1.0);"), "shader.mesh", "output-read"),
        (
            _mesh(
                "gl_MeshVerticesEXT[1].gl_Position = gl_MeshVerticesEXT[0].gl_Position;"
            ),
            "shader.mesh",
            "output-read",
        ),
        (
            _mesh("gl_PrimitiveLineIndicesEXT[0] = uvec2(0, 1);"),
            "shader.mesh",
            "index-array",
        ),
        (
            _mesh(
                "paint(0);",
                declarations="void paint(uint k) { color[k] = vec3(1.0); }",
            ),
            "shader.mesh",
            "helper-interface",
        ),
        (
            _mesh("color[0] = vec3(1.0);", layouts="layout(local_size_x = 1) in;"),
            "shader.mesh",
            "output-layout",
        ),
        (
            "#version 460\n#extension GL_EXT_mesh_shader : require\n"
            "layout(local_size_x = 1) in;\n"
            "void main() { EmitMeshTasksEXT(gl_NumWorkGroups.x, 1, 1); }\n",
            "shader.task",
            "num-work-groups",
        ),
    ],
)
def test_constructs_without_an_entry_parameter_form_are_refused(
    tmp_path, source, name, reason
):
    with pytest.raises(GlslMeshStageUnsupportedError) as raised:
        _translate(tmp_path, source, name, "directx")
    assert raised.value.reason_code == reason
    assert raised.value.code == f"{CODE_PREFIX}glsl-mesh-stage-unsupported"
    if reason != "num-work-groups":
        with pytest.raises(GlslMeshStageUnsupportedError):
            _translate(tmp_path, source, name, "metal")


def test_metal_refuses_component_writes_of_mesh_outputs(tmp_path):
    with pytest.raises(GlslMeshStageUnsupportedError) as raised:
        _translate(tmp_path, _mesh("color[1].y = 1.0;"), "a.mesh", "metal")
    assert raised.value.reason_code == "output-component-write"


HLSL_MESH = """struct V { float4 p : SV_Position; float2 uv : TEXCOORD0; float3 c : COLOR0; };
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void MSMain(out vertices V v[3], out indices uint3 t[1]) {
    SetMeshOutputCounts(3, 1);
    v[0].p = float4(0, 0, 0, 1); v[1].p = float4(1, 0, 0, 1); v[2].p = float4(0, 1, 0, 1);
    v[0].uv = float2(0, 0); v[0].c = float3(1, 0, 0);
    t[0] = uint3(0, 1, 2);
}
"""


def test_metal_mesh_vertex_members_link_by_user_keys(tmp_path):
    code = _flat(_translate(tmp_path, HLSL_MESH, "mesh.hlsl", "metal"))
    assert "float2 uv [[user(TEXCOORD0)]];" in code
    assert "float3 c [[user(COLOR0)]];" in code
    assert "[[attribute(" not in code
    _metal(tmp_path, code, "hlsl_mesh")


def test_metal_refuses_component_writes_of_hlsl_mesh_outputs(tmp_path):
    source = HLSL_MESH.replace("v[0].uv = float2(0, 0);", "v[0].uv.x = 1.0;")
    with pytest.raises(MetalLoweringRefused) as raised:
        _translate(tmp_path, source, "mesh.hlsl", "metal")
    assert "component writes of an output member" in str(raised.value)


def test_a_payload_store_before_the_dispatch_is_kept(tmp_path):
    # The GLSL importer used to fold the store into the dispatch, passing the stored value as the payload.
    source = (
        "#version 460\n#extension GL_EXT_mesh_shader : require\n"
        "layout(local_size_x = 1) in;\nstruct Task { uint count; };\n"
        "taskPayloadSharedEXT Task task;\n"
        "void main() { task.count = 3u; EmitMeshTasksEXT(3, 1, 1); }\n"
    )
    crossgl = _flat(_translate(tmp_path, source, "shader.task", "cgl"))
    assert "task.count = 3u; DispatchMesh(3, 1, 1);" in crossgl
    hlsl = _flat(_translate(tmp_path, source, "shader.task", "directx"))
    assert "task.count = 3u; DispatchMesh(3, 1, 1, task);" in hlsl
    _dxc(tmp_path, hlsl, "as_6_5", "ASMain")
    glsl = _flat(_translate(tmp_path, source, "shader.task", "opengl"))
    assert "task.count = 3u; EmitMeshTasksEXT(3, 1, 1);" in glsl
