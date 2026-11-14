"""Mesh outputs in the form the SPIR-V importer writes reach HLSL and Metal entry parameters.

The SPIR-V importer marks a mesh stage's outputs ``@output`` rather than qualifying them ``out``, declares the index
array it writes, and imports a ``Block`` output array as an array of its structure::

    VertexOutput vertexOutput[3] @output @location(0);
    uint3 gl_PrimitiveTriangleIndicesEXT[1] @output @gl_PrimitiveTriangleIndicesEXT;

The HLSL and Metal emitters dropped those declarations without moving the outputs into the vertex structure, so
``vertexOutput[0].color = ...`` referred to an undeclared name. Each member of a structure output now becomes a
member of the vertex structure, at the structure's location plus the locations of the members before it, as for
an arrayed ``out`` block (GLSL 4.60 section 4.4.2.1).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import GlslMeshStageUnsupportedError

MESH = """shader main {
    struct VertexOutput {
        float4 color;
        float2 uv;
    }

    mesh {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(triangles, max_vertices = 3, max_primitives = 1) out;
        VertexOutput vertexOutput[3] @output @location(0);
        float depth[3] @output @location(3);
        uint gl_LocalInvocationIndex @input @gl_LocalInvocationIndex;
        uint3 gl_PrimitiveTriangleIndicesEXT[1] @output @gl_PrimitiveTriangleIndicesEXT;
        void main() {
            SetMeshOutputCounts(3, 1);
            gl_MeshVerticesEXT[0].gl_Position = float4(0.0, -1.0, 0.0, 1.0);
            gl_MeshVerticesEXT[1].gl_Position = float4(-1.0, 1.0, 0.0, 1.0);
            gl_MeshVerticesEXT[2].gl_Position = float4(1.0, 1.0, 0.0, 1.0);
            vertexOutput[0].color = float4(0.0, 1.0, 0.0, 1.0);
            vertexOutput[1].color = float4(0.0, 0.0, 1.0, 1.0);
            vertexOutput[2].color = float4(1.0, 0.0, 0.0, 1.0);
            vertexOutput[0].uv = float2(0.0, 0.0);
            vertexOutput[1].uv = float2(1.0, 0.0);
            vertexOutput[2].uv = float2(0.0, 1.0);
            depth[0] = 0.5;
            depth[1] = 0.25;
            depth[2] = 0.75;
            gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex] = uint3(0, 1, 2);
            return;
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_structure_outputs_become_members_of_the_hlsl_vertex_structure(tmp_path):
    code = _translate(tmp_path, MESH, "directx")
    body = re.search(r"struct MeshVertex\s*\{(.*?)\};", code, re.S).group(1)
    assert dict(re.findall(r"(\w+)\s*:\s*(\w+)\s*;", body)) == {
        "position": "SV_POSITION",
        "color": "TEXCOORD0",
        "uv": "TEXCOORD1",
        "depth": "TEXCOORD3",
    }
    assert "meshVertices[2].color = float4(1.0, 0.0, 0.0, 1.0);" in code
    assert "meshVertices[1].uv = float2(1.0, 0.0);" in code
    assert "meshIndices[localInvocationIndex] = uint3(0, 1, 2);" in code
    assert "vertexOutput" not in code
    assert "gl_PrimitiveTriangleIndicesEXT" not in code
    if shutil.which("dxc"):
        (tmp_path / "mesh.hlsl").write_text(code, encoding="utf-8")
        _run(
            ["dxc", "-T", "ms_6_5", "-E", "MSMain", "mesh.hlsl", "-Fo", "mesh.dxil"],
            tmp_path,
        )


def test_structure_outputs_become_members_of_the_metal_vertex(tmp_path):
    code = _translate(tmp_path, MESH, "metal")
    assert "vertexOutput" not in code
    assert re.search(r"\bcolor\b", code) and re.search(r"\buv\b", code)
    if shutil.which("xcrun") and (
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "--find", "metal"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    ):
        (tmp_path / "mesh.metal").write_text(code, encoding="utf-8")
        _run(
            [
                *("xcrun", "-sdk", "macosx", "metal", "-std=metal3.0", "-c"),
                *("mesh.metal", "-o", "mesh.air"),
            ],
            tmp_path,
        )


@pytest.mark.parametrize("backend", ["directx", "metal"])
def test_a_structure_with_a_per_primitive_member_in_a_vertex_output_is_refused(
    tmp_path, backend
):
    source = MESH.replace("float2 uv;", "float2 uv @perprimitiveEXT;")
    with pytest.raises(GlslMeshStageUnsupportedError, match="per-primitive"):
        _translate(tmp_path, source, backend)
