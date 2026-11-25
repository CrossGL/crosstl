"""HLSL geometry entry points whose input primitive is an array of plain values lower to GLSL and SPIR-V.

``void main(line float4 position[2] : POSITION, inout PointStream<V> stream)`` takes each vertex as one ``float4``
rather than as a structure. The geometry lowering (:mod:`crosstl.translator.semantics.geometry_streams`) treated
the primitive itself as the vertex member, saw the array type ``float4[2]`` and refused the stage with "input vertex
member 'position' is an array or structure", so every HLSL geometry coverage probe was refused for OpenGL, GLSL ES
and Vulkan. Each vertex of such a primitive is one element of the array.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.geometry_streams import (
    GeometryStreamUnsupportedError,
)

GLSLANG = shutil.which("glslangValidator")
SPIRV_TOOLS = shutil.which("spirv-as") and shutil.which("spirv-val")

LINES = """struct GeometryOut {
    float4 position : SV_Position;
};

[maxvertexcount(1)]
void main(line float4 position[2] : POSITION, inout PointStream<GeometryOut> stream) {
    GeometryOut output;
    output.position = position[0] + position[1];
    stream.Append(output);
    stream.RestartStrip();
}
"""

INSTANCED_POINTS = """struct GeometryOut {
    float4 position : SV_Position;
};

[instance(2)]
[maxvertexcount(1)]
void main(point float4 position[1] : POSITION, uint id : SV_GSInstanceID,
          inout PointStream<GeometryOut> stream) {
    GeometryOut output;
    output.position = position[0] * float(id + 1);
    stream.Append(output);
    stream.RestartStrip();
}
"""

ADJACENT_COLOURS = """struct GeometryOut {
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[maxvertexcount(3)]
void main(triangleadj float3 color[6] : COLOR0, inout TriangleStream<GeometryOut> stream) {
    for (int k = 0; k < 6; k += 2) {
        GeometryOut output;
        output.position = float4(color[k], 1.0);
        output.color = color[k + 1];
        stream.Append(output);
    }
    stream.RestartStrip();
}
"""

TWO_STREAMS = """struct GeometryOut {
    float4 position : SV_Position;
};

struct ExtraOut {
    float4 extra : TEXCOORD0;
};

[maxvertexcount(2)]
void main(point float4 position[1] : POSITION, inout PointStream<GeometryOut> stream,
          inout PointStream<ExtraOut> extras) {
    GeometryOut output;
    output.position = position[0];
    stream.Append(output);
    ExtraOut extra;
    extra.extra = position[0] * 2.0;
    extras.Append(extra);
}
"""

STRUCT_VERTICES = """struct VertexOut {
    float4 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct GeometryOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[maxvertexcount(1)]
void main(line VertexOut input[2], inout PointStream<GeometryOut> stream) {
    GeometryOut output;
    output.position = input[0].position + input[1].position;
    output.uv = input[1].uv;
    stream.Append(output);
}
"""

NESTED_VALUES = """struct GeometryOut {
    float4 position : SV_Position;
};

[maxvertexcount(1)]
void main(point float2 uv[1][2] : TEXCOORD0, inout PointStream<GeometryOut> stream) {
    GeometryOut output;
    output.position = float4(uv[0][0], uv[0][1]);
    stream.Append(output);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "geometry.hlsl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _glslang(tmp_path, glsl):
    if GLSLANG is None:
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "translated.geom"
    path.write_text(glsl)
    result = subprocess.run(
        [GLSLANG, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr + glsl


def _spirv_val(tmp_path, assembly):
    if not SPIRV_TOOLS:
        pytest.skip("spirv-as and spirv-val are not on PATH")
    asm, binary = tmp_path / "translated.spvasm", tmp_path / "translated.spv"
    asm.write_text(assembly)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr + assembly


def test_a_line_of_positions_reads_one_input_array_element_per_vertex(tmp_path):
    glsl = _translate(tmp_path, LINES, "opengl")
    assert "layout(lines) in;" in glsl
    assert "layout(points, max_vertices = 1) out;" in glsl
    # POSITION is user semantic 0, so the input links with a vertex stage's location 0 output.
    assert re.search(r"layout\(location = 0\) in vec4 (\w+)\[2\];", glsl)
    name = re.search(r"in vec4 (\w+)\[2\];", glsl).group(1)
    assert f"({name}[0] + {name}[1])" in glsl
    assert "EmitVertex();" in glsl and "EndPrimitive();" in glsl
    _glslang(tmp_path, glsl)


def test_an_instanced_point_keeps_its_invocation_count_and_id(tmp_path):
    glsl = _translate(tmp_path, INSTANCED_POINTS, "opengl")
    assert "layout(points, invocations = 2) in;" in glsl
    assert "uint id = uint(gl_InvocationID);" in glsl
    assert re.search(r"in vec4 \w+\[1\];", glsl)
    _glslang(tmp_path, glsl)


def test_an_adjacent_triangle_of_colours_indexes_its_six_vertices(tmp_path):
    glsl = _translate(tmp_path, ADJACENT_COLOURS, "opengl")
    assert "layout(triangles_adjacency) in;" in glsl
    assert "layout(triangle_strip, max_vertices = 3) out;" in glsl
    name = re.search(r"in vec3 (\w+)\[6\];", glsl).group(1)
    assert f"{name}[k]" in glsl and f"{name}[(k + 1)]" in glsl
    _glslang(tmp_path, glsl)


def test_a_point_of_positions_feeds_two_vertex_streams(tmp_path):
    glsl = _translate(tmp_path, TWO_STREAMS, "opengl")
    assert re.search(r"layout\(location = \d+, stream = 1\) out vec4 \w+;", glsl)
    assert "EmitStreamVertex(0);" in glsl and "EmitStreamVertex(1);" in glsl
    _glslang(tmp_path, glsl)


def test_a_primitive_of_structures_still_reads_each_member_array(tmp_path):
    glsl = _translate(tmp_path, STRUCT_VERTICES, "opengl")
    assert re.search(r"layout\(location = 0\) in vec4 \w+\[2\];", glsl)
    assert re.search(r"in vec2 \w+\[2\];", glsl)
    _glslang(tmp_path, glsl)


@pytest.mark.parametrize("source", [LINES, INSTANCED_POINTS, ADJACENT_COLOURS])
def test_vulkan_output_of_a_value_primitive_validates(tmp_path, source):
    assembly = _translate(tmp_path, source, "vulkan")
    assert "OpExecutionMode %main OutputVertices" in assembly or re.search(
        r"OpExecutionMode %\w+ OutputVertices", assembly
    )
    _spirv_val(tmp_path, assembly)


def test_a_primitive_of_arrays_is_still_refused(tmp_path):
    with pytest.raises(GeometryStreamUnsupportedError, match="array or structure"):
        _translate(tmp_path, NESTED_VALUES, "opengl")
