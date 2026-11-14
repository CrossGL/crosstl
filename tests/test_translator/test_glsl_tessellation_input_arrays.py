"""GLSL output leaves the per-vertex dimension of a tessellation input implicit.

GLSL 4.60 section 4.3.4: a tessellation control or evaluation input that is not ``patch`` is an array indexed by
vertex whose size, if written, must be ``gl_MaxPatchVertices``. SPIR-V and HLSL size it by the patch's control
points, as in DXC's ``float4 in_var_POSITION[3]``, and glslang rejects that size ("tessellation input array size
must be gl_MaxPatchVertices or implicitly sized"). Outputs keep their size, which ``layout(vertices = N) out``
must match, and ``.length()`` still counts the written elements.
"""

from __future__ import annotations

import shutil
import subprocess

import crosstl

EVALUATION = """
shader Probe {{
    tessellation_evaluation {{
        layout(triangles, equal_spacing, ccw) in;
        float3 gl_TessCoord @input @gl_TessCoord @patch;
        {inputs}
        float4 gl_Position @output @gl_Position;
        void main() {{
            {body}
        }}
    }}
}}
"""

CONTROL = """
shader Probe {
    tessellation_control {
        layout(vertices = 3) out;
        float4 in_pos[3] @input @location(0);
        float4 out_pos[3] @output @location(0);
        void main() {
            out_pos[gl_InvocationID] = in_pos[gl_InvocationID];
            if (gl_InvocationID == 0) {
                gl_TessLevelOuter[0] = 4.0;
                gl_TessLevelOuter[1] = 4.0;
                gl_TessLevelOuter[2] = 4.0;
                gl_TessLevelInner[0] = 4.0;
            }
        }
    }
}
"""

SUM = "gl_Position = points[0] * gl_TessCoord.x + points[1] * gl_TessCoord.y;"


def _translate(tmp_path, source, backend="opengl"):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _evaluation(tmp_path, inputs, body=SUM):
    return _translate(tmp_path, EVALUATION.format(inputs=inputs, body=body))


def _glslang(tmp_path, code, suffix):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"checked.{suffix}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_evaluation_input_sized_by_the_patch_is_declared_unsized(tmp_path):
    code = _evaluation(tmp_path, "float4 points[3] @input @location(0);")

    assert "layout(location = 0) in vec4 points[];" in code
    assert "points[3]" not in code
    _glslang(tmp_path, code, "tese")


def test_control_input_is_unsized_and_its_output_keeps_the_patch_size(tmp_path):
    code = _translate(tmp_path, CONTROL)

    assert "layout(location = 0) in vec4 in_pos[];" in code
    assert "layout(location = 0) out vec4 out_pos[3];" in code
    _glslang(tmp_path, code, "tesc")


def test_inner_dimensions_of_a_per_vertex_input_keep_their_sizes(tmp_path):
    code = _evaluation(
        tmp_path,
        "float4 points[3][2] @input @location(0);",
        "gl_Position = points[0][1] * gl_TessCoord.x + points[2][0] * gl_TessCoord.z;",
    )

    assert "layout(location = 0) in vec4 points[][2];" in code
    _glslang(tmp_path, code, "tese")


def test_patch_inputs_keep_their_sizes(tmp_path):
    code = _evaluation(
        tmp_path,
        "float4 points[3] @input @location(0);\n"
        "        float4 weights[2] @input @location(1) @patch;",
        "gl_Position = points[0] * weights[1].x + points[1] * gl_TessCoord.y;",
    )

    assert "layout(location = 1) patch in vec4 weights[2];" in code
    assert "layout(location = 0) in vec4 points[];" in code
    _glslang(tmp_path, code, "tese")


def test_length_of_an_unsized_input_counts_the_written_elements(tmp_path):
    code = _evaluation(
        tmp_path,
        "float4 points[3] @input @location(0);",
        "int count = points.length();\n"
        "            gl_Position = points[0] * float(count) * gl_TessCoord.x;",
    )

    assert "int count = 3;" in code
    assert ".length()" not in code
    _glslang(tmp_path, code, "tese")


def test_length_of_a_local_array_with_an_input_name_is_unchanged(tmp_path):
    code = _evaluation(
        tmp_path,
        "float4 points[3] @input @location(0);",
        "float4 first = points[0];\n"
        "            {\n"
        "                float points[5];\n"
        "                points[0] = float(points.length());\n"
        "                first = first * points[0];\n"
        "            }\n"
        "            gl_Position = first * gl_TessCoord.x;",
    )

    assert "points.length()" in code
    _glslang(tmp_path, code, "tese")


def test_size_spelled_gl_max_patch_vertices_is_kept(tmp_path):
    code = _evaluation(
        tmp_path, "float4 points[gl_MaxPatchVertices] @input @location(0);"
    )

    assert "in vec4 points[gl_MaxPatchVertices];" in code
    _glslang(tmp_path, code, "tese")


def test_geometry_inputs_keep_their_sizes(tmp_path):
    source = """
shader Probe {
    geometry {
        layout(triangles) in;
        layout(points, max_vertices = 1) out;
        float4 points[3] @input @location(0);
        float4 gl_Position @output @gl_Position;
        void main() {
            gl_Position = points[0] + points[1] + points[2];
            EmitVertex();
            EndPrimitive();
        }
    }
}
"""
    code = _translate(tmp_path, source)

    assert "layout(location = 0) in vec4 points[3];" in code
    _glslang(tmp_path, code, "geom")
