"""Geometry stages in the rust-gpu output (:mod:`crosstl.translator.codegen.rust_gpu_geometry`).

CrossGL's GLSL form of a geometry stage (stage layouts, module-scope per-vertex input arrays, outputs written before
``EmitVertex()``) becomes one ``#[spirv(geometry(...))]`` entry point whose parameters are the interface. Each
generated crate compiles with rust-gpu when it is available.
"""

from __future__ import annotations

import re

import pytest

from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_codegen.test_rust_codegen import (
    assert_generated_rust_smoke_compiles,
    generate_code,
    parse_code,
    tokenize_code,
)


def _rust(source):
    return generate_code(parse_code(tokenize_code(source)))


def _geometry(layouts, body, declarations="", helpers=""):
    return f"""
        shader Probe {{
            {declarations}
            geometry {{
                {layouts}
                {helpers}
                void main() {{
                    {body}
                }}
            }}
        }}
        """


@pytest.mark.parametrize(
    "input_layout, count, mode",
    [
        ("points", 1, "input_points"),
        ("lines", 2, "input_lines"),
        ("lines_adjacency", 4, "input_lines_adjacency"),
        ("triangles", 3, "triangles"),
        ("triangles_adjacency", 6, "input_triangles_adjacency"),
    ],
)
def test_input_primitives_become_execution_modes(input_layout, count, mode, tmp_path):
    code = _rust(
        _geometry(
            f"layout({input_layout}) in;\nlayout(points, max_vertices = 1) out;",
            f"gl_Position = inPosition[{count - 1}];\nEmitVertex();\nEndPrimitive();",
            declarations=f"in vec4 inPosition[{count}] @location(0);",
        )
    )
    assert (
        f"#[spirv(geometry({mode}, output_points, output_vertices = 1))]" in code
    ), code
    assert re.search(
        rf"#\[spirv\(location = 0\)\]\s*inPosition\s*:\s*\[Vec4; {count}\]", code
    ), code
    assert "spirv_std::arch::emit_vertex()" in code
    assert "spirv_std::arch::end_primitive()" in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_outputs_invocations_and_varyings(tmp_path):
    code = _rust(
        _geometry(
            "layout(points, invocations = 2) in;\n"
            "layout(triangle_strip, max_vertices = 3) out;",
            "for (int k = 0; k < 3; k++) {\n"
            "    gl_Position = inPosition[0] * float(gl_InvocationID + k);\n"
            "    vColor = inColor[0];\n"
            "    vId = gl_PrimitiveIDIn;\n"
            "    EmitVertex();\n"
            "}\n"
            "EndPrimitive();",
            declarations="in vec4 inPosition[1] @location(0);\n"
            "in vec3 inColor[1] @location(1);\n"
            "out vec3 vColor;\n"
            "flat out int vId;",
        )
    )
    assert (
        "#[spirv(geometry(input_points, output_triangle_strip, output_vertices = 3, invocations = 2))]"
        in code
    ), code
    assert re.search(r"#\[spirv\(invocation_id\)\]\s*gl_InvocationID\s*:\s*i32", code)
    assert re.search(r"#\[spirv\(primitive_id\)\]\s*gl_PrimitiveIDIn\s*:\s*i32", code)
    assert re.search(r"#\[spirv\(location = 0\)\]\s*vColor\s*:\s*&mut Vec3", code)
    assert re.search(r"#\[spirv\(flat, location = 1\)\]\s*vId\s*:\s*&mut i32", code)
    assert "(*vColor) = " in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_helper_writes_outputs_through_the_threaded_reference(tmp_path):
    code = _rust(
        _geometry(
            "layout(points) in;\nlayout(points, max_vertices = 1) out;",
            "emit(inPosition[0]);",
            declarations="in vec4 inPosition[1] @location(0);",
            helpers="void emit(vec4 p) {\n    gl_Position = p;\n    EmitVertex();\n}",
        )
    )
    assert "pub fn emit(p: Vec4, gl_Position: &mut Vec4)" in code, code
    assert "emit(inPosition[0], gl_Position);" in code, code
    assert "(*gl_Position) = p;" in code, code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_stream_calls_take_the_stream_as_a_const_generic(tmp_path):
    code = _rust(
        _geometry(
            "layout(points) in;\nlayout(points, max_vertices = 1) out;",
            "EmitStreamVertex(0);\nEndStreamPrimitive(0);\noutp[0] = 1u;",
            declarations="",
        ).replace("void main()", "buffer uint* outp;\nvoid main()")
    )
    assert "spirv_std::arch::emit_stream_vertex::<0>()" in code
    assert "spirv_std::arch::end_stream_primitive::<0>()" in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


@pytest.mark.parametrize(
    "layouts, declarations, body, message",
    [
        (
            "layout(points) in;",
            "",
            "EmitVertex();",
            "one output primitive and max_vertices",
        ),
        (
            "layout(lines) in;\nlayout(points, max_vertices = 1) out;",
            "in vec4 inPosition[3] @location(0);",
            "gl_Position = inPosition[0];\nEmitVertex();",
            "has 3 elements; the input primitive has 2 vertices",
        ),
        (
            "layout(points) in;\nlayout(points, max_vertices = 1) out;",
            "in vec4 inPosition @location(0);",
            "gl_Position = inPosition;\nEmitVertex();",
            "is not an array of one value per input vertex",
        ),
        (
            "layout(points) in;\nlayout(points, max_vertices = 1) out;",
            "in vec4 inPosition[1] @location(0);\nnoperspective out vec4 vValue;",
            "vValue = inPosition[0];\nEmitVertex();",
            "rust-gpu has no attribute for that interpolation",
        ),
    ],
)
def test_geometry_values_rust_gpu_cannot_declare_are_refused(
    layouts, declarations, body, message
):
    with pytest.raises(RustLoweringRefused, match=message):
        _rust(_geometry(layouts, body, declarations=declarations))


def test_non_constant_stream_is_refused():
    source = _geometry(
        "layout(points) in;\nlayout(points, max_vertices = 1) out;",
        "EmitStreamVertex(k);",
    ).replace("void main()", "uniform int k;\nvoid main()")
    with pytest.raises(RustLoweringRefused, match="needs a constant stream number"):
        _rust(source)


def test_geometry_calls_outside_a_geometry_stage_are_refused():
    source = """
        shader Probe {
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main() {
                    EmitVertex();
                }
            }
        }
        """
    with pytest.raises(RustLoweringRefused, match="'EmitVertex' is a geometry-stage"):
        _rust(source)
