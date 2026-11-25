"""GLSL- and SPIR-V-form tessellation evaluation stages translate to Metal post-tessellation vertex functions.

The GLSL and SPIR-V importers write an evaluation stage as GLSL declares it: per-vertex input arrays, ``patch in``
variables, ``gl_TessCoord`` and ``gl_in[]``, read by a ``void main()`` that writes its outputs. The Metal generator
refused that form; it now rewrites it into the patch form a ``[[patch]]`` vertex function takes (MSL 4.1 section
5.1.5). GLSL leaves the patch's control-point count to the control stage, so without one the function's ``[[patch]]``
attribute gives no count.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import AssignmentNode, LiteralNode
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_glsl_tessellation import (
    rewrite_glsl_evaluation_entry,
)
from crosstl.translator.codegen.metal_post_tessellation import (
    MetalPostTessellationError,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

TRIANGLES = """#version 460
layout(triangles, fractional_odd_spacing, cw) in;
layout(location = 0) in vec4 position[];
layout(location = 2) in vec2 uv[];
layout(location = 1) patch in vec4 extra;
layout(location = 0) flat out int patchIndex;
layout(location = 1) out vec2 vuv;
void main() {
    vuv = uv[0] * gl_TessCoord.x + uv[1] * gl_TessCoord.y + uv[2] * gl_TessCoord.z;
    patchIndex = gl_PrimitiveID;
    gl_Position = position[0] * gl_TessCoord.x + position[1] * gl_TessCoord.y
        + position[2] * gl_TessCoord.z + extra;
}
"""

QUADS = """#version 460
layout(quads, equal_spacing, ccw) in;
layout(location = 0) in vec4 position[];
void main() {
    if (gl_TessCoord.z != 0.0) {
        gl_Position = vec4(0.0);
        return;
    }
    gl_Position = mix(mix(position[0], position[1], gl_TessCoord.x),
                      mix(position[3], position[2], gl_TessCoord.x), gl_TessCoord.y);
}
"""

GL_IN = """#version 450
layout(triangles, equal_spacing, cw) in;
void main() {
    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position
        + gl_TessCoord.z * gl_in[2].gl_Position;
}
"""

# The SPIR-V importer's form: stage-level @input and @output variables, the gl_PerVertex outputs included.
SPIRV_FORM = """
shader main {
    tessellation_evaluation {
        layout(triangles, equal_spacing, ccw) in;
        float4 gl_Position @output @gl_Position;
        float gl_PointSize @output @gl_PointSize;
        float gl_CullDistance[1] @output @gl_CullDistance;
        float4 position[32] @input @location(0);
        float3 coord @input @gl_TessCoord;
        float4 extra @input @location(1) @patch;
        void main() {
            gl_Position = position[0] * coord[0] + extra * coord[1];
            return;
        }
    }
}
"""

WITH_CONTROL_STAGE = """
shader main {
    tessellation_control {
        layout(vertices = 3) out;
        float4 point[32] @output @location(0);
        void main() { }
    }
    tessellation_evaluation {
        layout(triangles) in;
        float4 gl_Position @output @gl_Position;
        float4 position[32] @input @location(0);
        void main() { gl_Position = position[gl_PatchVerticesIn - 1]; }
    }
}
"""


def parse(code):
    return Parser(Lexer(code).get_tokens()).parse()


def compile_metal(tmp_path, source):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "patch.metal").write_text(source)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "patch.metal", "-o", "patch.air"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr


def translate(tmp_path, source, name="probe.tese"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal")


def test_triangles_become_a_patch_function_of_the_draws_patch_size(tmp_path):
    code = translate(tmp_path, TRIANGLES)

    # GLSL gives no control-point count without a control stage, and Metal then takes the draw's.
    assert "[[patch(triangle)]]\nvertex " in code
    assert "patch_control_point<" in code
    assert "float4 position [[attribute(0)]];" in code
    assert "float2 uv [[attribute(2)]];" in code
    assert "float4 extra [[attribute(1)]];" in code
    assert "float3 _crossglTessCoord [[position_in_patch]]" in code
    assert "uint _crossglPatchId [[patch_id]]" in code
    assert "int _crossglPrimitiveId = int(_crossglPatchId);" in code
    assert "min(uint(_crossglPatch.points.size()), 32u)" in code
    assert "[[flat]]" in code and "[[user(locn0)]]" in code
    assert "float4 gl_Position [[position]];" in code
    # The spacing and winding are the render pipeline's.
    assert "spacing" not in code and "cw" not in code.replace("ccw", "")
    compile_metal(tmp_path, code)


def test_quads_read_a_two_component_position_in_patch(tmp_path):
    code = translate(tmp_path, QUADS)

    assert "[[patch(quad)]]" in code
    assert "float2 _crossglTessCoord [[position_in_patch]]" in code
    # GLSL's gl_TessCoord is a vec3 whose third component is 0 in a quad.
    assert "float3 _crossglTessCoord3 = float3(_crossglTessCoord, 0.0);" in code
    assert code.count("return _crossglOutput;") == 2
    compile_metal(tmp_path, code)


def test_gl_in_positions_become_a_control_point_attribute(tmp_path):
    code = translate(tmp_path, GL_IN)

    assert "float4 gl_Position [[attribute(0)]];" in code
    assert "_crossglPoints[1].gl_Position" in code
    compile_metal(tmp_path, code)


def test_spirv_form_drops_the_unused_per_vertex_outputs(tmp_path):
    code = MetalCodeGen().generate(parse(SPIRV_FORM))

    assert "[[patch(triangle)]]" in code
    assert "float4 extra [[attribute(1)]];" in code
    assert "PointSize" not in code and "CullDistance" not in code
    assert "return _crossglOutput;" in code
    compile_metal(tmp_path, code)


def test_a_control_stage_fixes_the_patch_size():
    ast = parse(WITH_CONTROL_STAGE)
    stage = next(
        stage
        for stage_type, stage in ast.stages.items()
        if "evaluation" in str(stage_type).lower()
    )
    structs = rewrite_glsl_evaluation_entry(ast, stage, stage.entry_point)

    points = next(
        parameter
        for parameter in stage.entry_point.parameters
        if getattr(parameter.param_type, "name", None) == "OutputPatch"
    )
    assert points.param_type.generic_args[1].value == 3
    assert not stage.entry_point.get_annotation(
        "metal_post_tessellation_open_patch", False
    )
    assert [struct.name for struct in structs] == [
        "main_TessellationPoint",
        "main_TessellationOutput",
    ]
    # gl_PatchVerticesIn is the control stage's count.
    assignment = next(
        statement.expression
        for statement in stage.entry_point.body.statements
        if isinstance(getattr(statement, "expression", None), AssignmentNode)
    )
    index = assignment.value.object_expr.index_expr
    assert isinstance(index.left, LiteralNode) and index.left.value == 3


@pytest.mark.parametrize(
    ("source", "phrase"),
    [
        (GL_IN.replace("triangles,", "isolines,"), "isolines"),
        (GL_IN.replace("cw) in;", "cw, point_mode) in;"), "point_mode"),
        (
            GL_IN.replace("gl_TessCoord.x * gl_in", "gl_TessLevelOuter[0] * gl_in"),
            "host-bound buffer",
        ),
        (
            GL_IN.replace(
                "gl_TessCoord.x * gl_in", "float(gl_PatchVerticesIn) * gl_in"
            ),
            "no tessellation control stage",
        ),
        (
            TRIANGLES.replace(
                "void main() {", "vec4 first() { return position[0]; }\nvoid main() {"
            ),
            "helper 'first'",
        ),
        (
            TRIANGLES.replace(
                "patchIndex = gl_PrimitiveID;", "patchIndex = position.length();"
            ),
            "other than through a subscript",
        ),
    ],
)
def test_forms_metal_cannot_write_are_refused(tmp_path, source, phrase):
    with pytest.raises(MetalPostTessellationError, match=phrase):
        translate(tmp_path, source)
