"""A GLSL evaluation shader states the tessellation modes the program's control stage declares.

GLSL reads the domain, spacing, vertex order and point mode only from the evaluation shader's input layout (GLSL
4.60 section 4.4.1.2). When a CrossGL program states them on its ``tessellation_control`` entry, the evaluation
shader must repeat them; otherwise it silently tessellates with ``equal_spacing`` and ``ccw``. SPIR-V output reads
the same modes from the control stage, so both outputs must agree.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

PROGRAM = """
shader Probe {{
    tessellation_control {{
        layout(vertices = 3) out;
        void hs() @stage_entry {control} {{
            gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
            if (gl_InvocationID == 0) {{
                gl_TessLevelOuter[0] = 4.0;
                gl_TessLevelOuter[1] = 4.0;
                gl_TessLevelOuter[2] = 4.0;
                gl_TessLevelInner[0] = 4.0;
            }}
        }}
    }}
    tessellation_evaluation {{
        {evaluation}
        void ds() @stage_entry {{
            gl_Position = gl_in[0].gl_Position * gl_TessCoord.x + gl_in[1].gl_Position * gl_TessCoord.y;
        }}
    }}
}}
"""

CONTROL = (
    '@domain("tri") @partitioning("fractional_odd") @outputtopology("triangle_cw")'
)

SPIRV_MODES = {
    "triangles": "Triangles",
    "fractional_odd_spacing": "SpacingFractionalOdd",
    "fractional_even_spacing": "SpacingFractionalEven",
    "equal_spacing": "SpacingEqual",
    "cw": "VertexOrderCw",
    "ccw": "VertexOrderCcw",
    "point_mode": "PointMode",
}


def _translate(tmp_path, backend, control, evaluation, entry_point=None):
    path = tmp_path / "tess.cgl"
    path.write_text(PROGRAM.format(control=control, evaluation=evaluation))
    return crosstl.translate(
        str(path), backend=backend, format_output=False, entry_point=entry_point
    )


def _evaluation_layout(code):
    match = re.search(
        r"layout\(([^)]*)\) in;", code.split("GL_TESS_EVALUATION_SHADER")[-1]
    )
    assert match, code
    return [part.strip() for part in match.group(1).split(",")]


def _spirv_evaluation_modes(code):
    entry = re.search(r"OpEntryPoint TessellationEvaluation (%\w+)", code)
    assert entry, code
    return {*re.findall(rf"OpExecutionMode {entry.group(1)} (\w+)", code)}


def _glslang(tmp_path, code):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / "checked.tese"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "control, evaluation, expected",
    [
        (
            CONTROL,
            "layout(triangles) in;",
            ["triangles", "fractional_odd_spacing", "cw"],
        ),
        (CONTROL, "", ["triangles", "fractional_odd_spacing", "cw"]),
        (
            CONTROL,
            "layout(triangles, fractional_odd_spacing) in;",
            ["triangles", "fractional_odd_spacing", "cw"],
        ),
        (
            '@domain("tri") @partitioning("fractional_even") @outputtopology("point")',
            "layout(triangles) in;",
            ["triangles", "fractional_even_spacing", "point_mode"],
        ),
        ("", "layout(triangles) in;", ["triangles"]),
    ],
)
def test_evaluation_layout_states_the_control_stage_modes(
    tmp_path, control, evaluation, expected
):
    combined = _translate(tmp_path, "opengl", control, evaluation)
    stage = _translate(tmp_path, "opengl", control, evaluation, entry_point="ds")

    assert _evaluation_layout(combined) == expected
    assert _evaluation_layout(stage) == expected
    _glslang(tmp_path, stage)


@pytest.mark.parametrize(
    "control",
    [
        CONTROL,
        '@domain("tri") @partitioning("fractional_even") @outputtopology("point")',
    ],
)
def test_glsl_and_spirv_evaluation_stages_use_the_same_modes(tmp_path, control):
    glsl = _translate(tmp_path, "opengl", control, "layout(triangles) in;")
    spirv = _translate(tmp_path, "vulkan", control, "layout(triangles) in;")

    glsl_modes = {SPIRV_MODES[part] for part in _evaluation_layout(glsl)}
    # GLSL's defaults when the layout omits them (GLSL 4.60 section 4.4.1.2).
    if not glsl_modes & {
        "SpacingEqual",
        "SpacingFractionalEven",
        "SpacingFractionalOdd",
    }:
        glsl_modes.add("SpacingEqual")
    if not glsl_modes & {"VertexOrderCw", "VertexOrderCcw"}:
        glsl_modes.add("VertexOrderCcw")
    assert glsl_modes == _spirv_evaluation_modes(spirv)


@pytest.mark.parametrize(
    "evaluation, mode",
    [
        ("layout(triangles, equal_spacing) in;", "partitioning"),
        ("layout(triangles, ccw) in;", "winding"),
        ("layout(quads) in;", "domain"),
    ],
)
def test_conflicting_stage_modes_are_refused(tmp_path, evaluation, mode):
    with pytest.raises(Exception, match=f"Conflicting GLSL tessellation {mode}"):
        _translate(tmp_path, "opengl", CONTROL, evaluation)


EVALUATION_ONLY = """
shader Probe {{
    tessellation_evaluation {{
        void ds() @stage_entry {attributes} {{
            gl_Position = gl_in[0].gl_Position * gl_TessCoord.x + gl_in[1].gl_Position * gl_TessCoord.y;
        }}
    }}
}}
"""


def _translate_evaluation(tmp_path, backend, attributes):
    path = tmp_path / "evaluation.cgl"
    path.write_text(EVALUATION_ONLY.format(attributes=attributes))
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    "attributes, missing, unstated",
    [
        (
            '@domain("tri")',
            "spacing (@partitioning) and vertex order",
            {"SpacingEqual", "VertexOrderCcw"},
        ),
        (
            '@domain("quad") @partitioning("integer")',
            "vertex order (@outputtopology)",
            {"VertexOrderCcw"},
        ),
        (
            '@domain("tri") @outputtopology("triangle_cw")',
            "spacing (@partitioning) to",
            {"SpacingEqual"},
        ),
    ],
)
def test_hlsl_style_modes_no_stage_states_are_refused(
    tmp_path, attributes, missing, unstated
):
    # SPIR-V output leaves these modes to a control stage, as DXC does for a domain shader.
    spirv = _translate_evaluation(tmp_path, "vulkan", attributes)
    assert not _spirv_evaluation_modes(spirv) & unstated

    with pytest.raises(Exception, match=re.escape(f"leaves its {missing}")):
        _translate_evaluation(tmp_path, "opengl", attributes)


@pytest.mark.parametrize(
    "attributes, expected",
    [
        (
            '@domain("tri") @partitioning("integer") @outputtopology("triangle_cw")',
            ["triangles", "equal_spacing", "cw"],
        ),
        (
            '@domain("isoline") @partitioning("fractional_odd")',
            ["isolines", "fractional_odd_spacing"],
        ),
        (
            '@domain("quad") @partitioning("fractional_even") @outputtopology("point")',
            ["quads", "fractional_even_spacing", "point_mode"],
        ),
    ],
)
def test_hlsl_style_evaluation_stating_every_mode_translates(
    tmp_path, attributes, expected
):
    code = _translate_evaluation(tmp_path, "opengl", attributes)

    assert _evaluation_layout(code) == expected
    _glslang(tmp_path, code)


def test_hlsl_style_evaluation_takes_modes_from_the_control_stage(tmp_path):
    path = tmp_path / "tess.cgl"
    source = PROGRAM.format(control=CONTROL, evaluation="").replace(
        "void ds() @stage_entry", 'void ds() @stage_entry @domain("tri")'
    )
    path.write_text(source)

    for entry_point in (None, "ds"):
        code = crosstl.translate(
            str(path), backend="opengl", format_output=False, entry_point=entry_point
        )
        assert _evaluation_layout(code) == ["triangles", "fractional_odd_spacing", "cw"]
