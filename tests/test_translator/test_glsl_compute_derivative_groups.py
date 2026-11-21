"""GLSL output keeps the derivative group a compute stage declares.

``layout(derivative_group_quadsKHR) in;`` makes quad operations read 2x2 blocks of an 8x8 workgroup. GLSL output
used to write the extension the source enabled but drop the layout, so the quads became four consecutive subgroup
invocations and every quad swap read another invocation (#4840).
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused

GLSL_QUADS = """#version 450
#extension GL_KHR_shader_subgroup_quad : require
#extension GL_KHR_compute_shader_derivatives : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(derivative_group_quadsKHR) in;
layout(std430, binding = 0) buffer A { int fin[]; };
layout(std430, binding = 1) buffer B { int outp[]; };
void main() {
    uint gi = gl_LocalInvocationIndex;
    outp[gi] = subgroupQuadSwapVertical(fin[gi]) * 100 + subgroupQuadBroadcast(fin[gi], 3u);
}
"""


def _crossgl(group="derivative_group_quadsKHR", size="8, local_size_y = 8"):
    return f"""shader main {{
    buffer int* fin;
    buffer int* outp;
    compute {{
        layout(local_size_x = {size}, local_size_z = 1) in;
        layout({group}) in;
        void main() {{
            uint gi = gl_LocalInvocationIndex;
            outp[gi] = QuadReadAcrossY(fin[gi]) * 100 + QuadReadLaneAt(fin[gi], 3u);
        }}
    }}
}}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _assert_glslang_accepts(tmp_path, glsl):
    if not shutil.which("glslangValidator"):
        return
    (tmp_path / "out.comp").write_text(glsl)
    for command in (
        ["glslangValidator", "-S", "comp", "out.comp"],
        [
            "glslangValidator",
            "-V",
            "--target-env",
            "vulkan1.1",
            "-S",
            "comp",
            "-o",
            "out.spv",
            "out.comp",
        ],
    ):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr + glsl


def test_glsl_round_trip_keeps_the_quads_layout(tmp_path):
    glsl = _translate(tmp_path, GLSL_QUADS, ".comp")
    assert "layout(derivative_group_quadsKHR) in;" in glsl
    assert glsl.count("#extension GL_KHR_compute_shader_derivatives : require") == 1
    assert "subgroupQuadSwapVertical(" in glsl
    _assert_glslang_accepts(tmp_path, glsl)


@pytest.mark.parametrize(
    ("group", "extension"),
    [
        ("derivative_group_quadsKHR", "GL_KHR_compute_shader_derivatives"),
        ("derivative_group_linearKHR", "GL_KHR_compute_shader_derivatives"),
        ("derivative_group_quadsNV", "GL_NV_compute_shader_derivatives"),
        ("derivative_group_linearNV", "GL_NV_compute_shader_derivatives"),
    ],
)
def test_crossgl_derivative_group_writes_its_extension(tmp_path, group, extension):
    glsl = _translate(tmp_path, _crossgl(group), ".cgl")
    assert f"layout({group}) in;" in glsl
    assert glsl.count(f"#extension {extension} : require") == 1
    layout_line = glsl.index(f"layout({group}) in;")
    assert glsl.index("layout(local_size_x = 8") < layout_line < glsl.index("void main")
    _assert_glslang_accepts(tmp_path, glsl)


def test_derivative_group_raises_the_version_to_450(tmp_path):
    source = GLSL_QUADS.replace("#version 450", "#version 430")
    glsl = _translate(tmp_path, source, ".comp")
    assert glsl.startswith("#version 450")
    assert "layout(derivative_group_quadsKHR) in;" in glsl


def test_stage_without_a_derivative_group_writes_none(tmp_path):
    source = GLSL_QUADS.replace("layout(derivative_group_quadsKHR) in;\n", "").replace(
        "#extension GL_KHR_compute_shader_derivatives : require\n", ""
    )
    glsl = _translate(tmp_path, source, ".comp")
    assert "derivative_group" not in glsl
    assert "compute_shader_derivatives" not in glsl


@pytest.mark.parametrize(
    ("group", "size", "needs"),
    [
        ("derivative_group_quadsKHR", "6, local_size_y = 3", "even local_size_x"),
        ("derivative_group_linearKHR", "6, local_size_y = 1", "multiple of four"),
    ],
)
def test_workgroup_that_cannot_form_the_group_is_refused(tmp_path, group, size, needs):
    with pytest.raises(OpenGLLoweringRefused, match=needs):
        _translate(tmp_path, _crossgl(group, size), ".cgl")
