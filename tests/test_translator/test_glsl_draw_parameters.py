"""``gl_BaseVertex``, ``gl_BaseInstance`` and ``gl_DrawID`` for OpenGL and OpenGL ES targets."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import WebGLLoweringRefused

SOURCE = """shader Probe {
    struct VertexInput {
        vec4 position;
        int base @ gl_BaseVertex;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position * 2.0 + vec4(float(input.base), 0.0, 0.0, 0.0);
            return output;
        }
    }
}
"""

PARAMETERS = """shader Probe {
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        float shift(uint draw) {
            return float(draw + 1u);
        }
        VertexOutput main(vec4 position @ POSITION, int base @ gl_BaseVertex, uint draw @ gl_DrawID) {
            VertexOutput output;
            output.position = position + vec4(float(base), shift(draw), 0.0, 0.0);
            return output;
        }
    }
}
"""

HELPER = """shader Probe {
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        float offset() {
            return float(gl_DrawID + gl_BaseVertex);
        }
        VertexOutput main(vec4 position @ POSITION) {
            VertexOutput output;
            output.position = position + vec4(offset(), float(gl_BaseInstance), 0.0, 0.0);
            return output;
        }
    }
}
"""

METAL = """#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float4 position [[attribute(0)]];
};

vertex float4 probe(VertexIn in [[stage_in]], uint base [[base_vertex]], uint first [[base_instance]]) {
    return in.position + float4(float(base), float(first + 1u), 0.0, 0.0);
}
"""

GLSL_460 = """#version 460
layout(location = 0) in vec4 position;
void main() {
    gl_Position = position + vec4(float(gl_BaseVertex), float(gl_BaseInstance), float(gl_DrawID), 0.0);
}
"""

_CORE = re.compile(r"\bgl_(?:BaseVertex|BaseInstance|DrawID)\b")
_EXTENSION = "#extension GL_ARB_shader_draw_parameters : require"


def translate(tmp_path, backend, source=SOURCE, suffix=".cgl"):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate_glsl(tmp_path, glsl):
    if shutil.which("glslangValidator"):
        (tmp_path / "out.vert").write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", "-S", "vert", "out.vert"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_opengl_requires_the_draw_parameters_extension(tmp_path):
    glsl = translate(tmp_path, "opengl")
    assert "#extension GL_ARB_shader_draw_parameters : require" in glsl
    assert "gl_BaseVertexARB" in glsl
    _validate_glsl(tmp_path, glsl)


@pytest.mark.parametrize(
    ("source", "suffix", "names"),
    [
        (PARAMETERS, ".cgl", ("gl_BaseVertexARB", "gl_DrawIDARB")),
        (METAL, ".metal", ("gl_BaseVertexARB", "gl_BaseInstanceARB")),
    ],
    ids=["crossgl-parameters", "metal-attributes"],
)
def test_entry_parameters_read_the_extension_names(tmp_path, source, suffix, names):
    glsl = translate(tmp_path, "opengl", source, suffix)
    assert _EXTENSION in glsl
    assert all(name in glsl for name in names)
    assert not _CORE.search(glsl)
    _validate_glsl(tmp_path, glsl)


def test_helper_reads_share_one_extension_directive(tmp_path):
    glsl = translate(tmp_path, "opengl", HELPER)
    assert glsl.splitlines()[:2] == ["#version 450 core", _EXTENSION]
    assert glsl.count("GL_ARB_shader_draw_parameters") == 1
    assert "float((gl_DrawIDARB + gl_BaseVertexARB))" in glsl
    assert "float(gl_BaseInstanceARB)" in glsl
    _validate_glsl(tmp_path, glsl)


def test_glsl_460_keeps_the_core_names(tmp_path):
    glsl = translate(tmp_path, "opengl", GLSL_460, ".vert")
    assert glsl.startswith("#version 460")
    assert "GL_ARB_shader_draw_parameters" not in glsl
    assert sorted(set(_CORE.findall(glsl))) == [
        "gl_BaseInstance",
        "gl_BaseVertex",
        "gl_DrawID",
    ]
    _validate_glsl(tmp_path, glsl)


def test_glsl_es_has_no_base_vertex(tmp_path):
    with pytest.raises(WebGLLoweringRefused, match="base vertex"):
        translate(tmp_path, "webgl")
