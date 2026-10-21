"""WebGL 2 output: GLSL ES 3.00 varyings linked by name where the source links them by location.

GLSL ES 3.00 has ``layout(location)`` only on vertex inputs and fragment outputs, and links a vertex output to
the fragment input of the same name. Vulkan GLSL and SPIR-V link by location, so the WebGL emitter names those
varyings after their locations, and writes ``#version 300 es`` whatever version the source declared.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import WebGLLoweringRefused

STAGE_LOCAL_VERTEX_CGL = """
shader main {
    vertex {
        vec2 outUV @output @location(0);
        vec2 inUV @input @location(1);
        vec3 outColor @output @location(1);
        vec3 inPos @input @location(0);
        vec4 gl_Position @output @gl_Position;
        void main() {
            outUV = inUV;
            outColor = inPos;
            gl_Position = vec4(inPos, 1.0);
        }
    }
}
"""

STAGE_LOCAL_FRAGMENT_CGL = """
shader main {
    fragment {
        vec3 color @input @location(1);
        vec2 uv @input @location(0);
        vec4 fragColor @output @location(0);
        void main() {
            fragColor = vec4(color * uv.x, 1.0);
        }
    }
}
"""

VULKAN_VERTEX_GLSL = """#version 450
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 0) out vec2 outUV;
layout(location = 1) out vec3 outColor;
void main() {
    outUV = inUV;
    outColor = inPos;
    gl_Position = vec4(inPos, 1.0);
}
"""

VULKAN_FRAGMENT_GLSL = """#version 450
layout(location = 1) in vec3 inColor;
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outFragColor;
void main() {
    outFragColor = vec4(inColor * inUV.x, 1.0);
}
"""


def _translate(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="webgl", format_output=False)


def _declarations(code, direction):
    return {
        name: layout or ""
        for layout, name in re.findall(
            rf"^(layout\([^)]*\) )?{direction} (?:highp |mediump |lowp )?\w+ (\w+);",
            code,
            re.M,
        )
    }


def _link(tmp_path, vertex, fragment):
    if shutil.which("glslangValidator") is None:
        return
    (tmp_path / "linked.vert").write_text(vertex)
    (tmp_path / "linked.frag").write_text(fragment)
    result = subprocess.run(
        [
            "glslangValidator",
            "-l",
            str(tmp_path / "linked.vert"),
            str(tmp_path / "linked.frag"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("vertex_source", "fragment_source", "suffixes"),
    [
        (STAGE_LOCAL_VERTEX_CGL, STAGE_LOCAL_FRAGMENT_CGL, (".cgl", ".cgl")),
        (VULKAN_VERTEX_GLSL, VULKAN_FRAGMENT_GLSL, (".vert", ".frag")),
    ],
    ids=["stage-variables", "stage-structures"],
)
def test_location_linked_varyings_link_by_location_names(
    tmp_path, vertex_source, fragment_source, suffixes
):
    vertex = _translate(tmp_path, vertex_source, f"shader{suffixes[0]}")
    fragment = _translate(tmp_path, fragment_source, f"shader{suffixes[1]}")

    assert vertex.startswith("#version 300 es\n")
    assert fragment.startswith("#version 300 es\n")
    assert _declarations(vertex, "out") == {"cgl_location_0": "", "cgl_location_1": ""}
    assert _declarations(fragment, "in") == {"cgl_location_0": "", "cgl_location_1": ""}
    vertex_inputs = _declarations(vertex, "in")
    assert sorted(vertex_inputs.values()) == [
        "layout(location = 0) ",
        "layout(location = 1) ",
    ]
    assert re.search(r"cgl_location_0 = \w+;", vertex)
    assert re.search(r"cgl_location_1 \* cgl_location_0\.x", fragment)
    _link(tmp_path, vertex, fragment)


def test_varyings_without_locations_keep_their_names(tmp_path):
    vertex = _translate(
        tmp_path,
        "#version 300 es\nin vec3 inPos;\nout vec2 uv;\n"
        "void main() { uv = inPos.xy; gl_Position = vec4(inPos, 1.0); }\n",
        "shader.vert",
    )

    assert "out vec2 uv;" in vertex
    assert "cgl_location" not in vertex


def test_component_qualified_varyings_are_refused(tmp_path):
    source = STAGE_LOCAL_VERTEX_CGL.replace(
        "vec3 outColor @output @location(1);",
        "float outColor @output @location(1) @component(2);",
    ).replace("outColor = inPos;", "outColor = inPos.x;")

    with pytest.raises(WebGLLoweringRefused, match="cannot name part of a location"):
        _translate(tmp_path, source, "shader.cgl")


def test_location_names_the_program_already_uses_are_refused(tmp_path):
    source = STAGE_LOCAL_VERTEX_CGL.replace(
        "void main() {",
        "float cgl_location_1(float x) { return x; }\n        void main() {",
    )

    with pytest.raises(WebGLLoweringRefused, match="already declares"):
        _translate(tmp_path, source, "shader.cgl")
