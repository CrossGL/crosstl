"""Integer vertex outputs are declared ``flat`` when they are stage-level variables.

An integer cannot be interpolated. GLSL ES 3.00 section 4.3.6 requires a vertex output that is, or contains, an
integer to be ``flat``, as the fragment input it feeds must be (GLSL 4.60 section 4.5). The GLSL and WebGL emitters
wrote ``flat`` on such outputs when they are members of a stage output struct, but not when they are stage-level
``@output`` variables, the form the SPIR-V importer writes, so glslang rejected the WebGL vertex shader ("must be
qualified as flat out").
"""

import shutil
import subprocess

import pytest

import crosstl

VERTEX = """
shader main {
    vertex {
        vec3 inPos @input @location(0);
        int outInstance @output @location(0);
        uvec2 outCascade @output @location(1);
        vec2 outUV @output @location(2);
        vec4 gl_Position @output @gl_Position;
        void main() {
            outInstance = gl_InstanceID;
            outCascade = uvec2(uint(gl_VertexID), 1u);
            outUV = inPos.xy;
            gl_Position = vec4(inPos, 1.0);
        }
    }
}
"""

FRAGMENT = """
shader main {
    fragment {
        int instance @input @location(0);
        uvec2 cascade @input @location(1);
        vec2 uv @input @location(2);
        vec4 color @output @location(0);
        void main() {
            color = vec4(uv, float(instance), float(cascade.x + cascade.y));
        }
    }
}
"""


def _translate(tmp_path, source, backend, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _outputs(code, direction):
    return [
        line.strip()
        for line in code.splitlines()
        if f"{direction} " in line
        and line.strip().endswith(";")
        and "(" not in line.split(direction)[-1]
    ]


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (
            "webgl",
            {
                "flat out int cgl_location_0;",
                "flat out uvec2 cgl_location_1;",
                "out vec2 cgl_location_2;",
            },
        ),
        (
            "opengl",
            {
                "layout(location = 0) flat out int outInstance;",
                "layout(location = 1) flat out uvec2 outCascade;",
                "layout(location = 2) out vec2 outUV;",
            },
        ),
    ],
)
def test_integer_vertex_outputs_are_flat(tmp_path, backend, expected):
    code = _translate(tmp_path, VERTEX, backend, "shader.cgl")
    declared = set(_outputs(code, "out"))
    assert expected <= declared
    assert not any(
        "flat" in line and "vec2" in line and "uvec2" not in line for line in declared
    )


@pytest.mark.parametrize("backend", ["webgl", "opengl"])
def test_flat_integer_varyings_compile_and_link(tmp_path, backend):
    vertex = _translate(tmp_path, VERTEX, backend, "vertex.cgl")
    fragment = _translate(tmp_path, FRAGMENT, backend, "fragment.cgl")
    assert "flat in int" in fragment
    if shutil.which("glslangValidator") is None:
        return
    (tmp_path / "out.vert").write_text(vertex)
    (tmp_path / "out.frag").write_text(fragment)
    _run(["glslangValidator", str(tmp_path / "out.vert")])
    _run(["glslangValidator", str(tmp_path / "out.frag")])
    _run(
        [
            "glslangValidator",
            "-l",
            str(tmp_path / "out.vert"),
            str(tmp_path / "out.frag"),
        ]
    )


def test_explicitly_flat_float_output_stays_flat_once(tmp_path):
    source = VERTEX.replace(
        "vec2 outUV @output @location(2);", "vec2 outUV @output @location(2) @flat;"
    )
    code = _translate(tmp_path, source, "webgl", "shader.cgl")
    line = next(
        line
        for line in code.splitlines()
        if "cgl_location_2" in line and "out " in line
    )
    assert line.split().count("flat") == 1
