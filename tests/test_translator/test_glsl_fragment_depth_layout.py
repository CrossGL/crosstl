"""A ``gl_FragDepth`` redeclared with a depth layout keeps its built-in name in GLSL output.

GLSL 4.60 section 4.4.2.3 lets a fragment shader redeclare ``gl_FragDepth`` with ``depth_any``, ``depth_greater``,
``depth_less`` or ``depth_unchanged``. The declaration kept the name while the writes to it were renamed to the
escaped ``crossgl_gl_FragDepth`` whenever another pass had asked for the global's GLSL name first, so glslang
rejected the output with an undeclared identifier.
"""

import shutil
import subprocess

import pytest

import crosstl

CROSSGL = """shader Probe {{
    struct FragmentInput {{
        vec2 uv;
    }}
    out vec4 color @location(0);
    out float gl_FragDepth @{layout};
    fragment {{
        void main(FragmentInput input) {{
            color = vec4(input.uv, 0.0, 1.0);
            gl_FragDepth = input.uv.x * 0.5 + 0.25;
        }}
    }}
}}
"""

GLSL = """#version 460
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(depth_greater) out float gl_FragDepth;
void main() {
    color = vec4(uv, 0.0, 1.0);
    gl_FragDepth = uv.x * 0.5 + 0.25;
}
"""


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _glslang(tmp_path, code):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / "out.frag"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", "-S", "frag", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "layout", ["depth_any", "depth_greater", "depth_less", "depth_unchanged"]
)
def test_crossgl_depth_layout_writes_the_builtin(tmp_path, layout):
    glsl = _translate(tmp_path, "probe.cgl", CROSSGL.format(layout=layout))
    assert f"layout({layout}) out float gl_FragDepth;" in glsl
    assert "gl_FragDepth = ((uv.x * 0.5) + 0.25);" in glsl
    assert "crossgl_gl_FragDepth" not in glsl
    _glslang(tmp_path, glsl)


def test_glsl_depth_layout_round_trip_writes_the_builtin(tmp_path):
    glsl = _translate(tmp_path, "probe.frag", GLSL)
    assert "layout(depth_greater) out float gl_FragDepth;" in glsl
    assert "crossgl_gl_FragDepth" not in glsl
    _glslang(tmp_path, glsl)
