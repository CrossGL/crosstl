"""A conservative depth layout becomes a ``gl_FragDepth`` redeclaration in GLSL output.

GLSL 4.60 section 4.4.2.3 lets a fragment shader redeclare ``gl_FragDepth`` with ``depth_any``, ``depth_greater``,
``depth_less`` or ``depth_unchanged``, and allows a depth layout nowhere else. The declaration kept the name while
the writes to it were renamed to the escaped ``crossgl_gl_FragDepth`` whenever another pass had asked for the
global's GLSL name first, so glslang rejected the output with an undeclared identifier.

CrossGL also puts the layout on a stage-level ``@output @gl_FragDepth`` variable (the SPIR-V importer's form), on a
member of the structure a fragment entry returns (the HLSL importer's form), on an entry's ``out`` parameter and on
its return value. GLSL output wrote the stage-level form as an ordinary output carrying the layout, which glslang
rejects, and dropped the layout of the others, so the SPIR-V glslang compiles lost the depth execution mode. GLSL ES
output declared the layout without ``GL_EXT_conservative_depth``, which GLSL ES 3.00 needs.
"""

import re
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


def _glslang(tmp_path, code, execution_mode=None):
    """Compile ``code`` with glslang; with ``execution_mode``, check the SPIR-V module carries it."""
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / "out.frag"
    path.write_text(code)
    command = ["glslangValidator", "-S", "frag", str(path)]
    if execution_mode is not None:
        command[1:1] = ["-G", "-H", "-o", str(tmp_path / "out.spv")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    if execution_mode is not None:
        assert re.search(
            rf"ExecutionMode \d+ {execution_mode}\b", result.stdout
        ), result.stdout


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


_MODES = {
    "depth_greater": "DepthGreater",
    "depth_less": "DepthLess",
    "depth_unchanged": "DepthUnchanged",
}

STAGE_SCOPE = """shader Probe {{
    fragment {{
        vec4 color @output @location(0);
        vec2 uv @input @location(0);
        float depth @output @gl_FragDepth @{layout};
        void main() {{
            color = vec4(uv, 0.0, 1.0);
            depth = uv.x * 0.5 + 0.25;
        }}
    }}
}}
"""

STRUCT_MEMBER = """shader Probe {{
    struct FragmentInput {{
        vec2 uv @ location(0);
    }}
    struct FragmentOutput {{
        vec4 color @ location(0);
        float depth @ gl_FragDepth @ {layout};
    }}
    fragment {{
        FragmentOutput main(FragmentInput input) {{
            FragmentOutput output;
            output.color = vec4(input.uv, 0.0, 1.0);
            output.depth = input.uv.x * 0.5 + 0.25;
            return output;
        }}
    }}
}}
"""

OUT_PARAMETER = """shader Probe {{
    fragment {{
        void main(vec2 uv @location(0), out vec4 color @location(0), out float depth @gl_FragDepth @{layout}) {{
            color = vec4(uv, 0.0, 1.0);
            depth = uv.x * 0.5 + 0.25;
        }}
    }}
}}
"""

RETURN_VALUE = """shader Probe {{
    fragment {{
        float main(vec2 uv @location(0)) @gl_FragDepth @{layout} {{
            return uv.x * 0.5 + 0.25;
        }}
    }}
}}
"""

HLSL = """struct PSOut {
    float4 color : SV_Target;
    float depth : SV_DepthGreaterEqual;
};
PSOut main(float2 uv : TEXCOORD0) {
    PSOut o;
    o.color = float4(uv, 0.0, 1.0);
    o.depth = uv.x * 0.5 + 0.25;
    return o;
}
"""


def _webgl(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="webgl", format_output=False)


@pytest.mark.parametrize(
    "form",
    [STAGE_SCOPE, STRUCT_MEMBER, OUT_PARAMETER, RETURN_VALUE],
    ids=["stage-scope", "struct-member", "out-parameter", "return-value"],
)
@pytest.mark.parametrize("layout", ["depth_greater", "depth_less", "depth_unchanged"])
def test_every_crossgl_depth_output_form_redeclares_the_builtin(tmp_path, form, layout):
    glsl = _translate(tmp_path, "probe.cgl", form.format(layout=layout))
    assert glsl.count(f"layout({layout}) out float gl_FragDepth;") == 1
    assert "gl_FragDepth = ((" in glsl
    assert "out float depth" not in glsl
    _glslang(tmp_path, glsl, _MODES[layout])


@pytest.mark.parametrize(
    "form",
    [STAGE_SCOPE, STRUCT_MEMBER],
    ids=["stage-scope", "struct-member"],
)
def test_webgl_output_requires_the_es_conservative_depth_extension(tmp_path, form):
    glsl = _webgl(tmp_path, form.format(layout="depth_less"))
    assert glsl.startswith(
        "#version 300 es\n#extension GL_EXT_conservative_depth : require\n"
    )
    assert "layout(depth_less) out float gl_FragDepth;" in glsl
    _glslang(tmp_path, glsl)


def test_hlsl_depth_greater_equal_keeps_its_layout_in_glsl(tmp_path):
    glsl = _translate(tmp_path, "probe.hlsl", HLSL)
    assert "layout(depth_greater) out float gl_FragDepth;" in glsl
    _glslang(tmp_path, glsl, "DepthGreater")


def test_desktop_glsl_rises_to_420_for_a_depth_layout(tmp_path):
    source = GLSL.replace("#version 460", "#version 330 core")
    glsl = _translate(tmp_path, "probe.frag", source)
    assert glsl.startswith("#version 420 core\n")
    _glslang(tmp_path, glsl)


@pytest.mark.parametrize(
    "source, message",
    [
        (
            """shader Probe {
    fragment {
        vec4 color @output @location(0) @depth_less;
        void main() { color = vec4(1.0); }
    }
}
""",
            "depth layout 'depth_less' to 'color'",
        ),
        (
            """shader Probe {
    struct FragmentOutput {
        vec4 color @ location(0) @ depth_less;
    }
    fragment {
        FragmentOutput main() {
            FragmentOutput o;
            o.color = vec4(1.0);
            return o;
        }
    }
}
""",
            "depth layout 'depth_less' to 'color'",
        ),
        (
            """shader Probe {
    fragment {
        vec4 color @output @location(0);
        float depth @output @gl_FragDepth @depth_less @depth_greater;
        void main() { color = vec4(1.0); depth = 0.5; }
    }
}
""",
            "depth layouts depth_greater, depth_less at once",
        ),
        (
            """shader Probe {
    struct VOut {
        vec4 pos @ gl_Position @ depth_less;
    }
    vertex {
        VOut main(vec3 p @location(0)) {
            VOut o;
            o.pos = vec4(p, 1.0);
            return o;
        }
    }
}
""",
            "of the vertex stage",
        ),
    ],
    ids=["ordinary-output", "ordinary-member", "two-layouts", "vertex-output"],
)
def test_a_depth_layout_off_gl_fragdepth_is_refused(tmp_path, source, message):
    for backend in ("opengl", "webgl"):
        path = tmp_path / "probe.cgl"
        path.write_text(source)
        with pytest.raises(Exception, match="conservative depth") as error:
            crosstl.translate(str(path), backend=backend)
        assert message in str(error.value)
        assert "GLSL 4.60 section 4.4.2.3" in str(error.value)
