"""GLSL and WebGL output read the multiview view index ``gl_ViewIndex`` in the stages that declare it (#3578).

A fragment stage reads it through ``CROSSGL_VIEW_INDEX``: ``gl_ViewIndex`` under GL_EXT_multiview for Vulkan and
OpenGL SPIR-V, ``int(gl_ViewID_OVR)`` under GL_OVR_multiview2 for OpenGL and WebGL 2 drivers. Geometry and
tessellation stages keep ``gl_ViewIndex`` under GL_EXT_multiview. A vertex stage is refused: GL_OVR_multiview needs
its view count there, which the source does not state.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_refusals import (
    OpenGLLoweringRefused,
    WebGLLoweringRefused,
)
from crosstl.translator.codegen.glsl_view_index import lower_view_index

FRAGMENT = """
shader Probe {
    struct FragmentInput {
        vec2 uv;
        int view @ gl_ViewIndex;
    }
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, float(input.view), 1.0);
            return output;
        }
    }
}
"""

BARE_FRAGMENT = """
shader Probe {
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        FragmentOutput main() {
            FragmentOutput output;
            output.color = vec4(float(gl_ViewIndex));
            return output;
        }
    }
}
"""

VERTEX_AND_FRAGMENT = """
shader Probe {
    struct VertexInput {
        vec2 position;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
        vec2 uv;
    }
    struct FragmentOutput {
        vec4 color;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = vec4(input.position, 0.0, 1.0);
            output.uv = input.position;
            return output;
        }
    }
    fragment {
        FragmentOutput main(VertexOutput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, float(gl_ViewIndex), 1.0);
            return output;
        }
    }
}
"""

VERTEX = """
shader Probe {
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main() {
            VertexOutput output;
            output.position = vec4(float(gl_ViewIndex), 0.0, 0.0, 1.0);
            return output;
        }
    }
}
"""

GEOMETRY = """
shader Probe {
    geometry {
        layout(points) in;
        layout(points, max_vertices = 1) out;
        float4 gl_Position @output @gl_Position;
        float outValue @output @location(0);
        void main() {
            outValue = float(gl_ViewIndex);
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();
        }
    }
}
"""

HLSL_FRAGMENT = """
float4 main(float2 uv : TEXCOORD0, uint view : SV_ViewID) : SV_Target {
    return float4(uv, float(view), 1.0);
}
"""

PRELUDE = (
    "#if defined(GL_SPIRV) || defined(VULKAN)\n"
    "#extension GL_EXT_multiview : require\n"
    "#define CROSSGL_VIEW_INDEX gl_ViewIndex\n"
    "#else\n"
    "#extension GL_OVR_multiview2 : require\n"
    "#define CROSSGL_VIEW_INDEX int(gl_ViewID_OVR)\n"
    "#endif\n"
)


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _glslang(tmp_path, code, stage, *arguments):
    """Return glslang's log when it rejects ``code``, or ``""``."""
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / f"out.{stage}"
    path.write_text(code, encoding="utf-8")
    command = ["glslangValidator", *arguments, str(path)]
    if arguments:
        command[1:1] = ["-o", str(tmp_path / "out.spv")]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return "" if result.returncode == 0 else result.stdout + result.stderr


@pytest.mark.parametrize("source", [FRAGMENT, BARE_FRAGMENT])
def test_fragment_reads_the_view_index_through_a_profile_macro(tmp_path, source):
    code = _translate(tmp_path, source, "opengl")

    assert code.startswith("#version 450 core\n" + PRELUDE)
    assert "float(CROSSGL_VIEW_INDEX)" in code
    assert code.count("gl_ViewIndex") == 1
    for arguments in ((), ("-G",), ("-V",)):
        assert _glslang(tmp_path, code, "frag", *arguments) == ""


def test_fragment_view_index_is_the_spirv_view_index(tmp_path):
    if shutil.which("spirv-dis") is None:
        pytest.skip("spirv-dis is not installed")
    code = _translate(tmp_path, FRAGMENT, "opengl")
    for arguments in (("-G",), ("-V",)):
        assert _glslang(tmp_path, code, "frag", *arguments) == ""
        disassembly = subprocess.run(
            ["spirv-dis", str(tmp_path / "out.spv")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "BuiltIn ViewIndex" in disassembly
        assert "OpCapability MultiView" in disassembly


def test_webgl_fragment_reads_gl_view_id_ovr(tmp_path):
    code = _translate(tmp_path, FRAGMENT, "webgl")

    assert code.startswith("#version 300 es\n#extension GL_OVR_multiview2 : require\n")
    assert "float(int(gl_ViewID_OVR))" in code
    assert "gl_ViewIndex" not in code
    assert _glslang(tmp_path, code, "frag") == ""


def test_multi_stage_program_with_a_fragment_reader(tmp_path):
    code = _translate(tmp_path, VERTEX_AND_FRAGMENT, "opengl")

    assert PRELUDE in code
    assert "float(CROSSGL_VIEW_INDEX)" in code


@pytest.mark.parametrize(
    ("backend", "error"),
    [("opengl", OpenGLLoweringRefused), ("webgl", WebGLLoweringRefused)],
)
def test_vertex_reader_is_refused(tmp_path, backend, error):
    with pytest.raises(error, match="layout\\(num_views\\)"):
        _translate(tmp_path, VERTEX, backend)


def test_geometry_reader_keeps_gl_view_index(tmp_path):
    code = _translate(tmp_path, GEOMETRY, "opengl")

    assert "#extension GL_EXT_multiview : require\n" in code
    assert "CROSSGL_VIEW_INDEX" not in code
    assert "float(gl_ViewIndex)" in code
    for arguments in ((), ("-G",), ("-V",)):
        assert _glslang(tmp_path, code, "geom", *arguments) == ""


class _Codegen:
    """The parts of the GLSL emitter :func:`lower_view_index` reads."""

    GLSL_STAGE_GUARD_MACROS = GLSLCodeGen.GLSL_STAGE_GUARD_MACROS
    GLSL_TARGET_DISPLAY_NAME = "OpenGL"

    def __init__(self, stages):
        self.stages = stages
        self.refusals = []

    def glsl_stage_names(self, ast, target_stage=None):
        return set(self.stages)

    def glsl_stage_guard_macro(self, stage):
        return self.GLSL_STAGE_GUARD_MACROS.get(stage)

    def refuse_lowering(self, reason):
        self.refusals.append(reason)


GUARDED = """#version 460
#ifdef GL_VERTEX_SHADER
void main() { gl_Position = vec4(1.0); }
#endif
#ifdef GL_TASK_SHADER_EXT
#if defined(VULKAN)
// gl_ViewIndex in a comment is not a read
#endif
void main() { int v = gl_ViewIndex; }
#else
int unguarded = 0;
#endif
"""


def test_reads_are_attributed_to_their_stage_guard():
    codegen = _Codegen({"vertex", "task"})

    assert lower_view_index(codegen, None, None, GUARDED) == GUARDED
    assert codegen.refusals == [
        "'gl_ViewIndex': GLSL declares the multiview view index in no task shader"
    ]


def test_an_unguarded_read_counts_for_every_stage():
    codegen = _Codegen({"vertex", "fragment"})
    code = "#version 460\nint f() { return gl_ViewIndex; }\n"

    lower_view_index(codegen, None, None, code)

    assert len(codegen.refusals) == 1
    assert "vertex shader" in codegen.refusals[0]


def test_hlsl_sv_view_id_reaches_glsl(tmp_path):
    code = _translate(tmp_path, HLSL_FRAGMENT, "opengl", ".hlsl")

    assert PRELUDE in code
    assert "CROSSGL_VIEW_INDEX" in code.split(PRELUDE, 1)[1]
    assert _glslang(tmp_path, code, "frag") == ""
