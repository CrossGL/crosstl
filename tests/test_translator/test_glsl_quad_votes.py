"""CrossGL's quad votes QuadAll and QuadAny become GL_EXT_shader_quad_control calls in GLSL output.

The GLSL generator used to refuse both as wave intrinsics it did not recognize, so HLSL's QuadAll and QuadAny
(shader model 6.7) and CrossGL programs that use them could not reach OpenGL or Vulkan GLSL.
"""

import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.codegen.opengl_values import OpenGLSoftwareSubgroupError

VOTE = "#extension GL_KHR_shader_subgroup_vote : require"
QUAD_CONTROL = "#extension GL_EXT_shader_quad_control : require"


def _program(body, value_type="bool"):
    return f"""
    shader QuadVote {{
        compute {{
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
            buffer float* fin;
            buffer int* outp;
            void main() {{
                uint i = gl_GlobalInvocationID.x;
                {value_type} a = {"fin[i] > 0.0" if value_type == "bool" else "fin[i]"};
                {body}
            }}
        }}
    }}
    """


def _generate(source, **options):
    return GLSLCodeGen(**options).generate(crosstl.translator.parse(source))


def _glslang(tmp_path, code):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / "out.comp"
    path.write_text(code)
    for extra in (
        [],
        ["-V", "--target-env", "vulkan1.1", "-o", str(tmp_path / "o.spv")],
    ):
        result = subprocess.run(
            ["glslangValidator", *extra, "-S", "comp", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "operation, spelling",
    [("QuadAll", "subgroupQuadAll"), ("QuadAny", "subgroupQuadAny")],
)
def test_quad_vote_writes_the_quad_control_function(tmp_path, operation, spelling):
    glsl = _generate(_program(f"outp[i] = {operation}(a) ? 1 : 0;"))

    assert f"{spelling}(a)" in glsl
    # GL_EXT_shader_quad_control depends on GL_KHR_shader_subgroup_vote, which glslang requires to be enabled.
    lines = glsl.splitlines()
    assert lines.index(VOTE) < lines.index(QUAD_CONTROL)
    _glslang(tmp_path, glsl)


def test_quad_votes_and_wave_votes_share_one_vote_extension_line():
    glsl = _generate(
        _program(
            "outp[i] = (QuadAll(a) ? 4 : 0) + (QuadAny(a) ? 2 : 0) "
            "+ (WaveActiveAllTrue(a) ? 1 : 0);"
        )
    )

    assert glsl.count(VOTE) == 1
    assert glsl.count(QUAD_CONTROL) == 1


def test_hlsl_quad_votes_reach_glsl(tmp_path):
    source = tmp_path / "quad.hlsl"
    source.write_text("""RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<int> outp : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    bool a = fin[id.x] > 0.0;
    outp[id.x] = (QuadAll(a) ? 2 : 0) + (QuadAny(a) ? 1 : 0);
}
""")
    glsl = crosstl.translate(str(source), backend="opengl", format_output=False)

    assert "subgroupQuadAll(a)" in glsl
    assert "subgroupQuadAny(a)" in glsl
    assert QUAD_CONTROL in glsl
    _glslang(tmp_path, glsl)


def test_quad_vote_of_a_float_is_refused():
    with pytest.raises(OpenGLLoweringRefused, match="QuadAll requires a boolean"):
        _generate(_program("outp[i] = QuadAll(a) ? 1 : 0;", value_type="float"))


def test_software_subgroups_refuse_quad_votes():
    # Emulated subgroups have no quad lowering for the votes, so the generator fails closed.
    with pytest.raises(OpenGLSoftwareSubgroupError, match="QuadAny"):
        _generate(_program("outp[i] = QuadAny(a) ? 1 : 0;"), software_subgroup_width=32)
