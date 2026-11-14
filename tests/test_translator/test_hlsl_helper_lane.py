"""CrossGL's gl_HelperInvocation reaches HLSL as the Shader Model 6.6 IsHelperLane() intrinsic."""

import pytest

import crosstl.translator
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused

HELPER_READ = """
shader HelperRead {
    fragment {
        bool probe() { return gl_HelperInvocation; }
        vec4 main(vec4 color @ COLOR0) @ gl_FragColor {
            bool helper = gl_HelperInvocation;
            return (helper || probe()) ? vec4(0.0) : color;
        }
    }
}
"""


def generate(code, target_profile=None):
    return HLSLCodeGen(target_profile=target_profile).generate(
        crosstl.translator.parse(code)
    )


@pytest.mark.parametrize("target_profile", [None, "directx-12"])
def test_helper_invocation_reads_lower_to_is_helper_lane(target_profile):
    hlsl = generate(HELPER_READ, target_profile)

    assert "return IsHelperLane();" in hlsl
    assert "bool helper = IsHelperLane();" in hlsl
    assert "gl_HelperInvocation" not in hlsl


@pytest.mark.parametrize("target_profile", ["directx-10", "directx-11"])
def test_shader_model_5_profiles_refuse_helper_invocation(target_profile):
    with pytest.raises(
        DirectXLoweringRefused, match="IsHelperLane\\(\\) requires Shader Model 6.6"
    ):
        generate(HELPER_READ, target_profile)


def test_a_local_named_gl_helper_invocation_is_left_alone():
    hlsl = generate("""
        shader HelperShadow {
            fragment {
                vec4 main(vec4 color @ COLOR0) @ gl_FragColor {
                    bool gl_HelperInvocation = false;
                    return gl_HelperInvocation ? vec4(0.0) : color;
                }
            }
        }
    """)

    assert "bool gl_HelperInvocation = false;" in hlsl
    assert "IsHelperLane" not in hlsl
