"""CrossGL's gl_HelperInvocation reaches Slang as the IsHelperLane() intrinsic."""

import pytest

import crosstl.translator
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused

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


def generate(code):
    return SlangCodeGen().generate(crosstl.translator.parse(code))


def test_slang_helper_invocation_reads_lower_to_is_helper_lane():
    slang = generate(HELPER_READ)

    assert "return IsHelperLane();" in slang
    assert "bool helper = IsHelperLane();" in slang
    assert "gl_HelperInvocation" not in slang


def test_slang_local_named_gl_helper_invocation_is_left_alone():
    slang = generate("""
        shader HelperShadow {
            fragment {
                vec4 main(vec4 color @ COLOR0) @ gl_FragColor {
                    bool gl_HelperInvocation = false;
                    return gl_HelperInvocation ? vec4(0.0) : color;
                }
            }
        }
    """)

    assert "bool gl_HelperInvocation = false;" in slang
    assert "IsHelperLane" not in slang


def test_slang_refuses_unmapped_builtin_semantic():
    with pytest.raises(
        SlangLoweringRefused,
        match="Slang has no system value for the built-in @gl_HelperInvocation",
    ):
        generate("""
            shader Probe {
                struct FragmentInput {
                    vec2 uv;
                    bool helper @ gl_HelperInvocation;
                }
                fragment {
                    void main(FragmentInput input) {}
                }
            }
        """)


def test_slang_refuses_unmapped_builtin_entry_parameter():
    with pytest.raises(
        SlangLoweringRefused,
        match="Slang has no system value for the built-in @gl_VertexIndex",
    ):
        generate("""
            shader Probe {
                vertex {
                    vec4 main(vec3 position @ POSITION, int index @ gl_VertexIndex)
                        @ gl_Position {
                        return vec4(position, float(index));
                    }
                }
            }
        """)


def test_slang_keeps_workgroup_count_entry_parameter():
    slang = generate("""
        shader Probe {
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer uint* o;
                void main(uvec3 gid @ gl_GlobalInvocationID, uvec3 groups @ gl_NumWorkGroups) {
                    o[gid.x] = groups.x;
                }
            }
        }
    """)

    assert "uint3 groups = WorkgroupCount();" in slang
    assert "gl_NumWorkGroups" not in slang
