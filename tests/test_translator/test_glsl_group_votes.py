"""OpenGL output writes a stage's boolean group votes with GL_ARB_shader_group_vote (#3009).

GLSL 4.60 (section 8.18) took ``allInvocations``, ``anyInvocation`` and ``allInvocationsEqual`` from
GL_ARB_shader_group_vote, which OpenGL drivers without the GL_KHR_shader_subgroup extensions implement: Mesa's
llvmpipe rejects ``#extension GL_KHR_shader_subgroup_basic`` in a compute shader. A stage whose only subgroup
operations are votes on booleans writes them with the ARB suffix below 4.60 and with the core names from 4.60. A
stage that needs another GL_KHR_shader_subgroup operation, or votes on a value that is not a boolean, keeps
``subgroupAll``, ``subgroupAny`` and ``subgroupAllEqual``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

KHR_VOTE = "#extension GL_KHR_shader_subgroup_vote : require"
ARB_VOTE = "#extension GL_ARB_shader_group_vote : require"


def _probe(vote, argument="a0"):
    # The coverage probe of #3009, with the vote and its argument as parameters.
    return f"""shader Probe {{
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* outp;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            if (i % 3u != 0u) {{
                bool a0 = (fin[i + 0u] > 0.0);
                bool r = {vote}({argument});
                outp[i] = r ? 1 : 0;
            }}
        }}
    }}
}}"""


def _opengl(source):
    return GLSLCodeGen().generate(parse(source))


def _glslang(tmp_path, code):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / "vote.comp"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run([validator, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "vote", ["allInvocations", "anyInvocation", "allInvocationsEqual"]
)
def test_a_boolean_vote_needs_only_arb_shader_group_vote(tmp_path, vote):
    generated = _opengl(_probe(vote))

    assert generated.startswith(f"#version 450 core\n{ARB_VOTE}\n")
    assert "GL_KHR_shader_subgroup" not in generated
    assert f"bool r = {vote}ARB(a0);" in generated
    assert "subgroup" not in generated
    _glslang(tmp_path, generated)


@pytest.mark.parametrize(
    ("wave", "vote"),
    [
        ("WaveActiveAllTrue", "allInvocationsARB"),
        ("WaveActiveAnyTrue", "anyInvocationARB"),
        ("WaveActiveAllEqual", "allInvocationsEqualARB"),
    ],
)
def test_hlsl_wave_votes_on_booleans_use_the_group_vote_functions(tmp_path, wave, vote):
    generated = _opengl(_probe(wave, "!a0"))

    assert ARB_VOTE in generated and "GL_KHR_shader_subgroup" not in generated
    assert f"bool r = {vote}((!a0));" in generated
    _glslang(tmp_path, generated)


def test_glsl_460_writes_the_core_vote_functions(tmp_path):
    source = """#version 460
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Results { uint results[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    bool odd = (i & 1u) == 1u;
    bool a = allInvocations(odd);
    bool b = anyInvocation(odd);
    bool c = allInvocationsEqual(odd);
    results[i] = uint(a) + uint(b) * 2u + uint(c) * 4u;
}
"""
    path = tmp_path / "votes.comp"
    path.write_text(source, encoding="utf-8")
    generated = crosstl.translate(str(path), backend="opengl")

    assert generated.startswith("#version 460")
    assert "#extension" not in generated
    assert "bool a = allInvocations(odd);" in generated
    assert "bool b = anyInvocation(odd);" in generated
    assert "bool c = allInvocationsEqual(odd);" in generated
    _glslang(tmp_path, generated)


def test_a_vote_beside_another_subgroup_operation_keeps_khr_subgroup_vote():
    source = _probe("allInvocations").replace(
        "outp[i] = r ? 1 : 0;", "outp[i] = int(WaveActiveSum(i)) + (r ? 1 : 0);"
    )
    generated = _opengl(source)

    assert KHR_VOTE in generated and ARB_VOTE not in generated
    assert "bool r = subgroupAll(a0);" in generated


def test_all_equal_on_a_value_that_is_not_a_boolean_keeps_khr_subgroup_vote():
    generated = _opengl(_probe("allInvocationsEqual", "i"))

    assert KHR_VOTE in generated and ARB_VOTE not in generated
    assert "bool r = subgroupAllEqual(i);" in generated


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl")


# A uniform-block member ``int flag`` beside a helper's ``bool flag`` parameter: the name is not only a bool, so the
# vote stays subgroupAllEqual, which takes an int (review of #4666).
HLSL_BLOCK_MEMBER_SHARING_A_BOOL_NAME = """cbuffer Params : register(b0) { int flag; };
RWStructuredBuffer<int> outp : register(u0);
bool negate(bool flag) { return !flag; }
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outp[id.x] = WaveActiveAllEqual(flag) ? 1 : 0;
}
"""
GLSL_BLOCK_MEMBER_SHARING_A_BOOL_NAME = """#version 450
#extension GL_KHR_shader_subgroup_vote : require
layout(local_size_x = 64) in;
layout(std140, binding = 0) uniform Params { int flag; };
layout(std430, binding = 1) buffer Out { int outp[]; };
bool negate(bool flag) { return !flag; }
void main() {
    outp[gl_GlobalInvocationID.x] = subgroupAllEqual(flag) ? 1 : 0;
}
"""


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("vote.hlsl", HLSL_BLOCK_MEMBER_SHARING_A_BOOL_NAME),
        ("vote.comp", GLSL_BLOCK_MEMBER_SHARING_A_BOOL_NAME),
    ],
)
def test_all_equal_on_a_block_member_sharing_a_bool_name_keeps_khr_subgroup_vote(
    tmp_path, name, source
):
    generated = _translate(tmp_path, name, source)

    assert KHR_VOTE in generated and ARB_VOTE not in generated
    assert "subgroupAllEqual(flag)" in generated
    _glslang(tmp_path, generated)


def test_a_vote_beside_a_subgroup_mask_built_in_keeps_khr_subgroup_vote(tmp_path):
    source = """#version 450
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Out { uint outp[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    bool odd = (i & 1u) == 1u;
    bool r = subgroupAll(odd);
    outp[i] = gl_SubgroupEqMask.x + (r ? 1u : 0u);
}
"""
    generated = _translate(tmp_path, "vote.comp", source)

    assert KHR_VOTE in generated and ARB_VOTE not in generated
    assert "bool r = subgroupAll(odd);" in generated
    _glslang(tmp_path, generated)
