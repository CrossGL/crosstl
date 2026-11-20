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
