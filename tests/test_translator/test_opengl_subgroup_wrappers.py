"""Converged collective calls across resolved OpenGL helper graphs."""

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLSoftwareSubgroupError,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_votes import _canonical


def _generate(body, helpers):
    return GLSLCodeGen(software_subgroup_width=32).generate_stage(
        parse(_canonical(body, helpers)), "compute"
    )


def _helpers(depth, operation, value_type):
    callee = "leaf"
    helpers = (
        f"{value_type} leaf({value_type} value) {{ return {operation}(value); }}\n"
    )
    for index in range(depth):
        name = f"wrapper_{index}"
        helpers += (
            f"{value_type} {name}({value_type} value) {{ return {callee}(value); }}\n"
        )
        callee = name
    return helpers, callee


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize(
    "operation,value_type",
    [
        ("WaveActiveSum", "float"),
        ("WaveActiveAllTrue", "bool"),
        ("WaveActiveAnyTrue", "bool"),
    ],
)
def test_wrappers_compile_and_preserve_single_evaluation(
    tmp_path, depth, operation, value_type
):
    helpers, callee = _helpers(depth, operation, value_type)
    generated = _generate(
        f"uint counter = invocation; {value_type} result = {callee}({value_type}(counter++)); results[invocation] = uint(result);",
        helpers,
    )
    assert generated.count("counter++") == 1
    assert f"{value_type} {callee}(" in generated
    assert "GL_KHR_shader_subgroup" not in generated
    assert generated.count("memoryBarrierShared();") == generated.count("barrier();")
    _compile(generated, "opengl", tmp_path)


def test_wrapper_calls_in_uniform_entry_loop_compile(tmp_path):
    helpers, callee = _helpers(3, "WaveActiveSum", "float")
    generated = _generate(
        f"""float total = 0.0;
        for (uint i = 0u; i < 3u; ++i) {{
            float value = {callee}(float(invocation + i));
            total += value;
        }}
        results[invocation] = uint(total);""",
        helpers,
    )
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "if (value) { return leaf(value); } return false;",
        "return value ? leaf(value) : false;",
        "return value && leaf(value);",
        "for (uint i = 0u; i < uint(value); ++i) { value = leaf(value); } return value;",
    ],
)
def test_wrappers_reject_conditional_collective_calls(body):
    helpers = f"bool leaf(bool value) {{ return WaveActiveAnyTrue(value); }} bool wrapper(bool value) {{ {body} }}"
    with pytest.raises(OpenGLSoftwareSubgroupError) as raised:
        _generate(
            "bool result = wrapper(invocation != 0u); results[invocation] = uint(result);",
            helpers,
        )
    assert raised.value.reason == "helper-call-not-uniform"


@pytest.mark.parametrize("location", ["entry", "wrapper", "leaf"])
def test_wrappers_reject_preceding_divergent_returns(location):
    leaf_exit = "if (!value) { return false; }" if location == "leaf" else ""
    wrapper_exit = "if (!value) { return false; }" if location == "wrapper" else ""
    entry_exit = "if (invocation == 0u) { return; }" if location == "entry" else ""
    helpers = f"""bool leaf(bool value) {{ {leaf_exit} return WaveActiveAnyTrue(value); }}
    bool wrapper(bool value) {{ {wrapper_exit} return leaf(value); }}"""
    with pytest.raises(OpenGLSoftwareSubgroupError) as raised:
        _generate(
            entry_exit
            + "bool result = wrapper(invocation != 0u); results[invocation] = uint(result);",
            helpers,
        )
    assert raised.value.reason == "potentially-divergent-control-flow"


@pytest.mark.parametrize("mutual", [False, True])
def test_wrappers_reject_recursive_collective_graphs(mutual):
    if mutual:
        helpers = """float wrapper(float value) { return recurse(value); }
        float recurse(float value) { float result = WaveActiveSum(value); return wrapper(result); }"""
    else:
        helpers = """float wrapper(float value) {
            float result = WaveActiveSum(value); return wrapper(result);
        }"""
    with pytest.raises(OpenGLSoftwareSubgroupError) as raised:
        _generate(
            "float value = wrapper(float(invocation)); results[invocation] = uint(value);",
            helpers,
        )
    assert raised.value.reason == "helper-call-recursive"


def test_wrappers_preserve_resolved_nested_overload_identity(tmp_path):
    helpers = """float leaf(float value) { return WaveActiveSum(value); }
    uint leaf(uint value) { return WaveActiveMax(value); }
    float inner(float value) { return leaf(value); }
    float outer(float value) { return inner(value); }"""
    generated = _generate(
        "float value = outer(float(invocation)); results[invocation] = uint(value);",
        helpers,
    )
    assert "float leaf(float value)" in generated
    assert "uint leaf(uint value)" not in generated
    _compile(generated, "opengl", tmp_path)


def test_wrapper_root_overloads_remain_diagnostic():
    helpers = """float leaf(float value) { return WaveActiveSum(value); }
    float wrapper(float value) { return leaf(value); }
    uint wrapper(uint value) { return value; }"""
    with pytest.raises(OpenGLSoftwareSubgroupError) as raised:
        _generate(
            "float value = wrapper(float(invocation)); results[invocation] = uint(value);",
            helpers,
        )
    assert raised.value.reason == "helper-identity-ambiguous"


@pytest.mark.parametrize("condition", ["groups > 1u", "gl_WorkGroupID.x % 2u == 0u"])
def test_wrappers_in_immutable_workgroup_uniform_branches_compile(tmp_path, condition):
    helpers, callee = _helpers(2, "WaveActiveAnyTrue", "bool")
    source = _canonical(
        f"""bool result = false;
        if ({condition}) {{ result = {callee}(invocation != 0u); }}
        else {{ result = {callee}(invocation == 0u); }}
        results[invocation] = uint(result);""",
        helpers,
    ).replace(
        "uint invocation @gl_LocalInvocationIndex",
        "uint invocation @gl_LocalInvocationIndex, uint groups @gl_NumSubgroups",
    )
    generated = GLSLCodeGen(software_subgroup_width=32).generate_stage(
        parse(source), "compute"
    )
    assert "gl_NumSubgroups" not in generated
    _compile(generated, "opengl", tmp_path)


def test_wrapper_branch_rejects_raw_hardware_subgroup_count():
    helpers, callee = _helpers(2, "WaveActiveAnyTrue", "bool")
    with pytest.raises(OpenGLSoftwareSubgroupError) as raised:
        _generate(
            f"if (gl_NumSubgroups > 1u) {{ bool result = {callee}(invocation != 0u); }}",
            helpers,
        )
    assert raised.value.reason == "subgroup-builtin-unsupported"


def test_wrapper_branch_rejects_const_lane_parameter():
    helpers, callee = _helpers(2, "WaveActiveAnyTrue", "bool")
    source = _canonical(
        f"if (invocation != 0u) {{ bool result = {callee}(true); }}", helpers
    ).replace(
        "uint invocation @gl_LocalInvocationIndex",
        "const uint invocation @gl_LocalInvocationIndex",
    )
    with pytest.raises(OpenGLSoftwareSubgroupError):
        GLSLCodeGen(software_subgroup_width=32).generate_stage(parse(source), "compute")


@pytest.mark.parametrize(
    "prefix",
    [
        "groups = invocation;",
        "observe(groups);",
        "uint& alias = groups; alias = invocation;",
    ],
)
def test_wrapper_branch_proof_rejects_mutated_or_escaped_uniform_parameters(prefix):
    helpers, callee = _helpers(2, "WaveActiveAnyTrue", "bool")
    helpers += "void observe(inout uint value) { value = gl_LocalInvocationIndex; }"
    source = _canonical(
        prefix
        + f"if (groups > 1u) {{ bool result = {callee}(invocation != 0u); results[invocation] = uint(result); }}",
        helpers,
    ).replace(
        "uint invocation @gl_LocalInvocationIndex",
        "uint invocation @gl_LocalInvocationIndex, uint groups @gl_NumSubgroups",
    )
    with pytest.raises(OpenGLSoftwareSubgroupError):
        GLSLCodeGen(software_subgroup_width=32).generate_stage(parse(source), "compute")
