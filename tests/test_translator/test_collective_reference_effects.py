"""Writable reference arguments do not write their value-only selectors."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from crosstl.translator.codegen.GLSL_codegen import OpenGLSoftwareSubgroupError
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _canonical, _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_COLLECTIVE_REFERENCE_RUNTIME"
HELPER = "void add(inout uint result, uint lane, uint count) { result += WaveActiveSum(lane); if (count > 0u) { result += WaveActiveSum(lane + 1u); } }"


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "destination", ["totals[i]", "totals[1u - i]", "cells[i].value"]
)
@pytest.mark.parametrize("direction", ["inout", "out"])
def test_selected_reference_writes_preserve_uniform_indices(
    tmp_path, target, destination, direction
):
    helpers = "struct Cell { uint value; }; " + HELPER
    if direction == "out":
        helpers = helpers.replace("inout uint result", "out uint result").replace(
            "result += WaveActiveSum(lane);", "result = WaveActiveSum(lane);"
        )
    source = _canonical(
        "uint totals[2]; Cell cells[2]; totals[0] = 0u; totals[1] = 0u; cells[0].value = 0u; cells[1].value = 0u;"
        f" for (uint i = 0u; i < 2u; ++i) {{ add({destination}, invocation, i); }}"
        " results[invocation] = totals[0] + totals[1] + cells[0].value + cells[1].value;",
        helpers,
    )
    generator = _codegen(target)
    ast = parse(source)
    generated = generator.generate_stage(ast, "compute")
    assert generated == generator.generate_stage(ast, "compute")
    assert "WaveActiveSum(" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "body,helpers",
    [
        ("add(totals[i++], invocation, i);", ""),
        ("add(totals[++i], invocation, i);", ""),
        ("add(totals[i], invocation, i++);", ""),
        ("add(i, invocation, 0u);", ""),
        (
            "add(totals[change(i, invocation)], invocation, i);",
            "uint change(inout uint index, uint lane) { index = lane; return index; }",
        ),
        ("add(totals[unknown(i)], invocation, i);", ""),
        ("unknown(totals[i]); add(totals[i], invocation, i);", ""),
        (
            "add(totals[i], invocation, i); stop = invocation;",
            "",
        ),
        (
            "add(totals[i], invocation, i); change(stop, invocation);",
            "void change(inout uint limit, uint lane) { limit = lane; }",
        ),
        ("add(totals[i], invocation, i); uint i = invocation;", ""),
    ],
)
def test_reference_effects_retain_control_mutations(target, body, helpers):
    error = (
        DirectXSoftwareSubgroupError
        if target == "directx"
        else OpenGLSoftwareSubgroupError
    )
    source = _canonical(
        "uint totals[2]; totals[0] = 0u; totals[1] = 0u; uint stop = 2u;"
        f" for (uint i = 0u; i < stop; ++i) {{ {body} }}"
        " results[invocation] = totals[0] + totals[1];",
        HELPER + helpers,
    )
    with pytest.raises(error):
        _codegen(target).generate_stage(parse(source), "compute")


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_written_reference_root_does_not_remain_uniform(target):
    source = _canonical(
        "uint stop = 2u; add(stop, invocation, 0u);"
        " for (uint i = 0u; i < stop; ++i) { results[invocation] = WaveActiveSum(invocation); }",
        HELPER,
    )
    error = (
        DirectXSoftwareSubgroupError
        if target == "directx"
        else OpenGLSoftwareSubgroupError
    )
    with pytest.raises(error):
        _codegen(target).generate_stage(parse(source), "compute")


def _metal_source(record, depth):
    helpers = """void add(thread uint& result, uint value, uint count) {
    result += simd_sum(value);
    if (count > 0u) { result += simd_sum(value + 1u); }
}
"""
    callee = "add"
    for level in range(depth):
        name = f"forward{level}"
        helpers += f"void {name}(thread uint& result, uint value, uint count) {{ {callee}(result, value, count); }}\n"
        callee = name
    if record:
        declaration = "Cell totals[2];"
        initialize = "totals[i].value = lane + i; totals[i].guard = 0xDEADBEEFu;"
        count, destination = 2, "totals[i].value"
        store = "output[index] = totals[0].value; output[index + 1u] = totals[0].guard; output[index + 2u] = totals[1].value; output[index + 3u] = totals[1].guard;"
    else:
        declaration = "uint totals[4];"
        initialize = "totals[i] = lane + i;"
        count, destination = 4, "totals[i]"
        store = "for (short slot = 0; slot < 4; ++slot) { output[index + uint(slot)] = totals[slot]; }"
    return f"""#include <metal_stdlib>
using namespace metal;
struct Cell {{ uint value; uint guard; }};
{helpers}
kernel void accumulate(device uint* output [[buffer(0)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {{
    if (group.x >= 2u) {{ return; }}
    {declaration}
    for (uint i = 0u; i < {count}u; ++i) {{ {initialize} }}
    for (uint i = 0u; i < {count}u; ++i) {{
        for (short repeat = 0; repeat < 3; ++repeat) {{
            {callee}({destination}, lane + uint(repeat) + group.x, i % 2u);
        }}
    }}
    uint index = 4u + (group.x * 64u + lane) * 4u;
    {store}
}}
"""


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "mutation",
    [
        "group.x = invocation;",
        "change(group.x, invocation);",
        "unknown(group.x);",
        "uint values[2]; values[change(group.x, invocation)] = 0u;",
        "uint3 group = uint3(invocation, 0u, 0u);",
    ],
)
def test_entry_exit_facts_retain_writes_and_unknown_calls(target, mutation):
    source = _canonical(
        mutation + " if (group.x >= 2u) { return; }"
        " uint totals[2]; totals[0] = 0u; totals[1] = 0u;"
        " for (uint i = 0u; i < 2u; ++i) { add(totals[i], invocation + group.x, i); }"
        " results[invocation] = totals[0] + totals[1];",
        HELPER
        + " uint change(inout uint value, uint lane) { value = lane; return value; }",
    ).replace(
        "uint invocation @gl_LocalInvocationIndex",
        "uint invocation @gl_LocalInvocationIndex, uint3 group @gl_WorkGroupID",
    )
    error = (
        DirectXSoftwareSubgroupError
        if target == "directx"
        else OpenGLSoftwareSubgroupError
    )
    with pytest.raises(error):
        _codegen(target).generate_stage(parse(source), "compute")


def _request(root, target, record, depth):
    source, descriptor, package = _package(
        root, target, "uint", (32, 2, 1), source=_metal_source(record, depth)
    )
    guard = 0xDEADBEEF
    expected = [guard] * 777
    for group in range(2):
        for lane in range(64):
            first = lane // 32 * 32
            for slot in range(2 if record else 4):
                value = lane + slot
                for repeat in range(3):
                    value += sum(
                        neighbor + repeat + group
                        for neighbor in range(first, first + 32)
                    )
                    if slot % 2:
                        value += sum(
                            neighbor + repeat + group + 1
                            for neighbor in range(first, first + 32)
                        )
                expected[4 + (group * 64 + lane) * 4 + slot * (2 if record else 1)] = (
                    value
                )
    assert len(descriptor["bindings"]) == 1
    name = descriptor["bindings"][0]["name"]
    inputs = {name: {"dtype": "uint32", "shape": [777], "values": [guard] * 777}}
    outputs = {name: {"dtype": "uint32", "shape": [777], "values": expected}}
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [32, 2, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("depth", [0, 2])
def test_indexed_reference_helpers_translate_and_compile(
    tmp_path, target, record, depth
):
    _, request, _ = _request(tmp_path, target, record, depth)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("depth", [0, 2])
def test_indexed_reference_helpers_execute(tmp_path, record, depth):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required reference-effect execution")
    target = (
        "metal"
        if sys.platform == "darwin"
        else "directx" if sys.platform == "win32" else "opengl"
    )
    source, request, outputs = _request(tmp_path, target, record, depth)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="accumulate",
    )


def test_reference_effect_execution_is_required_in_existing_ci_job():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_collective_reference_effects.py" in step
    assert "-n auto" in step and "continue-on-error" not in step
