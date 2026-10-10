"""Finite additive induction preserves every software collective invocation."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLSoftwareSubgroupError,
)
from tests.test_translator.test_directx_software_reductions import _codegen, _source
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_COLLECTIVE_STRIDE_RUNTIME"


@pytest.fixture(params=["directx", "opengl"])
def stride_target(request):
    return request.param


def _generator(target):
    return (
        _codegen() if target == "directx" else GLSLCodeGen(software_subgroup_width=32)
    )


STRIDE_ERRORS = (DirectXSoftwareSubgroupError, OpenGLSoftwareSubgroupError)

CASES = {
    "ascending": ("short step = 0; step < 16; step += 8", (0, 8)),
    "descending": ("short step = 16; step >= 0; step -= 8", (16, 8, 0)),
    "negative-stride": ("int step = 7; step > -2; step += -3", (7, 4, 1)),
    "negative-subtract": ("int step = -3; step < 4; step -= -3", (-3, 0, 3)),
    "reversed-comparison": ("short step = 0; 16 > step; step += 8", (0, 8)),
    "zero-trip": ("short step = 16; step < 16; step += 8", ()),
    "narrow-boundary": (
        "short step = 32750; step < 32766; step += 8",
        (32750, 32758),
    ),
    "unit-boundary": ("short step = 32765; step < 32767; ++step", (32765, 32766)),
    "unit-descending": ("ushort step = 3; step > 0; --step", (3, 2, 1)),
}


def _canonical(header, body="", helpers=""):
    return _source(
        "uint",
        f"uint result = 0u; for ({header}) {{ {body} result += WaveActiveSum(invocation + uint(step)); }} outputWords[index] = result;",
        helpers,
    )


@pytest.mark.parametrize("case", CASES)
def test_fixed_stride_collectives_compile(tmp_path, case, stride_target):
    generator = _generator(stride_target)
    ast = parse(_canonical(CASES[case][0]))
    generated = generator.generate_stage(ast, "compute")
    assert generated == generator.generate_stage(ast, "compute")
    assert (
        "GroupMemoryBarrierWithGroupSync"
        if stride_target == "directx"
        else "barrier();"
    ) in generated
    assert "WaveActiveSum(" not in generated
    _compile(generated, stride_target, tmp_path)


@pytest.mark.parametrize(
    "header",
    [
        "int step = 0; step < 16; step += 0",
        "int step = 0; step < 16; step -= 8",
        "int step = 16; step > 0; step += 8",
        "short step = 32760; step < 32767; step += 8",
        "short step = -32760; step > -32768; step -= 9",
        "short step = 0; step <= 32767; step += 1",
        "short step = 32768; step < 32769; step += 1",
        "uint step = 4294967290u; step < 4294967295u; step += 8u",
        "uint step = 8u; step >= 0u; step -= 8u",
        "int step = 2147483640; step < 2147483647; step += 8",
        "int step = -2; step < 8u; step += 2",
        "int step = 0; step < 16; step += invocation",
        "int step = 0; step < invocation; step += 8",
        "int step = int(group.x); step < 16; step += 8",
        "int step = 0; step < 16; step += unknown()",
        "int step = 0; step < 16; step += (2147483647 + 8) - 2147483647",
        "int step = 0; step < 16; step += ((-2147483647 - 1) % -1) + 8",
    ],
)
def test_unproved_or_overflowing_induction_is_rejected(header, stride_target):
    with pytest.raises(STRIDE_ERRORS):
        _generator(stride_target).generate_stage(parse(_canonical(header)), "compute")


@pytest.mark.parametrize(
    "body",
    [
        "step = int(invocation);",
        "++step;",
        "int step = int(invocation);",
        "unknown(step);",
        "int& alias = step; alias = int(invocation);",
        "int* alias = &step; *alias = int(invocation);",
        "if (invocation == 0u) { break; }",
        "if (invocation == 0u) { continue; }",
        "if (invocation == 0u) { return; }",
    ],
)
def test_fixed_stride_proof_retains_mutation_and_exit_checks(body, stride_target):
    with pytest.raises(STRIDE_ERRORS):
        _generator(stride_target).generate_stage(
            parse(_canonical("int step = 0; step < 16; step += 8", body)),
            "compute",
        )


def test_constant_expression_stride_compiles(tmp_path, stride_target):
    generated = _generator(stride_target).generate_stage(
        parse(_canonical("int step = 0; step < 4 * 4; step += 2 * 4")),
        "compute",
    )
    _compile(generated, stride_target, tmp_path)


@pytest.mark.parametrize(
    "index", ["index", "index + 4u", "group.x * 128u + invocation"]
)
def test_collective_store_value_compiles(tmp_path, index):
    helper = "uint collect(uint value) { return WaveActiveSum(value); }"
    generated = _generator("opengl").generate_stage(
        parse(
            _source(
                "uint",
                f"buffer_store(outputWords, {index}, collect(invocation));",
                helper,
            )
        ),
        "compute",
    )
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "body,extra",
    [
        ("buffer_store(outputWords, index++, collect(invocation));", ""),
        ("buffer_store(outputWords, unknown(index), collect(invocation));", ""),
        (
            "buffer_store(outputWords, index, invocation == 0u ? collect(invocation) : 0u);",
            "",
        ),
        (
            "if (invocation == 0u) { buffer_store(outputWords, index, collect(invocation)); }",
            "",
        ),
        (
            "buffer_store(outputWords, index, collect(invocation));",
            "void buffer_store(RWStructuredBuffer<uint> data, uint offset, uint value) { data[offset] = value; }",
        ),
    ],
)
def test_collective_store_proof_rejects_unchecked_evaluation(body, extra):
    helper = "uint collect(uint value) { return WaveActiveSum(value); } " + extra
    with pytest.raises(OpenGLSoftwareSubgroupError):
        _generator("opengl").generate_stage(
            parse(_source("uint", body, helper)), "compute"
        )


@pytest.mark.parametrize(
    "header",
    [
        "short step = 32760; step <= 32767; ++step",
        "ushort step = 65530; step <= 65535; ++step",
        "short step = -32760; step >= -32768; --step",
    ],
)
def test_narrow_unit_loops_require_a_representable_terminating_update(
    header, stride_target
):
    with pytest.raises(STRIDE_ERRORS):
        _generator(stride_target).generate_stage(parse(_canonical(header)), "compute")


def _metal_source(case):
    return f"""#include <metal_stdlib>
using namespace metal;
uint collect(uint lane) {{
    uint result = 0u;
    for ({CASES[case][0]}) {{ result += simd_sum(lane + uint(step)); }}
    return result;
}}
kernel void accumulate(device uint* output [[buffer(0)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {{
    if (group.x >= 2u) {{ return; }}
    output[4u + group.x * 64u + lane] = collect(lane);
}}
"""


def _request(root, target, case):
    source, descriptor, package = _package(
        root, target, "uint", (32, 2, 1), source=_metal_source(case)
    )
    guard = 0xDEADBEEF
    expected = [guard] * 201
    for lane in range(128):
        first = (lane % 64) // 32 * 32
        expected[4 + lane] = (
            sum(
                neighbor + step
                for step in CASES[case][1]
                for neighbor in range(first, first + 32)
            )
            & 0xFFFFFFFF
        )
    assert len(descriptor["bindings"]) == 1
    name = descriptor["bindings"][0]["name"]
    inputs = {name: {"dtype": "uint32", "shape": [201], "values": [guard] * 201}}
    outputs = {name: {"dtype": "uint32", "shape": [201], "values": expected}}
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
@pytest.mark.parametrize("case", CASES)
def test_fixed_stride_helpers_translate_and_compile(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_fixed_stride_helpers_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required fixed-stride execution")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform, "opengl")
    source, request, outputs = _request(tmp_path, target, case)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="accumulate",
    )


def test_fixed_stride_execution_is_required_in_existing_ci_job():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_collective_loop_strides.py" in step
    assert "-n auto" in step and "continue-on-error" not in step
