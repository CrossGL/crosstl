"""Resolved subgroup overloads preserve convergence and integer payloads."""

import math
import os
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_integer_subgroup_shuffles import OFFSETS, SHAPES
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _canonical, _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_OVERLOAD_RUNTIME"


def _generate(body, helpers):
    return _codegen("directx").generate_stage(
        parse(_canonical(body, helpers)), "compute"
    )


@pytest.mark.parametrize("prototype", [False, True])
def test_source_overloads_keep_independent_uniform_arguments(tmp_path, prototype):
    helpers = """
    uint shifted(uint value, uint count) {
        for (uint i = 0u; i < count; ++i) { value = WaveShuffleDown(value, 1u); }
        return value;
    }
    float shifted(float value, uint count) { return value + float(count); }
    uint outer(uint value) { return shifted(value, 2u); }
    """
    if prototype:
        helpers = "uint shifted(uint value, uint count);\n" + helpers
    generated = _generate(
        """uint result = outer(invocation);
        if (invocation > 0u) { result += uint(shifted(float(invocation), invocation)); }
        results[invocation] = result;""",
        helpers,
    )
    assert "groupshared uint " in generated and "WaveReadLane" not in generated
    _compile(generated, "directx", tmp_path)


def test_overload_resolution_uses_lexical_shadow_types(tmp_path):
    generated = _generate(
        """uint value = invocation;
        uint result = shifted(value, 2u);
        if (invocation > 0u) {
            float value = float(invocation);
            result += uint(shifted(value, invocation));
        }
        result += shifted(value, 2u);
        results[invocation] = result;""",
        """uint shifted(uint value, uint count) {
            for (uint i = 0u; i < count; ++i) { value = WaveShuffleDown(value, 1u); }
            return value;
        }
        float shifted(float value, uint count) { return value + float(count); }""",
    )
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize(
    "body,reason",
    [
        ("results[invocation] = shifted(unknown, 1u);", "helper-identity-ambiguous"),
        (
            "if (invocation > 0u) { results[invocation] = shifted(invocation, 1u); }",
            "potentially-divergent-control-flow",
        ),
        (
            "results[invocation] = invocation > 0u ? shifted(invocation, 1u) : 0u;",
            "potentially-divergent-control-flow",
        ),
        (
            "bool result = invocation > 0u && shifted(invocation, 1u) > 0u;",
            "potentially-divergent-control-flow",
        ),
        (
            "results[invocation] = shifted(invocation, invocation);",
            "potentially-divergent-control-flow",
        ),
        (
            "results[invocation] = shifted(invocation, 1u); results[invocation] += shifted(invocation, invocation);",
            "potentially-divergent-control-flow",
        ),
    ],
)
def test_overloaded_collectives_retain_convergence_diagnostics(body, reason):
    helpers = """uint shifted(uint value, uint count) {
        for (uint i = 0u; i < count; ++i) { value = WaveShuffleDown(value, 1u); }
        return value;
    }
    float shifted(float value, uint count) { return value; }"""
    with pytest.raises(DirectXSoftwareSubgroupError) as raised:
        _generate(body, helpers)
    assert raised.value.reason == reason


@pytest.mark.parametrize("mutual", [False, True])
def test_overloaded_collectives_reject_recursive_graphs(mutual):
    helpers = """float shifted(float value) { return value; }
    uint shifted(uint value) {
        uint result = WaveShuffleDown(value, 1u);
        return RECURSE(result);
    }""".replace("RECURSE", "outer" if mutual else "shifted")
    if mutual:
        helpers += "uint outer(uint value) { return shifted(value); }"
    with pytest.raises(DirectXSoftwareSubgroupError) as raised:
        _generate("results[invocation] = shifted(invocation);", helpers)
    assert raised.value.reason == "helper-call-recursive"


@pytest.mark.parametrize("qualifier", ["out", "inout"])
def test_resolved_mutating_overload_invalidates_uniform_loop_bound(qualifier):
    helpers = f"""void modify({qualifier} uint count, uint value) {{ count = value; }}
    void modify(float count, uint value) {{}}
    uint shifted(uint value) {{ return WaveShuffleDown(value, 1u); }}
    float shifted(float value) {{ return value; }}"""
    with pytest.raises(DirectXSoftwareSubgroupError) as raised:
        _generate(
            """uint count = 2u;
            modify(count, invocation);
            for (uint i = 0u; i < count; ++i) { results[invocation] = shifted(invocation); }""",
            helpers,
        )
    assert raised.value.reason == "potentially-divergent-control-flow"


def _source(shape):
    size = math.prod(shape)
    fields = 2 * len(OFFSETS) + 1
    calls = []
    for field, offset in enumerate(OFFSETS):
        calls.append(f"""
    long s{field} = signed_wrapper(signed_value + long(counter++ - {field}u), ushort({offset}u));
    ulong u{field} = unsigned_wrapper(unsigned_value, ushort({offset}u));
    s{field} = {offset}u < active - lane ? s{field} : signed_value;
    u{field} = {offset}u < active - lane ? u{field} : unsigned_value;
    outputs[index * {fields}u + {2 * field + 4}u] = ulong(s{field});
    outputs[index * {fields}u + {2 * field + 5}u] = u{field};""")
    return f"""#include <metal_stdlib>
using namespace metal;
long shifted(long value, ushort delta) {{
    uint2 words = as_type<uint2>(value);
    return as_type<int64_t>(metal::simd_shuffle_down(words, delta));
}}
ulong shifted(ulong value, ushort delta) {{
    uint2 words = as_type<uint2>(value);
    return as_type<uint64_t>(metal::simd_shuffle_down(words, delta));
}}
long signed_wrapper(long value, ushort delta) {{ return shifted(value, delta); }}
ulong unsigned_wrapper(ulong value, ushort delta) {{ return shifted(value, delta); }}
kernel void overloaded_shuffles(const device long* signed_inputs [[buffer(0)]],
                                 const device ulong* unsigned_inputs [[buffer(1)]],
                                 device ulong* outputs [[buffer(2)]],
                                 uint invocation [[thread_index_in_threadgroup]],
                                 uint3 group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {{
    uint index = group.x * {size}u + invocation;
    uint active = min(32u, {size}u - (invocation / 32u) * 32u);
    long signed_value = signed_inputs[index];
    ulong unsigned_value = unsigned_inputs[index];
    uint counter = 0u;
    {''.join(calls)}
    outputs[index * {fields}u + {fields + 3}u] = ulong(counter);
}}
"""


def _request(root, target, shape):
    source, descriptor, package = _package(
        root, target, "ulong", shape, source=_source(shape)
    )
    size = math.prod(shape)
    mask = (1 << 64) - 1
    patterns = [
        0,
        1,
        mask,
        1 << 63,
        (1 << 63) - 1,
        0x123456789ABCDEF0,
        0xFEDCBA9876543210,
    ]
    signed_bits = [patterns[index % len(patterns)] for index in range(size * 3)]
    unsigned = [patterns[(index + 3) % len(patterns)] for index in range(size * 3)]
    signed = [value - (1 << 64) if value >= 1 << 63 else value for value in signed_bits]
    guard = 0xBAD01234567890AB
    wanted = [guard] * 4
    for group in range(3):
        for invocation in range(size):
            index = group * size + invocation
            active = min(32, size - invocation // 32 * 32)
            lane = invocation % 32
            for delta in OFFSETS:
                origin = index + delta if delta < active - lane else index
                wanted.extend([signed_bits[origin], unsigned[origin]])
            wanted.append(len(OFFSETS))
    wanted.extend([guard] * 4)

    def payload(dtype, values):
        return {"dtype": dtype, "shape": [len(values)], "values": values}

    inputs = _bound_values(
        descriptor,
        {
            "signed_inputs": payload("int64", signed),
            "unsigned_inputs": payload("uint64", unsigned),
            "outputs": payload("uint64", [guard] * len(wanted)),
        },
    )
    expected = _bound_values(descriptor, {"outputs": payload("uint64", wanted)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": list(shape)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("shape", SHAPES)
def test_overloaded_word_shuffles_package_and_compile(tmp_path, target, shape):
    _, request, _ = _request(tmp_path, target, shape)
    generated = request.artifact_path.read_text()
    if target == "directx":
        assert "groupshared uint2 " in generated
        assert "WaveReadLane" not in generated
        assert generated.count("counter++") == len(OFFSETS)
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("shape", SHAPES)
def test_overloaded_word_shuffles_execute(tmp_path, shape):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required overloaded shuffle execution")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target, shape)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="overloaded_shuffles",
    )


def test_overloaded_shuffle_execution_is_required_in_existing_native_jobs():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate exact partial subgroup execution"
    )
    assert "test_directx_subgroup_overloads.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step
