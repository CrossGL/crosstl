"""Integer shuffles retain payload width through project packages and native dispatch."""

import math
import os
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_partial import _canonical
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_INTEGER_SHUFFLE_RUNTIME"
KINDS = {
    "short": ("int16_t", "int16", 16, True),
    "ushort": ("uint16_t", "uint16", 16, False),
    "long": ("int64_t", "int64", 64, True),
    "ulong": ("uint64_t", "uint64", 64, False),
}
SHAPES = ((32, 1, 1), (7, 5, 1), (32, 2, 1))
OFFSETS = (0, 1, 31, 32, 65535, 1)
FIELDS = len(OFFSETS) + 1


def _source(kind, shape, target):
    size = math.prod(shape)
    width = KINDS[kind][2]
    canonical = target == "directx"
    source_type = KINDS[kind][0] if canonical else kind
    result_type = "uint64_t" if canonical else "ulong"
    mask = (1 << width) - 1
    calls = []
    for field, offset in enumerate(OFFSETS):
        calls.append(
            f"""
    {source_type} shifted{field} = wrapped(value + {source_type}(counter++ - {field}u), {offset}u);
    shifted{field} = {offset & 65535}u < active - lane ? shifted{field} : value;
    outputs[index * {FIELDS}u + {field + 4}u] = {result_type}(shifted{field}) & {mask}ul;"""
        )
    body = f"""
    uint index = group.x * {size}u + invocation;
    uint active = min(32u, {size}u - (invocation / 32u) * 32u);
    {source_type} value = inputs[index];
    uint counter = 0u;
    {''.join(calls)}
    outputs[index * {FIELDS}u + {len(OFFSETS) + 4}u] = {result_type}(counter);
"""
    if canonical:
        return f"""shader IntegerShuffles {{
    StructuredBuffer<{source_type}> inputs @register(t0);
    RWStructuredBuffer<uint64_t> outputs @register(u1);
    {source_type} shuffle_value({source_type} value, uint delta) {{
        return WaveShuffleDown(value, delta);
    }}
    {source_type} wrapped({source_type} value, uint delta) {{
        return shuffle_value(value, delta);
    }}
    compute {{
        @numthreads({shape[0]}, {shape[1]}, {shape[2]})
        void main(uint invocation @gl_LocalInvocationIndex,
                  uvec3 group @gl_WorkGroupID,
                  uint lane @gl_SubgroupInvocationID) {{
            {body}
        }}
    }}
}}"""
    shuffle = "return metal::simd_shuffle_down(value, delta);"
    if width == 64:
        shuffle = f"""
    uint2 words = as_type<uint2>(value);
    words.x = metal::simd_shuffle_down(words.x, delta);
    words.y = metal::simd_shuffle_down(words.y, delta);
    return as_type<{KINDS[kind][0]}>(words);"""
    return f"""#include <metal_stdlib>
using namespace metal;
{kind} shuffle_value({kind} value, ushort delta) {{
    {shuffle}
}}
{kind} wrapped({kind} value, ushort delta) {{ return shuffle_value(value, delta); }}
kernel void integer_shuffles(const device {kind}* inputs [[buffer(0)]],
                             device ulong* outputs [[buffer(1)]],
                             uint invocation [[thread_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {{
    {body}
}}
"""


def _values(kind, shape):
    _, _, width, signed = KINDS[kind]
    size = math.prod(shape)
    mask = (1 << width) - 1
    patterns = [0, 1, mask, 1 << (width - 1), (1 << (width - 1)) - 1]
    if width == 64:
        patterns.extend([0x123456789ABCDEF0, 0xFEDCBA9876543210])
    bits = [
        patterns[index % len(patterns)] if index % 3 else (index * 997 + 19) & mask
        for index in range(size * 3)
    ]
    values = [
        value - (1 << width) if signed and value >= 1 << (width - 1) else value
        for value in bits
    ]
    wanted = [0xBAD01234567890AB] * 4
    for group in range(3):
        for invocation in range(size):
            index = group * size + invocation
            active = min(32, size - invocation // 32 * 32)
            lane = invocation % 32
            for delta in OFFSETS:
                delta &= 65535
                wanted.append(
                    bits[index + delta] if delta < active - lane else bits[index]
                )
            wanted.append(len(OFFSETS))
    wanted.extend([0xBAD01234567890AB] * 4)
    return values, wanted


def _request(root, target, kind, shape):
    source, descriptor, package = _package(
        root,
        target,
        kind,
        shape,
        source=_source(kind, shape, target),
        source_backend="crossgl" if target == "directx" else "metal",
    )
    values, wanted = _values(kind, shape)
    dtype = KINDS[kind][1]
    if target == "opengl" and KINDS[kind][2] == 16:
        dtype = "int32" if KINDS[kind][3] else "uint32"

    def payload(dtype, data):
        return {"dtype": dtype, "shape": [len(data)], "values": data}

    inputs = {
        "inputs": payload(dtype, values),
        "outputs": payload("uint64", [0xBAD01234567890AB] * len(wanted)),
    }
    expected = _bound_values(
        descriptor,
        {"outputs": payload("uint64", wanted)},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": list(shape)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_integer_shuffles_package_and_compile(tmp_path, kind, shape, target):
    _, request, _ = _request(tmp_path, target, kind, shape)
    generated = request.artifact_path.read_text()
    if target == "directx":
        canonical = KINDS[kind][0]
        assert f"groupshared {canonical} " in generated
        assert f"[{math.prod(shape)}]" in generated
        assert "WaveReadLane" not in generated
        assert generated.count("GroupMemoryBarrierWithGroupSync();") == 2
        assert generated.count("counter++") == len(OFFSETS)
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_integer_shuffles_execute(tmp_path, kind, shape):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required integer shuffle execution")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target, kind, shape)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="integer_shuffles",
    )


@pytest.mark.parametrize("kind", KINDS)
def test_integer_shuffle_keeps_overflow_safe_canonical_offsets(tmp_path, kind):
    value_type = KINDS[kind][0]
    generated = _codegen("directx").generate_stage(
        parse(
            _canonical(
                (7, 5, 1),
                f"{value_type} value = {value_type}(inputs[invocation]); "
                f"{value_type} result = WaveShuffleDown(value, 4294967295u); "
                "outputs[invocation] = uint(result);",
            )
        ),
        "compute",
    )
    assert "delta < (activeCount - lane)" in generated
    assert "sourceValid ? lane + delta : lane" in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("operation", ["Sum", "Min", "Max", "Product"])
def test_integer_shuffle_support_does_not_enable_wider_reductions(kind, operation):
    value_type = KINDS[kind][0]
    with pytest.raises(DirectXSoftwareSubgroupError) as error:
        _codegen("directx").generate_stage(
            parse(
                _canonical(
                    (32, 1, 1),
                    f"{value_type} value = {value_type}(inputs[invocation]); "
                    f"{value_type} result = WaveActive{operation}(value); "
                    "outputs[invocation] = uint(result);",
                )
            ),
            "compute",
        )
    assert error.value.reason == "value-type-unsupported"


@pytest.mark.parametrize("kind", KINDS)
def test_integer_shuffle_still_requires_convergent_barriers(kind):
    value_type = KINDS[kind][0]
    with pytest.raises(DirectXSoftwareSubgroupError) as error:
        _codegen("directx").generate_stage(
            parse(
                _canonical(
                    (32, 1, 1),
                    f"if (invocation < 16u) {{ {value_type} value = {value_type}(inputs[invocation]); "
                    "outputs[invocation] = uint(WaveShuffleDown(value, 1u)); }",
                )
            ),
            "compute",
        )
    assert error.value.reason == "potentially-divergent-control-flow"


def test_integer_shuffle_gate_is_required_on_all_native_targets():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate exact partial subgroup execution"
    )
    assert "test_integer_subgroup_shuffles.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_integer_subgroup_shuffles.py",
        )
