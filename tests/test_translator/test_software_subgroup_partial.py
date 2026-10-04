"""Exact workgroup sizes retain only real lanes in software collectives."""

import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import (
    _check_outputs,
    _package,
)
from tests.test_translator.test_software_subgroup_votes import _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_PARTIAL_SUBGROUP_READBACKS"
SHAPES = (
    (1, 1, 1),
    (5, 1, 1),
    (31, 1, 1),
    (33, 1, 1),
    (63, 1, 1),
    (7, 5, 1),
    (1, 1, 35),
    (16, 2, 1),
    (1023, 1, 1),
)
FIELDS = 14


def _canonical(shape, body):
    return f"""shader Partial {{
        StructuredBuffer<uint> inputs @register(t0);
        RWStructuredBuffer<uint> outputs @register(u1);
        compute {{
            @numthreads({shape[0]}, {shape[1]}, {shape[2]})
            void main(uint invocation @gl_LocalInvocationIndex, uint groups @gl_NumSubgroups) {{
                {body}
            }}
        }}
    }}"""


@pytest.mark.parametrize("target", ["opengl", "directx"])
@pytest.mark.parametrize("shape", SHAPES)
def test_partial_collectives_compile_with_exact_storage(tmp_path, shape, target):
    generated = _codegen(target).generate_stage(
        parse(
            _canonical(
                shape,
                "uint value = inputs[invocation]; uint result = WaveActiveSum(value); result += WaveActiveProduct(value); result += WaveShuffleDown(value, 4294967295u); outputs[invocation] = result + groups;",
            )
        ),
        "compute",
    )
    count = math.prod(shape)
    assert f"[{count}]" in generated
    if count % 32:
        assert "activeCount" in generated
        assert "lane + stride < activeCount" in generated
    active = "activeCount" if count % 32 else "32u"
    assert f"delta < ({active} - lane)" in generated
    assert "sourceValid ? lane + delta : lane" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["opengl", "directx"])
@pytest.mark.parametrize("shape", [(5, 1, 1), (7, 5, 1)])
@pytest.mark.parametrize(
    "body",
    [
        "if (invocation == 0u) { return; } outputs[invocation] = WaveActiveSum(invocation);",
        "outputs[invocation] = invocation > 0u ? WaveActiveProduct(invocation) : 0u;",
        "for (uint i = 0u; i < invocation; ++i) { outputs[invocation] = WaveActiveSum(invocation); }",
    ],
)
def test_partial_layout_does_not_relax_barrier_participation(target, shape, body):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(parse(_canonical(shape, body)), "compute")


@pytest.mark.parametrize("target", ["opengl", "directx"])
def test_partial_layout_state_is_reset_between_generations(target):
    generator = _codegen(target)
    body = "outputs[invocation] = WaveActiveSum(invocation) + groups;"
    for shape in ((5, 1, 1), (64, 1, 1), (7, 5, 1), (32, 1, 1)):
        ast = parse(_canonical(shape, body))
        assert generator.generate_stage(ast, "compute") == _codegen(
            target
        ).generate_stage(ast, "compute")


def _source(size, kind, distant="65535"):
    read = (
        "inputWords[index]" if kind == "uint" else f"as_type<{kind}>(inputWords[index])"
    )
    factor = "2u" if kind == "uint" else f"{kind}(-1)"
    return f"""#include <metal_stdlib>
using namespace metal;
{kind} sum_value({kind} value) {{ return simd_sum(value); }}
{kind} wrapped_sum({kind} value) {{ return sum_value(value); }}
kernel void products(device uint* inputWords [[buffer(0)]],
                     device uint* outputWords [[buffer(1)]],
                     uint invocation [[thread_index_in_threadgroup]],
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sid [[simdgroup_index_in_threadgroup]],
                     uint groups [[simdgroups_per_threadgroup]],
                     uint width [[threads_per_simdgroup]]) {{
    uint index = gid.x * {size}u + invocation;
    {kind} value = {read};
    uint counter = 0u;
    {kind} first = wrapped_sum(value + {kind}(counter++));
    {kind} second = wrapped_sum(value + {kind}(counter++));
    {kind} minimum = simd_min(value);
    {kind} maximum = simd_max(value);
    {kind} factor = invocation % 3u == 0u ? {factor} : {kind}(1);
    {kind} product = simd_product(factor);
    {kind} shifted = simd_shuffle_down(value, 1);
    uint active = min(32u, {size}u - sid * 32u);
    shifted = lane + 1u < active ? shifted : value;
    {kind} distant = simd_shuffle_down(value, {distant});
    bool all_positive = simd_all(value > {kind}(0));
    bool any_positive = simd_any(value > {kind}(0));
    outputWords[index * {FIELDS}u] = as_type<uint>(first);
    outputWords[index * {FIELDS}u + 1u] = as_type<uint>(second);
    outputWords[index * {FIELDS}u + 2u] = as_type<uint>(minimum);
    outputWords[index * {FIELDS}u + 3u] = as_type<uint>(maximum);
    outputWords[index * {FIELDS}u + 4u] = as_type<uint>(product);
    outputWords[index * {FIELDS}u + 5u] = as_type<uint>(shifted);
    outputWords[index * {FIELDS}u + 6u] = as_type<uint>(distant);
    outputWords[index * {FIELDS}u + 7u] = uint(all_positive);
    outputWords[index * {FIELDS}u + 8u] = uint(any_positive);
    outputWords[index * {FIELDS}u + 9u] = lane;
    outputWords[index * {FIELDS}u + 10u] = sid;
    outputWords[index * {FIELDS}u + 11u] = groups;
    outputWords[index * {FIELDS}u + 12u] = width;
    outputWords[index * {FIELDS}u + 13u] = counter;
}}
"""


def _word(value, kind):
    return (
        struct.unpack("<I", struct.pack("<f", value))[0]
        if kind == "float"
        else value & 0xFFFFFFFF
    )


def _reference(size, kind):
    words, wanted = [], []
    for group in range(3):
        values = [
            (
                (2 + lane % 7)
                if group == 0
                else (
                    (32 + lane % 7 if kind == "uint" else -2 - lane % 7)
                    if group == 1
                    else lane % 9 - (0 if kind == "uint" else 4)
                )
            )
            for lane in range(size)
        ]
        if kind == "float":
            values = [value * 0.25 for value in values]
        words.extend(_word(value, kind) for value in values)
        for start in range(0, size, 32):
            active = values[start : start + 32]
            factors = [
                (2 if kind == "uint" else -1) if lane % 3 == 0 else 1
                for lane in range(start, start + len(active))
            ]
            for lane, value in enumerate(active):
                wanted.extend(
                    [
                        _word(sum(active), kind),
                        _word(sum(active) + len(active), kind),
                        _word(min(active), kind),
                        _word(max(active), kind),
                        _word(math.prod(factors), kind),
                        _word(active[min(lane + 1, len(active) - 1)], kind),
                        _word(value, kind),
                        int(all(item > 0 for item in active)),
                        int(any(item > 0 for item in active)),
                        lane,
                        start // 32,
                        (size + 31) // 32,
                        32,
                        2,
                    ]
                )
    return words, wanted


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
def test_partial_collectives_execute(tmp_path, shape, kind):
    _execute_collectives(tmp_path, shape, kind)


@pytest.mark.parametrize("shape", [(32, 1, 1), (64, 1, 1), (32, 4, 1)])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
@pytest.mark.parametrize("distance_from_wrap", [1, 32])
def test_shuffle_offsets_do_not_wrap(tmp_path, shape, kind, distance_from_wrap):
    distant = f"(0u - {distance_from_wrap}u * (groups / groups))"
    _execute_collectives(tmp_path, shape, kind, distant=distant)


def _execute_collectives(tmp_path, shape, kind, *, distant="65535"):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required partial subgroup readbacks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path, target, kind, shape, source=_source(size, kind, distant)
    )
    words, wanted = _reference(size, kind)
    guards = [0xBAD00000 + index for index in range(17)]

    def payload(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    inputs = {
        "inputWords": payload(words),
        "outputWords": payload([0xDEADBEEF] * len(wanted) + guards),
    }
    expected = _bound_values(
        descriptor,
        {"inputWords": payload(words), "outputWords": payload(wanted + guards)},
    )
    names = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding["name"]
        for binding in descriptor["bindings"]
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        {
            name: {key: value for key, value in data.items() if key != "values"}
            for name, data in expected.items()
        },
        {"workgroupCount": [3, 1, 1], "workgroupSize": list(shape)},
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(request.artifact_path.read_text(), target, compiled)
    executor = _executor(target)
    records = {}
    try:
        assert executor.is_available(request).available
        result = executor.run(request)
        assert result.status == "ok", result
        records["generated"] = {"outputs": result.outputs, "details": result.details}
        if target == "metal":
            original = tmp_path / "original"
            original.mkdir()
            original_source, original_module = _compile(source, target, original)
            state, native = _native_request(request)
            native = replace(
                native,
                artifact_path=original_source,
                module_path=original_module,
                entry_point="products",
            )
            actual = executor.runtime_adapter.runtime.dispatch(None, state, native)
            records["originalMetal"] = {
                "outputs": actual,
                "moduleSha256": (
                    hashlib.sha256(original_module.read_bytes()).hexdigest()
                ),
            }
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "kind": kind,
                    "distant": distant,
                    "shape": shape,
                    "inputs": inputs,
                    "expected": expected,
                    "descriptor": descriptor,
                    "records": records,
                    "sourceSha256": (
                        hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                    ),
                    "validationModuleSha256": (
                        hashlib.sha256(module.read_bytes()).hexdigest()
                    ),
                },
                indent=2,
            )
        )
        for record in records.values():
            _check_outputs(record["outputs"], expected, names, "uint")
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


def test_partial_native_gate_is_required_on_every_target():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate exact partial subgroup execution"
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step and "test_software_subgroup_partial.py" in step
    assert "-n auto" in step and "--basetemp=" in step and "--junitxml=" in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_software_subgroup_partial.py",
        )
