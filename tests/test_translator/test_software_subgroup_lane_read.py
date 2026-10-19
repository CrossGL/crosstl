"""Exact-width lane reads preserve payload bits and logical subgroup boundaries."""

import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _canonical, _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SOFTWARE_SUBGROUP_LANE_READ"
KINDS = ("uint", "int", "float", "uint2")


def test_lane_read_gate_is_required_on_native_demo_platforms():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate software subgroup products"
    )
    assert "tests/test_translator/test_software_subgroup_lane_read.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step
    assert (
        "--timeout-seconds 300" in step
        and "--junitxml=" in step
        and "--basetemp=" in step
    )


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "mutation",
    ("payloadCounter++", "++payloadCounter", "payloadCounter--", "--payloadCounter"),
)
def test_lane_read_compiles_with_single_argument_evaluation(
    tmp_path, target, kind, mutation
):
    source = _canonical(
        f"uint payloadCounter = invocation; uint laneCounter = invocation; "
        f"{kind} result = read_value({kind}({mutation}), laneCounter++ % 32u); "
        "results[invocation] = uint(result" + (".x" if kind == "uint2" else "") + ");",
        f"{kind} read_value({kind} value, uint index) {{ return WaveReadLaneAt(value, index); }}",
    )
    generator = _codegen(target)
    generated = generator.generate_stage(parse(source), "compute")
    assert generated.count(mutation) == generated.count("laneCounter++") == 1
    assert "WaveReadLaneAt(" not in generated and "subgroupShuffle(" not in generated
    assert "bool sourceValid = index < 32u;" in generated
    assert "sourceValid ? index : lane" in generated
    assert "subgroupBase + sourceLane" in generated
    _compile(generated, target, tmp_path)
    assert generator.generate_stage(parse(source), "compute") == generated


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize(
    "body",
    (
        "if (invocation < 32u) { uint value = READ; results[invocation] = value; }",
        "if (invocation == 0u) { return; } uint value = READ;",
        "uint value = invocation == 0u ? READ : 0u;",
        "for (uint i = 0u; i < invocation; ++i) { uint value = READ; }",
    ),
)
def test_lane_read_still_rejects_unproven_barrier_participation(target, body):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(_canonical(body.replace("READ", "WaveReadLaneAt(invocation, 0u)"))),
            "compute",
        )


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize("index", ("true", "1.0", "uint2(0u)"))
def test_lane_read_rejects_non_scalar_integer_indices(target, index):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(_canonical(f"uint value = WaveReadLaneAt(invocation, {index});")),
            "compute",
        )


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize("kind", ("bool", "double", "float2"))
def test_lane_read_rejects_unsupported_payloads(target, kind):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(
                _canonical(
                    f"{kind} value = {kind}(1); {kind} result = WaveReadLaneAt(value, 0u);"
                )
            ),
            "compute",
        )


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize("kind", ("half", "float16_t"))
def test_lane_read_moves_binary16_payloads(tmp_path, target, kind):
    # binary16 registers move between lanes unchanged on both targets.
    generated = _codegen(target).generate_stage(
        parse(
            _canonical(
                f"{kind} value = {kind}(1); {kind} result = WaveReadLaneAt(value, 0u); "
                "results[invocation] = uint(result);"
            )
        ),
        "compute",
    )
    assert "WaveReadLaneAt" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("opengl", "directx"))
@pytest.mark.parametrize("arguments", ("", "invocation", "invocation, 0u, 1u"))
def test_lane_read_rejects_wrong_arity(target, arguments):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(_canonical(f"uint result = WaveReadLaneAt({arguments});")), "compute"
        )


def _source(kind, size):
    load = (
        "uint2(inputWords[index * 2u], inputWords[index * 2u + 1u])"
        if kind == "uint2"
        else f"as_type<{kind}>(inputWords[index * 2u])"
    )
    encode_a = "a" if kind == "uint2" else "uint2(as_type<uint>(a), 0u)"
    encode_b = "b" if kind == "uint2" else "uint2(as_type<uint>(b), 0u)"
    return f"""#include <metal_stdlib>
using namespace metal;
{kind} read_value({kind} value, uint sourceLane) {{
    return simd_shuffle(value, sourceLane);
}}
kernel void lane_reads(device uint* inputWords [[buffer(0)]],
                       device uint* outputWords [[buffer(1)]],
                       uint invocation [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    uint lane = invocation % 32u;
    uint activeCount = min(32u, {size}u - (invocation - lane));
    {kind} value = {load};
    uint payloadCounter = 0u;
    uint laneCounter = 0u;
    for (uint round = 0u; round < 3u; ++round) {{
        uint sourceLane = (lane * 7u + round) % activeCount;
        {kind} a = read_value(payloadCounter++ == round * 2u ? value : {kind}(0),
                            laneCounter++ == round * 2u ? sourceLane : 0u);
        {kind} b = simd_broadcast(payloadCounter++ == round * 2u + 1u ? value : {kind}(0),
                                 laneCounter++ == round * 2u + 1u ? activeCount - 1u : 0u);
        uint2 aWords = {encode_a};
        uint2 bWords = {encode_b};
        outputWords[index * 18u + round * 4u] = aWords.x;
        outputWords[index * 18u + round * 4u + 1u] = aWords.y;
        outputWords[index * 18u + round * 4u + 2u] = bWords.x;
        outputWords[index * 18u + round * 4u + 3u] = bWords.y;
    }}
    uint2 replicated = uint2(payloadCounter++);
    uint2 decremented = uint2(--payloadCounter);
    outputWords[index * 18u + 12u] = payloadCounter;
    outputWords[index * 18u + 13u] = laneCounter;
    outputWords[index * 18u + 14u] = replicated.x;
    outputWords[index * 18u + 15u] = replicated.y;
    outputWords[index * 18u + 16u] = decremented.x;
    outputWords[index * 18u + 17u] = decremented.y;
}}
"""


def _inputs(count):
    special = (
        0x80000000,
        0,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        0xFFC05678,
        0xFFFFFFFF,
        0x80000001,
    )
    return [
        special[i % len(special)] if i % 3 == 0 else (0x3F000000 + i * 1009)
        for i in range(count * 2)
    ]


def _expected(words, kind, size, groups=3):
    result = []
    for group in range(groups):
        for invocation in range(size):
            lane = invocation % 32
            base = group * size + invocation - lane
            active = min(32, size - (invocation - lane))
            for round_number in range(3):
                for source_lane in ((lane * 7 + round_number) % active, active - 1):
                    source = (base + source_lane) * 2
                    result.extend(
                        (words[source], words[source + 1] if kind == "uint2" else 0)
                    )
            result.extend((6, 6, 6, 6, 6, 6))
    return result


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("kind", KINDS)
def test_lane_read_translates_through_project_packages(tmp_path, target, kind):
    _, descriptor, package = _package(
        tmp_path, target, kind, (7, 5, 1), source=_source(kind, 35)
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    if target != "metal":
        assert (
            "WaveReadLaneAt(" not in generated and "subgroupShuffle(" not in generated
        )
        assert "index < activeCount" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("shape", ((32, 1, 1), (7, 5, 1), (32, 4, 1)))
@pytest.mark.parametrize("kind", KINDS)
def test_lane_read_executes_on_device(tmp_path, shape, kind):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native lane reads")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path, target, kind, shape, source=_source(kind, size)
    )
    words = _inputs(size * 3)
    expected_words = _expected(words, kind, size)
    guards = [0xBAD00000 + i for i in range(17)]

    def payload(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    inputs = {
        "inputWords": payload(words),
        "outputWords": payload([0xDEADBEEF] * len(expected_words) + guards),
    }
    expected = _bound_values(
        descriptor,
        {"inputWords": payload(words), "outputWords": payload(expected_words + guards)},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        {
            name: {k: v for k, v in data.items() if k != "values"}
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
                entry_point="lane_reads",
            )
            records["originalMetal"] = {
                "outputs": executor.runtime_adapter.runtime.dispatch(
                    None, state, native
                ),
                "moduleSha256": (
                    hashlib.sha256(original_module.read_bytes()).hexdigest()
                ),
            }
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "kind": kind,
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
            assert record["outputs"] == expected
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
