"""Shuffle-and-fill preserves segments, fill-lane payloads and scratch reuse."""

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
from tests.test_translator.test_software_subgroup_lane_read import _inputs
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _canonical, _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SOFTWARE_SUBGROUP_SHUFFLE_FILL"
KINDS = ("uint", "int", "float", "bool", "uint2")
CASES = tuple(
    (modulo, delta)
    for modulo in (None, 2, 4, 8, 16, 32)
    for delta in (0, 1, (modulo or 32) - 1, modulo or 32)
)


def test_shuffle_fill_gate_is_required_on_native_demo_platforms():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate software subgroup products"
    )
    assert "tests/test_translator/test_software_subgroup_shuffle_fill.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step
    assert (
        "--timeout-seconds 180" in step
        and "--basetemp=" in step
        and "--junitxml=" in step
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("modulo", ("", ", moduloCounter++"))
def test_shuffle_fill_compiles_and_evaluates_each_argument_once(
    tmp_path, target, kind, modulo
):
    source = _canonical(
        "uint valueCounter = invocation; uint fillCounter = invocation; "
        "uint deltaCounter = 1u; uint moduloCounter = 8u; "
        f"{kind} result = WaveShuffleAndFillUp({kind}(valueCounter++), "
        f"{kind}(fillCounter++), deltaCounter++{modulo});"
    )
    generator = _codegen(target)
    generated = generator.generate_stage(parse(source), "compute")
    for name in ("valueCounter", "fillCounter", "deltaCounter"):
        assert generated.count(f"{name}++") == 1
    assert generated.count("moduloCounter++") == bool(modulo)
    assert "WaveReadLaneAt(" not in generated and "subgroupShuffle(" not in generated
    assert "delta <= segmentWidth" in generated
    assert "sourceLane = sourceValid ? sourceLane : lane" in generated
    assert "lane + segmentWidth - delta" in generated
    _compile(generated, target, tmp_path)
    assert generator.generate_stage(parse(source), "compute") == generated


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "arguments",
    (
        "",
        "1u",
        "1u, 2u",
        "1u, 2u, 1u, 8u, 0u",
        "1u, 1.0, 1u",
        "1u, 1u, true",
        "1u, 1u, 1u, uint2(8u)",
        "float2(1.0), float2(2.0), 1u",
        "double(1.0), double(2.0), 1u",
    ),
)
def test_shuffle_fill_rejects_unsupported_signatures(target, arguments):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(_canonical(f"uint result = WaveShuffleAndFillUp({arguments});")),
            "compute",
        )


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "body",
    (
        "if (invocation < 32u) { uint result = READ; }",
        "if (invocation == 0u) { return; } uint result = READ;",
        "uint result = invocation == 0u ? READ : 0u;",
        "for (uint i = 0u; i < invocation; ++i) { uint result = READ; }",
    ),
)
def test_shuffle_fill_rejects_unproven_barrier_participation(target, body):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(
                _canonical(
                    body.replace("READ", "WaveShuffleAndFillUp(invocation, 0u, 1u)")
                )
            ),
            "compute",
        )


@pytest.mark.parametrize("software", (False, True))
def test_owned_metal_shuffle_fill_reads_remote_fill_lanes(tmp_path, software):
    source = _canonical(
        "uint valueCounter = invocation; uint fillCounter = 100u + invocation; "
        "uint deltaCounter = 1u; "
        "uint result = simd_shuffle_and_fill_up(valueCounter++, fillCounter++, deltaCounter++); "
        "results[invocation] = result;"
    )
    generated = _codegen("directx", software=software).generate_stage(
        parse(source), "compute"
    )
    for name in ("valueCounter", "fillCounter", "deltaCounter"):
        assert generated.count(f"{name}++") == 1
    if software:
        assert "lane + segmentWidth - delta" in generated
        assert "WaveReadLaneAt(" not in generated
    else:
        assert "WaveReadLaneAt(fill, fillSourceLane)" in generated
        assert "segmentBase + modulo + segmentLane - delta" in generated
    _compile(generated, "directx", tmp_path)


def test_owned_complex_shuffle_reads_both_payloads_before_selection(tmp_path):
    source = _canonical(
        "complex64_t value; value.real = float(invocation); value.imag = -1.0; "
        "complex64_t fill; fill.real = 100.0 + float(invocation); fill.imag = -2.0; "
        "complex64_t result = simd_shuffle_and_fill_up(value, fill, 1u); "
        "results[invocation] = asuint(result.real);",
        "struct complex64_t { float real; float imag; };",
    )
    generated = _codegen("directx", software=False).generate_stage(
        parse(source), "compute"
    )
    helper = generated.split(
        "complex64_t __crossgl_complex64_wave_shuffle_and_fill_up(", 1
    )[1].split("\n}", 1)[0]
    assert helper.index("wave_read_lane_at(value, valueLane)") < helper.index(
        "if (useValue)"
    )
    assert helper.index("wave_read_lane_at(fill, fillLane)") < helper.index(
        "if (useValue)"
    )
    assert "WaveGetLaneCount() + lane - delta" in helper
    assert "return fill;" not in helper
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("kind", ("uint", "bool", "uint2"))
def test_metal_segmented_canonical_shuffle_preserves_modulo(tmp_path, kind):
    from crosstl.translator.codegen.metal_codegen import MetalCodeGen

    source = _canonical(
        "uint deltaCounter = 1u; uint moduloCounter = 8u; "
        f"{kind} result = WaveShuffleAndFillUp({kind}(invocation), {kind}(1), deltaCounter++, moduloCounter++);"
    )
    generated = MetalCodeGen().generate_stage(parse(source), "compute")
    assert "unsupported" not in generated
    assert "ushort(moduloCounter++)" in generated
    assert generated.count("deltaCounter++") == generated.count("moduloCounter++") == 1
    _compile(generated, "metal", tmp_path)


@pytest.mark.parametrize("modulo", ("true", "1.0", "uint2(8u)"))
def test_metal_segmented_shuffle_rejects_noninteger_modulo(modulo):
    from crosstl.translator.codegen.metal_codegen import MetalCodeGen

    generated = MetalCodeGen().generate_stage(
        parse(_canonical(f"uint result = WaveShuffleAndFillUp(1u, 2u, 1u, {modulo});")),
        "compute",
    )
    assert "unsupported Metal wave intrinsic" in generated and "modulo" in generated


def _source(kind, size):
    def intrinsic(extra):
        value = "uint(value)" if kind == "bool" else "value"
        fill = "uint(fill)" if kind == "bool" else "fill"
        result = f"simd_shuffle_and_fill_up({value}, {fill}, delta{extra})"
        return f"({result} != 0u)" if kind == "bool" else result

    def load(offset):
        if kind == "uint2":
            return f"uint2(inputWords[index * 4u + {offset}u], inputWords[index * 4u + {offset + 1}u])"
        if kind == "bool":
            return f"((inputWords[index * 4u + {offset}u] & 1u) != 0u)"
        return f"as_type<{kind}>(inputWords[index * 4u + {offset}u])"

    calls = []
    for case, (modulo, delta) in enumerate(CASES):
        extra = f", ushort({modulo}u)" if modulo else ""
        result = f"fill_value(value, fill, ushort({delta}u){extra})"
        words = (
            "result"
            if kind == "uint2"
            else (
                "uint2(uint(result), 0u)"
                if kind == "bool"
                else "uint2(as_type<uint>(result), 0u)"
            )
        )
        width = modulo or 32
        calls.append(f"""
        {{
            {kind} result = {result};
            uint sourceLane = lane - lane % {width}u + (lane % {width}u + {width}u - {delta}u) % {width}u;
            uint2 words = {words};
            if (sourceLane >= activeCount) {{ words = uint2(0xdeadbeefu); }}
            uint offset = index * {len(CASES) * 4}u + round * {len(CASES) * 2}u + {case * 2}u;
            outputWords[offset] = words.x;
            outputWords[offset + 1u] = words.y;
        }}
        """)
    return f"""#include <metal_stdlib>
using namespace metal;
{kind} fill_value({kind} value, {kind} fill, ushort delta) {{
    return {intrinsic("")};
}}
{kind} fill_value({kind} value, {kind} fill, ushort delta, ushort modulo) {{
    return {intrinsic(", modulo")};
}}
kernel void shuffle_fill(device uint* inputWords [[buffer(0)]],
                         device uint* outputWords [[buffer(1)]],
                         uint invocation [[thread_index_in_threadgroup]],
                         uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    uint lane = invocation % 32u;
    uint activeCount = min(32u, {size}u - (invocation - lane));
    {kind} value = {load(0)};
    {kind} fill = {load(2)};
    for (uint round = 0u; round < 2u; ++round) {{
        {''.join(calls)}
        {kind} temporary = value;
        value = fill;
        fill = temporary;
    }}
}}
"""


def _expected(words, kind, size):
    expected = []
    for group in range(3):
        for invocation in range(size):
            lane = invocation % 32
            base = group * size + invocation - lane
            active = min(32, size - (invocation - lane))
            for round_number in range(2):
                for modulo, delta in CASES:
                    width = modulo or 32
                    segment, segment_lane = divmod(lane, width)
                    source_lane = segment * width + (segment_lane - delta) % width
                    if source_lane >= active:
                        expected.extend((0xDEADBEEF, 0xDEADBEEF))
                        continue
                    fill = (segment_lane < delta) != bool(round_number)
                    offset = (base + source_lane) * 4 + (2 if fill else 0)
                    first = words[offset] & 1 if kind == "bool" else words[offset]
                    expected.extend(
                        (first, words[offset + 1] if kind == "uint2" else 0)
                    )
    return expected


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("kind", KINDS)
def test_shuffle_fill_translates_through_project_packages(tmp_path, target, kind):
    _, descriptor, package = _package(
        tmp_path, target, kind, (7, 5, 1), source=_source(kind, 35)
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert "unsupported" not in generated
    if target != "metal":
        assert (
            "WaveReadLaneAt(" not in generated and "subgroupShuffle(" not in generated
        )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("shape", ((32, 1, 1), (7, 5, 1), (32, 4, 1)))
@pytest.mark.parametrize("kind", KINDS)
def test_shuffle_fill_executes_on_device(tmp_path, shape, kind):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native shuffle-and-fill")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path, target, kind, shape, source=_source(kind, size)
    )
    words = _inputs(size * 6)
    expected_words = _expected(words, kind, size)
    guards = [0xBAD00000 + i for i in range(17)]

    def payload(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    inputs = {
        "inputWords": payload(words),
        "outputWords": payload([0xFA110000] * len(expected_words) + guards),
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
                entry_point="shuffle_fill",
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
