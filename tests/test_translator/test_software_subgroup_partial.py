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

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from tests.ci_helpers import assert_native_step_prerequisites, assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
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
    assert_native_step_prerequisites(step, job="portable-host")
    assert "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step and "test_software_subgroup_partial.py" in step
    assert "-n auto" in step and "--basetemp=" in step and "--junitxml=" in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_software_subgroup_partial.py",
        )


def _half_word(value):
    try:
        return int.from_bytes(struct.pack("<e", value), "little")
    except OverflowError:
        return 0xFC00 if value < 0 else 0x7C00


def _half_value(word):
    return struct.unpack("<e", word.to_bytes(2, "little"))[0]


def _half_sum_reference(words):
    lanes = list(words)
    while len(lanes) > 1:
        lanes = [
            (
                _half_word(_half_value(lanes[index]) + _half_value(lanes[index + 1]))
                if index + 1 < len(lanes)
                else lanes[index]
            )
            for index in range(0, len(lanes), 2)
        ]
    result = lanes[0]
    return 0x7E00 if result & 0x7FFF > 0x7C00 else result


def _half_sum_source(size):
    return f"""#include <metal_stdlib>
using namespace metal;
half sum_value(half value) {{ return simd_sum(value); }}
half wrapped_sum(half value) {{ return sum_value(value); }}
uint half_bits(half value) {{
    return isnan(value) ? 0x7e00u : uint(as_type<ushort>(value));
}}
kernel void half_sums(const device uint* inputs [[buffer(0)]],
                     device uint* outputs [[buffer(1)]],
                     uint invocation [[thread_index_in_threadgroup]],
                     uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    half value = as_type<half>(ushort(inputs[index]));
    uint counter = 0u;
    half first = wrapped_sum(counter++ == 0u ? value : half(0));
    half second = wrapped_sum(counter++ == 1u ? value : half(0));
    outputs[index * 3u] = half_bits(first);
    outputs[index * 3u + 1u] = half_bits(second);
    outputs[index * 3u + 2u] = counter;
}}
"""


def test_half_sum_reference_rounds_intermediate_results():
    assert _half_sum_reference([_half_word(0.1)] * 32) == 0x4266
    assert (
        _half_sum_reference([_half_word(value) for value in [2048, 1, -2048, 1] * 8])
        == 0x4800
    )
    assert _half_sum_reference([0x8000] * 31) == 0x8000
    assert _half_sum_reference([1] * 32) == 32
    assert _half_sum_reference([0x7BFF] * 32) == 0x7C00


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_half_and_float_sums_keep_separate_helpers(tmp_path, target):
    source = _canonical(
        (32, 1, 1),
        """
        half narrow = half(invocation);
        float wide = float(invocation);
        half first = WaveActiveSum(narrow);
        float second = WaveActiveSum(wide);
        outputs[invocation] = uint(first) + uint(second);
    """,
    )
    generated = _codegen(target).generate_stage(parse(source), "compute")
    assert (
        "scratch_float16_t" in generated
        if target == "directx"
        else "ScratchHalf" in generated
    )
    assert (
        "scratch_float[" in generated
        if target == "directx"
        else "ScratchFloat[" in generated
    )
    _compile(generated, target, tmp_path)


def test_masked_half_sum_requires_source_participation_contract():
    source = _canonical(
        (32, 1, 1),
        """
        half value = half(0.1);
        if (invocation < 16u) { value = WaveActiveSum(value); }
        outputs[invocation] = uint(value);
    """,
    )
    with pytest.raises(ValueError) as caught:
        _codegen("opengl").generate_stage(parse(source), "compute")
    assert caught.value.reason == "half-masked-participation-unproven"


def test_project_reports_unproven_masked_half_participation(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void masked_half(const device uint* inputs [[buffer(0)]],
                        device uint* outputs [[buffer(1)]],
                        uint invocation [[thread_index_in_threadgroup]]) {
    half value = as_type<half>(ushort(inputs[invocation]));
    if (invocation % 32u < 16u) { value = simd_sum(value); }
    outputs[invocation] = uint(as_type<ushort>(value));
}
"""
    (tmp_path / "masked.metal").write_text(source, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        targets=("opengl",),
        output_dir="out",
        workgroup_size=(32, 3, 1),
        source_options={
            "metal": {"target_options": {"opengl": {"software_subgroup_width": 32}}}
        },
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0
    (diagnostic,) = payload["diagnostics"]
    assert diagnostic["code"] == "project.translate.opengl-software-subgroup-invalid"
    assert (
        diagnostic["details"]["executionSpecialization"]["reason"]
        == "half-masked-participation-unproven"
    )


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("shape", [(32, 1, 1), (7, 5, 1), (32, 4, 1)])
def test_half_sum_package_preserves_typed_scratch_and_rounding(tmp_path, target, shape):
    _, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_half_sum_source(math.prod(shape))
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert "simd_sum(" not in generated and "WaveActiveSum(" not in generated
    if target == "directx":
        assert "groupshared float16_t" in generated
        assert "precise float16_t sum" in generated
    else:
        assert "crossglSoftwareSubgroupSumHalf" in generated
        assert "precise float sum" in generated
        assert "= crossgl_round_half1(sum);" in generated
    assert "stride <<= 1u" in generated
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _compile(generated, target, compiled)


@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (5, 1, 1), (31, 1, 1), (32, 1, 1), (7, 5, 1), (32, 4, 1)]
)
def test_half_sums_execute_with_partial_groups_and_scratch_reuse(tmp_path, shape):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required partial subgroup readbacks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    size = math.prod(shape)
    patterns = (
        [1.0] * 32,
        [0.1] * 32,
        [0.0] * 32,
        [-0.0] * 32,
        [2**-24] * 32,
        [-(2**-24)] * 32,
        [65504.0] * 32,
        [-65504.0] * 32,
        [math.inf, -math.inf] + [0.0] * 30,
        [math.nan] + [1.0] * 31,
        [2048.0, 1.0, -2048.0, 1.0] * 8,
        [float(lane - 16) / 16 for lane in range(32)],
    )
    words, wanted = [], []
    for pattern in patterns:
        group = [_half_word(pattern[index % 32]) for index in range(size)]
        words.extend(group)
        for start in range(0, size, 32):
            lanes = group[start : start + 32]
            result = _half_sum_reference(lanes)
            wanted.extend([result, result, 2] * len(lanes))
    wanted.extend([0xDEAD] * 32)
    source, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_half_sum_source(size)
    )

    def payload(data):
        return dict(dtype="uint32", shape=[len(data)], values=data)

    inputs = _bound_values(
        descriptor,
        {"inputs": payload(words), "outputs": payload([0xDEAD] * len(wanted))},
    )
    expected = _bound_values(descriptor, {"outputs": payload(wanted)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        dict(workgroupCount=[len(patterns), 1, 1], workgroupSize=list(shape)),
        observe_inputs=[name for name in inputs if name not in expected],
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="half_sums",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    observations = evidence["records"]["generated"]["details"]["inputObservations"]
    assert len(observations) == 1
    assert all(item["unchanged"] for item in observations.values())
