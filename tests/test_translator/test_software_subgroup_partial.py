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


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_masked_half_sum_requires_source_participation_contract(target):
    source = _canonical(
        (32, 1, 1),
        """
        half value = half(0.1);
        if (invocation < 16u) { value = WaveActiveSum(value); }
        outputs[invocation] = uint(value);
    """,
    )
    with pytest.raises(ValueError) as caught:
        _codegen(target).generate_stage(parse(source), "compute")
    assert caught.value.reason == "half-masked-participation-unproven"


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "guard",
    [
        # A lane term splits each subgroup.
        "sid == 1u && invocation % 32u < 16u",
        # Only the entry's own, never-written subgroup index is trusted.
        "copy == 1u",
        "sid + invocation == 1u",
    ],
)
def test_masked_half_sum_rejects_guards_that_split_subgroups(tmp_path, target, guard):
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void split(const device uint* inputs [[buffer(0)]],
                  device uint* outputs [[buffer(1)]],
                  uint invocation [[thread_index_in_threadgroup]],
                  uint sid [[simdgroup_index_in_threadgroup]]) {{
    half value = as_type<half>(ushort(inputs[invocation]));
    uint copy = sid;
    half total = half(0);
    if ({guard}) {{ total = simd_sum(value); }}
    outputs[invocation] = uint(as_type<ushort>(total));
}}
"""
    (tmp_path / "split.metal").write_text(source, encoding="utf-8")
    options = {"software_subgroup_width": 32}
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    payload = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            output_dir="out",
            workgroup_size=(32, 3, 1),
            source_options={"metal": {"target_options": {target: options}}},
        ),
        format_output=False,
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0
    (diagnostic,) = payload["diagnostics"]
    assert diagnostic["code"] == f"project.translate.{target}-software-subgroup-invalid"
    assert (
        diagnostic["details"]["executionSpecialization"]["reason"]
        == "half-masked-participation-unproven"
    )


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


# MLX's Prod<half> reductions call simd_product on binary16 values. Every
# pattern below keeps each partial product exact and inside binary16's range
# for any association order, so Metal's unspecified order and the software
# helper's pairwise tree give the same words.
HALF_PRODUCT_PATTERNS = (
    [1.0] * 32,
    [-1.0] * 32,
    [-1.0] * 31 + [2.0],
    [2.0] * 15 + [0.5] * 17,
    [-2.0, 0.5] * 15 + [1.0, 1.0],
    [0.0] + [1.0] * 31,
    [-0.0] + [-1.0] * 31,
    [2**-24] + [2.0] * 4 + [1.0] * 27,
    [65504.0] + [-1.0] * 31,
    [math.inf] + [-1.0] * 31,
    [math.inf, 0.0] + [1.0] * 30,
    [math.nan] + [1.0] * 31,
)


def _half_product_source(size):
    return f"""#include <metal_stdlib>
using namespace metal;
uint half_bits(half value) {{
    return isnan(value) ? 0x7e00u : uint(as_type<ushort>(value));
}}
kernel void half_products(const device uint* inputs [[buffer(0)]],
                          device uint* outputs [[buffer(1)]],
                          uint invocation [[thread_index_in_threadgroup]],
                          uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    half value = as_type<half>(ushort(inputs[index]));
    outputs[index] = half_bits(simd_product(value));
}}
"""


def _half_product_reference(words):
    product = math.prod(_half_value(word) for word in words)
    return 0x7E00 if math.isnan(product) else _half_word(product)


def test_half_product_reference_is_exact_for_every_order():
    for pattern in HALF_PRODUCT_PATTERNS:
        words = [_half_word(value) for value in pattern]
        values = [_half_value(word) for word in words]
        assert values == pattern or any(math.isnan(value) for value in pattern)
        # Powers of two whose growing and shrinking factors each stay inside
        # binary16 keep every partial product of every order exact.
        finite = [abs(value) for value in values if math.isfinite(value) and value]
        assert all(math.frexp(value)[0] == 0.5 for value in finite if value != 65504)
        assert math.prod(value for value in finite if value > 1) <= 65504
        assert math.prod(value for value in finite if value < 1) >= 2**-24
    assert _half_product_reference([_half_word(-1.0)] * 31 + [0x4000]) == 0xC000
    assert _half_product_reference([0x8000] + [_half_word(-1.0)] * 31) == 0x0000
    assert _half_product_reference([0x7C00, 0] + [0x3C00] * 30) == 0x7E00


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_half_product_package_rounds_every_node(tmp_path, target):
    _, descriptor, package = _package(
        tmp_path, target, "float", (32, 1, 1), source=_half_product_source(32)
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert "simd_product(" not in generated and "WaveActiveProduct(" not in generated
    if target == "directx":
        assert "precise float16_t product = " in generated
    else:
        assert "precise float product = " in generated
        assert "= crossgl_round_half1(product);" in generated
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _compile(generated, target, compiled)


@pytest.mark.parametrize("shape", [(5, 1, 1), (32, 1, 1), (7, 5, 1), (32, 4, 1)])
def test_half_products_execute_with_partial_groups(tmp_path, shape):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required partial subgroup readbacks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    size = math.prod(shape)
    words, wanted = [], []
    for pattern in HALF_PRODUCT_PATTERNS:
        group = [_half_word(pattern[index % 32]) for index in range(size)]
        words.extend(group)
        for start in range(0, size, 32):
            lanes = group[start : start + 32]
            wanted.extend([_half_product_reference(lanes)] * len(lanes))
    wanted.extend([0xDEAD] * 32)
    source, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_half_product_source(size)
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
        dict(
            workgroupCount=[len(HALF_PRODUCT_PATTERNS), 1, 1],
            workgroupSize=list(shape),
        ),
        observe_inputs=[name for name in inputs if name not in expected],
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="half_products",
    )


# MLX's non-precise block softmax reduces binary16 values, and its bfloat16
# overloads convert float reductions back, under ``simd_group_id == 0``. A guard
# on the subgroup index alone includes each subgroup completely or not at all,
# so the predicated reductions keep the unguarded rounding order.
GUARDED_HALF_FIELDS = 5
GUARDED_HALF_PATTERNS = (
    [lane / 8 - 2 for lane in range(32)],
    [-0.0] * 31 + [-1.0],
    [0.1] * 32,
    [2**-24] * 32,
    [math.inf, -math.inf] + [0.5] * 30,
    [(-1) ** lane * (lane + 1) * 0.25 for lane in range(32)],
    [2048.0, 1.0, -2048.0, 1.0] * 8,
)


def _guarded_half_source(size):
    return f"""#include <metal_stdlib>
using namespace metal;
uint half_bits(half value) {{
    return isnan(value) ? 0x7e00u : uint(as_type<ushort>(value));
}}
kernel void guarded_half(const device uint* inputs [[buffer(0)]],
                         device uint* outputs [[buffer(1)]],
                         uint invocation [[thread_index_in_threadgroup]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint sid [[simdgroup_index_in_threadgroup]]) {{
    uint index = group.x * {size}u + invocation;
    half value = as_type<half>(ushort(inputs[index]));
    half largest = simd_max(value);
    half smallest = simd_min(value);
    half total = half(-1);
    if (sid == 1u) {{
        total = simd_sum(value);
    }}
    half guarded = half(-2);
    if (sid != 1u) {{
        guarded = simd_max(value);
    }}
    float widened = float(value);
    half narrowed = half(-3);
    if (sid == 0u) {{
        narrowed = half(simd_sum(widened));
    }}
    outputs[index * {GUARDED_HALF_FIELDS}u] = half_bits(largest);
    outputs[index * {GUARDED_HALF_FIELDS}u + 1u] = half_bits(smallest);
    outputs[index * {GUARDED_HALF_FIELDS}u + 2u] = half_bits(total);
    outputs[index * {GUARDED_HALF_FIELDS}u + 3u] = half_bits(guarded);
    outputs[index * {GUARDED_HALF_FIELDS}u + 4u] = half_bits(narrowed);
}}
"""


def _guarded_half_reference(size):
    words, wanted = [], []
    for pattern in GUARDED_HALF_PATTERNS:
        group = [_half_word(pattern[index % 32]) for index in range(size)]
        words.extend(group)
        for start in range(0, size, 32):
            lanes = group[start : start + 32]
            values = [_half_value(word) for word in lanes]
            subgroup = start // 32
            largest = lanes[values.index(max(values))]
            smallest = lanes[values.index(min(values))]
            total = _half_sum_reference(lanes) if subgroup == 1 else 0xBC00
            guarded = largest if subgroup != 1 else 0xC000
            narrowed = 0xC200
            if subgroup == 0:
                # Every pattern's binary32 sum is exact, so any summation
                # order gives the same value before the single rounding.
                exact = sum(values)
                narrowed = 0x7E00 if math.isnan(exact) else _half_word(exact)
            wanted.extend([largest, smallest, total, guarded, narrowed] * len(lanes))
    return words, wanted


def test_guarded_half_reference_covers_edges():
    _, wanted = _guarded_half_reference(96)
    fields = [
        wanted[index::GUARDED_HALF_FIELDS] for index in range(GUARDED_HALF_FIELDS)
    ]
    largest, smallest, total, guarded, narrowed = fields
    assert 0x8000 in largest and 0x7C00 in largest and 0xFC00 in smallest
    assert 0x0001 in largest and 0x0020 in total and 0x0020 in narrowed
    # Rounding every binary16 partial gives 8 for this pattern, while one
    # rounding of the exact float sum gives 16, so the converted reduction
    # cannot reuse the binary16 helper.
    assert 0x4800 in total and 0x4C00 in narrowed
    assert 0x7E00 in total and 0x7E00 in narrowed
    assert {0xBC00, 0xC000, 0xC200} <= set(total) | set(guarded) | set(narrowed)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("shape", [(32, 1, 1), (32, 3, 1), (80, 1, 1)])
def test_guarded_half_package_predicates_reductions(tmp_path, target, shape):
    _, descriptor, package = _package(
        tmp_path,
        target,
        "float",
        shape,
        source=_guarded_half_source(math.prod(shape)),
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    for name in ("simd_sum(", "simd_max(", "simd_min(", "WaveActive"):
        assert name not in generated
    if math.prod(shape) > 32 or target == "directx":
        assert generated.count("crossglSoftwareSubgroupActive") >= 6
    if target == "directx":
        assert "float16_t __crossgl_software_subgroup_max_float16_t(" in generated
        assert "float16_t __crossgl_software_subgroup_min_float16_t(" in generated
        assert "uint leftBits = uint(asuint16(result));" in generated
        if math.prod(shape) > 32:
            assert "asfloat16(uint16_t(0x8000u))" in generated
            assert "asfloat16(uint16_t(0xfc00u))" in generated
    elif math.prod(shape) > 32:
        assert "crossglSoftwareSubgroupSumHalf(" in generated
        assert "uintBitsToFloat(0x80000000u)" in generated
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _compile(generated, target, compiled)


@pytest.mark.parametrize("shape", [(32, 1, 1), (32, 3, 1), (80, 1, 1)])
def test_guarded_half_reductions_execute(tmp_path, shape):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required partial subgroup readbacks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    size = math.prod(shape)
    words, wanted = _guarded_half_reference(size)
    wanted.extend([0xDEAD] * 32)
    source, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_guarded_half_source(size)
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
        dict(
            workgroupCount=[len(GUARDED_HALF_PATTERNS), 1, 1],
            workgroupSize=list(shape),
        ),
        observe_inputs=[name for name in inputs if name not in expected],
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="guarded_half",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    observations = evidence["records"]["generated"]["details"]["inputObservations"]
    assert len(observations) == 1
    assert all(item["unchanged"] for item in observations.values())


# MLX's RMS VJP reduces two partials under one ``simd_group_id == 0`` guard,
# inside a row loop whose bound is clamped before the loop. Every invocation
# runs each predicated reduction in source order, so a later payload sees the
# earlier result, and the statements between them still run only where the
# guard held. A second guard splits a subgroup.
GUARDED_SEQUENCE_FIELDS = 4
GUARDED_SEQUENCE_GROUPS = 3


def _guarded_sequence_source(size):
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void guarded_sequence(const device uint* inputs [[buffer(0)]],
                             device uint* outputs [[buffer(1)]],
                             uint invocation [[thread_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint sid [[simdgroup_index_in_threadgroup]]) {{
    uint index = group.x * {size}u + invocation;
    float value = as_type<float>(inputs[index]);
    float first = -1.0;
    float second = -2.0;
    float scaled = -3.0;
    uint count = 5u;
    uint end = group.x + 2u;
    if (end > 3u) {{
        end = 3u;
    }}
    for (uint step = group.x; step < end; ++step) {{
        if (sid == 0u) {{
            scaled = value * float(step + 1u);
            first = simd_sum(scaled);
            scaled = first + value;
            second = simd_max(scaled);
        }}
        if (invocation < 40u) {{
            count = simd_sum(count + step);
        }}
    }}
    outputs[index * {GUARDED_SEQUENCE_FIELDS}u] = as_type<uint>(first);
    outputs[index * {GUARDED_SEQUENCE_FIELDS}u + 1u] = as_type<uint>(second);
    outputs[index * {GUARDED_SEQUENCE_FIELDS}u + 2u] = as_type<uint>(scaled);
    outputs[index * {GUARDED_SEQUENCE_FIELDS}u + 3u] = count;
}}
"""


def _float_word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _guarded_sequence_reference(size):
    words, wanted = [], []
    for group in range(GUARDED_SEQUENCE_GROUPS):
        # The first group's guarded subgroup holds -0.0 only, which the sums
        # keep only if inactive invocations contribute -0.0. Other values are
        # small integers, so every binary32 sum is exact in any order.
        values = [
            -0.0 if group == 0 and lane < 32 else float((lane * 5 + group) % 11 - 5)
            for lane in range(size)
        ]
        words.extend(_float_word(value) for value in values)
        first, second, scaled = [-1.0] * size, [-2.0] * size, [-3.0] * size
        count = [5] * size
        guarded = range(min(32, size))
        for step in range(group, min(group + 2, 3)):
            total = -0.0
            for lane in guarded:
                scaled[lane] = values[lane] * (step + 1)
                total += scaled[lane]
            for lane in guarded:
                first[lane] = total
                scaled[lane] = total + values[lane]
            largest = max(scaled[lane] for lane in guarded)
            for lane in guarded:
                second[lane] = largest
            for start in range(0, size, 32):
                active = range(start, min(start + 32, size, 40))
                partial = sum(count[lane] + step for lane in active) % 2**32
                for lane in active:
                    count[lane] = partial
        for lane in range(size):
            wanted.extend(
                [
                    _float_word(first[lane]),
                    _float_word(second[lane]),
                    _float_word(scaled[lane]),
                    count[lane],
                ]
            )
    return words, wanted


def test_guarded_sequence_reference_covers_edges():
    _, wanted = _guarded_sequence_reference(80)
    first = wanted[0::GUARDED_SEQUENCE_FIELDS]
    count = wanted[3::GUARDED_SEQUENCE_FIELDS]
    # -0.0 survives the predicated sum, untouched lanes keep their values, and
    # the clamped bound runs the last group once.
    assert first[0] == 0x80000000 and _float_word(-1.0) in first
    assert count[79] == 5 and count[39] != count[40] and count[32] != count[0]
    _, single = _guarded_sequence_reference(32)
    assert 5 not in single[3::GUARDED_SEQUENCE_FIELDS]


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("shape", [(32, 1, 1), (64, 1, 1), (80, 1, 1)])
def test_guarded_sequence_package_predicates_reductions(tmp_path, target, shape):
    _, descriptor, package = _package(
        tmp_path,
        target,
        "float",
        shape,
        source=_guarded_sequence_source(math.prod(shape)),
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    for name in ("simd_sum(", "simd_max(", "WaveActive", "subgroupAdd"):
        assert name not in generated
    # The split guard is always predicated; the subgroup guard is when the
    # group holds more than one subgroup.
    actives = generated.count("bool crossglSoftwareSubgroupActive")
    assert actives == (1 if math.prod(shape) == 32 else 2)
    lines = [line.strip() for line in generated.splitlines()]
    if math.prod(shape) > 32:
        # The second payload follows the first result under the guard.
        first = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("first = crossglSoftwareSubgroupResult")
        )
        assert lines[first + 1].startswith("scaled = ")
        assert lines[first + 2].startswith("crossglSoftwareSubgroupInput")
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _compile(generated, target, compiled)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_guarded_sequence_rejects_divergent_loop_bounds(tmp_path, target):
    # A bound clamped only where an invocation-dependent guard holds is not
    # uniform, so the loop around the collectives stays rejected.
    source = _guarded_sequence_source(64).replace(
        "if (end > 3u) {", "if (invocation > 3u) {"
    )
    (tmp_path / "products.metal").write_text(source, encoding="utf-8")
    target_options = {"software_subgroup_width": 32}
    if target == "directx":
        target_options["relative_wave_shuffle_out_of_range"] = "self"
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("products.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(64, 1, 1),
            source_options={"metal": {"target_options": {target: target_options}}},
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    messages = [item["message"] for item in report["diagnostics"]]
    assert any("statically uniform" in message for message in messages), messages


@pytest.mark.parametrize("shape", [(32, 1, 1), (64, 1, 1), (80, 1, 1)])
def test_guarded_sequence_executes(tmp_path, shape):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required partial subgroup readbacks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    size = math.prod(shape)
    words, wanted = _guarded_sequence_reference(size)
    wanted.extend([0xDEADBEEF] * 32)
    source, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_guarded_sequence_source(size)
    )

    def payload(data):
        return dict(dtype="uint32", shape=[len(data)], values=data)

    inputs = _bound_values(
        descriptor,
        {"inputs": payload(words), "outputs": payload([0xDEADBEEF] * len(wanted))},
    )
    expected = _bound_values(descriptor, {"outputs": payload(wanted)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        dict(
            workgroupCount=[GUARDED_SEQUENCE_GROUPS, 1, 1],
            workgroupSize=list(shape),
        ),
        observe_inputs=[name for name in inputs if name not in expected],
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="guarded_sequence",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    observations = evidence["records"]["generated"]["details"]["inputObservations"]
    assert len(observations) == 1
    assert all(item["unchanged"] for item in observations.values())


# MLX's dot product sums SIMD-group partials under ``if (tid < SIMD_GROUPS)``,
# which leaves part of one subgroup inactive. Inactive invocations contribute
# each reduction's exact identity (-0.0 for sums, +/-infinity for min and max),
# so an all -0.0 sum stays -0.0 and the NaN payloads of inactive invocations
# never reach a result.
MASKED_LANE_FIELDS = 3
MASKED_LANE_GROUPS = 3
MASKED_LANE_IDENTITIES = {
    "opengl": (
        "uintBitsToFloat(0x80000000u)",
        "uintBitsToFloat(0x7f800000u)",
        "uintBitsToFloat(0xff800000u)",
    ),
    "directx": (
        "asfloat(0x80000000u)",
        "asfloat(0x7f800000u)",
        "asfloat(0xff800000u)",
    ),
}


def _masked_lane_source(size):
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void masked_lanes(const device uint* inputs [[buffer(0)]],
                         device uint* outputs [[buffer(1)]],
                         uint invocation [[thread_index_in_threadgroup]],
                         uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    float value = as_type<float>(inputs[index]);
    float total = 1.0;
    float low = 2.0;
    float high = 3.0;
    if (invocation >= 8u && invocation < 24u) {{
        total = simd_sum(value);
        low = simd_min(value);
        high = simd_max(value);
    }}
    outputs[index * {MASKED_LANE_FIELDS}u] = as_type<uint>(total);
    outputs[index * {MASKED_LANE_FIELDS}u + 1u] = as_type<uint>(low);
    outputs[index * {MASKED_LANE_FIELDS}u + 2u] = as_type<uint>(high);
}}
"""


def _masked_lane_reference(size):
    active = range(8, 24)
    words, wanted = [], []
    for group in range(MASKED_LANE_GROUPS):
        values = []
        for lane in range(size):
            if lane not in active:
                # Quiet NaNs of both signs and an infinity; none may leak.
                values.append((0x7FC00001, 0xFFC00002, 0x7F800000)[lane % 3])
            elif group == 0:
                values.append(0x80000000)
            elif group == 2 and lane == 13:
                values.append(0xFF800000)
            else:
                values.append(_float_word(float(lane * 3 % 7 - 3 + group)))
        words.extend(values)
        floats = [
            struct.unpack("<f", struct.pack("<I", values[lane]))[0] for lane in active
        ]
        total = -0.0
        for value in floats:
            total += value
        # Integers and infinities keep every sum exact in any order, and no
        # active set mixes +0.0 with -0.0, so min and max are unambiguous.
        result = [
            _float_word(total),
            _float_word(min(floats)),
            _float_word(max(floats)),
        ]
        for lane in range(size):
            if lane in active:
                wanted.extend(result)
            else:
                wanted.extend([_float_word(1.0), _float_word(2.0), _float_word(3.0)])
    return words, wanted


def test_masked_lane_reference_covers_edges():
    _, wanted = _masked_lane_reference(80)
    lane = MASKED_LANE_FIELDS * 8
    group = MASKED_LANE_FIELDS * 80
    assert wanted[lane : lane + 3] == [0x80000000] * 3
    assert wanted[group + lane : group + lane + 3] == [
        _float_word(19.0),
        _float_word(-2.0),
        _float_word(4.0),
    ]
    assert wanted[2 * group + lane : 2 * group + lane + 3] == [
        0xFF800000,
        0xFF800000,
        _float_word(5.0),
    ]
    assert wanted[:3] == [_float_word(1.0), _float_word(2.0), _float_word(3.0)]


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("shape", [(32, 1, 1), (64, 1, 1), (80, 1, 1)])
def test_masked_lane_package_uses_exact_identities(tmp_path, target, shape):
    _, descriptor, package = _package(
        tmp_path,
        target,
        "float",
        shape,
        source=_masked_lane_source(math.prod(shape)),
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    for name in ("simd_sum(", "simd_min(", "simd_max(", "WaveActive", "subgroupAdd"):
        assert name not in generated
    assert generated.count("bool crossglSoftwareSubgroupActive") == 1
    lines = [line.strip() for line in generated.splitlines()]
    inputs = [
        line.split(" = ", 1)[1].rstrip(";")
        for line in lines
        if line.startswith("float crossglSoftwareSubgroupInput")
    ]
    assert tuple(inputs) == MASKED_LANE_IDENTITIES[target]
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _compile(generated, target, compiled)


@pytest.mark.parametrize("shape", [(32, 1, 1), (64, 1, 1), (80, 1, 1)])
def test_masked_lanes_execute_with_exact_identities(tmp_path, shape):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required partial subgroup readbacks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    size = math.prod(shape)
    words, wanted = _masked_lane_reference(size)
    wanted.extend([0xDEADBEEF] * 32)
    source, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_masked_lane_source(size)
    )

    def payload(data):
        return dict(dtype="uint32", shape=[len(data)], values=data)

    inputs = _bound_values(
        descriptor,
        {"inputs": payload(words), "outputs": payload([0xDEADBEEF] * len(wanted))},
    )
    expected = _bound_values(descriptor, {"outputs": payload(wanted)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        dict(
            workgroupCount=[MASKED_LANE_GROUPS, 1, 1],
            workgroupSize=list(shape),
        ),
        observe_inputs=[name for name in inputs if name not in expected],
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="masked_lanes",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    observations = evidence["records"]["generated"]["details"]["inputObservations"]
    assert len(observations) == 1
    assert all(item["unchanged"] for item in observations.values())


# MLX's one-pass attention walks keys with ``for (i = simd_gid; i < N; i +=
# BN)``, BN being the number of SIMD groups, and reduces each key's score under
# a mask. Subgroups then disagree on the trip count; the lowering runs the
# workgroup's rounds in every invocation and predicates each key. With
# 2 * groups + 1 keys the first subgroup takes three rounds and the others two,
# and the masked keys differ between workgroups.
STRIDED_FIELDS = 3
STRIDED_GROUPS = 3


def _strided_subgroups(size):
    return -(-size // 32)


def _strided_keys(size):
    return 2 * _strided_subgroups(size) + 1


def _strided_source(size):
    groups = _strided_subgroups(size)
    total = size * STRIDED_GROUPS
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void strided_rounds(const device uint* inputs [[buffer(0)]],
                           device uint* outputs [[buffer(1)]],
                           const constant int& count [[buffer(2)]],
                           uint invocation [[thread_index_in_threadgroup]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint sid [[simdgroup_index_in_threadgroup]]) {{
    uint index = group.x * {size}u + invocation;
    float value = as_type<float>(inputs[index]);
    float total = -1.0;
    float last = -2.0;
    uint rounds = 0u;
    uint row = {total}u + group.x * uint(count);
    for (int i = sid; i < count; i += {groups}) {{
        bool use = true;
        if (inputs[row + uint(i)] == 0u) {{
            use = false;
        }}
        if (use) {{
            float score = -0.0;
            score += value * float(i + 1);
            score = simd_sum(score);
            last = score;
            total += score;
        }}
        rounds += 1u;
    }}
    outputs[index * {STRIDED_FIELDS}u] = as_type<uint>(total);
    outputs[index * {STRIDED_FIELDS}u + 1u] = as_type<uint>(last);
    outputs[index * {STRIDED_FIELDS}u + 2u] = rounds;
}}
"""


def _strided_reference(size):
    groups = _strided_subgroups(size)
    keys = _strided_keys(size)
    # The first subgroup of the first workgroup holds -0.0 only, so its scores
    # stay -0.0 only if the predicated sum keeps the sign. Small integer
    # products sum exactly in any order.
    values = [
        -0.0 if index < 32 else float(index * 5 % 7 - 3)
        for index in range(size * STRIDED_GROUPS)
    ]
    keep = [
        int((group + key) % 3 != 1)
        for group in range(STRIDED_GROUPS)
        for key in range(keys)
    ]
    words = [_float_word(value) for value in values] + keep
    wanted = []
    for group in range(STRIDED_GROUPS):
        for invocation in range(size):
            sid = invocation // 32
            first = group * size + sid * 32
            lanes = range(first, group * size + min(size, sid * 32 + 32))
            total, last, rounds = -1.0, -2.0, 0
            for key in range(sid, keys, groups):
                if keep[group * keys + key]:
                    score = -0.0
                    for lane in lanes:
                        score += -0.0 + values[lane] * (key + 1)
                    last = score
                    total += score
                rounds += 1
            wanted += [_float_word(total), _float_word(last), rounds]
    return words, wanted


def test_strided_reference_covers_divergent_rounds():
    for size in (64, 80, 1024):
        _, wanted = _strided_reference(size)
        rounds = wanted[2::STRIDED_FIELDS]
        groups = _strided_subgroups(size)
        # The first subgroup takes one more round than the others.
        assert [rounds[sid * 32] for sid in range(groups)] == [3] + [2] * (groups - 1)
        assert wanted[1] == 0x80000000
        lasts = wanted[1::STRIDED_FIELDS]
        assert len(set(lasts)) > 3


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("shape", [(64, 1, 1), (80, 1, 1), (1024, 1, 1)])
def test_strided_package_predicates_rounds(tmp_path, target, shape):
    size = math.prod(shape)
    _, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_strided_source(size)
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    for name in ("simd_sum(", "WaveActive", "subgroupAdd"):
        assert name not in generated
    groups = _strided_subgroups(size)
    assert f"crossglSoftwareSubgroupBase += {groups})" in generated
    assert generated.count("bool crossglSoftwareSubgroupLoopActive") == 1
    assert generated.count("bool crossglSoftwareSubgroupActive") == 1
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _compile(generated, target, compiled)


@pytest.mark.parametrize("shape", [(64, 1, 1), (80, 1, 1), (1024, 1, 1)])
def test_strided_rounds_execute(tmp_path, shape):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required partial subgroup readbacks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    size = math.prod(shape)
    words, wanted = _strided_reference(size)
    wanted.extend([0xDEADBEEF] * 32)
    source, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_strided_source(size)
    )

    def payload(data):
        return dict(dtype="uint32", shape=[len(data)], values=data)

    inputs = _bound_values(
        descriptor,
        {"inputs": payload(words), "outputs": payload([0xDEADBEEF] * len(wanted))},
    )
    # DirectX prefixes constant members with the entry name.
    (count,) = (
        binding["name"]
        for binding in descriptor["bindings"]
        if binding["scalarLayout"].get("memberName", binding["name"]).endswith("count")
    )
    inputs[count] = dict(dtype="int32", shape=[1], values=[_strided_keys(size)])
    expected = _bound_values(descriptor, {"outputs": payload(wanted)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        dict(workgroupCount=[STRIDED_GROUPS, 1, 1], workgroupSize=list(shape)),
        observe_inputs=[name for name in inputs if name not in expected],
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="strided_rounds",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    observations = evidence["records"]["generated"]["details"]["inputObservations"]
    assert len(observations) == 2
    assert all(item["unchanged"] for item in observations.values())
