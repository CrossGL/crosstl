"""Product reductions preserve logical lanes, rounding and scratch reuse."""

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
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from crosstl.translator import parse
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_votes import _canonical, _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SOFTWARE_SUBGROUP_PRODUCT"


def test_product_gate_is_required_on_every_native_target():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate software subgroup products"
    )
    assert "test_software_subgroup_product.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step
    assert (
        "--timeout-seconds 180" in step
        and "--junitxml=" in step
        and "--basetemp=" in step
    )
    for event in ("pull_request", "push"):
        assert (
            "tests/test_translator/test_software_subgroup_product.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )


def _word(value):
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return _word(math.copysign(math.inf, value))


def _float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _input_words(kind):
    if kind != "float":
        return [
            value & 0xFFFFFFFF
            for group in range(16)
            for value in (
                ([1] * 31 + [0])
                if group == 0
                else (
                    ([-1] * group + [1] * (32 - group))
                    if group < 8
                    else [
                        0x7FFFFFFF if lane % 3 else lane + group for lane in range(32)
                    ]
                )
            )
        ]
    patterns = [
        [1.0] * 32,
        [-1.0] + [1.0] * 31,
        [-1.0] * 32,
        [0.0] + [1.0] * 31,
        [-0.0] + [1.0] * 31,
        [-0.0, -1.0] + [1.0] * 30,
        [math.nan] + [1.0] * 31,
        [math.nan] * 32,
        [math.inf] + [1.0] * 31,
        [-math.inf] + [1.0] * 31,
        [math.inf, 0.0] + [1.0] * 30,
        [2.0, 0.5] * 16,
        [0.5] * 32,
        [2.0] * 32,
        [1.0 + (lane - 16) * 2.0**-12 for lane in range(32)],
        [2.0**100, 2.0**100, 2.0**-100, 2.0**-100] + [1.0] * 28,
    ]
    return [_word(value) for pattern in patterns for value in pattern]


def _reference(words, kind):
    lanes = list(words)
    while len(lanes) > 1:
        lanes = [
            _word(_float(a) * _float(b)) if kind == "float" else (a * b) & 0xFFFFFFFF
            for a, b in zip(lanes[::2], lanes[1::2])
        ]
    return lanes[0]


def test_product_reference_preserves_adjacent_pair_rounding():
    words = _input_words("float")
    assert _reference(words[14 * 32 : 15 * 32], "float") == 0x3F7EFB30
    assert _reference(words[15 * 32 :], "float") & 0x7FFFFFFF > 0x7F800000
    assert _reference([0xFFFFFFFF] * 31 + [1], "int") == 0xFFFFFFFF


def _source(kind, size):
    cast = (
        f"as_type<{kind}>(inputWords[index])" if kind != "uint" else "inputWords[index]"
    )
    return f"""#include <metal_stdlib>
using namespace metal;
{kind} product_value({kind} value) {{ return simd_product(value); }}
{kind} wrapped_product({kind} value) {{ return product_value(value); }}
kernel void products(device uint* inputWords [[buffer(0)]],
                     device uint* outputWords [[buffer(1)]],
                     uint invocation [[thread_index_in_threadgroup]],
                     uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    {kind} value = {cast};
    uint counter = 0u;
    {kind} first = wrapped_product(counter++ == 0u ? value : {kind}(1));
    {kind} second = wrapped_product(counter++ == 1u ? value : {kind}(1));
    outputWords[index * 3u] = as_type<uint>(first);
    outputWords[index * 3u + 1u] = as_type<uint>(second);
    outputWords[index * 3u + 2u] = counter;
}}
"""


def _package(
    root,
    target,
    kind,
    shape,
    *,
    source=None,
    software_subgroups=True,
    index_range_assertions=(),
    workgroup_access_assertions=(),
):
    if source is None:
        source = _source(kind, math.prod(shape))
    (root / "products.metal").write_text(source, encoding="utf-8")
    options = {}
    if target != "metal" and software_subgroups:
        target_options = {"software_subgroup_width": 32}
        if target == "directx":
            target_options["relative_wave_shuffle_out_of_range"] = "self"
        options = {"metal": {"target_options": {target: target_options}}}
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("products.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=shape,
            source_options=options,
            index_range_assertions=index_range_assertions,
            workgroup_access_assertions=workgroup_access_assertions,
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    (root / "artifacts.json").write_text(json.dumps(manifest))
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    return source, descriptor, package


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
def test_product_compiles_and_evaluates_operand_once(tmp_path, target, kind):
    generated = _codegen(target).generate_stage(
        parse(
            _canonical(
                f"uint counter = invocation; {kind} value = WaveActiveProduct({kind}(counter++)); results[invocation] = uint(value);"
            )
        ),
        "compute",
    )
    assert generated.count("counter++") == 1
    assert "WaveActiveProduct(" not in generated and "subgroupMul(" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "body",
    [
        "if (invocation < 16u) { uint value = WaveActiveProduct(invocation); }",
        "if (invocation == 0u) { return; } uint value = WaveActiveProduct(invocation);",
        "uint value = invocation > 0u ? WaveActiveProduct(invocation) : 0u;",
    ],
)
def test_product_rejects_unproven_barrier_participation(target, body):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(parse(_canonical(body)), "compute")


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "kind",
    ["bool", "double", "float2", "int64_t", "half", "float16_t", "uint16_t", "int8_t"],
)
def test_product_rejects_unsupported_payloads(target, kind):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(
                _canonical(
                    f"{kind} input = {kind}(1); {kind} result = WaveActiveProduct(input);"
                )
            ),
            "compute",
        )


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
def test_product_translates_through_project_packages(tmp_path, target, kind):
    _, descriptor, package = _package(tmp_path, target, kind, (32, 4, 1))
    _compile(
        (package / descriptor["artifact"]["packagePath"]).read_text(), target, tmp_path
    )


def _check_outputs(actual, expected, names, kind):
    assert actual[names["inputWords"]] == expected[names["inputWords"]]
    output = actual[names["outputWords"]]
    wanted = expected[names["outputWords"]]
    assert output["dtype"] == wanted["dtype"] and output["shape"] == wanted["shape"]
    assert len(output["values"]) == len(wanted["values"])
    for index, (got, want) in enumerate(zip(output["values"], wanted["values"])):
        if (
            kind == "float"
            and index < 512 * 3
            and index % 3 < 2
            and want & 0x7FFFFFFF > 0x7F800000
        ):
            assert got & 0x7FFFFFFF > 0x7F800000, (index, got, want)
        else:
            assert got == want, (index, got, want)


@pytest.mark.parametrize("shape", [(32, 1, 1), (32, 4, 1)])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
def test_product_executes_on_device(tmp_path, shape, kind):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native products")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, descriptor, package = _package(tmp_path, target, kind, shape)
    words = _input_words(kind)
    guards = [0xBAD00000 + index for index in range(17)]
    expected_words = []
    for start in range(0, len(words), 32):
        product = _reference(words[start : start + 32], kind)
        expected_words.extend([product, product, 2] * 32)

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
    names = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding["name"]
        for binding in descriptor["bindings"]
    }
    readbacks = {
        name: {key: value for key, value in data.items() if key != "values"}
        for name, data in expected.items()
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        readbacks,
        {
            "workgroupCount": [len(words) // math.prod(shape), 1, 1],
            "workgroupSize": list(shape),
        },
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(request.artifact_path.read_text(), target, compiled)
    assert module.is_file() and module.stat().st_size
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
            _check_outputs(record["outputs"], expected, names, kind)
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
