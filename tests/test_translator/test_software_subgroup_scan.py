"""Prefix scans preserve logical lane order, identities and scratch reuse."""

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
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import _float, _package, _word
from tests.test_translator.test_software_subgroup_votes import _canonical, _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SOFTWARE_SUBGROUP_SCAN"
OPERATIONS = (
    ("WavePrefixSum", "+", False),
    ("WavePrefixInclusiveSum", "+", True),
    ("WavePrefixProduct", "*", False),
    ("WavePrefixInclusiveProduct", "*", True),
)


def test_scan_gate_shares_required_native_product_job():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate software subgroup products"
    )
    assert "tests/test_translator/test_software_subgroup_scan.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step
    assert (
        "--timeout-seconds 180" in step
        and "--junitxml=" in step
        and "--basetemp=" in step
    )
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_software_subgroup_scan.py",
        )


def _prefixes(words, kind, operator, inclusive):
    lanes = list(words)
    stride = 1
    while stride < len(lanes):
        previous = lanes
        lanes = list(previous)
        for lane in range(len(previous)):
            if not lane & stride:
                continue
            source = (lane // (2 * stride)) * (2 * stride) + stride - 1
            left, right = previous[source], previous[lane]
            if kind == "float":
                left, right = _float(left), _float(right)
            value = left + right if operator == "+" else left * right
            lanes[lane] = _word(value) if kind == "float" else value & 0xFFFFFFFF
        stride *= 2
    identity = 0 if operator == "+" else 1
    if kind == "float":
        identity = 0x80000000 if operator == "+" else _word(1.0)
    exclusive = [identity, *lanes[:-1]]
    if not inclusive:
        return exclusive
    result = []
    for left, right in zip(exclusive, words):
        if kind == "float":
            left, right = _float(left), _float(right)
        value = left + right if operator == "+" else left * right
        result.append(_word(value) if kind == "float" else value & 0xFFFFFFFF)
    return result


def test_scan_integer_reference_matches_sequential_modular_prefixes():
    words = [0xFFFFFFFF, 0x80000000, 3, 0x7FFFFFFF, 1, 0, 19] * 4
    for operator in ("+", "*"):
        for inclusive in (False, True):
            expected = []
            accumulator = 0 if operator == "+" else 1
            for value in words:
                before = accumulator
                accumulator = (
                    accumulator + value if operator == "+" else accumulator * value
                ) & 0xFFFFFFFF
                expected.append(accumulator if inclusive else before)
            assert _prefixes(words, "uint", operator, inclusive) == expected


def test_scan_reference_preserves_identity_and_float_rounding_order():
    words = [_word(value) for value in [2**24, 1.0, -(2**24), 1.0]]
    assert [_float(v) for v in _prefixes(words, "float", "+", True)] == [
        2**24,
        2**24,
        0.0,
        1.0,
    ]
    assert _prefixes([0x80000000], "float", "+", True) == [0x80000000]
    assert _prefixes([0x80000000], "float", "+", False) == [0x80000000]
    assert _prefixes([0x80000000], "float", "*", False) == [_word(1.0)]


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
@pytest.mark.parametrize("operation,operator,inclusive", OPERATIONS)
def test_scan_compiles_and_evaluates_operand_once(
    tmp_path, target, kind, operation, operator, inclusive
):
    generator = _codegen(target)
    ast = parse(
        _canonical(
            f"uint counter = invocation; {kind} result = scan_value({kind}(counter++)); "
            "results[invocation] = uint(result);",
            f"{kind} scan_value({kind} value) {{ return {operation}(value); }}",
        )
    )
    generated = generator.generate_stage(ast, "compute")
    assert generated.count("counter++") == 1
    assert operation + "(" not in generated
    assert "subgroupInclusive" not in generated and "subgroupExclusive" not in generated
    assert "(lane & ~(2u * stride - 1u)) + stride - 1u" in generated
    assert "invocation - 1u" in generated
    assert "stride <<= 1u" in generated
    if kind == "float":
        assert "precise float result" in generated
    _compile(generated, target, tmp_path)
    assert generator.generate_stage(ast, "compute") == generated


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("operation,operator,inclusive", OPERATIONS)
@pytest.mark.parametrize(
    "body",
    [
        "if (invocation < 16u) { uint value = SCAN; }",
        "if (invocation == 0u) { return; } uint value = SCAN;",
        "uint value = invocation == 0u ? SCAN : 0u;",
        "for (uint i = 0u; i < invocation; ++i) { uint value = SCAN; }",
    ],
)
def test_scan_rejects_unproven_barrier_participation(
    target, operation, operator, inclusive, body
):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(_canonical(body.replace("SCAN", f"{operation}(invocation)"))),
            "compute",
        )


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("operation,operator,inclusive", OPERATIONS)
@pytest.mark.parametrize(
    "kind",
    ["bool", "double", "float2", "int64_t", "half", "float16_t", "uint16_t", "int8_t"],
)
def test_scan_rejects_unsupported_source_widths(
    target, operation, operator, inclusive, kind
):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(
            parse(
                _canonical(
                    f"{kind} input = {kind}(1); {kind} value = {operation}(input);"
                )
            ),
            "compute",
        )


def _source(kind, size):
    return f"""#include <metal_stdlib>
using namespace metal;
{kind} sum_value({kind} value) {{ return simd_prefix_exclusive_sum(value); }}
{kind} sum_inclusive({kind} value) {{ return simd_prefix_inclusive_sum(value); }}
{kind} product_value({kind} value) {{ return simd_prefix_exclusive_product(value); }}
{kind} product_inclusive({kind} value) {{ return simd_prefix_inclusive_product(value); }}
{kind} wrapped_sum({kind} value) {{ return sum_value(value); }}
kernel void scans(device uint* inputWords [[buffer(0)]],
                  device uint* outputWords [[buffer(1)]],
                  uint invocation [[thread_index_in_threadgroup]],
                  uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    {kind} value = as_type<{kind}>(inputWords[index]);
    uint counter = 0u;
    for (uint round = 0u; round < 2u; ++round) {{
        {kind} a = wrapped_sum(counter++ == round * 4u ? value : {kind}(0));
        {kind} b = sum_inclusive(counter++ == round * 4u + 1u ? value : {kind}(0));
        {kind} c = product_value(counter++ == round * 4u + 2u ? value : {kind}(0));
        {kind} d = product_inclusive(counter++ == round * 4u + 3u ? value : {kind}(0));
        outputWords[index * 9u + round * 4u] = as_type<uint>(a);
        outputWords[index * 9u + round * 4u + 1u] = as_type<uint>(b);
        outputWords[index * 9u + round * 4u + 2u] = as_type<uint>(c);
        outputWords[index * 9u + round * 4u + 3u] = as_type<uint>(d);
    }}
    outputWords[index * 9u + 8u] = counter;
}}
"""


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
def test_scan_translates_through_project_packages(tmp_path, target, kind):
    _, descriptor, package = _package(
        tmp_path, target, kind, (7, 5, 1), source=_source(kind, 35)
    )
    _compile(
        (package / descriptor["artifact"]["packagePath"]).read_text(), target, tmp_path
    )


def _inputs(kind, count):
    integer = [0xFFFFFFFF, 0x80000000, 3, 0x7FFFFFFF, 1, 0, 19]
    patterns = [
        [1.0, -1.0, 0.5, 2.0],
        [0.0, -0.0, -1.0, 1.0],
        [1.0 + (lane - 16) * 2.0**-12 for lane in range(32)],
        [2**24, 1.0, -(2**24), 1.0],
        [math.inf, 1.0, -math.inf, 0.0],
        [math.nan, 1.0, -1.0],
    ]
    if kind != "float":
        return [integer[index % len(integer)] for index in range(count)]
    return [
        _word(
            patterns[(index // 32) % len(patterns)][
                (index % 32) % len(patterns[(index // 32) % len(patterns)])
            ]
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("shape", [(32, 1, 1), (7, 5, 1), (32, 4, 1)])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
def test_scan_executes_on_device(tmp_path, shape, kind):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native prefix scans")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path, target, kind, shape, source=_source(kind, size)
    )
    words = _inputs(kind, size * 6)
    guards = [0xBAD00000 + index for index in range(17)]
    expected_words = []
    for group in range(6):
        for start in range(0, size, 32):
            lanes = words[group * size + start : group * size + min(start + 32, size)]
            prefixes = [
                _prefixes(lanes, kind, op, inclusive) for _, op, inclusive in OPERATIONS
            ]
            for lane in range(len(lanes)):
                results = [prefix[lane] for prefix in prefixes]
                expected_words.extend(results * 2 + [8])

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
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        {
            name: {k: v for k, v in data.items() if k != "values"}
            for name, data in expected.items()
        },
        {"workgroupCount": [6, 1, 1], "workgroupSize": list(shape)},
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
                entry_point="scans",
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
            actual = record["outputs"]
            assert actual[names["inputWords"]] == expected[names["inputWords"]]
            got = actual[names["outputWords"]]
            wanted = expected[names["outputWords"]]
            assert got["dtype"] == wanted["dtype"] and got["shape"] == wanted["shape"]
            assert len(got["values"]) == len(wanted["values"])
            for index, (a, b) in enumerate(zip(got["values"], wanted["values"])):
                if (
                    kind == "float"
                    and index < len(expected_words)
                    and index % 9 < 8
                    and b & 0x7FFFFFFF > 0x7F800000
                ):
                    assert a & 0x7FFFFFFF > 0x7F800000, (index, a, b)
                else:
                    assert a == b, (index, a, b)
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
