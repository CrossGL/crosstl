"""Read-only helper returns do not weaken collective convergence checks."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import FunctionCallNode, IdentifierNode, NamedType
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from crosstl.translator.codegen.uniform_returns import UniformReturnAnalysis
from tests.runtime_helpers import _prepare_native_package
from tests.test_translator.test_directx_software_reductions import _codegen, _source
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_UNIFORM_RETURN_RUNTIME"


def _body(expression):
    return f"""uint stop = {expression};
        if (stop == 0u) {{ return; }}
        uint value = 0u;
        for (uint step = 0u; step < stop; ++step) {{
            value += WaveActiveSum(invocation + step);
        }}
        outputWords[index] = value;"""


@pytest.mark.parametrize(
    "expression", ["bound(group.x)", "bound(group.x + 1u)", "bound(2u)"]
)
@pytest.mark.parametrize(
    "helper",
    [
        "uint bound(uint value) { return value; }",
        "uint bound(uint value) { uint copy = value + 1u; return min(copy, 3u); }",
        "uint bound(uint value) { if (value > 2u) { return 2u; } return value; }",
        "uint bound(uint value) { if (value > 2u) { return 2u; } else { return value; } }",
        "uint inner(uint value) { return value + 1u; } uint bound(uint value) { return inner(value); }",
        "uint bound(uint value) { return value > 2u ? 2u : value; }",
    ],
)
def test_read_only_return_bounds_compile(tmp_path, expression, helper):
    source = parse(_source("uint", _body(expression), helper))
    generator = _codegen()
    generated = generator.generate_stage(source, "compute")
    assert generator.generate_stage(source, "compute") == generated
    assert "WaveActiveSum" not in generated
    assert "GroupMemoryBarrierWithGroupSync" in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize(
    "expression,helper",
    [
        ("bound(invocation)", "uint bound(uint value) { return value; }"),
        ("bound(group.x)", "uint bound(uint value) { return inputWords[0]; }"),
        (
            "bound(group.x)",
            "uint bound(uint value) { return gl_LocalInvocationIndex; }",
        ),
        (
            "bound(group.x)",
            "uint bound(uint value) { outputWords[0] = value; return value; }",
        ),
        ("bound(group.x)", "uint bound(uint value) { return unknown(value); }"),
        ("bound(group.x)", "uint bound(uint value) { return bound(value); }"),
        (
            "bound(group.x)",
            "uint inner(uint value) { return bound(value); } uint bound(uint value) { return inner(value); }",
        ),
        (
            "bound(group.x)",
            "uint bound(inout uint value) { value = 2u; return value; }",
        ),
        (
            "bound(group.x)",
            "uint bound(uint value) { uint& alias = value; alias = 2u; return value; }",
        ),
        (
            "bound(group.x)",
            "uint bound(uint value) { uint* alias = &value; return *alias; }",
        ),
        (
            "bound(group.x)",
            "uint bound(uint value) { volatile uint copy = value; return copy; }",
        ),
        ("bound(group.x)", "uint bound(uint value) { uint copy; return copy; }"),
        (
            "bound(group.x)",
            "uint bound(uint value) { if (value > 0u) { return value; } }",
        ),
        (
            "bound(group.x, invocation)",
            "uint bound(uint value, uint lane) { if (lane > 0u) { return 1u; } return value; }",
        ),
        (
            "bound(group.x, invocation++)",
            "uint bound(uint value, uint ignored) { return value; }",
        ),
        (
            "bound(group.x, unknown())",
            "uint bound(uint value, uint ignored) { return value; }",
        ),
        (
            "bound(group.x)",
            "uint min(uint value, uint limit) { return inputWords[value]; } uint bound(uint value) { return min(value, 2u); }",
        ),
    ],
)
def test_return_proofs_reject_unsafe_or_varying_helpers(expression, helper):
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(
            parse(_source("uint", _body(expression), helper)), "compute"
        )


def test_unused_lane_value_does_not_make_the_return_vary(tmp_path):
    helper = "uint bound(uint value, uint ignored) { return value + 1u; }"
    generated = _codegen().generate_stage(
        parse(_source("uint", _body("bound(group.x, invocation)"), helper)), "compute"
    )
    _compile(generated, "directx", tmp_path)


def test_uniform_return_facts_are_not_shared_between_callers():
    helper = "uint bound(uint value) { return value + 1u; }"
    body = "uint good = bound(group.x); outputWords[index] = good;" + _body(
        "bound(invocation)"
    )
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(parse(_source("uint", body, helper)), "compute")


def test_return_proofs_use_the_resolved_source_overload(tmp_path):
    helper = """uint bound(uint value) { return value + 1u; }
        uint bound(float value) { return inputWords[uint(value)]; }"""
    generated = _codegen().generate_stage(
        parse(_source("uint", _body("bound(group.x)"), helper)), "compute"
    )
    _compile(generated, "directx", tmp_path)
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(
            parse(_source("uint", _body("bound(float(group.x))"), helper)), "compute"
        )


@pytest.mark.parametrize("field", ["uint* value", "volatile uint value"])
def test_aggregate_return_proofs_exclude_pointer_and_volatile_fields(field):
    ast = parse(f"""shader Invalid {{ struct Metadata {{ {field}; }};
        uint bound(Metadata value) {{ return value.value; }} }}""")
    function = ast.functions[0]
    call = FunctionCallNode(IdentifierNode("bound"), [IdentifierNode("metadata")])
    analysis = UniformReturnAnalysis(
        {id(call): function}, set(), {s.name: s for s in ast.structs}
    )
    assert analysis.dependencies(call) is None


SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Parameters { uint group_limit; uint iterations; };
uint limit(constant Parameters& params) { return params.group_limit; }
uint count(constant Parameters& params) { return params.iterations; }
uint nested_count(constant Parameters& params) { return count(params); }
kernel void accumulate(constant Parameters& params [[buffer(0)]],
                       device uint* output [[buffer(1)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
    if (group.x >= limit(params)) { return; }
    uint result = 0u;
    for (uint step = 0u; step < nested_count(params); ++step) {
        result += simd_sum(lane + step);
    }
    output[group.x * 64u + lane] = result;
}
"""


SCALAR_SOURCE = """#include <metal_stdlib>
using namespace metal;
uint limit(constant uint& value) { return value; }
uint count(constant uint& value) { return value; }
uint nested_count(constant uint& value) { return count(value); }
kernel void accumulate(constant uint& group_limit [[buffer(0)]],
                       constant uint& iterations [[buffer(1)]],
                       device uint* output [[buffer(2)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
    if (group.x >= limit(group_limit)) { return; }
    uint result = 0u;
    for (uint step = 0u; step < nested_count(iterations); ++step) {
        result += simd_sum(lane + step);
    }
    output[group.x * 64u + lane] = result;
}
"""


def _report(root, target, source):
    (root / "kernel.metal").write_text(source)
    options = (
        {
            "target_options": {
                "directx": {
                    "software_subgroup_width": 32,
                    "relative_wave_shuffle_out_of_range": "self",
                }
            }
        }
        if target == "directx"
        else {}
    )
    options["preserve_resource_origins"] = True
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("kernel.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(32, 2, 1),
            source_options={"metal": options},
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    assert report.to_json()["diagnostics"] == [], report.to_json()
    return report


def _request(root, target, iterations=3):
    return _buffer_request(root, target, SCALAR_SOURCE, iterations)


def _buffer_request(root, target, source, iterations):
    report = _report(root, target, source)
    descriptor, package = _prepare_native_package(report, root)
    names = {
        binding["provenance"]["sourceResource"]["parameter"]: binding["name"]
        for binding in descriptor["bindings"]
    }
    wanted = [
        sum(
            sum(range(lane // 32 * 32, lane // 32 * 32 + 32)) + 32 * step
            for step in range(iterations)
        )
        for group in range(2)
        for lane in range(64)
    ]
    guard = [0x5A17BEEF] * (64 + 17)

    def value(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    outputs = {names["output"]: value(wanted + guard)}
    inputs = {
        names["output"]: value([0xDEADBEEF] * 128 + guard),
    }
    if "counts" in names:
        inputs[names["counts"]] = value([2, iterations])
    else:
        inputs[names["group_limit"]] = value([2])
        inputs[names["iterations"]] = value([iterations])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [32, 2, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, outputs


@pytest.mark.parametrize("target", ["metal", "directx"])
def test_aggregate_metadata_helpers_translate_and_compile(tmp_path, target):
    report = _report(tmp_path, target, SOURCE)
    artifact = report.to_json()["artifacts"][0]
    _compile((tmp_path / artifact["path"]).read_text(), target, tmp_path)


@pytest.mark.parametrize("target", ["metal", "directx"])
def test_constant_reference_helpers_package_and_compile(tmp_path, target):
    request, _ = _request(tmp_path, target)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("iterations", [0, 1, 3])
def test_constant_reference_helpers_execute(tmp_path, iterations):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required helper-return execution")
    if sys.platform not in {"darwin", "win32"}:
        pytest.skip("This execution control covers Metal and DirectX")
    target = "metal" if sys.platform == "darwin" else "directx"
    request, expected = _request(tmp_path, target, iterations)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=SCALAR_SOURCE if target == "metal" else None,
        original_entry="accumulate",
    )


def test_uniform_return_execution_is_required_in_existing_ci_job():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert 'CROSTL_REQUIRE_UNIFORM_RETURN_RUNTIME: "1"' in step
    assert "tests/test_translator/test_subgroup_uniform_returns.py" in step
    assert "-n auto" in step and "continue-on-error" not in step


def _storage_source(expression, helpers="", *, qualifier="@constant", prefix=""):
    return (
        _source("uint", prefix + _body(expression), helpers)
        .replace("StructuredBuffer<uint> inputWords @register(t0);", "")
        .replace(
            "void main(uint invocation",
            f"void main(StructuredBuffer<uint> counts {qualifier} @buffer(0), uint invocation",
        )
    )


@pytest.mark.parametrize(
    "expression,helpers,prefix",
    [
        ("counts[group.x]", "", ""),
        (
            "load(counts, group.x)",
            "uint load(StructuredBuffer<uint> values, uint offset) { return values[offset]; }",
            "",
        ),
        (
            "load(alias, group.x)",
            "uint load(StructuredBuffer<uint> values, uint offset) { return values[offset]; }",
            "StructuredBuffer<uint> alias = counts;",
        ),
        (
            "load(counts, group.x)",
            "uint inner(StructuredBuffer<uint> values, uint offset) { return values[offset]; } uint load(StructuredBuffer<uint> values, uint offset) { return inner(values, offset); }",
            "",
        ),
        (
            "load(group.x, counts)",
            "uint load(uint offset, StructuredBuffer<uint> values) { StructuredBuffer<uint> alias = values; return alias[offset]; }",
            "",
        ),
    ],
)
def test_constant_storage_return_dependencies_compile(
    tmp_path, expression, helpers, prefix
):
    generator = _codegen()
    ast = parse(_storage_source(expression, helpers, prefix=prefix))
    generated = generator.generate_stage(ast, "compute")
    assert generator.generate_stage(ast, "compute") == generated
    assert "GroupMemoryBarrierWithGroupSync" in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize(
    "expression,qualifier,prefix,helper",
    [
        ("load(counts, group.x)", "", "", ""),
        ("load(counts, group.x)", "@readonly", "", ""),
        ("load(counts, invocation)", "@constant", "", ""),
        ("counts[invocation]", "@constant", "", ""),
        ("load(counts, group.x)", "@constant", "unknown(counts);", ""),
        (
            "load(alias, group.x)",
            "@constant",
            "StructuredBuffer<uint> alias = counts; alias = unknown();",
            "",
        ),
        (
            "load(counts, group.x)",
            "@constant",
            "",
            "uint load(StructuredBuffer<uint> values, uint offset) { values[0] = offset; return values[offset]; }",
        ),
        (
            "load(counts, group.x)",
            "@constant",
            "",
            "uint load(StructuredBuffer<uint> values, uint offset) { return values[gl_LocalInvocationIndex]; }",
        ),
    ],
)
def test_storage_dependencies_do_not_infer_immutability_or_uniform_indices(
    expression, qualifier, prefix, helper
):
    helper = (
        helper
        or "uint load(StructuredBuffer<uint> values, uint offset) { return values[offset]; }"
    )
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(
            parse(
                _storage_source(expression, helper, qualifier=qualifier, prefix=prefix)
            ),
            "compute",
        )


def _record_source(initializer, expression="metadata.count", prefix=""):
    helpers = (
        "struct Metadata { uint count; uint offset; }; Metadata build(uint count) { "
        + initializer
        + " }"
    )
    return _source(
        "uint",
        "Metadata metadata = build(group.x + 1u);" + prefix + _body(expression),
        helpers,
    )


def test_fully_initialized_private_aggregate_returns_compile(tmp_path):
    initializer = (
        "Metadata result; result.offset = 0u; result.count = count; return result;"
    )
    generated = _codegen().generate_stage(parse(_record_source(initializer)), "compute")
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize(
    "initializer",
    [
        "Metadata result; result.count = count; return result;",
        "Metadata result; result.count = result.offset; result.offset = 0u; return result;",
        "Metadata result; result.count = count; result.count = 1u; result.offset = 0u; return result;",
        "Metadata result; result.count = count; result.offset = 0u; result.offset = 1u; return result;",
        "Metadata result; result.count = count; unknown(result); result.offset = 0u; return result;",
        "Metadata result; result.count = count; Metadata& alias = result; result.offset = 0u; return result;",
        "Metadata result; result.count = count; result.offset += 1u; return result;",
        "Metadata result; if (count > 0u) { result.count = count; } result.offset = 0u; return result;",
    ],
)
def test_partial_escaped_or_mutated_aggregate_returns_remain_unproven(initializer):
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(parse(_record_source(initializer)), "compute")


def test_private_aggregate_writes_in_the_caller_invalidate_uniformity():
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(
            parse(
                _record_source(
                    "Metadata result; result.count = count; result.offset = 0u; return result;",
                    prefix="metadata.count = invocation;",
                )
            ),
            "compute",
        )


@pytest.mark.parametrize("wide_type", ["int64", "uint64", "int64_t", "uint64_t"])
def test_record_storage_dependencies_preserve_wide_source_types(tmp_path, wide_type):
    helper = f"struct Bounds {{ uint count; {wide_type} stride; }}; uint load(StructuredBuffer<Bounds> values, uint offset) {{ return values[offset].count; }}"
    source = _storage_source("load(counts, group.x)", helper).replace(
        "StructuredBuffer<uint> counts", "StructuredBuffer<Bounds> counts"
    )
    _compile(_codegen().generate_stage(parse(source), "compute"), "directx", tmp_path)


def test_resource_return_proof_requires_every_callers_immutable_binding():
    helper = "uint load(StructuredBuffer<uint> values, uint offset) { return values[offset]; }"
    source = _storage_source(
        "load(other, group.x)", helper, prefix="uint good = load(counts, group.x);"
    ).replace("void main(", "void main(StructuredBuffer<uint> other @buffer(2), ")
    with pytest.raises(DirectXSoftwareSubgroupError):
        _codegen().generate_stage(parse(source), "compute")


@pytest.mark.parametrize("name", ["uint&", "volatile uint", "custom::uint", "uint*"])
def test_source_integer_aliases_do_not_erase_indirection_or_qualifiers(name):
    assert not UniformReturnAnalysis({}, set()).value_type(NamedType(name))


@pytest.mark.parametrize("name", ["min16int", "min16uint"])
def test_minimum_precision_integer_locals_remain_value_types(name):
    assert UniformReturnAnalysis({}, set()).value_type(NamedType(name))


def test_collective_helper_receives_checked_resource_and_record_facts(tmp_path):
    helper = """struct Bounds { uint count; };
        Bounds make_bounds(uint count) { Bounds result; result.count = count; return result; }
        uint load(StructuredBuffer<uint> values, uint offset) { return values[offset]; }
        uint reduce(StructuredBuffer<uint> values, Bounds bounds, uint lane) {
            uint result = 0u;
            for (uint step = 0u; step < load(values, bounds.count); ++step) {
                result += WaveActiveSum(lane + step);
            }
            return result;
        }"""
    source = _storage_source("1u", helper).replace(
        _body("1u"),
        "Bounds bounds = make_bounds(group.x); outputWords[index] = reduce(counts, bounds, invocation);",
    )
    _compile(_codegen().generate_stage(parse(source), "compute"), "directx", tmp_path)


STORAGE_SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Destination { device uint* data; };
uint read_count(constant uint* words, uint offset) { return words[offset]; }
kernel void accumulate(constant uint* counts [[buffer(0)]],
                       device uint* output [[buffer(1)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
    Destination destination;
    destination.data = output;
    if (group.x >= read_count(counts, 0u)) { return; }
    uint result = 0u;
    for (uint step = 0u; step < read_count(counts, 1u); ++step) {
        result += simd_sum(lane + step);
    }
    destination.data[group.x * 64u + lane] = result;
}
"""


@pytest.mark.parametrize("target", ["metal", "directx"])
def test_generated_resource_load_helpers_package_and_compile(tmp_path, target):
    request, _ = _buffer_request(tmp_path, target, STORAGE_SOURCE, 3)
    source = request.artifact_path.read_text()
    if target == "directx":
        assert "crosstl_resource_load" in source
        assert "GroupMemoryBarrierWithGroupSync" in source
    _compile(source, target, tmp_path)


@pytest.mark.parametrize("iterations", [0, 1, 3])
def test_generated_resource_load_helpers_execute(tmp_path, iterations):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required helper-return execution")
    if sys.platform not in {"darwin", "win32"}:
        pytest.skip("This execution control covers Metal and DirectX")
    target = "metal" if sys.platform == "darwin" else "directx"
    request, expected = _buffer_request(tmp_path, target, STORAGE_SOURCE, iterations)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=STORAGE_SOURCE if target == "metal" else None,
        original_entry="accumulate",
    )
