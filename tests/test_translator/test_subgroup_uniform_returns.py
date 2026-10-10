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
from crosstl.translator.ast import (
    FunctionCallNode,
    IdentifierNode,
    NamedType,
    PointerType,
    PrimitiveType,
)
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLSoftwareSubgroupError,
)
from crosstl.translator.codegen.uniform_returns import UniformReturnAnalysis
from tests.runtime_helpers import _prepare_native_package
from tests.test_translator.test_directx_software_reductions import _codegen, _source
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_UNIFORM_RETURN_RUNTIME"


@pytest.fixture(params=["directx", "opengl"])
def return_target(request):
    return request.param


def _return_codegen(target):
    return (
        _codegen() if target == "directx" else GLSLCodeGen(software_subgroup_width=32)
    )


RETURN_ERRORS = (DirectXSoftwareSubgroupError, OpenGLSoftwareSubgroupError)


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
def test_read_only_return_bounds_compile(tmp_path, expression, helper, return_target):
    source = parse(_source("uint", _body(expression), helper))
    generator = _return_codegen(return_target)
    generated = generator.generate_stage(source, "compute")
    assert generator.generate_stage(source, "compute") == generated
    assert "WaveActiveSum" not in generated
    assert (
        "GroupMemoryBarrierWithGroupSync"
        if return_target == "directx"
        else "barrier();"
    ) in generated
    _compile(generated, return_target, tmp_path)


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
def test_return_proofs_reject_unsafe_or_varying_helpers(
    expression, helper, return_target
):
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
            parse(_source("uint", _body(expression), helper)), "compute"
        )


def test_unused_lane_value_does_not_make_the_return_vary(tmp_path, return_target):
    helper = "uint bound(uint value, uint ignored) { return value + 1u; }"
    generated = _return_codegen(return_target).generate_stage(
        parse(_source("uint", _body("bound(group.x, invocation)"), helper)), "compute"
    )
    _compile(generated, return_target, tmp_path)


def test_uniform_return_facts_are_not_shared_between_callers(return_target):
    helper = "uint bound(uint value) { return value + 1u; }"
    body = "uint good = bound(group.x); outputWords[index] = good;" + _body(
        "bound(invocation)"
    )
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
            parse(_source("uint", body, helper)), "compute"
        )


def test_return_proofs_use_the_resolved_source_overload(tmp_path, return_target):
    helper = """uint bound(uint value) { return value + 1u; }
        uint bound(float value) { return inputWords[uint(value)]; }"""
    generated = _return_codegen(return_target).generate_stage(
        parse(_source("uint", _body("bound(group.x)"), helper)), "compute"
    )
    _compile(generated, return_target, tmp_path)
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
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


def _report(root, target, source, *, diagnostic=None):
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
    if target == "opengl":
        options["target_options"] = {"opengl": {"software_subgroup_width": 32}}
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
    assert [item["code"] for item in report.to_json()["diagnostics"]] == (
        [] if diagnostic is None else [diagnostic]
    ), report.to_json()
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


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_aggregate_metadata_helpers_translate_and_compile(tmp_path, target):
    report = _report(tmp_path, target, SOURCE)
    artifact = report.to_json()["artifacts"][0]
    _compile((tmp_path / artifact["path"]).read_text(), target, tmp_path)


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_constant_reference_helpers_package_and_compile(tmp_path, target):
    request, _ = _request(tmp_path, target)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("iterations", [0, 1, 3])
def test_constant_reference_helpers_execute(tmp_path, iterations):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required helper-return execution")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform, "opengl")
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
    tmp_path, expression, helpers, prefix, return_target
):
    generator = _return_codegen(return_target)
    source = _storage_source(expression, helpers, prefix=prefix)
    if return_target == "opengl":
        source = source.replace(
            "StructuredBuffer<uint> values", "constant uint* values"
        ).replace("StructuredBuffer<uint> alias", "constant uint* alias")
    ast = parse(source)
    generated = generator.generate_stage(ast, "compute")
    assert generator.generate_stage(ast, "compute") == generated
    assert (
        "GroupMemoryBarrierWithGroupSync"
        if return_target == "directx"
        else "barrier();"
    ) in generated
    _compile(generated, return_target, tmp_path)


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
    expression, qualifier, prefix, helper, return_target
):
    helper = (
        helper
        or "uint load(StructuredBuffer<uint> values, uint offset) { return values[offset]; }"
    )
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
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


def test_fully_initialized_private_aggregate_returns_compile(tmp_path, return_target):
    initializer = (
        "Metadata result; result.offset = 0u; result.count = count; return result;"
    )
    generated = _return_codegen(return_target).generate_stage(
        parse(_record_source(initializer)), "compute"
    )
    _compile(generated, return_target, tmp_path)


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
def test_partial_escaped_or_mutated_aggregate_returns_remain_unproven(
    initializer, return_target
):
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
            parse(_record_source(initializer)), "compute"
        )


def test_private_aggregate_writes_in_the_caller_invalidate_uniformity(return_target):
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(
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


def test_resource_return_proof_requires_every_callers_immutable_binding(return_target):
    helper = "uint load(StructuredBuffer<uint> values, uint offset) { return values[offset]; }"
    source = _storage_source(
        "load(other, group.x)", helper, prefix="uint good = load(counts, group.x);"
    ).replace("void main(", "void main(StructuredBuffer<uint> other @buffer(2), ")
    with pytest.raises(RETURN_ERRORS):
        _return_codegen(return_target).generate_stage(parse(source), "compute")


@pytest.mark.parametrize("name", ["uint&", "volatile uint", "custom::uint", "uint*"])
def test_source_integer_aliases_do_not_erase_indirection_or_qualifiers(name):
    assert not UniformReturnAnalysis({}, set()).value_type(NamedType(name))


@pytest.mark.parametrize("name", ["min16int", "min16uint"])
def test_minimum_precision_integer_locals_remain_value_types(name):
    assert UniformReturnAnalysis({}, set()).value_type(NamedType(name))


def test_collective_helper_receives_checked_resource_and_record_facts(
    tmp_path, return_target
):
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
    if return_target == "opengl":
        source = source.replace(
            "StructuredBuffer<uint> values", "constant uint* values"
        )
    _compile(
        _return_codegen(return_target).generate_stage(parse(source), "compute"),
        return_target,
        tmp_path,
    )


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
        pytest.skip("Generated handle indices still require an OpenGL range proof")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform, "opengl")
    request, expected = _buffer_request(tmp_path, target, STORAGE_SOURCE, iterations)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=STORAGE_SOURCE if target == "metal" else None,
        original_entry="accumulate",
    )


def test_generated_handle_uniformity_does_not_authorize_index_narrowing(tmp_path):
    report = _report(
        tmp_path,
        "opengl",
        STORAGE_SOURCE,
        diagnostic="project.translate.opengl-index-type-unsupported",
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert (
        report["diagnostics"][0]["details"]["indexConversion"]["reason"]
        == "index-range-unproven"
    )


BUFFER_SOURCE = (
    STORAGE_SOURCE.replace("struct Destination { device uint* data; };", "")
    .replace("    Destination destination;\n    destination.data = output;\n", "")
    .replace("destination.data[", "output[")
)

_ACCUMULATION = """uint result = 0u;
    for (uint step = 0u; step < read_count(counts, 1u); ++step) {
        result += simd_sum(lane + step);
    }"""
COLLECTIVE_SOURCE = BUFFER_SOURCE.replace(
    "kernel void accumulate(",
    "uint accumulate_values(constant uint* counts, uint lane) { "
    "if (read_count(counts, 1u) == 0u) { return 0u; } "
    + _ACCUMULATION
    + " return result; }\nkernel void accumulate(",
).replace(
    "    " + _ACCUMULATION,
    "    uint result = accumulate_values(counts, lane);",
)


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize(
    "source", [BUFFER_SOURCE, COLLECTIVE_SOURCE], ids=["entry", "helper"]
)
def test_constant_storage_helpers_package_and_compile(tmp_path, target, source):
    request, _ = _buffer_request(tmp_path, target, source, 3)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("iterations", [0, 1, 3])
@pytest.mark.parametrize(
    "source", [BUFFER_SOURCE, COLLECTIVE_SOURCE], ids=["entry", "helper"]
)
def test_constant_storage_helpers_execute(tmp_path, iterations, source):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required helper-return execution")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform, "opengl")
    request, expected = _buffer_request(tmp_path, target, source, iterations)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="accumulate",
    )


@pytest.mark.parametrize(
    "body,helpers,expected",
    [
        ("return 1u;", "", True),
        ("return values[0];", "", True),
        ("uint value = values[0]; outputWords[0] = value; return value;", "", True),
        ("outputWords[0] = values[0]; return 1u;", "", True),
        (
            "return read(values);",
            "uint read(StructuredBuffer<uint> data) { return data[0]; }",
            True,
        ),
        ("return copy(values[0]);", "uint copy(uint value) { return value; }", True),
        ("values[0] = 1u; return 1u;", "", False),
        ("values[0]++; return 1u;", "", False),
        ("unknown(values); return 1u;", "", False),
        ("unknown(values[0]); return 1u;", "", False),
        ("uint& alias = values[0]; return alias;", "", False),
        ("StructuredBuffer<uint> alias = values; return alias[0];", "", False),
        (
            "return read(values);",
            "uint read(StructuredBuffer<uint> data) { data[0] = 1u; return data[0]; }",
            False,
        ),
        (
            "return read(values);",
            "uint read(StructuredBuffer<uint> data) { return inspect(data); }",
            False,
        ),
        (
            "return copy(values[0]);",
            "uint copy(inout uint value) { value++; return value; }",
            False,
        ),
    ],
)
@pytest.mark.parametrize("pointer", [False, True])
def test_read_only_resource_effects_follow_bodies(body, helpers, expected, pointer):
    ast = parse(
        "shader Effect { RWStructuredBuffer<uint> outputWords; "
        + helpers
        + " uint inspect(StructuredBuffer<uint> values) { "
        + body
        + " } }"
    )
    functions = {function.name: function for function in ast.functions}
    if pointer:
        for function in functions.values():
            for parameter in function.parameters:
                if getattr(parameter.param_type, "name", None) == "StructuredBuffer":
                    parameter.param_type = PointerType(
                        PrimitiveType("uint"),
                        is_mutable=False,
                        address_space="constant",
                        access_mode="read",
                    )
    calls = {
        id(node): functions[node.function.name]
        for node in ast.walk()
        if isinstance(node, FunctionCallNode) and node.function.name in functions
    }
    analysis = UniformReturnAnalysis(calls, set())
    for _ in range(2):
        assert (
            analysis.read_only_resource_parameter(functions["inspect"], 0) is expected
        )


@pytest.mark.parametrize(
    "space,access,mutable,expected",
    [
        ("constant", "read", False, True),
        ("constant", None, True, True),
        ("constant", "write", True, False),
        ("device", "read", False, True),
        ("storage", "read", False, True),
        ("threadgroup", "read", False, False),
        ("thread", "read", False, False),
        (None, "read", False, False),
        ("device", "read_write", False, False),
        ("device", "write", False, False),
        ("device", "read", True, False),
    ],
)
def test_storage_pointer_proofs_preserve_access_contract(
    space, access, mutable, expected
):
    type_ = PointerType(
        PrimitiveType("uint"),
        is_mutable=mutable,
        address_space=space,
        access_mode=access,
    )
    assert UniformReturnAnalysis({}, set()).resource_type(type_) is expected


@pytest.mark.parametrize("qualifier", ["volatile", "coherent", "writeonly"])
def test_storage_pointer_proofs_reject_memory_qualifiers(qualifier):
    type_ = PointerType(
        PrimitiveType("uint"),
        is_mutable=False,
        address_space="constant",
        access_mode="read",
    )
    type_.qualifiers = [qualifier]
    assert not UniformReturnAnalysis({}, set()).resource_type(type_)


@pytest.mark.parametrize("qualifier", ["out", "inout"])
def test_resource_output_parameters_do_not_preserve_caller_facts(qualifier):
    function = parse(
        "shader Effect { uint inspect(StructuredBuffer<uint> values) { return 1u; } }"
    ).functions[0]
    function.parameters[0].qualifiers = [qualifier]
    assert not UniformReturnAnalysis({}, set()).read_only_resource_parameter(
        function, 0
    )


@pytest.mark.parametrize(
    "body,expected",
    [("return values[0];", True), ("values[0] = 1u; return 1u;", False)],
)
def test_resource_effects_accept_statement_list_bodies(body, expected):
    function = parse(
        "shader Effect { uint inspect(StructuredBuffer<uint> values) { " + body + " } }"
    ).functions[0]
    function.body = function.body.statements
    assert (
        UniformReturnAnalysis({}, set()).read_only_resource_parameter(function, 0)
        is expected
    )


@pytest.mark.parametrize(
    "replacement",
    [
        ("constant uint*", "const device uint*"),
        ("read_count(counts, 1u)", "read_count(counts, lane)"),
        ("return words[offset];", "return unknown(words, offset);"),
    ],
)
def test_source_storage_returns_require_immutable_uniform_inputs(tmp_path, replacement):
    report = _report(
        tmp_path,
        "opengl",
        BUFFER_SOURCE.replace(*replacement),
        diagnostic="project.translate.opengl-software-subgroup-invalid",
    ).to_json()
    assert report["summary"]["translatedCount"] == 0


def test_return_analysis_resets_between_programs(return_target):
    generator = _return_codegen(return_target)
    helper = "uint bound(uint value) { return value + 1u; }"
    safe = parse(_source("uint", _body("bound(group.x)"), helper))
    expected = generator.generate_stage(safe, "compute")
    with pytest.raises(RETURN_ERRORS):
        generator.generate_stage(
            parse(_source("uint", _body("bound(invocation)"), helper)), "compute"
        )
    assert generator.generate_stage(safe, "compute") == expected
