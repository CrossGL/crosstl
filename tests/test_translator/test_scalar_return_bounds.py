"""Helper return bounds retain source conversions and lexical ownership."""

import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    inspect_runtime_package,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import (
    FunctionCallNode,
    IdentifierNode,
    PrimitiveType,
    ReferenceType,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.scalar_return_bounds import scalar_return_interval
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metadata_preconditions import TARGETS, _package
from tests.test_translator.test_project_metadata_entry_scope import _config, _packages


def _generator(declarations):
    shader = parse("shader Bounds { " + declarations + " }")
    generator = GLSLCodeGen()
    for function in shader.functions:
        generator.function_definitions[function.name] = function
        generator.glsl_function_overloads_by_name.setdefault(function.name, []).append(
            function
        )
        generator.function_return_types[function.name] = function.return_type
    return generator


def _call(name, *arguments):
    return FunctionCallNode(
        IdentifierNode(name), [IdentifierNode(argument) for argument in arguments]
    )


@pytest.mark.parametrize(
    "stride,expected", [((-2, -1), (-32, 0)), ((0, 0), (0, 0)), ((1, 3), (0, 48))]
)
def test_scalar_helper_keeps_mixed_signedness_and_return_conversion(stride, expected):
    generator = _generator(
        "int offset(uint lane, int stride) { return lane * stride; }"
    )
    generator.local_variable_source_types.update(lane="uint", stride="int")
    assert (
        scalar_return_interval(
            generator,
            _call("offset", "lane", "stride"),
            {"lane": (0, 16), "stride": stride},
            {},
        )
        == expected
    )


def test_nested_helpers_resolve_source_overloads_in_the_callee_frame():
    generator = _generator("""
        int inner(int value) { return value - 2; }
        uint inner(uint value) { return value + 100u; }
        int offset(uint lane, int stride) { return inner(int(lane)) * stride; }
    """)
    generator.local_variable_source_types.update(
        lane="uint", stride="int", value="uint64_t"
    )
    assert scalar_return_interval(
        generator,
        _call("offset", "lane", "stride"),
        {"lane": (0, 3), "stride": (2, 2)},
        {"value": 1000000},
    ) == (-4, 2)
    assert generator.local_variable_source_types["value"] == "uint64_t"


def test_nested_overloads_retain_readonly_reference_source_identity():
    generator = _generator(
        "int inner(int x) { return 1; } int inner(int x) { return 2; } int outer(int x) { return inner(x); }"
    )
    reference = ReferenceType(PrimitiveType("int"), address_space="constant")
    generator.glsl_function_overloads_by_name["inner"][0].parameters[
        0
    ].param_type = reference
    generator.glsl_function_overloads_by_name["inner"][0].parameters[0].qualifiers = [
        "in"
    ]
    generator.function_definitions["outer"].parameters[0].param_type = reference
    generator.function_definitions["outer"].parameters[0].qualifiers = ["in"]
    generator.local_variable_source_types["value"] = reference
    assert scalar_return_interval(
        generator, _call("outer", "value"), {"value": (0, 3)}, {}
    ) == (1, 1)


@pytest.mark.parametrize("body", ["return hidden;", "return lane + hidden;"])
def test_helper_cannot_capture_caller_constants_or_flow_facts(body):
    generator = _generator(f"int offset(int lane) {{ {body} }}")
    generator.local_variable_source_types.update(lane="int", hidden="int")
    call = _call("offset", "lane")
    assert (
        scalar_return_interval(
            generator, call, {"lane": (0, 3), "hidden": (0, 0)}, {"hidden": 0}
        )
        is None
    )
    assert (
        generator.glsl_constant_return_integer_interval(call, {"lane": 0, "hidden": 0})
        is None
    )


@pytest.mark.parametrize(
    "declarations",
    [
        "int offset(int lane) { return lane + 2147483647; }",
        "int offset(int lane) { return offset(lane); }",
        "int offset(int lane) { return external(lane); }",
        "int offset(int lane) { return ++lane; }",
        "int offset(int lane) { lane += 1; return lane; }",
        "int offset(inout int lane) { return lane; }",
    ],
)
def test_unknown_or_effectful_helpers_cannot_establish_bounds(declarations):
    generator = _generator(declarations)
    generator.local_variable_source_types["lane"] = "int"
    assert (
        scalar_return_interval(generator, _call("offset", "lane"), {"lane": (0, 3)}, {})
        is None
    )
    assert generator._glsl_return_bounds_active is False


def test_ambiguous_overloads_do_not_supply_a_return_range():
    generator = _generator(
        "int offset(int64_t x) { return 1; } int offset(uint64_t x) { return 2; }"
    )
    generator.local_variable_source_types["lane"] = "int"
    assert (
        scalar_return_interval(generator, _call("offset", "lane"), {"lane": (0, 3)}, {})
        is None
    )


def test_parameter_conversion_precedes_arithmetic_and_return_conversion():
    generator = _generator("int offset(uint value) { return value + 2u; }")
    generator.local_variable_source_types["input"] = "int"
    assert scalar_return_interval(
        generator, _call("offset", "input"), {"input": (-3, -1)}, {}
    ) == (-1, 1)


def test_call_sites_do_not_share_argument_ranges():
    generator = _generator("int offset(int x) { return x * 2; }")
    generator.local_variable_source_types["x"] = "int"
    call = _call("offset", "x")
    for bounds, expected in [
        ((0, 3), (0, 6)),
        ((5, 9), (10, 18)),
        ((2**30, 2**30), None),
    ]:
        assert scalar_return_interval(generator, call, {"x": bounds}, {}) == expected


def test_helper_analysis_does_not_accept_unchecked_index_assertions():
    generator = _generator("int offset(int x) { return x; }")
    generator.local_variable_source_types["x"] = "int64_t"
    generator.set_index_range_assertions([dict(expression="x", minimum=0, maximum=3)])
    assert scalar_return_interval(generator, _call("offset", "x"), {}, {}) is None


def test_nested_call_depth_is_bounded():
    generator = _generator(
        " ".join(f"int call{i}(int x) {{ return call{i + 1}(x); }}" for i in range(40))
        + " int call40(int x) { return x; }"
    )
    generator.local_variable_source_types["x"] = "int"
    assert (
        scalar_return_interval(generator, _call("call0", "x"), {"x": (0, 1)}, {})
        is None
    )


HEADER = """#include <metal_stdlib>
using namespace metal;
int source_offset(uint lane, constant long& stride) {
    return lane * int(stride);
}
"""
KERNEL = """kernel void checked_offset(
    constant long& stride [[buffer(0)]],
    constant long& base [[buffer(1)]],
    const device int* values [[buffer(2)]],
    device int* results [[buffer(3)]],
    uint lane [[thread_position_in_grid]]) {
    if (lane < 4) results[lane] = values[base + source_offset(lane, stride)];
}
"""
GUARD = 123456
CASES = [(-2, 7), (-1, 7), (0, 2), (1, 2), (3, 2)]


def _source(nested):
    if not nested:
        return HEADER + KERNEL
    return HEADER.replace("int source_offset", "int inner_offset") + """
int source_offset(uint lane, constant long& stride) {
    return inner_offset(lane, stride);
}
""" + KERNEL


def _conditions(stride, base):
    minimum, maximum = (-2, -1) if stride < 0 else (0, 3)
    return [
        dict(
            entryPoint="checked_offset",
            parameter="stride",
            minimum=minimum,
            maximum=maximum,
        ),
        dict(entryPoint="checked_offset", parameter="base", minimum=base, maximum=base),
    ]


@pytest.mark.parametrize(
    "fault", ["signed-overflow", "negative-index", "wide-input", "unchecked-assertion"]
)
def test_invalid_helper_indices_produce_structured_rejection(tmp_path, fault):
    source = _source(False)
    conditions = _conditions(-2, 7)
    options = {}
    if fault == "signed-overflow":
        source = source.replace("lane * int(stride)", "int(lane) + 2147483647")
    elif fault == "negative-index":
        conditions[1].update(minimum=0, maximum=0)
    else:
        source = source.replace("if (lane < 4) ", "")
        conditions = _conditions(1, 2)
        if fault == "unchecked-assertion":
            options["index_range_assertions"] = [
                dict(expression="lane", minimum=0, maximum=3)
            ]
    config = _config(
        tmp_path,
        "opengl",
        source=source,
        conditions=conditions,
        entries=("checked_offset",),
        **options,
    )
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    ), report


@pytest.mark.parametrize("rename", [False, True])
def test_pre_emission_helper_bounds_keep_exact_caller_types(tmp_path, rename):
    source = _source(False).replace("if (lane < 4) ", "")
    if rename:
        source = source.replace(
            "source_offset(uint lane,", "source_offset(uint base,"
        ).replace("return lane * int(stride);", "return base * int(stride);")
    config = _config(
        tmp_path,
        "opengl",
        source=source,
        conditions=_conditions(-2, 7),
        entries=("checked_offset",),
    )
    (region,) = plan_dispatch_regions((4, 1, 1), (4, 1, 1))
    config = replace(
        config,
        source_options={
            "metal": dict(
                preserve_resource_origins=True,
                target_options={"opengl": dict(dispatch_region=region.to_json())},
            )
        },
    )
    payload, _, descriptors = _packages(config)
    assert payload["summary"]["translatedCount"] == 1
    assert len(descriptors) == 1
    assert not config.index_range_assertions


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("stride,base", CASES)
@pytest.mark.parametrize("nested", [False, True])
def test_source_helper_ranges_survive_project_packages(
    tmp_path, target, stride, base, nested
):
    _, package = _package(
        tmp_path, target, source=_source(nested), conditions=_conditions(stride, base)
    )
    assert inspect_runtime_package(package / "runtime-package.json")["success"]


@pytest.mark.parametrize("stride,base", CASES)
@pytest.mark.parametrize("nested", [False, True])
def test_source_helper_returns_execute_natively(tmp_path, stride, base, nested):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source = _source(nested)
    descriptor, package = _package(
        tmp_path, target, source=source, conditions=_conditions(stride, base)
    )
    inputs, outputs = {}, {}
    values = [101 + 7 * index for index in range(16)]
    expected = [values[base + lane * stride] for lane in range(4)] + [GUARD] * 4
    for binding in descriptor["bindings"]:
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        words = {
            "stride": [stride],
            "base": [base],
            "values": values,
            "results": [GUARD] * 8,
        }[parameter]
        dtype = "int64" if parameter in {"stride", "base"} else "int32"
        inputs[binding["name"]] = dict(dtype=dtype, shape=[len(words)], values=words)
        if parameter == "results":
            outputs[binding["name"]] = dict(dtype=dtype, shape=[8], values=expected)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
        observe_inputs=[
            item["name"] for item in descriptor["bindings"] if item["access"] == "read"
        ],
    )
    native = tmp_path / "native"
    native.mkdir()
    _execute(
        request,
        outputs,
        native,
        validate=_validate,
        original_source=source,
        original_entry="checked_offset",
    )
    details = json.loads((native / "evidence.json").read_text())["records"][
        "generated"
    ]["details"]
    assert len(details["inputObservations"]) == 3
    assert all(item["unchanged"] for item in details["inputObservations"].values())
