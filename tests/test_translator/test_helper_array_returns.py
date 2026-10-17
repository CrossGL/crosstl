"""Checked array facts and bounded vector-return helpers preserve indexing."""

import json
import os
import sys
from copy import deepcopy

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.translator.ast import (
    NamedType,
    PointerType,
    PrimitiveType,
    ResourceMemoryQualifierNode,
)
from crosstl.translator.codegen.scalar_return_bounds import vector_return_intervals
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metadata_preconditions import TARGETS, _package
from tests.test_translator.test_project_metadata_entry_scope import _config
from tests.test_translator.test_scalar_return_bounds import _call, _generator


def _array_generator(body):
    generator = _generator("int2 offsets(int lane, int* values) { " + body + " }")
    parameter = generator.function_definitions["offsets"].parameters[1]
    parameter.param_type = PointerType(PrimitiveType("int"), address_space="constant")
    parameter.qualifiers = ["constant"]
    generator.local_variable_source_types.update(
        lane="int", values=parameter.param_type
    )
    return generator


def _bounds(body, facts=None):
    generator = _array_generator(body)
    return vector_return_intervals(
        generator,
        _call("offsets", "lane", "values"),
        "ivec2",
        (
            facts
            if facts is not None
            else {
                "lane": (0, 5),
                "@metadata:values[0]": (2, 2),
                "@metadata:values[1]": (3, 3),
            }
        ),
        {},
    )


def test_checked_array_arguments_and_vector_components():
    assert _bounds("return int2(lane / values[1], lane % values[1]);") == (
        (0, 1),
        (0, 2),
    )


@pytest.mark.parametrize(
    "body",
    [
        "values[0] = 1; return int2(0);",
        "return int2(lane / values[2]);",
        "return int2(lane / 0);",
        "return int2(lane + 2147483647);",
        "int2 result = int2(0); for (int d = 0; d < 1000; ++d) { result.x += d; } return result;",
        "int2 result = int2(0); while (lane > 0) { result.x += lane; --lane; } return result;",
        "int2 result = int2(0); unknown(values); return result;",
        "int2 result = int2(0); for (int d = 0; d < 2; ++d) { int lane = 1; result.x += lane; } return result;",
    ],
)
def test_unknown_effects_uncovered_elements_and_unbounded_state_fail_closed(body):
    assert _bounds(body) is None


def test_loop_joins_zero_and_nonzero_trip_paths():
    assert _bounds(
        "int2 result = int2(0); for (int d = 1; d >= 0 && lane > 0; --d) { result.x += lane % values[d]; lane /= values[d]; } return result;"
    ) == ((0, 3), (0, 0))


def test_unchecked_array_values_do_not_establish_helper_bounds():
    assert _bounds("return int2(values[0]);", {"lane": (0, 5)}) is None
    assert _bounds("return int2(values[0]);", {}) is None


@pytest.mark.parametrize(
    "fault",
    [
        "out",
        "inout",
        "volatile",
        "typed-volatile",
        "element-volatile",
        "mutable",
        "width",
        "ambiguous",
    ],
)
def test_array_helper_requires_unique_immutable_source_contract(fault):
    generator = _array_generator("return int2(values[0]);")
    function = generator.function_definitions["offsets"]
    parameter = function.parameters[1]
    if fault in {"out", "inout", "volatile"}:
        parameter.qualifiers.append(fault)
    elif fault == "typed-volatile":
        parameter.param_type.resource_qualifiers = [
            ResourceMemoryQualifierNode("volatile")
        ]
    elif fault == "element-volatile":
        parameter.param_type.pointee_type.qualifiers = ["volatile"]
    elif fault == "mutable":
        parameter.param_type.address_space = "device"
    elif fault == "width":
        parameter.param_type = PointerType(
            PrimitiveType("int64_t"), address_space="constant"
        )
    else:
        other = deepcopy(function)
        other.parameters[1].param_type = PointerType(
            PrimitiveType("int"), is_mutable=False, address_space="device"
        )
        generator.glsl_function_overloads_by_name["offsets"].append(other)
        generator.local_variable_source_types["values"] = NamedType(
            "StructuredBuffer", [PrimitiveType("int")]
        )
    assert (
        vector_return_intervals(
            generator,
            _call("offsets", "lane", "values"),
            "ivec2",
            {"lane": (0, 1), "@metadata:values[0]": (3, 3)},
            {},
        )
        is None
    )


@pytest.mark.parametrize(
    "body",
    [
        "int2 result = int2(0); for (int d = 0; d < 2; ++d) { result.x += d; } return result + int2(d);",
        "const int2 result = int2(0); result.x = 1; return result;",
        "int2 result = int2(0); for (int d = 0; d < 2; ++d) { values[d] = 1; } return result;",
        "return offsets(lane, values);",
    ],
)
def test_vector_helpers_reject_scope_escape_const_writes_and_recursion(body):
    assert _bounds(body) is None


def test_checked_arrays_forward_through_nested_vector_helpers():
    generator = _generator("""
        int2 inner(int lane, int* values) { return int2(lane / values[1], lane % values[1]); }
        int2 outer(int lane, int* values) { int2 result = inner(lane, values); result.x += values[0]; return result; }
    """)
    for function in generator.function_definitions.values():
        function.parameters[1].param_type = PointerType(
            PrimitiveType("int"), address_space="constant"
        )
        function.parameters[1].qualifiers = ["constant"]
    generator.local_variable_source_types.update(
        lane="int",
        values=generator.function_definitions["outer"].parameters[1].param_type,
    )
    assert vector_return_intervals(
        generator,
        _call("outer", "lane", "values"),
        "ivec2",
        {
            "lane": (0, 5),
            "@metadata:values[0]": (-2, -2),
            "@metadata:values[1]": (3, 3),
        },
        {},
    ) == ((-2, -1), (0, 2))


BODIES = {
    "direct": (
        """int2 offsets(uint flat, constant int* shape, constant long* a, constant long* b) {
        uint row = flat / uint(shape[1]);
        uint column = flat % uint(shape[1]);
        return int2(int(a[0]) * int(row) + int(a[1]) * int(column),
                    int(b[0]) * int(row) + int(b[1]) * int(column));
    }"""
    ),
    "loop": (
        """int2 offsets(uint flat, constant int* shape, constant long* a, constant long* b) {
        int2 result = int2(0);
        for (int d = 1; d >= 0 && flat > 0; --d) {
            result += int2(int(a[d]), int(b[d])) * int(flat % uint(shape[d]));
            flat /= uint(shape[d]);
        }
        return result;
    }"""
    ),
}
KERNEL = """kernel void indexed_layout(
    constant int* shape [[buffer(0)]], constant long* src_strides [[buffer(1)]],
    constant long* dst_strides [[buffer(2)]], constant long& src_base [[buffer(3)]],
    constant long& dst_base [[buffer(4)]], const device int* src [[buffer(5)]],
    device int* dst [[buffer(6)]], uint lane [[thread_position_in_grid]]) {
    if (lane < 6) {
        int2 at = offsets(lane, shape, src_strides, dst_strides);
        dst[dst_base + at.y] = src[src_base + at.x];
    }
}
"""
GUARD = 123456


def _inputs(reverse):
    return {
        "shape": ("int32", [2, 3]),
        "src_strides": ("int64", [3, -1 if reverse else 1]),
        "dst_strides": ("int64", [1, 2]),
        "src_base": ("int64", [5]),
        "dst_base": ("int64", [3]),
    }


def _source(case):
    return "#include <metal_stdlib>\nusing namespace metal;\n" + BODIES[case] + KERNEL


def _conditions(reverse):
    return [
        dict(
            entryPoint="indexed_layout",
            parameter=name,
            minimum=word,
            maximum=word,
            **({"element": i} if len(words) > 1 else {}),
        )
        for name, (_, words) in _inputs(reverse).items()
        for i, word in enumerate(words)
    ]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", BODIES)
@pytest.mark.parametrize("reverse", [False, True])
def test_vector_helper_metadata_survives_project_packages(
    tmp_path, target, case, reverse
):
    _package(tmp_path, target, source=_source(case), conditions=_conditions(reverse))


@pytest.mark.parametrize(
    "fault", ["missing-element", "zero-divisor", "negative-offset"]
)
def test_invalid_vector_helper_indices_are_rejected(tmp_path, fault):
    conditions = _conditions(False)
    if fault == "missing-element":
        conditions = [
            item
            for item in conditions
            if not (item["parameter"] == "shape" and item.get("element") == 1)
        ]
    else:
        parameter = "shape" if fault == "zero-divisor" else "dst_base"
        for item in conditions:
            if item["parameter"] == parameter:
                item.update(
                    minimum=0 if fault == "zero-divisor" else -100,
                    maximum=0 if fault == "zero-divisor" else -100,
                )
    config = _config(
        tmp_path,
        "opengl",
        source=_source("direct"),
        conditions=conditions,
        entries=("indexed_layout",),
    )
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("case", BODIES)
@pytest.mark.parametrize("reverse", [False, True])
def test_vector_array_helpers_execute_natively(tmp_path, case, reverse):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source = _source(case)
    descriptor, package = _package(
        tmp_path, target, source=source, conditions=_conditions(reverse)
    )
    expected = [GUARD] * 14
    for lane in range(6):
        row, column = divmod(lane, 3)
        expected[3 + row + column * 2] = 101 + 7 * (
            5 + row * 3 + column * (-1 if reverse else 1)
        )
    data = {
        **_inputs(reverse),
        "src": ("int32", [101 + 7 * i for i in range(20)]),
        "dst": ("int32", [GUARD] * 14),
    }
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        name = binding["provenance"]["sourceResource"]["parameter"]
        dtype, words = data[name]
        inputs[binding["name"]] = dict(dtype=dtype, shape=[len(words)], values=words)
        if name == "dst":
            outputs[binding["name"]] = dict(dtype=dtype, shape=[14], values=expected)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [2, 1, 1], "workgroupSize": [4, 1, 1]},
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
        original_entry="indexed_layout",
    )
    details = json.loads((native / "evidence.json").read_text())["records"][
        "generated"
    ]["details"]
    assert len(details["metadataPreconditions"]) == 8
    assert len(details["inputObservations"]) == 6
    assert all(item["unchanged"] for item in details["inputObservations"].values())
