"""Scalar helper state stays source-typed and separate from its caller."""

import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.translator.ast import PrimitiveType, ReferenceType
from crosstl.translator.codegen.scalar_return_bounds import scalar_return_interval
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_project_metadata_entry_scope import _config, _packages
from tests.test_translator.test_scalar_return_bounds import _call, _generator


def _bounds(body, facts=None):
    generator = _generator("int offset(int lane, int stride) { " + body + " }")
    generator.local_variable_source_types.update(lane="int", stride="int")
    facts = facts if facts is not None else {"lane": (0, 3), "stride": (-2, -1)}
    original = dict(facts)
    result = scalar_return_interval(
        generator, _call("offset", "lane", "stride"), facts, {}
    )
    assert facts == original
    assert generator._glsl_return_bounds_active is False
    return result


@pytest.mark.parametrize(
    "body,expected",
    [
        ("int at = lane * stride; return at;", (-6, 0)),
        ("int at = lane; at *= stride; return at;", (-6, 0)),
        ("lane += 1; return lane;", (1, 4)),
        ("lane *= stride; return lane;", (-6, 0)),
        (
            "int at = 0; for (int d = 0; d < 3; ++d) { at += lane * stride; } return at;",
            (-18, 0),
        ),
        (
            "int at = 0; for (int d = 0; d < lane; ++d) { at += stride; } return at;",
            (-6, 0),
        ),
        ("int at = 7; for (int d = 0; d < 0; ++d) { at += lane; } return at;", (7, 7)),
        ("int at = 0; for (int d = 0; d < 64; ++d) { at += 1; } return at;", (64, 64)),
        (
            "int at = 0; for (int d = 0; d < 2; ++d) { int value = d + 1; at += value; } return at;",
            (3, 3),
        ),
        (
            "int at = 0; for (int d = 0; d < 2; ++d) { for (int e = 0; e < 3; ++e) { at += 1; } } return at;",
            (6, 6),
        ),
        ("uint at = uint(lane); at *= uint(stride); return int(at);", (-6, 0)),
        ("int at = 0; for (int d = 0; d < 2; ++d) { return 1; } return at;", (1, 1)),
        ("if (lane == 0) return 7; return lane;", (0, 7)),
        ("if (lane == 0) return 7; else return 3;", (3, 7)),
        ("int at = 1; { at += 2; } return at;", (3, 3)),
        ("for (int d = 0; d < 3; ++d) { if (d == lane) return d; } return 9;", (0, 9)),
    ],
)
def test_scalar_local_state_has_source_typed_exit_bounds(body, expected):
    assert _bounds(body) == expected


@pytest.mark.parametrize(
    "body",
    [
        "int at = lane; unknown(at); return at;",
        "int at = lane; int& ref = at; return ref;",
        "int at = lane; int* ref = &at; return at;",
        "int at = lane; int at = 2; return at;",
        "int at = 0; for (int d = 0; d < 2; ++d) { int lane = 1; at += lane; } return at;",
        "const int at = lane; at += 1; return at;",
        "volatile int at = lane; return at;",
        "int at = 0; for (int d = 0; d < 65; ++d) { at += 1; } return at;",
        "int at = 0; for (int d = 0; d < 2; --d) { at += 1; } return at;",
        "int at = 0; for (int d = 0; d < unknown(lane); ++d) { at += 1; } return at;",
        "int at = 0; for (int d = 0; d < 2; ++d) { break; } return at;",
        "if (lane == 0) return 7;",
        "if (lane == 0) return 7; return unknown(lane);",
        "const int at = 1; if (lane == 0) at += 1; return at;",
        "if (lane == 0) return 7; return lane + 2147483647;",
        "int at = lane; at += 2147483647; return at;",
        "int at = lane; at /= 0; return at;",
        "lane += 2147483647; return lane;",
        "int at = offset(lane, stride); return at;",
        "int at = 0; for (int d = 0; d < 2; ++d) { at += d; } return at + d;",
        "int at = 0; for (const int d = 0; d < 2; ++d) { at += d; } return at;",
        "int at = 0; for (int d = 0; d < 2; ++d) { for (int e = 0; e < 2; ++e) { hidden += 1; } } return at;",
    ],
)
def test_unsupported_scalar_state_cannot_establish_a_proof(body):
    assert _bounds(body) is None


@pytest.mark.parametrize("trip_count", range(9))
@pytest.mark.parametrize("stride", [-7, -1, 0, 1, 7])
def test_local_loop_bounds_cover_every_exact_execution(trip_count, stride):
    bounds = _bounds(
        f"int at = 0; for (int d = 0; d < {trip_count}; ++d) {{ at += lane * stride; }} return at;",
        {"lane": (0, 10), "stride": (stride, stride)},
    )
    expected = [sum(lane * stride for _ in range(trip_count)) for lane in range(11)]
    assert bounds == (min(expected), max(expected))


def test_nested_scalar_parameter_updates_do_not_change_caller_state():
    generator = _generator("""
        int inner(int lane) { lane += 2; return lane; }
        int outer(int lane) { int value = inner(lane); return value + lane; }
    """)
    generator.local_variable_source_types.update(lane="int", value="uint64_t")
    facts, constants = {"lane": (0, 3)}, {"value": 1000000}
    for _ in range(2):
        assert scalar_return_interval(
            generator, _call("outer", "lane"), facts, constants
        ) == (2, 8)
    assert facts == {"lane": (0, 3)}
    assert constants == {"value": 1000000}
    assert generator.local_variable_source_types["value"] == "uint64_t"


@pytest.mark.parametrize(
    "qualifier", ["const", "out", "inout", "volatile", "reference", "const-reference"]
)
def test_scalar_parameter_writes_require_owned_mutable_values(qualifier):
    generator = _generator("int offset(int lane) { lane += 1; return lane; }")
    parameter = generator.function_definitions["offset"].parameters[0]
    if qualifier in {"reference", "const-reference"}:
        parameter.param_type = ReferenceType(
            PrimitiveType("int"), is_mutable=qualifier == "reference"
        )
    else:
        parameter.qualifiers = [qualifier]
    generator.local_variable_source_types["lane"] = parameter.param_type
    assert (
        scalar_return_interval(generator, _call("offset", "lane"), {"lane": (0, 3)}, {})
        is None
    )


def test_scalar_helper_vector_parameter_state_stays_inside_callee():
    generator = _generator(
        "int offset(uint3 position) { int at = 0; "
        "for (int d = 0; d < 2; ++d) { at += int(position.z % 3u); position.z /= 3u; } "
        "return at + int(position.x); }"
    )
    generator.local_variable_source_types["position"] = "uint3"
    generator.local_variable_types["position"] = "uvec3"
    facts = {"position.x": (1, 2), "position.y": (0, 0), "position.z": (0, 8)}
    original = dict(facts)
    for _ in range(2):
        assert scalar_return_interval(
            generator, _call("offset", "position"), facts, {}
        ) == (1, 6)
    assert facts == original


EXPRESSION = "long(flat / uint(shape[1])) * strides[0] + long(flat % uint(shape[1])) * strides[1]"
BODIES = {
    "expression": f"return {EXPRESSION};",
    "early-return": f"if (flat == 0) return 0; return {EXPRESSION};",
    "loop-return": (
        f"for (int d = 0; d < 2; ++d) {{ if (d == 0) return {EXPRESSION}; }} return 0;"
    ),
    "local": f"long at = {EXPRESSION}; return at;",
    "loop": (
        """long at = 0; uint remaining = flat;
        for (int d = 1; d >= 0; --d) {
            at += long(remaining % uint(shape[d])) * strides[d];
            remaining /= uint(shape[d]);
        }
        return at;"""
    ),
    "parameter": (
        """long at = 0;
        for (int d = 1; d >= 0; --d) {
            at += long(flat % uint(shape[d])) * strides[d];
            flat /= uint(shape[d]);
        }
        return at;"""
    ),
}
METADATA = {
    "shape": ("int32", [2, 3]),
    "strides": ("int64", [3, -1]),
    "base": ("int64", [2]),
}


def _source(case):
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + (
            "long offset(uint flat, constant int* shape, constant long* strides) { "
            + BODIES[case]
            + " }\n"
        )
        + """kernel void checked_copy(
        constant int* shape [[buffer(0)]], constant long* strides [[buffer(1)]],
        constant long& base [[buffer(2)]], const device uint* src [[buffer(3)]],
        device uint* dst [[buffer(4)]], uint flat [[thread_position_in_grid]]) {
        if (flat < 6) {
            long index = offset(flat, shape, strides);
            dst[flat] = src[base + index];
            dst[6 + flat] = flat;
        }
    }
    """
    )


def _local_config(root, target, case):
    conditions = [
        dict(
            entryPoint="checked_copy",
            parameter=parameter,
            minimum=value,
            maximum=value,
            **({"element": index} if len(values) > 1 else {}),
        )
        for parameter, (_, values) in METADATA.items()
        for index, value in enumerate(values)
    ]
    config = _config(
        root,
        target,
        source=_source(case),
        entries=("checked_copy",),
        conditions=conditions,
    )
    options = dict(config.source_options["metal"])
    if target == "opengl":
        (region,) = plan_dispatch_regions((6, 1, 1), (1, 1, 1))
        options["target_options"] = {target: dict(dispatch_region=region.to_json())}
    return replace(config, workgroup_size=(1, 1, 1), source_options={"metal": options})


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("case", BODIES)
def test_scalar_local_helpers_translate_with_checked_array_facts(
    tmp_path, target, case
):
    config = _local_config(tmp_path, target, case)
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    assert not config.index_range_assertions


@pytest.mark.parametrize("case", BODIES)
def test_scalar_local_helpers_reject_missing_metadata(tmp_path, case):
    config = replace(_local_config(tmp_path, "opengl", case), metadata_preconditions=())
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("case", BODIES)
def test_scalar_local_helpers_execute_natively(tmp_path, case):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    config = _local_config(tmp_path, target, case)
    _, package, (descriptor,) = _packages(config)
    values = [17 + 13 * i for i in range(6)]
    guard = 0xABCDEF01
    expected = [values[i] for i in (2, 1, 0, 5, 4, 3)] + list(range(6)) + [guard] * 32
    data = {
        **METADATA,
        "src": ("uint32", values),
        "dst": ("uint32", [guard] * len(expected)),
    }
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        name = binding["provenance"]["sourceResource"]["parameter"]
        dtype, items = data[name]
        inputs[binding["name"]] = dict(dtype=dtype, shape=[len(items)], values=items)
        if name == "dst":
            outputs[binding["name"]] = dict(
                dtype=dtype, shape=[len(expected)], values=expected
            )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        dict(workgroupCount=[6, 1, 1], workgroupSize=[1, 1, 1]),
        observe_inputs=[
            binding["name"]
            for binding in descriptor["bindings"]
            if binding["access"] == "read"
        ],
    )
    assert not request.execution_plan.diagnostics
    native = tmp_path / "native"
    native.mkdir()
    _execute(
        request,
        outputs,
        native,
        validate=_validate,
        original_source=_source(case),
        original_entry="checked_copy",
    )
    details = json.loads((native / "evidence.json").read_text())["records"][
        "generated"
    ]["details"]
    assert len(details["metadataPreconditions"]) == 5
    assert len(details["inputObservations"]) == 4
    assert all(item["unchanged"] for item in details["inputObservations"].values())
