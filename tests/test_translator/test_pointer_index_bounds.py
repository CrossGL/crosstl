"""Checked pointer bases permit signed relative indices without losing bounds."""

import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.translator.ast import ForNode
from crosstl.translator.codegen import GLSL_codegen
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metadata_index_bounds import (
    GUARD,
    SOURCE,
    _index_config,
    _index_request,
)
from tests.test_translator.test_project_metadata_entry_scope import _config, _packages
from tests.test_translator.test_scalar_return_bounds import _generator

CONDITIONS = (
    dict(entryPoint="read_value", parameter="index", minimum=-2, maximum=-1),
    dict(entryPoint="read_value", parameter="offset", minimum=2, maximum=4),
)
BODIES = {
    "load": "values += offset; result[2] = values[index];",
    "alias": (
        "const device uint* shifted = values + offset; result[2] = shifted[index];"
    ),
    "narrow": "values += offset; result[2] = values[int(index)];",
    "member": (
        "values += offset; long2 at = long2(index, index); result[2] = values[at.y];"
    ),
    "store": "result += offset; result[index] = values[offset + index];",
    "constant": "values += 4; result[2] = values[-1];",
}


def _source(case):
    return SOURCE.replace("ulong", "long").replace(
        "result[0] = values[offset + index];", BODIES[case]
    )


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("case", BODIES)
def test_checked_base_keeps_signed_relative_indices(tmp_path, target, case):
    _, package, (descriptor,) = _packages(
        _index_config(
            tmp_path, source=_source(case), conditions=CONDITIONS, target=target
        )
    )
    assert (package / descriptor["artifact"]["packagePath"]).is_file()
    for index in (-2, -1):
        for offset in (2, 4):
            request, _ = _index_request(descriptor, package, index, offset)
            assert not request.execution_plan.diagnostics


@pytest.mark.parametrize(
    "bounds,accepted",
    [((0, 2), True), ((-1, -1), False), ((2147483648, 2147483648), False)],
)
def test_proved_zero_base_preserves_absolute_index_assertion_status(
    tmp_path, monkeypatch, bounds, accepted
):
    decisions = []
    original = GLSL_codegen.decide_index_narrowing

    def record(**kwargs):
        decision = original(**kwargs)
        value_range = kwargs["value_range"]
        if value_range is not None and value_range.status == "asserted":
            decisions.append((kwargs, decision))
        return decision

    monkeypatch.setattr(GLSL_codegen, "decide_index_narrowing", record)
    config = replace(
        _index_config(
            tmp_path,
            source=_source("load"),
            conditions=({**CONDITIONS[1], "minimum": 0, "maximum": 0},),
        ),
        index_range_assertions=[
            dict(expression="index", minimum=bounds[0], maximum=bounds[1])
        ],
    )
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == int(accepted), report
    assert decisions
    assert all(arguments["base_range"] is None for arguments, _ in decisions)
    if accepted:
        assert all(decision.range_status == "asserted" for _, decision in decisions)
    else:
        assert all(not decision.accepted for _, decision in decisions)


@pytest.mark.parametrize(
    "offset,reason",
    [
        (0, "index-range-unproven"),
        (2, "composed-pointer-index-range-unproven"),
        (8, "composed-pointer-index-range-unproven"),
    ],
)
def test_checked_pointer_base_does_not_prove_an_unknown_index(tmp_path, offset, reason):
    report = translate_project(
        _index_config(
            tmp_path,
            source=_source("load"),
            conditions=({**CONDITIONS[1], "minimum": offset, "maximum": offset},),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    diagnostic = report["diagnostics"][0]
    assert diagnostic["code"] == "project.translate.opengl-index-type-unsupported"
    assert diagnostic["details"]["indexConversion"]["reason"] == reason
    assert diagnostic["details"]["indexConversion"]["rangeStatus"] == "unproven"


@pytest.mark.parametrize(
    "fault",
    [
        "missing-base",
        "missing-index",
        "negative-sum",
        "overflow",
        "mutation",
        "shadow",
        "escape",
        "assertion",
        "positive-assertion",
        "derived-assertion",
        "local-assertion",
    ],
)
def test_relative_index_requires_a_complete_checked_address(tmp_path, fault):
    conditions = list(CONDITIONS)
    source = _source("load")
    if fault == "missing-base":
        conditions = conditions[:1]
    elif fault == "missing-index" or "assertion" in fault:
        conditions = conditions[1:]
        if fault == "derived-assertion":
            source = source.replace("values[index]", "values[index + 1]")
        elif fault == "local-assertion":
            source = source.replace(
                "values += offset;", "values += offset; long at = index + 1;"
            ).replace("values[index]", "values[at]")
    elif fault == "negative-sum":
        conditions[1] = {**conditions[1], "minimum": 0, "maximum": 1}
    elif fault == "overflow":
        conditions[0] = {**conditions[0], "minimum": 2147483647, "maximum": 2147483647}
    elif fault == "shadow":
        source = source.replace(
            BODIES["load"], "{ long index = -99; " + BODIES["load"] + " }"
        )
    elif fault == "escape":
        source = source.replace(
            "kernel void", "void overwrite(thread long& at) { at = -99; }\nkernel void"
        )
        source = source.replace(
            "values += offset;", "values += offset; long at = index; overwrite(at);"
        ).replace("values[index]", "values[at]")
    else:
        source = source.replace(
            "values += offset;", "values += offset; long at = index; at = -99;"
        ).replace("values[index]", "values[at]")
    config = _index_config(tmp_path, source=source, conditions=conditions)
    if "assertion" in fault:
        minimum, maximum = (-2, -1) if fault == "assertion" else (0, 0)
        config = replace(
            config,
            index_range_assertions=[
                dict(expression="index", minimum=minimum, maximum=maximum)
            ],
        )
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 0
    diagnostic = report["diagnostics"][0]
    assert diagnostic["code"] == "project.translate.opengl-index-type-unsupported"
    assert diagnostic["details"]["indexConversion"]["reason"] in {
        "composed-pointer-index-range-unproven",
        "pointer-offset-range-unproven",
    }


@pytest.mark.parametrize("kind", ["int", "long"])
def test_negative_absolute_subscript_is_still_rejected(tmp_path, kind):
    source = SOURCE.replace(
        "result[0] = values[offset + index];", f"result[2] = values[{kind}(-1)];"
    )
    report = translate_project(
        _index_config(tmp_path, source=source), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert (
        report["diagnostics"][0]["details"]["indexConversion"]["reason"]
        == "negative-index"
    )


@pytest.mark.parametrize(
    "bounds,reason,status",
    [
        ((-1, -1), "negative-index", "out-of-range"),
        ((-3, -1), "negative-index", "out-of-range"),
        ((-1, 0), "index-range-may-be-negative", "unproven"),
        ((-1, 5), "index-range-may-be-negative", "unproven"),
    ],
)
def test_index_diagnostic_distinguishes_possible_and_proven_negative_values(
    tmp_path, bounds, reason, status
):
    source = _source("load").replace("values += offset; ", "")
    conditions = ({**CONDITIONS[0], "minimum": bounds[0], "maximum": bounds[1]},)
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=conditions),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    diagnostic = report["diagnostics"][0]["details"]["indexConversion"]
    assert diagnostic["reason"] == reason
    assert diagnostic["rangeStatus"] == status
    assert (
        diagnostic["sourceRange"]["minimum"],
        diagnostic["sourceRange"]["maximum"],
    ) == bounds


@pytest.mark.parametrize("case", ["load", "alias", "member", "store"])
@pytest.mark.parametrize("index", [-2, -1])
@pytest.mark.parametrize("offset", [2, 4])
def test_signed_relative_indices_execute_natively(tmp_path, case, index, offset):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native index execution"
        )
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source = _source(case)
    _, package, (descriptor,) = _packages(
        _index_config(tmp_path, source=source, conditions=CONDITIONS, target=target)
    )
    expected = [GUARD] * 8
    expected[index + offset if case == "store" else 2] = 17 + 13 * (index + offset)
    request, outputs = _index_request(
        descriptor, package, index, offset, expected_words=expected
    )
    _execute(
        request,
        outputs,
        tmp_path,
        validate=_validate,
        original_source=source,
        original_entry="read_value",
    )


def _pointer_loop_bounds(body, *, start=32, step=2, condition="i < 4", helpers=""):
    generator = _generator(
        helpers + "void run(device int* values, int64_t stride) { "
        f"device int* current = values + {start}; "
        f"for (int i = 0; {condition}; ++i) {{ {body} }} }}"
    )
    function = generator.function_definitions["run"]
    generator.glsl_workgroup_proof_initial_intervals = lambda _: {
        "values": (0, 0),
        "stride": (step, step),
    }
    generator.annotate_glsl_control_flow_intervals(function)
    (loop,) = (
        node for node in generator.walk_ast(function.body) if isinstance(node, ForNode)
    )
    return getattr(loop.body, "statements", loop.body)[0]._glsl_control_flow_intervals


@pytest.mark.parametrize("step", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("operator", ["+=", "-="])
def test_local_storage_pointer_loop_bounds_include_every_entry(step, operator):
    facts = _pointer_loop_bounds(f"current; current {operator} stride;", step=step)
    delta = step if operator == "+=" else -step
    expected = [32 + iteration * delta for iteration in range(4)]
    assert facts["current"] == (min(expected), max(expected))


@pytest.mark.parametrize(
    "body,start,step,condition",
    [
        ("current; current += stride;", 2147483644, 1, "i < 4"),
        ("current; current -= stride;", 3, 1, "i < 4"),
        ("current; current += stride; stride += 1;", 32, 1, "i < 4"),
        ("current; if (i == 0) { current += stride; }", 32, 1, "i < 4"),
        ("current; current += stride; current += 1;", 32, 1, "i < 4"),
        ("current; current += i;", 32, 1, "i < 4"),
        ("current; unknown(&current); current += stride;", 32, 1, "i < 4"),
        ("current; current += stride;", 32, 1, "i < unknown(stride)"),
        ("current; device int*& alias = current; current += stride;", 32, 1, "i < 4"),
        ("current; device int* current = values; current += stride;", 32, 1, "i < 4"),
    ],
)
def test_local_pointer_loop_bounds_reject_unproved_updates(
    body, start, step, condition
):
    assert "current" not in _pointer_loop_bounds(
        body, start=start, step=step, condition=condition
    )


@pytest.mark.parametrize("qualifier", ["", "inout "])
def test_local_pointer_loop_distinguishes_value_and_reference_calls(qualifier):
    facts = _pointer_loop_bounds(
        "current; move(current); current += stride;",
        helpers=f"void move({qualifier}device int* pointer) {{ pointer += 10; }} ",
    )
    if qualifier:
        assert "current" not in facts
    else:
        assert facts["current"] == (32, 38)


@pytest.mark.parametrize(
    "body",
    [
        "return int(value) + 2;",
        "int result = int(value); result += 2; return result;",
        "value += 2; return int(value);",
        "int result = int(value); for (int j = 0; j < 2; ++j) { result += 1; } return result;",
    ],
)
def test_pure_invariant_helper_return_bounds_pointer_loop(body):
    facts = _pointer_loop_bounds(
        "current; current += stride;",
        condition="i < bound(stride)",
        helpers=f"int bound(int64_t value) {{ {body} }} ",
    )
    assert facts["current"] == (32, 38)


@pytest.mark.parametrize(
    "helper,condition,body",
    [
        (
            "int bound(int64_t value) { unknown(value); return 4; }",
            "i < bound(stride)",
            "",
        ),
        ("int bound(int64_t value) { return bound(value); }", "i < bound(stride)", ""),
        (
            "int bound(int64_t value) { return int(value) + 2147483647; }",
            "i < bound(stride)",
            "",
        ),
        (
            "int bound(int64_t value) { return int(value) / 0; }",
            "i < bound(stride)",
            "",
        ),
        (
            "int bound(inout int64_t value) { value += 1; return 4; }",
            "i < bound(stride)",
            "",
        ),
        ("int bound(int64_t value) { return 4; }", "i < bound(++stride)", ""),
        ("int bound(int64_t value) { return 4; }", "i < bound(stride)", "stride += 1;"),
        (
            "int bound(int64_t value) { return 4; }",
            "i < bound(stride)",
            "{ int64_t stride = 2; }",
        ),
        (
            "int bound(int value) { for (int j = 0; j < value; ++j) { unknown(); } return 1; }",
            "i < 4 && bound(i) > 0",
            "",
        ),
    ],
)
def test_effectful_or_changing_helper_calls_cannot_bound_pointer_loop(
    helper, condition, body
):
    facts = _pointer_loop_bounds(
        "current; " + body + " current += stride;", condition=condition, helpers=helper
    )
    assert "current" not in facts


LOOP_BOUNDS = {
    "literal": "2u",
    "metadata": "(size + 127ul) / 128ul",
    "helper": "blocks(size, 128u)",
}


def _loop_source(bound):
    return """#include <metal_stdlib>
using namespace metal;
ulong blocks(ulong size, uint width) { return (size + width - 1ul) / width; }
kernel void nested_copy(const device uint* src [[buffer(0)]],
                        device uint* dst [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        constant long& stride [[buffer(3)]],
                        constant long& base [[buffer(4)]],
                        uint tid [[thread_position_in_grid]]) {
    for (uint r = 0; r < BOUND; ++r) {
        uint index = r * 128u + tid * 4u;
        const device uint* current = src + base + long(index) * stride;
        for (int i = 0; i < 4; ++i) {
            if (index < size) { dst[index] = *current; }
            ++index;
            current += stride;
        }
    }
}
""".replace("BOUND", LOOP_BOUNDS[bound])


def _loop_config(root, target, bound, stride):
    conditions = [
        dict(entryPoint="nested_copy", parameter=name, minimum=value, maximum=value)
        for name, value in (
            ("size", 129),
            ("stride", stride),
            ("base", 512 if stride < 0 else 0),
        )
    ]
    config = _config(
        root,
        target,
        source=_loop_source(bound),
        entries=("nested_copy",),
        conditions=conditions,
    )
    metal = dict(config.source_options["metal"])
    if target == "opengl":
        (region,) = plan_dispatch_regions((32, 1, 1), (32, 1, 1))
        metal["target_options"] = {"opengl": {"dispatch_region": region.to_json()}}
    return replace(config, workgroup_size=(32, 1, 1), source_options={"metal": metal})


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("bound", LOOP_BOUNDS)
@pytest.mark.parametrize("stride", [-2, 2])
def test_local_pointer_loops_translate_with_checked_metadata(
    tmp_path, target, bound, stride
):
    config = _loop_config(tmp_path, target, bound, stride)
    report = translate_project(config, format_output=False).to_json()
    assert not config.index_range_assertions
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]


@pytest.mark.parametrize("parameter", ["stride", "base"])
@pytest.mark.parametrize("bound", LOOP_BOUNDS)
def test_local_pointer_loop_requires_checked_offset_inputs(tmp_path, parameter, bound):
    config = _loop_config(tmp_path, "opengl", bound, 2)
    config = replace(
        config,
        metadata_preconditions=tuple(
            item
            for item in config.metadata_preconditions
            if item.parameter != parameter
        ),
    )
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("bound", LOOP_BOUNDS)
@pytest.mark.parametrize("stride", [-2, 2])
def test_local_pointer_loops_execute_natively(tmp_path, bound, stride):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native pointer checks"
        )
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    config = _loop_config(tmp_path, target, bound, stride)
    _, package, (descriptor,) = _packages(config)
    # All formed pointers, including inactive lanes and terminal updates, stay
    # inside this input allocation. Only 129 values are dereferenced and copied.
    values = [17 + 13 * index for index in range(513)]
    base = 512 if stride < 0 else 0
    expected = [values[base + index * stride] for index in range(129)] + [GUARD] * 32
    data = {
        "src": ("uint32", values),
        "dst": ("uint32", [GUARD] * len(expected)),
        "size": ("uint64", [129]),
        "stride": ("int64", [stride]),
        "base": ("int64", [base]),
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
        dict(workgroupCount=[1, 1, 1], workgroupSize=[32, 1, 1]),
        observe_inputs=[
            item["name"] for item in descriptor["bindings"] if item["access"] == "read"
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
        original_source=_loop_source(bound),
        original_entry="nested_copy",
    )
    details = json.loads((native / "evidence.json").read_text())["records"][
        "generated"
    ]["details"]
    assert len(details["metadataPreconditions"]) == 3
    assert len(details["inputObservations"]) == 4
    assert all(item["unchanged"] for item in details["inputObservations"].values())
