"""Checked pointer bases permit signed relative indices without losing bounds."""

import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.translator.ast import ForNode, ReturnNode
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

ARRAY_BOUND_BODIES = {
    "addressed": "rows[0] += 1; return rows[0][0] + rows[0][0];",
    "read": "return rows[0][0] + rows[1][0];",
    "advance": "rows[0] += 1; rows[1] += 2; return rows[0][0] + rows[1][0];",
    "loop": (
        "for (int i = 0; i < 3; ++i) { rows[0] += 1; } return rows[0][0] + rows[1][0];"
    ),
    "loop-reads": (
        "uint total = 0; for (int i = 0; i < 3; ++i) { total += rows[0][0]; rows[0] += 1; } return total + rows[1][0];"
    ),
    "nested": (
        "for (int i = 0; i < 2; ++i) { for (int j = 0; j < 2; ++j) { rows[j] += i + 1; } } return rows[0][0] + rows[1][0];"
    ),
    "branch": (
        "if (selector == 0) { rows[0] += 1; } else { rows[1] += 2; } return rows[0][0] + rows[1][0];"
    ),
    "forward": "return advance_rows(rows);",
    "saved": (
        "const device uint* saved = rows[0]; rows[0] += 1; return saved[0] + rows[1][0];"
    ),
    "multiple-calls": "return rows[0][0] + rows[1][0];",
}
ARRAY_BOUND_DIRECTX_DIAGNOSTICS = {
    "saved": "pointer-element-escape",
    "multiple-calls": "element-reassignment-unsupported",
}


def _array_bounds_source(case, *, body=None):
    body = ARRAY_BOUND_BODIES[case] if body is None else body
    helpers = ""
    if "advance_rows(" in body:
        helpers += """uint advance_rows(const device uint* rows[2]) {
    rows[0] += 1; rows[1] += 2;
    return rows[0][0] + rows[1][0];
}
"""
    if "invalidate_rows(" in body:
        helpers += "void invalidate_rows(const device uint* rows[2]) { rows[0] += 2147483647; }\n"
    second_call = ""
    if case == "multiple-calls":
        second_call = """
        for (int i = 0; i < 2; ++i) { rows[i] = values + base + 4 + i * 2; }
        result[2] += consume(rows, uint(index));
        """
    source = (
        """#include <metal_stdlib>
using namespace metal;
long source_offset(constant long& offset) { return offset; }
HELPERS
uint consume(const device uint* rows[2], uint selector) {
    BODY
}
kernel void read_value(const device uint* values [[buffer(0)]],
                       constant long& index [[buffer(1)]],
                       constant long& offset [[buffer(2)]],
                       device uint* result [[buffer(3)]]) {
    long base = source_offset(offset);
    const device uint* rows[2];
    for (int i = 0; i < 2; ++i) { rows[i] = values + base + i * 2; }
    result[2] = consume(rows, uint(index));
    SECOND_CALL
}
""".replace("BODY", body)
        .replace("HELPERS", helpers)
        .replace("SECOND_CALL", second_call)
    )
    if case == "addressed":
        source = source.replace("rows[2]", "rows[1]")
        source = source.replace(
            "const device uint* rows[1];\n    for (int i = 0; i < 2; ++i) { rows[i] = values + base + i * 2; }",
            "const device uint* row = values + base;",
        ).replace("consume(rows, uint(index))", "consume(&row, uint(index))")
    return source


def _array_bounds_config(root, case, target="opengl", *, body=None):
    return replace(
        _index_config(
            root,
            source=_array_bounds_source(case, body=body),
            target=target,
            conditions=(
                dict(entryPoint="read_value", parameter="index", minimum=0, maximum=1),
                dict(entryPoint="read_value", parameter="offset", minimum=0, maximum=3),
            ),
        ),
        output_dir="out",
    )


@pytest.mark.parametrize(
    "case,target",
    [
        (case, target)
        for case in ARRAY_BOUND_BODIES
        for target in ("metal", "directx", "opengl")
        if not (target == "directx" and case in ARRAY_BOUND_DIRECTX_DIAGNOSTICS)
    ],
)
def test_pointer_array_bounds_follow_source_updates(tmp_path, case, target):
    payload = translate_project(
        _array_bounds_config(tmp_path, case, target), format_output=False
    ).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert not payload["diagnostics"]


@pytest.mark.parametrize("case,reason", ARRAY_BOUND_DIRECTX_DIAGNOSTICS.items())
def test_pointer_array_bounds_report_remaining_directx_restrictions(
    tmp_path, case, reason
):
    payload = translate_project(
        _array_bounds_config(tmp_path, case, "directx"), format_output=False
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert (
        payload["diagnostics"][0]["details"]["resourcePointerArray"]["reason"] == reason
    )
    assert not list((tmp_path / "out").rglob("*.hlsl"))


@pytest.mark.parametrize(
    "mutation",
    (
        "rows[0] += 2147483647;",
        "rows[0] -= 4;",
        "rows[selector] += 2147483647;",
        "if (selector == 0) { rows[0] -= 4; }",
        "invalidate_rows(rows);",
        "for (int i = 0; i < 2; ++i) { rows[0] += 1073741824; }",
        "for (int i = 0; i < 2; ++i) { if (i == 0) { rows[0] -= 4; } uint observed = rows[0][0]; rows[0] += 4; }",
        "while (selector != 0) { rows[0] += 1; selector -= 1; }",
    ),
)
def test_pointer_array_bounds_reject_unproven_mutations(tmp_path, mutation):
    payload = translate_project(
        _array_bounds_config(
            tmp_path, "read", body=mutation + ARRAY_BOUND_BODIES["read"]
        ),
        format_output=False,
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert any(item["severity"] == "error" for item in payload["diagnostics"])
    assert not list((tmp_path / "out").rglob("*.glsl"))


def test_pointer_array_bounds_require_checked_metadata(tmp_path):
    config = replace(_array_bounds_config(tmp_path, "read"), metadata_preconditions=())
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert not list((tmp_path / "out").rglob("*.glsl"))


@pytest.mark.parametrize("case", ("read", "addressed"))
def test_pointer_array_bounds_invalidate_other_argument_effects(tmp_path, case):
    extent = 1 if case == "addressed" else 2
    argument = "&row" if case == "addressed" else "rows"
    source = (
        _array_bounds_source(case)
        .replace(
            "uint consume(",
            f"uint change_rows(const device uint* rows[{extent}]) {{ rows[0] += 2147483647; return 0; }}\nuint consume(",
        )
        .replace(
            f"consume({argument}, uint(index))",
            f"consume({argument}, change_rows({argument}))",
        )
    )
    config = _array_bounds_config(tmp_path, case)
    (tmp_path / "multi.metal").write_text(source)
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert not list((tmp_path / "out").rglob("*.glsl"))


@pytest.mark.parametrize("offset", (-1, 4))
def test_pointer_array_bounds_validate_uploads(tmp_path, offset):
    _, package, (descriptor,) = _packages(_array_bounds_config(tmp_path, "read"))
    with pytest.raises(NativeLoaderDispatchError, match="outside"):
        _index_request(descriptor, package, 0, offset)


@pytest.mark.parametrize("case", ARRAY_BOUND_BODIES)
def test_pointer_array_bounds_execute_natively(tmp_path, case):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native array bounds"
        )
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    if target == "directx" and case in ARRAY_BOUND_DIRECTX_DIAGNOSTICS:
        pytest.skip(
            f"DirectX rejects this source: {ARRAY_BOUND_DIRECTX_DIAGNOSTICS[case]}"
        )
    config = _array_bounds_config(tmp_path, case, target)
    _, package, (descriptor,) = _packages(config)
    values = [17 + position * 13 for position in range(16)]
    for offset, selector in ((0, 0), (1, 1), (3, 0)):
        first, second = offset, offset + 2
        if case == "addressed":
            first = second = offset + 1
        elif case in {"advance", "forward"}:
            first += 1
            second += 2
        elif case == "loop":
            first += 3
        elif case == "nested":
            first += 3
            second += 3
        elif case == "branch":
            first += selector == 0
            second += 2 * (selector != 0)
        expected_words = [GUARD] * 8
        expected_words[2] = values[first] + values[second]
        if case == "loop-reads":
            expected_words[2] += values[first + 1] + values[first + 2]
        if case == "multiple-calls":
            expected_words[2] += values[first + 4] + values[second + 4]
        request, expected = _index_request(
            descriptor, package, selector, offset, expected_words=expected_words
        )
        assert not request.execution_plan.diagnostics
        run = tmp_path / f"run-{offset}-{selector}"
        run.mkdir()
        _execute(
            request,
            expected,
            run,
            validate=_validate,
            original_source=_array_bounds_source(case),
            original_entry="read_value",
        )
        evidence = json.loads((run / "evidence.json").read_text())
        for record in evidence["records"].values():
            observations = record["details"]["inputObservations"]
            assert len(observations) == 3
            assert all(item["unchanged"] for item in observations.values())


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


OPERAND_TYPES = {
    "uint8": ("uchar", "32", "int", "8", "short", "1", 256),
    "int8": ("char", "16", "int", "8", "short", "1", 128),
    "uint16": ("ushort", "256", "int", "256", "int", "256", 256),
    "int16": ("short", "128", "int", "256", "int", "128", 256),
    "uint32": ("uint", "2147483648u", "long", "2l", "long", "16777216l", 256),
    "int32": ("int", "1073741824", "long", "2l", "long", "8388608l", 256),
    "uint64": ("ulong", "4294967296ul", "int", "2", "ulong", "33554432ul", 256),
    "int64": ("long", "4294967296l", "uint", "2u", "long", "33554432l", 256),
}

STORED_OPERANDS = {
    "rebased-alias": ("input += 2;", "1l", 3),
    "repeated-rebase": ("input += 1; input += 2;", "1l", 4),
    "loop-rebase": ("for (int i = 0; i < 3; ++i) input += 2;", "1l", 7),
    "relative-alias": ("input += 4;", "-1l", 3),
    "branch-rebase": (
        "if (input[0] != 0) { input += 2; } else { input += 4; }",
        "1l",
        3,
    ),
    "saved-pointer": (
        "const device uint* saved = input; input += 4; input = saved + 2;",
        "1l",
        3,
    ),
    "offset-name-collision": ("long input_offset = 1; input += 2;", "input_offset", 3),
    "nested-alias": (
        "input += 2; const device uint* intermediate = input + 1l; input = intermediate;",
        "1l",
        4,
    ),
    "helper-char": ("int groups = 257;", "typed_offset(groups)", 1),
    "helper-uchar": ("int groups = 257;", "typed_offset(groups)", 1),
    "helper-short": ("int groups = 65537;", "typed_offset(groups)", 1),
    "helper-ushort": ("int groups = 65537;", "typed_offset(groups)", 1),
    "store-uint8": ("int groups = 257; uchar narrowed = groups;", "long(narrowed)", 1),
    "store-int8": ("int groups = 257; char narrowed = groups;", "long(narrowed)", 1),
    "store-uint16": (
        "int groups = 65537; ushort narrowed = groups;",
        "long(narrowed)",
        1,
    ),
    "store-int16": (
        "int groups = 65537; short narrowed = groups;",
        "long(narrowed)",
        1,
    ),
    "store-uint32": (
        "ulong groups = 4294967297ul; uint narrowed = groups;",
        "long(narrowed)",
        1,
    ),
    "store-int32": (
        "long groups = 4294967297l; int narrowed = groups;",
        "long(narrowed)",
        1,
    ),
    "store-uint64": (
        "long groups = -1l; ulong narrowed = groups;",
        "long(narrowed / 18446744073709551615ul)",
        1,
    ),
    "store-int64": (
        "ulong groups = 18446744073709551615ul; long narrowed = groups;",
        "narrowed + 2l",
        1,
    ),
    "assign-int16": (
        "int groups = 65537; short narrowed = 0; narrowed = groups;",
        "long(narrowed)",
        1,
    ),
    "add-int16": (
        "int groups = 32770; short narrowed = 32767; narrowed += groups;",
        "long(narrowed)",
        1,
    ),
    "subtract-int16": (
        "int groups = 65535; short narrowed = 0; narrowed -= groups;",
        "long(narrowed)",
        1,
    ),
    "vector-int16": (
        "int groups = 65537; short2 narrowed = short2(groups, groups);",
        "long(narrowed.x)",
        1,
    ),
    "component-int16": (
        "int groups = 65537; short2 narrowed = short2(0); narrowed.x = groups;",
        "long(narrowed.x)",
        1,
    ),
    "component-add-int16": (
        "int groups = 32770; short2 narrowed = short2(32767); narrowed.x += groups;",
        "long(narrowed.x)",
        1,
    ),
}


def _operand_source(kind):
    global_type, width, local_type, groups, product_type, divisor, _ = OPERAND_TYPES[
        kind
    ]
    return f"""#include <metal_stdlib>
using namespace metal;
constant {global_type} width = {width};
kernel void select_value(const device uint* input [[buffer(0)]],
                         device uint* output [[buffer(1)]]) {{
    const {local_type} groups = {groups};
    const {product_type} product = groups * width;
    const long at = long(product / {divisor});
    const device uint* selected = input + at;
    output[0] = selected[0];
}}
"""


def _stored_operand_source(kind):
    declarations, expression, _ = STORED_OPERANDS[kind]
    source = _replace_operand_body(declarations, expression).replace(
        "constant uchar width = 32;\n", ""
    )
    if kind.startswith("helper-"):
        scalar = kind.split("-", 1)[1]
        helper = f"long typed_offset(int value) {{ {scalar}2 stored = {scalar}2(value); return long(stored.x); }}\n"
        source = source.replace("kernel void", helper + "kernel void")
    return source


def _replace_operand_body(declarations, expression):
    source = _operand_source("uint8")
    start = source.index("    const int groups")
    end = source.index("    const device uint* selected", start)
    return (
        source[:start]
        + f"    {declarations}\n    const long at = {expression};\n"
        + source[end:]
    )


def _operand_config(root, target, kind, *, source=None):
    return replace(
        _config(
            root,
            target,
            source=_operand_source(kind) if source is None else source,
            entries=("select_value",),
            conditions=(),
        ),
        workgroup_size=(1, 1, 1),
    )


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("kind", OPERAND_TYPES)
def test_pointer_bounds_preserve_promoted_operand_types(tmp_path, target, kind):
    config = _operand_config(tmp_path, target, kind)
    assert not config.index_range_assertions
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    assert not report["diagnostics"]


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("kind", STORED_OPERANDS)
def test_pointer_bounds_preserve_stored_operand_values(tmp_path, target, kind):
    config = _operand_config(
        tmp_path, target, "uint8", source=_stored_operand_source(kind)
    )
    assert not config.index_range_assertions
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    assert not report["diagnostics"]


@pytest.mark.parametrize(
    "scalar,bits", [("char", 8), ("uchar", 8), ("short", 16), ("ushort", 16)]
)
def test_helper_vector_storage_cannot_authorize_negative_addresses(
    tmp_path, scalar, bits
):
    source = _stored_operand_source(f"helper-{scalar}").replace(
        "return long(stored.x);", f"return long(stored.x) - {1 << bits}l;"
    )
    report = translate_project(
        _operand_config(tmp_path, "opengl", "uint8", source=source), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("name", ["groups", "count", "width"])
def test_operand_analysis_restores_caller_type_state(name):
    generator = _generator("""
        int offset(uint8 width) {
            const int groups = 8;
            const int count = groups * width;
            return count;
        }
    """)
    function = generator.function_definitions["offset"]
    generator.local_variable_types[name] = "uint64_t"
    generator.local_variable_source_types[name] = "uint8"
    types = dict(generator.local_variable_types)
    source_types = dict(generator.local_variable_source_types)
    generator.glsl_workgroup_proof_initial_intervals = lambda _: {"width": (32, 32)}
    generator.annotate_glsl_control_flow_intervals(function)
    (returned,) = (
        node
        for node in generator.walk_ast(function.body)
        if isinstance(node, ReturnNode)
    )
    assert returned._glsl_control_flow_intervals["count"] == (256, 256)
    assert generator.local_variable_types == types
    assert generator.local_variable_source_types == source_types


@pytest.mark.parametrize(
    "source_type,bits,signed",
    [
        ("int8", 8, True),
        ("uint8", 8, False),
        ("int16", 16, True),
        ("uint16", 16, False),
        ("int", 32, True),
        ("uint", 32, False),
    ],
)
@pytest.mark.parametrize("assignment", [False, True])
@pytest.mark.parametrize("boundary", ["sign", "wrap", "negative"])
def test_stored_operand_ranges_include_every_converted_value(
    source_type, bits, signed, assignment, boundary
):
    modulus = 1 << bits
    start = {"sign": modulus // 2 - 2, "wrap": modulus - 2, "negative": -2}[boundary]
    values = [value % modulus for value in range(start, start + 5)]
    if signed:
        values = [
            value - modulus if value >= modulus // 2 else value for value in values
        ]
    body = f"{source_type} narrowed = input_value;"
    if assignment:
        body = f"{source_type} narrowed = 0; narrowed = input_value;"
    generator = _generator(
        f"int offset(int64 input_value) {{ {body} return int(narrowed); }}"
    )
    function = generator.function_definitions["offset"]
    generator.glsl_workgroup_proof_initial_intervals = lambda _: {
        "input_value": (start, start + 4)
    }
    generator.annotate_glsl_control_flow_intervals(function)
    (returned,) = (
        node
        for node in generator.walk_ast(function.body)
        if isinstance(node, ReturnNode)
    )
    assert returned._glsl_control_flow_intervals["narrowed"] == (
        min(values),
        max(values),
    )


@pytest.mark.parametrize(
    "declarations,expression",
    [
        ("int groups = 2147483647;", "groups + 1"),
        ("uint groups = 4294967295u;", "groups + 1u"),
        ("long groups = 9223372036854775807l;", "groups + 1l"),
        ("int groups = int(input[0]);", "groups * width"),
        ("int groups = 8; { long groups = -1; output[0] = input[groups]; }", "groups"),
        ("int groups = 8; groups = -1;", "groups"),
        ("int groups = 8; unknown(&groups);", "groups"),
        (
            "const int groups = 65537; const short narrowed = groups;",
            "long(narrowed) - 65536l",
        ),
        (
            "int groups = 65537; short narrowed = 0; narrowed = groups;",
            "long(narrowed) - 65536l",
        ),
        (
            "int groups = 32770; short narrowed = 32767; narrowed += groups;",
            "long(narrowed) - 65536l",
        ),
        (
            "int groups = 65537; short2 narrowed = short2(groups);",
            "long(narrowed.x) - 65536l",
        ),
        (
            "int groups = 65537; short2 narrowed = short2(0); narrowed.x = groups;",
            "long(narrowed.x) - 65536l",
        ),
        (
            "int groups = 32770; short2 narrowed = short2(32767); narrowed.x += groups;",
            "long(narrowed.x) - 65536l",
        ),
        ("int groups = 2147483647; groups += 1;", "long(groups) - 2147483647l"),
    ],
)
def test_operand_types_do_not_authorize_unproved_addresses(
    tmp_path, declarations, expression
):
    source = _replace_operand_body(declarations, expression)
    report = translate_project(
        _operand_config(tmp_path, "opengl", "uint8", source=source), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(item["severity"] == "error" for item in report["diagnostics"])


@pytest.mark.parametrize("kind", [*OPERAND_TYPES, *STORED_OPERANDS])
def test_typed_pointer_operands_execute_natively(tmp_path, kind):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native pointer checks"
        )
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    if kind in STORED_OPERANDS:
        source, index = _stored_operand_source(kind), STORED_OPERANDS[kind][-1]
    else:
        source, index = _operand_source(kind), OPERAND_TYPES[kind][-1]
    _, package, (descriptor,) = _packages(
        _operand_config(tmp_path, target, "uint8", source=source)
    )
    values = [17 + 13 * index for index in range(257)]
    expected = [values[index]] + [GUARD] * 32
    inputs, outputs, observed = {}, {}, []
    for binding in descriptor["bindings"]:
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        data = values if parameter == "input" else [GUARD] * len(expected)
        inputs[binding["name"]] = dict(dtype="uint32", shape=[len(data)], values=data)
        if parameter == "output":
            outputs[binding["name"]] = dict(
                dtype="uint32", shape=[len(expected)], values=expected
            )
        else:
            observed.append(binding["name"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        dict(workgroupCount=[1, 1, 1], workgroupSize=[1, 1, 1]),
        observe_inputs=observed,
    )
    assert not request.execution_plan.diagnostics
    native = tmp_path / "native"
    native.mkdir()
    _execute(
        request,
        outputs,
        native,
        validate=_validate,
        original_source=source,
        original_entry="select_value",
    )
    details = json.loads((native / "evidence.json").read_text())["records"][
        "generated"
    ]["details"]
    assert len(details["inputObservations"]) == 1
    assert all(item["unchanged"] for item in details["inputObservations"].values())
