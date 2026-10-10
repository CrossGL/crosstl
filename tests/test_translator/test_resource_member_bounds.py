"""Resource-handle field bounds survive value construction and helper calls."""

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
from crosstl.translator.ast import FunctionCallNode, IdentifierNode, MemberAccessNode
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_RESOURCE_MEMBER_BOUNDS"
CASES = (
    "zero",
    "shift",
    "inline",
    "nested",
    "copy",
    "assign",
    "member-assign",
    "rebase",
    "branch",
    "multiple-calls",
    "swizzle-fields",
)


def _source(case):
    declarations = """
struct Params { uint bias; uint scale; };
struct Cursor { constant Params* row; };
struct Envelope { Cursor cursor; };
uint sum(Cursor cursor) { return cursor.row[0].bias + cursor.row[0].scale; }
uint nested_sum(Envelope envelope) { return sum(envelope.cursor); }
"""
    body = "Cursor cursor{params}; uint value = sum(cursor);"
    if case == "shift":
        body = "Cursor cursor{params + 1}; uint value = sum(cursor);"
    elif case == "inline":
        body = "uint value = sum(Cursor{params + 1});"
    elif case == "nested":
        body = (
            "Envelope envelope{Cursor{params + 1}}; uint value = nested_sum(envelope);"
        )
    elif case == "copy":
        body = "Cursor cursor{params + 1}; Cursor copied = cursor; uint value = sum(copied);"
    elif case == "assign":
        body = "Cursor cursor{params}; Cursor copied{params + 1}; cursor = copied; uint value = sum(cursor);"
    elif case == "member-assign":
        body = "Envelope envelope{Cursor{params}}; envelope.cursor = Cursor{params + 1}; uint value = nested_sum(envelope);"
    elif case == "rebase":
        body = "Cursor cursor{params}; cursor.row += 1; uint value = sum(cursor);"
    elif case == "branch":
        body = "Cursor cursor{params}; if ((tid & 1u) != 0u) { cursor.row = params + 1; } uint value = sum(cursor);"
    elif case == "multiple-calls":
        body = "Cursor first{params}; Cursor second{params + 1}; uint value = sum(first) + sum(second);"
    elif case == "unknown":
        body = "Cursor cursor{params + tid}; uint value = sum(cursor);"
    elif case == "mutated":
        declarations += (
            "void move(thread Cursor& cursor, uint offset) { cursor.row += offset; }"
        )
        body = "Cursor cursor{params}; move(cursor, tid); uint value = sum(cursor);"
    elif case in {"swizzle-fields", "swizzle-mutation"}:
        declarations += "struct Palette { constant Params* r; constant Params* x; };"
        offset = "1" if case == "swizzle-fields" else "tid"
        body = (
            "Palette palette{params, params + 1}; "
            f"palette.r = params + {offset}; "
            "uint value = palette.r[0].bias + palette.x[0].scale;"
        )
    elif case == "unknown-second":
        body = "Cursor first{params}; Cursor second{params + tid}; uint value = sum(first) + sum(second);"
    elif case == "wide":
        body = "Cursor cursor{params + 4294967296ul}; uint value = sum(cursor);"
    elif case in {"wrapped-shift", "wrapped-index", "wrapped-local"}:
        expression = "(4294967295u + 1u) / 4u - 1u"
        if case == "wrapped-shift":
            body = (
                f"Cursor cursor{{params + ({expression})}}; uint value = sum(cursor);"
            )
        else:
            if case == "wrapped-local":
                declarations = declarations.replace(
                    "return cursor.row",
                    f"uint index = {expression}; return cursor.row",
                    1,
                )
                expression = "index"
            declarations = declarations.replace(
                "cursor.row[0]", f"cursor.row[{expression}]"
            )
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void parameters(constant Params* params [[buffer(0)]],
                       device uint* results [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {{
    {body}
    results[4u + (tid & 3u)] = value + tid;
}}
"""


def _request(root, target, case):
    source = _source(case)
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    guard = 0x5A1B2C3D

    def value(words, shape=None):
        return {"dtype": "uint32", "shape": shape or [len(words)], "values": words}

    words = []
    for invocation in range(4):
        base = 30 if case == "zero" else 108
        if case == "branch":
            base = 108 if invocation & 1 else 30
        elif case == "multiple-calls":
            base = 138
        words.append(base + invocation)
    inputs = _bound_values(
        descriptor,
        {
            "params": value([[11, 19], [101, 7]], [2, 2]),
            "results": value([guard] * 4 + [0] * 4 + [guard] * 4),
        },
    )
    expected = _bound_values(
        descriptor, {"results": value([guard] * 4 + words + [guard] * 4)}
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, expected


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", ["opengl", "directx", "metal"])
def test_resource_member_bounds_compile(tmp_path, target, case):
    request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_resource_member_bounds_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required resource member execution")
    target = {"linux": "opengl", "darwin": "metal", "win32": "directx"}[sys.platform]
    request, expected = _request(tmp_path, target, case)
    options = {}
    if target == "metal":
        options = {
            "original_source": _source(case),
            "original_entry": "parameters",
            "metal_compile_flags": ("-Wall", "-Wextra", "-Werror"),
        }
    _execute(request, expected, tmp_path, **options)


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "mutated",
        "unknown-second",
        "wide",
        "swizzle-mutation",
        "wrapped-shift",
        "wrapped-index",
        "wrapped-local",
    ],
)
def test_resource_member_bounds_reject_unproven_offsets(tmp_path, case):
    (tmp_path / "source.metal").write_text(_source(case))
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("source.metal",),
            targets=("opengl",),
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1, report
    assert any(
        d["code"] == "project.translate.opengl-index-type-unsupported"
        for d in report["diagnostics"]
    ), report
    assert not list((tmp_path / "out").rglob("*.glsl"))


def _summary(source, name="make", constants=None):
    ast = parse("shader Fields { struct Item { int64_t offset; }; " + source + " }")
    generator = GLSLCodeGen()
    generator.struct_member_types = {"Item": {"offset": "int64_t"}}
    functions = generator.collect_functions(ast)
    generator.function_definitions = {function.name: function for function in functions}
    generator.glsl_function_overloads_by_name = {}
    for function in functions:
        generator.glsl_function_overloads_by_name.setdefault(function.name, []).append(
            function
        )
    return generator.glsl_simple_struct_constructor_member_intervals(
        FunctionCallNode(IdentifierNode(name), []), {}, constants or {}
    )


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "Item make() { Item result; result.offset = 3; return result; }",
            {"offset": (3, 3)},
        ),
        (
            "Item make() { Item result; result.offset = 3; result.offset += 2; return result; }",
            {"offset": (5, 5)},
        ),
        (
            "Item make() { Item result; result.offset = 3; result.offset -= 2; return result; }",
            {"offset": (1, 1)},
        ),
        (
            "Item make() { Item result; result.offset = 3; result.offset = 9; return result; }",
            {"offset": (9, 9)},
        ),
        (
            "Item make() { Item result; result.offset = 3; if (result.offset > 1) { result.offset = 9; } return result; }",
            {},
        ),
        ("Item make() { return make(); }", {}),
        (
            "Item make() { Item result; result.offset = uint(4294967295u + 1u); return result; }",
            {},
        ),
        (
            "Item make() { Item result; result.offset = 9223372036854775807l; result.offset += 1; return result; }",
            {},
        ),
        (
            "Item make() { Item result; result.offset = 3; Item& alias = result; alias.offset = 9; return result; }",
            {},
        ),
        (
            "Item narrow(uint x) { Item result; result.offset = x; return result; } Item make() { return narrow(4294967296l); }",
            {},
        ),
        (
            "Item shift(Item item) { item.offset += 2; return item; } Item make() { Item result; result.offset = 3; return shift(result); }",
            {"offset": (5, 5)},
        ),
        (
            "Item copy(inout Item item) { return item; } Item make() { Item result; result.offset = 3; return copy(result); }",
            {},
        ),
        (
            "int change(inout Item item) { item.offset = 9; return 1; } Item make() { Item result; result.offset = 3; result.offset = change(result); return result; }",
            {},
        ),
    ],
)
def test_straight_line_member_summaries(source, expected):
    assert _summary(source) == expected


@pytest.mark.parametrize(
    "body,expected",
    [
        ("return shift(shift(item, 4), 2);", {"offset": (6, 6)}),
        ("return shift(shift(shift(item, 4), 2), -1);", {"offset": (5, 5)}),
        ("return shift(shift(item, unknown), 2);", {}),
        ("return shift(shift(item, 2147483648l), -1);", {}),
        ("return shift(shift(item, 4), unknown++);", {}),
        ("item.offset = 9223372036854775807l; return shift(shift(item, 1), -1);", {}),
    ],
)
def test_nested_member_summaries_preserve_argument_contracts(body, expected):
    source = (
        "Item shift(Item item, int delta) { item.offset += delta; return item; } "
        "Item make() { Item item; item.offset = 0; " + body + " }"
    )
    assert _summary(source) == expected


@pytest.mark.parametrize(
    "helpers",
    [
        "Item shift(Item item, int delta) { return shift(item, delta); }",
        "Item shift(Item item, int delta) { return step(item, delta); } "
        "Item step(Item item, int delta) { return shift(item, delta); }",
        "Item shift(Item item, int delta) { return shift(shift(item, delta), delta); }",
    ],
)
def test_recursive_member_summaries_remain_unknown(helpers):
    assert (
        _summary(
            helpers
            + "Item make() { Item item; item.offset = 0; return shift(item, 2); }"
        )
        == {}
    )


@pytest.mark.parametrize(
    "parameter,body,argument,expected",
    [
        ("const int* unused", "item.offset += delta;", "buffer", {"offset": (2, 2)}),
        ("device int* unused", "item.offset += delta;", "buffer", {"offset": (2, 2)}),
        ("int* unused", "item.offset += delta;", "buffer", {}),
        ("inout Item unused", "item.offset += delta;", "item", {}),
        ("out Item unused", "item.offset += delta;", "item", {}),
        ("Item& unused", "item.offset += delta;", "item", {}),
        ("int* unused", "item.offset += delta;", "buffer++", {}),
        ("int* buffer", "item.offset += *buffer;", "buffer", {}),
        ("inout Item alias", "alias.offset = 9; item.offset += delta;", "item", {}),
    ],
)
def test_member_summaries_ignore_only_unused_reference_parameters(
    parameter, body, argument, expected
):
    assert (
        _summary(
            f"Item shift(Item item, int delta, {parameter}) {{ {body} return item; }}"
            f"Item make() {{ Item item; item.offset = 0; return shift(item, 2, {argument}); }}"
        )
        == expected
    )


def test_finite_nested_member_summaries_do_not_repeat_argument_analysis(monkeypatch):
    original = GLSLCodeGen.glsl_simple_struct_constructor_member_intervals
    calls = 0

    def observe(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        GLSLCodeGen, "glsl_simple_struct_constructor_member_intervals", observe
    )
    expression = "item"
    for _ in range(16):
        expression = f"shift({expression}, 1)"
    assert _summary(
        "Item shift(Item item, int delta) { item.offset += delta; return item; } "
        "Item make() { Item item; item.offset = 0; return " + expression + "; }"
    ) == {"offset": (16, 16)}
    assert calls < 100


@pytest.mark.parametrize("known", [False, True])
def test_struct_fields_are_not_vector_component_aliases(known):
    generator = GLSLCodeGen()
    generator.struct_member_types = {"Fields": {"r": "int64_t", "x": "int64_t"}}
    generator.local_variable_source_types = {"value": "Fields"}
    expression = MemberAccessNode(IdentifierNode("value"), "r")
    intervals = {"value.x": (1, 1)}
    if known:
        intervals["value.r"] = (7, 9)
    expression._glsl_control_flow_intervals = intervals
    result = generator.glsl_index_flow_range(expression)
    assert (None if result is None else (result.minimum, result.maximum)) == (
        (7, 9) if known else None
    )
    assert generator.glsl_private_pointer_interval(expression, intervals, {}) == (
        (7, 9) if known else None
    )
    assert generator.glsl_interval_expression_keys(expression) == {"value.r"}
    assert generator.glsl_interval_mutation_target_keys(expression) == {"value.r"}


@pytest.mark.parametrize(
    "source",
    [
        "Item take(uint index) { Item result; result.offset = index; return result; } "
        "Item make() { return take(unbounded); }",
        "Item make() { uint index; Item result; result.offset = index; return result; }",
    ],
)
def test_member_summaries_do_not_capture_caller_constants(source):
    assert _summary(source, constants={"index": 17}) == {}


def test_member_summary_rejects_overflow_inside_argument():
    source = (
        "Item take(uint index) { Item result; result.offset = index; return result; } "
        "Item make() { return take((4294967295u + 1u) / 4u - 1u); }"
    )
    assert _summary(source) == {}


def test_resource_member_bounds_ci_contract():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_resource_member_bounds.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
