"""Owned aggregate fields retain checked values across helper calls."""

import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.translator.ast import (
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PrimitiveType,
    ReferenceType,
)
from crosstl.translator.codegen.scalar_return_bounds import (
    aggregate_call_intervals,
    scalar_return_interval,
)
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_pointer_index_bounds import _operand_config
from tests.test_translator.test_project_metadata_entry_scope import _packages
from tests.test_translator.test_scalar_return_bounds import _call, _generator


def _aggregate_generator(declarations, *, scalar="int64"):
    generator = _generator(declarations)
    generator.struct_member_types["Cursor"] = {"offset": scalar, "unused": "int"}
    generator.local_variable_source_types.update(cursor="Cursor", step="int")
    return generator


def test_readonly_aggregate_getter_uses_only_initialized_declared_fields():
    generator = _aggregate_generator(
        "int64 locate(inout Cursor state) { return state.offset; }"
    )
    call = _call("locate", "cursor")
    assert scalar_return_interval(generator, call, {"cursor.offset": (2, 7)}, {}) == (
        2,
        7,
    )
    assert (
        scalar_return_interval(generator, call, {"cursor.unused": (2, 7)}, {}) is None
    )
    assert scalar_return_interval(generator, call, {"state.offset": (2, 7)}, {}) is None
    assert generator._glsl_return_bounds_active is False


@pytest.mark.parametrize(
    "scalar,initial,step,expected",
    [
        ("int64", (1, 2), (2, 3), (3, 5)),
        ("int16", (32767, 32767), (1, 1), (-32768, -32768)),
        ("uint16", (65535, 65535), (1, 1), (0, 0)),
    ],
)
def test_aggregate_update_applies_source_storage_conversion(
    scalar, initial, step, expected
):
    generator = _aggregate_generator(
        "void advance(inout Cursor state, int step) { state.offset += step; }",
        scalar=scalar,
    )
    intervals = {"cursor.offset": initial, "step": step}
    assert aggregate_call_intervals(
        generator, _call("advance", "cursor", "step"), intervals, {}
    ) == {"cursor.offset": expected}
    assert intervals == {"cursor.offset": initial, "step": step}


def test_aggregate_helper_loop_and_nested_updates_keep_callee_ownership():
    generator = _aggregate_generator("""
        void advance(inout Cursor state, int n) { state.offset += n; }
        void repeat(inout Cursor state, int n) {
            for (int i = 0; i < 3; ++i) { advance(state, n); }
        }
    """)
    assert aggregate_call_intervals(
        generator,
        _call("repeat", "cursor", "step"),
        {"cursor.offset": (0, 0), "step": (2, 2)},
        {},
    ) == {"cursor.offset": (6, 6)}


@pytest.mark.parametrize(
    "declaration",
    [
        "void advance(inout Cursor state, int n) { state.offset += hidden; }",
        "void advance(inout Cursor state, int n) { unknown(state); }",
        "void advance(inout Cursor state, int n) { advance(state, n); }",
        "void advance(inout Cursor state, int n) { if (n > 0) state.offset += n; }",
        "void advance(inout Cursor state, int n) { state.offset += 9223372036854775807l; }",
    ],
)
def test_aggregate_updates_reject_unmodelled_effects(declaration):
    generator = _aggregate_generator(declaration)
    assert (
        aggregate_call_intervals(
            generator,
            _call("advance", "cursor", "step"),
            {"cursor.offset": (1, 1), "step": (2, 2), "hidden": (0, 0)},
            {"hidden": 0},
        )
        is None
    )


def test_aliased_aggregate_parameters_cannot_supply_updates():
    generator = _aggregate_generator("""
        void advance(inout Cursor first, inout Cursor second) {
            first.offset += 1; second.offset += first.offset;
        }
    """)
    assert (
        aggregate_call_intervals(
            generator,
            _call("advance", "cursor", "cursor"),
            {"cursor.offset": (1, 1)},
            {},
        )
        is None
    )


def test_value_parameter_updates_do_not_change_the_callers_aggregate():
    generator = _aggregate_generator("""
        void advance(inout Cursor state, int n) { state.offset += n; }
        int64 locate(Cursor state) { advance(state, 2); return state.offset; }
    """)
    intervals = {"cursor.offset": (1, 1)}
    assert scalar_return_interval(
        generator, _call("locate", "cursor"), intervals, {}
    ) == (3, 3)
    assert (
        aggregate_call_intervals(generator, _call("locate", "cursor"), intervals, {})
        == {}
    )
    assert intervals == {"cursor.offset": (1, 1)}


def test_readonly_field_reference_cannot_alias_a_mutated_aggregate():
    generator = _aggregate_generator("""
        void advance(inout Cursor state, int64 prior) {
            state.offset += 1; state.offset += prior;
        }
    """)
    function = generator.function_definitions["advance"]
    function.parameters[1].param_type = ReferenceType(
        PrimitiveType("int64"), address_space="thread", is_mutable=False
    )
    call = FunctionCallNode(
        IdentifierNode("advance"),
        [
            IdentifierNode("cursor"),
            MemberAccessNode(IdentifierNode("cursor"), "offset"),
        ],
    )
    assert (
        aggregate_call_intervals(generator, call, {"cursor.offset": (1, 1)}, {}) is None
    )


@pytest.mark.parametrize(
    "body",
    [
        "state.offset += state.offset;",
        "state.offset += state.unused;",
        "state.offset *= n;",
        "state.offset = n;",
        "state.offset += n; state.offset += n;",
    ],
)
def test_additive_summary_requires_one_state_independent_update(body):
    generator = _aggregate_generator(
        f"void advance(inout Cursor state, int n) {{ {body} }}"
    )
    assert (
        aggregate_call_intervals(
            generator,
            _call("advance", "cursor", "step"),
            {"cursor.offset": (1, 1), "cursor.unused": (1, 1), "step": (1, 1)},
            {},
            additive=True,
        )
        is None
    )


CASES = {
    "getter": ("cursor.offset = 1;", 1),
    "advance": ("cursor.next(2);", 2),
    "repeat": ("cursor.repeat(1);", 3),
    "copy": ("cursor.next(1); Cursor copy = cursor; copy.next(2);", 1),
    "sequence": ("cursor.next(3); cursor.next(-1);", 2),
    "by-value": ("cursor.next(1); mutate_copy(cursor);", 1),
    "caller-loop": ("for (int i = 0; i < 3; ++i) cursor.next(1);", 3),
    "field-loop": ("for (int i = 0; i < 3; ++i) cursor.offset += 1;", 3),
    "long-loop": ("for (int i = 0; i < 128; ++i) cursor.next(1);", 128),
    "strided-loop": ("for (long i = 1; i < 9; i += 3) cursor.next(2);", 6),
    "inclusive-stride": ("for (long i = 1; i <= 10; i += 3) cursor.next(2);", 8),
    "zero-trip-stride": ("for (long i = 10; i < 9; i += 3) cursor.next(-1);", 0),
    "varying-start": ("for (long i = long(tid % 3); i < 9; i += 3) cursor.next(2);", 6),
    "varying-stride": (
        "short step = short(tid % 2) + 2; for (long i = long(tid % 3); i < 7; i += step) cursor.next(1);",
        (4, 2, 3, 3),
    ),
}


def _source(body):
    return f"""#include <metal_stdlib>
using namespace metal;
struct Cursor {{
    long offset = 0;
    Cursor(int) thread {{}}
    void next(int n) thread {{ offset += n; }}
    void repeat(int n) thread {{ for (int i = 0; i < 3; ++i) next(n); }}
    long location() thread {{ return offset; }}
}};
void mutate_copy(Cursor value) {{ value.next(2); }}
kernel void select_value(const device uint* input [[buffer(0)]],
                         device uint* output [[buffer(1)]],
                         uint tid [[thread_index_in_threadgroup]]) {{
    Cursor cursor(0);
    {body}
    const long at = cursor.location();
    const device uint* selected = input + at;
    output[tid] = selected[0];
}}
"""


def _config(root, target, source):
    return replace(
        _operand_config(root, target, "uint8", source=source), workgroup_size=(4, 1, 1)
    )


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("case", CASES)
def test_private_aggregate_state_survives_project_packages(tmp_path, target, case):
    config = _config(tmp_path, target, _source(CASES[case][0]))
    assert not config.index_range_assertions
    report, _, descriptors = _packages(config)
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    assert not report["diagnostics"] and len(descriptors) == 1


@pytest.mark.parametrize(
    "body",
    [
        "cursor.next(-1);",
        "cursor.offset = -1;",
        "cursor.offset = 1; thread Cursor& alias = cursor; alias.offset = -1;",
        "for (int i = 0; i < 2; ++i) cursor.next(-1);",
        "for (int i = 0; i < 2; ++i) cursor.next(cursor.offset + 1);",
        "for (int i = 0; i < 2; ++i) { cursor.next(1); cursor.offset = -1; }",
        "int step = 1; for (int i = 0; i < 3; i += step) { cursor.next(1); step = 0; }",
        "for (int i = 2147483646; i < 2147483647; i += 2) cursor.next(1);",
        "for (short i = 32766; i < 32767; i += 2) cursor.next(1);",
        "for (int i = 1; i < 3; i += i) cursor.next(1);",
        "for (int i = 0; i < 3; i += 2) { cursor.next(1); i = 0; }",
    ],
)
def test_unsafe_or_unproved_aggregate_offsets_remain_diagnostic(tmp_path, body):
    config = _config(tmp_path, "opengl", _source(body))
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert payload["diagnostics"]


def test_volatile_aggregate_cannot_supply_helper_return_bounds(tmp_path):
    source = _source("cursor.offset = 1;").replace(
        "Cursor cursor(0);", "volatile Cursor cursor(0);"
    )
    config = _config(tmp_path, "opengl", source)
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert payload["diagnostics"]


@pytest.mark.parametrize("scalar,initial", [("int", 2147483646), ("short", 32766)])
def test_caller_loop_cannot_prove_a_range_across_storage_overflow(
    tmp_path, scalar, initial
):
    source = _source("for (int i = 0; i < 2; ++i) cursor.next(1);").replace(
        "long offset = 0;", f"{scalar} offset = {initial};"
    )
    config = _config(tmp_path, "opengl", source)
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert payload["diagnostics"]


@pytest.mark.parametrize("case", CASES)
def test_private_aggregate_helpers_execute_natively(tmp_path, case):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip("set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native execution")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    body, index = CASES[case]
    source = _source(body)
    config = _config(tmp_path, target, source)
    _, package, (descriptor,) = _packages(config)
    values = [17 + 13 * index for index in range(132)]
    guard = 0xABCDEF01
    indices = index if isinstance(index, tuple) else (index,) * 4
    expected = [values[item] for item in indices] + [guard] * 32
    inputs, outputs, observed = {}, {}, []
    for binding in descriptor["bindings"]:
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        data = values if parameter == "input" else [guard] * len(expected)
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
        dict(workgroupCount=[1, 1, 1], workgroupSize=[4, 1, 1]),
        observe_inputs=observed,
    )
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
