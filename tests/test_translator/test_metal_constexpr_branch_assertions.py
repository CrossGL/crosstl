"""Static assertions in instantiated Metal templates: static-member constants and discarded ``if constexpr`` branches.

``constexpr int stride = decltype(tile)::kElemsPerTile * 32;`` reads a constant static data member, so it is a
constant expression ([expr.const]) and an assertion over it is checked at translation time, as MLX's attention
kernels size their score exchange. During the instantiation of a template, the substatement an ``if constexpr``
discards is not instantiated ([stmt.if]/2), so an assertion that would fail there never fires. A false assertion
in the taken branch, and one in a discarded branch of a non-template function, which is fully checked, are still
refused.
"""

import os
import sys

import pytest

from crosstl.backend.Metal.constexpr_branches import (
    discarded_constexpr_branch_spans,
    substitute_static_member_constants,
)
from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_CONSTEXPR_BRANCH_ASSERTIONS"
ENTRY = "exchange_int"
WORKGROUP = (32, 1, 1)
GROUPS = 3
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [(index * 2654435761 + 11) % 100003 for index in range(GROUPS * 32)]

# (WM, WN, N). Each layout satisfies the assertion of its taken branch and would fail the assertions of the
# branches it discards: under (4, 1, 32) the exchange needs 256 of 128 slots; under (1, 2, 16) the rows fill 32
# of 64 slots; neither has WN == 4.
LAYOUTS = {"rows": (4, 1, 32), "exchange": (1, 2, 16)}


def _source(wm, wn, n):
    return f"""#include <metal_stdlib>
using namespace metal;

template <typename T, int R, int C>
struct Tile {{
  static constexpr constant int kElemsPerTile = R * C;
  T vals[R * C];
}};

template <typename T, int WM, int WN, int N>
[[kernel]] void exchange(
    const device T* values [[buffer(0)]],
    device T* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {{
  Tile<T, 2, 2> tile;
  for (int i = 0; i < 4; i++) {{
    tile.vals[i] = values[index] + T(i);
  }}
  constexpr int stride = decltype(tile)::kElemsPerTile * 8;
  constexpr int capacity = N * 4;
  T bonus = T(0);
  if constexpr (WN == 2) {{
    constexpr int size = WM * 2 * stride;
    static_assert(size * sizeof(T) <= capacity * sizeof(T), "exchange must fit");
    bonus = tile.vals[3] + T(size);
  }} else if constexpr (WN == 1) {{
    static_assert(WM * stride == capacity, "rows must fill the buffer");
    bonus = tile.vals[1] + T(stride);
  }} else {{
    static_assert(WN == 4, "unsupported layout");
    bonus = T(-1);
  }}
  results[index] = tile.vals[0] + bonus + T(Tile<T, 2, 2>::kElemsPerTile);
}}

template [[host_name("{ENTRY}")]] [[kernel]] void exchange<int, {wm}, {wn}, {n}>(
    const device int*, device int*, uint);
"""


def _expected(wm, wn):
    stride = 4 * 8
    if wn == 2:
        bonuses = [value + 3 + wm * 2 * stride for value in VALUES]
    else:
        bonuses = [value + 1 + stride for value in VALUES]
    return [value + bonus + 4 for value, bonus in zip(VALUES, bonuses)] + GUARD


def _request(root, target, layout):
    wm, wn, n = LAYOUTS[layout]
    source, descriptor, package = _package(
        root,
        target,
        "int",
        WORKGROUP,
        source=_source(wm, wn, n),
        software_subgroups=False,
    )
    initial = [-0x21524111] * len(VALUES) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "int32", "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "int32", "shape": [len(initial)], "values": initial},
        },
    )
    expected = _expected(wm, wn)
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "int32", "shape": [len(expected)], "values": expected}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


def _translate(root, target, source, entry=ENTRY):
    (root / "branches.metal").write_text(source)
    return translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("branches.metal",),
            targets=(target,),
            output_dir="out",
            entry_points={"branches.metal": (entry,)},
            workgroup_size=WORKGROUP,
        ),
        format_output=False,
    ).to_json()


def _assertion_failures(report):
    return [
        item
        for item in report["diagnostics"]
        if item["code"] == "project.translate.metal-static-assertion"
    ]


def _values(table):
    return lambda owner, member: table.get((owner, member))


def test_static_members_fold_only_to_their_values():
    lookup = _values(
        {
            ("Tile_int_2_2", "kElemsPerTile"): "4",
            ("Bias", "low"): "-3",
            ("Frag<float,1>", "kRows"): "short(2)",
        }
    )
    types = {"tile": "Tile_int_2_2"}
    assert (
        substitute_static_member_constants(
            "decltype(tile)::kElemsPerTile * 32 + Bias::low - Frag<float, 1>::kRows",
            types,
            lookup,
        )
        == "4 * 32 + (-3) - (short(2))"
    )
    for expression in (
        "decltype((tile))::kElemsPerTile",
        "decltype(other)::kElemsPerTile",
        "Tile_int_2_2::kElemsPerTile(1)",
        "Tile_int_2_2::kElemsPerTile<2>",
        "Tile_int_2_2::unknown",
        "tile.kElemsPerTile",
    ):
        assert (
            substitute_static_member_constants(expression, types, lookup) == expression
        ), expression


def _discarded(code, values):
    def evaluate(start, end):
        return values.get(code[start:end].strip())

    return [
        code[start:end]
        for start, end in discarded_constexpr_branch_spans(
            code, [(0, len(code))], evaluate
        )
    ]


def test_discarded_branches_follow_folded_conditions_and_else_if_chains():
    chain = "if constexpr (A) { a; } else if constexpr (B) { b; } else { c; }"
    assert _discarded(chain, {"A": True}) == ["if constexpr (B) { b; } else { c; }"]
    assert _discarded(chain, {"A": False, "B": False}) == ["{ a; }", "{ b; }"]
    assert _discarded(chain, {"A": False, "B": True}) == ["{ a; }", "{ c; }"]
    # An unknown outer condition leaves its branches alone, yet the inner one still folds.
    assert _discarded(chain, {"B": True}) == ["{ c; }"]
    nested = "if constexpr (A) { if constexpr (B) { b; } } else { c; }"
    assert _discarded(nested, {"A": False, "B": True}) == [
        "{ if constexpr (B) { b; } }"
    ]
    assert _discarded("if constexpr (A) { a; }", {"A": True}) == []
    assert _discarded("if (A) { a; } else { c; }", {"A": True}) == []


def test_undelimited_branches_are_never_discarded():
    for code in (
        "if constexpr (A) a; else { c; }",
        "if constexpr (A) { a; } else c;",
        "if constexpr (A) { a; } else if constexpr (B) b;",
        "if constexpr (A) { a;",
    ):
        assert _discarded(code, {"A": False, "B": False}) == [], code
        assert _discarded(code, {"A": True, "B": True}) == [], code


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_instantiated_branch_assertions_compile(tmp_path, target, layout):
    _, request, _ = _request(tmp_path, target, layout)
    generated = request.artifact_path.read_text()
    assert "static_assert" not in generated and "decltype" not in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_false_assertion_in_the_taken_branch_is_refused(tmp_path, target):
    report = _translate(tmp_path, target, _source(4, 2, 32))
    assert report["summary"]["translatedCount"] == 0
    (failure,) = _assertion_failures(report)
    details = failure["details"]["staticAssertion"]
    assert details["reason"] == "assertion-failed", details
    assert details["assertionMessage"] == "exchange must fit"
    # size = WM * 2 * stride = 4 * 2 * 32 and capacity = N * 4 = 128 slots of sizeof(int) bytes.
    assert details["resolvedExpression"] == "256 * 4 <= 128 * 4", details


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_discarded_branch_of_a_non_template_kernel_is_still_checked(tmp_path, target):
    source = """#include <metal_stdlib>
using namespace metal;

[[kernel]] void plain(device int* results [[buffer(0)]],
                      uint index [[thread_position_in_grid]]) {
  if constexpr (sizeof(int) == 2) {
    static_assert(sizeof(int) == 2, "never true");
    results[index] = 1;
  } else {
    results[index] = 2;
  }
}
"""
    report = _translate(tmp_path, target, source, entry="plain")
    assert report["summary"]["translatedCount"] == 0
    (failure,) = _assertion_failures(report)
    assert failure["details"]["staticAssertion"]["reason"] == "assertion-failed"


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_instantiated_branch_assertions_execute(tmp_path, layout):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native branch assertions")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target, layout)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_branch_assertions_are_required_on_all_native_platforms():
    from pathlib import Path

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_metal_constexpr_branch_assertions.py" in step
    assert "if:" not in step and "continue-on-error" not in step
