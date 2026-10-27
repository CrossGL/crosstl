"""Template deduction from member-alias argument types, and member-alias chains.

Template argument deduction compares a parameter type with the type of the call argument ([temp.deduct.call]). A
reference returned by another struct's method is typed through that struct's member alias, so
``Tile<T, 2>::frag_at`` yields ``Frag<T, 8, 8>::frag_type``, which must compare equal to the ``metal::vec<Atype, 2>``
that a member alias template ``dtype_frag_t<Atype>`` names. A vector shorthand such as ``uint2`` is
``metal::vec<uint, 2>`` (Metal Shading Language Specification, section 2.2). A qualified name such as
``tile_t::Frag_t::kElemCols`` names a static member of the struct that the member alias ``Tile::Frag_t`` denotes
([class.qual]), and a member typedef ``typename Frag_t::frag_type`` names a member of that struct too. MLX's
attention kernels use all of these in ``BaseMMAFrag::mma`` and in their score masks.

A statement that follows a trailing comment (``constexpr short n = 8; // size``) starts after the comment, so a local
``using`` alias declared on the next line is still seen when template arguments are deduced.
"""

import os
import sys

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.project.pipeline import _metal_statement_spans
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_MEMBER_ALIAS_DEDUCTION"
ENTRY = "mads_int"
WORKGROUP = (32, 1, 1)
GROUPS = 2
ROW = 6
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 2003) - 1001 for index in range(GROUPS * 32 * ROW)]

SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T, int R, int C>
struct Frag {
  static_assert(R == 4, "only the 8x8 specialization is used");
};

template <typename T>
struct Frag<T, 8, 8> {
  static constant constexpr const short kElemCols = (8 * 8) / 32;
  typedef metal::vec<T, kElemCols> frag_type;
  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemCols>;

  template <typename Atype, typename Btype, typename Ctype>
  static constexpr void mad(
      thread frag_type& D,
      thread dtype_frag_t<Atype>& A,
      thread dtype_frag_t<Btype>& B,
      thread dtype_frag_t<Ctype>& C) {
    D = frag_type(A) * frag_type(B) + frag_type(C);
  }
};

template <typename T, int N, class Frag_ = Frag<T, 8, 8>>
struct Tile {
  using Frag_t = Frag_;
  typedef typename Frag_t::frag_type frag_type;
  frag_type val_frags[N];
  constexpr thread frag_type& frag_at(const short i) thread {
    return val_frags[i];
  }
};

template <typename T>
[[kernel]] void mads(
    const device T* values [[buffer(0)]],
    device T* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  constexpr short kFragSize = 8; // the fragment edge
  using F = Frag<T, kFragSize, kFragSize>;
  const device T* row = values + index * 6;
  Tile<T, 2, F> acc;
  Tile<T, 2, F> lhs;
  acc.frag_at(0) = metal::vec<T, 2>(row[0], row[1]);
  lhs.frag_at(0) = metal::vec<T, 2>(row[2], row[3]);
  lhs.frag_at(1) = metal::vec<T, 2>(row[4], row[5]);
  F::mad(acc.frag_at(1), lhs.frag_at(0), lhs.frag_at(1), acc.frag_at(0));
  uint2 scale = uint2(index % 3 + 1, 2);
  F::mad(acc.frag_at(0), scale, lhs.frag_at(1), acc.frag_at(1));
  using tile_t = decltype(acc);
  T total = T(0);
  for (short jj = 0; jj < tile_t::Frag_t::kElemCols; jj++) {
    total += T(jj + 1) * (acc.val_frags[0][jj] + T(3) * acc.val_frags[1][jj]);
  }
  results[index] = total;
}

template [[host_name("mads_int")]] [[kernel]] void mads<int>(
    const device int*, device int*, uint);
"""


def _expected():
    results = []
    for index in range(GROUPS * 32):
        r = VALUES[index * ROW : (index + 1) * ROW]
        acc0, l0, l1 = r[0:2], r[2:4], r[4:6]
        acc1 = [l0[i] * l1[i] + acc0[i] for i in range(2)]
        scale = [index % 3 + 1, 2]
        acc0 = [scale[i] * l1[i] + acc1[i] for i in range(2)]
        results.append(sum((jj + 1) * (acc0[jj] + 3 * acc1[jj]) for jj in range(2)))
    return results + GUARD


def _request(root, target):
    source, descriptor, package = _package(
        root, target, "int", WORKGROUP, source=SOURCE, software_subgroups=False
    )
    expected = _expected()
    initial = [-0x21524111] * (len(expected) - len(GUARD)) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "int32", "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "int32", "shape": [len(initial)], "values": initial},
        },
    )
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


def _translate(root, target, source):
    (root / "mads.metal").write_text(source)
    return translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("mads.metal",),
            targets=(target,),
            output_dir="out",
            entry_points={"mads.metal": (ENTRY,)},
            workgroup_size=WORKGROUP,
        ),
        format_output=False,
    ).to_json()


def test_statement_spans_start_after_trailing_comments():
    source = (
        "{ constexpr short n = 8; // the edge\n"
        "  using F = Frag<T, n>; /* lead */ F x;\n"
        "  // only a comment\n}"
    )
    statements = [
        source[start:end]
        for start, end in _metal_statement_spans(source, 1, len(source))
    ]
    assert statements == [
        "constexpr short n = 8;",
        "using F = Frag<T, n>;",
        "F x;",
    ]


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_member_alias_deduction_compiles(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    for spelling in ("frag_at", "Frag_t", "dtype_frag_t", "tile_t"):
        assert spelling not in generated, spelling
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "replacement",
    (
        # Atype and Btype must be the same element type for a shared parameter.
        (
            "thread dtype_frag_t<Btype>& B,",
            "thread dtype_frag_t<Atype>& B,",
        ),
        # A three-component shorthand is not a two-component fragment.
        ("uint2 scale = uint2(index % 3 + 1, 2);", "uint3 scale = uint3(1, 2, 3);"),
    ),
)
def test_inconsistent_deductions_stay_refused(tmp_path, target, replacement):
    report = _translate(tmp_path, target, SOURCE.replace(*replacement))
    assert report["summary"]["translatedCount"] == 0
    messages = [item["message"] for item in report["diagnostics"]]
    assert any("did not bind consistently" in message for message in messages), messages


def test_member_alias_deduction_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native member alias deduction")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_member_alias_deduction_is_required_on_all_native_platforms():
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
    assert "test_metal_member_alias_deduction.py" in step
    assert "if:" not in step and "continue-on-error" not in step
