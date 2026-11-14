"""Member calls as template member call arguments, array-element receivers and alias-typed locals.

A template member call such as ``Frag<T>::load(tile.frag_at(0), src, 1)`` is lowered once the types of its arguments
are known, so a member call inside the argument list is lowered first, in its own lexical scope, and the argument is
typed as the lowered expression. An element of an array of structs is an object of that struct ([expr.sub]), so
``tiles[k].frag_at(0)`` and ``tiles[k].weighted()`` are calls on that object; an index that would need lowering
itself or has side effects keeps the call refused. A local declared through an alias of a struct-scoped type
(``using frag_t = typename Frag<T>::frag_type; frag_t extra;``) has the aliased type. MLX's attention kernels load
their query tiles with ``MMAFrag_t::load(Qtiles[dd].frag_at(0, 0), ...)`` and their masks into such a local.

Their row reductions add a second shape. Inside ``Tile::row_reduce``, ``Frag_t::template row_reduce<Op>(frag_at(i),
&vals[i * Frag_t::kElemRows])`` names a member of ``Frag_t``, so the ``template`` disambiguator never makes it a call
of the enclosing method; ``Frag_t`` is the default template argument ``Frag<T, 1>``, whose static members fold
through the struct materialized for it, also in the array extent of ``thread T vals[kRowsPerThread]`` and in other
structs' static initializers; and a parameter spelled ``thread const frag_type&`` is a vector whose components are
typed. Direct3D and OpenGL view ``&vals[i]`` as the array parameter plus an element offset.
"""

import os
import sys

import pytest

from crosstl.backend.Metal.array_element_receivers import (
    array_element_receiver_end,
    array_rank_at,
)
from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_MEMBER_CALL_ARGUMENTS"
ENTRY = "gather_int"
WORKGROUP = (32, 1, 1)
GROUPS = 2
K = 3
ROW = 6 + 2 * K
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
# float32 guards, each exactly representable so that a float32 buffer holds it unchanged.
FLOAT_GUARD = [1831565568.0, -7.0, 19088640.0, -94742016.0]
VALUES = [((index * 2654435761) % 2003) - 1001 for index in range(GROUPS * 32 * ROW)]

ELEMENT_LOAD = "Frag<T>::load(tiles[k].frag_at(0), row + 4 + 2 * k, 1);"


def _source(element_load=ELEMENT_LOAD):
    return f"""#include <metal_stdlib>
using namespace metal;

template <typename T>
struct Frag {{
  typedef metal::vec<T, 2> frag_type;
  template <typename SrcPtrType, typename StrX>
  static constexpr void load(thread frag_type& dst, SrcPtrType src, StrX str_x) {{
    dst[0] = static_cast<T>(src[0]);
    dst[1] = static_cast<T>(src[str_x]);
  }}
}};

template <typename T, int N>
struct Tile {{
  typedef typename Frag<T>::frag_type frag_type;
  frag_type val_frags[N];
  constexpr thread frag_type& frag_at(const short i) thread {{ return val_frags[i]; }}
  constexpr const thread frag_type& frag_at(const short i) const thread {{
    return val_frags[i];
  }}
  T weighted() const thread {{
    T total = T(0);
    for (short i = 0; i < N; i++) {{
      total += T(i + 1) * (val_frags[i].x - val_frags[i].y);
    }}
    return total;
  }}
}};

short next_index(short k) {{
  return k;
}}

template <typename T, int K>
[[kernel]] void gather(
    const device T* values [[buffer(0)]],
    device T* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {{
  const device T* row = values + index * (6 + 2 * K);
  Tile<T, 2> tile;
  Tile<T, 2> tiles[K];
  Frag<T>::load(tile.frag_at(0), row, 1);
  Frag<T>::load(tile.frag_at(1), row + 2, 1);
  for (short k = 0; k < K; k++) {{
    {element_load}
    tiles[k].frag_at(1) = tile.frag_at(k % 2);
  }}
  using frag_t = typename Frag<T>::frag_type;
  frag_t extra;
  Frag<T>::load(extra, row + 4 + 2 * K, 1);
  results[index] = tile.weighted() + tiles[K - 1].weighted() + T(3) * extra.x + extra.y;
}}

template [[host_name("{ENTRY}")]] [[kernel]] void gather<int, {K}>(
    const device int*, device int*, uint);
"""


ROW_REDUCE_ENTRY = "rowred_float"
ROW_REDUCE_SOURCE = """#include <metal_stdlib>
using namespace metal;

struct MaxOp {
  template <typename T>
  static constexpr T apply(T x, T y) { return metal::max(x, y); }
};

template <typename T, int R>
struct Frag {
  static constexpr constant int kElemRows = R;
  typedef metal::vec<T, 2> frag_type;
  template <typename Op>
  static constexpr void row_reduce(thread const frag_type& inp_vals, thread T* reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals[1]);
    reduced_vals[0] = Op::apply(reduced_vals[0], thr_reduce);
  }
};

template <typename T, int N, class Frag_ = Frag<T, 1>>
struct Tile {
  using Frag_t = Frag_;
  typedef typename Frag_t::frag_type frag_type;
  static constexpr constant int kRowsPerThread = N * Frag_t::kElemRows;
  frag_type val_frags[N];
  constexpr const thread frag_type& frag_at(const short i) const thread { return val_frags[i]; }
  template <typename Op>
  void row_reduce(thread T vals[kRowsPerThread]) const thread {
    for (short i = 0; i < N; ++i) {
      Frag_t::template row_reduce<Op>(frag_at(i), &vals[i * Frag_t::kElemRows]);
    }
  }
};

template <typename T, int N>
[[kernel]] void rowred(const device T* values [[buffer(0)]], device T* results [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
  Tile<T, N> tile;
  for (short i = 0; i < N; ++i) {
    tile.val_frags[i] = metal::vec<T, 2>(values[index * 2 * N + 2 * i], values[index * 2 * N + 2 * i + 1]);
  }
  T maxima[N];
  for (short i = 0; i < N; ++i) {
    maxima[i] = T(-1000);
  }
  tile.template row_reduce<MaxOp>(maxima);
  T total = T(0);
  for (short i = 0; i < N; ++i) {
    total += T(i + 1) * maxima[i];
  }
  results[index] = total;
}

template [[host_name("rowred_float")]] [[kernel]] void rowred<float, 2>(
    const device float*, device float*, uint);
"""
ROW_REDUCE_VALUES = [
    float(((index * 37) % 101) - 50) for index in range(GROUPS * 32 * 4)
]


def _row_reduce_expected():
    results = []
    for index in range(GROUPS * 32):
        row = ROW_REDUCE_VALUES[index * 4 : index * 4 + 4]
        results.append(max(row[0], row[1]) + 2 * max(row[2], row[3]))
    return results + FLOAT_GUARD


def _translate(root, target, source, entry):
    (root / "calls.metal").write_text(source)
    return translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("calls.metal",),
            targets=(target,),
            output_dir="out",
            entry_points={"calls.metal": (entry,)},
            workgroup_size=WORKGROUP,
        ),
        format_output=False,
    ).to_json()


def _expected():
    results = []
    for index in range(GROUPS * 32):
        row = VALUES[index * ROW : (index + 1) * ROW]
        tile = [row[0:2], row[2:4]]
        last = [row[4 + 2 * (K - 1) : 6 + 2 * (K - 1)], tile[(K - 1) % 2]]
        extra = row[4 + 2 * K : 6 + 2 * K]

        def weighted(frags):
            return sum((i + 1) * (frag[0] - frag[1]) for i, frag in enumerate(frags))

        results.append(weighted(tile) + weighted(last) + 3 * extra[0] + extra[1])
    return results + GUARD


def _request(root, target):
    source, descriptor, package = _package(
        root, target, "int", WORKGROUP, source=_source(), software_subgroups=False
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


def test_array_element_receivers_index_every_dimension_with_plain_indices():
    code = "S grid[2][3]; grid[i][j + 1].m(); grid[i].m(); grid[f(i)][0].m();"
    declaration_end = code.index("[")
    assert array_rank_at(code, declaration_end) == 2
    element = code.index("grid[i][j")
    end = array_element_receiver_end(code, code.index("[", element), 2)
    assert code[element:end] == "grid[i][j + 1]"
    for receiver in ("grid[i].m", "grid[f(i)][0]"):
        start = code.index(receiver)
        assert array_element_receiver_end(code, code.index("[", start), 2) is None
    for text in (
        "a[k++].m()",
        "a[k = 1].m()",
        "a[k += 1].m()",
        "a[b.c].m()",
        "a[p->q].m()",
        "a[(k)].m()",
        "a[k, j].m()",
        "a[c ? 1 : 0].m()",
        "a[k]->m()",
        "a[k](1)",
        "a[].m()",
    ):
        assert array_element_receiver_end(text, 1, 1) is None, text
    assert array_element_receiver_end("a[k == 1 && j <= 2].m()", 1, 1) == len(
        "a[k == 1 && j <= 2]"
    )
    assert array_element_receiver_end("a[k].m()", 1, 0) is None


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_member_call_arguments_compile(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert "frag_at" not in generated and "frag_t" not in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "element_load",
    (
        # The index calls a function, so it is not moved into the lowered call.
        "Frag<T>::load(tiles[next_index(k)].frag_at(0), row + 4 + 2 * k, 1);",
        # The index has a side effect that the lowered call would duplicate or drop.
        "short j = k; Frag<T>::load(tiles[j++].frag_at(0), row + 4 + 2 * k, 1);",
    ),
)
def test_element_receivers_with_unsafe_indices_stay_refused(
    tmp_path, target, element_load
):
    report = _translate(tmp_path, target, _source(element_load), ENTRY)
    assert report["summary"]["translatedCount"] == 0
    codes = {item["code"] for item in report["diagnostics"]}
    assert "project.translate.metal-struct-method" in codes, codes


def _row_reduce_request(root, target="metal"):
    source, descriptor, package = _package(
        root,
        target,
        "float",
        WORKGROUP,
        source=ROW_REDUCE_SOURCE,
        software_subgroups=False,
    )
    expected = _row_reduce_expected()
    initial = [-12345.0] * (len(expected) - len(GUARD)) + FLOAT_GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {
                "dtype": "float32",
                "shape": [len(ROW_REDUCE_VALUES)],
                "values": ROW_REDUCE_VALUES,
            },
            "results": {
                "dtype": "float32",
                "shape": [len(initial)],
                "values": initial,
            },
        },
    )
    outputs = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "float32",
                "shape": [len(expected)],
                "values": expected,
            }
        },
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
    return request, outputs


def test_qualified_template_member_row_reduction_compiles_for_metal(tmp_path):
    request, _ = _row_reduce_request(tmp_path)
    generated = request.artifact_path.read_text()
    for spelling in ("template", "::", "kRowsPerThread[", "Frag<"):
        assert spelling not in generated, spelling
    assert "static constant constexpr int kRowsPerThread = 2;" in generated
    _compile(generated, "metal", tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_row_reduction_through_array_parameters_compiles(tmp_path, target):
    request, _ = _row_reduce_request(tmp_path, target)
    generated = request.artifact_path.read_text()
    # The fragment reduction receives the tile's array and an element offset.
    assert "reduced_vals_base" in generated
    _compile(generated, target, tmp_path)


def test_qualified_template_member_row_reduction_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the native row reduction")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    request, expected = _row_reduce_request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=ROW_REDUCE_SOURCE,
        original_entry=ROW_REDUCE_ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_member_call_arguments_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native member call arguments")
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


def test_member_call_arguments_are_required_on_all_native_platforms():
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
    assert "test_metal_member_call_arguments.py" in step
    assert "if:" not in step and "continue-on-error" not in step
