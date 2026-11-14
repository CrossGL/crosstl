"""Function-local ``conditional_t`` aliases select their branch wherever they are used.

``using In = metal::conditional_t<C, A, B>;`` names ``A`` when the constant ``C`` is true and ``B`` otherwise
([meta.trans.other]). Kernels then use the alias inside other template arguments, for example
``Frag<In, 8, 8>`` and ``Tile<In, 1, 2, Frag<In, 8, 8>>``, and name member types through a local alias of a
specialization, as in ``using frag_t = typename Frag_in_t::frag_type;``. The Metal frontend selected a branch
only when the whole argument was a ``conditional_t`` and only in scopes with local constants, so
``Frag<conditional_t<1 == 2, int, int>, 8, 8>`` became a struct of its own whose static members were never
materialized. ``frag_t`` stayed an unresolved spelling. MLX's ``steel_attention`` declares
``MMAInType = metal::conditional_t<WN == 2, T, AccumType>`` and ``frag_t = typename MMAFrag_mask_t::frag_type``.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_LOCAL_CONDITIONAL_ALIASES"
ENTRY = "conditional_tiles_int_1"
WORKGROUP = (32, 1, 1)
GROUPS = 2
THREADS = GROUPS * WORKGROUP[0]
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 2003) - 1001 for index in range(THREADS * 4)]

SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T, int R, int C>
struct Frag {};

template <typename T>
struct Frag<T, 8, 8> {
  typedef metal::vec<T, 2> frag_type;

  template <typename SrcPtrType>
  static void load(thread frag_type& dst, SrcPtrType src, int stride) {
    for (short j = 0; j < 2; j++) {
      dst[j] = static_cast<T>(src[j * stride]);
    }
  }
};

template <typename T, int R, int C, class Frag_t_>
struct Tile {
  using Frag_t = Frag_t_;
  typedef typename Frag_t::frag_type frag_type;
  frag_type val_frags[R * C];

  // As MLX's MMA tiles, the member names its thread address space, so its
  // fragments bind to Frag's thread references on every Metal compiler.
  template <typename U, int stride>
  void load(const device U* src) thread {
    for (short i = 0; i < R * C; ++i) {
      Frag_t::load(val_frags[i], &src[i * 2 * stride], stride);
    }
  }
};

template <typename T, int WN>
[[kernel]] void conditional_tiles(
    const device T* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  using Accum = int;
  using In = metal::conditional_t<WN == 2, short, Accum>;
  using Frag_in_t = Frag<In, 8, 8>;
  Tile<In, 1, 2, Frag_in_t> tile;
  tile.template load<T, 1>(values + index * 4);
  using frag_t = typename Frag_in_t::frag_type;
  frag_t extra;
  Frag_in_t::load(extra, values + index * 4 + 1, 2);
  results[index] = tile.val_frags[0][0] * 7 + tile.val_frags[1][1] * 3 - extra[0] + 5 * extra[1];
}

template [[host_name("conditional_tiles_int_1")]] [[kernel]] void conditional_tiles<int, 1>(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]);
"""


def _expected():
    results = []
    for index in range(THREADS):
        v = VALUES[4 * index : 4 * index + 4]
        results.append(v[0] * 7 + v[3] * 3 - v[1] + 5 * v[3])
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


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_conditional_aliases_select_one_specialization(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    # One fragment specialization, named for the selected type, serves the tile and the local alias.
    assert "conditional_t<" not in generated and "_conditional_t_" not in generated
    assert "Frag_int_8_8" in generated and "Tile_int_1_2_Frag_int_8_8" in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


def test_conditional_aliases_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native conditional aliases")
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


def test_conditional_aliases_are_required_on_all_native_platforms():
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
    assert "test_metal_local_conditional_aliases.py" in step
    assert "if:" not in step and "continue-on-error" not in step
