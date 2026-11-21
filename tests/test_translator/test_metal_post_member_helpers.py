"""Helpers that only a lowered member body calls see the members they call.

A function template is instantiated where a call selects it ([temp.inst]). Here the only call to
``tile_mad<T, N>`` is in ``Block<T, N>::run``, whose locals ``Tile<T, N> d, a, b`` name the same
specialization ``Tile<int, 4>`` that ``run`` itself calls members on. The project pipeline finds the
call after member lowering has turned ``run`` into a free function, so ``tile_mad_int_4`` was
materialized after ``Tile_int_4``'s members were lowered and dropped. Its body kept ``d.at(i)``,
``a.get(i)`` and ``b.template scaled<3>(i)``: the Metal output called members ``Tile_int_4`` no longer
declared, and DirectX and OpenGL refused ``b.scaled<3>`` because no member specialization was
materialized. The helper is now lowered with the struct. MLX's complex GEMM tiles reach
``tile_matmad`` in the same way.
"""

import os
import re
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_POST_MEMBER_HELPERS"
ENTRY = "post_member_helpers_int"
WIDTH = 4
WORKGROUP = (32, 1, 1)
GROUPS = 2
THREADS = GROUPS * WORKGROUP[0]
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 1999) - 999 for index in range(THREADS * 2 * WIDTH)]

SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T, int N>
struct Tile {
  T vals[N];

  METAL_FUNC thread T& at(const short i) thread {
    return vals[i];
  }

  METAL_FUNC T get(const short i) const thread {
    return vals[i];
  }

  template <int K>
  METAL_FUNC T scaled(const short i) const thread {
    return vals[i] * T(K);
  }

  METAL_FUNC void clear() thread {
    for (short i = 0; i < N; ++i) {
      vals[i] = T(0);
    }
  }

  METAL_FUNC void set(const short i, T value) thread {
    vals[i] = value;
  }
};

template <typename T, int N>
METAL_FUNC void tile_mad(
    thread Tile<T, N>& d,
    thread Tile<T, N>& a,
    thread Tile<T, N>& b) {
  for (short i = 0; i < N; ++i) {
    d.at(i) += a.get(i) * b.template scaled<3>(i);
  }
}

template <typename T, int N>
struct Block {
  METAL_FUNC T run(const device T* src) thread {
    Tile<T, N> d, a, b;
    d.clear();
    a.clear();
    b.clear();
    for (short i = 0; i < N; ++i) {
      d.set(i, T(i + 1));
      a.set(i, src[i]);
      b.set(i, src[i + N]);
    }
    tile_mad(d, a, b);
    T total = T(0);
    for (short i = 0; i < N; ++i) {
      total += d.get(i) * T(i + 2);
    }
    return total;
  }
};

template <typename T>
[[kernel]] void post_member_helpers(
    const device T* values [[buffer(0)]],
    device T* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  Block<T, 4> block;
  results[index] = block.run(values + index * 8);
}

template [[host_name("post_member_helpers_int")]] [[kernel]] void post_member_helpers<int>(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]);
"""

# A member call whose receiver is one of the tiles: `d.at(`, `b.template scaled<3>(`.
MEMBER_CALL = re.compile(
    r"\b[dab]\s*\.\s*(?:template\s+)?(?:at|get|scaled|clear|set)\b"
)


def _expected():
    results = []
    for index in range(THREADS):
        a = VALUES[2 * WIDTH * index : 2 * WIDTH * index + WIDTH]
        b = VALUES[2 * WIDTH * index + WIDTH : 2 * WIDTH * (index + 1)]
        d = [i + 1 + a[i] * b[i] * 3 for i in range(WIDTH)]
        results.append(sum(d[i] * (i + 2) for i in range(WIDTH)))
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
def test_post_member_helpers_call_lowered_members(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    # The helper reads and writes the tile's storage and calls the lowered members.
    assert not MEMBER_CALL.search(generated), MEMBER_CALL.search(generated)
    assert re.search(r"\bTile_+int_+4_+get\(\s*a\b", generated)
    assert re.search(r"\bTile_+int_+4_+scaled_+3\(\s*b\b", generated)
    assert re.search(r"\bd\.vals\[[^\]]*\]\s*\+=", generated)
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


def test_post_member_helpers_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native post-member helpers")
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


def test_post_member_helpers_are_required_on_all_native_platforms():
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
    assert "test_metal_post_member_helpers.py" in step
    assert "if:" not in step and "continue-on-error" not in step
