"""Metal struct members declared in lists and specializations spelled twice.

``short lo, hi;`` declares two data members, so a method that names either
one reads its own object's member. ``Tile<float, 2, Frag<float, 2>>`` and
the same tile spelled after ``Frag<float, 2>`` was materialized name one
type, so they share one concrete struct.
"""

import os
import re
import shutil
import sys

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_MEMBER_SPELLINGS"
TARGETS = ("metal", "directx", "opengl")
TOOLS = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}

_WINDOWS = """#include <metal_stdlib>
using namespace metal;

template <typename T>
struct Window {
  T lo, hi;
  T width() const thread { return hi - lo; }
};

template <>
struct Window<short> {
  short lo, hi;
  short width() const thread { return short(hi - lo + 1); }
};

kernel void widths(device int* result [[buffer(0)]],
                   uint tid [[thread_position_in_grid]]) {
  Window<int> a;
  a.lo = int(tid);
  a.hi = int(3 * tid + 2);
  Window<short> b;
  b.lo = short(tid);
  b.hi = short(2 * tid + 5);
  result[tid] = a.width() * 100 + int(b.width());
}
"""

_TILES = """#include <metal_stdlib>
using namespace metal;

template <typename T, int R>
struct Frag {
  T v[R];
};

template <typename T, int M, typename F = Frag<T, 2>>
struct Tile {
  F frags[M];
};

template <typename T, int M>
void tile_add(thread Tile<T, M>& d, thread Tile<T, M>& s) {
  for (int i = 0; i < M; ++i) {
    d.frags[i].v[0] += s.frags[i].v[0];
    d.frags[i].v[1] += s.frags[i].v[1];
  }
}

template <typename T, int BM>
struct Block {
  static constant constexpr const short kFrag = 2;
  static constant constexpr const short TM = BM / kFrag;
  using Frag_t = Frag<T, kFrag>;
  Tile<T, TM, Frag_t> acc;

  void step(T x) thread {
    Tile<T, TM, Frag_t> p;
    for (int i = 0; i < TM; ++i) {
      p.frags[i].v[0] = x;
      p.frags[i].v[1] = x + T(1);
    }
    tile_add(acc, p);
  }
};

kernel void tiles(device float* result [[buffer(0)]],
                  uint tid [[thread_position_in_grid]]) {
  Block<float, 4> b;
  for (int i = 0; i < 2; ++i) {
    b.acc.frags[i].v[0] = float(i);
    b.acc.frags[i].v[1] = 0.0f;
  }
  b.step(float(tid));
  result[2 * tid] = b.acc.frags[1].v[0];
  result[2 * tid + 1] = b.acc.frags[0].v[1];
}
"""

_SOURCES = {"windows": (_WINDOWS, "widths"), "tiles": (_TILES, "tiles")}
THREADS = 8


def _expected(case):
    if case == "windows":
        return [(2 * tid + 2) * 100 + (tid + 6) for tid in range(THREADS)]
    return [value for tid in range(THREADS) for value in (1.0 + tid, tid + 1.0)]


def _translate(tmp_path, source, target):
    (tmp_path / "members.metal").write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("members.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 0, report["diagnostics"][:2]
    return (tmp_path / report["artifacts"][0]["path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", sorted(_SOURCES))
def test_member_spellings_translate_and_compile(tmp_path, target, case):
    generated = _translate(tmp_path, _SOURCES[case][0], target)
    if case == "windows":
        if target == "metal":
            # Both declarators of each list are members of the receiver.
            assert "self.hi - self.lo" in generated
            assert re.search(r"-\s*lo\b", generated) is None
    else:
        tiles = re.findall(r"\bstruct (Tile_\w+)", generated)
        assert len(tiles) == 1, tiles
    if shutil.which(TOOLS[target]):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", sorted(_SOURCES))
def test_member_spellings_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required member-spelling execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, entry = _SOURCES[case]
    original, descriptor, package = _package(
        tmp_path, target, "float", (1, 1, 1), source=source, software_subgroups=False
    )
    dtype = "int32" if case == "windows" else "float32"
    count = THREADS if case == "windows" else 2 * THREADS
    outputs = _bound_values(
        descriptor,
        {"result": {"dtype": dtype, "shape": [count], "values": _expected(case)}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {"result": {"dtype": dtype, "shape": [count], "values": [0] * count}},
        ),
        outputs,
        {"workgroupCount": [THREADS, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=original,
        original_entry=entry,
    )


def test_member_spelling_execution_is_required():
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
    assert "test_metal_member_spellings.py" in step
