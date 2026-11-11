"""Storage views of lane records, records of one 32-bit scalar type.

``struct Pair { float re; float im; }`` has no padding, so an aligned byte
view copy between ``Pair`` elements is a copy of whole elements, and a
``float`` view of ``Pair`` storage addresses one member per lane. Tile
loaders written for a generic element type use both views.
"""

import os
import shutil
import struct
import sys

import pytest

import crosstl
from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import AssignmentNode, FunctionNode, PointerReinterpretNode
from crosstl.translator.codegen.lane_records import (
    lane_record_layout,
    lower_record_byte_copies,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_LANE_RECORD_VIEWS"
TARGETS = ("metal", "directx", "opengl")
TOOLS = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}
WORKGROUP = (32, 1, 1)

_PAIR = """#include <metal_stdlib>
using namespace metal;

struct Pair {
  float re;
  float im;
};
"""

# The tile loader of a generic GEMM: byte records of two elements, aligned to
# one element's size, copied from storage into a workgroup tile.
_COPY = _PAIR + """
struct Loader {
  threadgroup Pair* dst;
  const device Pair* src;

  struct alignas(sizeof(Pair)) ReadVector {
    uchar v[sizeof(Pair) * 2];
  };

  Loader(threadgroup Pair* dst_, const device Pair* src_) : dst(dst_), src(src_) {}
};

void Loader__copy(const thread Loader& self, uint i) {
  *((threadgroup Loader::ReadVector*)(&self.dst[2 * i])) =
      *((const device Loader::ReadVector*)(&self.src[2 * i]));
}

kernel void copy_pairs(
    const device Pair* src [[buffer(0)]],
    device Pair* result [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup Pair tile[64];
  Loader loader(tile, src);
  Loader__copy(loader, tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  result[2 * tid] = tile[63 - 2 * tid];
  result[2 * tid + 1] = tile[62 - 2 * tid];
}
"""

# A float view of a workgroup tile of pairs, read through a helper and
# written one lane at a time.
_LANES = _PAIR + """
struct Tiles {
  threadgroup Pair* pairs;
};

float lane_sum(const threadgroup float* lanes, uint k) {
  return lanes[2 * k] + lanes[2 * k + 1];
}

kernel void lane_views(
    const device Pair* src [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup Pair tile[32];
  Tiles tiles{tile};
  tiles.pairs[tid] = src[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* lanes = reinterpret_cast<threadgroup float*>(tiles.pairs);
  float total = lane_sum(lanes, 31 - tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  lanes[2 * tid + 1] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  result[2 * tid] = tile[tid].re;
  result[2 * tid + 1] = tile[tid].im;
}
"""

_SOURCES = {"copy": (_COPY, "copy_pairs"), "lanes": (_LANES, "lane_views")}


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _inputs():
    return [
        (_f32(0.25 * index + 0.5), _f32(-1.5 * index + 0.75)) for index in range(64)
    ]


def _expected(case):
    pairs = _inputs()
    if case == "copy":
        reordered = []
        for tid in range(32):
            reordered.extend((pairs[63 - 2 * tid], pairs[62 - 2 * tid]))
        return [lane for pair in reordered for lane in pair]
    values = []
    for tid in range(32):
        source = pairs[31 - tid]
        values.extend((pairs[tid][0], _f32(source[0] + source[1])))
    return values


def _translate(tmp_path, source, target):
    path = tmp_path / "lanes.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"threadgroup_size": WORKGROUP},
    )


def _report(tmp_path, source, target):
    (tmp_path / "lanes.metal").write_text(source, encoding="utf-8")
    return translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("lanes.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=WORKGROUP,
        ),
        format_output=False,
    ).to_json()


def _functions(ast):
    return [node for node in ast.walk() if isinstance(node, FunctionNode)]


def test_lane_records_are_padding_free_records_of_one_word_scalar():
    ast = parse("""shader main {
    struct Pair { float re; float im; }
    struct Triple { uint a; uint b; uint c; }
    struct Mixed { float a; int b; }
    struct Halves { half a; half b; }
    struct Wide { double a; }
    struct Flags { bool a; bool b; }
    struct Nested { Pair p; }
    compute { void main() {} }
}""")
    layouts = {struct.name: lane_record_layout(struct) for struct in ast.structs}
    assert layouts["Pair"] == ("float", 2)
    assert layouts["Triple"] == ("uint", 3)
    for name in ("Mixed", "Halves", "Wide", "Flags", "Nested"):
        assert layouts[name] is None, name


def test_byte_copy_of_lane_records_becomes_element_copies(tmp_path):
    path = tmp_path / "copy.metal"
    path.write_text(_COPY, encoding="utf-8")
    ast = parse(crosstl.translate(str(path), backend="cgl", format_output=False))
    lower_record_byte_copies(ast)
    copy = next(f for f in _functions(ast) if f.name == "Loader__copy")
    assignments = [n for n in copy.body.walk() if isinstance(n, AssignmentNode)]
    # Two Pair elements of eight bytes fill the sixteen-byte view.
    assert len(assignments) == 2
    assert not any(isinstance(n, PointerReinterpretNode) for n in copy.body.walk())


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", sorted(_SOURCES))
def test_lane_record_views_translate_and_compile(tmp_path, target, case):
    generated = _translate(tmp_path, _SOURCES[case][0], target)
    if case == "copy":
        # The byte view is gone; whole elements move instead.
        assert "ReadVector*" not in generated
    elif target == "metal":
        assert "reinterpret_cast<threadgroup float*>" in generated
    else:
        # Each lane is read and written as the member that holds it.
        assert ".re" in generated and ".im" in generated
    if shutil.which(TOOLS[target]):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size


_REFUSED_COPIES = {
    # The view's alignment exceeds the element's size, so element addresses
    # do not prove it.
    "over_aligned": ("alignas(sizeof(Pair))", "alignas(16)"),
    # A record of mixed scalar types is not a lane record.
    "mixed_record": ("float im;", "int im;"),
    # A view that is not a whole number of elements.
    "partial_element": ("uchar v[sizeof(Pair) * 2];", "uchar v[12];"),
}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", sorted(_REFUSED_COPIES))
def test_unproven_byte_copies_stay_refused(tmp_path, target, case):
    old, new = _REFUSED_COPIES[case]
    source = _COPY.replace(old, new)
    if case == "partial_element":
        source = source.replace("alignas(sizeof(Pair))", "alignas(4)")
    assert source != _COPY
    report = _report(tmp_path, source, target)
    assert report["summary"]["failedCount"] == 1
    codes = {item["code"] for item in report["diagnostics"]}
    assert codes & {
        "project.translate.pointer-reinterpret-unsupported",
        "project.translate.resource-aggregate-unsupported",
    }, codes


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_lane_view_of_another_scalar_stays_refused(tmp_path, target):
    # A uint view of float lanes would need a bit conversion per lane.
    source = _LANES.replace(
        "threadgroup float* lanes = reinterpret_cast<threadgroup float*>",
        "threadgroup uint* lanes = reinterpret_cast<threadgroup uint*>",
    ).replace(
        "float lane_sum(const threadgroup float* lanes",
        "float lane_sum(const threadgroup uint* lanes",
    )
    report = _report(tmp_path, source, target)
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.resource-aggregate-unsupported"
        for item in report["diagnostics"]
    )


def _request(root, target, case):
    source, entry = _SOURCES[case]
    original, descriptor, package = _package(
        root,
        target,
        "float",
        WORKGROUP,
        source=source,
        software_subgroups=False,
    )
    pairs = _inputs()
    count = 64 if case == "copy" else 32
    inputs = {
        "src": {
            "dtype": "float32",
            "shape": [64, 2],
            "values": [lane for pair in pairs for lane in pair],
        },
        "result": {
            "dtype": "float32",
            "shape": [count, 2] if case == "copy" else [2 * count],
            "values": [-99.0] * (2 * count),
        },
    }
    expected = _expected(case)
    outputs = _bound_values(
        descriptor,
        {
            "result": {
                "dtype": "float32",
                "shape": inputs["result"]["shape"],
                "values": expected,
            }
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, entry, request, outputs


@pytest.mark.parametrize("case", sorted(_SOURCES))
def test_lane_record_views_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required lane-record execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, entry, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry=entry,
    )


def test_lane_record_execution_is_required():
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
    assert "test_lane_record_views.py" in step
    assert "if:" not in step and "continue-on-error" not in step
