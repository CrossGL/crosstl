"""Loop-carried locals and origin-restricted storage reads in subgroup proofs.

A uniform loop may advance a local on every trip, as a block-mask offset
``offset += count == 0 ? step : 0;``. The DirectX and OpenGL software
subgroup proofs keep such a local uniform on every trip when each write is
uniform given the facts at the trip's start (a fixpoint), every invocation
runs the same trips and no trip ends early.

A resource-aggregate load reads the binding its handle's identity selects.
When the handle's origins are proven, the load's value depends only on those
bindings, so a read of an immutable binding stays uniform even when another
binding of the same element type is not immutable.
"""

import os

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.ast import (
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    UnaryOpNode,
)
from crosstl.translator.codegen.subgroup_loop_carried import (
    carried_fixpoint,
    carried_write,
    follow_carried_write,
)
from crosstl.translator.codegen.uniform_returns import (
    UniformReturnAnalysis,
    _StorageDependency,
)

DIRECTX_ERROR = "project.translate.directx-software-subgroup-invalid"
OPENGL_ERROR = "project.translate.opengl-software-subgroup-invalid"

CARRIED_KERNEL = """#include <metal_stdlib>
using namespace metal;

kernel void masked_products(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    const device bool* mask [[buffer(2)]],
    constant int& T [[buffer(3)]],
    constant int& step [[buffer(4)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 A, B, C;
  simdgroup_load(A, src + sg * 64, 8);
  B = make_filled_simdgroup_matrix<float, 8>(0.5f);
  B.thread_elements()[0] = float(int(lane % 3) - 1);
  C = make_filled_simdgroup_matrix<float, 8>(0.0f);
  int offset = T * step;
  int other = 0;
  short count = 2;
  for (int k = 0; k < T; k++) {
    if (mask[offset]) {
      simdgroup_multiply_accumulate(C, A, B, C);
    }
    count--;
    offset += count == 0 ? step : 0;
    count = count == 0 ? short(2) : count;
  }
  simdgroup_store(C, dst + sg * 64, 8);
}
"""

ORIGIN_KERNEL = """#include <metal_stdlib>
using namespace metal;

struct View {
  const device uint* data;
};

[[kernel]] void flagged_sums(
    const device uint* flags [[buffer(0)]],
    const device uint* values [[buffer(1)]],
    device uint* results [[buffer(2)]],
    const device uint* other [[buffer(3)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  View view{values};
  uint flag = flags[group.x];
  uint base = group.x * 128u;
  if (flag == 0u) {
    results[base + lid] = 0u;
    return;
  }
  threadgroup_barrier(mem_flags::mem_none);
  results[base + lid] = simd_sum(view.data[base + lid]) + flag + other[0];
}
"""

REDECLARED_KERNEL = """#include <metal_stdlib>
using namespace metal;

[[kernel]] void picked_sums(
    const device uint* flags [[buffer(0)]],
    const device uint* values [[buffer(1)]],
    device uint* results [[buffer(2)]],
    constant int& mode [[buffer(3)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  uint flag = 0u;
  if (mode > 0) {
    uint pick = flags[group.x];
    flag = pick;
  } else {
    uint pick = flags[group.x + 1u];
    flag = pick;
  }
  uint base = group.x * 128u;
  if (flag == 0u) {
    results[base + lid] = 0u;
    return;
  }
  threadgroup_barrier(mem_flags::mem_none);
  results[base + lid] = simd_sum(values[base + lid]) + flag;
}
"""


def _translate(
    root, source, target, entry, immutable=(), workgroup_size=(32, 2, 1), ranges=()
):
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.metal").write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("kernel.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=workgroup_size,
            index_range_assertions=ranges if target == "opengl" else (),
            immutable_storage=[
                {"entryPoint": entry, "parameter": name} for name in immutable
            ],
            source_options={
                "metal": {
                    "preserve_resource_origins": True,
                    "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                    "cooperative_matrix_fragment_mapping_provenance": (
                        "unit_lane_coordinates"
                    ),
                    "target_options": {
                        "directx": {
                            "cooperative_matrix_software_lowering": True,
                            "software_subgroup_width": 32,
                            "relative_wave_shuffle_out_of_range": "self",
                        },
                        "opengl": {
                            "cooperative_matrix_software_lowering": True,
                            "software_subgroup_width": 32,
                        },
                    },
                }
            },
        ),
        format_output=False,
    )
    return report.to_json(), root


def _validate(text, target, tmp_path):
    if target == "directx":
        if os.environ.get("CROSTL_SKIP_NATIVE_VALIDATORS") == "1":
            return
        from tests.test_translator.test_directx_float_atomics import _compile

        work = tmp_path / "validation"
        work.mkdir()
        _compile(text, work, profile="cs_6_2")
    else:
        from tests.test_translator.test_metal_builtin_ownership import _compile

        _compile(text, "opengl", tmp_path)


def _accepted(tmp_path, source, target, entry, immutable, **options):
    report, root = _translate(tmp_path, source, target, entry, immutable, **options)
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]
    (artifact,) = report["artifacts"]
    text = (root / artifact["path"]).read_text()
    assert "simdgroup_" not in text and "simd_sum" not in text
    _validate(text, target, tmp_path)
    return text


def _rejected(tmp_path, source, target, entry, immutable, **options):
    report, _root = _translate(tmp_path, source, target, entry, immutable, **options)
    assert report["summary"]["translatedCount"] == 0, report
    codes = {item["code"] for item in report["diagnostics"]}
    assert codes == {DIRECTX_ERROR if target == "directx" else OPENGL_ERROR}, codes


def _rewritten(source, rewrites):
    for old, new in rewrites:
        assert old in source, old
        source = source.replace(old, new, 1)
    return source


TARGETS = ("directx", "opengl")

# --- Loop-carried locals ----------------------------------------------------


@pytest.mark.parametrize("target", TARGETS)
def test_accepts_a_mask_offset_the_uniform_loop_advances(tmp_path, target):
    text = _accepted(tmp_path, CARRIED_KERNEL, target, "masked_products", ("mask",))
    assert "offset" in text and "count" in text


CARRIED_REJECTIONS = {
    # A trip writes a value that differs between lanes.
    "divergent-value": [
        (
            "offset += count == 0 ? step : 0;",
            "offset += count == 0 ? step : int(lane % 2);",
        )
    ],
    # A uniform value written only by some lanes.
    "divergent-guard": [("    count--;\n", "    if (lane % 2 == 0) { count--; }\n")],
    # Uniform on the first trip only: ``other`` turns divergent at the end of
    # the trip, so the fixpoint drops ``offset`` on its second round.
    "second-trip": [
        ("offset += count == 0 ? step : 0;", "offset = other;\n    other = int(lane);")
    ],
    # A trip may end early, so the trips can differ.
    "early-exit": [("    count--;\n", "    if (T > 100) { break; }\n    count--;\n")],
    # Not every lane runs the same trips.
    "divergent-trips": [("k < T;", "k < T + int(lane % 2);")],
    # A write outside the loop makes it an ordinary written local.
    "written-outside": [
        ("  for (int k = 0;", "  offset = int(lane);\n  for (int k = 0;")
    ],
    # A step of a local that was divergent where the loop starts.
    "divergent-start": [("short count = 2;", "short count = short(lane % 2 + 1);")],
}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", sorted(CARRIED_REJECTIONS))
def test_rejects_unproven_loop_carried_locals(tmp_path, target, case):
    source = _rewritten(CARRIED_KERNEL, CARRIED_REJECTIONS[case])
    _rejected(tmp_path, source, target, "masked_products", ("mask",))


@pytest.mark.parametrize("target", TARGETS)
def test_rejects_a_carried_index_into_storage_without_a_contract(tmp_path, target):
    _rejected(tmp_path, CARRIED_KERNEL, target, "masked_products", ())


def _identifier(name):
    return IdentifierNode(name)


def test_carried_write_kinds():
    candidates = {"x"}

    def identity(operator):
        return operator

    one = LiteralNode(1, None)
    assert carried_write(AssignmentNode(_identifier("x"), one), candidates, identity)[
        :2
    ] == ("assign", "x")
    assert carried_write(
        AssignmentNode(_identifier("x"), one, "+="), candidates, identity
    )[:2] == ("compound", "x")
    assert carried_write(UnaryOpNode("--", _identifier("x")), candidates, identity)[
        :2
    ] == ("step", "x")
    assert (
        carried_write(AssignmentNode(_identifier("y"), one), candidates, identity)
        is None
    )
    assert (
        carried_write(UnaryOpNode("-", _identifier("x")), candidates, identity) is None
    )


def test_follow_carried_write_needs_a_uniform_local_for_updates():
    def identity(operator):
        return operator

    def uniform(value):
        return isinstance(value, LiteralNode)

    one = LiteralNode(1, None)
    lane = _identifier("lane")
    names = {"x"}
    assert follow_carried_write(
        AssignmentNode(_identifier("x"), one, "+="), {"x"}, names, uniform, identity
    )
    assert names == {"x"}
    assert not follow_carried_write(
        AssignmentNode(_identifier("x"), lane, "+="), {"x"}, names, uniform, identity
    )
    assert names == set()
    # A divergent local stays divergent after a uniform step.
    assert not follow_carried_write(
        UnaryOpNode("++", _identifier("x")), {"x"}, names, uniform, identity
    )
    # A plain write of a uniform value makes it uniform again.
    assert follow_carried_write(
        AssignmentNode(_identifier("x"), one), {"x"}, names, uniform, identity
    )
    assert names == {"x"}


def test_carried_fixpoint_shrinks_until_a_trip_preserves_the_set():
    # ``a = b; b = lane;`` keeps ``a`` for one trip only.
    def trip(names):
        if "b" in names:
            names.add("a")
        else:
            names.discard("a")
        names.discard("b")

    assert carried_fixpoint({"a", "b"}, {"a", "b", "n"}, trip) == frozenset()

    def stable(names):
        pass

    assert carried_fixpoint({"c"}, {"c", "n"}, stable) == {"c"}
    # A carried local divergent where the loop starts is never kept.
    assert carried_fixpoint({"c"}, {"n"}, stable) == frozenset()


# --- Origin-restricted storage reads ---------------------------------------

# GLSL indexes with 32-bit ints; the handles' 64-bit offsets stay small here.
ORIGIN = {
    "workgroup_size": (128, 1, 1),
    "ranges": tuple(
        {
            "source": "kernel.metal",
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_{access}_uint",
            "minimum": 0,
            "maximum": 2**31 - 1,
        }
        for access in ("load", "store")
    ),
}


@pytest.mark.parametrize("target", TARGETS)
def test_accepts_a_read_of_immutable_storage_sharing_its_accessor(tmp_path, target):
    text = _accepted(
        tmp_path, ORIGIN_KERNEL, target, "flagged_sums", ("flags",), **ORIGIN
    )
    # The flag read goes through the accessor that also serves the values,
    # the results and ``other``, none of which is immutable.
    assert "crosstl_resource_load_uint" in text


@pytest.mark.parametrize("target", TARGETS)
def test_rejects_a_read_of_mutable_storage_sharing_its_accessor(tmp_path, target):
    # ``other`` carries no immutable-storage contract.
    source = _rewritten(
        ORIGIN_KERNEL, [("uint flag = flags[group.x];", "uint flag = other[group.x];")]
    )
    _rejected(tmp_path, source, target, "flagged_sums", ("flags",), **ORIGIN)


@pytest.mark.parametrize("target", TARGETS)
def test_rejects_the_shared_accessor_read_without_a_contract(tmp_path, target):
    _rejected(tmp_path, ORIGIN_KERNEL, target, "flagged_sums", (), **ORIGIN)


def test_storage_read_dependencies_follow_the_named_bindings():
    call = FunctionCallNode(
        _identifier("load"),
        [_identifier("ref"), _identifier("i"), _identifier("r0"), _identifier("r1")],
    )
    call.storage_reads = ("r1",)
    analysis = UniformReturnAnalysis({}, set())
    assert analysis.dependencies(call) == frozenset({0, 1, _StorageDependency(3)})
    assert analysis.uniform_arguments(call, {"r1"}) is not None
    assert analysis.uniform_arguments(call, {"r0"}) is None
    # A named binding missing from the call, or one in the reference or index
    # position, fails closed.
    call.storage_reads = ("r2",)
    assert analysis.dependencies(call) is None
    call.storage_reads = ("ref",)
    assert analysis.dependencies(call) is None
    call.storage_reads = ()
    assert analysis.dependencies(call) is None


# --- Locals declared in several scopes ---------------------------------------

REDECLARED = {"workgroup_size": (128, 1, 1)}


@pytest.mark.parametrize("target", TARGETS)
def test_accepts_a_flag_from_locals_declared_in_both_branches(tmp_path, target):
    _accepted(
        tmp_path, REDECLARED_KERNEL, target, "picked_sums", ("flags",), **REDECLARED
    )


@pytest.mark.parametrize("target", TARGETS)
def test_rejects_a_redeclared_local_that_is_written(tmp_path, target):
    source = _rewritten(
        REDECLARED_KERNEL,
        [("    flag = pick;\n  }\n", "    pick += lid;\n    flag = pick;\n  }\n")],
    )
    _rejected(tmp_path, source, target, "picked_sums", ("flags",), **REDECLARED)
