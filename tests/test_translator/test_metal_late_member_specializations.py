"""Struct specializations that only an instantiated member template body names are lowered.

A member function template body is instantiated when a call selects it ([temp.inst]). Here
``Tile<T, R>::row_reduce<Op>`` calls ``Frag<T>::template row_reduce<Op>(...)``, and nothing else names
``Frag<int>``, so the specialization first appears in the free function that member lowering emits for
``Tile_int_2::row_reduce<MaxOp>``. The struct materializer then produced ``Frag_int`` after member lowering
had finished: the qualified call reached the parser as ``Frag_int::row_reduce<MaxOp>(...)`` and became an
assignment to a call, and ``MaxOp::apply`` had already been dropped because nothing called it while members
were lowered. The DirectX and OpenGL routes emitted an undeclared ``apply(...)``: a qualified call whose
owner is a struct without that member lost its owner while the call name was mapped, and is now refused.
MLX's attention tiles reduce through ``MMAFrag_t::template row_reduce<Op>`` in the same way.
"""

import os
import re
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_LATE_MEMBER_SPECIALIZATIONS"
ENTRY = "late_members_int"
WORKGROUP = (32, 1, 1)
GROUPS = 2
THREADS = GROUPS * WORKGROUP[0]
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 2003) - 1001 for index in range(THREADS * 4)]

SOURCE = """#include <metal_stdlib>
using namespace metal;

struct MaxOp {
  template <typename T>
  static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  static constexpr T apply(T x, T y) {
    return x + y;
  }
};

template <typename T>
struct Frag {
  typedef metal::vec<T, 2> frag_type;

  template <typename Op>
  static void row_reduce(thread const frag_type& inp, thread T* reduced) {
    T r = Op::apply(inp.x, inp.y);
    reduced[0] = Op::apply(reduced[0], r);
  }
};

template <typename T, int R>
struct Tile {
  metal::vec<T, 2> frags[R];

  template <typename Op>
  void row_reduce(thread T vals[R]) const thread {
    for (short i = 0; i < R; ++i) {
      Frag<T>::template row_reduce<Op>(frags[i], &vals[i]);
    }
  }
};

template <typename T>
[[kernel]] void late_members(
    const device T* values [[buffer(0)]],
    device T* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  Tile<T, 2> tile;
  tile.frags[0] = metal::vec<T, 2>(values[index * 4], values[index * 4 + 1]);
  tile.frags[1] = metal::vec<T, 2>(values[index * 4 + 2], values[index * 4 + 3]);
  T maxima[2] = {T(-5000), T(-5000)};
  T sums[2] = {T(3), T(-4)};
  tile.template row_reduce<MaxOp>(maxima);
  tile.template row_reduce<SumOp>(sums);
  results[index] = maxima[0] * 7 + maxima[1] * 3 - sums[0] + 5 * sums[1];
}

template [[host_name("late_members_int")]] [[kernel]] void late_members<int>(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]);
"""


def _expected():
    results = []
    for index in range(THREADS):
        v = VALUES[4 * index : 4 * index + 4]
        maxima = (max(v[0], v[1]), max(v[2], v[3]))
        sums = (3 + v[0] + v[1], -4 + v[2] + v[3])
        results.append(maxima[0] * 7 + maxima[1] * 3 - sums[0] + 5 * sums[1])
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
def test_late_member_specializations_are_lowered(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    # Every reduction calls a materialized operator; no qualified or bare call survives.
    assert not re.search(r"(?<![\w:])apply\s*\(", generated)
    assert "::row_reduce" not in generated and "crosstl_member_layer" not in generated
    # GLSL reserves identifiers that contain "__", so its helper names use single underscores.
    assert re.search(r"\bMaxOp_+apply_+int\(", generated)
    assert re.search(r"\bSumOp_+apply_+int\(", generated)
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


def test_late_member_specializations_reach_crossgl(tmp_path):
    from crosstl._crosstl import translate

    # The CrossGL route uses the frontend's own member-lowering layer.
    path = tmp_path / "late_members.metal"
    path.write_text(SOURCE, encoding="utf-8")
    generated = translate(str(path), backend="cgl")
    assert "Frag_int__row_reduce__MaxOp(" in generated
    assert "Frag_int__row_reduce__SumOp(" in generated
    assert not re.search(r"(?<![\w:])apply\s*\(", generated)
    assert "Frag_int::" not in generated and "crosstl_member_layer" not in generated


UNDECLARED_MEMBER_SOURCE = """#include <metal_stdlib>
using namespace metal;

struct MaxOp {
  int unused;
};

[[kernel]] void undeclared_member(device float* out [[buffer(0)]], uint index [[thread_position_in_grid]]) {
  out[index] = MaxOp::apply(float(index), 2.0f);
}
"""


@pytest.mark.parametrize("backend", ("cgl", "directx", "opengl"))
def test_undeclared_static_member_call_is_refused(tmp_path, backend):
    from crosstl._crosstl import translate
    from crosstl.backend.Metal.MetalCrossGLCodeGen import (
        MetalStructMethodCallResolutionError,
    )

    # `MaxOp::apply` names no member; dropping the owner would call an undeclared `apply`.
    path = tmp_path / "undeclared_member.metal"
    path.write_text(UNDECLARED_MEMBER_SOURCE, encoding="utf-8")
    with pytest.raises(MetalStructMethodCallResolutionError, match=r"MaxOp::apply"):
        translate(str(path), backend=backend)


def test_namespace_qualified_calls_keep_their_mapping(tmp_path):
    from crosstl._crosstl import translate

    source = UNDECLARED_MEMBER_SOURCE.replace(
        "MaxOp::apply(float(index), 2.0f)", "metal::max(float(index), 2.0f)"
    )
    path = tmp_path / "namespace_call.metal"
    path.write_text(source, encoding="utf-8")
    assert "max(float(index), 2.0f)" in translate(str(path), backend="cgl")


def test_late_member_specializations_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(
            f"set {REQUIRE_ENV}=1 for required native late member specializations"
        )
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


def test_late_member_specializations_are_required_on_all_native_platforms():
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
    assert "test_metal_late_member_specializations.py" in step
    assert "if:" not in step and "continue-on-error" not in step
