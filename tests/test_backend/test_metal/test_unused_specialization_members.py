"""Lowered members of class template specializations exist only when the program uses them ([temp.inst])."""

import re

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

# `Scan<pair2>::total` is ill-formed (a prefix sum of a struct), but nothing
# calls it, so a C++ compiler never instantiates its definition.
SOURCE = """#include <metal_stdlib>
using namespace metal;

struct pair2 {
  float a;
  float b;
};

template <typename U>
struct Scan {
  U total(U x) thread {
    return simd_prefix_inclusive_sum(x);
  }
  U pass(U x) thread {
    return x;
  }
};

kernel void k(device pair2* out [[buffer(0)]],
              device float* values [[buffer(1)]],
              uint lane [[thread_index_in_simdgroup]]) {
  Scan<pair2> op;
  out[lane] = op.pass(out[lane]);
  Scan<float> float_op;
  values[lane] = float_op.total(values[lane]);
}
"""

# `twice` reaches `once` only from a member nobody calls; `apply` reaches
# `inner` from a used member, a static member is called through its qualified
# name, and an operator member is bound by code generation without a name.
CHAIN_SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T>
struct Ops {
  T once(T x) thread { return x + x; }
  T twice(T x) thread { return once(once(x)); }
  T inner(T x) thread { return x * x; }
  T apply(T x) thread { return inner(x) + T(1); }
  static T scale(T x) { return x * T(3); }
  T operator()(T x) thread { return x - T(1); }
};

struct Plain {
  float unused(float x) thread { return x; }
};

kernel void k(device float* values [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  Ops<float> ops;
  values[i] = ops.apply(values[i]) + Ops<float>::scale(values[i]) + ops(values[i]);
}
"""


def _definitions(code):
    return set(re.findall(r"static inline \w+ (\w+)\(", code))


def test_unused_specialization_member_is_not_instantiated():
    code = MetalPreprocessor().preprocess(SOURCE)
    definitions = _definitions(code)
    assert "Scan_pair2__pass" in definitions
    assert "Scan_float__total" in definitions
    assert "Scan_pair2__total" not in definitions
    assert "Scan_float__pass" not in definitions


def test_unused_specialization_member_does_not_reach_targets(tmp_path):
    path = tmp_path / "unit.metal"
    path.write_text(SOURCE)
    options = {"threadgroup_size": (32, 1, 1)}
    hlsl = crosstl.translate(str(path), backend="directx", source_options=options)
    assert "Scan_pair2__total" not in hlsl
    assert hlsl.count("WavePrefixSum") + hlsl.count("WavePrefixInclusiveSum") >= 1
    assert "Scan_pair2__pass" in hlsl
    glsl = crosstl.translate(str(path), backend="opengl", source_options=options)
    assert "Scan_pair2_total" not in glsl and "Scan_pair2__total" not in glsl


def test_pruning_follows_calls_between_members():
    code = MetalPreprocessor().preprocess(CHAIN_SOURCE)
    definitions = _definitions(code)
    assert {"Ops_float__apply", "Ops_float__inner", "Ops_float__scale"} <= definitions
    assert "Ops_float__operator_call" in definitions
    assert not {"Ops_float__once", "Ops_float__twice"} & definitions
    # Members of a concrete struct are ordinary definitions, not instantiations.
    assert "Plain__unused" in definitions
