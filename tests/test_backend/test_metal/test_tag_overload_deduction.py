"""Overloads of a template member that differ in where a tag parameter sits.

``mma(C, C, A, tag, B, B, tag)`` and ``mma(C, C, A, A, tag, B, tag)`` take the
same number of arguments. Deducing ``bool_constant<b>`` from a vector argument
(or ``vec<T, 8>`` from a tag) fails, so only one overload is a candidate for
each call.
"""

import re

import pytest

from crosstl._crosstl import translate
from crosstl.backend.Metal.overload_deduction import deduction_pattern_mismatch
from tests.test_translator.test_metal_builtin_ownership import _compile

KERNEL = """#include <metal_stdlib>
using namespace metal;
struct Frag {
  template <typename U>
  using frag_t = typename metal::vec<U, 8>;
  template <typename CType, typename AType, typename BType, bool ta, bool tb>
  static void mma(thread frag_t<CType>& C0, thread frag_t<CType>& C1,
                  const thread frag_t<AType>& A, metal::bool_constant<ta>,
                  const thread frag_t<BType>& B0, const thread frag_t<BType>& B1,
                  metal::bool_constant<tb>) {
    for (short i = 0; i < 8; i++) {
      C0[i] += CType(ta ? A[7 - i] : A[i]) * CType(B0[i]);
      C1[i] += CType(A[i]) * CType(tb ? B1[7 - i] : B1[i]);
    }
  }
  template <typename CType, typename AType, typename BType, bool ta, bool tb>
  static void mma(thread frag_t<CType>& C0, thread frag_t<CType>& C1,
                  const thread frag_t<AType>& A0, const thread frag_t<AType>& A1,
                  metal::bool_constant<ta>, const thread frag_t<BType>& B,
                  metal::bool_constant<tb>) {
    for (short i = 0; i < 8; i++) {
      C0[i] += CType(A0[i]) * CType(B[i]) + CType(100);
      C1[i] += CType(A1[i]) * CType(B[i]) + CType(200);
    }
  }
};
kernel void k(device float* dst [[buffer(0)]],
              uint gid [[thread_position_in_grid]]) {
  metal::vec<float, 8> c0 = metal::vec<float, 8>(0);
  metal::vec<float, 8> c1 = metal::vec<float, 8>(0);
  metal::vec<half, 8> a;
  for (short i = 0; i < 8; i++) a[i] = half(gid + i);
  Frag::mma(c0, c1, a, metal::bool_constant<true>{}, a, a,
            metal::bool_constant<false>{});
  Frag::mma(c0, c1, a, a, metal::bool_constant<false>{}, a,
            metal::bool_constant<true>{});
  for (short i = 0; i < 8; i++) {
    dst[gid * 16 + i] = c0[i];
    dst[gid * 16 + 8 + i] = c1[i];
  }
}
"""


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_each_call_selects_the_overload_its_tags_deduce(tmp_path, target):
    path = tmp_path / "tags.metal"
    path.write_text(KERNEL, encoding="utf-8")
    generated = translate(str(path), backend=target)
    instantiations = set(re.findall(r"\bFrag_+mma_+(\w+?)\s*\(", generated))
    assert instantiations == {
        "float_half_half_true_false",
        "float_half_half_false_true",
    }
    # Both bodies are emitted: the `+ 100` / `+ 200` overload exactly once.
    assert len(re.findall(r"\b100\b", generated)) == 1
    assert len(re.findall(r"\b200\b", generated)) == 1
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-Wno-unused-function",),
    )


@pytest.mark.parametrize(
    ("declared", "argument", "mismatch"),
    [
        ("metal::vec<T,8>", "metal::bool_constant<true>", True),
        ("metal::bool_constant<b>", "metal::vec<half,8>", True),
        ("metal::bool_constant<b>", "float", True),
        ("metal::vec<T,8>", "metal::vec<half,8>", False),
        ("vec<T,4>", "float4", False),
        ("metal::bool_constant<b>", "metal::integral_constant<bool,true>", False),
        # Template parameters outside a template-id, explicitly supplied ones
        # and struct arguments are left to the existing binding checks.
        ("T", "metal::bool_constant<true>", False),
        ("metal::vec<U,8>", "metal::bool_constant<true>", False),
        ("Tile<T,1,2>", "Tile_float_1_2", False),
    ],
)
def test_deduction_mismatch_is_reported_only_when_it_must_fail(
    declared, argument, mismatch
):
    assert deduction_pattern_mismatch(declared, argument, {"T", "b"}) is mismatch
