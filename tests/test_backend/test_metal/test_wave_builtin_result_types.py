"""A builtin SIMD-group call keeps the builtin's type beside a user overload of its name.

MSL lets a program overload a builtin SIMD-group function, as MLX's ``bf16_math.h`` does for bfloat. A call whose
arguments no user overload matches calls the builtin template (MSL 3.2 section 6.9.2), so ``int + simd_prefix_
inclusive_sum(int)`` is integer arithmetic. The import lowered such a call to the builtin but typed it as the one user
overload of the same arity, so it converted the other operand to bfloat and rounded integers above 256.
"""

import pytest

import crosstl

SOURCE = """
#include <metal_stdlib>
using namespace metal;

inline bfloat simd_prefix_inclusive_sum(bfloat data) {
  return bfloat(simd_prefix_inclusive_sum(float(data)));
}

kernel void prefix_total(
    device int* out [[buffer(0)]],
    device float* scaled [[buffer(1)]],
    uint lane [[thread_index_in_simdgroup]]) {
  int before = 300;
  const int n = int(lane);
  const int through = before + simd_prefix_inclusive_sum(n);
  out[lane] = through;
  const bfloat b = bfloat(lane);
  scaled[lane] = float(b + simd_prefix_inclusive_sum(b));
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "prefix.metal"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend)


def test_integer_builtin_call_keeps_integer_arithmetic(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "const int through = before + WavePrefixInclusiveSum(n);" in crossgl
    assert "bfloat16(before)" not in crossgl


def test_matching_user_overload_keeps_its_bfloat_type(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "float(b + simd_prefix_inclusive_sum(b))" in crossgl
    assert "WavePrefixInclusiveSum(b)" not in crossgl


@pytest.mark.parametrize(
    "backend, integer_sum",
    [
        ("metal", "const int through = before + simd_prefix_inclusive_sum(n);"),
        ("opengl", "const int through = (before + subgroupInclusiveAdd(n));"),
    ],
)
def test_targets_add_the_builtin_sum_to_the_integer(tmp_path, backend, integer_sum):
    code = _translate(tmp_path, backend)
    assert integer_sum in code
    assert "bfloat(before)" not in code and "bfloat16(before)" not in code
