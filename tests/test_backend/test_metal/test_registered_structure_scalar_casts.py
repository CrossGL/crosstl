"""Explicit scalar conversions of a registered structure representation from Metal input.

``complex_t<float>`` is the representation of the registered ``complex64_t`` structure conversion contract
(``crosstl.translator.structure_conversions``). Its conversion to a scalar is a member template
(``template <typename U> operator U()``), which member lowering does not turn into a free helper, so an explicit
``T(value)`` or ``static_cast<T>(value)`` has no lowered helper to call. The importer keeps the scalar constructor,
and every code generator lowers it through the contract's value conversion of the real part. Any other aggregate
without a lowered helper is still refused rather than emitted as a scalar constructor over a struct value.
"""

import re

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalSourceOverloadResolutionError,
)

SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
struct STRUCT {
  T real;
  T imag;
  template <typename U>
  constexpr operator U() const thread {
    return static_cast<U>(real);
  }
};

template <typename U, typename T>
inline U cast_to(T val) {
  return static_cast<U>(val);
}

kernel void probe(device const STRUCT<float>* src [[buffer(0)]],
                  device uint* dst [[buffer(1)]],
                  device float* real_out [[buffer(2)]],
                  uint tid [[thread_position_in_grid]]) {
  STRUCT<float> value = src[tid];
  dst[tid] = cast_to<uint>(value);
  real_out[tid] = CONVERSION;
}
"""

CONVERSIONS = {
    "static_cast": "static_cast<float>(value)",
    "functional": "float(value)",
}

EXPECTED_REAL_PARTS = {
    "opengl": ("uint((val).real)", "(value).real"),
    "directx": ("uint((val).real)", "(value).real"),
    "metal": ("uint((val).real)", "(value).real"),
}


def _translate(tmp_path, struct_name, conversion, backend):
    path = tmp_path / "probe.metal"
    path.write_text(
        SOURCE.replace("STRUCT", struct_name).replace("CONVERSION", conversion),
        encoding="utf-8",
    )
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", sorted(EXPECTED_REAL_PARTS))
@pytest.mark.parametrize("spelling", sorted(CONVERSIONS))
def test_registered_representation_converts_its_real_part(tmp_path, backend, spelling):
    code = _translate(tmp_path, "complex_t", CONVERSIONS[spelling], backend)
    for fragment in EXPECTED_REAL_PARTS[backend]:
        assert fragment in code
    # No scalar constructor over the struct value itself survives.
    assert re.search(r"\buint\(val\)", code) is None
    assert re.search(r"\bfloat\(value\)", code) is None


@pytest.mark.parametrize("spelling", sorted(CONVERSIONS))
def test_unregistered_aggregate_without_helper_is_refused(tmp_path, spelling):
    with pytest.raises(
        MetalSourceOverloadResolutionError,
        match="no readonly scalar conversion helper",
    ):
        _translate(tmp_path, "pair_t", CONVERSIONS[spelling], "cgl")
