"""CUDA and HIP intrinsics that round down, up or toward zero are refused with a typed diagnostic.

The importers used to write ``__fadd_rd(a, b)`` as ``a + b``, which rounds to nearest, so the result could differ
from the source's by one ULP with no diagnostic. See crosstl/backend/CUDA/directed_rounding.py.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.backend.CUDA.directed_rounding import (
    DIRECTED_ROUNDING_INTRINSIC,
    DirectedRoundingUnsupportedError,
    refuse_directed_rounding,
)
from crosstl.translator.diagnostics import TranslationDiagnostic

KERNEL = """{prelude}__global__ void k(const float *a, const int *n, float *o) {{
    unsigned int i = threadIdx.x;
    o[i] = {call};
}}
"""

DIRECTED_CALLS = [
    "__fadd_rd(a[i], a[i + 1])",
    "__fsub_ru(a[i], a[i + 1])",
    "__fmul_rz(a[i], a[i + 1])",
    "__fdiv_rd(a[i], a[i + 1])",
    "__frcp_ru(a[i])",
    "__fsqrt_rz(a[i])",
    "__fmaf_rd(a[i], a[i + 1], a[i + 2])",
    "__fmaf_ieee_ru(a[i], a[i + 1], a[i + 2])",
    "__int2float_rz(n[i])",
    "__uint2float_rd((unsigned int)n[i])",
    "__ll2float_ru((long long)n[i])",
]

NEAREST_CALLS = [
    "__fadd_rn(a[i], a[i + 1])",
    "__fmul_rn(a[i], a[i + 1])",
    "__fsqrt_rn(a[i])",
    "__int2float_rn(n[i])",
]

# Conversions to an integer round to an integral value first, which floor, ceil and trunc express exactly.
INTEGRAL_CALLS = [
    "(float)__float2int_rd(a[i])",
    "(float)__float2int_ru(a[i])",
    "(float)__float2int_rz(a[i])",
]


def translate_source(tmp_path, suffix, source, backend="cgl"):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    ("suffix", "prelude"),
    [(".cu", ""), (".hip", "#include <hip/hip_runtime.h>\n")],
    ids=["cuda", "hip"],
)
@pytest.mark.parametrize("call", DIRECTED_CALLS)
def test_a_directed_rounding_intrinsic_is_refused(tmp_path, suffix, prelude, call):
    source = KERNEL.format(prelude=prelude, call=call)
    with pytest.raises(DirectedRoundingUnsupportedError) as raised:
        translate_source(tmp_path, suffix, source)
    error = raised.value
    assert isinstance(error, TranslationDiagnostic)
    assert error.code.endswith("directed-rounding-unsupported")
    assert "crossgl.directed-rounding" in error.missing_capabilities
    assert call.split("(")[0] in str(error)


@pytest.mark.parametrize(
    ("suffix", "prelude"),
    [(".cu", ""), (".hip", "#include <hip/hip_runtime.h>\n")],
    ids=["cuda", "hip"],
)
@pytest.mark.parametrize("call", NEAREST_CALLS + INTEGRAL_CALLS)
def test_round_to_nearest_and_integral_conversions_still_translate(
    tmp_path, suffix, prelude, call
):
    source = KERNEL.format(prelude=prelude, call=call)
    output = translate_source(tmp_path, suffix, source)
    assert "o[i] =" in output


@pytest.mark.parametrize(
    "name",
    [
        "__dadd_rd",
        "__dmul_ru",
        "__drcp_rz",
        "__dsqrt_rd",
        "__fma_ru",
        "__float2half_rz",
        "__float2bfloat16_rd",
        "__double2float_ru",
        "__double2half_rd",
        "__ull2double_rz",
        "__short2half_ru",
        "__ushort2bfloat16_rd",
    ],
)
def test_the_pattern_covers_each_family(name):
    assert DIRECTED_ROUNDING_INTRINSIC.match(name)
    with pytest.raises(DirectedRoundingUnsupportedError):
        refuse_directed_rounding(f"::{name}")


@pytest.mark.parametrize(
    "name",
    ["__fadd_rn", "__float2int_rd", "__half2int_ru", "__fdividef", "fadd_rd", None],
)
def test_other_names_pass(name):
    refuse_directed_rounding(name)
