"""CUDA and HIP intrinsics that round down, up or toward zero.

A call with a ``float`` or ``double`` result imports as a call of a CrossGL function that computes the directed result
exactly (crosstl/backend/CUDA/directed_rounding_helpers.py); test_directed_rounding_exact.py checks those functions
against exact rational arithmetic. The conversions to ``half`` and ``bfloat16`` have no such function yet and are
refused with a typed diagnostic, as every directed intrinsic was before: writing them as the plain operation rounds
to nearest, one ULP away from the source's result with no diagnostic.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.backend.CUDA.directed_rounding import (
    DIRECTED_ROUNDING_INTRINSIC,
    DirectedRoundingLibrary,
    DirectedRoundingUnsupportedError,
    directed_rounding_call,
    refuse_directed_rounding,
)
from crosstl.translator.diagnostics import TranslationDiagnostic

KERNEL = """{prelude}__global__ void k(const float *a, const int *n, float *o) {{
    unsigned int i = threadIdx.x;
    o[i] = {call};
}}
"""

DOUBLE_KERNEL = """{prelude}__global__ void k(const double *a, const long long *n, double *o, float *f) {{
    unsigned int i = threadIdx.x;
    o[i] = {call};
    f[i] = __double2float_rd(a[i]) + __double2float_ru(a[i]) + __double2float_rz(a[i]);
}}
"""

LANGUAGES = pytest.mark.parametrize(
    ("suffix", "prelude"),
    [(".cu", ""), (".hip", "#include <hip/hip_runtime.h>\n")],
    ids=["cuda", "hip"],
)

FLOAT_CALLS = [
    ("__fadd_rd(a[i], a[i + 1])", "fadd_rd(f32(a[i]), f32(a[(i + 1)]))"),
    ("__fsub_ru(a[i], a[i + 1])", "fsub_ru(f32(a[i]), f32(a[(i + 1)]))"),
    ("__fmul_rz(a[i], a[i + 1])", "fmul_rz(f32(a[i]), f32(a[(i + 1)]))"),
    ("__fdiv_rd(a[i], a[i + 1])", "fdiv_rd(f32(a[i]), f32(a[(i + 1)]))"),
    ("__frcp_ru(a[i])", "frcp_ru(f32(a[i]))"),
    ("__fsqrt_rz(a[i])", "fsqrt_rz(f32(a[i]))"),
    (
        "__fmaf_rd(a[i], a[i + 1], a[i + 2])",
        "fmaf_rd(f32(a[i]), f32(a[(i + 1)]), f32(a[(i + 2)]))",
    ),
    (
        "__fmaf_ieee_ru(a[i], a[i + 1], a[i + 2])",
        "fmaf_ieee_ru(f32(a[i]), f32(a[(i + 1)]), f32(a[(i + 2)]))",
    ),
    ("__int2float_rz(n[i])", "int2float_rz(i32(n[i]))"),
    ("__uint2float_rd((unsigned int)n[i])", "uint2float_rd(u32(u32(n[i])))"),
    ("__ll2float_ru((long long)n[i])", "ll2float_ru(i64(i64(n[i])))"),
]

DOUBLE_CALLS = [
    "__dadd_rd(a[i], a[i + 1])",
    "__dsub_rz(a[i], a[i + 1])",
    "__dmul_ru(a[i], a[i + 1])",
    "__ddiv_rz(a[i], a[i + 1])",
    "__drcp_rd(a[i])",
    "__dsqrt_ru(a[i])",
    "__fma_rz(a[i], a[i + 1], a[i + 2])",
    "__ll2double_rd(n[i])",
    "__ull2double_ru((unsigned long long)n[i])",
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

REFUSED_CALLS = [
    "__half2float(__float2half_rz(a[i]))",
    "__half2float(__float2half_rd(a[i]))",
    "__half2float(__int2half_ru(n[i]))",
    "__half2float(__ushort2half_rd((unsigned short)n[i]))",
]


def translate_source(tmp_path, suffix, source, backend="cgl"):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def translate_crossgl(tmp_path, crossgl, backend):
    path = tmp_path / "imported.cgl"
    path.write_text(crossgl, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@LANGUAGES
@pytest.mark.parametrize(
    ("call", "imported"), FLOAT_CALLS, ids=[c for c, _ in FLOAT_CALLS]
)
def test_a_float_directed_intrinsic_imports_as_an_exact_function(
    tmp_path, suffix, prelude, call, imported
):
    crossgl = translate_source(
        tmp_path, suffix, KERNEL.format(prelude=prelude, call=call)
    )
    assert f"o[i] = {imported};" in crossgl
    assert call.split("(")[0] not in crossgl
    wrapper = imported.split("(")[0]
    assert crossgl.count(f" {wrapper}(") == 2  # defined once and called once


@LANGUAGES
@pytest.mark.parametrize("call", DOUBLE_CALLS)
def test_a_double_directed_intrinsic_imports_as_an_exact_function(
    tmp_path, suffix, prelude, call
):
    crossgl = translate_source(
        tmp_path, suffix, DOUBLE_KERNEL.format(prelude=prelude, call=call)
    )
    intrinsic = call.split("(")[0]
    assert f" {intrinsic[2:]}(" in crossgl
    assert intrinsic not in crossgl and "__double2float" not in crossgl


@LANGUAGES
def test_the_imported_functions_translate_to_every_target(tmp_path, suffix, prelude):
    single = " + ".join(call for call, _ in FLOAT_CALLS)
    crossgl = translate_source(
        tmp_path, suffix, KERNEL.format(prelude=prelude, call=single)
    )
    for backend in ("cuda", "hip", "metal", "directx", "opengl", "vulkan"):
        assert translate_crossgl(tmp_path, crossgl, backend)
    double = " + ".join(DOUBLE_CALLS)
    crossgl = translate_source(
        tmp_path, suffix, DOUBLE_KERNEL.format(prelude=prelude, call=double)
    )
    for backend in ("cuda", "hip", "directx", "opengl", "vulkan"):
        assert translate_crossgl(tmp_path, crossgl, backend)
    # Metal has no double (MSL 4.1 section 2.1) and says so.
    with pytest.raises(TranslationDiagnostic, match="double"):
        translate_crossgl(tmp_path, crossgl, "metal")


@LANGUAGES
@pytest.mark.parametrize("call", NEAREST_CALLS + INTEGRAL_CALLS)
def test_round_to_nearest_and_integral_conversions_still_translate(
    tmp_path, suffix, prelude, call
):
    crossgl = translate_source(
        tmp_path, suffix, KERNEL.format(prelude=prelude, call=call)
    )
    assert "o[i] =" in crossgl
    assert "directed_" not in crossgl


@LANGUAGES
@pytest.mark.parametrize("call", REFUSED_CALLS)
def test_a_directed_conversion_to_half_is_refused(tmp_path, suffix, prelude, call):
    include = (
        "#include <cuda_fp16.h>\n" if suffix == ".cu" else "#include <hip/hip_fp16.h>\n"
    )
    source = KERNEL.format(prelude=prelude + include, call=call)
    with pytest.raises(DirectedRoundingUnsupportedError) as raised:
        translate_source(tmp_path, suffix, source)
    error = raised.value
    assert isinstance(error, TranslationDiagnostic)
    assert error.code.endswith("directed-rounding-unsupported")
    assert "crossgl.directed-rounding" in error.missing_capabilities
    assert call.split("(")[1] in str(error)


@LANGUAGES
def test_the_functions_take_no_name_the_program_uses(tmp_path, suffix, prelude):
    # The kernel's buffer ``a`` and the user function ``fadd_rd`` keep their names; the CrossGL functions and their
    # parameters move aside, since Metal output refuses a parameter that shadows a global resource.
    source = (
        f"{prelude}__device__ float fadd_rd(float x) {{ return x + 1.0f; }}\n"
        "__global__ void k(const float *a, const float *b, float *o) {\n"
        "    unsigned int i = threadIdx.x;\n"
        "    o[i] = __fadd_rd(a[i], b[i]) + fadd_rd(a[i]) + __fmaf_rz(a[i], b[i], 1.0f);\n"
        "}\n"
    )
    crossgl = translate_source(tmp_path, suffix, source)
    assert "fadd_rd_(f32(a[i]), f32(b[i]))" in crossgl
    assert "fadd_rd(a[i])" in crossgl
    assert "f32 fadd_rd_(f32 a_, f32 b_) {" in crossgl
    assert "f32 a," not in crossgl and "u32 a," not in crossgl
    assert translate_crossgl(tmp_path, crossgl, "metal")


def test_a_call_with_the_wrong_number_of_arguments_is_refused():
    with pytest.raises(DirectedRoundingUnsupportedError, match="takes 2 arguments"):
        directed_rounding_call(DirectedRoundingLibrary(), "__fadd_rd", ["x"])
    assert (
        directed_rounding_call(DirectedRoundingLibrary(), "__fadd_rn", ["x", "y"])
        is None
    )
    assert directed_rounding_call(None, "__fadd_rd", ["x", "y"]) is None


def test_only_the_functions_a_program_calls_are_written():
    library = DirectedRoundingLibrary()
    library.call("__int2float_rd", ["n"])
    text = library.render()
    assert "f32 int2float_rd(i32 x) {" in text
    assert "directed_from_u32_f32" in text and "directed_round_f32" in text
    assert "u64" not in text and "f64" not in text
    assert "directed_sum_f32" not in text and "directed_div_f32" not in text


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
    [
        "__dadd_rd",
        "__fmaf_ieee_rz",
        "__double2float_ru",
        "__ull2double_rz",
        "__uint2float_rd",
    ],
)
def test_the_library_builds_the_float_and_double_families(name):
    assert DirectedRoundingLibrary().handles(f"::{name}")


@pytest.mark.parametrize(
    "name",
    [
        "__float2half_rz",
        "__double2bfloat16_rd",
        "__short2half_ru",
        "__fadd_rn",
        "__float2int_rd",
        "fadd_rd",
    ],
)
def test_the_library_leaves_other_names(name):
    assert not DirectedRoundingLibrary().handles(name)


@pytest.mark.parametrize(
    "name",
    ["__fadd_rn", "__float2int_rd", "__half2int_ru", "__fdividef", "fadd_rd", None],
)
def test_other_names_pass(name):
    refuse_directed_rounding(name)
