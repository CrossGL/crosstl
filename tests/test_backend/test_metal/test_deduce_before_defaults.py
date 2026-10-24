"""A function template's defaulted parameter that the call can deduce takes the deduced type, not its default.

``template <typename T, typename U = float> T f(U factor, T value)`` called as ``f<int>(half(2), 3)`` binds T to int
explicitly and deduces U as half from the argument; the default applies only to a parameter that is neither supplied
nor deducible ([temp.deduct.call], [temp.arg.explicit]/4). Translation used to materialize U = float, so the
helper's ``sizeof(U)`` read 4 instead of 2 and the half argument was widened.
"""

import pytest

import crosstl

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

DEDUCED_DEFAULT = PRELUDE + """
template <typename T, typename U = float>
T scale_by(U factor, T value) {
    return value * T(factor) + T(sizeof(U));
}

kernel void scale_kernel(device int* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = scale_by<int>(half(2), 3);
}
"""

DEPENDENT_DEFAULT = PRELUDE + """
template <typename I, bool STRIDED>
struct indexer_t {
    typedef I IndexT;
    I stride;
    I operator()(I i) const { return STRIDED ? i * stride : i; }
};

template <typename T, ushort N, typename Indexer, typename IndexT = typename Indexer::IndexT>
T sum_values(Indexer indexer, constant IndexT& count, device const T* src) {
    T total = 0;
    for (IndexT i = 0; i < count; ++i) {
        total += src[indexer(i)];
    }
    return total * T(N);
}

kernel void sum_kernel(constant uint& count [[buffer(0)]],
                       device const float* src [[buffer(1)]],
                       device float* dst [[buffer(2)]]) {
    indexer_t<uint, false> indexer;
    indexer.stride = 1;
    dst[0] = sum_values<float, 4>(indexer, count, src);
}
"""

PARTIAL_EXPLICIT = PRELUDE + """
template <typename T, typename U>
T convert_to(U value) {
    return T(value);
}

kernel void convert_kernel(device float* values [[buffer(0)]],
                           device int* ints [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    ints[gid] = convert_to<int>(values[gid]);
}
"""


def translate(tmp_path, source, backend, name="kernel.metal"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize(
    ("backend", "signature"),
    [
        ("directx", "int scale_by_int_half(float16_t factor, int value)"),
        ("opengl", "int scale_by_int_half("),
    ],
)
def test_deducible_default_takes_the_deduced_type(tmp_path, backend, signature):
    code = translate(tmp_path, DEDUCED_DEFAULT, backend)
    assert signature in code
    assert "scale_by_int(" not in code
    # sizeof(U) is sizeof(half).
    assert "int(2)" in code
    assert "int(4)" not in code


def test_deducible_default_translates_to_spirv(tmp_path):
    assert "OpFunctionCall %int" in translate(tmp_path, DEDUCED_DEFAULT, "vulkan")


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_default_naming_a_deduced_parameter_is_applied_after_deduction(
    tmp_path, backend
):
    code = translate(tmp_path, DEPENDENT_DEFAULT, backend)
    if backend != "vulkan":
        assert "sum_values_float_4_indexer_t_uint_0_uint" in code
    else:
        assert "OpEntryPoint GLCompute" in code


def test_partially_explicit_call_keeps_its_arguments_in_crossgl(tmp_path):
    crossgl = translate(tmp_path, PARTIAL_EXPLICIT, "cgl")
    assert "generic<T, U> T convert_to(U value)" in crossgl
    assert "convert_to<int>(buffer_load(values, gid))" in crossgl
    # A sanitized ``convert_to<int>`` names no CrossGL function.
    assert "convert_to_u3cint_u3e" not in crossgl

    hlsl = translate(tmp_path, crossgl, "directx", name="roundtrip.cgl")
    assert "int convert_to_int_float(float value)" in hlsl
    assert "convert_to_int_float(values" in hlsl
