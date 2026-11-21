"""CUDA and HIP array extents import as the values of their integral constant expressions.

``float s[N * 2]`` imported as ``array<f32, N_mul_2>``, ``__shared__ float tile[32][32 + 1]`` as
``array<array<f32, cuda_array_extent_32_plus_1>, 32>`` and ``float t[N]`` as ``array<f32, N>``, which the CrossGL
parser reads as a type, so every target refused the array (the ``constant-array-size`` coverage cell of CUDA and
HIP input). The importers now fold an extent with C++ integer arithmetic, in the scopes the source declares its
constants in, write an extent of names they cannot evaluate (a template value parameter) as the expression, and
refuse any other extent with a typed import refusal instead of writing an identifier no program declares.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.backend.CUDA.array_extents import (
    IntegerConstants,
    crossgl_array_extent,
    expression_extent,
    fold_integer_constant,
)
from crosstl.backend.import_refusals import ImportRefused

TARGETS = ("cuda", "hip", "metal", "directx", "opengl", "vulkan")


def _constants(**values):
    constants = IntegerConstants()
    for name, value in values.items():
        constants.declare(name, value)
    return constants


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        ("4", 4),
        ("0x10", 16),
        ("010", 8),
        ("N * 2", 8),
        ("(N + 1) * (N - 1)", 15),
        ("-7 / 2", -3),
        ("-7 % 2", -1),
        ("7 / -2", -3),
        ("1 << N", 16),
        ("256 >> 2 | 1", 65),
        ("N > 2 ? 3 : 5", 3),
        ("~0 & 0xff", 255),
        ("!0 + !N", 1),
        ("N + 2 * 3 - 1", 9),
    ],
)
def test_folds_with_cpp_integer_arithmetic(expression, value):
    assert fold_integer_constant(expression, _constants(N=(4, False)))[0] == value


@pytest.mark.parametrize(
    "expression",
    [
        "M",  # not a constant
        "N / 0",
        "N % 0",
        "0u - 1u",  # wraps around in C++: left to CrossGL
        "-1 < 0u",  # the usual arithmetic conversions make -1 unsigned
        "1 << 64",
        "N * 2.0f",
        "sizeof(float) * N",
        "Config::N",
    ],
)
def test_leaves_what_it_cannot_evaluate_exactly(expression):
    assert fold_integer_constant(expression, _constants(N=(4, False))) is None


def test_extent_keeps_an_unknown_expression():
    constants = _constants(N=(4, False))
    refused = []

    def extent(size):
        def other(text):
            expression = expression_extent(text)
            if expression is None:
                refused.append(text)
                return "refused"
            return expression

        return crossgl_array_extent(size, constants, other)

    assert extent("N * 2") == "8"
    assert extent("K") == "K"
    # An extent of names the importer cannot evaluate is written as the expression, for CrossGL to substitute.
    assert extent("K + 1") == "(K + 1)"
    assert extent("(K * 2) + 1u") == "((K * 2) + 1u)"
    assert extent("(K << 0x2)") == "(K << 2)"
    assert extent("-K + 010") == "(-K + 8)"
    assert extent("N - 4") == "(N - 4)"
    # Any other extent has no CrossGL spelling and is refused.
    for other in (
        "FFT::storage_size",
        "cosize_v<Layout>",
        "sizeof(float) * K",
        "K > 2 ? 4 : 8",
        "f(K)",
        "K K",
        "(K",
        "4u - 5u",
    ):
        assert extent(other) == "refused", other
    assert refused == [
        "FFT::storage_size",
        "cosize_v<Layout>",
        "sizeof(float) * K",
        "K > 2 ? 4 : 8",
        "f(K)",
        "K K",
        "(K",
        "4u - 5u",
    ]


def test_an_inner_declaration_hides_a_constant():
    constants = _constants(N=(4, False))
    constants.push()
    constants.declare("N")
    assert fold_integer_constant("N", constants) is None
    constants.pop()
    assert fold_integer_constant("N", constants) == (4, False)


CUDA_SOURCE = """
#define BS 8
constexpr int N = 4;
static const unsigned M = N * 2u;
__global__ void k(float* out) {
    const int TILE = 4;
    __shared__ float tile[TILE][TILE + 1];
    float a[N * 2];
    float b[M];
    float c[BS];
    {
        const int N = 2;
        float inner[N + 1];
        inner[0] = 1.0f;
        out[1] = inner[0];
    }
    float after[N + 1];
    int idx = threadIdx.x;
    tile[idx % TILE][0] = 1.0f;
    a[idx % 8] = 2.0f;
    b[0] = 1.0f;
    c[0] = 3.0f;
    after[0] = 4.0f;
    __syncthreads();
    out[idx] = tile[0][0] + a[0] + b[0] + c[0] + after[0];
}
"""


def _import(tmp_path, source, suffix):
    path = tmp_path / f"extents{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="cgl", format_output=False)


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_imported_extents_are_constant_in_every_target(tmp_path, language):
    if language == "hip":
        source = "#include <hip/hip_runtime.h>\n" + CUDA_SOURCE.replace(
            "threadIdx.x", "hipThreadIdx_x"
        )
        crossgl = _import(tmp_path, source, ".hip")
    else:
        crossgl = _import(tmp_path, CUDA_SOURCE, ".cu")

    assert "var<workgroup> tile: array<array<f32, 5>, 4>;" in crossgl
    assert "var a: array<f32, 8>;" in crossgl
    assert "var b: array<f32, 8>;" in crossgl
    assert "var c: array<f32, 8>;" in crossgl
    # The block-scope N = 2 sizes only the array of its block.
    assert "var inner: array<f32, 3>;" in crossgl
    assert "var after: array<f32, 5>;" in crossgl
    assert "_mul_" not in crossgl and "_plus_" not in crossgl

    path = tmp_path / "extents.cgl"
    path.write_text(crossgl)
    for target in TARGETS:
        assert crosstl.translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_a_template_value_parameter_hides_a_namespace_constant(tmp_path, suffix):
    source = """
constexpr int N = 4;
template <int N>
__device__ float first(float x) {
    float local[N];
    local[0] = x;
    return local[0];
}
"""
    crossgl = _import(tmp_path, source, suffix)
    assert "generic<N> f32 first(f32 x)" in crossgl
    assert "var local: array<f32, N>;" in crossgl


TEMPLATE_EXTENT_SOURCE = """
template <int K>
__device__ float spread(float x) {
    float s[K * 2 + 1];
    for (int i = 0; i < K * 2 + 1; ++i) {
        s[i] = x * i;
    }
    return s[K * 2];
}
__global__ void k(float* out) { out[0] = spread<3>(1.0f); }
"""


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_a_template_value_extent_is_written_as_an_expression(tmp_path, suffix):
    source = TEMPLATE_EXTENT_SOURCE
    if suffix == ".hip":
        source = "#include <hip/hip_runtime.h>\n" + source
    crossgl = _import(tmp_path, source, suffix)
    assert "var s: array<f32, ((K * 2) + 1)>;" in crossgl
    assert "_mul_" not in crossgl and "_plus_" not in crossgl

    path = tmp_path / "template_extent.cgl"
    path.write_text(crossgl)
    # CrossGL substitutes K = 3 into the extent, so every target sizes the array with 7 elements.
    for target in TARGETS:
        code = crosstl.translate(str(path), backend=target, format_output=False)
        if target == "vulkan":
            assert "OpTypeArray" in code
        else:
            assert "K_mul_2" not in code and "K * 2" not in code
            assert "3 * 2" in code


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
@pytest.mark.parametrize(
    "extent", ["Config::size", "sizeof(float4) / 4", "Wide<int>::value"]
)
def test_an_extent_crossgl_cannot_spell_is_refused(tmp_path, suffix, extent):
    source = (
        "struct Config { static constexpr int size = 4; };\n"
        "template <typename T> struct Wide { static constexpr int value = 4; };\n"
        f"__global__ void k(float* out) {{ float s[{extent}]; s[0] = 1.0f; out[0] = s[0]; }}\n"
    )
    if suffix == ".hip":
        source = "#include <hip/hip_runtime.h>\n" + source
    with pytest.raises(ImportRefused, match="array extent") as raised:
        _import(tmp_path, source, suffix)
    language = "cuda" if suffix == ".cu" else "hip"
    assert raised.value.code == f"project.translate.{language}-import-unsupported"


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_an_extent_that_names_an_undeclared_macro_is_refused(tmp_path, suffix):
    # CHUNK_K and SKEW are macros of a header the importer did not read: written as the expression, every target
    # compiler rejected the undeclared names, so the import is refused and names them.
    source = (
        "__global__ void k(float* out) {\n"
        "    float s[CHUNK_K * 2 + SKEW];\n"
        "    s[0] = 1.0f;\n"
        "    out[0] = s[0];\n"
        "}\n"
    )
    if suffix == ".hip":
        source = "#include <hip/hip_runtime.h>\n" + source
    with pytest.raises(ImportRefused, match="'CHUNK_K', 'SKEW', which the program"):
        _import(tmp_path, source, suffix)


ENUM_EXTENT_SOURCE = """
enum { FIRST, TILE = 16, NEXT };
enum class Scoped { NEXT = 2 };
__global__ void k(float* out) {
    float s[TILE * 2];
    float t[NEXT - FIRST];
    s[0] = 1.0f;
    t[0] = s[0];
    out[0] = t[0];
}
"""


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_an_enumerator_extent_folds(tmp_path, suffix):
    source = ENUM_EXTENT_SOURCE
    if suffix == ".hip":
        source = "#include <hip/hip_runtime.h>\n" + source
    crossgl = _import(tmp_path, source, suffix)
    # FIRST is 0 and NEXT is 17 ([dcl.enum]/2); the scoped enum's NEXT needs its enum's name, so it does not hide.
    assert "var s: array<f32, 32>;" in crossgl
    assert "var t: array<f32, 17>;" in crossgl
