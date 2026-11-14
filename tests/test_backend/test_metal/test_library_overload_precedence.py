"""An enabled constrained template competes with the Metal library's overloads wherever lookup finds both.

Beside ``using namespace metal;`` at global scope, a call of a user template named like a library function also finds
the library's non-template overloads. A library overload whose parameter types are the argument types is as good a
match as the template's specialization, and C++ prefers the non-template function ([over.match.best]), so
``max(x, 1.0f)`` on a ``float x`` calls ``metal::max`` although the float-only user ``max`` is enabled. Where the
library has no such overload, the template's exact match wins. Every source here compiles with the Metal compiler,
and the expectations follow the functions its AIR calls.
"""

import re

import pytest

import crosstl
from crosstl.backend.Metal.library_overloads import (
    LIBRARY_OVERLOAD_NAMES,
    library_overload_precedence,
    library_parameter_lists,
    library_type_spelling,
)
from crosstl.backend.Metal.preprocessor import MetalTemplateSpecializationError

FLOAT_MAX = """#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
T max(T a, T b) { return a > b ? a : b + T(100); }
"""

# The AIR calls air.fast_fmax.f32: the library's max(float, float) wins.
LIBRARY_WINS = (
    FLOAT_MAX
    + """kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = max(o[i], 1.0f);
}
"""
)

# The library has no bfloat max, so the template serves the bfloat call; float4 disables it.
TEMPLATE_WINS_FOR_BFLOAT = (
    FLOAT_MAX
    + """kernel void k(device float4* o [[buffer(0)]], device bfloat* q [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
  o[i] = max(o[i], float4(1.0f));
  q[i] = max(q[i], bfloat(1.0f));
}
"""
)

# The template matches (float, int) exactly; every library max needs a conversion.
TEMPLATE_WINS_WITHOUT_CONVERSION = """#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
T max(T a, int b) { return a > T(b) ? a : T(b) + T(100); }
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = max(o[i], 4);
}
"""

# MLX's integral-only isnan: the template serves int, the library serves float.
INTEGRAL_ISNAN = """#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
bool isnan(T) { return false; }
kernel void k(device float* o [[buffer(0)]], device int* p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  o[i] = isnan(p[i]) ? 1.0f : (isnan(o[i]) ? 2.0f : 3.0f);
}
"""

# Inside ns the template hides the library, so it serves the float call.
NAMESPACED = """#include <metal_stdlib>
using namespace metal;
namespace ns {
template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
T max(T a, T b) { return a > b ? a : b + T(100); }
float h(float a) { return max(a, 1.0f); }
}
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = ns::h(o[i]);
}
"""

# A non-const reference parameter ties with max(float, float) for an lvalue, so the library wins.
REFERENCE_PARAMETER = """#include <metal_stdlib>
using namespace metal;
template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
T max(thread T& a, T b) { return a > b ? a : b + T(100); }
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float x = o[i];
  o[i] = max(x, 1.0f);
}
"""

# The template's third parameter is a non-deduced alias the importer does not reduce; the library's exact
# fma(float, float, float) wins whatever it names, so the AIR calls air.fma.f32.
NON_DEDUCED_PARAMETER = """#include <metal_stdlib>
using namespace metal;
template <typename U, typename V>
using common_t = decltype(U(0) + V(0));
template <typename T, typename U, enable_if_t<is_floating_point_v<T>, bool> = true>
common_t<T, U> fma(const T x, const U y, const common_t<T, U> z) { return x * y + z + T(100); }
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float v = o[i];
  o[i] = fma(v, v, 2.0f);
}
"""

BACKENDS = ["metal", "directx", "opengl"]


def _translate(tmp_path, source, backend):
    path = tmp_path / "precedence.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _kernel(code):
    """The entry point's text: everything after the last ``k(``, ``main(`` or ``CSMain(`` header."""
    starts = [match.start() for match in re.finditer(r"\b(?:k|main|CSMain)\s*\(", code)]
    assert starts, code
    return " ".join(code[starts[-1] :].split())


@pytest.mark.parametrize("backend", BACKENDS)
def test_exact_library_overload_beats_the_enabled_template(tmp_path, backend):
    code = _translate(tmp_path, LIBRARY_WINS, backend)

    assert "100" not in code
    assert not re.search(r"\bmax_\w+\(", code)


@pytest.mark.parametrize("backend", BACKENDS)
def test_template_serves_types_the_library_lacks(tmp_path, backend):
    code = _translate(tmp_path, TEMPLATE_WINS_FOR_BFLOAT, backend)

    assert code.count("100") == 1
    assert re.search(r"\bmax_bf\w*\(", _kernel(code))
    assert not re.search(r"\bmax_float4?\w*\(", code)


@pytest.mark.parametrize("backend", BACKENDS)
def test_template_wins_where_every_library_overload_converts(tmp_path, backend):
    code = _translate(tmp_path, TEMPLATE_WINS_WITHOUT_CONVERSION, backend)

    assert re.search(r"\bmax_float\(", _kernel(code))
    assert code.count("100") == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_integral_template_and_library_float_overload_split_the_calls(
    tmp_path, backend
):
    code = _translate(tmp_path, INTEGRAL_ISNAN, backend)

    kernel = _kernel(code)
    assert re.search(r"\bisnan_int\(", kernel)
    assert re.search(r"(?<![\w.])isnan\(", kernel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_namespaced_template_hides_the_library(tmp_path, backend):
    code = _translate(tmp_path, NAMESPACED, backend)

    assert re.search(r"\bmax_float\(a, 1\.0f?\)", code)
    assert code.count("100") == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_library_beats_a_template_taking_a_reference(tmp_path, backend):
    code = _translate(tmp_path, REFERENCE_PARAMETER, backend)

    assert "100" not in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_library_beats_a_template_with_a_non_deduced_parameter(tmp_path, backend):
    code = _translate(tmp_path, NON_DEDUCED_PARAMETER, backend)

    assert "100" not in code
    assert not re.search(r"\bfma_\w+\(", code)


def test_reference_template_without_a_library_match_is_refused(tmp_path):
    source = REFERENCE_PARAMETER.replace(
        "max(x, 1.0f)", "float(max(b, bfloat(1.0f)))"
    ).replace("float x = o[i];", "bfloat b = bfloat(o[i]);")

    with pytest.raises(MetalTemplateSpecializationError, match="unproven"):
        _translate(tmp_path, source, "metal")


def test_table_lists_every_scalar_and_vector_max():
    lists = library_parameter_lists("max")

    for element in ("char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"):
        for width in (1, 2, 3, 4, 8, 16):
            spelling = library_type_spelling(element, width)
            assert (spelling, spelling) in lists
    assert ("float4", "float4") in lists and ("half", "half") in lists
    assert ("bfloat", "bfloat") not in lists
    assert ("float", "float") in library_parameter_lists("fmax")
    assert {"isnan", "clamp", "select", "dot", "ldexp"} <= LIBRARY_OVERLOAD_NAMES
    assert len(LIBRARY_OVERLOAD_NAMES) == 100


@pytest.mark.parametrize(
    ("name", "arguments", "parameters", "expected"),
    [
        ("max", ["float", "float"], ["float", "float"], "library"),
        ("max", ["bfloat", "bfloat"], ["bfloat", "bfloat"], "template"),
        ("isnan", ["int"], ["int"], "template"),
        ("isnan", ["float"], ["float"], "library"),
        ("max", ["float", "int"], ["float", "int"], "template"),
        ("max", ["float", "int"], ["float", "float"], "unproven"),
        ("max", ["float", "float"], [None, "float"], "library"),
        ("max", ["bfloat", "bfloat"], [None, "bfloat"], "unproven"),
        ("max", ["float", "float"], ["thread float &", "float"], "library"),
        ("max", ["bfloat", "bfloat"], ["thread bfloat &", "bfloat"], "unproven"),
        ("modf", ["float", "float"], ["float", "float"], "unproven"),
        ("ldexp", ["float", "int"], ["float", "int"], "library"),
        ("max", ["float"], ["float"], "template"),
        ("not_a_library_function", ["float"], ["float"], "template"),
    ],
)
def test_library_overload_precedence(name, arguments, parameters, expected):
    assert library_overload_precedence(name, arguments, parameters) == expected


def test_library_type_spelling_folds_fixed_width_aliases():
    assert library_type_spelling("int32_t", 1) == "int"
    assert library_type_spelling("uint16_t", 4) == "ushort4"
    assert library_type_spelling("bfloat16_t", 2) == "bfloat2"
    assert library_type_spelling("size_t", 1) == "ulong"


def test_library_type_spelling_refuses_unknown_elements():
    assert library_type_spelling("float", 3) == "float3"
    assert library_type_spelling("bfloat", 1) == "bfloat"
    assert library_type_spelling("uchar4", 1) is None
    assert library_type_spelling("MyScalar", 1) is None
