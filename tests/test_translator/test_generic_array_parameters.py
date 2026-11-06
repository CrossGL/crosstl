"""A specialised generic function substitutes its type argument in array parameters and local arrays (#3160)."""

import pytest

import crosstl.translator
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.slang_codegen import SlangCodeGen

GENERIC_ARRAYS = """
shader GenericArrays {
    buffer float* results;
    T first<T>(in T v[2]) { return v[0] + v[1]; }
    T corner<T>(in T a[2][3]) { return a[0][0] + a[1][2]; }
    T twice<T>(T x) { T tmp[2]; tmp[0] = x; tmp[1] = x; return tmp[0] + tmp[1]; }
    compute {
        @ numthreads(1, 1, 1)
        void main() {
            float a[2] = {1.0, 2.0};
            float c[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            results[0] = first<float>(a) + corner<float>(c) + twice<float>(1.5);
        }
    }
}
"""


@pytest.mark.parametrize(
    ("generator", "parameters"),
    [
        (
            HLSLCodeGen,
            [
                "float first_float(in float v[2])",
                "float corner_float(in float a[2][3])",
            ],
        ),
        (
            GLSLCodeGen,
            ["float first_float(float v[2])", "float corner_float(float a[2][3])"],
        ),
        (
            MetalCodeGen,
            ["float first_float(float v[2])", "float corner_float(float a[2][3])"],
        ),
        (
            SlangCodeGen,
            [
                "float first_float(in float v[2])",
                "float corner_float(in float a[2][3])",
            ],
        ),
    ],
)
def test_specialised_generic_function_substitutes_array_element_type(
    generator, parameters
):
    code = generator().generate(crosstl.translator.parse(GENERIC_ARRAYS))

    for parameter in parameters:
        assert parameter in code
    assert "float tmp[2];" in code
    for unsubstituted in ("T v[2]", "T a[2][3]", "T tmp[2]"):
        assert unsubstituted not in code
