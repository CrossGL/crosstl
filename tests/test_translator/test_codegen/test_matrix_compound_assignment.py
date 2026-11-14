"""CrossGL ``*=`` with a matrix right operand is a product in every generator.

GLSL 4.60 section 5.9: ``m *= n`` is ``m = m * n`` (the linear-algebra product) and ``v *= m`` is ``v = v * m``.
HLSL and Slang define ``*=`` on matrices component-wise and reject ``vector *= matrix``; MSL has no ``*=`` that takes
two matrices. Each generator must write the same code as for the explicit product.
"""

import pytest

from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.matrix_compound_assignment import (
    MatrixCompoundAssignmentError,
)
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GENERATORS = {
    "directx": HLSLCodeGen,
    "metal": MetalCodeGen,
    "slang": SlangCodeGen,
    "glsl": GLSLCodeGen,
}


def kernel(statements, *, declarations=""):
    return f"""
shader MatrixCompound {{
    struct Frame {{
        mat3 basis;
        vec3 axis;
    }}

    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        {declarations}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            mat2 m = mat2(float(iin[i]), 1.0, 2.0, 3.0);
            mat2 n = mat2(1.0, 2.0, 0.0, 1.0);
            vec2 v = vec2(1.0, 1.0);
            mat2 ms[2];
            ms[0] = m;
            ms[1] = n;
            Frame frame;
            frame.basis = mat3(1.0);
            frame.axis = vec3(1.0, 2.0, 3.0);
            {statements}
            outp[i] = m[0][1] + v.x + ms[1][0][1] + frame.basis[0][0] + frame.axis.y;
        }}
    }}
}}
"""


def generate(backend, source):
    ast = Parser(Lexer(source).get_tokens()).parse()
    return GENERATORS[backend]().generate(ast)


COMPOUND = (
    "m *= n; v *= m; ms[1] *= m; frame.basis *= mat3(2.0); frame.axis *= frame.basis;"
)
EXPLICIT = (
    "m = m * n; v = v * m; ms[1] = ms[1] * m; frame.basis = frame.basis * mat3(2.0); "
    "frame.axis = frame.axis * frame.basis;"
)


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_matrix_compound_multiply_matches_the_explicit_product(backend):
    compound = generate(backend, kernel(COMPOUND))
    explicit = generate(backend, kernel(EXPLICIT))

    assert compound == explicit
    assert "m = mul(m, n);" in compound
    assert "v = mul(v, m);" in compound
    assert "*= m" not in compound and "*= n" not in compound


def test_metal_matrix_compound_multiply_is_a_product():
    compound = generate("metal", kernel(COMPOUND))
    explicit = generate(
        "metal",
        kernel(
            EXPLICIT.replace("v = v * m;", "v *= m;").replace(
                "frame.axis = frame.axis * frame.basis;", "frame.axis *= frame.basis;"
            )
        ),
    )

    # MSL defines vector *= matrix as the row-vector product, so only matrix targets change.
    assert compound == explicit
    assert "m = m * n;" in compound
    assert "v *= m;" in compound
    assert "*= n" not in compound


@pytest.mark.parametrize("backend", ["directx", "metal", "slang"])
def test_scalar_compound_multiply_of_a_matrix_is_unchanged(backend):
    output = generate(backend, kernel("m *= 2.0; v *= 3.0;"))

    assert "m *= 2.0" in output
    assert "v *= 3.0" in output


def test_glsl_keeps_the_compound_product():
    output = generate("glsl", kernel(COMPOUND))

    assert "m *= n;" in output
    assert "v *= m;" in output


@pytest.mark.parametrize("backend", ["directx", "metal", "slang"])
@pytest.mark.parametrize(
    "statement",
    ["ms[next()] *= n;", "ms[i++ & 1u] *= n;"],
    ids=["call-index", "increment-index"],
)
def test_matrix_compound_multiply_refuses_targets_with_side_effects(backend, statement):
    source = kernel(
        statement,
        declarations="int counter; int next() { counter = counter + 1; return counter & 1; }",
    )

    with pytest.raises(MatrixCompoundAssignmentError) as raised:
        generate(backend, source)

    assert raised.value.code.endswith("matrix-compound-assignment-unsupported")
    assert raised.value.target == backend
