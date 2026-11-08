"""Mixed-precision cooperative-matrix multiply-accumulate, and targets that refuse cooperative matrices.

MSL ``simdgroup_multiply_accumulate(d, a, b, c)`` and SPIR-V ``OpCooperativeMatrixMulAddKHR`` take the component
types of A, B, the accumulator and the result independently: half inputs with a float accumulator is the common
form. CrossGL used to demand one component type for all four, so such kernels could not be translated at all, even
from Metal back to Metal.
"""

import pytest

import crosstl
from crosstl.translator.codegen.cooperative_matrix_refusals import (
    CooperativeMatrixLoweringUnsupportedError,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import CrossGLFunctionBodyParseError, Parser
from tests.test_translator.test_metal_builtin_ownership import _compile


def _result_type(parameters, expression):
    code = f"""
    shader MixedPrecision {{
        void main({parameters}) {{
            auto result = {expression};
        }}
    }}
    """
    ast = Parser(Lexer(code).tokens, strict_function_bodies=True).parse()
    return ast.functions[0].body.statements[0].initial_value.result_type


def _matrix(element, rows=8, cols=8):
    return f"CooperativeMatrix<{element}, {rows}, {cols}>"


@pytest.mark.parametrize(
    ("left", "right", "accumulator"),
    [
        ("float16", "float16", "float"),
        ("float", "float16", "float"),
        ("float16", "float", "float16"),
    ],
)
def test_multiply_accumulate_result_has_the_accumulator_component_type(
    left, right, accumulator
):
    result = _result_type(
        f"{_matrix(left)} a, {_matrix(right)} b, {_matrix(accumulator)} c",
        "cooperative_matrix_multiply_accumulate(a, b, c)",
    )
    assert result.element_type.name == accumulator
    assert (result.rows, result.cols) == (8, 8)


def test_multiply_accumulate_result_has_the_destination_component_type():
    result = _result_type(
        f"{_matrix('float16')} d, {_matrix('float16')} a, {_matrix('float16')} b, "
        f"{_matrix('float')} c",
        "cooperative_matrix_multiply_accumulate(d, a, b, c)",
    )
    assert result.element_type.name == "float16"


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (
            f"{_matrix('float16', 8, 4)} a, {_matrix('float16', 8, 8)} b, "
            f"{_matrix('float')} c",
            "incompatible inner dimension",
        ),
        (
            f"{_matrix('float16', 8, 8)} a, {_matrix('float16', 8, 4)} b, "
            f"{_matrix('float')} c",
            "incompatible column count",
        ),
    ],
)
def test_mixed_precision_multiply_accumulate_still_checks_shapes(parameters, message):
    with pytest.raises(CrossGLFunctionBodyParseError, match=message):
        _result_type(parameters, "cooperative_matrix_multiply_accumulate(a, b, c)")


def test_plain_multiply_still_needs_one_component_type():
    # A product without an accumulator has no operand that names its result type.
    with pytest.raises(
        CrossGLFunctionBodyParseError, match="incompatible component type"
    ):
        _result_type(
            f"{_matrix('float')} a, {_matrix('float16')} b",
            "cooperative_matrix_multiply(a, b)",
        )


METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;
kernel void mixed(device const AIN* a [[buffer(0)]], device const BIN* b [[buffer(1)]],
                  device DOUT* d [[buffer(2)]], uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_matrix<AIN, 8, 8> A;
    simdgroup_matrix<BIN, 8, 8> B;
    simdgroup_matrix<float, 8, 8> C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<DOUT, 8, 8> D;
    simdgroup_load(A, a, 8);
    simdgroup_load(B, b, 8);
    simdgroup_multiply_accumulate(D, A, B, C);
    simdgroup_store(D, d, 8);
}
"""

MIXED = {
    "half-inputs": ("half", "half", "float"),
    "mixed-inputs": ("float", "half", "float"),
    "half-destination": ("half", "half", "half"),
}


def _metal_source(tmp_path, case):
    a, b, d = MIXED[case]
    source = METAL_KERNEL.replace("AIN", a).replace("BIN", b).replace("DOUT", d)
    path = tmp_path / "mixed.metal"
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize("case", MIXED)
def test_metal_mixed_precision_multiply_accumulate_round_trips(tmp_path, case):
    a, b, d = MIXED[case]
    metal = crosstl.translate(
        str(_metal_source(tmp_path, case)), backend="metal", format_output=False
    )
    for name, element in (("A", a), ("B", b), ("C", "float"), ("D", d)):
        assert f"simdgroup_matrix<{element}, 8, 8> {name}" in metal
    assert "simdgroup_multiply_accumulate(D, A, B, C);" in metal
    _compile(metal, "metal", tmp_path)


TARGETS_WITHOUT_COOPERATIVE_MATRICES = ("cuda", "hip", "slang", "rust", "mojo", "wgsl")


@pytest.mark.parametrize("backend", TARGETS_WITHOUT_COOPERATIVE_MATRICES)
def test_targets_without_cooperative_matrices_refuse_them(tmp_path, backend):
    # These emitters used to write the CrossGL type, or a Python representation of
    # the type or of the operation node, into the program.
    with pytest.raises(CooperativeMatrixLoweringUnsupportedError) as caught:
        crosstl.translate(
            str(_metal_source(tmp_path, "half-inputs")),
            backend=backend,
            format_output=False,
            source_options={"threadgroup_size": (32, 1, 1)},
        )
    assert "operation multiply_accumulate" in caught.value.constructs
    assert "type CooperativeMatrix<float16, 8, 8, subgroup>" in caught.value.constructs
    assert caught.value.target == backend


@pytest.mark.parametrize("backend", TARGETS_WITHOUT_COOPERATIVE_MATRICES)
def test_cooperative_matrix_declaration_alone_is_refused(tmp_path, backend):
    path = tmp_path / "declaration.cgl"
    path.write_text(
        """
        shader Declaration {
            compute {
                layout(local_size_x = 32) in;
                void main() {
                    CooperativeMatrix<float, 8, 8> unused;
                }
            }
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(
        CooperativeMatrixLoweringUnsupportedError,
        match=r"type CooperativeMatrix<float, 8, 8, subgroup>",
    ):
        crosstl.translate(str(path), backend=backend, format_output=False)
