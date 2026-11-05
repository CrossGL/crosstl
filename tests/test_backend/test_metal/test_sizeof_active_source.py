"""`sizeof` of aggregates folds against the source being preprocessed.

[expr.sizeof]/2 gives a class operand its object size. Integral constant
initializers such as `16 / sizeof(T)` are often folded without a resolver for
the declaration's position, so the frontend resolves the operand against the
file scope of the active source: concrete structs, aliases and class template
specializations.
"""

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor

SOURCE = """
template <typename T>
struct Pair_t;

template <typename T>
struct Pair_t {
  T re;
  T im;
  Pair_t(T r, T i) thread : re(r), im(i) {}
};

using cplx = Pair_t<float>;

struct Quad {
  float a;
  float b;
  float c;
  float d;
};

constant constexpr int kPad = 16 / sizeof(cplx);
"""


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("16 / sizeof(Quad)", "16 / 16"),
        ("16 / sizeof(cplx)", "16 / 8"),
        ("sizeof(Pair_t<float>) * 2", "8 * 2"),
        # Unknown and dependent names stay unevaluated.
        ("16 / sizeof(T)", "16 / sizeof(T)"),
        ("sizeof(Missing)", "sizeof(Missing)"),
    ],
)
def test_sizeof_operands_resolve_against_the_active_source(expression, expected):
    preprocessor = MetalPreprocessor()
    previous = preprocessor._enter_sizeof_source(SOURCE)
    try:
        assert preprocessor._replace_concrete_sizeof_expressions(expression) == expected
    finally:
        preprocessor._active_sizeof_source = previous


def test_sizeof_of_an_aggregate_needs_an_active_source():
    preprocessor = MetalPreprocessor()
    assert (
        preprocessor._replace_concrete_sizeof_expressions("16 / sizeof(Quad)")
        == "16 / sizeof(Quad)"
    )
    assert preprocessor._evaluate_static_integral_expression("16 / sizeof(float2)") == (
        True,
        2,
    )
