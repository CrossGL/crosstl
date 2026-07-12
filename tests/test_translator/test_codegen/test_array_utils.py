import pytest

from crosstl.translator.ast import (
    BinaryOpNode,
    ConstantNode,
    ConstructorNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    TernaryOpNode,
    UnaryOpNode,
)
from crosstl.translator.codegen.array_utils import (
    collect_literal_int_constants,
    evaluate_literal_int_expression,
)

INT = PrimitiveType("int")


def int_literal(value):
    return LiteralNode(value, INT)


def uint_literal(value):
    return LiteralNode(value, PrimitiveType("uint"))


@pytest.mark.parametrize(
    ("operator", "left", "right", "expected"),
    [
        ("%", -5, 2, -1),
        ("<<", 3, 2, 12),
        (">>", 12, 2, 3),
        ("&", 6, 3, 2),
        ("|", 4, 1, 5),
        ("^", 7, 3, 4),
        ("==", 4, 4, 1),
        ("!=", 4, 5, 1),
        ("<", 4, 5, 1),
        ("<=", 5, 5, 1),
        (">", 5, 4, 1),
        (">=", 5, 5, 1),
        ("&&", 2, 3, 1),
        ("||", 0, 3, 1),
    ],
)
def test_evaluate_literal_integer_binary_expression(
    operator,
    left,
    right,
    expected,
):
    expression = BinaryOpNode(
        int_literal(left),
        operator,
        int_literal(right),
    )

    assert evaluate_literal_int_expression(expression) == expected


@pytest.mark.parametrize(
    ("operator", "operand", "expected"), [("~", 3, -4), ("!", 0, 1)]
)
def test_evaluate_literal_integer_unary_expression(operator, operand, expected):
    expression = UnaryOpNode(operator, int_literal(operand))

    assert evaluate_literal_int_expression(expression) == expected


def test_evaluate_literal_integer_conditional_short_circuits_unselected_branch():
    expression = TernaryOpNode(
        BinaryOpNode(int_literal(3), ">", int_literal(1)),
        int_literal(54),
        BinaryOpNode(int_literal(1), "/", int_literal(0)),
    )

    assert evaluate_literal_int_expression(expression) == 54


@pytest.mark.parametrize(
    ("operator", "left", "expected"),
    [("&&", 0, 0), ("||", 1, 1)],
)
def test_evaluate_literal_integer_logical_expression_short_circuits(
    operator,
    left,
    expected,
):
    expression = BinaryOpNode(
        int_literal(left),
        operator,
        BinaryOpNode(int_literal(1), "/", int_literal(0)),
    )

    assert evaluate_literal_int_expression(expression) == expected


def test_evaluate_literal_integer_conditional_accepts_boolean_literal():
    expression = TernaryOpNode(
        LiteralNode(True, PrimitiveType("bool")),
        int_literal(4),
        int_literal(5),
    )

    assert evaluate_literal_int_expression(expression) == 4


def test_evaluate_literal_integer_unsigned_arithmetic_is_not_folded_as_signed():
    unsigned_one = ConstructorNode(PrimitiveType("uint"), [int_literal(1)])
    expression = TernaryOpNode(
        BinaryOpNode(
            BinaryOpNode(unsigned_one, "-", int_literal(2)),
            "<",
            int_literal(0),
        ),
        int_literal(4),
        int_literal(5),
    )

    assert evaluate_literal_int_expression(expression) is None


def test_evaluate_typed_unsigned_literal_arithmetic_is_not_folded_as_signed():
    expression = TernaryOpNode(
        BinaryOpNode(
            BinaryOpNode(uint_literal(1), "-", int_literal(2)),
            "<",
            int_literal(0),
        ),
        int_literal(4),
        int_literal(5),
    )

    assert evaluate_literal_int_expression(expression) is None


def test_collect_literal_integer_constants_preserves_unsigned_provenance():
    constants = [
        ConstantNode("SIGNED_N", PrimitiveType("int"), int_literal(2)),
        ConstantNode("UNSIGNED_N", PrimitiveType("uint"), int_literal(1)),
    ]

    resolved = collect_literal_int_constants(constants)
    unsigned_identifier = IdentifierNode("UNSIGNED_N")

    assert resolved == {"SIGNED_N": 2, "UNSIGNED_N": 1}
    assert evaluate_literal_int_expression(unsigned_identifier, resolved) == 1
    assert (
        evaluate_literal_int_expression(
            BinaryOpNode(unsigned_identifier, "-", int_literal(2)),
            resolved,
        )
        is None
    )
