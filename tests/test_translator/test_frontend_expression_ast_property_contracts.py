from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    BinaryOpNode,
    BuiltinVariableNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PointerAccessNode,
    PrimitiveType,
    RangeNode,
    SwizzleNode,
    TernaryOpNode,
    VectorType,
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def _int_type():
    return PrimitiveType("int")


def _uint_type():
    return PrimitiveType("uint")


def _float_type():
    return PrimitiveType("float")


def _bool_type():
    return PrimitiveType("bool")


def _id(name, expression_type=None):
    return IdentifierNode(name, expression_type=expression_type)


def _int(value):
    return LiteralNode(value, _int_type())


def _assert_walks_once_with(root, expected_nodes):
    walked = list(root.walk())

    assert len({id(node) for node in walked}) == len(walked)
    for node in expected_nodes:
        assert node in walked


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    first_value=st.integers(min_value=0, max_value=64),
    array_size=st.integers(min_value=1, max_value=16),
    index=st.integers(min_value=0, max_value=15),
)
def test_generated_access_and_array_expression_nodes_preserve_structural_aliases(
    suffix,
    first_value,
    array_size,
    index,
):
    array_type = ArrayType(_int_type(), _int(array_size))
    array_literal = ArrayLiteralNode(
        [
            _int(first_value),
            BinaryOpNode(_int(first_value + 1), "+", _int(first_value + 2)),
        ],
        expression_type=array_type,
    )
    array_access = ArrayAccessNode(
        array_literal,
        _int(index),
        expression_type=_int_type(),
    )
    member_access = MemberAccessNode(
        array_access,
        f"field_{suffix}",
        expression_type=NamedType(f"Payload_{suffix}"),
    )
    pointer_access = PointerAccessNode(
        _id(f"payloadPtr_{suffix}", expression_type=NamedType(f"Payload_{suffix}")),
        f"value_{suffix}",
        expression_type=VectorType(_float_type(), 4),
    )
    swizzle = SwizzleNode(
        pointer_access,
        "xy",
        expression_type=VectorType(_float_type(), 2),
    )
    expression = BinaryOpNode(
        member_access,
        "+",
        swizzle,
        expression_type=VectorType(_float_type(), 2),
    )

    _assert_walks_once_with(
        expression,
        [
            expression.expression_type,
            expression.expression_type.element_type,
            member_access,
            member_access.expression_type,
            array_access,
            array_access.expression_type,
            array_literal,
            array_type,
            array_type.element_type,
            array_type.size,
            *array_literal.elements,
            array_literal.elements[1].left,
            array_literal.elements[1].right,
            array_access.index_expr,
            swizzle,
            swizzle.expression_type,
            swizzle.expression_type.element_type,
            pointer_access,
            pointer_access.expression_type,
            pointer_access.expression_type.element_type,
            pointer_access.pointer_expr,
            pointer_access.pointer_expr.expression_type,
        ],
    )

    assert array_access.array is array_access.array_expr
    assert array_access.index is array_access.index_expr
    assert member_access.object is member_access.object_expr
    assert expression.vtype is expression.expression_type

    expression.bind_parent_links()

    assert expression.expression_type.parent is expression
    assert expression.expression_type.element_type.parent is expression.expression_type
    assert member_access.parent is expression
    assert member_access.expression_type.parent is member_access
    assert array_access.parent is member_access
    assert array_access.expression_type.parent is array_access
    assert array_literal.parent is array_access
    assert array_type.parent is array_literal
    assert array_type.element_type.parent is array_type
    assert array_type.size.parent is array_type
    assert all(element.parent is array_literal for element in array_literal.elements)
    assert array_literal.elements[1].left.parent is array_literal.elements[1]
    assert array_literal.elements[1].right.parent is array_literal.elements[1]
    assert array_access.index_expr.parent is array_access
    assert swizzle.parent is expression
    assert swizzle.expression_type.parent is swizzle
    assert pointer_access.parent is swizzle
    assert pointer_access.expression_type.parent is pointer_access
    assert pointer_access.pointer_expr.parent is pointer_access
    assert (
        pointer_access.pointer_expr.expression_type.parent
        is pointer_access.pointer_expr
    )


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    start=st.integers(min_value=0, max_value=8),
    end=st.integers(min_value=9, max_value=32),
    inclusive=st.booleans(),
    component=st.sampled_from(["x", "y", "z"]),
)
def test_generated_ternary_range_and_builtin_nodes_bind_operands_and_types(
    suffix,
    start,
    end,
    inclusive,
    component,
):
    builtin = BuiltinVariableNode(
        f"workgroup_id_{suffix}",
        component=component,
        expression_type=VectorType(_uint_type(), 3),
    )
    condition = BinaryOpNode(
        builtin,
        "<",
        _int(end),
        expression_type=_bool_type(),
    )
    true_range = RangeNode(
        _int(start),
        _int(end),
        inclusive=inclusive,
        expression_type=NamedType("Range"),
    )
    false_range = RangeNode(
        _int(0),
        _int(start),
        inclusive=False,
        expression_type=NamedType("Range"),
    )
    expression = TernaryOpNode(
        condition,
        true_range,
        false_range,
        expression_type=NamedType("Range"),
    )

    _assert_walks_once_with(
        expression,
        [
            expression.expression_type,
            condition,
            condition.expression_type,
            builtin,
            builtin.expression_type,
            builtin.expression_type.element_type,
            condition.right,
            true_range,
            true_range.expression_type,
            true_range.start,
            true_range.end,
            false_range,
            false_range.expression_type,
            false_range.start,
            false_range.end,
        ],
    )

    assert builtin.component == component
    assert true_range.inclusive is inclusive
    assert false_range.inclusive is False

    expression.bind_parent_links()

    assert expression.expression_type.parent is expression
    assert condition.parent is expression
    assert condition.expression_type.parent is condition
    assert builtin.parent is condition
    assert builtin.expression_type.parent is builtin
    assert builtin.expression_type.element_type.parent is builtin.expression_type
    assert condition.right.parent is condition
    assert true_range.parent is expression
    assert true_range.expression_type.parent is true_range
    assert true_range.start.parent is true_range
    assert true_range.end.parent is true_range
    assert false_range.parent is expression
    assert false_range.expression_type.parent is false_range
    assert false_range.start.parent is false_range
    assert false_range.end.parent is false_range
