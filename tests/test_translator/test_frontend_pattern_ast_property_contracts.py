from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    BinaryOpNode,
    BlockNode,
    ConstructorPatternNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IdentifierPatternNode,
    LiteralNode,
    LiteralPatternNode,
    MatchArmNode,
    MatchNode,
    NamedType,
    PrimitiveType,
    ReturnNode,
    StructPatternNode,
    WildcardPatternNode,
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def _int_type():
    return PrimitiveType("int")


def _bool_type():
    return PrimitiveType("bool")


def _id(name):
    return IdentifierNode(name)


def _int(value):
    return LiteralNode(value, _int_type())


def _bool(value):
    return LiteralNode(bool(value), _bool_type())


def _assert_walks_once_with(root, expected_nodes):
    walked = list(root.walk())

    assert len({id(node) for node in walked}) == len(walked)
    for node in expected_nodes:
        assert node in walked


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    literal_value=st.integers(min_value=0, max_value=64),
    guard_threshold=st.integers(min_value=65, max_value=128),
    enabled=st.booleans(),
)
def test_generated_match_patterns_walk_and_bind_all_pattern_children(
    suffix,
    literal_value,
    guard_threshold,
    enabled,
):
    identifier_pattern = IdentifierPatternNode(f"value_{suffix}")
    literal_pattern = LiteralPatternNode(_int(literal_value))
    wildcard_pattern = WildcardPatternNode()
    constructor_pattern = ConstructorPatternNode(
        f"Result_{suffix}::Ok",
        [identifier_pattern, literal_pattern, wildcard_pattern],
    )
    flag_pattern = LiteralPatternNode(_bool(enabled))
    struct_pattern = StructPatternNode(
        f"Command_{suffix}::Dispatch",
        {
            "payload": constructor_pattern,
            "enabled": flag_pattern,
        },
        has_rest=True,
    )
    guard = BinaryOpNode(
        _id(f"value_{suffix}"),
        ">",
        _int(guard_threshold),
        expression_type=_bool_type(),
    )
    result_call = FunctionCallNode(
        _id(f"emit_{suffix}"),
        [_id(f"value_{suffix}")],
        generic_args=[NamedType(f"Payload_{suffix}")],
    )
    arm = MatchArmNode(
        struct_pattern,
        guard,
        ExpressionStatementNode(result_call, is_tail_expression=True),
    )
    match = MatchNode(_id(f"command_{suffix}"), [arm])

    _assert_walks_once_with(
        match,
        [
            match.expression,
            arm,
            struct_pattern,
            constructor_pattern,
            identifier_pattern,
            literal_pattern,
            literal_pattern.literal,
            wildcard_pattern,
            flag_pattern,
            flag_pattern.literal,
            guard,
            guard.expression_type,
            guard.left,
            guard.right,
            arm.body,
            result_call,
            result_call.function,
            result_call.arguments[0],
            result_call.generic_args[0],
        ],
    )

    assert struct_pattern.has_rest is True
    assert list(struct_pattern.field_patterns) == ["payload", "enabled"]
    assert arm.body.is_tail_expression is True

    match.bind_parent_links()

    assert match.expression.parent is match
    assert arm.parent is match
    assert struct_pattern.parent is arm
    assert constructor_pattern.parent is struct_pattern
    assert identifier_pattern.parent is constructor_pattern
    assert literal_pattern.parent is constructor_pattern
    assert literal_pattern.literal.parent is literal_pattern
    assert wildcard_pattern.parent is constructor_pattern
    assert flag_pattern.parent is struct_pattern
    assert flag_pattern.literal.parent is flag_pattern
    assert guard.parent is arm
    assert guard.expression_type.parent is guard
    assert guard.left.parent is guard
    assert guard.right.parent is guard
    assert arm.body.parent is arm
    assert result_call.parent is arm.body
    assert result_call.function.parent is result_call
    assert result_call.arguments[0].parent is result_call
    assert result_call.generic_args[0].parent is result_call


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    return_value=st.integers(min_value=0, max_value=128),
)
def test_generated_match_arm_block_bodies_bind_statement_children(
    suffix,
    return_value,
):
    arm = MatchArmNode(
        IdentifierPatternNode(f"item_{suffix}"),
        None,
        BlockNode([ReturnNode(_int(return_value))]),
    )
    match = MatchNode(_id(f"input_{suffix}"), [arm])

    _assert_walks_once_with(
        match,
        [
            match.expression,
            arm,
            arm.pattern,
            arm.body,
            arm.body.statements[0],
            arm.body.statements[0].value,
            arm.body.statements[0].value.literal_type,
        ],
    )

    assert arm.guard is None

    match.bind_parent_links()

    assert arm.parent is match
    assert arm.pattern.parent is arm
    assert arm.body.parent is arm
    assert arm.body.statements[0].parent is arm.body
    assert arm.body.statements[0].value.parent is arm.body.statements[0]
    assert arm.body.statements[0].value.literal_type.parent is (
        arm.body.statements[0].value
    )
