from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import (
    BinaryOpNode,
    ConstructorNode,
    ConstructorPatternNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IdentifierPatternNode,
    MatchNode,
    ReturnNode,
    StructPatternNode,
    WildcardPatternNode,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def assert_identifier(node, name):
    assert isinstance(node, IdentifierNode)
    assert node.name == name


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    guard_operator=st.sampled_from((">", ">=", "!=", "==")),
)
def test_generated_match_constructor_and_struct_patterns_preserve_ir(
    suffix,
    guard_operator,
):
    code = f"""
    shader PatternContracts_{suffix} {{
        fn inspect_{suffix}(
            result_{suffix}: Result_{suffix}<int, Error_{suffix}>,
            command_{suffix}: Command_{suffix},
            threshold_{suffix}: int
        ) -> int {{
            match result_{suffix} {{
                Result_{suffix}::Ok(value_{suffix})
                    if value_{suffix} {guard_operator} threshold_{suffix}
                    => value_{suffix},
                Result_{suffix}::Err(_) => 0,
            }}

            match command_{suffix} {{
                Command_{suffix}::Draw {{
                    payload: Result_{suffix}::Ok(draw_value_{suffix}),
                    amount_{suffix}
                }} if amount_{suffix} > 0
                    => draw_value_{suffix} + amount_{suffix},
                Command_{suffix}::Skip {{ .. }} => 0,
            }}

            return 0;
        }}
    }}
    """

    ast = parse_code(code)
    result_match, command_match, return_statement = ast.functions[0].body.statements

    assert isinstance(result_match, MatchNode)
    assert_identifier(result_match.expression, f"result_{suffix}")
    assert len(result_match.arms) == 2

    ok_pattern = result_match.arms[0].pattern
    assert isinstance(ok_pattern, ConstructorPatternNode)
    assert ok_pattern.type_name == f"Result_{suffix}::Ok"
    assert len(ok_pattern.arguments) == 1
    assert isinstance(ok_pattern.arguments[0], IdentifierPatternNode)
    assert ok_pattern.arguments[0].name == f"value_{suffix}"

    ok_guard = result_match.arms[0].guard
    assert isinstance(ok_guard, BinaryOpNode)
    assert ok_guard.operator == guard_operator
    assert_identifier(ok_guard.left, f"value_{suffix}")
    assert_identifier(ok_guard.right, f"threshold_{suffix}")
    assert result_match.arms[0].body.is_tail_expression is True

    err_pattern = result_match.arms[1].pattern
    assert isinstance(err_pattern, ConstructorPatternNode)
    assert err_pattern.type_name == f"Result_{suffix}::Err"
    assert len(err_pattern.arguments) == 1
    assert isinstance(err_pattern.arguments[0], WildcardPatternNode)

    assert isinstance(command_match, MatchNode)
    assert_identifier(command_match.expression, f"command_{suffix}")
    assert len(command_match.arms) == 2

    draw_pattern = command_match.arms[0].pattern
    assert isinstance(draw_pattern, StructPatternNode)
    assert draw_pattern.type_name == f"Command_{suffix}::Draw"
    assert list(draw_pattern.field_patterns) == [
        "payload",
        f"amount_{suffix}",
    ]
    assert draw_pattern.has_rest is False

    payload_pattern = draw_pattern.field_patterns["payload"]
    assert isinstance(payload_pattern, ConstructorPatternNode)
    assert payload_pattern.type_name == f"Result_{suffix}::Ok"
    assert isinstance(payload_pattern.arguments[0], IdentifierPatternNode)
    assert payload_pattern.arguments[0].name == f"draw_value_{suffix}"

    shorthand_pattern = draw_pattern.field_patterns[f"amount_{suffix}"]
    assert isinstance(shorthand_pattern, IdentifierPatternNode)
    assert shorthand_pattern.name == f"amount_{suffix}"

    draw_guard = command_match.arms[0].guard
    assert isinstance(draw_guard, BinaryOpNode)
    assert draw_guard.operator == ">"
    assert command_match.arms[0].body.is_tail_expression is True
    assert isinstance(command_match.arms[0].body.expression, BinaryOpNode)

    skip_pattern = command_match.arms[1].pattern
    assert isinstance(skip_pattern, StructPatternNode)
    assert skip_pattern.type_name == f"Command_{suffix}::Skip"
    assert skip_pattern.field_patterns == {}
    assert skip_pattern.has_rest is True

    assert isinstance(return_statement, ReturnNode)


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    field_order=st.permutations(("color", "depth", "coverage")),
)
def test_generated_namespaced_constructor_expressions_preserve_named_arguments(
    suffix,
    field_order,
):
    field_values = {
        "color": f"color_{suffix}",
        "depth": f"depth_{suffix}",
        "coverage": f"coverage_{suffix}",
    }
    constructor_fields = ", ".join(
        (
            field_values[field_name]
            if field_name == "color"
            else f"{field_name}: {field_values[field_name]}"
        )
        for field_name in field_order
    )
    code = f"""
    shader ConstructorContracts_{suffix} {{
        fn build_{suffix}(
            color_{suffix}: vec4,
            depth_{suffix}: float,
            coverage_{suffix}: float
        ) -> int {{
            Output_{suffix} out_{suffix} = RenderOutput_{suffix}::Clear {{
                {constructor_fields}
            }};
            return 0;
        }}
    }}
    """

    ast = parse_code(code)
    constructor = ast.functions[0].body.statements[0].initial_value
    expected_fields = [
        field_values[field_name] if field_name == "color" else field_name
        for field_name in field_order
    ]

    assert isinstance(constructor, ConstructorNode)
    assert constructor.constructor_type.name == f"RenderOutput_{suffix}::Clear"
    assert constructor.arguments == []
    assert list(constructor.named_arguments) == expected_fields
    for field_name, expected_field in zip(field_order, expected_fields):
        assert_identifier(
            constructor.named_arguments[expected_field],
            field_values[field_name],
        )


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    value=st.integers(min_value=0, max_value=31),
)
def test_generated_match_block_tail_expressions_preserve_semicolon_contract(
    suffix,
    value,
):
    code = f"""
    shader MatchTailContracts_{suffix} {{
        fn convert_{suffix}(op_{suffix}: VectorOp_{suffix})
            -> Result_{suffix}<int, Error_{suffix}> {{
            match op_{suffix} {{
                VectorOp_{suffix}::Cross => {{
                    Result_{suffix}::Ok({value})
                }},
                _ => Result_{suffix}::Err(Error_{suffix}::InvalidInput),
            }}
        }}
    }}
    """

    ast = parse_code(code)
    match_statement = ast.functions[0].body.statements[0]
    assert isinstance(match_statement, MatchNode)

    cross_body = match_statement.arms[0].body
    cross_tail = cross_body.statements[-1]
    assert isinstance(cross_tail, ExpressionStatementNode)
    assert cross_tail.is_tail_expression is True
    assert isinstance(cross_tail.expression, FunctionCallNode)
    assert cross_tail.expression.function.name == f"Result_{suffix}::Ok"
    assert cross_tail.expression.arguments[0].value == value

    fallback_pattern = match_statement.arms[1].pattern
    assert isinstance(fallback_pattern, WildcardPatternNode)
    fallback_body = match_statement.arms[1].body
    assert fallback_body.is_tail_expression is True
    assert isinstance(fallback_body.expression, FunctionCallNode)
    assert fallback_body.expression.function.name == f"Result_{suffix}::Err"
