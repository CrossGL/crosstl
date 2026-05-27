from dataclasses import dataclass

from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    ArrayAccessNode,
    BinaryOpNode,
    BlockNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    IdentifierNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


@dataclass(frozen=True)
class PrecedenceCase:
    lower_operator: str
    higher_operator: str


PRECEDENCE_CASES = (
    PrecedenceCase("+", "*"),
    PrecedenceCase("-", "%"),
    PrecedenceCase("<<", "+"),
    PrecedenceCase("&", "=="),
    PrecedenceCase("^", "&"),
    PrecedenceCase("|", "^"),
    PrecedenceCase("&&", "=="),
    PrecedenceCase("||", "&&"),
)

ASSIGNMENT_OPERATORS = ("=", "+=", "-=", "*=", "/=", "%=", "^=", "|=", "&=")


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def assert_identifier(node, name):
    assert isinstance(node, IdentifierNode)
    assert node.name == name


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    precedence_case=st.sampled_from(PRECEDENCE_CASES),
)
def test_generated_binary_operator_precedence_preserves_tree_shape(
    suffix,
    precedence_case,
):
    code = f"""
    shader ExpressionPrecedence_{suffix} {{
        int combine_{suffix}(int a_{suffix}, int b_{suffix}, int c_{suffix}) {{
            int result_{suffix} =
                a_{suffix} {precedence_case.lower_operator}
                b_{suffix} {precedence_case.higher_operator}
                c_{suffix};
            return result_{suffix};
        }}
    }}
    """

    ast = parse_code(code)
    initial_value = ast.functions[0].body.statements[0].initial_value

    assert isinstance(initial_value, BinaryOpNode)
    assert initial_value.operator == precedence_case.lower_operator
    assert_identifier(initial_value.left, f"a_{suffix}")

    assert isinstance(initial_value.right, BinaryOpNode)
    assert initial_value.right.operator == precedence_case.higher_operator
    assert_identifier(initial_value.right.left, f"b_{suffix}")
    assert_identifier(initial_value.right.right, f"c_{suffix}")


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    assignment_operator=st.sampled_from(ASSIGNMENT_OPERATORS),
    postfix_operator=st.sampled_from(("++", "--")),
)
def test_generated_ternary_assignment_and_postfix_chains_preserve_ir(
    suffix,
    assignment_operator,
    postfix_operator,
):
    code = f"""
    shader ExpressionChains_{suffix} {{
        int mutate_{suffix}(
            int index_{suffix},
            int slot_{suffix},
            bool choose_{suffix},
            int fallback_{suffix}
        ) {{
            int values_{suffix}[4];
            int payload_{suffix} = choose_{suffix}
                ? table_{suffix}[index_{suffix}].inner.value
                : fallback_{suffix};
            values_{suffix}[slot_{suffix}] {assignment_operator} payload_{suffix};
            table_{suffix}[index_{suffix}].inner.count{postfix_operator};
            return values_{suffix}[slot_{suffix}];
        }}
    }}
    """

    ast = parse_code(code)
    statements = ast.functions[0].body.statements

    payload_initializer = statements[1].initial_value
    assert isinstance(payload_initializer, TernaryOpNode)
    assert_identifier(payload_initializer.condition, f"choose_{suffix}")
    assert_identifier(payload_initializer.false_expr, f"fallback_{suffix}")

    true_value = payload_initializer.true_expr
    assert isinstance(true_value, MemberAccessNode)
    assert true_value.member == "value"
    inner_access = true_value.object_expr
    assert isinstance(inner_access, MemberAccessNode)
    assert inner_access.member == "inner"
    indexed_table = inner_access.object_expr
    assert isinstance(indexed_table, ArrayAccessNode)
    assert_identifier(indexed_table.array_expr, f"table_{suffix}")
    assert_identifier(indexed_table.index_expr, f"index_{suffix}")

    assignment_statement = statements[2]
    assert isinstance(assignment_statement, ExpressionStatementNode)
    assignment = assignment_statement.expression
    assert assignment.operator == assignment_operator
    assert isinstance(assignment.target, ArrayAccessNode)
    assert_identifier(assignment.target.array_expr, f"values_{suffix}")
    assert_identifier(assignment.target.index_expr, f"slot_{suffix}")
    assert_identifier(assignment.value, f"payload_{suffix}")

    postfix_statement = statements[3]
    assert isinstance(postfix_statement, ExpressionStatementNode)
    postfix = postfix_statement.expression
    assert isinstance(postfix, UnaryOpNode)
    assert postfix.operator == postfix_operator
    assert postfix.is_postfix is True
    assert isinstance(postfix.operand, MemberAccessNode)
    assert postfix.operand.member == "count"

    return_statement = statements[4]
    assert isinstance(return_statement, ReturnNode)
    assert isinstance(return_statement.value, ArrayAccessNode)


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    for_update_operator=st.sampled_from(("+=", "-=", "^=", "<<=", ">>=")),
    branch_operator=st.sampled_from(("+=", "-=", "|=", "&=")),
)
def test_generated_control_flow_statements_preserve_structural_ir(
    suffix,
    for_update_operator,
    branch_operator,
):
    code = f"""
    shader ControlFlow_{suffix} {{
        int scan_{suffix}(int limit_{suffix}, int stride_{suffix}) {{
            int total_{suffix} = 0;
            for (
                int i_{suffix} = 0;
                i_{suffix} < limit_{suffix};
                i_{suffix} {for_update_operator} stride_{suffix}
            ) {{
                if (((i_{suffix} & 1) == 0)) {{
                    total_{suffix} {branch_operator} i_{suffix};
                }} else {{
                    total_{suffix} -= stride_{suffix};
                }}
            }}
            while (total_{suffix} < limit_{suffix}) {{
                total_{suffix}++;
            }}
            do {{
                total_{suffix}--;
            }} while (total_{suffix} > 0);
            switch (total_{suffix}) {{
                case 0:
                    total_{suffix} = stride_{suffix};
                    break;
                case 1:
                    total_{suffix} += stride_{suffix};
                    break;
                default:
                    total_{suffix} = limit_{suffix};
            }}
            return total_{suffix};
        }}
    }}
    """

    ast = parse_code(code)
    statements = ast.functions[0].body.statements

    assert isinstance(statements[0], VariableNode)

    for_statement = statements[1]
    assert isinstance(for_statement, ForNode)
    assert isinstance(for_statement.init, VariableNode)
    assert for_statement.init.name == f"i_{suffix}"
    assert isinstance(for_statement.condition, BinaryOpNode)
    assert for_statement.condition.operator == "<"
    assert for_statement.update.operator == for_update_operator
    assert isinstance(for_statement.body, BlockNode)

    if_statement = for_statement.body.statements[0]
    assert isinstance(if_statement, IfNode)
    assert isinstance(if_statement.condition, BinaryOpNode)
    assert if_statement.condition.operator == "=="
    assert isinstance(if_statement.then_branch, BlockNode)
    assert isinstance(if_statement.else_branch, BlockNode)
    then_assignment = if_statement.then_branch.statements[0].expression
    assert then_assignment.operator == branch_operator

    while_statement = statements[2]
    assert isinstance(while_statement, WhileNode)
    assert isinstance(while_statement.condition, BinaryOpNode)
    assert while_statement.condition.operator == "<"

    do_while_statement = statements[3]
    assert isinstance(do_while_statement, DoWhileNode)
    assert isinstance(do_while_statement.condition, BinaryOpNode)
    assert do_while_statement.condition.operator == ">"

    switch_statement = statements[4]
    assert isinstance(switch_statement, SwitchNode)
    assert_identifier(switch_statement.expression, f"total_{suffix}")
    assert len(switch_statement.cases) == 3
    assert switch_statement.cases[0].value.value == 0
    assert switch_statement.cases[1].value.value == 1
    assert switch_statement.cases[2].value is None

    assert isinstance(statements[5], ReturnNode)
