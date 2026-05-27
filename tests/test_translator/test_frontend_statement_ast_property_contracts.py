from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    PrimitiveType,
    ReturnNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
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
    initial_value=st.integers(min_value=0, max_value=16),
    limit=st.integers(min_value=17, max_value=64),
)
def test_generated_for_and_assignment_nodes_preserve_alias_and_parent_contracts(
    suffix,
    initial_value,
    limit,
):
    initializer = VariableNode(
        f"i_{suffix}",
        _int_type(),
        initial_value=_int(initial_value),
    )
    condition = BinaryOpNode(_id(f"i_{suffix}"), "<", _int(limit))
    update = UnaryOpNode("++", _id(f"i_{suffix}"), is_postfix=True)
    assignment = AssignmentNode(
        _id(f"sum_{suffix}"),
        BinaryOpNode(_id(f"sum_{suffix}"), "+", _id(f"i_{suffix}")),
        operator="+=",
    )
    body = BlockNode([assignment, ContinueNode()])
    loop = ForNode(initializer, condition, update, body)

    _assert_walks_once_with(
        loop,
        [
            initializer,
            initializer.var_type,
            initializer.initial_value,
            condition,
            condition.left,
            condition.right,
            update,
            update.operand,
            body,
            assignment,
            assignment.target,
            assignment.value,
            assignment.value.left,
            assignment.value.right,
            body.statements[1],
        ],
    )

    assert assignment.left is assignment.target
    assert assignment.right is assignment.value

    loop.bind_parent_links()

    assert initializer.parent is loop
    assert initializer.var_type.parent is initializer
    assert initializer.initial_value.parent is initializer
    assert condition.parent is loop
    assert condition.left.parent is condition
    assert condition.right.parent is condition
    assert update.parent is loop
    assert update.operand.parent is update
    assert body.parent is loop
    assert assignment.parent is body
    assert assignment.target.parent is assignment
    assert assignment.value.parent is assignment
    assert assignment.value.left.parent is assignment.value
    assert assignment.value.right.parent is assignment.value
    assert body.statements[1].parent is body


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    primary_value=st.integers(min_value=0, max_value=32),
    secondary_value=st.integers(min_value=33, max_value=64),
    default_value=st.integers(min_value=65, max_value=96),
)
def test_generated_if_switch_and_else_if_lists_walk_structural_children(
    suffix,
    primary_value,
    secondary_value,
    default_value,
):
    condition = BinaryOpNode(_id(f"value_{suffix}"), "==", _int(primary_value))
    then_branch = BlockNode([ReturnNode(_int(primary_value))])
    else_branch = BlockNode([ReturnNode(_int(default_value))])
    if_node = IfNode(condition, then_branch, else_branch)
    if_node.else_if_conditions = [
        BinaryOpNode(_id(f"value_{suffix}"), "==", _int(secondary_value))
    ]
    if_node.else_if_bodies = [BlockNode([ReturnNode(_int(secondary_value))])]

    switch = SwitchNode(
        _id(f"value_{suffix}"),
        [
            CaseNode(_int(primary_value), [BreakNode()]),
            CaseNode(_int(secondary_value), [ExpressionStatementNode(if_node)]),
        ],
        default_case=BlockNode([ReturnNode(_int(default_value))]),
    )

    _assert_walks_once_with(
        switch,
        [
            switch.expression,
            *switch.cases,
            switch.cases[0].value,
            switch.cases[0].statements[0],
            switch.cases[1].value,
            switch.cases[1].statements[0],
            if_node,
            condition,
            condition.left,
            condition.right,
            then_branch,
            then_branch.statements[0],
            then_branch.statements[0].value,
            else_branch,
            else_branch.statements[0],
            else_branch.statements[0].value,
            if_node.else_if_conditions[0],
            if_node.else_if_conditions[0].left,
            if_node.else_if_conditions[0].right,
            if_node.else_if_bodies[0],
            if_node.else_if_bodies[0].statements[0],
            if_node.else_if_bodies[0].statements[0].value,
            switch.default_case,
            switch.default_case.statements[0],
            switch.default_case.statements[0].value,
        ],
    )

    assert if_node.if_condition is if_node.condition
    assert if_node.if_body is if_node.then_branch
    assert if_node.else_body is if_node.else_branch

    switch.bind_parent_links()

    assert switch.expression.parent is switch
    assert switch.cases[0].parent is switch
    assert switch.cases[0].value.parent is switch.cases[0]
    assert switch.cases[0].statements[0].parent is switch.cases[0]
    assert switch.cases[1].statements[0].parent is switch.cases[1]
    assert if_node.parent is switch.cases[1].statements[0]
    assert condition.parent is if_node
    assert then_branch.parent is if_node
    assert else_branch.parent is if_node
    assert if_node.else_if_conditions[0].parent is if_node
    assert if_node.else_if_bodies[0].parent is if_node
    assert switch.default_case.parent is switch
    assert switch.default_case.statements[0].parent is switch.default_case


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    enabled=st.booleans(),
    start=st.integers(min_value=0, max_value=8),
    end=st.integers(min_value=9, max_value=32),
)
def test_generated_loop_variants_bind_iterables_conditions_and_bodies(
    suffix,
    enabled,
    start,
    end,
):
    for_in_body = BlockNode([ExpressionStatementNode(_id(f"consume_{suffix}"))])
    for_in = ForInNode(f"item_{suffix}", _id(f"items_{suffix}"), for_in_body)
    while_body = BlockNode([BreakNode()])
    while_node = WhileNode(
        BinaryOpNode(_id(f"index_{suffix}"), "<", _int(end)),
        while_body,
    )
    do_body = BlockNode([ContinueNode()])
    do_while = DoWhileNode(do_body, _bool(enabled))
    infinite_loop = LoopNode(
        BlockNode([ReturnNode(_id(f"done_{suffix}"))]),
        label=f"retry_{suffix}",
    )
    wrapper = BlockNode([for_in, while_node, do_while, infinite_loop])

    _assert_walks_once_with(
        wrapper,
        [
            for_in,
            for_in.iterable,
            for_in_body,
            for_in_body.statements[0],
            for_in_body.statements[0].expression,
            while_node,
            while_node.condition,
            while_node.condition.left,
            while_node.condition.right,
            while_body,
            while_body.statements[0],
            do_while,
            do_body,
            do_body.statements[0],
            do_while.condition,
            infinite_loop,
            infinite_loop.body,
            infinite_loop.body.statements[0],
            infinite_loop.body.statements[0].value,
        ],
    )

    assert for_in.pattern == f"item_{suffix}"
    assert infinite_loop.label == f"retry_{suffix}"

    wrapper.bind_parent_links()

    assert for_in.parent is wrapper
    assert for_in.iterable.parent is for_in
    assert for_in.body.parent is for_in
    assert for_in_body.statements[0].parent is for_in_body
    assert for_in_body.statements[0].expression.parent is for_in_body.statements[0]
    assert while_node.parent is wrapper
    assert while_node.condition.parent is while_node
    assert while_node.body.parent is while_node
    assert while_body.statements[0].parent is while_body
    assert do_while.parent is wrapper
    assert do_while.body.parent is do_while
    assert do_while.condition.parent is do_while
    assert infinite_loop.parent is wrapper
    assert infinite_loop.body.parent is infinite_loop
    assert infinite_loop.body.statements[0].parent is infinite_loop.body
    assert (
        infinite_loop.body.statements[0].value.parent
        is infinite_loop.body.statements[0]
    )
