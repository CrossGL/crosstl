"""Checked scalar facts preserve observable control flow."""

import pytest

from crosstl.translator.ast import (
    BinaryOpNode,
    BlockNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    ReturnNode,
    UnaryOpNode,
    VariableNode,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_metadata_index_bounds import (
    CONDITIONS,
    SOURCE,
    _index_config,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_project_metadata_entry_scope import _packages
from tests.test_translator.test_scalar_return_bounds import _generator


def _loop(kind="int", count=0, start=0, operator="<", qualifiers=()):
    bound = VariableNode(
        "limit", kind, LiteralNode(count, kind), qualifiers=list(qualifiers)
    )
    init = VariableNode("i", "int", LiteralNode(start, "int"))
    loop = ForNode(
        init,
        BinaryOpNode(IdentifierNode("i"), operator, IdentifierNode("limit")),
        UnaryOpNode("++", IdentifierNode("i")),
        BlockNode([]),
    )
    function = FunctionNode("run", "void", [], BlockNode([bound, loop]))
    return function, loop


@pytest.mark.parametrize(
    "start,count,operator,expected",
    [
        (0, 0, "<", True),
        (0, 1, "<", False),
        (1, 0, "<=", True),
        (0, 0, "<=", False),
        (0, 1, ">", True),
        (0, 0, ">=", False),
        (0, 1, "==", True),
        (0, 0, "!=", True),
        (-1, 0, ">", True),
    ],
)
def test_exact_signed_bound_controls(start, count, operator, expected):
    generator = GLSLCodeGen()
    function, loop = _loop(count=count, start=start, operator=operator)
    generator.annotate_glsl_control_flow_intervals(function)
    assert loop._glsl_checked_zero_trip is expected


@pytest.mark.parametrize(
    "kind,qualifiers",
    [("uint", ()), ("uint64_t", ()), ("int16_t", ()), ("int", ("volatile",))],
)
def test_bound_type_or_volatile_does_not_establish_reachability(kind, qualifiers):
    function, loop = _loop(kind=kind, qualifiers=qualifiers, start=-1, operator=">")
    generator = GLSLCodeGen()
    generator.annotate_glsl_control_flow_intervals(function)
    assert not loop._glsl_checked_zero_trip


@pytest.mark.parametrize(
    "which",
    [
        "initializer",
        "condition",
        "cast-condition",
        "unsigned",
        "overflow",
        "shadow-init",
        "volatile-init",
        "assignment-init",
    ],
)
def test_unmodeled_evaluation_stays_reachable(which):
    function, loop = _loop()
    call = FunctionCallNode(IdentifierNode("touch"), [])
    if which == "initializer":
        loop.init.initial_value = call
    elif which == "condition":
        loop.condition.right = UnaryOpNode(
            "++", IdentifierNode("limit"), is_postfix=True
        )
    elif which == "cast-condition":
        loop.condition.right = FunctionCallNode(IdentifierNode("int"), [call])
    elif which == "unsigned":
        loop.condition.right = LiteralNode(0, "uint")
        loop.init.initial_value = LiteralNode(-1, "int")
        loop.condition.op = loop.condition.operator = ">"
    elif which == "overflow":
        loop.condition.right = BinaryOpNode(
            LiteralNode(2147483647, "int"), "+", LiteralNode(1, "int")
        )
    elif which == "shadow-init":
        function.body.statements.insert(
            0, VariableNode("i", "int", LiteralNode(0, "int"))
        )
        loop.init.initial_value = IdentifierNode("i")
    elif which == "volatile-init":
        loop.init.qualifiers = ["volatile"]
    else:
        from crosstl.translator.ast import AssignmentNode

        loop.init = AssignmentNode(IdentifierNode("limit"), LiteralNode(0, "int"))
    generator = GLSLCodeGen()
    generator.annotate_glsl_control_flow_intervals(function)
    assert not loop._glsl_checked_zero_trip


@pytest.mark.parametrize(
    "body",
    [
        "for (int i=0; i<0; ++i) { result[0] = values[ulong(i)]; } result[1]=7;",
        "int limit=int(offset % 4ul); for (int i=0; i<limit; ++i) { result[0] = values[ulong(i)]; } result[1]=7;",
        "int limit=int(offset % 4ul); if (limit>0) { for (int i=0; i<limit; ++i) { result[0]=values[ulong(i)]; } } result[1]=7;",
    ],
)
def test_public_checked_zero_trip_compiles(tmp_path, body):
    source = SOURCE.replace(
        "result[0] = values[offset + index];", "values += index; " + body
    )
    conditions = (CONDITIONS[0], {**CONDITIONS[1], "minimum": 16, "maximum": 16})
    _, package, (descriptor,) = _packages(
        _index_config(tmp_path, source=source, conditions=conditions)
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert "for (" not in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "kind,qualifiers,known",
    [
        ("int", (), True),
        ("uint", (), False),
        ("int64_t", (), False),
        ("int", ("volatile",), False),
    ],
)
def test_checked_branch_requires_exact_signed_lexical_type(kind, qualifiers, known):
    value = VariableNode(
        "limit", kind, LiteralNode(0, kind), qualifiers=list(qualifiers)
    )
    statement = IfNode(
        BinaryOpNode(IdentifierNode("limit"), ">", LiteralNode(0, "int")),
        BlockNode([]),
        BlockNode([]),
    )
    function = FunctionNode("run", "void", [], BlockNode([value, statement]))
    generator = GLSLCodeGen()
    generator.annotate_glsl_control_flow_intervals(function)
    assert statement._glsl_checked_static_if[0] is known
    if known:
        assert statement._glsl_checked_static_if[1] is statement.else_body


def test_reannotation_discards_previous_loop_and_branch_decisions():
    function, loop = _loop()
    statement = IfNode(
        BinaryOpNode(IdentifierNode("limit"), ">", LiteralNode(0, "int")),
        BlockNode([]),
        BlockNode([]),
    )
    function.body.statements.insert(1, statement)
    generator = GLSLCodeGen()
    generator.annotate_glsl_control_flow_intervals(function)
    assert loop._glsl_checked_zero_trip
    assert statement._glsl_checked_static_if == (True, statement.else_body)
    function.body.statements[0].initial_value = LiteralNode(1, "int")
    generator.annotate_glsl_control_flow_intervals(function)
    assert not loop._glsl_checked_zero_trip
    assert statement._glsl_checked_static_if == (True, statement.if_body)


@pytest.mark.parametrize(
    "body,facts,unreachable",
    [
        ("if (flag == 0) return;", {"flag": (0, 0)}, True),
        ("if (flag == 0) return;", {"flag": (1, 1)}, False),
        ("if (flag == 0) return;", {"flag": (0, 1)}, False),
        ("if (flag == 0) return;", {}, False),
        ("if (flag == 0) { return; } else { return; }", {}, True),
        ("{ if (flag == 0) return; }", {"flag": (0, 0)}, True),
        ("if (flag == 0) return; else if (flag == 1) return;", {"flag": (0, 1)}, True),
        ("if (flag == 0) return; else if (flag == 1) return;", {"flag": (0, 2)}, False),
        ("unknown(&flag); if (flag == 0) return;", {"flag": (0, 0)}, False),
        ("if (++flag == 0) return;", {"flag": (0, 0)}, False),
        ("for (int i = 0; i < flag; ++i) { return; }", {"flag": (0, 1)}, False),
    ],
)
def test_only_proven_return_paths_remove_the_following_statement(
    body, facts, unreachable
):
    generator = _generator(f"void run(int flag) {{ {body} marker(); }}")
    function = generator.function_definitions["run"]
    generator.glsl_workgroup_proof_initial_intervals = lambda _function: dict(facts)
    generator.annotate_glsl_control_flow_intervals(function)
    marker = function.body.statements[-1]
    assert getattr(marker, "_glsl_checked_unreachable", False) is unreachable


def test_return_reachability_is_recomputed_for_each_caller_context():
    generator = _generator("void run(int flag) { if (flag == 0) return; marker(); }")
    function = generator.function_definitions["run"]
    for flag in (0, 1, 0, 1):
        generator.glsl_workgroup_proof_initial_intervals = lambda _function: {
            "flag": (flag, flag)
        }
        generator.annotate_glsl_control_flow_intervals(function)
        assert function.body.statements[-1]._glsl_checked_unreachable is (flag == 0)


def test_a_shared_statement_reached_by_another_branch_is_not_removed():
    marker = FunctionCallNode(IdentifierNode("marker"), [])
    branch = IfNode(
        IdentifierNode("unknown"),
        BlockNode([ReturnNode(), marker]),
        BlockNode([marker]),
    )
    function = FunctionNode("run", "void", [], BlockNode([branch]))
    generator = GLSLCodeGen()
    generator.annotate_glsl_control_flow_intervals(function)
    assert not marker._glsl_checked_unreachable


def test_replayed_inner_loop_is_not_removed_when_later_iterations_execute_it():
    generator = _generator("""
        void run(Cursor cursor) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < i; ++j) { cursor.offset += 1; }
            }
            cursor.offset;
        }
    """)
    generator.struct_member_types["Cursor"] = {"offset": "int64"}
    function = generator.function_definitions["run"]
    generator.glsl_workgroup_proof_initial_intervals = lambda _function: {
        "cursor.offset": (0, 0)
    }
    generator.annotate_glsl_control_flow_intervals(function)
    loops = [
        node for node in generator.walk_ast(function.body) if isinstance(node, ForNode)
    ]
    assert len(loops) == 2
    assert all(not node._glsl_checked_zero_trip for node in loops)
    low, high = function.body.statements[-1]._glsl_control_flow_intervals[
        "cursor.offset"
    ]
    assert 0 <= low <= 3 == high


@pytest.mark.parametrize("location", ["condition", "cast", "later-else-if"])
def test_branch_does_not_drop_observable_condition_evaluation(location):
    limit = VariableNode("limit", "int", LiteralNode(0, "int"))
    effect = UnaryOpNode("++", IdentifierNode("limit"), is_postfix=True)
    if location == "cast":
        effect = FunctionCallNode(IdentifierNode("int"), [effect])
    statement = IfNode(
        BinaryOpNode(effect, ">", LiteralNode(0, "int")), BlockNode([]), BlockNode([])
    )
    if location == "later-else-if":
        statement.else_if_conditions = [statement.condition]
        statement.else_if_bodies = [BlockNode([])]
        statement.condition = BinaryOpNode(
            IdentifierNode("limit"), ">", LiteralNode(0, "int")
        )
    generator = GLSLCodeGen()
    generator.annotate_glsl_control_flow_intervals(
        FunctionNode("run", "void", [], BlockNode([limit, statement]))
    )
    assert not statement._glsl_checked_static_if[0]


NATIVE_BODIES = (
    "for (int i=0; i<0; ++i) { result[0]=values[ulong(i)]; } result[1]=7;",
    "int limit=int(offset%4ul); for (int i=0; i<limit; ++i) { result[0]=values[ulong(i)]; } result[1]=7;",
    "int limit=int(offset%3ul); for (int i=0; i<limit; ++i) { result[0]=values[ulong(i)]; } result[1]=7;",
    "int limit=int(offset%4ul); if (limit>0) { for (int i=0; i<limit; ++i) { result[0]=values[ulong(i)]; } } else { result[3]=11; } result[1]=7;",
    "int limit=int(offset%4ul); int marker=11; if (limit==0) { int marker=13; result[3]=marker; } else { result[3]=19; } result[4]=marker; result[1]=7;",
    "int calls=0; for (int i=(++calls); i<0; ++i) { result[0]=values[0]; } result[3]=calls; result[1]=7;",
    "int calls=0; for (int i=0; i<calls++; ++i) { result[0]=values[0]; } result[3]=calls; result[1]=7;",
)
