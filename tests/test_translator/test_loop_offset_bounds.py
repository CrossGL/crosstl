"""Checked loop-carried offsets retain finite source bounds."""

import pytest

from crosstl.project import translate_project
from crosstl.translator.ast import (
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ConstructorNode,
    ForNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
    UnaryOpNode,
    VariableNode,
    VectorType,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_metadata_index_bounds import SOURCE, _index_config


def _integer(value, kind="int"):
    return LiteralNode(value, PrimitiveType(kind))


@pytest.mark.parametrize(
    "kind", ["int", "uint", "int64_t", "uint64_t", "int16_t", "uint16_t"]
)
@pytest.mark.parametrize(
    "operator,step",
    [("++", 1), ("--", -1), ("PRE_INCREMENT", 1), ("POST_DECREMENT", -1)],
)
def test_checked_increment_retains_lexical_bounds(kind, operator, step):
    observed = IdentifierNode("count")
    function = FunctionNode(
        "run",
        "void",
        [],
        BlockNode(
            [
                VariableNode("count", kind, _integer(3)),
                UnaryOpNode(operator, IdentifierNode("count")),
                observed,
            ]
        ),
    )
    GLSLCodeGen().annotate_glsl_control_flow_intervals(function)
    assert observed._glsl_control_flow_intervals["count"] == (3 + step, 3 + step)


@pytest.mark.parametrize(
    "kind,value,operator",
    [
        ("uint", 0, "--"),
        ("uint", 4294967295, "++"),
        ("int", 2147483647, "++"),
        ("int", -2147483648, "--"),
        ("int64_t", 9223372036854775807, "++"),
        ("uint64_t", 18446744073709551615, "++"),
    ],
)
def test_increment_does_not_prove_wrapping_arithmetic(kind, value, operator):
    observed = IdentifierNode("count")
    function = FunctionNode(
        "run",
        "void",
        [],
        BlockNode(
            [
                VariableNode("count", kind, _integer(value, kind)),
                UnaryOpNode(operator, IdentifierNode("count")),
                observed,
            ]
        ),
    )
    GLSLCodeGen().annotate_glsl_control_flow_intervals(function)
    assert "count" not in observed._glsl_control_flow_intervals


@pytest.mark.parametrize(
    "kind,qualifiers",
    [
        ("int&", []),
        ("int*", []),
        ("int", ["volatile"]),
        ("volatile int", []),
        ("int", ["inout"]),
    ],
)
def test_increment_does_not_retain_alias_or_volatile_bounds(kind, qualifiers):
    observed = IdentifierNode("count")
    function = FunctionNode(
        "run",
        "void",
        [],
        BlockNode(
            [
                VariableNode("count", kind, _integer(3), qualifiers=qualifiers),
                UnaryOpNode("++", IdentifierNode("count")),
                observed,
            ]
        ),
    )
    GLSLCodeGen().annotate_glsl_control_flow_intervals(function)
    assert "count" not in observed._glsl_control_flow_intervals


def _offset_loop(
    *, start=0, limit=3, kind="int64_t", step=2, position_type="int64_t", qualifiers=()
):
    inside, after = IdentifierNode("position"), IdentifierNode("position")
    loop = ForNode(
        VariableNode("i", kind, _integer(start)),
        BinaryOpNode(IdentifierNode("i"), "<", _integer(limit)),
        UnaryOpNode("++", IdentifierNode("i")),
        BlockNode(
            [
                inside,
                AssignmentNode(IdentifierNode("position"), _integer(step), "+="),
            ]
        ),
    )
    function = FunctionNode(
        "run",
        "void",
        [],
        BlockNode(
            [
                VariableNode(
                    "position", position_type, _integer(4), qualifiers=list(qualifiers)
                ),
                loop,
                after,
            ]
        ),
    )
    return function, loop, inside, after


@pytest.mark.parametrize("kind", ["int", "uint", "int64_t", "uint64_t"])
@pytest.mark.parametrize(
    "step,inside_bounds,after_bounds",
    [
        (2, (4, 8), (4, 10)),
        (0, (4, 4), (4, 4)),
        (-1, (2, 4), (1, 4)),
    ],
)
def test_additive_loop_intervals_cover_entry_and_exit(
    kind, step, inside_bounds, after_bounds
):
    function, _, inside, after = _offset_loop(kind=kind, step=step)
    GLSLCodeGen().annotate_glsl_control_flow_intervals(function)
    assert inside._glsl_control_flow_intervals["position"] == inside_bounds
    assert after._glsl_control_flow_intervals["position"] == after_bounds


def _component_loop(step=2):
    function, loop, inside, after = _offset_loop(step=step)
    vector_type = VectorType(PrimitiveType("int"), 2)
    position = function.body.statements[0]
    position.var_type = vector_type
    position.initial_value = ConstructorNode(vector_type, [_integer(4), _integer(9)])
    loop.body.statements[-1].target = MemberAccessNode(IdentifierNode("position"), "x")
    loop.condition = BinaryOpNode(
        loop.condition, "&&", BinaryOpNode(IdentifierNode("i"), "<", _integer(2))
    )
    return function, loop, inside, after


@pytest.mark.parametrize(
    "step,bounds,exit_bounds", [(2, (4, 6), (4, 10)), (-1, (3, 4), (1, 4))]
)
def test_conjunction_bounded_component_updates(step, bounds, exit_bounds):
    function, loop, inside, after = _component_loop(step)
    loop.body.statements.append(
        AssignmentNode(
            MemberAccessNode(IdentifierNode("position"), "y"), _integer(1), "+="
        )
    )
    GLSLCodeGen().annotate_glsl_control_flow_intervals(function)
    assert inside._glsl_control_flow_intervals["position.x"] == bounds
    assert after._glsl_control_flow_intervals["position.x"] == exit_bounds
    assert inside._glsl_control_flow_intervals["position.y"] == (9, 10)
    assert after._glsl_control_flow_intervals["position.y"] == (9, 12)


@pytest.mark.parametrize(
    "fault",
    [
        "whole-write",
        "swizzle-write",
        "duplicate-update",
        "escape",
        "shadow",
        "volatile",
        "overflow",
        "condition-write",
        "disjunction",
    ],
)
def test_component_loop_proof_rejects_unsafe_updates(fault):
    function, loop, inside, after = _component_loop()
    target = lambda: MemberAccessNode(IdentifierNode("position"), "x")
    if fault == "whole-write":
        loop.body.statements.insert(
            1, AssignmentNode(IdentifierNode("position"), IdentifierNode("unknown"))
        )
    elif fault == "swizzle-write":
        loop.body.statements.insert(
            1,
            AssignmentNode(
                MemberAccessNode(IdentifierNode("position"), "xy"),
                IdentifierNode("unknown"),
            ),
        )
    elif fault == "duplicate-update":
        loop.body.statements.append(loop.body.statements[-1])
    elif fault == "escape":
        function.body.statements.insert(1, UnaryOpNode("&", target()))
    elif fault == "shadow":
        loop.body.statements.insert(
            1, BlockNode([VariableNode("position", "int", _integer(0))])
        )
    elif fault == "volatile":
        function.body.statements[0].qualifiers = ["volatile"]
    elif fault == "overflow":
        loop.body.statements[-1].value = _integer(2147483647)
    elif fault == "condition-write":
        loop.condition.right = UnaryOpNode("++", target())
    else:
        loop.condition.op = loop.condition.operator = "||"
    GLSLCodeGen().annotate_glsl_control_flow_intervals(function)
    assert "position.x" not in inside._glsl_control_flow_intervals
    assert "position.x" not in after._glsl_control_flow_intervals


@pytest.mark.parametrize(
    "position_type,qualifiers",
    [
        ("int64_t&", ()),
        ("int64_t", ("volatile",)),
        ("uint*", ("volatile",)),
        ("volatile uint*", ()),
        ("float", ()),
    ],
)
def test_loop_does_not_borrow_pointer_facts_for_unstable_or_scalar_binding(
    position_type, qualifiers
):
    function, _, inside, after = _offset_loop(
        position_type=position_type, qualifiers=qualifiers
    )
    generator = GLSLCodeGen()
    generator.glsl_entry_pointer_parameters = {id(function): {"position"}}
    generator.annotate_glsl_control_flow_intervals(function)
    assert "position" not in inside._glsl_control_flow_intervals
    assert "position" not in after._glsl_control_flow_intervals


@pytest.mark.parametrize(
    "case",
    [
        "terminal-overflow",
        "negative-start",
        "initial-overflow",
        "mutable-bound",
        "reference-bound",
        "volatile-bound",
        "escaped-position",
        "shadow-counter",
        "shadow-position",
        "unknown-stride",
        "stride-overflow",
        "reused-update",
    ],
)
def test_unproved_loop_does_not_restore_offset(case):
    function, loop, _, after = _offset_loop()
    if case == "terminal-overflow":
        loop.init.var_type = "int"
        loop.condition.op = loop.condition.operator = "<="
        loop.condition.right = _integer(2147483647)
    elif case == "negative-start":
        loop.init.initial_value = _integer(-1)
    elif case == "initial-overflow":
        loop.init.initial_value = BinaryOpNode(_integer(2147483647), "+", _integer(1))
        loop.condition.right = _integer(2147483650, "int64_t")
    elif case in {"mutable-bound", "reference-bound", "volatile-bound"}:
        kind = "int64_t&" if case == "reference-bound" else "int64_t"
        qualifiers = ["volatile"] if case == "volatile-bound" else []
        function.body.statements.insert(
            0, VariableNode("limit", kind, _integer(3), qualifiers=qualifiers)
        )
        loop.condition.right = IdentifierNode("limit")
        if case == "mutable-bound":
            loop.body.statements.append(UnaryOpNode("++", IdentifierNode("limit")))
    elif case == "escaped-position":
        function.body.statements.insert(1, UnaryOpNode("&", IdentifierNode("position")))
    elif case.startswith("shadow-"):
        name = "i" if case == "shadow-counter" else "position"
        loop.body.statements.insert(
            0, BlockNode([VariableNode(name, "int", _integer(0))])
        )
    elif case == "unknown-stride":
        loop.body.statements[-1].value = IdentifierNode("unknown")
    elif case == "reused-update":
        loop.body.statements.append(loop.body.statements[-1])
    else:
        loop.body.statements[-1].value = BinaryOpNode(
            _integer(2147483647), "+", _integer(1)
        )
    GLSLCodeGen().annotate_glsl_control_flow_intervals(function)
    assert "position" not in after._glsl_control_flow_intervals


@pytest.mark.parametrize("kind", ["int", "long", "ulong"])
@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("step", [1, 4])
def test_bounded_pointer_loop_translates(tmp_path, kind, count, step):
    body = (
        "values += offset; "
        f"for ({kind} b=0; b<{count}; ++b) {{ result[0] += values[0]; values += {step}; }} "
        "result[1] = values[0];"
    )
    report = translate_project(
        _index_config(
            tmp_path, source=SOURCE.replace("result[0] = values[offset + index];", body)
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]


@pytest.mark.parametrize(
    "change",
    [
        "++count;",
        "if (index > 1) { ++count; }",
        "if (index > 1) { count++; }",
    ],
)
def test_bounded_conditional_loop_count_translates(tmp_path, change):
    body = (
        f"long count=0; {change} values += offset; "
        "for (long b=0; b<count; ++b) { result[0] += values[0]; values += 1; } "
        "result[1] = values[0];"
    )
    report = translate_project(
        _index_config(
            tmp_path, source=SOURCE.replace("result[0] = values[offset + index];", body)
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]


@pytest.mark.parametrize(
    "body",
    [
        "values += offset; for (long b=0; b<3; ++b) { result[0]=values[0]; values += 2147483647; }",
        "values += offset; long count=3; for (long b=0; b<count; ++b) { result[0]=values[0]; values += 1; ++count; }",
        "values += offset; for (long b=0; b<3; ++b) { result[0]=values[0]; values += 1; --b; }",
        "values += offset; for (long b=0; b<3; ++b) { result[0]=values[0]; if (index>0) { values += 1; } }",
        "values += offset; for (long b=0; b<3; ++b) { result[0]=values[0]; values += 1; values += index; }",
        "values += offset; uint3 limits=uint3(3); for (long b=0; b<limits.x; ++b) { result[0]=values[0]; values += 1; limits.x += 1; }",
        "values += offset; uint3 strides=uint3(1); for (long b=0; b<3; ++b) { result[0]=values[0]; values += strides.x; strides.x += 1; }",
    ],
)
def test_unknown_or_overflowing_pointer_loop_is_rejected(tmp_path, body):
    report = translate_project(
        _index_config(
            tmp_path, source=SOURCE.replace("result[0] = values[offset + index];", body)
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert report["diagnostics"]


NATIVE_LOOPS = {
    "zero": ("for (long b=0; b<0; ++b) { total += values[0]; values += 1; }", (), 0),
    "stride": (
        "for (long b=0; b<3; ++b) { total += values[0]; values += 2; }",
        (0, 2, 4),
        6,
    ),
    "inclusive": (
        "for (ulong b=1; b<=3; ++b) { total += values[0]; values += 3; }",
        (0, 3, 6),
        9,
    ),
    "decreasing": (
        "values += 8; for (long b=0; b<3; ++b) { total += values[0]; values -= 2; }",
        (8, 6, 4),
        2,
    ),
    "continue": (
        "for (long b=0; b<3; ++b) { if (b==1) { continue; } total += values[0]; values += 1; }",
        (0, 1),
        2,
    ),
    "break": (
        "for (long b=0; b<3; ++b) { if (b==1) { break; } total += values[0]; values += 1; }",
        (0,),
        1,
    ),
    "wide-start": (
        "for (long b=4294967296l; b<4294967299l; ++b) { total += values[0]; values += 1; }",
        (0, 1, 2),
        3,
    ),
    "conditional": (
        "long count=0; if (index>1) { ++count; } for (long b=0; b<count; ++b) { total += values[0]; values += 1; }",
        None,
        None,
    ),
}
