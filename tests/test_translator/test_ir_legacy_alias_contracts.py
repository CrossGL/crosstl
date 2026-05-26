from crosstl.translator.ast import (
    ArrayAccessNode,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
    UnaryOpNode,
    VariableNode,
)


def test_variable_nodes_preserve_legacy_type_and_semantic_aliases():
    variable = VariableNode(
        "position",
        PrimitiveType("float4"),
        attributes=[AttributeNode("position")],
    )

    assert variable.var_type is variable.vtype
    assert variable.semantic == "position"


def test_assignment_and_expression_nodes_preserve_codegen_aliases():
    target = IdentifierNode("value")
    assigned_value = LiteralNode(1, PrimitiveType("int"))
    assignment = AssignmentNode(target, assigned_value, operator="+=")

    assert assignment.left is target
    assert assignment.right is assigned_value
    assert assignment.operator == "+="

    binary = BinaryOpNode(IdentifierNode("a"), "*", IdentifierNode("b"))
    unary = UnaryOpNode("-", IdentifierNode("a"))

    assert binary.op == "*"
    assert unary.op == "-"


def test_call_and_access_nodes_preserve_codegen_aliases():
    callee = IdentifierNode("sample")
    argument = IdentifierNode("uv")
    call = FunctionCallNode(callee, [argument])

    assert call.name is callee
    assert call.args == [argument]

    receiver = IdentifierNode("material")
    member_access = MemberAccessNode(receiver, "albedo")

    assert member_access.object is receiver
    assert member_access.member == "albedo"

    array_expr = IdentifierNode("values")
    index_expr = IdentifierNode("i")
    array_access = ArrayAccessNode(array_expr, index_expr)

    assert array_access.array is array_expr
    assert array_access.index is index_expr
