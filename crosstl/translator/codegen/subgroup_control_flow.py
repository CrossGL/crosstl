"""Converge pure, subgroup-guarded returns before software barrier analysis."""

from copy import deepcopy

from ..ast import (
    BinaryOpNode,
    BlockNode,
    CastNode,
    ConstructorNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    PrimitiveType,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)

_FLOATS = {
    "float",
    "float32_t",
    "half",
    "float16_t",
    "bfloat",
    "bfloat16",
    "bfloat16_t",
}
_SCALARS = {"bool", "int", "uint", "int32_t", "uint32_t"} | _FLOATS
_VOTES = {"WaveActiveAnyTrue", "WaveActiveAllTrue"}
_REDUCTIONS = {"WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax"}


def _type_name(value):
    return value if isinstance(value, str) else getattr(value, "name", None)


def _statements(body):
    if isinstance(body, BlockNode):
        return list(body.statements)
    if isinstance(body, list):
        return list(body)
    return [] if body is None else [body]


def _operation(node):
    if isinstance(node, WaveOpNode):
        return node.operation
    if isinstance(node, FunctionCallNode):
        function = node.function
        return (
            function if isinstance(function, str) else getattr(function, "name", None)
        )
    return None


def _safe_value(node, names, map_operator, declared_functions, *, speculate=False):
    def safe(child):
        return _safe_value(
            child, names, map_operator, declared_functions, speculate=speculate
        )

    def safe_conversion(type_name):
        return type_name in _SCALARS and (
            not speculate or type_name in {"bool"} | _FLOATS
        )

    if isinstance(node, LiteralNode):
        return True
    if isinstance(node, IdentifierNode):
        return node.name in names
    if isinstance(node, CastNode):
        return safe_conversion(_type_name(node.target_type)) and safe(node.expression)
    if isinstance(node, ConstructorNode):
        return (
            safe_conversion(_type_name(node.constructor_type))
            and not node.named_arguments
            and len(node.arguments) == 1
            and safe(node.arguments[0])
        )
    if isinstance(node, FunctionCallNode):
        return (
            _operation(node) in _SCALARS
            and _operation(node) not in declared_functions
            and safe_conversion(_operation(node))
            and len(node.arguments) == 1
            and not getattr(node, "generic_args", None)
            and safe(node.arguments[0])
        )
    if isinstance(node, UnaryOpNode):
        return map_operator(node.operator) in {"!", "~", "+", "-"} and safe(
            node.operand
        )
    if isinstance(node, BinaryOpNode):
        return map_operator(node.operator) in {
            "+",
            "-",
            "*",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "&",
            "|",
            "^",
        } and all(safe(part) for part in (node.left, node.right))
    return False


def _collective_value(node, declared_functions):
    """Look through pure floating conversions without removing their rounding."""
    while True:
        if isinstance(node, CastNode) and _type_name(node.target_type) in _FLOATS:
            node = node.expression
        elif (
            isinstance(node, ConstructorNode)
            and _type_name(node.constructor_type) in _FLOATS
            and not node.named_arguments
            and len(node.arguments) == 1
        ):
            node = node.arguments[0]
        elif (
            isinstance(node, FunctionCallNode)
            and _operation(node) in _FLOATS
            and _operation(node) not in declared_functions
            and not getattr(node, "generic_args", None)
            and len(node.arguments) == 1
        ):
            node = node.arguments[0]
        else:
            return node


def _guarded_return_plan(function, declared_names, map_operator, declared_functions):
    if _type_name(function.return_type) not in _SCALARS - {"bool"}:
        return None
    parameters = function.parameters or []
    # Unused receiver/resource parameters need not disqualify a pure method.
    # Only by-value scalar parameters may occur in speculated expressions.
    names = {
        parameter.name
        for parameter in parameters
        if _type_name(parameter.param_type) in _SCALARS
        and not set(parameter.qualifiers or []) & {"out", "inout", "volatile"}
    }
    names.update({"NAN", "INFINITY"} - declared_names)
    statements = _statements(function.body)
    alias = None
    if statements and isinstance(statements[0], VariableNode):
        alias = statements.pop(0)
        if _type_name(alias.var_type) != "bool" or alias.name in names:
            return None
    if not statements or not isinstance(statements[0], IfNode):
        return None
    branch = statements[0]
    if branch.else_if_conditions or branch.else_if_bodies:
        return None
    early = _statements(branch.then_branch)
    tail = (
        _statements(branch.else_branch)
        if len(statements) == 1
        else statements[1:] if branch.else_branch is None else []
    )
    if (
        len(early) != 1
        or len(tail) != 1
        or not isinstance(early[0], ReturnNode)
        or not isinstance(tail[0], ReturnNode)
    ):
        return None
    condition = branch.condition
    vote = condition
    if isinstance(vote, UnaryOpNode) and map_operator(vote.operator) == "!":
        vote = vote.operand
    if alias is not None:
        if not isinstance(vote, IdentifierNode) or vote.name != alias.name:
            return None
        vote = alias.initial_value
    reduction = _collective_value(tail[0].value, declared_functions)
    if _operation(vote) not in _VOTES or _operation(reduction) not in _REDUCTIONS:
        return None
    if _operation(vote) in declared_names or _operation(reduction) in declared_names:
        return None
    for call in (vote, reduction):
        if len(call.arguments) != 1 or not _safe_value(
            call.arguments[0],
            names,
            map_operator,
            declared_functions,
            speculate=call is reduction,
        ):
            return None
    if not _safe_value(early[0].value, names, map_operator, declared_functions):
        return None
    return alias, condition, tail[0].value, early[0].value


def converge_subgroup_guarded_returns(ast, walk_ast, map_operator):
    """Copy and lower proven pure return patterns; never mutate the caller's AST.

    A vote is uniform within each logical subgroup, not across a workgroup.
    Computing both collectives before selection preserves active subgroup inputs
    while every invocation reaches the software barriers. Memory reads, unknown
    calls, mutation and potentially trapping arithmetic are not speculated.
    """
    nodes = list(walk_ast(ast))
    declared_functions = {node.name for node in nodes if isinstance(node, FunctionNode)}
    declared_names = {
        node.name
        for node in nodes
        if not isinstance(node, IdentifierNode)
        and isinstance(getattr(node, "name", None), str)
    }
    plans = [
        (node, plan)
        for node in nodes
        if isinstance(node, FunctionNode)
        and (
            plan := _guarded_return_plan(
                node, declared_names, map_operator, declared_functions
            )
        )
        is not None
    ]
    if not plans:
        return ast
    reserved = {
        node.name for node in nodes if isinstance(getattr(node, "name", None), str)
    }
    memo = {}
    result = deepcopy(ast, memo)

    def fresh(base):
        name = base
        while name in reserved:
            name += "_"
        reserved.add(name)
        return name

    for function, plan in plans:
        alias, condition, reduction, early_value = deepcopy(plan, memo)
        function = memo[id(function)]
        guard_name = fresh("__crossgl_subgroup_guard")
        result_name = fresh("__crossgl_subgroup_result")
        location = getattr(condition, "source_location", None)
        statements = [alias] if alias is not None else []
        statements.extend(
            [
                VariableNode(
                    guard_name,
                    PrimitiveType("bool"),
                    condition,
                    source_location=location,
                ),
                VariableNode(
                    result_name,
                    function.return_type,
                    reduction,
                    source_location=getattr(reduction, "source_location", None),
                ),
                ReturnNode(
                    TernaryOpNode(
                        IdentifierNode(guard_name),
                        early_value,
                        IdentifierNode(result_name),
                    ),
                    source_location=location,
                ),
            ]
        )
        if isinstance(function.body, BlockNode):
            function.body.statements = statements
        else:
            function.body = statements
    return result
