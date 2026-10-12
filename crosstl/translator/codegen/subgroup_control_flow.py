"""Converge proven subgroup control flow before software barrier analysis."""

from copy import deepcopy

from ..arithmetic_conversions import ArithmeticScalarKind, source_integer_shape
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    CastNode,
    ConstructorNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
    ReturnNode,
    StructMemberNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from .loop_induction import (
    checked_induction_value,
    fixed_stride_loop_terminates,
    geometric_loop_terminates,
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


def converge_subgroup_conditional_calls(ast, entries, walk_ast, map_operator):
    """Stage guarded inputs around pure subgroup wrappers in compute entries.

    Only a subgroup-uniform predicate may be lifted: every participating lane
    in an enabled subgroup must retain its original input. Disabled subgroups
    execute the same barrier schedule using private zero inputs. Their memory
    accesses and stores stay guarded, and their results are never published.
    """
    nodes = list(walk_ast(ast))
    functions = {}
    for node in nodes:
        if isinstance(node, FunctionNode):
            functions.setdefault(node.name, []).append(node)
    operations = (
        _REDUCTIONS
        | _VOTES
        | {
            "WavePrefixSum",
            "WavePrefixProduct",
            "WavePrefixInclusiveSum",
            "WavePrefixInclusiveProduct",
            "WaveShuffleDown",
            "WaveReadLaneAt",
            "WaveShuffleAndFillUp",
        }
    )
    summaries = {}
    scalar_types = _SCALARS | {
        "char",
        "uchar",
        "short",
        "ushort",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
    }
    value_structs = {
        node.name: {
            member.name: _type_name(member.member_type) for member in node.members
        }
        for node in nodes
        if isinstance(node, StructNode)
        and node.members
        and not (node.generic_params or node.inheritance or node.attributes)
        and len({member.name for member in node.members}) == len(node.members)
        and all(
            isinstance(member, StructMemberNode)
            and _type_name(member.member_type) in scalar_types
            and not (
                member.attributes or member.resource_qualifiers or member.default_value
            )
            and not getattr(member.member_type, "qualifiers", [])
            for member in node.members
        )
    }
    value_types = scalar_types | set(value_structs)
    empty_types = {
        node.name for node in nodes if isinstance(node, StructNode) and not node.members
    }

    def value_type(node, types):
        if isinstance(node, IdentifierNode):
            return types.get(node.name)
        if isinstance(node, MemberAccessNode):
            return value_structs.get(value_type(node.object_expr, types), {}).get(
                node.member
            )
        if isinstance(node, LiteralNode):
            return _type_name(node.literal_type)
        if isinstance(node, CastNode):
            return _type_name(node.target_type)
        if isinstance(node, ConstructorNode):
            return _type_name(node.constructor_type)
        if isinstance(node, UnaryOpNode):
            return (
                "bool"
                if map_operator(node.operator) == "!"
                else value_type(node.operand, types)
            )
        if isinstance(node, BinaryOpNode):
            if map_operator(node.operator) in {
                "==",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
                "&&",
                "||",
            }:
                return "bool"
            left, right = value_type(node.left, types), value_type(node.right, types)
            return (
                left
                if left == right
                else (
                    "float"
                    if left in _FLOATS or right in _FLOATS
                    else (
                        "uint"
                        if left in {"uint", "uint32_t"} or right in {"uint", "uint32_t"}
                        else "int"
                    )
                )
            )
        if isinstance(node, TernaryOpNode):
            left, right = value_type(node.true_expr, types), value_type(
                node.false_expr, types
            )
            return left if left == right else None
        operation = _operation(node)
        if operation in functions:
            function = resolve_function(node, types)
            return _type_name(function.return_type) if function is not None else None
        if operation in value_types:
            return operation
        if operation in {"isnan", "isinf"} | _VOTES:
            return "bool"
        if operation in {"asfloat", "asint", "asuint"}:
            return {"asfloat": "float", "asint": "int", "asuint": "uint"}[operation]
        if operation in operations and node.arguments:
            return value_type(node.arguments[0], types)
        return None

    def integer_constant(node):
        type_name = value_type(node, {})
        shape = source_integer_shape(type_name) if type_name else None
        if shape is None or shape[2] != 1 or shape[1] > 64:
            return None
        unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
        if isinstance(node, LiteralNode) and type(node.value) is int:
            value = node.value
        elif isinstance(node, UnaryOpNode) and map_operator(node.operator) in {
            "+",
            "-",
        }:
            value = integer_constant(node.operand)
            if value is None or unsigned and map_operator(node.operator) == "-":
                return None
            value *= -1 if map_operator(node.operator) == "-" else 1
        else:
            return None
        minimum = 0 if unsigned else -(1 << (shape[1] - 1))
        maximum = (1 << (shape[1] - (not unsigned))) - 1
        return value if minimum <= value <= maximum else None

    def converts_safely(target, value, types):
        source = value_type(value, types)
        if target in value_structs:
            return source == target
        constant = integer_constant(value)
        if constant is not None and target in scalar_types:
            if target in _FLOATS | {"bool"}:
                return True
            shape = source_integer_shape(target)
            unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
            return (
                (0 if unsigned else -(1 << (shape[1] - 1)))
                <= constant
                <= (1 << (shape[1] - (not unsigned))) - 1
            )
        return (
            source in scalar_types
            and target in scalar_types
            and (target in _FLOATS | {"bool"} or source not in _FLOATS)
        )

    def resolve_function(call, types):
        candidates = [
            function
            for function in functions.get(_operation(call), ())
            if len(function.parameters) == len(call.arguments)
        ]
        if len(candidates) > 1:
            candidates = [
                function
                for function in candidates
                if all(
                    (
                        _type_name(parameter.param_type) in scalar_types
                        and value_type(argument, types) in scalar_types
                    )
                    or (
                        _type_name(parameter.param_type) in value_types | empty_types
                        and value_type(argument, types)
                        == _type_name(parameter.param_type)
                    )
                    for parameter, argument in zip(function.parameters, call.arguments)
                )
            ]
        return candidates[0] if len(candidates) == 1 else None

    def expression_inputs(node, parameters, types, visiting, initialized):
        if (
            isinstance(node, FunctionCallNode)
            and _operation(node) in value_structs
            and _operation(node) not in functions
            and not node.generic_args
        ):
            node = ConstructorNode(NamedType(_operation(node)), node.arguments)
        if isinstance(node, IdentifierNode) and node.name in parameters:
            return (
                (parameters[node.name], False)
                if types[node.name] in value_types and node.name in initialized
                else None
            )
        if isinstance(node, MemberAccessNode) and isinstance(
            node.object_expr, IdentifierNode
        ):
            name = node.object_expr.name
            if (
                name in parameters
                and value_type(node, types) in scalar_types
                and (name in initialized or (name, node.member) in initialized)
            ):
                return parameters[name], False
            return None
        if isinstance(node, LiteralNode):
            return frozenset(), False
        if integer_constant(node) is not None:
            return frozenset(), False
        if (
            isinstance(node, (UnaryOpNode, BinaryOpNode))
            and checked_induction_value(node, {}) is not None
        ):
            return frozenset(), False
        if isinstance(node, CastNode) and converts_safely(
            _type_name(node.target_type), node.expression, types
        ):
            return expression_inputs(
                node.expression, parameters, types, visiting, initialized
            )
        if (
            isinstance(node, ConstructorNode)
            and not node.named_arguments
            and len(node.arguments) == 1
            and converts_safely(
                _type_name(node.constructor_type), node.arguments[0], types
            )
        ):
            return expression_inputs(
                node.arguments[0], parameters, types, visiting, initialized
            )
        if (
            isinstance(node, ConstructorNode)
            and _type_name(node.constructor_type) in value_structs
        ):
            fields = value_structs[_type_name(node.constructor_type)]
            if (
                node.named_arguments
                or len(fields) != len(node.arguments)
                or not all(
                    converts_safely(field_type, argument, types)
                    for field_type, argument in zip(fields.values(), node.arguments)
                )
            ):
                return None
            proofs = [
                expression_inputs(argument, parameters, types, visiting, initialized)
                for argument in node.arguments
            ]
            if any(proof is None for proof in proofs):
                return None
            return frozenset().union(*(proof[0] for proof in proofs)), any(
                proof[1] for proof in proofs
            )
        children = None
        if isinstance(node, UnaryOpNode):
            operator = map_operator(node.operator)
            operand_type = value_type(node.operand, types)
            if (
                operator in {"!", "+"}
                or operator == "-"
                and (
                    operand_type in _FLOATS
                    or checked_induction_value(node, {}) is not None
                )
                or operator == "~"
                and operand_type in scalar_types - _FLOATS
            ):
                children = [node.operand]
        elif isinstance(node, BinaryOpNode):
            operator = map_operator(node.operator)
            operands = {value_type(node.left, types), value_type(node.right, types)}
            if (
                operator in {
                    "==",
                    "!=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                    "&&",
                    "||",
                    "&",
                    "|",
                    "^",
                }
                or operator in {"+", "-", "*"}
                and operands <= scalar_types
                and operands & _FLOATS
            ):
                children = [node.left, node.right]
        elif isinstance(node, TernaryOpNode):
            children = [node.condition, node.true_expr, node.false_expr]
        if children is not None:
            proofs = [
                expression_inputs(child, parameters, types, visiting, initialized)
                for child in children
            ]
            if any(proof is None for proof in proofs):
                return None
            return frozenset().union(*(proof[0] for proof in proofs)), any(
                proof[1] for proof in proofs
            )
        operation = _operation(node)
        if (
            isinstance(node, FunctionCallNode)
            and operation in scalar_types
            and operation not in functions
            and len(node.arguments) == 1
            and not node.generic_args
            and converts_safely(operation, node.arguments[0], types)
        ):
            return expression_inputs(
                node.arguments[0], parameters, types, visiting, initialized
            )
        if (
            isinstance(node, FunctionCallNode)
            and operation in {"isnan", "isinf", "asfloat", "asint", "asuint"}
            and operation not in functions
            and len(node.arguments) == 1
            and not node.generic_args
            and value_type(node.arguments[0], types)
            in {"float", "float32_t", "int", "int32_t", "uint", "uint32_t"}
        ):
            return expression_inputs(
                node.arguments[0], parameters, types, visiting, initialized
            )
        if operation in operations and operation not in functions:
            required = range(len(node.arguments))
            collective = True
        elif isinstance(node, FunctionCallNode):
            proof = wrapper(node, visiting, allow_pure=True, types=types)
            if proof is None:
                return None
            _function, required, collective = proof
            if any(
                _type_name(parameter.param_type) in value_types
                and not converts_safely(
                    _type_name(parameter.param_type), argument, types
                )
                for parameter, argument in zip(_function.parameters, node.arguments)
            ):
                return None
        else:
            return None
        dependencies = set()
        for index, argument in enumerate(node.arguments):
            if index not in required:
                if not (
                    isinstance(argument, LiteralNode)
                    or isinstance(argument, IdentifierNode)
                    and argument.name in parameters
                ):
                    return None
                continue
            proof = expression_inputs(
                argument, parameters, types, visiting, initialized
            )
            if proof is None:
                return None
            if operation in operations and index == 0 and not proof[0]:
                return None
            dependencies.update(proof[0])
            collective |= proof[1]
        return frozenset(dependencies), collective

    def body_inputs(
        body,
        parameters,
        types,
        visiting,
        result_type,
        constants=None,
        controls=frozenset(),
        initialized=None,
    ):
        # Names mark whole values; (name, field) tracks partial local construction.
        initialized = set(parameters if initialized is None else initialized)
        parameters, types, constants = (
            dict(parameters),
            dict(types),
            dict(constants or {}),
        )
        dependencies, collective = set(), False
        for statement in _statements(body):
            member = None
            if isinstance(statement, ExpressionStatementNode):
                statement = statement.expression
            if isinstance(statement, (VariableNode, AssignmentNode, ReturnNode)):
                if isinstance(statement, VariableNode):
                    if (
                        statement.name in types
                        or statement.name in controls
                        or statement.attributes
                        or set(statement.qualifiers or []) - {"const", "thread"}
                        or set(getattr(statement.var_type, "qualifiers", []) or [])
                        - {"const", "thread"}
                    ):
                        return None
                    name, target, value = (
                        statement.name,
                        _type_name(statement.var_type),
                        statement.initial_value,
                    )
                    if value is None and target in value_structs:
                        parameters[name], types[name] = frozenset(), target
                        initialized.discard(name)
                        continue
                elif isinstance(statement, AssignmentNode):
                    destination = statement.target
                    if isinstance(destination, MemberAccessNode):
                        member = destination.member
                        destination = destination.object_expr
                    if (
                        not isinstance(destination, IdentifierNode)
                        or destination.name not in parameters
                        or destination.name in controls
                        or map_operator(statement.operator) != "="
                    ):
                        return None
                    name = destination.name
                    target, value = types[name], statement.value
                    if member is not None:
                        target = value_structs.get(target, {}).get(member)
                        if target is None:
                            return None
                else:
                    name, target, value = None, result_type, statement.value
                if isinstance(value, ArrayLiteralNode) and target in value_structs:
                    value = ConstructorNode(NamedType(target), value.elements)
                proof = expression_inputs(
                    value, parameters, types, visiting, initialized
                )
                if (
                    proof is None
                    or target is not None
                    and not converts_safely(target, value, types)
                ):
                    return None
                if name is not None:
                    if member is None:
                        parameters[name], types[name] = proof[0], target
                        initialized.add(name)
                    else:
                        parameters[name] |= proof[0]
                        initialized.add((name, member))
                        if all(
                            (name, field) in initialized
                            for field in value_structs[types[name]]
                        ):
                            initialized.add(name)
                    constants.pop(name, None)
                    if isinstance(statement, VariableNode) and "const" in set(
                        statement.qualifiers or []
                    ) | set(getattr(statement.var_type, "qualifiers", []) or []):
                        constant = checked_induction_value(value, constants)
                        if constant is not None:
                            constants[name] = constant
            elif isinstance(statement, (BlockNode, IfNode)):
                if isinstance(statement, BlockNode):
                    branches, conditions = [statement], []
                elif (
                    isinstance(statement.condition, LiteralNode)
                    and type(statement.condition.value) is bool
                    and not statement.else_if_conditions
                    and not statement.else_if_bodies
                ):
                    branches = [
                        (
                            statement.then_branch
                            if statement.condition.value
                            else statement.else_branch
                        )
                    ]
                    conditions = []
                else:
                    branches = [
                        statement.then_branch,
                        *statement.else_if_bodies,
                        statement.else_branch,
                    ]
                    conditions = [statement.condition, *statement.else_if_conditions]
                proofs = [
                    expression_inputs(
                        condition, parameters, types, visiting, initialized
                    )
                    for condition in conditions
                ]
                results = [
                    body_inputs(
                        branch,
                        parameters,
                        types,
                        visiting,
                        result_type,
                        constants,
                        controls,
                        initialized,
                    )
                    for branch in branches
                ]
                if any(proof is None for proof in [*proofs, *results]):
                    return None
                proofs += [result[:2] for result in results]
                if conditions and any(proof[1] for proof in proofs):
                    return None
                proof = frozenset().union(*(proof[0] for proof in proofs)), any(
                    proof[1] for proof in proofs
                )
                for name in parameters:
                    parameters[name] |= frozenset().union(
                        *(result[2][name] for result in results)
                    )
                initialized.intersection_update(*(result[4] for result in results))
                constants = {
                    name: value
                    for name, value in constants.items()
                    if all(result[3].get(name) == value for result in results)
                }
            elif isinstance(statement, ForNode):
                initial, condition, update = (
                    statement.init,
                    statement.condition,
                    statement.update,
                )
                if (
                    not isinstance(initial, VariableNode)
                    or initial.name in parameters
                    or initial.name in controls
                    or initial.attributes
                    or initial.qualifiers
                    or not isinstance(condition, BinaryOpNode)
                    or not isinstance(condition.left, IdentifierNode)
                    or condition.left.name != initial.name
                    or not isinstance(update, (AssignmentNode, UnaryOpNode))
                ):
                    return None
                target = (
                    update.target
                    if isinstance(update, AssignmentNode)
                    else update.operand
                )
                if (
                    not isinstance(target, IdentifierNode)
                    or target.name != initial.name
                ):
                    return None
                terminate = (
                    geometric_loop_terminates
                    if map_operator(update.operator) in {"*=", "<<="}
                    else fixed_stride_loop_terminates
                )
                if not terminate(
                    _type_name(initial.var_type),
                    initial.initial_value,
                    condition.right,
                    update,
                    constants,
                    map_operator(condition.operator),
                ):
                    return None
                if any(
                    isinstance(child, ReturnNode) for child in walk_ast(statement.body)
                ):
                    return None
                nested_parameters = {**parameters, initial.name: frozenset()}
                nested_types = {**types, initial.name: _type_name(initial.var_type)}
                loop_controls = (
                    controls
                    | {initial.name}
                    | {
                        child.name
                        for root in (condition.right, update)
                        for child in walk_ast(root)
                        if isinstance(child, IdentifierNode)
                    }
                )
                result = body_inputs(
                    statement.body,
                    nested_parameters,
                    nested_types,
                    visiting,
                    result_type,
                    constants,
                    loop_controls,
                    initialized | {initial.name},
                )
                if result is None:
                    return None
                proof = result[:2]
                for name in parameters:
                    parameters[name] |= result[2][name]
                initialized.intersection_update(result[4])
                constants = {
                    name: value
                    for name, value in constants.items()
                    if result[3].get(name) == value
                }
            else:
                return None
            dependencies.update(proof[0])
            collective |= proof[1]
        return frozenset(dependencies), collective, parameters, constants, initialized

    def wrapper(call, visiting=frozenset(), *, allow_pure=False, types=None):
        if getattr(call, "generic_args", None):
            return None
        function = resolve_function(call, types or {})
        if function is None:
            return None
        key = id(function)
        if key in visiting:
            return None
        if key not in summaries:
            statements = _statements(function.body)
            proof = None
            if (
                statements
                and isinstance(statements[-1], ReturnNode)
                and _type_name(function.return_type) in value_types
                and not getattr(function, "generic_params", None)
                and not any(
                    set(parameter.qualifiers or []) & {"out", "volatile"}
                    or set(getattr(parameter.param_type, "qualifiers", []) or [])
                    & {"volatile"}
                    for parameter in function.parameters
                )
            ):
                proof = body_inputs(
                    statements,
                    {
                        parameter.name: frozenset({i})
                        for i, parameter in enumerate(function.parameters)
                    },
                    {
                        parameter.name: _type_name(parameter.param_type)
                        for parameter in function.parameters
                    },
                    visiting | {key},
                    _type_name(function.return_type),
                )
            if proof is not None:
                for index, parameter in enumerate(function.parameters):
                    if (
                        index not in proof[0]
                        and _type_name(parameter.param_type) in empty_types
                    ):
                        continue
                    if _type_name(parameter.param_type) not in value_types or (
                        set(parameter.qualifiers or [])
                        | set(getattr(parameter.param_type, "qualifiers", []) or [])
                    ) & {
                        "out",
                        "inout",
                        "volatile",
                        "device",
                        "constant",
                        "threadgroup",
                        "groupshared",
                    }:
                        proof = None
                        break
            summaries[key] = proof[:2] if proof is not None else None
        proof = summaries[key]
        if proof is None or not (allow_pure or proof[1]):
            return None
        return (function, *proof) if allow_pure else (function, proof[0])

    def root_name(node):
        while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
            node = (
                node.array_expr
                if isinstance(node, ArrayAccessNode)
                else node.object_expr
            )
        if isinstance(node, UnaryOpNode) and map_operator(node.operator) == "&":
            return root_name(node.operand)
        return node.name if isinstance(node, IdentifierNode) else None

    declared = {
        node.name
        for node in nodes
        if not isinstance(node, IdentifierNode)
        and isinstance(getattr(node, "name", None), str)
    }
    plans = []
    for stage, entry, _config in entries:
        if stage != "compute":
            continue
        names = {"gl_SubgroupID"} - declared
        for parameter in entry.parameters:
            if any(
                attribute.name == "gl_SubgroupID" for attribute in parameter.attributes
            ):
                names.add(parameter.name)
        for node in walk_ast(entry.body):
            if isinstance(node, VariableNode):
                names.discard(node.name)
                if isinstance(node.var_type, (PointerType, ReferenceType)):
                    names.discard(root_name(node.initial_value))
            elif isinstance(node, AssignmentNode):
                names.discard(root_name(node.target))
            elif isinstance(node, UnaryOpNode) and map_operator(node.operator) in {
                "++",
                "--",
                "&",
            }:
                names.discard(root_name(node.operand))
            elif isinstance(node, FunctionCallNode):
                for argument in node.arguments:
                    names.discard(root_name(argument))
        if not names:
            continue
        for node in walk_ast(entry.body):
            if (
                not isinstance(node, IfNode)
                or node.else_branch is not None
                or node.else_if_conditions
                or node.else_if_bodies
                or not _safe_value(node.condition, names, map_operator, set(functions))
                or not any(
                    isinstance(child, IdentifierNode) and child.name in names
                    for child in walk_ast(node.condition)
                )
            ):
                continue
            body = _statements(node.then_branch)
            if not body or not isinstance(body[0], VariableNode):
                continue
            declaration = body[0]
            if (
                _type_name(declaration.var_type) not in value_types
                or declaration.attributes
                or set(declaration.qualifiers or []) - {"const", "thread"}
                or set(getattr(declaration.var_type, "qualifiers", []) or [])
                - {"const", "thread"}
            ):
                continue
            call = declaration.initial_value
            if not isinstance(call, FunctionCallNode):
                continue
            proof = wrapper(call)
            if proof is None:
                continue
            function, _required = proof
            staged = {
                index
                for index, parameter in enumerate(function.parameters)
                if _type_name(parameter.param_type) in value_types
            }
            if any(
                not isinstance(argument, IdentifierNode)
                for index, argument in enumerate(call.arguments)
                if index not in staged
            ):
                continue
            # Nested collectives in argument evaluation are not staged by this pass.
            if any(
                _operation(child) in operations
                or isinstance(child, FunctionCallNode)
                and wrapper(child) is not None
                for argument in call.arguments
                for child in walk_ast(argument)
            ):
                continue
            plans.append((node, function, staged))
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

    def zero_value(type_):
        fields = value_structs.get(_type_name(type_))
        if fields:
            return ConstructorNode(
                type_,
                [
                    zero_value(PrimitiveType(field_type))
                    for field_type in fields.values()
                ],
            )
        return LiteralNode(False if _type_name(type_) == "bool" else 0, type_)

    for node, function, required in plans:
        branch = memo[id(node)]
        body = _statements(branch.then_branch)
        declaration = body[0]
        call = declaration.initial_value
        location = getattr(branch, "source_location", None)
        guard = fresh("__crossgl_subgroup_enabled")
        statements = [
            VariableNode(
                guard, PrimitiveType("bool"), branch.condition, source_location=location
            )
        ]
        inputs = []
        arguments = []
        for index, argument in enumerate(call.arguments):
            if index not in required:
                arguments.append(argument)
                continue
            name = fresh("__crossgl_subgroup_input")
            type_ = deepcopy(function.parameters[index].param_type)
            statements.append(
                VariableNode(name, type_, zero_value(type_), source_location=location)
            )
            inputs.append(
                AssignmentNode(IdentifierNode(name), argument, source_location=location)
            )
            arguments.append(IdentifierNode(name))
        call.arguments = arguments
        call.args = arguments
        statements.extend(
            [
                IfNode(
                    IdentifierNode(guard), BlockNode(inputs), source_location=location
                ),
                declaration,
                IfNode(
                    IdentifierNode(guard), BlockNode(body[1:]), source_location=location
                ),
            ]
        )
        replacement = BlockNode(statements, source_location=location)
        # Replace every legacy alias of the original branch using the same node.
        for parent in walk_ast(result):
            for key, value in list(getattr(parent, "__dict__", {}).items()):
                if value is branch:
                    setattr(parent, key, replacement)
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        if child is branch:
                            value[index] = replacement
    return result
