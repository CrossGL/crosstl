"""Converge proven subgroup control flow before software barrier analysis."""

from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    CastNode,
    ConstructorNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    ReturnNode,
    StructNode,
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
    empty_types = {
        node.name for node in nodes if isinstance(node, StructNode) and not node.members
    }

    def expression_inputs(node, parameters, visiting):
        if isinstance(node, IdentifierNode) and node.name in parameters:
            return frozenset({parameters[node.name]}), False
        if isinstance(node, LiteralNode):
            return frozenset(), False
        if isinstance(node, CastNode) and _type_name(node.target_type) in _FLOATS:
            return expression_inputs(node.expression, parameters, visiting)
        if (
            isinstance(node, ConstructorNode)
            and _type_name(node.constructor_type) in _FLOATS | {"bool"}
            and not node.named_arguments
            and len(node.arguments) == 1
        ):
            return expression_inputs(node.arguments[0], parameters, visiting)
        operation = _operation(node)
        if (
            isinstance(node, FunctionCallNode)
            and operation in _FLOATS | {"bool"}
            and operation not in functions
            and len(node.arguments) == 1
            and not node.generic_args
        ):
            return expression_inputs(node.arguments[0], parameters, visiting)
        if operation in operations and operation not in functions:
            required = range(len(node.arguments))
            collective = True
        elif isinstance(node, FunctionCallNode):
            proof = wrapper(node, visiting)
            if proof is None:
                return None
            _function, required = proof
            collective = True
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
            proof = expression_inputs(argument, parameters, visiting)
            if proof is None:
                return None
            if operation in operations and index == 0 and not proof[0]:
                return None
            dependencies.update(proof[0])
            collective |= proof[1]
        return frozenset(dependencies), collective

    def wrapper(call, visiting=frozenset()):
        if getattr(call, "generic_args", None):
            return None
        candidates = [
            function
            for function in functions.get(_operation(call), ())
            if len(function.parameters) == len(call.arguments)
        ]
        if len(candidates) != 1:
            return None
        function = candidates[0]
        key = id(function)
        if key in visiting:
            return None
        if key not in summaries:
            statements = _statements(function.body)
            proof = None
            if (
                len(statements) == 1
                and isinstance(statements[0], ReturnNode)
                and _type_name(function.return_type) in _SCALARS
                and not getattr(function, "generic_params", None)
                and not any(
                    set(parameter.qualifiers or []) & {"out", "volatile"}
                    or set(getattr(parameter.param_type, "qualifiers", []) or [])
                    & {"volatile"}
                    for parameter in function.parameters
                )
            ):
                proof = expression_inputs(
                    statements[0].value,
                    {
                        parameter.name: i
                        for i, parameter in enumerate(function.parameters)
                    },
                    visiting | {key},
                )
            if proof is not None and proof[1]:
                for index, parameter in enumerate(function.parameters):
                    if (
                        index not in proof[0]
                        and _type_name(parameter.param_type) in empty_types
                    ):
                        continue
                    if (
                        _type_name(parameter.param_type) not in _SCALARS
                        or set(parameter.qualifiers or [])
                        & {"out", "inout", "volatile"}
                        or set(getattr(parameter.param_type, "qualifiers", []) or [])
                        & {"volatile"}
                    ):
                        proof = None
                        break
            summaries[key] = proof[0] if proof is not None and proof[1] else None
        required = summaries[key]
        return (function, required) if required is not None else None

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
                _type_name(declaration.var_type) not in _SCALARS
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
                if _type_name(parameter.param_type) in _SCALARS
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
            type_ = PrimitiveType(_type_name(function.parameters[index].param_type))
            statements.append(
                VariableNode(
                    name, type_, LiteralNode(0, type_), source_location=location
                )
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
