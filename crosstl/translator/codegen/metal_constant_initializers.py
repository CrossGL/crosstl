"""Fold member-wise constructor helpers in constant initializers for Metal.

Metal sources initialize ``static constexpr constant`` data members with
converting constructors, such as ``static constexpr constant U init = U(0);``.
The CrossGL import spells such a constructor as a helper function that
declares a record, assigns each member once from its parameters and returns
the record. A call to that helper is not a constant expression in Metal, and
the helper is emitted after the records that use it. When the helper has
exactly that shape and the arguments are constants, the call is the aggregate
whose members are the assigned expressions with the arguments substituted,
which is the initializer Metal needs.
"""

from copy import deepcopy

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .pointer_reinterpret import scalar_storage_layout


def _call_name(call):
    function = getattr(call, "function", None)
    name = getattr(function, "name", function)
    return name if isinstance(name, str) else None


def _type_name(type_node):
    return getattr(type_node, "name", None) if type_node is not None else None


def _constant(node, parameters, generator, depth=0):
    """Whether ``node`` is built only from literals, parameters and casts."""

    if depth > 32:
        return False
    if isinstance(node, LiteralNode):
        return True
    if isinstance(node, IdentifierNode):
        return node.name in parameters
    if isinstance(node, CastNode):
        return _constant(node.expression, parameters, generator, depth + 1)
    if isinstance(node, UnaryOpNode):
        return getattr(node, "op", None) in {"-", "+", "~", "!"} and _constant(
            node.operand, parameters, generator, depth + 1
        )
    if isinstance(node, BinaryOpNode):
        return _constant(node.left, parameters, generator, depth + 1) and _constant(
            node.right, parameters, generator, depth + 1
        )
    if isinstance(node, TernaryOpNode):
        return all(
            _constant(part, parameters, generator, depth + 1)
            for part in (node.condition, node.true_expr, node.false_expr)
        )
    if isinstance(node, FunctionCallNode):
        # A scalar conversion such as ``float(x)``; never a user function.
        name = _call_name(node)
        return (
            name is not None
            and name not in generator.functions_by_name
            and (name == "bool" or scalar_storage_layout(name) is not None)
            and len(node.arguments) == 1
            and _constant(node.arguments[0], parameters, generator, depth + 1)
        )
    return False


def _substitute(node, arguments):
    if isinstance(node, IdentifierNode) and node.name in arguments:
        return deepcopy(arguments[node.name])
    result = deepcopy(node)
    stack = [result]
    while stack:
        current = stack.pop()
        for key, value in list(vars(current).items()):
            if key in {"parent", "annotations", "source_location"}:
                continue
            if isinstance(value, IdentifierNode) and value.name in arguments:
                setattr(current, key, deepcopy(arguments[value.name]))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, IdentifierNode) and item.name in arguments:
                        value[index] = deepcopy(arguments[item.name])
                    elif hasattr(item, "__dict__"):
                        stack.append(item)
            elif hasattr(value, "__dict__") and not isinstance(value, type):
                stack.append(value)
    return result


def member_wise_constructor_values(generator, call):
    """Return ``[(member, type, expression)]`` for a foldable helper call."""

    if not isinstance(call, FunctionCallNode):
        return None
    function = generator.functions_by_name.get(_call_name(call))
    overloads = generator.function_overloads_by_name.get(_call_name(call), [])
    if function is None or len(overloads) != 1:
        return None
    parameters = [parameter.name for parameter in function.parameters]
    if len(parameters) != len(call.arguments) or any(
        not _constant(argument, (), generator) for argument in call.arguments
    ):
        return None
    statements = list(getattr(function.body, "statements", function.body) or ())
    if len(statements) < 2:
        return None
    declaration, *assignments, result = statements
    record = generator.structs_by_name.get(
        _type_name(getattr(declaration, "var_type", None))
    )
    if (
        not isinstance(declaration, VariableNode)
        or declaration.initial_value is not None
        or record is None
        or _type_name(function.return_type) != record.name
        or not isinstance(result, ReturnNode)
        or not isinstance(result.value, IdentifierNode)
        or result.value.name != declaration.name
    ):
        return None
    members = [
        member
        for member in record.members
        if "static" not in {
            str(getattr(item, "name", item)) for item in member.attributes or ()
        }
    ]
    assigned = {}
    for statement in assignments:
        assignment = (
            statement.expression
            if isinstance(statement, ExpressionStatementNode)
            else statement
        )
        target = getattr(assignment, "target", None)
        if (
            not isinstance(assignment, AssignmentNode)
            or getattr(assignment, "operator", "=") != "="
            or not isinstance(target, MemberAccessNode)
            or not isinstance(target.object, IdentifierNode)
            or target.object.name != declaration.name
            or target.member in assigned
            or not _constant(assignment.value, set(parameters), generator)
        ):
            return None
        assigned[target.member] = assignment.value
    if [member.name for member in members] != list(assigned):
        # Every member is assigned once, in declaration order.
        return None
    arguments = dict(zip(parameters, call.arguments))
    return [
        (member.name, member.member_type, _substitute(assigned[member.name], arguments))
        for member in members
    ]


def metal_constant_initializer(generator, value, expected_type):
    """Render a constant initializer, folding member-wise constructor calls."""

    values = member_wise_constructor_values(generator, value)
    if values is None:
        return None
    record_folded_call(generator, value)
    rendered = [
        generator.generate_expression_with_expected(expression, member_type)
        for _member, member_type, expression in values
    ]
    return "{" + ", ".join(rendered) + "}"


def record_folded_call(generator, call):
    """Remember a helper call that a constant initializer replaced."""

    folded = getattr(generator, "metal_folded_constructor_calls", None)
    if folded is None:
        folded = generator.metal_folded_constructor_calls = set()
    folded.add(id(call))


def folded_helper_is_unused(generator, function):
    """Whether every call of ``function`` was folded into an initializer.

    Metal compiles with ``-Werror``, so a ``static`` helper whose only calls
    were folded would fail as an unused function.
    """

    folded = getattr(generator, "metal_folded_constructor_calls", None)
    name = getattr(function, "name", None)
    if not folded or not name:
        return False
    roots = [
        getattr(candidate, "body", None)
        for overloads in generator.function_overloads_by_name.values()
        for candidate in overloads
    ]
    roots.extend(
        member.default_value
        for record in generator.structs_by_name.values()
        for member in getattr(record, "members", ()) or ()
        if getattr(member, "default_value", None) is not None
    )
    calls = [
        node
        for root in roots
        if root is not None
        for node in generator.iter_ast_nodes(root)
        if isinstance(node, FunctionCallNode) and _call_name(node) == name
    ]
    return bool(calls) and all(id(call) in folded for call in calls)


def _static_initializer_calls(generator, records):
    names = []
    for record in records:
        for member in getattr(record, "members", ()) or ():
            value = getattr(member, "default_value", None)
            if value is None:
                continue
            for node in generator.iter_ast_nodes(value):
                if isinstance(node, FunctionCallNode):
                    name = _call_name(node)
                    if name and name not in names:
                        names.append(name)
    return names


def restore_static_initializer_helpers(generator, ast, scoped_ast):
    """Keep helpers that retained records call from member initializers.

    Entry scoping keeps the functions the entry reaches through calls. A
    record that survives scoping may still call a helper from a constant
    member initializer, so the scoped program keeps that helper and its
    callees, as they appear in ``ast``.
    """

    records = list(getattr(scoped_ast, "structs", ()) or ())
    for stage in getattr(scoped_ast, "stages", {}).values():
        records.extend(getattr(stage, "local_structs", ()) or ())
    retained = {
        getattr(function, "name", None)
        for function in getattr(scoped_ast, "functions", ()) or ()
    }
    originals = {}
    for function in getattr(ast, "functions", ()) or ():
        originals.setdefault(getattr(function, "name", None), []).append(function)
    wanted = [
        name
        for name in _static_initializer_calls(generator, records)
        if name in originals and name not in retained
    ]
    for name in wanted:
        callees = set()
        for function in originals[name]:
            callees |= generator.entry_reachable_function_names(ast, function)
        for restored in [name, *sorted(callees)]:
            if restored in retained or restored not in originals:
                continue
            retained.add(restored)
            scoped_ast.functions.extend(deepcopy(originals[restored]))
