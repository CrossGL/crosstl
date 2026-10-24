"""Prove values identical across the invocations of one software subgroup.

Software subgroups lower collectives to workgroup barriers, so every
invocation must reach each of them. ``converge_subgroup_early_exits``
therefore predicates the statements around collectives instead of skipping
them, and invocations that left still run the collectives. A relative
shuffle tolerates that, but a reduction, vote or broadcast combines every
lane of a subgroup. It still refines the source when the lanes of each
subgroup agree on the predicate: a subgroup then runs the collective with
all of its lanes, as in the source, or publishes nothing.

A name is *subgroup-uniform* in a function when every invocation of one
subgroup holds the same value for it:

* entry parameters bound to a workgroup or subgroup system value
  (``gl_WorkGroupID``, ``gl_SubgroupID`` and the like) or to read-only
  ``constant`` memory;
* helper parameters whose argument is subgroup-uniform at every call from a
  reachable function;
* locals each of whose declarations has a subgroup-uniform initializer and
  that nothing assigns, increments, takes the address of or passes to a
  user function.

Expressions combine such names with literals, side-effect-free operators,
conversions, a few pure builtins, pure user functions and element reads of
read-only ``constant`` memory at uniform positions. The analysis is
conservative: anything else is treated as varying.
"""

import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    ConstructorNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

# System values every invocation of a software subgroup shares.
SUBGROUP_UNIFORM_SEMANTICS = frozenset(
    {
        "gl_WorkGroupID",
        "gl_NumWorkGroups",
        "gl_WorkGroupSize",
        "gl_NumSubgroups",
        "gl_SubgroupSize",
        "gl_SubgroupID",
        "SV_GroupID",
    }
)

_UNIFORM_OPERATORS = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "%",
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
        "<<",
        ">>",
    }
)
# Builtins whose result depends only on their arguments.
_PURE_BUILTINS = frozenset({"min", "max", "clamp", "abs", "select"})
_CONVERSIONS = re.compile(
    r"(?:u?int|float|half|bool|bfloat|u?char|u?short|u?long)"
    r"(?:8|16|32|64)?(?:_t)?(?:[234])?"
)
_WRITING_UNARY = frozenset({"++", "--", "&"})


def _call_name(node):
    function = node.function
    return function if isinstance(function, str) else getattr(function, "name", None)


def _root_name(node):
    while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
        node = (
            node.array_expr if isinstance(node, ArrayAccessNode) else node.object_expr
        )
    return node.name if isinstance(node, IdentifierNode) else None


def _semantic(parameter):
    semantic = getattr(parameter, "semantic", None)
    if semantic:
        return str(semantic)
    for attribute in getattr(parameter, "attributes", None) or []:
        name = getattr(attribute, "name", None)
        if isinstance(name, str) and (
            name.startswith("gl_") or name.upper().startswith("SV_")
        ):
            return name
    return None


def _qualifiers(parameter):
    return set(getattr(parameter, "qualifiers", None) or []) | set(
        getattr(getattr(parameter, "param_type", None), "qualifiers", None) or []
    )


def _constant_memory(parameter):
    """Whether ``parameter`` addresses read-only ``constant`` memory."""
    kind = parameter.param_type
    return (
        isinstance(kind, (PointerType, ReferenceType))
        and getattr(kind, "address_space", None) == "constant"
        and not (isinstance(kind, ReferenceType) and kind.is_mutable)
    )


def _value_parameter(parameter):
    """Whether a callee parameter receives a copy its caller cannot see change."""
    return not isinstance(parameter.param_type, (PointerType, ReferenceType)) and not (
        _qualifiers(parameter) & {"out", "inout", "thread", "threadgroup", "device"}
    )


def _entry_uniform(parameter):
    if _semantic(parameter) in SUBGROUP_UNIFORM_SEMANTICS:
        return True
    if _constant_memory(parameter):
        return True
    return "constant" in _qualifiers(parameter) and _value_parameter(parameter)


class _Uniformity:
    """Subgroup-uniform names of one function."""

    def __init__(self, names, constant_roots, pure, user_functions, map_operator):
        self.names = frozenset(names)
        self.constant_roots = frozenset(constant_roots)
        self.pure = frozenset(pure)
        self.user_functions = frozenset(user_functions)
        self.map_operator = map_operator

    def with_names(self, names):
        return _Uniformity(
            self.names | set(names),
            self.constant_roots,
            self.pure,
            self.user_functions,
            self.map_operator,
        )

    def expression(self, node):
        """Whether ``node`` has one value in every invocation of a subgroup."""
        if isinstance(node, LiteralNode):
            return True
        if isinstance(node, IdentifierNode):
            return node.name in self.names
        if isinstance(node, CastNode):
            return self.expression(node.expression)
        if isinstance(node, MemberAccessNode):
            return self.expression(node.object_expr)
        if isinstance(node, ArrayAccessNode):
            return (
                _root_name(node) in self.constant_roots
                and self.expression(node.array_expr)
                and self.expression(node.index_expr)
            )
        if isinstance(node, UnaryOpNode):
            return self.map_operator(node.operator) in {
                "+",
                "-",
                "!",
                "~",
            } and self.expression(node.operand)
        if isinstance(node, BinaryOpNode):
            return self.map_operator(node.operator) in _UNIFORM_OPERATORS and all(
                self.expression(part) for part in (node.left, node.right)
            )
        if isinstance(node, TernaryOpNode):
            return all(
                self.expression(part)
                for part in (node.condition, node.true_expr, node.false_expr)
            )
        if isinstance(node, ConstructorNode):
            return all(self.expression(part) for part in node.arguments or [])
        if isinstance(node, FunctionCallNode):
            name = _call_name(node)
            if not isinstance(name, str) or getattr(node, "generic_args", None):
                return False
            if name in self.user_functions:
                known = name in self.pure
            else:
                known = name in _PURE_BUILTINS or _CONVERSIONS.fullmatch(name)
            return bool(known) and all(
                self.expression(argument) for argument in node.arguments
            )
        return False


def _written_names(function, functions, walk_ast, map_operator):
    """Names ``function`` assigns, increments, addresses or lends to callees."""
    written = set()
    for node in walk_ast(function.body):
        if isinstance(node, AssignmentNode):
            written.add(_root_name(node.target))
        elif isinstance(node, UnaryOpNode) and (
            map_operator(node.operator) in _WRITING_UNARY
        ):
            written.add(_root_name(node.operand))
        elif isinstance(node, FunctionCallNode) and _call_name(node) in functions:
            for index, argument in enumerate(node.arguments):
                if any(
                    index >= len(callee.parameters or [])
                    or not _value_parameter(callee.parameters[index])
                    for callee in functions[_call_name(node)]
                ):
                    written.add(_root_name(argument))
    written.discard(None)
    return written


def _local_uniformity(
    function, parameters, functions, pure, walk_ast, map_operator, written
):
    """The ``_Uniformity`` of ``function`` given its uniform ``parameters``."""
    all_parameters = {parameter.name for parameter in function.parameters or []}
    constant_roots = {
        parameter.name
        for parameter in function.parameters or []
        if parameter.name in parameters
        and _constant_memory(parameter)
        and parameter.name not in written
    }
    declarations = {}
    for node in walk_ast(function.body):
        if isinstance(node, VariableNode):
            declarations.setdefault(node.name, []).append(node)
    names = set(parameters) - written
    while True:
        uniformity = _Uniformity(names, constant_roots, pure, functions, map_operator)
        added = {
            name
            for name, nodes in declarations.items()
            if name not in names
            and name not in written
            and name not in all_parameters
            and all(
                node.initial_value is not None
                and not isinstance(node.var_type, (PointerType, ReferenceType))
                and uniformity.expression(node.initial_value)
                for node in nodes
            )
        }
        if not added:
            return uniformity
        names |= added


def subgroup_uniformity(ast, entries, walk_ast, map_operator, pure=frozenset()):
    """Map ``id(function)`` to the ``_Uniformity`` of each reachable function.

    ``entries`` are the code generator's ``(stage, function, config)``
    records; only compute entries seed the analysis. ``pure`` names user
    overload sets that only compute a value from their arguments.
    """
    functions = {}
    for node in walk_ast(ast):
        if isinstance(node, FunctionNode):
            functions.setdefault(node.name, []).append(node)
    roots = [entry for stage, entry, _config in entries if stage == "compute"]
    reachable, pending = {}, list(roots)
    while pending:
        function = pending.pop()
        if id(function) in reachable:
            continue
        reachable[id(function)] = function
        for node in walk_ast(function.body):
            if isinstance(node, FunctionCallNode) and _call_name(node) in functions:
                pending.extend(functions[_call_name(node)])
    root_ids = {id(entry) for entry in roots}
    written = {
        key: _written_names(function, functions, walk_ast, map_operator)
        for key, function in reachable.items()
    }
    # Optimistic for helpers: parameters lose uniformity at varying calls.
    parameters = {}
    for key, function in reachable.items():
        parameters[key] = {
            parameter.name
            for parameter in function.parameters or []
            if (
                _entry_uniform(parameter)
                if key in root_ids
                else _value_parameter(parameter) or _constant_memory(parameter)
            )
        }
    while True:
        result = {
            key: _local_uniformity(
                function,
                parameters[key],
                functions,
                pure,
                walk_ast,
                map_operator,
                written[key],
            )
            for key, function in reachable.items()
        }
        changed = False
        for key, function in reachable.items():
            for node in walk_ast(function.body):
                if not (
                    isinstance(node, FunctionCallNode) and _call_name(node) in functions
                ):
                    continue
                for callee in functions[_call_name(node)]:
                    if id(callee) in root_ids or id(callee) not in parameters:
                        continue
                    for index, parameter in enumerate(callee.parameters or []):
                        if parameter.name not in parameters[id(callee)]:
                            continue
                        if index >= len(node.arguments) or not result[key].expression(
                            node.arguments[index]
                        ):
                            parameters[id(callee)].discard(parameter.name)
                            changed = True
        if not changed:
            return result
