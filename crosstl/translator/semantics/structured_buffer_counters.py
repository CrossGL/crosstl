"""Hidden ``RWStructuredBuffer`` counters as explicit counter buffers, for targets that have no hidden counter.

HLSL lets a UAV created with a counter (``D3D11_BUFFER_UAV_FLAG_COUNTER``) bump that counter through
``IncrementCounter()``, which returns the value before the increment, and ``DecrementCounter()``, which returns the
value after the decrement (the HLSL reference pages for both methods). CrossGL spells them
``buffer_increment_counter(b)`` and ``buffer_decrement_counter(b)``.

Metal, GLSL, CUDA and HIP have no hidden counter. A host binds it as a buffer of its own, as DXC does for SPIR-V
(a separate ``counter.var.<name>`` storage buffer) and as CrossTL already does for the counter of an
``AppendStructuredBuffer``. :func:`materialize_structured_buffer_counters` makes that buffer explicit in the AST:

- each ``RWStructuredBuffer`` whose counter is used gets a sibling ``RWStructuredBuffer<uint> <name>Counter``
  declaration (an array of buffers gets an array of counters of the same size), with no binding of its own, so the
  target allocates the next free one;
- ``buffer_increment_counter(b)`` becomes ``atomicAdd(bCounter[0], 1u)`` and ``buffer_decrement_counter(b)``
  becomes ``atomicAdd(bCounter[0], 4294967295u) - 1u``, the modular decrement that returns the new value;
- a function whose ``RWStructuredBuffer`` parameter reaches a counter gets a counter parameter right after it, and
  every call passes the counter of its argument.

A use the pass cannot follow back to one declared buffer, such as a counter of a buffer that is not an
``RWStructuredBuffer``, a buffer argument other than a name or an indexed name, an index with side effects that would
have to be evaluated twice, or an overload that cannot be told apart by its arity, is reported through
``refuse(reason)`` and left in place.
"""

from __future__ import annotations

from copy import copy, deepcopy

from ..ast import (
    AST_CHILD_FIELD_EXCLUSIONS,
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    NamedType,
    ParameterNode,
    PrimitiveType,
    UnaryOpNode,
    VariableNode,
)

__all__ = ["COUNTER_INTRINSICS", "materialize_structured_buffer_counters"]

COUNTER_INTRINSICS = frozenset({"buffer_increment_counter", "buffer_decrement_counter"})
_COUNTED_KIND = "RWStructuredBuffer"
_UINT_MAX = 0xFFFFFFFF


def _call_name(node):
    if not isinstance(node, FunctionCallNode):
        return None
    function = getattr(node, "function", None)
    if isinstance(function, str):
        return function
    name = getattr(function, "name", None)
    return name if isinstance(name, str) else None


def _buffer_kind(type_node):
    """The resource name of a (possibly arrayed) buffer type, such as ``RWStructuredBuffer``, or ``None``."""
    while isinstance(type_node, ArrayType):
        type_node = type_node.element_type
    if isinstance(type_node, NamedType):
        return type_node.name
    if isinstance(type_node, str):
        return type_node.split("<", 1)[0].split("[", 1)[0].strip() or None
    return None


def _declared_type(node):
    for attribute in ("var_type", "vtype", "param_type"):
        value = getattr(node, attribute, None)
        if value is not None:
            return value
    return None


def _counter_type(buffer_type):
    """``RWStructuredBuffer<uint>``, arrayed like ``buffer_type``."""
    counter = NamedType(_COUNTED_KIND, [PrimitiveType("uint")])
    if isinstance(buffer_type, ArrayType):
        arrayed = copy(buffer_type)
        arrayed.element_type = _counter_type(buffer_type.element_type)
        return arrayed
    return counter


def _uint(value):
    return LiteralNode(value, PrimitiveType("uint"))


def _children(node):
    for field, value in vars(node).items():
        if field in AST_CHILD_FIELD_EXCLUSIONS:
            continue
        yield field, value


def _walk(root):
    """Every AST node reachable from ``root`` once, in pre-order, without following metadata backlinks."""
    visited, stack = set(), [root]
    while stack:
        value = stack.pop()
        if isinstance(value, (list, tuple, set)):
            stack.extend(reversed(list(value)))
            continue
        if isinstance(value, dict):
            stack.extend(reversed(list(value.values())))
            continue
        if not isinstance(value, ASTNode) or id(value) in visited:
            continue
        visited.add(id(value))
        yield value
        stack.extend(reversed([child for _, child in _children(value)]))


def _replace_node(root, old, new):
    """Point every field and list slot that holds ``old`` (aliases included) at ``new``."""
    for node in _walk(root):
        for field, value in list(_children(node)):
            if value is old:
                setattr(node, field, new)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if item is old:
                        value[index] = new


def _has_side_effects(expr):
    for node in _walk(expr):
        if isinstance(node, (FunctionCallNode, AssignmentNode)):
            return True
        if isinstance(node, UnaryOpNode) and getattr(node, "operator", None) in {
            "++",
            "--",
        }:
            return True
    return False


class _Program:
    """Functions, buffer declarations and scopes of one program, with the counters the pass decided on."""

    def __init__(self, ast, refuse):
        self.ast = ast
        self.refuse = refuse or (lambda _reason: None)
        self.global_decls = {}
        self.functions = (
            []
        )  # (function, stage-local declarations by name), each function once
        self.overloads = {}
        self.stage_lists = []
        for node in getattr(ast, "global_variables", None) or []:
            self.global_decls.setdefault(getattr(node, "name", None), node)
        for function in getattr(ast, "functions", None) or []:
            self._add_function(function, {})
        for stage in (getattr(ast, "stages", None) or {}).values():
            local_variables = getattr(stage, "local_variables", None) or []
            self.stage_lists.append(local_variables)
            scope = {}
            for node in local_variables:
                scope.setdefault(getattr(node, "name", None), node)
            for function in getattr(stage, "local_functions", None) or []:
                self._add_function(function, scope)
            entry = getattr(stage, "entry_point", None)
            if entry is not None:
                self._add_function(entry, scope)
        self.position = {
            id(function): i for i, (function, _) in enumerate(self.functions)
        }
        self.used_names = {
            node.name
            for node in _walk(ast)
            if isinstance(getattr(node, "name", None), str)
        }
        self.counted_decls = set()  # names of declared buffers whose counter is used
        self.counted_params = set()  # (function position, parameter name)

    def _add_function(self, function, stage_scope):
        if any(known is function for known, _ in self.functions):
            return
        self.functions.append((function, stage_scope))
        self.overloads.setdefault(getattr(function, "name", None), []).append(function)

    @staticmethod
    def parameters(function):
        return list(getattr(function, "parameters", None) or [])

    def local_names(self, function):
        names = {getattr(p, "name", None) for p in self.parameters(function)}
        for node in _walk(getattr(function, "body", None)):
            if isinstance(node, VariableNode):
                names.add(node.name)
        return names

    def resolve(self, function, stage_scope, locals_, expr):
        """``(key, index)`` of the counter a buffer operand needs, or the reason it has none.

        ``key`` is ``("param", function position, name)`` or ``("decl", name)``; ``index`` selects one buffer of an
        array of buffers, else it is ``None``.
        """
        index, base = None, expr
        if isinstance(expr, ArrayAccessNode):
            base, index = expr.array_expr, expr.index_expr
        if not isinstance(base, IdentifierNode):
            return f"buffer operand '{expr}' is not a buffer name or an indexed buffer name"
        name = base.name
        for parameter in self.parameters(function):
            if parameter.name == name:
                if _buffer_kind(_declared_type(parameter)) != _COUNTED_KIND:
                    return f"'{name}' is not an {_COUNTED_KIND}"
                return ("param", self.position[id(function)], name), index
        if name in locals_:
            return f"'{name}' is a local variable, not a declared {_COUNTED_KIND}"
        declaration = stage_scope.get(name, self.global_decls.get(name))
        if declaration is None:
            return f"'{name}' is not a declared buffer"
        if _buffer_kind(_declared_type(declaration)) != _COUNTED_KIND:
            return f"'{name}' is not an {_COUNTED_KIND}"
        return ("decl", name), index

    def callee(self, call):
        """The one user function a call reaches by name and arity, ``None`` for a built-in, or a refusal reason."""
        candidates = self.overloads.get(_call_name(call))
        if not candidates:
            return None
        arity = len(call.arguments)
        matching = [f for f in candidates if len(self.parameters(f)) == arity]
        if len(matching) > 1:
            return f"overloads of '{_call_name(call)}' with {arity} parameters"
        return matching[0] if matching else None

    def counted_indices(self, function):
        position = self.position[id(function)]
        return [
            index
            for index, parameter in enumerate(self.parameters(function))
            if (position, parameter.name) in self.counted_params
        ]

    def mark(self, key):
        """Record that the buffer ``key`` names needs its counter; ``True`` when that is new."""
        target, item = (
            (self.counted_params, key[1:])
            if key[0] == "param"
            else (self.counted_decls, key[1])
        )
        if item in target:
            return False
        target.add(item)
        return True


def _analyze(program):
    """Mark every buffer whose counter is reached, following buffer arguments into callees to a fixed point."""
    changed = True
    while changed:
        changed = False
        for function, stage_scope in program.functions:
            locals_ = program.local_names(function)
            for node in _walk(getattr(function, "body", None)):
                if not isinstance(node, FunctionCallNode):
                    continue
                if _call_name(node) in COUNTER_INTRINSICS:
                    operands = node.arguments[:1] if len(node.arguments) == 1 else []
                else:
                    callee = program.callee(node)
                    if not isinstance(callee, ASTNode):
                        continue
                    operands = [
                        node.arguments[i] for i in program.counted_indices(callee)
                    ]
                for operand in operands:
                    resolved = program.resolve(function, stage_scope, locals_, operand)
                    if isinstance(resolved, tuple):
                        changed |= program.mark(resolved[0])


def _counter_names(program):
    """A fresh counter name for every counted declaration and parameter, chosen in a stable order."""
    names, taken = {}, set(program.used_names)

    def fresh(base, reserve):
        candidate, suffix = f"{base}Counter", 1
        while candidate in taken:
            candidate = f"{base}Counter_{suffix}"
            suffix += 1
        if reserve:
            taken.add(candidate)
        return candidate

    for name in sorted(program.counted_decls):
        names[("decl", name)] = fresh(name, reserve=True)
    # A parameter's counter is local to its function, so only program names and buffer counters are avoided.
    for position, parameter_name in sorted(program.counted_params):
        names[("param", position, parameter_name)] = fresh(
            parameter_name, reserve=False
        )
    return names


def _counter_operand(names, key, index, copy_index=False):
    counter = IdentifierNode(names[key])
    if index is None:
        return counter
    return ArrayAccessNode(counter, deepcopy(index) if copy_index else index)


def _declare_counters(program, names):
    for declarations in [
        getattr(program.ast, "global_variables", None)
    ] + program.stage_lists:
        for position in range(len(declarations or []) - 1, -1, -1):
            node = declarations[position]
            name = getattr(node, "name", None)
            if name not in program.counted_decls:
                continue
            if _buffer_kind(_declared_type(node)) != _COUNTED_KIND:
                continue
            counter_type = _counter_type(_declared_type(node))
            declarations.insert(
                position + 1, VariableNode(names[("decl", name)], counter_type)
            )


def _add_counter_parameters(program, names, counted_indices):
    for function, _stage_scope in program.functions:
        position = program.position[id(function)]
        parameters = function.parameters
        for index in reversed(counted_indices[id(function)]):
            parameter = parameters[index]
            counter = ParameterNode(
                names[("param", position, parameter.name)],
                _counter_type(_declared_type(parameter)),
            )
            parameters.insert(index + 1, counter)


def _lower_counter_call(program, names, function, stage_scope, locals_, node):
    """The explicit atomic for one counter call, or the reason it has none."""
    name = _call_name(node)
    if len(node.arguments) != 1:
        return f"{name} takes exactly one buffer operand"
    resolved = program.resolve(function, stage_scope, locals_, node.arguments[0])
    if not isinstance(resolved, tuple):
        return f"{name}: {resolved}"
    element = ArrayAccessNode(_counter_operand(names, *resolved), _uint(0))
    if name == "buffer_increment_counter":
        return FunctionCallNode(IdentifierNode("atomicAdd"), [element, _uint(1)])
    previous = FunctionCallNode(
        IdentifierNode("atomicAdd"), [element, _uint(_UINT_MAX)]
    )
    return BinaryOpNode(previous, "-", _uint(1))


def _pass_counters(
    program, names, function, stage_scope, locals_, node, counted_indices
):
    """Insert the counters a user-function call must pass; returns refusal reasons."""
    callee = program.callee(node)
    if isinstance(callee, str):
        if any(
            counted_indices[id(f)] for f in program.overloads.get(_call_name(node), [])
        ):
            return [f"buffer counter passed to {callee}"]
        return []
    if callee is None:
        return []
    reasons = []
    for index in reversed(counted_indices[id(callee)]):
        resolved = program.resolve(
            function, stage_scope, locals_, node.arguments[index]
        )
        if not isinstance(resolved, tuple):
            reasons.append(f"buffer counter for '{callee.name}': {resolved}")
            continue
        if resolved[1] is not None and _has_side_effects(resolved[1]):
            reasons.append(
                f"buffer counter for '{callee.name}': the buffer index has side effects"
            )
            continue
        operand = _counter_operand(names, *resolved, copy_index=True)
        node.arguments.insert(index + 1, operand)
    return reasons


def _rewrite_function(program, names, function, stage_scope, counted_indices):
    """Lower the counter calls of one function and pass counters to its callees; returns refusal reasons."""
    reasons = []
    locals_ = program.local_names(function)
    # Innermost calls first, so an outer rewrite reuses operands that are already lowered.
    for node in reversed(list(_walk(getattr(function, "body", None)))):
        if not isinstance(node, FunctionCallNode):
            continue
        if _call_name(node) in COUNTER_INTRINSICS:
            lowered = _lower_counter_call(
                program, names, function, stage_scope, locals_, node
            )
            if isinstance(lowered, str):
                reasons.append(lowered)
            else:
                _replace_node(function, node, lowered)
            continue
        reasons.extend(
            _pass_counters(
                program, names, function, stage_scope, locals_, node, counted_indices
            )
        )
    return reasons


def _uses_counters(ast):
    return any(_call_name(node) in COUNTER_INTRINSICS for node in _walk(ast))


def materialize_structured_buffer_counters(ast, refuse=None):
    """``ast`` with every hidden ``RWStructuredBuffer`` counter it uses made an explicit counter buffer.

    ``ast`` is not mutated; a program that uses no counter is returned as it is. ``refuse`` is called with the
    reason for each use that cannot be lowered, which stays in place.
    """
    if not _uses_counters(ast):
        return ast
    result = deepcopy(ast)
    program = _Program(result, refuse)
    _analyze(program)
    names = _counter_names(program)
    # Taken before any parameter is inserted, while every call still has its source arity.
    counted_indices = {
        id(function): program.counted_indices(function)
        for function, _ in program.functions
    }
    reasons = []
    for function, stage_scope in program.functions:
        reasons.extend(
            _rewrite_function(program, names, function, stage_scope, counted_indices)
        )
    _add_counter_parameters(program, names, counted_indices)
    _declare_counters(program, names)
    for reason in dict.fromkeys(reasons):
        program.refuse(reason)
    return result
