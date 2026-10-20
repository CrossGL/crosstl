"""Array parameters passed by value, for Metal, whose C array parameters are pointers.

GLSL, HLSL and SPIR-V pass an ``in`` array parameter by value: the callee works on its own copy, and its writes
never reach the caller's array (GLSL 4.60 section 6.1.1, "Function Calling Conventions"). The Metal emitter
declares the parameter with C syntax, ``float3 v[2]``, and C++ adjusts such a parameter to a pointer to the
caller's first element ([dcl.fct]/5), so a write in the callee changed the caller's array.

:func:`copy_written_array_parameters` gives a Metal callee that may write an ``in`` array parameter a copy of its
own: the parameter is renamed, and a local with the parameter's name and type is initialized from it at the start
of the body. A parameter counts as written when it is the root of an assignment target or an increment, or of an
argument to an ``out`` or ``inout`` parameter of a user function or to a built-in that may write memory. Parameters
the body only reads stay pointers, which reads cannot tell apart from copies. ``out`` and ``inout`` array
parameters already have the caller's array as their storage, and arrays of pointers or resources are handles
rather than values.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name, is_pure_callee
from .opaque_structs import _is_opaque
from .rewrite import child_slots, get_child

__all__ = ["copy_written_array_parameters"]

_POINTER_ARRAY_TARGETS = frozenset({"metal"})
_PARAMETER_PREFIX = "cgl_array_argument"
_WRITING_QUALIFIERS = frozenset({"out", "inout"})


def copy_written_array_parameters(program, target) -> int:
    """Give callees that write an ``in`` array parameter a local copy; returns how many parameters were copied."""
    if not isinstance(program, A.ShaderNode) or str(target or "").lower() not in _POINTER_ARRAY_TARGETS:
        return 0
    entry_points = {
        id(stage.entry_point)
        for stage in _stages(program)
        if getattr(stage, "entry_point", None) is not None
    }
    functions = list(getattr(program, "functions", None) or [])
    for stage in _stages(program):
        functions.extend(getattr(stage, "local_functions", None) or [])
    writers = _writing_positions(functions)
    names = _declared_names(program)
    copied = 0
    for function in functions:
        if id(function) in entry_points:
            continue
        statements = _body_statements(function)
        if statements is None:
            continue
        written = _written_roots(statements, writers)
        copies = []
        for parameter in getattr(function, "parameters", None) or []:
            if not _is_by_value_array(parameter) or parameter.name not in written:
                continue
            renamed = _fresh_name(names, parameter.name)
            local = A.VariableNode(
                name=parameter.name,
                var_type=parameter.param_type,
                initial_value=A.IdentifierNode(renamed),
            )
            local.source_location = getattr(parameter, "source_location", None)
            parameter.name = renamed
            copies.append(local)
        statements[0:0] = copies
        copied += len(copies)
    return copied


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _body_statements(function):
    body = getattr(function, "body", None)
    if isinstance(body, A.BlockNode):
        return body.statements
    if isinstance(body, list):
        return body
    return None


def _is_by_value_array(parameter):
    if not isinstance(getattr(parameter, "param_type", None), A.ArrayType):
        return False
    if parameter.param_type.size is None:
        return False
    element = parameter.param_type
    while isinstance(element, A.ArrayType):
        element = element.element_type
    # Pointers and resources are handles, not values GLSL copies; their arrays keep the emitter's own lowering.
    if isinstance(element, (A.PointerType, A.ReferenceType)) or _is_opaque(element):
        return False
    qualifiers = {str(qualifier).lower() for qualifier in getattr(parameter, "qualifiers", None) or []}
    return not qualifiers & _WRITING_QUALIFIERS


def _writing_positions(functions):
    """Argument positions, per user function name, that some overload declares ``out`` or ``inout``."""
    positions = {}
    for function in functions:
        written = positions.setdefault(function.name, set())
        for index, parameter in enumerate(getattr(function, "parameters", None) or []):
            qualifiers = {str(qualifier).lower() for qualifier in getattr(parameter, "qualifiers", None) or []}
            if qualifiers & _WRITING_QUALIFIERS:
                written.add(index)
    return positions


def _written_roots(statements, writers):
    written = set()
    stack, seen = list(statements), set()
    while stack:
        node = stack.pop()
        if id(node) in seen or not isinstance(node, A.ASTNode):
            continue
        seen.add(id(node))
        if isinstance(node, A.AssignmentNode):
            written.add(_root_name(node.target))
        elif isinstance(node, A.UnaryOpNode) and node.op in ("++", "--"):
            written.add(_root_name(node.operand))
        elif isinstance(node, A.FunctionCallNode):
            name = call_name(node)
            for index, argument in enumerate(node.arguments or []):
                if name in writers:
                    if index in writers[name]:
                        written.add(_root_name(argument))
                elif not is_pure_callee(name):
                    written.add(_root_name(argument))
        stack.extend(get_child(node, field, key) for field, key in child_slots(node))
    written.discard(None)
    return written


def _root_name(expr):
    while True:
        if isinstance(expr, A.IdentifierNode):
            return expr.name
        if isinstance(expr, A.ArrayAccessNode):
            expr = expr.array_expr
        elif isinstance(expr, A.MemberAccessNode):
            expr = expr.object_expr
        elif isinstance(expr, A.SwizzleNode):
            expr = expr.vector_expr
        else:
            return None


def _declared_names(program):
    names = set()
    stack, seen = [program], set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        name = getattr(current, "name", None)
        if isinstance(name, str):
            names.add(name)
        stack.extend(get_child(current, field, key) for field, key in child_slots(current))
    return names


def _fresh_name(names, base):
    index = 0
    while f"{_PARAMETER_PREFIX}_{base}_{index}" in names:
        index += 1
    name = f"{_PARAMETER_PREFIX}_{base}_{index}"
    names.add(name)
    return name
