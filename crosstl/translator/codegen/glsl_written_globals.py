"""Unqualified module-scope variables that the program writes are private globals in GLSL.

CrossGL's unqualified module-scope variable of a plain value type has one copy per invocation, which every function
the invocation calls reads and writes, the storage of a WGSL ``var<private>`` or an HLSL ``static`` global
(:mod:`..semantics.device_private_globals`). The GLSL generator writes an unqualified global as a ``uniform``, the
right form for a value the host sets and the shader only reads. GLSL forbids every store to a uniform ("uniform
variables are read-only", GLSL 4.60 section 4.3.5), so a global the program stores to became an assignment that
glslang refuses (``l-value required "counter" (can't modify a uniform)``), and GLSL ES also refuses the uniform's
initializer (``'uniform' : cannot initialize this type of qualifier``).

:func:`written_unqualified_global_ids` finds the module- and stage-scope variables without a storage qualifier
whose name some function assigns, increments, decrements, takes the address of, or passes as an ``out`` or
``inout`` argument (of a CrossGL function, or of the GLSL built-ins with output parameters). The generator writes
them without a storage qualifier, the GLSL form of a global private to each invocation (section 4.3: a global
without a storage qualifier is read-write memory of its own invocation). A write inside a function that declares a
parameter or a local of the same name stores to that declaration, so it does not count.
"""

from __future__ import annotations

from .. import ast as A
from .ast_walk import walk_ast_values

__all__ = ["written_unqualified_global_ids"]

_WRITING_UNARY_OPERATORS = frozenset({"++", "--", "&"})
_OUTPUT_QUALIFIERS = frozenset({"out", "inout"})
# GLSL built-ins whose arguments at these positions are ``out`` parameters (GLSL 4.60 sections 8.3 and 8.8).
_INTRINSIC_OUTPUT_ARGUMENTS = {
    "modf": (1,),
    "frexp": (1,),
    "sincos": (1, 2),
    "uaddCarry": (2,),
    "usubBorrow": (2,),
    "umulExtended": (2, 3),
    "imulExtended": (2, 3),
}


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _functions(program):
    yield from getattr(program, "functions", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_functions", None) or []
        entry = getattr(stage, "entry_point", None)
        if entry is not None:
            yield entry


def _root_name(expression):
    while isinstance(expression, (A.MemberAccessNode, A.ArrayAccessNode)):
        expression = (
            expression.object_expr
            if isinstance(expression, A.MemberAccessNode)
            else expression.array_expr
        )
    return expression.name if isinstance(expression, A.IdentifierNode) else None


def _callee_name(call):
    callee = getattr(call, "function", None)
    if isinstance(callee, A.IdentifierNode):
        return callee.name
    return callee if isinstance(callee, str) else getattr(call, "name", None)


def _lower_names(values):
    return {str(value).lower() for value in values or []}


def _output_positions(functions):
    positions = {
        name: set(indices) for name, indices in _INTRINSIC_OUTPUT_ARGUMENTS.items()
    }
    for function in functions:
        for index, parameter in enumerate(getattr(function, "parameters", None) or []):
            if (
                _lower_names(getattr(parameter, "qualifiers", None))
                & _OUTPUT_QUALIFIERS
            ):
                positions.setdefault(function.name, set()).add(index)
    return positions


def _written_and_declared_names(function, output_positions):
    written = set()
    declared = {
        parameter.name for parameter in getattr(function, "parameters", None) or []
    }
    for node in walk_ast_values(getattr(function, "body", None)):
        if isinstance(node, A.VariableNode):
            declared.add(node.name)
        elif isinstance(node, A.AssignmentNode):
            written.add(_root_name(node.target))
        elif isinstance(node, A.UnaryOpNode):
            if str(node.operator) in _WRITING_UNARY_OPERATORS:
                written.add(_root_name(node.operand))
        elif isinstance(node, A.FunctionCallNode):
            arguments = list(getattr(node, "arguments", None) or [])
            for index in output_positions.get(_callee_name(node), ()):
                if index < len(arguments):
                    written.add(_root_name(arguments[index]))
    written.discard(None)
    return written, declared


def written_unqualified_global_ids(program) -> set:
    """Ids of the module- and stage-scope variables without qualifiers that some function writes."""
    candidates = [
        variable
        for variable in [
            *(getattr(program, "global_variables", None) or []),
            *(
                variable
                for stage in _stages(program)
                for variable in getattr(stage, "local_variables", None) or []
            ),
        ]
        if type(variable) is A.VariableNode
        and getattr(variable, "name", None)
        and not getattr(variable, "qualifiers", None)
    ]
    if not candidates:
        return set()
    functions = [
        function
        for function in _functions(program)
        if isinstance(function, A.FunctionNode)
    ]
    output_positions = _output_positions(functions)
    written = set()
    for function in functions:
        names, declared = _written_and_declared_names(function, output_positions)
        written |= names - declared
    return {id(variable) for variable in candidates if variable.name in written}
