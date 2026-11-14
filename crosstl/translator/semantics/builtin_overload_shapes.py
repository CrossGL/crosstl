"""Calls that a user overload of a built-in's name cannot accept, because an argument has another vector shape.

A program may overload a built-in function with its own: llama.cpp's Metal sources declare
``static inline float dot(float x, float y)`` beside the library's ``dot(floatN, floatN)``, and call both. Overload
resolution sends ``dot(sumft, float4(1.0f))`` to the library, because no implicit conversion turns a vector into a
scalar (the Metal Shading Language Specification, "Implicit Type Conversions"; GLSL 4.60 section 4.1.10 converts
only between types of the same shape). The library's overloads are generic over the vector width (GLSL 4.60
section 8 writes them for ``genFType``), so where a user overload would need a scalar widened to a vector, the
built-in matches exactly and wins overload resolution too. A generator that renames such a user function must
therefore leave such a call with the built-in.

:func:`calls_only_builtins_accept` finds them: a call whose arguments, where their types are known, differ in vector
width from the parameters of every user overload of that name and argument count. A call whose argument types are
unknown is left to the user overloads, as the argument count alone decides it.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name, type_from_node
from .scoped_walk import ScopedWalker, parameter_type, program_stages

__all__ = ["calls_only_builtins_accept"]


def _functions(program):
    yield from getattr(program, "functions", None) or []
    for stage in program_stages(program):
        yield from getattr(stage, "local_functions", None) or []


def _overload_widths(program, names):
    """``{name: {arity: [widths per overload]}}``; a width is ``None`` where the parameter's shape is unknown."""
    overloads = {}
    for function in _functions(program):
        name = getattr(function, "name", None)
        if name not in names:
            continue
        parameters = getattr(function, "parameters", None) or []
        widths = []
        for parameter in parameters:
            value = type_from_node(parameter_type(parameter))
            widths.append(value.lanes if value is not None else None)
        overloads.setdefault(name, {}).setdefault(len(widths), []).append(widths)
    return overloads


class _ShapeWalker(ScopedWalker):
    def __init__(self, program, overloads):
        super().__init__(program)
        self.overloads = overloads
        self.builtin_calls = set()

    def visit(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return
        candidates = self.overloads.get(call_name(node))
        if not candidates:
            return
        arguments = list(getattr(node, "arguments", None) or [])
        signatures = candidates.get(len(arguments))
        if not signatures:
            return
        widths = []
        for argument in arguments:
            value = self.scope.infer(argument)
            widths.append(value.lanes if value is not None else None)
        if not any(_accepts(signature, widths) for signature in signatures):
            self.builtin_calls.add(id(node))


def _accepts(parameters, arguments):
    return all(
        parameter is None or argument is None or parameter == argument
        for parameter, argument in zip(parameters, arguments)
    )


def calls_only_builtins_accept(program, names) -> set[int]:
    """``id()`` of each call to a function in ``names`` that no user overload of its argument count accepts.

    Call the function before renaming, while the calls and the user functions still share their names.
    """
    if not isinstance(program, A.ShaderNode):
        return set()
    overloads = _overload_widths(program, set(names))
    if not overloads:
        return set()
    walker = _ShapeWalker(program, overloads)
    walker.run_program()
    return walker.builtin_calls
