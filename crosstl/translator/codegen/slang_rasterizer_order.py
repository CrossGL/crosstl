"""Slang rasterizer-ordered views for CrossGL's ``@rasterizer_ordered`` resources.

CrossGL marks a storage buffer or storage image whose accesses must happen in rasterization order with
``@ rasterizer_ordered`` (HLSL's rasterizer-ordered views, GLSL's GL_ARB_fragment_shader_interlock critical
section). Slang has HLSL's ``RasterizerOrdered*`` resource types, which slangc compiles for SPIR-V to the
``PixelInterlockOrderedEXT`` execution mode with ``OpBeginInvocationInterlockEXT`` and
``OpEndInvocationInterlockEXT`` around the fragment shader. The generator declared the plain ``RW*`` type, so
overlapping fragments raced on the resource.

A ``@ rasterizer_ordered`` declaration whose Slang type has no rasterizer-ordered form, such as a read-only
resource, is refused.

A function parameter that receives a rasterizer-ordered resource is declared with the rasterizer-ordered type too:
slangc converts neither a ``RasterizerOrderedTexture2D`` to a ``RWTexture2D`` parameter nor the reverse (E30019,
"type mismatch"). :func:`rasterizer_ordered_parameters` finds those parameters, through helpers that pass the
resource on, and a parameter that receives both a rasterizer-ordered and a plain resource is refused, since one
Slang parameter type cannot take both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    VariableNode,
)
from .array_utils import split_array_type_suffix

__all__ = [
    "RasterizerOrderedParameters",
    "is_slang_rasterizer_ordered",
    "rasterizer_ordered_parameters",
    "slang_rasterizer_ordered_parameter_type",
    "slang_rasterizer_ordered_type",
]

_RASTERIZER_ORDERED_TYPES = {
    "RWTexture1D": "RasterizerOrderedTexture1D",
    "RWTexture1DArray": "RasterizerOrderedTexture1DArray",
    "RWTexture2D": "RasterizerOrderedTexture2D",
    "RWTexture2DArray": "RasterizerOrderedTexture2DArray",
    "RWTexture3D": "RasterizerOrderedTexture3D",
    "RWBuffer": "RasterizerOrderedBuffer",
    "RWStructuredBuffer": "RasterizerOrderedStructuredBuffer",
    "RWByteAddressBuffer": "RasterizerOrderedByteAddressBuffer",
}
_ORDERED_TYPES = frozenset(_RASTERIZER_ORDERED_TYPES.values())


def is_slang_rasterizer_ordered(node) -> bool:
    """Whether the declaration ``node`` carries ``@ rasterizer_ordered``."""
    return any(
        str(getattr(attribute, "name", "") or "").lower() == "rasterizer_ordered"
        for attribute in getattr(node, "attributes", None) or ()
    )


def _ordered_form(mapped_type):
    """``(resource name, rasterizer-ordered spelling or None)`` for the Slang type ``mapped_type``."""
    base_type, array_suffix = (
        split_array_type_suffix(mapped_type)
        if "[" in mapped_type and mapped_type.endswith("]")
        else (mapped_type, "")
    )
    resource_name, bracket, generic_arguments = base_type.partition("<")
    resource_name = resource_name.strip()
    if resource_name in _ORDERED_TYPES:
        return resource_name, mapped_type
    ordered = _RASTERIZER_ORDERED_TYPES.get(resource_name)
    if ordered is None:
        return resource_name, None
    return resource_name, f"{ordered}{bracket}{generic_arguments}{array_suffix}"


def slang_rasterizer_ordered_type(codegen, mapped_type, node):
    """``mapped_type`` as its ``RasterizerOrdered*`` form when ``node`` is ``@ rasterizer_ordered``."""
    if node is None or not isinstance(mapped_type, str):
        return mapped_type
    if not is_slang_rasterizer_ordered(node):
        return mapped_type
    resource_name, ordered = _ordered_form(mapped_type)
    if ordered is None:
        name = getattr(node, "name", None)
        return codegen.refuse_lowering(
            f"Slang @rasterizer_ordered: '{name}' is a {resource_name}, which has no rasterizer-ordered "
            "form; only read-write buffers and storage textures do",
            mapped_type,
        )
    return ordered


@dataclass(frozen=True)
class RasterizerOrderedParameters:
    """The ``(function name, parameter name)`` pairs that receive a rasterizer-ordered resource.

    ``mixed`` holds the pairs that also receive a resource that is not rasterizer-ordered.
    """

    ordered: frozenset = field(default_factory=frozenset)
    mixed: frozenset = field(default_factory=frozenset)


def _argument_name(argument):
    while isinstance(argument, ArrayAccessNode):
        argument = argument.array_expr
    return argument.name if isinstance(argument, IdentifierNode) else None


def _callee_name(call):
    callee = call.function
    if isinstance(callee, IdentifierNode):
        callee = callee.name
    return callee if isinstance(callee, str) else None


def _call_arguments(functions_by_name, functions):
    """``(caller, callee, arity, index, argument name)`` for each argument of a call to a user function."""
    arguments = []
    for function in functions:
        for node in function.walk():
            if not isinstance(node, FunctionCallNode):
                continue
            callee = _callee_name(node)
            if callee not in functions_by_name:
                continue
            call_arguments = list(node.arguments or [])
            for index, argument in enumerate(call_arguments):
                arguments.append(
                    (
                        function.name,
                        callee,
                        len(call_arguments),
                        index,
                        _argument_name(argument),
                    )
                )
    return arguments


def _receiving_parameters(functions_by_name, callee, arity, index):
    for function in functions_by_name[callee]:
        parameters = list(getattr(function, "parameters", None) or [])
        if len(parameters) == arity:
            yield (callee, parameters[index].name)


def rasterizer_ordered_parameters(ast) -> RasterizerOrderedParameters:
    """The function parameters of ``ast`` that receive a rasterizer-ordered resource, directly or through helpers."""
    nodes = list(ast.walk())
    functions = [node for node in nodes if isinstance(node, FunctionNode)]
    functions_by_name = {}
    for function in functions:
        functions_by_name.setdefault(function.name, []).append(function)
    ordered_resources = {
        node.name
        for node in nodes
        if isinstance(node, VariableNode) and is_slang_rasterizer_ordered(node)
    }
    ordered = {
        (function.name, parameter.name)
        for function in functions
        for parameter in getattr(function, "parameters", None) or []
        if is_slang_rasterizer_ordered(parameter)
    }
    arguments = _call_arguments(functions_by_name, functions)

    def is_ordered_argument(caller, name):
        return name is not None and (
            (caller, name) in ordered or name in ordered_resources
        )

    changed = True
    while changed:
        changed = False
        for caller, callee, arity, index, name in arguments:
            if not is_ordered_argument(caller, name):
                continue
            for key in _receiving_parameters(functions_by_name, callee, arity, index):
                if key not in ordered:
                    ordered.add(key)
                    changed = True

    mixed = {
        key
        for caller, callee, arity, index, name in arguments
        if not is_ordered_argument(caller, name)
        for key in _receiving_parameters(functions_by_name, callee, arity, index)
        if key in ordered
    }
    return RasterizerOrderedParameters(frozenset(ordered), frozenset(mixed))


def slang_rasterizer_ordered_parameter_type(codegen, function_name, mapped_type, param):
    """The Slang type that declares parameter ``param`` of ``function_name``, whose resource type is ``mapped_type``.

    An explicit ``@ rasterizer_ordered`` parameter takes the rasterizer-ordered form as a declaration does. A
    parameter that receives rasterizer-ordered resources takes it when its type has one; one that also receives a
    plain resource is refused.
    """
    if param is None or not isinstance(mapped_type, str):
        return mapped_type
    if is_slang_rasterizer_ordered(param):
        return slang_rasterizer_ordered_type(codegen, mapped_type, param)
    parameters = getattr(codegen, "slang_rasterizer_ordered_parameters", None)
    key = (function_name, getattr(param, "name", None))
    if parameters is None or key not in parameters.ordered:
        return mapped_type
    _, ordered = _ordered_form(mapped_type)
    if ordered is None:
        return mapped_type
    if key in parameters.mixed:
        return codegen.refuse_lowering(
            f"Slang parameter '{key[1]}' of '{function_name}' receives both a rasterizer-ordered and a plain "
            "resource; slangc converts neither to the other (E30019), so one parameter type cannot take both",
            mapped_type,
        )
    return ordered
