"""Arguments of Metal parameters that the generator declares as address-space references.

A helper-function parameter in a named address space is emitted as a reference, such as ``constant T& value``
(MSL 4.1 §4: a variable in the device or constant address space is passed by pointer or reference). A reference
binds the argument object itself, and no temporary can be created in the device or constant address space, so the
argument cannot be wrapped in a value conversion: ``f(bfloat(value))`` for a ``constant bfloat&`` parameter has no
viable overload. An argument whose type already is the parameter's type is passed unchanged.

The generator computes with 8- and 16-bit integers in a widened representation (``uint`` for ``uint8``), but an
object in the device or constant address space has the storage type (``constant uchar& value [[buffer(1)]]``). A
reference parameter in one of those address spaces is therefore declared with the storage type, so that it can bind
the buffer object; reading it yields the stored value, which the body widens as it does for any buffer read. A
program-scope constant of a narrow integer type is declared in the widened representation, so it cannot bind to
such a parameter, and the call is refused.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    PointerType,
    ReferenceType,
    UnaryOpNode,
)

__all__ = [
    "BUFFER_ADDRESS_SPACES",
    "binds_address_space_reference",
    "binds_buffer_reference",
    "buffer_reference_value_type",
    "has_parameter_value_type",
    "is_program_scope_object",
]

BUFFER_ADDRESS_SPACES = frozenset({"constant", "device"})


def binds_address_space_reference(generator, func_name, index):
    """Whether parameter ``index`` of the only overload of ``func_name`` is emitted as an address-space reference."""

    parameters = _parameters(generator, func_name)
    if parameters is None or index >= len(parameters):
        return False
    parameter = parameters[index]
    raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
    if isinstance(raw_type, ReferenceType):
        return True
    if isinstance(raw_type, (PointerType, ArrayType)) or generator.is_array_type_node(
        raw_type
    ):
        return False
    return generator.parameter_address_space(parameter) is not None


def has_parameter_value_type(generator, argument, parameter_type):
    """Whether ``argument`` has the Metal value type of ``parameter_type``, ignoring references and ``const``."""

    argument_type = generator.expression_result_type(argument)
    if argument_type is None or parameter_type is None:
        return False
    if isinstance(argument_type, ReferenceType):
        argument_type = argument_type.referenced_type
    expected = _value_type(
        generator.map_type(generator.type_name_string(parameter_type))
    )
    actual = _value_type(generator.map_type(argument_type))
    return bool(expected) and expected == actual


def _value_type(mapped):
    text = str(mapped or "").strip()
    while text.endswith("&"):
        text = text[:-1].rstrip()
    if text.startswith("const "):
        text = text[len("const ") :].lstrip()
    return text


def buffer_reference_value_type(generator, raw_type, address_space, mapped_type):
    """The type a ``device`` or ``constant`` value reference parameter declares: the storage type of a narrow integer."""

    if address_space not in BUFFER_ADDRESS_SPACES:
        return mapped_type
    return generator.metal_native_narrow_bitcast_storage_type(raw_type) or mapped_type


def binds_buffer_reference(generator, func_name, index):
    """Whether the parameter is a ``device`` or ``constant`` value reference, which binds the buffer object."""

    if not binds_address_space_reference(generator, func_name, index):
        return False
    parameter = _parameters(generator, func_name)[index]
    raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
    if isinstance(raw_type, ReferenceType):
        return False
    return generator.parameter_address_space(parameter) in BUFFER_ADDRESS_SPACES


def _parameters(generator, func_name):
    overloads = generator.function_overloads_by_name.get(func_name, [])
    if len(overloads) != 1:
        return None
    return list(
        getattr(overloads[0], "parameters", getattr(overloads[0], "params", [])) or []
    )


def is_program_scope_object(generator, argument):
    """Whether ``argument`` names a program-scope constant or variable, or an element or member of one."""

    root = _root_name(argument)
    if root is None:
        return False
    return root in (
        getattr(generator, "metal_program_scope_constant_types", None) or {}
    ) or root in (getattr(generator, "metal_program_scope_value_globals", None) or ())


def _root_name(expression):
    while True:
        if isinstance(expression, IdentifierNode):
            return expression.name
        if isinstance(expression, ArrayAccessNode):
            expression = expression.array_expr
        elif isinstance(expression, MemberAccessNode):
            expression = expression.object_expr
        elif isinstance(expression, PointerAccessNode):
            expression = expression.pointer_expr
        elif (
            isinstance(expression, UnaryOpNode)
            and expression.operator == "*"
            and not getattr(expression, "is_postfix", False)
        ):
            expression = expression.operand
        elif (
            isinstance(expression, FunctionCallNode)
            and _call_name(expression) == "buffer_load"
            and getattr(expression, "args", None)
        ):
            expression = expression.args[0]
        else:
            return None


def _call_name(call):
    name = getattr(call, "name", None)
    return name.name if isinstance(name, IdentifierNode) else name
