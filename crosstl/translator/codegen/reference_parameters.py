"""Pass ``out`` and ``inout`` value parameters by reference in CUDA and HIP.

CUDA and HIP have no ``out``/``inout`` parameter qualifiers. A helper function
that writes such a parameter must receive a reference, otherwise the write is
made to a local copy and never reaches the caller.

Every value type is declared ``T&``: scalars, vectors, matrices, structs,
enums and half vectors alike. Arrays decay to pointers, so writes through them
already reach the caller, and pointers, references and resources such as
structured buffers and textures keep their declaration, as
``value_array_parameters`` treats array elements. A struct the program
declares is a value whatever its name.

A reference binds only to an lvalue, and CUDA and HIP build a swizzle of
several components, such as ``v.xy``, as a new vector. Passing one to an
``out`` or ``inout`` parameter is refused instead of being written as a call
that does not compile.
"""

from ..ast import ArrayAccessNode, ArrayType, FunctionType, MemberAccessNode
from .array_utils import split_array_type_suffix
from .value_array_parameters import (
    _call_name,
    copyable_element_type,
    parameter_qualifier_names,
)

WRITABLE_PARAMETER_QUALIFIERS = frozenset({"out", "inout"})


def parameter_is_writable(parameter):
    """Return whether ``parameter`` is declared ``out`` or ``inout``."""
    return bool(parameter_qualifier_names(parameter) & WRITABLE_PARAMETER_QUALIFIERS)


def is_reference_value_type(raw_type, type_name, struct_names=frozenset()):
    """Return whether a written parameter of this type is declared ``T&``.

    Arrays and function types keep their declaration, and so do pointers,
    references and resources (``copyable_element_type``).
    """
    if isinstance(raw_type, (ArrayType, FunctionType)):
        return False
    _base_type, array_suffix = split_array_type_suffix(type_name)
    if array_suffix:
        return False
    return copyable_element_type(raw_type, type_name, struct_names)


def reference_declarator(declaration, name):
    """Turn ``T name`` into ``T& name``."""
    suffix = f" {name}"
    if not declaration.endswith(suffix):
        raise ValueError(
            f"cannot declare parameter '{name}' as a reference: '{declaration}'"
        )
    return f"{declaration[: -len(suffix)]}& {name}"


def _multi_component_swizzle(argument, swizzle_components):
    """The first swizzle of several components that ``argument`` goes through."""
    expr = argument
    while True:
        if isinstance(expr, MemberAccessNode):
            components = swizzle_components(expr)
            if components is not None and len(components) > 1:
                return expr
            expr = expr.object_expr
        elif isinstance(expr, ArrayAccessNode):
            expr = expr.array_expr
        else:
            return None


def refuse_swizzle_reference_arguments(
    call, arguments, parameters_by_name, swizzle_components, backend
):
    """Refuse a swizzle of several components passed to an ``out``/``inout`` value.

    ``parameters_by_name`` maps function names to ``(qualifiers, kind)`` per
    parameter and overload (``collect_function_parameters_by_name``), and
    ``swizzle_components`` returns the components of a vector swizzle or
    ``None``.
    """
    name = _call_name(call)
    overloads = parameters_by_name.get(name, ())
    for index, argument in enumerate(arguments):
        written = set()
        for parameters in overloads:
            if index < len(parameters):
                qualifiers, kind = parameters[index]
                if kind is None:
                    written |= qualifiers & WRITABLE_PARAMETER_QUALIFIERS
        if not written:
            continue
        swizzle = _multi_component_swizzle(argument, swizzle_components)
        if swizzle is None:
            continue
        qualifier = "out" if "out" in written else "inout"
        raise ValueError(
            f"{backend} codegen cannot pass the swizzle '.{swizzle.member}' "
            f"as argument {index + 1} of '{name}', an {qualifier} parameter: "
            f"{backend} builds a swizzle of several components as a new "
            "vector, which a reference cannot bind to. Copy it into a local "
            "variable, pass that and assign it back."
        )
