"""Concrete SPIR-V structures for the instances of CrossGL generic structures.

SPIR-V has no generic types: every ``OpTypeStruct`` has concrete members. The generator declared a generic
structure only when its members needed no type argument, so ``generic<T> struct Box { T value; }`` used as
``Box<float>`` mapped to "Unknown type Box<float>, using float as default" and the program was refused.

:func:`materialize_generic_struct_instance` declares the instance on first use: a copy of the definition named as
the other generators name it (``Box_float``), whose members take the instance's type and value arguments
(``T items[N]`` in ``Ring<float, 4>`` holds ``float[4]``). The generator then lowers it as any other structure.
:func:`generic_struct_instance_spelling` gives back the CrossGL spelling of an instance's structure, so a generic
function that takes a ``Box<T>`` still deduces ``T`` from an argument of type ``Box_float``.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import ArrayNode, StructNode
from .enum_utils import generic_type_parts, substitute_generic_value_names
from .generic_struct_utils import (
    generic_struct_specialization_name,
    substitute_generic_type_name_with_arrays,
)

__all__ = [
    "generic_struct_instance_spelling",
    "materialize_generic_struct_instance",
]

_TYPE_FIELDS = ("member_type", "var_type", "vtype")


def _parameter_names(definition):
    return [
        str(getattr(parameter, "name", parameter))
        for parameter in getattr(definition, "generic_params", None) or []
    ]


def _instances(generator):
    instances = getattr(generator, "generic_struct_instance_spellings", None)
    if instances is None:
        instances = generator.generic_struct_instance_spellings = {}
    return instances


def _is_concrete(argument, open_parameters):
    """Whether a type argument names none of the program's unbound generic parameters."""
    base, nested = generic_type_parts(argument)
    if base.strip() in open_parameters:
        return False
    return all(_is_concrete(item, open_parameters) for item in nested)


def _substitute_member(generator, member, substitutions):
    if isinstance(member, ArrayNode):
        element = member.element_type
        if not isinstance(element, str):
            element = generator.convert_type_node_to_string(element)
        member.element_type = substitute_generic_type_name_with_arrays(
            element, substitutions
        )
        if member.size is not None:
            size = member.size
            if not isinstance(size, (str, int)):
                size = generator.convert_type_node_to_string(size)
            member.size = substitute_generic_value_names(str(size), substitutions)
        return
    for field in _TYPE_FIELDS:
        value = getattr(member, field, None)
        if value is None:
            continue
        if not isinstance(value, str):
            value = generator.convert_type_node_to_string(value)
        setattr(
            member,
            field,
            substitute_generic_type_name_with_arrays(value, substitutions),
        )


def materialize_generic_struct_instance(generator, type_str, open_parameters=()):
    """Declare the structure of the generic instance ``type_str`` (``Box<float>``); returns its name or ``None``.

    ``open_parameters`` are generic parameter names that are not bound where the type is read; an instance that
    names one is not concrete, and is left to the caller.
    """
    base, arguments = generic_type_parts(str(type_str or ""))
    if not arguments:
        return None
    declarations = generator.struct_declarations
    definition = declarations.get(base)
    if not isinstance(definition, StructNode):
        return None
    parameters = _parameter_names(definition)
    if not parameters or len(arguments) != len(parameters):
        return None
    if not all(_is_concrete(argument, set(open_parameters)) for argument in arguments):
        return None
    spelling = f"{base}<{', '.join(argument.strip() for argument in arguments)}>"
    name = generic_struct_specialization_name(spelling)
    if name in declarations:
        return name if name in _instances(generator) else None
    instance = deepcopy(definition)
    instance.name = name
    instance.generic_params = []
    substitutions = dict(zip(parameters, (argument.strip() for argument in arguments)))
    for member in getattr(instance, "members", None) or []:
        _substitute_member(generator, member, substitutions)
    declarations[name] = instance
    _instances(generator)[name] = spelling
    return name


def generic_struct_instance_spelling(generator, struct_name):
    """The CrossGL spelling (``Box<float>``) of a materialized instance's structure, else ``None``."""
    return _instances(generator).get(str(struct_name))
