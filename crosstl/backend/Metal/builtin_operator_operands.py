"""Operators whose operands have no class or enumeration type.

C++ [over.match.oper]/1, which the Metal Shading Language inherits: "If no operand of an operator in an expression
has a type that is a class or an enumeration, the operator is assumed to be a built-in operator". [over.oper]/6
requires every operator function to have a parameter of class or enumeration type (or a reference to one), so no
declared free operator can be selected for such an expression, and its overload candidates need not be ranked.

Metal scalars, vectors (``vec<T, N>`` and the ``packed_`` forms are Clang extended vectors, not classes),
pointers and arrays are not class types. A name with a scalar spelling that the program itself declares as a
structure or an enumeration is treated as a class, so a source that names a structure ``bfloat16_t`` still ranks
its operators. Ranking every candidate for each such operand pair was most of the importer's free-operator cost on
large kernels.
"""

from __future__ import annotations

from .MetalAst import InitializerListNode

__all__ = ("operands_have_no_class_type",)


def _declared_aggregate(codegen, type_text) -> bool:
    name = codegen.normalized_metal_type(type_text)
    return (
        name in codegen.struct_member_types
        or name in codegen.metal_enum_arithmetic_types
        or name in codegen.type_alias_enum_declarations
    )


def _builtin_descriptor(descriptor) -> bool:
    """Whether a ``metal_source_overload_type_descriptor`` describes a type that is not a class or enumeration."""
    if descriptor is None:
        return False
    kind = descriptor[0]
    if kind == "scalar" or kind == "pointer":
        return True
    return kind == "vector" and descriptor[2][0] == "scalar"


def operands_have_no_class_type(codegen, arguments, argument_types) -> bool:
    """Whether no operand of an operator has a class or enumeration type, so the built-in operator applies.

    ``argument_types`` are the operands' ``metal_source_overload_value_type`` spellings. Braced initializer lists
    and operands whose type is unknown are never proven built-in.
    """
    for argument, type_text in zip(arguments, argument_types):
        if isinstance(argument, InitializerListNode) or not type_text:
            return False
        if not _builtin_descriptor(
            codegen.metal_source_overload_type_descriptor(type_text)
        ):
            return False
        if _declared_aggregate(codegen, type_text):
            return False
    return len(arguments) == len(argument_types)
