"""Helpers for ordering constants around user-defined type declarations."""

from ..ast import (
    ArrayType,
    FunctionType,
    GenericType,
    MatrixType,
    NamedType,
    PointerType,
    ReferenceType,
    StructNode,
    VectorType,
)
from .array_utils import split_array_type_suffix


def partition_constants_by_struct_dependency(constants, structs):
    """Split constants into pre-struct and post-struct declaration groups."""
    struct_names = {
        getattr(node, "name", None)
        for node in structs or []
        if isinstance(node, StructNode) and getattr(node, "name", None)
    }
    leading_constants = []
    struct_dependent_constants = []

    for constant in constants or []:
        const_type = getattr(
            constant, "const_type", getattr(constant, "vtype", None)
        )
        if type_references_names(const_type, struct_names):
            struct_dependent_constants.append(constant)
        else:
            leading_constants.append(constant)

    return leading_constants, struct_dependent_constants


def type_references_names(type_node, names):
    """Return whether a type node directly or transitively names any target."""
    if not names or type_node is None:
        return False

    if isinstance(type_node, str):
        base_type, _array_suffix = split_array_type_suffix(type_node)
        return base_type.strip() in names

    if isinstance(type_node, ArrayType):
        return type_references_names(type_node.element_type, names)

    if isinstance(type_node, (VectorType, MatrixType)):
        return type_references_names(type_node.element_type, names)

    if isinstance(type_node, PointerType):
        return type_references_names(type_node.pointee_type, names)

    if isinstance(type_node, ReferenceType):
        return type_references_names(type_node.referenced_type, names)

    if isinstance(type_node, FunctionType):
        return type_references_names(type_node.return_type, names) or any(
            type_references_names(param_type, names)
            for param_type in getattr(type_node, "param_types", []) or []
        )

    name = getattr(type_node, "name", None)
    if name in names:
        return True

    if isinstance(type_node, GenericType):
        return any(
            type_references_names(constraint, names)
            for constraint in getattr(type_node, "constraints", []) or []
        )

    return any(
        type_references_names(arg, names)
        for arg in getattr(type_node, "generic_args", []) or []
    )
