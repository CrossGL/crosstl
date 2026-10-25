"""Helpers for ordering constants around user-defined type declarations."""

from ..ast import (
    ArrayType,
    ASTNode,
    FunctionType,
    GenericType,
    IdentifierNode,
    MatrixType,
    NamedType,
    PointerType,
    ReferenceType,
    StructNode,
    VariableNode,
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
    # Names a later initializer must not reference before the structs: the
    # structs themselves and every constant declared after them.
    after_structs = set(struct_names)

    for constant in constants or []:
        const_type = getattr(constant, "const_type", getattr(constant, "vtype", None))
        initializer = getattr(
            constant, "value", getattr(constant, "initial_value", None)
        )
        if type_references_names(const_type, struct_names) or (
            expression_reference_names(initializer) & after_structs
        ):
            struct_dependent_constants.append(constant)
            name = getattr(constant, "name", None)
            if isinstance(name, str):
                after_structs.add(name)
        else:
            leading_constants.append(constant)

    return leading_constants, struct_dependent_constants


def expression_reference_names(expression):
    """Return the identifiers and type names an expression refers to.

    Member names (``s.a``) and swizzles are not references. ``parent`` links
    are not followed, so only the expression itself is searched.
    """
    names = set()
    pending = [expression]
    seen = set()
    while pending:
        node = pending.pop()
        if isinstance(node, (list, tuple)):
            pending.extend(node)
            continue
        if not isinstance(node, ASTNode) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (IdentifierNode, VariableNode, NamedType)):
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
        for key, value in vars(node).items():
            if key != "parent" and isinstance(value, (ASTNode, list, tuple)):
                pending.append(value)
    return names


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
