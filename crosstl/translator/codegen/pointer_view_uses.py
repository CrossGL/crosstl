"""Classify how the pointer produced by a storage view is used.

A source cast such as ``(device T*)readonly_bytes`` drops pointee constness.
Metal can reproduce that cast exactly only when the source storage is writable,
but a view whose pointer is only ever read can be re-spelled as a ``const``
view without changing any observable value.  This module proves that narrow
property from the view's syntactic ancestors; anything it does not recognise is
reported as a possible write so callers keep failing closed.
"""

_POINTER_ARITHMETIC_OPERATORS = {"+", "-"}
_PURE_UNARY_OPERATORS = {"-", "+", "!", "~"}


def _operator(node):
    return getattr(node, "operator", getattr(node, "op", None))


def _is_const_pointer_type(type_node):
    from ..ast import PointerType

    return isinstance(type_node, PointerType) and not getattr(
        type_node, "is_mutable", True
    )


def _declares_const_pointee(declaration):
    """Return whether a local pointer declaration has a const pointee.

    CrossGL spells ``const device T* p`` with a ``const`` declaration
    qualifier, and the Metal generator refuses writes through such locals.
    """

    from ..ast import PointerType

    var_type = getattr(declaration, "var_type", None)
    if not isinstance(var_type, PointerType):
        return False
    qualifiers = {
        str(getattr(qualifier, "name", qualifier)).lower()
        for qualifier in getattr(declaration, "qualifiers", ()) or ()
    }
    return "const" in qualifiers or _is_const_pointer_type(var_type)


def _is_reference_type(type_node):
    return type_node is None or type_node.__class__.__name__ == "ReferenceType"


def _declares_reference(target):
    """Return whether an assignment target declares a reference binding."""

    from ..ast import BinaryOpNode, VariableNode

    if isinstance(target, BinaryOpNode):
        # ``T & name = value`` parses with a binary ``&`` target.
        return _operator(target) == "&"
    if isinstance(target, VariableNode):
        return _is_reference_type(getattr(target, "var_type", None)) and (
            getattr(target, "var_type", None) is not None
        )
    return False


def _climb_pointer_arithmetic(node, ancestors):
    """Skip ``view + offset`` / ``view - offset`` wrappers around ``node``."""

    from ..ast import BinaryOpNode

    index = 0
    while index < len(ancestors):
        parent = ancestors[index]
        if not isinstance(parent, BinaryOpNode):
            break
        if _operator(parent) not in _POINTER_ARITHMETIC_OPERATORS:
            break
        left = getattr(parent, "left", None)
        right = getattr(parent, "right", None)
        if left is not node and not (right is node and _operator(parent) == "+"):
            break
        node = parent
        index += 1
    return node, index


def _is_value_read(element, parent):
    """Return whether ``element`` (an lvalue) is consumed only as a value."""

    from ..ast import AssignmentNode, BinaryOpNode, UnaryOpNode, VariableNode

    if isinstance(parent, AssignmentNode):
        target = getattr(parent, "target", None)
        return (
            getattr(parent, "value", None) is element
            and target is not element
            and not _declares_reference(target)
        )
    if isinstance(parent, VariableNode):
        return getattr(
            parent, "initial_value", None
        ) is element and not _is_reference_type(getattr(parent, "var_type", None))
    if isinstance(parent, BinaryOpNode):
        return True
    if isinstance(parent, UnaryOpNode):
        return _operator(parent) in _PURE_UNARY_OPERATORS and not getattr(
            parent, "is_postfix", False
        )
    return False


def metal_pointer_view_is_read_only_use(expression, ancestors):
    """Prove that no write can go through the pointer ``expression`` yields.

    ``ancestors`` lists the syntactic parents of ``expression``, nearest first.
    """

    from ..ast import ArrayAccessNode, PointerReinterpretNode, UnaryOpNode, VariableNode

    node, index = _climb_pointer_arithmetic(expression, ancestors)
    if index >= len(ancestors):
        return False
    parent = ancestors[index]
    if isinstance(parent, PointerReinterpretNode):
        # The enclosing view is validated on its own; a const view never
        # writes through the pointer it reinterprets.
        return getattr(parent, "expression", None) is node and _is_const_pointer_type(
            getattr(parent, "target_type", None)
        )
    if isinstance(parent, VariableNode):
        return getattr(parent, "initial_value", None) is node and (
            _declares_const_pointee(parent)
        )
    if isinstance(parent, ArrayAccessNode):
        if getattr(parent, "array", None) is not node:
            return False
    elif isinstance(parent, UnaryOpNode):
        if _operator(parent) != "*" or getattr(parent, "operand", None) is not node:
            return False
    else:
        return False
    if index + 1 >= len(ancestors):
        return False
    return _is_value_read(parent, ancestors[index + 1])
