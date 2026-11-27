"""Dereferences of storage buffers in the WGSL generator.

A CrossGL ``buffer T*`` is declared in WGSL as ``var<storage, read_write> name: array<T>``, or passed to a helper as
``ptr<storage, array<T>, read_write>``. Neither is a pointer to an element, so ``*name`` and ``*(name + i)`` do not
validate. They are element accesses: ``*buf`` reads ``buf[0]``, and ``*(buf + i)`` or ``*(i + buf)`` reads
``buf[i]``. Pointer locals (``wgsl_pointer_locals``) are real WGSL pointers and keep their ``*``.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    ASTNode,
    BinaryOpNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    UnaryOpNode,
)
from ..refusals import WGSLRefusal

_INTEGER_SCALARS = frozenset({"int", "uint", "i32", "u32"})


def buffer_pointer_module_names(codegen, global_variables):
    """Module-scope ``buffer T*`` variables, which WGSL declares as ``array<T>`` storage bindings."""
    return {
        variable.name
        for variable in global_variables
        if getattr(variable, "name", "")
        and codegen.is_buffer_pointer_type(
            getattr(variable, "var_type", None), getattr(variable, "qualifiers", ())
        )
    }


def is_buffer_identifier(codegen, expr) -> bool:
    """Whether ``expr`` names a storage buffer: a buffer parameter, or a module buffer no local shadows."""
    if not isinstance(expr, IdentifierNode):
        return False
    if codegen.is_pointer_identifier(expr.name):
        return True
    return expr.name in getattr(
        codegen, "_buffer_pointer_module_names", ()
    ) and not codegen.is_local_identifier(expr.name)


def _is_integer(codegen, expr) -> bool:
    return codegen.scalar_type_name(codegen.expression_type(expr)) in _INTEGER_SCALARS


def _is_negative_constant(expr) -> bool:
    """A negative integer literal, ``-1`` or a literal ``-1`` value: an index before the buffer's first element."""
    if isinstance(expr, UnaryOpNode) and expr.operator == "-" and not expr.is_postfix:
        return isinstance(expr.operand, LiteralNode)
    if isinstance(expr, LiteralNode):
        try:
            return int(str(expr.value).rstrip("uUlL")) < 0
        except ValueError:
            return False
    return False


def _buffer_and_offset(codegen, expr):
    """Split ``buf``, ``buf + i``, ``i + buf`` and ``buf + i + j`` into the buffer and its element offset.

    Returns ``(buffer, offset)``, with ``offset`` ``None`` for the buffer itself, or ``None`` when ``expr`` is not
    a buffer plus integer offsets.
    """
    if is_buffer_identifier(codegen, expr):
        return expr, None
    if not isinstance(expr, BinaryOpNode) or expr.operator != "+":
        return None
    for base, step, step_first in (
        (expr.left, expr.right, False),
        (expr.right, expr.left, True),
    ):
        split = _buffer_and_offset(codegen, base)
        if (
            split is None
            or not _is_integer(codegen, step)
            or _is_negative_constant(step)
        ):
            continue
        buffer, offset = split
        if offset is None:
            return buffer, step
        # Keep the source order of the offsets, so side effects in them run in the same order.
        return buffer, (
            BinaryOpNode(step, "+", offset)
            if step_first
            else BinaryOpNode(offset, "+", step)
        )
    return None


def _mentions_buffer(codegen, expr) -> bool:
    if is_buffer_identifier(codegen, expr):
        return True
    if isinstance(expr, ArrayAccessNode):
        # ``buf[i]`` is an element value, not the buffer.
        return _mentions_buffer(codegen, expr.index_expr)
    if not isinstance(expr, ASTNode):
        return False
    for value in vars(expr).values():
        children = value if isinstance(value, (list, tuple)) else (value,)
        for child in children:
            if isinstance(child, ASTNode) and _mentions_buffer(codegen, child):
                return True
    return False


def buffer_dereference_access(codegen, expr):
    """The element access for a dereference of a storage buffer, or ``None`` when ``expr`` is not one.

    Raises ``WGSLRefusal`` for a dereference of an expression built from a buffer that is not the buffer plus
    integer offsets, such as ``*(buf - 1)`` or ``*(buf + -1)``, which have no element-access form.
    """
    if not (
        isinstance(expr, UnaryOpNode) and expr.operator == "*" and not expr.is_postfix
    ):
        return None
    split = _buffer_and_offset(codegen, expr.operand)
    if split is None:
        if _mentions_buffer(codegen, expr.operand):
            raise WGSLRefusal(
                "WGSL target can only write a dereference of a storage buffer as an "
                "element access, `*buf` or `*(buf + i)` with an integer offset i"
            )
        return None
    buffer, offset = split
    if offset is None:
        offset = LiteralNode(0, PrimitiveType("int"))
    return ArrayAccessNode(buffer, offset)
