"""Value types of SPIR-V storage-buffer subscripts.

A structured buffer or storage pointer lowers to a block whose runtime array holds the elements,
so the block type itself has no array element type. Expression typing therefore resolved
``buffer[i]`` to nothing, and generic specialization could not deduce ``T`` from a call such as
``bump(buffer[i])``.
"""

from typing import Optional

from ..ast import ArrayAccessNode, MemberAccessNode


def storage_buffer_subscript_element_type_name(generator, expr) -> Optional[str]:
    """The CrossGL element type ``buffer[i]`` reads, or ``None`` if ``expr`` is no such read.

    ``buffer`` must be a single structured buffer or storage pointer. Subscripting an array of
    buffers selects a buffer, and byte-address buffers have no element subscript.
    """
    if not isinstance(expr, ArrayAccessNode):
        return None
    array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
    if array_expr is None:
        return None
    buffer_type = generator.storage_buffer_expression_type_name(array_expr)
    type_info = generator.storage_buffer_resource_type_info(buffer_type)
    if type_info is None:
        return None
    if type_info.get("array_dimensions") or type_info.get("byte_address"):
        return None
    element_type_name = type_info.get("element_type_name")
    if not isinstance(element_type_name, str) or not element_type_name.strip():
        return None
    return element_type_name


def reads_storage_buffer_element(generator, expr) -> bool:
    """Whether ``expr`` reads a structured-buffer element, or a member or component of one.

    Such an expression is a value of the element type. Its access chain carries the buffer's
    metadata, so a storage-buffer type query on it names the whole buffer instead.
    """
    node = expr
    while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
        if isinstance(node, ArrayAccessNode):
            if storage_buffer_subscript_element_type_name(generator, node) is not None:
                return True
            node = getattr(node, "array", getattr(node, "array_expr", None))
        else:
            node = getattr(node, "object", getattr(node, "object_expr", None))
    return False
