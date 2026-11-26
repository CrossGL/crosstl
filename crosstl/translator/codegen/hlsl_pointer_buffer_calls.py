"""CrossGL ``buffer_load`` and ``buffer_store`` through storage pointers in HLSL.

A helper that takes ``device T* data`` receives an HLSL resource and an
element offset, not a buffer of its own. ``buffer_load(data, i)`` and
``buffer_store(data, i, v)`` on such a pointer mean ``data[i]`` and
``data[i] = v``, so they render through the same pointer access as the
indexing forms: the offset, the view's element type and the root resource's
own store form (``buffer[index] = value`` for typed and structured buffers,
``Store`` for byte-address buffers) all come from the pointer binding. Read as
a resource method instead, the store came out as ``data.Store(i, v)``, which
no typed or structured HLSL buffer has.
"""

from ..ast import ArrayAccessNode, AssignmentNode

POINTER_BUFFER_CALLS = frozenset({"buffer_load", "buffer_store"})


def hlsl_pointer_buffer_call(
    generator, func_name, args, *, rendered_value=None, statement_context=False
):
    """Render ``buffer_load``/``buffer_store`` on a storage pointer, or ``None``.

    Returns ``None`` when the first operand is an HLSL buffer resource itself
    (the resource method forms apply) or is not a storage pointer.
    """
    if func_name not in POINTER_BUFFER_CALLS:
        return None
    if len(args) < (3 if func_name == "buffer_store" else 2):
        return None
    pointer = args[0]
    if generator.hlsl_buffer_helper_resource_type(pointer) is not None:
        return None
    if generator.hlsl_resource_pointer_binding(pointer) is None:
        return None
    element = ArrayAccessNode(pointer, args[1])
    if func_name == "buffer_load":
        return generator.generate_expression(element)
    if rendered_value is not None:
        access = generator.generate_hlsl_resource_pointer_access(
            pointer, args[1], require_write=True
        )
        if access is None:
            return None
        return f"{access} = {rendered_value}"
    store = generator.generate_assignment(
        AssignmentNode(element, args[2], "="), statement_context=statement_context
    )
    return store.rstrip().rstrip(";") if isinstance(store, str) else store
