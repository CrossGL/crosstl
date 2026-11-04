"""Rows of fixed multidimensional private arrays passed as private pointers.

For a local ``float x[R][C]``, the C++ expression ``x[r]`` is an lvalue of type ``float[C]``, and passing it where a
``thread float*`` is expected converts it to a pointer to the row's first element ([conv.array]). The view covers
exactly the ``C`` elements of that row ([dcl.array]).

HLSL and GLSL lower a private pointer parameter to an ``inout`` fixed array plus a base offset. Both accept the row
itself as such an argument: ``x[r]`` is an lvalue of the row's array type in HLSL and in GLSL 4.30 and later (GLSL
4.60 section 4.1.9, "Arrays"). GLSL evaluates every argument exactly once, at call time, and copies an ``out`` or
``inout`` value back to the l-value found then (GLSL 4.60 section 6.1.1, "Function Calling Conventions"); DXC follows
the same convention for HLSL ``inout`` parameters. So a row view lowers to the row l-value with offset zero and extent
``C``, and a view into the row (``&x[r][c]``, ``x[r] + c``) adds its offset as for a one-dimensional array.

Only a row of the innermost dimension decays to a pointer to scalars. ``x[a]`` of ``float x[A][B][C]`` decays to a
pointer to ``float[C]``, which is not a private scalar view, so it is not recognized here.
"""

from ..ast import ArrayAccessNode, ArrayType, IdentifierNode, LiteralNode, VariableNode


def array_row_shape(array_type, literal_int):
    """The extents and innermost element type of a fixed multidimensional array type, or None.

    The extents are the literal extents of every dimension, outermost first, so the last one is the row length.
    ``literal_int`` resolves a size expression to an int, or None.
    """
    if not isinstance(array_type, ArrayType):
        return None
    extents = []
    element = array_type
    while isinstance(element, ArrayType):
        extent = literal_int(element.size)
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            return None
        extents.append(extent)
        element = element.element_type
    if len(extents) < 2:
        return None
    return tuple(extents), element


def array_row_view(expression, shape_of):
    """``(root, extent, element_type, indices)`` when ``expression`` is a row of a known local array.

    ``shape_of`` returns a local array name's :func:`array_row_shape`, or None for any other name. ``indices`` are
    the subscripts that select the row, outermost first. The expression must index every dimension but the innermost,
    and a literal subscript must lie within its dimension ([expr.add]); otherwise the view has no fixed backing row.
    """
    indices = []
    node = expression
    while isinstance(node, ArrayAccessNode):
        indices.append(node.index)
        node = node.array
    if not indices or not isinstance(node, (IdentifierNode, VariableNode)):
        return None
    shape = shape_of(getattr(node, "name", None))
    if shape is None:
        return None
    extents, element_type = shape
    indices.reverse()
    if len(indices) != len(extents) - 1:
        return None
    for index, extent in zip(indices, extents):
        value = (
            getattr(index, "value", None) if isinstance(index, LiteralNode) else None
        )
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and not 0 <= value < extent
        ):
            return None
    return node.name, extents[-1], element_type, tuple(indices)


def aliased_row_view(view_calls):
    """First call that binds a row view and another view of the same local array, or None.

    Each ``inout`` argument is copied back independently, so two views of one array lose each other's writes when
    the rows coincide. Returns ``(call, first_parameter, second_parameter, root)`` in deterministic order.
    """
    for caller_name in sorted(view_calls):
        for call in view_calls[caller_name]:
            seen = {}
            rows = set()
            for parameter_name, binding in call["bindings"].items():
                if binding.get("root_kind") not in {"local", "local_word_view"}:
                    continue
                root = binding.get("root")
                if binding.get("row_view"):
                    rows.add(root)
                if root in seen and root in rows:
                    return call, seen[root], parameter_name, root
                seen.setdefault(root, parameter_name)
    return None
