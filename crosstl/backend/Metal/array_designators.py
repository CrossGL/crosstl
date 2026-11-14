"""Designated array initializers in Metal sources: ``{[2] = a, [4 ... 7] = b, c}``.

Clang accepts the C array designator ``[i] = v`` (C11 6.7.9) in C++, and so in the Metal Shading Language, which
is based on C++14. It also accepts the GNU range form ``[first ... last] = v`` (GCC manual, "Designated
Initializers"). CrossGL initializer lists are positional, so the importer resolves every designator to element
indices and writes the positional list:

- a designator sets the current index, and a positional element takes the current index; each initializer then
  moves the current index one past the last element it initialized (C11 6.7.9p17);
- an element that no initializer names is value-initialized, that is zero (C11 6.7.9p21);
- a range initializes each element from ``first`` to ``last`` with a single evaluation of its value, so the value
  is repeated only when it is free of side effects;
- an array of unknown size takes its length from the largest index initialized (C11 6.7.9p22).

An initializer that names an element twice overrides the earlier one (C11 6.7.9p19), and an implementation need
not evaluate the overridden one. The importer refuses that case rather than pick an evaluation. It also refuses
designator chains such as ``[1].x = v`` and ``[1][2] = v``, bounds that are not integral constants, and indices
outside the array.
"""

from __future__ import annotations

from ..common_ast import (
    DesignatedInitializerNode,
    InitializerListNode,
    VectorConstructorNode,
)

__all__ = ["has_array_designators", "positional_array_initializer"]

ARRAY_DESIGNATORS = ("index", "range")


def has_array_designators(initializer) -> bool:
    """True when ``initializer`` is a brace list with an element that starts with ``[i]`` or ``[a ... b]``."""
    return isinstance(initializer, InitializerListNode) and any(
        isinstance(element, DesignatedInitializerNode)
        and element.designators
        and element.designators[0][0] in ARRAY_DESIGNATORS
        for element in initializer.elements
    )


def _integral(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bound(codegen, bound):
    """The value of a designator bound, with the template arguments of the specialization being generated."""
    return _integral(
        codegen.evaluate_concrete_array_extent(codegen.format_array_extent(bound))
    )


def _zero(codegen, element_type, inner_extents):
    """A value-initialized element: ``T(0)`` for a scalar, vector or matrix, ``{}`` for an aggregate."""
    if (
        not inner_extents
        and element_type
        and (
            codegen.metal_scalar_arithmetic_type_info(element_type) is not None
            or codegen.metal_vector_component_parts(element_type) is not None
            or codegen.metal_matrix_type_parts(element_type) is not None
        )
    ):
        return VectorConstructorNode(element_type, [0])
    return InitializerListNode([])


class _Refused(Exception):
    pass


def _slot_range(codegen, element, current, extent):
    """The first and last index ``element`` initializes, and its value."""
    if not isinstance(element, DesignatedInitializerNode):
        first = last = current
        value = element
    else:
        if len(element.designators) != 1 or element.designators[0][0] not in (
            ARRAY_DESIGNATORS
        ):
            raise _Refused("a designator chain such as '[i].x = v' or '[i][j] = v'")
        designator = element.designators[0]
        first = _bound(codegen, designator[1])
        last = _bound(codegen, designator[2]) if designator[0] == "range" else first
        if first is None or last is None:
            raise _Refused("a designator bound that is not an integral constant")
        if last < first:
            raise _Refused(f"the empty designator range [{first} ... {last}]")
        value = element.value
        if (
            last > first
            and not codegen.discarded_expression_is_proven_side_effect_free(value)
        ):
            raise _Refused(
                "a range designator whose value may have side effects, which the range evaluates once"
            )
    if first < 0 or (extent is not None and last >= extent):
        raise _Refused(f"an initializer of index {last} outside the array")
    return first, last, value


def _positional(codegen, element_type, extents, initializer):
    extent = None
    if extents and extents[0] is not None:
        extent = _integral(
            codegen.evaluate_concrete_array_extent(
                codegen.format_array_extent(extents[0])
            )
        )
        if extent is None:
            raise _Refused("an array extent that is not an integral constant")
    inner_extents = list(extents[1:]) if extents else []
    slots = {}
    current = 0
    for element in initializer.elements:
        first, last, value = _slot_range(codegen, element, current, extent)
        if inner_extents and has_array_designators(value):
            value = _positional(codegen, element_type, inner_extents, value)
        for index in range(first, last + 1):
            if index in slots:
                raise _Refused(f"two initializers of index {index}")
            slots[index] = value
        current = last + 1
    length = max(slots) + 1 if slots else 0
    return InitializerListNode(
        [
            (
                slots[index]
                if index in slots
                else _zero(codegen, element_type, inner_extents)
            )
            for index in range(length)
        ]
    )


def positional_array_initializer(codegen, element_type, extents, initializer):
    """``initializer`` for an array of ``element_type`` and ``extents`` with its designators resolved to positions.

    ``extents`` lists the array's dimensions, outermost first, with None for an unknown size. A designator the
    importer cannot resolve is refused through ``codegen.refuse_import``, and an empty list stands in for it.
    """
    try:
        return _positional(codegen, element_type, list(extents or []), initializer)
    except _Refused as refusal:
        return codegen.refuse_import(
            f"Metal designated array initializer with {refusal}",
            InitializerListNode([]),
        )
