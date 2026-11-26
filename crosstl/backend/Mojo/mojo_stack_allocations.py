"""Mojo ``stack_allocation`` calls as the CrossGL workgroup arrays they allocate.

``stack_allocation[N, T, address_space=AddressSpace.SHARED]()`` (``std.memory``) returns a pointer to ``N`` values of
``T`` that every thread of the block shares, so ``tile[i]`` reads and writes the block's array. CrossGL spells that
array ``var<workgroup> tile: array<T, N>;`` in the kernel, which every generator writes as its own workgroup memory
(Mojo's generator writes the same ``stack_allocation`` back). The parser reads the call as a constructor of the
type text ``stack_allocation[...]``. Any other ``stack_allocation`` (thread-local, or with an alignment) is refused,
since CrossGL has no pointer to a thread's stack.
"""

from __future__ import annotations

import re

__all__ = ["shared_stack_allocation", "stack_allocation_declaration"]

_STACK_ALLOCATION = re.compile(
    r"^stack_allocation\[\s*(?P<count>\d+|[A-Za-z_]\w*)\s*,\s*(?P<element>.+?)\s*"
    r"(?:,\s*(?P<options>[A-Za-z_]\w*\s*=.*))?\]$"
)
_SHARED_SPACE = re.compile(r"^address_space\s*=\s*AddressSpace\.SHARED$")


def shared_stack_allocation(expr):
    """``(count, Mojo element type)`` of ``stack_allocation[count, T, address_space=SHARED]()``, else ``None``.

    Returns ``False`` for a ``stack_allocation`` call that is not one such shared allocation.
    """
    type_text = getattr(expr, "type_name", None) or getattr(expr, "vector_type", None)
    if not isinstance(type_text, str) or getattr(expr, "args", None):
        return None
    if not type_text.strip().startswith("stack_allocation["):
        return None
    match = _STACK_ALLOCATION.match(type_text.strip())
    if match is None or match.group("options") is None:
        return False
    if not _SHARED_SPACE.match(match.group("options").strip()):
        return False
    return match.group("count"), match.group("element").strip()


def stack_allocation_declaration(generator, name, expr):
    """The CrossGL declaration of ``var name = stack_allocation[...]()``, a refusal, or ``None`` for another value."""
    allocation = shared_stack_allocation(expr)
    if allocation is None:
        return None
    if allocation is False:
        return generator.refuse_import(
            f"the Mojo allocation '{name} = {expr.type_name}()' has no CrossGL form: only a "
            "stack_allocation in AddressSpace.SHARED, a workgroup array, has one",
            placeholder=f"int {name}",
        )
    count, element = allocation
    element_type = generator.map_type(element)
    if not isinstance(element_type, str) or "[" in element_type:
        return generator.refuse_import(
            f"the Mojo shared allocation '{name}' of '{element}' has an element type with no CrossGL form",
            placeholder=f"int {name}",
        )
    return f"var<workgroup> {name}: array<{element_type}, {count}>"
