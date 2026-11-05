"""Bind the local storage-pointer declarations of a function body to the storage they point into.

A function decides which access a storage-pointer parameter needs by following the local pointers derived from it:
``device float* p = data + 1`` makes ``p`` an alias of ``data``, so a store through ``p`` writes ``data``. The
alias table is keyed by name and does not follow scopes, and a body may declare one name more than once, as in

.. code-block:: c

    if (mode == 0) { device float* p = data + 1; total += p[0]; }
    else           { device float* p = data + 2; total += p[0]; }

Rebinding ``p`` to each declaration in turn never settles, because each one undoes the other. Here a name declared
by several declarations is bound once per pass, from all of them together:

* When every declaration gives the same binding, the name has that binding, exactly as with a single declaration.
* When they disagree, the name takes the first binding whose root is the preferred root (the parameter under
  analysis), and otherwise the first one in walk order. A use of the name may refer to any of the declarations, so
  treating it as an alias of the parameter whenever one of them is keeps the parameter's required access
  conservative: it can only report a write that a more precise analysis would rule out, never miss one.

The passes stop when no declaration's binding changes. Pointers that are defined through each other in a cycle can
keep producing new offsets, so the passes are also capped; the caller then has no settled table and must assume the
most demanding access.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

__all__ = ["bind_local_storage_pointers"]

Binding = Dict[str, Any]


def bind_local_storage_pointers(
    declarations: Sequence[Any],
    aliases: dict[str, Binding],
    is_storage_pointer_declaration: Callable[[Any, dict[str, Binding]], bool],
    binding_of: Callable[[Any, dict[str, Binding]], Binding | None],
    preferred_root: str | None = None,
) -> bool:
    """Add the binding of every local storage pointer in ``declarations`` to ``aliases``; whether they settled.

    ``declarations`` are the variable declarations of the body in walk order, and ``aliases`` starts with the
    bindings of the parameters. ``is_storage_pointer_declaration(node, aliases)`` tells whether a declaration
    declares a storage pointer, and ``binding_of(expression, aliases)`` returns the binding of a pointer expression,
    or ``None`` when it does not point into tracked storage. A declaration whose initializer has no binding leaves
    its name as it was. ``aliases`` is updated in place after every declaration, so a later declaration in the same
    pass sees the bindings of the earlier ones. Returns ``False`` when the bindings were still changing after the
    last allowed pass; ``aliases`` then holds the bindings of that pass.
    """
    by_name: dict[str, list[int]] = {}
    for index, node in enumerate(declarations):
        by_name.setdefault(node.name, []).append(index)
    bound: dict[int, Binding] = {}
    for _ in range(len(declarations) + 2):
        changed = False
        for index, node in enumerate(declarations):
            if not is_storage_pointer_declaration(node, aliases):
                continue
            binding = binding_of(getattr(node, "initial_value", None), aliases)
            if binding is None or bound.get(index) == binding:
                continue
            bound[index] = binding
            changed = True
            aliases[node.name] = _merged_binding(
                [bound[i] for i in by_name[node.name] if i in bound], preferred_root
            )
        if not changed:
            return True
    return False


def _merged_binding(bindings: list[Binding], preferred_root: str | None) -> Binding:
    """The binding of a name whose declarations give ``bindings`` (at least one), in walk order."""
    first = bindings[0]
    if all(binding == first for binding in bindings[1:]):
        return first
    for binding in bindings:
        if preferred_root is not None and binding.get("root") == preferred_root:
            return binding
    return first
