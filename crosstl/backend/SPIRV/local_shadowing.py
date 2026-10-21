"""Keep the importer's locals from hiding the functions a SPIR-V function calls.

A value the importer assigns to a local (a phi variable, or a result evaluated where it is defined) is named after
its ``OpName`` or, without one, after its id. Assembly may name an id after the operation that defines it, as in
``%dot = OpDot ...``; a CrossGL local called ``dot`` would then hide the ``dot`` built-in for the rest of its scope,
including the call in its own initializer, and every target compiler would reject the program.
:func:`rename_shadowing_locals` renames such a local after the function body is built, when every call is known.
"""

from __future__ import annotations

from ..common_ast import ASTNode, FunctionCallNode, VariableNode

__all__ = ["rename_shadowing_locals"]


def _nodes(root):
    seen = set()
    pending = [root]
    while pending:
        item = pending.pop()
        if isinstance(item, (list, tuple)):
            pending.extend(reversed(item))
            continue
        if not isinstance(item, ASTNode) or id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        pending.extend(reversed(list(vars(item).values())))


def rename_shadowing_locals(body, local_names, taken_names):
    """Rename each local in ``local_names`` (SPIR-V id to name) that a call in ``body`` names.

    The new name is ``value_<name>``, or ``value_<name>_N`` if that is taken; ``taken_names`` gains it. Declarations
    and references are found by their ``spirv_id``.
    """
    nodes = list(_nodes(body))
    called = {
        node.name
        for node in nodes
        if isinstance(node, FunctionCallNode) and isinstance(node.name, str)
    }
    renames = {}
    for value_id, name in sorted(local_names.items()):
        if name not in called:
            continue
        new_name = f"value_{name}"
        suffix = 1
        while new_name in taken_names or new_name in called:
            new_name = f"value_{name}_{suffix}"
            suffix += 1
        taken_names.add(new_name)
        renames[value_id] = (name, new_name)
    if not renames:
        return
    for node in nodes:
        if not isinstance(node, VariableNode):
            continue
        rename = renames.get(getattr(node, "spirv_id", None))
        if rename is None:
            continue
        old_name, new_name = rename
        text = str(node.name)
        if text == old_name or text.startswith(f"{old_name}["):
            node.name = new_name + text[len(old_name) :]
