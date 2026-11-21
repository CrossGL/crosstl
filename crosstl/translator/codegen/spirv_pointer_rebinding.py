"""Assigning a new value to a local pointer into a storage buffer or local array.

SPIR-V's logical addressing model has no pointer variables: the generator keeps each CrossGL pointer local as its
root resource and a Function-storage element offset (:class:`~.spirv_values.PointerOffsetState`). ``p += n`` and
``p -= n`` already update that offset. ``p = q``, ``p = q + n`` and ``p = p - n`` re-bind ``p`` to another pointer
into the same root: ``p``'s offset takes the right-hand pointer's offset, and the root stays. A pointer into another
root would need a pointer variable that logical SPIR-V cannot express, so it is refused.
"""

from __future__ import annotations

from ..ast import IdentifierNode, VariableNode

__all__ = ["rebind_storage_pointer_alias"]


def _target_name(target):
    if isinstance(target, (IdentifierNode, VariableNode)):
        return target.name
    return target if isinstance(target, str) else None


def rebind_storage_pointer_alias(gen, target, value) -> bool:
    """Lower ``target = value`` when ``target`` is a local pointer alias; return whether it was handled."""
    name = _target_name(target)
    if name is None or name not in gen.resource_alias_variables:
        return False
    alias = gen.local_variables.get(name)
    if alias is None:
        return False
    aliases = gen.storage_buffer_pointer_offset_aliases
    state = aliases.get(gen.storage_buffer_pointer_alias_key(alias))
    if state is None:
        return False
    source = gen.variable_pointer_from_expression(value)
    if source is None:
        return False
    # Evaluating ``value`` may have given ``target`` its offset cell (``p = p + n``), so its state is read again.
    state = aliases[gen.storage_buffer_pointer_alias_key(alias)]
    source_state = aliases.get(gen.storage_buffer_pointer_alias_key(source))
    source_root = source_state.root_pointer if source_state is not None else source
    if source_root.id != state.root_pointer.id:
        gen.refuse_lowering(
            f"cannot point SPIR-V pointer {name} into another resource: logical SPIR-V keeps each "
            "pointer local at the resource it was declared from"
        )
        return True
    if source_state is not None and source_state.offset_variable is not None:
        offset_type = gen.variable_value_types[source_state.offset_variable.id]
        offset = gen.load_from_variable(source_state.offset_variable, offset_type)
    else:
        offset_type = gen.register_primitive_type("uint")
        offset = gen.register_constant(0, offset_type)
    state = gen.materialize_storage_buffer_pointer_offset_alias(
        alias, state, offset_type
    )
    target_type = gen.variable_value_types[state.offset_variable.id]
    gen.store_to_variable(
        state.offset_variable, gen.convert_value_to_type(offset, target_type)
    )
    return True
