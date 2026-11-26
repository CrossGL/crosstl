"""The ``[[vk::offset(N)]]`` of the members of a struct that a push-constant block holds.

A push-constant ``ConstantBuffer<T>``, or a push-constant ``T``, lays out the members of the struct ``T``. slangc keeps
a struct member's ``[[vk::offset(N)]]`` in every buffer it lays the struct out in, for SPIR-V: the std430 layout of a
push constant or a structured buffer and the std140 layout of a constant buffer. SPIR-V output places the members of
every struct type at the byte their ``@ offset(N)`` names, so Slang output writes ``[[vk::offset(N)]]`` for such a
member of a struct that a push-constant block holds, and slangc lays the block out at the bytes CrossGL declares.

:mod:`..semantics.member_placements` refuses the placements this cannot write: an ``@ align(N)`` that moves a member,
a placement inside a struct that ``T`` holds, and an offset that is not a multiple of the member's std430 alignment,
which slangc moves to that alignment.
"""

from __future__ import annotations

from .. import ast as A
from ..semantics.constant_buffers import constant_buffer_element
from .buffer_member_layout import member_explicit_offset
from .push_constant_blocks import is_push_constant

__all__ = ["slang_member_offset_prefix", "slang_push_constant_struct_names"]


def _variables(program):
    yield from getattr(program, "global_variables", None) or []
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        yield from getattr(stage, "local_variables", None) or []


def _struct_name(type_node):
    if isinstance(type_node, A.NamedType) and not type_node.generic_args:
        return type_node.name
    if isinstance(type_node, str) and type_node.isidentifier():
        return type_node
    return None


def slang_push_constant_struct_names(program) -> frozenset[str]:
    """The names of the structs that a push-constant ``ConstantBuffer<T>`` or a push-constant ``T`` of ``program``
    holds."""
    names = set()
    for variable in _variables(program):
        if not is_push_constant(variable):
            continue
        var_type = getattr(variable, "var_type", None)
        element = constant_buffer_element(var_type)
        name = _struct_name(var_type if element is None else element)
        if name is not None:
            names.add(name)
    return frozenset(names)


def slang_member_offset_prefix(codegen, struct_name, member) -> str:
    """``[[vk::offset(N)]] `` for a member with ``@ offset(N)`` of a struct a push-constant block holds, else ``""``."""
    if struct_name not in getattr(codegen, "slang_push_constant_struct_names", ()):
        return ""
    offset = member_explicit_offset(member)
    return f"[[vk::offset({offset})]] " if offset is not None else ""
