"""HLSL atomics whose destination is a group-shared variable.

HLSL's ``Interlocked*`` functions take any group-shared scalar ``int`` or ``uint`` as their destination, as they
take a typed UAV element (HLSL intrinsic function reference, "InterlockedAdd": ``dest`` is "the destination
address"; group-shared memory and UAVs are the two resources atomics act on). The DirectX generator recognized
group-shared destinations only when they were reached through a workgroup pointer, so a CrossGL atomic on a
``shared`` variable that the semantic pass did not rewrite into statement form, such as ``atomicLoad``,
``atomicStore``, a bare ``atomicExchange`` or an atomic nested in an expression, was printed under its CrossGL
name, which DXC rejects.

:func:`is_hlsl_groupshared_atomic_target` tells the generator that the destination's root variable is declared
group-shared, so the call takes the same interlocked lowering as a buffer element. A function parameter or private
local that shadows a group-shared name is not group-shared.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, IdentifierNode, MemberAccessNode, VariableNode
from ..semantics.atomic_forms import shared_variable_names

__all__ = ["hlsl_groupshared_variable_names", "is_hlsl_groupshared_atomic_target"]

_GROUPSHARED_QUALIFIERS = frozenset(
    {"groupshared", "shared", "threadgroup", "workgroup"}
)


def hlsl_groupshared_variable_names(program) -> frozenset:
    """The names the program declares group-shared at any scope."""
    return frozenset(shared_variable_names(program, _GROUPSHARED_QUALIFIERS))


def _root_name(target):
    while True:
        if isinstance(target, ArrayAccessNode):
            target = target.array_expr
        elif isinstance(target, MemberAccessNode):
            target = target.object_expr
        else:
            break
    if isinstance(target, str):
        return target
    if isinstance(target, (IdentifierNode, VariableNode)):
        return target.name
    return None


def is_hlsl_groupshared_atomic_target(codegen, target) -> bool:
    """Whether the atomic destination ``target`` lies in a group-shared variable."""
    name = _root_name(target)
    names = getattr(codegen, "hlsl_groupshared_variable_names", frozenset())
    if name is None or name not in names:
        return False
    # A hoisted function-scope group-shared variable is registered as a local under its alias; any other local
    # or parameter of that name shadows the group-shared one.
    if name in getattr(codegen, "local_variable_types", {}):
        return name in getattr(codegen, "current_identifier_aliases", {})
    return True
