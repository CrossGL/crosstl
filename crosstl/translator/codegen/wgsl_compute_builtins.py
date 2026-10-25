"""Compute built-in inputs narrower than WGSL's ``vec3<u32>``.

WGSL declares the compute built-in inputs ``global_invocation_id``, ``local_invocation_id``, ``workgroup_id`` and
``num_workgroups`` as ``vec3<u32>`` and nothing else (WGSL section 13.3.1.1). MSL accepts ``uint`` and ``uint2``
for the corresponding attributes (``thread_position_in_grid`` and the like, MSL section 5.2.3.6), and HLSL accepts
``uint``, ``uint2``, ``int`` and ``int2`` forms of ``SV_DispatchThreadID`` and its siblings. Those components are the
leading components of the three-dimensional value.

:func:`lower_narrow_compute_builtins` receives such an entry-point parameter as ``vec3<u32>`` under a fresh name, and
declares the source parameter as a local at the top of the body, initialized with the leading components converted
to its type. Reads and writes of the parameter then use the local, which WGSL, unlike a parameter, lets the body
write.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..semantics.hub_types import type_from_node

__all__ = ["VECTOR_COMPUTE_BUILTINS", "lower_narrow_compute_builtins"]

VECTOR_COMPUTE_BUILTINS = frozenset(
    {"global_invocation_id", "local_invocation_id", "workgroup_id", "num_workgroups"}
)
_COMPONENTS = "xyz"


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if hasattr(stages, "values") else list(stages)


def _identifiers(program):
    names = set()
    for node in program.walk():
        for field in ("name", "member"):
            value = getattr(node, field, None)
            if isinstance(value, str):
                names.add(value)
    return names


def _fresh_name(base, taken):
    name, index = base, 1
    while name in taken:
        name, index = f"{base}_{index}", index + 1
    taken.add(name)
    return name


def _narrow_integer_type(type_node):
    """The lane count of an integer scalar or vector type other than ``vec3<u32>``; ``None`` otherwise."""
    value = type_from_node(type_node)
    if value is None or value.kind not in ("i", "u") or value.lanes > 3:
        return None
    if value.kind == "u" and value.bits == 32 and value.lanes == 3:
        return None
    return value.lanes


def lower_narrow_compute_builtins(program, builtin_of) -> int:
    """Receive narrow compute built-in parameters of entry points as ``vec3<u32>``, in place; return the count.

    ``builtin_of(attribute_name)`` names the WGSL built-in a parameter attribute denotes, or returns ``None``.
    """
    if program is None or not hasattr(program, "walk"):
        return 0
    taken = None
    count = 0
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        if not isinstance(entry, A.FunctionNode) or not isinstance(
            entry.body, A.BlockNode
        ):
            continue
        declarations = []
        for index, parameter in enumerate(entry.parameters or []):
            builtins = {
                builtin_of(str(getattr(attribute, "name", attribute)))
                for attribute in parameter.attributes or []
            }
            builtin = next(iter(builtins & VECTOR_COMPUTE_BUILTINS), None)
            lanes = _narrow_integer_type(parameter.param_type)
            if builtin is None or lanes is None:
                continue
            if taken is None:
                taken = _identifiers(program)
            received = copy.copy(parameter)
            received.name = _fresh_name(f"{parameter.name}_{builtin}", taken)
            received.param_type = A.VectorType(A.PrimitiveType("uint"), 3)
            entry.parameters[index] = received
            components = A.IdentifierNode(received.name)
            if lanes < 3:
                components = A.SwizzleNode(components, _COMPONENTS[:lanes])
            declarations.append(
                A.VariableNode(
                    parameter.name,
                    parameter.param_type,
                    A.CastNode(components, parameter.param_type),
                )
            )
            count += 1
        entry.body.statements[0:0] = declarations
    return count
