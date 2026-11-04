"""Integer built-in inputs whose source type is not the WGSL one.

WGSL declares the compute built-in inputs ``global_invocation_id``, ``local_invocation_id``, ``workgroup_id`` and
``num_workgroups`` as ``vec3<u32>``, and ``vertex_index``, ``instance_index``, ``local_invocation_index``,
``sample_index`` and ``primitive_index`` as ``u32``, and nothing else (WGSL section 13.3.1.1). MSL accepts ``uint``
and ``uint2`` for the three-dimensional attributes (``thread_position_in_grid`` and the like, MSL section 5.2.3.6),
HLSL accepts ``uint``, ``uint2``, ``int`` and ``int2`` forms of ``SV_DispatchThreadID`` and its siblings, and CrossGL
types ``gl_VertexID``, ``gl_PrimitiveID`` and their siblings ``int`` as GLSL does. A narrower value holds the leading
components of the three-dimensional one.

:func:`lower_integer_builtin_inputs` receives such an entry-point parameter in the WGSL type under a fresh name,
and declares the source parameter as a local at the top of the body, initialized with the leading components
converted to its type. Reads and writes of the parameter then use the local, which WGSL, unlike a parameter, lets
the body write.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..semantics.hub_types import type_from_node

__all__ = ["SCALAR_BUILTINS", "VECTOR_COMPUTE_BUILTINS", "lower_integer_builtin_inputs"]

VECTOR_COMPUTE_BUILTINS = frozenset(
    {"global_invocation_id", "local_invocation_id", "workgroup_id", "num_workgroups"}
)
SCALAR_BUILTINS = frozenset(
    {
        "vertex_index",
        "instance_index",
        "local_invocation_index",
        "sample_index",
        "primitive_index",
    }
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


def _narrow_integer_type(type_node, wgsl_lanes):
    """The lane count of an integer type that ``wgsl_lanes`` lanes of ``u32`` hold and that is not that type."""
    value = type_from_node(type_node)
    if value is None or value.kind not in ("i", "u") or value.lanes > wgsl_lanes:
        return None
    if value.kind == "u" and value.bits == 32 and value.lanes == wgsl_lanes:
        return None
    return value.lanes


def lower_integer_builtin_inputs(program, builtin_of) -> int:
    """Receive integer built-in parameters of entry points in their WGSL types, in place; return the count.

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
            builtin = next(
                iter(builtins & (VECTOR_COMPUTE_BUILTINS | SCALAR_BUILTINS)), None
            )
            if builtin is None:
                continue
            wgsl_lanes = 3 if builtin in VECTOR_COMPUTE_BUILTINS else 1
            lanes = _narrow_integer_type(parameter.param_type, wgsl_lanes)
            if lanes is None:
                continue
            if taken is None:
                taken = _identifiers(program)
            received = copy.copy(parameter)
            received.name = _fresh_name(f"{parameter.name}_{builtin}", taken)
            received.param_type = (
                A.VectorType(A.PrimitiveType("uint"), 3)
                if wgsl_lanes == 3
                else A.PrimitiveType("uint")
            )
            entry.parameters[index] = received
            components = A.IdentifierNode(received.name)
            if lanes < wgsl_lanes:
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
