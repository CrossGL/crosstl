"""Entry-point parameters that receive the workgroup size, which WGSL has no built-in input for.

A CrossGL entry-point parameter marked ``@gl_WorkGroupSize``, the import of MSL ``[[threads_per_threadgroup]]``
(MSL section 5.2.3.6), receives the size of the dispatching workgroup. WGSL's compute built-in inputs are
``local_invocation_id``, ``local_invocation_index``, ``global_invocation_id``, ``workgroup_id``, ``num_workgroups`` and
the subgroup ones (WGSL section 13.3.1.1); the size is the entry's ``@workgroup_size`` instead, which the generator
spells as constants or pipeline overrides and which ``gl_WorkGroupSize`` reads.

:func:`lower_workgroup_size_inputs` removes such a parameter from the entry point and declares it as a local at the
top of the body, initialized from ``gl_WorkGroupSize``: its leading components, converted to the parameter's type.
A parameter of another type, which holds no workgroup size, is left for the generator to refuse.
"""

from __future__ import annotations

from .. import ast as A
from ..semantics.hub_types import type_from_node

__all__ = ["lower_workgroup_size_inputs"]

_WORKGROUP_SIZE_ATTRIBUTES = frozenset({"gl_workgroupsize", "threads_per_threadgroup"})
_COMPONENTS = "xyz"


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if hasattr(stages, "values") else list(stages)


def _receives_workgroup_size(parameter) -> bool:
    return any(
        str(getattr(attribute, "name", attribute)).lower() in _WORKGROUP_SIZE_ATTRIBUTES
        for attribute in getattr(parameter, "attributes", None) or []
    )


def _lanes(type_node):
    value = type_from_node(type_node)
    if value is None or value.kind not in ("i", "u") or value.lanes > 3:
        return None
    return value.lanes


def lower_workgroup_size_inputs(program) -> int:
    """Read entry-point workgroup-size parameters from ``gl_WorkGroupSize``, in place; return the count."""
    if program is None:
        return 0
    count = 0
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        if not isinstance(entry, A.FunctionNode) or not isinstance(
            entry.body, A.BlockNode
        ):
            continue
        kept = []
        declarations = []
        for parameter in entry.parameters or []:
            lanes = (
                _lanes(parameter.param_type)
                if _receives_workgroup_size(parameter)
                else None
            )
            if lanes is None:
                kept.append(parameter)
                continue
            size = A.IdentifierNode("gl_WorkGroupSize")
            if lanes < 3:
                size = A.SwizzleNode(size, _COMPONENTS[:lanes])
            declarations.append(
                A.VariableNode(
                    parameter.name,
                    parameter.param_type,
                    A.CastNode(size, parameter.param_type),
                )
            )
        if not declarations:
            continue
        entry.parameters = kept
        entry.body.statements[0:0] = declarations
        count += len(declarations)
    return count
