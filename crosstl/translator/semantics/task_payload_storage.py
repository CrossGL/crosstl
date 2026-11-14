"""Module-scope task payloads declared in task-payload storage for GLSL and SPIR-V targets.

An HLSL amplification shader, and the CrossGL the HLSL importer writes for it, declares its payload as
``groupshared`` storage at module scope and passes it to ``DispatchMesh(x, y, z, payload)`` (HLSL shader model 6.5).
GLSL and SPIR-V declare a task payload in its own storage class instead, ``taskPayloadSharedEXT`` and
``TaskPayloadWorkgroupEXT``, which the invocations of a task workgroup share and which ``EmitMeshTasksEXT`` hands
to the mesh workgroups it launches (GL_EXT_mesh_shader, SPV_EXT_mesh_shader). The GLSL and SPIR-V emitters declare a
payload in that storage when it is a local of the task entry (:mod:`..codegen.task_payload_variables`) or carries
the ``taskPayloadSharedEXT`` attribute, as the GLSL importer writes it; a module-scope ``groupshared`` payload was
refused.

:func:`declare_task_payload_storage` gives each module-scope ``groupshared`` variable that a task stage passes to
``DispatchMesh`` the ``taskPayloadSharedEXT`` attribute in place of its ``groupshared`` qualifier. A payload that
another stage also uses is left alone, and the emitters refuse it as before.
"""

from __future__ import annotations

from .. import ast as A
from .stage_varyings import GLSL_TARGETS, SPIRV_TARGETS, slots
from .stage_varyings import stage_name as _stage_name
from .stage_varyings import stages as _stages

__all__ = ["declare_task_payload_storage"]

_TARGETS = GLSL_TARGETS | SPIRV_TARGETS
_TASK_STAGES = frozenset({"task", "amplification", "object"})


def _functions(program, stage):
    functions = [stage.entry_point]
    for function in [
        *(getattr(stage, "local_functions", None) or []),
        *(getattr(program, "functions", None) or []),
    ]:
        if isinstance(function, A.FunctionNode) and function is not stage.entry_point:
            functions.append(function)
    return functions


def _identifiers(program, stage):
    names = set()
    for function in _functions(program, stage):
        for _p, _f, _k, node in slots(getattr(function, "body", None)):
            if isinstance(node, A.IdentifierNode):
                names.add(node.name)
    return names


def _payload_names(program, stage):
    names = set()
    for function in _functions(program, stage):
        for _p, _f, _k, node in slots(getattr(function, "body", None)):
            if (
                isinstance(node, A.MeshOpNode)
                and node.operation == "DispatchMesh"
                and len(node.arguments) == 4
                and isinstance(node.arguments[3], A.IdentifierNode)
            ):
                names.add(node.arguments[3].name)
    return names


def declare_task_payload_storage(program, target) -> int:
    """Declare the module-scope ``groupshared`` task payloads of ``program`` in task-payload storage."""
    if (
        not isinstance(program, A.ShaderNode)
        or str(target or "").lower() not in _TARGETS
    ):
        return 0
    stages = [
        stage
        for stage in _stages(program)
        if getattr(stage, "entry_point", None) is not None
    ]
    changed = 0
    for stage in stages:
        if _stage_name(stage) not in _TASK_STAGES:
            continue
        payloads = _payload_names(program, stage)
        if not payloads:
            continue
        others = set()
        for other in stages:
            if other is not stage:
                others |= _identifiers(program, other)
        for variable in getattr(program, "global_variables", None) or []:
            qualifiers = [str(q) for q in getattr(variable, "qualifiers", None) or []]
            if (
                not isinstance(variable, A.VariableNode)
                or variable.name not in payloads
                or variable.name in others
                or "groupshared" not in qualifiers
            ):
                continue
            variable.qualifiers = [q for q in qualifiers if q != "groupshared"]
            variable.attributes = list(variable.attributes or []) + [
                A.AttributeNode("taskPayloadSharedEXT")
            ]
            changed += 1
    return changed
