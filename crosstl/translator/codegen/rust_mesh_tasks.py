"""``DispatchMesh`` in a task stage, lowered to rust-gpu's mesh-task launch.

A CrossGL task (amplification, object) entry ends with ``DispatchMesh(x, y, z)`` or ``DispatchMesh(x, y, z,
payload)``. rust-gpu spells the operation, ``OpEmitMeshTasksEXT`` (SPV_EXT_mesh_shader), as two functions in
``spirv_std::arch`` (rust-gpu 0.11.0, ``crates/spirv-std/src/arch/mesh_shading.rs``)::

    pub unsafe fn emit_mesh_tasks_ext(group_count_x: u32, group_count_y: u32, group_count_z: u32) -> !
    pub unsafe fn emit_mesh_tasks_ext_payload<T>(group_count_x: u32, group_count_y: u32, group_count_z: u32,
                                                 payload: &mut T) -> !

Both are ``unsafe`` and end the invocation. The group counts are ``u32``, so a count of another integer type is
converted.

The payload is the task workgroup's shared payload variable itself: SPV_EXT_mesh_shader requires the operand to be an
``OpVariable`` in ``TaskPayloadWorkgroupEXT`` storage, and rust-gpu rejects a borrow of any other storage ("Payload
OpVariable must have a storage class of TaskPayloadWorkgroupEXT"). The operand must therefore name a program-scope
``groupshared`` (or ``taskPayloadSharedEXT``) variable that no stage variable, parameter or local of the entry
shadows; anything else is refused, as the SPIR-V, GLSL, HLSL and Metal generators refuse it. ``DispatchMesh(x, y, z)`` passes the stage's one
``taskPayloadSharedEXT`` variable when it has one, as GLSL's ``EmitMeshTasksEXT`` and the SPIR-V generator do, and is
refused when two would be candidates.
"""

from __future__ import annotations

from .. import ast as A

__all__ = [
    "TASK_STAGES",
    "generate_rust_dispatch_mesh",
    "rust_dispatch_mesh_payload_roots",
    "rust_task_payload_declarations",
]

TASK_STAGES = frozenset({"task", "amplification", "object"})
_ARCH = "spirv_std::arch"
_WORKGROUP = frozenset({"shared", "workgroup", "groupshared"})
_TASK_PAYLOAD = frozenset(
    {"taskpayloadshared", "taskpayloadsharedext", "task_payload_shared"}
)


def _metadata(node):
    names = {
        str(qualifier).strip().lower()
        for qualifier in getattr(node, "qualifiers", None) or []
    }
    for attribute in getattr(node, "attributes", None) or []:
        names.add(str(getattr(attribute, "name", attribute)).strip().lower())
    return names


def _is_task_payload(node):
    return bool(_metadata(node) & _TASK_PAYLOAD)


def _is_workgroup(node):
    return bool(_metadata(node) & (_WORKGROUP | _TASK_PAYLOAD))


def _children(node):
    if isinstance(node, (list, tuple)):
        return list(node)
    if isinstance(node, dict):
        return list(node.values())
    return [
        value for key, value in getattr(node, "__dict__", {}).items() if key != "parent"
    ]


def _body_declarations(body):
    """The variable declarations anywhere in ``body``, in source order."""
    declarations = []
    seen = set()
    stack = [body]
    while stack:
        node = stack.pop()
        if (
            node is None
            or isinstance(node, (str, int, float, bool))
            or id(node) in seen
        ):
            continue
        seen.add(id(node))
        if (
            isinstance(node, A.VariableNode)
            and getattr(node, "vtype", None) is not None
        ):
            declarations.append(node)
        stack.extend(reversed(_children(node)))
    return declarations


def rust_task_payload_declarations(generator, func, stage_node, shader_type):
    """The program-scope task payload candidate each name in a task entry resolves to, or ``None`` outside a task
    stage.

    A name maps to its program variable when that is ``groupshared`` and nothing nearer declares the name: a stage
    variable, an entry parameter or any declaration in the entry's body shadows it and maps the name to ``None``, so
    a payload operand is never taken for workgroup storage it might not name.
    """
    if shader_type not in TASK_STAGES:
        return None
    declarations = {}
    for node in getattr(generator, "rust_program_variables", None) or []:
        name = getattr(node, "name", None)
        if isinstance(name, str):
            declarations[name] = node if _is_workgroup(node) else None
    nearer = (
        list(getattr(stage_node, "local_variables", None) or [])
        + list(getattr(func, "parameters", None) or getattr(func, "params", None) or [])
        + _body_declarations(getattr(func, "body", None))
    )
    for node in nearer:
        name = getattr(node, "name", None)
        if isinstance(name, str):
            declarations[name] = None
    return declarations


def _payload_name(argument):
    if isinstance(argument, (A.IdentifierNode, A.VariableNode)):
        name = getattr(argument, "name", None)
        return name if isinstance(name, str) else None
    return None


def _resolved_payload(generator, arguments):
    """``(name, None)`` for the payload a ``DispatchMesh`` passes, ``(None, None)`` for none, or ``(None, reason)``."""
    declarations = getattr(generator, "current_rust_task_declarations", None) or {}
    if len(arguments) == 4:
        name = _payload_name(arguments[3])
        if not (name and declarations.get(name) is not None):
            return None, (
                "DispatchMesh payload argument: the payload must be a program-scope groupshared variable, because "
                "rust-gpu's emit_mesh_tasks_ext_payload passes the mesh workgroups the task payload variable "
                "(TaskPayloadWorkgroupEXT storage) and rejects any other"
            )
        return name, None
    payloads = sorted(
        name
        for name, node in declarations.items()
        if node is not None and _is_task_payload(node)
    )
    if len(payloads) > 1:
        return None, (
            "DispatchMesh without a payload argument: the stage has more than one taskPayloadSharedEXT variable "
            f"({', '.join(payloads)}), so the payload the mesh workgroups read is ambiguous"
        )
    return (payloads[0] if payloads else None), None


def _group_count(generator, argument):
    """The Rust ``u32`` for one group-count operand."""
    text = generator.generate_expression(argument)
    if generator.literal_int_value(argument) is not None:
        return text
    scalar = generator.normalize_scalar_type(generator.expression_result_type(argument))
    if scalar in (None, "u32"):
        return text
    return f"({text}) as u32"


def generate_rust_dispatch_mesh(generator, arguments):
    """The Rust statement expression for ``DispatchMesh`` with already validated ``arguments``."""
    arguments = list(arguments or [])
    payload, refusal = _resolved_payload(generator, arguments)
    if refusal:
        return generator.refuse_lowering(refusal, "()")
    counts = ", ".join(_group_count(generator, argument) for argument in arguments[:3])
    if payload is None:
        return f"unsafe {{ {_ARCH}::emit_mesh_tasks_ext({counts}) }}"
    location = generator.generate_expression(A.IdentifierNode(payload))
    return (
        f"unsafe {{ {_ARCH}::emit_mesh_tasks_ext_payload({counts}, &mut {location}) }}"
    )


def rust_dispatch_mesh_payload_roots(generator, node):
    """The binding a ``DispatchMesh`` call borrows with ``&mut``, which must therefore be declared ``let mut``."""
    if (
        not isinstance(node, A.MeshOpNode)
        or getattr(node, "operation", None) != "DispatchMesh"
    ):
        return set()
    payload, _refusal = _resolved_payload(
        generator, list(getattr(node, "arguments", []) or [])
    )
    return {payload} if payload else set()
