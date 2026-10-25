"""Workgroup variables that a task entry passes to ``DispatchMesh`` as its payload.

A CrossGL task entry declares its payload as ``groupshared`` storage and passes it to
``DispatchMesh(x, y, z, payload)``, as HLSL does. GLSL and SPIR-V instead declare the payload in task-payload storage
(``taskPayloadSharedEXT``, ``TaskPayloadWorkgroupEXT``), which a task workgroup shares like ``shared``/``Workgroup``
storage and which ``EmitMeshTasksEXT`` passes on (GL_EXT_mesh_shader, SPV_EXT_mesh_shader). Those generators declare
a ``groupshared`` variable named by a ``DispatchMesh`` payload operand in task-payload storage instead of workgroup
storage, so the same IR reaches every target.
"""

from __future__ import annotations

__all__ = ["dispatch_mesh_payload_names"]


def _children(node):
    if isinstance(node, (list, tuple)):
        yield from node
        return
    if isinstance(node, dict):
        yield from node.values()
        return
    for key, value in getattr(node, "__dict__", {}).items():
        if key != "parent":
            yield value


def _name(node):
    for attribute in ("name", "identifier"):
        value = getattr(node, attribute, None)
        if isinstance(value, str):
            return value
    return None


def dispatch_mesh_payload_names(root) -> set:
    """The names that ``DispatchMesh`` calls under ``root`` pass directly as their payload operand."""
    names = set()
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or isinstance(node, (str, int, float, bool)):
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        operation = getattr(node, "operation", None) or getattr(node, "name", None)
        arguments = getattr(node, "arguments", None) or getattr(node, "args", None)
        if operation == "DispatchMesh" and isinstance(arguments, list):
            if len(arguments) >= 4:
                name = _name(arguments[3])
                if name:
                    names.add(name)
        stack.extend(_children(node))
    return names
