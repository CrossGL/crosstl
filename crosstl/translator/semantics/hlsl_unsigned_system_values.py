"""HLSL entry-point parameters for the system values DXIL delivers only as ``uint``.

CrossGL types the vertex, instance, primitive and sample indices ``int``, as GLSL's ``gl_VertexID``,
``gl_InstanceID``, ``gl_PrimitiveID`` and ``gl_SampleID`` are, and the Metal import reads MSL's ``uint``
``[[vertex_id]]`` the same way. DXIL requires ``SV_VertexID``, ``SV_InstanceID``, ``SV_PrimitiveID`` and
``SV_SampleIndex`` to be ``uint``, and DXC's validator rejects an ``int`` entry-point parameter carrying one ("SV_VertexID
must be uint").

For HLSL targets :func:`deliver_unsigned_system_values` declares each such ``int`` parameter of a stage entry point
as a ``uint`` parameter with a fresh name and converts it into the original name once at the start of the entry
point (``int vid = int(vid_bits);``), so the body and the helpers it calls keep reading an ``int``. An index is
never negative, so the conversion keeps its value.
"""

from __future__ import annotations

from .. import ast as A
from .builtin_values import glsl_builtin_name

__all__ = ["deliver_unsigned_system_values"]

TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
_UNSIGNED = frozenset(
    {
        "gl_VertexID",
        "gl_VertexIndex",
        "gl_InstanceID",
        "gl_InstanceIndex",
        "gl_PrimitiveID",
        "gl_SampleID",
    }
)


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _builtin(parameter):
    for attribute in getattr(parameter, "attributes", None) or []:
        name = glsl_builtin_name(attribute)
        if name is not None:
            return name
    return None


def _is_int(type_node):
    return getattr(type_node, "name", type_node) == "int" and not getattr(
        type_node, "generic_args", None
    )


def _names(function):
    names = {
        getattr(parameter, "name", None) for parameter in function.parameters or []
    }
    stack = [getattr(function, "body", None)]
    while stack:
        node = stack.pop()
        if isinstance(node, A.VariableNode):
            names.add(node.name)
        if isinstance(node, A.ASTNode):
            stack.extend(
                value
                for key, value in vars(node).items()
                if key != "parent" and isinstance(value, A.ASTNode)
            )
            for value in vars(node).values():
                if isinstance(value, list):
                    stack.extend(item for item in value if isinstance(item, A.ASTNode))
    return names


def deliver_unsigned_system_values(program, target) -> int:
    """Declare the ``int`` unsigned system-value parameters of ``program``'s entry points ``uint``, in place."""
    if str(target or "").lower() not in TARGETS or not isinstance(
        program, A.ShaderNode
    ):
        return 0
    changed = 0
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        body = getattr(entry, "body", None)
        if not isinstance(body, A.BlockNode):
            continue
        taken = None
        conversions = []
        for parameter in list(entry.parameters or []):
            if _builtin(parameter) not in _UNSIGNED or not _is_int(
                parameter.param_type
            ):
                continue
            if taken is None:
                taken = _names(entry)
            raw, index = f"{parameter.name}_bits", 1
            while raw in taken:
                index += 1
                raw = f"{parameter.name}_bits{index}"
            taken.add(raw)
            conversions.append(
                A.VariableNode(
                    name=parameter.name,
                    var_type=A.PrimitiveType("int"),
                    initial_value=A.FunctionCallNode(
                        A.IdentifierNode("int"), [A.IdentifierNode(raw)]
                    ),
                )
            )
            parameter.name = raw
            parameter.param_type = A.PrimitiveType("uint")
            changed += 1
        body.statements[0:0] = conversions
    return changed
