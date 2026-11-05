"""OpenGL's and Direct3D's instance index on Metal's base-inclusive ``[[instance_id]]``.

Metal's ``[[instance_id]]`` includes the draw's base instance: "The per-instance identifier, which includes the base
instance value if one is specified" (Metal Shading Language Specification 3.2, table 5.2), as Vulkan's
``InstanceIndex`` does. OpenGL's ``gl_InstanceID`` counts from zero: "The value of <baseinstance> does not affect
the shader-visible value of gl_InstanceID" (GL_ARB_base_instance; OpenGL 4.6 core profile, section 10.4), and so does
Direct3D's ``SV_InstanceID``, which CrossGL reads as ``gl_InstanceID`` (slangc, for one, lowers ``SV_InstanceID``
to MSL ``instance_id - base_instance``). The Metal generator bound ``gl_InstanceID`` to ``[[instance_id]]``, so every
draw with a non-zero base instance read shifted instance numbers.

:func:`offset_metal_draw_base_indices` rewrites each vertex entry point that takes a ``gl_InstanceID`` parameter
(those of stage-input structures are parameters by then): the parameter is dropped, the entry point takes
``[[instance_id]]`` and ``[[base_instance]]`` (reusing parameters that already carry them), and the body starts by
declaring the dropped parameter as their difference, which is exact since ``[[instance_id]]`` counts from the base.
``gl_InstanceIndex`` keeps ``[[instance_id]]``. The vertex index needs no change: OpenGL's ``gl_VertexID`` includes
the base vertex (OpenGL 4.6, section 11.1.3.9), as ``[[vertex_id]]`` does. The input AST is not modified.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import (
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    FunctionCallNode,
    IdentifierNode,
    ParameterNode,
    PrimitiveType,
    VariableNode,
)
from .stage_utils import collect_stage_entry_records

__all__ = ["offset_metal_draw_base_indices"]

# Metal attribute -> (base-inclusive CrossGL semantic, base attribute, base CrossGL semantic, generated-name stem).
_KINDS = {
    "instance_id": ("gl_InstanceIndex", "base_instance", "gl_BaseInstance", "Instance"),
}


def _counts_from_zero(semantic):
    """The Metal attribute of a built-in ``semantic`` that leaves out the base instance, else ``None``."""
    text = str(semantic)
    # HLSL semantics are case-insensitive; CrossGL also spells SV_InstanceID as sv_instance_id.
    if text == "gl_InstanceID" or text.lower() in ("sv_instanceid", "sv_instance_id"):
        return "instance_id"
    return None


def _type_name(type_node):
    if isinstance(type_node, str):
        return type_node
    name = getattr(type_node, "name", None)
    return name if isinstance(name, str) else None


def _iter_nodes(root):
    """Yield every AST node reachable from ``root`` once, in a deterministic order."""
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or isinstance(node, (str, int, float, bool)):
            continue
        if isinstance(node, (list, tuple)):
            stack.extend(reversed(node))
            continue
        if isinstance(node, dict):
            stack.extend(reversed(list(node.values())))
            continue
        if not hasattr(node, "__dict__") or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(reversed(list(vars(node).values())))


def _taken_names(function):
    names = set()
    for node in _iter_nodes(function):
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
        if isinstance(node, IdentifierNode):
            names.add(str(node.identifier))
    return names


def _fresh(stem, taken):
    candidate, suffix = stem, 1
    while candidate in taken:
        suffix += 1
        candidate = f"{stem}{suffix}"
    taken.add(candidate)
    return candidate


class _Entry:
    """One vertex entry point's index and base parameters, by Metal attribute."""

    def __init__(self, function, codegen):
        self.function = function
        self.exclusive = []
        self.inclusive = {}
        self.bases = {}
        for parameter in getattr(function, "parameters", None) or []:
            semantic = codegen.semantic_from_node(parameter)
            if semantic is None:
                continue
            attribute = _counts_from_zero(semantic)
            if attribute is not None:
                # Any other type is left to the generator, which refuses it for the attribute.
                if _type_name(getattr(parameter, "param_type", None)) in (
                    "int",
                    "uint",
                ):
                    self.exclusive.append((parameter, attribute))
                continue
            attribute = codegen.canonical_metal_semantic(semantic)
            if attribute in _KINDS:
                self.inclusive.setdefault(attribute, parameter.name)
            for kind, (_semantic, base, _base_semantic, _stem) in _KINDS.items():
                if attribute == base:
                    self.bases.setdefault(kind, parameter.name)

    def source(self, attribute, taken):
        """``(index, base)`` parameter names for ``attribute``, adding the parameters that are missing."""
        semantic, _base, base_semantic, stem = _KINDS[attribute]
        if attribute not in self.inclusive:
            self.inclusive[attribute] = self.add(
                f"_crossgl{stem}Index", semantic, taken
            )
        if attribute not in self.bases:
            self.bases[attribute] = self.add(
                f"_crossglBase{stem}", base_semantic, taken
            )
        return self.inclusive[attribute], self.bases[attribute]

    def add(self, stem, semantic, taken):
        name = _fresh(stem, taken)
        self.function.parameters.append(
            ParameterNode(
                name, PrimitiveType("uint"), attributes=[AttributeNode(semantic)]
            )
        )
        return name

    def rewrite(self):
        taken = _taken_names(self.function)
        dropped = {id(parameter) for parameter, _attribute in self.exclusive}
        self.function.parameters = [
            parameter
            for parameter in self.function.parameters
            if id(parameter) not in dropped
        ]
        prologue = []
        for parameter, attribute in self.exclusive:
            index, base = self.source(attribute, taken)
            difference = BinaryOpNode(IdentifierNode(index), "-", IdentifierNode(base))
            param_type = deepcopy(parameter.param_type)
            value = FunctionCallNode(
                IdentifierNode(_type_name(param_type)), [difference]
            )
            prologue.append(
                VariableNode(parameter.name, param_type, initial_value=value)
            )
        body = self.function.body
        if isinstance(body, BlockNode):
            body.statements[0:0] = prologue
        elif isinstance(body, list):
            body[0:0] = prologue
        else:
            self.function.body = BlockNode(prologue)


def _vertex_entries(ast):
    seen = set()
    for _id, _stage, function in collect_stage_entry_records(ast, None, {"vertex"}):
        if id(function) not in seen:
            seen.add(id(function))
            yield function


def offset_metal_draw_base_indices(ast, codegen):
    """Return ``ast`` with each vertex entry point's ``gl_InstanceID`` computed from Metal's ``[[instance_id]]``.

    ``codegen`` is the Metal emitter, whose ``semantic_from_node`` and ``canonical_metal_semantic`` name the
    parameters' attributes. Returns ``ast`` itself when no vertex entry point takes ``gl_InstanceID``.
    """
    if not any(
        _Entry(function, codegen).exclusive for function in _vertex_entries(ast)
    ):
        return ast
    ast = deepcopy(ast)
    for function in _vertex_entries(ast):
        entry = _Entry(function, codegen)
        if entry.exclusive:
            entry.rewrite()
    return ast
