"""Constant buffers declared as ``ConstantBuffer<T>`` resources, for targets that bind a uniform struct as one.

CrossGL spells an HLSL ``ConstantBuffer<T> name : register(bN)`` (and Slang's ``ConstantBuffer<T>``) as a module
variable of type ``ConstantBuffer<T>``: one constant buffer whose contents are a ``T``, read as ``name.member``.
SPIR-V and Metal bind exactly that through a uniform variable of the struct type: SPIR-V declares a ``Uniform``
variable whose ``Block`` type is ``T`` (Vulkan 1.0 section 15.1.1 / SPIR-V 2.16.2, as DXC lowers
``ConstantBuffer<T>``), and Metal declares ``constant T &name [[buffer(N)]]``. Their emitters already lower a
``uniform T name`` declaration that way, keeping its binding attributes.

:func:`lower_constant_buffer_resources` rewrites each such declaration into ``uniform T name`` and keeps its
attributes. Member reads need no change, because ``name.member`` already reads a member of a ``T``.

An array of constant buffers (``ConstantBuffer<T> lights[4]``) is an array of descriptors, one buffer each, which
SPIR-V spells as a ``Uniform`` variable of an array of ``Block`` structs and Metal only through an argument buffer.
A ``uniform T name[N]`` declaration instead lowers to one buffer holding the array, so array declarations are left
in place and reported through ``refuse(name)``.
"""

from __future__ import annotations

from copy import copy

from ..ast import ArrayType, NamedType, StageMap

__all__ = ["constant_buffer_element", "lower_constant_buffer_resources"]


def constant_buffer_element(type_node):
    """The element type of a ``ConstantBuffer<T>`` type, or ``None`` for any other type."""
    if isinstance(type_node, NamedType):
        if type_node.name == "ConstantBuffer" and len(type_node.generic_args) == 1:
            return type_node.generic_args[0]
        return None
    if isinstance(type_node, str):
        text = type_node.strip()
        if text.startswith("ConstantBuffer<") and text.endswith(">"):
            inner = text[len("ConstantBuffer<") : -1].strip()
            return NamedType(inner) if inner else None
    return None


def _holds_constant_buffers(type_node):
    while isinstance(type_node, ArrayType):
        type_node = type_node.element_type
    return constant_buffer_element(type_node) is not None


def _lowered_declarations(nodes, refuse):
    """``nodes`` with each ``ConstantBuffer<T>`` declaration replaced by a ``uniform T`` copy, or ``None``."""
    lowered, changed = [], False
    for node in nodes or []:
        declared = getattr(node, "var_type", None) or getattr(node, "vtype", None)
        uniform_type = constant_buffer_element(declared)
        if uniform_type is None:
            if _holds_constant_buffers(declared) and refuse is not None:
                refuse(node.name)
            lowered.append(node)
            continue
        replacement = copy(node)
        for attribute in ("var_type", "vtype"):
            if hasattr(replacement, attribute):
                setattr(replacement, attribute, uniform_type)
        qualifiers = list(getattr(node, "qualifiers", None) or [])
        if "uniform" not in qualifiers:
            qualifiers.append("uniform")
        replacement.qualifiers = qualifiers
        lowered.append(replacement)
        changed = True
    return lowered if changed else None


def lower_constant_buffer_resources(ast, refuse=None):
    """``ast`` with module and stage ``ConstantBuffer<T>`` declarations as ``uniform T``; ``ast`` is not mutated.

    ``refuse`` is called with the name of each array of constant buffers, which stays declared as it was.
    """
    globals_ = _lowered_declarations(getattr(ast, "global_variables", None), refuse)
    stages = getattr(ast, "stages", None) or {}
    # A StageMap holds several entry points per stage key (one per kernel), so keep every (key, stage) pair.
    lowered_stages, stages_changed = StageMap(), False
    for stage_type, stage in stages.items():
        local_variables = _lowered_declarations(
            getattr(stage, "local_variables", None), refuse
        )
        if local_variables is not None:
            stage = copy(stage)
            stage.local_variables = local_variables
            stages_changed = True
        lowered_stages.append(stage_type, stage)
    if globals_ is None and not stages_changed:
        return ast
    result = copy(ast)
    if globals_ is not None:
        result.global_variables = globals_
    if stages_changed:
        result.stages = lowered_stages
    return result
