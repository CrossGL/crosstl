"""Depth textures that Vulkan GLSL only names at their comparison sites, typed as CrossGL ``texture*Shadow``.

GL_KHR_vulkan_glsl declares a separate depth texture as a plain ``texture2D`` and turns it into a depth texture only
where it is combined with a sampler: ``texture(sampler2DShadow(shadowMap, compareSampler), ...)``.  glslang follows
that usage and declares such a texture with ``Depth`` 1 in SPIR-V.  CrossGL states it in the type instead
(``texture2DShadow``), so HLSL declares ``Texture2D<float>``, MSL ``depth2d<float>`` and SPIR-V an ``OpTypeImage`` with
``Depth`` 1 for it.

:func:`retype_shadow_textures` therefore gives each texture *slot* (a global, or a function parameter) the shadow type
when every use of it, directly or through the arguments it is passed to, is a ``sampler*Shadow`` constructor.  Slots
that one call joins (an argument and the parameter it binds) must agree on the type, so they are merged first and
decided together.  A slot that is also combined with a non-shadow constructor (``sampler2D(t, s)``) keeps its plain
type: CrossGL samples a plain ``texture2D`` with either kind of sampler, while a ``texture2DShadow`` is only compared.
"""

from __future__ import annotations

import re

from crosstl.backend.common_ast import (
    ArrayAccessNode,
    ASTNode,
    FunctionCallNode,
    FunctionNode,
    VariableNode,
)
from crosstl.backend.usage_slots import UsageSlots

_PLAIN_TEXTURE = re.compile(r"texture(1D|2D|Cube)(Array)?")
_SHADOW_CONSTRUCTOR = re.compile(r"sampler(1D|2D|Cube)(Array)?Shadow")
_PLAIN_CONSTRUCTOR = re.compile(
    r"[iu]?sampler(1D|2D|3D|Cube|2DRect|Buffer)(MS)?(Array)?"
)
# GL_EXT_samplerless_texture_functions queries a texture without a sampler; a depth texture answers them unchanged.
_NEUTRAL_QUERIES = frozenset({"textureSize", "textureQueryLevels", "textureSamples"})


def _call_name(node):
    name = getattr(node, "name", None)
    if isinstance(name, VariableNode):
        return name.name
    return name if isinstance(name, str) else None


def _root_name(expression):
    """Name of the variable an argument denotes, looking through array indexing (``shadowMaps[i]``)."""

    while isinstance(expression, ArrayAccessNode):
        expression = getattr(expression, "array", None)
    if isinstance(expression, VariableNode) and isinstance(expression.name, str):
        return expression.name
    return None


def _children(node):
    for value in vars(node).values():
        if isinstance(value, ASTNode):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, ASTNode):
                    yield item


def _calls(node):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, FunctionCallNode):
            yield current
        stack.extend(reversed(list(_children(current))))


def _base_type(vtype):
    return vtype if isinstance(vtype, str) else None


def _global_textures(shader):
    seen = set()
    for attribute in ("uniforms", "global_variables"):
        for variable in getattr(shader, attribute, None) or []:
            if id(variable) in seen or not isinstance(variable, VariableNode):
                continue
            seen.add(id(variable))
            if _PLAIN_TEXTURE.fullmatch(_base_type(variable.vtype) or ""):
                yield variable


def retype_shadow_textures(shader) -> None:
    """Give the plain textures that are only ever compared their CrossGL ``texture*Shadow`` type, in place."""

    functions = [
        function
        for function in getattr(shader, "functions", None) or []
        if isinstance(function, FunctionNode)
    ]
    globals_by_name = {variable.name: variable for variable in _global_textures(shader)}
    if not globals_by_name and not any(
        _PLAIN_TEXTURE.fullmatch(_base_type(getattr(param, "vtype", None)) or "")
        for function in functions
        for param in function.params or []
    ):
        return

    slots = UsageSlots()
    for function in functions:
        local_slots = {
            param.name: ("param", id(function), index)
            for index, param in enumerate(function.params or [])
            if _PLAIN_TEXTURE.fullmatch(_base_type(getattr(param, "vtype", None)) or "")
        }

        def slot_of(expression):
            name = _root_name(expression)
            if name in local_slots:
                return local_slots[name]
            if name in globals_by_name:
                return ("global", name)
            return None

        for call in _calls(function):
            name = _call_name(call)
            args = getattr(call, "args", None) or []
            if not name or not args:
                continue
            if _SHADOW_CONSTRUCTOR.fullmatch(name) or _PLAIN_CONSTRUCTOR.fullmatch(
                name
            ):
                slot = slot_of(args[0])
                if slot is not None:
                    slots.use(slot, "shadow" if "Shadow" in name else "plain")
                continue
            callees = [
                callee
                for callee in functions
                if callee.name == name and len(callee.params or []) == len(args)
            ]
            if not callees:
                # Any other built-in that reads the texture itself (texelFetch) needs the plain type.
                slot = slot_of(args[0])
                if slot is not None and name not in _NEUTRAL_QUERIES:
                    slots.use(slot, "plain")
                continue
            # A user function joins each texture argument with the parameter it binds. Overloads that share the name
            # are joined too: deciding them apart would need overload resolution, and joining only keeps types plain.
            for callee in callees:
                for index, (param, arg) in enumerate(zip(callee.params, args)):
                    arg_slot = slot_of(arg)
                    if arg_slot is None or not _PLAIN_TEXTURE.fullmatch(
                        _base_type(getattr(param, "vtype", None)) or ""
                    ):
                        continue
                    slots.join(arg_slot, ("param", id(callee), index))

    for name, variable in globals_by_name.items():
        if slots.only(("global", name), "shadow"):
            variable.vtype = f"{variable.vtype}Shadow"
    for function in functions:
        for index, param in enumerate(function.params or []):
            if _PLAIN_TEXTURE.fullmatch(
                _base_type(getattr(param, "vtype", None)) or ""
            ) and slots.only(("param", id(function), index), "shadow"):
                param.vtype = f"{param.vtype}Shadow"
