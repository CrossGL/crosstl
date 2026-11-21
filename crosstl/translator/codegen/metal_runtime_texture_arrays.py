"""Argument buffers for the CrossGL texture arrays of unknown length that a Metal program indexes at run time.

CrossGL declares a texture array without a length (``uniform sampler2D tex[];``, ``texture2D tex[];``) as GLSL with
GL_EXT_nonuniform_qualifier and HLSL do: a runtime descriptor array (SPIR-V ``RuntimeDescriptorArray``, an
``OpTypeRuntimeArray`` of images) whose length the host picks when it binds the descriptors. A Metal entry point binds
a fixed number of textures to its argument table, so ``array<texture2d<float>, N> tex [[texture(0)]]`` has a length
the shader fixes. A texture array of unknown length exists in Metal only inside an argument buffer: a buffer of
structures that each hold a texture, which the host encodes (MSL specification, "Argument Buffers").

The Metal generator sized such an array from the indices that reach it, which turned the runtime array into a fixed
one (``tex[int(x) & 1]`` became ``array<texture2d<float>, 2>``), and refused an index with no bound.
:func:`lower_runtime_texture_arrays` instead declares the array as an argument buffer when a run-time index reaches it:

* a structure ``<name>_slot`` holds the element: ``texture2D texture`` for a separate texture, and ``texture2D texture;
  sampler sampler_state`` for a combined sampler, whose sampler state the host binds per element as it does for a
  combined image sampler descriptor;
* the declaration becomes ``StructuredBuffer<<name>_slot> <name>``, which the Metal generator writes as
  ``const device <name>_slot* <name> [[buffer(n)]]``. An HLSL ``register(tN)`` stays, since the generator already
  binds the ``t`` registers of buffers to buffer slots. A GLSL ``binding`` names a texture unit, which in OpenGL is a
  namespace apart from the buffer binding points (GLSL 4.60 section 4.4.6), so the generator gives the argument buffer
  a free buffer slot instead of one that another buffer may hold;
* each element ``tex[i]`` becomes ``tex[i].texture``, and a sampling call on a combined sampler's element takes
  ``tex[i].sampler_state`` after it, unless the call already names a sampler there (``texture(tex[i], s, P)``).

An array that only constant indices reach keeps its sized form, as GLSL sizes such an array implicitly (GLSL 4.60
section 4.1.9). So does an array whose shape this pass does not rewrite: one passed whole to a function, one whose
name a function also declares, a combined sampler element used other than as a texture call's first argument or
sampled through an index with side effects, and a shadow sampler. HLSL sources reach this pass through the HLSL
importer, which writes ``Texture2D tex[]`` as the separate ``texture2D tex[]``.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    ASTNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    ParameterNode,
    PrimitiveType,
    StructMemberNode,
    StructNode,
    VariableNode,
)
from ..semantics.rewrite import child_slots, get_child, is_pure, set_child
from .image_access_contracts import TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES

__all__ = ["lower_runtime_texture_arrays"]

TEXTURE_MEMBER = "texture"
SAMPLER_MEMBER = "sampler_state"

_DIMENSIONS = ("1D", "1DArray", "2D", "2DArray", "3D", "Cube", "CubeArray")
# A combined sampler's element: the separate texture type of the same dimension and component type.
_COMBINED_TEXTURES = {
    f"{prefix}sampler{dim}": f"{prefix}texture{dim}"
    for prefix in ("", "i", "u")
    for dim in _DIMENSIONS
}
_SEPARATE_TEXTURES = frozenset(_COMBINED_TEXTURES.values())
_SAMPLER_TYPES = frozenset(
    {
        "sampler",
        "samplerShadow",
        "SamplerState",
        "SamplerComparisonState",
        "comparison_sampler",
    }
)
# Calls that read a combined sampler's texture without its sampler state (GLSL 4.60 sections 8.9.1 and 8.9.2).
_TEXEL_CALLS = frozenset(
    {
        "texelFetch",
        "texelFetchOffset",
        "textureSize",
        "textureQueryLevels",
        "textureSamples",
    }
)


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _functions(program):
    roots = list(getattr(program, "functions", None) or [])
    for stage in _stages(program):
        roots += list(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            roots.append(stage.entry_point)
    return roots


def _walk(node, seen):
    if not isinstance(node, ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key), seen)


def _callee(call):
    function = getattr(call, "function", None)
    if isinstance(function, IdentifierNode):
        return function.name
    return function if isinstance(function, str) else ""


def _type_name(type_node):
    return str(getattr(type_node, "name", "") or "")


def _element(variable):
    """``(texture type name, combined)`` for a texture array of unknown length, else None."""
    var_type = getattr(variable, "var_type", None)
    if not isinstance(var_type, ArrayType) or var_type.size is not None:
        return None
    element = var_type.element_type
    if isinstance(element, ArrayType):
        return None
    name = _type_name(element)
    if name in _COMBINED_TEXTURES:
        return _COMBINED_TEXTURES[name], True
    if name in _SEPARATE_TEXTURES and not getattr(element, "generic_args", None):
        return name, False
    return None


def _candidates(program):
    """The program-scope and stage-local texture arrays of unknown length, by name."""
    declared = {}
    counts = {}
    holders = [program, *_stages(program)]
    for holder in holders:
        field = "global_variables" if holder is program else "local_variables"
        for variable in getattr(holder, field, None) or []:
            name = getattr(variable, "name", None)
            counts[name] = counts.get(name, 0) + 1
            element = _element(variable) if isinstance(variable, VariableNode) else None
            if element is not None:
                declared[name] = element
    return {name: v for name, v in declared.items() if counts[name] == 1}


def _constant_names(program):
    names = set()
    for constant in getattr(program, "constants", None) or []:
        value = getattr(constant, "value", getattr(constant, "initial_value", None))
        if isinstance(value, LiteralNode) and isinstance(value.value, int):
            names.add(constant.name)
    return names


def _is_constant_index(index, constants):
    if isinstance(index, LiteralNode):
        return isinstance(index.value, int) and not isinstance(index.value, bool)
    return isinstance(index, IdentifierNode) and index.name in constants


def _function_declared_names(functions):
    names = set()
    for function in functions:
        for node in _walk(function, set()):
            if isinstance(node, (ParameterNode, VariableNode)):
                names.add(getattr(node, "name", None))
    return names


def _is_sampler_type(type_node):
    if isinstance(type_node, ArrayType):
        return _is_sampler_type(type_node.element_type)
    return _type_name(type_node) in _SAMPLER_TYPES


def _program_sampler_names(program):
    names = set()
    holders = [program, *_stages(program)]
    for holder in holders:
        field = "global_variables" if holder is program else "local_variables"
        for variable in getattr(holder, field, None) or []:
            if _is_sampler_type(getattr(variable, "var_type", None)):
                names.add(variable.name)
    return names


def _function_sampler_names(function):
    names = set()
    for node in _walk(function, set()):
        if isinstance(node, ParameterNode):
            declared = getattr(node, "param_type", getattr(node, "vtype", None))
        elif isinstance(node, VariableNode):
            declared = node.var_type
        else:
            continue
        if _is_sampler_type(declared):
            names.add(node.name)
    return names


def _base_name(expression):
    while isinstance(expression, (ArrayAccessNode, MemberAccessNode)):
        expression = (
            expression.array_expr
            if isinstance(expression, ArrayAccessNode)
            else expression.object_expr
        )
    return expression.name if isinstance(expression, IdentifierNode) else None


class _Uses:
    """Every reference to each candidate array in the program's functions."""

    def __init__(self, functions, names, program_samplers=frozenset()):
        self.accesses = {name: [] for name in names}
        self.references = {name: 0 for name in names}
        # The texture call whose first argument is each access, by the access's id.
        self.calls = {}
        # The ids of the calls that already pass a sampler after the texture.
        self.explicit_samplers = set()
        for function in functions:
            samplers = program_samplers | _function_sampler_names(function)
            for node in _walk(function, set()):
                if isinstance(node, IdentifierNode) and node.name in self.references:
                    self.references[node.name] += 1
                elif isinstance(node, ArrayAccessNode):
                    base = node.array_expr
                    if isinstance(base, IdentifierNode) and base.name in self.accesses:
                        self.accesses[base.name].append(node)
                elif isinstance(node, FunctionCallNode):
                    arguments = list(node.arguments or [])
                    if arguments and isinstance(arguments[0], ArrayAccessNode):
                        self.calls[id(arguments[0])] = node
                        if len(arguments) > 1 and _base_name(arguments[1]) in samplers:
                            self.explicit_samplers.add(id(node))


def _combined_use_supported(access, call, uses):
    if call is None:
        return False
    name = _callee(call)
    if name in _TEXEL_CALLS or id(call) in uses.explicit_samplers:
        return True
    return name in TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES and is_pure(
        access.index_expr
    )


def _lowered(program):
    """``{name: (texture type, combined)}`` for the arrays this pass rewrites."""
    candidates = _candidates(program)
    if not candidates:
        return {}
    functions = _functions(program)
    shadowed = _function_declared_names(functions)
    candidates = {n: v for n, v in candidates.items() if n not in shadowed}
    if not candidates:
        return {}
    uses = _Uses(functions, set(candidates), _program_sampler_names(program))
    constants = _constant_names(program)
    lowered = {}
    for name, (texture, combined) in sorted(candidates.items()):
        accesses = uses.accesses[name]
        # Every reference indexes the array, and a run-time index reaches it.
        if not accesses or len(accesses) != uses.references[name]:
            continue
        if all(_is_constant_index(a.index_expr, constants) for a in accesses):
            continue
        if combined and not all(
            _combined_use_supported(a, uses.calls.get(id(a)), uses) for a in accesses
        ):
            continue
        lowered[name] = (texture, combined)
    return lowered


def _slot_name(name, taken):
    slot = f"{name}_slot"
    suffix = 1
    while slot in taken:
        suffix += 1
        slot = f"{name}_slot{suffix}"
    taken.add(slot)
    return slot


def _taken_type_names(program):
    taken = set()
    holders = [program, *_stages(program)]
    for holder in holders:
        for field in ("structs", "local_structs"):
            for struct in getattr(holder, field, None) or []:
                taken.add(getattr(struct, "name", None))
    return taken


def _slot_struct(slot, texture, combined):
    members = [StructMemberNode(TEXTURE_MEMBER, NamedType(texture))]
    if combined:
        members.append(StructMemberNode(SAMPLER_MEMBER, PrimitiveType("sampler")))
    return StructNode(slot, members)


def _replace_declarations(program, slots):
    holders = [program, *_stages(program)]
    for holder in holders:
        field = "global_variables" if holder is program else "local_variables"
        variables = getattr(holder, field, None)
        if not variables:
            continue
        for index, variable in enumerate(variables):
            slot = slots.get(getattr(variable, "name", None))
            if slot is None or not isinstance(variable, VariableNode):
                continue
            variables[index] = VariableNode(
                variable.name,
                NamedType("StructuredBuffer", [NamedType(slot)]),
                attributes=[
                    deepcopy(attribute)
                    for attribute in variable.attributes or []
                    if str(getattr(attribute, "name", "")).lower() == "register"
                ],
            )


def _rewrite_accesses(program, lowered):
    functions = _functions(program)
    uses = _Uses(functions, set(lowered), _program_sampler_names(program))
    replacements = {}
    for name, (_, combined) in lowered.items():
        for access in uses.accesses[name]:
            replacements[id(access)] = MemberAccessNode(access, TEXTURE_MEMBER)
            call = uses.calls.get(id(access))
            if (
                combined
                and call is not None
                and id(call) not in uses.explicit_samplers
                and _callee(call) in TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES
            ):
                sampler = MemberAccessNode(deepcopy(access), SAMPLER_MEMBER)
                call.arguments.insert(1, sampler)
                if getattr(call, "args", None) is not call.arguments:
                    call.args = call.arguments
    for function in functions:
        _replace(function, replacements, set())


def _replace(node, replacements, seen):
    if not isinstance(node, ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        replacement = replacements.get(id(child))
        if replacement is not None:
            set_child(node, field, key, replacement)
            seen.add(id(replacement))
            # The access itself is the member's object; its index may hold other accesses.
            _replace(child.index_expr, replacements, seen)
            continue
        _replace(child, replacements, seen)


def lower_runtime_texture_arrays(ast):
    """``ast``, or a copy in which each run-time indexed texture array of unknown length is an argument buffer."""
    if not hasattr(ast, "global_variables"):
        return ast
    lowered = _lowered(ast)
    if not lowered:
        return ast
    ast = deepcopy(ast)
    taken = _taken_type_names(ast)
    slots = {}
    structs = list(getattr(ast, "structs", None) or [])
    for name, (texture, combined) in sorted(lowered.items()):
        slots[name] = _slot_name(name, taken)
        structs.append(_slot_struct(slots[name], texture, combined))
    ast.structs = structs
    _replace_declarations(ast, slots)
    _rewrite_accesses(ast, lowered)
    return ast
