"""Bound samplers for the CrossGL combined image samplers a Metal program samples.

A CrossGL combined sampler (``sampler2D``, ``samplerCube``, ``sampler2DShadow``, ...) is GLSL's: one uniform names a
texture unit, and the host binds both a texture and a sampler object to that unit (OpenGL 4.6 section 8.2), or one
combined image sampler descriptor (Vulkan ``VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER``). The sampler state, filtering,
addressing, mipmap selection, LOD clamps and the depth comparison, is the host's. MSL has no combined type: a
function takes a ``texture2d`` and a ``sampler`` argument, each bound by the host to its own argument table slot
(MSL 3.2 sections 2.9 and 2.10). The Metal generator sampled combined samplers through an inline
``sampler(mag_filter::linear, min_filter::linear)``, which ignores the host's sampler: mipmapped textures were read at
level 0 only, every address mode became ``clamp_to_edge``, and a shadow comparison used no compare function.

:func:`bind_combined_samplers` declares, for each program-scope combined sampler that a sampling call reads (or that
a call passes on to a helper), a program-scope ``sampler`` named ``<texture>Sampler`` with the texture's binding
attributes. The generator pairs a texture with the sampler of that name, so the entry point takes
``sampler <texture>Sampler [[sampler(n)]]`` and samples through it; ``n`` is the texture's binding, as the HLSL
generator uses ``s<n>`` beside ``t<n>``. A program that already declares a value of that name keeps it. A helper
function's combined sampler parameter likewise takes a ``sampler <parameter>Sampler`` parameter after it, and every
call passes the sampler of its argument.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    ASTNode,
    AttributeNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    ParameterNode,
    PrimitiveType,
    ShaderNode,
    VariableNode,
)
from ..semantics.rewrite import child_slots, get_child

__all__ = ["bind_combined_samplers"]

_DIMENSIONS = ("1D", "1DArray", "2D", "2DArray", "3D", "Cube", "CubeArray")
COMBINED_SAMPLER_TYPES = frozenset(
    [f"{prefix}sampler{dim}" for prefix in ("", "i", "u") for dim in _DIMENSIONS]
    + ["sampler1DShadow", "sampler1DArrayShadow", "sampler2DShadow"]
    + ["sampler2DArrayShadow", "samplerCubeShadow", "samplerCubeArrayShadow"]
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
_BINDING_ATTRIBUTES = frozenset({"binding", "set", "register", "vk_binding"})


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _type_name(type_node):
    if isinstance(type_node, ArrayType):
        return _type_name(type_node.element_type)
    return str(getattr(type_node, "name", type_node) or "")


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


def _argument_name(argument):
    if isinstance(argument, ArrayAccessNode):
        return _argument_name(argument.array_expr)
    return argument.name if isinstance(argument, IdentifierNode) else None


def _functions(program):
    roots = list(getattr(program, "functions", None) or [])
    for stage in _stages(program):
        roots += list(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            roots.append(stage.entry_point)
    return roots


_SAMPLER_TYPES = frozenset(
    {
        "sampler",
        "samplerShadow",
        "SamplerState",
        "SamplerComparisonState",
        "comparison_sampler",
    }
)


def _sampler_names(program, function=None):
    names = {
        v.name
        for v in getattr(program, "global_variables", None) or []
        if _type_name(getattr(v, "var_type", None)) in _SAMPLER_TYPES
    }
    for parameter in getattr(function, "parameters", None) or []:
        if _type_name(getattr(parameter, "param_type", None)) in _SAMPLER_TYPES:
            names.add(parameter.name)
    return names


def _sampler_members(program):
    """Structure member names that hold a sampler or an array of samplers."""
    return {
        member.name
        for struct in getattr(program, "structs", None) or []
        for member in getattr(struct, "members", None) or []
        if _type_name(
            getattr(member, "member_type", None) or getattr(member, "vtype", None)
        )
        in _SAMPLER_TYPES
    }


def _is_sampler(argument, samplers, members):
    if isinstance(argument, ArrayAccessNode):
        return _is_sampler(argument.array_expr, samplers, members)
    if isinstance(argument, MemberAccessNode):
        return argument.member in members
    if isinstance(argument, FunctionCallNode):
        return _callee(argument) == "sampler"
    return _argument_name(argument) in samplers


def _sampled_in(program, roots, candidates, helpers):
    """The candidates a call samples through their own sampler state, or passes to a helper that does."""
    sampled = set()
    seen = set()
    members = _sampler_members(program)
    globals_ = {
        v.name: _type_name(getattr(v, "var_type", None))
        for v in getattr(program, "global_variables", None) or []
    }
    for root in roots:
        samplers = _sampler_names(program, root)
        types = dict(globals_)
        for parameter in getattr(root, "parameters", None) or []:
            types[parameter.name] = _type_name(getattr(parameter, "param_type", None))
        for node in _walk(getattr(root, "body", None), seen):
            if not isinstance(node, FunctionCallNode):
                continue
            arguments = list(getattr(node, "arguments", None) or [])
            callee = _callee(node)
            if (callee, len(arguments)) in helpers:
                # Only the parameters the helper samples with their own sampler state need one.
                indices = helpers[(callee, len(arguments))]
                names = [_argument_name(arguments[i]) for i in indices]
            elif (
                callee.startswith("texture")
                and arguments
                and callee not in _TEXEL_CALLS
            ):
                explicit = len(arguments) >= 3 and _is_sampler(
                    arguments[1], samplers, members
                )
                name = _argument_name(arguments[0])
                # MSL takes no offsets on cube textures; the generator refuses such a call.
                refused = "Offset" in callee and "Cube" in types.get(name, "")
                names = [] if explicit or refused else [name]
            else:
                names = []
            sampled.update(name for name in names if name in candidates)
    return sampled


def _declared_names(program):
    names = set()
    for node in _walk(program, set()):
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _binding_key(attributes):
    return tuple(
        sorted(
            (
                str(a.name).lower(),
                tuple(str(getattr(x, "value", x)) for x in a.arguments or []),
            )
            for a in attributes
        )
    )


def _binding_attributes(node):
    return [
        attribute
        for attribute in getattr(node, "attributes", None) or []
        if str(getattr(attribute, "name", "")).lower() in _BINDING_ATTRIBUTES
    ]


def _sampler_for(texture, name, taken):
    attributes = [
        AttributeNode(attribute.name, deepcopy(list(attribute.arguments or [])))
        for attribute in _binding_attributes(texture)
    ]
    if _binding_key(attributes) in taken:
        # A declared sampler already holds the texture's binding; the generator gives this one a free slot.
        attributes = []
    sampler_type = PrimitiveType("sampler")
    if isinstance(texture.var_type, ArrayType):
        sampler_type = ArrayType(sampler_type, deepcopy(texture.var_type.size))
    return VariableNode(
        name=name,
        var_type=sampler_type,
        attributes=attributes,
        qualifiers=[q for q in texture.qualifiers or [] if str(q) == "uniform"],
    )


def bind_combined_samplers(ast):
    """``ast``, or a copy declaring a bound sampler for each combined sampler a sampling call reads."""
    if not isinstance(ast, ShaderNode):
        return ast
    textures = {
        variable.name: variable
        for variable in getattr(ast, "global_variables", None) or []
        if isinstance(variable, VariableNode)
        and _type_name(variable.var_type) in COMBINED_SAMPLER_TYPES
    }
    if not textures:
        return ast
    helpers = _helper_sampler_parameters(ast)
    sampled = _sampled_in(ast, _functions(ast), set(textures), helpers)
    declared = _declared_names(ast)
    needed = sorted(name for name in sampled if f"{name}Sampler" not in declared)
    if not needed and not helpers:
        return ast
    ast = deepcopy(ast)
    globals_ = list(ast.global_variables)
    taken = {
        _binding_key(_binding_attributes(v))
        for v in globals_
        if _type_name(getattr(v, "var_type", None)) in _SAMPLER_TYPES
        and _binding_attributes(v)
    }
    by_name = {getattr(v, "name", None): v for v in globals_}
    # After every other declaration, so that the samplers the program declares keep the slots they had.
    for name in needed:
        globals_.append(_sampler_for(by_name[name], f"{name}Sampler", taken))
    ast.global_variables = globals_
    if helpers:
        _thread_helper_samplers(ast)
    return ast


def _helper_sampler_parameters(program):
    """``{(name, parameter count): [index, ...]}``: the combined sampler parameters each helper samples.

    A helper that passes a parameter on to a helper sampling it samples it too, so the map grows to a fixed point.
    """
    entries = {id(getattr(stage, "entry_point", None)) for stage in _stages(program)}
    helpers = [f for f in _functions(program) if id(f) not in entries]
    result = {}
    changed = True
    while changed:
        changed = False
        for function in helpers:
            parameters = list(getattr(function, "parameters", None) or [])
            candidates = {
                p.name: i
                for i, p in enumerate(parameters)
                if _type_name(getattr(p, "param_type", None)) in COMBINED_SAMPLER_TYPES
            }
            local = {p.name for p in parameters}
            sampled = _sampled_in(program, [function], set(candidates), result)
            indices = sorted(
                candidates[name] for name in sampled if f"{name}Sampler" not in local
            )
            key = (function.name, len(parameters))
            if indices and indices != result.get(key):
                result[key] = indices
                changed = True
    return result


def _sampler_argument(argument):
    if isinstance(argument, ArrayAccessNode):
        inner = _sampler_argument(argument.array_expr)
        return (
            None
            if inner is None
            else ArrayAccessNode(inner, deepcopy(argument.index_expr))
        )
    if isinstance(argument, IdentifierNode):
        return IdentifierNode(f"{argument.name}Sampler")
    return None


def _thread_helper_samplers(program):
    helpers = _helper_sampler_parameters(program)
    functions = _functions(program)
    for function in functions:
        for node in _walk(getattr(function, "body", None), set()):
            if not isinstance(node, FunctionCallNode):
                continue
            arguments = list(getattr(node, "arguments", None) or [])
            indices = helpers.get((_callee(node), len(arguments)))
            if not indices:
                continue
            for index in reversed(indices):
                sampler = _sampler_argument(arguments[index])
                if sampler is None:
                    raise ValueError(
                        f"Metal call to '{_callee(node)}' passes a combined sampler that names no texture, so "
                        "no sampler can be passed with it"
                    )
                arguments.insert(index + 1, sampler)
            node.arguments = arguments
            node.args = arguments
    for function in functions:
        indices = helpers.get((function.name, len(function.parameters or [])))
        if not indices:
            continue
        parameters = list(function.parameters)
        for index in reversed(indices):
            texture = parameters[index]
            sampler_type = PrimitiveType("sampler")
            if isinstance(texture.param_type, ArrayType):
                sampler_type = ArrayType(
                    sampler_type, deepcopy(texture.param_type.size)
                )
            parameters.insert(
                index + 1, ParameterNode(f"{texture.name}Sampler", sampler_type)
            )
        function.parameters = parameters
