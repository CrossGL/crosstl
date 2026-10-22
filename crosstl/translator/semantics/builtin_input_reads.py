"""Built-in inputs read by name, passed to the HLSL and MSL entry point and on to the helpers that read them.

GLSL reads a shader's built-in inputs as global variables, anywhere in the stage, helpers included, and the CrossGL
import keeps those reads. They are a fragment stage's ``gl_FragCoord``, ``gl_FrontFacing``, ``gl_PointCoord``,
``gl_SampleID`` and ``gl_PrimitiveID`` (GLSL 4.60 section 7.1.5), and a vertex stage's ``gl_VertexID`` and
``gl_InstanceID``, or Vulkan GLSL's ``gl_VertexIndex`` and ``gl_InstanceIndex`` (GL_KHR_vulkan_glsl). HLSL and MSL
have no such globals. A built-in input is a parameter of the entry point carrying a system-value semantic
(``SV_IsFrontFace``, ``SV_InstanceID``) or an attribute (``[[front_facing]]``, ``[[instance_id]]``), and a helper
sees it only as an argument. The generators bound only ``gl_FragCoord`` and ``gl_VertexIndex``. Metal bound both
only in the entry point, and HLSL bound ``gl_VertexIndex`` only there. Every other read was emitted verbatim, as an
undeclared identifier.

:func:`pass_builtin_inputs` rewrites each vertex and fragment stage for those targets, in place:

* the entry point takes each built-in input that it or a helper reads as a parameter carrying the built-in
  (``bool cgl_FrontFacing @ gl_FrontFacing``), or reuses a parameter or input-structure member that already
  carries it. HLSL and MSL deliver the sample, primitive, vertex and instance indices as ``uint``, so these arrive
  in a ``uint`` parameter and are converted to GLSL's ``int`` once at the start of the entry point;
* each helper that reads one, directly or through the helpers it calls, takes it as a trailing parameter, and every
  call passes it on.

The built-in attributes are then spelled for the target by :mod:`.builtin_values`, which also refuses the ones the
target lacks (HLSL has no point-sprite coordinate). The HLSL generator already threads ``gl_FragCoord`` through
helpers itself, so for HLSL that one is left to it. A built-in that a generator binds only in the entry point
(``gl_VertexIndex``, and ``gl_FragCoord`` for Metal) is passed here as soon as a helper reads it.

MSL's ``[[vertex_id]]`` and ``[[instance_id]]`` include the draw's base vertex and base instance, as Vulkan's
indices do. HLSL's ``SV_VertexID`` and ``SV_InstanceID`` count from the draw's start, so the HLSL generator reads
Vulkan's indices as if the draw started at vertex and instance 0. DXC makes the same assumption for them unless told
otherwise (``-fvk-support-nonzero-base-vertex`` and ``-fvk-support-nonzero-base-instance``).
"""

from __future__ import annotations

from .. import ast as A
from .builtin_values import glsl_builtin_name
from .hub_types import call_name
from .rewrite import child_slots, get_child, set_child

__all__ = ["pass_builtin_inputs"]

_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
TARGETS = _HLSL_TARGETS | {"metal"}
_INPUTS = {
    "fragment": {
        "gl_FragCoord": "vec4",
        "gl_FrontFacing": "bool",
        "gl_PointCoord": "vec2",
        "gl_SampleID": "int",
        "gl_PrimitiveID": "int",
    },
    "vertex": {
        "gl_VertexID": "int",
        "gl_VertexIndex": "int",
        "gl_InstanceID": "int",
        "gl_InstanceIndex": "int",
    },
}
# HLSL SV_SampleIndex, SV_PrimitiveID, SV_VertexID and SV_InstanceID, and the MSL attributes for them, are uint.
_UNSIGNED = frozenset(
    {
        "gl_SampleID",
        "gl_PrimitiveID",
        "gl_VertexID",
        "gl_VertexIndex",
        "gl_InstanceID",
        "gl_InstanceIndex",
    }
)
# Built-ins a generator binds itself: everywhere, or only when the entry point alone reads them.
_BOUND_EVERYWHERE = {"hlsl": frozenset({"gl_FragCoord"}), "metal": frozenset()}
_BOUND_IN_ENTRY = {
    "hlsl": frozenset({"gl_VertexIndex"}),
    "metal": frozenset({"gl_FragCoord", "gl_VertexIndex"}),
}


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _stage_name(stage):
    value = getattr(stage, "stage", None)
    name = str(getattr(value, "value", value) or "").lower()
    return "fragment" if name == "pixel" else name


def _walk(node, seen=None):
    seen = set() if seen is None else seen
    if not isinstance(node, A.ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key), seen)


def _reads(function, names):
    return {
        node.name
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.IdentifierNode) and node.name in names
    }


def _declared(function):
    names = {
        getattr(parameter, "name", None) for parameter in function.parameters or []
    }
    for node in _walk(getattr(function, "body", None)):
        if isinstance(node, A.VariableNode):
            names.add(node.name)
    return names


def _program_names(program):
    names = set()
    for variable in getattr(program, "global_variables", None) or []:
        names.add(variable.name)
    for stage in _stages(program):
        for variable in getattr(stage, "local_variables", None) or []:
            names.add(variable.name)
    return names


def _all_names(program):
    names = _program_names(program)
    for struct in getattr(program, "structs", None) or []:
        names.add(struct.name)
    for stage in _stages(program):
        functions = list(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            functions.append(stage.entry_point)
        for function in functions:
            names.add(function.name)
            names |= _declared(function)
    for function in getattr(program, "functions", None) or []:
        names.add(function.name)
        names |= _declared(function)
    return names


def _builtin_of(node):
    for attribute in getattr(node, "attributes", None) or []:
        name = glsl_builtin_name(attribute)
        if name is not None:
            return name
    return None


def _existing_source(program, entry, builtin):
    """An expression for ``builtin`` the entry point already receives, or ``None``."""
    structs = {
        struct.name: struct for struct in getattr(program, "structs", None) or []
    }
    for parameter in entry.parameters or []:
        if _builtin_of(parameter) == builtin:
            return A.IdentifierNode(parameter.name)
        struct = structs.get(getattr(parameter.param_type, "name", None))
        for member in getattr(struct, "members", None) or []:
            if _builtin_of(member) == builtin:
                return A.MemberAccessNode(A.IdentifierNode(parameter.name), member.name)
    return None


def _fresh(preferred, taken):
    name, index = preferred, 0
    while name in taken:
        index += 1
        name = f"{preferred}_{index}"
    taken.add(name)
    return name


def _replace_reads(function, replacements):
    """Replace each read of a name in ``replacements`` with a fresh copy of its expression."""

    def visit(node):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if isinstance(child, A.IdentifierNode) and child.name in replacements:
                set_child(node, field, key, _copy(replacements[child.name]))
            elif isinstance(child, A.ASTNode):
                visit(child)

    if getattr(function, "body", None) is not None:
        visit(function.body)


def _copy(expression):
    if isinstance(expression, A.MemberAccessNode):
        return A.MemberAccessNode(_copy(expression.object_expr), expression.member)
    return A.IdentifierNode(expression.name)


def _passes(target, builtin, helper_reads):
    family = "hlsl" if target in _HLSL_TARGETS else "metal"
    if builtin in _BOUND_EVERYWHERE[family]:
        return False
    return builtin in helper_reads or builtin not in _BOUND_IN_ENTRY[family]


def pass_builtin_inputs(program, target) -> int:
    """Pass the fragment built-in inputs ``program`` reads by name as parameters; returns the stages changed."""
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    program_names = _program_names(program)
    changed = 0
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        inputs = _INPUTS.get(_stage_name(stage))
        if inputs is None or not isinstance(getattr(entry, "body", None), A.BlockNode):
            continue
        helpers = [
            function
            for function in list(getattr(program, "functions", None) or []) + list(
                getattr(stage, "local_functions", None) or []
            )
            if function is not entry and getattr(function, "body", None) is not None
        ]
        candidates = set(inputs) - program_names
        direct = {
            function.name: _reads(function, candidates - _declared(function))
            for function in [entry, *helpers]
        }
        helper_reads = (
            set().union(*(direct[helper.name] for helper in helpers))
            if helpers
            else set()
        )
        builtins = sorted(
            builtin
            for builtin in set().union(*direct.values())
            if _passes(target, builtin, helper_reads)
        )
        if not builtins:
            continue
        _thread(program, entry, helpers, direct, builtins, inputs, _all_names(program))
        changed += 1
    return changed


def _thread(program, entry, helpers, direct, builtins, inputs, taken):
    sources, conversions = {}, []
    for builtin in builtins:
        existing = _existing_source(program, entry, builtin)
        if existing is not None:
            sources[builtin] = existing
            continue
        name = _fresh(f"cgl_{builtin[3:]}", taken)
        type_node = A.PrimitiveType(inputs[builtin])
        attribute = [A.AttributeNode(builtin)]
        if builtin in _UNSIGNED:
            raw = _fresh(f"{name}_bits", taken)
            entry.parameters = list(entry.parameters or []) + [
                A.ParameterNode(raw, A.PrimitiveType("uint"), attributes=attribute)
            ]
            conversions.append(
                A.VariableNode(
                    name=name,
                    var_type=type_node,
                    initial_value=A.FunctionCallNode(
                        A.IdentifierNode("int"), [A.IdentifierNode(raw)]
                    ),
                )
            )
        else:
            entry.parameters = list(entry.parameters or []) + [
                A.ParameterNode(name, type_node, attributes=attribute)
            ]
        sources[builtin] = A.IdentifierNode(name)
    entry.body.statements[0:0] = conversions

    # The built-ins each helper needs: the ones it reads, and the ones the helpers it calls need.
    calls = {
        function.name: {
            call_name(node)
            for node in _walk(function.body)
            if isinstance(node, A.FunctionCallNode)
        }
        for function in [entry, *helpers]
    }
    needs = {
        helper.name: set(direct[helper.name]) & set(builtins) for helper in helpers
    }
    changed = True
    while changed:
        changed = False
        for helper in helpers:
            inherited = set()
            for callee in calls[helper.name]:
                inherited |= needs.get(callee, set())
            if not inherited <= needs[helper.name]:
                needs[helper.name] |= inherited
                changed = True

    parameter_names = {}
    for helper in helpers:
        names = {}
        for builtin in builtins:
            if builtin in needs[helper.name]:
                names[builtin] = _fresh(f"cgl_{builtin[3:]}", taken)
                helper.parameters = list(helper.parameters or []) + [
                    A.ParameterNode(names[builtin], A.PrimitiveType(inputs[builtin]))
                ]
        parameter_names[helper.name] = names

    for function in [entry, *helpers]:
        local = (
            sources
            if function is entry
            else {
                builtin: A.IdentifierNode(name)
                for builtin, name in parameter_names[function.name].items()
            }
        )
        for call in [
            node
            for node in _walk(function.body)
            if isinstance(node, A.FunctionCallNode)
        ]:
            needed = needs.get(call_name(call))
            if needed:
                arguments = list(call.arguments or []) + [
                    _copy(local[builtin]) for builtin in builtins if builtin in needed
                ]
                # ``args`` aliases ``arguments`` on call nodes.
                call.arguments = call.args = arguments
        _replace_reads(function, local)
