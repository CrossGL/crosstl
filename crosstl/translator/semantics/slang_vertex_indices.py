"""Vulkan's vertex and instance indices on Slang's Direct3D system values.

GL_KHR_vulkan_glsl replaced OpenGL's ``gl_InstanceID`` with ``gl_InstanceIndex``, which counts from the draw's first
instance: ``gl_InstanceIndex`` is ``gl_InstanceID + gl_BaseInstance`` (GL_KHR_vulkan_glsl, "gl_InstanceIndex"), and
``gl_VertexIndex`` is the vertex index OpenGL's ``gl_VertexID`` holds (OpenGL 4.6 core profile, section 11.1.3.9).
Slang has no system value for either name. Its ``SV_InstanceID`` is Direct3D's and counts from zero: slangc lowers it
to SPIR-V ``InstanceIndex - BaseInstance`` and to MSL ``instance_id - base_instance``. The Slang generator copied a
read of ``gl_InstanceIndex`` or ``gl_VertexIndex`` verbatim, which slangc rejects as an undeclared identifier, and
refused an entry-point parameter carrying either.

:func:`spell_slang_vertex_indices` rewrites the vertex stages of a program bound for Slang, in place, into the OpenGL
built-ins the Slang generator binds (``gl_InstanceID`` to ``SV_InstanceID``, ``gl_BaseInstance`` to
``SV_StartInstanceLocation``, ``gl_VertexID`` to ``SV_VertexID``):

- a read of ``gl_InstanceIndex`` becomes ``int(gl_InstanceID + gl_BaseInstance)``, and a read of ``gl_VertexIndex``
  becomes ``gl_VertexID``; :mod:`.builtin_input_reads` then passes them to the helpers that read them;
- an entry-point parameter carrying ``@gl_InstanceIndex`` is replaced by parameters carrying ``@gl_InstanceID`` and
  ``@gl_BaseInstance`` (an existing one is reused), and the body starts by declaring it as their sum; a parameter
  carrying ``@gl_VertexIndex`` carries ``@gl_VertexID`` instead;
- a stage-level input carrying ``@gl_InstanceIndex`` (the SPIR-V importer's form) is replaced by inputs carrying
  ``@gl_InstanceID`` and ``@gl_BaseInstance``, and each read of it by their sum; one carrying ``@gl_VertexIndex``
  carries ``@gl_VertexID`` instead.

Both indices therefore reach Slang as every ``gl_VertexID`` and ``gl_InstanceID`` does.
"""

from __future__ import annotations

from .. import ast as A
from .builtin_values import glsl_builtin_name
from .rewrite import child_slots, get_child, set_child

__all__ = ["spell_slang_vertex_indices"]

_INSTANCE_INDEX = "gl_InstanceIndex"
_VERTEX_INDEX = "gl_VertexIndex"


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _is_vertex(stage):
    value = getattr(stage, "stage", None)
    return str(getattr(value, "value", value) or "").lower() == "vertex"


def _walk(node, seen=None):
    seen = set() if seen is None else seen
    if not isinstance(node, A.ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key), seen)


def _names(program):
    """Every name ``program`` declares or reads, so a generated parameter cannot shadow one."""
    names = set()
    for node in _walk(program):
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _declared(program, function):
    names = {
        variable.name for variable in getattr(program, "global_variables", None) or []
    }
    for stage in _stages(program):
        names.update(
            variable.name for variable in getattr(stage, "local_variables", None) or []
        )
    names.update(parameter.name for parameter in function.parameters or [])
    names.update(
        node.name
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.VariableNode)
    )
    return names


def _instance_index():
    return A.FunctionCallNode(
        A.IdentifierNode("int"),
        [
            A.BinaryOpNode(
                A.IdentifierNode("gl_InstanceID"),
                "+",
                A.IdentifierNode("gl_BaseInstance"),
            )
        ],
    )


def _replacement(name):
    if name == _INSTANCE_INDEX:
        return _instance_index()
    return A.IdentifierNode("gl_VertexID")


def _replace_reads(function, names):
    """Replace each read of a name in ``names`` in ``function``'s body; returns the number replaced."""
    count = 0

    def visit(node):
        nonlocal count
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if isinstance(child, A.IdentifierNode) and child.name in names:
                set_child(node, field, key, _replacement(child.name))
                count += 1
            elif isinstance(child, A.ASTNode):
                visit(child)

    if getattr(function, "body", None) is not None:
        visit(function.body)
    return count


def _builtin_of(parameter):
    for attribute in getattr(parameter, "attributes", None) or []:
        name = glsl_builtin_name(attribute)
        if name is not None:
            return attribute, name
    return None, None


def _fresh(preferred, taken):
    name, index = preferred, 0
    while name in taken:
        index += 1
        name = f"{preferred}_{index}"
    taken.add(name)
    return name


def _rewrite_parameters(entry, taken):
    """Spell the entry point's ``@gl_InstanceIndex`` and ``@gl_VertexIndex`` parameters; returns the number changed."""
    sources = {}
    instance_parameters = []
    changed = 0
    for parameter in entry.parameters or []:
        attribute, builtin = _builtin_of(parameter)
        if builtin == _VERTEX_INDEX:
            attribute.name = "gl_VertexID"
            changed += 1
        elif builtin == _INSTANCE_INDEX:
            # Any other type keeps the attribute, which the Slang generator refuses.
            if getattr(parameter.param_type, "name", None) in ("int", "uint"):
                instance_parameters.append(parameter)
        elif builtin in ("gl_InstanceID", "gl_BaseInstance"):
            sources.setdefault(builtin, parameter.name)
    if not instance_parameters:
        return changed
    changed += len(instance_parameters)
    dropped = {id(parameter) for parameter in instance_parameters}
    parameters = [p for p in entry.parameters if id(p) not in dropped]
    for builtin, stem in (
        ("gl_InstanceID", "cgl_InstanceID"),
        ("gl_BaseInstance", "cgl_BaseInstance"),
    ):
        if builtin not in sources:
            sources[builtin] = _fresh(stem, taken)
            parameters.append(
                A.ParameterNode(
                    sources[builtin],
                    A.PrimitiveType("uint"),
                    attributes=[A.AttributeNode(builtin)],
                )
            )
    entry.parameters = parameters
    prologue = []
    for parameter in instance_parameters:
        total = A.BinaryOpNode(
            A.IdentifierNode(sources["gl_InstanceID"]),
            "+",
            A.IdentifierNode(sources["gl_BaseInstance"]),
        )
        type_name = getattr(parameter.param_type, "name", None)
        value = (
            A.FunctionCallNode(A.IdentifierNode(type_name), [total])
            if isinstance(type_name, str)
            else total
        )
        prologue.append(
            A.VariableNode(
                name=parameter.name, var_type=parameter.param_type, initial_value=value
            )
        )
    entry.body.statements[0:0] = prologue
    return changed


def _rewrite_stage_variables(stage, functions, taken):
    """Spell the stage's ``@gl_InstanceIndex`` and ``@gl_VertexIndex`` input variables; returns the number changed.

    :func:`.stage_interface_structs.gather_stage_interface_variables` later turns the stage-level built-in inputs
    into entry-point parameters, so an instance index becomes a ``gl_InstanceID`` and a ``gl_BaseInstance`` input
    whose sum replaces each read of it.
    """
    variables = list(getattr(stage, "local_variables", None) or [])
    changed = 0
    base = None
    for variable in variables:
        attribute, builtin = _builtin_of(variable)
        if builtin == _VERTEX_INDEX:
            attribute.name = "gl_VertexID"
            changed += 1
        elif builtin == "gl_BaseInstance" and base is None:
            base = variable.name
    for variable in variables:
        attribute, builtin = _builtin_of(variable)
        if builtin != _INSTANCE_INDEX:
            continue
        var_type = getattr(variable, "var_type", None)
        if getattr(var_type, "name", None) not in ("int", "uint"):
            continue
        if base is None:
            base = _fresh("base_instance", taken)
            stage.local_variables.append(
                A.VariableNode(
                    base,
                    A.PrimitiveType("uint"),
                    attributes=[
                        A.AttributeNode("input"),
                        A.AttributeNode("gl_BaseInstance"),
                    ],
                )
            )
        name = variable.name
        variable.name = _fresh("instance_id", taken)
        attribute.name = "gl_InstanceID"
        changed += 1

        def total(index=variable.name, type_name=var_type.name, base=base):
            converted = A.FunctionCallNode(
                A.IdentifierNode(type_name), [A.IdentifierNode(base)]
            )
            return A.FunctionCallNode(
                A.IdentifierNode(type_name),
                [A.BinaryOpNode(A.IdentifierNode(index), "+", converted)],
            )

        for function in functions:
            if name in _declared_locals(function):
                continue
            _replace_name(function, name, total)
    return changed


def _declared_locals(function):
    names = {parameter.name for parameter in function.parameters or []}
    names.update(
        node.name
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.VariableNode)
    )
    return names


def _replace_name(function, name, replacement):
    def visit(node):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if isinstance(child, A.IdentifierNode) and child.name == name:
                set_child(node, field, key, replacement())
            elif isinstance(child, A.ASTNode):
                visit(child)

    if getattr(function, "body", None) is not None:
        visit(function.body)


def spell_slang_vertex_indices(program, target) -> int:
    """Spell ``gl_InstanceIndex`` and ``gl_VertexIndex`` through OpenGL's built-ins for Slang; returns the rewrites."""
    if str(target or "").lower() != "slang" or not isinstance(program, A.ShaderNode):
        return 0
    taken = None
    changed = 0
    vertex_stages = [stage for stage in _stages(program) if _is_vertex(stage)]
    if not vertex_stages:
        return 0
    functions = list(getattr(program, "functions", None) or [])
    for stage in vertex_stages:
        entry = getattr(stage, "entry_point", None)
        stage_functions = list(getattr(stage, "local_functions", None) or [])
        if entry is not None:
            stage_functions.append(entry)
        if getattr(stage, "local_variables", None):
            if taken is None:
                taken = _names(program)
            changed += _rewrite_stage_variables(
                stage, functions + stage_functions, taken
            )
        for function in functions + stage_functions:
            names = {_INSTANCE_INDEX, _VERTEX_INDEX} - _declared(program, function)
            if names:
                changed += _replace_reads(function, names)
        if entry is not None and isinstance(getattr(entry, "body", None), A.BlockNode):
            if taken is None:
                taken = _names(program)
            changed += _rewrite_parameters(entry, taken)
    return changed
