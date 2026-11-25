"""GLSL-style ray tracing interfaces in Slang output, rewritten into the DXR form Slang declares.

The GLSL and SPIR-V importers write a ray tracing stage's payload, hit attributes and callable data as module-scope
variables marked ``@rayPayloadEXT``, ``@rayPayloadInEXT``, ``@hitAttributeEXT``, ``@callableDataEXT`` and
``@callableDataInEXT``, and ``ReportHit(t, kind)`` without the attributes (GL_EXT_ray_tracing,
:mod:`crosstl.translator.semantics.hlsl_ray_interfaces`). Slang takes the HLSL forms: a hit stage receives the payload
and the attributes as entry parameters, ``TraceRay`` takes a ``RayDesc`` and the payload, ``CallShader`` the callable
data and ``ReportHit`` the attributes. The Slang generator copied the GLSL form, so slangc rejected a write to the
interface variable (a non-``static`` global is a uniform, error E30011) and ``ReportHit`` without its attribute
argument (error E30439).

:func:`lower_slang_glsl_ray_interfaces` runs the HLSL generator's rewrite,
:func:`~crosstl.translator.semantics.hlsl_ray_interfaces.lower_glsl_ray_interfaces`, on a program bound for Slang
that has an interface the Slang generator cannot write: a module-scope ray interface variable, or a stage-scope hit
attribute variable of an intersection stage. What that rewrite refuses (the object and world transform matrices,
``gl_CullMaskEXT``, interface variables shared by several stages) is refused for Slang with the same reason. Any other
program keeps the Slang generator's own forms: it writes GLSL-spelled ray built-ins as Slang intrinsics and a
stage-scope interface variable as an entry parameter, and :func:`report_placeholder_attributes` gives
``ReportHit(t, kind)`` the placeholder attributes the rewrite would.
"""

from __future__ import annotations

from .. import ast as A
from ..refusals import SlangRefusal
from ..semantics.hlsl_ray_interfaces import (
    RayInterfaceUnsupportedError,
    lower_glsl_ray_interfaces,
)
from ..semantics.rewrite import child_slots, get_child

__all__ = ["lower_slang_glsl_ray_interfaces"]

# The annotations GL_EXT_ray_tracing's interface storage classes import as.
_INTERFACE_ANNOTATIONS = frozenset(
    {
        "raypayloadext",
        "raypayloadinext",
        "hitattributeext",
        "callabledataext",
        "callabledatainext",
        "raypayloadnv",
        "raypayloadinnv",
        "hitattributenv",
        "callabledatanv",
        "callabledatainnv",
    }
)


def _is_interface_variable(variable):
    return any(
        str(getattr(attribute, "name", "")).lower() in _INTERFACE_ANNOTATIONS
        for attribute in getattr(variable, "attributes", None) or []
    )


def _walk(node):
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key))


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _functions(program):
    yield from getattr(program, "functions", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_functions", None) or []
        if getattr(stage, "entry_point", None) is not None:
            yield stage.entry_point


def needs_dxr_rewrite(program):
    """Whether ``program`` has a ray tracing interface the Slang generator writes only after the DXR rewrite.

    That is a module-scope interface variable (the GLSL and SPIR-V importers' form) or a stage-scope hit attribute
    variable of an intersection stage, which ``ReportHit(t, kind)`` reports.
    """
    if any(
        _is_interface_variable(variable)
        for variable in getattr(program, "global_variables", None) or []
    ):
        return True
    return any(
        getattr(stage, "stage", None) == A.ShaderStage.RAY_INTERSECTION
        and any(
            _is_interface_variable(variable)
            for variable in getattr(stage, "local_variables", None) or []
        )
        for stage in _stages(program)
    )


def _taken_names(program):
    taken = {struct.name for struct in getattr(program, "structs", None) or []}
    taken.update(variable.name for variable in program.global_variables or [])
    for function in _functions(program):
        taken.add(function.name)
        taken.update(parameter.name for parameter in function.parameters or [])
        taken.update(
            node.name for node in _walk(function) if isinstance(node, A.VariableNode)
        )
    for stage in _stages(program):
        taken.update(
            variable.name for variable in getattr(stage, "local_variables", None) or []
        )
    return taken


def _unused(taken, base):
    name, suffix = base, 1
    while name in taken:
        name, suffix = f"{base}_{suffix}", suffix + 1
    taken.add(name)
    return name


def _declared_hit_attribute_types(program, generator):
    """The structure types the hit stages of ``program`` receive their hit attributes as."""
    struct_names = {struct.name for struct in getattr(program, "structs", None) or []}
    declared = set()
    for stage in _stages(program):
        stage_name = generator.get_stage_name(stage.stage)
        entry = getattr(stage, "entry_point", None)
        candidates = [
            *(getattr(entry, "parameters", None) or []),
            *(getattr(stage, "local_variables", None) or []),
        ]
        for candidate in candidates:
            if (
                generator.slang_ray_semantic_role(candidate, stage_name)
                != "hit_attribute"
            ):
                continue
            value_type = getattr(candidate, "param_type", None) or getattr(
                candidate, "var_type", None
            )
            if isinstance(value_type, A.NamedType) and value_type.name in struct_names:
                declared.add(value_type.name)
    return declared


def report_placeholder_attributes(program, generator):
    """Give each ``ReportHit(t, kind)`` of ``program`` placeholder attributes, in place.

    GL_EXT_ray_tracing's ``reportIntersectionEXT(t, kind)`` in a stage without a ``hitAttributeEXT`` variable reports
    attributes no stage may read, while Slang's ``ReportHit`` needs them (error E30439). The placeholder is a
    ``static`` value of the hit attribute type the program's hit stages declare, or, when they declare none, the
    4-byte structure the DXR rewrite reports.
    """
    calls = [
        node
        for function in _functions(program)
        for node in _walk(function)
        if isinstance(node, A.RayTracingOpNode)
        and node.operation == "ReportHit"
        and len(node.arguments) == 2
    ]
    if not calls:
        return
    taken = _taken_names(program)
    declared = _declared_hit_attribute_types(program, generator)
    if len(declared) == 1:
        struct_name = next(iter(declared))
    else:
        # With several declared types the generator refuses the mismatch for each ReportHit.
        struct_name = _unused(taken, "CrossGLHitAttributesUnused")
        program.structs.append(
            A.StructNode(
                struct_name, [A.StructMemberNode("unused", A.PrimitiveType("uint"))]
            )
        )
    attributes = A.VariableNode(
        _unused(taken, "crossgl_hit_attributes"),
        A.NamedType(struct_name),
        qualifiers=["static"],
    )
    program.global_variables.append(attributes)
    for call in calls:
        call.arguments.append(A.IdentifierNode(attributes.name))


def lower_slang_glsl_ray_interfaces(program, generator):
    """Rewrite the GLSL-style ray tracing interfaces of ``program`` into their Slang (DXR) form, in place.

    ``generator`` is the :class:`~crosstl.translator.codegen.slang_codegen.SlangCodeGen` that writes ``program``.
    """
    if not needs_dxr_rewrite(program):
        report_placeholder_attributes(program, generator)
        return program
    try:
        lower_glsl_ray_interfaces(program)
    except RayInterfaceUnsupportedError as error:
        detail = error.reason.removeprefix(f"DirectX {error.stage} stage: ")
        raise SlangRefusal(f"Slang {error.stage} stage: {detail}") from error
    return program
