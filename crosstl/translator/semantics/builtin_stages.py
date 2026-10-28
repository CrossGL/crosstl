"""``gl_`` built-in values are refused in a stage or direction GLSL does not declare them in.

GLSL declares each built-in variable for particular stages, as an input or as an output (GLSL 4.60 section 7.1,
"Built-In Language Variables", and the extensions listed in :mod:`.builtin_names`). Each generator mapped a built-in
it recognised for the stage and treated any other ``gl_`` attribute as a user varying: a vertex output
``@gl_SubgroupSize`` became an interface location on SPIR-V, the user semantic ``gl_SubgroupSize`` on HLSL and an
``out`` variable on GLSL, and a fragment input ``@gl_LaunchIDEXT`` read a varying no stage writes.

:func:`refuse_misplaced_builtins` checks the ``gl_`` attributes of every entry point's return value and parameters,
of the members of the structures they take or return (nested structures and arrays included) and of the stage
variables declared ``in``/``out`` (or ``@input``/``@output``). A built-in used in a stage, or in a direction, that
GLSL does not declare it for is refused with :class:`~.builtin_values.BuiltinValueUnsupportedError`. CrossGL's
conventions for values one stage writes and the next reads widen GLSL's tables:

* a vertex input ``@gl_Position`` is the vertex attribute HLSL spells ``SV_Position``, and a fragment input
  ``@gl_Position`` is the fragment's window position, as HLSL's pixel input ``SV_Position`` (the DirectX importer
  writes ``@gl_Position`` for both);
* a fragment input ``@gl_PointSize`` is a member of a vertex output structure the pixel stage also takes, which no
  generator reads;
* a fragment input ``@gl_FragColor``/``@gl_FragData`` reads a colour attachment (:mod:`.colour_attachment_inputs`),
  and a fragment input ``@gl_SampleMask`` is the coverage mask, as HLSL's ``SV_Coverage`` input;
* a value used both as an input and as an output of one stage (an ``inout`` parameter, or a structure the entry
  point takes and returns) needs only one of the two directions, because the generators split it.

The names themselves are checked by :func:`~.builtin_names.resolve_builtin_names`, which runs first. Stages
other than a ``ShaderStage`` this module names are not checked.
"""

from __future__ import annotations

from enum import Enum

from .. import ast as A
from .builtin_values import (
    _BOTH,
    _INPUT,
    _OUTPUT,
    BuiltinValueUnsupportedError,
    _parameter_direction,
    _stages,
    _type_name,
    _variable_direction,
)
from .hub_types import iter_structs

__all__ = ["BUILTIN_STAGE_RULES", "builtin_stage_rule", "refuse_misplaced_builtins"]

_STAGE_CODES = {
    "V": ("vertex",),
    "TC": ("tessellation_control",),
    "TE": ("tessellation_evaluation",),
    "G": ("geometry",),
    "F": ("fragment",),
    "C": ("compute",),
    "TASK": ("task", "amplification", "object"),
    "MESH": ("mesh",),
    "RG": ("ray_generation",),
    "IS": ("ray_intersection",),
    "AH": ("ray_any_hit",),
    "CH": ("ray_closest_hit",),
    "MS": ("ray_miss",),
    "CA": ("ray_callable",),
}
_ALL = " ".join(_STAGE_CODES)

# (built-in names, stages reading them as inputs, stages writing them as outputs).
_RULE_TEXT = (
    # GLSL 4.60 section 7.1.1, vertex inputs; GL_ARB_shader_draw_parameters; GL_EXT_mesh_shader for gl_DrawID.
    (
        "gl_VertexID gl_VertexIndex gl_InstanceIndex gl_BaseVertex gl_BaseInstance gl_BaseVertexARB "
        "gl_BaseInstanceARB",
        "V",
        "",
    ),
    ("gl_InstanceID", "V IS AH CH", ""),
    ("gl_DrawID gl_DrawIDARB", "V TASK MESH", ""),
    # Per-vertex values (gl_PerVertex), sections 7.1.1 to 7.1.4 and 7.1.5 for the fragment's clip distances;
    # GL_NV_viewport_array2, GL_NV_stereo_view_rendering and GL_NVX_multiview_per_view_attributes. The vertex and
    # fragment inputs of gl_Position and the fragment input of gl_PointSize are CrossGL conventions (see above).
    ("gl_Position", "V TC TE G F", "V TC TE G MESH"),
    ("gl_ClipDistance gl_CullDistance gl_PointSize", "TC TE G F", "V TC TE G MESH"),
    (
        "gl_PositionPerViewNV gl_ClipDistancePerViewNV gl_CullDistancePerViewNV "
        "gl_SecondaryPositionNV",
        "TC TE G",
        "V TC TE G MESH",
    ),
    (
        "gl_LayerPerViewNV gl_ViewportMaskPerViewNV gl_ViewportMaskNV gl_SecondaryViewportMaskNV",
        "",
        "V TE G MESH",
    ),
    # Tessellation and geometry, sections 7.1.2 to 7.1.4; ray-tracing inputs from GL_EXT_ray_tracing.
    ("gl_PatchVerticesIn", "TC TE", ""),
    ("gl_PrimitiveID", "TC TE G F IS AH CH", "G MESH"),
    ("gl_PrimitiveIDIn", "G", ""),
    ("gl_InvocationID", "TC G", ""),
    ("gl_TessLevelOuter gl_TessLevelInner", "TE", "TC"),
    ("gl_TessCoord", "TE", ""),
    # GL_ARB_shader_viewport_layer_array lets the vertex and tessellation evaluation stages write these too.
    ("gl_Layer gl_ViewportIndex", "F", "V TE G MESH"),
    # Fragment inputs, section 7.1.5, and the fragment-only extensions.
    (
        "gl_FragCoord gl_FrontFacing gl_IsFrontFace gl_PointCoord gl_SampleID gl_SamplePosition "
        "gl_SampleMaskIn gl_HelperInvocation gl_BaryCoordEXT gl_BaryCoordNoPerspEXT gl_BaryCoordNV "
        "gl_BaryCoordNoPerspNV gl_BaryCoordNoPerspAMD gl_BaryCoordNoPerspCentroidAMD "
        "gl_BaryCoordNoPerspSampleAMD gl_BaryCoordSmoothAMD gl_BaryCoordSmoothCentroidAMD "
        "gl_BaryCoordSmoothSampleAMD gl_BaryCoordPullModelAMD gl_ShadingRateEXT gl_FragSizeEXT "
        "gl_FragInvocationCountEXT gl_FullyCoveredEXT",
        "F",
        "",
    ),
    ("gl_SampleMask", "F", "F"),
    # A fragment input colour reads an attachment (CrossGL convention, see above).
    ("gl_FragColor gl_FragData", "F", "F"),
    ("gl_FragDepth gl_FragStencilRefARB gl_FragStencilRefEXT", "", "F"),
    ("gl_PrimitiveShadingRateEXT", "", "V G MESH"),
    # Built-in uniform state, section 7.4, and the values every stage reads.
    (
        "gl_NumSamples gl_SubgroupSize gl_SubgroupInvocationID gl_SubgroupEqMask gl_SubgroupGeMask "
        "gl_SubgroupGtMask gl_SubgroupLeMask gl_SubgroupLtMask gl_DeviceIndex gl_WarpIDNV gl_SMIDNV "
        "gl_WarpsPerSMNV gl_SMCountNV",
        _ALL,
        "",
    ),
    ("gl_ViewIndex", "V TC TE G F TASK MESH", ""),
    # Compute inputs, section 7.1.6, which GL_EXT_mesh_shader also gives task and mesh shaders;
    # GL_KHR_shader_subgroup_basic. Metal's quad-group indices exist in kernels and fragment functions.
    (
        "gl_NumWorkGroups gl_WorkGroupSize gl_WorkGroupID gl_LocalInvocationID gl_GlobalInvocationID "
        "gl_LocalInvocationIndex gl_NumSubgroups gl_SubgroupID",
        "C TASK MESH",
        "",
    ),
    ("gl_QuadGroupID gl_QuadGroupInvocationID", "C F TASK MESH", ""),
    # GL_EXT_mesh_shader and GL_NV_mesh_shader.
    (
        "gl_PrimitivePointIndicesEXT gl_PrimitiveLineIndicesEXT gl_PrimitiveTriangleIndicesEXT "
        "gl_CullPrimitiveEXT gl_PrimitiveCountNV gl_PrimitiveIndicesNV",
        "",
        "MESH",
    ),
    ("gl_TaskCountNV", "", "TASK"),
    ("gl_MeshViewCountNV gl_MeshViewIndicesNV", "TASK MESH", ""),
    # GL_EXT_ray_tracing, GL_EXT_ray_cull_mask, GL_EXT_ray_tracing_position_fetch and GL_NV_ray_tracing.
    ("gl_LaunchIDEXT gl_LaunchSizeEXT gl_LaunchIDNV gl_LaunchSizeNV", "RG IS AH CH MS CA", ""),
    (
        "gl_InstanceCustomIndexEXT gl_GeometryIndexEXT gl_ObjectRayOriginEXT gl_ObjectRayDirectionEXT "
        "gl_ObjectToWorldEXT gl_WorldToObjectEXT gl_ObjectToWorld3x4EXT gl_WorldToObject3x4EXT",
        "IS AH CH",
        "",
    ),
    (
        "gl_WorldRayOriginEXT gl_WorldRayDirectionEXT gl_RayTminEXT gl_RayTmaxEXT gl_IncomingRayFlagsEXT "
        "gl_CullMaskEXT",
        "IS AH CH MS",
        "",
    ),
    ("gl_HitTEXT gl_HitKindEXT gl_HitTriangleVertexPositionsEXT", "AH CH", ""),
)


def _stage_names(codes):
    return frozenset(name for code in codes.split() for name in _STAGE_CODES[code])


def _build_rules():
    rules = {}
    for names, inputs, outputs in _RULE_TEXT:
        rule = {_INPUT: _stage_names(inputs), _OUTPUT: _stage_names(outputs)}
        for name in names.split():
            rules[name] = rule
    return rules


BUILTIN_STAGE_RULES = _build_rules()
_RULES_LOWERED = {name.lower(): rule for name, rule in BUILTIN_STAGE_RULES.items()}
_KNOWN_STAGES = frozenset(name for names in _STAGE_CODES.values() for name in names)
_INDEXED_BASES = ("gl_fragcolor", "gl_clipdistance", "gl_culldistance")


def builtin_stage_rule(name):
    """``{"input": stages, "output": stages}`` for the ``gl_`` built-in ``name`` in any letter case, or ``None``."""
    lowered = str(name or "").lower()
    if lowered.startswith("gl_fragdata["):
        lowered = "gl_fragdata"
    for base in _INDEXED_BASES:
        if lowered.startswith(base) and lowered[len(base) :].isdigit():
            lowered = base
    return _RULES_LOWERED.get(lowered)


def refuse_misplaced_builtins(program, target) -> int:
    """Refuse ``gl_`` built-ins used outside their stages or directions; returns how many were checked.

    Raises :class:`~.builtin_values.BuiltinValueUnsupportedError` for the first misplaced built-in.
    """
    if not isinstance(program, A.ShaderNode):
        return 0
    checker = _Checker(str(target or ""), {s.name: s for s in iter_structs(program)})
    for stage in _stages(program):
        stage_name = _stage_name(getattr(stage, "stage", None))
        if stage_name not in _KNOWN_STAGES:
            continue
        uses = {}
        function = getattr(stage, "entry_point", None)
        if function is not None:
            name = getattr(function, "name", "main")
            checker.check(function, stage_name, _OUTPUT, f"the return value of '{name}'")
            checker.collect(getattr(function, "return_type", None), _OUTPUT, uses)
            for parameter in getattr(function, "parameters", None) or []:
                direction = _parameter_direction(parameter)
                checker.check(
                    parameter,
                    stage_name,
                    direction,
                    f"parameter '{getattr(parameter, 'name', '')}' of '{name}'",
                )
                checker.collect(getattr(parameter, "param_type", None), direction, uses)
        for variable in getattr(stage, "local_variables", None) or []:
            direction = _variable_direction(variable)
            if direction is None:
                continue
            checker.check(
                variable, stage_name, direction, f"variable '{getattr(variable, 'name', '')}'"
            )
            checker.collect(getattr(variable, "var_type", None), direction, uses)
        for struct_name in sorted(uses):
            struct = checker.structs[struct_name]
            direction = _BOTH if len(uses[struct_name]) > 1 else next(iter(uses[struct_name]))
            for member in getattr(struct, "members", None) or []:
                checker.check(
                    member,
                    stage_name,
                    direction,
                    f"member '{getattr(member, 'name', '')}' of '{struct_name}'",
                )
    return checker.checked


class _Checker:
    def __init__(self, target, structs):
        self.target = target
        self.structs = structs
        self.checked = 0

    def check(self, node, stage, direction, context):
        for attribute in getattr(node, "attributes", None) or []:
            name = str(getattr(attribute, "name", "") or "")
            rule = builtin_stage_rule(name) if name.lower().startswith("gl_") else None
            if rule is None:
                continue
            self.checked += 1
            directions = (_INPUT, _OUTPUT) if direction == _BOTH else (direction,)
            if any(stage in rule[d] for d in directions):
                continue
            raise BuiltinValueUnsupportedError(
                f"@{name} on {context} is not a {stage} {_describe(directions)} in GLSL; "
                f"{_where(name, rule)} (GLSL 4.60 section 7.1)",
                self.target,
            )

    def collect(self, type_node, direction, uses):
        """Note ``direction`` for the structure ``type_node`` names and the structures nested in it."""
        directions = (_INPUT, _OUTPUT) if direction == _BOTH else (direction,)
        while isinstance(type_node, A.ArrayType):
            type_node = type_node.element_type
        name = _type_name(type_node)
        struct = self.structs.get(name)
        if struct is None or uses.get(name, set()).issuperset(directions):
            return
        uses.setdefault(name, set()).update(directions)
        for member in getattr(struct, "members", None) or []:
            self.collect(getattr(member, "member_type", None), direction, uses)


def _stage_name(stage):
    value = stage.value if isinstance(stage, Enum) else stage
    return str(value or "").lower()


def _describe(directions):
    return " or ".join("input" if d == _INPUT else "output" for d in directions)


def _where(name, rule):
    parts = []
    for direction, label in ((_INPUT, "an input"), (_OUTPUT, "an output")):
        stages = sorted(rule[direction])
        if stages:
            parts.append(f"{label} of {', '.join(stages)}")
    return f"GLSL declares {name} as " + " and ".join(parts)
