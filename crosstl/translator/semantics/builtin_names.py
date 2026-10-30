"""``gl_`` built-in value names: a name CrossGL does not define is refused, an alias takes its canonical spelling.

GLSL reserves every identifier that starts with ``gl_`` (GLSL 4.60 section 3.7, "Identifiers"), and CrossGL spells
built-in values with those names (``@gl_FragCoord``). Each generator looks such an attribute up in its own tables and
treats a name it does not know as a user varying: SPIR-V gave ``@gl_FragBogus`` an interface location, HLSL emitted
it as the user semantic ``gl_FragBogus`` and GLSL declared an ``out`` variable, so a misspelt built-in became a varying
no stage reads. CrossGL's alias ``@gl_IsFrontFace`` (written by the Slang, Rust and Mojo importers) and GLSL's
``@gl_FragStencilRefARB`` (GL_ARB_shader_stencil_export) became user varyings the same way on SPIR-V.

:func:`resolve_builtin_names` checks the attributes of every function (its return value), parameter, structure
member and stage or module variable:

* a ``gl_`` name that is neither a GLSL built-in variable (GLSL 4.60 section 7.1, "Built-In Language Variables",
  and the extensions CrossTL reads: KHR subgroups, multiview, device groups, draw parameters, stencil export,
  barycentrics, fragment shading rate and density, fully covered fragments, EXT and NV mesh shading, NV per-view
  attributes, NV SM built-ins and EXT and NV ray tracing) nor a CrossGL extension of that namespace is refused with
  :class:`~.builtin_values.BuiltinValueUnsupportedError`. The CrossGL extensions are the indexed colour outputs
  ``gl_FragColorN`` and ``gl_FragData[N]``, the indexed distance members ``gl_ClipDistanceN`` and
  ``gl_CullDistanceN`` (HLSL ``SV_ClipDistanceN``), and Metal's quad-group indices ``gl_QuadGroupID`` and
  ``gl_QuadGroupInvocationID``. Names are matched without regard to case, as the generators match them;
* the aliases ``gl_IsFrontFace`` and ``gl_FragStencilRefARB`` are rewritten to ``gl_FrontFacing`` and
  ``gl_FragStencilRefEXT``, the spellings every generator reads.

Whether the target can express a known built-in is left to the generators and to
:func:`~.builtin_values.canonicalize_builtin_values`.
"""

from __future__ import annotations

import re

from .. import ast as A
from .builtin_values import BuiltinValueUnsupportedError
from .hub_types import iter_functions, iter_structs

__all__ = ["CROSSGL_GL_BUILTIN_NAMES", "is_crossgl_gl_builtin", "resolve_builtin_names"]

_GLSL_CORE = (
    # GLSL 4.60 section 7.1, including the compatibility profile's fragment outputs.
    "gl_VertexID gl_InstanceID gl_VertexIndex gl_InstanceIndex gl_DrawID gl_BaseVertex gl_BaseInstance "
    "gl_Position gl_PointSize gl_ClipDistance gl_CullDistance gl_PatchVerticesIn gl_PrimitiveID "
    "gl_PrimitiveIDIn gl_InvocationID gl_TessLevelOuter gl_TessLevelInner gl_TessCoord gl_Layer "
    "gl_ViewportIndex gl_FragCoord gl_FrontFacing gl_PointCoord gl_SampleID gl_SamplePosition "
    "gl_SampleMaskIn gl_SampleMask gl_HelperInvocation gl_FragDepth gl_NumSamples gl_NumWorkGroups "
    "gl_WorkGroupSize gl_WorkGroupID gl_LocalInvocationID gl_GlobalInvocationID gl_LocalInvocationIndex "
    "gl_FragColor gl_FragData"
)
_GLSL_EXTENSIONS = (
    # GL_KHR_shader_subgroup_basic and _ballot.
    "gl_NumSubgroups gl_SubgroupID gl_SubgroupSize gl_SubgroupInvocationID gl_SubgroupEqMask "
    "gl_SubgroupGeMask gl_SubgroupGtMask gl_SubgroupLeMask gl_SubgroupLtMask "
    # GL_EXT_multiview, GL_EXT_device_group and GL_ARB_shader_draw_parameters.
    "gl_ViewIndex gl_DeviceIndex gl_DrawIDARB gl_BaseVertexARB gl_BaseInstanceARB "
    # GL_ARB_shader_stencil_export, GL_EXT_fragment_shader_barycentric, GL_NV_fragment_shader_barycentric and
    # GL_AMD_shader_explicit_vertex_parameter.
    "gl_FragStencilRefARB gl_BaryCoordEXT gl_BaryCoordNoPerspEXT gl_BaryCoordNV gl_BaryCoordNoPerspNV "
    "gl_BaryCoordNoPerspAMD gl_BaryCoordNoPerspCentroidAMD gl_BaryCoordNoPerspSampleAMD "
    "gl_BaryCoordSmoothAMD gl_BaryCoordSmoothCentroidAMD gl_BaryCoordSmoothSampleAMD "
    "gl_BaryCoordPullModelAMD "
    # GL_EXT_fragment_shading_rate, GL_EXT_fragment_invocation_density and GL_NV_conservative_raster_underestimation
    # as exposed by SPV_EXT_fragment_fully_covered.
    "gl_PrimitiveShadingRateEXT gl_ShadingRateEXT gl_FragSizeEXT gl_FragInvocationCountEXT "
    "gl_FullyCoveredEXT "
    # GL_EXT_mesh_shader and GL_NV_mesh_shader.
    "gl_PrimitivePointIndicesEXT gl_PrimitiveLineIndicesEXT gl_PrimitiveTriangleIndicesEXT "
    "gl_CullPrimitiveEXT gl_PrimitiveCountNV gl_PrimitiveIndicesNV gl_TaskCountNV gl_MeshViewCountNV "
    "gl_MeshViewIndicesNV "
    # GL_NV_stereo_view_rendering, GL_NV_viewport_array2 and GL_NVX_multiview_per_view_attributes.
    "gl_PositionPerViewNV gl_ClipDistancePerViewNV gl_CullDistancePerViewNV gl_LayerPerViewNV "
    "gl_ViewportMaskPerViewNV gl_ViewportMaskNV gl_SecondaryPositionNV gl_SecondaryViewportMaskNV "
    # GL_NV_shader_sm_builtins.
    "gl_WarpIDNV gl_SMIDNV gl_WarpsPerSMNV gl_SMCountNV "
    # GL_EXT_ray_tracing, GL_EXT_ray_cull_mask, GL_EXT_ray_tracing_position_fetch and GL_NV_ray_tracing.
    "gl_LaunchIDEXT gl_LaunchSizeEXT gl_InstanceCustomIndexEXT gl_GeometryIndexEXT gl_WorldRayOriginEXT "
    "gl_WorldRayDirectionEXT gl_ObjectRayOriginEXT gl_ObjectRayDirectionEXT gl_RayTminEXT gl_RayTmaxEXT "
    "gl_IncomingRayFlagsEXT gl_HitTEXT gl_HitKindEXT gl_ObjectToWorldEXT gl_WorldToObjectEXT "
    "gl_ObjectToWorld3x4EXT gl_WorldToObject3x4EXT gl_CullMaskEXT gl_HitTriangleVertexPositionsEXT "
    "gl_LaunchIDNV gl_LaunchSizeNV"
)
_CROSSGL_EXTENSIONS = (
    "gl_FragStencilRefEXT gl_IsFrontFace gl_QuadGroupID gl_QuadGroupInvocationID"
)

CROSSGL_GL_BUILTIN_NAMES = frozenset(
    " ".join((_GLSL_CORE, _GLSL_EXTENSIONS, _CROSSGL_EXTENSIONS)).split()
)
_LOWERED = frozenset(name.lower() for name in CROSSGL_GL_BUILTIN_NAMES)
_INDEXED = re.compile(
    r"(?:gl_fragcolor|gl_clipdistance|gl_culldistance)\d+|gl_fragdata\[\d+\]"
)
_ALIASES = {
    "gl_isfrontface": "gl_FrontFacing",
    "gl_fragstencilrefarb": "gl_FragStencilRefEXT",
}


def is_crossgl_gl_builtin(name) -> bool:
    """Whether the ``gl_`` attribute name ``name`` is a CrossGL built-in value, in any letter case."""
    lowered = str(name or "").lower()
    return lowered in _LOWERED or _INDEXED.fullmatch(lowered) is not None


def resolve_builtin_names(program, target) -> int:
    """Refuse unknown ``gl_`` built-in names and rewrite aliases in place; returns how many were rewritten.

    Raises :class:`~.builtin_values.BuiltinValueUnsupportedError` for a ``gl_`` attribute CrossGL does not define.
    """
    if not isinstance(program, A.ShaderNode):
        return 0
    target = str(target or "")
    rewritten = 0
    for node, context in _interface_nodes(program):
        attributes = getattr(node, "attributes", None)
        if not attributes:
            continue
        for index, attribute in enumerate(list(attributes)):
            name = str(getattr(attribute, "name", "") or "")
            lowered = name.lower()
            if not lowered.startswith("gl_"):
                continue
            if not is_crossgl_gl_builtin(name):
                raise BuiltinValueUnsupportedError(
                    f"Unknown built-in value @{name} on {context}: GLSL reserves the gl_ prefix "
                    "(GLSL 4.60 section 3.7) and neither GLSL nor CrossGL defines this built-in",
                    target,
                )
            canonical = _ALIASES.get(lowered)
            if canonical is not None:
                attributes[index] = A.AttributeNode(
                    canonical, list(getattr(attribute, "arguments", None) or [])
                )
                rewritten += 1
    return rewritten


def _interface_nodes(program):
    for function in iter_functions(program):
        name = getattr(function, "name", "function")
        yield function, f"the return value of '{name}'"
        for parameter in getattr(function, "parameters", None) or []:
            yield parameter, f"parameter '{getattr(parameter, 'name', '')}' of '{name}'"
    for struct in iter_structs(program):
        struct_name = getattr(struct, "name", "")
        for member in getattr(struct, "members", None) or []:
            yield member, f"member '{getattr(member, 'name', '')}' of '{struct_name}'"
    yield from (
        (variable, f"variable '{getattr(variable, 'name', '')}'")
        for variable in _variables(program)
    )


def _variables(program):
    yield from getattr(program, "global_variables", None) or []
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        yield from getattr(stage, "local_variables", None) or []
