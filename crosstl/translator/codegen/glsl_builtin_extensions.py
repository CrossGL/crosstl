"""``#extension`` directives for the extension built-ins a GLSL program names.

Programs imported from SPIR-V, HLSL or Metal reach built-ins that GLSL declares only under an extension:
``gl_BaryCoordEXT`` (HLSL ``SV_Barycentrics``, SPIR-V ``BaryCoordKHR``) exists only once
``GL_EXT_fragment_shader_barycentric`` is enabled, and ``gl_ShadingRateEXT`` only under
``GL_EXT_fragment_shading_rate``. A compiler rejects the program when the directive is missing ("required extension
not requested"), so the emitter scans the rendered program and requires each extension one of its built-ins needs.
A directive the program already has, with any behaviour, is left as it stands.
"""

import re

__all__ = ["BUILTIN_EXTENSIONS", "require_builtin_extensions"]

_BARYCENTRIC_EXT = "GL_EXT_fragment_shader_barycentric"
_BARYCENTRIC_NV = "GL_NV_fragment_shader_barycentric"
_SHADING_RATE = "GL_EXT_fragment_shading_rate"
_DENSITY = "GL_EXT_fragment_invocation_density"
_DRAW_PARAMETERS = "GL_ARB_shader_draw_parameters"
_AMD_BARYCENTRIC = "GL_AMD_shader_explicit_vertex_parameter"
_SUBGROUP_BASIC = "GL_KHR_shader_subgroup_basic"
_SUBGROUP_BALLOT = "GL_KHR_shader_subgroup_ballot"
_SM_BUILTINS = "GL_NV_shader_sm_builtins"
_UNDERESTIMATION = "GL_NV_conservative_raster_underestimation"

# Each extension's "New Built-in Variables" and "New Built-in Constants" sections (the Khronos GLSL extension
# registry, GLSL_EXT_fragment_shader_barycentric.txt, GLSL_NV_fragment_shader_barycentric.txt,
# GLSL_EXT_fragment_shading_rate.txt and GLSL_EXT_fragment_invocation_density.txt).
BUILTIN_EXTENSIONS = {
    "gl_BaryCoordEXT": _BARYCENTRIC_EXT,
    "gl_BaryCoordNoPerspEXT": _BARYCENTRIC_EXT,
    "gl_BaryCoordNV": _BARYCENTRIC_NV,
    "gl_BaryCoordNoPerspNV": _BARYCENTRIC_NV,
    "gl_ShadingRateEXT": _SHADING_RATE,
    "gl_PrimitiveShadingRateEXT": _SHADING_RATE,
    "gl_ShadingRateFlag2VerticalPixelsEXT": _SHADING_RATE,
    "gl_ShadingRateFlag4VerticalPixelsEXT": _SHADING_RATE,
    "gl_ShadingRateFlag2HorizontalPixelsEXT": _SHADING_RATE,
    "gl_ShadingRateFlag4HorizontalPixelsEXT": _SHADING_RATE,
    "gl_FragSizeEXT": _DENSITY,
    "gl_FragInvocationCountEXT": _DENSITY,
    # GL_ARB_shader_draw_parameters, "New Built-in Variables"; GLSL 4.60 makes them core without the suffix.
    "gl_BaseVertexARB": _DRAW_PARAMETERS,
    "gl_BaseInstanceARB": _DRAW_PARAMETERS,
    "gl_DrawIDARB": _DRAW_PARAMETERS,
    # GL_AMD_shader_explicit_vertex_parameter, GL_KHR_shader_subgroup (whose ballot extension implicitly enables
    # the basic one), GL_NV_shader_sm_builtins and GL_NV_conservative_raster_underestimation.
    "gl_BaryCoordNoPerspAMD": _AMD_BARYCENTRIC,
    "gl_BaryCoordNoPerspCentroidAMD": _AMD_BARYCENTRIC,
    "gl_BaryCoordNoPerspSampleAMD": _AMD_BARYCENTRIC,
    "gl_BaryCoordSmoothAMD": _AMD_BARYCENTRIC,
    "gl_BaryCoordSmoothCentroidAMD": _AMD_BARYCENTRIC,
    "gl_BaryCoordSmoothSampleAMD": _AMD_BARYCENTRIC,
    "gl_BaryCoordPullModelAMD": _AMD_BARYCENTRIC,
    "gl_NumSubgroups": _SUBGROUP_BASIC,
    "gl_SubgroupID": _SUBGROUP_BASIC,
    "gl_SubgroupSize": _SUBGROUP_BASIC,
    "gl_SubgroupInvocationID": _SUBGROUP_BASIC,
    "gl_SubgroupEqMask": _SUBGROUP_BALLOT,
    "gl_SubgroupGeMask": _SUBGROUP_BALLOT,
    "gl_SubgroupGtMask": _SUBGROUP_BALLOT,
    "gl_SubgroupLeMask": _SUBGROUP_BALLOT,
    "gl_SubgroupLtMask": _SUBGROUP_BALLOT,
    "gl_WarpIDNV": _SM_BUILTINS,
    "gl_SMIDNV": _SM_BUILTINS,
    "gl_WarpsPerSMNV": _SM_BUILTINS,
    "gl_SMCountNV": _SM_BUILTINS,
    "gl_FragFullyCoveredNV": _UNDERESTIMATION,
}

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STRING = re.compile(r'"(?:\\.|[^"\\])*"')
_IDENTIFIER = re.compile(r"\bgl_\w+\b")
_EXTENSION_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*extension[ \t]+(\w+)", re.M)
_VERSION = re.compile(r"^[ \t]*#[ \t]*version[^\n]*\n", re.M)


def require_builtin_extensions(code, stage_extensions=()):
    """``code`` with an ``#extension <name> : require`` line for each extension its built-ins need.

    ``stage_extensions`` lists, in order, the extensions a built-in needs only in the stage the generator wrote it
    in, such as ``gl_Layer`` written by a vertex shader; they are required before the ones found by name.
    """
    if not isinstance(code, str):
        return code
    text = _STRING.sub('""', _COMMENT.sub(" ", code))
    declared = set(_EXTENSION_DIRECTIVE.findall(text))
    required = []
    for extension in stage_extensions:
        if extension not in declared:
            declared.add(extension)
            required.append(extension)
    for match in _IDENTIFIER.finditer(text):
        extension = BUILTIN_EXTENSIONS.get(match.group(0))
        if extension is not None and extension not in declared:
            declared.add(extension)
            required.append(extension)
    if not required:
        return code
    lines = "".join(f"#extension {extension} : require\n" for extension in required)
    version = _VERSION.search(code)
    if version is None:
        return lines + code
    return code[: version.end()] + lines + code[version.end() :]
