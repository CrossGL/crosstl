"""Built-in variables and functions a GLSL target profile does not have.

Imported programs name built-ins of the language they came from: Vulkan GLSL's ``gl_ViewIndex`` (HLSL's
``SV_ViewID``), or GLSL 4.x variables that GLSL ES 3.00 lacks. Emitting them for a profile without them yields a shader its compiler rejects, so the OpenGL and
WebGL emitters scan the rendered program and refuse each one with the reason the profile cannot express it.
"""

import re

_VIEW_INDEX = (
    "the multiview view index exists only in Vulkan GLSL (GL_EXT_multiview); "
    "GL_OVR_multiview also needs the view count (layout(num_views)), which the source does not state"
)

_QUAD_GROUP = "GLSL has no quad-group index; only a compute shader derives it, from gl_LocalInvocationIndex"

# Desktop OpenGL GLSL 4.60 (GL_KHR_vulkan_glsl section "Removed features" lists the Vulkan-only additions).
OPENGL_UNAVAILABLE_BUILTINS = {
    "gl_ViewIndex": _VIEW_INDEX,
    "gl_DeviceIndex": (
        "the device-group index exists only in Vulkan GLSL (GL_EXT_device_group)"
    ),
    # Metal's quad-group indices; compute shaders read them from gl_LocalInvocationIndex instead.
    "gl_QuadGroupID": _QUAD_GROUP,
    "gl_QuadGroupInvocationID": _QUAD_GROUP,
}

_ES_CLIP = "GLSL ES 3.00 has no clip or cull distances"
_ES_BARYCENTRIC = "GLSL ES 3.00 has no fragment barycentrics"
_ES_SAMPLE = "GLSL ES 3.00 has no per-sample shading variables (GLSL ES 3.20)"
_ES_LAYER = "GLSL ES 3.00 has no layered or multi-viewport rendering"
_ES_DENSITY = "GLSL ES 3.00 has no fragment density map queries (GL_EXT_fragment_invocation_density)"
_ES_SHADING_RATE = (
    "GLSL ES 3.00 has no fragment shading rate (GL_EXT_fragment_shading_rate)"
)

# OpenGL ES Shading Language 3.00, section 7 (built-in variables) against GLSL 4.60 section 7.
WEBGL_UNAVAILABLE_BUILTINS = {
    **OPENGL_UNAVAILABLE_BUILTINS,
    "gl_ClipDistance": _ES_CLIP,
    "gl_CullDistance": _ES_CLIP,
    "gl_BaryCoordEXT": _ES_BARYCENTRIC,
    "gl_BaryCoordNoPerspEXT": _ES_BARYCENTRIC,
    "gl_BaryCoordNV": _ES_BARYCENTRIC,
    "gl_BaryCoordNoPerspNV": _ES_BARYCENTRIC,
    "gl_PrimitiveID": "GLSL ES 3.00 has no primitive ID (GLSL ES 3.20)",
    "gl_Layer": _ES_LAYER,
    "gl_ViewportIndex": _ES_LAYER,
    "gl_SampleID": _ES_SAMPLE,
    "gl_SamplePosition": _ES_SAMPLE,
    "gl_SampleMaskIn": _ES_SAMPLE,
    "gl_SampleMask": _ES_SAMPLE,
    "gl_NumSamples": _ES_SAMPLE,
    "gl_HelperInvocation": "GLSL ES 3.00 has no helper-invocation query (GLSL ES 3.10)",
    "gl_DrawID": "GLSL ES 3.00 has no draw index",
    "gl_BaseVertex": "GLSL ES 3.00 has no base vertex",
    "gl_BaseInstance": "GLSL ES 3.00 has no base instance",
    # GL_EXT_fragment_invocation_density and GL_EXT_fragment_shading_rate are not WebGL 2 extensions.
    "gl_FragSizeEXT": _ES_DENSITY,
    "gl_FragInvocationCountEXT": _ES_DENSITY,
    "gl_ShadingRateEXT": _ES_SHADING_RATE,
    "gl_PrimitiveShadingRateEXT": _ES_SHADING_RATE,
    # Vendor extensions of desktop GLSL that are not WebGL 2 extensions.
    **{
        name: (
            "GLSL ES 3.00 has no explicit vertex parameters (GL_AMD_shader_explicit_vertex_parameter)"
        )
        for name in (
            "gl_BaryCoordNoPerspAMD",
            "gl_BaryCoordNoPerspCentroidAMD",
            "gl_BaryCoordNoPerspSampleAMD",
            "gl_BaryCoordSmoothAMD",
            "gl_BaryCoordSmoothCentroidAMD",
            "gl_BaryCoordSmoothSampleAMD",
            "gl_BaryCoordPullModelAMD",
        )
    },
    **{
        name: "GLSL ES 3.00 has no SM built-ins (GL_NV_shader_sm_builtins)"
        for name in ("gl_WarpIDNV", "gl_SMIDNV", "gl_WarpsPerSMNV", "gl_SMCountNV")
    },
    "gl_FragFullyCoveredNV": (
        "GLSL ES 3.00 has no full-coverage query (GL_NV_conservative_raster_underestimation)"
    ),
}

WEBGL_UNAVAILABLE_FUNCTION_PATTERNS = (
    (
        re.compile(r"\bsparse\w*ARB\b"),
        "GLSL ES 3.00 has no sparse texture residency queries (GL_ARB_sparse_texture2)",
    ),
    (
        re.compile(r"\b(?:rayQueryEXT|accelerationStructureEXT)\b"),
        "GLSL ES 3.00 has no ray queries or acceleration structures (GL_EXT_ray_query is not a WebGL 2 extension)",
    ),
)

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STRING = re.compile(r'"(?:\\.|[^"\\])*"')
_IDENTIFIER = re.compile(r"\bgl_\w+\b")


def unavailable_builtin_reasons(code, builtins, patterns=()):
    """Return one refusal reason per built-in of ``builtins`` that ``code`` references, in first-use order."""
    if not isinstance(code, str):
        return []
    text = _STRING.sub('""', _COMMENT.sub(" ", code))
    reasons = []
    for match in _IDENTIFIER.finditer(text):
        name = match.group(0)
        reason = builtins.get(name)
        if reason is None:
            continue
        entry = f"'{name}': {reason}"
        if entry not in reasons:
            reasons.append(entry)
    for pattern, reason in patterns:
        found = pattern.search(text)
        if found is not None:
            entry = f"'{found.group(0)}': {reason}"
            if entry not in reasons:
                reasons.append(entry)
    return reasons
