"""The GLSL profile of the output, and the ``#version`` and other preprocessor lines it takes from the program.

The target chooses the profile: the ``opengl`` backend writes desktop GLSL, its ``gles`` target (``opengl-es``) writes
GLSL ES for OpenGL ES 3.x, and the ``webgl`` backend writes GLSL ES 3.00. A CrossGL ``#version`` line, which a GLSL
or GLSL ES source passes on, chooses the version output starts from: a line of the other profile maps to the first
version of the output's profile that has its constructs (:func:`_profile_version_line`), so a WebGL 2 or OpenGL ES
source gives desktop GLSL for OpenGL. The generator then raises the version to the lowest one of the profile that has
every construct the program uses.

Each profile lacks the other's extensions:

- Desktop output leaves out a GLSL ES ``#extension`` line. When desktop GLSL has the extension's constructs in its
  core language, the version rises to the first desktop version that has them all; when desktop GLSL has no form of
  it, such as ``samplerExternalOES`` or framebuffer fetch, the line is refused.
- GLSL ES output leaves out a ``GL_ARB_`` line whose constructs GLSL ES has, and refuses one GLSL ES has no form
  of.

An extension line that disables the extension is left out of the other profile's output.
"""

from __future__ import annotations

import re

from ..ast import PreprocessorNode
from ..refusals import OpenGLRefusal

__all__ = [
    "DESKTOP_EXTENSIONS_IN_ES_CORE",
    "DESKTOP_VERSIONS_OF_ES",
    "ES_EXTENSIONS_IN_DESKTOP_CORE",
    "ES_EXTENSIONS_WITHOUT_DESKTOP_FORM",
    "GLSL_ES_TARGET_PROFILES",
    "glsl_source_directives",
    "glsl_target_profile_is_es",
]

# The target names that select GLSL ES output (``crosstl.translator.codegen`` registers them for ``opengl``).
GLSL_ES_TARGET_PROFILES = frozenset({"gles", "opengles", "opengl-es"})

# GLSL ES extensions whose constructs desktop GLSL has in its core language, with the first desktop version that has
# all of them (GLSL 4.60 section 1.2, "Changes", per version).
ES_EXTENSIONS_IN_DESKTOP_CORE = {
    # Derivatives: GLSL 1.10.
    "GL_OES_standard_derivatives": 110,
    # Texture buffers: GLSL 1.40.
    "GL_EXT_texture_buffer": 140,
    "GL_OES_texture_buffer": 140,
    # Geometry shaders, interface blocks on stage inputs and outputs, and 2D multisample array samplers: GLSL 1.50.
    "GL_EXT_geometry_shader": 150,
    "GL_OES_geometry_shader": 150,
    "GL_EXT_geometry_point_size": 150,
    "GL_OES_geometry_point_size": 150,
    "GL_EXT_shader_io_blocks": 150,
    "GL_OES_shader_io_blocks": 150,
    "GL_OES_texture_storage_multisample_2d_array": 150,
    # Tessellation, GL_ARB_gpu_shader5, sample shading, interpolateAt* and cube map arrays: GLSL 4.00.
    "GL_EXT_tessellation_shader": 400,
    "GL_OES_tessellation_shader": 400,
    "GL_EXT_tessellation_point_size": 400,
    "GL_OES_tessellation_point_size": 400,
    "GL_EXT_gpu_shader5": 400,
    "GL_OES_gpu_shader5": 400,
    "GL_OES_sample_variables": 400,
    "GL_OES_shader_multisample_interpolation": 400,
    "GL_EXT_texture_cube_map_array": 400,
    "GL_OES_texture_cube_map_array": 400,
    # Separate shader objects (stage interface matching by location): GLSL 4.10.
    "GL_EXT_separate_shader_objects": 410,
    # Conservative depth layouts and image atomics: GLSL 4.20.
    "GL_EXT_conservative_depth": 420,
    "GL_OES_shader_image_atomic": 420,
    # gl_CullDistance: GLSL 4.50 (gl_ClipDistance is in every desktop version this generator writes).
    "GL_EXT_clip_cull_distance": 450,
}

# GLSL ES extensions desktop GLSL has no form of, each with what it adds.
ES_EXTENSIONS_WITHOUT_DESKTOP_FORM = {
    "GL_OES_EGL_image_external": "samplerExternalOES",
    "GL_OES_EGL_image_external_essl3": "samplerExternalOES",
    "GL_EXT_YUV_target": "__samplerExternal2DY2YEXT and YUV fragment outputs",
    "GL_EXT_primitive_bounding_box": "gl_BoundingBox",
    "GL_OES_primitive_bounding_box": "gl_BoundingBox",
    "GL_EXT_shader_framebuffer_fetch": "inout fragment outputs (framebuffer fetch)",
    "GL_EXT_shader_framebuffer_fetch_non_coherent": (
        "inout fragment outputs (framebuffer fetch)"
    ),
    "GL_ARM_shader_framebuffer_fetch": "gl_LastFragColorARM",
    "GL_ARM_shader_framebuffer_fetch_depth_stencil": "gl_LastFragDepthARM",
    "GL_EXT_shader_pixel_local_storage": "pixel local storage blocks",
    "GL_EXT_shader_pixel_local_storage2": "pixel local storage blocks",
}

# Desktop (ARB) extensions whose constructs GLSL ES 3.20 has in its core language (GLSL ES 3.20 section 1.1, and the
# GLSL ES 3.00 and 3.10 specifications). GLSL output takes GL_ARB_shading_language_420pack for ``layout(binding)``
# only. GLSL ES output leaves their lines out and raises its version for the constructs the program uses, as it does
# for every construct, so a line that enables an extension the program does not use costs no version.
DESKTOP_EXTENSIONS_IN_ES_CORE = frozenset(
    {
        "GL_ARB_arrays_of_arrays",
        "GL_ARB_compute_shader",
        "GL_ARB_explicit_attrib_location",
        "GL_ARB_explicit_uniform_location",
        "GL_ARB_gpu_shader5",
        "GL_ARB_sample_shading",
        "GL_ARB_separate_shader_objects",
        "GL_ARB_shader_atomic_counters",
        "GL_ARB_shader_bit_encoding",
        "GL_ARB_shader_image_load_store",
        "GL_ARB_shader_image_size",
        "GL_ARB_shader_storage_buffer_object",
        "GL_ARB_shading_language_420pack",
        "GL_ARB_shading_language_packing",
        "GL_ARB_tessellation_shader",
        "GL_ARB_texture_buffer_object",
        "GL_ARB_texture_cube_map_array",
        "GL_ARB_texture_gather",
        "GL_ARB_texture_multisample",
    }
)

# The desktop version a GLSL ES ``#version`` maps to in desktop output (:func:`_profile_version_line`).
DESKTOP_VERSIONS_OF_ES = {100: 330, 300: 330, 310: 450, 320: 450}

_EXTENSION_LINE = re.compile(r"^\s*#\s*extension\s+([A-Za-z_]\w*)\s*:\s*(\w+)")


def glsl_target_profile_is_es(target_profile):
    """Whether ``target_profile``, a target name the ``opengl`` backend registers, asks for GLSL ES output."""
    if target_profile is None:
        return False
    name = str(target_profile).strip().lower().replace("_", "-")
    if name not in GLSL_ES_TARGET_PROFILES:
        raise OpenGLRefusal(
            f"unknown GLSL target profile '{target_profile}'; GLSL output names "
            f"{', '.join(sorted(GLSL_ES_TARGET_PROFILES))} for GLSL ES"
        )
    return True


def _directive_line(directive):
    if not isinstance(directive, PreprocessorNode):
        return str(directive).strip()
    if directive.directive == "precision":
        return f"precision {directive.content};" if directive.content else "precision;"
    return f"#{directive.directive} {directive.content}".strip()


def _foreign_extension(name, is_es):
    """``(minimum, missing)`` for an extension of the other profile: the version it needs, or what it adds.

    Both are ``None`` for an extension of the output's own profile.
    """
    if is_es:
        if not name.startswith("GL_ARB_"):
            return None, None
        if name in DESKTOP_EXTENSIONS_IN_ES_CORE:
            return 0, None
        return None, "a desktop OpenGL extension"
    if name in ES_EXTENSIONS_WITHOUT_DESKTOP_FORM:
        return None, ES_EXTENSIONS_WITHOUT_DESKTOP_FORM[name]
    return ES_EXTENSIONS_IN_DESKTOP_CORE.get(name), None


def _profile_lines(version_line, lines, is_es, refusal_type):
    """``lines`` without the other profile's extensions, and the lowest version their constructs need.

    An extension the output's profile has no form of raises ``refusal_type``, the generator's lowering refusal.
    """
    kept = []
    minimum = 0
    profile = "GLSL ES" if is_es else "desktop GLSL"
    for line in lines:
        match = _EXTENSION_LINE.match(line)
        if match is None:
            kept.append(line)
            continue
        name, behavior = match.groups()
        version, missing = _foreign_extension(name, is_es)
        if version is None and missing is None:
            kept.append(line)
            continue
        if behavior == "disable":
            continue
        if missing is not None:
            raise refusal_type(
                [
                    f"{profile} has no form of the extension {name} ({missing}); the target is {version_line}"
                ]
            )
        minimum = max(minimum, version)
    return kept, minimum


def _profile_version_line(codegen, version_line):
    """The program's ``version_line`` in the output's profile: the first version of it with the line's constructs.

    GLSL ES 1.00 and 3.00 map to GLSL 3.30, which has the constructs of GLSL ES 3.00 (stage interfaces with
    ``layout(location)``, uniform blocks, integer types and the GLSL 1.30 texture functions; the packing functions are
    newer on desktop, and the generator raises the version when the program calls them). GLSL ES 3.10 and 3.20 map to
    GLSL 4.50, whose OpenGL version took in GL_ARB_ES3_1_compatibility (``gl_HelperInvocation`` and ``mix()`` with
    integer and boolean operands) and which has the compute, storage buffer, image and binding constructs of GLSL 4.30.
    A desktop line gives GLSL ES output its lowest version, GLSL ES 3.00.
    """
    number = codegen.glsl_version_number(version_line)
    is_es = codegen.glsl_version_is_es(version_line) or number == 100
    if codegen.glsl_es_target:
        return version_line if is_es and number != 100 else "#version 300 es"
    if not is_es:
        return version_line
    return f"#version {DESKTOP_VERSIONS_OF_ES.get(number, 450)} core"


def glsl_source_directives(codegen, ast, target_stage):
    """Return ``(version_line, lines)``: the ``#version`` line output starts from and the other preprocessor lines.

    The version line is the program's own in the output's profile, the generator's default when it has none, or the
    version an earlier rendering of the program had to rise to. Each profile leaves out the other's extensions and
    starts from a version that has their constructs.
    """
    version_line = None
    lines = []
    for directive in getattr(ast, "preprocessors", []) or []:
        if not codegen.should_emit_glsl_preprocessor_directive(directive):
            continue
        line = _directive_line(directive)
        if line.startswith("#version") and version_line is None:
            version_line = line
        elif line and codegen.should_emit_glsl_preprocessor_line(line):
            lines.append(line)
    # Without a line of its own, the program starts from the lowest version of the profile and output chooses
    # every version-dependent form so that the version can rise (:mod:`.glsl_version_floor`).
    codegen.glsl_version_line_is_default = version_line is None
    if version_line is None:
        version_line = codegen.default_glsl_version_line(ast, target_stage)
    version_line = _profile_version_line(codegen, version_line)
    required_version_line = getattr(codegen, "glsl_required_version_line", None)
    if required_version_line is not None:
        version_line = required_version_line
    lines, minimum = _profile_lines(
        version_line,
        lines,
        codegen.glsl_version_is_es(version_line),
        codegen.LOWERING_REFUSAL_TYPE,
    )
    if minimum:
        # Only desktop output raises its version for extensions (GLSL ES output raises it per construct).
        version_line = codegen.glsl_version_line_with_minimum(version_line, minimum)
    return version_line, lines
