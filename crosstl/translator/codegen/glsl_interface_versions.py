"""The GLSL versions of built-in variables, qualifiers and declaration forms, read from rendered program lines.

:mod:`.glsl_version_requirements` lists the versions of built-in functions, types, layout qualifiers on resources
and shader stages. This module adds the constructs a version line also has to cover once output starts from the
lowest version of its profile (:mod:`.glsl_version_floor`):

- built-in variables that a stage gains after GLSL 3.30 or GLSL ES 3.00, such as ``gl_SampleID`` (GLSL 4.00),
  ``gl_Layer`` in a fragment shader (GLSL 4.30) or ``gl_HelperInvocation`` (GLSL 4.50), from the per-stage
  built-in variable lists of GLSL 4.60 and GLSL ES 3.20 section 7.1;
- the ``precise`` and ``sample`` qualifiers (GLSL 4.00, section 4.7 and 4.5), the fragment-depth redeclaration
  (GLSL 4.20, section 4.4.2.3), ``early_fragment_tests`` (GLSL 4.20), geometry ``invocations`` and ``stream``
  (GLSL 4.00), the transform-feedback, ``offset`` and ``align`` layouts (GLSL 4.40, section 4.4.2.1 and 4.4.5);
- initializer lists (GLSL 4.20, section 4.1.11) and a sampler array indexed by a non-constant expression (GLSL
  4.00 made it dynamically uniform, section 4.1.7.1);
- a redeclared ``gl_PerVertex`` block (GLSL 4.10 section 7.1, with separate shader objects; GLSL ES 3.20 or
  GL_EXT_shader_io_blocks).

A built-in variable that an extension declares below its core version lists that extension, so a program that
enables it keeps its version. glslang declares a few of them only from a version of its own, which is the minimum
used here: the GL_ARB_shader_draw_parameters variables from GLSL 4.40, and the GL_EXT_fragment_shader_barycentric
and GL_EXT_fragment_shading_rate variables from GLSL 4.50.
"""

from __future__ import annotations

import re

from .glsl_version_requirements import GLSLRequirement

__all__ = [
    "INITIALIZER_LIST",
    "PER_VERTEX_REDECLARATION",
    "interface_line_requirements",
    "sampler_array_names",
]

_GPU_SHADER5 = ("GL_ARB_gpu_shader5",)
_ES_GPU_SHADER5 = ("GL_EXT_gpu_shader5", "GL_OES_gpu_shader5")
_SAMPLE_SHADING = ("GL_ARB_sample_shading",)
_ES_SAMPLE_VARIABLES = ("GL_OES_sample_variables",)
_ES_GEOMETRY = ("GL_EXT_geometry_shader", "GL_OES_geometry_shader")
_ANY_STAGE = "*"


def _variable(feature, desktop, es, desktop_extensions=(), es_extensions=()):
    return GLSLRequirement(
        f"built-in variable '{feature}'",
        desktop,
        es,
        desktop_extensions,
        es_extensions,
    )


_SAMPLE_VARIABLE = {
    "fragment": (400, 320, _SAMPLE_SHADING, _ES_SAMPLE_VARIABLES),
}
# Built-in variable -> {stage: (desktop, es, desktop extensions, es extensions)}. A stage that is not listed has
# the variable in the baseline (or not at all, which glslang reports as an undeclared identifier).
_BUILTIN_VARIABLES = {
    "gl_SampleID": _SAMPLE_VARIABLE,
    "gl_SamplePosition": _SAMPLE_VARIABLE,
    "gl_SampleMaskIn": _SAMPLE_VARIABLE,
    "gl_SampleMask": _SAMPLE_VARIABLE,
    "gl_NumSamples": _SAMPLE_VARIABLE,
    # GLSL ES fragment shaders read gl_PrimitiveID from GLSL ES 3.20 (or the geometry shader extensions); desktop
    # GLSL has it from GLSL 1.50.
    "gl_PrimitiveID": {"fragment": (150, 320, (), _ES_GEOMETRY)},
    "gl_Layer": {
        "fragment": (430, 320, (), _ES_GEOMETRY),
        "vertex": (410, None, ("GL_ARB_shader_viewport_layer_array",)),
        "tessellation_evaluation": (410, None, ("GL_ARB_shader_viewport_layer_array",)),
    },
    "gl_ViewportIndex": {
        "fragment": (430, None, (), ("GL_OES_viewport_array",)),
        "geometry": (410, None, ("GL_ARB_viewport_array",), ("GL_OES_viewport_array",)),
        "vertex": (410, None, ("GL_ARB_shader_viewport_layer_array",)),
        "tessellation_evaluation": (410, None, ("GL_ARB_shader_viewport_layer_array",)),
    },
    "gl_HelperInvocation": {"fragment": (450, 310)},
    "gl_CullDistance": {_ANY_STAGE: (450, None, (), ("GL_EXT_clip_cull_distance",))},
    "gl_InvocationID": {"geometry": (400, 320, _GPU_SHADER5, _ES_GEOMETRY)},
    "gl_BaseVertexARB": {_ANY_STAGE: (440, None)},
    "gl_BaseInstanceARB": {_ANY_STAGE: (440, None)},
    "gl_DrawIDARB": {_ANY_STAGE: (440, None)},
    "gl_BaseVertex": {_ANY_STAGE: (460, None)},
    "gl_BaseInstance": {_ANY_STAGE: (460, None)},
    "gl_DrawID": {_ANY_STAGE: (460, None)},
    "gl_BaryCoordEXT": {_ANY_STAGE: (450, 320)},
    "gl_BaryCoordNoPerspEXT": {_ANY_STAGE: (450, 320)},
    "gl_ShadingRateEXT": {_ANY_STAGE: (450, 310)},
    "gl_PrimitiveShadingRateEXT": {_ANY_STAGE: (450, 320)},
}
_VARIABLE_USE = re.compile(
    r"\b(" + "|".join(sorted(_BUILTIN_VARIABLES, key=len, reverse=True)) + r")\b"
)


def _variable_requirement(name, stage):
    """The requirement of built-in variable ``name`` in ``stage`` (the highest of its stages when unknown)."""
    stages = _BUILTIN_VARIABLES[name]
    if isinstance(stage, str):
        entry = stages.get(stage) or stages.get(_ANY_STAGE)
    else:
        # A line of an artifact that carries several stages without a guard: use the stage that needs most.
        entry = max(stages.values(), key=lambda item: item[0] or 0)
    if entry is None:
        return None
    return _variable(name, *entry)


_QUALIFIER_REQUIREMENTS = {
    "precise": GLSLRequirement(
        "the precise qualifier", 400, 320, _GPU_SHADER5, _ES_GPU_SHADER5
    ),
    "sample": GLSLRequirement(
        "the sample interpolation qualifier",
        400,
        320,
        _GPU_SHADER5,
        ("GL_OES_shader_multisample_interpolation",),
    ),
}
_QUALIFIER_USE = re.compile(r"(?:^|[\s(])(precise|sample)\s+(?=[A-Za-z_])")
_LAYOUT_ID_REQUIREMENTS = (
    (
        frozenset({"early_fragment_tests"}),
        GLSLRequirement(
            "layout(early_fragment_tests)",
            420,
            310,
            ("GL_ARB_shader_image_load_store",),
        ),
    ),
    (
        frozenset({"invocations"}),
        GLSLRequirement(
            "a geometry invocations layout", 400, 320, _GPU_SHADER5, _ES_GEOMETRY
        ),
    ),
    (
        frozenset({"stream"}),
        GLSLRequirement("a geometry stream layout", 400, None, _GPU_SHADER5),
    ),
    (
        frozenset({"xfb_buffer", "xfb_offset", "xfb_stride"}),
        GLSLRequirement(
            "a transform feedback layout", 440, None, ("GL_ARB_enhanced_layouts",)
        ),
    ),
    (
        frozenset({"offset", "align"}),
        GLSLRequirement(
            "an offset or align layout", 440, None, ("GL_ARB_enhanced_layouts",)
        ),
    ),
)
INITIALIZER_LIST = GLSLRequirement(
    "an initializer list", 420, None, ("GL_ARB_shading_language_420pack",)
)
# ``= {`` after a declarator, which is how an initializer list starts (a block or a function body never follows
# an ``=``).
_INITIALIZER_LIST = re.compile(r"=\s*\{")
DYNAMIC_SAMPLER_INDEX = GLSLRequirement(
    "a sampler array indexed by a non-constant expression",
    400,
    320,
    _GPU_SHADER5,
    _ES_GPU_SHADER5,
)
_SAMPLER_ARRAY_DECLARATION = re.compile(
    r"\buniform\s+(?:(?:highp|mediump|lowp)\s+)?[iu]?sampler\w*\s+([A-Za-z_]\w*)\s*\["
)
_LITERAL_INDEX = re.compile(r"\s*\d+[uU]?\s*")
# Redeclaring the built-in ``gl_PerVertex`` input or output block came with separate shader objects (GLSL 4.10
# section 7.1); glslang refuses it below GLSL 4.10 without GL_ARB_separate_shader_objects. GLSL ES has stage
# interface blocks, and so the redeclaration, from GLSL ES 3.20 or GL_EXT_shader_io_blocks (on GLSL ES 3.10).
PER_VERTEX_REDECLARATION = GLSLRequirement(
    "a gl_PerVertex block redeclaration",
    410,
    320,
    ("GL_ARB_separate_shader_objects",),
    ("GL_EXT_shader_io_blocks", "GL_OES_shader_io_blocks"),
)
_PER_VERTEX_REDECLARATION = re.compile(
    r"^(?:layout\s*\([^()]*(?:\([^()]*\)[^()]*)*\)\s*)?(?:[a-z]+\s+)*(?:in|out)\s+gl_PerVertex\b"
)


def sampler_array_names(body):
    """The names of the sampler arrays a rendered program declares."""
    return frozenset(_SAMPLER_ARRAY_DECLARATION.findall(body))


def _indexes_sampler_array_dynamically(stripped, sampler_arrays):
    if not sampler_arrays or _SAMPLER_ARRAY_DECLARATION.search(stripped):
        return False
    for name in sampler_arrays:
        for match in re.finditer(rf"\b{re.escape(name)}\s*\[", stripped):
            start = match.end()
            depth = 1
            for index in range(start, len(stripped)):
                char = stripped[index]
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        if not _LITERAL_INDEX.fullmatch(stripped[start:index]):
                            return True
                        break
    return False


def interface_line_requirements(stripped, stage, layout_ids=None, sampler_arrays=()):
    """The requirements of one rendered, comment-free line of ``stage`` that the resource tables do not cover.

    ``layout_ids`` are the line's layout qualifier ids when it starts with ``layout(...)``, and
    ``sampler_arrays`` the program's sampler array names (:func:`sampler_array_names`).
    """
    requirements = []
    for match in _VARIABLE_USE.finditer(stripped):
        requirement = _variable_requirement(match.group(1), stage)
        if requirement is not None:
            requirements.append(requirement)
    for match in _QUALIFIER_USE.finditer(stripped):
        requirements.append(_QUALIFIER_REQUIREMENTS[match.group(1)])
    if layout_ids:
        for ids, requirement in _LAYOUT_ID_REQUIREMENTS:
            if ids & layout_ids:
                requirements.append(requirement)
    if _INITIALIZER_LIST.search(stripped):
        requirements.append(INITIALIZER_LIST)
    if _indexes_sampler_array_dynamically(stripped, sampler_arrays):
        requirements.append(DYNAMIC_SAMPLER_INDEX)
    if _PER_VERTEX_REDECLARATION.match(stripped):
        requirements.append(PER_VERTEX_REDECLARATION)
    return requirements
