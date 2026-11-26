"""Probe templates for GLSL ES 3.20, the shading language of OpenGL ES 3.2.

GLSL ES 3.20 is GLSL with fewer types and a few more rules, so a probe is GLSL's: a kernel template runs in
:class:`~tools.coverage.languages.GLES`'s kernel, and a stage program is GLSL's under ``#version 320 es`` with the
default precisions declared after its extensions (:func:`program`). :data:`OWN` holds the templates GLSL ES spells
otherwise, and :data:`MISSING` the features it does not have, each with the part of the GLSL ES 3.20 specification
that says so. Both specifications are built from one source (KhronosGroup/GLSL ``chapters/``), where
``ifdef::GLSL[]`` marks the text only desktop GLSL has.
"""

from __future__ import annotations

import dataclasses
import re

from .base import ABSENT, Template
from .glsl import GLSL
from .webgl import WEBGL

VERSION = "#version 320 es"
# A fragment shader has no default precision for floats, and most opaque types have none in any stage (GLSL ES 3.20,
# "Default Precision Qualifiers"), so every probe declares them.
PRECISION = ("precision highp float;", "precision highp int;")
# The opaque types a probe may declare, which take a default precision statement of their own.
_OPAQUE = re.compile(
    r"\b[iu]?(?:sampler(?:2D|3D|Cube|Buffer)\w*|image(?:2D|3D|Cube|Buffer)\w*|texture(?:2D|3D|Cube)\w*"
    r"|subpassInput\w*|sampler(?:Shadow)?)\b"
)

_RAY_TRACING = "GL_EXT_ray_tracing applies to desktop GLSL 4.60 and later only (its Dependencies section)"
_MESH = "GL_EXT_mesh_shader applies to desktop GLSL 4.50 and later only (its Dependencies section)"
_KEYWORD = "is a keyword GLSL ES 3.20 reserves for future use (Keywords)"
_DRAW = "GLSL ES 3.20 has no gl_DrawID, gl_BaseVertex or gl_BaseInstance (Vertex Shader Special Variables)"
# OpenGL ES extensions add clip and cull distances (EXT_clip_cull_distance), viewport indices (OES_viewport_array),
# fragment shader interlock (NV_fragment_shader_interlock) and 64-bit integers (NV_gpu_shader5), so those probes stay
# in the row even where glslang does not take the extension for GLSL ES. No OpenGL ES extension adds these two.
_STENCIL = (
    "GLSL ES 3.20 has no stencil reference output (Fragment Shader Special Variables), and the Khronos registry "
    "(xml/gl.xml) lists ARB_shader_stencil_export and AMD_shader_stencil_export for desktop OpenGL only"
)
_UNSIZED = (
    "a user-declared GLSL ES 3.20 array must not remain unsized (Arrays); GL_EXT_nonuniform_qualifier, which declares "
    "resource arrays of unknown length, requires GL_KHR_vulkan_glsl, which is GLSL for Vulkan, not OpenGL ES"
)

# What GLSL has and GLSL ES 3.20 does not, with the reason.
MISSING: dict[str, str] = {
    "double": f"double {_KEYWORD}",
    "texture-1d": f"sampler1D {_KEYWORD}",
    "texture-1d-array": f"sampler1DArray {_KEYWORD}",
    "storage-image-1d": f"image1D {_KEYWORD}",
    "base-instance": _DRAW,
    "base-vertex": _DRAW,
    "draw-index": _DRAW,
    "geometry-streams": (
        "GLSL ES 3.20 has no stream layout qualifier (Geometry Outputs)"
    ),
    "stencil-export": _STENCIL,
    "unsized-array-of-textures": _UNSIZED,
    **dict.fromkeys(
        (
            "ray-generation",
            "intersection",
            "any-hit",
            "closest-hit",
            "miss",
            "callable",
            "ray-callable-data",
            "ray-custom-attributes",
            "ray-payload",
            "ray-reported-attributes",
            "ray-shader-record",
        ),
        _RAY_TRACING,
    ),
    **dict.fromkeys(
        (
            "task",
            "mesh",
            "mesh-cull-primitive",
            "mesh-layer",
            "mesh-local-invocation-index",
            "mesh-output-lines",
            "mesh-output-points",
            "mesh-per-primitive-output",
            "mesh-primitive-id",
            "mesh-task-payload",
            "mesh-viewport-index",
            "mesh-workgroup-id",
            "task-local-invocation-index",
            "task-payload-array",
            "task-workgroup-id",
        ),
        _MESH,
    ),
}


def _extended(name: str, extension: str) -> Template:
    """GLSL's stage program for ``name``, with the GLSL ES extension that adds the feature to GLSL ES enabled."""
    text = GLSL[name].program
    version, rest = text.split("\n", 1)
    return Template(program=f"{version}\n#extension {extension} : require\n{rest}")


def _interlocked(name: str) -> Template:
    """GLSL's interlock probe ``name`` with NV_fragment_shader_interlock, the OpenGL ES extension (OpenGL ES 3.1 and
    later) that GLSL's ARB_fragment_shader_interlock, which requires GLSL 4.20, is to OpenGL.
    """
    text = GLSL[name].program
    for arb, nv in (
        ("GL_ARB_fragment_shader_interlock", "GL_NV_fragment_shader_interlock"),
        ("InvocationInterlockARB(", "InvocationInterlockNV("),
    ):
        if arb not in text:
            raise ValueError(f"the GLSL {name} probe has no {arb}")
        text = text.replace(arb, nv)
    return dataclasses.replace(GLSL[name], program=text)


# The templates GLSL ES spells otherwise than GLSL. It has no initializer lists, so arrays are initialized with
# array constructors, and no implicit conversions (Variables and Types), as in WebGL's GLSL ES 3.00 probes. Clip and
# cull distances, conservative depth, viewport indices, linear interpolation and fragment shader interlock come from
# GLSL ES extensions.
OWN: dict[str, Template] = {
    **{
        name: WEBGL[name]
        for name in (
            "array-initializer",
            "constant-array",
            "reserved-global-name",
            "implicit-conversion",
        )
    },
    "clip-distance": _extended("clip-distance", "GL_EXT_clip_cull_distance"),
    "cull-distance": _extended("cull-distance", "GL_EXT_clip_cull_distance"),
    "conservative-depth": _extended("conservative-depth", "GL_EXT_conservative_depth"),
    "viewport-index": _extended("viewport-index", "GL_OES_viewport_array"),
    "noperspective-interpolation": _extended(
        "noperspective-interpolation", "GL_NV_shader_noperspective_interpolation"
    ),
    **{
        name: _interlocked(name)
        for name in ("rasterizer-ordered-buffer", "rasterizer-ordered-image")
    },
}


def with_precisions(source: str) -> str:
    """``source`` with a default precision statement for each opaque type it names, after its last precision
    statement."""
    types = sorted(set(_OPAQUE.findall(source)))
    if not types:
        return source
    lines = source.splitlines(keepends=True)
    last = max(i for i, line in enumerate(lines) if line.startswith("precision "))
    added = [f"precision highp {kind};\n" for kind in types]
    return "".join(lines[: last + 1] + added + lines[last + 1 :])


def program(text: str) -> str:
    """A GLSL stage program as GLSL ES 3.20: its version line, its extensions, then the default precisions, which
    follow every ``#extension`` directive because none may follow a declaration (GLSL ES 3.20, "Preprocessor").
    """
    lines = text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("#version "):
        raise ValueError(f"a GLSL program begins with its #version line: {text[:40]!r}")
    head = 1
    while head < len(lines) and lines[head].startswith("#"):
        head += 1
    precision = [f"{line}\n" for line in PRECISION]
    return with_precisions(
        "".join([VERSION + "\n", *lines[1:head], *precision, *lines[head:]])
    )


def _es(template: Template) -> Template:
    if template is ABSENT or not template.program:
        return template
    return dataclasses.replace(template, program=program(template.program))


GLES: dict[str, Template] = {
    name: _es(template)
    for name, template in {
        **GLSL,
        **OWN,
        **dict.fromkeys(MISSING, ABSENT),
    }.items()
}
