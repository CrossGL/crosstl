"""Probe templates for WebGL 2, whose shading language is GLSL ES 3.00.

The kernel around a body is :class:`~tools.coverage.languages.WebGL`'s: a fragment shader that reads uniform arrays
and writes a local array. GLSL ES 3.00 has only the vertex and fragment stages and none of compute shaders, storage
buffers, images, shared variables or runtime-sized arrays, which GLSL ES 3.10 adds; its scalar types are ``bool``,
``int``, ``uint`` and ``float`` (GLSL ES 3.00.6, sections 2 and 4.1). Those features are absent.
"""

from __future__ import annotations

from .base import ABSENT, C_FAMILY, Template
from .composites import WEBGL as _COMPOSITES

_HEADER = "#version 300 es\nprecision highp float;\n"
_ABSENT = (
    "compute",
    "runtime-array-length",
    "half",
    "double",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int64",
    "uint64",
    "shared-memory",
    "local-invocation-id",
    "workgroup-id",
    "local-invocation-index",
    "num-workgroups",
    "read-only-buffer",
    "storage-image",
    "tessellation-control",
    "tessellation-evaluation",
    "geometry",
    "ray-generation",
    "intersection",
    "any-hit",
    "closest-hit",
    "miss",
    "callable",
    "task",
    "mesh",
)

WEBGL = {
    **C_FAMILY,
    **dict.fromkeys(_ABSENT, ABSENT),
    **_COMPOSITES,
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", "const float SCALE = 2.0;"
    ),
    # GLSL ES 3.00 has uniform blocks but no binding layouts.
    "uniform-buffer": Template(
        "outp[i] = fin[i] * scale;",
        "layout(std140) uniform Params {\n    float scale;\n};",
    ),
    "sampled-texture": Template(
        "outp[i] = textureLod(tex, vec2(fin[i], 0.5), 0.0).x;",
        "uniform highp sampler2D tex;",
    ),
    "vertex": Template(
        program="#version 300 es\nin vec4 position;\nvoid main() {\n    gl_Position = position * 2.0;\n}\n"
    ),
    "fragment": Template(
        program=_HEADER
        + "in vec2 uv;\nlayout(location = 0) out vec4 color;\nvoid main() {\n    color = vec4(uv, 0.0, 1.0);\n}\n"
    ),
    "discard": Template(
        program=_HEADER
        + "in vec2 uv;\nlayout(location = 0) out vec4 color;\nvoid main() {\n    if (uv.x < 0.5) {\n"
        "        discard;\n    }\n    color = vec4(uv, 0.0, 1.0);\n}\n"
    ),
}
