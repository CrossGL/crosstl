"""Probe templates for WebGL 2, whose shading language is GLSL ES 3.00.

The kernel around a body is :class:`~tools.coverage.languages.WebGL`'s: a fragment shader that reads uniform arrays
and writes a local array. GLSL ES 3.00 has only the vertex and fragment stages and none of compute shaders, storage
buffers, images, shared variables or runtime-sized arrays, which GLSL ES 3.10 adds; its scalar types are ``bool``,
``int``, ``uint`` and ``float`` (GLSL ES 3.00.6, sections 2 and 4.1). Those features are absent.
"""

from __future__ import annotations

from .abstractions import FAMILY as _ABSTRACTIONS
from .abstractions import GLSL as _ABSTRACTIONS_GLSL
from .aggregates import FAMILY as _AGGREGATES
from .base import ABSENT, C_FAMILY, Template
from .blocks import WEBGL as _BLOCKS
from .booleans import WEBGL as _BOOLEANS
from .calls import WEBGL as _CALLS
from .columns import WEBGL as _COLUMNS
from .composites import WEBGL as _COMPOSITES
from .constructors import GLSL as _CONSTRUCTORS
from .edges import FAMILY as _EDGES
from .elements import WEBGL as _ELEMENTS
from .evaluation import WEBGL as _EVALUATION
from .forms import FAMILY as _FORMS
from .forms import GLSL as _FORMS_GLSL
from .fragments import WEBGL as _FRAGMENTS
from .helpers import WEBGL as _HELPERS
from .identifiers import WEBGL as _IDENTIFIERS
from .idioms import FAMILY as _IDIOMS
from .idioms import GLSL as _IDIOMS_GLSL
from .initializers import FAMILY as _INITIALIZERS
from .initializers import GLSL as _INITIALIZERS_GLSL
from .integers import WEBGL as _INTEGERS
from .interfaces import WEBGL as _INTERFACES
from .layouts import WEBGL as _LAYOUTS
from .lexical import FAMILY as _LEXICAL
from .lexical import WEBGL as _LEXICAL_WEBGL
from .orientation import WEBGL as _ORIENTATION
from .packing import WEBGL as _PACKING
from .pipeline import WEBGL as _PIPELINE
from .portability import FAMILY as _PORTABILITY
from .portability import GLSL as _PORTABILITY_GLSL
from .portability import WEBGL as _PORTABILITY_WEBGL
from .rasterization import WEBGL as _RASTERIZATION
from .records import WEBGL as _RECORDS
from .relations import GLSL as _RELATIONS
from .rows import WEBGL as _ROWS
from .scopes import WEBGL as _SCOPES
from .subgroups import WEBGL as _SUBGROUPS
from .values import FAMILY as _VALUES
from .vectors import GLSL as _VECTORS
from .draws import WEBGL as _DRAWS
from .vertices import WEBGL as _VERTICES
from .workgroups import WEBGL as _WORKGROUPS

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
    **_AGGREGATES,
    **_VALUES,
    **_EDGES,
    **_LEXICAL,
    **_LEXICAL_WEBGL,
    **_FORMS,
    **_FORMS_GLSL,
    **_IDIOMS,
    **_IDIOMS_GLSL,
    **_ABSTRACTIONS,
    **_ABSTRACTIONS_GLSL,
    **_PORTABILITY,
    **_PORTABILITY_GLSL,
    **_PORTABILITY_WEBGL,
    **_INITIALIZERS,
    **_INITIALIZERS_GLSL,
    **_HELPERS,
    **_LAYOUTS,
    **_ELEMENTS,
    **_BLOCKS,
    **_WORKGROUPS,
    **_RECORDS,
    **_IDENTIFIERS,
    **_INTEGERS,
    **_COLUMNS,
    **_BOOLEANS,
    **_PACKING,
    **_ORIENTATION,
    **_ROWS,
    **_VECTORS,
    **_RELATIONS,
    **_FRAGMENTS,
    **_INTERFACES,
    **_VERTICES,
    **_DRAWS,
    **_RASTERIZATION,
    **_PIPELINE,
    **_SUBGROUPS,
    **dict.fromkeys(_ABSENT, ABSENT),
    **_COMPOSITES,
    **_CONSTRUCTORS,
    **_SCOPES,
    **_CALLS,
    **_EVALUATION,
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
