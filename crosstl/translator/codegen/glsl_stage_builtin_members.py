"""Built-in members of GLSL vertex output and fragment input structures beyond the ones GLSL 4.60 core declares.

GLSL has no stage structures: each member of a vertex entry's return structure becomes an ``out`` variable or a
built-in, and each member of a fragment entry's input structure an ``in`` variable or a built-in. The generator knew
only the ``gl_PerVertex`` outputs (``gl_Position``, ``gl_PointSize`` and the distances) and seven fragment inputs,
and declared any other ``gl_`` member as a user varying with no location. A vertex shader writing ``@gl_Layer`` then
emitted ``flat out uint layer;`` and never selected a layer, and a fragment input ``@gl_HelperInvocation`` read a
varying no stage writes, although the same built-in taken as an entry parameter was read correctly.

* Vertex outputs ``gl_Layer`` and ``gl_ViewportIndex`` are written to the built-ins, which the vertex and
  tessellation evaluation stages have under ``GL_ARB_shader_viewport_layer_array`` (GLSL 4.60 core declares them
  for geometry shaders only). The program requires that extension, also when a vertex or tessellation
  evaluation function stores into them directly, and a value stored into them is converted to
  the ``int`` GLSL declares them with (:mod:`.glsl_stage_output_values`).
* Every other CrossGL ``gl_`` vertex output reaching the user-varying fallback is refused. The remaining ones,
  ``gl_PrimitiveShadingRateEXT`` and the NVIDIA multi-view, stereo and viewport-mask outputs, are either Vulkan GLSL
  only (``GL_EXT_fragment_shading_rate``) or per-view arrays a scalar member cannot carry.
* A fragment entry parameter reading the distances takes their first elements, and one reading ``gl_PointSize``,
  which no GLSL fragment shader has, is refused.
* A fragment input member names the built-in an entry parameter with the same attribute reads
  (:meth:`~.GLSL_codegen.GLSLCodeGen.stage_input_builtin_alias`) for every built-in GLSL declares as a fragment
  input. The CrossGL conventions handled elsewhere keep their paths: the distances, the colour attachment inputs
  ``gl_FragColor``/``gl_FragData``, and ``gl_PointSize``, which a fragment stage taking the vertex output structure
  carries without reading.
"""

from __future__ import annotations

import re

from ..semantics.builtin_names import is_crossgl_gl_builtin
from ..semantics.builtin_stages import builtin_stage_rule
from ..stage_utils import normalize_stage_name

__all__ = [
    "VIEWPORT_LAYER_EXTENSION",
    "fragment_member_builtin_alias",
    "fragment_parameter_builtin_expression",
    "is_vertex_layer_output",
    "note_builtin_store",
    "note_vertex_builtin_output",
    "refuse_unmapped_vertex_output",
]

VIEWPORT_LAYER_EXTENSION = "GL_ARB_shader_viewport_layer_array"

# GL_ARB_shader_viewport_layer_array, "Additions to Chapter 7 of the OpenGL Shading Language Specification".
_VERTEX_LAYER_OUTPUTS = frozenset({"gl_Layer", "gl_ViewportIndex"})

# Fragment inputs whose CrossGL conventions other modules lower (see the module docstring).
_FRAGMENT_CONVENTION_INPUTS = frozenset(
    {
        "gl_PointSize",
        "gl_FragColor",
        "gl_FragData",
        "gl_ClipDistance",
        "gl_CullDistance",
    }
)
_INDEX_SUFFIX = re.compile(r"\[\d+\]$|\d+$")
_VECTOR_WIDTH = re.compile(r"vec([234])")


def is_vertex_layer_output(name) -> bool:
    """Whether ``name`` is a layer or viewport built-in the vertex stage writes under the extension."""
    return str(name or "") in _VERTEX_LAYER_OUTPUTS


def note_vertex_builtin_output(codegen, name):
    """Record the extension a vertex output built-in ``name`` needs, once per program."""
    if not is_vertex_layer_output(name):
        return
    _require(codegen)


def note_builtin_store(codegen, name):
    """Record the extension a store into ``name`` needs in the stage being generated, such as a helper's."""
    if not is_vertex_layer_output(name):
        return
    stage = normalize_stage_name(getattr(codegen, "current_stage_entry_type", None))
    if stage in ("vertex", "tessellation_evaluation"):
        _require(codegen)


def _require(codegen):
    required = getattr(codegen, "glsl_stage_builtin_extensions", None)
    if required is None:
        required = codegen.glsl_stage_builtin_extensions = []
    if VIEWPORT_LAYER_EXTENSION not in required:
        required.append(VIEWPORT_LAYER_EXTENSION)


def refuse_unmapped_vertex_output(codegen, semantic, mapped):
    """Refuse a CrossGL ``gl_`` built-in that would otherwise become a vertex output varying."""
    text = str(semantic or "").strip()
    name = str(mapped or "")
    if not name.startswith("gl_") or not is_crossgl_gl_builtin(text):
        return
    codegen.refuse_lowering(
        f"OpenGL GLSL has no vertex output built-in for @{text}; "
        "it would become a user varying no stage reads as the built-in"
    )


def fragment_member_builtin_alias(codegen, semantic, mapped):
    """The GLSL expression a fragment input member with the built-in ``semantic`` reads, or ``None``."""
    name = str(mapped or "")
    if not name.startswith("gl_"):
        return None
    text = str(semantic or "").strip()
    if _INDEX_SUFFIX.sub("", name) in _FRAGMENT_CONVENTION_INPUTS:
        return None
    rule = builtin_stage_rule(text) if is_crossgl_gl_builtin(text) else None
    if rule is None and is_crossgl_gl_builtin(name):
        rule = builtin_stage_rule(name)
    if rule is None or "fragment" not in rule["input"]:
        return None
    return codegen.stage_input_builtin_alias(semantic, "fragment")


def fragment_parameter_builtin_expression(codegen, builtin, base_type, stage):
    """The value a scalar or vector fragment entry parameter reading ``builtin`` holds, or ``None``.

    GLSL declares the fragment's distances as ``in float gl_ClipDistance[]`` (GLSL 4.60 section 7.1.5), so a
    ``float`` parameter reads the first element and a ``vecN`` one the first ``N``; an array parameter names the
    array itself. GLSL has no fragment ``gl_PointSize``, so a fragment parameter reading it is refused.
    """
    if normalize_stage_name(stage) != "fragment":
        return None
    if builtin == "gl_PointSize":
        return codegen.refuse_lowering(
            "OpenGL GLSL has no fragment input gl_PointSize; "
            "a fragment parameter @gl_PointSize reads a value no stage passes"
        )
    if builtin not in ("gl_ClipDistance", "gl_CullDistance"):
        return None
    if base_type == "float":
        return f"{builtin}[0]"
    width = _VECTOR_WIDTH.fullmatch(str(base_type))
    if width is None:
        return None
    count = int(width.group(1))
    elements = ", ".join(f"{builtin}[{index}]" for index in range(count))
    return f"vec{count}({elements})"
