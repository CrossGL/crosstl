"""Texture reads that report tile residency: CrossGL's ``textureResidency`` and ``textureResident``.

A sparse (tiled, partially resident) texture may leave some of its tiles unmapped. A read can report whether every
texel it touched was resident, so that a shader can fall back to a coarser level. The targets express this per read:

* GLSL: the GL_ARB_sparse_texture2 functions (``sparseTextureARB``, ``sparseTexelFetchARB``,
  ``sparseTextureGatherARB``, ``sparseImageLoadARB`` and their kin) return an ``int`` residency code and write the
  texel to an ``out`` operand; ``sparseTexelsResidentARB(code)`` tests the code. GL_ARB_sparse_texture_clamp adds
  the clamped forms. Both are desktop OpenGL extensions; GLSL ES has neither.
* MSL: the ``sparse_sample``, ``sparse_sample_compare``, ``sparse_read``, ``sparse_gather`` and
  ``sparse_gather_compare`` texture members return ``sparse_color<T>``, whose ``value()`` is the texel and whose
  ``resident()`` is the residency (Metal Shading Language specification, section 6.12, MSL 2.3 and later).
* HLSL: the trailing ``out uint Status`` operand of the Shader Model 5 tiled-resource overloads, tested with
  ``CheckAccessFullyMapped``.
* SPIR-V: the ``OpImageSparse*`` instructions return a residency code and the texel, tested with
  ``OpImageSparseTexelsResident`` (``SparseResidency`` capability).

CrossGL wraps an ordinary texture read::

    int code = textureResidency(texture(t, s, P), texel);   // texel = texture(t, s, P); code = its residency
    bool resident = textureResident(code);

The first operand is a direct call to a texture read and the second the lvalue that receives its texel. The read
stays an ordinary CrossGL texture operation, so sampler pairing, shadow-compare legalization and implicit-LOD
rewriting treat it as any other read. An emitter lowers the read through its ordinary texture path and then turns
the call it produced into the target's residency-reporting form. The code is opaque: only ``textureResident``
interprets it. A read the target has no residency-reporting form of is refused with
:class:`TextureResidencyUnsupportedError` rather than emitted without its residency.
"""

from __future__ import annotations

import re

from ..ast import FunctionCallNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .texture_lod_clamp import join_call_text, split_call_text

__all__ = [
    "GLSL_RESIDENCY_EXTENSION",
    "GLSL_SPARSE_FUNCTIONS",
    "TEXTURE_RESIDENCY_OPERATION",
    "TEXTURE_RESIDENCY_READS",
    "TEXTURE_RESIDENT_OPERATION",
    "TextureResidencyUnsupportedError",
    "emit_metal_texture_residency_helpers",
    "glsl_texture_residency_call",
    "glsl_texture_resident_call",
    "is_texture_residency_operation",
    "metal_texture_residency_call",
    "metal_texture_resident_call",
    "split_texture_residency_call",
]

TEXTURE_RESIDENCY_OPERATION = "textureResidency"
TEXTURE_RESIDENT_OPERATION = "textureResident"

# The CrossGL texture reads a residency query may wrap: those GL_ARB_sparse_texture2, MSL, HLSL and SPIR-V all have
# a residency-reporting form of. Projected reads, size and level queries and image stores have none.
TEXTURE_RESIDENCY_READS = frozenset(
    {
        "texture",
        "textureOffset",
        "textureLod",
        "textureLodOffset",
        "textureGrad",
        "textureGradOffset",
        "textureClamp",
        "textureOffsetClamp",
        "textureGradClamp",
        "textureGradOffsetClamp",
        "textureCompare",
        "textureCompareOffset",
        "textureCompareLod",
        "textureCompareLodOffset",
        "textureCompareGrad",
        "textureCompareGradOffset",
        "textureCompareClamp",
        "textureCompareOffsetClamp",
        "textureCompareGradClamp",
        "textureCompareGradOffsetClamp",
        "texelFetch",
        "texelFetchOffset",
        "textureGather",
        "textureGatherOffset",
        "textureGatherOffsets",
        "textureGatherCompare",
        "textureGatherCompareOffset",
        "textureGatherCompareOffsets",
        "imageLoad",
    }
)

GLSL_RESIDENCY_EXTENSION = "GL_ARB_sparse_texture2"
# The GLSL call a read renders to, its GL_ARB_sparse_texture2 (or _clamp) form, and the operand count before the
# ``out`` texel when an optional trailing operand (a bias or a gather component) follows it; ``None`` when the
# texel is always the last operand.
GLSL_SPARSE_FUNCTIONS = {
    "texture": ("sparseTextureARB", 2),
    "textureOffset": ("sparseTextureOffsetARB", 3),
    "textureLod": ("sparseTextureLodARB", None),
    "textureLodOffset": ("sparseTextureLodOffsetARB", None),
    "textureGrad": ("sparseTextureGradARB", None),
    "textureGradOffset": ("sparseTextureGradOffsetARB", None),
    "texelFetch": ("sparseTexelFetchARB", None),
    "texelFetchOffset": ("sparseTexelFetchOffsetARB", None),
    "textureGather": ("sparseTextureGatherARB", 2),
    "textureGatherOffset": ("sparseTextureGatherOffsetARB", 3),
    "textureGatherOffsets": ("sparseTextureGatherOffsetsARB", 3),
    "imageLoad": ("sparseImageLoadARB", None),
    "textureClampARB": ("sparseTextureClampARB", 3),
    "textureOffsetClampARB": ("sparseTextureOffsetClampARB", 4),
    "textureGradClampARB": ("sparseTextureGradClampARB", None),
    "textureGradOffsetClampARB": ("sparseTextureGradOffsetClampARB", None),
}

# MSL texture members with a sparse form returning sparse_color<T> (MSL specification, section 6.12).
_METAL_SPARSE_METHODS = frozenset(
    {"sample", "sample_compare", "read", "gather", "gather_compare"}
)
_SIMPLE_OPERAND = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_METAL_HELPER = "crosstl_texture_residency"
# One overload per address space the texel may live in; the residency code is 0 when every texel was resident.
_METAL_HELPER_TEXT = "".join(
    f"template <typename T, typename U>\n"
    f"inline int {_METAL_HELPER}(sparse_color<T> result, {space} U& texel) {{\n"
    f"  texel = U(result.value());\n"
    f"  return result.resident() ? 0 : 1;\n"
    f"}}\n"
    for space in ("thread", "threadgroup", "device")
)


def is_texture_residency_operation(operation) -> bool:
    """Whether ``operation`` is ``textureResidency`` or ``textureResident``."""
    return operation in (TEXTURE_RESIDENCY_OPERATION, TEXTURE_RESIDENT_OPERATION)


class TextureResidencyUnsupportedError(TranslationDiagnostic):
    """A texture residency query the target cannot express for the read it wraps."""

    def __init__(self, target, operation, reason, *, source_location=None):
        self.operation = operation
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.texture-residency",)
        super().__init__(
            f"texture operation '{operation}' reports tile residency, which {target} cannot carry here: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-texture-residency-unsupported",
        )


def _call_name(node):
    function = getattr(node, "function", None)
    if isinstance(function, str):
        return function
    return getattr(function, "name", None) if function is not None else None


def split_texture_residency_call(target, args, call_node=None):
    """``(read, texel)`` of ``textureResidency(read, texel)``; refuses a call of any other shape."""
    args = list(args or [])
    location = getattr(call_node, "source_location", None)
    if len(args) != 2:
        raise TextureResidencyUnsupportedError(
            target,
            TEXTURE_RESIDENCY_OPERATION,
            f"it takes a texture read and the texel it writes, got {len(args)} operand(s)",
            source_location=location,
        )
    read, texel = args
    read_name = _call_name(read) if isinstance(read, FunctionCallNode) else None
    if read_name not in TEXTURE_RESIDENCY_READS:
        raise TextureResidencyUnsupportedError(
            target,
            TEXTURE_RESIDENCY_OPERATION,
            "its first operand must be a direct call to a texture read with a residency-reporting form "
            f"({', '.join(sorted(TEXTURE_RESIDENCY_READS))}), got "
            f"{read_name or type(read).__name__}",
            source_location=location,
        )
    return read, texel


def _resident_code(target, args, call_node):
    args = list(args or [])
    if len(args) != 1:
        raise TextureResidencyUnsupportedError(
            target,
            TEXTURE_RESIDENT_OPERATION,
            f"it takes one residency code, got {len(args)} operand(s)",
            source_location=getattr(call_node, "source_location", None),
        )
    return args[0]


def _has_pending_refusal(codegen):
    refusals = getattr(codegen, "refusals", None)
    return bool(getattr(refusals, "reasons", None))


def _render_read(codegen, read):
    """The read through the emitter's ordinary texture path, outside any expected-type context of the query."""
    saved = codegen.__dict__.get("current_expression_expected_type")
    if "current_expression_expected_type" in codegen.__dict__:
        codegen.current_expression_expected_type = None
    try:
        return codegen.generate_expression(read)
    finally:
        if "current_expression_expected_type" in codegen.__dict__:
            codegen.current_expression_expected_type = saved


def _refuse(codegen, target, base_text, reason, call_node):
    """Raise, unless the read was itself refused: its pending refusal already names the actual cause."""
    if _has_pending_refusal(codegen):
        return base_text
    raise TextureResidencyUnsupportedError(
        target,
        TEXTURE_RESIDENCY_OPERATION,
        reason,
        source_location=getattr(call_node, "source_location", None),
    )


def metal_texture_residency_call(codegen, args, call_node=None):
    """MSL for ``textureResidency``: the read's ``sparse_*`` member, unpacked by a helper into texel and code."""
    read, texel = split_texture_residency_call("metal", args, call_node)
    base_text = _render_read(codegen, read)
    parts = split_call_text(base_text)
    callee = parts[0] if parts else ""
    receiver, _, method = callee.rpartition(".")
    if not receiver or method not in _METAL_SPARSE_METHODS:
        return _refuse(
            codegen,
            "metal",
            base_text,
            f"the read lowered to '{base_text}', not to a texture member with a sparse form "
            f"({', '.join(sorted(_METAL_SPARSE_METHODS))}; MSL specification, section 6.12)",
            call_node,
        )
    sparse_text = join_call_text(f"{receiver}.sparse_{method}", parts[1])
    codegen.__dict__.setdefault("metal_texture_residency_helpers", {})[
        _METAL_HELPER
    ] = _METAL_HELPER_TEXT
    return f"{_METAL_HELPER}({sparse_text}, {codegen.generate_expression(texel)})"


def metal_texture_resident_call(codegen, args, call_node=None):
    """MSL for ``textureResident``: the helper's code is 0 when every texel was resident."""
    code = codegen.generate_expression(_resident_code("metal", args, call_node))
    if not _SIMPLE_OPERAND.fullmatch(code):
        code = f"({code})"
    return f"({code} == 0)"


def emit_metal_texture_residency_helpers(codegen):
    """The residency helpers the program calls."""
    return "".join(codegen.__dict__.get("metal_texture_residency_helpers", {}).values())


def _is_cube_array_shadow_read(codegen, read, glsl_arguments):
    """Whether a GLSL ``texture`` call reads a ``samplerCubeArrayShadow``, whose third operand is the reference."""
    if glsl_arguments and str(glsl_arguments[0]).startswith("samplerCubeArrayShadow"):
        return True
    resource_type = None
    lookup = getattr(codegen, "texture_argument_resource_type", None)
    read_args = list(getattr(read, "arguments", None) or [])
    if lookup is not None and read_args:
        resource_type = lookup(read_args[0])
    return str(resource_type or "").split("[", 1)[0] == "samplerCubeArrayShadow"


def glsl_texture_residency_call(codegen, args, call_node=None):
    """GLSL for ``textureResidency``: the read's GL_ARB_sparse_texture2 function with the texel ``out`` operand."""
    read, texel = split_texture_residency_call("opengl", args, call_node)
    base_text = _render_read(codegen, read)
    parts = split_call_text(base_text)
    entry = GLSL_SPARSE_FUNCTIONS.get(parts[0]) if parts else None
    if entry is None:
        return _refuse(
            codegen,
            "opengl",
            base_text,
            f"the read lowered to '{base_text}', which GL_ARB_sparse_texture2 has no sparse form of",
            call_node,
        )
    function, operands_before_trailer = entry
    arguments = list(parts[1])
    texel_text = codegen.generate_expression(texel)
    compare_read = "Compare" in str(_call_name(read))
    has_trailer = (
        operands_before_trailer is not None
        and not compare_read
        and len(arguments) == operands_before_trailer + 1
        and not (
            parts[0] == "texture"
            and _is_cube_array_shadow_read(codegen, read, arguments)
        )
    )
    if has_trailer:
        # A bias or a gather component follows the texel (GL_ARB_sparse_texture2, section 8.9).
        arguments.insert(len(arguments) - 1, texel_text)
    else:
        arguments.append(texel_text)
    return join_call_text(function, arguments)


def glsl_texture_resident_call(codegen, args, call_node=None):
    """GLSL for ``textureResident``."""
    code = _resident_code("opengl", args, call_node)
    return f"sparseTexelsResidentARB({codegen.generate_expression(code)})"
