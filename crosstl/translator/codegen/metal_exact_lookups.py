"""Exact name tables for the Metal generator, with a refusal for every name outside them.

The generator looked names up as ``mapping.get(name, name)``, which writes a name the table does not know into the
output as it stands. For semantics that is silent: the Metal compiler is a Clang, and Clang ignores an attribute it
does not know with only a warning (``-Wunknown-attributes``), so ``float h(float a) [[bogus]]`` compiles and the
declaration loses whatever binding the source semantic stood for. Each lookup is now exact:

- A semantic ``semantic_map`` does not rename is written as a Metal attribute only when it already is one: an
  argument-free attribute of :data:`METAL_ATTRIBUTE_NAMES` or one of the parameterized forms of
  :data:`_METAL_ATTRIBUTE_FORMS` (Metal Shading Language Specification 3.2, sections 5.2 and 5.8). Any other
  semantic is refused instead of being written as an attribute Metal ignores.
- The key :func:`metal_semantic_key` classifies a semantic by. It is the ``semantic_map`` spelling for a semantic
  the table maps and the semantic's own text otherwise; callers compare it with Metal attribute names, and it
  never reaches the output (``map_semantic`` writes semantics through :func:`metal_unmapped_semantic_attribute`).
- A SPIR-V ``BuiltIn`` decoration names a CrossGL built-in variable through :data:`SPIRV_BUILTIN_SEMANTICS`
  (SPIR-V 1.6 section 3.21); a decoration outside it is refused rather than written as a Metal attribute.
- An identifier no local rename covers is written as itself only when Metal can spell it unchanged
  (:func:`metal_unrenamed_identifier`).

Every refusal goes through the generator's ``refuse_lowering``, so the program is reported with
:class:`~.metal_refusals.MetalLoweringRefused` and never returned.
"""

from __future__ import annotations

import re

from .metal_keyword_identifiers import METAL_KEYWORD_IDENTIFIERS

__all__ = [
    "METAL_ATTRIBUTE_NAMES",
    "METAL_MATRIX_ORDER_ATTRIBUTES",
    "SPIRV_BUILTIN_SEMANTICS",
    "is_metal_attribute_spelling",
    "metal_semantic_index",
    "metal_semantic_key",
    "metal_spirv_builtin_semantic",
    "metal_unmapped_semantic_attribute",
    "metal_unrenamed_identifier",
]

# Argument-free Metal attributes a CrossGL semantic may spell directly (MSL 3.2 section 5.2 tables 5.2 to 5.8 for
# stage inputs and outputs, 5.2.3.7 for intersection functions, 5.1 for function attributes, 5.4 for
# interpolation qualifiers and 4.4 for the imageblock address space).
METAL_ATTRIBUTE_NAMES = frozenset(
    {
        # Vertex and post-tessellation vertex inputs (tables 5.2 and 5.3).
        "amplification_count",
        "amplification_id",
        "base_instance",
        "base_vertex",
        "instance_id",
        "patch_id",
        "position_in_patch",
        "vertex_id",
        # Vertex outputs (table 5.4).
        "clip_distance",
        "invariant",
        "point_size",
        "position",
        "render_target_array_index",
        "shared",
        "viewport_array_index",
        # Fragment inputs and outputs (tables 5.5 and 5.6).
        "barycentric_coord",
        "front_facing",
        "point_coord",
        "primitive_id",
        "sample_id",
        "sample_mask",
        "stencil",
        # Kernel, object and mesh inputs (tables 5.8 to 5.10).
        "dispatch_quadgroups_per_threadgroup",
        "dispatch_simdgroups_per_threadgroup",
        "dispatch_threads_per_threadgroup",
        "grid_origin",
        "grid_size",
        "payload",
        "primitive_culled",
        "quadgroup_index_in_threadgroup",
        "quadgroups_per_threadgroup",
        "simdgroup_index_in_threadgroup",
        "simdgroups_per_threadgroup",
        "thread_execution_width",
        "thread_index_in_quadgroup",
        "thread_index_in_simdgroup",
        "thread_index_in_threadgroup",
        "thread_position_in_grid",
        "thread_position_in_threadgroup",
        "threadgroup_position_in_grid",
        "threadgroups_per_grid",
        "threads_per_grid",
        "threads_per_simdgroup",
        "threads_per_threadgroup",
        # Intersection function inputs and results (section 5.2.3.7).
        "accept_intersection",
        "continue_search",
        "curve_parameter",
        "direction",
        "distance",
        "geometry_id",
        "geometry_intersection_function_table_offset",
        "instance_intersection_function_table_offset",
        "max_distance",
        "min_distance",
        "object_to_world_transform",
        "opaque",
        "origin",
        "primitive_data",
        "user_instance_id",
        "world_space_direction",
        "world_space_origin",
        "world_to_object_transform",
        # Function attributes, interpolation qualifiers and the imageblock address space.
        "center_no_perspective",
        "center_perspective",
        "centroid_no_perspective",
        "centroid_perspective",
        "early_fragment_tests",
        "flat",
        "imageblock_data",
        "sample_no_perspective",
        "sample_perspective",
        "stage_in",
        "threadgroup_imageblock",
        "visible",
    }
)

# Matrix order qualifiers (GLSL 4.60 section 4.4.5; HLSL row_major and column_major): layout, never a semantic.
METAL_MATRIX_ORDER_ATTRIBUTES = frozenset({"row_major", "column_major"})

_INDEX = r"(?:0|[1-9][0-9]*)"
_NAME = r"[A-Za-z_][A-Za-z0-9_]*"

# Parameterized attributes (MSL 3.2 sections 5.2.1 to 5.2.3 and 5.8): stage-input and colour locations, user
# varyings, depth qualifiers, resource and raster-order-group indices, and function constants.
_METAL_ATTRIBUTE_FORMS = re.compile(
    rf"attribute\({_INDEX}\)"
    rf"|color\({_INDEX}\)(?:,\s*index\({_INDEX}\))?"
    rf"|user\({_NAME}\)"
    r"|depth\((?:any|greater|less)\)"
    rf"|(?:buffer|texture|sampler|threadgroup|id|raster_order_group)\({_INDEX}\)"
    rf"|function_constant\({_NAME}\)"
    rf"|imageblock_data\({_NAME}\)"
    r"|sample_mask,\s*post_depth_coverage"
)

_COLOR_INDEX = re.compile(rf"color\((?P<index>{_INDEX})\)")
_ATTRIBUTE_INDEX = re.compile(rf"attribute\((?P<index>{_INDEX})\)")
_SEMANTIC_INDEX_FORMS = {"color": _COLOR_INDEX, "attribute": _ATTRIBUTE_INDEX}

_IDENTIFIER = re.compile(rf"{_NAME}(?:::{_NAME})*")

# Keywords an expression may name unchanged: the values ``nullptr``, ``true``, ``false`` and ``this``, and the
# scalar types ``half`` and ``bfloat``, which a type operand (a template or ``sizeof`` argument) spells.
_METAL_VALUE_KEYWORDS = frozenset(
    {"nullptr", "true", "false", "this", "half", "bfloat"}
)

# SPIR-V 1.6 section 3.21: the BuiltIn decorations a vertex or fragment interface variable can carry, as the
# CrossGL built-in variable each one is.
SPIRV_BUILTIN_SEMANTICS = {
    "BaryCoordKHR": "gl_BaryCoordEXT",
    "BaryCoordNoPerspKHR": "gl_BaryCoordNoPerspEXT",
    "BaseInstance": "gl_BaseInstance",
    "BaseVertex": "gl_BaseVertex",
    "ClipDistance": "gl_ClipDistance",
    "CullDistance": "gl_CullDistance",
    "DrawIndex": "gl_DrawID",
    "FragCoord": "gl_FragCoord",
    "FragDepth": "gl_FragDepth",
    "FragStencilRefEXT": "gl_FragStencilRefEXT",
    "FrontFacing": "gl_FrontFacing",
    "HelperInvocation": "gl_HelperInvocation",
    "InstanceIndex": "gl_InstanceIndex",
    "Layer": "gl_Layer",
    "PointCoord": "gl_PointCoord",
    "PointSize": "gl_PointSize",
    "Position": "gl_Position",
    "PrimitiveId": "gl_PrimitiveID",
    "SampleId": "gl_SampleID",
    "SampleMask": "gl_SampleMask",
    "SamplePosition": "gl_SamplePosition",
    "VertexIndex": "gl_VertexIndex",
    "ViewIndex": "gl_ViewIndex",
    "ViewportIndex": "gl_ViewportIndex",
}


def _unbracketed(text):
    if text.startswith("[[") and text.endswith("]]"):
        return text[2:-2]
    return text


def is_metal_attribute_spelling(text):
    """Whether ``text``, with or without its ``[[ ]]``, is a Metal attribute a declaration may carry."""
    inner = _unbracketed(str(text)).strip()
    return inner in METAL_ATTRIBUTE_NAMES or _METAL_ATTRIBUTE_FORMS.fullmatch(inner)


def metal_semantic_key(generator, semantic):
    """The key ``semantic`` is classified by: its ``semantic_map`` spelling, or its own text, without ``[[ ]]``.

    The key is compared with Metal attribute names and never written: an unmapped semantic whose text is not a
    Metal attribute classifies as none of them, and :func:`metal_unmapped_semantic_attribute` refuses it where a
    semantic is written.
    """
    if semantic is None:
        return None
    text = str(semantic)
    mapped = generator.semantic_map.get(text)
    return _unbracketed(text if mapped is None else mapped)


def metal_semantic_index(generator, semantic, form):
    """The index of a ``color(N)`` or ``attribute(N)`` semantic (``form`` names which), or None for any other."""
    key = metal_semantic_key(generator, semantic)
    if key is None:
        return None
    match = _SEMANTIC_INDEX_FORMS[form].fullmatch(key)
    return int(match.group("index")) if match else None


def metal_unmapped_semantic_attribute(generator, semantic):
    """A semantic no table renames, written as a Metal attribute: the semantic itself when it is one.

    Anything else is refused and ``None`` returned: Metal would ignore it as an unknown attribute, so the declaration
    would silently lose the meaning the semantic gives it in the source.
    """
    text = str(semantic)
    if is_metal_attribute_spelling(text):
        return text
    generator.refuse_lowering(
        f"semantic '{text}' names no Metal attribute; the Metal compiler ignores an unknown attribute, so the "
        "declaration would lose the binding the semantic gives it"
    )
    return None


def metal_spirv_builtin_semantic(generator, builtin):
    """The CrossGL built-in variable a SPIR-V ``BuiltIn`` decoration names, or a refusal (None) for another."""
    if builtin is None:
        return None
    text = str(builtin)
    semantic = SPIRV_BUILTIN_SEMANTICS.get(text)
    if semantic is not None:
        return semantic
    generator.refuse_lowering(
        f"SPIR-V BuiltIn {text} has no Metal stage input or output (SPIR-V 1.6 section 3.21)"
    )
    return None


def metal_unrenamed_identifier(generator, name):
    """``name`` written as itself, when no local rename covers it: refused unless Metal can spell it unchanged.

    A plain or ``::``-qualified identifier passes unless it is a Metal or C++ keyword
    (:data:`~.metal_keyword_identifiers.METAL_KEYWORD_IDENTIFIERS`) other than the values and scalar types an
    expression spells; a keyword-spelled program name would change meaning or fail to parse. The keyword renaming
    passes and ``collect_metal_local_identifier_remaps`` rename every parameter and local that needs it, so a name
    refused here is one no pass renamed.
    """
    if not isinstance(name, str):
        return name
    if name in _METAL_VALUE_KEYWORDS:
        return name
    if _IDENTIFIER.fullmatch(name) and name not in METAL_KEYWORD_IDENTIFIERS:
        return name
    return generator.refuse_lowering(
        f"identifier '{name}' has no Metal spelling: it is not an identifier Metal leaves to programs",
        name,
    )
