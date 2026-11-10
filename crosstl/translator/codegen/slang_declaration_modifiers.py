"""Slang spellings of the declaration modifiers CrossGL keeps as attributes.

CrossGL records a declaration's storage and precision modifiers as attributes, ``@ row_major``, ``@ precise`` or
``@ packoffset(c12)``, next to the attributes that name a semantic. Slang spells them like HLSL:

- ``row_major`` and ``column_major`` prefix a matrix declaration (HLSL "Variable Syntax", storage class modifiers).
  CrossGL keeps the HLSL meaning, as the DirectX generator does: an HLSL ``float4x4`` with ``mul(m, v)`` is the
  CrossGL ``mat4`` with ``m * v``, and both qualifiers name the order of the same mathematical rows and columns.
  slangc decorates the member ``RowMajor`` or ``ColMajor`` and refuses the qualifier on a type that is not a matrix
  (E39026, "matrix layout modifier on non-matrix type"). GLSL lets the qualifier reach the matrices inside a struct
  member (GLSL 4.60 section 4.4.5); Slang has no spelling for that, so such a member is refused.
- ``precise`` prefixes a variable or a struct member (HLSL ``precise``, SPIR-V ``NoContraction``).
- ``packoffset(c<N>[.<component>])`` follows a constant-buffer member, where slangc places it as DXC does.

The other attributes in :data:`SLANG_DECLARATION_METADATA_ATTRIBUTES` never name a semantic:

- GLSL precision qualifiers set a minimum precision (GLSL ES 3.00 section 4.5.2) that Slang's 32-bit types meet.
- GLSL ``invariant`` has no HLSL or Slang spelling; the DirectX generator drops it in the same way.
- ``buffer_layout`` is the layout contract the buffer planners read.
- ``hlsl_program_constant`` and ``source_resource`` record where a declaration came from.
"""

from __future__ import annotations

import re

from .array_utils import split_array_type_suffix

__all__ = [
    "MATRIX_ORIENTATIONS",
    "SLANG_DECLARATION_METADATA_ATTRIBUTES",
    "is_slang_declaration_metadata_attribute",
    "slang_matrix_orientation",
    "slang_member_modifier_prefix",
    "slang_packoffset_suffix",
    "slang_precise_prefix",
]

MATRIX_ORIENTATIONS = ("row_major", "column_major")

SLANG_DECLARATION_METADATA_ATTRIBUTES = frozenset(
    {
        *MATRIX_ORIENTATIONS,
        "precise",
        "packoffset",
        "invariant",
        "highp",
        "mediump",
        "lowp",
        "buffer_layout",
        "hlsl_program_constant",
        "source_resource",
    }
)

_SLANG_MATRIX = re.compile(
    r"(?:float|double|half|int|uint|bool|min16float|min10float|min16int|min16uint|"
    r"float16_t|float64_t|int16_t|uint16_t|int64_t|uint64_t)[1-4]x[1-4]"
)
_MATRIX_TEMPLATE = re.compile(r"matrix\s*<.*>")


def _name(value) -> str:
    return str(getattr(value, "name", value) or "").lower()


def _modifier_names(node) -> list[str]:
    """Qualifiers, then attributes, of ``node`` in declaration order, lower-cased."""
    names = [_name(qualifier) for qualifier in getattr(node, "qualifiers", None) or ()]
    names.extend(_name(attr) for attr in getattr(node, "attributes", None) or ())
    return names


def is_slang_declaration_metadata_attribute(attr) -> bool:
    """Whether ``attr`` is a declaration modifier or record rather than a semantic."""
    return _name(attr) in SLANG_DECLARATION_METADATA_ATTRIBUTES


def slang_precise_prefix(node) -> str:
    """``precise `` for a declaration whose arithmetic must not be contracted or reordered."""
    return "precise " if "precise" in _modifier_names(node) else ""


def slang_matrix_orientation(node):
    """The ``row_major`` or ``column_major`` qualifier ``node`` declares, or ``None``.

    The first one written wins, as in the DirectX generator.
    """
    for name in _modifier_names(node):
        if name in MATRIX_ORIENTATIONS:
            return name
    return None


def _is_slang_matrix_type(type_text: str) -> bool:
    text = str(type_text or "").strip()
    return bool(_SLANG_MATRIX.fullmatch(text) or _MATRIX_TEMPLATE.fullmatch(text))


def _struct_holds_matrix(codegen, struct_name, seen) -> bool:
    struct = codegen.user_structs_by_name.get(struct_name)
    if struct is None or struct_name in seen:
        return False
    seen = seen | {struct_name}
    for member in getattr(struct, "members", None) or ():
        member_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if member_type is None:
            continue
        base, _suffix = split_array_type_suffix(
            str(codegen.convert_type(codegen.convert_type_node_to_string(member_type)))
        )
        if _is_slang_matrix_type(base) or _struct_holds_matrix(codegen, base, seen):
            return True
    return False


def slang_member_modifier_prefix(codegen, node, slang_type, context) -> str:
    """``precise`` and matrix-orientation qualifiers of a struct, block or module-scope declaration.

    ``slang_type`` is the Slang spelling of the declared type, with any array suffix. An orientation on a scalar
    or vector has no effect (GLSL 4.60 section 4.4.5) and is dropped; on a struct that holds a matrix it is
    refused, because Slang can only qualify the matrix declarations themselves.
    """
    prefix = slang_precise_prefix(node)
    orientation = slang_matrix_orientation(node)
    if orientation is None:
        return prefix
    base, _suffix = split_array_type_suffix(str(slang_type or ""))
    base = base.strip()
    if _is_slang_matrix_type(base):
        return f"{prefix}{orientation} "
    if _struct_holds_matrix(codegen, base, frozenset()):
        codegen.refuse_lowering(
            f"Slang cannot apply {orientation} to the matrices inside struct '{base}' of {context}; "
            "only a matrix declaration takes a matrix layout qualifier"
        )
    return prefix


def slang_packoffset_suffix(codegen, member) -> str:
    """`` : packoffset(c<N>)`` for a constant-buffer member that names its register."""
    for attr in getattr(member, "attributes", None) or ():
        if _name(attr) != "packoffset":
            continue
        arguments = getattr(attr, "arguments", None) or getattr(attr, "args", None)
        if not arguments:
            continue
        return f" : packoffset({codegen.attribute_value_to_string(arguments[0])})"
    return ""
