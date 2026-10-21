"""GLSL std140 uniform-block layouts for backends that must reproduce them.

A GLSL uniform block is laid out with the std140 rules (GLSL 4.60 section
7.6.2.2), which neither HLSL constant-buffer packing nor the natural layout of
Metal follow:

* a scalar is aligned to its size, a two-component vector to twice that and a
  three- or four-component vector to four times that; a ``vec3`` takes 12
  bytes, so a following ``float`` packs into its last word (rules 1-3);
* an array element, and every column (or row, for ``row_major``) of a matrix,
  is aligned to its vector's alignment rounded up to 16 bytes, and an array or
  matrix takes ``count * stride`` bytes: the member after ``float a[2]`` starts
  at byte 32, the one after a ``mat2`` at byte 32 (rules 4-8);
* a struct is aligned to its widest member rounded up to 16 bytes and its size
  is rounded up to that alignment (rules 9-10);
* ``layout(offset = N)`` places a member at byte ``N``, which must be a
  multiple of its alignment and must not reach back into the previous member;
  ``layout(align = N)`` raises a member's alignment to ``N`` (section 4.4.5).

CrossGL keeps a uniform block imported from GLSL as a ``cbuffer`` marked
``@ buffer_layout(std140)``; the shapes are the :class:`HlslLayoutType` values
of :mod:`hlsl_buffer_layout`, so a backend may place a buffer with either
rules and compare them.
"""

import re

from ..ast import ArrayType, MatrixType, NamedType, PrimitiveType, VectorType
from .hlsl_buffer_layout import (
    BUFFER_LAYOUT_ATTRIBUTE,
    HLSL_BUFFER_LAYOUTS,
    HlslLayoutError,
    HlslMemberPlacement,
    align_to,
    array_type,
    cbuffer_array_stride,
    cbuffer_struct_placements,
    oriented_matrix_type,
    scalar_type,
    struct_placements,
    struct_type,
    vector_type,
)

STD140_LAYOUT = "std140"
STD430_LAYOUT = "std430"
SCALAR_BLOCK_LAYOUT = "scalar"
# Layouts a ``@ buffer_layout`` attribute may name.
DECLARED_BUFFER_LAYOUTS = HLSL_BUFFER_LAYOUTS + (
    STD140_LAYOUT,
    STD430_LAYOUT,
    SCALAR_BLOCK_LAYOUT,
    "metal",
)
VEC4_ALIGNMENT = 16

CROSSGL_SCALARS = {
    "float": "float",
    "int": "int",
    "uint": "uint",
    "bool": "bool",
    "double": "double",
}
CROSSGL_VECTOR_PREFIXES = {
    "vec": "float",
    "ivec": "int",
    "uvec": "uint",
    "bvec": "bool",
    "dvec": "double",
}
CROSSGL_MATRIX_PREFIXES = {"mat": "float", "dmat": "double"}


def declared_buffer_layout(attributes):
    """The layout a ``@ buffer_layout(...)`` attribute names (any rule set), or ``None``."""
    for attribute in attributes or []:
        if str(getattr(attribute, "name", "")).lower() != BUFFER_LAYOUT_ATTRIBUTE:
            continue
        arguments = list(
            getattr(attribute, "arguments", None)
            or getattr(attribute, "args", None)
            or []
        )
        if not arguments:
            continue
        argument = arguments[0]
        value = getattr(argument, "name", getattr(argument, "value", argument))
        value = str(value).strip().strip('"').lower()
        if value in DECLARED_BUFFER_LAYOUTS:
            return value
    return None


def member_explicit_alignment(member):
    """The byte alignment a member's ``align`` qualifier names, or ``None``."""
    for attribute in getattr(member, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() != "align":
            continue
        arguments = list(
            getattr(attribute, "arguments", None)
            or getattr(attribute, "args", None)
            or []
        )
        if len(arguments) != 1:
            raise HlslLayoutError(
                f"member '{getattr(member, 'name', '')}' has a malformed align qualifier"
            )
        argument = arguments[0]
        value = getattr(argument, "value", getattr(argument, "name", argument))
        try:
            alignment = int(str(value).strip(), 0)
        except ValueError:
            raise HlslLayoutError(
                f"member '{getattr(member, 'name', '')}' has a non-constant align '{value}'"
            ) from None
        if alignment <= 0 or alignment & (alignment - 1):
            raise HlslLayoutError(
                f"member '{getattr(member, 'name', '')}' has an align of {alignment}, "
                "which is not a power of two"
            )
        return alignment
    return None


# std140 rules ----------------------------------------------------------------
def std140_vector_alignment(scalar_size, components):
    if components == 1:
        return scalar_size
    if components == 2:
        return 2 * scalar_size
    return 4 * scalar_size


def std140_matrix_vector_stride(layout_type):
    """Distance between the stored columns (rows, for ``row_major``) of a matrix (rules 5 and 7)."""
    return align_to(
        std140_vector_alignment(layout_type.scalar_size, layout_type.components),
        VEC4_ALIGNMENT,
    )


def std140_alignment(layout_type):
    """The std140 base alignment of a value."""
    kind = layout_type.kind
    if kind == "scalar":
        return layout_type.scalar_size
    if kind == "vector":
        return std140_vector_alignment(layout_type.scalar_size, layout_type.components)
    if kind == "matrix":
        return std140_matrix_vector_stride(layout_type)
    if kind == "array":
        return align_to(std140_alignment(layout_type.element), VEC4_ALIGNMENT)
    if kind == "struct":
        alignment = VEC4_ALIGNMENT
        for (_name, member), explicit in zip(
            layout_type.members, member_alignments(layout_type)
        ):
            alignment = max(alignment, std140_alignment(member), explicit or 0)
        return align_to(alignment, VEC4_ALIGNMENT)
    raise HlslLayoutError(f"unsupported std140 value kind '{kind}'")


def std140_array_stride(layout_type):
    """The element stride of a std140 array (rules 4, 6, 8 and 10)."""
    element = layout_type.element
    return align_to(std140_size(element), std140_alignment(layout_type))


def std140_size(layout_type):
    """The bytes a std140 value takes, including the padding no other member may use."""
    kind = layout_type.kind
    if kind in ("scalar", "vector"):
        return layout_type.scalar_size * layout_type.components
    if kind == "matrix":
        return std140_matrix_vector_stride(layout_type) * layout_type.registers
    if kind == "array":
        return std140_array_stride(layout_type) * layout_type.count
    if kind == "struct":
        _placements, size = std140_struct_placements(layout_type)
        return size
    raise HlslLayoutError(f"unsupported std140 value kind '{kind}'")


def member_alignments(layout_type):
    """The ``align`` qualifier of every member of a struct, ``None`` where there is none."""
    return layout_type.alignments or (None,) * len(layout_type.members)


def std140_struct_placements(layout_type, root=False):
    """Member placements and size of a struct, or of a uniform block when ``root``.

    The size of a nested struct is rounded up to its alignment; a block's size
    ends after its last member.
    """
    offsets = layout_type.offsets or (None,) * len(layout_type.members)
    alignments = member_alignments(layout_type)
    placements = []
    cursor = 0
    for (name, member_type), offset, explicit_alignment in zip(
        layout_type.members, offsets, alignments
    ):
        alignment = std140_alignment(member_type)
        if offset is not None:
            if offset % alignment:
                raise HlslLayoutError(
                    f"offset {offset} of '{name}' is not a multiple of its std140 "
                    f"alignment {alignment}"
                )
            if offset < cursor:
                raise HlslLayoutError(
                    f"offset {offset} of '{name}' lies within or before the previous "
                    f"member of '{layout_type.struct_name}'"
                )
            cursor = offset
        if explicit_alignment:
            alignment = max(alignment, explicit_alignment)
        start = align_to(cursor, alignment)
        size = std140_size(member_type)
        placements.append(HlslMemberPlacement(name, member_type, start, size))
        cursor = start + size
    if root:
        return placements, cursor
    return placements, align_to(cursor, std140_alignment(layout_type))


def layout_struct_placements(layout_type, layout, root=False):
    """Member placements and size of a struct under ``layout`` (an HLSL layout or std140)."""
    if layout == STD140_LAYOUT:
        return std140_struct_placements(layout_type, root)
    return struct_placements(layout_type, layout)


def layout_array_stride(layout_type, layout):
    if layout == STD140_LAYOUT:
        return std140_array_stride(layout_type)
    return cbuffer_array_stride(layout_type)


# CrossGL types -------------------------------------------------------------
def crossgl_layout_type(
    type_name, struct_members, row_major=False, stack=(), constant_value=None
):
    """The layout shape of a CrossGL type: a type node or a spelling such as ``vec3``, ``mat3x4`` or ``Light[4]``.

    ``struct_members(name)`` returns the ``(member_name, type, row_major,
    offset, alignment)`` tuples of a struct, or ``None`` when ``name`` is not a
    struct. ``row_major`` orients a matrix (or the matrices of an array).
    ``constant_value(expression)`` evaluates an array size that is not a
    literal, returning ``None`` when it is not a constant.
    """
    if not isinstance(type_name, str):
        return crossgl_type_node_layout_type(
            type_name, struct_members, row_major, stack, constant_value
        )
    type_name = re.sub(r"\s+", " ", str(type_name)).strip()
    array_match = re.fullmatch(r"(.+?)\s*\[\s*([^\]]*)\s*\]", type_name)
    if array_match is not None:
        count_text = array_match.group(2).strip()
        if not count_text:
            raise HlslLayoutError("unsized arrays in uniform blocks are not supported")
        try:
            count = int(count_text, 0)
        except ValueError:
            raise HlslLayoutError(
                f"array size '{count_text}' in a uniform block is not a constant"
            ) from None
        element = crossgl_layout_type(
            array_match.group(1), struct_members, row_major, stack, constant_value
        )
        if element.kind == "array":
            raise HlslLayoutError(
                "multidimensional arrays in uniform blocks are not supported"
            )
        return array_type(element, count)
    if type_name in CROSSGL_SCALARS:
        return scalar_type(CROSSGL_SCALARS[type_name])
    vector_match = re.fullmatch(r"(vec|ivec|uvec|bvec|dvec)([234])", type_name)
    if vector_match is not None:
        return vector_type(
            CROSSGL_VECTOR_PREFIXES[vector_match.group(1)], int(vector_match.group(2))
        )
    matrix_match = re.fullmatch(r"(mat|dmat)([234])(?:x([234]))?", type_name)
    if matrix_match is not None:
        columns = int(matrix_match.group(2))
        rows = int(matrix_match.group(3) or columns)
        return oriented_matrix_type(
            CROSSGL_MATRIX_PREFIXES[matrix_match.group(1)], columns, rows, row_major
        )
    members = struct_members(type_name)
    if members is None:
        raise HlslLayoutError(f"type '{type_name}' has no uniform-block layout")
    if type_name in stack:
        raise HlslLayoutError(f"recursive struct '{type_name}' in a uniform block")
    return crossgl_struct_layout_type(
        type_name, members, struct_members, (*stack, type_name), constant_value
    )


def crossgl_type_node_layout_type(
    type_node, struct_members, row_major=False, stack=(), constant_value=None
):
    if isinstance(type_node, PrimitiveType):
        name = str(type_node.name)
        if name not in CROSSGL_SCALARS:
            raise HlslLayoutError(f"type '{name}' has no uniform-block layout")
        return scalar_type(CROSSGL_SCALARS[name])
    if isinstance(type_node, VectorType):
        element = crossgl_type_node_layout_type(type_node.element_type, struct_members)
        return vector_type(element.scalar, int(type_node.size))
    if isinstance(type_node, MatrixType):
        element = crossgl_type_node_layout_type(type_node.element_type, struct_members)
        if element.scalar not in ("float", "double"):
            raise HlslLayoutError(
                f"{element.scalar} matrices have no uniform-block layout"
            )
        return oriented_matrix_type(
            element.scalar, int(type_node.cols), int(type_node.rows), row_major
        )
    if isinstance(type_node, ArrayType):
        size = type_node.size
        if size is None:
            raise HlslLayoutError("unsized arrays in uniform blocks are not supported")
        count = size if isinstance(size, int) else None
        if count is None and constant_value is not None:
            count = constant_value(size)
        if count is None:
            value = getattr(size, "value", None)
            if isinstance(value, int) and not isinstance(value, bool):
                count = value
        if count is None:
            raise HlslLayoutError("array sizes in uniform blocks must be constants")
        element = crossgl_type_node_layout_type(
            type_node.element_type, struct_members, row_major, stack, constant_value
        )
        if element.kind == "array":
            raise HlslLayoutError(
                "multidimensional arrays in uniform blocks are not supported"
            )
        return array_type(element, int(count))
    if isinstance(type_node, NamedType):
        return crossgl_layout_type(
            str(type_node.name), struct_members, row_major, stack, constant_value
        )
    return crossgl_layout_type(
        str(type_node), struct_members, row_major, stack, constant_value
    )


def crossgl_struct_layout_type(
    name, members, struct_members, stack=(), constant_value=None
):
    """The layout shape of a struct or block from its ``(name, type, row_major, offset, alignment)`` members."""
    members = list(members)
    return struct_type(
        name,
        [
            (
                member[0],
                crossgl_layout_type(
                    member[1], struct_members, member[2], stack, constant_value
                ),
            )
            for member in members
        ],
        [member[3] for member in members],
        [member[4] for member in members],
    )


# HLSL constant buffers -------------------------------------------------------
def hlsl_packoffsets_for_std140(layout_type):
    """``packoffset`` byte offsets that put the members of a constant buffer at their std140 offsets.

    Returns ``None`` when HLSL packing already places every member there, and
    raises :class:`HlslLayoutError` when HLSL cannot reproduce the block: a
    member's std140 offset breaks HLSL packing for its type, or a struct or
    array inside a member is laid out differently by the two rule sets
    (``packoffset`` places only the members of the buffer itself).
    """
    std140, _size = std140_struct_placements(layout_type, root=True)
    hlsl, _hlsl_size = cbuffer_struct_placements(
        struct_type(layout_type.struct_name, layout_type.members)
    )
    for placement in std140:
        check_std140_interior_matches_hlsl(placement.name, placement.layout_type)
    if [placement.offset for placement in std140] == [
        placement.offset for placement in hlsl
    ]:
        return None
    explicit = struct_type(
        layout_type.struct_name,
        layout_type.members,
        [placement.offset for placement in std140],
    )
    # Raises when a std140 offset is not a legal HLSL placement of its type.
    cbuffer_struct_placements(explicit)
    return [placement.offset for placement in std140]


def check_std140_interior_matches_hlsl(name, layout_type):
    """Refuse a value whose inner layout (array stride, struct members) differs between std140 and HLSL."""
    kind = layout_type.kind
    if kind == "array":
        std140_stride = std140_array_stride(layout_type)
        hlsl_stride = cbuffer_array_stride(layout_type)
        if std140_stride != hlsl_stride:
            raise HlslLayoutError(
                f"array '{name}' has a std140 stride of {std140_stride} bytes but an "
                f"HLSL constant-buffer stride of {hlsl_stride}"
            )
        check_std140_interior_matches_hlsl(name, layout_type.element)
        return
    if kind == "matrix":
        if std140_matrix_vector_stride(layout_type) != 16 * (
            1 + (layout_type.scalar_size * layout_type.components - 1) // 16
        ):
            raise HlslLayoutError(
                f"matrix '{name}' has a std140 vector stride that HLSL does not use"
            )
        return
    if kind != "struct":
        return
    std140, _size = std140_struct_placements(layout_type)
    hlsl, _hlsl_size = cbuffer_struct_placements(
        struct_type(layout_type.struct_name, layout_type.members)
    )
    std140_offsets = [placement.offset for placement in std140]
    hlsl_offsets = [placement.offset for placement in hlsl]
    if std140_offsets != hlsl_offsets:
        differing = next(s.name for s, h in zip(std140, hlsl) if s.offset != h.offset)
        raise HlslLayoutError(
            f"member '{name}' is a struct '{layout_type.struct_name}' whose member "
            f"'{differing}' sits at a different byte under std140 than under HLSL "
            "constant-buffer packing"
        )
    for placement in std140:
        check_std140_interior_matches_hlsl(
            f"{name}.{placement.name}", placement.layout_type
        )
