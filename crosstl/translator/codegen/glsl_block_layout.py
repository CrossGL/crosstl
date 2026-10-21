"""GLSL std140, std430 and scalar block layouts for backends that must reproduce them.

A GLSL uniform block is laid out with the std140 rules (GLSL 4.60 section
7.6.2.2), a Vulkan push-constant block with the std430 rules unless it names
another layout (GL_KHR_vulkan_glsl), and a block that names ``scalar`` with
the rules of GL_EXT_scalar_block_layout. None of them is HLSL constant-buffer
packing or the natural layout of Metal:

* std140 and std430 align a scalar to its size, a two-component vector to
  twice that and a three- or four-component vector to four times that; a
  ``vec3`` takes 12 bytes, so a following ``float`` packs into its last word
  (rules 1-3);
* an array element, and every column (or row, for ``row_major``) of a matrix,
  is aligned like its element (std430) or like its element rounded up to 16
  bytes (std140), and an array or matrix takes ``count * stride`` bytes
  (rules 4-8): ``float a[2]`` takes 8 bytes under std430 and 32 under std140;
* a struct is aligned to its widest member, rounded up to 16 bytes under
  std140, and its size is rounded up to that alignment (rules 9-10);
* the scalar layout aligns every value to the size of its scalar components;
  an array's stride is its element size rounded up to that alignment, and the
  padding after the last element of an array, or after the last member of a
  struct, is not reserved: the next member may start in it (the offsets
  glslang assigns);
* ``layout(offset = N)`` places a member at byte ``N``, which must be a
  multiple of its alignment and must not reach back into the previous member;
  ``layout(align = N)`` raises a member's alignment to ``N`` (section 4.4.5).

CrossGL keeps such a block as a ``cbuffer`` marked ``@ buffer_layout(std140)``
(or ``std430``, ``scalar``); the shapes are the :class:`HlslLayoutType` values
of :mod:`hlsl_buffer_layout`, so a backend may place a buffer with either rule
set and compare them.

The same planner computes the ``metal`` layout of an MSL ``constant`` buffer,
the natural layout of its type (MSL 3.2 section 2.2 and table 2.4): it follows
std430 except that a three-component vector takes 16 bytes, so a following
scalar starts after it, and that a packed vector (``packed_float3``, a
``@ packed`` member in CrossGL) is aligned to its scalar and takes exactly its
components. MSL's one-byte ``bool`` has no word layout and is refused.
"""

import re
from dataclasses import replace

from ..ast import ArrayType, MatrixType, NamedType, PrimitiveType, VectorType
from .hlsl_buffer_layout import (
    BUFFER_LAYOUT_ATTRIBUTE,
    HLSL_BUFFER_LAYOUTS,
    HlslLayoutError,
    HlslMemberPlacement,
    align_to,
    array_type,
    cbuffer_array_stride,
    cbuffer_matrix_vector_stride,
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
METAL_LAYOUT = "metal"
# Layouts a ``@ buffer_layout`` attribute may name.
DECLARED_BUFFER_LAYOUTS = HLSL_BUFFER_LAYOUTS + (
    STD140_LAYOUT,
    STD430_LAYOUT,
    SCALAR_BLOCK_LAYOUT,
    METAL_LAYOUT,
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


# Block layout rules ---------------------------------------------------------
GLSL_BLOCK_LAYOUTS = (STD140_LAYOUT, STD430_LAYOUT, SCALAR_BLOCK_LAYOUT)
# Every layout the block planner below computes.
BLOCK_LAYOUTS = GLSL_BLOCK_LAYOUTS + (METAL_LAYOUT,)


def require_block_layout(layout):
    if layout not in BLOCK_LAYOUTS:
        raise HlslLayoutError(f"'{layout}' is not a block layout")


def block_vector_alignment(scalar_size, components, layout):
    if layout == SCALAR_BLOCK_LAYOUT or components == 1:
        return scalar_size
    if components == 2:
        return 2 * scalar_size
    return 4 * scalar_size


def check_block_value(layout_type, layout):
    """Refuse a value ``layout`` has no layout for: a packed vector outside the metal and scalar rules, or an MSL
    ``bool``, which takes one byte."""
    if layout_type.packed and layout not in (METAL_LAYOUT, SCALAR_BLOCK_LAYOUT):
        raise HlslLayoutError(
            f"a packed vector has no {layout} layout: {layout} aligns a vector of two or more components to "
            "more than its scalar"
        )
    if layout == METAL_LAYOUT and layout_type.scalar == "bool":
        raise HlslLayoutError(
            "an MSL bool in a constant buffer takes one byte, which no word layout reproduces"
        )


def packed_layout_type(layout_type, member_name=""):
    """``layout_type`` stored in MSL's packed form: a vector, or an array of vectors, of a ``@ packed`` member."""
    if layout_type.kind == "vector":
        return replace(layout_type, packed=True)
    if layout_type.kind == "array" and layout_type.element.kind == "vector":
        return replace(layout_type, element=replace(layout_type.element, packed=True))
    raise HlslLayoutError(
        f"member '{member_name}' is marked packed, but only vectors and arrays of vectors have a packed form"
    )


def block_matrix_vector_stride(layout_type, layout):
    """Distance between the stored columns (rows, for ``row_major``) of a matrix (rules 5 and 7)."""
    if layout == SCALAR_BLOCK_LAYOUT:
        return layout_type.scalar_size * layout_type.components
    stride = block_vector_alignment(
        layout_type.scalar_size, layout_type.components, layout
    )
    if layout == STD140_LAYOUT:
        return align_to(stride, VEC4_ALIGNMENT)
    return stride


def block_alignment(layout_type, layout):
    """The base alignment of a value under ``layout``."""
    kind = layout_type.kind
    check_block_value(layout_type, layout)
    if kind == "scalar" or layout_type.packed:
        return layout_type.scalar_size
    if kind == "vector":
        return block_vector_alignment(
            layout_type.scalar_size, layout_type.components, layout
        )
    if kind == "matrix":
        if layout == SCALAR_BLOCK_LAYOUT:
            return layout_type.scalar_size
        return block_matrix_vector_stride(layout_type, layout)
    if kind == "array":
        alignment = block_alignment(layout_type.element, layout)
        if layout == STD140_LAYOUT:
            return align_to(alignment, VEC4_ALIGNMENT)
        return alignment
    if kind == "struct":
        alignment = VEC4_ALIGNMENT if layout == STD140_LAYOUT else 1
        for (_name, member), explicit in zip(
            layout_type.members, member_alignments(layout_type)
        ):
            alignment = max(alignment, block_alignment(member, layout), explicit or 0)
        if layout == STD140_LAYOUT:
            return align_to(alignment, VEC4_ALIGNMENT)
        return alignment
    raise HlslLayoutError(f"unsupported {layout} value kind '{kind}'")


def block_array_stride(layout_type, layout):
    """The element stride of an array (rules 4, 6, 8 and 10)."""
    return align_to(
        block_size(layout_type.element, layout), block_alignment(layout_type, layout)
    )


def block_size(layout_type, layout):
    """The bytes a value takes: no other member may start before its offset plus this size."""
    kind = layout_type.kind
    check_block_value(layout_type, layout)
    if kind in ("scalar", "vector"):
        if (
            layout == METAL_LAYOUT
            and layout_type.components == 3
            and not layout_type.packed
        ):
            # An MSL three-component vector takes the size of four (MSL 3.2 table 2.4).
            return layout_type.scalar_size * 4
        return layout_type.scalar_size * layout_type.components
    if kind == "matrix":
        if layout == SCALAR_BLOCK_LAYOUT:
            return (
                layout_type.scalar_size * layout_type.components * layout_type.registers
            )
        return block_matrix_vector_stride(layout_type, layout) * layout_type.registers
    if kind == "array":
        stride = block_array_stride(layout_type, layout)
        if layout == SCALAR_BLOCK_LAYOUT:
            # The padding after the last element is not reserved.
            return stride * (layout_type.count - 1) + block_size(
                layout_type.element, layout
            )
        return stride * layout_type.count
    if kind == "struct":
        _placements, size = block_struct_placements(layout_type, layout)
        return size
    raise HlslLayoutError(f"unsupported {layout} value kind '{kind}'")


def member_alignments(layout_type):
    """The ``align`` qualifier of every member of a struct, ``None`` where there is none."""
    return layout_type.alignments or (None,) * len(layout_type.members)


def block_struct_placements(layout_type, layout, root=False):
    """Member placements and size of a struct, or of a block when ``root``.

    The size of a nested struct is rounded up to its alignment, except under
    the scalar layout; a block's size ends after its last member.
    """
    require_block_layout(layout)
    offsets = layout_type.offsets or (None,) * len(layout_type.members)
    alignments = member_alignments(layout_type)
    placements = []
    cursor = 0
    for (name, member_type), offset, explicit_alignment in zip(
        layout_type.members, offsets, alignments
    ):
        alignment = block_alignment(member_type, layout)
        if offset is not None:
            if offset % alignment:
                raise HlslLayoutError(
                    f"offset {offset} of '{name}' is not a multiple of its {layout} "
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
        size = block_size(member_type, layout)
        placements.append(HlslMemberPlacement(name, member_type, start, size))
        cursor = start + size
    if root or layout == SCALAR_BLOCK_LAYOUT:
        return placements, cursor
    return placements, align_to(cursor, block_alignment(layout_type, layout))


def layout_struct_placements(layout_type, layout, root=False):
    """Member placements and size of a struct under ``layout`` (an HLSL or a block layout)."""
    if layout in BLOCK_LAYOUTS:
        return block_struct_placements(layout_type, layout, root)
    return struct_placements(layout_type, layout)


def layout_array_stride(layout_type, layout):
    if layout in BLOCK_LAYOUTS:
        return block_array_stride(layout_type, layout)
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
    # The HLSL-style spellings some importers emit: unambiguous for vectors and square matrices only.
    alias_match = re.fullmatch(
        r"(float|double|int|uint|bool)([234])(?:x([234]))?", type_name
    )
    if alias_match is not None:
        scalar, size, second = alias_match.groups()
        if second is None:
            return vector_type(CROSSGL_SCALARS[scalar], int(size))
        if second != size:
            raise HlslLayoutError(
                f"the matrix spelling '{type_name}' does not say which size is the column count; "
                f"write it as mat{size}x{second} or mat{second}x{size}"
            )
        if scalar not in ("float", "double"):
            raise HlslLayoutError(f"{scalar} matrices have no uniform-block layout")
        return oriented_matrix_type(scalar, int(size), int(size), row_major)
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
    """The layout shape of a struct or block from its ``(name, type, row_major, offset, alignment)`` members.

    A sixth item, when present, says whether the member is ``@ packed``.
    """
    members = list(members)
    layouts = []
    for member in members:
        layout = crossgl_layout_type(
            member[1], struct_members, member[2], stack, constant_value
        )
        if len(member) > 5 and member[5]:
            layout = packed_layout_type(layout, member[0])
        layouts.append((member[0], layout))
    return struct_type(
        name,
        layouts,
        [member[3] for member in members],
        [member[4] for member in members],
    )


# HLSL constant buffers -------------------------------------------------------
def hlsl_packoffsets_for_block(layout_type, layout):
    """``packoffset`` byte offsets that put the members of a constant buffer at their ``layout`` offsets.

    Returns ``None`` when HLSL packing already places every member there, and
    raises :class:`HlslLayoutError` when HLSL cannot reproduce the block: a
    member's offset breaks HLSL packing for its type, or a struct, array or
    matrix inside a member is laid out differently by the two rule sets
    (``packoffset`` places only the members of the buffer itself).
    """
    placements, _size = block_struct_placements(layout_type, layout, root=True)
    hlsl, _hlsl_size = cbuffer_struct_placements(
        struct_type(layout_type.struct_name, layout_type.members)
    )
    for placement in placements:
        check_block_interior_matches_hlsl(placement.name, placement.layout_type, layout)
    if [placement.offset for placement in placements] == [
        placement.offset for placement in hlsl
    ]:
        return None
    explicit = struct_type(
        layout_type.struct_name,
        layout_type.members,
        [placement.offset for placement in placements],
    )
    # Raises when an offset is not a legal HLSL placement of its type.
    cbuffer_struct_placements(explicit)
    return [placement.offset for placement in placements]


def check_block_interior_matches_hlsl(name, layout_type, layout):
    """Refuse a value whose inner layout (array stride, matrix stride, struct members) differs from HLSL's."""
    kind = layout_type.kind
    if kind == "array":
        stride = block_array_stride(layout_type, layout)
        hlsl_stride = cbuffer_array_stride(layout_type)
        if stride != hlsl_stride:
            raise HlslLayoutError(
                f"array '{name}' has a {layout} stride of {stride} bytes but an "
                f"HLSL constant-buffer stride of {hlsl_stride}"
            )
        check_block_interior_matches_hlsl(name, layout_type.element, layout)
        return
    if kind == "matrix":
        stride = block_matrix_vector_stride(layout_type, layout)
        if stride != cbuffer_matrix_vector_stride(layout_type):
            raise HlslLayoutError(
                f"matrix '{name}' has a {layout} vector stride of {stride} bytes, "
                "which HLSL constant buffers do not use"
            )
        return
    if kind != "struct":
        return
    placements, size = block_struct_placements(layout_type, layout)
    hlsl, hlsl_size = cbuffer_struct_placements(
        struct_type(layout_type.struct_name, layout_type.members)
    )
    offsets = [placement.offset for placement in placements]
    hlsl_offsets = [placement.offset for placement in hlsl]
    if offsets != hlsl_offsets:
        differing = next(
            s.name for s, h in zip(placements, hlsl) if s.offset != h.offset
        )
        raise HlslLayoutError(
            f"member '{name}' is a struct '{layout_type.struct_name}' whose member "
            f"'{differing}' sits at a different byte under {layout} than under HLSL "
            "constant-buffer packing"
        )
    for placement in placements:
        check_block_interior_matches_hlsl(
            f"{name}.{placement.name}", placement.layout_type, layout
        )
