"""SPIR-V uniform blocks laid out at the byte offsets their source declares.

A SPIR-V ``Uniform`` block carries its layout in ``Offset``, ``ArrayStride`` and ``MatrixStride`` decorations
(SPIR-V 1.6 section 2.16.2, Vulkan 1.3 section 15.6.4), so it can hold any layout the host wrote, not only GLSL's
std140. CrossGL keeps the layout of a constant block in ``@ buffer_layout(...)`` and in the ``packoffset``,
``offset`` and ``align`` qualifiers of its members:

* an HLSL constant buffer (``hlsl_cbuffer``) packs members into 16-byte registers: ``float a[2]; float b;`` puts
  ``b`` at byte 20 and ``float c; float2 d;`` puts ``d`` at byte 4, where std140 puts them at 32 and 8 (see
  :mod:`hlsl_buffer_layout`);
* a ``packoffset(c1.y)`` or ``layout(offset = 20)`` qualifier places a member at the byte it names (see
  :mod:`buffer_member_layout`), and ``layout(align = 32)`` raises its alignment.

:func:`placed_uniform_block` computes those placements with the shared planners (:mod:`glsl_block_layout`,
:mod:`hlsl_buffer_layout`). A block whose bytes match std140 everywhere keeps the generator's std140 path;
any other block is declared through layout clones decorated with the planned offsets and strides by
:func:`decorate_placed_uniform_block`. Such offsets break the extended alignment rules Vulkan applies to uniform
buffers by default, and every one of them satisfies the scalar block layout (VK_EXT_scalar_block_layout, core in
Vulkan 1.2), so the module records the requirement as ``OpSourceExtension "GL_EXT_scalar_block_layout"``, as
glslang does for a GLSL block that needs it. A layout no planner reproduces raises
:class:`UnsupportedSPIRVFeatureError`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..ast import ArrayNode, ArrayType, FunctionNode
from .array_utils import evaluate_literal_int_expression
from .buffer_member_layout import member_explicit_offset, member_is_row_major
from .glsl_block_layout import (
    GLSL_BLOCK_LAYOUTS,
    STD140_LAYOUT,
    block_matrix_vector_stride,
    block_struct_placements,
    crossgl_struct_layout_type,
    declared_buffer_layout,
    layout_array_stride,
    layout_struct_placements,
    member_explicit_alignment,
)
from .hlsl_buffer_layout import (
    HLSL_CBUFFER_LAYOUT,
    HlslLayoutError,
    HlslLayoutType,
    cbuffer_matrix_vector_stride,
    cbuffer_struct_placements,
)

__all__ = [
    "SCALAR_BLOCK_LAYOUT_SOURCE_EXTENSION",
    "PlacedUniformBlock",
    "decorate_placed_uniform_block",
    "placed_uniform_block",
]

SCALAR_BLOCK_LAYOUT_SOURCE_EXTENSION = "GL_EXT_scalar_block_layout"
PLACED_LAYOUTS = (HLSL_CBUFFER_LAYOUT,) + GLSL_BLOCK_LAYOUTS


@dataclass(frozen=True)
class PlacedUniformBlock:
    """The layout of a uniform block that SPIR-V must decorate member by member.

    ``layout`` names the rules nested values follow, ``layout_type`` is the block's shape with its explicit
    offsets, ``placements`` its members' offsets, and ``clone_key`` the key of the layout clones that carry the
    decorations, distinct from the clones of the std140, std430 and scalar paths.
    """

    layout: str
    layout_type: HlslLayoutType
    placements: tuple
    clone_key: str


def _unsupported(feature, message, capability):
    from .SPIRV_codegen import UnsupportedSPIRVFeatureError

    return UnsupportedSPIRVFeatureError(
        feature, message, missing_capabilities=(capability,)
    )


def _block_members(node):
    return [
        member
        for member in getattr(node, "members", None) or []
        if getattr(member, "name", None) and not isinstance(member, FunctionNode)
    ]


def _member_tuple(member):
    """``(name, type, row_major, offset, alignment)`` of a block or struct member."""
    if isinstance(member, ArrayNode):
        element_type = getattr(member, "element_type", getattr(member, "vtype", None))
        member_type = ArrayType(element_type, getattr(member, "size", None))
    elif getattr(member, "member_type", None) is not None:
        member_type = member.member_type
    else:
        member_type = getattr(member, "var_type", None) or getattr(member, "vtype")
    return (
        member.name,
        member_type,
        member_is_row_major(member),
        member_explicit_offset(member),
        member_explicit_alignment(member),
    )


def _has_explicit_placement(members):
    return any(
        member_explicit_offset(member) is not None
        or member_explicit_alignment(member) is not None
        for member in members
    )


def _block_layout_type(codegen, node, members):
    structs = getattr(codegen, "struct_declarations", None) or {}

    def struct_members(name):
        struct = structs.get(name)
        if struct is None:
            return None
        return [
            (*_member_tuple(member)[:3], None, member_explicit_alignment(member))
            for member in _block_members(struct)
        ]

    constants = codegen.active_literal_int_constants()
    return crossgl_struct_layout_type(
        node.name,
        [_member_tuple(member) for member in members],
        struct_members,
        (node.name,),
        lambda expression: evaluate_literal_int_expression(expression, constants),
    )


def _root_placements(layout_type, layout):
    if layout == HLSL_CBUFFER_LAYOUT:
        placements, _size = cbuffer_struct_placements(layout_type)
    else:
        placements, _size = block_struct_placements(layout_type, layout, root=True)
    return tuple(placements)


def _matrix_stride(layout_type, layout):
    if layout == HLSL_CBUFFER_LAYOUT:
        return cbuffer_matrix_vector_stride(layout_type)
    return block_matrix_vector_stride(layout_type, layout)


def _value_signature(layout_type, layout):
    """The bytes a value occupies, as nested offsets and strides, under ``layout``."""
    kind = layout_type.kind
    if kind == "array":
        return (
            "array",
            layout_type.count,
            layout_array_stride(layout_type, layout),
            _value_signature(layout_type.element, layout),
        )
    if kind == "matrix":
        return (
            "matrix",
            layout_type.row_major,
            _matrix_stride(layout_type, layout),
        )
    if kind == "struct":
        placements, _size = layout_struct_placements(layout_type, layout)
        return ("struct", _placements_signature(placements, layout))
    return (kind, layout_type.scalar, layout_type.components)


def _placements_signature(placements, layout):
    return tuple(
        (
            placement.name,
            placement.offset,
            _value_signature(placement.layout_type, layout),
        )
        for placement in placements
    )


def _validator_extent(layout_type, layout, placed_size):
    """The bytes ``spirv-val`` counts for a value when it checks that members do not overlap.

    SPIR-V Tools sizes a column-major matrix as ``columns * MatrixStride``, padding included, and an array as
    ``(count - 1) * ArrayStride`` plus its element's size; other values take the bytes their layout gives them.
    """
    kind = layout_type.kind
    if kind == "matrix" and not layout_type.row_major:
        return layout_type.registers * _matrix_stride(layout_type, layout)
    if kind == "array":
        element = layout_type.element
        element_size = _validator_extent(
            element, layout, _placed_value_size(element, layout)
        )
        stride = layout_array_stride(layout_type, layout)
        return (layout_type.count - 1) * stride + element_size
    return placed_size


def _placed_value_size(layout_type, layout):
    kind = layout_type.kind
    if kind in ("scalar", "vector"):
        return layout_type.scalar_size * layout_type.components
    if kind == "matrix":
        stride = _matrix_stride(layout_type, layout)
        return stride * (layout_type.registers - 1) + (
            layout_type.scalar_size * layout_type.components
        )
    if kind == "array":
        stride = layout_array_stride(layout_type, layout)
        return stride * (layout_type.count - 1) + _placed_value_size(
            layout_type.element, layout
        )
    _placements, size = layout_struct_placements(layout_type, layout)
    return size


def _check_validator_extents(struct_name, placements, layout):
    """Refuse members that start inside the padding ``spirv-val`` counts as part of the member before them.

    A column-major matrix whose last column leaves room for the next member (HLSL packs ``float k`` after a
    ``float3x3`` into its third register) has no SPIR-V spelling as a matrix member: ``spirv-val`` counts the
    whole last column stride and reports the members as overlapping.
    """
    ordered = sorted(placements, key=lambda placement: placement.offset)
    for previous, placement in zip(ordered, ordered[1:]):
        end = previous.offset + _validator_extent(
            previous.layout_type, layout, previous.size
        )
        if placement.offset < end:
            raise HlslLayoutError(
                f"member '{placement.name}' of '{struct_name}' at byte {placement.offset} lies in the padding "
                f"SPIR-V counts as part of '{previous.name}' (bytes {previous.offset} to {end}): a column-major "
                "matrix member takes its whole last column stride"
            )
    for placement in placements:
        nested = placement.layout_type
        while nested.kind == "array":
            nested = nested.element
        if nested.kind == "struct":
            nested_placements, _size = layout_struct_placements(nested, layout)
            _check_validator_extents(nested.struct_name, nested_placements, layout)


def placed_uniform_block(codegen, node) -> PlacedUniformBlock | None:
    """The placements of a ``cbuffer`` whose bytes differ from std140, or ``None`` for the std140 path.

    Blocks without explicit placements that declare std430 or scalar keep their own path, and so does a block
    the planners place exactly as std140 does.
    """
    layout = declared_buffer_layout(getattr(node, "attributes", None)) or STD140_LAYOUT
    members = _block_members(node)
    explicit = _has_explicit_placement(members)
    if layout != HLSL_CBUFFER_LAYOUT and not explicit:
        return None
    if layout not in PLACED_LAYOUTS:
        raise _unsupported(
            "uniform block layout",
            f"SPIR-V uniform block '{node.name}' declares the {layout} layout, which the SPIR-V emitter does not "
            "place yet",
            f"spirv.uniform_block.{layout}_layout",
        )
    try:
        layout_type = _block_layout_type(codegen, node, members)
        placements = _root_placements(layout_type, layout)
        std140_type = replace(layout_type, offsets=(), alignments=())
        std140 = _root_placements(std140_type, STD140_LAYOUT)
        if _placements_signature(placements, layout) == _placements_signature(
            std140, STD140_LAYOUT
        ):
            return None
        _check_validator_extents(node.name, placements, layout)
    except HlslLayoutError as exc:
        raise _unsupported(
            "uniform block layout",
            f"Cannot reproduce the {layout} layout of uniform block '{node.name}' in SPIR-V: {exc}",
            f"spirv.uniform_block.{layout}_layout",
        ) from exc
    return PlacedUniformBlock(layout, layout_type, placements, f"placed_{layout}")


def _decorate_once(codegen, type_id, decoration, value, decorated):
    """Record ``decoration value`` on a shared clone type, refusing a clone two layouts would decorate apart."""
    key = (type_id.id, decoration)
    previous = decorated.get(key)
    if previous is not None:
        if previous != value:
            raise _unsupported(
                "uniform block layout",
                f"the SPIR-V type {type_id.type.base_type} would need {decoration} {previous} and {value} in "
                "one module",
                "spirv.uniform_block.shared_layout_type",
            )
        return False
    decorated[key] = value
    codegen.decorations.append(f"OpDecorate %{type_id.id} {decoration} {value}")
    return True


def _decorate_value(codegen, type_id, layout_type, layout, decorated):
    """Decorate the array strides and nested struct offsets of a value of ``layout_type``."""
    kind = layout_type.kind
    if kind == "array":
        array_info = codegen.array_type_info_from_type(type_id)
        if array_info is None:
            raise HlslLayoutError(f"{type_id.type.base_type} is not an array type")
        _decorate_once(
            codegen,
            type_id,
            "ArrayStride",
            layout_array_stride(layout_type, layout),
            decorated,
        )
        _decorate_value(codegen, array_info[0], layout_type.element, layout, decorated)
        return
    if kind != "struct" or type_id.id in codegen.placed_uniform_structs:
        return
    codegen.placed_uniform_structs.add(type_id.id)
    placements, _size = layout_struct_placements(layout_type, layout)
    _decorate_members(codegen, type_id, placements, layout, decorated)


def _innermost_matrix(layout_type):
    while layout_type.kind == "array":
        layout_type = layout_type.element
    return layout_type if layout_type.kind == "matrix" else None


def _decorate_members(codegen, struct_type, placements, layout, decorated):
    members = codegen.current_struct_members.get(struct_type.type.base_type) or []
    if len(members) != len(placements):
        raise HlslLayoutError(
            f"{struct_type.type.base_type} has {len(members)} SPIR-V members but {len(placements)} placements"
        )
    for index, ((member_type, _name), placement) in enumerate(zip(members, placements)):
        codegen.decorations.append(
            f"OpMemberDecorate %{struct_type.id} {index} Offset {placement.offset}"
        )
        matrix = _innermost_matrix(placement.layout_type)
        if matrix is not None:
            orientation = "RowMajor" if matrix.row_major else "ColMajor"
            codegen.decorations.append(
                f"OpMemberDecorate %{struct_type.id} {index} {orientation}"
            )
            codegen.decorations.append(
                f"OpMemberDecorate %{struct_type.id} {index} MatrixStride "
                f"{_matrix_stride(matrix, layout)}"
            )
        _decorate_value(codegen, member_type, placement.layout_type, layout, decorated)


def decorate_placed_uniform_block(codegen, block_type, plan: PlacedUniformBlock):
    """Decorate ``block_type`` (built from ``plan.clone_key`` clones) with the offsets and strides of ``plan``."""
    if codegen.type_contains_bfloat16(block_type):
        codegen.require_bfloat16_storage(uniform=True)
    decorated = codegen.placed_uniform_decorations
    codegen.decorations.append(f"OpDecorate %{block_type.id} Block")
    try:
        _decorate_members(codegen, block_type, plan.placements, plan.layout, decorated)
    except HlslLayoutError as exc:
        raise _unsupported(
            "uniform block layout",
            f"Cannot decorate the {plan.layout} layout of uniform block {block_type.type.base_type} in SPIR-V: "
            f"{exc}",
            f"spirv.uniform_block.{plan.layout}_layout",
        ) from exc
    codegen.require_source_extension(SCALAR_BLOCK_LAYOUT_SOURCE_EXTENSION)
