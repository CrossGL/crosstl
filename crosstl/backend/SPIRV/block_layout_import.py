"""The CrossGL buffer layout that reproduces the explicit layout of a SPIR-V uniform or push-constant block.

SPIR-V spells a block's layout out: every member of a ``Uniform`` or ``PushConstant`` block has an ``Offset``, every
array an ``ArrayStride``, and every matrix a ``MatrixStride`` and a ``RowMajor`` or ``ColMajor`` orientation (SPIR-V
1.6 section 2.16.2 and the Vulkan "Offset and Stride Assignment" rules). CrossGL keeps a block's layout as a named
rule set (``@ buffer_layout``), which every generator reproduces with its own declarations or with explicit loads.
An importer that dropped the decorations left the generators to their default rules: a std430 push-constant block
read as std140, or as the natural MSL layout, reads its members from other bytes than the host wrote.

:func:`spirv_block_buffer_layout` checks every decoration of a block against the std140, std430 and scalar rules
(GLSL 4.60 section 7.6.2.2 and GL_EXT_scalar_block_layout) and the HLSL constant-buffer packing rules, and returns
the first rule set that reproduces all of them: std140 first for a uniform block and std430 first for push
constants, the rules GLSL gives them by default. A decoration the module omits constrains nothing, and a block
without any ``Offset`` keeps CrossGL's default; a block whose decorations no rule set reproduces, or whose members
this module cannot size, is refused.
"""

from __future__ import annotations

from ...translator.codegen.glsl_block_layout import (
    GLSL_BLOCK_LAYOUTS,
    SCALAR_BLOCK_LAYOUT,
    STD140_LAYOUT,
    STD430_LAYOUT,
    block_matrix_vector_stride,
    layout_array_stride,
    layout_struct_placements,
)
from ...translator.codegen.hlsl_buffer_layout import (
    HLSL_CBUFFER_LAYOUT,
    HlslLayoutError,
    array_type,
    cbuffer_matrix_vector_stride,
    oriented_matrix_type,
    scalar_type,
    struct_type,
    vector_type,
)
from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["SpirvBlockLayoutUnsupportedError", "spirv_block_buffer_layout"]

UNIFORM_LAYOUTS = (
    STD140_LAYOUT,
    STD430_LAYOUT,
    SCALAR_BLOCK_LAYOUT,
    HLSL_CBUFFER_LAYOUT,
)
PUSH_CONSTANT_LAYOUTS = (
    STD430_LAYOUT,
    STD140_LAYOUT,
    SCALAR_BLOCK_LAYOUT,
    HLSL_CBUFFER_LAYOUT,
)
# SPIR-V scalar spellings of the importer, by the scalar names of the layout rules.
_SCALARS = {
    "float": "float",
    "double": "double",
    "int": "int",
    "uint": "uint",
    "i64": "long",
    "u64": "ulong",
}


class SpirvBlockLayoutUnsupportedError(TranslationDiagnostic):
    """A SPIR-V block whose explicit layout no CrossGL buffer layout reproduces."""

    def __init__(self, block, detail):
        self.missing_capabilities = ("spirv.import.block_layout",)
        super().__init__(
            f"SPIR-V block '{block}' {detail}",
            target="cgl",
            code=f"{CODE_PREFIX}spirv-block-layout-unsupported",
        )


class _Unknown(Exception):
    """A member type the layout rules do not size."""


class _Decorations:
    """The layout decorations of one module, and the array lengths of its types."""

    def __init__(self, types, decorations, member_decorations, array_length):
        self.types = types
        self.decorations = decorations
        self.member_decorations = member_decorations
        self.array_length = array_length

    def member(self, struct_id, index, name):
        for member, decoration, operands in self.member_decorations.get(struct_id, ()):
            if str(member) == str(index) and decoration == name:
                return _literal(operands)
        return None

    def has_member(self, struct_id, index, name):
        return any(
            str(member) == str(index) and decoration == name
            for member, decoration, _operands in self.member_decorations.get(
                struct_id, ()
            )
        )

    def type_decoration(self, type_id, name):
        for decoration, operands in self.decorations.get(type_id, ()):
            if decoration == name:
                return _literal(operands)
        return None

    def offsets_anywhere(self, type_id, seen=()):
        """Whether the type, or a struct it holds, decorates a member with ``Offset``."""
        if type_id in seen:
            return False
        info = self.types.get(type_id, {})
        if info.get("kind") == "struct":
            if any(
                decoration == "Offset"
                for _member, decoration, _operands in self.member_decorations.get(
                    type_id, ()
                )
            ):
                return True
            return any(
                self.offsets_anywhere(member, (*seen, type_id))
                for member in info.get("member_types", ())
            )
        if info.get("kind") == "array":
            return self.offsets_anywhere(info.get("element_type"), (*seen, type_id))
        return False


def _literal(operands):
    if not operands:
        return None
    try:
        return int(str(operands[0]), 0)
    except ValueError:
        return None


def _scalar(info):
    name = _SCALARS.get(info.get("name"))
    if name is None:
        raise _Unknown(f"a {info.get('name')} member")
    return name


def _layout_type(module, type_id, row_major=False):
    """The layout shape of a SPIR-V type; ``row_major`` orients a matrix or the matrices of an array."""
    info = module.types.get(type_id, {})
    kind = info.get("kind")
    if kind == "scalar":
        return scalar_type(_scalar(info))
    if kind == "vector":
        component = module.types.get(info.get("component_type"), {})
        return vector_type(_scalar(component), int(info.get("component_count")))
    if kind == "matrix":
        column = module.types.get(info.get("column_type"), {})
        component = module.types.get(column.get("component_type"), {})
        return oriented_matrix_type(
            _scalar(component),
            int(info.get("column_count")),
            int(column.get("component_count")),
            row_major,
        )
    if kind == "array":
        count = module.array_length(info.get("length_id"))
        if count is None:
            raise _Unknown("an array whose length is not a constant")
        return array_type(
            _layout_type(module, info.get("element_type"), row_major), count
        )
    if kind == "struct":
        members = []
        for index, member_type in enumerate(info.get("member_types", ())):
            orientation = module.has_member(type_id, index, "RowMajor")
            members.append(
                (f"m{index}", _layout_type(module, member_type, orientation))
            )
        return struct_type(type_id, members)
    raise _Unknown(f"a {kind or 'unknown'} member")


def _matrix_stride(layout_type, layout):
    if layout in GLSL_BLOCK_LAYOUTS:
        return block_matrix_vector_stride(layout_type, layout)
    return cbuffer_matrix_vector_stride(layout_type)


def _innermost_matrix(layout_type):
    while layout_type.kind == "array":
        layout_type = layout_type.element
    return layout_type if layout_type.kind == "matrix" else None


def _mismatch(module, type_id, layout_type, layout, path, root=False):
    """The first decoration of ``type_id`` that ``layout`` does not reproduce, as a phrase, or ``None``."""
    info = module.types.get(type_id, {})
    if info.get("kind") == "array":
        stride = module.type_decoration(type_id, "ArrayStride")
        expected = layout_array_stride(layout_type, layout)
        if stride is not None and stride != expected:
            return f"{path} has ArrayStride {stride}, not {expected}"
        return _mismatch(
            module, info.get("element_type"), layout_type.element, layout, f"{path}[]"
        )
    if info.get("kind") != "struct":
        return None
    placements, _size = layout_struct_placements(layout_type, layout, root=root)
    for index, (member_type, placement) in enumerate(
        zip(info.get("member_types", ()), placements)
    ):
        where = f"{path}.{index}" if path else f"member {index}"
        offset = module.member(type_id, index, "Offset")
        if offset is not None and offset != placement.offset:
            return f"{where} has Offset {offset}, not {placement.offset}"
        matrix = _innermost_matrix(placement.layout_type)
        if matrix is not None:
            stride = module.member(type_id, index, "MatrixStride")
            expected = _matrix_stride(matrix, layout)
            if stride is not None and stride != expected:
                return f"{where} has MatrixStride {stride}, not {expected}"
        found = _mismatch(module, member_type, placement.layout_type, layout, where)
        if found:
            return found
    return None


def spirv_block_buffer_layout(
    block,
    struct_id,
    push_constant,
    types,
    decorations,
    member_decorations,
    array_length,
):
    """The ``@ buffer_layout`` of a uniform (``push_constant`` false) or push-constant block, or ``None``.

    ``array_length(length_id)`` returns the integer value of an array length operand, or ``None``.
    """
    module = _Decorations(types, decorations, member_decorations, array_length)
    if not module.offsets_anywhere(struct_id):
        return None
    try:
        layout_type = _layout_type(module, struct_id)
    except (_Unknown, HlslLayoutError, TypeError, ValueError) as error:
        raise SpirvBlockLayoutUnsupportedError(
            block, f"has explicit offsets the layout rules cannot check: {error}"
        ) from None
    reasons = []
    for layout in PUSH_CONSTANT_LAYOUTS if push_constant else UNIFORM_LAYOUTS:
        try:
            reason = _mismatch(module, struct_id, layout_type, layout, "", root=True)
        except HlslLayoutError as error:
            reason = str(error)
        if reason is None:
            return layout
        reasons.append(f"{layout}: {reason}")
    raise SpirvBlockLayoutUnsupportedError(
        block,
        "has Offset, ArrayStride or MatrixStride decorations that no std140, std430, scalar or HLSL "
        f"constant-buffer layout reproduces ({'; '.join(reasons)})",
    )
