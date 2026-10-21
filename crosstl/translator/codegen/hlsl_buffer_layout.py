"""HLSL buffer memory layouts for backends that must reproduce them.

HLSL places constant-buffer members with the constant packing rules of DXC and
structured-buffer elements with four-byte (eight-byte for 64-bit) alignment.
Neither matches the natural layout of Metal or the std140/std430 rules of GLSL.
CrossGL programs imported from HLSL therefore mark their buffers with
``@ buffer_layout(hlsl_cbuffer)`` or ``@ buffer_layout(hlsl_structured)`` and a
backend uses this module to compute the byte offsets the host writes.

Constant buffers (DXC):

* scalars and vectors are aligned to their component size and move to the next
  16-byte register when they would straddle one;
* every array element and every matrix row (row-major) or column
  (column-major) starts a new register, so array elements have a stride of a
  multiple of 16 bytes and the last element is not padded;
* a struct starts a new register, its members follow the same rules relative to
  its start, and the next member may pack directly after its last member.

Structured buffers place every member at its component alignment with no
16-byte rules; a struct is aligned to its largest member and its size is
rounded up to that alignment.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

BUFFER_LAYOUT_ATTRIBUTE = "buffer_layout"
HLSL_CBUFFER_LAYOUT = "hlsl_cbuffer"
HLSL_STRUCTURED_LAYOUT = "hlsl_structured"
HLSL_BUFFER_LAYOUTS = (HLSL_CBUFFER_LAYOUT, HLSL_STRUCTURED_LAYOUT)

CBUFFER_REGISTER_SIZE = 16

HLSL_SCALAR_SIZES = {
    "bool": 4,
    "int": 4,
    "uint": 4,
    "float": 4,
    "double": 8,
    "long": 8,
    "ulong": 8,
}


class HlslLayoutError(ValueError):
    """A buffer member whose HLSL layout cannot be reproduced."""


@dataclass(frozen=True)
class HlslLayoutType:
    """The memory shape of one HLSL buffer value.

    ``kind`` is ``scalar``, ``vector``, ``matrix``, ``struct`` or ``array``.
    Matrices are described by their in-memory vectors: ``registers`` vectors of
    ``components`` scalars (rows for row-major storage, columns otherwise).
    ``row_major`` says which: a row-major matrix of R rows and C columns has
    ``registers == R`` and ``components == C``.

    ``offsets`` holds the explicit byte offsets of the members of a constant
    buffer (``packoffset`` or ``offset`` qualifiers), ``None`` for a member
    without one, and is empty when no member has one.
    """

    kind: str
    scalar: str = ""
    components: int = 1
    registers: int = 0
    struct_name: str = ""
    members: Tuple[Tuple[str, "HlslLayoutType"], ...] = ()
    element: Optional["HlslLayoutType"] = None
    count: int = 0
    row_major: bool = False
    offsets: Tuple[Optional[int], ...] = ()

    @property
    def scalar_size(self):
        return HLSL_SCALAR_SIZES[self.scalar]


@dataclass(frozen=True)
class HlslMemberPlacement:
    name: str
    layout_type: HlslLayoutType
    offset: int
    size: int


def align_to(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def buffer_layout_from_attributes(attributes):
    """Return the HLSL buffer layout named by ``@ buffer_layout(...)``, if any."""
    for attribute in attributes or []:
        if str(getattr(attribute, "name", "")).lower() != BUFFER_LAYOUT_ATTRIBUTE:
            continue
        arguments = list(getattr(attribute, "arguments", None) or [])
        if not arguments:
            continue
        argument = arguments[0]
        value = getattr(argument, "name", getattr(argument, "value", argument))
        value = str(value).strip().strip('"').lower()
        if value in HLSL_BUFFER_LAYOUTS:
            return value
    return None


def scalar_type(scalar):
    if scalar not in HLSL_SCALAR_SIZES:
        raise HlslLayoutError(f"unsupported HLSL buffer scalar type '{scalar}'")
    return HlslLayoutType("scalar", scalar=scalar)


def vector_type(scalar, components):
    scalar_type(scalar)
    return HlslLayoutType("vector", scalar=scalar, components=components)


def matrix_type(scalar, registers, components, row_major=False):
    scalar_type(scalar)
    return HlslLayoutType(
        "matrix",
        scalar=scalar,
        components=components,
        registers=registers,
        row_major=row_major,
    )


def oriented_matrix_type(scalar, columns, rows, row_major=False):
    """The layout type of a matrix of ``columns`` columns and ``rows`` rows in the given orientation."""
    if row_major:
        return matrix_type(scalar, rows, columns, row_major=True)
    return matrix_type(scalar, columns, rows)


def struct_type(name, members, offsets=()):
    members = tuple(members)
    offsets = tuple(offsets or ())
    if not any(offset is not None for offset in offsets):
        offsets = ()
    elif len(offsets) != len(members):
        raise HlslLayoutError(f"struct '{name}' has offsets for some members only")
    return HlslLayoutType("struct", struct_name=name, members=members, offsets=offsets)


def contains_row_major_matrix(layout_type):
    """Whether a value holds a row-major matrix anywhere."""
    if layout_type.kind == "matrix":
        return layout_type.row_major
    if layout_type.kind == "array":
        return contains_row_major_matrix(layout_type.element)
    if layout_type.kind == "struct":
        return any(
            contains_row_major_matrix(member) for _name, member in layout_type.members
        )
    return False


def array_type(element, count):
    if count is None or count <= 0:
        raise HlslLayoutError("HLSL buffer arrays need a positive constant size")
    return HlslLayoutType("array", element=element, count=count)


def cbuffer_placement(layout_type, offset):
    """Place a value at the first legal constant-buffer offset at or after ``offset``.

    Returns ``(start, size)`` where ``size`` excludes trailing padding that a
    following member may reuse.
    """
    kind = layout_type.kind
    if kind in ("scalar", "vector"):
        size = layout_type.scalar_size * layout_type.components
        start = align_to(offset, layout_type.scalar_size)
        if (
            start // CBUFFER_REGISTER_SIZE != (
                start + size - 1
            ) // CBUFFER_REGISTER_SIZE
        ):
            start = align_to(start, CBUFFER_REGISTER_SIZE)
        return start, size
    if kind == "matrix":
        start = align_to(offset, CBUFFER_REGISTER_SIZE)
        vector_size = layout_type.scalar_size * layout_type.components
        stride = cbuffer_matrix_vector_stride(layout_type)
        return start, stride * (layout_type.registers - 1) + vector_size
    if kind == "array":
        start = align_to(offset, CBUFFER_REGISTER_SIZE)
        _element_start, element_size = cbuffer_placement(layout_type.element, 0)
        stride = cbuffer_array_stride(layout_type)
        return start, stride * (layout_type.count - 1) + element_size
    if kind == "struct":
        start = align_to(offset, CBUFFER_REGISTER_SIZE)
        _members, size = cbuffer_struct_placements(layout_type)
        return start, size
    raise HlslLayoutError(f"unsupported HLSL buffer value kind '{kind}'")


def cbuffer_matrix_vector_stride(layout_type):
    """Distance between the in-memory vectors of a constant-buffer matrix.

    Every row (row-major) or column (column-major) starts a new register, and
    a vector of three or four 64-bit components spans two registers.
    """
    vector_size = layout_type.scalar_size * layout_type.components
    return align_to(vector_size, CBUFFER_REGISTER_SIZE)


def cbuffer_array_stride(layout_type):
    _start, element_size = cbuffer_placement(layout_type.element, 0)
    return align_to(element_size, CBUFFER_REGISTER_SIZE)


def cbuffer_struct_placements(layout_type):
    """Member placements of a struct (or constant buffer) relative to its start.

    Members with ``packoffset`` sit at the bytes it names, which must satisfy
    the packing rules for their type, and must not overlap.
    """
    if layout_type.offsets:
        return cbuffer_explicit_placements(layout_type)
    placements = []
    cursor = 0
    for name, member_type in layout_type.members:
        start, size = cbuffer_placement(member_type, cursor)
        placements.append(HlslMemberPlacement(name, member_type, start, size))
        cursor = start + size
    return placements, cursor


def cbuffer_explicit_placements(layout_type):
    if None in layout_type.offsets:
        raise HlslLayoutError(
            f"constant buffer '{layout_type.struct_name}' mixes packoffset members "
            "with members placed in declaration order"
        )
    placements = []
    for (name, member_type), offset in zip(layout_type.members, layout_type.offsets):
        start, size = cbuffer_placement(member_type, offset)
        if start != offset:
            raise HlslLayoutError(
                f"packoffset of '{name}' at byte {offset} breaks the constant-buffer "
                f"packing of its type, which starts at byte {start}"
            )
        placements.append(HlslMemberPlacement(name, member_type, start, size))
    check_disjoint_placements(layout_type.struct_name, placements)
    return placements, max(
        (placement.offset + placement.size for placement in placements), default=0
    )


def check_disjoint_placements(struct_name, placements):
    ordered = sorted(placements, key=lambda placement: placement.offset)
    for previous, placement in zip(ordered, ordered[1:]):
        if placement.offset < previous.offset + previous.size:
            raise HlslLayoutError(
                f"member '{placement.name}' of '{struct_name}' at byte {placement.offset} "
                f"overlaps member '{previous.name}'"
            )


def structured_size_align(layout_type):
    """``(size, alignment)`` of a value in a structured buffer."""
    kind = layout_type.kind
    if kind == "scalar":
        return layout_type.scalar_size, layout_type.scalar_size
    if kind == "vector":
        return (
            layout_type.scalar_size * layout_type.components,
            layout_type.scalar_size,
        )
    if kind == "matrix":
        return (
            layout_type.scalar_size * layout_type.components * layout_type.registers,
            layout_type.scalar_size,
        )
    if kind == "array":
        size, alignment = structured_size_align(layout_type.element)
        return size * layout_type.count, alignment
    if kind == "struct":
        _members, size, alignment = structured_struct_placements(layout_type)
        return size, alignment
    raise HlslLayoutError(f"unsupported HLSL buffer value kind '{kind}'")


def structured_struct_placements(layout_type):
    """Member placements, size and alignment of a struct in a structured buffer."""
    placements = []
    cursor = 0
    max_alignment = 1
    for name, member_type in layout_type.members:
        size, alignment = structured_size_align(member_type)
        start = align_to(cursor, alignment)
        placements.append(HlslMemberPlacement(name, member_type, start, size))
        cursor = start + size
        max_alignment = max(max_alignment, alignment)
    return placements, align_to(cursor, max_alignment), max_alignment


def struct_placements(layout_type, layout):
    """Member placements and size of a struct under one HLSL buffer layout."""
    if layout == HLSL_CBUFFER_LAYOUT:
        return cbuffer_struct_placements(layout_type)
    placements, size, _alignment = structured_struct_placements(layout_type)
    return placements, size
