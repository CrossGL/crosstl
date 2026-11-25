"""Metal unions for the CrossGL unions that other languages' importers write.

CrossGL has no union type. A struct marked ``@union_layout(size, alignment, byte_order, abi)``, whose members are each
marked ``@union_member_layout(offset, size, alignment)``, stands for one. The Metal importer writes this contract with
``metal`` as its ABI, and the CUDA and HIP importers with ``cuda`` and ``hip``: nvcc and hipcc lay a union out by the
C++ rules, every member at the union's first byte (C++ [class.union]/2). MSL is C++14 with the same union rule, so a
CUDA or HIP union is the same union in Metal when Metal gives each member, and the union itself, the recorded layout.
The Metal generator refused every union whose ABI was not ``metal``, so a CUDA kernel that writes one member of a
union and reads another had no Metal translation.

:func:`verify_foreign_union_layout` accepts a union of any ABI when:

* every member has, in Metal, the size and alignment the contract records for it, by the scalar sizes of MSL 3.2
  section 2.1, the vector sizes and alignments of section 2.2 (``float3`` is 16 bytes aligned to 16, where CUDA's
  ``float3`` is 12 bytes aligned to 4) and the array and structure rules of C++, as
  :func:`crosstl.backend.Metal.type_layout.metal_type_layout` computes them; and
* the union's size and alignment are the largest member's size rounded up to the largest member alignment, and that
  alignment, which is what MSL gives a union without an alignment specifier.

The ``metal`` ABI keeps its own check: the Metal importer records Metal's layout. Any other union is refused with
:meth:`~crosstl.translator.codegen.metal_codegen.MetalCodeGen.metal_union_layout_error`, naming the member whose
Metal layout differs or cannot be computed (a pointer, a resource, a member of a structure with an explicit layout).
"""

from __future__ import annotations

import re

from ..ast import ArrayType, StructNode

__all__ = ["NATIVE_UNION_ABIS", "verify_foreign_union_layout"]

# The ABIs whose recorded layout is Metal's own: the Metal importer's.
NATIVE_UNION_ABIS = frozenset({"metal", "msl"})

_EXTENT = re.compile(r"\[\s*(\d+)\s*\]")


def _round(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def _attribute_names(codegen, node):
    return {
        codegen.normalized_metal_abi_attribute_name(attribute)
        for attribute in getattr(node, "attributes", None) or ()
    }


def _member_type_text(codegen, member):
    type_node = getattr(member, "member_type", None)
    if type_node is not None:
        return codegen.convert_type_node_to_string(type_node)
    return getattr(member, "vtype", None)


def _text_layout(codegen, type_text, resolving):
    """``(size, alignment)`` in Metal of the CrossGL type spelled ``type_text``, or ``None``."""
    from ...backend.Metal.type_layout import metal_type_layout

    if not type_text or "*" in str(type_text) or "&" in str(type_text):
        return None
    mapped = str(codegen.map_type(str(type_text))).strip()
    layout = metal_type_layout(mapped)
    if layout is not None:
        return layout
    base, suffix = mapped, ""
    if "[" in mapped:
        base, suffix = mapped[: mapped.index("[")].strip(), mapped[mapped.index("[") :]
    extents = [int(value) for value in _EXTENT.findall(suffix)]
    if suffix and (not extents or _EXTENT.sub("", suffix).strip()):
        return None
    struct = codegen.structs_by_name.get(base)
    if not isinstance(struct, StructNode) or base in resolving:
        return None
    layout = _struct_layout(codegen, struct, resolving | {base})
    if layout is None:
        return None
    size, alignment = layout
    for extent in reversed(extents):
        if extent <= 0:
            return None
        size = _round(size, alignment) * extent
    return size, alignment


def _member_layout(codegen, member, resolving):
    type_node = getattr(member, "member_type", None)
    if isinstance(type_node, ArrayType) and type_node.size is None:
        return None
    return _text_layout(codegen, _member_type_text(codegen, member), resolving)


def _struct_layout(codegen, struct, resolving):
    """The Metal layout of a nested structure: a plain one by C++ rules, a union contract by its members."""
    names = _attribute_names(codegen, struct)
    is_union = "union_layout" in names
    if names - {"union_layout"}:
        # An explicit alignment, a packed or HLSL buffer layout: the emitted structure is not the plain one.
        return None
    offset, alignment = 0, 1
    for member in getattr(struct, "members", None) or ():
        if "static" in _attribute_names(codegen, member):
            continue
        layout = _member_layout(codegen, member, resolving)
        if layout is None:
            return None
        member_size, member_alignment = layout
        if is_union:
            offset = max(offset, member_size)
        else:
            offset = _round(offset, member_alignment) + member_size
        alignment = max(alignment, member_alignment)
    if offset == 0:
        return None
    return _round(offset, alignment), alignment


def verify_foreign_union_layout(codegen, node, contract):
    """Refuse the union ``node`` unless Metal gives it, and each member, the layout ``contract`` records."""
    source_abi = contract["source_abi"]
    members = {
        getattr(member, "name", ""): member
        for member in getattr(node, "members", None) or ()
    }
    largest_size, largest_alignment = 0, 1
    for name, recorded in contract["members"].items():
        member = members.get(name)
        layout = (
            _member_layout(codegen, member, frozenset({node.name}))
            if member is not None
            else None
        )
        if layout is None:
            raise codegen.metal_union_layout_error(
                node,
                "member-layout-unknown",
                f"its {source_abi} layout cannot be compared with Metal's: the member's Metal layout is not known",
                member,
            )
        if layout != (recorded["size"], recorded["alignment"]):
            raise codegen.metal_union_layout_error(
                node,
                "member-layout-mismatch",
                f"it is {recorded['size']} bytes aligned to {recorded['alignment']} in the {source_abi} layout, "
                f"but {layout[0]} bytes aligned to {layout[1]} in Metal",
                member,
            )
        largest_size = max(largest_size, layout[0])
        largest_alignment = max(largest_alignment, layout[1])
    expected = (_round(largest_size, largest_alignment), largest_alignment)
    if expected != (contract["size"], contract["alignment"]):
        raise codegen.metal_union_layout_error(
            node,
            "union-layout-mismatch",
            f"it is {contract['size']} bytes aligned to {contract['alignment']} in the {source_abi} layout, "
            f"but {expected[0]} bytes aligned to {expected[1]} in Metal",
        )
