"""Matrix and array fields of structures nested in Metal stage-interface structures.

Metal accepts a nested structure as a member of a ``[[stage_in]]`` structure or of a vertex function's return
structure, but every member of it must itself be a scalar, a vector or another such structure: a matrix or an array
anywhere in the nesting makes the outer structure invalid ("invalid return type ... for vertex function", "type ...
is not valid for attribute 'stage_in'"). The emitter already splits a matrix or array member of a stage-interface
structure into one field per column or element (``metal_stage_io_member_lowering``); a CrossGL struct varying, which
the GLSL import produces for an instance interface block (``out Data { mat2 basis; } data;``), nests the matrix one
level down, so the split never applied and the translation did not compile.

The split is applied to every structure reachable from a stage-interface structure through its members, keyed by
structure name like the existing lowering, so declarations, member reads and writes, and constructor calls of that
structure are rewritten wherever it is used. A split field has the size, alignment and offset of the column or
element it replaces (Metal Shading Language Specification, section 2.2, "Matrix Data Types", and section 2.12,
"Arrays"), so a value of the structure keeps its memory layout. Nested structures that the emitter lays out from an
explicit plan (HLSL buffer layouts, unions, ``alignas``) keep their declaration, since the split does not apply to
their emission.
"""

from __future__ import annotations

from ..ast import StructNode
from .array_utils import split_array_type_suffix

__all__ = ["collect_metal_nested_stage_io_member_lowerings"]


def _member_struct_name(codegen, member, structs_by_name):
    """The user structure a member holds, directly or as array elements, else None."""
    raw_type = codegen.struct_member_raw_type(member)
    if not raw_type:
        return None
    base_type, _suffix = split_array_type_suffix(str(raw_type))
    while "[" in base_type:
        base_type, _suffix = split_array_type_suffix(base_type)
    base_type = base_type.strip()
    return base_type if base_type in structs_by_name else None


def _has_explicit_layout(codegen, struct) -> bool:
    name = struct.name
    if name in (getattr(codegen, "metal_hlsl_struct_plans", None) or {}):
        return True
    if name in (getattr(codegen, "metal_union_layouts", None) or {}):
        return True
    if name in (getattr(codegen, "lowered_glsl_buffer_block_struct_names", None) or ()):
        return True
    if name in (getattr(codegen, "glsl_buffer_block_struct_names", None) or ()):
        return True
    return codegen.metal_struct_alignment(struct) is not None


def collect_metal_nested_stage_io_member_lowerings(
    codegen, structs, stage_io_names, lowerings
):
    """Add split-member lowerings for structures nested in ``stage_io_names`` structures to ``lowerings``.

    ``lowerings`` maps a structure name to ``{member name: lowering}``; it is updated in place and returned.
    Traversal is in declaration order so the result does not depend on set iteration order.
    """
    structs_by_name = {
        struct.name: struct
        for struct in structs or []
        if isinstance(struct, StructNode) and getattr(struct, "name", None)
    }
    pending = [
        struct.name
        for struct in structs_by_name.values()
        if struct.name in stage_io_names
    ]
    visited = set(pending)
    while pending:
        struct = structs_by_name[pending.pop(0)]
        for member in getattr(struct, "members", []) or []:
            nested_name = _member_struct_name(codegen, member, structs_by_name)
            if nested_name is None or nested_name in visited:
                continue
            visited.add(nested_name)
            nested = structs_by_name[nested_name]
            if _has_explicit_layout(codegen, nested):
                continue
            pending.append(nested_name)
            for nested_member in getattr(nested, "members", []) or []:
                lowering = codegen.metal_stage_io_member_lowering(
                    nested_member, {}, nested_name
                )
                if lowering is not None:
                    lowerings.setdefault(nested_name, {})[nested_member.name] = lowering
    return lowerings
