"""Physical layouts for fixed std140 scalar/vector parameter blocks."""

from __future__ import annotations

import re
import struct
from typing import Any, Mapping, Sequence

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_SCALARS = {
    "float": ("float32", 4),
    "int": ("int32", 4),
    "uint": ("uint32", 4),
    "bool": ("uint32", 4),
    "int64_t": ("int64", 8),
    "uint64_t": ("uint64", 8),
}
_VECTORS = {
    "vec": "float",
    "ivec": "int",
    "uvec": "uint",
    "bvec": "bool",
    "i64vec": "int64_t",
    "u64vec": "uint64_t",
}
_PACK_CODES = {
    "float": "f",
    "int": "i",
    "uint": "I",
    "bool": "I",
    "int64_t": "q",
    "uint64_t": "Q",
}
MAX_UNIFORM_ARRAY_LENGTH = 4096
# Each storage layout names the target declaration packing it describes.
UNIFORM_BLOCK_STORAGE_LAYOUTS = {"opengl": "std140", "directx": "hlsl-constant-buffer"}


def _member_declaration(declaration: Any) -> tuple[Any, Any, int | None]:
    if not isinstance(declaration, (tuple, list)) or len(declaration) not in (2, 3):
        raise ValueError(
            "Uniform members require (type, name) or (type, name, length)."
        )
    length = declaration[2] if len(declaration) == 3 else None
    if length is not None and (
        type(length) is not int or not 1 <= length <= MAX_UNIFORM_ARRAY_LENGTH
    ):
        raise ValueError("Uniform member arrays require 1-4096 elements.")
    return declaration[0], declaration[1], length


def _checked_name(name: Any, names: set[str]) -> str:
    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) or name in names:
        raise ValueError("Uniform member names must be unique identifiers.")
    names.add(name)
    return name


def std140_block_layout(
    type_name: str,
    declarations: Sequence[tuple[str, str] | tuple[str, str, int | None]],
) -> dict[str, Any]:
    """Describe target declarations, not source-language struct packing.

    The payload is little-endian uint32 transport, including all padding. Member
    records retain the actual shader types; words are not numeric conversions.
    Fixed-length arrays of scalars and vectors follow std140: their alignment and
    element stride round up to 16 bytes, and members carry ``arrayLength`` and
    ``arrayStrideBytes``. Matrices, nested structures and explicit member layouts
    are rejected.
    """
    if not isinstance(type_name, str) or not _IDENTIFIER.fullmatch(type_name):
        raise ValueError("Uniform block type must be an identifier.")
    if not declarations:
        raise ValueError("Uniform blocks require at least one member.")
    members = []
    names = set()
    offset = 0
    alignment = 16
    for declaration in declarations:
        physical_type, name, length = _member_declaration(declaration)
        _checked_name(name, names)
        if not isinstance(physical_type, str):
            raise ValueError("Uniform member type must be a scalar or vector.")
        base_type, width = physical_type, 1
        vector = re.fullmatch(
            r"(vec|ivec|uvec|bvec|i64vec|u64vec)([2-4])", physical_type
        )
        if vector:
            base_type, width = _VECTORS[vector[1]], int(vector[2])
        if base_type not in _SCALARS:
            raise ValueError(f"Unsupported uniform member type: {physical_type}.")
        element_type, scalar_size = _SCALARS[base_type]
        member_alignment = scalar_size * (4 if width == 3 else width)
        size = scalar_size * width
        member = {
            "name": name,
            "physicalType": physical_type,
            "elementType": element_type,
            "vectorWidth": width,
        }
        if length is not None:
            member_alignment = _align(member_alignment, 16)
            stride = _align(size, member_alignment)
            size = stride * length
        offset = _align(offset, member_alignment)
        member.update(
            {
                "offsetBytes": offset,
                "sizeBytes": size,
                "alignmentBytes": member_alignment,
            }
        )
        if length is not None:
            member.update({"arrayLength": length, "arrayStrideBytes": stride})
        members.append(member)
        offset += size
        alignment = max(alignment, member_alignment)
    return _block_record(type_name, members, "std140", alignment, offset)


def hlsl_constant_buffer_block_layout(
    type_name: str,
    declarations: Sequence[tuple[str, str] | tuple[str, str, int | None]],
) -> dict[str, Any]:
    """Describe an HLSL ``ConstantBuffer<T>`` of scalars and scalar arrays.

    Legacy constant-buffer packing places scalars at their natural alignment
    without crossing a 16-byte register, starts every array element on a new
    register, and lets the next member share the last element's register.
    ``bool`` occupies one 32-bit word. Vectors, matrices and nested structures
    are rejected.
    """
    if not isinstance(type_name, str) or not _IDENTIFIER.fullmatch(type_name):
        raise ValueError("Constant buffer type must be an identifier.")
    if not declarations:
        raise ValueError("Constant buffers require at least one member.")
    members = []
    names = set()
    offset = 0
    for declaration in declarations:
        physical_type, name, length = _member_declaration(declaration)
        _checked_name(name, names)
        if not isinstance(physical_type, str) or physical_type not in _SCALARS:
            raise ValueError(
                f"Unsupported constant buffer member type: {physical_type}."
            )
        element_type, scalar_size = _SCALARS[physical_type]
        member = {
            "name": name,
            "physicalType": physical_type,
            "elementType": element_type,
            "vectorWidth": 1,
        }
        if length is None:
            # Natural alignment of 4- and 8-byte scalars never crosses a register.
            offset = _align(offset, scalar_size)
            member_alignment, size = scalar_size, scalar_size
        else:
            offset = _align(offset, 16)
            member_alignment, size = 16, 16 * (length - 1) + scalar_size
        member.update(
            {
                "offsetBytes": offset,
                "sizeBytes": size,
                "alignmentBytes": member_alignment,
            }
        )
        if length is not None:
            member.update({"arrayLength": length, "arrayStrideBytes": 16})
        members.append(member)
        offset += size
    return _block_record(type_name, members, "hlsl-constant-buffer", 16, offset)


def _block_record(type_name, members, storage_layout, alignment, offset):
    return {
        "physicalType": type_name,
        "payloadEncoding": "uint32-le-words",
        "elementType": "uint32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": alignment,
        "memberOffsetBytes": 0,
        "storageLayout": storage_layout,
        "runtimeSized": False,
        "blockSizeBytes": _align(offset, alignment),
        "blockMembers": members,
    }


def _expected_block_layout(layout: Mapping[str, Any], storage_layout: str):
    members = layout.get("blockMembers")
    if not isinstance(members, list) or not all(
        isinstance(member, Mapping) for member in members
    ):
        raise ValueError("Uniform block members must be a list of member records.")
    declarations = [
        (
            (member.get("physicalType"), member.get("name"), member.get("arrayLength"))
            if "arrayLength" in member
            else (member.get("physicalType"), member.get("name"))
        )
        for member in members
    ]
    builder = (
        std140_block_layout
        if storage_layout == "std140"
        else hlsl_constant_buffer_block_layout
    )
    return builder(layout.get("physicalType"), declarations)


def validate_std140_block_layout(layout: Mapping[str, Any]) -> int:
    """Recompute packing and reject missing, extra or contradictory metadata."""
    expected = _expected_block_layout(layout, "std140")
    if not _exact_layout(layout, expected):
        raise ValueError("Uniform block metadata does not match std140 member packing.")
    return expected["blockSizeBytes"]


def validate_uniform_block_layout(layout: Mapping[str, Any]) -> int:
    """Validate a std140 block or an HLSL constant buffer by its storage layout."""
    storage_layout = layout.get("storageLayout")
    if storage_layout not in UNIFORM_BLOCK_STORAGE_LAYOUTS.values():
        raise ValueError("Uniform block storage layout is not supported.")
    expected = _expected_block_layout(layout, storage_layout)
    if not _exact_layout(layout, expected):
        raise ValueError(
            f"Uniform block metadata does not match {storage_layout} member packing."
        )
    return expected["blockSizeBytes"]


def _pack_scalar(payload: bytearray, offset: int, member, base_type: str, value):
    if base_type == "bool":
        if type(value) is not bool:
            raise ValueError(f"Uniform member {member['name']} requires bool values.")
        value = int(value)
    elif (base_type == "float" and type(value) not in (int, float)) or (
        base_type != "float" and type(value) is not int
    ):
        raise ValueError(
            f"Uniform member {member['name']} requires {member['elementType']} values."
        )
    try:
        struct.pack_into("<" + _PACK_CODES[base_type], payload, offset, value)
    except (OverflowError, struct.error) as exc:
        raise ValueError(
            f"Uniform member {member['name']} is outside {member['elementType']} storage."
        ) from exc


def _pack_element(payload: bytearray, offset: int, member, value):
    physical_type, width = member["physicalType"], member["vectorWidth"]
    vector = re.fullmatch(r"(vec|ivec|uvec|bvec|i64vec|u64vec)([2-4])", physical_type)
    base_type = _VECTORS[vector[1]] if vector else physical_type
    if width == 1:
        _pack_scalar(payload, offset, member, base_type, value)
        return
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(
            f"Uniform member {member['name']} requires {width} components."
        )
    size = _SCALARS[base_type][1]
    for component, item in enumerate(value):
        _pack_scalar(payload, offset + component * size, member, base_type, item)


def pack_uniform_block(
    layout: Mapping[str, Any], values: Mapping[str, Any]
) -> list[int]:
    """Pack named member values into a validated block's uint32 words.

    Scalars take Python numbers (``bool`` members take booleans), vectors take
    ``vectorWidth`` components and arrays take exactly ``arrayLength``
    elements. Padding is zero. Use the words with dtype='uint32'.
    """
    size = validate_uniform_block_layout(layout)
    members = layout["blockMembers"]
    if not isinstance(values, Mapping) or values.keys() != {
        member["name"] for member in members
    }:
        raise ValueError(
            "Uniform block values must supply exactly the reflected members."
        )
    payload = bytearray(size)
    for member in members:
        value = values[member["name"]]
        length = member.get("arrayLength")
        if length is None:
            _pack_element(payload, member["offsetBytes"], member, value)
            continue
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise ValueError(
                f"Uniform member {member['name']} requires {length} elements."
            )
        for index, element in enumerate(value):
            _pack_element(
                payload,
                member["offsetBytes"] + index * member["arrayStrideBytes"],
                member,
                element,
            )
    return list(struct.unpack(f"<{size // 4}I", payload))


def _exact_layout(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, Mapping)
            and actual.keys() == expected.keys()
            and all(
                _exact_layout(actual[key], value) for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_exact_layout(left, right) for left, right in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _align(size: int, alignment: int) -> int:
    return (size + alignment - 1) // alignment * alignment
