"""Physical layouts and packing for flat mixed-scalar storage records."""

from __future__ import annotations

import re
import struct
from typing import Any, Mapping, Sequence

from .buffer_requirements import valid_minimum_binding_size
from .uniform_layout import _exact_layout

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_SCALARS = {
    "float": ("float32", 4, "f"),
    "int": ("int32", 4, "i"),
    "uint": ("uint32", 4, "I"),
    "int64_t": ("int64", 8, "q"),
    "uint64_t": ("uint64", 8, "Q"),
}
# Metal stores ``bool`` in one byte. HLSL structured buffers and std430 blocks
# store it in one 32-bit word, and any nonzero word reads as true.
_BOOLEAN_SIZES = {"metal-buffer": 1, "hlsl-structured-buffer": 4, "std430": 4}
_STORAGE_LAYOUTS = frozenset(_BOOLEAN_SIZES)
MAX_STORAGE_RECORD_ARRAY_LENGTH = 4096


def _field_declaration(declaration: Any) -> tuple[Any, Any, int | None]:
    if not isinstance(declaration, (tuple, list)) or len(declaration) not in (2, 3):
        raise ValueError(
            "Storage record fields require (type, name) or (type, name, length)."
        )
    physical_type, name = declaration[0], declaration[1]
    length = declaration[2] if len(declaration) == 3 else None
    if length is not None and (
        type(length) is not int
        or not 1 <= length <= MAX_STORAGE_RECORD_ARRAY_LENGTH
        or physical_type == "bool"
    ):
        raise ValueError(
            "Storage record arrays require 1-4096 32/64-bit scalar elements."
        )
    return physical_type, name, length


def storage_record_layout(
    type_name: str,
    declarations: Sequence[tuple[str, str] | tuple[str, str, int | None]],
    *,
    storage_layout: str,
) -> dict[str, Any]:
    """Describe target record arrays, not uniform-block or vector packing.

    Fields are 32/64-bit scalars, fixed-length arrays of them, or ``bool``.
    These have natural alignment in all three supported storage layouts, and
    array elements are tightly strided. Only ``bool`` differs by layout: one
    byte in Metal buffers, one 32-bit word in HLSL structured buffers and
    std430 blocks. Record strides include trailing alignment padding.
    Transport words preserve physical bits; they are not record member types.
    """
    if not isinstance(type_name, str) or not _IDENTIFIER.fullmatch(type_name):
        raise ValueError("Storage record type must be an identifier.")
    if (
        type_name in _SCALARS
        or type_name == "bool"
        or not isinstance(storage_layout, str)
        or storage_layout not in _STORAGE_LAYOUTS
    ):
        raise ValueError("Unsupported storage record type or storage layout.")
    if not 1 <= len(declarations) <= 64:
        raise ValueError("Storage records require 1-64 scalar fields.")
    members = []
    names = set()
    offset, alignment = 0, 1
    for declaration in declarations:
        physical_type, name, length = _field_declaration(declaration)
        if physical_type == "bool":
            dtype, size = "bool", _BOOLEAN_SIZES[storage_layout]
        elif isinstance(physical_type, str) and physical_type in _SCALARS:
            dtype, size, _ = _SCALARS[physical_type]
        else:
            raise ValueError(
                "Storage record fields require supported 32/64-bit scalars or bool."
            )
        if (
            not isinstance(name, str)
            or not _IDENTIFIER.fullmatch(name)
            or name in names
        ):
            raise ValueError("Storage record field names must be unique identifiers.")
        names.add(name)
        offset = (offset + size - 1) // size * size
        member = {
            "name": name,
            "physicalType": physical_type,
            "elementType": dtype,
            "offsetBytes": offset,
            "sizeBytes": size * (length or 1),
            "alignmentBytes": size,
        }
        if length is not None:
            member["arrayLength"] = length
            member["arrayStrideBytes"] = size
        members.append(member)
        offset += member["sizeBytes"]
        alignment = max(alignment, size)
    stride = (offset + alignment - 1) // alignment * alignment
    return {
        "physicalType": type_name,
        "elementType": "record",
        "payloadEncoding": "uint32-le-words",
        "elementSizeBytes": stride,
        "elementStrideBytes": stride,
        "alignmentBytes": alignment,
        "memberOffsetBytes": 0,
        "storageLayout": storage_layout,
        "runtimeSized": True,
        "structMembers": members,
    }


def validate_storage_record_layout(layout: Mapping[str, Any]) -> int:
    """Recompute every member offset, scalar representation and record stride."""
    members = layout.get("structMembers")
    if not isinstance(members, list) or not all(
        isinstance(m, Mapping) for m in members
    ):
        raise ValueError("Storage record members must be a list of field records.")
    expected = storage_record_layout(
        layout.get("physicalType"),
        [
            (
                member.get("physicalType"),
                member.get("name"),
                member.get("arrayLength"),
            )
            for member in members
        ],
        storage_layout=layout.get("storageLayout"),
    )
    if "memberName" in layout:
        name = layout["memberName"]
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise ValueError("Storage array member name must be an identifier.")
        expected["memberName"] = name
    if "minimumBindingSizeBytes" in layout:
        minimum = layout["minimumBindingSizeBytes"]
        if not valid_minimum_binding_size(minimum):
            raise ValueError(
                "Minimum storage binding size must be a positive byte count."
            )
        expected["minimumBindingSizeBytes"] = minimum
    if not _exact_layout(layout, expected):
        raise ValueError(
            "Storage record metadata does not match target scalar packing."
        )
    return expected["elementStrideBytes"]


def _pack_field(payload: bytearray, offset: int, member: Mapping[str, Any], value):
    dtype = member["elementType"]
    if dtype == "bool":
        if type(value) is not bool:
            raise ValueError(f"Storage field {member['name']} requires a bool value.")
        code = "B" if member["alignmentBytes"] == 1 else "I"
        struct.pack_into("<" + code, payload, offset, int(value))
        return
    _, _, code = _SCALARS[member["physicalType"]]
    if (dtype == "float32" and type(value) not in (int, float)) or (
        dtype != "float32" and type(value) is not int
    ):
        raise ValueError(f"Storage field {member['name']} requires a {dtype} value.")
    try:
        struct.pack_into("<" + code, payload, offset, value)
    except (OverflowError, struct.error) as exc:
        raise ValueError(
            f"Storage field {member['name']} is outside {dtype} storage."
        ) from exc


def pack_storage_records(
    layout: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> list[int]:
    """Pack named records into little-endian uint32 words with zeroed padding.

    Array fields take a list or tuple with exactly ``arrayLength`` values and
    ``bool`` fields take Python booleans, stored as 0 or 1 in their reflected
    width. Use the words with dtype='uint32' and shape=(record_count,
    stride // 4). Callers needing particular floating-point bit patterns may
    supply physical words directly through the same validated reflected layout.
    """
    stride = validate_storage_record_layout(layout)
    if not isinstance(records, Sequence) or isinstance(
        records, (str, bytes, bytearray)
    ):
        raise ValueError("Storage record values must be a sequence of field mappings.")
    if not records:
        raise ValueError("Storage record values must not be empty.")
    members = layout["structMembers"]
    names = {member["name"] for member in members}
    payload = bytearray(stride * len(records))
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.keys() != names:
            raise ValueError(
                f"Storage record {index} must supply exactly the reflected fields."
            )
        for member in members:
            value = record[member["name"]]
            offset = index * stride + member["offsetBytes"]
            length = member.get("arrayLength")
            if length is None:
                _pack_field(payload, offset, member, value)
                continue
            if not isinstance(value, (list, tuple)) or len(value) != length:
                raise ValueError(
                    f"Storage field {member['name']} requires {length} values."
                )
            for position, element in enumerate(value):
                _pack_field(
                    payload,
                    offset + position * member["arrayStrideBytes"],
                    member,
                    element,
                )
    return list(struct.unpack("<" + "I" * (len(payload) // 4), payload))
