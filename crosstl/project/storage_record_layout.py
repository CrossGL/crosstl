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
_STORAGE_LAYOUTS = {"metal-buffer", "hlsl-structured-buffer", "std430"}


def storage_record_layout(
    type_name: str,
    declarations: Sequence[tuple[str, str]],
    *,
    storage_layout: str,
) -> dict[str, Any]:
    """Describe target record arrays, not uniform-block or vector packing.

    These scalar types have natural field alignment in all three supported
    storage layouts. Record strides include trailing alignment padding.
    Transport words preserve physical bits; they are not record member types.
    """
    if not isinstance(type_name, str) or not _IDENTIFIER.fullmatch(type_name):
        raise ValueError("Storage record type must be an identifier.")
    if (
        type_name in _SCALARS
        or not isinstance(storage_layout, str)
        or storage_layout not in _STORAGE_LAYOUTS
    ):
        raise ValueError("Unsupported storage record type or storage layout.")
    if not 1 <= len(declarations) <= 64:
        raise ValueError("Storage records require 1-64 scalar fields.")
    members = []
    names = set()
    offset, alignment = 0, 1
    for physical_type, name in declarations:
        if not isinstance(physical_type, str) or physical_type not in _SCALARS:
            raise ValueError(
                "Storage record fields require supported 32/64-bit scalars."
            )
        if (
            not isinstance(name, str)
            or not _IDENTIFIER.fullmatch(name)
            or name in names
        ):
            raise ValueError("Storage record field names must be unique identifiers.")
        names.add(name)
        dtype, size, _ = _SCALARS[physical_type]
        offset = (offset + size - 1) // size * size
        members.append(
            {
                "name": name,
                "physicalType": physical_type,
                "elementType": dtype,
                "offsetBytes": offset,
                "sizeBytes": size,
                "alignmentBytes": size,
            }
        )
        offset += size
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
        [(member.get("physicalType"), member.get("name")) for member in members],
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


def pack_storage_records(
    layout: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> list[int]:
    """Pack named records into little-endian uint32 words with zeroed padding.

    Use the words with dtype='uint32' and shape=(record_count, stride // 4).
    Callers needing particular floating-point bit patterns may supply physical
    words directly through the same validated reflected layout.
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
            dtype, _, code = _SCALARS[member["physicalType"]]
            if (dtype == "float32" and type(value) not in (int, float)) or (
                dtype != "float32" and type(value) is not int
            ):
                raise ValueError(
                    f"Storage field {member['name']} requires a {dtype} value."
                )
            try:
                struct.pack_into(
                    "<" + code, payload, index * stride + member["offsetBytes"], value
                )
            except (OverflowError, struct.error) as exc:
                raise ValueError(
                    f"Storage field {member['name']} is outside {dtype} storage."
                ) from exc
    return list(struct.unpack("<" + "I" * (len(payload) // 4), payload))
