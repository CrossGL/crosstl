"""Physical layouts for fixed std140 scalar/vector parameter blocks."""

from __future__ import annotations

import re
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


def std140_block_layout(
    type_name: str, declarations: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    """Describe target declarations, not source-language struct packing.

    The payload is little-endian uint32 transport, including all padding. Member
    records retain the actual shader types; words are not numeric conversions.
    Arrays, matrices, nested structures and explicit member layouts are rejected.
    """
    if not isinstance(type_name, str) or not _IDENTIFIER.fullmatch(type_name):
        raise ValueError("Uniform block type must be an identifier.")
    if not declarations:
        raise ValueError("Uniform blocks require at least one member.")
    members = []
    names = set()
    offset = 0
    alignment = 16
    for physical_type, name in declarations:
        if (
            not isinstance(name, str)
            or not _IDENTIFIER.fullmatch(name)
            or name in names
        ):
            raise ValueError("Uniform member names must be unique identifiers.")
        names.add(name)
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
        offset = _align(offset, member_alignment)
        size = scalar_size * width
        members.append(
            {
                "name": name,
                "physicalType": physical_type,
                "elementType": element_type,
                "vectorWidth": width,
                "offsetBytes": offset,
                "sizeBytes": size,
                "alignmentBytes": member_alignment,
            }
        )
        offset += size
        alignment = max(alignment, member_alignment)
    return {
        "physicalType": type_name,
        "payloadEncoding": "uint32-le-words",
        "elementType": "uint32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": alignment,
        "memberOffsetBytes": 0,
        "storageLayout": "std140",
        "runtimeSized": False,
        "blockSizeBytes": _align(offset, alignment),
        "blockMembers": members,
    }


def validate_std140_block_layout(layout: Mapping[str, Any]) -> int:
    """Recompute packing and reject missing, extra or contradictory metadata."""
    members = layout.get("blockMembers")
    if not isinstance(members, list) or not all(
        isinstance(member, Mapping) for member in members
    ):
        raise ValueError("Uniform block members must be a list of member records.")
    expected = std140_block_layout(
        layout.get("physicalType"),
        [(member.get("physicalType"), member.get("name")) for member in members],
    )
    if not _exact_layout(layout, expected):
        raise ValueError("Uniform block metadata does not match std140 member packing.")
    return expected["blockSizeBytes"]


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
