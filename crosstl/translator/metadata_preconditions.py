"""Source-owned integer metadata contracts for checked native dispatch."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .arithmetic_conversions import ArithmeticScalarKind, source_integer_shape
from .artifact_contracts import (
    METADATA_PRECONDITION_PREFIX,
    artifact_contract_headers,
    with_artifact_contract_header,
)
from .ast import NamedType, PointerType, PrimitiveType, ReferenceType
from .resource_identity import source_resource_attribute, source_resource_identity

MAX_METADATA_PRECONDITIONS = 256
MAX_METADATA_PRECONDITION_BYTES = 65536
INTEGER_METADATA_TYPES = {
    f"{prefix}int{bits}": (
        0 if prefix else -(1 << (bits - 1)),
        (1 << (bits if prefix else bits - 1)) - 1,
    )
    for prefix in ("", "u")
    for bits in (8, 16, 32, 64)
}
INTEGER_METADATA_BYTES = {
    f"{prefix}int{bits}": bits // 8 for prefix in ("", "u") for bits in (8, 16, 32, 64)
}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class MetadataPreconditionError(ValueError):
    project_diagnostic_code = "project.translate.metadata-precondition-unsupported"


def _identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 1024
        or not _IDENTIFIER.fullmatch(value)
    ):
        raise ValueError(f"Metadata precondition {field} requires an identifier.")
    return value


def _bounds(minimum: Any, maximum: Any) -> None:
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or not -(1 << 63) <= minimum <= maximum <= (1 << 64) - 1
    ):
        raise ValueError(
            "Metadata precondition bounds require an ordered integer range."
        )


@dataclass(frozen=True)
class MetadataRangePrecondition:
    """A requirement on a scalar input, one immutable array element, or one
    integer field of an immutable constant record."""

    entry_point: str
    parameter: str
    minimum: int
    maximum: int
    source: str = "*"
    element: int | None = None
    member: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.entry_point, "entry_point")
        _identifier(self.parameter, "parameter")
        _bounds(self.minimum, self.maximum)
        _element(self.element)
        _member(self.member, self.element)
        if (
            not isinstance(self.source, str)
            or not 1 <= len(self.source) <= 4096
            or self.source != self.source.strip()
            or any(ord(char) < 32 for char in self.source)
        ):
            raise ValueError(
                "Metadata precondition source requires a nonempty pattern."
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "entryPoint": self.entry_point,
            "parameter": self.parameter,
            "minimum": self.minimum,
            "maximum": self.maximum,
            **({"element": self.element} if self.element is not None else {}),
            **({"member": self.member} if self.member is not None else {}),
        }


def _element(value: Any) -> None:
    """Keep the byte offset representable for every supported integer width."""
    if value is not None and (type(value) is not int or not 0 <= value < (1 << 61)):
        raise ValueError("Metadata element requires a nonnegative bounded integer.")


def _member(value: Any, element: Any) -> None:
    """A member names one field of the record; it cannot also index an array."""
    if value is None:
        return
    _identifier(value, "member")
    if element is not None:
        raise ValueError("Metadata precondition selects an element or a member.")


def _selector(element, member):
    """One requirement's position within its resource."""
    if member is not None:
        return ("member", member)
    if element is not None:
        return ("element", element)
    return None


def _add_selector(selectors, identity, selector):
    """A resource takes one whole-scalar requirement, or distinct elements, or
    distinct record members, never a mixture of these."""
    previous = selectors.setdefault(identity, set())
    kinds = {None if item is None else item[0] for item in previous}
    kind = None if selector is None else selector[0]
    if selector in previous or previous and (kind is None or kinds != {kind}):
        raise ValueError(
            "Duplicate or ambiguous metadata source parameter requirements."
        )
    previous.add(selector)


def parse_metadata_preconditions(
    records: Sequence[Any] | None, *, field_name: str = "metadata_preconditions"
) -> tuple[MetadataRangePrecondition, ...]:
    """Reject unknown or ambiguous requirements before source translation."""
    if records is None:
        return ()
    if (
        not isinstance(records, (list, tuple))
        or len(records) > MAX_METADATA_PRECONDITIONS
    ):
        raise ValueError(
            f"{field_name} requires at most {MAX_METADATA_PRECONDITIONS} records."
        )
    result, keys = [], {}
    for record in records:
        if isinstance(record, MetadataRangePrecondition):
            condition = record
        elif isinstance(record, Mapping):
            if set(record) - {
                "source",
                "entryPoint",
                "parameter",
                "minimum",
                "maximum",
                "element",
                "member",
            } or not {"entryPoint", "parameter", "minimum", "maximum"} <= set(record):
                raise ValueError(f"{field_name} has unsupported or missing fields.")
            condition = MetadataRangePrecondition(
                entry_point=record["entryPoint"],
                parameter=record["parameter"],
                minimum=record["minimum"],
                maximum=record["maximum"],
                source=record.get("source", "*"),
                element=record.get("element"),
                member=record.get("member"),
            )
            if "element" in record and record["element"] is None:
                raise ValueError("Explicit metadata element cannot be null.")
            if "member" in record and record["member"] is None:
                raise ValueError("Explicit metadata member cannot be null.")
        else:
            raise ValueError(f"{field_name} requires metadata precondition objects.")
        key = (condition.source, condition.entry_point, condition.parameter)
        _add_selector(keys, key, _selector(condition.element, condition.member))
        result.append(condition)
    return tuple(result)


def checked_metadata_preconditions(records: Any) -> list[dict[str, Any]]:
    """Validate resolved contracts without trusting generated resource names."""
    if (
        not isinstance(records, (list, tuple))
        or len(records) > MAX_METADATA_PRECONDITIONS
    ):
        raise ValueError("Resolved metadata preconditions require a bounded sequence.")
    result, indices, names, selectors, types = [], {}, {}, {}, {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) - {"element", "member"} != {
            "sourceResource",
            "dtype",
            "minimum",
            "maximum",
        }:
            raise ValueError("Unsupported resolved metadata precondition fields.")
        element = record.get("element")
        _element(element)
        if "element" in record and element is None:
            raise ValueError("Explicit metadata element cannot be null.")
        member = record.get("member")
        if "member" in record and member is None:
            raise ValueError("Explicit metadata member cannot be null.")
        _member(member, element)
        source = record["sourceResource"]
        if (
            not isinstance(source, Mapping)
            or set(source)
            != {"schemaVersion", "backend", "entryPoint", "parameter", "parameterIndex"}
            or type(source["schemaVersion"]) is not int
            or source["schemaVersion"] != 1
        ):
            raise ValueError("Unsupported metadata source resource identity.")
        source_resource_attribute(
            source["backend"],
            source["entryPoint"],
            source["parameter"],
            source["parameterIndex"],
        )
        _bounds(record["minimum"], record["maximum"])
        dtype = record["dtype"]
        if not isinstance(dtype, str) or dtype not in INTEGER_METADATA_TYPES:
            raise ValueError("Metadata preconditions require an integer scalar type.")
        low, high = INTEGER_METADATA_TYPES[dtype]
        if not low <= record["minimum"] <= record["maximum"] <= high:
            raise ValueError(
                "Metadata precondition bounds exceed the source integer type."
            )
        index = (source["backend"], source["entryPoint"], source["parameterIndex"])
        name = (source["backend"], source["entryPoint"], source["parameter"])
        if indices.get(index, name) != name or names.get(name, index) != index:
            raise ValueError(
                "Duplicate or ambiguous metadata source resource identity."
            )
        indices[index] = name
        names[name] = index
        # Array elements share one element type; record fields have their own.
        type_key = index if member is None else (index, member)
        if types.get(type_key, dtype) != dtype:
            raise ValueError("Metadata elements require one source integer type.")
        types[type_key] = dtype
        _add_selector(selectors, index, _selector(element, member))
        result.append({**record, "sourceResource": dict(source)})
    return sorted(
        result,
        key=lambda item: (
            item["sourceResource"]["backend"],
            item["sourceResource"]["entryPoint"],
            item["sourceResource"]["parameterIndex"],
            item.get("element", -1),
            item.get("member", ""),
        ),
    )


def metadata_precondition_header(records: Any) -> str:
    """Serialize checked obligations, not caller-trusted index assertions."""
    normalized = checked_metadata_preconditions(records)
    if not normalized:
        return ""
    header = METADATA_PRECONDITION_PREFIX + json.dumps(
        {"schemaVersion": 1, "requirements": normalized},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if len(header.encode("utf-8")) > MAX_METADATA_PRECONDITION_BYTES:
        raise ValueError("Metadata precondition header exceeds its byte limit.")
    return header + "\n"


def with_metadata_precondition_header(source: str, records: Any) -> str:
    """Retain every existing leading artifact contract when adding obligations."""
    if METADATA_PRECONDITION_PREFIX.rstrip() in source:
        raise ValueError("Source already contains a metadata precondition marker.")
    return with_artifact_contract_header(source, metadata_precondition_header(records))


def parse_metadata_precondition_header(source: str) -> list[dict[str, Any]]:
    """Require one canonical header; reject markers hidden in source text."""
    line = artifact_contract_headers(source).get(METADATA_PRECONDITION_PREFIX)
    if line is None:
        return []
    if len(line.encode("utf-8")) > MAX_METADATA_PRECONDITION_BYTES:
        raise ValueError("Metadata precondition header exceeds its byte limit.")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate metadata precondition key.")
            result[key] = value
        return result

    try:
        record = json.loads(
            line[len(METADATA_PRECONDITION_PREFIX) :],
            object_pairs_hook=unique,
        )
    except (ValueError, RecursionError) as error:
        raise ValueError("Invalid metadata precondition JSON.") from error
    if (
        not isinstance(record, dict)
        or set(record) != {"schemaVersion", "requirements"}
        or type(record["schemaVersion"]) is not int
        or record["schemaVersion"] != 1
        or not isinstance(record["requirements"], list)
        or not record["requirements"]
    ):
        raise ValueError("Unsupported metadata precondition header schema.")
    return checked_metadata_preconditions(record["requirements"])


def _record_field(reference, member, structs):
    """The one declared integer field ``member`` of a constant record reference."""
    if (
        not isinstance(reference, ReferenceType)
        or reference.address_space != "constant"
        or reference.is_mutable
    ):
        raise MetadataPreconditionError(
            "Metadata members require a constant record reference."
        )
    record = reference.referenced_type
    declarations = [
        struct
        for struct in structs or ()
        if isinstance(record, NamedType)
        and not record.generic_args
        and getattr(struct, "name", None) == record.name
    ]
    if len(declarations) != 1 or getattr(declarations[0], "generic_params", None):
        raise MetadataPreconditionError(
            "Metadata members require one declared non-generic record type."
        )
    fields = [
        field
        for field in getattr(declarations[0], "members", None) or ()
        if getattr(field, "name", None) == member
    ]
    if len(fields) != 1:
        raise MetadataPreconditionError(
            f"Metadata member {member} is not one field of {record.name}."
        )
    field = fields[0]
    field_type = getattr(field, "member_type", None) or getattr(field, "var_type", None)
    # ``int C;`` parses as a primitive, ``int64 C;`` as a named scalar; arrays,
    # vectors and nested records do not qualify.
    if not (
        isinstance(field_type, PrimitiveType)
        and field_type.size_bits is None
        or isinstance(field_type, NamedType)
        and not field_type.generic_args
    ):
        raise MetadataPreconditionError(
            "Metadata members require a supported integer scalar field."
        )
    return field, field_type


def resolve_metadata_preconditions(function, requirements, structs=()):
    """Resolve original names against one selected entry's constant references.

    ``structs`` holds the record declarations visible to the entry; a member
    requirement must name one integer field of its record parameter's type.
    No optimizer fact is attached here. Consumers must enforce the resulting
    obligations before using them as proofs about uploaded metadata.
    """
    result = []
    for requirement in requirements:
        matches = []
        for parameter in function.parameters:
            origin = source_resource_identity(parameter)
            if origin is not None and (
                origin["entryPoint"] == requirement.entry_point
                and origin["parameter"] == requirement.parameter
            ):
                matches.append((parameter, origin))
        if len(matches) != 1:
            raise MetadataPreconditionError(
                f"Metadata parameter {requirement.entry_point}.{requirement.parameter} "
                "requires one selected source-resource identity."
            )
        parameter, origin = matches[0]
        reference = parameter.param_type
        qualified = [parameter, reference]
        if requirement.member is not None:
            field, scalar = _record_field(reference, requirement.member, structs)
            qualified.extend((reference.referenced_type, field))
        elif requirement.element is not None:
            if (
                isinstance(reference, NamedType)
                and reference.name == "StructuredBuffer"
                and len(reference.generic_args) == 1
            ):
                scalar = reference.generic_args[0]
            elif (
                isinstance(reference, PointerType)
                and not reference.is_mutable
                and reference.address_space in {"constant", "device", "storage"}
            ):
                scalar = reference.pointee_type
            else:
                raise MetadataPreconditionError(
                    "Metadata elements require a read-only scalar buffer."
                )
        elif (
            not isinstance(reference, ReferenceType)
            or reference.address_space != "constant"
            or reference.is_mutable
        ):
            raise MetadataPreconditionError(
                "Metadata preconditions currently require constant scalar references."
            )
        else:
            scalar = reference.referenced_type
        if any(
            "volatile" in (getattr(node, "qualifiers", ()) or ())
            for node in (*qualified, scalar)
        ):
            raise MetadataPreconditionError(
                "Volatile metadata cannot provide a checked upload contract."
            )
        shape = source_integer_shape(getattr(scalar, "name", ""))
        if shape is None or shape[2] != 1 or shape[1] not in (8, 16, 32, 64):
            raise MetadataPreconditionError(
                "Metadata preconditions require a supported integer scalar."
            )
        prefix = "u" if shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER else ""
        result.append(
            dict(
                sourceResource={"schemaVersion": 1, **origin},
                dtype=f"{prefix}int{shape[1]}",
                minimum=requirement.minimum,
                maximum=requirement.maximum,
                **(
                    {"element": requirement.element}
                    if requirement.element is not None
                    else {}
                ),
                **(
                    {"member": requirement.member}
                    if requirement.member is not None
                    else {}
                ),
            )
        )
    try:
        return checked_metadata_preconditions(result)
    except ValueError as exc:
        raise MetadataPreconditionError(str(exc)) from exc


def validate_codegen_metadata_preconditions(entries, records, structs=()):
    """Bind proof obligations to the actual selected compute entry and its types."""
    records = checked_metadata_preconditions(records)
    if not records:
        return records
    if len(entries) != 1 or entries[0][0] != "compute":
        raise MetadataPreconditionError(
            "Metadata proofs require one selected compute entry."
        )
    resolved = resolve_metadata_preconditions(
        entries[0][1],
        [
            MetadataRangePrecondition(
                entry_point=item["sourceResource"]["entryPoint"],
                parameter=item["sourceResource"]["parameter"],
                minimum=item["minimum"],
                maximum=item["maximum"],
                element=item.get("element"),
                member=item.get("member"),
            )
            for item in records
        ],
        structs,
    )
    if resolved != records:
        raise MetadataPreconditionError(
            "Metadata proof identity or type differs from the selected source."
        )
    return records


RECORD_MEMBER_LAYOUTS = {"metal-buffer", "std140", "hlsl-constant-buffer"}


def metadata_member_offset(layout, requirement):
    """The byte offset of one checked field within a reflected constant record.

    The layout must be the artifact's own reflection of a whole record that is
    uploaded as little-endian words: a Metal record buffer, an OpenGL std140
    block or a DirectX constant buffer. The field must be one unpacked integer
    scalar of the required type, aligned to its width and inside the record.
    """
    if not isinstance(layout, Mapping) or not isinstance(requirement, Mapping):
        raise ValueError("Metadata members require a reflected record layout.")
    width = INTEGER_METADATA_BYTES.get(requirement.get("dtype"))
    storage = layout.get("storageLayout")
    if storage == "metal-buffer":
        fields, size = layout.get("structMembers"), layout.get("elementSizeBytes")
        whole_record = (
            layout.get("elementType") == "record"
            and "blockMembers" not in layout
            and layout.get("elementStrideBytes") == size
        )
    else:
        fields, size = layout.get("blockMembers"), layout.get("blockSizeBytes")
        whole_record = (
            storage in RECORD_MEMBER_LAYOUTS
            and "structMembers" not in layout
            and layout.get("runtimeSized") is False
        )
    if (
        width is None
        or not whole_record
        or layout.get("memberOffsetBytes") != 0
        or layout.get("payloadEncoding") != "uint32-le-words"
        or "storageEncoding" in layout
        or "memberName" in layout
        or not isinstance(fields, list)
        or type(size) is not int
        or size <= 0
    ):
        raise ValueError("Metadata members require a reflected constant record.")
    matches = [
        field
        for field in fields
        if isinstance(field, Mapping) and field.get("name") == requirement.get("member")
    ]
    if len(matches) != 1:
        raise ValueError("Metadata member is missing or ambiguous in the record.")
    (field,) = matches
    offset = field.get("offsetBytes")
    if (
        field.get("elementType") != requirement["dtype"]
        or field.get("sizeBytes") != width
        or field.get("vectorWidth", 1) != 1
        or any(key in field for key in ("arrayLength", "arrayStrideBytes"))
        or type(offset) is not int
        or offset < 0
        or offset % width
        or offset + width > size
    ):
        raise ValueError("Metadata members require one reflected integer scalar field.")
    return offset


def apply_metadata_preconditions(resources, records):
    """Match obligations to independently reflected read-only integer layouts."""
    pending = []
    for requirement in checked_metadata_preconditions(records):
        matches = [
            resource
            for resource in resources
            if resource.get("metadata", {}).get("provenance", {}).get("sourceResource")
            == requirement["sourceResource"]
        ]
        if len(matches) != 1:
            raise ValueError("Metadata precondition resource is missing or ambiguous.")
        resource = matches[0]
        layout = resource.get("scalarLayout", {})
        width = INTEGER_METADATA_BYTES[requirement["dtype"]]
        if "member" in requirement:
            if (
                resource.get("access") != "read"
                or resource.get("kind") != "constant-buffer"
            ):
                raise ValueError(
                    "Metadata members require a reflected read-only constant record."
                )
            metadata_member_offset(layout, requirement)
            pending.append((resource, requirement))
            continue
        array = "element" in requirement
        if (
            resource.get("access") != "read"
            or resource.get("kind")
            not in (
                {"constant-buffer", "buffer"}
                if array
                else {"constant-buffer", "uniform"}
            )
            or layout.get("elementType") != requirement["dtype"]
            or layout.get("elementSizeBytes") != width
            or layout.get("elementStrideBytes") != width
            or layout.get("memberOffsetBytes") != 0
            or layout.get("vectorWidth", 1) != 1
            or layout.get("runtimeSized") is not array
            or any(
                key in layout
                for key in (
                    "members",
                    "structMembers",
                    "componentCount",
                    "payloadEncoding",
                )
            )
            or "storageEncoding" in layout
        ):
            raise ValueError(
                "Metadata preconditions require a reflected read-only integer scalar."
            )
        pending.append((resource, requirement))
    for resource, _ in pending:
        resource["metadata"]["provenance"].pop("metadataElementPreconditions", None)
        resource["metadata"]["provenance"].pop("metadataMemberPreconditions", None)
    for resource, requirement in pending:
        provenance = resource["metadata"]["provenance"]
        if "member" in requirement:
            provenance.setdefault("metadataMemberPreconditions", []).append(requirement)
        elif "element" in requirement:
            provenance.setdefault("metadataElementPreconditions", []).append(
                requirement
            )
        else:
            provenance["metadataPrecondition"] = requirement


def binding_metadata_preconditions(bindings):
    """Read descriptor obligations without treating ordinary provenance as proof."""
    records = []
    for item in bindings:
        provenance = item.get("provenance", {})
        if "metadataPrecondition" in provenance:
            record = provenance["metadataPrecondition"]
            if not isinstance(record, Mapping) or {"element", "member"} & set(record):
                raise ValueError("Scalar metadata requires a scalar precondition.")
            records.append(record)
        for key, selector, label in (
            ("metadataElementPreconditions", "element", "Array metadata"),
            ("metadataMemberPreconditions", "member", "Record metadata"),
        ):
            if key not in provenance:
                continue
            selected = provenance[key]
            if (
                not isinstance(selected, list)
                or not selected
                or any(
                    not isinstance(record, Mapping) or selector not in record
                    for record in selected
                )
            ):
                raise ValueError(f"{label} requires explicit {selector} preconditions.")
            records.extend(selected)
    return checked_metadata_preconditions(records)
