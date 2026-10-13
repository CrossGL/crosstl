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
from .ast import NamedType, PointerType, ReferenceType
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
    """A requirement attached to a scalar input or one immutable array element."""

    entry_point: str
    parameter: str
    minimum: int
    maximum: int
    source: str = "*"
    element: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.entry_point, "entry_point")
        _identifier(self.parameter, "parameter")
        _bounds(self.minimum, self.maximum)
        _element(self.element)
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
        }


def _element(value: Any) -> None:
    """Keep the byte offset representable for every supported integer width."""
    if value is not None and (type(value) is not int or not 0 <= value < (1 << 61)):
        raise ValueError("Metadata element requires a nonnegative bounded integer.")


def _add_selector(selectors, identity, element):
    previous = selectors.setdefault(identity, set())
    if element in previous or previous and (element is None or None in previous):
        raise ValueError(
            "Duplicate or ambiguous metadata source parameter requirements."
        )
    previous.add(element)


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
            } or not {"entryPoint", "parameter", "minimum", "maximum"} <= set(record):
                raise ValueError(f"{field_name} has unsupported or missing fields.")
            condition = MetadataRangePrecondition(
                entry_point=record["entryPoint"],
                parameter=record["parameter"],
                minimum=record["minimum"],
                maximum=record["maximum"],
                source=record.get("source", "*"),
                element=record.get("element"),
            )
            if "element" in record and record["element"] is None:
                raise ValueError("Explicit metadata element cannot be null.")
        else:
            raise ValueError(f"{field_name} requires metadata precondition objects.")
        key = (condition.source, condition.entry_point, condition.parameter)
        _add_selector(keys, key, condition.element)
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
        if not isinstance(record, Mapping) or set(record) - {"element"} != {
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
        if types.get(index, dtype) != dtype:
            raise ValueError("Metadata elements require one source integer type.")
        types[index] = dtype
        _add_selector(selectors, index, element)
        result.append({**record, "sourceResource": dict(source)})
    return sorted(
        result,
        key=lambda item: (
            item["sourceResource"]["backend"],
            item["sourceResource"]["entryPoint"],
            item["sourceResource"]["parameterIndex"],
            item.get("element", -1),
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


def resolve_metadata_preconditions(function, requirements):
    """Resolve original names against one selected entry's constant references.

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
        if requirement.element is not None:
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
            for node in (parameter, reference, scalar)
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
            )
        )
    try:
        return checked_metadata_preconditions(result)
    except ValueError as exc:
        raise MetadataPreconditionError(str(exc)) from exc


def validate_codegen_metadata_preconditions(entries, records):
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
            )
            for item in records
        ],
    )
    if resolved != records:
        raise MetadataPreconditionError(
            "Metadata proof identity or type differs from the selected source."
        )
    return records


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
    for resource, requirement in pending:
        provenance = resource["metadata"]["provenance"]
        if "element" in requirement:
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
            if not isinstance(record, Mapping) or "element" in record:
                raise ValueError("Scalar metadata requires a scalar precondition.")
            records.append(record)
        if "metadataElementPreconditions" in provenance:
            elements = provenance["metadataElementPreconditions"]
            if (
                not isinstance(elements, list)
                or not elements
                or any(
                    not isinstance(record, Mapping) or "element" not in record
                    for record in elements
                )
            ):
                raise ValueError(
                    "Array metadata requires explicit element preconditions."
                )
            records.extend(elements)
    return checked_metadata_preconditions(records)
