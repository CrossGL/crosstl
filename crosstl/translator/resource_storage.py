"""Explicit logical encodings for physically distinct shader buffer storage."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, NoReturn, Sequence

RESOURCE_STORAGE_PREFIX = "// crosstl-resource-storage: "
MAX_RESOURCE_STORAGE_HEADER_BYTES = 65536
BINARY16_STORAGE = {
    "logicalElementType": "float16",
    "encoding": "ieee754-binary16",
}


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate resource storage key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-finite resource storage value {value!r}.")


def _validated_resources(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or not 1 <= len(value) <= 256:
        raise ValueError("Resource storage requires between 1 and 256 resources.")
    result = {}
    for name, encoding in value.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_]\w*", name, re.ASCII) is None
        ):
            raise ValueError("Resource storage names must be source identifiers.")
        if not isinstance(encoding, Mapping) or dict(encoding) != BINARY16_STORAGE:
            raise ValueError(f"Unsupported logical storage encoding for {name!r}.")
        result[name] = dict(encoding)
    return result


def resource_storage_header(resources: Mapping[str, Mapping[str, str]]) -> str:
    """Serialize a source contract; it must precede all shader source text."""
    payload = {"schemaVersion": 1, "resources": _validated_resources(resources)}
    header = RESOURCE_STORAGE_PREFIX + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if len(header.encode("utf-8")) > MAX_RESOURCE_STORAGE_HEADER_BYTES:
        raise ValueError("Resource storage header exceeds the byte limit.")
    return header + "\n"


def parse_resource_storage_header(source: str) -> dict[str, dict[str, str]]:
    """Read the reserved first-line contract, rejecting misplaced declarations."""
    marker = RESOURCE_STORAGE_PREFIX.rstrip()
    if marker not in source:
        return {}
    first_line, _, remaining = source.partition("\n")
    if not first_line.startswith(RESOURCE_STORAGE_PREFIX) or marker in remaining:
        raise ValueError("Resource storage must be declared once, on the first line.")
    if len(first_line.encode("utf-8")) > MAX_RESOURCE_STORAGE_HEADER_BYTES:
        raise ValueError("Resource storage header exceeds the byte limit.")
    try:
        payload = json.loads(
            first_line[len(RESOURCE_STORAGE_PREFIX) :],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"Invalid resource storage JSON: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schemaVersion", "resources"}
        or type(payload["schemaVersion"]) is not int
        or payload["schemaVersion"] != 1
    ):
        raise ValueError("Unsupported resource storage header schema.")
    return _validated_resources(payload["resources"])


def encoded_storage_dtype(
    layout: Mapping[str, Any], *, target: str, resource_kind: str, logical_dtype: str
) -> str:
    """Validate a codec and return its physical dtype, without changing bytes."""
    if "storageEncoding" not in layout:
        return logical_dtype
    encoding = layout["storageEncoding"]
    if (
        not isinstance(encoding, Mapping)
        or dict(encoding) != BINARY16_STORAGE
        or logical_dtype != "float16"
        or target != "directx"
        or resource_kind != "buffer"
        or layout.get("elementType") != "uint16"
        or layout.get("storageLayout") != "hlsl-structured-buffer"
        or layout.get("runtimeSized") is not True
    ):
        raise ValueError(
            "Binary16 bit storage requires a logical float16 buffer with native "
            "DirectX uint16 structured storage."
        )
    return "uint16"


def apply_resource_storage(
    resources: list[dict[str, Any]], contracts: Mapping[str, Mapping[str, str]]
) -> None:
    """Attach only contracts whose physical layouts were independently reflected."""
    pending = []
    for name, encoding in contracts.items():
        matches = [resource for resource in resources if resource.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Resource storage name {name!r} is missing or ambiguous.")
        resource = matches[0]
        layout = resource.get("scalarLayout")
        if not isinstance(layout, Mapping):
            raise ValueError(f"Resource storage for {name!r} has no concrete layout.")
        encoded = {**layout, "storageEncoding": dict(encoding)}
        encoded_storage_dtype(
            encoded,
            target="directx",
            resource_kind=resource["kind"],
            logical_dtype=encoding["logicalElementType"],
        )
        pending.append((resource, encoded))
    for resource, encoded in pending:
        resource["scalarLayout"] = encoded
