"""Source parameter identities retained through shader resource lowering.

These records describe provenance, not trusted bounds or dispatch preconditions.
The enclosing artifact supplies the source path and content hash.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .ast import LiteralNode

RESOURCE_IDENTITY_ATTRIBUTE = "source_resource"
RESOURCE_IDENTITY_PREFIX = "// crosstl-resource-origin: "
MAX_RESOURCE_IDENTITY_BYTES = 65536
MAX_RESOURCE_IDENTITIES = 256
_SOURCE_FIELDS = {"backend", "entryPoint", "parameter", "parameterIndex"}
_COMMENT_OR_STRING = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\r\n]*', re.DOTALL
)


def _name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 1024
        or any(ord(char) < 32 or char in '\\"' for char in value)
    ):
        raise ValueError("Resource identity names require nonempty, unescaped text.")
    return value


def _source(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
        raise ValueError("Unsupported source resource identity fields.")
    for field in ("backend", "entryPoint", "parameter"):
        _name(value[field])
    if re.fullmatch(r"[a-z][a-z0-9_-]*", value["backend"]) is None:
        raise ValueError("Source resource backend must be a canonical backend name.")
    index = value["parameterIndex"]
    if type(index) is not int or not 0 <= index <= 65535:
        raise ValueError("Source resource parameter index must be an unsigned integer.")
    return dict(value)


def source_resource_attribute(
    backend: str, entry_point: str, parameter: str, index: int
) -> str:
    """Emit a canonical attribute before identifiers are renamed by the frontend."""
    source = _source(
        dict(
            backend=backend,
            entryPoint=entry_point,
            parameter=parameter,
            parameterIndex=index,
        )
    )
    arguments = [
        source[key] for key in ("backend", "entryPoint", "parameter", "parameterIndex")
    ]
    return (
        "@source_resource("
        + ", ".join(json.dumps(value, ensure_ascii=False) for value in arguments)
        + ") "
    )


def source_resource_identity(node: Any) -> dict[str, Any] | None:
    attributes = [
        attr
        for attr in getattr(node, "attributes", []) or []
        if getattr(attr, "name", None) == RESOURCE_IDENTITY_ATTRIBUTE
    ]
    if not attributes:
        return None
    if len(attributes) != 1 or len(attributes[0].arguments) != 4:
        raise ValueError(
            "A resource requires one source_resource attribute with four literals."
        )
    arguments = attributes[0].arguments
    if not all(isinstance(arg, LiteralNode) for arg in arguments):
        raise ValueError("Source resource identity arguments must be literals.")
    return _source(
        dict(
            zip(
                ("backend", "entryPoint", "parameter", "parameterIndex"),
                (arg.value for arg in arguments),
            )
        )
    )


def resource_identity_marker(
    node: Any, target_name: str, *, entry_point: str | None = None
) -> str:
    """Bind an attributed declaration to its actual emitted target resource."""
    source = source_resource_identity(node)
    if source is None:
        return ""
    _name(target_name)
    if entry_point is not None:
        _name(entry_point)
    payload = dict(
        schemaVersion=1,
        source=source,
        target=dict(name=target_name, entryPoint=entry_point),
    )
    return (
        RESOURCE_IDENTITY_PREFIX
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate resource identity key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite resource identity value {value!r}.")


def parse_resource_identities(source: str) -> list[dict[str, Any]]:
    """Read standalone markers, rejecting reserved markers in strings or comments."""
    reserved = RESOURCE_IDENTITY_PREFIX.rstrip()
    count = source.count(reserved)
    if not count:
        return []
    if count > MAX_RESOURCE_IDENTITIES:
        raise ValueError("Too many source resource identities.")
    records = []
    byte_count = 0
    for match in _COMMENT_OR_STRING.finditer(source):
        token = match.group()
        if not token.startswith(RESOURCE_IDENTITY_PREFIX):
            continue
        line_start = source.rfind("\n", 0, match.start()) + 1
        if source[line_start : match.start()].strip():
            raise ValueError("Resource identity markers require a standalone line.")
        byte_count += len(token.encode("utf-8"))
        if byte_count > MAX_RESOURCE_IDENTITY_BYTES:
            raise ValueError("Resource identity metadata exceeds the byte limit.")
        try:
            record = json.loads(
                token[len(RESOURCE_IDENTITY_PREFIX) :],
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"Invalid resource identity JSON: {exc}") from exc
        if (
            not isinstance(record, dict)
            or set(record) != {"schemaVersion", "source", "target"}
            or type(record["schemaVersion"]) is not int
            or record["schemaVersion"] != 1
        ):
            raise ValueError("Unsupported resource identity schema.")
        record["source"] = _source(record["source"])
        target = record["target"]
        if not isinstance(target, dict) or set(target) != {"name", "entryPoint"}:
            raise ValueError("Unsupported target resource identity fields.")
        _name(target["name"])
        if target["entryPoint"] is not None:
            _name(target["entryPoint"])
        records.append(record)
    if len(records) != count:
        raise ValueError("Misplaced or malformed source resource identity marker.")
    return records


def apply_resource_identities(resources, records, entry_points) -> None:
    """Attach identities only after matching independently reflected declarations."""
    pending = []
    targets = set()
    sources = set()
    source_names = set()
    entries = {entry.get("name") for entry in entry_points}
    for record in records:
        target = record["target"]
        key = (target["name"], target["entryPoint"])
        source = record["source"]
        source_key = (source["backend"], source["entryPoint"], source["parameterIndex"])
        source_name = (source["backend"], source["entryPoint"], source["parameter"])
        if key in targets or source_key in sources or source_name in source_names:
            raise ValueError("Duplicate or ambiguous source resource identity.")
        targets.add(key)
        sources.add(source_key)
        source_names.add(source_name)
        if target["entryPoint"] is not None and target["entryPoint"] not in entries:
            raise ValueError("Resource identity target entry point is missing.")
        matches = [
            resource
            for resource in resources
            if resource.get("name") == target["name"]
            and (resource.get("metadata") or {}).get("entryPoint")
            == target["entryPoint"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Resource identity target {key!r} is missing or ambiguous."
            )
        resource = matches[0]
        if "sourceResource" in (resource.get("metadata") or {}).get("provenance", {}):
            raise ValueError("Resource already has a source identity.")
        pending.append((resource, {"schemaVersion": 1, **source}))
    for resource, identity in pending:
        metadata = dict(resource.get("metadata", {}))
        metadata["provenance"] = {
            **metadata.get("provenance", {}),
            "sourceResource": identity,
        }
        resource["metadata"] = metadata
