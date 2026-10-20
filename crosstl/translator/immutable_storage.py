"""Checked contracts that named read-only storage is immutable for one dispatch.

A read-only storage view is not immutable storage: another binding of the same
dispatch may alias it and write. A project may state, per source entry point and
parameter, that it will never bind such a parameter to memory that any writable
binding of the dispatch can reach. Only the software-subgroup uniformity proofs
consume the contract, and only for the parameters it names. Generated artifacts
carry the contract in a leading header so that reflection, packaging and every
native dispatch path can enforce it before execution.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .artifact_contracts import (
    IMMUTABLE_STORAGE_PREFIX,
    artifact_contract_headers,
    with_artifact_contract_header,
)
from .ast import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
)
from .resource_identity import source_resource_attribute, source_resource_identity

MAX_IMMUTABLE_STORAGE = 256
MAX_IMMUTABLE_STORAGE_BYTES = 65536
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_FIELDS = {"source", "entryPoint", "parameter"}
_IDENTITY_FIELDS = {
    "schemaVersion",
    "backend",
    "entryPoint",
    "parameter",
    "parameterIndex",
}
# Pointer offsets keep the view inside the same immutable binding.
_OFFSET_OPERATORS = {"+=", "-="}
_WRITE_UNARY_OPERATORS = {"++", "--", "&"}


class ImmutableStorageError(ValueError):
    project_diagnostic_code = "project.translate.immutable-storage-unsupported"


def _identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 1024
        or not _IDENTIFIER.fullmatch(value)
    ):
        raise ValueError(f"Immutable storage {field} requires an identifier.")
    return value


@dataclass(frozen=True)
class ImmutableStorageContract:
    """One source entry parameter that no writable binding may alias."""

    entry_point: str
    parameter: str
    source: str = "*"

    def __post_init__(self) -> None:
        _identifier(self.entry_point, "entry_point")
        _identifier(self.parameter, "parameter")
        if (
            not isinstance(self.source, str)
            or not 1 <= len(self.source) <= 4096
            or self.source != self.source.strip()
            or any(ord(char) < 32 for char in self.source)
        ):
            raise ValueError("Immutable storage source requires a nonempty pattern.")

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "entryPoint": self.entry_point,
            "parameter": self.parameter,
        }


def parse_immutable_storage(
    records: Sequence[Any] | None, *, field_name: str = "immutable_storage"
) -> tuple[ImmutableStorageContract, ...]:
    """Reject unknown fields and duplicate parameters before translation."""
    if records is None:
        return ()
    if not isinstance(records, (list, tuple)) or len(records) > MAX_IMMUTABLE_STORAGE:
        raise ValueError(
            f"{field_name} requires at most {MAX_IMMUTABLE_STORAGE} records."
        )
    result, keys = [], set()
    for record in records:
        if isinstance(record, ImmutableStorageContract):
            contract = record
        elif isinstance(record, Mapping):
            if set(record) - _FIELDS or not {"entryPoint", "parameter"} <= set(record):
                raise ValueError(f"{field_name} has unsupported or missing fields.")
            contract = ImmutableStorageContract(
                entry_point=record["entryPoint"],
                parameter=record["parameter"],
                source=record.get("source", "*"),
            )
        else:
            raise ValueError(f"{field_name} requires immutable storage objects.")
        key = (contract.source, contract.entry_point, contract.parameter)
        if key in keys:
            raise ValueError(f"{field_name} names one parameter more than once.")
        keys.add(key)
        result.append(contract)
    return tuple(result)


def checked_immutable_storage(records: Any) -> list[dict[str, Any]]:
    """Validate resolved contracts without trusting generated resource names."""
    if not isinstance(records, (list, tuple)) or len(records) > MAX_IMMUTABLE_STORAGE:
        raise ValueError("Resolved immutable storage requires a bounded sequence.")
    result, indices, names = [], {}, {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"sourceResource"}:
            raise ValueError("Unsupported resolved immutable storage fields.")
        source = record["sourceResource"]
        if (
            not isinstance(source, Mapping)
            or set(source) != _IDENTITY_FIELDS
            or type(source["schemaVersion"]) is not int
            or source["schemaVersion"] != 1
        ):
            raise ValueError("Unsupported immutable storage source resource identity.")
        source_resource_attribute(
            source["backend"],
            source["entryPoint"],
            source["parameter"],
            source["parameterIndex"],
        )
        index = (source["backend"], source["entryPoint"], source["parameterIndex"])
        name = (source["backend"], source["entryPoint"], source["parameter"])
        if index in indices or name in names:
            raise ValueError("Duplicate immutable storage source resource identity.")
        indices[index] = name
        names[name] = index
        result.append({"sourceResource": dict(source)})
    return sorted(result, key=lambda item: _identity_key(item["sourceResource"]))


def _identity_key(source):
    return (source["backend"], source["entryPoint"], source["parameterIndex"])


def immutable_storage_header(records: Any) -> str:
    normalized = checked_immutable_storage(records)
    if not normalized:
        return ""
    header = IMMUTABLE_STORAGE_PREFIX + json.dumps(
        {"schemaVersion": 1, "resources": normalized},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if len(header.encode("utf-8")) > MAX_IMMUTABLE_STORAGE_BYTES:
        raise ValueError("Immutable storage header exceeds its byte limit.")
    return header + "\n"


def with_immutable_storage_header(source: str, records: Any) -> str:
    """Retain every existing leading artifact contract when adding this one."""
    if IMMUTABLE_STORAGE_PREFIX.rstrip() in source:
        raise ValueError("Source already contains an immutable storage marker.")
    return with_artifact_contract_header(source, immutable_storage_header(records))


def parse_immutable_storage_header(source: str) -> list[dict[str, Any]]:
    """Require one canonical header; reject markers hidden in source text."""
    line = artifact_contract_headers(source).get(IMMUTABLE_STORAGE_PREFIX)
    if line is None:
        return []
    if len(line.encode("utf-8")) > MAX_IMMUTABLE_STORAGE_BYTES:
        raise ValueError("Immutable storage header exceeds its byte limit.")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate immutable storage key.")
            result[key] = value
        return result

    try:
        record = json.loads(
            line[len(IMMUTABLE_STORAGE_PREFIX) :], object_pairs_hook=unique
        )
    except (ValueError, RecursionError) as error:
        raise ValueError("Invalid immutable storage JSON.") from error
    if (
        not isinstance(record, dict)
        or set(record) != {"schemaVersion", "resources"}
        or type(record["schemaVersion"]) is not int
        or record["schemaVersion"] != 1
        or not isinstance(record["resources"], list)
        or not record["resources"]
    ):
        raise ValueError("Unsupported immutable storage header schema.")
    resources = checked_immutable_storage(record["resources"])
    if resources != record["resources"]:
        raise ValueError("Immutable storage header is not in canonical order.")
    return resources


def _call_name(call):
    function = getattr(call, "function", None)
    return getattr(function, "name", function)


def _write_roots(body):
    """Ids of every node on the addressed path of a write or address-of."""
    roots = set()
    for node in body.walk():
        target = None
        if isinstance(node, AssignmentNode):
            target = node.target
        elif isinstance(node, UnaryOpNode) and node.operator in _WRITE_UNARY_OPERATORS:
            target = node.operand
        while target is not None:
            roots.add(id(target))
            if isinstance(target, ArrayAccessNode):
                target = target.array_expr
            elif isinstance(target, MemberAccessNode):
                target = target.object_expr
            elif isinstance(target, SwizzleNode):
                target = target.vector_expr
            else:
                target = None
    return roots


def _check_parameter_uses(function, name, user_functions, structs, visiting=()):
    """Admit only element reads, in-place offsets and checked helper arguments.

    Any other use, such as a cast, a copy into a local or a write through an
    element, could produce a writable alias inside the kernel that the
    dispatch-time check cannot see, so it rejects the contract. Passing the
    whole view to a user function is admitted only when every same-arity
    candidate receives it as read-only storage and itself obeys these rules.
    """
    body = getattr(function, "body", None)
    if body is None:
        raise ImmutableStorageError("Immutable storage requires an entry body.")
    written = _write_roots(body)
    allowed = set()
    forwarded = []
    for node in body.walk():
        if isinstance(node, VariableNode) and node.name == name:
            raise ImmutableStorageError(
                f"Immutable storage parameter '{name}' is shadowed by a local."
            )
        if (
            isinstance(node, ArrayAccessNode)
            and isinstance(node.array_expr, IdentifierNode)
            and node.array_expr.name == name
            and id(node) not in written
        ):
            allowed.add(id(node.array_expr))
        elif (
            isinstance(node, FunctionCallNode)
            and _call_name(node) == "buffer_load"
            and "buffer_load" not in user_functions
            and len(node.arguments) == 2
            and isinstance(node.arguments[0], IdentifierNode)
            and node.arguments[0].name == name
        ):
            allowed.add(id(node.arguments[0]))
        elif isinstance(node, FunctionCallNode) and _call_name(node) in user_functions:
            forwarded.extend(
                (node, position, argument)
                for position, argument in enumerate(node.arguments or ())
                if isinstance(argument, IdentifierNode)
                and argument.name == name
                and id(argument) not in written
            )
        elif (
            isinstance(node, AssignmentNode)
            and node.operator in _OFFSET_OPERATORS
            and isinstance(node.target, IdentifierNode)
            and node.target.name == name
        ):
            allowed.add(id(node.target))
    for call, position, argument in forwarded:
        _check_helper_argument(
            call, position, name, user_functions, structs, (*visiting, function)
        )
        allowed.add(id(argument))
    for node in body.walk():
        if (
            isinstance(node, IdentifierNode)
            and node.name == name
            and id(node) not in allowed
        ):
            raise ImmutableStorageError(
                f"Immutable storage parameter '{name}' may only be read by element "
                "or offset in place."
            )


def _check_helper_argument(call, position, name, user_functions, structs, visiting):
    """Follow one immutable view into every helper the call may select."""
    arity = len(call.arguments or ())
    callees = [
        callee
        for callee in user_functions.get(_call_name(call), ())
        if len(getattr(callee, "parameters", None) or ()) == arity
    ]
    if not callees:
        raise ImmutableStorageError(
            f"Immutable storage parameter '{name}' is passed to an unresolved helper."
        )
    for callee in callees:
        if any(callee is item for item in visiting):
            raise ImmutableStorageError(
                f"Immutable storage parameter '{name}' reaches a recursive helper."
            )
        parameter = callee.parameters[position]
        if not _read_only_storage(parameter, structs):
            raise ImmutableStorageError(
                f"Immutable storage parameter '{name}' is passed to helper "
                f"parameter '{parameter.name}', which is not read-only storage."
            )
        _check_parameter_uses(callee, parameter.name, user_functions, structs, visiting)


def _read_only_storage(parameter, structs):
    from .codegen.uniform_returns import UniformReturnAnalysis

    analysis = UniformReturnAnalysis({}, set(), structs)
    return analysis.resource_type(parameter.param_type, parameter, read_only=True)


def resolve_immutable_storage(function, contracts, structs=None, user_functions=None):
    """Bind contracts to one selected entry's read-only storage parameters.

    ``user_functions`` maps each user function name to its definitions, as
    ``program_functions`` returns; helpers absent from it are not followed.
    """
    structs = structs if isinstance(structs, Mapping) else _struct_map(structs)
    user_functions = dict(user_functions or {})
    result = []
    for contract in contracts:
        matches = []
        for parameter in getattr(function, "parameters", []) or []:
            origin = source_resource_identity(parameter)
            if origin is not None and (
                origin["entryPoint"] == contract.entry_point
                and origin["parameter"] == contract.parameter
            ):
                matches.append((parameter, origin))
        if len(matches) != 1:
            raise ImmutableStorageError(
                f"Immutable storage parameter {contract.entry_point}."
                f"{contract.parameter} requires one selected source-resource identity."
            )
        parameter, origin = matches[0]
        if not _read_only_storage(parameter, structs):
            raise ImmutableStorageError(
                f"Immutable storage parameter {contract.entry_point}."
                f"{contract.parameter} requires read-only, non-volatile, "
                "non-coherent storage of value elements."
            )
        _check_parameter_uses(function, parameter.name, user_functions, structs)
        result.append({"sourceResource": {"schemaVersion": 1, **origin}})
    try:
        return checked_immutable_storage(result)
    except ValueError as exc:
        raise ImmutableStorageError(str(exc)) from exc


def _struct_map(structs):
    return {
        getattr(item, "name", None): item
        for item in structs or ()
        if getattr(item, "name", None)
    }


def program_functions(ast):
    """User functions by name, in source order.

    A user ``buffer_load`` is then not the builtin, and immutable views passed
    to helpers can be followed into every same-named definition.
    """
    functions = {}
    for node in ast.walk() if hasattr(ast, "walk") else ():
        if type(node).__name__ == "FunctionNode" and getattr(node, "name", None):
            functions.setdefault(node.name, []).append(node)
    return {name: tuple(items) for name, items in functions.items()}


def codegen_immutable_storage_names(entries, records, ast=None):
    """Validate artifact contracts against the selected entry; return its names.

    The returned names are the entry's own storage parameters. Codegen may treat
    exactly those as immutable in software-subgroup uniformity proofs.
    """
    records = checked_immutable_storage(records)
    if not records:
        return records, frozenset()
    if len(entries) != 1 or entries[0][0] != "compute":
        raise ImmutableStorageError(
            "Immutable storage requires one selected compute entry."
        )
    function = entries[0][1]
    resolved = resolve_immutable_storage(
        function,
        [
            ImmutableStorageContract(
                entry_point=item["sourceResource"]["entryPoint"],
                parameter=item["sourceResource"]["parameter"],
            )
            for item in records
        ],
        getattr(ast, "structs", ()),
        program_functions(ast),
    )
    if resolved != records:
        raise ImmutableStorageError(
            "Immutable storage identity differs from the selected source."
        )
    identities = [item["sourceResource"] for item in records]
    names = frozenset(
        parameter.name
        for parameter in function.parameters
        for origin in (source_resource_identity(parameter),)
        if origin is not None and {"schemaVersion": 1, **origin} in identities
    )
    return records, names


def apply_immutable_storage(resources, records):
    """Mark the reflected read-only storage resources each contract names."""
    pending = []
    for requirement in checked_immutable_storage(records):
        matches = [
            resource
            for resource in resources
            if resource.get("metadata", {}).get("provenance", {}).get("sourceResource")
            == requirement["sourceResource"]
        ]
        if len(matches) != 1:
            raise ValueError("Immutable storage resource is missing or ambiguous.")
        resource = matches[0]
        if resource.get("access") != "read" or resource.get("kind") not in {
            "buffer",
            "constant-buffer",
        }:
            raise ValueError(
                "Immutable storage requires a reflected read-only buffer resource."
            )
        pending.append((resource, requirement))
    for resource, requirement in pending:
        resource["metadata"]["provenance"]["immutableStorage"] = requirement


def binding_immutable_storage(bindings):
    """Read descriptor obligations without treating ordinary provenance as proof."""
    records = []
    for item in bindings:
        provenance = item.get("provenance", {}) or {}
        if "immutableStorage" not in provenance:
            continue
        record = provenance["immutableStorage"]
        if not isinstance(record, Mapping) or record.get(
            "sourceResource"
        ) != provenance.get("sourceResource"):
            raise ValueError(
                "Immutable storage provenance differs from its source resource."
            )
        records.append(record)
    return checked_immutable_storage(records)
