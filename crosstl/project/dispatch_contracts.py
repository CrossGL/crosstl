"""Load and evaluate bounded host dispatch contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

DISPATCH_CONTRACT_KIND = "crosstl-host-dispatch-contract"
DISPATCH_CONTRACT_EVALUATION_KIND = "crosstl-evaluated-host-dispatch-contract"
DISPATCH_CONTRACT_SCHEMA_VERSION = 1
MAX_DISPATCH_VARIANTS = 4096
MAX_EXPRESSION_DEPTH = 64

_ERROR_PREFIX = "project.dispatch-contract"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_INPUT_ROLES = frozenset(("shape", "scalar", "dtype", "feature"))
_VALUE_TYPES = frozenset(("integer", "number", "string", "boolean"))
_DISPATCH_GEOMETRY_FIELDS = frozenset(("workgroupCount", "globalSize", "gridSize"))
_BINARY_INTEGER_OPERATORS = frozenset(
    ("add", "subtract", "multiply", "floorDiv", "ceilDiv")
)
_COMPARISON_OPERATORS = frozenset(("eq", "ne", "lt", "le", "gt", "ge"))
_VARIADIC_INTEGER_OPERATORS = frozenset(("min", "max"))
_BOOLEAN_OPERATORS = frozenset(("all", "any"))
_OPERATORS = (
    _BINARY_INTEGER_OPERATORS
    | _COMPARISON_OPERATORS
    | _VARIADIC_INTEGER_OPERATORS
    | _BOOLEAN_OPERATORS
    | frozenset(("not", "select"))
)


class DispatchContractError(ValueError):
    """A dispatch contract cannot be validated or evaluated safely."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str = "$",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = (
            code if code.startswith(f"{_ERROR_PREFIX}.") else f"{_ERROR_PREFIX}.{code}"
        )
        self.message = message
        self.path = path
        self.details = dict(details or {})
        super().__init__(f"{path}: {message} ({self.code})")

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.details:
            payload["details"] = copy.deepcopy(self.details)
        return payload


@dataclass(frozen=True)
class DispatchContentIdentity:
    """Canonical SHA-256 identity for one dispatch contract manifest."""

    digest: str
    algorithm: str = "sha256"

    def to_json(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "value": self.digest}

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.digest}"


@dataclass(frozen=True)
class DispatchInputDeclaration:
    """A typed input supplied by a bounded workload record."""

    name: str
    role: str
    value_type: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "type": self.value_type,
        }
        if self.provenance:
            payload["provenance"] = copy.deepcopy(dict(self.provenance))
        return payload


@dataclass(frozen=True)
class DispatchCapabilityDeclaration:
    """A typed device property required by contract evaluation."""

    name: str
    value_type: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "type": self.value_type}
        if self.provenance:
            payload["provenance"] = copy.deepcopy(dict(self.provenance))
        return payload


@dataclass(frozen=True)
class DispatchValueRecord:
    """One finite workload or device-capability assignment."""

    record_id: str
    values: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.record_id,
            "values": copy.deepcopy(dict(self.values)),
        }
        if self.provenance:
            payload["provenance"] = copy.deepcopy(dict(self.provenance))
        return payload


@dataclass(frozen=True)
class DispatchContractBranch:
    """One exhaustive, mutually exclusive host dispatch decision branch."""

    branch_id: str
    when: Any
    workgroup_size: tuple[Any, Any, Any]
    dispatch_field: str
    dispatch_size: tuple[Any, Any, Any]
    source: Any | None = None
    entry_point: Any | None = None
    subgroup_width: Any | None = None
    specialization_constants: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.branch_id,
            "when": copy.deepcopy(self.when),
            "workgroupSize": copy.deepcopy(list(self.workgroup_size)),
            "dispatch": {self.dispatch_field: copy.deepcopy(list(self.dispatch_size))},
        }
        if self.source is not None:
            payload["source"] = copy.deepcopy(self.source)
        if self.entry_point is not None:
            payload["entryPoint"] = copy.deepcopy(self.entry_point)
        if self.subgroup_width is not None:
            payload["subgroupWidth"] = copy.deepcopy(self.subgroup_width)
        if self.specialization_constants:
            payload["specializationConstants"] = {
                name: copy.deepcopy(value)
                for name, value in sorted(self.specialization_constants.items())
            }
        if self.provenance:
            payload["provenance"] = copy.deepcopy(dict(self.provenance))
        return payload


@dataclass(frozen=True)
class DispatchContract:
    """A source/entry selection rule with finite decision branches."""

    contract_id: str
    branches: tuple[DispatchContractBranch, ...]
    source: Any | None = None
    entry_point: Any | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.contract_id,
            "branches": [
                branch.to_json()
                for branch in sorted(self.branches, key=lambda item: item.branch_id)
            ],
        }
        if self.source is not None:
            payload["source"] = copy.deepcopy(self.source)
        if self.entry_point is not None:
            payload["entryPoint"] = copy.deepcopy(self.entry_point)
        if self.provenance:
            payload["provenance"] = copy.deepcopy(dict(self.provenance))
        return payload


@dataclass(frozen=True)
class DispatchContractManifest:
    """Validated, bounded host dispatch contract records."""

    schema_version: int
    content_identity: DispatchContentIdentity
    inputs: tuple[DispatchInputDeclaration, ...]
    workloads: tuple[DispatchValueRecord, ...]
    capabilities: tuple[DispatchCapabilityDeclaration, ...]
    devices: tuple[DispatchValueRecord, ...]
    contracts: tuple[DispatchContract, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _manifest_payload(
            inputs=self.inputs,
            workloads=self.workloads,
            capabilities=self.capabilities,
            devices=self.devices,
            contracts=self.contracts,
            provenance=self.provenance,
        )

    def evaluate(
        self, *, max_variants: int = MAX_DISPATCH_VARIANTS
    ) -> DispatchContractEvaluation:
        return evaluate_dispatch_contract(self, max_variants=max_variants)


@dataclass(frozen=True)
class EvaluatedDispatchVariant:
    """A concrete backend-neutral execution contract."""

    variant_id: str
    artifact_id: str
    contract_id: str
    branch_id: str
    workload_id: str
    device_id: str
    inputs: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    source: str
    entry_point: str
    workgroup_size: tuple[int, int, int]
    subgroup_width: int | None
    specialization_constants: Mapping[str, Any]
    dispatch_field: str
    dispatch_size: tuple[int, int, int]
    provenance: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "variantId": self.variant_id,
            "artifactId": self.artifact_id,
            "contractId": self.contract_id,
            "branchId": self.branch_id,
            "workload": {
                "id": self.workload_id,
                "inputs": copy.deepcopy(dict(self.inputs)),
            },
            "device": {
                "id": self.device_id,
                "capabilities": copy.deepcopy(dict(self.capabilities)),
            },
            "source": self.source,
            "entryPoint": self.entry_point,
            "workgroupSize": list(self.workgroup_size),
            "specializationConstants": copy.deepcopy(
                dict(self.specialization_constants)
            ),
            "dispatch": {self.dispatch_field: list(self.dispatch_size)},
            "provenance": copy.deepcopy(dict(self.provenance)),
        }
        if self.subgroup_width is not None:
            payload["subgroupWidth"] = self.subgroup_width
        return payload


@dataclass(frozen=True)
class DispatchContractEvaluation(Sequence[EvaluatedDispatchVariant]):
    """Deterministically ordered variants produced from one manifest."""

    manifest_content_identity: DispatchContentIdentity
    variants: tuple[EvaluatedDispatchVariant, ...]
    manifest_source: str | None = None

    def __getitem__(self, index: int) -> EvaluatedDispatchVariant:
        return self.variants[index]

    def __len__(self) -> int:
        return len(self.variants)

    def __iter__(self) -> Iterator[EvaluatedDispatchVariant]:
        return iter(self.variants)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": DISPATCH_CONTRACT_EVALUATION_KIND,
            "schemaVersion": DISPATCH_CONTRACT_SCHEMA_VERSION,
            "manifestContentIdentity": self.manifest_content_identity.to_json(),
            "variantCount": len(self.variants),
            "variants": [variant.to_json() for variant in self.variants],
        }
        if self.manifest_source is not None:
            payload["manifestSource"] = self.manifest_source
        return payload


class _DuplicateJsonKey(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise _DuplicateJsonKey(key)
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value}")


def load_dispatch_contract(path: str | Path) -> DispatchContractManifest:
    """Load and validate a UTF-8 JSON host dispatch contract manifest."""

    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DispatchContractError(
            "read-failed",
            f"Could not read dispatch contract manifest: {exc}",
            details={"source": str(manifest_path)},
        ) from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey as exc:
        raise DispatchContractError(
            "duplicate-key",
            f"JSON object key {exc.key!r} is declared more than once.",
            details={"source": str(manifest_path), "key": exc.key},
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise DispatchContractError(
            "json-invalid",
            f"Dispatch contract manifest is not valid JSON: {exc}",
            details={"source": str(manifest_path)},
        ) from exc
    return parse_dispatch_contract(payload, source=manifest_path)


def parse_dispatch_contract(
    payload: Mapping[str, Any], *, source: str | Path | None = None
) -> DispatchContractManifest:
    """Validate parsed JSON records as a versioned dispatch contract manifest."""

    root = _expect_mapping(payload, "$", "manifest")
    _check_fields(
        root,
        required=(
            "kind",
            "schemaVersion",
            "inputs",
            "workloads",
            "capabilities",
            "devices",
            "contracts",
        ),
        optional=("provenance",),
        path="$",
    )
    if root["kind"] != DISPATCH_CONTRACT_KIND:
        raise DispatchContractError(
            "kind-invalid",
            f"kind must be {DISPATCH_CONTRACT_KIND!r}.",
            path="$.kind",
        )
    if type(root["schemaVersion"]) is not int:
        raise DispatchContractError(
            "schema-invalid",
            "schemaVersion must be an integer.",
            path="$.schemaVersion",
        )
    if root["schemaVersion"] != DISPATCH_CONTRACT_SCHEMA_VERSION:
        raise DispatchContractError(
            "schema-version-unsupported",
            (
                "Unsupported dispatch contract schemaVersion "
                f"{root['schemaVersion']!r}; expected {DISPATCH_CONTRACT_SCHEMA_VERSION}."
            ),
            path="$.schemaVersion",
        )

    provenance = _provenance(root.get("provenance", {}), "$.provenance")
    inputs = _parse_input_declarations(root["inputs"])
    input_by_name = {item.name: item for item in inputs}
    capabilities = _parse_capability_declarations(root["capabilities"])
    capability_by_name = {item.name: item for item in capabilities}
    workloads = _parse_value_records(
        root["workloads"],
        declarations=input_by_name,
        collection_name="workloads",
        missing_code="missing-input",
    )
    devices = _parse_value_records(
        root["devices"],
        declarations=capability_by_name,
        collection_name="devices",
        missing_code="missing-capability",
    )
    contracts = _parse_contracts(
        root["contracts"],
        input_names=frozenset(input_by_name),
        capability_names=frozenset(capability_by_name),
    )
    normalized = _manifest_payload(
        inputs=inputs,
        workloads=workloads,
        capabilities=capabilities,
        devices=devices,
        contracts=contracts,
        provenance=provenance,
    )
    digest = hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()
    return DispatchContractManifest(
        schema_version=DISPATCH_CONTRACT_SCHEMA_VERSION,
        content_identity=DispatchContentIdentity(digest=digest),
        inputs=inputs,
        workloads=workloads,
        capabilities=capabilities,
        devices=devices,
        contracts=contracts,
        provenance=provenance,
        source=str(source) if source is not None else None,
    )


def evaluate_dispatch_contract(
    manifest: DispatchContractManifest,
    *,
    max_variants: int = MAX_DISPATCH_VARIANTS,
) -> DispatchContractEvaluation:
    """Evaluate all finite workload/device records before target generation."""

    if not isinstance(manifest, DispatchContractManifest):
        raise TypeError("manifest must be a DispatchContractManifest")
    current_digest = hashlib.sha256(
        _canonical_json(manifest.to_json()).encode("utf-8")
    ).hexdigest()
    if current_digest != manifest.content_identity.digest:
        raise DispatchContractError(
            "content-identity-mismatch",
            "Dispatch contract contents changed after identity calculation.",
            details={
                "expected": manifest.content_identity.to_json(),
                "actual": {"algorithm": "sha256", "value": current_digest},
            },
        )
    if type(max_variants) is not int or not 0 < max_variants <= MAX_DISPATCH_VARIANTS:
        raise DispatchContractError(
            "variant-limit-invalid",
            f"max_variants must be between 1 and {MAX_DISPATCH_VARIANTS}.",
            path="max_variants",
        )

    upper_bound = (
        len(manifest.contracts) * len(manifest.workloads) * len(manifest.devices)
    )
    if upper_bound > max_variants:
        raise DispatchContractError(
            "variant-expansion-limit-exceeded",
            (
                f"Dispatch contract may produce {upper_bound} variants, exceeding "
                f"the configured limit of {max_variants}."
            ),
            details={"upperBound": upper_bound, "limit": max_variants},
        )

    input_declarations = {item.name: item for item in manifest.inputs}
    capability_declarations = {item.name: item for item in manifest.capabilities}
    variants: list[EvaluatedDispatchVariant] = []
    seen_variant_ids: dict[str, tuple[str, dict[str, str]]] = {}
    seen_execution_records: dict[str, dict[str, str]] = {}

    for contract in sorted(manifest.contracts, key=lambda item: item.contract_id):
        for workload in sorted(
            manifest.workloads,
            key=lambda item: (_canonical_json(item.values), item.record_id),
        ):
            for device in sorted(
                manifest.devices,
                key=lambda item: (_canonical_json(item.values), item.record_id),
            ):
                variant = _evaluate_variant(
                    manifest,
                    contract,
                    workload,
                    device,
                    input_declarations=input_declarations,
                    capability_declarations=capability_declarations,
                )
                identity_payload = _variant_identity_payload(variant)
                identity_json = _canonical_json(identity_payload)
                existing = seen_variant_ids.get(variant.variant_id)
                context = {
                    "contractId": contract.contract_id,
                    "workloadId": workload.record_id,
                    "deviceId": device.record_id,
                }
                if existing is not None:
                    code = (
                        "variant-duplicate"
                        if existing[0] == identity_json
                        else "variant-identity-collision"
                    )
                    raise DispatchContractError(
                        code,
                        (
                            "Evaluated dispatch variants do not have unique stable "
                            "identities."
                        ),
                        details={"first": existing[1], "second": context},
                    )
                execution_json = _canonical_json(_execution_payload(variant))
                if execution_json in seen_execution_records:
                    raise DispatchContractError(
                        "variant-duplicate",
                        "Two input records evaluate to the same execution variant.",
                        details={
                            "first": seen_execution_records[execution_json],
                            "second": context,
                        },
                    )
                seen_variant_ids[variant.variant_id] = (identity_json, context)
                seen_execution_records[execution_json] = context
                variants.append(variant)

    return DispatchContractEvaluation(
        manifest_content_identity=manifest.content_identity,
        manifest_source=manifest.source,
        variants=tuple(sorted(variants, key=lambda item: item.variant_id)),
    )


def _parse_input_declarations(value: Any) -> tuple[DispatchInputDeclaration, ...]:
    records = _non_empty_list(value, "$.inputs", "inputs")
    declarations: list[DispatchInputDeclaration] = []
    seen: set[str] = set()
    for index, record_value in enumerate(records):
        path = f"$.inputs[{index}]"
        record = _expect_mapping(record_value, path, "input declaration")
        _check_fields(
            record,
            required=("name", "role", "type"),
            optional=("provenance",),
            path=path,
        )
        name = _identifier(record["name"], f"{path}.name")
        if name in seen:
            raise DispatchContractError(
                "declaration-duplicate",
                f"Input {name!r} is declared more than once.",
                path=f"{path}.name",
            )
        seen.add(name)
        role = record["role"]
        value_type = record["type"]
        if not isinstance(role, str) or role not in _INPUT_ROLES:
            raise DispatchContractError(
                "input-role-invalid",
                f"Input role must be one of {sorted(_INPUT_ROLES)!r}.",
                path=f"{path}.role",
            )
        if not isinstance(value_type, str) or value_type not in _VALUE_TYPES:
            raise DispatchContractError(
                "value-type-invalid",
                f"Input type must be one of {sorted(_VALUE_TYPES)!r}.",
                path=f"{path}.type",
            )
        required_type = {"shape": "integer", "dtype": "string", "feature": "boolean"}
        if role in required_type and value_type != required_type[role]:
            raise DispatchContractError(
                "input-type-conflict",
                f"{role!r} inputs must use type {required_type[role]!r}.",
                path=f"{path}.type",
            )
        if role == "scalar" and value_type not in {"integer", "number"}:
            raise DispatchContractError(
                "input-type-conflict",
                "Scalar inputs must use type 'integer' or 'number'.",
                path=f"{path}.type",
            )
        declarations.append(
            DispatchInputDeclaration(
                name=name,
                role=role,
                value_type=value_type,
                provenance=_provenance(
                    record.get("provenance", {}), f"{path}.provenance"
                ),
            )
        )
    return tuple(declarations)


def _parse_capability_declarations(
    value: Any,
) -> tuple[DispatchCapabilityDeclaration, ...]:
    records = _expect_list(value, "$.capabilities", "capabilities")
    declarations: list[DispatchCapabilityDeclaration] = []
    seen: set[str] = set()
    for index, record_value in enumerate(records):
        path = f"$.capabilities[{index}]"
        record = _expect_mapping(record_value, path, "capability declaration")
        _check_fields(
            record,
            required=("name", "type"),
            optional=("provenance",),
            path=path,
        )
        name = _identifier(record["name"], f"{path}.name")
        if name in seen:
            raise DispatchContractError(
                "declaration-duplicate",
                f"Capability {name!r} is declared more than once.",
                path=f"{path}.name",
            )
        seen.add(name)
        value_type = record["type"]
        if not isinstance(value_type, str) or value_type not in _VALUE_TYPES:
            raise DispatchContractError(
                "value-type-invalid",
                f"Capability type must be one of {sorted(_VALUE_TYPES)!r}.",
                path=f"{path}.type",
            )
        declarations.append(
            DispatchCapabilityDeclaration(
                name=name,
                value_type=value_type,
                provenance=_provenance(
                    record.get("provenance", {}), f"{path}.provenance"
                ),
            )
        )
    return tuple(declarations)


def _parse_value_records(
    value: Any,
    *,
    declarations: Mapping[
        str, DispatchInputDeclaration | DispatchCapabilityDeclaration
    ],
    collection_name: str,
    missing_code: str,
) -> tuple[DispatchValueRecord, ...]:
    records = _non_empty_list(value, f"$.{collection_name}", collection_name)
    parsed: list[DispatchValueRecord] = []
    seen_ids: set[str] = set()
    expected_names = set(declarations)
    for index, record_value in enumerate(records):
        path = f"$.{collection_name}[{index}]"
        record = _expect_mapping(record_value, path, f"{collection_name} record")
        _check_fields(
            record,
            required=("id", "values"),
            optional=("provenance",),
            path=path,
        )
        record_id = _identifier(record["id"], f"{path}.id")
        if record_id in seen_ids:
            raise DispatchContractError(
                "record-id-duplicate",
                f"Record id {record_id!r} is declared more than once.",
                path=f"{path}.id",
            )
        seen_ids.add(record_id)
        values = _expect_mapping(record["values"], f"{path}.values", "record values")
        names = set(values)
        missing = sorted(expected_names - names)
        unknown = sorted(names - expected_names)
        if missing:
            raise DispatchContractError(
                missing_code,
                f"Record is missing declared values: {', '.join(missing)}.",
                path=f"{path}.values",
                details={"missing": missing},
            )
        if unknown:
            raise DispatchContractError(
                "record-value-unknown",
                f"Record contains undeclared values: {', '.join(unknown)}.",
                path=f"{path}.values",
                details={"unknown": unknown},
            )
        normalized_values: dict[str, Any] = {}
        for name in sorted(declarations):
            declaration = declarations[name]
            normalized_values[name] = _declared_value(
                values[name],
                declaration.value_type,
                path=f"{path}.values.{name}",
                positive=(
                    isinstance(declaration, DispatchInputDeclaration)
                    and declaration.role == "shape"
                ),
            )
        parsed.append(
            DispatchValueRecord(
                record_id=record_id,
                values=normalized_values,
                provenance=_provenance(
                    record.get("provenance", {}), f"{path}.provenance"
                ),
            )
        )
    return tuple(parsed)


def _parse_contracts(
    value: Any,
    *,
    input_names: frozenset[str],
    capability_names: frozenset[str],
) -> tuple[DispatchContract, ...]:
    records = _non_empty_list(value, "$.contracts", "contracts")
    contracts: list[DispatchContract] = []
    seen_ids: set[str] = set()
    for index, record_value in enumerate(records):
        record_path = f"$.contracts[{index}]"
        record = _expect_mapping(record_value, record_path, "contract")
        _check_fields(
            record,
            required=("id", "branches"),
            optional=("source", "entryPoint", "provenance"),
            path=record_path,
        )
        contract_id = _identifier(record["id"], f"{record_path}.id")
        if contract_id in seen_ids:
            raise DispatchContractError(
                "contract-id-duplicate",
                f"Contract id {contract_id!r} is declared more than once.",
                path=f"{record_path}.id",
            )
        seen_ids.add(contract_id)
        path = f"contracts[{contract_id}]"
        source = _optional_expression(
            record,
            "source",
            path=f"{path}.source",
            input_names=input_names,
            capability_names=capability_names,
        )
        entry_point = _optional_expression(
            record,
            "entryPoint",
            path=f"{path}.entryPoint",
            input_names=input_names,
            capability_names=capability_names,
        )
        branches = _parse_branches(
            record["branches"],
            contract_id=contract_id,
            has_contract_source=source is not None,
            has_contract_entry_point=entry_point is not None,
            input_names=input_names,
            capability_names=capability_names,
        )
        contracts.append(
            DispatchContract(
                contract_id=contract_id,
                branches=branches,
                source=source,
                entry_point=entry_point,
                provenance=_provenance(
                    record.get("provenance", {}), f"{record_path}.provenance"
                ),
            )
        )
    return tuple(contracts)


def _parse_branches(
    value: Any,
    *,
    contract_id: str,
    has_contract_source: bool,
    has_contract_entry_point: bool,
    input_names: frozenset[str],
    capability_names: frozenset[str],
) -> tuple[DispatchContractBranch, ...]:
    records = _non_empty_list(
        value, f"contracts[{contract_id}].branches", "contract branches"
    )
    branches: list[DispatchContractBranch] = []
    seen_ids: set[str] = set()
    for index, record_value in enumerate(records):
        record_path = f"$.contracts[{contract_id!r}].branches[{index}]"
        record = _expect_mapping(record_value, record_path, "contract branch")
        _check_fields(
            record,
            required=("id", "when", "workgroupSize", "dispatch"),
            optional=(
                "source",
                "entryPoint",
                "subgroupWidth",
                "specializationConstants",
                "provenance",
            ),
            path=record_path,
        )
        branch_id = _identifier(record["id"], f"{record_path}.id")
        if branch_id in seen_ids:
            raise DispatchContractError(
                "branch-id-duplicate",
                f"Branch id {branch_id!r} is declared more than once.",
                path=f"{record_path}.id",
            )
        seen_ids.add(branch_id)
        path = f"contracts[{contract_id}].branches[{branch_id}]"
        has_branch_source = "source" in record
        has_branch_entry_point = "entryPoint" in record
        if has_contract_source and has_branch_source:
            raise DispatchContractError(
                "field-conflict",
                "source must be declared at either contract or branch scope, not both.",
                path=f"{path}.source",
            )
        if has_contract_entry_point and has_branch_entry_point:
            raise DispatchContractError(
                "field-conflict",
                (
                    "entryPoint must be declared at either contract or branch scope, "
                    "not both."
                ),
                path=f"{path}.entryPoint",
            )
        if not has_contract_source and not has_branch_source:
            raise DispatchContractError(
                "field-missing",
                "Every branch must resolve a source.",
                path=path,
            )
        if not has_contract_entry_point and not has_branch_entry_point:
            raise DispatchContractError(
                "field-missing",
                "Every branch must resolve an entryPoint.",
                path=path,
            )
        when = copy.deepcopy(record["when"])
        _validate_expression(
            when,
            path=f"{path}.when",
            input_names=input_names,
            capability_names=capability_names,
        )
        source = _optional_expression(
            record,
            "source",
            path=f"{path}.source",
            input_names=input_names,
            capability_names=capability_names,
        )
        entry_point = _optional_expression(
            record,
            "entryPoint",
            path=f"{path}.entryPoint",
            input_names=input_names,
            capability_names=capability_names,
        )
        workgroup_size = _expression_vector(
            record["workgroupSize"],
            path=f"{path}.workgroupSize",
            input_names=input_names,
            capability_names=capability_names,
        )
        subgroup_width = _optional_expression(
            record,
            "subgroupWidth",
            path=f"{path}.subgroupWidth",
            input_names=input_names,
            capability_names=capability_names,
        )
        specialization_constants = _specialization_expressions(
            record.get("specializationConstants", {}),
            path=f"{path}.specializationConstants",
            input_names=input_names,
            capability_names=capability_names,
        )
        dispatch = _expect_mapping(record["dispatch"], f"{path}.dispatch", "dispatch")
        geometry_fields = set(dispatch)
        if (
            len(geometry_fields) != 1
            or not geometry_fields <= _DISPATCH_GEOMETRY_FIELDS
        ):
            raise DispatchContractError(
                "dispatch-geometry-conflict",
                (
                    "dispatch must declare exactly one of workgroupCount, globalSize, "
                    "or gridSize."
                ),
                path=f"{path}.dispatch",
            )
        dispatch_field = next(iter(geometry_fields))
        dispatch_size = _expression_vector(
            dispatch[dispatch_field],
            path=f"{path}.dispatch.{dispatch_field}",
            input_names=input_names,
            capability_names=capability_names,
        )
        branches.append(
            DispatchContractBranch(
                branch_id=branch_id,
                when=when,
                source=source,
                entry_point=entry_point,
                workgroup_size=workgroup_size,
                subgroup_width=subgroup_width,
                specialization_constants=specialization_constants,
                dispatch_field=dispatch_field,
                dispatch_size=dispatch_size,
                provenance=_provenance(
                    record.get("provenance", {}), f"{record_path}.provenance"
                ),
            )
        )
    return tuple(branches)


def _optional_expression(
    record: Mapping[str, Any],
    name: str,
    *,
    path: str,
    input_names: frozenset[str],
    capability_names: frozenset[str],
) -> Any | None:
    if name not in record:
        return None
    expression = copy.deepcopy(record[name])
    _validate_expression(
        expression,
        path=path,
        input_names=input_names,
        capability_names=capability_names,
    )
    return expression


def _expression_vector(
    value: Any,
    *,
    path: str,
    input_names: frozenset[str],
    capability_names: frozenset[str],
) -> tuple[Any, Any, Any]:
    values = _expect_list(value, path, "expression vector")
    if len(values) != 3:
        raise DispatchContractError(
            "geometry-rank-invalid",
            "Geometry vectors must contain exactly three expressions.",
            path=path,
        )
    copied = tuple(copy.deepcopy(item) for item in values)
    for index, expression in enumerate(copied):
        _validate_expression(
            expression,
            path=f"{path}[{index}]",
            input_names=input_names,
            capability_names=capability_names,
        )
    return copied  # type: ignore[return-value]


def _specialization_expressions(
    value: Any,
    *,
    path: str,
    input_names: frozenset[str],
    capability_names: frozenset[str],
) -> dict[str, Any]:
    expressions = _expect_mapping(value, path, "specializationConstants")
    parsed: dict[str, Any] = {}
    for name, expression in expressions.items():
        normalized_name = _specialization_selector(name, f"{path}.{name}")
        copied = copy.deepcopy(expression)
        _validate_expression(
            copied,
            path=f"{path}.{normalized_name}",
            input_names=input_names,
            capability_names=capability_names,
        )
        parsed[normalized_name] = copied
    return parsed


def _validate_expression(
    expression: Any,
    *,
    path: str,
    input_names: frozenset[str],
    capability_names: frozenset[str],
    depth: int = 0,
) -> None:
    if depth > MAX_EXPRESSION_DEPTH:
        raise DispatchContractError(
            "expression-depth-exceeded",
            f"Expressions may not exceed {MAX_EXPRESSION_DEPTH} nested nodes.",
            path=path,
        )
    if _is_json_scalar(expression):
        return
    if not isinstance(expression, Mapping):
        raise DispatchContractError(
            "expression-invalid",
            "Expression must be a JSON scalar or expression object.",
            path=path,
        )
    keys = set(expression) - {"provenance"}
    if "op" in keys:
        if keys != {"op", "args"}:
            raise DispatchContractError(
                "expression-invalid",
                "Operator expressions must declare only op and args.",
                path=path,
            )
        form = "op"
    elif len(keys) == 1:
        form = next(iter(keys))
    else:
        raise DispatchContractError(
            "expression-invalid",
            "Expression objects must declare exactly one expression form.",
            path=path,
        )
    _provenance(expression.get("provenance", {}), f"{path}.provenance")
    if form == "literal":
        if not _is_json_scalar(expression[form]):
            raise DispatchContractError(
                "literal-invalid",
                "literal must contain a finite JSON scalar.",
                path=f"{path}.literal",
            )
        return
    if form == "input":
        name = expression[form]
        if not isinstance(name, str) or name not in input_names:
            raise DispatchContractError(
                "input-reference-unknown",
                f"Unknown workload input reference {name!r}.",
                path=f"{path}.input",
            )
        return
    if form == "capability":
        name = expression[form]
        if not isinstance(name, str) or name not in capability_names:
            raise DispatchContractError(
                "capability-reference-unknown",
                f"Unknown device capability reference {name!r}.",
                path=f"{path}.capability",
            )
        return
    if form != "op":
        raise DispatchContractError(
            "expression-form-unknown",
            f"Unknown expression form {form!r}.",
            path=path,
        )
    _check_fields(
        expression,
        required=("op", "args"),
        optional=("provenance",),
        path=path,
    )
    operator = expression["op"]
    if not isinstance(operator, str) or operator not in _OPERATORS:
        raise DispatchContractError(
            "operator-unknown",
            f"Unknown expression operator {operator!r}.",
            path=f"{path}.op",
        )
    args = _expect_list(expression["args"], f"{path}.args", "operator arguments")
    if operator in _BINARY_INTEGER_OPERATORS | _COMPARISON_OPERATORS and len(args) != 2:
        expected = 2
    elif operator == "not" and len(args) != 1:
        expected = 1
    elif operator == "select" and len(args) != 3:
        expected = 3
    elif operator in _VARIADIC_INTEGER_OPERATORS | _BOOLEAN_OPERATORS and not args:
        expected = -1
    else:
        expected = 0
    if expected:
        description = "at least one" if expected == -1 else str(expected)
        raise DispatchContractError(
            "operator-arity-invalid",
            f"Operator {operator!r} requires {description} argument(s).",
            path=f"{path}.args",
        )
    for index, argument in enumerate(args):
        _validate_expression(
            argument,
            path=f"{path}.args[{index}]",
            input_names=input_names,
            capability_names=capability_names,
            depth=depth + 1,
        )


def _evaluate_variant(
    manifest: DispatchContractManifest,
    contract: DispatchContract,
    workload: DispatchValueRecord,
    device: DispatchValueRecord,
    *,
    input_declarations: Mapping[str, DispatchInputDeclaration],
    capability_declarations: Mapping[str, DispatchCapabilityDeclaration],
) -> EvaluatedDispatchVariant:
    predicate_traces: list[dict[str, Any]] = []
    predicate_results: list[dict[str, Any]] = []
    matches: list[DispatchContractBranch] = []
    for branch in sorted(contract.branches, key=lambda item: item.branch_id):
        path = f"contracts[{contract.contract_id}].branches[{branch.branch_id}].when"
        result = _evaluate_expression(
            branch.when,
            path=path,
            inputs=workload.values,
            capabilities=device.values,
            traces=predicate_traces,
        )
        if type(result) is not bool:
            raise DispatchContractError(
                "predicate-type-invalid",
                "Branch predicates must evaluate to boolean values.",
                path=path,
                details={"result": result},
            )
        predicate_results.append({"branchId": branch.branch_id, "matched": result})
        if result:
            matches.append(branch)
    context = {
        "contractId": contract.contract_id,
        "workloadId": workload.record_id,
        "deviceId": device.record_id,
    }
    if not matches:
        raise DispatchContractError(
            "branch-unmatched",
            "No host dispatch branch matched the workload and device records.",
            path=f"contracts[{contract.contract_id}].branches",
            details=context,
        )
    if len(matches) != 1:
        raise DispatchContractError(
            "branch-conflict",
            "More than one host dispatch branch matched the same records.",
            path=f"contracts[{contract.contract_id}].branches",
            details={
                **context,
                "matchingBranches": sorted(branch.branch_id for branch in matches),
            },
        )
    branch = matches[0]
    path = f"contracts[{contract.contract_id}].branches[{branch.branch_id}]"
    traces = predicate_traces
    source_expression = (
        contract.source if contract.source is not None else branch.source
    )
    source_path = (
        f"contracts[{contract.contract_id}].source"
        if contract.source is not None
        else f"{path}.source"
    )
    entry_expression = (
        contract.entry_point if contract.entry_point is not None else branch.entry_point
    )
    entry_path = (
        f"contracts[{contract.contract_id}].entryPoint"
        if contract.entry_point is not None
        else f"{path}.entryPoint"
    )
    source = _evaluate_expression(
        source_expression,
        path=source_path,
        inputs=workload.values,
        capabilities=device.values,
        traces=traces,
    )
    source = _source_path(source, source_path)
    entry_point = _evaluate_expression(
        entry_expression,
        path=entry_path,
        inputs=workload.values,
        capabilities=device.values,
        traces=traces,
    )
    entry_point = _non_empty_string(entry_point, entry_path, "entryPoint")
    workgroup_size = _evaluate_geometry_vector(
        branch.workgroup_size,
        path=f"{path}.workgroupSize",
        inputs=workload.values,
        capabilities=device.values,
        traces=traces,
    )
    subgroup_width = None
    if branch.subgroup_width is not None:
        subgroup_path = f"{path}.subgroupWidth"
        subgroup_width = _positive_integer(
            _evaluate_expression(
                branch.subgroup_width,
                path=subgroup_path,
                inputs=workload.values,
                capabilities=device.values,
                traces=traces,
            ),
            subgroup_path,
        )
    constants: dict[str, Any] = {}
    for name, expression in sorted(branch.specialization_constants.items()):
        constant_path = f"{path}.specializationConstants.{name}"
        result = _evaluate_expression(
            expression,
            path=constant_path,
            inputs=workload.values,
            capabilities=device.values,
            traces=traces,
        )
        if not _is_json_scalar(result):
            raise DispatchContractError(
                "specialization-value-invalid",
                "Specialization constants must evaluate to finite JSON scalars.",
                path=constant_path,
            )
        constants[name] = result
    dispatch_size = _evaluate_geometry_vector(
        branch.dispatch_size,
        path=f"{path}.dispatch.{branch.dispatch_field}",
        inputs=workload.values,
        capabilities=device.values,
        traces=traces,
    )

    execution = {
        "source": source,
        "entryPoint": entry_point,
        "workgroupSize": list(workgroup_size),
        "subgroupWidth": subgroup_width,
        "specializationConstants": constants,
        "dispatch": {branch.dispatch_field: list(dispatch_size)},
    }
    compile_contract = {
        name: execution[name]
        for name in (
            "source",
            "entryPoint",
            "workgroupSize",
            "subgroupWidth",
            "specializationConstants",
        )
    }
    identity_payload = {
        "contractId": contract.contract_id,
        "inputs": dict(workload.values),
        "capabilities": dict(device.values),
        "execution": execution,
    }
    variant_id = _sha256_identity(identity_payload)
    artifact_id = _sha256_identity(compile_contract)
    manifest_provenance: dict[str, Any] = {
        "contentIdentity": manifest.content_identity.to_json(),
        "record": copy.deepcopy(dict(manifest.provenance)),
    }
    if manifest.source is not None:
        manifest_provenance["source"] = manifest.source
    provenance = {
        "manifest": manifest_provenance,
        "inputDeclarations": {
            name: copy.deepcopy(dict(declaration.provenance))
            for name, declaration in sorted(input_declarations.items())
        },
        "workload": copy.deepcopy(dict(workload.provenance)),
        "capabilityDeclarations": {
            name: copy.deepcopy(dict(declaration.provenance))
            for name, declaration in sorted(capability_declarations.items())
        },
        "device": copy.deepcopy(dict(device.provenance)),
        "contract": copy.deepcopy(dict(contract.provenance)),
        "branch": copy.deepcopy(dict(branch.provenance)),
        "predicateResults": predicate_results,
        "expressions": sorted(traces, key=lambda item: item["path"]),
    }
    return EvaluatedDispatchVariant(
        variant_id=variant_id,
        artifact_id=artifact_id,
        contract_id=contract.contract_id,
        branch_id=branch.branch_id,
        workload_id=workload.record_id,
        device_id=device.record_id,
        inputs=dict(workload.values),
        capabilities=dict(device.values),
        source=source,
        entry_point=entry_point,
        workgroup_size=workgroup_size,
        subgroup_width=subgroup_width,
        specialization_constants=constants,
        dispatch_field=branch.dispatch_field,
        dispatch_size=dispatch_size,
        provenance=provenance,
    )


def _evaluate_expression(
    expression: Any,
    *,
    path: str,
    inputs: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    traces: list[dict[str, Any]],
) -> Any:
    provenance: Mapping[str, Any] = {}
    if _is_json_scalar(expression):
        result = expression
        kind = "literal"
        reference = None
    else:
        provenance = expression.get("provenance", {})
        if "literal" in expression:
            result = expression["literal"]
            kind = "literal"
            reference = None
        elif "input" in expression:
            name = expression["input"]
            result = inputs[name]
            kind = "input"
            reference = name
        elif "capability" in expression:
            name = expression["capability"]
            result = capabilities[name]
            kind = "capability"
            reference = name
        else:
            operator = expression["op"]
            args = expression["args"]
            result = _evaluate_operator(
                operator,
                args,
                path=path,
                inputs=inputs,
                capabilities=capabilities,
                traces=traces,
            )
            kind = "operator"
            reference = operator
    trace: dict[str, Any] = {
        "path": path,
        "kind": kind,
        "provenance": copy.deepcopy(dict(provenance)),
        "result": copy.deepcopy(result),
    }
    if reference is not None:
        trace["reference"] = reference
    traces.append(trace)
    return result


def _evaluate_operator(
    operator: str,
    args: list[Any],
    *,
    path: str,
    inputs: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    traces: list[dict[str, Any]],
) -> Any:
    def evaluate(index: int) -> Any:
        return _evaluate_expression(
            args[index],
            path=f"{path}.args[{index}]",
            inputs=inputs,
            capabilities=capabilities,
            traces=traces,
        )

    if operator == "select":
        condition = evaluate(0)
        _boolean_operand(condition, f"{path}.args[0]")
        return evaluate(1 if condition else 2)

    if operator in _BOOLEAN_OPERATORS:
        expected = operator == "all"
        for index in range(len(args)):
            value = _boolean_operand(evaluate(index), f"{path}.args[{index}]")
            if value is not expected:
                return not expected
        return expected

    values = [evaluate(index) for index in range(len(args))]
    if operator in _BINARY_INTEGER_OPERATORS | _VARIADIC_INTEGER_OPERATORS:
        integers = [
            _integer_operand(value, f"{path}.args[{index}]")
            for index, value in enumerate(values)
        ]
        if operator == "add":
            return integers[0] + integers[1]
        if operator == "subtract":
            return integers[0] - integers[1]
        if operator == "multiply":
            return integers[0] * integers[1]
        if operator in {"floorDiv", "ceilDiv"}:
            if integers[1] <= 0:
                raise DispatchContractError(
                    "division-invalid",
                    f"{operator} requires a positive divisor.",
                    path=f"{path}.args[1]",
                )
            if operator == "floorDiv":
                return integers[0] // integers[1]
            return -(-integers[0] // integers[1])
        return min(integers) if operator == "min" else max(integers)
    if operator == "not":
        return not _boolean_operand(values[0], f"{path}.args[0]")
    if operator in _COMPARISON_OPERATORS:
        left, right = values
        if operator in {"eq", "ne"}:
            equal = _comparison_equal(left, right)
            return equal if operator == "eq" else not equal
        _ordered_comparison_operands(left, right, path)
        if operator == "lt":
            return left < right
        if operator == "le":
            return left <= right
        if operator == "gt":
            return left > right
        return left >= right
    raise DispatchContractError(
        "operator-unknown",
        f"Unknown expression operator {operator!r}.",
        path=f"{path}.op",
    )


def _evaluate_geometry_vector(
    expressions: Sequence[Any],
    *,
    path: str,
    inputs: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    traces: list[dict[str, Any]],
) -> tuple[int, int, int]:
    values = tuple(
        _positive_integer(
            _evaluate_expression(
                expression,
                path=f"{path}[{index}]",
                inputs=inputs,
                capabilities=capabilities,
                traces=traces,
            ),
            f"{path}[{index}]",
        )
        for index, expression in enumerate(expressions)
    )
    return values  # type: ignore[return-value]


def _manifest_payload(
    *,
    inputs: Sequence[DispatchInputDeclaration],
    workloads: Sequence[DispatchValueRecord],
    capabilities: Sequence[DispatchCapabilityDeclaration],
    devices: Sequence[DispatchValueRecord],
    contracts: Sequence[DispatchContract],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": DISPATCH_CONTRACT_KIND,
        "schemaVersion": DISPATCH_CONTRACT_SCHEMA_VERSION,
        "inputs": [
            item.to_json() for item in sorted(inputs, key=lambda item: item.name)
        ],
        "workloads": [
            item.to_json()
            for item in sorted(workloads, key=lambda item: item.record_id)
        ],
        "capabilities": [
            item.to_json() for item in sorted(capabilities, key=lambda item: item.name)
        ],
        "devices": [
            item.to_json() for item in sorted(devices, key=lambda item: item.record_id)
        ],
        "contracts": [
            item.to_json()
            for item in sorted(contracts, key=lambda item: item.contract_id)
        ],
    }
    if provenance:
        payload["provenance"] = copy.deepcopy(dict(provenance))
    return payload


def _variant_identity_payload(variant: EvaluatedDispatchVariant) -> dict[str, Any]:
    return {
        "contractId": variant.contract_id,
        "inputs": dict(variant.inputs),
        "capabilities": dict(variant.capabilities),
        "execution": _execution_payload(variant),
    }


def _execution_payload(variant: EvaluatedDispatchVariant) -> dict[str, Any]:
    return {
        "source": variant.source,
        "entryPoint": variant.entry_point,
        "workgroupSize": list(variant.workgroup_size),
        "subgroupWidth": variant.subgroup_width,
        "specializationConstants": dict(variant.specialization_constants),
        "dispatch": {variant.dispatch_field: list(variant.dispatch_size)},
    }


def _sha256_identity(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise DispatchContractError(
            "json-value-invalid",
            f"Manifest contains a value that cannot be represented safely as JSON: {exc}",
        ) from exc


def _expect_mapping(value: Any, path: str, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DispatchContractError(
            "schema-invalid", f"{label} must be a JSON object.", path=path
        )
    return value


def _expect_list(value: Any, path: str, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise DispatchContractError(
            "schema-invalid", f"{label} must be a JSON array.", path=path
        )
    return value


def _non_empty_list(value: Any, path: str, label: str) -> list[Any]:
    records = _expect_list(value, path, label)
    if not records:
        raise DispatchContractError(
            "schema-invalid", f"{label} must not be empty.", path=path
        )
    return records


def _check_fields(
    record: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str],
    path: str,
) -> None:
    missing = sorted(set(required) - set(record))
    unknown = sorted(set(record) - set(required) - set(optional))
    if missing:
        raise DispatchContractError(
            "field-missing",
            f"Required field(s) missing: {', '.join(missing)}.",
            path=path,
            details={"missing": missing},
        )
    if unknown:
        raise DispatchContractError(
            "field-unknown",
            f"Unknown field(s): {', '.join(unknown)}.",
            path=path,
            details={"unknown": unknown},
        )


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise DispatchContractError(
            "identifier-invalid",
            "Identifiers must match [A-Za-z_][A-Za-z0-9_.-]*.",
            path=path,
        )
    return value


def _specialization_selector(value: Any, path: str) -> str:
    if isinstance(value, str) and value.isdigit() and str(int(value)) == value:
        return value
    return _identifier(value, path)


def _non_empty_string(value: Any, path: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise DispatchContractError(
            "result-string-invalid",
            f"{label} must evaluate to a non-empty string without control characters.",
            path=path,
        )
    return value


def _source_path(value: Any, path: str) -> str:
    source = _non_empty_string(value, path, "source")
    if "\\" in source:
        raise DispatchContractError(
            "source-path-invalid",
            "source must use repository-relative POSIX path separators.",
            path=path,
        )
    parsed = PurePosixPath(source)
    if (
        source == "."
        or ":" in source
        or parsed.is_absolute()
        or ".." in parsed.parts
        or str(parsed) != source
    ):
        raise DispatchContractError(
            "source-path-invalid",
            "source must be a normalized repository-relative path.",
            path=path,
        )
    return source


def _provenance(value: Any, path: str) -> dict[str, Any]:
    provenance = _expect_mapping(value, path, "provenance")
    copied = copy.deepcopy(dict(provenance))
    try:
        _canonical_json(copied)
    except DispatchContractError as exc:
        raise DispatchContractError(
            "provenance-invalid", exc.message, path=path
        ) from exc
    return copied


def _declared_value(
    value: Any, value_type: str, *, path: str, positive: bool = False
) -> Any:
    valid = False
    if value_type == "integer":
        valid = type(value) is int
    elif value_type == "number":
        valid = _is_number(value)
    elif value_type == "string":
        valid = isinstance(value, str)
    elif value_type == "boolean":
        valid = type(value) is bool
    if not valid:
        raise DispatchContractError(
            "record-value-type-invalid",
            f"Value must have declared type {value_type!r}.",
            path=path,
        )
    if positive and value <= 0:
        raise DispatchContractError(
            "shape-value-invalid",
            "Shape dimensions must be positive integers.",
            path=path,
        )
    if value_type == "string" and not value:
        raise DispatchContractError(
            "record-value-invalid", "String values must not be empty.", path=path
        )
    return value


def _is_number(value: Any) -> bool:
    return type(value) in {int, float} and (type(value) is int or math.isfinite(value))


def _is_json_scalar(value: Any) -> bool:
    return type(value) in {bool, int, str} or (
        type(value) is float and math.isfinite(value)
    )


def _integer_operand(value: Any, path: str) -> int:
    if type(value) is not int:
        raise DispatchContractError(
            "integer-operand-required",
            "Arithmetic operators require integer operands.",
            path=path,
        )
    return value


def _boolean_operand(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise DispatchContractError(
            "boolean-operand-required",
            "Boolean operators require boolean operands.",
            path=path,
        )
    return value


def _positive_integer(value: Any, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise DispatchContractError(
            "geometry-value-invalid",
            "Workgroup, subgroup, and dispatch geometry must be positive integers.",
            path=path,
            details={"result": value},
        )
    return value


def _comparison_equal(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        return left == right
    return type(left) is type(right) and left == right


def _ordered_comparison_operands(left: Any, right: Any, path: str) -> None:
    numeric = _is_number(left) and _is_number(right)
    strings = isinstance(left, str) and isinstance(right, str)
    if not numeric and not strings:
        raise DispatchContractError(
            "comparison-operands-invalid",
            "Ordered comparisons require two numbers or two strings.",
            path=path,
        )
