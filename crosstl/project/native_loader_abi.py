"""Versioned native loader ABI descriptors for ready runtime loader units."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

NATIVE_LOADER_ABI_KIND = "crosstl-native-loader-abi-descriptor"
NATIVE_LOADER_ABI_VERSION = 1

_ERROR_PREFIX = "project.native-loader-abi"
_ACCESS_VALUES = frozenset(("read", "write", "read_write"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_C_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]+")
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class NativeLoaderABIError(ValueError):
    """A runtime loader unit cannot produce a complete native ABI descriptor."""

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
        self.details = copy.deepcopy(dict(details or {}))
        super().__init__(f"{path}: {message} ({self.code})")

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": "error",
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.details:
            payload["details"] = copy.deepcopy(self.details)
        return payload


@dataclass(frozen=True)
class NativeLoaderABIDescriptor:
    """Repository-neutral native loader data derived from one ready load unit."""

    unit_id: str
    target: str
    stage: str | None
    entry_point: Mapping[str, Any]
    artifact: Mapping[str, Any]
    source: Mapping[str, Any]
    bindings: tuple[Mapping[str, Any], ...]
    scalar_layout: Mapping[str, Any]
    specialization_constants: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    abi_version: int = NATIVE_LOADER_ABI_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.abi_version,
            "kind": NATIVE_LOADER_ABI_KIND,
            "abiVersion": self.abi_version,
            "unitId": self.unit_id,
            "target": self.target,
            "stage": self.stage,
            "entryPoint": copy.deepcopy(dict(self.entry_point)),
            "artifact": copy.deepcopy(dict(self.artifact)),
            "source": copy.deepcopy(dict(self.source)),
            "bindings": [copy.deepcopy(dict(binding)) for binding in self.bindings],
            "scalarLayout": copy.deepcopy(dict(self.scalar_layout)),
            "specializationConstants": [
                copy.deepcopy(dict(constant))
                for constant in self.specialization_constants
            ],
            "provenance": copy.deepcopy(dict(self.provenance)),
        }


def build_native_loader_abi_descriptor(
    loader: Mapping[str, Any],
    *,
    load_unit_id: str | None = None,
) -> dict[str, Any]:
    """Build one native ABI descriptor from a ready loader manifest or load unit."""

    unit = _select_load_unit(loader, load_unit_id=load_unit_id)
    _validate_load_unit_shape(unit)
    _validate_load_unit_readiness(unit)

    target = _required_string(unit.get("target"), path="$.target")
    unit_id = _required_string(unit.get("id"), path="$.id")
    host_interface = _ready_host_interface(unit.get("hostInterface"))
    entry_point = _selected_entry_point(unit, host_interface)
    bindings = _binding_descriptors(target, host_interface.get("resources"))
    scalar_layout = _scalar_layout_descriptor(host_interface, bindings)
    specialization_constants = _specialization_descriptors(unit, host_interface)
    artifact = _artifact_descriptor(unit)
    source = _source_descriptor(unit)
    provenance = _json_mapping(unit.get("provenance"), path="$.provenance")

    descriptor = NativeLoaderABIDescriptor(
        unit_id=unit_id,
        target=target,
        stage=entry_point.get("stage"),
        entry_point=entry_point,
        artifact=artifact,
        source=source,
        bindings=tuple(bindings),
        scalar_layout=scalar_layout,
        specialization_constants=tuple(specialization_constants),
        provenance=provenance,
    )
    return descriptor.to_json()


def generate_native_loader_declarations(descriptor: Mapping[str, Any]) -> str:
    """Render deterministic C declarations and immutable descriptor data."""

    normalized = _validate_descriptor(descriptor)
    symbol = _descriptor_symbol(normalized)
    guard = f"{symbol.upper()}_H"
    bindings = normalized["bindings"]

    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "#ifndef CROSSTL_NATIVE_LOADER_ABI_VERSION",
        f"#define CROSSTL_NATIVE_LOADER_ABI_VERSION {NATIVE_LOADER_ABI_VERSION}u",
        f"#elif CROSSTL_NATIVE_LOADER_ABI_VERSION != {NATIVE_LOADER_ABI_VERSION}u",
        '#error "Conflicting CrossTL native loader ABI version"',
        "#endif",
        "",
        "#ifndef CROSSTL_NATIVE_LOADER_ABI_V1_TYPES",
        "#define CROSSTL_NATIVE_LOADER_ABI_V1_TYPES",
        "",
        "typedef enum CrossTLNativeLoaderAccess {",
        "    CROSSTL_NATIVE_LOADER_ACCESS_NONE = 0,",
        "    CROSSTL_NATIVE_LOADER_ACCESS_READ = 1,",
        "    CROSSTL_NATIVE_LOADER_ACCESS_WRITE = 2,",
        "    CROSSTL_NATIVE_LOADER_ACCESS_READ_WRITE = 3",
        "} CrossTLNativeLoaderAccess;",
        "",
        "typedef struct CrossTLNativeLoaderBindingDescriptor {",
        "    const char *name;",
        "    const char *resource_kind;",
        "    const char *type_name;",
        "    const char *binding_namespace;",
        "    uint32_t set_index;",
        "    uint32_t binding_index;",
        "    CrossTLNativeLoaderAccess access;",
        "    const char *scalar_layout_json;",
        "    const char *provenance_json;",
        "} CrossTLNativeLoaderBindingDescriptor;",
        "",
        "typedef struct CrossTLNativeLoaderUnitDescriptor {",
        "    uint32_t abi_version;",
        "    const char *unit_id;",
        "    const char *target;",
        "    const char *stage;",
        "    const char *entry_point;",
        "    const char *entry_execution_json;",
        "    const char *artifact_path;",
        "    const char *artifact_format;",
        "    const char *artifact_hash_algorithm;",
        "    const char *artifact_hash_value;",
        "    uint64_t artifact_size_bytes;",
        "    const char *source_path;",
        "    const char *source_backend;",
        "    size_t binding_count;",
        "    const CrossTLNativeLoaderBindingDescriptor *bindings;",
        "    const char *scalar_layout_json;",
        "    const char *specialization_constants_json;",
        "    const char *provenance_json;",
        "    const char *source_artifact_path;",
        "    const char *source_hash_algorithm;",
        "    const char *source_hash_value;",
        "    const char *source_remap_json;",
        "} CrossTLNativeLoaderUnitDescriptor;",
        "",
        "#endif /* CROSSTL_NATIVE_LOADER_ABI_V1_TYPES */",
        "",
    ]

    if bindings:
        lines.extend(
            [
                "static const CrossTLNativeLoaderBindingDescriptor",
                f"    {symbol}_bindings[] = {{",
            ]
        )
        for binding in bindings:
            coordinates = binding["coordinates"]
            lines.extend(
                [
                    "    {",
                    f"        {_c_string(binding['name'])},",
                    f"        {_c_string(binding['kind'])},",
                    f"        {_c_string(binding['type'])},",
                    f"        {_c_string(binding['namespace'])},",
                    f"        {coordinates['set']}u,",
                    f"        {coordinates['binding']}u,",
                    f"        {_c_access(binding.get('access'))},",
                    f"        {_c_json_string(binding.get('scalarLayout'))},",
                    f"        {_c_json_string(binding.get('provenance', {}))}",
                    "    },",
                ]
            )
        lines.extend(["};", ""])

    artifact = normalized["artifact"]
    artifact_hash = artifact["hash"]
    entry_point = normalized["entryPoint"]
    source = normalized["source"]
    source_hash = source.get("hash")
    binding_pointer = f"{symbol}_bindings" if bindings else "NULL"
    lines.extend(
        [
            f"static const CrossTLNativeLoaderUnitDescriptor {symbol} = {{",
            f"    {NATIVE_LOADER_ABI_VERSION}u,",
            f"    {_c_string(normalized['unitId'])},",
            f"    {_c_string(normalized['target'])},",
            f"    {_c_nullable_string(normalized.get('stage'))},",
            f"    {_c_string(entry_point['name'])},",
            f"    {_c_json_string(entry_point.get('executionConfig', {}))},",
            f"    {_c_string(artifact['packagePath'])},",
            f"    {_c_string(artifact['format'])},",
            f"    {_c_string(artifact_hash['algorithm'])},",
            f"    {_c_string(artifact_hash['value'])},",
            f"    UINT64_C({artifact['sizeBytes']}),",
            f"    {_c_nullable_string(source.get('path'))},",
            f"    {_c_nullable_string(source.get('backend'))},",
            f"    {len(bindings)}u,",
            f"    {binding_pointer},",
            f"    {_c_json_string(normalized['scalarLayout'])},",
            f"    {_c_json_string(normalized['specializationConstants'])},",
            f"    {_c_json_string(normalized['provenance'])},",
            f"    {_c_nullable_string(source.get('artifactPath'))},",
            f"    {_c_nullable_string(source_hash.get('algorithm') if source_hash else None)},",
            f"    {_c_nullable_string(source_hash.get('value') if source_hash else None)},",
            f"    {_c_nullable_json_string(source.get('remap'))}",
            "};",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            f"#endif /* {guard} */",
            "",
        ]
    )
    return "\n".join(lines)


def _pipeline_loader_contract() -> tuple[int, str, frozenset[str]]:
    # Import lazily so pipeline.py can call this module without an import cycle.
    from .pipeline import (  # pylint: disable=import-outside-toplevel
        REPORT_SCHEMA_VERSION,
        RUNTIME_LOADER_MANIFEST_KIND,
        RUNTIME_LOADER_MANIFEST_LOAD_UNIT_FIELDS,
    )

    return (
        REPORT_SCHEMA_VERSION,
        RUNTIME_LOADER_MANIFEST_KIND,
        RUNTIME_LOADER_MANIFEST_LOAD_UNIT_FIELDS,
    )


def _select_load_unit(
    loader: Mapping[str, Any], *, load_unit_id: str | None
) -> Mapping[str, Any]:
    if not isinstance(loader, Mapping):
        raise NativeLoaderABIError(
            "input-invalid",
            "Native loader ABI input must be a runtime loader manifest or load unit object.",
        )

    schema_version, manifest_kind, _unit_fields = _pipeline_loader_contract()
    if "loadUnits" not in loader and loader.get("kind") != manifest_kind:
        if load_unit_id is not None:
            raise NativeLoaderABIError(
                "load-unit-selection-invalid",
                "load_unit_id may only select a unit from a runtime loader manifest.",
                path="$.loadUnitId",
            )
        return loader

    if loader.get("schemaVersion") != schema_version:
        raise NativeLoaderABIError(
            "manifest-schema-invalid",
            f"Runtime loader manifest schemaVersion must be {schema_version}.",
            path="$.schemaVersion",
            details={"actual": loader.get("schemaVersion"), "expected": schema_version},
        )
    if loader.get("kind") != manifest_kind:
        raise NativeLoaderABIError(
            "manifest-kind-invalid",
            f"Native loader ABI input manifest must have kind {manifest_kind!r}.",
            path="$.kind",
            details={"actual": loader.get("kind"), "expected": manifest_kind},
        )
    if loader.get("success") is not True:
        raise NativeLoaderABIError(
            "manifest-failed",
            "Runtime loader manifest must be successful before ABI generation.",
            path="$.success",
        )
    units = loader.get("loadUnits")
    if not isinstance(units, list) or not all(
        isinstance(unit, Mapping) for unit in units
    ):
        raise NativeLoaderABIError(
            "load-units-invalid",
            "Runtime loader manifest loadUnits must be a list of objects.",
            path="$.loadUnits",
        )
    if load_unit_id is None:
        if len(units) != 1:
            raise NativeLoaderABIError(
                "load-unit-selection-required",
                "A manifest with zero or multiple load units requires load_unit_id.",
                path="$.loadUnits",
                details={
                    "availableUnitIds": [_unit_identifier(unit) for unit in units]
                },
            )
        return units[0]
    if not isinstance(load_unit_id, str) or not load_unit_id:
        raise NativeLoaderABIError(
            "load-unit-selection-invalid",
            "load_unit_id must be a non-empty string.",
            path="$.loadUnitId",
        )
    matches = [unit for unit in units if unit.get("id") == load_unit_id]
    if len(matches) != 1:
        raise NativeLoaderABIError(
            "load-unit-not-found",
            f"Runtime loader manifest does not contain load unit {load_unit_id!r}.",
            path="$.loadUnits",
            details={"availableUnitIds": [_unit_identifier(unit) for unit in units]},
        )
    return matches[0]


def _validate_load_unit_shape(unit: Mapping[str, Any]) -> None:
    _schema_version, _manifest_kind, expected_fields = _pipeline_loader_contract()
    actual_fields = frozenset(unit)
    missing = sorted(expected_fields - actual_fields)
    unsupported = sorted(actual_fields - expected_fields)
    if missing or unsupported:
        raise NativeLoaderABIError(
            "load-unit-schema-invalid",
            "Runtime loader unit does not match RUNTIME_LOADER_MANIFEST_LOAD_UNIT_FIELDS.",
            details={"missingFields": missing, "unsupportedFields": unsupported},
        )


def _validate_load_unit_readiness(unit: Mapping[str, Any]) -> None:
    blockers = unit.get("blockers")
    if not isinstance(blockers, list):
        raise NativeLoaderABIError(
            "blockers-invalid",
            "Runtime loader unit blockers must be a list.",
            path="$.blockers",
        )
    if blockers:
        raise NativeLoaderABIError(
            "load-unit-blocked",
            "Blocked runtime loader units cannot produce native ABI declarations.",
            path="$.blockers",
            details={"blockers": copy.deepcopy(blockers)},
        )
    validation = unit.get("validation")
    if not isinstance(validation, Mapping):
        raise NativeLoaderABIError(
            "validation-invalid",
            "Runtime loader unit validation must be an object.",
            path="$.validation",
        )
    if validation.get("loadReady") is not True:
        raise NativeLoaderABIError(
            "load-unit-not-ready",
            "Runtime loader unit validation.loadReady must be true.",
            path="$.validation.loadReady",
        )
    if validation.get("hostInterface") != "ready":
        raise NativeLoaderABIError(
            "host-interface-not-ready",
            "Runtime loader unit validation.hostInterface must be ready.",
            path="$.validation.hostInterface",
        )


def _ready_host_interface(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeLoaderABIError(
            "host-interface-invalid",
            "Runtime loader unit hostInterface must be an object.",
            path="$.hostInterface",
        )
    if value.get("status") != "ready":
        raise NativeLoaderABIError(
            "host-interface-not-ready",
            "Runtime loader unit hostInterface.status must be ready.",
            path="$.hostInterface.status",
        )
    for field_name in (
        "entryPoints",
        "resources",
        "constants",
        "specializationConstants",
    ):
        if not isinstance(value.get(field_name), list):
            raise NativeLoaderABIError(
                "host-interface-invalid",
                f"Runtime loader unit hostInterface.{field_name} must be a list.",
                path=f"$.hostInterface.{field_name}",
            )
    for count_field, records_field in (
        ("entryPointCount", "entryPoints"),
        ("resourceCount", "resources"),
        ("constantCount", "constants"),
        ("specializationConstantCount", "specializationConstants"),
    ):
        count = value.get(count_field)
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or count != len(value[records_field])
        ):
            raise NativeLoaderABIError(
                "host-interface-count-invalid",
                f"Runtime loader unit hostInterface.{count_field} must match {records_field}.",
                path=f"$.hostInterface.{count_field}",
                details={"actual": count, "expected": len(value[records_field])},
            )
    return value


def _selected_entry_point(
    unit: Mapping[str, Any], host_interface: Mapping[str, Any]
) -> dict[str, Any]:
    entries = host_interface["entryPoints"]
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise NativeLoaderABIError(
                "entry-point-invalid",
                "Reflected entry points must be objects.",
                path=f"$.hostInterface.entryPoints[{index}]",
            )
        _required_string(
            entry.get("name"), path=f"$.hostInterface.entryPoints[{index}].name"
        )
        _json_mapping(
            entry.get("executionConfig"),
            path=f"$.hostInterface.entryPoints[{index}].executionConfig",
        )

    selected = unit.get("entryPoint")
    if selected is not None and not isinstance(selected, Mapping):
        raise NativeLoaderABIError(
            "entry-point-selection-invalid",
            "Runtime loader unit entryPoint must be an object or null.",
            path="$.entryPoint",
        )
    selected_name = selected.get("target") if isinstance(selected, Mapping) else None
    if selected_name is None:
        if len(entries) != 1:
            raise NativeLoaderABIError(
                "entry-point-selection-required",
                "A load unit with zero or multiple reflected entries requires entryPoint.target.",
                path="$.entryPoint",
                details={
                    "availableEntryPoints": [
                        entry.get("name")
                        for entry in entries
                        if isinstance(entry, Mapping)
                    ]
                },
            )
        selected_entry = entries[0]
    else:
        selected_name = _required_string(selected_name, path="$.entryPoint.target")
        matches = [entry for entry in entries if entry.get("name") == selected_name]
        if len(matches) != 1:
            raise NativeLoaderABIError(
                "entry-point-not-found",
                f"Reflected host interface does not contain entry point {selected_name!r}.",
                path="$.entryPoint.target",
                details={
                    "availableEntryPoints": [entry.get("name") for entry in entries]
                },
            )
        selected_entry = matches[0]

    stage = selected_entry.get("stage") or unit.get("stage")
    if stage is not None:
        stage = _required_string(stage, path="$.hostInterface.entryPoints[].stage")
    declared_stage = selected.get("stage") if isinstance(selected, Mapping) else None
    if declared_stage is not None:
        declared_stage = _required_string(declared_stage, path="$.entryPoint.stage")
    if declared_stage is not None and stage is not None and declared_stage != stage:
        raise NativeLoaderABIError(
            "entry-point-stage-mismatch",
            "Selected entry-point stage conflicts with reflected host-interface metadata.",
            path="$.entryPoint.stage",
            details={"selected": declared_stage, "reflected": stage},
        )
    unit_stage = unit.get("stage")
    if unit_stage is not None:
        unit_stage = _required_string(unit_stage, path="$.stage")
    if unit_stage is not None and stage is not None and unit_stage != stage:
        raise NativeLoaderABIError(
            "entry-point-stage-mismatch",
            "Load-unit stage conflicts with reflected host-interface metadata.",
            path="$.stage",
            details={"loadUnit": unit_stage, "reflected": stage},
        )
    return {
        "name": selected_entry["name"],
        "stage": stage,
        "executionConfig": _json_mapping(
            selected_entry.get("executionConfig"), path="$.entryPoint.executionConfig"
        ),
        "provenance": _json_mapping(
            selected_entry.get("provenance", {}), path="$.entryPoint.provenance"
        ),
    }


def _binding_descriptors(target: str, value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise NativeLoaderABIError(
            "bindings-invalid",
            "Runtime loader host-interface resources must be a list.",
            path="$.hostInterface.resources",
        )
    bindings: list[dict[str, Any]] = []
    coordinates: dict[tuple[str, int, int], str] = {}
    for index, resource in enumerate(value):
        path = f"$.hostInterface.resources[{index}]"
        if not isinstance(resource, Mapping):
            raise NativeLoaderABIError(
                "binding-invalid", "Reflected resource must be an object.", path=path
            )
        name = _required_string(resource.get("name"), path=f"{path}.name")
        kind = _required_string(resource.get("kind"), path=f"{path}.kind")
        type_name = _required_string(resource.get("type"), path=f"{path}.type")
        set_index = _coordinate(resource.get("set"), path=f"{path}.set")
        binding_index = _coordinate(resource.get("binding"), path=f"{path}.binding")
        access = resource.get("access")
        if access is not None and access not in _ACCESS_VALUES:
            raise NativeLoaderABIError(
                "binding-access-invalid",
                "Binding access must be read, write, read_write, or null.",
                path=f"{path}.access",
                details={"access": access},
            )
        namespace = _binding_namespace(target, resource)
        coordinate = (namespace, set_index, binding_index)
        if coordinate in coordinates:
            raise NativeLoaderABIError(
                "binding-coordinate-duplicate",
                "Two reflected resources use the same native binding coordinate.",
                path=path,
                details={
                    "resource": name,
                    "conflictingResource": coordinates[coordinate],
                    "namespace": namespace,
                    "set": set_index,
                    "binding": binding_index,
                },
            )
        coordinates[coordinate] = name
        metadata = _json_mapping(resource.get("metadata", {}), path=f"{path}.metadata")
        scalar_layout_value = resource.get("scalarLayout")
        if scalar_layout_value is None:
            scalar_layout_value = metadata.get("scalarLayout")
        scalar_layout = (
            _json_mapping(scalar_layout_value, path=f"{path}.scalarLayout")
            if scalar_layout_value is not None
            else None
        )
        provenance = _json_mapping(
            resource.get("provenance", metadata.get("provenance", {})),
            path=f"{path}.provenance",
        )
        bindings.append(
            {
                "name": name,
                "kind": kind,
                "type": type_name,
                "namespace": namespace,
                "coordinates": {"set": set_index, "binding": binding_index},
                "access": access,
                "scalarLayout": scalar_layout,
                "provenance": provenance,
            }
        )
    return sorted(
        bindings,
        key=lambda binding: (
            binding["namespace"],
            binding["coordinates"]["set"],
            binding["coordinates"]["binding"],
            binding["name"],
        ),
    )


def _binding_namespace(target: str, resource: Mapping[str, Any]) -> str:
    metadata = resource.get("metadata")
    if isinstance(metadata, Mapping):
        explicit = metadata.get("bindingNamespace", metadata.get("registerClass"))
        if isinstance(explicit, str) and explicit:
            return explicit.lower()
    kind = str(resource.get("kind") or "").lower()
    access = resource.get("access")
    type_name = str(resource.get("type") or "").lower().replace(" ", "")
    if target == "directx":
        if kind == "sampler":
            return "sampler"
        if kind in {"constant-buffer", "uniform"}:
            return "cbv"
        if access in {"write", "read_write"} or type_name.startswith("rw"):
            return "uav"
        return "srv"
    if target in {"opengl", "webgl"}:
        if kind in {"sampler", "texture"}:
            return "texture"
        if kind == "storage-texture":
            return "image"
        if kind in {"constant-buffer", "uniform"}:
            return "uniform-buffer"
        if kind == "buffer":
            return "storage-buffer"
    return kind


def _scalar_layout_descriptor(
    host_interface: Mapping[str, Any], bindings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    constants = []
    for index, constant in enumerate(host_interface["constants"]):
        path = f"$.hostInterface.constants[{index}]"
        if not isinstance(constant, Mapping):
            raise NativeLoaderABIError(
                "scalar-layout-invalid",
                "Scalar layout constants must be objects.",
                path=path,
            )
        constants.append(_json_mapping(constant, path=path))
    constants.sort(key=lambda value: _canonical_json(value))
    binding_layouts = [
        {"binding": binding["name"], "layout": copy.deepcopy(binding["scalarLayout"])}
        for binding in bindings
        if binding.get("scalarLayout") is not None
    ]
    return {"constants": constants, "bindings": binding_layouts}


def _specialization_descriptors(
    unit: Mapping[str, Any], host_interface: Mapping[str, Any]
) -> list[dict[str, Any]]:
    unit_constants = unit.get("specializationConstants")
    reflected_constants = host_interface.get("specializationConstants")
    if not isinstance(unit_constants, list):
        raise NativeLoaderABIError(
            "specialization-constants-invalid",
            "Runtime loader unit specializationConstants must be a list.",
            path="$.specializationConstants",
        )
    values = unit_constants if unit_constants else reflected_constants
    if not isinstance(values, list):
        raise NativeLoaderABIError(
            "specialization-constants-invalid",
            "Reflected specializationConstants must be a list.",
            path="$.hostInterface.specializationConstants",
        )
    result = []
    identities: set[tuple[str, Any]] = set()
    for index, constant in enumerate(values):
        path = f"$.specializationConstants[{index}]"
        if not isinstance(constant, Mapping):
            raise NativeLoaderABIError(
                "specialization-constant-invalid",
                "Specialization constants must be objects.",
                path=path,
            )
        constant_id = constant.get("id", constant.get("constantId"))
        name = constant.get("name")
        if constant_id is not None:
            if (
                not isinstance(constant_id, int)
                or isinstance(constant_id, bool)
                or constant_id < 0
                or constant_id > _UINT32_MAX
            ):
                raise NativeLoaderABIError(
                    "specialization-id-invalid",
                    "Specialization constant id must be a uint32 value or null.",
                    path=f"{path}.id",
                )
            identity = ("id", constant_id)
        elif isinstance(name, str) and name:
            identity = ("name", name)
        else:
            raise NativeLoaderABIError(
                "specialization-identity-missing",
                "Specialization constant must have an id or non-empty name.",
                path=path,
            )
        if identity in identities:
            raise NativeLoaderABIError(
                "specialization-identity-duplicate",
                "Specialization constant identities must be unique.",
                path=path,
                details={"identity": list(identity)},
            )
        identities.add(identity)
        result.append(_json_mapping(constant, path=path))
    return sorted(
        result,
        key=lambda constant: (
            constant.get("id", _UINT32_MAX + 1),
            str(constant.get("name") or ""),
            _canonical_json(constant),
        ),
    )


def _artifact_descriptor(unit: Mapping[str, Any]) -> dict[str, Any]:
    artifact_hash = _sha256_hash(
        unit.get("hash"),
        path="$.hash",
        code="artifact-hash-invalid",
        label="Runtime loader unit hash",
    )
    size_bytes = _uint64(
        unit.get("sizeBytes"),
        path="$.sizeBytes",
        code="artifact-size-invalid",
        label="Runtime loader unit sizeBytes",
    )
    return {
        "packagePath": _required_string(unit.get("packagePath"), path="$.packagePath"),
        "format": _required_string(unit.get("artifactFormat"), path="$.artifactFormat"),
        "hash": artifact_hash,
        "sizeBytes": size_bytes,
    }


def _source_descriptor(unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": _optional_string(unit.get("source"), path="$.source"),
        "artifactPath": _optional_string(unit.get("sourcePath"), path="$.sourcePath"),
        "backend": _optional_string(unit.get("sourceBackend"), path="$.sourceBackend"),
        "hash": _optional_hash(unit.get("sourceHash"), path="$.sourceHash"),
        "remap": _source_remap_descriptor(
            unit.get("sourceRemap"),
            path="$.sourceRemap",
            code_prefix="source-remap",
            label="Source remap",
        ),
    }


def _optional_hash(value: Any, *, path: str) -> dict[str, str] | None:
    if value is None:
        return None
    return _sha256_hash(
        value,
        path=path,
        code="source-hash-invalid",
        label="Source hash",
    )


def _validate_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise NativeLoaderABIError(
            "descriptor-invalid", "Native loader ABI descriptor must be an object."
        )
    required_fields = frozenset(
        (
            "schemaVersion",
            "kind",
            "abiVersion",
            "unitId",
            "target",
            "stage",
            "entryPoint",
            "artifact",
            "source",
            "bindings",
            "scalarLayout",
            "specializationConstants",
            "provenance",
        )
    )
    actual_fields = frozenset(descriptor)
    missing = sorted(required_fields - actual_fields)
    unsupported = sorted(actual_fields - required_fields, key=lambda item: str(item))
    if missing or unsupported:
        raise NativeLoaderABIError(
            "descriptor-schema-invalid",
            "Native loader ABI descriptor has an invalid field set.",
            details={"missingFields": missing, "unsupportedFields": unsupported},
        )
    if (
        type(descriptor.get("schemaVersion")) is not int
        or descriptor.get("schemaVersion") != NATIVE_LOADER_ABI_VERSION
        or type(descriptor.get("abiVersion")) is not int
        or descriptor.get("abiVersion") != NATIVE_LOADER_ABI_VERSION
        or descriptor.get("kind") != NATIVE_LOADER_ABI_KIND
    ):
        raise NativeLoaderABIError(
            "descriptor-version-invalid",
            "Native loader ABI descriptor version or kind is unsupported.",
        )
    normalized = _json_mapping(descriptor, path="$")
    _required_string(normalized.get("unitId"), path="$.unitId")
    _required_string(normalized.get("target"), path="$.target")
    if normalized.get("stage") is not None:
        _required_string(normalized.get("stage"), path="$.stage")
    entry_point = _mapping_with_fields(
        normalized.get("entryPoint"),
        fields=frozenset(("name", "stage", "executionConfig", "provenance")),
        path="$.entryPoint",
        code="descriptor-entry-point-invalid",
        label="Native loader ABI entryPoint",
    )
    _required_string(entry_point.get("name"), path="$.entryPoint.name")
    entry_stage = _optional_string(entry_point.get("stage"), path="$.entryPoint.stage")
    if entry_stage != normalized.get("stage"):
        raise NativeLoaderABIError(
            "descriptor-entry-point-invalid",
            "Native loader ABI entryPoint.stage must match stage.",
            path="$.entryPoint.stage",
        )
    _json_mapping(
        entry_point.get("executionConfig"), path="$.entryPoint.executionConfig"
    )
    _json_mapping(entry_point.get("provenance"), path="$.entryPoint.provenance")

    _validate_descriptor_artifact(normalized.get("artifact"))
    _validate_descriptor_source(normalized.get("source"))
    bindings = _validate_descriptor_bindings(normalized.get("bindings"))
    _validate_descriptor_scalar_layout(normalized.get("scalarLayout"), bindings)
    _validate_descriptor_specialization_constants(
        normalized.get("specializationConstants")
    )
    if not isinstance(normalized.get("provenance"), Mapping):
        raise NativeLoaderABIError(
            "descriptor-provenance-invalid",
            "Native loader ABI provenance must be an object.",
            path="$.provenance",
        )
    return normalized


def _validate_descriptor_artifact(value: Any) -> None:
    artifact = _mapping_with_fields(
        value,
        fields=frozenset(("packagePath", "format", "hash", "sizeBytes")),
        path="$.artifact",
        code="descriptor-artifact-invalid",
        label="Native loader ABI artifact",
    )
    _required_string(artifact.get("packagePath"), path="$.artifact.packagePath")
    _required_string(artifact.get("format"), path="$.artifact.format")
    _sha256_hash(
        artifact.get("hash"),
        path="$.artifact.hash",
        code="descriptor-artifact-hash-invalid",
        label="Native loader ABI artifact hash",
    )
    _uint64(
        artifact.get("sizeBytes"),
        path="$.artifact.sizeBytes",
        code="descriptor-artifact-size-invalid",
        label="Native loader ABI artifact sizeBytes",
    )


def _validate_descriptor_source(value: Any) -> None:
    source = _mapping_with_fields(
        value,
        fields=frozenset(("path", "artifactPath", "backend", "hash", "remap")),
        path="$.source",
        code="descriptor-source-invalid",
        label="Native loader ABI source",
    )
    _optional_string(source.get("path"), path="$.source.path")
    _optional_string(source.get("artifactPath"), path="$.source.artifactPath")
    _optional_string(source.get("backend"), path="$.source.backend")
    source_hash = source.get("hash")
    if source_hash is not None:
        _sha256_hash(
            source_hash,
            path="$.source.hash",
            code="descriptor-source-hash-invalid",
            label="Native loader ABI source hash",
        )
    _source_remap_descriptor(
        source.get("remap"),
        path="$.source.remap",
        code_prefix="descriptor-source-remap",
        label="Native loader ABI source remap",
    )


def _validate_descriptor_bindings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise NativeLoaderABIError(
            "descriptor-bindings-invalid",
            "Native loader ABI bindings must be a list.",
            path="$.bindings",
        )
    bindings: list[Mapping[str, Any]] = []
    names: set[str] = set()
    coordinates: set[tuple[str, int, int]] = set()
    for index, binding_value in enumerate(value):
        path = f"$.bindings[{index}]"
        binding = _mapping_with_fields(
            binding_value,
            fields=frozenset(
                (
                    "name",
                    "kind",
                    "type",
                    "namespace",
                    "coordinates",
                    "access",
                    "scalarLayout",
                    "provenance",
                )
            ),
            path=path,
            code="descriptor-binding-invalid",
            label="Native loader ABI binding",
        )
        name = _required_string(binding.get("name"), path=f"{path}.name")
        _required_string(binding.get("kind"), path=f"{path}.kind")
        _required_string(binding.get("type"), path=f"{path}.type")
        namespace = _required_string(binding.get("namespace"), path=f"{path}.namespace")
        coordinate_value = _mapping_with_fields(
            binding.get("coordinates"),
            fields=frozenset(("set", "binding")),
            path=f"{path}.coordinates",
            code="descriptor-binding-coordinate-invalid",
            label="Native loader ABI binding coordinates",
        )
        set_index = _coordinate(
            coordinate_value.get("set"), path=f"{path}.coordinates.set"
        )
        binding_index = _coordinate(
            coordinate_value.get("binding"), path=f"{path}.coordinates.binding"
        )
        access = binding.get("access")
        if access is not None and access not in _ACCESS_VALUES:
            raise NativeLoaderABIError(
                "descriptor-binding-access-invalid",
                "Native loader ABI binding access must be read, write, read_write, or null.",
                path=f"{path}.access",
            )
        scalar_layout = binding.get("scalarLayout")
        if scalar_layout is not None:
            _json_mapping(scalar_layout, path=f"{path}.scalarLayout")
        _json_mapping(binding.get("provenance"), path=f"{path}.provenance")
        if name in names:
            raise NativeLoaderABIError(
                "descriptor-binding-name-duplicate",
                "Native loader ABI binding names must be unique.",
                path=f"{path}.name",
                details={"name": name},
            )
        coordinate = (namespace, set_index, binding_index)
        if coordinate in coordinates:
            raise NativeLoaderABIError(
                "descriptor-binding-coordinate-duplicate",
                "Native loader ABI binding coordinates must be unique within a namespace.",
                path=f"{path}.coordinates",
                details={
                    "namespace": namespace,
                    "set": set_index,
                    "binding": binding_index,
                },
            )
        names.add(name)
        coordinates.add(coordinate)
        bindings.append(binding)
    return bindings


def _validate_descriptor_scalar_layout(
    value: Any, bindings: Sequence[Mapping[str, Any]]
) -> None:
    scalar_layout = _mapping_with_fields(
        value,
        fields=frozenset(("constants", "bindings")),
        path="$.scalarLayout",
        code="descriptor-scalar-layout-invalid",
        label="Native loader ABI scalarLayout",
    )
    constants = scalar_layout.get("constants")
    if not isinstance(constants, list):
        raise NativeLoaderABIError(
            "descriptor-scalar-layout-invalid",
            "Native loader ABI scalarLayout.constants must be a list.",
            path="$.scalarLayout.constants",
        )
    for index, constant in enumerate(constants):
        _json_mapping(constant, path=f"$.scalarLayout.constants[{index}]")

    binding_layouts = scalar_layout.get("bindings")
    if not isinstance(binding_layouts, list):
        raise NativeLoaderABIError(
            "descriptor-scalar-layout-invalid",
            "Native loader ABI scalarLayout.bindings must be a list.",
            path="$.scalarLayout.bindings",
        )
    actual_layouts: dict[str, Mapping[str, Any]] = {}
    for index, layout_value in enumerate(binding_layouts):
        path = f"$.scalarLayout.bindings[{index}]"
        layout = _mapping_with_fields(
            layout_value,
            fields=frozenset(("binding", "layout")),
            path=path,
            code="descriptor-scalar-layout-invalid",
            label="Native loader ABI scalar binding layout",
        )
        binding_name = _required_string(layout.get("binding"), path=f"{path}.binding")
        layout_data = _json_mapping(layout.get("layout"), path=f"{path}.layout")
        if binding_name in actual_layouts:
            raise NativeLoaderABIError(
                "descriptor-scalar-layout-invalid",
                "Native loader ABI scalar binding layout names must be unique.",
                path=f"{path}.binding",
            )
        actual_layouts[binding_name] = layout_data

    expected_layouts = {
        str(binding["name"]): binding["scalarLayout"]
        for binding in bindings
        if binding.get("scalarLayout") is not None
    }
    if actual_layouts != expected_layouts:
        raise NativeLoaderABIError(
            "descriptor-scalar-layout-invalid",
            "Native loader ABI scalarLayout.bindings must match binding scalar layouts.",
            path="$.scalarLayout.bindings",
        )


def _validate_descriptor_specialization_constants(value: Any) -> None:
    if not isinstance(value, list):
        raise NativeLoaderABIError(
            "descriptor-specialization-invalid",
            "Native loader ABI specializationConstants must be a list.",
            path="$.specializationConstants",
        )
    for index, constant in enumerate(value):
        _json_mapping(constant, path=f"$.specializationConstants[{index}]")


def _mapping_with_fields(
    value: Any,
    *,
    fields: frozenset[str],
    path: str,
    code: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeLoaderABIError(
            code,
            f"{label} must be an object.",
            path=path,
        )
    actual_fields = frozenset(value)
    missing = sorted(fields - actual_fields)
    unsupported = sorted(actual_fields - fields, key=lambda item: str(item))
    if missing or unsupported:
        raise NativeLoaderABIError(
            code,
            f"{label} has an invalid field set.",
            path=path,
            details={"missingFields": missing, "unsupportedFields": unsupported},
        )
    return value


def _sha256_hash(
    value: Any,
    *,
    path: str,
    code: str,
    label: str,
) -> dict[str, str]:
    hash_value = _mapping_with_fields(
        value,
        fields=frozenset(("algorithm", "value")),
        path=path,
        code=code,
        label=label,
    )
    algorithm = hash_value.get("algorithm")
    digest = hash_value.get("value")
    if (
        algorithm != "sha256"
        or not isinstance(digest, str)
        or not _SHA256_RE.fullmatch(digest)
    ):
        raise NativeLoaderABIError(
            code,
            f"{label} must contain a lowercase SHA-256 digest.",
            path=path,
            details={"algorithm": algorithm, "value": digest},
        )
    return {"algorithm": algorithm, "value": digest}


def _uint64(value: Any, *, path: str, code: str, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _UINT64_MAX
    ):
        raise NativeLoaderABIError(
            code,
            f"{label} must be a uint64 value.",
            path=path,
            details={"value": value},
        )
    return value


def _source_remap_descriptor(
    value: Any,
    *,
    path: str,
    code_prefix: str,
    label: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise NativeLoaderABIError(
            f"{code_prefix}-invalid",
            f"{label} must be an object or null.",
            path=path,
        )
    integrity_fields = frozenset(("sourcePath", "packagePath", "hash", "sizeBytes"))
    inspection_fields = frozenset(("sourcePath", "packagePath", "status"))
    actual_fields = frozenset(value)
    if actual_fields not in (integrity_fields, inspection_fields):
        raise NativeLoaderABIError(
            f"{code_prefix}-invalid",
            f"{label} has an invalid field set.",
            path=path,
            details={
                "actualFields": sorted(actual_fields, key=lambda item: str(item)),
                "acceptedFieldSets": [
                    sorted(integrity_fields),
                    sorted(inspection_fields),
                ],
            },
        )
    remap = value
    source_path = _required_string(remap.get("sourcePath"), path=f"{path}.sourcePath")
    package_path = _required_string(
        remap.get("packagePath"), path=f"{path}.packagePath"
    )
    if actual_fields == inspection_fields:
        status = _required_string(remap.get("status"), path=f"{path}.status")
        if status not in {"ready", "failed"}:
            raise NativeLoaderABIError(
                f"{code_prefix}-status-invalid",
                f"{label} status must be ready or failed.",
                path=f"{path}.status",
                details={"status": status},
            )
        return {
            "sourcePath": source_path,
            "packagePath": package_path,
            "status": status,
        }
    remap_hash = _sha256_hash(
        remap.get("hash"),
        path=f"{path}.hash",
        code=f"{code_prefix}-hash-invalid",
        label=f"{label} hash",
    )
    size_bytes = _uint64(
        remap.get("sizeBytes"),
        path=f"{path}.sizeBytes",
        code=f"{code_prefix}-size-invalid",
        label=f"{label} sizeBytes",
    )
    return {
        "sourcePath": source_path,
        "packagePath": package_path,
        "hash": remap_hash,
        "sizeBytes": size_bytes,
    }


def _json_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeLoaderABIError(
            "json-object-invalid", "Value must be a JSON object.", path=path
        )
    normalized = _json_value(value, path=path)
    assert isinstance(normalized, dict)
    return normalized


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise NativeLoaderABIError(
                "json-value-invalid", "JSON numbers must be finite.", path=path
            )
        return value
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str) or not key:
                raise NativeLoaderABIError(
                    "json-key-invalid",
                    "JSON object keys must be non-empty strings.",
                    path=path,
                )
            result[key] = _json_value(value[key], path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise NativeLoaderABIError(
        "json-value-invalid",
        "Value must contain only JSON-compatible data.",
        path=path,
        details={"type": type(value).__name__},
    )


def _coordinate(value: Any, *, path: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _UINT32_MAX
    ):
        raise NativeLoaderABIError(
            "binding-coordinate-invalid",
            "Binding coordinates must be uint32 values.",
            path=path,
            details={"value": value},
        )
    return value


def _required_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise NativeLoaderABIError(
            "string-invalid", "Value must be a non-empty trimmed string.", path=path
        )
    return value


def _optional_string(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, path=path)


def _unit_identifier(unit: Mapping[str, Any]) -> Any:
    return unit.get("id") if isinstance(unit, Mapping) else None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _descriptor_symbol(descriptor: Mapping[str, Any]) -> str:
    slug = _C_IDENTIFIER_RE.sub("_", str(descriptor["unitId"])).strip("_").lower()
    if not slug or slug[0].isdigit():
        slug = f"unit_{slug}"
    identity = _canonical_json(
        {
            "abiVersion": descriptor["abiVersion"],
            "unitId": descriptor["unitId"],
            "target": descriptor["target"],
            "artifactHash": descriptor["artifact"]["hash"],
        }
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"crosstl_native_loader_{slug}_{suffix}"


def _c_access(value: Any) -> str:
    return {
        None: "CROSSTL_NATIVE_LOADER_ACCESS_NONE",
        "read": "CROSSTL_NATIVE_LOADER_ACCESS_READ",
        "write": "CROSSTL_NATIVE_LOADER_ACCESS_WRITE",
        "read_write": "CROSSTL_NATIVE_LOADER_ACCESS_READ_WRITE",
    }[value]


def _c_json_string(value: Any) -> str:
    return _c_string(_canonical_json(value))


def _c_nullable_json_string(value: Any) -> str:
    return "NULL" if value is None else _c_json_string(value)


def _c_nullable_string(value: Any) -> str:
    return "NULL" if value is None else _c_string(value)


def _c_string(value: Any) -> str:
    encoded = str(value).encode("utf-8")
    characters = ['"']
    for byte in encoded:
        if byte == 34:
            characters.append(r"\"")
        elif byte == 92:
            characters.append(r"\\")
        elif byte == 10:
            characters.append(r"\n")
        elif byte == 13:
            characters.append(r"\r")
        elif byte == 9:
            characters.append(r"\t")
        elif 32 <= byte <= 126:
            characters.append(chr(byte))
        else:
            characters.append(f"\\{byte:03o}")
    characters.append('"')
    return "".join(characters)
