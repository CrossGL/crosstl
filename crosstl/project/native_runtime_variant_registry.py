"""Generate native exact-lookup registries for loader ABI execution wrappers."""

from __future__ import annotations

import copy
import math
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .native_loader_abi import (
    NativeLoaderABIError,
    _descriptor_symbol,
    _execution_workgroup_size,
    _validate_descriptor,
)
from .pipeline import (
    RUNTIME_VARIANT_REGISTRY_KIND,
    lookup_runtime_variant,
)

_ERROR_PREFIX = "project.native-runtime-variant-registry"
_SUPPORTED_TARGETS = frozenset(("directx", "opengl"))
_SUPPORTED_ARTIFACT_FORMATS = {
    "directx": frozenset(("HLSL source", "DXIL", "DXIL binary")),
    "opengl": frozenset(("GLSL source", "SPIR-V binary")),
}
_UINT32_MAX = (1 << 32) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_DTYPE_ALIASES = {
    "bool": "bool",
    "boolean": "bool",
    "int": "int32",
    "i32": "int32",
    "int32": "int32",
    "int32_t": "int32",
    "uint": "uint32",
    "u32": "uint32",
    "uint32": "uint32",
    "uint32_t": "uint32",
    "float": "float32",
    "f32": "float32",
    "float32": "float32",
}
_UNIT_FIELDS = frozenset(("descriptor", "executionHeader"))
_C_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]+")
_PORTABLE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class NativeRuntimeVariantRegistryError(ValueError):
    """A runtime variant registry cannot form a native execution header."""

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
        """Return the stable diagnostic representation."""

        payload: dict[str, Any] = {
            "severity": "error",
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.details:
            payload["details"] = copy.deepcopy(self.details)
        return payload


def generate_native_runtime_variant_registry(
    registry: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> str:
    """Render a deterministic C++17 exact-lookup execution registry.

    ``units`` associates each validated native-loader ABI descriptor with the
    package-relative execution header that defines its generated wrapper.
    Every registry variant must be ready and must match exactly one unit.
    """

    variants = _validated_registry(registry)
    normalized_units = _validated_units(units)
    entries = [
        _native_variant_entry(key, record, normalized_units, index=index)
        for index, (key, record) in enumerate(sorted(variants.items()))
    ]
    lines = [
        "#ifndef CROSSTL_NATIVE_RUNTIME_VARIANT_REGISTRY_HPP",
        "#define CROSSTL_NATIVE_RUNTIME_VARIANT_REGISTRY_HPP",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
    ]
    for execution_header in sorted({entry["executionHeader"] for entry in entries}):
        lines.append(f'#include "{execution_header}"')
    lines.extend(
        [
            "",
            "typedef CrossTLNativeLoaderExecutionResult",
            "(*CrossTLNativeRuntimeVariantExecute)(",
            "    const CrossTLNativeLoaderExecutionRequest *request,",
            "    const CrossTLNativeLoaderAdapter *adapter);",
            "",
            "typedef struct CrossTLNativeRuntimeVariantEntry {",
            "    const char *key;",
            "    const char *unit_id;",
            "    const char *target;",
            "    const char *entry_point;",
            "    uint32_t workgroup_size[3];",
            "    uint32_t has_subgroup_width;",
            "    uint32_t subgroup_width;",
            "    size_t specialization_count;",
            "    const CrossTLNativeLoaderSpecializationRequest *specializations;",
            "    const CrossTLNativeLoaderUnitDescriptor *unit;",
            "    CrossTLNativeRuntimeVariantExecute execute;",
            "} CrossTLNativeRuntimeVariantEntry;",
            "",
        ]
    )
    for entry in entries:
        lines.extend(_specialization_declarations(entry))
    lines.extend(
        [
            "static const CrossTLNativeRuntimeVariantEntry",
            f"crosstl_native_runtime_variants[{len(entries)}] = {{",
        ]
    )
    for entry in entries:
        workgroup_size = entry["workgroupSize"]
        lines.extend(
            [
                "    {",
                f"        {_c_string(entry['key'])},",
                f"        {_c_string(entry['unitId'])},",
                f"        {_c_string(entry['target'])},",
                f"        {_c_string(entry['entryPoint'])},",
                "        {"
                f"{workgroup_size[0]}u, "
                f"{workgroup_size[1]}u, "
                f"{workgroup_size[2]}u"
                "},",
                f"        {1 if entry['subgroupWidth'] is not None else 0}u,",
                f"        {entry['subgroupWidth'] or 0}u,",
                f"        {len(entry['specializations'])}u,",
                (
                    f"        {entry['symbol']}_specializations,"
                    if entry["specializations"]
                    else "        NULL,"
                ),
                f"        &{entry['descriptorSymbol']},",
                f"        {entry['executeSymbol']},",
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            "static inline const CrossTLNativeRuntimeVariantEntry *",
            "crosstl_native_runtime_variant_lookup(const char *key) {",
            "    size_t index = 0u;",
            "    if (key == NULL) {",
            "        return NULL;",
            "    }",
            f"    for (index = 0u; index < {len(entries)}u; ++index) {{",
            "        if (crosstl_native_loader_strings_equal(",
            "                key, crosstl_native_runtime_variants[index].key)) {",
            "            return &crosstl_native_runtime_variants[index];",
            "        }",
            "    }",
            "    return NULL;",
            "}",
            "",
            "static inline int crosstl_native_runtime_variant_is_registered(",
            "    const CrossTLNativeRuntimeVariantEntry *variant) {",
            "    size_t index = 0u;",
            "    if (variant == NULL) {",
            "        return 0;",
            "    }",
            f"    for (index = 0u; index < {len(entries)}u; ++index) {{",
            "        if (variant == &crosstl_native_runtime_variants[index]) {",
            "            return 1;",
            "        }",
            "    }",
            "    return 0;",
            "}",
            "",
            "static inline CrossTLNativeLoaderExecutionRequest",
            "crosstl_native_runtime_variant_make_request(",
            "    const CrossTLNativeRuntimeVariantEntry *variant,",
            "    size_t binding_count,",
            "    const CrossTLNativeLoaderBindingRequest *bindings,",
            "    const uint32_t workgroup_count[3]) {",
            "    CrossTLNativeLoaderExecutionRequest request = {};",
            "    request.abi_version = CROSSTL_NATIVE_LOADER_ABI_VERSION;",
            "    request.binding_count = binding_count;",
            "    request.bindings = bindings;",
            "    if (!crosstl_native_runtime_variant_is_registered(variant)) {",
            "        return request;",
            "    }",
            "    request.target = variant->target;",
            "    request.specialization_count = variant->specialization_count;",
            "    request.specializations = variant->specializations;",
            "    if (workgroup_count != NULL) {",
            "        request.dispatch.workgroup_count[0] = workgroup_count[0];",
            "        request.dispatch.workgroup_count[1] = workgroup_count[1];",
            "        request.dispatch.workgroup_count[2] = workgroup_count[2];",
            "    }",
            "    request.dispatch.workgroup_size[0] = variant->workgroup_size[0];",
            "    request.dispatch.workgroup_size[1] = variant->workgroup_size[1];",
            "    request.dispatch.workgroup_size[2] = variant->workgroup_size[2];",
            "    return request;",
            "}",
            "",
            "static inline int crosstl_native_runtime_variant_bytes_equal(",
            "    const void *left, const void *right, size_t size) {",
            "    const uint8_t *left_bytes = static_cast<const uint8_t *>(left);",
            "    const uint8_t *right_bytes = static_cast<const uint8_t *>(right);",
            "    size_t index = 0u;",
            "    if (left == NULL || right == NULL) {",
            "        return left == right && size == 0u;",
            "    }",
            "    for (index = 0u; index < size; ++index) {",
            "        if (left_bytes[index] != right_bytes[index]) {",
            "            return 0;",
            "        }",
            "    }",
            "    return 1;",
            "}",
            "",
            "static inline CrossTLNativeLoaderExecutionResult",
            "crosstl_native_runtime_variant_execute(",
            "    const CrossTLNativeRuntimeVariantEntry *variant,",
            "    const CrossTLNativeLoaderExecutionRequest *request,",
            "    const CrossTLNativeLoaderAdapter *adapter) {",
            "    size_t index = 0u;",
            "    if (!crosstl_native_runtime_variant_is_registered(variant) ||",
            "        request == NULL || adapter == NULL) {",
            "        return crosstl_native_loader_execution_failure(",
            "            CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "            CROSSTL_NATIVE_LOADER_CODE_INVALID_ARGUMENT,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX, 0);",
            "    }",
            "    if (request->abi_version != CROSSTL_NATIVE_LOADER_ABI_VERSION ||",
            "        adapter->abi_version != CROSSTL_NATIVE_LOADER_ABI_VERSION) {",
            "        return crosstl_native_loader_execution_failure(",
            "            CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "            CROSSTL_NATIVE_LOADER_CODE_ABI_VERSION_MISMATCH,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX, 0);",
            "    }",
            "    if (!crosstl_native_loader_strings_equal(",
            "            request->target, variant->target) ||",
            "        !crosstl_native_loader_strings_equal(",
            "            adapter->target, variant->target)) {",
            "        return crosstl_native_loader_execution_failure(",
            "            CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "            CROSSTL_NATIVE_LOADER_CODE_TARGET_MISMATCH,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX, 0);",
            "    }",
            "    if (request->dispatch.workgroup_size[0] !=",
            "            variant->workgroup_size[0] ||",
            "        request->dispatch.workgroup_size[1] !=",
            "            variant->workgroup_size[1] ||",
            "        request->dispatch.workgroup_size[2] !=",
            "            variant->workgroup_size[2]) {",
            "        return crosstl_native_loader_execution_failure(",
            "            CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "            CROSSTL_NATIVE_LOADER_CODE_WORKGROUP_SIZE_MISMATCH,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX, 0);",
            "    }",
            "    if (request->specialization_count !=",
            "            variant->specialization_count) {",
            "        return crosstl_native_loader_execution_failure(",
            "            CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "            CROSSTL_NATIVE_LOADER_CODE_SPECIALIZATION_COUNT_MISMATCH,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX, 0);",
            "    }",
            "    if (variant->specialization_count != 0u &&",
            "        request->specializations == NULL) {",
            "        return crosstl_native_loader_execution_failure(",
            "            CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "            CROSSTL_NATIVE_LOADER_CODE_INVALID_ARGUMENT,",
            "            CROSSTL_NATIVE_LOADER_NO_INDEX, 0u, 0);",
            "    }",
            "    for (index = 0u; index < variant->specialization_count; ++index) {",
            "        const CrossTLNativeLoaderSpecializationRequest *actual =",
            "            &request->specializations[index];",
            "        const CrossTLNativeLoaderSpecializationRequest *expected =",
            "            &variant->specializations[index];",
            "        if (actual->has_id != expected->has_id ||",
            "            actual->id != expected->id ||",
            "            !crosstl_native_loader_strings_equal(",
            "                actual->name, expected->name)) {",
            "            return crosstl_native_loader_execution_failure(",
            "                CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "                CROSSTL_NATIVE_LOADER_CODE_SPECIALIZATION_IDENTITY_MISMATCH,",
            "                CROSSTL_NATIVE_LOADER_NO_INDEX, index, 0);",
            "        }",
            "        if (!crosstl_native_loader_strings_equal(",
            "                actual->type_name, expected->type_name)) {",
            "            return crosstl_native_loader_execution_failure(",
            "                CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "                CROSSTL_NATIVE_LOADER_CODE_SPECIALIZATION_TYPE_MISMATCH,",
            "                CROSSTL_NATIVE_LOADER_NO_INDEX, index, 0);",
            "        }",
            "        if (actual->payload == NULL ||",
            "            actual->payload_size_bytes == 0u) {",
            "            return crosstl_native_loader_execution_failure(",
            "                CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "                CROSSTL_NATIVE_LOADER_CODE_SPECIALIZATION_PAYLOAD_MISSING,",
            "                CROSSTL_NATIVE_LOADER_NO_INDEX, index, 0);",
            "        }",
            "        if (actual->payload_size_bytes !=",
            "                expected->payload_size_bytes ||",
            "            !crosstl_native_runtime_variant_bytes_equal(",
            "                actual->payload,",
            "                expected->payload,",
            "                expected->payload_size_bytes)) {",
            "            return crosstl_native_loader_execution_failure(",
            "                CROSSTL_NATIVE_LOADER_PHASE_VALIDATE_REQUEST,",
            "                CROSSTL_NATIVE_LOADER_CODE_SPECIALIZATION_VALUE_MISMATCH,",
            "                CROSSTL_NATIVE_LOADER_NO_INDEX, index, 0);",
            "        }",
            "    }",
            "    CrossTLNativeLoaderExecutionRequest exact_request = *request;",
            "    CrossTLNativeLoaderAdapter exact_adapter = *adapter;",
            "    exact_request.abi_version = CROSSTL_NATIVE_LOADER_ABI_VERSION;",
            "    exact_request.target = variant->target;",
            "    exact_request.specialization_count = variant->specialization_count;",
            "    exact_request.specializations = variant->specializations;",
            "    exact_request.dispatch.workgroup_size[0] =",
            "        variant->workgroup_size[0];",
            "    exact_request.dispatch.workgroup_size[1] =",
            "        variant->workgroup_size[1];",
            "    exact_request.dispatch.workgroup_size[2] =",
            "        variant->workgroup_size[2];",
            "    exact_adapter.abi_version = CROSSTL_NATIVE_LOADER_ABI_VERSION;",
            "    exact_adapter.target = variant->target;",
            "    return variant->execute(&exact_request, &exact_adapter);",
            "}",
            "",
            "#endif /* CROSSTL_NATIVE_RUNTIME_VARIANT_REGISTRY_HPP */",
            "",
        ]
    )
    return "\n".join(lines)


def _validated_registry(
    registry: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(registry, Mapping):
        raise NativeRuntimeVariantRegistryError(
            "registry-invalid",
            "Runtime variant registry must be an object.",
        )
    variants = registry.get("variants")
    if (
        registry.get("kind") != RUNTIME_VARIANT_REGISTRY_KIND
        or not isinstance(variants, Mapping)
        or not variants
    ):
        raise NativeRuntimeVariantRegistryError(
            "registry-invalid",
            "A non-empty runtime variant registry is required.",
        )
    first_key = sorted(variants, key=lambda value: str(value))[0]
    lookup = lookup_runtime_variant(registry, first_key)
    if not lookup.get("success"):
        raise NativeRuntimeVariantRegistryError(
            "registry-invalid",
            "Runtime variant registry validation failed.",
            details={"lookup": lookup},
        )
    normalized: dict[str, Mapping[str, Any]] = {}
    for key, record in variants.items():
        path = f"$.variants[{key!r}]"
        if not isinstance(key, str) or not isinstance(record, Mapping):
            raise NativeRuntimeVariantRegistryError(
                "variant-invalid",
                "Runtime variant entries require string keys and object records.",
                path=path,
            )
        if record.get("status") != "ready":
            raise NativeRuntimeVariantRegistryError(
                "variant-blocked",
                "Native execution registries require every variant to be ready.",
                path=path,
                details={"status": record.get("status")},
            )
        normalized[key] = record
    return normalized


def _validated_units(
    units: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes, bytearray)):
        raise NativeRuntimeVariantRegistryError(
            "units-invalid",
            "Native runtime variant units must be a sequence.",
            path="$.units",
        )
    normalized: dict[str, dict[str, Any]] = {}
    for index, unit in enumerate(units):
        path = f"$.units[{index}]"
        if not isinstance(unit, Mapping) or set(unit) != _UNIT_FIELDS:
            raise NativeRuntimeVariantRegistryError(
                "unit-invalid",
                "Native runtime variant units require descriptor and executionHeader.",
                path=path,
            )
        try:
            descriptor = _validate_descriptor(unit["descriptor"])
        except NativeLoaderABIError as exc:
            raise NativeRuntimeVariantRegistryError(
                "descriptor-invalid",
                "Native loader ABI descriptor validation failed.",
                path=f"{path}.descriptor",
                details={"abiDiagnostic": exc.to_json()},
            ) from exc
        execution_header = _safe_header_path(
            unit["executionHeader"], path=f"{path}.executionHeader"
        )
        unit_id = descriptor["unitId"]
        _native_string(unit_id, path=f"{path}.descriptor.unitId")
        if unit_id in normalized:
            raise NativeRuntimeVariantRegistryError(
                "unit-duplicate",
                "Native runtime variant unit IDs must be unique.",
                path=f"{path}.descriptor.unitId",
                details={"unitId": unit_id},
            )
        symbol = _descriptor_symbol(descriptor)
        normalized[unit_id] = {
            "descriptor": descriptor,
            "executionHeader": execution_header,
            "descriptorSymbol": symbol,
            "executeSymbol": f"{symbol}_execute",
        }
    if not normalized:
        raise NativeRuntimeVariantRegistryError(
            "units-invalid",
            "At least one native runtime variant unit is required.",
            path="$.units",
        )
    return normalized


def _native_variant_entry(
    key: str,
    record: Mapping[str, Any],
    units: Mapping[str, Mapping[str, Any]],
    *,
    index: int,
) -> dict[str, Any]:
    path = f"$.variants[{key!r}]"
    artifact = _mapping(record.get("artifact"), path=f"{path}.artifact")
    unit_id = artifact.get("id")
    if not isinstance(unit_id, str) or unit_id not in units:
        raise NativeRuntimeVariantRegistryError(
            "unit-not-found",
            "Runtime variant artifact does not match a native loader unit.",
            path=f"{path}.artifact.id",
            details={"unitId": unit_id, "availableUnits": sorted(units)},
        )
    unit = units[unit_id]
    descriptor = unit["descriptor"]
    target = _mapping(record.get("target"), path=f"{path}.target")
    expected = {
        "target": descriptor["target"],
        "stage": descriptor["stage"],
        "entryPoint": descriptor["entryPoint"]["name"],
        "path": descriptor["artifact"]["packagePath"],
        "format": descriptor["artifact"]["format"],
        "hash": descriptor["artifact"]["hash"],
        "sizeBytes": descriptor["artifact"]["sizeBytes"],
    }
    actual = {
        "target": target.get("backend"),
        "stage": target.get("stage"),
        "entryPoint": target.get("entryPoint"),
        "path": artifact.get("path"),
        "format": artifact.get("format"),
        "hash": artifact.get("hash"),
        "sizeBytes": artifact.get("sizeBytes"),
    }
    if actual != expected:
        raise NativeRuntimeVariantRegistryError(
            "unit-mismatch",
            "Runtime variant metadata does not match its native loader descriptor.",
            path=path,
            details={"expected": expected, "actual": actual},
        )
    if descriptor["target"] not in _SUPPORTED_TARGETS:
        raise NativeRuntimeVariantRegistryError(
            "target-unsupported",
            "Native runtime variant execution currently supports DirectX and OpenGL.",
            path=f"{path}.target.backend",
            details={"target": descriptor["target"]},
        )
    _validate_native_execution_contract(descriptor, path=path)
    execution = _mapping(record.get("execution"), path=f"{path}.execution")
    workgroup_size = _dimensions(
        execution.get("workgroupSize"), path=f"{path}.execution.workgroupSize"
    )
    try:
        descriptor_workgroup_size = _execution_workgroup_size(
            descriptor["entryPoint"]["executionConfig"]
        )
    except NativeLoaderABIError as exc:
        raise NativeRuntimeVariantRegistryError(
            "execution-invalid",
            "Native loader workgroup metadata is invalid.",
            path=f"{path}.execution",
            details={"abiDiagnostic": exc.to_json()},
        ) from exc
    if descriptor_workgroup_size != workgroup_size:
        raise NativeRuntimeVariantRegistryError(
            "workgroup-size-mismatch",
            "Runtime variant workgroup size does not match the native loader unit.",
            path=f"{path}.execution.workgroupSize",
            details={
                "registry": list(workgroup_size),
                "descriptor": (
                    list(descriptor_workgroup_size)
                    if descriptor_workgroup_size is not None
                    else None
                ),
            },
        )
    subgroup_width = execution.get("subgroupWidth")
    if subgroup_width is not None:
        subgroup_width = _uint32(
            subgroup_width,
            path=f"{path}.execution.subgroupWidth",
            positive=True,
        )
    specializations = _variant_specializations(
        record.get("specializationConstants"),
        descriptor["specializationConstants"],
        path=f"{path}.specializationConstants",
        target=descriptor["target"],
        artifact_format=descriptor["artifact"]["format"],
    )
    symbol_key = _C_IDENTIFIER_RE.sub("_", f"variant_{index}").strip("_").lower()
    return {
        "key": key,
        "unitId": unit_id,
        "target": descriptor["target"],
        "entryPoint": descriptor["entryPoint"]["name"],
        "workgroupSize": workgroup_size,
        "subgroupWidth": subgroup_width,
        "specializations": specializations,
        "executionHeader": unit["executionHeader"],
        "descriptorSymbol": unit["descriptorSymbol"],
        "executeSymbol": unit["executeSymbol"],
        "symbol": f"crosstl_native_runtime_{symbol_key}",
    }


def _variant_specializations(
    values: Any,
    descriptor_values: Sequence[Mapping[str, Any]],
    *,
    path: str,
    target: str,
    artifact_format: str,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise NativeRuntimeVariantRegistryError(
            "specializations-invalid",
            "Runtime variant specializationConstants must be a list.",
            path=path,
        )
    if descriptor_values and (
        (target == "directx" and artifact_format != "HLSL source")
        or (target == "opengl" and artifact_format != "SPIR-V binary")
    ):
        raise NativeRuntimeVariantRegistryError(
            "specialization-mechanism-unsupported",
            "The native target adapter cannot apply specializations to this artifact format.",
            path=path,
            details={"target": target, "artifactFormat": artifact_format},
        )
    descriptor_by_identity: dict[tuple[str, Any], Mapping[str, Any]] = {}
    descriptor_names: set[str] = set()
    for index, constant in enumerate(descriptor_values):
        item_path = f"{path}.descriptor[{index}]"
        identity = _specialization_identity(constant, path=item_path)
        if identity in descriptor_by_identity:
            raise NativeRuntimeVariantRegistryError(
                "descriptor-specialization-duplicate",
                "Native loader specialization identities must be unique.",
                path=item_path,
                details={"identity": list(identity)},
            )
        constant_id = constant.get("id", constant.get("constantId"))
        name = constant.get("name")
        if name is not None:
            _native_string(name, path=f"{item_path}.name")
            if not name:
                raise NativeRuntimeVariantRegistryError(
                    "specialization-name-invalid",
                    "Native loader specialization names must not be empty.",
                    path=f"{item_path}.name",
                )
            if name in descriptor_names:
                raise NativeRuntimeVariantRegistryError(
                    "descriptor-specialization-name-duplicate",
                    "Native loader specialization names must be unique.",
                    path=f"{item_path}.name",
                    details={"name": name},
                )
            descriptor_names.add(name)
        if target == "directx":
            if constant_id is None:
                raise NativeRuntimeVariantRegistryError(
                    "specialization-id-required",
                    "DirectX HLSL specialization requires a numeric constant ID.",
                    path=f"{item_path}.id",
                )
            if not isinstance(name, str) or not _PORTABLE_IDENTIFIER_RE.fullmatch(name):
                raise NativeRuntimeVariantRegistryError(
                    "specialization-name-invalid",
                    "DirectX HLSL specialization requires a portable identifier name.",
                    path=f"{item_path}.name",
                    details={"name": name},
                )
        elif constant_id is None:
            raise NativeRuntimeVariantRegistryError(
                "specialization-id-required",
                "OpenGL SPIR-V specialization requires a numeric constant ID.",
                path=f"{item_path}.id",
            )
        descriptor_dtype = _dtype(constant.get("dtype"), path=f"{item_path}.dtype")
        if constant.get("dtype") != descriptor_dtype:
            raise NativeRuntimeVariantRegistryError(
                "descriptor-specialization-type-noncanonical",
                "Native loader specialization types must use canonical ABI names.",
                path=f"{item_path}.dtype",
                details={
                    "dtype": constant.get("dtype"),
                    "canonicalDtype": descriptor_dtype,
                },
            )
        descriptor_by_identity[identity] = constant
    result_by_identity: dict[tuple[str, Any], dict[str, Any]] = {}
    seen: set[tuple[str, Any]] = set()
    for index, constant_value in enumerate(values):
        item_path = f"{path}[{index}]"
        constant = _mapping(constant_value, path=item_path)
        identity = _specialization_identity(constant, path=item_path)
        if identity in seen:
            raise NativeRuntimeVariantRegistryError(
                "specialization-duplicate",
                "Runtime variant specialization identities must be unique.",
                path=item_path,
            )
        seen.add(identity)
        descriptor = descriptor_by_identity.get(identity)
        if descriptor is None:
            raise NativeRuntimeVariantRegistryError(
                "specialization-mismatch",
                "Runtime variant specialization does not exist in the loader unit.",
                path=item_path,
                details={"identity": list(identity)},
            )
        dtype = _dtype(constant.get("dtype"), path=f"{item_path}.dtype")
        descriptor_dtype = _dtype(
            descriptor.get("dtype"), path=f"{item_path}.descriptor.dtype"
        )
        if dtype != descriptor_dtype:
            raise NativeRuntimeVariantRegistryError(
                "specialization-type-mismatch",
                "Runtime variant specialization type does not match the loader unit.",
                path=f"{item_path}.dtype",
                details={"registry": dtype, "descriptor": descriptor_dtype},
            )
        if constant.get("id", constant.get("constantId")) != descriptor.get(
            "id", descriptor.get("constantId")
        ) or constant.get("name") != descriptor.get("name"):
            raise NativeRuntimeVariantRegistryError(
                "specialization-identity-mismatch",
                "Runtime variant specialization identity does not match the loader unit.",
                path=item_path,
                details={
                    "registry": {
                        "id": constant.get("id", constant.get("constantId")),
                        "name": constant.get("name"),
                    },
                    "descriptor": {
                        "id": descriptor.get("id", descriptor.get("constantId")),
                        "name": descriptor.get("name"),
                    },
                },
            )
        encoded = _encoded_specialization(
            constant.get("value"), dtype=dtype, path=f"{item_path}.value"
        )
        descriptor_value = _concrete_specialization_value(descriptor)
        if descriptor_value is not None:
            descriptor_encoded = _encoded_specialization(
                descriptor_value,
                dtype=dtype,
                path=f"{item_path}.descriptor.value",
            )
            if descriptor_encoded != encoded:
                raise NativeRuntimeVariantRegistryError(
                    "specialization-value-mismatch",
                    "Runtime variant specialization value does not match the loader unit.",
                    path=f"{item_path}.value",
                    details={
                        "registry": constant.get("value"),
                        "descriptor": descriptor_value,
                    },
                )
        result_by_identity[identity] = {
            "hasId": constant.get("id", constant.get("constantId")) is not None,
            "id": constant.get("id", constant.get("constantId")) or 0,
            "name": constant.get("name"),
            "typeName": descriptor["dtype"],
            **encoded,
        }
    if set(descriptor_by_identity) != seen:
        raise NativeRuntimeVariantRegistryError(
            "specialization-count-mismatch",
            "Runtime variant specialization set does not match the loader unit.",
            path=path,
            details={
                "registry": [list(identity) for identity in sorted(seen)],
                "descriptor": [
                    list(identity) for identity in sorted(descriptor_by_identity)
                ],
            },
        )
    return [
        result_by_identity[
            _specialization_identity(descriptor, path=f"{path}.descriptor[{index}]")
        ]
        for index, descriptor in enumerate(descriptor_values)
    ]


def _specialization_declarations(entry: Mapping[str, Any]) -> list[str]:
    specializations = entry["specializations"]
    if not specializations:
        return []
    symbol = entry["symbol"]
    lines = []
    for index, constant in enumerate(specializations):
        lines.append(
            "static const uint32_t "
            f"{symbol}_specialization_value_{index} = "
            f"0x{constant['bits']:08x}u;"
        )
    lines.extend(
        [
            "static const CrossTLNativeLoaderSpecializationRequest",
            f"{symbol}_specializations[{len(specializations)}] = {{",
        ]
    )
    for index, constant in enumerate(specializations):
        lines.append(
            "    {"
            f"{1 if constant['hasId'] else 0}u, "
            f"{constant['id']}u, "
            f"{_c_nullable_string(constant['name'])}, "
            f"{_c_string(constant['typeName'])}, "
            f"&{symbol}_specialization_value_{index}, "
            "sizeof(uint32_t)"
            "},"
        )
    lines.extend(("};", ""))
    return lines


def _encoded_specialization(value: Any, *, dtype: str, path: str) -> dict[str, Any]:
    if dtype == "bool":
        if type(value) is not bool:
            raise NativeRuntimeVariantRegistryError(
                "specialization-value-invalid",
                "Boolean specializations require true or false.",
                path=path,
            )
        return {"bits": 1 if value else 0}
    if dtype == "uint32":
        return {"bits": _uint32(value, path=path)}
    if dtype == "int32":
        if type(value) is not int or not (_INT32_MIN <= value <= _INT32_MAX):
            raise NativeRuntimeVariantRegistryError(
                "specialization-value-invalid",
                "int32 specializations require a signed 32-bit integer.",
                path=path,
            )
        return {"bits": value & _UINT32_MAX}
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise NativeRuntimeVariantRegistryError(
            "specialization-value-invalid",
            "float32 specializations require a finite numeric value.",
            path=path,
        )
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise NativeRuntimeVariantRegistryError(
            "specialization-value-invalid",
            "float32 specialization is outside the representable range.",
            path=path,
        ) from exc
    if not math.isfinite(numeric):
        raise NativeRuntimeVariantRegistryError(
            "specialization-value-invalid",
            "float32 specializations require a finite numeric value.",
            path=path,
        )
    try:
        packed = struct.pack("<f", numeric)
    except (OverflowError, struct.error) as exc:
        raise NativeRuntimeVariantRegistryError(
            "specialization-value-invalid",
            "float32 specialization is outside the representable range.",
            path=path,
        ) from exc
    return {"bits": struct.unpack("<I", packed)[0]}


def _concrete_specialization_value(constant: Mapping[str, Any]) -> Any:
    for field_name in ("concreteValue", "value", "defaultValue", "default"):
        value = constant.get(field_name)
        if value is not None:
            return value
    return None


def _specialization_identity(
    constant: Mapping[str, Any],
    *,
    path: str,
) -> tuple[str, Any]:
    constant_id = constant.get("id", constant.get("constantId"))
    name = constant.get("name")
    if constant_id is not None:
        return ("id", _uint32(constant_id, path=f"{path}.id"))
    if isinstance(name, str) and name and name == name.strip():
        return ("name", name)
    raise NativeRuntimeVariantRegistryError(
        "specialization-identity-invalid",
        "Specializations require a numeric ID or non-empty name.",
        path=path,
    )


def _safe_header_path(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise NativeRuntimeVariantRegistryError(
            "execution-header-invalid",
            "Execution header must be a non-empty package-relative path.",
            path=path,
        )
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or "\\" in value
        or '"' in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or any(part in ("", ".", "..") for part in candidate.parts)
        or candidate.suffix not in (".h", ".hpp")
        or candidate.as_posix() != value
    ):
        raise NativeRuntimeVariantRegistryError(
            "execution-header-invalid",
            "Execution header must be a normalized package-relative header path.",
            path=path,
            details={"value": value},
        )
    return value


def _validate_native_execution_contract(
    descriptor: Mapping[str, Any],
    *,
    path: str,
) -> None:
    target = descriptor["target"]
    stage = descriptor["stage"]
    artifact_format = descriptor["artifact"]["format"]
    entry_point = descriptor["entryPoint"]["name"]
    if stage != "compute":
        raise NativeRuntimeVariantRegistryError(
            "stage-unsupported",
            "Native runtime variant execution supports compute-stage units only.",
            path=f"{path}.target.stage",
            details={"stage": stage},
        )
    if artifact_format not in _SUPPORTED_ARTIFACT_FORMATS[target]:
        raise NativeRuntimeVariantRegistryError(
            "artifact-format-unsupported",
            "Native runtime variant artifact format is unsupported by its target adapter.",
            path=f"{path}.artifact.format",
            details={
                "target": target,
                "artifactFormat": artifact_format,
                "supportedFormats": sorted(_SUPPORTED_ARTIFACT_FORMATS[target]),
            },
        )
    _native_string(
        descriptor["artifact"]["packagePath"],
        path=f"{path}.artifact.path",
    )
    _native_string(entry_point, path=f"{path}.target.entryPoint")
    if target == "directx" and not _PORTABLE_IDENTIFIER_RE.fullmatch(entry_point):
        raise NativeRuntimeVariantRegistryError(
            "entry-point-invalid",
            "DirectX native execution requires a portable shader identifier.",
            path=f"{path}.target.entryPoint",
            details={"entryPoint": entry_point},
        )
    if (
        target == "opengl"
        and artifact_format == "GLSL source"
        and entry_point != "main"
    ):
        raise NativeRuntimeVariantRegistryError(
            "entry-point-unsupported",
            "OpenGL GLSL source execution requires the main entry point.",
            path=f"{path}.target.entryPoint",
            details={"entryPoint": entry_point},
        )


def _native_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise NativeRuntimeVariantRegistryError(
            "native-string-invalid",
            "Native registry strings must not contain embedded null bytes.",
            path=path,
        )
    return value


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeRuntimeVariantRegistryError(
            "object-invalid",
            "Expected an object.",
            path=path,
        )
    return value


def _dimensions(value: Any, *, path: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 3
    ):
        raise NativeRuntimeVariantRegistryError(
            "workgroup-size-invalid",
            "Workgroup size must contain exactly three positive uint32 values.",
            path=path,
        )
    return tuple(
        _uint32(component, path=f"{path}[{index}]", positive=True)
        for index, component in enumerate(value)
    )


def _uint32(value: Any, *, path: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or not (minimum <= value <= _UINT32_MAX):
        qualifier = "positive " if positive else ""
        raise NativeRuntimeVariantRegistryError(
            "uint32-invalid",
            f"Value must be a {qualifier}uint32 integer.",
            path=path,
        )
    return value


def _dtype(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in _DTYPE_ALIASES:
        raise NativeRuntimeVariantRegistryError(
            "specialization-type-unsupported",
            "Native variant specializations support bool, int32, uint32, and float32.",
            path=path,
            details={"dtype": value},
        )
    return _DTYPE_ALIASES[value.strip().lower()]


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


def _c_nullable_string(value: Any) -> str:
    return "NULL" if value is None else _c_string(value)
