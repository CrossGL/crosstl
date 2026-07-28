"""Deterministic deferred native compilation request contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from .pipeline import decode_runtime_variant_key

NATIVE_DEFERRED_COMPILATION_KIND = "crosstl-native-deferred-compilation-request"
NATIVE_DEFERRED_COMPILATION_REQUEST_KIND = NATIVE_DEFERRED_COMPILATION_KIND
NATIVE_DEFERRED_COMPILATION_VERSION = 1

_ERROR_PREFIX = "project.native-deferred-compilation"
_UNRESOLVED_VALUE = "@unresolved"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DIRECTX_PROFILE_RE = re.compile(r"^cs_6_[0-9]$")
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_WINDOWS_INVALID_PATH_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_PATH_STEMS = frozenset(
    (
        "AUX",
        "CON",
        "NUL",
        "PRN",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    )
)
_REQUEST_FIELDS = frozenset(
    (
        "schemaVersion",
        "kind",
        "source",
        "includes",
        "target",
        "variant",
        "expectedLoaderDescriptor",
        "requestHash",
    )
)
_SOURCE_FIELDS = frozenset(("path", "format", "hash", "sizeBytes"))
_HASH_FIELDS = frozenset(("algorithm", "value"))
_TARGET_FIELDS = frozenset(
    ("backend", "profile", "stage", "entryPoint", "outputFormat")
)
_VARIANT_FIELDS = frozenset(
    (
        "key",
        "typeArguments",
        "valueArguments",
        "compileDefines",
        "specializationValues",
        "execution",
    )
)
_SPECIALIZATION_FIELDS = frozenset(("id", "name", "value"))
_EXECUTION_FIELDS = frozenset(("workgroupSize", "subgroupWidth"))
_DESCRIPTOR_FIELDS = frozenset(("path", "hash", "sizeBytes"))
_TARGET_FORMATS = {
    "directx": {
        "source": "HLSL source",
        "output": "DXIL binary",
    },
    "opengl": {
        "source": "GLSL source",
        "output": "SPIR-V binary",
    },
}


class NativeDeferredCompilationError(ValueError):
    """A bounded variant cannot form a safe deferred compilation request."""

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
        self.details = _stable_details(details or {})
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


def build_native_deferred_compilation_request(
    source: Mapping[str, Any],
    includes: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    variant: Mapping[str, Any],
    expected_loader_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic request for a fully resolved native variant."""

    candidate = {
        "schemaVersion": NATIVE_DEFERRED_COMPILATION_VERSION,
        "kind": NATIVE_DEFERRED_COMPILATION_KIND,
        "source": source,
        "includes": includes,
        "target": target,
        "variant": variant,
        "expectedLoaderDescriptor": expected_loader_descriptor,
        "requestHash": {"algorithm": "sha256", "value": "0" * 64},
    }
    return _normalize_request(candidate, verify_request_hash=False)


def validate_native_deferred_compilation_request(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize a deferred native compilation request."""

    return _normalize_request(payload, verify_request_hash=True)


def _normalize_request(
    payload: Mapping[str, Any],
    *,
    verify_request_hash: bool,
) -> dict[str, Any]:
    request = _mapping_with_fields(
        payload,
        fields=_REQUEST_FIELDS,
        path="$",
        code="request-schema-invalid",
        label="Deferred native compilation request",
    )
    if (
        type(request.get("schemaVersion")) is not int
        or request.get("schemaVersion") != NATIVE_DEFERRED_COMPILATION_VERSION
        or request.get("kind") != NATIVE_DEFERRED_COMPILATION_KIND
    ):
        raise NativeDeferredCompilationError(
            "request-version-invalid",
            "Deferred native compilation request version or kind is unsupported.",
            details={
                "kind": request.get("kind"),
                "schemaVersion": request.get("schemaVersion"),
            },
        )

    source = _source_record(request.get("source"), path="$.source")
    includes = _include_records(request.get("includes"), source=source)
    target = _target_record(request.get("target"))
    variant = _variant_record(request.get("variant"))
    descriptor = _descriptor_record(request.get("expectedLoaderDescriptor"))
    request_hash = _sha256_hash(
        request.get("requestHash"),
        path="$.requestHash",
        code="request-hash-invalid",
        label="Deferred native compilation request hash",
    )

    _validate_target_source_binding(target, source)
    _validate_include_source_binding(includes, source)
    _validate_variant_key_binding(variant, target)
    _validate_global_path_identity(source, includes, descriptor)

    normalized = {
        "schemaVersion": NATIVE_DEFERRED_COMPILATION_VERSION,
        "kind": NATIVE_DEFERRED_COMPILATION_KIND,
        "source": source,
        "includes": includes,
        "target": target,
        "variant": variant,
        "expectedLoaderDescriptor": descriptor,
        "requestHash": request_hash,
    }
    expected_hash = _request_hash(normalized)
    if verify_request_hash and request_hash != expected_hash:
        raise NativeDeferredCompilationError(
            "request-hash-mismatch",
            "Deferred native compilation request hash does not match its contents.",
            path="$.requestHash",
            details={
                "expectedHash": expected_hash,
                "actualHash": request_hash,
            },
        )
    normalized["requestHash"] = expected_hash
    return normalized


def _source_record(value: Any, *, path: str) -> dict[str, Any]:
    record = _mapping_with_fields(
        value,
        fields=_SOURCE_FIELDS,
        path=path,
        code="source-record-invalid",
        label="Deferred compilation source record",
    )
    return {
        "path": _portable_path(record.get("path"), path=f"{path}.path"),
        "format": _required_string(
            record.get("format"),
            path=f"{path}.format",
            code="source-format-invalid",
            label="Source format",
        ),
        "hash": _sha256_hash(
            record.get("hash"),
            path=f"{path}.hash",
            code="source-hash-invalid",
            label="Source hash",
        ),
        "sizeBytes": _uint64(
            record.get("sizeBytes"),
            path=f"{path}.sizeBytes",
            code="source-size-invalid",
            label="Source sizeBytes",
        ),
    }


def _include_records(value: Any, *, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise NativeDeferredCompilationError(
            "includes-invalid",
            "Deferred compilation includes must be a sequence of source records.",
            path="$.includes",
        )
    records = [
        _source_record(item, path=f"$.includes[{index}]")
        for index, item in enumerate(value)
    ]
    records.sort(key=lambda item: (item["path"].casefold(), item["path"]))

    seen: dict[str, str] = {source["path"].casefold(): source["path"]}
    for index, record in enumerate(records):
        path = record["path"]
        portable_key = path.casefold()
        previous = seen.get(portable_key)
        if previous is not None:
            code = (
                "include-path-duplicate"
                if previous == path
                else "include-path-case-collision"
            )
            raise NativeDeferredCompilationError(
                code,
                "Deferred compilation source and include paths must be unique under portable case folding.",
                path=f"$.includes[{index}].path",
                details={"path": path, "conflictingPath": previous},
            )
        seen[portable_key] = path
    return records


def _target_record(value: Any) -> dict[str, Any]:
    target = _mapping_with_fields(
        value,
        fields=_TARGET_FIELDS,
        path="$.target",
        code="target-invalid",
        label="Deferred compilation target",
    )
    backend = _required_string(
        target.get("backend"),
        path="$.target.backend",
        code="target-backend-invalid",
        label="Target backend",
    )
    if backend not in _TARGET_FORMATS:
        raise NativeDeferredCompilationError(
            "target-backend-unsupported",
            "Deferred native compilation supports only DirectX and OpenGL.",
            path="$.target.backend",
            details={"backend": backend, "supportedBackends": sorted(_TARGET_FORMATS)},
        )
    profile = _optional_string(
        target.get("profile"),
        path="$.target.profile",
        code="target-profile-invalid",
        label="Target profile",
    )
    if backend == "directx" and (
        profile is None or not _DIRECTX_PROFILE_RE.fullmatch(profile)
    ):
        raise NativeDeferredCompilationError(
            "target-profile-unsupported",
            "DirectX deferred compilation requires a cs_6_x compute profile.",
            path="$.target.profile",
            details={"backend": backend, "profile": profile},
        )
    if backend == "opengl" and profile is not None:
        raise NativeDeferredCompilationError(
            "target-profile-unsupported",
            "OpenGL deferred compilation derives its profile from the packaged GLSL source.",
            path="$.target.profile",
            details={"backend": backend, "profile": profile},
        )
    stage = _required_string(
        target.get("stage"),
        path="$.target.stage",
        code="target-stage-invalid",
        label="Target stage",
    )
    if stage != "compute":
        raise NativeDeferredCompilationError(
            "target-stage-unsupported",
            "Deferred native compilation currently supports only compute stages.",
            path="$.target.stage",
            details={"stage": stage},
        )
    entry_point = _identifier(
        target.get("entryPoint"),
        path="$.target.entryPoint",
        code="target-entry-point-invalid",
        label="Target entry point",
    )
    if backend == "opengl" and entry_point != "main":
        raise NativeDeferredCompilationError(
            "target-entry-point-unsupported",
            "OpenGL GLSL deferred compilation requires the main entry point.",
            path="$.target.entryPoint",
            details={"entryPoint": entry_point},
        )
    output_format = _required_string(
        target.get("outputFormat"),
        path="$.target.outputFormat",
        code="target-output-format-invalid",
        label="Target output format",
    )
    expected_output = _TARGET_FORMATS[backend]["output"]
    if output_format != expected_output:
        raise NativeDeferredCompilationError(
            "target-output-format-unsupported",
            "Deferred compilation output format is incompatible with the target backend.",
            path="$.target.outputFormat",
            details={
                "backend": backend,
                "outputFormat": output_format,
                "expectedOutputFormat": expected_output,
            },
        )
    return {
        "backend": backend,
        "profile": profile,
        "stage": stage,
        "entryPoint": entry_point,
        "outputFormat": output_format,
    }


def _variant_record(value: Any) -> dict[str, Any]:
    variant = _mapping_with_fields(
        value,
        fields=_VARIANT_FIELDS,
        path="$.variant",
        code="variant-invalid",
        label="Deferred compilation variant",
    )
    key = _required_string(
        variant.get("key"),
        path="$.variant.key",
        code="variant-key-invalid",
        label="Variant key",
    )
    type_arguments = _string_mapping(
        variant.get("typeArguments"),
        path="$.variant.typeArguments",
        code="variant-type-arguments-invalid",
        label="Variant type arguments",
        allow_empty_values=False,
    )
    value_arguments = _scalar_mapping(
        variant.get("valueArguments"),
        path="$.variant.valueArguments",
        code="variant-value-arguments-invalid",
        label="Variant value arguments",
    )
    compile_defines = _string_mapping(
        variant.get("compileDefines"),
        path="$.variant.compileDefines",
        code="variant-compile-defines-invalid",
        label="Variant compile defines",
        allow_empty_values=True,
    )
    specializations = _specialization_values(variant.get("specializationValues"))
    execution = _execution_identity(variant.get("execution"))
    return {
        "key": key,
        "typeArguments": type_arguments,
        "valueArguments": value_arguments,
        "compileDefines": compile_defines,
        "specializationValues": specializations,
        "execution": execution,
    }


def _specialization_values(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise NativeDeferredCompilationError(
            "variant-specialization-values-invalid",
            "Variant specialization values must be a sequence.",
            path="$.variant.specializationValues",
        )
    normalized: list[dict[str, Any]] = []
    seen_ids: dict[int, int] = {}
    seen_names: dict[str, int] = {}
    for index, item in enumerate(value):
        path = f"$.variant.specializationValues[{index}]"
        record = _mapping_with_fields(
            item,
            fields=_SPECIALIZATION_FIELDS,
            path=path,
            code="variant-specialization-value-invalid",
            label="Variant specialization value",
        )
        constant_id = record.get("id")
        if constant_id is not None:
            constant_id = _uint32(
                constant_id,
                path=f"{path}.id",
                code="variant-specialization-id-invalid",
                label="Specialization id",
                allow_zero=True,
            )
        name = record.get("name")
        if name is not None:
            name = _identifier(
                name,
                path=f"{path}.name",
                code="variant-specialization-name-invalid",
                label="Specialization name",
            )
        if constant_id is None and name is None:
            raise NativeDeferredCompilationError(
                "variant-specialization-identity-missing",
                "A specialization value must have an id or name.",
                path=path,
            )
        if constant_id is not None and constant_id in seen_ids:
            raise NativeDeferredCompilationError(
                "variant-specialization-identity-duplicate",
                "Specialization ids must be unique.",
                path=f"{path}.id",
                details={
                    "id": constant_id,
                    "firstIndex": seen_ids[constant_id],
                    "duplicateIndex": index,
                },
            )
        if name is not None and name in seen_names:
            raise NativeDeferredCompilationError(
                "variant-specialization-identity-duplicate",
                "Specialization names must be unique.",
                path=f"{path}.name",
                details={
                    "name": name,
                    "firstIndex": seen_names[name],
                    "duplicateIndex": index,
                },
            )
        if constant_id is not None:
            seen_ids[constant_id] = index
        if name is not None:
            seen_names[name] = index
        normalized.append(
            {
                "id": constant_id,
                "name": name,
                "value": _concrete_scalar(
                    record.get("value"),
                    path=f"{path}.value",
                    code="variant-specialization-value-invalid",
                    label="Specialization value",
                ),
            }
        )
    normalized.sort(
        key=lambda item: (
            item["id"] is None,
            item["id"] if item["id"] is not None else 0,
            item["name"] or "",
            _canonical_json(item["value"]),
        )
    )
    return normalized


def _execution_identity(value: Any) -> dict[str, Any]:
    execution = _mapping_with_fields(
        value,
        fields=_EXECUTION_FIELDS,
        path="$.variant.execution",
        code="variant-execution-invalid",
        label="Variant execution identity",
    )
    workgroup = execution.get("workgroupSize")
    if not isinstance(workgroup, Sequence) or isinstance(
        workgroup, (str, bytes, bytearray)
    ):
        raise NativeDeferredCompilationError(
            "variant-workgroup-size-invalid",
            "Variant workgroupSize must contain exactly three uint32 components.",
            path="$.variant.execution.workgroupSize",
        )
    components = list(workgroup)
    if len(components) != 3:
        raise NativeDeferredCompilationError(
            "variant-workgroup-size-invalid",
            "Variant workgroupSize must contain exactly three uint32 components.",
            path="$.variant.execution.workgroupSize",
            details={"componentCount": len(components)},
        )
    normalized_workgroup = [
        _uint32(
            component,
            path=f"$.variant.execution.workgroupSize[{index}]",
            code="variant-workgroup-size-invalid",
            label="Workgroup size component",
            allow_zero=False,
        )
        for index, component in enumerate(components)
    ]
    subgroup = execution.get("subgroupWidth")
    if subgroup is not None:
        subgroup = _uint32(
            subgroup,
            path="$.variant.execution.subgroupWidth",
            code="variant-subgroup-width-invalid",
            label="Subgroup width",
            allow_zero=False,
        )
    return {
        "workgroupSize": normalized_workgroup,
        "subgroupWidth": subgroup,
    }


def _descriptor_record(value: Any) -> dict[str, Any]:
    descriptor = _mapping_with_fields(
        value,
        fields=_DESCRIPTOR_FIELDS,
        path="$.expectedLoaderDescriptor",
        code="loader-descriptor-invalid",
        label="Expected native-loader descriptor",
    )
    return {
        "path": _portable_path(
            descriptor.get("path"),
            path="$.expectedLoaderDescriptor.path",
        ),
        "hash": _sha256_hash(
            descriptor.get("hash"),
            path="$.expectedLoaderDescriptor.hash",
            code="loader-descriptor-hash-invalid",
            label="Expected native-loader descriptor hash",
        ),
        "sizeBytes": _uint64(
            descriptor.get("sizeBytes"),
            path="$.expectedLoaderDescriptor.sizeBytes",
            code="loader-descriptor-size-invalid",
            label="Expected native-loader descriptor sizeBytes",
        ),
    }


def _validate_target_source_binding(
    target: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    expected_format = _TARGET_FORMATS[target["backend"]]["source"]
    if source["format"] != expected_format:
        raise NativeDeferredCompilationError(
            "source-format-target-mismatch",
            "Deferred compilation source format is incompatible with the target backend.",
            path="$.source.format",
            details={
                "backend": target["backend"],
                "sourceFormat": source["format"],
                "expectedSourceFormat": expected_format,
            },
        )


def _validate_include_source_binding(
    includes: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> None:
    for index, include in enumerate(includes):
        if include["format"] == source["format"]:
            continue
        raise NativeDeferredCompilationError(
            "include-format-source-mismatch",
            "Deferred compilation includes must use the source language format.",
            path=f"$.includes[{index}].format",
            details={
                "includeFormat": include["format"],
                "sourceFormat": source["format"],
            },
        )


def _validate_variant_key_binding(
    variant: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    try:
        decoded = decode_runtime_variant_key(variant["key"])
    except ValueError as exc:
        raise NativeDeferredCompilationError(
            "variant-key-invalid",
            "Variant key must use the canonical runtime variant key schema.",
            path="$.variant.key",
            details={"error": str(exc)},
        ) from exc
    if decoded.get("sourceUnit") == _UNRESOLVED_VALUE:
        raise NativeDeferredCompilationError(
            "variant-key-unresolved",
            "Variant key sourceUnit must be fully resolved.",
            path="$.variant.key",
        )
    if decoded.get("sourceEntry") == _UNRESOLVED_VALUE:
        raise NativeDeferredCompilationError(
            "variant-key-unresolved",
            "Variant key sourceEntry must be fully resolved.",
            path="$.variant.key",
        )
    expected = {
        "target": target["backend"],
        "targetProfile": target["profile"],
        "execution": variant["execution"],
        "typeArguments": variant["typeArguments"],
        "valueArguments": variant["valueArguments"],
        "specializationConstants": variant["specializationValues"],
        "defines": variant["compileDefines"],
    }
    mismatches = [
        field for field, value in expected.items() if decoded.get(field) != value
    ]
    if mismatches:
        raise NativeDeferredCompilationError(
            "variant-key-binding-mismatch",
            "Variant key does not match the bounded compilation inputs.",
            path="$.variant.key",
            details={"mismatchedFields": sorted(mismatches)},
        )


def _validate_global_path_identity(
    source: Mapping[str, Any],
    includes: Sequence[Mapping[str, Any]],
    descriptor: Mapping[str, Any],
) -> None:
    source_paths = {source["path"].casefold(): source["path"]}
    source_paths.update(
        {include["path"].casefold(): include["path"] for include in includes}
    )
    descriptor_path = descriptor["path"]
    conflict = source_paths.get(descriptor_path.casefold())
    if conflict is not None:
        raise NativeDeferredCompilationError(
            "loader-descriptor-path-collision",
            "Expected native-loader descriptor path must not collide with a source path.",
            path="$.expectedLoaderDescriptor.path",
            details={"path": descriptor_path, "conflictingPath": conflict},
        )


def _mapping_with_fields(
    value: Any,
    *,
    fields: frozenset[str],
    path: str,
    code: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be an object.",
            path=path,
        )
    actual_fields = frozenset(value)
    missing = sorted(fields - actual_fields)
    unsupported = sorted(actual_fields - fields, key=lambda item: str(item))
    if missing or unsupported:
        raise NativeDeferredCompilationError(
            code,
            f"{label} has an invalid field set.",
            path=path,
            details={
                "missingFields": missing,
                "unsupportedFields": unsupported,
            },
        )
    return value


def _sha256_hash(
    value: Any,
    *,
    path: str,
    code: str,
    label: str,
) -> dict[str, str]:
    hash_record = _mapping_with_fields(
        value,
        fields=_HASH_FIELDS,
        path=path,
        code=code,
        label=label,
    )
    algorithm = hash_record.get("algorithm")
    digest = hash_record.get("value")
    if (
        algorithm != "sha256"
        or not isinstance(digest, str)
        or not _SHA256_RE.fullmatch(digest)
    ):
        raise NativeDeferredCompilationError(
            code,
            f"{label} must contain an exact lowercase SHA-256 digest.",
            path=path,
            details={"algorithm": algorithm, "value": digest},
        )
    return {"algorithm": "sha256", "value": digest}


def _uint64(value: Any, *, path: str, code: str, label: str) -> int:
    if type(value) is not int or value < 0 or value > _UINT64_MAX:
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be a uint64 value.",
            path=path,
            details={"value": value},
        )
    return value


def _uint32(
    value: Any,
    *,
    path: str,
    code: str,
    label: str,
    allow_zero: bool,
) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum or value > _UINT32_MAX:
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be a {'non-negative' if allow_zero else 'positive'} uint32 value.",
            path=path,
            details={"value": value},
        )
    return value


def _required_string(
    value: Any,
    *,
    path: str,
    code: str,
    label: str,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be a non-empty trimmed string.",
            path=path,
            details={"value": value},
        )
    return value


def _optional_string(
    value: Any,
    *,
    path: str,
    code: str,
    label: str,
) -> str | None:
    if value is None:
        return None
    return _required_string(
        value,
        path=path,
        code=code,
        label=label,
    )


def _identifier(
    value: Any,
    *,
    path: str,
    code: str,
    label: str,
) -> str:
    normalized = _required_string(
        value,
        path=path,
        code=code,
        label=label,
    )
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be a portable identifier.",
            path=path,
            details={"value": normalized},
        )
    return normalized


def _string_mapping(
    value: Any,
    *,
    path: str,
    code: str,
    label: str,
    allow_empty_values: bool,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be an object.",
            path=path,
        )
    normalized: dict[str, str] = {}
    for key in sorted(value, key=lambda item: str(item)):
        name = _identifier(
            key,
            path=path,
            code=code,
            label=f"{label} key",
        )
        item = value[key]
        if (
            not isinstance(item, str)
            or item != item.strip()
            or (not allow_empty_values and not item)
            or item == _UNRESOLVED_VALUE
        ):
            raise NativeDeferredCompilationError(
                code,
                f"{label} values must be concrete trimmed strings.",
                path=f"{path}.{name}",
                details={"value": item},
            )
        normalized[name] = item
    return normalized


def _scalar_mapping(
    value: Any,
    *,
    path: str,
    code: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be an object.",
            path=path,
        )
    normalized: dict[str, Any] = {}
    for key in sorted(value, key=lambda item: str(item)):
        name = _identifier(
            key,
            path=path,
            code=code,
            label=f"{label} key",
        )
        normalized[name] = _concrete_scalar(
            value[key],
            path=f"{path}.{name}",
            code=code,
            label=f"{label} value",
        )
    return normalized


def _concrete_scalar(
    value: Any,
    *,
    path: str,
    code: str,
    label: str,
) -> Any:
    if value is None:
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be fully resolved and non-null.",
            path=path,
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be finite.",
            path=path,
            details={"value": repr(value)},
        )
    if not isinstance(value, (bool, int, float, str)):
        raise NativeDeferredCompilationError(
            code,
            f"{label} must be a concrete JSON scalar.",
            path=path,
            details={"type": type(value).__name__},
        )
    if value == _UNRESOLVED_VALUE:
        raise NativeDeferredCompilationError(
            code,
            f"{label} must not use the unresolved marker.",
            path=path,
        )
    return value


def _portable_path(value: Any, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        raise NativeDeferredCompilationError(
            "path-invalid",
            "Package paths must use portable relative path syntax.",
            path=path,
            details={"path": value},
        )
    components = value.split("/")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    nonportable_components = [
        component
        for component in components
        if (
            component.endswith((" ", "."))
            or component.partition(".")[0].upper() in _WINDOWS_RESERVED_PATH_STEMS
            or any(
                ord(character) < 32 or character in _WINDOWS_INVALID_PATH_CHARACTERS
                for character in component
            )
        )
    ]
    if (
        any(component in {"", ".", ".."} for component in components)
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or nonportable_components
    ):
        raise NativeDeferredCompilationError(
            "path-invalid",
            "Package paths must be portable, normalized, relative, and traversal-free.",
            path=path,
            details={
                "path": value,
                "nonportableComponents": nonportable_components,
            },
        )
    return posix_path.as_posix()


def _request_hash(value: Mapping[str, Any]) -> dict[str, str]:
    hash_payload = {key: item for key, item in value.items() if key != "requestHash"}
    digest = hashlib.sha256(_canonical_json(hash_payload).encode("utf-8")).hexdigest()
    return {"algorithm": "sha256", "value": digest}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _stable_details(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _stable_detail_value(item)
        for key, item in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _stable_detail_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return _stable_details(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_detail_value(item) for item in value]
    return repr(value)
