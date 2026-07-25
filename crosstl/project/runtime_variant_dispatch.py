"""Build native dispatch requests from exact runtime variant selections."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .native_loader_abi import (
    NATIVE_LOADER_ABI_VERSION,
    NativeLoaderABIError,
    _validate_descriptor,
)
from .native_loader_abi_package import (
    NATIVE_LOADER_ABI_PACKAGE_KIND,
    NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
    NATIVE_LOADER_ABI_PACKAGE_VERSION,
)
from .native_loader_dispatch import (
    NativeLoaderDispatchError,
    build_native_loader_dispatch_request,
)
from .pipeline import lookup_runtime_variant
from .runtime_verification import RuntimeDispatchGeometry, RuntimeExecutionRequest

_ERROR_PREFIX = "project.runtime-variant-dispatch"
_PACKAGE_FIELDS = frozenset(
    (
        "schemaVersion",
        "kind",
        "abiVersion",
        "sourceLoaderManifest",
        "sourceLoaderManifestHash",
        "success",
        "summary",
        "units",
        "targetAdapters",
        "runtimeVariantRegistry",
        "generatedFiles",
    )
)
_PACKAGE_UNIT_FIELDS = frozenset(
    (
        "unitId",
        "target",
        "stage",
        "entryPoint",
        "artifact",
        "descriptorPath",
        "descriptorHash",
        "descriptorSizeBytes",
        "declarationsPath",
        "declarationsHash",
        "executionABIPath",
        "executionABIHash",
    )
)
_AVAILABLE_RUNTIME_VARIANT_REGISTRY_FIELDS = frozenset(
    ("available", "path", "hash", "registryHash", "variantCount", "nativeHeader")
)
_UNAVAILABLE_RUNTIME_VARIANT_REGISTRY_FIELDS = frozenset(
    (*_AVAILABLE_RUNTIME_VARIANT_REGISTRY_FIELDS, "reason")
)
_AVAILABLE_NATIVE_HEADER_FIELDS = frozenset(("available", "path", "hash"))
_UNAVAILABLE_NATIVE_HEADER_FIELDS = frozenset(("available", "reason"))
_TARGET_UNAVAILABLE_NATIVE_HEADER_FIELDS = frozenset(
    ("available", "reason", "unavailableTargets")
)
_SHA256_FIELDS = frozenset(("algorithm", "value"))
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


class _DuplicateJSONKeyError(ValueError):
    """A JSON object contains a key more than once."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


class _InvalidJSONConstantError(ValueError):
    """A JSON document contains a non-standard numeric constant."""


class RuntimeVariantDispatchError(ValueError):
    """An exact runtime variant cannot form a verified native dispatch request."""

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


def build_runtime_variant_dispatch_request(
    registry: Mapping[str, Any],
    key: str,
    native_loader_package: str | os.PathLike[str],
    input_values: Mapping[str, Any] | Sequence[Any],
    output_values: Mapping[str, Any] | Sequence[Any],
    dispatch_geometry: RuntimeDispatchGeometry | Mapping[str, Any] | Sequence[int],
) -> RuntimeExecutionRequest:
    """Select one exact variant and build its verified native dispatch request.

    Specialization values and workgroup size are authoritative properties of
    the selected registry record. Runtime callers provide resource values and
    dispatch extents, but cannot override either identity.
    """

    registry_snapshot = _snapshot_caller_value(
        registry,
        path="$.runtimeVariantRegistry",
        label="Runtime variant registry",
    )
    input_snapshot = _snapshot_caller_value(
        input_values,
        path="$.inputValues",
        label="Runtime input values",
    )
    output_snapshot = _snapshot_caller_value(
        output_values,
        path="$.outputValues",
        label="Runtime output values",
    )
    geometry_snapshot = _snapshot_caller_value(
        dispatch_geometry,
        path="$.dispatchGeometry",
        label="Dispatch geometry",
    )

    selected = _selected_record(registry_snapshot, key)
    package, package_root = _load_package(native_loader_package)
    packaged_registry = _load_packaged_registry(package, package_root)
    _verify_registry_binding(
        registry_snapshot,
        selected,
        packaged_registry,
        key=key,
        reference=package["runtimeVariantRegistry"],
    )
    package_unit = _select_package_unit(package, selected)
    descriptor = _load_descriptor(package_root, package_unit)
    _cross_check_identities(selected, package_unit, descriptor)

    specialization_values = _selected_specialization_values(selected, descriptor)
    selected_geometry = _selected_dispatch_geometry(
        geometry_snapshot,
        selected_execution=selected["execution"],
        entry_point=selected["target"]["entryPoint"],
    )
    try:
        return build_native_loader_dispatch_request(
            descriptor,
            package_root,
            input_snapshot,
            output_snapshot,
            selected_geometry,
            specialization_values,
            expected_target=selected["target"]["backend"],
        )
    except NativeLoaderDispatchError as exc:
        raise RuntimeVariantDispatchError(
            "request-invalid",
            "The selected runtime variant could not form a native dispatch request.",
            path=exc.path,
            details={
                "requestedKey": key,
                "dispatchDiagnostic": exc.to_json(),
            },
        ) from exc


def _snapshot_caller_value(value: Any, *, path: str, label: str) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise RuntimeVariantDispatchError(
            "input-snapshot-failed",
            f"{label} could not be isolated from caller-owned mutable state.",
            path=path,
            details={"error": str(exc)},
        ) from exc


def _selected_record(registry: Mapping[str, Any], key: str) -> dict[str, Any]:
    lookup = lookup_runtime_variant(registry, key)
    if not lookup.get("success") or not isinstance(lookup.get("record"), Mapping):
        diagnostics = lookup.get("diagnostics")
        raise RuntimeVariantDispatchError(
            "lookup-failed",
            "No lookup-ready runtime variant exactly matches the requested key.",
            path="$.key",
            details={
                "requestedKey": key if isinstance(key, str) else None,
                "lookupStatus": lookup.get("status"),
                "lookupDiagnostics": (
                    copy.deepcopy(diagnostics) if isinstance(diagnostics, list) else []
                ),
            },
        )
    selected = copy.deepcopy(dict(lookup["record"]))
    _validate_selected_record(selected)
    return selected


def _validate_selected_record(selected: Mapping[str, Any]) -> None:
    target = selected.get("target")
    if not isinstance(target, Mapping):
        raise RuntimeVariantDispatchError(
            "variant-target-invalid",
            "Selected runtime variant target identity must be an object.",
            path="$.selectedVariant.target",
        )
    for field in ("backend", "stage", "entryPoint"):
        value = target.get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeVariantDispatchError(
                "variant-target-invalid",
                f"Selected runtime variant target {field} must be a non-empty string.",
                path=f"$.selectedVariant.target.{field}",
            )

    artifact = selected.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeVariantDispatchError(
            "variant-artifact-invalid",
            "Selected runtime variant artifact identity must be an object.",
            path="$.selectedVariant.artifact",
        )
    for field in ("id", "path", "format"):
        value = artifact.get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeVariantDispatchError(
                "variant-artifact-invalid",
                f"Selected runtime variant artifact {field} must be a non-empty string.",
                path=f"$.selectedVariant.artifact.{field}",
            )
    _validated_sha256(
        artifact.get("hash"),
        path="$.selectedVariant.artifact.hash",
    )
    size_bytes = artifact.get("sizeBytes")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        raise RuntimeVariantDispatchError(
            "variant-artifact-invalid",
            "Selected runtime variant artifact sizeBytes must be a non-negative integer.",
            path="$.selectedVariant.artifact.sizeBytes",
        )
    if not isinstance(selected.get("execution"), Mapping):
        raise RuntimeVariantDispatchError(
            "execution-invalid",
            "Selected runtime variant execution identity must be an object.",
            path="$.selectedVariant.execution",
        )


def _load_package(
    value: str | os.PathLike[str],
) -> tuple[dict[str, Any], Path]:
    try:
        supplied = Path(os.fspath(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeVariantDispatchError(
            "package-path-invalid",
            "Native loader package must be a package directory or manifest path.",
            path="$.nativeLoaderPackage",
        ) from exc
    try:
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeVariantDispatchError(
            "package-unavailable",
            "Native loader package does not resolve to an available path.",
            path="$.nativeLoaderPackage",
            details={"package": str(supplied), "error": str(exc)},
        ) from exc

    if resolved.is_dir():
        package_root = resolved
        unresolved_manifest = package_root / NATIVE_LOADER_ABI_PACKAGE_MANIFEST
        try:
            manifest_path = unresolved_manifest.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeVariantDispatchError(
                "package-manifest-unavailable",
                "Native loader package manifest is unavailable.",
                path="$.nativeLoaderPackage",
                details={"manifestPath": str(unresolved_manifest), "error": str(exc)},
            ) from exc
        _require_contained(
            manifest_path,
            package_root,
            code="package-manifest-path-escape",
            path="$.nativeLoaderPackage",
            message="Native loader package manifest resolves outside the package root.",
        )
    elif resolved.is_file():
        manifest_path = resolved
        package_root = resolved.parent.resolve(strict=True)
    else:
        raise RuntimeVariantDispatchError(
            "package-path-invalid",
            "Native loader package path must resolve to a directory or regular file.",
            path="$.nativeLoaderPackage",
            details={"package": str(supplied)},
        )

    payload = _read_json_object(
        manifest_path,
        code_prefix="package-manifest",
        path="$.nativeLoaderPackage",
    )
    _validate_package(payload)
    return payload, package_root


def _validate_package(package: Mapping[str, Any]) -> None:
    actual_fields = frozenset(package)
    missing = sorted(_PACKAGE_FIELDS - actual_fields)
    unsupported = sorted(actual_fields - _PACKAGE_FIELDS, key=str)
    if missing or unsupported:
        raise RuntimeVariantDispatchError(
            "package-schema-invalid",
            "Native loader package manifest has an invalid field set.",
            path="$.packageManifest",
            details={"missingFields": missing, "unsupportedFields": unsupported},
        )
    if package.get("kind") != NATIVE_LOADER_ABI_PACKAGE_KIND:
        raise RuntimeVariantDispatchError(
            "package-kind-invalid",
            "Native loader package kind is unsupported.",
            path="$.packageManifest.kind",
            details={"kind": package.get("kind")},
        )
    if (
        type(package.get("schemaVersion")) is not int
        or package.get("schemaVersion") != NATIVE_LOADER_ABI_PACKAGE_VERSION
        or type(package.get("abiVersion")) is not int
        or package.get("abiVersion") != NATIVE_LOADER_ABI_VERSION
    ):
        raise RuntimeVariantDispatchError(
            "package-version-invalid",
            "Native loader package schema or ABI version is unsupported.",
            path="$.packageManifest.schemaVersion",
            details={
                "schemaVersion": package.get("schemaVersion"),
                "abiVersion": package.get("abiVersion"),
            },
        )
    if package.get("success") is not True:
        raise RuntimeVariantDispatchError(
            "package-unsuccessful",
            "Native loader package must report successful generation.",
            path="$.packageManifest.success",
            details={"success": package.get("success")},
        )
    summary = package.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeVariantDispatchError(
            "package-schema-invalid",
            "Native loader package summary must be an object.",
            path="$.packageManifest.summary",
        )
    runtime_variant_count = summary.get("runtimeVariantCount")
    if (
        not isinstance(runtime_variant_count, int)
        or isinstance(runtime_variant_count, bool)
        or runtime_variant_count < 0
    ):
        raise RuntimeVariantDispatchError(
            "package-schema-invalid",
            "Native loader package runtimeVariantCount must be a non-negative integer.",
            path="$.packageManifest.summary.runtimeVariantCount",
        )
    if not isinstance(package.get("sourceLoaderManifestHash"), Mapping):
        raise RuntimeVariantDispatchError(
            "package-schema-invalid",
            "Native loader package sourceLoaderManifestHash must be an object.",
            path="$.packageManifest.sourceLoaderManifestHash",
        )
    _validated_sha256(
        package["sourceLoaderManifestHash"],
        path="$.packageManifest.sourceLoaderManifestHash",
    )
    for field in ("targetAdapters", "generatedFiles"):
        if not isinstance(package.get(field), list):
            raise RuntimeVariantDispatchError(
                "package-schema-invalid",
                f"Native loader package {field} must be a list.",
                path=f"$.packageManifest.{field}",
            )
    units = package.get("units")
    if not isinstance(units, list) or not units:
        raise RuntimeVariantDispatchError(
            "package-units-invalid",
            "Native loader package units must be a non-empty list.",
            path="$.packageManifest.units",
        )
    for index, unit in enumerate(units):
        _validate_package_unit(unit, index=index)


def _validate_package_unit(value: Any, *, index: int) -> None:
    path = f"$.packageManifest.units[{index}]"
    if not isinstance(value, Mapping):
        raise RuntimeVariantDispatchError(
            "package-unit-invalid",
            "Native loader package units must be objects.",
            path=path,
        )
    actual_fields = frozenset(value)
    missing = sorted(_PACKAGE_UNIT_FIELDS - actual_fields)
    unsupported = sorted(actual_fields - _PACKAGE_UNIT_FIELDS, key=str)
    if missing or unsupported:
        raise RuntimeVariantDispatchError(
            "package-unit-schema-invalid",
            "Native loader package unit has an invalid field set.",
            path=path,
            details={"missingFields": missing, "unsupportedFields": unsupported},
        )
    for field in (
        "unitId",
        "target",
        "stage",
        "entryPoint",
        "descriptorPath",
        "declarationsPath",
        "executionABIPath",
    ):
        item = value.get(field)
        if field == "stage" and item is None:
            continue
        if not isinstance(item, str) or not item or item != item.strip():
            raise RuntimeVariantDispatchError(
                "package-unit-schema-invalid",
                f"Native loader package unit {field} must be a non-empty string.",
                path=f"{path}.{field}",
            )
    if not isinstance(value.get("artifact"), Mapping):
        raise RuntimeVariantDispatchError(
            "package-unit-schema-invalid",
            "Native loader package unit artifact must be an object.",
            path=f"{path}.artifact",
        )
    for field in ("descriptorHash", "declarationsHash", "executionABIHash"):
        _validated_sha256(value.get(field), path=f"{path}.{field}")
    descriptor_size = value.get("descriptorSizeBytes")
    if (
        not isinstance(descriptor_size, int)
        or isinstance(descriptor_size, bool)
        or descriptor_size < 0
        or descriptor_size > _UINT64_MAX
    ):
        raise RuntimeVariantDispatchError(
            "package-unit-schema-invalid",
            "Native loader descriptorSizeBytes must be a uint64 integer.",
            path=f"{path}.descriptorSizeBytes",
        )


def _load_packaged_registry(
    package: Mapping[str, Any],
    package_root: Path,
) -> dict[str, Any]:
    reference = package.get("runtimeVariantRegistry")
    path = "$.packageManifest.runtimeVariantRegistry"
    if not isinstance(reference, Mapping):
        raise RuntimeVariantDispatchError(
            "package-registry-invalid",
            "Native loader package runtimeVariantRegistry must be an object.",
            path=path,
        )
    available = reference.get("available")
    expected_fields = (
        _AVAILABLE_RUNTIME_VARIANT_REGISTRY_FIELDS
        if available is True
        else (
            _UNAVAILABLE_RUNTIME_VARIANT_REGISTRY_FIELDS
            if available is False
            else frozenset()
        )
    )
    if not expected_fields or frozenset(reference) != expected_fields:
        raise RuntimeVariantDispatchError(
            "package-registry-invalid",
            "Native loader package runtimeVariantRegistry has an invalid field set.",
            path=path,
            details={
                "available": available,
                "actualFields": sorted(reference, key=str),
                "expectedFields": sorted(expected_fields),
            },
        )
    if available is False:
        reason = reference.get("reason")
        if not isinstance(reason, str) or not reason or reason != reason.strip():
            raise RuntimeVariantDispatchError(
                "package-registry-invalid",
                "Unavailable runtime variant registry reason must be a non-empty string.",
                path=f"{path}.reason",
            )
        raise RuntimeVariantDispatchError(
            "package-registry-unavailable",
            "Native loader package does not contain a ready runtime variant registry.",
            path=path,
            details={"reason": reason},
        )

    variant_count = reference.get("variantCount")
    if (
        not isinstance(variant_count, int)
        or isinstance(variant_count, bool)
        or variant_count < 0
    ):
        raise RuntimeVariantDispatchError(
            "package-registry-invalid",
            "Native loader package runtime variant count must be a non-negative integer.",
            path=f"{path}.variantCount",
        )
    registry_hash = _validated_sha256(
        reference.get("registryHash"),
        path=f"{path}.registryHash",
    )
    registry_content = _read_verified_package_file(
        package_root,
        reference.get("path"),
        reference.get("hash"),
        field_path=f"{path}.path",
        label="Runtime variant registry",
        code_prefix="package-registry",
    )
    packaged_registry = _decode_json_object(
        registry_content,
        code_prefix="package-registry",
        path=f"{path}.path",
    )
    packaged_registry_hash = _validated_sha256(
        packaged_registry.get("registryHash"),
        path="$.packagedRuntimeVariantRegistry.registryHash",
    )
    if packaged_registry_hash != registry_hash:
        raise RuntimeVariantDispatchError(
            "package-registry-identity-mismatch",
            "Packaged runtime variant registry identity does not match the package manifest.",
            path=f"{path}.registryHash",
            details={
                "manifestSHA256": registry_hash,
                "registrySHA256": packaged_registry_hash,
            },
        )

    variants = packaged_registry.get("variants")
    if not isinstance(variants, Mapping):
        raise RuntimeVariantDispatchError(
            "package-registry-invalid",
            "Packaged runtime variant registry variants must be an object.",
            path="$.packagedRuntimeVariantRegistry.variants",
        )
    registry_summary = packaged_registry.get("summary")
    registry_summary_count = (
        registry_summary.get("variantCount")
        if isinstance(registry_summary, Mapping)
        else None
    )
    package_summary = package["summary"]
    package_summary_count = package_summary.get("runtimeVariantCount")
    counts = {
        "manifest": variant_count,
        "packageSummary": package_summary_count,
        "registrySummary": registry_summary_count,
        "registryVariants": len(variants),
    }
    if (
        not isinstance(registry_summary_count, int)
        or isinstance(registry_summary_count, bool)
        or len(set(counts.values())) != 1
    ):
        raise RuntimeVariantDispatchError(
            "package-registry-count-mismatch",
            "Runtime variant counts disagree across the native loader package.",
            path=f"{path}.variantCount",
            details={"variantCounts": counts},
        )

    _validate_native_header_reference(
        package_root,
        reference.get("nativeHeader"),
        registry_path=reference.get("path"),
    )
    return packaged_registry


def _validate_native_header_reference(
    package_root: Path,
    value: Any,
    *,
    registry_path: Any,
) -> None:
    path = "$.packageManifest.runtimeVariantRegistry.nativeHeader"
    if not isinstance(value, Mapping):
        raise RuntimeVariantDispatchError(
            "package-registry-header-invalid",
            "Native runtime variant registry header metadata must be an object.",
            path=path,
        )
    available = value.get("available")
    if available is True:
        expected_fields = _AVAILABLE_NATIVE_HEADER_FIELDS
    elif available is False and "unavailableTargets" in value:
        expected_fields = _TARGET_UNAVAILABLE_NATIVE_HEADER_FIELDS
    elif available is False:
        expected_fields = _UNAVAILABLE_NATIVE_HEADER_FIELDS
    else:
        expected_fields = frozenset()
    if not expected_fields or frozenset(value) != expected_fields:
        raise RuntimeVariantDispatchError(
            "package-registry-header-invalid",
            "Native runtime variant registry header metadata has an invalid field set.",
            path=path,
            details={
                "available": available,
                "actualFields": sorted(value, key=str),
                "expectedFields": sorted(expected_fields),
            },
        )
    if available is True:
        header_path = value.get("path")
        if header_path == registry_path:
            raise RuntimeVariantDispatchError(
                "package-registry-header-invalid",
                "Native registry header and JSON registry must use distinct package paths.",
                path=f"{path}.path",
            )
        _read_verified_package_file(
            package_root,
            header_path,
            value.get("hash"),
            field_path=f"{path}.path",
            label="Native runtime variant registry header",
            code_prefix="package-registry-header",
        )
        return

    reason = value.get("reason")
    if not isinstance(reason, str) or not reason or reason != reason.strip():
        raise RuntimeVariantDispatchError(
            "package-registry-header-invalid",
            "Unavailable native registry header reason must be a non-empty string.",
            path=f"{path}.reason",
        )
    if "unavailableTargets" not in value:
        return
    unavailable_targets = value.get("unavailableTargets")
    if (
        not isinstance(unavailable_targets, list)
        or not unavailable_targets
        or any(
            not isinstance(target, str) or not target or target != target.strip()
            for target in unavailable_targets
        )
        or unavailable_targets != sorted(set(unavailable_targets))
    ):
        raise RuntimeVariantDispatchError(
            "package-registry-header-invalid",
            "Unavailable native registry targets must be sorted unique strings.",
            path=f"{path}.unavailableTargets",
        )


def _verify_registry_binding(
    registry: Mapping[str, Any],
    selected: Mapping[str, Any],
    packaged_registry: Mapping[str, Any],
    *,
    key: str,
    reference: Mapping[str, Any],
) -> None:
    manifest_hash = _validated_sha256(
        reference.get("registryHash"),
        path="$.packageManifest.runtimeVariantRegistry.registryHash",
    )
    supplied_hash = _validated_sha256(
        registry.get("registryHash"),
        path="$.runtimeVariantRegistry.registryHash",
    )
    if supplied_hash != manifest_hash:
        raise RuntimeVariantDispatchError(
            "registry-package-mismatch",
            "Supplied runtime variant registry does not belong to the native loader package.",
            path="$.runtimeVariantRegistry.registryHash",
            details={
                "suppliedSHA256": supplied_hash,
                "packageSHA256": manifest_hash,
            },
        )
    try:
        packaged_selected = _selected_record(packaged_registry, key)
    except RuntimeVariantDispatchError as exc:
        raise RuntimeVariantDispatchError(
            "package-registry-invalid",
            "Packaged runtime variant registry cannot select the requested key.",
            path="$.packageManifest.runtimeVariantRegistry.path",
            details={"registryDiagnostic": exc.to_json()},
        ) from exc
    if selected != packaged_selected:
        raise RuntimeVariantDispatchError(
            "registry-package-mismatch",
            "Supplied and packaged runtime variant records disagree.",
            path="$.runtimeVariantRegistry.variants",
            details={"requestedKey": key},
        )


def _select_package_unit(
    package: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = selected.get("artifact")
    artifact_id = artifact.get("id") if isinstance(artifact, Mapping) else None
    if not isinstance(artifact_id, str) or not artifact_id:
        raise RuntimeVariantDispatchError(
            "variant-artifact-identity-invalid",
            "Selected runtime variant is missing a stable artifact identity.",
            path="$.selectedVariant.artifact.id",
        )
    provenance = selected.get("provenance")
    provenance_id = (
        provenance.get("artifactId") if isinstance(provenance, Mapping) else None
    )
    if provenance_id != artifact_id:
        raise RuntimeVariantDispatchError(
            "variant-artifact-identity-mismatch",
            "Selected runtime variant artifact identities disagree.",
            path="$.selectedVariant.provenance.artifactId",
            details={"artifactId": artifact_id, "provenanceArtifactId": provenance_id},
        )
    matches = [
        unit
        for unit in package["units"]
        if isinstance(unit, Mapping) and unit.get("unitId") == artifact_id
    ]
    if len(matches) != 1:
        raise RuntimeVariantDispatchError(
            "package-unit-match-invalid",
            "Native loader package must contain exactly one unit for the selected artifact.",
            path="$.packageManifest.units",
            details={"artifactId": artifact_id, "matchCount": len(matches)},
        )
    return copy.deepcopy(dict(matches[0]))


def _load_descriptor(
    package_root: Path,
    package_unit: Mapping[str, Any],
) -> dict[str, Any]:
    relative = _safe_package_path(
        package_unit["descriptorPath"],
        path="$.packageUnit.descriptorPath",
    )
    unresolved = package_root.joinpath(*relative.parts)
    try:
        descriptor_path = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeVariantDispatchError(
            "descriptor-unavailable",
            "Native loader ABI descriptor is unavailable.",
            path="$.packageUnit.descriptorPath",
            details={
                "descriptorPath": package_unit["descriptorPath"],
                "error": str(exc),
            },
        ) from exc
    _require_contained(
        descriptor_path,
        package_root,
        code="descriptor-path-escape",
        path="$.packageUnit.descriptorPath",
        message="Native loader ABI descriptor resolves outside the package root.",
    )
    if not descriptor_path.is_file():
        raise RuntimeVariantDispatchError(
            "descriptor-not-file",
            "Native loader ABI descriptor must resolve to a regular file.",
            path="$.packageUnit.descriptorPath",
        )
    try:
        content = descriptor_path.read_bytes()
    except OSError as exc:
        raise RuntimeVariantDispatchError(
            "descriptor-read-failed",
            "Native loader ABI descriptor could not be read.",
            path="$.packageUnit.descriptorPath",
            details={"error": str(exc)},
        ) from exc
    expected_size = package_unit.get("descriptorSizeBytes")
    if expected_size is not None and len(content) != expected_size:
        raise RuntimeVariantDispatchError(
            "descriptor-size-mismatch",
            "Native loader ABI descriptor byte size does not match the package unit.",
            path="$.packageUnit.descriptorSizeBytes",
            details={
                "expectedSizeBytes": expected_size,
                "actualSizeBytes": len(content),
            },
        )
    expected_hash = _validated_sha256(
        package_unit["descriptorHash"],
        path="$.packageUnit.descriptorHash",
    )
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeVariantDispatchError(
            "descriptor-hash-mismatch",
            "Native loader ABI descriptor hash does not match the package unit.",
            path="$.packageUnit.descriptorHash.value",
            details={"expectedSHA256": expected_hash, "actualSHA256": actual_hash},
        )
    descriptor = _decode_json_object(
        content,
        code_prefix="descriptor",
        path="$.packageUnit.descriptorPath",
    )
    try:
        return _validate_descriptor(descriptor)
    except NativeLoaderABIError as exc:
        raise RuntimeVariantDispatchError(
            "descriptor-schema-invalid",
            "Native loader ABI descriptor validation failed.",
            path=exc.path,
            details={"descriptorDiagnostic": exc.to_json()},
        ) from exc


def _cross_check_identities(
    selected: Mapping[str, Any],
    package_unit: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> None:
    artifact = selected["artifact"]
    target = selected["target"]
    source = selected.get("source")
    provenance = selected.get("provenance")
    if not isinstance(source, Mapping):
        raise RuntimeVariantDispatchError(
            "source-identity-invalid",
            "Selected runtime variant source identity must be an object.",
            path="$.selectedVariant.source",
        )
    if not isinstance(provenance, Mapping):
        raise RuntimeVariantDispatchError(
            "variant-artifact-identity-invalid",
            "Selected runtime variant provenance must be an object.",
            path="$.selectedVariant.provenance",
        )
    checks = (
        (
            "$.packageUnit.unitId",
            artifact["id"],
            package_unit["unitId"],
            "unit identity",
        ),
        (
            "$.descriptor.unitId",
            artifact["id"],
            descriptor["unitId"],
            "descriptor unit identity",
        ),
        (
            "$.descriptor.target",
            target["backend"],
            descriptor["target"],
            "target",
        ),
        (
            "$.descriptor.stage",
            target["stage"],
            descriptor["stage"],
            "stage",
        ),
        (
            "$.descriptor.entryPoint.name",
            target["entryPoint"],
            descriptor["entryPoint"]["name"],
            "entry point",
        ),
        (
            "$.descriptor.artifact.packagePath",
            artifact["path"],
            descriptor["artifact"]["packagePath"],
            "artifact path",
        ),
        (
            "$.descriptor.artifact.format",
            artifact["format"],
            descriptor["artifact"]["format"],
            "artifact format",
        ),
        (
            "$.descriptor.artifact.hash",
            artifact["hash"],
            descriptor["artifact"]["hash"],
            "artifact hash",
        ),
        (
            "$.descriptor.artifact.sizeBytes",
            artifact["sizeBytes"],
            descriptor["artifact"]["sizeBytes"],
            "artifact size",
        ),
        (
            "$.packageUnit.target",
            descriptor["target"],
            package_unit["target"],
            "package target",
        ),
        (
            "$.packageUnit.stage",
            descriptor["stage"],
            package_unit["stage"],
            "package stage",
        ),
        (
            "$.packageUnit.entryPoint",
            descriptor["entryPoint"]["name"],
            package_unit["entryPoint"],
            "package entry point",
        ),
        (
            "$.packageUnit.artifact",
            descriptor["artifact"],
            package_unit["artifact"],
            "package artifact",
        ),
        (
            "$.descriptor.source.path",
            source.get("unit"),
            descriptor["source"]["path"],
            "source unit",
        ),
        (
            "$.descriptor.source.backend",
            source.get("backend"),
            descriptor["source"]["backend"],
            "source backend",
        ),
        (
            "$.descriptor.source.hash",
            provenance.get("sourceHash"),
            descriptor["source"]["hash"],
            "source hash",
        ),
        (
            "$.descriptor.source.remap",
            provenance.get("sourceRemap"),
            descriptor["source"]["remap"],
            "source remap",
        ),
    )
    for path, expected, actual, label in checks:
        if expected != actual:
            raise RuntimeVariantDispatchError(
                "identity-mismatch",
                f"Selected runtime variant {label} does not match the ABI package.",
                path=path,
                details={
                    "expected": copy.deepcopy(expected),
                    "actual": copy.deepcopy(actual),
                },
            )

    binding_interface = selected.get("bindingInterface")
    if not isinstance(binding_interface, Mapping):
        raise RuntimeVariantDispatchError(
            "binding-interface-invalid",
            "Selected runtime variant binding interface must be an object.",
            path="$.selectedVariant.bindingInterface",
        )
    interface_checks = (
        ("status", "ready", binding_interface.get("status")),
        ("artifactFormat", artifact["format"], binding_interface.get("artifactFormat")),
    )
    for field, expected, actual in interface_checks:
        if expected != actual:
            raise RuntimeVariantDispatchError(
                "binding-interface-mismatch",
                "Selected runtime variant binding interface does not match its artifact.",
                path=f"$.selectedVariant.bindingInterface.{field}",
                details={"expected": expected, "actual": actual},
            )
    interface_entry = binding_interface.get("entryPoint")
    if not isinstance(interface_entry, Mapping) or (
        interface_entry.get("name") != target["entryPoint"]
        or interface_entry.get("stage") != target["stage"]
    ):
        raise RuntimeVariantDispatchError(
            "entry-point-mismatch",
            "Selected runtime variant binding entry point does not match the target identity.",
            path="$.selectedVariant.bindingInterface.entryPoint",
        )

    descriptor_execution = _descriptor_execution_identity(descriptor["entryPoint"])
    if selected.get("execution") != descriptor_execution:
        raise RuntimeVariantDispatchError(
            "execution-mismatch",
            "Selected runtime variant execution identity does not match the ABI descriptor.",
            path="$.selectedVariant.execution",
            details={
                "selectedExecution": copy.deepcopy(selected.get("execution")),
                "descriptorExecution": descriptor_execution,
            },
        )


def _descriptor_execution_identity(entry_point: Mapping[str, Any]) -> dict[str, Any]:
    config = entry_point.get("executionConfig")
    if not isinstance(config, Mapping):
        raise RuntimeVariantDispatchError(
            "execution-invalid",
            "Native loader ABI entry-point executionConfig must be an object.",
            path="$.descriptor.entryPoint.executionConfig",
        )
    workgroup_candidates: list[list[int]] = []
    for field in (
        "workgroupSize",
        "workgroup_size",
        "numthreads",
        "localSize",
        "local_size",
    ):
        if field in config and config.get(field) is not None:
            workgroup_candidates.append(
                _positive_dimensions(
                    config[field],
                    path=f"$.descriptor.entryPoint.executionConfig.{field}",
                    label="workgroup size",
                )
            )
    component_fields = ("local_size_x", "local_size_y", "local_size_z")
    if any(field in config for field in component_fields):
        workgroup_candidates.append(
            _positive_dimensions(
                [config.get(field, 1) for field in component_fields],
                path="$.descriptor.entryPoint.executionConfig.local_size",
                label="workgroup size",
            )
        )
    padded_workgroups = {_padded_dimensions(value) for value in workgroup_candidates}
    if len(padded_workgroups) > 1:
        raise RuntimeVariantDispatchError(
            "execution-ambiguous",
            "Native loader ABI descriptor contains conflicting workgroup sizes.",
            path="$.descriptor.entryPoint.executionConfig",
        )
    workgroup = list(next(iter(padded_workgroups))) if padded_workgroups else None

    subgroup_values = set()
    for field in ("subgroupWidth", "subgroup_width", "waveSize", "wave_size"):
        if field not in config or config.get(field) is None:
            continue
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeVariantDispatchError(
                "execution-invalid",
                "Native loader ABI subgroup width must be a positive integer.",
                path=f"$.descriptor.entryPoint.executionConfig.{field}",
            )
        subgroup_values.add(value)
    if len(subgroup_values) > 1:
        raise RuntimeVariantDispatchError(
            "execution-ambiguous",
            "Native loader ABI descriptor contains conflicting subgroup widths.",
            path="$.descriptor.entryPoint.executionConfig",
        )
    subgroup = next(iter(subgroup_values)) if subgroup_values else None
    return {"workgroupSize": workgroup, "subgroupWidth": subgroup}


def _selected_specialization_values(
    selected: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> dict[int, Any]:
    selected_constants = selected.get("specializationConstants")
    descriptor_constants = descriptor.get("specializationConstants")
    if not isinstance(selected_constants, list) or not isinstance(
        descriptor_constants, list
    ):
        raise RuntimeVariantDispatchError(
            "specialization-schema-invalid",
            "Runtime variant and ABI specialization constants must be lists.",
            path="$.selectedVariant.specializationConstants",
        )

    selected_by_identity: dict[tuple[int, str | None], Mapping[str, Any]] = {}
    selected_ids: dict[int, str | None] = {}
    selected_names: dict[str, int] = {}
    for index, constant in enumerate(selected_constants):
        path = f"$.selectedVariant.specializationConstants[{index}]"
        if not isinstance(constant, Mapping):
            raise RuntimeVariantDispatchError(
                "specialization-schema-invalid",
                "Selected specialization constants must be objects.",
                path=path,
            )
        if constant.get("runtimeRole") != "pipeline-specialization":
            raise RuntimeVariantDispatchError(
                "specialization-role-mismatch",
                "Selected specialization constant is not a pipeline-time value.",
                path=f"{path}.runtimeRole",
                details={"runtimeRole": constant.get("runtimeRole")},
            )
        constant_id = constant.get("id")
        name = constant.get("name")
        if (
            not isinstance(constant_id, int)
            or isinstance(constant_id, bool)
            or not 0 <= constant_id <= _UINT32_MAX
            or (name is not None and (not isinstance(name, str) or not name))
        ):
            raise RuntimeVariantDispatchError(
                "specialization-identity-invalid",
                "Selected specialization constants require an unambiguous uint32 id.",
                path=path,
            )
        identity = (constant_id, name)
        if (
            identity in selected_by_identity
            or constant_id in selected_ids
            or (
                isinstance(name, str)
                and name in selected_names
                and selected_names[name] != constant_id
            )
        ):
            raise RuntimeVariantDispatchError(
                "specialization-identity-ambiguous",
                "Selected specialization constant identity is ambiguous.",
                path=path,
                details={"id": constant_id, "name": name},
            )
        selected_by_identity[identity] = constant
        selected_ids[constant_id] = name
        if isinstance(name, str):
            selected_names[name] = constant_id

    descriptor_by_identity: dict[tuple[int, str | None], Mapping[str, Any]] = {}
    descriptor_ids: set[int] = set()
    for index, constant in enumerate(descriptor_constants):
        path = f"$.descriptor.specializationConstants[{index}]"
        if not isinstance(constant, Mapping):
            raise RuntimeVariantDispatchError(
                "specialization-schema-invalid",
                "ABI specialization constants must be objects.",
                path=path,
            )
        constant_id = constant.get("id", constant.get("constantId"))
        name = constant.get("name")
        if (
            not isinstance(constant_id, int)
            or isinstance(constant_id, bool)
            or not 0 <= constant_id <= _UINT32_MAX
            or (name is not None and (not isinstance(name, str) or not name))
        ):
            raise RuntimeVariantDispatchError(
                "specialization-identity-invalid",
                "ABI specialization constants require an unambiguous uint32 id.",
                path=path,
            )
        identity = (constant_id, name)
        if identity in descriptor_by_identity or constant_id in descriptor_ids:
            raise RuntimeVariantDispatchError(
                "specialization-identity-ambiguous",
                "ABI specialization constant identity is ambiguous.",
                path=path,
                details={"id": constant_id, "name": name},
            )
        descriptor_by_identity[identity] = constant
        descriptor_ids.add(constant_id)

    if set(selected_by_identity) != set(descriptor_by_identity):
        raise RuntimeVariantDispatchError(
            "specialization-identity-mismatch",
            "Selected specialization identities do not match the ABI descriptor.",
            path="$.selectedVariant.specializationConstants",
            details={
                "selectedIdentities": _identity_payload(selected_by_identity),
                "descriptorIdentities": _identity_payload(descriptor_by_identity),
            },
        )

    values: dict[int, Any] = {}
    for identity, selected_constant in sorted(
        selected_by_identity.items(),
        key=lambda item: (item[0][0], item[0][1] or ""),
    ):
        descriptor_constant = descriptor_by_identity[identity]
        selected_dtype = selected_constant.get("dtype")
        descriptor_dtype = descriptor_constant.get("dtype")
        if selected_dtype != descriptor_dtype:
            raise RuntimeVariantDispatchError(
                "specialization-type-mismatch",
                "Selected specialization type does not match the ABI descriptor.",
                path="$.selectedVariant.specializationConstants",
                details={
                    "id": identity[0],
                    "selectedType": selected_dtype,
                    "descriptorType": descriptor_dtype,
                },
            )
        if "value" not in selected_constant or selected_constant.get("value") is None:
            raise RuntimeVariantDispatchError(
                "specialization-value-missing",
                "Selected runtime variant specialization value is unavailable.",
                path="$.selectedVariant.specializationConstants",
                details={"id": identity[0], "name": identity[1]},
            )
        values[identity[0]] = copy.deepcopy(selected_constant["value"])
    return values


def _selected_dispatch_geometry(
    value: RuntimeDispatchGeometry | Mapping[str, Any] | Sequence[int],
    *,
    selected_execution: Any,
    entry_point: str,
) -> RuntimeDispatchGeometry | dict[str, Any]:
    if not isinstance(selected_execution, Mapping):
        raise RuntimeVariantDispatchError(
            "execution-invalid",
            "Selected runtime variant execution identity must be an object.",
            path="$.selectedVariant.execution",
        )
    selected_workgroup = _positive_dimensions(
        selected_execution.get("workgroupSize"),
        path="$.selectedVariant.execution.workgroupSize",
        label="workgroup size",
    )

    if isinstance(value, RuntimeDispatchGeometry):
        _reject_workgroup_override(value.workgroup_size, selected_workgroup)
        _positive_dimensions(
            value.workgroup_count,
            path="$.dispatchGeometry.workgroupCount",
            label="workgroup count",
        )
        if value.entry_point is not None and value.entry_point != entry_point:
            raise RuntimeVariantDispatchError(
                "entry-point-mismatch",
                "Dispatch geometry entry point does not match the selected variant.",
                path="$.dispatchGeometry.entryPoint",
                details={
                    "expectedEntryPoint": entry_point,
                    "actualEntryPoint": value.entry_point,
                },
            )
        return replace(
            value,
            entry_point=entry_point,
            workgroup_size=tuple(selected_workgroup),
        )
    if isinstance(value, Mapping):
        result = copy.deepcopy(dict(value))
        _reject_workgroup_override(result.get("workgroupSize"), selected_workgroup)
        if result.get("workgroupCount") is not None:
            _positive_dimensions(
                result["workgroupCount"],
                path="$.dispatchGeometry.workgroupCount",
                label="workgroup count",
            )
        if result.get("entryPoint") not in (None, entry_point):
            raise RuntimeVariantDispatchError(
                "entry-point-mismatch",
                "Dispatch geometry entry point does not match the selected variant.",
                path="$.dispatchGeometry.entryPoint",
                details={
                    "expectedEntryPoint": entry_point,
                    "actualEntryPoint": result.get("entryPoint"),
                },
            )
        result["entryPoint"] = entry_point
        result["workgroupSize"] = list(selected_workgroup)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        workgroup_count = _positive_dimensions(
            value,
            path="$.dispatchGeometry.workgroupCount",
            label="workgroup count",
        )
        return {
            "entryPoint": entry_point,
            "workgroupCount": workgroup_count,
            "workgroupSize": list(selected_workgroup),
        }
    return value  # type: ignore[return-value]


def _reject_workgroup_override(
    supplied: Any,
    selected: Sequence[int],
) -> None:
    if supplied in (None, (), []):
        return
    supplied_dimensions = _positive_dimensions(
        supplied,
        path="$.dispatchGeometry.workgroupSize",
        label="workgroup size",
    )
    if _padded_dimensions(supplied_dimensions) != _padded_dimensions(selected):
        raise RuntimeVariantDispatchError(
            "workgroup-size-override",
            "Dispatch geometry cannot override the selected variant workgroup size.",
            path="$.dispatchGeometry.workgroupSize",
            details={
                "selectedWorkgroupSize": list(selected),
                "suppliedWorkgroupSize": supplied_dimensions,
            },
        )


def _positive_dimensions(value: Any, *, path: str, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeVariantDispatchError(
            "execution-invalid",
            f"Runtime variant {label} must be a one-to-three component sequence.",
            path=path,
        )
    dimensions = list(value)
    if not 1 <= len(dimensions) <= 3 or any(
        not isinstance(item, int)
        or isinstance(item, bool)
        or not 0 < item <= _UINT32_MAX
        for item in dimensions
    ):
        raise RuntimeVariantDispatchError(
            "execution-invalid",
            f"Runtime variant {label} must contain one to three positive uint32 integers.",
            path=path,
            details={"value": copy.deepcopy(dimensions)},
        )
    return dimensions


def _padded_dimensions(value: Sequence[int]) -> tuple[int, int, int]:
    return tuple([*value, *([1] * (3 - len(value)))])  # type: ignore[return-value]


def _read_verified_package_file(
    package_root: Path,
    relative_value: Any,
    hash_value: Any,
    *,
    field_path: str,
    label: str,
    code_prefix: str,
) -> bytes:
    relative = _safe_package_path(
        relative_value,
        path=field_path,
        code=f"{code_prefix}-path-invalid",
        label=f"{label} package path",
    )
    unresolved = package_root.joinpath(*relative.parts)
    try:
        resolved = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeVariantDispatchError(
            f"{code_prefix}-unavailable",
            f"{label} is unavailable.",
            path=field_path,
            details={"packagePath": relative_value, "error": str(exc)},
        ) from exc
    _require_contained(
        resolved,
        package_root,
        code=f"{code_prefix}-path-escape",
        path=field_path,
        message=f"{label} resolves outside the package root.",
    )
    if not resolved.is_file():
        raise RuntimeVariantDispatchError(
            f"{code_prefix}-not-file",
            f"{label} must resolve to a regular file.",
            path=field_path,
        )
    try:
        content = resolved.read_bytes()
    except OSError as exc:
        raise RuntimeVariantDispatchError(
            f"{code_prefix}-read-failed",
            f"{label} could not be read.",
            path=field_path,
            details={"error": str(exc)},
        ) from exc
    expected_hash = _validated_sha256(
        hash_value,
        path=field_path.rsplit(".", 1)[0] + ".hash",
    )
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeVariantDispatchError(
            f"{code_prefix}-hash-mismatch",
            f"{label} hash does not match the package manifest.",
            path=field_path.rsplit(".", 1)[0] + ".hash.value",
            details={
                "expectedSHA256": expected_hash,
                "actualSHA256": actual_hash,
            },
        )
    return content


def _safe_package_path(
    value: Any,
    *,
    path: str,
    code: str = "descriptor-path-invalid",
    label: str = "Descriptor package path",
) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        raise RuntimeVariantDispatchError(
            code,
            f"{label} must use portable relative path syntax.",
            path=path,
            details={"packagePath": value},
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
        raise RuntimeVariantDispatchError(
            code,
            f"{label} must be portable, normalized, relative, and traversal-free.",
            path=path,
            details={
                "packagePath": value,
                "nonportableComponents": nonportable_components,
            },
        )
    return posix_path


def _require_contained(
    resolved: Path,
    root: Path,
    *,
    code: str,
    path: str,
    message: str,
) -> None:
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeVariantDispatchError(code, message, path=path) from exc


def _validated_sha256(value: Any, *, path: str) -> str:
    if (
        not isinstance(value, Mapping)
        or frozenset(value) != _SHA256_FIELDS
        or value.get("algorithm") != "sha256"
        or not isinstance(value.get("value"), str)
        or len(value["value"]) != 64
        or any(character not in "0123456789abcdef" for character in value["value"])
    ):
        raise RuntimeVariantDispatchError(
            "hash-invalid",
            "Content identity must be a lowercase SHA-256 hash object.",
            path=path,
        )
    return value["value"]


def _read_json_object(
    path_value: Path,
    *,
    code_prefix: str,
    path: str,
) -> dict[str, Any]:
    try:
        content = path_value.read_bytes()
    except OSError as exc:
        raise RuntimeVariantDispatchError(
            f"{code_prefix}-read-failed",
            "JSON input could not be read.",
            path=path,
            details={"inputPath": str(path_value), "error": str(exc)},
        ) from exc
    return _decode_json_object(content, code_prefix=code_prefix, path=path)


def _decode_json_object(
    content: bytes,
    *,
    code_prefix: str,
    path: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except _DuplicateJSONKeyError as exc:
        raise RuntimeVariantDispatchError(
            f"{code_prefix}-json-invalid",
            "Input JSON objects must not contain duplicate keys.",
            path=path,
            details={"duplicateKey": exc.key},
        ) from exc
    except _InvalidJSONConstantError as exc:
        raise RuntimeVariantDispatchError(
            f"{code_prefix}-json-invalid",
            "Input must use finite standard JSON numbers.",
            path=path,
            details={"constant": str(exc)},
        ) from exc
    except (UnicodeError, ValueError) as exc:
        raise RuntimeVariantDispatchError(
            f"{code_prefix}-json-invalid",
            "Input must contain valid finite UTF-8 JSON.",
            path=path,
            details={"error": str(exc)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeVariantDispatchError(
            f"{code_prefix}-schema-invalid",
            "JSON input must contain an object.",
            path=path,
        )
    return copy.deepcopy(dict(payload))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _InvalidJSONConstantError(value)


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _InvalidJSONConstantError(value)
    return parsed


def _identity_payload(
    values: Mapping[tuple[int, str | None], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"id": constant_id, "name": name}
        for constant_id, name in sorted(
            values,
            key=lambda identity: (identity[0], identity[1] or ""),
        )
    ]
