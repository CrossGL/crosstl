"""Deterministic native loader ABI packages for runtime loader manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from .native_loader_abi import (
    NATIVE_LOADER_ABI_VERSION,
    NativeLoaderABIError,
    build_native_loader_abi_descriptor,
    generate_native_loader_declarations,
    generate_native_loader_execution_abi,
)
from .native_runtime_variant_registry import (
    NativeRuntimeVariantRegistryError,
    generate_native_runtime_variant_registry,
)
from .native_target_adapters import (
    NativeLoaderTargetAdapterError,
    generate_native_loader_target_adapter,
    native_loader_target_adapter_targets,
)
from .pipeline import build_runtime_variant_registry

NATIVE_LOADER_ABI_PACKAGE_KIND = "crosstl-native-loader-abi-package"
NATIVE_LOADER_ABI_PACKAGE_VERSION = 3
NATIVE_LOADER_ABI_PACKAGE_MANIFEST = "native-loader-abi-package.json"
NATIVE_RUNTIME_VARIANT_REGISTRY_PATH = "runtime/runtime-variant-registry.json"
NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH = "native-runtime-variant-registry.hpp"

_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
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


def build_native_loader_abi_package(
    loader_manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Write descriptors, execution interfaces, and available target adapters.

    Descriptors, declarations, execution interfaces, adapters, and the exact
    variant registry are generated before any output is written. Structural
    unit failures therefore prevent package publication. When the registry is
    ready, every referenced target artifact is also verified before writing.
    """

    manifest_path = _filesystem_path(
        loader_manifest_path,
        field="Runtime loader manifest path",
        error_path="$.loaderManifestPath",
    )
    package_root = _filesystem_path(
        output_dir,
        field="Native loader ABI output directory",
        error_path="$.outputDirectory",
    )
    if _same_path(package_root / NATIVE_LOADER_ABI_PACKAGE_MANIFEST, manifest_path):
        raise NativeLoaderABIError(
            "path-collision",
            "Native loader ABI output manifest must not overwrite its input manifest.",
            path="$.outputDirectory",
            details={"manifestPath": str(manifest_path)},
        )
    manifest, source_bytes = _read_loader_manifest(manifest_path)
    unit_ids = _load_unit_ids(manifest)

    generated_units: list[dict[str, Any]] = []
    generated_contents: list[tuple[str, str | bytes, str]] = []
    native_registry_units: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        descriptor = build_native_loader_abi_descriptor(
            manifest,
            load_unit_id=unit_id,
        )
        declarations = generate_native_loader_declarations(descriptor)
        execution_abi = generate_native_loader_execution_abi(descriptor)
        descriptor_text = _json_text(descriptor)
        descriptor_path, declarations_path, execution_abi_path = _unit_output_paths(
            descriptor
        )
        generated_contents.extend(
            (
                (
                    descriptor_path,
                    descriptor_text,
                    "native-loader-abi-descriptor",
                ),
                (
                    declarations_path,
                    declarations,
                    "native-loader-c-declarations",
                ),
                (
                    execution_abi_path,
                    execution_abi,
                    "native-loader-execution-abi",
                ),
            )
        )
        generated_units.append(
            {
                "unitId": descriptor["unitId"],
                "target": descriptor["target"],
                "stage": descriptor.get("stage"),
                "entryPoint": descriptor["entryPoint"]["name"],
                "artifact": descriptor["artifact"],
                "descriptorPath": descriptor_path,
                "descriptorHash": _content_hash(descriptor_text.encode("utf-8")),
                "descriptorSizeBytes": len(descriptor_text.encode("utf-8")),
                "declarationsPath": declarations_path,
                "declarationsHash": _content_hash(declarations.encode("utf-8")),
                "executionABIPath": execution_abi_path,
                "executionABIHash": _content_hash(execution_abi.encode("utf-8")),
            }
        )
        native_registry_units.append(
            {
                "descriptor": descriptor,
                "executionHeader": execution_abi_path,
            }
        )

    generated_units.sort(key=lambda unit: (unit["target"], unit["unitId"]))
    target_adapters = _generate_target_adapters(
        {unit["target"] for unit in generated_units},
        generated_contents,
    )
    runtime_variant_registry = _generate_runtime_variant_registry(
        manifest_path,
        native_registry_units,
        target_adapters,
        generated_contents,
    )
    if runtime_variant_registry["available"]:
        _package_runtime_artifacts(
            manifest_path,
            native_registry_units,
            generated_contents,
        )
    generated_contents.sort(key=lambda item: item[0])
    _validate_generated_output_paths(
        package_root,
        manifest_path,
        generated_contents,
    )
    package = {
        "schemaVersion": NATIVE_LOADER_ABI_PACKAGE_VERSION,
        "kind": NATIVE_LOADER_ABI_PACKAGE_KIND,
        "abiVersion": NATIVE_LOADER_ABI_VERSION,
        "sourceLoaderManifest": str(manifest_path),
        "sourceLoaderManifestHash": _content_hash(source_bytes),
        "success": True,
        "summary": {
            "unitCount": len(generated_units),
            "targetCount": len({unit["target"] for unit in generated_units}),
            "targetAdapterCount": sum(
                adapter["available"] for adapter in target_adapters
            ),
            "unavailableTargetAdapterCount": sum(
                not adapter["available"] for adapter in target_adapters
            ),
            "runtimeVariantCount": runtime_variant_registry["variantCount"],
            "generatedFileCount": 1 + len(generated_contents),
        },
        "units": generated_units,
        "targetAdapters": target_adapters,
        "runtimeVariantRegistry": runtime_variant_registry,
        "generatedFiles": [
            {
                "path": NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
                "kind": "native-loader-abi-package-manifest",
            },
            *[
                {
                    "path": path,
                    "kind": kind,
                }
                for path, _content, kind in generated_contents
            ],
        ],
    }

    _invalidate_package_manifest(package_root)
    for relative_path, content, _kind in generated_contents:
        _write_output(package_root, relative_path, content)
    _write_output(package_root, NATIVE_LOADER_ABI_PACKAGE_MANIFEST, _json_text(package))
    return package


def _generate_runtime_variant_registry(
    manifest_path: Path,
    units: list[dict[str, Any]],
    target_adapters: list[dict[str, Any]],
    generated_contents: list[tuple[str, str | bytes, str]],
) -> dict[str, Any]:
    registry = build_runtime_variant_registry(manifest_path)
    summary = registry.get("summary")
    variant_count = (
        summary.get("variantCount") if isinstance(summary, Mapping) else None
    )
    if (
        not isinstance(variant_count, int)
        or isinstance(variant_count, bool)
        or variant_count < 0
    ):
        raise NativeLoaderABIError(
            "runtime-variant-registry-generation-failed",
            "Runtime variant registry summary is missing a valid variant count.",
            path="$.summary.variantCount",
        )
    registry_hash = registry.get("registryHash")
    if not isinstance(registry_hash, Mapping):
        raise NativeLoaderABIError(
            "runtime-variant-registry-generation-failed",
            "Runtime variant registry is missing its content identity.",
            path="$.registryHash",
        )

    registry_text = _json_text(registry)
    generated_contents.append(
        (
            NATIVE_RUNTIME_VARIANT_REGISTRY_PATH,
            registry_text,
            "runtime-variant-registry",
        )
    )
    result = {
        "available": (
            registry.get("success") is True and registry.get("status") == "ready"
        ),
        "path": NATIVE_RUNTIME_VARIANT_REGISTRY_PATH,
        "hash": _content_hash(registry_text.encode("utf-8")),
        "registryHash": dict(registry_hash),
        "variantCount": variant_count,
    }
    if result["available"] is not True:
        result["reason"] = "runtime-variant-registry-unavailable"
        result["nativeHeader"] = {
            "available": False,
            "reason": "runtime-variant-registry-unavailable",
        }
        return result

    unavailable_targets = sorted(
        str(adapter.get("target"))
        for adapter in target_adapters
        if adapter.get("available") is not True
    )
    if unavailable_targets:
        result["nativeHeader"] = {
            "available": False,
            "reason": "target-adapter-unavailable",
            "unavailableTargets": unavailable_targets,
        }
        return result

    try:
        native_header = generate_native_runtime_variant_registry(registry, units)
    except NativeRuntimeVariantRegistryError as exc:
        if (
            exc.code.endswith(".specialization-mechanism-unsupported")
            and exc.details.get("target") == "opengl"
            and exc.details.get("artifactFormat") == "GLSL source"
        ):
            result["nativeHeader"] = {
                "available": False,
                "reason": "specialization-requires-deferred-compilation",
            }
            return result
        raise NativeLoaderABIError(
            "native-runtime-variant-registry-generation-failed",
            "Runtime variant registry could not form a native execution header.",
            path=exc.path,
            details={"diagnostic": exc.to_json()},
        ) from exc
    generated_contents.append(
        (
            NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH,
            native_header,
            "native-runtime-variant-registry",
        )
    )
    result["nativeHeader"] = {
        "available": True,
        "path": NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH,
        "hash": _content_hash(native_header.encode("utf-8")),
    }
    return result


def _package_runtime_artifacts(
    manifest_path: Path,
    units: list[dict[str, Any]],
    generated_contents: list[tuple[str, str | bytes, str]],
) -> None:
    try:
        source_root = manifest_path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NativeLoaderABIError(
            "artifact-root-unavailable",
            "Runtime loader artifact root is unavailable.",
            path="$.loaderManifestPath",
            details={"manifestPath": str(manifest_path), "error": str(exc)},
        ) from exc

    packaged: dict[str, bytes] = {}
    for index, unit in enumerate(units):
        descriptor = unit["descriptor"]
        artifact = descriptor["artifact"]
        artifact_path = f"$.loadUnits[{index}].artifact.packagePath"
        relative = _portable_package_path(
            artifact["packagePath"],
            path=artifact_path,
        )
        unresolved = source_root.joinpath(*relative.parts)
        try:
            resolved = unresolved.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise NativeLoaderABIError(
                "artifact-unavailable",
                "Runtime loader artifact is unavailable.",
                path=artifact_path,
                details={
                    "packagePath": artifact["packagePath"],
                    "error": str(exc),
                },
            ) from exc
        try:
            resolved.relative_to(source_root)
        except ValueError as exc:
            raise NativeLoaderABIError(
                "artifact-path-escape",
                "Runtime loader artifact resolves outside the package root.",
                path=artifact_path,
                details={"packagePath": artifact["packagePath"]},
            ) from exc
        if not resolved.is_file():
            raise NativeLoaderABIError(
                "artifact-not-file",
                "Runtime loader artifact must resolve to a regular file.",
                path=artifact_path,
                details={"packagePath": artifact["packagePath"]},
            )
        try:
            content = resolved.read_bytes()
        except OSError as exc:
            raise NativeLoaderABIError(
                "artifact-read-failed",
                "Runtime loader artifact could not be read.",
                path=artifact_path,
                details={
                    "packagePath": artifact["packagePath"],
                    "error": str(exc),
                },
            ) from exc
        if len(content) != artifact["sizeBytes"]:
            raise NativeLoaderABIError(
                "artifact-size-mismatch",
                "Runtime loader artifact size does not match its descriptor.",
                path=f"{artifact_path}.sizeBytes",
                details={
                    "packagePath": artifact["packagePath"],
                    "expectedSizeBytes": artifact["sizeBytes"],
                    "actualSizeBytes": len(content),
                },
            )
        expected_hash = artifact["hash"]["value"]
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            raise NativeLoaderABIError(
                "artifact-hash-mismatch",
                "Runtime loader artifact hash does not match its descriptor.",
                path=f"{artifact_path}.hash.value",
                details={
                    "packagePath": artifact["packagePath"],
                    "expectedSHA256": expected_hash,
                    "actualSHA256": actual_hash,
                },
            )
        package_path = relative.as_posix()
        previous = packaged.get(package_path)
        if previous is not None and previous != content:
            raise NativeLoaderABIError(
                "artifact-path-collision",
                "Runtime loader artifacts map different content to one package path.",
                path=artifact_path,
                details={"packagePath": package_path},
            )
        packaged[package_path] = content

    generated_contents.extend(
        (path, content, "runtime-target-artifact")
        for path, content in sorted(packaged.items())
    )


def _generate_target_adapters(
    targets: set[str],
    generated_contents: list[tuple[str, str | bytes, str]],
) -> list[dict[str, Any]]:
    supported = set(native_loader_target_adapter_targets())
    adapters: list[dict[str, Any]] = []
    for target in sorted(targets):
        if target not in supported:
            adapters.append(
                {
                    "target": target,
                    "available": False,
                    "reason": "target-adapter-unavailable",
                }
            )
            continue
        try:
            content = generate_native_loader_target_adapter(target)
        except NativeLoaderTargetAdapterError as exc:
            raise NativeLoaderABIError(
                "target-adapter-generation-failed",
                f"Native loader target adapter could not be generated: {exc.message}",
                path="$.loadUnits",
                details={"target": target, "diagnostic": exc.to_json()},
            ) from exc
        path = _target_adapter_output_path(target)
        generated_contents.append((path, content, "native-loader-target-adapter"))
        adapters.append(
            {
                "target": target,
                "available": True,
                "path": path,
                "hash": _content_hash(content.encode("utf-8")),
            }
        )
    return adapters


def _read_loader_manifest(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise NativeLoaderABIError(
            "manifest-read-failed",
            f"Runtime loader manifest could not be read: {exc}",
            path="$.loaderManifestPath",
            details={"manifestPath": str(path)},
        ) from exc
    try:
        payload = json.loads(
            source_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeError, ValueError) as exc:
        raise NativeLoaderABIError(
            "manifest-parse-failed",
            f"Runtime loader manifest must contain UTF-8 JSON: {exc}",
            path="$",
            details={"manifestPath": str(path)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise NativeLoaderABIError(
            "input-invalid",
            "Runtime loader manifest must be a JSON object.",
            path="$",
        )
    return payload, source_bytes


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object member {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r}")
    return parsed


def _load_unit_ids(manifest: Mapping[str, Any]) -> list[str]:
    units = manifest.get("loadUnits")
    if not isinstance(units, list) or not units:
        raise NativeLoaderABIError(
            "load-units-invalid",
            "Runtime loader manifest loadUnits must be a non-empty list of objects.",
            path="$.loadUnits",
        )

    unit_ids: list[str] = []
    seen: set[str] = set()
    for index, unit in enumerate(units):
        path = f"$.loadUnits[{index}]"
        if not isinstance(unit, Mapping):
            raise NativeLoaderABIError(
                "load-units-invalid",
                "Runtime loader manifest loadUnits must contain objects.",
                path=path,
            )
        unit_id = unit.get("id")
        if not isinstance(unit_id, str) or not unit_id or unit_id != unit_id.strip():
            raise NativeLoaderABIError(
                "load-unit-id-invalid",
                "Runtime loader unit id must be a non-empty trimmed string.",
                path=f"{path}.id",
            )
        if unit_id in seen:
            raise NativeLoaderABIError(
                "load-unit-id-duplicate",
                f"Runtime loader manifest contains duplicate load unit {unit_id!r}.",
                path=f"{path}.id",
                details={"unitId": unit_id},
            )
        seen.add(unit_id)
        unit_ids.append(unit_id)
    return sorted(unit_ids)


def _unit_output_paths(descriptor: Mapping[str, Any]) -> tuple[str, str, str]:
    target = _path_component(str(descriptor["target"]), fallback="target")
    unit = _path_component(str(descriptor["unitId"]), fallback="load-unit")
    directory = f"targets/{target}"
    declarations_base = f"{directory}/{unit}.native-loader-abi"
    return (
        f"{declarations_base}.json",
        f"{declarations_base}.h",
        f"{directory}/{unit}.native-loader-execution.h",
    )


def _target_adapter_output_path(target: str) -> str:
    component = _path_component(target, fallback="target")
    return f"targets/{component}/native-loader-target-adapter.hpp"


def _path_component(value: str, *, fallback: str) -> str:
    readable = _PATH_COMPONENT_RE.sub("-", value).strip("-._").lower()
    readable = readable[:64].rstrip("-._") or fallback
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def _portable_package_path(value: Any, *, path: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        raise NativeLoaderABIError(
            "artifact-path-invalid",
            "Runtime loader artifact path must use portable relative syntax.",
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
        raise NativeLoaderABIError(
            "artifact-path-invalid",
            "Runtime loader artifact path must be portable, normalized, relative, and traversal-free.",
            path=path,
            details={
                "packagePath": value,
                "nonportableComponents": nonportable_components,
            },
        )
    return posix_path


def _filesystem_path(value: Any, *, field: str, error_path: str) -> Path:
    try:
        path = Path(value)
    except (TypeError, ValueError) as exc:
        raise NativeLoaderABIError(
            "path-invalid",
            f"{field} must be a filesystem path.",
            path=error_path,
        ) from exc
    if not str(path):
        raise NativeLoaderABIError(
            "path-invalid",
            f"{field} must not be empty.",
            path=error_path,
        )
    return path


def _same_path(first: Path, second: Path) -> bool:
    try:
        return first.resolve(strict=False) == second.resolve(strict=False)
    except OSError:
        return first.absolute() == second.absolute()


def _validate_generated_output_paths(
    package_root: Path,
    manifest_path: Path,
    generated_contents: list[tuple[str, str | bytes, str]],
) -> None:
    try:
        resolved_root = package_root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise NativeLoaderABIError(
            "output-root-invalid",
            "Native loader ABI output directory could not be resolved.",
            path="$.outputDirectory",
            details={"outputDirectory": str(package_root), "error": str(exc)},
        ) from exc

    package_manifest_path = package_root / NATIVE_LOADER_ABI_PACKAGE_MANIFEST
    package_manifest_key = NATIVE_LOADER_ABI_PACKAGE_MANIFEST.casefold()
    output_paths: list[tuple[str, Path, str]] = []
    for relative_path, _content, _kind in generated_contents:
        output_path = package_root / relative_path
        portable_key = PurePosixPath(relative_path).as_posix().casefold()
        try:
            resolved_output = output_path.resolve(strict=False)
            resolved_output.relative_to(resolved_root)
        except ValueError as exc:
            raise NativeLoaderABIError(
                "output-path-escape",
                "Native loader ABI output resolves outside the package root.",
                path="$.outputDirectory",
                details={
                    "outputDirectory": str(package_root),
                    "relativePath": relative_path,
                },
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise NativeLoaderABIError(
                "output-path-invalid",
                "Native loader ABI output path could not be resolved.",
                path="$.outputDirectory",
                details={
                    "outputDirectory": str(package_root),
                    "relativePath": relative_path,
                    "error": str(exc),
                },
            ) from exc
        if portable_key == package_manifest_key or _same_path(
            output_path, package_manifest_path
        ):
            raise NativeLoaderABIError(
                "output-path-collision",
                "Native loader ABI output collides with its package manifest.",
                path="$.outputDirectory",
                details={
                    "firstRelativePath": NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
                    "secondRelativePath": relative_path,
                },
            )
        if _same_path(output_path, manifest_path):
            raise NativeLoaderABIError(
                "path-collision",
                "Native loader ABI output must not overwrite its input manifest.",
                path="$.outputDirectory",
                details={
                    "manifestPath": str(manifest_path),
                    "relativePath": relative_path,
                },
            )
        for previous_relative, previous_path, previous_key in output_paths:
            if portable_key == previous_key or _same_path(output_path, previous_path):
                raise NativeLoaderABIError(
                    "output-path-collision",
                    "Native loader ABI outputs resolve to the same filesystem path.",
                    path="$.outputDirectory",
                    details={
                        "firstRelativePath": previous_relative,
                        "secondRelativePath": relative_path,
                    },
                )
        output_paths.append((relative_path, output_path, portable_key))


def _invalidate_package_manifest(root: Path) -> None:
    manifest_path = root / NATIVE_LOADER_ABI_PACKAGE_MANIFEST
    try:
        if manifest_path.exists() or manifest_path.is_symlink():
            manifest_path.unlink()
    except OSError as exc:
        raise NativeLoaderABIError(
            "package-write-failed",
            f"Native loader ABI package output could not be written: {exc}",
            path="$.outputDirectory",
            details={
                "outputDirectory": str(root),
                "relativePath": NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
            },
        ) from exc


def _write_output(root: Path, relative_path: str, content: str | bytes) -> None:
    output_path = root / relative_path
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8") if isinstance(content, str) else content
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
        temporary_path.replace(output_path)
    except (OSError, UnicodeError) as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise NativeLoaderABIError(
            "package-write-failed",
            f"Native loader ABI package output could not be written: {exc}",
            path="$.outputDirectory",
            details={
                "outputDirectory": str(root),
                "relativePath": relative_path,
            },
        ) from exc


def _json_text(value: Mapping[str, Any]) -> str:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise NativeLoaderABIError(
            "json-invalid",
            f"Native loader ABI data must be JSON serializable: {exc}",
        ) from exc


def _content_hash(content: bytes) -> dict[str, str]:
    return {"algorithm": "sha256", "value": hashlib.sha256(content).hexdigest()}
