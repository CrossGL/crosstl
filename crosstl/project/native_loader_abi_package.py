"""Deterministic native loader ABI packages for runtime loader manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .native_loader_abi import (
    NATIVE_LOADER_ABI_VERSION,
    NativeLoaderABIError,
    build_native_loader_abi_descriptor,
    generate_native_loader_declarations,
)

NATIVE_LOADER_ABI_PACKAGE_KIND = "crosstl-native-loader-abi-package"
NATIVE_LOADER_ABI_PACKAGE_VERSION = 1
NATIVE_LOADER_ABI_PACKAGE_MANIFEST = "native-loader-abi-package.json"

_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def build_native_loader_abi_package(
    loader_manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Write descriptors and C declarations for every ready loader unit.

    All descriptors and declarations are built before any output is written. A
    blocked, incomplete, or duplicate load unit therefore prevents publication
    of a package manifest.
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
    generated_contents: list[tuple[str, str]] = []
    for unit_id in unit_ids:
        descriptor = build_native_loader_abi_descriptor(
            manifest,
            load_unit_id=unit_id,
        )
        declarations = generate_native_loader_declarations(descriptor)
        descriptor_text = _json_text(descriptor)
        descriptor_path, declarations_path = _unit_output_paths(descriptor)
        generated_contents.extend(
            (
                (descriptor_path, descriptor_text),
                (declarations_path, declarations),
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
                "declarationsPath": declarations_path,
                "declarationsHash": _content_hash(declarations.encode("utf-8")),
            }
        )

    generated_units.sort(key=lambda unit: (unit["target"], unit["unitId"]))
    generated_contents.sort(key=lambda item: item[0])
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
            "generatedFileCount": 1 + len(generated_contents),
        },
        "units": generated_units,
        "generatedFiles": [
            {
                "path": NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
                "kind": "native-loader-abi-package-manifest",
            },
            *[
                {
                    "path": path,
                    "kind": (
                        "native-loader-abi-descriptor"
                        if path.endswith(".json")
                        else "native-loader-c-declarations"
                    ),
                }
                for path, _content in generated_contents
            ],
        ],
    }

    for relative_path, content in generated_contents:
        _write_output(package_root, relative_path, content)
    _write_output(package_root, NATIVE_LOADER_ABI_PACKAGE_MANIFEST, _json_text(package))
    return package


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
        payload = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
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


def _unit_output_paths(descriptor: Mapping[str, Any]) -> tuple[str, str]:
    target = _path_component(str(descriptor["target"]), fallback="target")
    unit = _path_component(str(descriptor["unitId"]), fallback="load-unit")
    base = f"targets/{target}/{unit}.native-loader-abi"
    return f"{base}.json", f"{base}.h"


def _path_component(value: str, *, fallback: str) -> str:
    readable = _PATH_COMPONENT_RE.sub("-", value).strip("-._").lower()
    readable = readable[:64].rstrip("-._") or fallback
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


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


def _write_output(root: Path, relative_path: str, content: str) -> None:
    output_path = root / relative_path
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_bytes(content.encode("utf-8"))
        temporary_path.replace(output_path)
    except (OSError, UnicodeError) as exc:
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
