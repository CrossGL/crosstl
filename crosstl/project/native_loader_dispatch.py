"""Build native runtime requests from loader ABI descriptors."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from crosstl.project.native_loader_abi import (
    NativeLoaderABIError,
    _validate_descriptor,
)
from crosstl.project.runtime_verification import (
    RuntimeAdapterContract,
    RuntimeArtifactSelector,
    RuntimeDispatchGeometry,
    RuntimeEntryPoint,
    RuntimeExecutionRequest,
    RuntimeFixture,
    RuntimeResourceBinding,
    RuntimeSpecializationConstant,
    RuntimeTolerance,
    RuntimeValue,
    prepare_runtime_execution,
)

_ERROR_PREFIX = "project.native-loader-dispatch"
_COMPUTE_STAGE = "compute"
_TARGET_ARTIFACT_FORMATS = {
    "directx": "HLSL source",
    "opengl": "GLSL source",
}
_TARGET_RESOURCE_NAMESPACES = {
    "directx": {
        ("buffer", "read"): "srv",
        ("buffer", "write"): "uav",
        ("buffer", "read_write"): "uav",
        ("constant-buffer", "read"): "cbv",
    },
    "opengl": {
        ("buffer", "read"): "storage-buffer",
        ("buffer", "write"): "storage-buffer",
        ("buffer", "read_write"): "storage-buffer",
        ("constant-buffer", "read"): "uniform-buffer",
    },
}
_BUFFER_KIND_ALIASES = {
    "buffer": "buffer",
    "storage-buffer": "buffer",
    "constant-buffer": "constant-buffer",
    "constantbuffer": "constant-buffer",
    "uniform": "constant-buffer",
}
_BUFFER_DTYPE_ALIASES = {
    "float": "float32",
    "f32": "float32",
    "float32": "float32",
    "float32_t": "float32",
    "int": "int32",
    "i32": "int32",
    "int32": "int32",
    "int32_t": "int32",
    "uint": "uint32",
    "u32": "uint32",
    "uint32": "uint32",
    "uint32_t": "uint32",
}
_SPECIALIZATION_DTYPE_ALIASES = {
    **_BUFFER_DTYPE_ALIASES,
    "bool": "bool",
    "boolean": "bool",
}
_DTYPE_SIZES = {"float32": 4, "int32": 4, "uint32": 4}
_VALUE_FIELDS = frozenset(
    ("name", "kind", "dtype", "shape", "values", "value", "tolerance", "metadata")
)
_ALIAS_METADATA_FIELDS = frozenset(("aliases", "resourceAliases", "bindingAliases"))
_ENTRY_POINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UINT32_MAX = (1 << 32) - 1


class NativeLoaderDispatchError(ValueError):
    """A native loader descriptor cannot form a safe runtime request."""

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
        """Return the stable diagnostic representation for reports and tooling."""

        payload: dict[str, Any] = {
            "severity": "error",
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.details:
            payload["details"] = copy.deepcopy(self.details)
        return payload


def build_native_loader_dispatch_request(
    descriptor: Mapping[str, Any],
    runtime_package_root: str | os.PathLike[str],
    input_values: Mapping[str, Any] | Sequence[Any],
    output_values: Mapping[str, Any] | Sequence[Any],
    dispatch_geometry: RuntimeDispatchGeometry | Mapping[str, Any] | Sequence[int],
    specialization_values: Mapping[int, Any] | None = None,
    *,
    expected_target: str | None = None,
) -> RuntimeExecutionRequest:
    """Build and preflight one DirectX or OpenGL native runtime request.

    The descriptor remains the source of truth for artifact identity, resource
    coordinates, entry-point metadata, and specialization identities. Callers
    provide values only under the exact reflected binding names and numeric
    specialization ids.
    """

    normalized = _validated_descriptor(descriptor)
    target = _validated_target(normalized, expected_target=expected_target)
    entry_point = _validated_entry_point(normalized, target=target)
    artifact_path, package_root = _verified_artifact(
        normalized["artifact"], runtime_package_root
    )

    inputs = _typed_values(input_values, role="input")
    outputs = _typed_values(output_values, role="output")
    _validate_value_names(inputs, outputs, normalized["bindings"])
    resource_bindings = _resource_bindings(
        normalized["bindings"],
        target=target,
        inputs=inputs,
        outputs=outputs,
    )
    constants = _specialization_constants(
        normalized["specializationConstants"],
        specialization_values,
        target=target,
    )
    dispatch = _dispatch_geometry(
        dispatch_geometry,
        entry_point=entry_point["name"],
        reflected_workgroup_size=_reflected_workgroup_size(entry_point),
    )

    runtime_entry_point = RuntimeEntryPoint(
        name=entry_point["name"],
        stage=_COMPUTE_STAGE,
        execution_config=copy.deepcopy(entry_point["executionConfig"]),
        workgroup_size=dispatch.workgroup_size,
        metadata={
            "provenance": copy.deepcopy(entry_point["provenance"]),
            "nativeLoaderUnitId": normalized["unitId"],
        },
    )
    contract = RuntimeAdapterContract(
        contract_id=(
            f"{normalized['unitId']}:native-loader-abi-v{normalized['abiVersion']}"
        ),
        entry_points=(runtime_entry_point,),
        resource_bindings=tuple(resource_bindings),
        specialization_constants=tuple(constants),
        dispatch=dispatch,
        metadata={
            "nativeLoaderABI": {
                "kind": normalized["kind"],
                "abiVersion": normalized["abiVersion"],
                "unitId": normalized["unitId"],
            },
            "scalarLayout": copy.deepcopy(normalized["scalarLayout"]),
            "provenance": copy.deepcopy(normalized["provenance"]),
        },
    )
    artifact = _runtime_artifact(normalized)
    fixture = RuntimeFixture(
        id=f"{normalized['unitId']}:native-loader-dispatch",
        selector=RuntimeArtifactSelector(
            target=target,
            stage=_COMPUTE_STAGE,
            path=normalized["artifact"]["packagePath"],
            artifact_id=normalized["unitId"],
        ),
        entry_point=entry_point["name"],
        inputs=tuple(
            inputs[name] for name in _binding_value_order(resource_bindings, inputs)
        ),
        expected_outputs=tuple(
            outputs[name] for name in _binding_value_order(resource_bindings, outputs)
        ),
        adapter_contract=contract,
        metadata={
            "nativeLoaderUnitId": normalized["unitId"],
            "artifactHash": copy.deepcopy(normalized["artifact"]["hash"]),
        },
    )
    request = RuntimeExecutionRequest(
        fixture=fixture,
        artifact=artifact,
        artifact_path=artifact_path,
        project_root=package_root,
        adapter_contract=contract,
    )
    plan = prepare_runtime_execution(request)
    setup_errors = [
        copy.deepcopy(dict(diagnostic))
        for diagnostic in plan.diagnostics
        if isinstance(diagnostic, Mapping) and diagnostic.get("severity") == "error"
    ]
    if setup_errors:
        raise NativeLoaderDispatchError(
            "execution-plan-invalid",
            "Native loader values produced runtime execution setup errors.",
            details={"diagnostics": setup_errors},
        )
    return replace(request, execution_plan=plan)


def _validated_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return _validate_descriptor(descriptor)
    except NativeLoaderABIError as exc:
        raise NativeLoaderDispatchError(
            "descriptor-invalid",
            "Native loader ABI descriptor validation failed.",
            path=exc.path,
            details={"abiDiagnostic": exc.to_json()},
        ) from exc


def _validated_target(
    descriptor: Mapping[str, Any], *, expected_target: str | None
) -> str:
    target = descriptor["target"]
    if target not in _TARGET_ARTIFACT_FORMATS:
        raise NativeLoaderDispatchError(
            "target-unsupported",
            "Native loader dispatch supports DirectX and OpenGL targets only.",
            path="$.target",
            details={
                "target": target,
                "supportedTargets": sorted(_TARGET_ARTIFACT_FORMATS),
            },
        )
    if expected_target is not None:
        if not isinstance(expected_target, str) or not expected_target.strip():
            raise NativeLoaderDispatchError(
                "expected-target-invalid",
                "Expected target must be a non-empty string.",
                path="$.expectedTarget",
                details={"value": expected_target},
            )
        expected = expected_target.strip().lower()
        if expected not in _TARGET_ARTIFACT_FORMATS:
            raise NativeLoaderDispatchError(
                "expected-target-invalid",
                "Expected target must identify DirectX or OpenGL.",
                path="$.expectedTarget",
                details={"value": expected_target},
            )
        if expected != target:
            raise NativeLoaderDispatchError(
                "target-mismatch",
                "Native loader descriptor target does not match the expected target.",
                path="$.target",
                details={"expectedTarget": expected, "actualTarget": target},
            )

    expected_format = _TARGET_ARTIFACT_FORMATS[target]
    actual_format = descriptor["artifact"]["format"]
    if actual_format != expected_format:
        raise NativeLoaderDispatchError(
            "artifact-format-unsupported",
            f"{target} native dispatch requires {expected_format} artifacts.",
            path="$.artifact.format",
            details={
                "target": target,
                "expectedFormat": expected_format,
                "actualFormat": actual_format,
            },
        )
    if descriptor["stage"] != _COMPUTE_STAGE:
        raise NativeLoaderDispatchError(
            "stage-unsupported",
            "Native loader dispatch supports compute-stage descriptors only.",
            path="$.stage",
            details={"stage": descriptor["stage"]},
        )
    return target


def _validated_entry_point(
    descriptor: Mapping[str, Any], *, target: str
) -> Mapping[str, Any]:
    entry_point = descriptor["entryPoint"]
    name = entry_point["name"]
    if not _ENTRY_POINT_RE.fullmatch(name):
        raise NativeLoaderDispatchError(
            "entry-point-invalid",
            "Native loader entry point must be a portable shader identifier.",
            path="$.entryPoint.name",
            details={"entryPoint": name},
        )
    if target == "opengl" and name != "main":
        raise NativeLoaderDispatchError(
            "entry-point-unsupported",
            "OpenGL compute source dispatch requires the selected entry point to be main.",
            path="$.entryPoint.name",
            details={"target": target, "entryPoint": name},
        )
    return entry_point


def _verified_artifact(
    artifact: Mapping[str, Any], runtime_package_root: str | os.PathLike[str]
) -> tuple[Path, Path]:
    root = _resolved_package_root(runtime_package_root)
    package_path = artifact["packagePath"]
    relative = _safe_package_path(package_path)
    unresolved = root.joinpath(*relative.parts)
    try:
        resolved = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NativeLoaderDispatchError(
            "artifact-unavailable",
            "Native loader artifact does not resolve to an available package file.",
            path="$.artifact.packagePath",
            details={"packagePath": package_path, "error": str(exc)},
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise NativeLoaderDispatchError(
            "artifact-path-escape",
            "Native loader artifact resolves outside the runtime package root.",
            path="$.artifact.packagePath",
            details={"packagePath": package_path},
        ) from exc
    if not resolved.is_file():
        raise NativeLoaderDispatchError(
            "artifact-not-file",
            "Native loader artifact must resolve to a regular file.",
            path="$.artifact.packagePath",
            details={"packagePath": package_path},
        )

    digest = hashlib.sha256()
    actual_size = 0
    try:
        with resolved.open("rb") as artifact_file:
            while chunk := artifact_file.read(1024 * 1024):
                actual_size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise NativeLoaderDispatchError(
            "artifact-read-failed",
            "Native loader artifact could not be read for identity verification.",
            path="$.artifact.packagePath",
            details={"packagePath": package_path, "error": str(exc)},
        ) from exc

    expected_size = artifact["sizeBytes"]
    if actual_size != expected_size:
        raise NativeLoaderDispatchError(
            "artifact-size-mismatch",
            "Native loader artifact byte size does not match the descriptor.",
            path="$.artifact.sizeBytes",
            details={
                "expectedSizeBytes": expected_size,
                "actualSizeBytes": actual_size,
                "packagePath": package_path,
            },
        )
    actual_hash = digest.hexdigest()
    expected_hash = artifact["hash"]["value"]
    if actual_hash != expected_hash:
        raise NativeLoaderDispatchError(
            "artifact-hash-mismatch",
            "Native loader artifact SHA-256 does not match the descriptor.",
            path="$.artifact.hash.value",
            details={
                "expectedSHA256": expected_hash,
                "actualSHA256": actual_hash,
                "packagePath": package_path,
            },
        )
    return resolved, root


def _resolved_package_root(value: str | os.PathLike[str]) -> Path:
    try:
        root = Path(os.fspath(value))
    except (TypeError, ValueError) as exc:
        raise NativeLoaderDispatchError(
            "package-root-invalid",
            "Runtime package root must be a filesystem path.",
            path="$.runtimePackageRoot",
        ) from exc
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NativeLoaderDispatchError(
            "package-root-unavailable",
            "Runtime package root does not resolve to an available directory.",
            path="$.runtimePackageRoot",
            details={"packageRoot": str(root), "error": str(exc)},
        ) from exc
    if not resolved.is_dir():
        raise NativeLoaderDispatchError(
            "package-root-not-directory",
            "Runtime package root must resolve to a directory.",
            path="$.runtimePackageRoot",
            details={"packageRoot": str(root)},
        )
    return resolved


def _safe_package_path(value: str) -> PurePosixPath:
    if "\\" in value or "\x00" in value:
        raise NativeLoaderDispatchError(
            "artifact-path-invalid",
            "Native loader package paths must use portable relative path syntax.",
            path="$.artifact.packagePath",
            details={"packagePath": value},
        )
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise NativeLoaderDispatchError(
            "artifact-path-invalid",
            "Native loader package path must use normalized path components.",
            path="$.artifact.packagePath",
            details={"packagePath": value},
        )
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or value == "."
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise NativeLoaderDispatchError(
            "artifact-path-invalid",
            "Native loader package path must be a normalized relative path without traversal.",
            path="$.artifact.packagePath",
            details={"packagePath": value},
        )
    return posix_path


def _typed_values(
    values: Mapping[str, Any] | Sequence[Any], *, role: str
) -> dict[str, RuntimeValue]:
    path = "$.inputValues" if role == "input" else "$.outputValues"
    records: list[tuple[str | None, Any, str]] = []
    if isinstance(values, Mapping):
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise NativeLoaderDispatchError(
                    "value-name-invalid",
                    "Runtime value map keys must be non-empty binding names.",
                    path=path,
                    details={"name": key},
                )
            records.append((key, value, f"{path}.{key}"))
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        records.extend(
            (None, value, f"{path}[{index}]") for index, value in enumerate(values)
        )
    else:
        raise NativeLoaderDispatchError(
            "value-collection-invalid",
            "Runtime values must be a name map or a list of typed values.",
            path=path,
        )

    result: dict[str, RuntimeValue] = {}
    for key, value, value_path in records:
        runtime_value = _typed_value(value, key=key, role=role, path=value_path)
        if runtime_value.name in result:
            raise NativeLoaderDispatchError(
                "value-duplicate",
                "Runtime binding values must have unique names.",
                path=value_path,
                details={"name": runtime_value.name, "role": role},
            )
        result[runtime_value.name] = runtime_value
    return result


def _typed_value(value: Any, *, key: str | None, role: str, path: str) -> RuntimeValue:
    if isinstance(value, RuntimeValue):
        runtime_value = value
    elif isinstance(value, Mapping):
        unsupported = sorted(set(value) - _VALUE_FIELDS, key=str)
        if unsupported:
            raise NativeLoaderDispatchError(
                "value-schema-invalid",
                "Typed runtime value contains unsupported fields.",
                path=path,
                details={"unsupportedFields": unsupported},
            )
        if "values" in value and "value" in value:
            raise NativeLoaderDispatchError(
                "value-schema-invalid",
                "Typed runtime value cannot define both values and value.",
                path=path,
            )
        name = value.get("name", key)
        tolerance = _runtime_tolerance(value.get("tolerance"), path=f"{path}.tolerance")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise NativeLoaderDispatchError(
                "value-metadata-invalid",
                "Typed runtime value metadata must be an object.",
                path=f"{path}.metadata",
            )
        runtime_value = RuntimeValue(
            name=name,
            kind=value.get("kind", "buffer"),
            dtype=value.get("dtype"),
            shape=_value_shape(value.get("shape"), path=f"{path}.shape"),
            values=value.get("values", value.get("value")),
            tolerance=tolerance,
            metadata=copy.deepcopy(dict(metadata)),
        )
    else:
        raise NativeLoaderDispatchError(
            "value-invalid",
            "Runtime binding values must be RuntimeValue instances or typed objects.",
            path=path,
        )

    if not isinstance(runtime_value.name, str) or not runtime_value.name:
        raise NativeLoaderDispatchError(
            "value-name-invalid",
            "Typed runtime value requires a non-empty binding name.",
            path=f"{path}.name",
        )
    if key is not None and runtime_value.name != key:
        raise NativeLoaderDispatchError(
            "value-name-mismatch",
            "Typed runtime value name must match its map key exactly.",
            path=f"{path}.name",
            details={"mapKey": key, "valueName": runtime_value.name},
        )
    if runtime_value.kind != "buffer":
        raise NativeLoaderDispatchError(
            "value-kind-unsupported",
            "Native loader dispatch currently accepts buffer runtime values only.",
            path=f"{path}.kind",
            details={"name": runtime_value.name, "kind": runtime_value.kind},
        )
    metadata = runtime_value.metadata
    if not isinstance(metadata, Mapping):
        raise NativeLoaderDispatchError(
            "value-metadata-invalid",
            "Typed runtime value metadata must be an object.",
            path=f"{path}.metadata",
        )
    aliases = sorted(_ALIAS_METADATA_FIELDS.intersection(metadata))
    if aliases:
        raise NativeLoaderDispatchError(
            "value-alias-unsupported",
            "Native loader dispatch binds exact descriptor names and does not accept value aliases.",
            path=f"{path}.metadata",
            details={"aliasFields": aliases, "name": runtime_value.name},
        )
    dtype = _buffer_dtype(runtime_value.dtype, path=f"{path}.dtype")
    shape = _value_shape(runtime_value.shape, path=f"{path}.shape")
    if not shape:
        raise NativeLoaderDispatchError(
            "value-shape-invalid",
            "Typed runtime buffer shape must contain at least one positive dimension.",
            path=f"{path}.shape",
            details={"name": runtime_value.name},
        )
    if role == "input" and runtime_value.values is None:
        raise NativeLoaderDispatchError(
            "input-data-missing",
            "Input runtime buffers require typed values.",
            path=f"{path}.values",
            details={"name": runtime_value.name},
        )
    if runtime_value.values is not None:
        flattened_values = _flatten_values(runtime_value.values)
        actual_count = len(flattened_values)
        expected_count = math.prod(shape)
        if actual_count != expected_count:
            raise NativeLoaderDispatchError(
                "value-size-mismatch",
                "Runtime buffer value count does not match its declared shape.",
                path=f"{path}.values",
                details={
                    "name": runtime_value.name,
                    "expectedCount": expected_count,
                    "actualCount": actual_count,
                },
            )
        _validate_buffer_values(
            flattened_values,
            dtype=dtype,
            path=f"{path}.values",
            name=runtime_value.name,
        )
    return RuntimeValue(
        name=runtime_value.name,
        kind="buffer",
        dtype=dtype,
        shape=shape,
        values=runtime_value.values,
        tolerance=runtime_value.tolerance,
        metadata=copy.deepcopy(dict(metadata)),
    )


def _runtime_tolerance(value: Any, *, path: str) -> RuntimeTolerance | None:
    if value is None or isinstance(value, RuntimeTolerance):
        return value
    if not isinstance(value, Mapping):
        raise NativeLoaderDispatchError(
            "value-tolerance-invalid",
            "Runtime output tolerance must be an object.",
            path=path,
        )
    unsupported = sorted(set(value) - {"absolute", "relative"}, key=str)
    if unsupported:
        raise NativeLoaderDispatchError(
            "value-tolerance-invalid",
            "Runtime output tolerance contains unsupported fields.",
            path=path,
            details={"unsupportedFields": unsupported},
        )
    absolute = value.get("absolute", 0.0)
    relative = value.get("relative", 0.0)
    for name, item in (("absolute", absolute), ("relative", relative)):
        try:
            finite = math.isfinite(float(item))
        except (OverflowError, TypeError, ValueError):
            finite = False
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or item < 0
            or not finite
        ):
            raise NativeLoaderDispatchError(
                "value-tolerance-invalid",
                "Runtime output tolerance values must be finite and non-negative.",
                path=f"{path}.{name}",
                details={"value": item},
            )
    return RuntimeTolerance(absolute=float(absolute), relative=float(relative))


def _value_shape(value: Any, *, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise NativeLoaderDispatchError(
            "value-shape-invalid",
            "Runtime buffer shape must be a list of positive integers.",
            path=path,
        )
    shape = tuple(value)
    for index, item in enumerate(shape):
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise NativeLoaderDispatchError(
                "value-shape-invalid",
                "Runtime buffer shape dimensions must be positive integers.",
                path=f"{path}[{index}]",
                details={"value": item},
            )
    return shape


def _buffer_dtype(value: Any, *, path: str) -> str:
    normalized = re.sub(r"\s+", "", str(value or "")).lower()
    dtype = _BUFFER_DTYPE_ALIASES.get(normalized)
    if dtype is None:
        raise NativeLoaderDispatchError(
            "value-dtype-unsupported",
            "Native runtime buffers support float32, int32, and uint32 values only.",
            path=path,
            details={"dtype": value},
        )
    return dtype


def _flatten_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Sequence):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten_values(item))
        return result
    return [value]


def _validate_buffer_values(
    values: Sequence[Any], *, dtype: str, path: str, name: str
) -> None:
    for index, value in enumerate(values):
        valid = False
        if dtype == "int32":
            valid = (
                isinstance(value, int)
                and not isinstance(value, bool)
                and -(1 << 31) <= value < (1 << 31)
            )
        elif dtype == "uint32":
            valid = (
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= _UINT32_MAX
            )
        elif dtype == "float32":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            if valid:
                try:
                    numeric_value = float(value)
                    valid = math.isfinite(numeric_value)
                    if valid:
                        struct.pack("<f", numeric_value)
                except (OverflowError, struct.error, ValueError):
                    valid = False
        if not valid:
            raise NativeLoaderDispatchError(
                "value-data-invalid",
                "Runtime buffer value is incompatible with its declared dtype.",
                path=f"{path}[{index}]",
                details={"name": name, "dtype": dtype, "value": value},
            )


def _validate_value_names(
    inputs: Mapping[str, RuntimeValue],
    outputs: Mapping[str, RuntimeValue],
    bindings: Sequence[Mapping[str, Any]],
) -> None:
    overlap = sorted(set(inputs).intersection(outputs))
    if overlap:
        raise NativeLoaderDispatchError(
            "value-duplicate",
            "A binding cannot be provided as both an input and an output.",
            path="$.outputValues",
            details={"bindings": overlap},
        )
    binding_by_name = {binding["name"]: binding for binding in bindings}
    read_only_outputs = sorted(
        name
        for name in outputs
        if name in binding_by_name and binding_by_name[name].get("access") == "read"
    )
    if read_only_outputs:
        raise NativeLoaderDispatchError(
            "read-only-output",
            "Read-only descriptor bindings cannot be allocated as runtime outputs.",
            path="$.outputValues",
            details={"bindings": read_only_outputs},
        )

    read_write_inputs = sorted(
        name
        for name in inputs
        if name in binding_by_name
        and binding_by_name[name].get("access") == "read_write"
    )
    if read_write_inputs:
        raise NativeLoaderDispatchError(
            "read-write-input-unsupported",
            "Read-write descriptor bindings cannot currently combine input initialization and output readback.",
            path="$.inputValues",
            details={"bindings": read_write_inputs},
        )

    expected_inputs = {
        binding["name"] for binding in bindings if binding.get("access") == "read"
    }
    expected_outputs = {
        binding["name"]
        for binding in bindings
        if binding.get("access") in {"write", "read_write"}
    }
    extra_inputs = sorted(set(inputs) - expected_inputs)
    extra_outputs = sorted(set(outputs) - expected_outputs)
    if extra_inputs or extra_outputs:
        raise NativeLoaderDispatchError(
            "value-extra",
            "Runtime values contain names not accepted by the descriptor binding contract.",
            path="$.inputValues",
            details={
                "extraInputs": extra_inputs,
                "extraOutputs": extra_outputs,
            },
        )
    missing_inputs = sorted(expected_inputs - set(inputs))
    missing_outputs = sorted(expected_outputs - set(outputs))
    if missing_inputs or missing_outputs:
        raise NativeLoaderDispatchError(
            "value-missing",
            "Runtime values do not cover every required descriptor binding.",
            details={
                "missingInputs": missing_inputs,
                "missingOutputs": missing_outputs,
            },
        )


def _resource_bindings(
    bindings: Sequence[Mapping[str, Any]],
    *,
    target: str,
    inputs: Mapping[str, RuntimeValue],
    outputs: Mapping[str, RuntimeValue],
) -> list[RuntimeResourceBinding]:
    if not bindings:
        raise NativeLoaderDispatchError(
            "resource-unsupported",
            "Native loader dispatch requires at least one reflected buffer resource.",
            path="$.bindings",
        )
    result = []
    for index, binding in enumerate(bindings):
        path = f"$.bindings[{index}]"
        name = binding["name"]
        raw_kind = str(binding["kind"]).strip().lower().replace("_", "-")
        kind = _BUFFER_KIND_ALIASES.get(raw_kind)
        access = binding.get("access")
        if kind is None or access not in {"read", "write", "read_write"}:
            raise NativeLoaderDispatchError(
                "resource-unsupported",
                "Native loader dispatch currently supports reflected buffer resources with explicit access.",
                path=path,
                details={
                    "target": target,
                    "name": name,
                    "kind": binding["kind"],
                    "access": access,
                },
            )
        expected_namespace = _TARGET_RESOURCE_NAMESPACES[target].get((kind, access))
        if expected_namespace is None or binding["namespace"] != expected_namespace:
            raise NativeLoaderDispatchError(
                "resource-namespace-unsupported",
                "Descriptor resource namespace is not executable by the selected runtime adapter.",
                path=f"{path}.namespace",
                details={
                    "target": target,
                    "name": name,
                    "kind": binding["kind"],
                    "access": access,
                    "expectedNamespace": expected_namespace,
                    "actualNamespace": binding["namespace"],
                },
            )
        coordinates = binding["coordinates"]
        if coordinates["set"] != 0:
            raise NativeLoaderDispatchError(
                "resource-set-unsupported",
                "DirectX and OpenGL native runtime adapters currently require resource set zero.",
                path=f"{path}.coordinates.set",
                details={"target": target, "name": name, "set": coordinates["set"]},
            )
        value = inputs.get(name) if access == "read" else outputs.get(name)
        if value is None:
            raise NativeLoaderDispatchError(
                "value-missing",
                "Runtime value is missing for a required descriptor binding.",
                path=path,
                details={"name": name},
            )
        scalar_layout = _validated_scalar_layout(
            binding.get("scalarLayout"),
            runtime_value=value,
            path=f"{path}.scalarLayout",
        )
        result.append(
            RuntimeResourceBinding(
                binding_id=name,
                name=name,
                kind=kind,
                type_name=binding["type"],
                set=coordinates["set"],
                binding=coordinates["binding"],
                access=access,
                value=name,
                metadata={
                    "required": True,
                    "bindingNamespace": binding["namespace"],
                    "namespace": binding["namespace"],
                    "coordinates": copy.deepcopy(coordinates),
                    "scalarLayout": scalar_layout,
                    "byteStride": scalar_layout["elementStrideBytes"],
                    "provenance": copy.deepcopy(binding["provenance"]),
                },
            )
        )
    return result


def _validated_scalar_layout(
    layout: Any, *, runtime_value: RuntimeValue, path: str
) -> dict[str, Any]:
    if not isinstance(layout, Mapping):
        raise NativeLoaderDispatchError(
            "resource-layout-unsupported",
            "Native runtime buffer bindings require a concrete scalar layout.",
            path=path,
            details={"binding": runtime_value.name},
        )
    element_type = _buffer_dtype(layout.get("elementType"), path=f"{path}.elementType")
    element_size = layout.get("elementSizeBytes")
    element_stride = layout.get("elementStrideBytes")
    if (
        not isinstance(element_size, int)
        or isinstance(element_size, bool)
        or element_size <= 0
        or not isinstance(element_stride, int)
        or isinstance(element_stride, bool)
        or element_stride <= 0
    ):
        raise NativeLoaderDispatchError(
            "resource-layout-invalid",
            "Scalar layout element size and stride must be positive integers.",
            path=path,
            details={
                "binding": runtime_value.name,
                "elementSizeBytes": element_size,
                "elementStrideBytes": element_stride,
            },
        )
    if (
        element_type != runtime_value.dtype
        or element_size != _DTYPE_SIZES[runtime_value.dtype]
    ):
        raise NativeLoaderDispatchError(
            "resource-layout-mismatch",
            "Runtime value dtype is incompatible with the descriptor scalar layout.",
            path=path,
            details={
                "binding": runtime_value.name,
                "valueDtype": runtime_value.dtype,
                "layoutElementType": element_type,
                "layoutElementSizeBytes": element_size,
            },
        )
    if element_stride != element_size:
        raise NativeLoaderDispatchError(
            "resource-layout-unsupported",
            "Native loader dispatch requires tightly packed scalar buffer layouts.",
            path=f"{path}.elementStrideBytes",
            details={
                "binding": runtime_value.name,
                "elementSizeBytes": element_size,
                "elementStrideBytes": element_stride,
            },
        )
    return copy.deepcopy(dict(layout))


def _specialization_constants(
    constants: Sequence[Mapping[str, Any]],
    values: Mapping[int, Any] | None,
    *,
    target: str,
) -> list[RuntimeSpecializationConstant]:
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise NativeLoaderDispatchError(
            "specialization-values-invalid",
            "Specialization values must be an object keyed by numeric id.",
            path="$.specializationValues",
        )
    explicit_values: dict[int, Any] = {}
    for constant_id, value in values.items():
        if (
            not isinstance(constant_id, int)
            or isinstance(constant_id, bool)
            or not 0 <= constant_id <= _UINT32_MAX
        ):
            raise NativeLoaderDispatchError(
                "specialization-id-invalid",
                "Specialization value keys must be uint32 numeric ids.",
                path="$.specializationValues",
                details={"id": constant_id},
            )
        explicit_values[constant_id] = value

    by_id: dict[int, tuple[int, Mapping[str, Any]]] = {}
    runtime_keys: dict[str, int] = {}
    for index, constant in enumerate(constants):
        path = f"$.specializationConstants[{index}]"
        if not isinstance(constant, Mapping):
            raise NativeLoaderDispatchError(
                "specialization-invalid",
                "Specialization constant descriptors must be objects.",
                path=path,
            )
        constant_id = constant.get("id", constant.get("constantId"))
        if (
            not isinstance(constant_id, int)
            or isinstance(constant_id, bool)
            or not 0 <= constant_id <= _UINT32_MAX
        ):
            raise NativeLoaderDispatchError(
                "specialization-id-missing",
                "Native dispatch specialization constants require stable uint32 ids.",
                path=f"{path}.id",
                details={"id": constant_id},
            )
        if constant_id in by_id:
            previous_index = by_id[constant_id][0]
            raise NativeLoaderDispatchError(
                "specialization-id-ambiguous",
                "Specialization constant id is declared more than once.",
                path=f"{path}.id",
                details={"id": constant_id, "indices": [previous_index, index]},
            )
        name = constant.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise NativeLoaderDispatchError(
                "specialization-name-invalid",
                "Specialization constant name must be a non-empty string or null.",
                path=f"{path}.name",
                details={"id": constant_id, "name": name},
            )
        runtime_key = name if name is not None else str(constant_id)
        if runtime_key in runtime_keys:
            raise NativeLoaderDispatchError(
                "specialization-name-ambiguous",
                "Specialization constants resolve to the same native runtime key.",
                path=f"{path}.name",
                details={
                    "runtimeKey": runtime_key,
                    "indices": [runtime_keys[runtime_key], index],
                },
            )
        runtime_keys[runtime_key] = index
        by_id[constant_id] = (index, constant)

    extra = sorted(set(explicit_values) - set(by_id))
    if extra:
        raise NativeLoaderDispatchError(
            "specialization-value-extra",
            "Specialization values contain ids absent from the descriptor.",
            path="$.specializationValues",
            details={"ids": extra},
        )

    result = []
    for constant_id, (index, constant) in sorted(by_id.items()):
        path = f"$.specializationConstants[{index}]"
        required = constant.get("required", False)
        if not isinstance(required, bool):
            raise NativeLoaderDispatchError(
                "specialization-required-invalid",
                "Specialization constant required must be a boolean.",
                path=f"{path}.required",
                details={"id": constant_id, "value": required},
            )
        dtype = _specialization_dtype(constant.get("dtype"), path=f"{path}.dtype")
        has_descriptor_value = "value" in constant and constant.get("value") is not None
        has_default = "default" in constant and constant.get("default") is not None
        if has_descriptor_value:
            _validate_specialization_value(
                constant["value"], dtype=dtype, path=f"{path}.value"
            )
        if has_default:
            _validate_specialization_value(
                constant["default"], dtype=dtype, path=f"{path}.default"
            )
        has_explicit = constant_id in explicit_values
        if has_explicit:
            _validate_specialization_value(
                explicit_values[constant_id],
                dtype=dtype,
                path=f"$.specializationValues[{constant_id}]",
            )
        if required and not (has_explicit or has_descriptor_value or has_default):
            raise NativeLoaderDispatchError(
                "specialization-value-missing",
                "Required specialization constant has no explicit, descriptor, or default value.",
                path=f"$.specializationValues[{constant_id}]",
                details={"id": constant_id, "name": constant.get("name")},
            )

        if target == "directx":
            if not has_descriptor_value:
                raise NativeLoaderDispatchError(
                    "specialization-unmaterialized",
                    "DirectX dispatch requires specialization constants to be materialized in HLSL source.",
                    path=path,
                    details={"id": constant_id, "name": constant.get("name")},
                )
            compiled_value = constant["value"]
            if has_explicit and explicit_values[constant_id] != compiled_value:
                raise NativeLoaderDispatchError(
                    "specialization-override-unsupported",
                    "DirectX dispatch cannot override a specialization value materialized in HLSL source.",
                    path=f"$.specializationValues[{constant_id}]",
                    details={
                        "id": constant_id,
                        "materializedValue": compiled_value,
                        "explicitValue": explicit_values[constant_id],
                    },
                )

        if has_explicit:
            resolved_value = explicit_values[constant_id]
            source = "explicit"
        elif has_descriptor_value:
            resolved_value = constant["value"]
            source = "descriptor"
        elif has_default:
            resolved_value = None
            source = "default"
        else:
            resolved_value = None
            source = "unbound"
        provenance = {
            "source": source,
            "constantId": constant_id,
            "descriptorProvenance": copy.deepcopy(constant.get("provenance", {})),
        }
        if has_descriptor_value:
            provenance["descriptorValue"] = copy.deepcopy(constant["value"])
        if has_default:
            provenance["defaultValue"] = copy.deepcopy(constant["default"])
        metadata = {
            "descriptor": copy.deepcopy(dict(constant)),
            "descriptorProvenance": copy.deepcopy(constant.get("provenance", {})),
        }
        kind = "specialization-constant"
        if target == "directx":
            kind = "compile-time-constant"
            metadata["mechanism"] = "compiled"
        result.append(
            RuntimeSpecializationConstant(
                name=constant.get("name"),
                constant_id=constant_id,
                kind=kind,
                dtype=dtype,
                source_type=constant.get("sourceType"),
                value=resolved_value,
                default=constant.get("default") if has_default else None,
                required=required,
                overridden=has_explicit,
                override_status=source,
                status=(
                    "bound"
                    if source in {"explicit", "descriptor"}
                    else "defaulted" if source == "default" else "unbound"
                ),
                value_provenance=provenance,
                metadata=metadata,
            )
        )
    return result


def _specialization_dtype(value: Any, *, path: str) -> str:
    normalized = re.sub(r"\s+", "", str(value or "")).lower()
    dtype = _SPECIALIZATION_DTYPE_ALIASES.get(normalized)
    if dtype is None:
        raise NativeLoaderDispatchError(
            "specialization-dtype-unsupported",
            "Specialization constants require bool or 32-bit scalar dtypes.",
            path=path,
            details={"dtype": value},
        )
    return dtype


def _validate_specialization_value(value: Any, *, dtype: str, path: str) -> None:
    valid = False
    if dtype == "bool":
        valid = isinstance(value, bool)
    elif dtype == "int32":
        valid = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and -(1 << 31) <= value < (1 << 31)
        )
    elif dtype == "uint32":
        valid = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= _UINT32_MAX
        )
    elif dtype == "float32":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                numeric_value = float(value)
                valid = math.isfinite(numeric_value)
                if valid:
                    struct.pack("<f", numeric_value)
            except (OverflowError, struct.error, TypeError, ValueError):
                valid = False
    if not valid:
        raise NativeLoaderDispatchError(
            "specialization-value-invalid",
            "Specialization value is incompatible with its declared dtype.",
            path=path,
            details={"dtype": dtype, "value": value},
        )


def _reflected_workgroup_size(entry_point: Mapping[str, Any]) -> tuple[int, ...]:
    execution_config = entry_point["executionConfig"]
    candidates: list[tuple[str, tuple[int, ...]]] = []
    for key in (
        "workgroupSize",
        "workgroup_size",
        "numthreads",
        "localSize",
        "local_size",
    ):
        if key not in execution_config or execution_config[key] is None:
            continue
        candidates.append(
            (
                key,
                _positive_dimensions(
                    execution_config[key],
                    path=f"$.entryPoint.executionConfig.{key}",
                    label="workgroup size",
                ),
            )
        )
    component_keys = ("local_size_x", "local_size_y", "local_size_z")
    if any(key in execution_config for key in component_keys):
        candidates.append(
            (
                "local_size_x/y/z",
                _positive_dimensions(
                    [execution_config.get(key, 1) for key in component_keys],
                    path="$.entryPoint.executionConfig.local_size_x/y/z",
                    label="workgroup size",
                ),
            )
        )
    if not candidates:
        return ()
    first_name, first = candidates[0]
    for candidate_name, candidate in candidates[1:]:
        if _padded_dimensions(first) != _padded_dimensions(candidate):
            raise NativeLoaderDispatchError(
                "workgroup-size-ambiguous",
                "Reflected entry-point workgroup sizes disagree.",
                path="$.entryPoint.executionConfig",
                details={
                    "firstField": first_name,
                    "firstValue": list(first),
                    "conflictingField": candidate_name,
                    "conflictingValue": list(candidate),
                },
            )
    return first


def _dispatch_geometry(
    value: RuntimeDispatchGeometry | Mapping[str, Any] | Sequence[int],
    *,
    entry_point: str,
    reflected_workgroup_size: tuple[int, ...],
) -> RuntimeDispatchGeometry:
    path = "$.dispatchGeometry"
    if isinstance(value, RuntimeDispatchGeometry):
        dispatch_entry_point = value.entry_point
        workgroup_count = value.workgroup_count
        supplied_workgroup_size = value.workgroup_size
        global_size = value.global_size
        grid_size = value.grid_size
        metadata = value.metadata
    elif isinstance(value, Mapping):
        allowed = {
            "entryPoint",
            "workgroupCount",
            "workgroupSize",
            "globalSize",
            "gridSize",
            "metadata",
        }
        unsupported = sorted(set(value) - allowed, key=str)
        if unsupported:
            raise NativeLoaderDispatchError(
                "dispatch-schema-invalid",
                "Dispatch geometry contains unsupported fields.",
                path=path,
                details={"unsupportedFields": unsupported},
            )
        dispatch_entry_point = value.get("entryPoint")
        workgroup_count = value.get("workgroupCount")
        supplied_workgroup_size = value.get("workgroupSize", ())
        global_size = value.get("globalSize", ())
        grid_size = value.get("gridSize", ())
        metadata = value.get("metadata", {})
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        dispatch_entry_point = None
        workgroup_count = value
        supplied_workgroup_size = ()
        global_size = ()
        grid_size = ()
        metadata = {}
    else:
        raise NativeLoaderDispatchError(
            "dispatch-invalid",
            "Dispatch geometry must provide one to three positive workgroup counts.",
            path=path,
        )
    if dispatch_entry_point is not None and dispatch_entry_point != entry_point:
        raise NativeLoaderDispatchError(
            "entry-point-mismatch",
            "Dispatch entry point does not match the native loader descriptor.",
            path=f"{path}.entryPoint",
            details={
                "expectedEntryPoint": entry_point,
                "actualEntryPoint": dispatch_entry_point,
            },
        )
    if not isinstance(metadata, Mapping):
        raise NativeLoaderDispatchError(
            "dispatch-metadata-invalid",
            "Dispatch geometry metadata must be an object.",
            path=f"{path}.metadata",
        )
    counts = _positive_dimensions(
        workgroup_count, path=f"{path}.workgroupCount", label="workgroup count"
    )
    supplied = ()
    if supplied_workgroup_size:
        supplied = _positive_dimensions(
            supplied_workgroup_size,
            path=f"{path}.workgroupSize",
            label="workgroup size",
        )
    if (
        reflected_workgroup_size
        and supplied
        and _padded_dimensions(reflected_workgroup_size) != _padded_dimensions(supplied)
    ):
        raise NativeLoaderDispatchError(
            "workgroup-size-mismatch",
            "Dispatch workgroup size does not match reflected entry-point metadata.",
            path=f"{path}.workgroupSize",
            details={
                "reflectedWorkgroupSize": list(reflected_workgroup_size),
                "dispatchWorkgroupSize": list(supplied),
            },
        )
    workgroup_size = reflected_workgroup_size or supplied
    derived_global_size: tuple[int, ...] = ()
    if workgroup_size:
        padded_counts = _padded_dimensions(counts)
        padded_size = _padded_dimensions(workgroup_size)
        derived_global_size = tuple(
            count * size for count, size in zip(padded_counts, padded_size)
        )
    for field_name, candidate in (("globalSize", global_size), ("gridSize", grid_size)):
        if not candidate:
            continue
        dimensions = _positive_dimensions(
            candidate, path=f"{path}.{field_name}", label=field_name
        )
        if derived_global_size and _padded_dimensions(dimensions) != _padded_dimensions(
            derived_global_size
        ):
            raise NativeLoaderDispatchError(
                "dispatch-size-mismatch",
                "Dispatch global dimensions conflict with workgroup count and size.",
                path=f"{path}.{field_name}",
                details={
                    "expectedSize": list(derived_global_size),
                    "actualSize": list(dimensions),
                },
            )
    return RuntimeDispatchGeometry(
        entry_point=entry_point,
        workgroup_size=workgroup_size,
        workgroup_count=counts,
        global_size=derived_global_size,
        grid_size=derived_global_size,
        metadata={
            **copy.deepcopy(dict(metadata)),
            "source": "native-loader-dispatch",
            "workgroupSizeSource": (
                "reflection"
                if reflected_workgroup_size
                else "dispatch" if supplied else "unavailable"
            ),
        },
    )


def _positive_dimensions(value: Any, *, path: str, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise NativeLoaderDispatchError(
            "dispatch-dimensions-invalid",
            f"Dispatch {label} must be a list of one to three positive integers.",
            path=path,
        )
    dimensions = tuple(value)
    if not 1 <= len(dimensions) <= 3:
        raise NativeLoaderDispatchError(
            "dispatch-dimensions-invalid",
            f"Dispatch {label} must contain one to three dimensions.",
            path=path,
            details={"dimensionCount": len(dimensions)},
        )
    for index, item in enumerate(dimensions):
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise NativeLoaderDispatchError(
                "dispatch-dimensions-invalid",
                f"Dispatch {label} dimensions must be positive integers.",
                path=f"{path}[{index}]",
                details={"value": item},
            )
    return dimensions


def _padded_dimensions(value: Sequence[int]) -> tuple[int, int, int]:
    padded = tuple(value) + (1,) * (3 - len(value))
    return padded[:3]


def _runtime_artifact(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    artifact = descriptor["artifact"]
    source = descriptor["source"]
    return {
        "id": descriptor["unitId"],
        "source": source.get("path"),
        "sourceBackend": source.get("backend"),
        "path": artifact["packagePath"],
        "packagePath": artifact["packagePath"],
        "target": descriptor["target"],
        "stage": descriptor["stage"],
        "status": "translated",
        "entryPoint": descriptor["entryPoint"]["name"],
        "artifactFormat": artifact["format"],
        "hash": copy.deepcopy(artifact["hash"]),
        "sizeBytes": artifact["sizeBytes"],
        "provenance": copy.deepcopy(descriptor["provenance"]),
        "nativeLoaderABI": {
            "kind": descriptor["kind"],
            "abiVersion": descriptor["abiVersion"],
            "unitId": descriptor["unitId"],
        },
    }


def _binding_value_order(
    bindings: Sequence[RuntimeResourceBinding], values: Mapping[str, RuntimeValue]
) -> list[str]:
    return [binding.name for binding in bindings if binding.name in values]
