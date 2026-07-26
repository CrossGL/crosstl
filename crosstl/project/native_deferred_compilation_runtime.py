"""Compile and execute bounded native deferred variants."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .directx_toolchain import (
    dxc_compiler_arguments_for_source,
    dxc_profile_for_source,
)
from .host_reflection import reflect_target_host_interface
from .native_deferred_compilation import (
    validate_native_deferred_compilation_request,
)
from .native_deferred_compilation_cache import (
    lookup_native_deferred_compilation_cache,
    publish_native_deferred_compilation_cache,
)
from .native_deferred_compilation_package import (
    materialize_native_deferred_compilation_inputs,
)
from .native_loader_abi import (
    NativeLoaderABIError,
    _binding_descriptors,
    _validate_descriptor,
)
from .native_loader_dispatch import build_native_loader_dispatch_request
from .runtime_verification import (
    RuntimeDispatchGeometry,
    RuntimeExecutionRequest,
    RuntimeExecutorAvailability,
    RuntimeExecutorResult,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
    native_runtime_parity_adapter,
)

NATIVE_DEFERRED_COMPILATION_RESULT_KIND = "crosstl-native-deferred-compilation-result"
NATIVE_DEFERRED_COMPILATION_RESULT_VERSION = 1

_ERROR_PREFIX = "project.native-deferred-compilation-runtime"
_DXIL_FORMAT = "DXIL binary"
_SPIRV_FORMAT = "SPIR-V binary"
_OUTPUT_SUFFIXES = {
    _DXIL_FORMAT: ".dxil",
    _SPIRV_FORMAT: ".spv",
}
_TOOLCHAINS = {
    "directx": ("dxc", ("--version",)),
    "opengl": ("glslangValidator", ("--version",)),
}
_EXECUTION_WORKGROUP_FIELDS = (
    "workgroupSize",
    "workgroup_size",
    "numthreads",
    "localSize",
    "local_size",
)
_EXECUTION_SUBGROUP_FIELDS = (
    "subgroupWidth",
    "subgroup_width",
    "requiredSubgroupWidth",
    "required_subgroup_width",
    "waveSize",
)


class NativeDeferredCompilationRuntimeError(ValueError):
    """A bounded deferred variant cannot be compiled or executed safely."""

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


@dataclass(frozen=True)
class _Toolchain:
    name: str
    executable: Path
    version: str
    executable_hash: Mapping[str, str]
    probe_command: tuple[str, ...]
    probe_result: Mapping[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "executableHash": copy.deepcopy(dict(self.executable_hash)),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            **self.identity,
            "resolvedExecutable": str(self.executable),
            "probe": {
                "command": list(self.probe_command),
                **copy.deepcopy(dict(self.probe_result)),
            },
        }


def compile_native_deferred_compilation_request(
    request: Mapping[str, Any],
    package_root: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    *,
    workspace_root: str | os.PathLike[str] | None = None,
    command_runner: Callable[[Sequence[str]], Any] | None = None,
    tool_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Compile one verified bounded request or reuse its exact cache entry."""

    normalized = validate_native_deferred_compilation_request(request)
    runner = command_runner or _run_command
    resolver = tool_resolver or shutil.which
    workspace_parent = _workspace_parent(workspace_root)
    prefix = f"crosstl-deferred-{normalized['requestHash']['value'][:12]}-"

    with tempfile.TemporaryDirectory(
        prefix=prefix,
        dir=workspace_parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        materialized = materialize_native_deferred_compilation_inputs(
            normalized,
            package_root,
            temporary_root / "inputs",
        )
        descriptor = _load_and_validate_descriptor(
            Path(materialized["interface"]["path"])
        )
        interface = _verify_source_interface(
            normalized,
            descriptor,
            materialized,
        )
        toolchain = _resolve_toolchain(
            normalized,
            runner=runner,
            resolver=resolver,
        )
        cached = lookup_native_deferred_compilation_cache(
            cache_root,
            normalized,
            toolchain.identity,
        )
        compilation_command: tuple[str, ...] | None = None
        compilation_result: dict[str, Any]
        cache_status: str

        if cached is not None:
            cache_status = "hit"
            compilation_result = {
                "status": "cache-hit",
                "command": None,
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }
        else:
            output_format = normalized["target"]["outputFormat"]
            output_path = temporary_root / (
                "compiled" + _OUTPUT_SUFFIXES[output_format]
            )
            compilation_command = _compiler_command(
                normalized,
                materialized,
                toolchain,
                output_path=output_path,
            )
            raw_result = _invoke_command(
                runner,
                compilation_command,
                code="compiler-invocation-failed",
                message="Deferred native compiler could not be invoked.",
                path="$.compiler",
            )
            command_result = _command_result(compilation_command, raw_result)
            compilation_result = {
                "status": "compiled" if command_result["returncode"] == 0 else "failed",
                **command_result,
            }
            if command_result["returncode"] != 0:
                raise NativeDeferredCompilationRuntimeError(
                    "compiler-failed",
                    "Deferred native compilation failed.",
                    path="$.compiler",
                    details={
                        "target": normalized["target"]["backend"],
                        **command_result,
                    },
                )
            output_bytes = _read_compiler_output(
                output_path,
                output_format=output_format,
            )
            _verify_toolchain_unchanged(toolchain)
            cached = publish_native_deferred_compilation_cache(
                cache_root,
                normalized,
                toolchain.identity,
                output_bytes,
            )
            cache_status = "published"

        assert cached is not None
        if cache_status == "hit":
            _verify_toolchain_unchanged(toolchain)
        runtime_descriptor, runtime_package_root = _compiled_runtime_descriptor(
            descriptor,
            normalized,
            cached,
            toolchain,
        )
        output = cached["entry"]["output"]
        return {
            "schemaVersion": NATIVE_DEFERRED_COMPILATION_RESULT_VERSION,
            "kind": NATIVE_DEFERRED_COMPILATION_RESULT_KIND,
            "success": True,
            "requestHash": copy.deepcopy(normalized["requestHash"]),
            "request": copy.deepcopy(normalized),
            "target": copy.deepcopy(normalized["target"]),
            "variant": copy.deepcopy(normalized["variant"]),
            "toolchain": toolchain.to_json(),
            "interface": interface,
            "compiler": compilation_result,
            "cache": {
                "status": cache_status,
                "entryPath": cached["entryPath"],
                "outputPath": cached["outputPath"],
                "cacheKey": copy.deepcopy(cached["entry"]["cacheKey"]),
            },
            "output": {
                "path": cached["outputPath"],
                "format": output["format"],
                "sizeBytes": output["sizeBytes"],
                "hash": copy.deepcopy(output["hash"]),
            },
            "runtimePackageRoot": str(runtime_package_root),
            "runtimeDescriptor": runtime_descriptor,
            "provenance": {
                "source": copy.deepcopy(materialized["source"]),
                "includes": copy.deepcopy(materialized["includes"]),
                "includeDirectories": copy.deepcopy(materialized["includeDirectories"]),
                "compileCommand": (
                    list(compilation_command)
                    if compilation_command is not None
                    else None
                ),
            },
            "diagnostics": [],
        }


def build_native_deferred_compilation_dispatch_request(
    compilation: Mapping[str, Any],
    input_values: Mapping[str, Any] | Sequence[Any],
    output_values: Mapping[str, Any] | Sequence[Any],
    dispatch_geometry: RuntimeDispatchGeometry | Mapping[str, Any] | Sequence[int],
) -> RuntimeExecutionRequest:
    """Build one native runtime request from a successful compilation result."""

    result = _validated_compilation_result(compilation)
    specialization_values: dict[int, Any] = {}
    for index, constant in enumerate(result["variant"]["specializationValues"]):
        constant_id = constant.get("id")
        if constant_id is None:
            raise NativeDeferredCompilationRuntimeError(
                "specialization-id-missing",
                "Native deferred execution requires numeric specialization ids.",
                path=f"$.compilation.variant.specializationValues[{index}].id",
                details={"name": constant.get("name")},
            )
        specialization_values[constant_id] = copy.deepcopy(constant["value"])
    return build_native_loader_dispatch_request(
        result["runtimeDescriptor"],
        result["runtimePackageRoot"],
        input_values,
        output_values,
        dispatch_geometry,
        specialization_values,
        expected_target=result["target"]["backend"],
    )


def execute_native_deferred_compilation_request(
    request: Mapping[str, Any],
    package_root: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    input_values: Mapping[str, Any] | Sequence[Any],
    output_values: Mapping[str, Any] | Sequence[Any],
    dispatch_geometry: RuntimeDispatchGeometry | Mapping[str, Any] | Sequence[int],
    *,
    runtime_adapter: Any | None = None,
    workspace_root: str | os.PathLike[str] | None = None,
    command_runner: Callable[[Sequence[str]], Any] | None = None,
    tool_resolver: Callable[[str], str | None] | None = None,
) -> RuntimeExecutorResult:
    """Compile, bind, and dispatch one bounded native variant."""

    compilation = compile_native_deferred_compilation_request(
        request,
        package_root,
        cache_root,
        workspace_root=workspace_root,
        command_runner=command_runner,
        tool_resolver=tool_resolver,
    )
    dispatch_request = build_native_deferred_compilation_dispatch_request(
        compilation,
        input_values,
        output_values,
        dispatch_geometry,
    )
    adapter = runtime_adapter or native_runtime_parity_adapter(
        compilation["target"]["backend"]
    )
    executor = _deferred_runtime_executor(
        adapter,
        target=compilation["target"]["backend"],
    )
    availability = executor.is_available(dispatch_request)
    if not isinstance(availability, RuntimeExecutorAvailability):
        raise NativeDeferredCompilationRuntimeError(
            "runtime-availability-invalid",
            "Native runtime adapter returned an invalid availability result.",
            path="$.runtimeAdapter",
        )
    if not availability.available:
        raise NativeDeferredCompilationRuntimeError(
            "runtime-unavailable",
            "Native runtime adapter is unavailable for deferred execution.",
            path="$.runtimeAdapter",
            details={
                "target": compilation["target"]["backend"],
                "reason": availability.reason,
                "availability": copy.deepcopy(dict(availability.details)),
            },
        )
    result = executor.run(dispatch_request)
    if not isinstance(result, RuntimeExecutorResult):
        raise NativeDeferredCompilationRuntimeError(
            "runtime-result-invalid",
            "Native runtime adapter returned an invalid execution result.",
            path="$.runtimeAdapter",
        )
    return replace(
        result,
        details={
            **dict(result.details),
            "nativeDeferredCompilation": _runtime_compilation_report(compilation),
        },
    )


def _deferred_runtime_executor(adapter: Any, *, target: str) -> Any:
    if callable(getattr(adapter, "is_available", None)) and callable(
        getattr(adapter, "run", None)
    ):
        return adapter
    parity_methods = ("prepare_buffers", "dispatch", "collect_outputs")
    if all(callable(getattr(adapter, method, None)) for method in parity_methods):
        return RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"{target}-deferred-native",
                target=target,
                executor=target,
                adapter_kind=f"{target}-native-runtime",
            ),
            runtime_adapter=adapter,
        )
    raise NativeDeferredCompilationRuntimeError(
        "runtime-adapter-invalid",
        "Deferred execution requires a runtime executor or parity adapter.",
        path="$.runtimeAdapter",
        details={"target": target},
    )


def _workspace_parent(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    try:
        parent = Path(value)
        parent.mkdir(parents=True, exist_ok=True)
        resolved = parent.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise NativeDeferredCompilationRuntimeError(
            "workspace-invalid",
            "Deferred compilation workspace could not be prepared.",
            path="$.workspaceRoot",
            details={"workspaceRoot": str(value), "error": str(exc)},
        ) from exc
    if not resolved.is_dir():
        raise NativeDeferredCompilationRuntimeError(
            "workspace-invalid",
            "Deferred compilation workspace must be a directory.",
            path="$.workspaceRoot",
            details={"workspaceRoot": str(resolved)},
        )
    return str(resolved)


def _load_and_validate_descriptor(path: Path) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise NativeDeferredCompilationRuntimeError(
            "descriptor-read-failed",
            "Deferred compilation interface descriptor could not be read.",
            path="$.expectedLoaderDescriptor",
            details={"descriptorPath": str(path), "error": str(exc)},
        ) from exc
    try:
        decoded = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, TypeError) as exc:
        raise NativeDeferredCompilationRuntimeError(
            "descriptor-json-invalid",
            "Deferred compilation interface descriptor must be strict UTF-8 JSON.",
            path="$.expectedLoaderDescriptor",
            details={"descriptorPath": str(path), "error": str(exc)},
        ) from exc
    try:
        return _validate_descriptor(decoded)
    except NativeLoaderABIError as exc:
        raise NativeDeferredCompilationRuntimeError(
            "descriptor-invalid",
            "Deferred compilation interface descriptor failed ABI validation.",
            path=exc.path,
            details={"descriptorDiagnostic": exc.to_json()},
        ) from exc


def _verify_source_interface(
    request: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    materialized: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    target = request["target"]
    source = request["source"]
    expected_artifact = {
        "packagePath": source["path"],
        "format": source["format"],
        "hash": source["hash"],
        "sizeBytes": source["sizeBytes"],
    }
    _record_mismatch(
        mismatches,
        "$.descriptor.target",
        target["backend"],
        descriptor["target"],
    )
    _record_mismatch(
        mismatches,
        "$.descriptor.stage",
        target["stage"],
        descriptor["stage"],
    )
    _record_mismatch(
        mismatches,
        "$.descriptor.entryPoint.name",
        target["entryPoint"],
        descriptor["entryPoint"]["name"],
    )
    _record_mismatch(
        mismatches,
        "$.descriptor.artifact",
        expected_artifact,
        descriptor["artifact"],
    )
    descriptor_execution = _descriptor_execution_identity(
        descriptor["entryPoint"]["executionConfig"]
    )
    _record_mismatch(
        mismatches,
        "$.descriptor.entryPoint.executionConfig",
        request["variant"]["execution"],
        descriptor_execution,
    )
    _verify_descriptor_specialization_values(request, descriptor, mismatches)
    if mismatches:
        _raise_interface_drift(mismatches, phase="descriptor")

    source_path = Path(materialized["source"]["path"])
    reflected = reflect_target_host_interface(
        source_path,
        target=target["backend"],
        artifact_format=source["format"],
        stage=target["stage"],
    )
    if not isinstance(reflected, Mapping) or reflected.get("status") != "ready":
        raise NativeDeferredCompilationRuntimeError(
            "source-reflection-failed",
            "Deferred compilation source interface could not be reflected exactly.",
            path="$.source",
            details={
                "sourcePath": str(source_path),
                "reflection": copy.deepcopy(reflected),
            },
        )

    entries = [
        entry
        for entry in reflected.get("entryPoints", [])
        if isinstance(entry, Mapping) and entry.get("name") == target["entryPoint"]
    ]
    if len(entries) != 1:
        _raise_interface_drift(
            [
                {
                    "path": "$.reflection.entryPoints",
                    "expected": target["entryPoint"],
                    "observed": [
                        entry.get("name")
                        for entry in reflected.get("entryPoints", [])
                        if isinstance(entry, Mapping)
                    ],
                }
            ],
            phase="source-reflection",
        )
    reflected_execution = _descriptor_execution_identity(
        entries[0].get("executionConfig", {}),
        allow_missing_subgroup=True,
    )
    reflected_execution["subgroupWidth"] = descriptor_execution["subgroupWidth"]
    _record_mismatch(
        mismatches,
        "$.reflection.entryPoints[].executionConfig",
        request["variant"]["execution"],
        reflected_execution,
    )

    try:
        reflected_bindings = _binding_descriptors(
            target["backend"],
            reflected.get("resources"),
        )
    except NativeLoaderABIError as exc:
        raise NativeDeferredCompilationRuntimeError(
            "source-interface-invalid",
            "Reflected source resources cannot form the expected loader interface.",
            path=exc.path,
            details={"descriptorDiagnostic": exc.to_json()},
        ) from exc
    expected_bindings = [
        _binding_interface_identity(binding) for binding in descriptor["bindings"]
    ]
    observed_bindings = [
        _binding_interface_identity(binding) for binding in reflected_bindings
    ]
    _record_mismatch(
        mismatches,
        "$.reflection.resources",
        _sorted_json_records(expected_bindings),
        _sorted_json_records(observed_bindings),
    )
    _record_mismatch(
        mismatches,
        "$.reflection.constants",
        _sorted_json_records(descriptor["scalarLayout"]["constants"]),
        _sorted_json_records(reflected.get("constants", [])),
    )
    if target["backend"] == "opengl":
        _record_mismatch(
            mismatches,
            "$.reflection.specializationConstants",
            _specialization_interface_identity(descriptor["specializationConstants"]),
            _specialization_interface_identity(
                reflected.get("specializationConstants", [])
            ),
        )
    if mismatches:
        _raise_interface_drift(mismatches, phase="source-reflection")

    return {
        "status": "verified",
        "descriptor": copy.deepcopy(request["expectedLoaderDescriptor"]),
        "source": copy.deepcopy(request["source"]),
        "reflection": {
            "status": reflected["status"],
            "parser": reflected.get("parser"),
            "entryPointCount": reflected.get("entryPointCount"),
            "resourceCount": reflected.get("resourceCount"),
            "constantCount": reflected.get("constantCount"),
            "specializationConstantCount": reflected.get("specializationConstantCount"),
        },
    }


def _verify_descriptor_specialization_values(
    request: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    mismatches: list[dict[str, Any]],
) -> None:
    requested = request["variant"]["specializationValues"]
    declared = descriptor["specializationConstants"]
    declared_by_identity: dict[tuple[Any, Any], Mapping[str, Any]] = {
        (item.get("id"), item.get("name")): item for item in declared
    }
    requested_identities = {(item.get("id"), item.get("name")) for item in requested}
    declared_identities = set(declared_by_identity)
    if requested_identities != declared_identities:
        mismatches.append(
            {
                "path": "$.descriptor.specializationConstants",
                "expected": sorted(
                    (list(identity) for identity in requested_identities),
                    key=_canonical_json,
                ),
                "observed": sorted(
                    (list(identity) for identity in declared_identities),
                    key=_canonical_json,
                ),
            }
        )
        return
    for index, item in enumerate(requested):
        identity = (item.get("id"), item.get("name"))
        descriptor_item = declared_by_identity.get(identity)
        if descriptor_item is None:
            mismatches.append(
                {
                    "path": "$.descriptor.specializationConstants" f"[{index}]",
                    "expected": copy.deepcopy(item),
                    "observed": None,
                }
            )
            continue
        if request["target"]["backend"] != "directx":
            continue
        _record_mismatch(
            mismatches,
            f"$.descriptor.specializationConstants[{index}].value",
            item["value"],
            descriptor_item.get("value"),
        )


def _descriptor_execution_identity(
    value: Mapping[str, Any],
    *,
    allow_missing_subgroup: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeDeferredCompilationRuntimeError(
            "execution-interface-invalid",
            "Loader execution configuration must be an object.",
            path="$.descriptor.entryPoint.executionConfig",
        )
    workgroups = [
        list(value[field])
        for field in _EXECUTION_WORKGROUP_FIELDS
        if field in value and value[field] is not None
    ]
    component_fields = ("local_size_x", "local_size_y", "local_size_z")
    if any(field in value for field in component_fields):
        workgroups.append([value.get(field, 1) for field in component_fields])
    if not workgroups or any(
        len(item) != 3
        or any(type(component) is not int or component <= 0 for component in item)
        for item in workgroups
    ):
        raise NativeDeferredCompilationRuntimeError(
            "execution-interface-invalid",
            "Loader execution configuration requires one positive workgroup size.",
            path="$.descriptor.entryPoint.executionConfig",
        )
    if any(item != workgroups[0] for item in workgroups[1:]):
        raise NativeDeferredCompilationRuntimeError(
            "execution-interface-ambiguous",
            "Loader execution configuration contains conflicting workgroup sizes.",
            path="$.descriptor.entryPoint.executionConfig",
            details={"workgroupSizes": workgroups},
        )
    subgroup_values = [
        value[field]
        for field in _EXECUTION_SUBGROUP_FIELDS
        if field in value and value[field] is not None
    ]
    if any(type(item) is not int or item <= 0 for item in subgroup_values):
        raise NativeDeferredCompilationRuntimeError(
            "execution-interface-invalid",
            "Loader subgroup width must be a positive integer.",
            path="$.descriptor.entryPoint.executionConfig",
        )
    if any(item != subgroup_values[0] for item in subgroup_values[1:]):
        raise NativeDeferredCompilationRuntimeError(
            "execution-interface-ambiguous",
            "Loader execution configuration contains conflicting subgroup widths.",
            path="$.descriptor.entryPoint.executionConfig",
            details={"subgroupWidths": subgroup_values},
        )
    subgroup = subgroup_values[0] if subgroup_values else None
    if subgroup is None and not allow_missing_subgroup:
        subgroup = None
    return {"workgroupSize": workgroups[0], "subgroupWidth": subgroup}


def _binding_interface_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: copy.deepcopy(binding.get(field))
        for field in (
            "name",
            "kind",
            "type",
            "namespace",
            "coordinates",
            "access",
            "scalarLayout",
        )
    }


def _specialization_interface_identity(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for item in values:
        records.append(
            {
                "id": item.get("id", item.get("constantId")),
                "name": item.get("name"),
                "dtype": _scalar_type_name(item.get("dtype")),
            }
        )
    return _sorted_json_records(records)


def _scalar_type_name(value: Any) -> str:
    normalized = re.sub(r"\s+", "", str(value or "")).lower()
    aliases = {
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
        "float32_t": "float32",
    }
    return aliases.get(normalized, normalized)


def _resolve_toolchain(
    request: Mapping[str, Any],
    *,
    runner: Callable[[Sequence[str]], Any],
    resolver: Callable[[str], str | None],
) -> _Toolchain:
    backend = request["target"]["backend"]
    name, probe_arguments = _TOOLCHAINS[backend]
    resolved_value = resolver(name)
    if not isinstance(resolved_value, str) or not resolved_value:
        raise NativeDeferredCompilationRuntimeError(
            "toolchain-unavailable",
            "Required deferred compilation toolchain is unavailable.",
            path="$.toolchain",
            details={"target": backend, "tool": name},
        )
    try:
        executable = Path(resolved_value).resolve(strict=True)
        metadata = executable.stat()
        content = executable.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise NativeDeferredCompilationRuntimeError(
            "toolchain-unavailable",
            "Deferred compilation toolchain executable could not be verified.",
            path="$.toolchain",
            details={
                "tool": name,
                "resolvedExecutable": resolved_value,
                "error": str(exc),
            },
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or not content:
        raise NativeDeferredCompilationRuntimeError(
            "toolchain-invalid",
            "Deferred compilation toolchain must resolve to a non-empty regular file.",
            path="$.toolchain",
            details={"tool": name, "resolvedExecutable": str(executable)},
        )
    executable_hash = _sha256(content)
    probe_command = (str(executable), *probe_arguments)
    probe_result = _command_result(
        probe_command,
        _invoke_command(
            runner,
            probe_command,
            code="toolchain-probe-failed",
            message="Deferred compilation toolchain version probe could not run.",
            path="$.toolchain",
        ),
    )
    if probe_result["returncode"] != 0:
        raise NativeDeferredCompilationRuntimeError(
            "toolchain-probe-failed",
            "Deferred compilation toolchain version probe failed.",
            path="$.toolchain",
            details={"tool": name, **probe_result},
        )
    version = _toolchain_version(probe_result)
    return _Toolchain(
        name=name,
        executable=executable,
        version=version,
        executable_hash=executable_hash,
        probe_command=probe_command,
        probe_result={
            "returncode": probe_result["returncode"],
            "stdout": probe_result["stdout"],
            "stderr": probe_result["stderr"],
        },
    )


def _toolchain_version(result: Mapping[str, Any]) -> str:
    lines = [
        line.strip()
        for value in (result.get("stdout"), result.get("stderr"))
        for line in str(value or "").splitlines()
        if line.strip()
    ]
    if not lines:
        raise NativeDeferredCompilationRuntimeError(
            "toolchain-version-missing",
            "Deferred compilation toolchain did not report a version.",
            path="$.toolchain.version",
        )
    version = lines[0]
    if len(version) > 4096 or any(ord(character) < 32 for character in version):
        raise NativeDeferredCompilationRuntimeError(
            "toolchain-version-invalid",
            "Deferred compilation toolchain reported an invalid version.",
            path="$.toolchain.version",
        )
    return version


def _compiler_command(
    request: Mapping[str, Any],
    materialized: Mapping[str, Any],
    toolchain: _Toolchain,
    *,
    output_path: Path,
) -> tuple[str, ...]:
    source_path = Path(materialized["source"]["path"])
    source_text = _read_source_text(source_path)
    defines = tuple(
        f"-D{name}={value}"
        for name, value in sorted(request["variant"]["compileDefines"].items())
    )
    include_directories = sorted(
        {
            str(Path(materialized["sourceRoot"])),
            *(str(Path(value)) for value in materialized["includeDirectories"]),
        }
    )
    directx_include_arguments = tuple(
        argument for directory in include_directories for argument in ("-I", directory)
    )
    target = request["target"]
    if target["backend"] == "directx":
        profile = dxc_profile_for_source(target["profile"], source_text)
        if profile != target["profile"]:
            raise NativeDeferredCompilationRuntimeError(
                "target-profile-incompatible",
                "Requested DirectX profile cannot preserve the packaged source types.",
                path="$.target.profile",
                details={
                    "requestedProfile": target["profile"],
                    "requiredProfile": profile,
                },
            )
        compiler_arguments = dxc_compiler_arguments_for_source(source_text)
        return (
            str(toolchain.executable),
            "-T",
            profile,
            *compiler_arguments,
            "-E",
            target["entryPoint"],
            *defines,
            *directx_include_arguments,
            "-Fo",
            str(output_path),
            str(source_path),
        )
    opengl_include_arguments = tuple(
        f"-I{directory}" for directory in include_directories
    )
    return (
        str(toolchain.executable),
        "--target-env",
        "opengl",
        "-S",
        "comp",
        "-e",
        target["entryPoint"],
        *defines,
        *opengl_include_arguments,
        "-o",
        str(output_path),
        str(source_path),
    )


def _read_source_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise NativeDeferredCompilationRuntimeError(
            "source-read-failed",
            "Materialized deferred compilation source could not be read.",
            path="$.source",
            details={"sourcePath": str(path), "error": str(exc)},
        ) from exc


def _read_compiler_output(path: Path, *, output_format: str) -> bytes:
    try:
        metadata = path.lstat()
        content = path.read_bytes()
    except FileNotFoundError as exc:
        raise NativeDeferredCompilationRuntimeError(
            "compiler-output-missing",
            "Deferred compiler succeeded without producing its declared output.",
            path="$.compiler.output",
            details={"outputPath": str(path)},
        ) from exc
    except OSError as exc:
        raise NativeDeferredCompilationRuntimeError(
            "compiler-output-read-failed",
            "Deferred compiler output could not be read.",
            path="$.compiler.output",
            details={"outputPath": str(path), "error": str(exc)},
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise NativeDeferredCompilationRuntimeError(
            "compiler-output-invalid",
            "Deferred compiler output must be a regular file.",
            path="$.compiler.output",
            details={"outputPath": str(path)},
        )
    if output_format == _DXIL_FORMAT:
        if len(content) < 32 or content[:4] != b"DXBC":
            raise NativeDeferredCompilationRuntimeError(
                "dxil-output-invalid",
                "Deferred DirectX compiler output is not a complete DXIL container.",
                path="$.compiler.output",
                details={"outputPath": str(path), "sizeBytes": len(content)},
            )
        return content
    if (
        output_format != _SPIRV_FORMAT
        or len(content) < 20
        or len(content) % 4
        or content[:4] != b"\x03\x02#\x07"
    ):
        raise NativeDeferredCompilationRuntimeError(
            "spirv-output-invalid",
            "Deferred OpenGL compiler output is not a complete SPIR-V module.",
            path="$.compiler.output",
            details={"outputPath": str(path), "sizeBytes": len(content)},
        )
    return content


def _verify_toolchain_unchanged(toolchain: _Toolchain) -> None:
    try:
        observed = _sha256(toolchain.executable.read_bytes())
    except OSError as exc:
        raise NativeDeferredCompilationRuntimeError(
            "toolchain-changed",
            "Deferred compilation toolchain could not be re-verified.",
            path="$.toolchain",
            details={
                "resolvedExecutable": str(toolchain.executable),
                "error": str(exc),
            },
        ) from exc
    if observed != toolchain.executable_hash:
        raise NativeDeferredCompilationRuntimeError(
            "toolchain-changed",
            "Deferred compilation toolchain changed during compilation.",
            path="$.toolchain",
            details={
                "resolvedExecutable": str(toolchain.executable),
                "expectedHash": copy.deepcopy(dict(toolchain.executable_hash)),
                "observedHash": observed,
            },
        )


def _compiled_runtime_descriptor(
    descriptor: Mapping[str, Any],
    request: Mapping[str, Any],
    cache_hit: Mapping[str, Any],
    toolchain: _Toolchain,
) -> tuple[dict[str, Any], Path]:
    entry_path = Path(cache_hit["entryPath"])
    output_path = Path(cache_hit["outputPath"])
    runtime_package_root = entry_path.parent
    try:
        package_path = output_path.relative_to(runtime_package_root).as_posix()
    except ValueError as exc:
        raise NativeDeferredCompilationRuntimeError(
            "cache-output-path-invalid",
            "Compiled cache output is outside its entry package.",
            path="$.cache.outputPath",
            details={
                "entryPath": str(entry_path),
                "outputPath": str(output_path),
            },
        ) from exc
    output = cache_hit["entry"]["output"]
    compiled = copy.deepcopy(dict(descriptor))
    compiled["artifact"] = {
        "packagePath": package_path,
        "format": output["format"],
        "hash": copy.deepcopy(output["hash"]),
        "sizeBytes": output["sizeBytes"],
    }
    provenance = copy.deepcopy(dict(compiled["provenance"]))
    provenance["deferredCompilation"] = {
        "requestHash": copy.deepcopy(request["requestHash"]),
        "toolchain": toolchain.identity,
        "outputHash": copy.deepcopy(output["hash"]),
    }
    compiled["provenance"] = provenance
    try:
        return _validate_descriptor(compiled), runtime_package_root
    except NativeLoaderABIError as exc:
        raise NativeDeferredCompilationRuntimeError(
            "compiled-descriptor-invalid",
            "Compiled deferred artifact cannot form a native loader descriptor.",
            path=exc.path,
            details={"descriptorDiagnostic": exc.to_json()},
        ) from exc


def _validated_compilation_result(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeDeferredCompilationRuntimeError(
            "result-invalid",
            "Deferred compilation result must be an object.",
            path="$.compilation",
        )
    required = {
        "schemaVersion",
        "kind",
        "success",
        "requestHash",
        "request",
        "target",
        "variant",
        "runtimePackageRoot",
        "runtimeDescriptor",
    }
    missing = sorted(required - set(value))
    if (
        missing
        or value.get("schemaVersion") != NATIVE_DEFERRED_COMPILATION_RESULT_VERSION
        or value.get("kind") != NATIVE_DEFERRED_COMPILATION_RESULT_KIND
        or value.get("success") is not True
    ):
        raise NativeDeferredCompilationRuntimeError(
            "result-invalid",
            "Deferred compilation result is incomplete or unsupported.",
            path="$.compilation",
            details={"missingFields": missing},
        )
    runtime_root = value.get("runtimePackageRoot")
    if not isinstance(runtime_root, str) or not runtime_root:
        raise NativeDeferredCompilationRuntimeError(
            "result-invalid",
            "Deferred compilation result has no runtime package root.",
            path="$.compilation.runtimePackageRoot",
        )
    try:
        descriptor = _validate_descriptor(value.get("runtimeDescriptor"))
    except NativeLoaderABIError as exc:
        raise NativeDeferredCompilationRuntimeError(
            "result-invalid",
            "Deferred compilation result contains an invalid runtime descriptor.",
            path=exc.path,
            details={"descriptorDiagnostic": exc.to_json()},
        ) from exc
    try:
        normalized_request = validate_native_deferred_compilation_request(
            value.get("request")
        )
    except (TypeError, ValueError) as exc:
        raise NativeDeferredCompilationRuntimeError(
            "result-request-invalid",
            "Deferred compilation result does not retain a valid source request.",
            path="$.compilation.request",
            details={"error": str(exc)},
        ) from exc
    mismatches: list[dict[str, Any]] = []
    _record_mismatch(
        mismatches,
        "$.compilation.requestHash",
        normalized_request["requestHash"],
        value.get("requestHash"),
    )
    _record_mismatch(
        mismatches,
        "$.compilation.target",
        normalized_request["target"],
        value.get("target"),
    )
    _record_mismatch(
        mismatches,
        "$.compilation.variant",
        normalized_request["variant"],
        value.get("variant"),
    )
    _record_mismatch(
        mismatches,
        "$.compilation.runtimeDescriptor.target",
        normalized_request["target"]["backend"],
        descriptor["target"],
    )
    _record_mismatch(
        mismatches,
        "$.compilation.runtimeDescriptor.artifact.format",
        normalized_request["target"]["outputFormat"],
        descriptor["artifact"]["format"],
    )
    descriptor_provenance = descriptor.get("provenance", {})
    deferred_provenance = (
        descriptor_provenance.get("deferredCompilation", {})
        if isinstance(descriptor_provenance, Mapping)
        else {}
    )
    observed_request_hash = (
        deferred_provenance.get("requestHash")
        if isinstance(deferred_provenance, Mapping)
        else None
    )
    _record_mismatch(
        mismatches,
        "$.compilation.runtimeDescriptor.provenance.deferredCompilation.requestHash",
        normalized_request["requestHash"],
        observed_request_hash,
    )
    if mismatches:
        raise NativeDeferredCompilationRuntimeError(
            "result-request-drift",
            "Deferred compilation result no longer matches its hashed source request.",
            path="$.compilation",
            details={"mismatches": mismatches},
        )
    result = copy.deepcopy(dict(value))
    result["request"] = normalized_request
    result["requestHash"] = copy.deepcopy(normalized_request["requestHash"])
    result["target"] = copy.deepcopy(normalized_request["target"])
    result["variant"] = copy.deepcopy(normalized_request["variant"])
    result["runtimeDescriptor"] = descriptor
    return result


def _runtime_compilation_report(
    compilation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        field: copy.deepcopy(compilation[field])
        for field in (
            "schemaVersion",
            "kind",
            "success",
            "requestHash",
            "target",
            "variant",
            "toolchain",
            "interface",
            "compiler",
            "cache",
            "output",
            "provenance",
            "diagnostics",
        )
    }


def _record_mismatch(
    mismatches: list[dict[str, Any]],
    path: str,
    expected: Any,
    observed: Any,
) -> None:
    if expected == observed:
        return
    mismatches.append(
        {
            "path": path,
            "expected": copy.deepcopy(expected),
            "observed": copy.deepcopy(observed),
        }
    )


def _raise_interface_drift(
    mismatches: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> None:
    raise NativeDeferredCompilationRuntimeError(
        "interface-drift",
        "Deferred compilation source no longer matches its expected loader interface.",
        path="$.expectedLoaderDescriptor",
        details={
            "phase": phase,
            "mismatches": [copy.deepcopy(dict(item)) for item in mismatches],
        },
    )


def _sorted_json_records(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (copy.deepcopy(dict(value)) for value in values),
        key=_canonical_json,
    )


def _command_result(command: Sequence[str], result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        returncode = result.get("returncode", result.get("status", 0))
        if isinstance(returncode, str):
            returncode = (
                0 if returncode.strip().lower() in {"ok", "pass", "passed"} else 1
            )
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
    else:
        returncode = getattr(result, "returncode", 0)
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
    return {
        "command": [str(item) for item in command],
        "returncode": int(returncode),
        "stdout": "" if stdout is None else str(stdout),
        "stderr": "" if stderr is None else str(stderr),
    }


def _invoke_command(
    runner: Callable[[Sequence[str]], Any],
    command: Sequence[str],
    *,
    code: str,
    message: str,
    path: str,
) -> Any:
    try:
        return runner(command)
    except Exception as exc:
        raise NativeDeferredCompilationRuntimeError(
            code,
            message,
            path=path,
            details={
                "command": [str(item) for item in command],
                "error": str(exc),
            },
        ) from exc


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=False,
    )


def _sha256(content: bytes) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(content).hexdigest(),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Invalid JSON constant: {value}")
