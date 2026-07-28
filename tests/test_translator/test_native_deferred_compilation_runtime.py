import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_deferred_compilation import (
    build_native_deferred_compilation_request,
)
from crosstl.project.native_deferred_compilation_runtime import (
    NATIVE_DEFERRED_COMPILATION_RESULT_KIND,
    NativeDeferredCompilationRuntimeError,
    build_native_deferred_compilation_dispatch_request,
    compile_native_deferred_compilation_request,
    execute_native_deferred_compilation_request,
)
from crosstl.project.native_loader_abi import (
    NATIVE_LOADER_ABI_KIND,
    NATIVE_LOADER_ABI_VERSION,
    _binding_descriptors,
)
from crosstl.project.native_runtime_drivers import OpenGLComputeRuntime
from crosstl.project.pipeline import encode_runtime_variant_key
from crosstl.project.runtime_verification import (
    RuntimeExecutionRequest,
    RuntimeExecutorAvailability,
    RuntimeExecutorResult,
    native_runtime_parity_adapter,
)


def _is_relative_to(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _hash(content):
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(content).hexdigest(),
    }


def _source(backend, *, with_include=False):
    include = b"#include <common/constants.hlsl>\n" if with_include else b""
    if backend == "directx":
        return include + b"""
StructuredBuffer<float> input_values : register(t0);
RWStructuredBuffer<float> output_values : register(u1);

[numthreads(1, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    output_values[tid.x] = input_values[tid.x] + 1.0f;
}
""".strip() + b"\n"
    return b"""
#version 450 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) readonly buffer InputBlock {
    float input_values[];
};
layout(std430, binding = 1) writeonly buffer OutputBlock {
    float output_values[];
};
void main() {
    uint index = gl_GlobalInvocationID.x;
    output_values[index] = input_values[index] + 1.0;
}
""".strip() + b"\n"


def _write_package(
    tmp_path,
    backend="directx",
    *,
    descriptor_execution=None,
    descriptor_specializations=None,
    drift_binding=False,
    request_specializations=None,
    with_include=False,
):
    package_root = tmp_path / "package"
    source_suffix = "hlsl" if backend == "directx" else "comp.glsl"
    source_format = "HLSL source" if backend == "directx" else "GLSL source"
    output_format = "DXIL binary" if backend == "directx" else "SPIR-V binary"
    entry_point = "CSMain" if backend == "directx" else "main"
    profile = "cs_6_2" if backend == "directx" else None
    source_path = package_root / "sources" / f"copy.{source_suffix}"
    source_path.parent.mkdir(parents=True)
    source_bytes = _source(backend, with_include=with_include)
    source_path.write_bytes(source_bytes)
    includes = []
    if with_include:
        include_bytes = b"#define COPY_INCREMENT 1\n"
        include_path = package_root / "vendor" / "common" / "constants.hlsl"
        include_path.parent.mkdir(parents=True)
        include_path.write_bytes(include_bytes)
        includes.append(
            {
                "path": "vendor/common/constants.hlsl",
                "format": source_format,
                "hash": _hash(include_bytes),
                "sizeBytes": len(include_bytes),
            }
        )

    reflected = reflect_target_host_interface(
        source_path,
        target=backend,
        artifact_format=source_format,
        stage="compute",
    )
    assert reflected["status"] == "ready"
    bindings = _binding_descriptors(backend, reflected["resources"])
    if drift_binding:
        bindings[0]["name"] = "different_resource"
    execution = {"workgroupSize": [1, 1, 1], "subgroupWidth": None}
    request_specializations = copy.deepcopy(request_specializations or [])
    descriptor_specializations = copy.deepcopy(descriptor_specializations or [])

    descriptor = {
        "schemaVersion": NATIVE_LOADER_ABI_VERSION,
        "kind": NATIVE_LOADER_ABI_KIND,
        "abiVersion": NATIVE_LOADER_ABI_VERSION,
        "unitId": f"copy:{backend}",
        "target": backend,
        "stage": "compute",
        "entryPoint": {
            "name": entry_point,
            "stage": "compute",
            "executionConfig": copy.deepcopy(
                descriptor_execution
                if descriptor_execution is not None
                else next(
                    item["executionConfig"]
                    for item in reflected["entryPoints"]
                    if item["name"] == entry_point
                )
            ),
            "provenance": {"sourceEntry": "copy_float"},
        },
        "artifact": {
            "packagePath": f"sources/copy.{source_suffix}",
            "format": source_format,
            "hash": _hash(source_bytes),
            "sizeBytes": len(source_bytes),
        },
        "source": {
            "path": "kernels/copy.metal",
            "artifactPath": f"generated/copy.{source_suffix}",
            "backend": "metal",
            "hash": None,
            "remap": None,
        },
        "bindings": bindings,
        "scalarLayout": {
            "constants": copy.deepcopy(reflected["constants"]),
            "bindings": [
                {
                    "binding": binding["name"],
                    "layout": copy.deepcopy(binding["scalarLayout"]),
                }
                for binding in bindings
                if binding.get("scalarLayout") is not None
            ],
        },
        "specializationConstants": descriptor_specializations,
        "provenance": {"pipeline": "metal-to-crossgl", "target": backend},
    }
    descriptor_path = package_root / "descriptors" / "copy.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_bytes = (
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor_path.write_bytes(descriptor_bytes)

    variant_key = encode_runtime_variant_key(
        "kernels/copy.metal",
        "copy_float",
        backend,
        target_profile=profile,
        execution=execution,
        type_arguments={"T": "float"},
        value_arguments={"N": 4},
        specialization_constants=request_specializations,
        defines={},
    )
    request = build_native_deferred_compilation_request(
        {
            "path": f"sources/copy.{source_suffix}",
            "format": source_format,
            "hash": _hash(source_bytes),
            "sizeBytes": len(source_bytes),
        },
        includes,
        {
            "backend": backend,
            "profile": profile,
            "stage": "compute",
            "entryPoint": entry_point,
            "outputFormat": output_format,
        },
        {
            "key": variant_key,
            "typeArguments": {"T": "float"},
            "valueArguments": {"N": 4},
            "compileDefines": {},
            "specializationValues": request_specializations,
            "execution": execution,
        },
        {
            "path": "descriptors/copy.json",
            "hash": _hash(descriptor_bytes),
            "sizeBytes": len(descriptor_bytes),
        },
    )
    return package_root, request


class _Compiler:
    def __init__(
        self,
        backend,
        *,
        fail=False,
        invalid_output=False,
        missing_output=False,
        raise_on_compile=False,
    ):
        self.backend = backend
        self.fail = fail
        self.invalid_output = invalid_output
        self.missing_output = missing_output
        self.raise_on_compile = raise_on_compile
        self.commands = []

    @property
    def compile_commands(self):
        return [command for command in self.commands if "--version" not in command]

    def __call__(self, command):
        command = tuple(str(item) for item in command)
        self.commands.append(command)
        if "--version" in command:
            return {
                "returncode": 0,
                "stdout": (
                    "dxc version 1.9.0\n"
                    if self.backend == "directx"
                    else "Glslang Version: 16.4.0\n"
                ),
                "stderr": "",
            }
        if self.fail:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": "compile failed",
            }
        if self.raise_on_compile:
            raise OSError("compiler process could not start")
        output_flag = "-Fo" if self.backend == "directx" else "-o"
        output_path = Path(command[command.index(output_flag) + 1])
        if self.missing_output:
            pass
        elif self.invalid_output:
            output_path.write_bytes(b"invalid")
        elif self.backend == "directx":
            output_path.write_bytes(b"DXBC" + (b"\0" * 60))
        else:
            output_path.write_bytes(
                b"\x03\x02#\x07"
                + b"\x00\x06\x01\x00"
                + b"\x00\x00\x00\x00"
                + b"\x02\x00\x00\x00"
                + b"\x00\x00\x00\x00"
            )
        return {"returncode": 0, "stdout": "", "stderr": ""}


def _tool(tmp_path, backend):
    name = "dxc" if backend == "directx" else "glslangValidator"
    path = tmp_path / "tools" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{name}-fixture".encode())
    return path


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_compiles_and_reuses_exact_success_cache(tmp_path, backend):
    package_root, request = _write_package(tmp_path, backend)
    tool = _tool(tmp_path, backend)
    compiler = _Compiler(backend)
    cache_root = tmp_path / "cache"

    first = compile_native_deferred_compilation_request(
        request,
        package_root,
        cache_root,
        command_runner=compiler,
        tool_resolver=lambda name: str(tool),
    )
    second = compile_native_deferred_compilation_request(
        request,
        package_root,
        cache_root,
        command_runner=compiler,
        tool_resolver=lambda name: str(tool),
    )

    assert first["kind"] == NATIVE_DEFERRED_COMPILATION_RESULT_KIND
    assert first["success"] is True
    assert first["cache"]["status"] == "published"
    assert second["cache"]["status"] == "hit"
    assert second["compiler"]["status"] == "cache-hit"
    assert first["output"] == second["output"]
    assert len(compiler.compile_commands) == 1
    assert first["runtimeDescriptor"]["artifact"]["format"] == (
        "DXIL binary" if backend == "directx" else "SPIR-V binary"
    )
    assert first["interface"]["status"] == "verified"

    command = compiler.compile_commands[0]
    if backend == "directx":
        assert command[1:5] == ("-T", "cs_6_2", "-E", "CSMain")
        assert "-Fo" in command
    else:
        assert command[1:7] == (
            "--target-env",
            "opengl",
            "-S",
            "comp",
            "-e",
            "main",
        )
        assert command[7].startswith("-I")
        assert "-o" in command


def test_interface_drift_fails_before_toolchain_or_cache(tmp_path):
    package_root, request = _write_package(
        tmp_path,
        "directx",
        drift_binding=True,
    )
    tool = _tool(tmp_path, "directx")
    compiler = _Compiler("directx")

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        compile_native_deferred_compilation_request(
            request,
            package_root,
            tmp_path / "cache",
            command_runner=compiler,
            tool_resolver=lambda name: str(tool),
        )

    assert raised.value.code.endswith("interface-drift")
    assert raised.value.details["phase"] == "source-reflection"
    assert compiler.commands == []
    assert not (tmp_path / "cache").exists()


def test_descriptor_execution_drift_fails_before_toolchain_or_cache(tmp_path):
    package_root, request = _write_package(
        tmp_path,
        "directx",
        descriptor_execution={"workgroupSize": [2, 1, 1]},
    )
    tool = _tool(tmp_path, "directx")
    compiler = _Compiler("directx")

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        compile_native_deferred_compilation_request(
            request,
            package_root,
            tmp_path / "cache",
            command_runner=compiler,
            tool_resolver=lambda name: str(tool),
        )

    assert raised.value.code.endswith("interface-drift")
    assert raised.value.details["phase"] == "descriptor"
    assert raised.value.details["mismatches"][0]["path"] == (
        "$.descriptor.entryPoint.executionConfig"
    )
    assert compiler.commands == []
    assert not (tmp_path / "cache").exists()


def test_directx_specialization_value_drift_fails_before_compilation(tmp_path):
    requested = [{"id": 7, "name": "mode", "value": 1}]
    package_root, request = _write_package(
        tmp_path,
        "directx",
        request_specializations=requested,
        descriptor_specializations=[
            {"id": 7, "name": "mode", "dtype": "uint32", "value": 2}
        ],
    )
    tool = _tool(tmp_path, "directx")
    compiler = _Compiler("directx")

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        compile_native_deferred_compilation_request(
            request,
            package_root,
            tmp_path / "cache",
            command_runner=compiler,
            tool_resolver=lambda name: str(tool),
        )

    assert raised.value.code.endswith("interface-drift")
    assert raised.value.details["phase"] == "descriptor"
    assert raised.value.details["mismatches"] == [
        {
            "path": "$.descriptor.specializationConstants[0].value",
            "expected": 1,
            "observed": 2,
        }
    ]
    assert compiler.commands == []
    assert not (tmp_path / "cache").exists()


def test_compiler_failure_is_structured_and_not_cached(tmp_path):
    package_root, request = _write_package(tmp_path, "directx")
    tool = _tool(tmp_path, "directx")
    compiler = _Compiler("directx", fail=True)
    cache_root = tmp_path / "cache"

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        compile_native_deferred_compilation_request(
            request,
            package_root,
            cache_root,
            command_runner=compiler,
            tool_resolver=lambda name: str(tool),
        )

    assert raised.value.code.endswith("compiler-failed")
    assert raised.value.details["returncode"] == 1
    assert raised.value.details["stderr"] == "compile failed"
    assert not cache_root.exists()


def test_compiler_invocation_failure_is_structured_and_not_cached(tmp_path):
    package_root, request = _write_package(tmp_path, "directx")
    tool = _tool(tmp_path, "directx")
    compiler = _Compiler("directx", raise_on_compile=True)
    cache_root = tmp_path / "cache"

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        compile_native_deferred_compilation_request(
            request,
            package_root,
            cache_root,
            command_runner=compiler,
            tool_resolver=lambda name: str(tool),
        )

    assert raised.value.code.endswith("compiler-invocation-failed")
    assert raised.value.path == "$.compiler"
    assert raised.value.details["error"] == "compiler process could not start"
    assert not cache_root.exists()


def test_successful_compiler_exit_without_output_is_not_cached(tmp_path):
    package_root, request = _write_package(tmp_path, "directx")
    tool = _tool(tmp_path, "directx")
    compiler = _Compiler("directx", missing_output=True)
    cache_root = tmp_path / "cache"

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        compile_native_deferred_compilation_request(
            request,
            package_root,
            cache_root,
            command_runner=compiler,
            tool_resolver=lambda name: str(tool),
        )

    assert raised.value.code.endswith("compiler-output-missing")
    assert raised.value.path == "$.compiler.output"
    assert len(compiler.compile_commands) == 1
    assert not cache_root.exists()


@pytest.mark.parametrize(
    ("backend", "code"),
    [
        ("directx", "dxil-output-invalid"),
        ("opengl", "spirv-output-invalid"),
    ],
)
def test_invalid_compiler_output_is_not_published(tmp_path, backend, code):
    package_root, request = _write_package(tmp_path, backend)
    tool = _tool(tmp_path, backend)
    compiler = _Compiler(backend, invalid_output=True)
    cache_root = tmp_path / "cache"

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        compile_native_deferred_compilation_request(
            request,
            package_root,
            cache_root,
            command_runner=compiler,
            tool_resolver=lambda name: str(tool),
        )

    assert raised.value.code.endswith(code)
    assert not cache_root.exists()


def test_toolchain_change_during_compilation_is_not_cached(tmp_path):
    package_root, request = _write_package(tmp_path, "directx")
    tool = _tool(tmp_path, "directx")
    compiler = _Compiler("directx")
    cache_root = tmp_path / "cache"

    def mutating_compiler(command):
        result = compiler(command)
        if "--version" not in command:
            tool.write_bytes(b"dxc-replaced-during-compilation")
        return result

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        compile_native_deferred_compilation_request(
            request,
            package_root,
            cache_root,
            command_runner=mutating_compiler,
            tool_resolver=lambda name: str(tool),
        )

    assert raised.value.code.endswith("toolchain-changed")
    assert raised.value.details["expectedHash"] != raised.value.details["observedHash"]
    assert len(compiler.compile_commands) == 1
    assert not cache_root.exists()


def test_cache_hit_reverifies_toolchain_before_returning_artifact(tmp_path):
    package_root, request = _write_package(tmp_path, "directx")
    tool = _tool(tmp_path, "directx")
    cache_root = tmp_path / "cache"
    initial_compiler = _Compiler("directx")
    initial = compile_native_deferred_compilation_request(
        request,
        package_root,
        cache_root,
        command_runner=initial_compiler,
        tool_resolver=lambda name: str(tool),
    )
    output_bytes = Path(initial["output"]["path"]).read_bytes()
    probe_commands = []

    def mutating_probe(command):
        command = tuple(str(item) for item in command)
        probe_commands.append(command)
        assert "--version" in command
        tool.write_bytes(b"dxc-replaced-after-identity-check")
        return {
            "returncode": 0,
            "stdout": "dxc version 1.9.0\n",
            "stderr": "",
        }

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        compile_native_deferred_compilation_request(
            request,
            package_root,
            cache_root,
            command_runner=mutating_probe,
            tool_resolver=lambda name: str(tool),
        )

    assert raised.value.code.endswith("toolchain-changed")
    assert len(probe_commands) == 1
    assert Path(initial["output"]["path"]).read_bytes() == output_bytes


def test_compiler_receives_only_isolated_source_and_include_paths(tmp_path):
    package_root, request = _write_package(
        tmp_path,
        "directx",
        with_include=True,
    )
    tool = _tool(tmp_path, "directx")
    compiler = _Compiler("directx")
    workspace_root = tmp_path / "workspace"

    compile_native_deferred_compilation_request(
        request,
        package_root,
        tmp_path / "cache",
        workspace_root=workspace_root,
        command_runner=compiler,
        tool_resolver=lambda name: str(tool),
    )

    command = compiler.compile_commands[0]
    include_paths = [
        Path(command[index + 1])
        for index, argument in enumerate(command)
        if argument == "-I"
    ]
    source_path = Path(command[-1])
    output_path = Path(command[command.index("-Fo") + 1])
    resolved_workspace = workspace_root.resolve()
    resolved_package = package_root.resolve()

    assert len(include_paths) == 2
    assert all(_is_relative_to(path, resolved_workspace) for path in include_paths)
    assert any(path.name == "vendor" for path in include_paths)
    assert _is_relative_to(source_path, resolved_workspace)
    assert _is_relative_to(output_path, resolved_workspace)
    assert not _is_relative_to(source_path, resolved_package)
    assert all(not _is_relative_to(path, resolved_package) for path in include_paths)


def test_builds_compiled_native_loader_dispatch_request(tmp_path):
    package_root, request = _write_package(tmp_path, "directx")
    tool = _tool(tmp_path, "directx")
    compilation = compile_native_deferred_compilation_request(
        request,
        package_root,
        tmp_path / "cache",
        command_runner=_Compiler("directx"),
        tool_resolver=lambda name: str(tool),
    )

    runtime_request = build_native_deferred_compilation_dispatch_request(
        compilation,
        {
            "input_values": {
                "dtype": "float32",
                "shape": [4],
                "values": [1.0, 2.0, 3.0, 4.0],
            }
        },
        {
            "output_values": {
                "dtype": "float32",
                "shape": [4],
                "values": [2.0, 3.0, 4.0, 5.0],
            }
        },
        (4, 1, 1),
    )

    assert isinstance(runtime_request, RuntimeExecutionRequest)
    assert runtime_request.artifact["artifactFormat"] == "DXIL binary"
    assert runtime_request.artifact_identity.hash_value == (
        compilation["output"]["hash"]["value"]
    )
    assert runtime_request.artifact_path.read_bytes().startswith(b"DXBC")


def test_dispatch_rejects_target_drift_from_retained_request(tmp_path):
    package_root, request = _write_package(tmp_path, "directx")
    tool = _tool(tmp_path, "directx")
    compilation = compile_native_deferred_compilation_request(
        request,
        package_root,
        tmp_path / "cache",
        command_runner=_Compiler("directx"),
        tool_resolver=lambda name: str(tool),
    )
    compilation["target"]["profile"] = "cs_6_0"

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        build_native_deferred_compilation_dispatch_request(
            compilation,
            {},
            {},
            (1, 1, 1),
        )

    assert raised.value.code.endswith("result-request-drift")
    assert raised.value.details["mismatches"] == [
        {
            "path": "$.compilation.target",
            "expected": request["target"],
            "observed": compilation["target"],
        }
    ]


def test_dispatch_rejects_specialization_drift_from_retained_request(tmp_path):
    specializations = [{"id": 7, "name": "mode", "value": 1}]
    package_root, request = _write_package(
        tmp_path,
        "directx",
        request_specializations=specializations,
        descriptor_specializations=[
            {"id": 7, "name": "mode", "dtype": "uint32", "value": 1}
        ],
    )
    tool = _tool(tmp_path, "directx")
    compilation = compile_native_deferred_compilation_request(
        request,
        package_root,
        tmp_path / "cache",
        command_runner=_Compiler("directx"),
        tool_resolver=lambda name: str(tool),
    )
    compilation["variant"]["specializationValues"][0]["value"] = 2

    with pytest.raises(NativeDeferredCompilationRuntimeError) as raised:
        build_native_deferred_compilation_dispatch_request(
            compilation,
            {},
            {},
            (1, 1, 1),
        )

    assert raised.value.code.endswith("result-request-drift")
    assert raised.value.details["mismatches"] == [
        {
            "path": "$.compilation.variant",
            "expected": request["variant"],
            "observed": compilation["variant"],
        }
    ]


def test_execute_runs_compiled_request_through_supplied_adapter(tmp_path):
    package_root, request = _write_package(tmp_path, "directx")
    tool = _tool(tmp_path, "directx")

    class Adapter:
        def __init__(self):
            self.request = None

        def is_available(self, runtime_request):
            self.request = runtime_request
            return RuntimeExecutorAvailability(True, details={"runtime": "fixture"})

        def run(self, runtime_request):
            assert runtime_request is self.request
            return RuntimeExecutorResult(
                outputs={
                    "output_values": {
                        "dtype": "float32",
                        "shape": [1],
                        "values": [2.0],
                    }
                },
                details={"nativeRuntimeDispatch": {"status": "passed"}},
            )

    adapter = Adapter()
    result = execute_native_deferred_compilation_request(
        request,
        package_root,
        tmp_path / "cache",
        {
            "input_values": {
                "dtype": "float32",
                "shape": [1],
                "values": [1.0],
            }
        },
        {
            "output_values": {
                "dtype": "float32",
                "shape": [1],
                "values": [2.0],
            }
        },
        (1, 1, 1),
        runtime_adapter=adapter,
        command_runner=_Compiler("directx"),
        tool_resolver=lambda name: str(tool),
    )

    assert adapter.request.artifact["artifactFormat"] == "DXIL binary"
    assert result.outputs["output_values"]["values"] == [2.0]
    assert result.details["nativeDeferredCompilation"]["success"] is True


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None,
    reason="glslangValidator is unavailable",
)
def test_real_opengl_toolchain_compiles_and_validates_cached_output(tmp_path):
    package_root, request = _write_package(tmp_path, "opengl")

    result = compile_native_deferred_compilation_request(
        request,
        package_root,
        tmp_path / "cache",
    )

    output_path = Path(result["output"]["path"])
    assert output_path.read_bytes().startswith(b"\x03\x02#\x07")
    spirv_validator = shutil.which("spirv-val")
    if spirv_validator is not None:
        validation = subprocess.run(
            [spirv_validator, "--target-env", "opengl4.5", str(output_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert validation.returncode == 0, validation.stderr


def test_deferred_compilation_contracts_are_public_project_apis():
    import crosstl.project as project

    expected = {
        "NATIVE_DEFERRED_COMPILATION_REQUEST_KIND",
        "NATIVE_DEFERRED_COMPILATION_RESULT_KIND",
        "NativeDeferredCompilationError",
        "NativeDeferredCompilationCacheError",
        "NativeDeferredCompilationPackageError",
        "NativeDeferredCompilationRuntimeError",
        "build_native_deferred_compilation_request",
        "validate_native_deferred_compilation_request",
        "materialize_native_deferred_compilation_inputs",
        "lookup_native_deferred_compilation_cache",
        "publish_native_deferred_compilation_cache",
        "compile_native_deferred_compilation_request",
        "build_native_deferred_compilation_dispatch_request",
        "execute_native_deferred_compilation_request",
    }

    assert expected <= set(project.__all__)
    assert all(hasattr(project, name) for name in expected)


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_real_deferred_compilation_dispatch(tmp_path, backend):
    required_target = os.environ.get("CROSSTL_REQUIRE_DEFERRED_NATIVE_TARGET")
    missing = []
    if backend == "directx":
        if sys.platform != "win32":
            missing.append("Windows")
        if shutil.which("dxc") is None:
            missing.append("dxc")
        if importlib.util.find_spec("compushady") is None:
            missing.append("compushady")
    else:
        if shutil.which("glslangValidator") is None:
            missing.append("glslangValidator")
        for module_name in ("moderngl", "OpenGL"):
            if importlib.util.find_spec(module_name) is None:
                missing.append(module_name)
    if missing:
        message = f"{backend} deferred runtime prerequisites unavailable: {missing}"
        if required_target == backend:
            pytest.fail(message)
        pytest.skip(message)

    package_root, request = _write_package(tmp_path, backend)
    input_name = "input_values" if backend == "directx" else "InputBlock"
    output_name = "output_values" if backend == "directx" else "OutputBlock"
    context = None
    runtime_adapter = None
    try:
        if backend == "opengl":
            import moderngl

            try:
                context = moderngl.create_standalone_context(
                    backend="egl",
                    require=430,
                )
            except Exception as exc:
                message = f"OpenGL EGL context unavailable: {exc}"
                if required_target == backend:
                    pytest.fail(message)
                pytest.skip(message)
            runtime_adapter = native_runtime_parity_adapter(
                "opengl",
                runtime=OpenGLComputeRuntime(
                    context_factory=lambda module: context,
                    release_context=False,
                ),
            )
        try:
            result = execute_native_deferred_compilation_request(
                request,
                package_root,
                tmp_path / "cache",
                {
                    input_name: {
                        "dtype": "float32",
                        "shape": [4],
                        "values": [1.0, 2.0, 3.0, 4.0],
                    }
                },
                {
                    output_name: {
                        "dtype": "float32",
                        "shape": [4],
                        "values": [2.0, 3.0, 4.0, 5.0],
                    }
                },
                (4, 1, 1),
                runtime_adapter=runtime_adapter,
            )
        except NativeDeferredCompilationRuntimeError as exc:
            if exc.code.endswith("runtime-unavailable") and required_target != backend:
                pytest.skip(str(exc))
            raise

        if context is not None:
            probe = context.buffer(data=b"\x01\x02\x03\x04")
            try:
                assert probe.read() == b"\x01\x02\x03\x04"
            finally:
                probe.release()
    finally:
        if context is not None:
            context.release()

    assert result.outputs[output_name]["values"] == [2.0, 3.0, 4.0, 5.0]
    report = result.details["nativeDeferredCompilation"]
    assert report["target"]["backend"] == backend
    assert report["cache"]["status"] == "published"
    assert report["interface"]["status"] == "verified"
