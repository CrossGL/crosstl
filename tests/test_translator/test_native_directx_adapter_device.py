from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import crosstl
from crosstl.project import (
    generate_native_loader_execution_abi,
    generate_native_loader_target_adapter,
)
from crosstl.project.native_loader_abi import (
    NATIVE_LOADER_ABI_KIND,
    NATIVE_LOADER_ABI_VERSION,
)

_RUN_DEVICE_TEST = "CROSSTL_RUN_NATIVE_DIRECTX_ADAPTER_DEVICE_TEST"
_DXC_INCLUDE_DIR = "CROSSTL_DXC_INCLUDE_DIR"
_DXC_DLL_DIR = "CROSSTL_DXC_DLL_DIR"
_FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "native_loader"
    / "directx"
    / "specialized_output.glsl"
)
_ARTIFACT_PATH = Path("artifacts/directx/specialized_output.hlsl")
_ELEMENT_COUNT = 8
_WORKGROUP_SIZE = 4
_SPECIALIZATION_ID = 7
_SPECIALIZATION_NAME = "MULTIPLIER"
_SPECIALIZATION_VALUE = 11
_DEFAULT_VALUE = 3
_INPUT_VALUES = (2, 7, 1, 9, 4, 13, 6, 5)
_EXPECTED_VALUES = tuple(value * _SPECIALIZATION_VALUE for value in _INPUT_VALUES)
_DEFAULT_EXPECTED_VALUES = tuple(value * _DEFAULT_VALUE for value in _INPUT_VALUES)


def _descriptor(artifact: bytes) -> dict:
    scalar_layout = {
        "alignmentBytes": 4,
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "elementType": "uint32",
        "runtimeSized": True,
        "storageLayout": "hlsl-structured-buffer",
    }
    return {
        "schemaVersion": NATIVE_LOADER_ABI_VERSION,
        "kind": NATIVE_LOADER_ABI_KIND,
        "abiVersion": NATIVE_LOADER_ABI_VERSION,
        "unitId": "native-directx-specialized-output-device-test",
        "target": "directx",
        "stage": "compute",
        "entryPoint": {
            "name": "CSMain",
            "stage": "compute",
            "executionConfig": {
                "numthreads": [_WORKGROUP_SIZE, 1, 1],
            },
            "provenance": {},
        },
        "artifact": {
            "packagePath": _ARTIFACT_PATH.as_posix(),
            "format": "HLSL source",
            "hash": {
                "algorithm": "sha256",
                "value": hashlib.sha256(artifact).hexdigest(),
            },
            "sizeBytes": len(artifact),
        },
        "source": {
            "path": "shaders/specialized_output.glsl",
            "artifactPath": "out/directx/specialized_output.hlsl",
            "backend": "opengl",
            "hash": None,
            "remap": None,
        },
        "bindings": [
            {
                "name": "input_values",
                "kind": "buffer",
                "type": "StructuredBuffer<uint>",
                "namespace": "srv",
                "coordinates": {"set": 0, "binding": 0},
                "access": "read",
                "scalarLayout": scalar_layout,
                "provenance": {},
            },
            {
                "name": "output_values",
                "kind": "buffer",
                "type": "RWStructuredBuffer<uint>",
                "namespace": "uav",
                "coordinates": {"set": 0, "binding": 1},
                "access": "write",
                "scalarLayout": scalar_layout,
                "provenance": {},
            },
        ],
        "scalarLayout": {
            "constants": [],
            "bindings": [
                {
                    "binding": "input_values",
                    "layout": scalar_layout,
                },
                {
                    "binding": "output_values",
                    "layout": scalar_layout,
                },
            ],
        },
        "specializationConstants": [
            {
                "id": _SPECIALIZATION_ID,
                "name": _SPECIALIZATION_NAME,
                "dtype": "uint32",
                "defaultValue": _DEFAULT_VALUE,
            }
        ],
        "provenance": {},
    }


def _execution_symbol(execution_header: str) -> str:
    match = re.search(
        r"static inline CrossTLNativeLoaderExecutionResult\s+"
        r"([A-Za-z_]\w*_execute)\(",
        execution_header,
    )
    assert match is not None
    return match.group(1)


def _unit_symbol(execution_header: str) -> str:
    match = re.search(
        r"static const CrossTLNativeLoaderUnitDescriptor\s+" r"([A-Za-z_]\w*)\s*=",
        execution_header,
    )
    assert match is not None
    return match.group(1)


def _translate_fixture() -> bytes:
    generated = crosstl.translate(
        str(_FIXTURE),
        backend="directx",
        format_output=False,
        source_backend="opengl",
    )
    return generated.encode("utf-8")


def _write_package(tmp_path: Path) -> tuple[Path, Path, str, str, bytes]:
    artifact = _translate_fixture()
    descriptor = _descriptor(artifact)
    package_root = tmp_path / "package"
    artifact_path = package_root / _ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(artifact)

    execution_header = generate_native_loader_execution_abi(descriptor)
    adapter_header = generate_native_loader_target_adapter("directx")
    build_root = tmp_path / "build"
    build_root.mkdir()
    (build_root / "execution.h").write_text(execution_header, encoding="utf-8")
    (build_root / "directx-adapter.h").write_text(
        adapter_header,
        encoding="utf-8",
    )
    return (
        package_root,
        build_root,
        _execution_symbol(execution_header),
        _unit_symbol(execution_header),
        artifact,
    )


def _harness_source(execute_symbol: str, unit_symbol: str) -> str:
    inputs = ", ".join(f"{value}u" for value in _INPUT_VALUES)
    expected = ", ".join(f"{value}u" for value in _EXPECTED_VALUES)
    return textwrap.dedent(f"""\
        #include <cstdint>
        #include <cstdio>

        #include "execution.h"
        #include "directx-adapter.h"

        static int report_failure(
            const char *operation,
            const CrossTLDirectXNativeLoaderContext &context,
            int32_t status) {{
            std::fprintf(
                stderr,
                "%s failed: status=%d hresult=0x%08x diagnostic=%s\\n",
                operation,
                status,
                static_cast<unsigned int>(
                    crosstl_directx_native_loader_last_hresult(&context)),
                crosstl_directx_native_loader_last_diagnostic(&context));
            return 1;
        }}

        static int verify_live_pipeline_shutdown_is_rejected(
            CrossTLDirectXNativeLoaderContext &context,
            const CrossTLNativeLoaderAdapter &adapter) {{
            void *artifact = nullptr;
            int32_t status = adapter.load_artifact(
                adapter.context,
                &{unit_symbol},
                &artifact);
            if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {{
                return report_failure("lifecycle artifact load", context, status);
            }}

            const uint32_t specialization_value = {_SPECIALIZATION_VALUE}u;
            const CrossTLNativeLoaderSpecializationDescriptor specialization = {{
                1u,
                {_SPECIALIZATION_ID}u,
                "{_SPECIALIZATION_NAME}",
                "uint32",
            }};
            const CrossTLNativeLoaderSpecializationRequest request = {{
                1u,
                {_SPECIALIZATION_ID}u,
                "{_SPECIALIZATION_NAME}",
                "uint32",
                &specialization_value,
                sizeof(specialization_value),
            }};
            status = adapter.apply_specialization(
                adapter.context, artifact, &specialization, &request);
            if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {{
                adapter.unload_artifact(adapter.context, artifact);
                return report_failure(
                    "lifecycle specialization", context, status);
            }}

            void *pipeline = nullptr;
            status = adapter.create_pipeline(
                adapter.context,
                artifact,
                &{unit_symbol},
                &pipeline);
            if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {{
                adapter.unload_artifact(adapter.context, artifact);
                return report_failure("lifecycle pipeline", context, status);
            }}

            status = crosstl_directx_native_loader_context_shutdown(&context);
            if (status != CROSSTL_DIRECTX_NATIVE_LOADER_PIPELINES_ACTIVE ||
                !context.initialized ||
                context.device == nullptr ||
                context.active_pipeline_count != 1u) {{
                adapter.destroy_pipeline(adapter.context, pipeline);
                adapter.unload_artifact(adapter.context, artifact);
                std::fprintf(
                    stderr,
                    "context shutdown did not reject a live pipeline: "
                    "status=%d initialized=%d active=%zu\\n",
                    status,
                    context.initialized ? 1 : 0,
                    context.active_pipeline_count);
                return 1;
            }}

            status = adapter.destroy_pipeline(adapter.context, pipeline);
            if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK ||
                context.active_pipeline_count != 0u) {{
                adapter.unload_artifact(adapter.context, artifact);
                return report_failure(
                    "lifecycle pipeline destruction", context, status);
            }}
            status = adapter.unload_artifact(adapter.context, artifact);
            if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {{
                return report_failure("lifecycle artifact unload", context, status);
            }}
            return 0;
        }}

        int main(int argc, char **argv) {{
            if (argc != 2) {{
                std::fprintf(stderr, "expected the package root argument\\n");
                return 2;
            }}

            CrossTLDirectXNativeLoaderContext context = {{}};
            int32_t status = crosstl_directx_native_loader_context_initialize(
                &context, argv[1]);
            if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {{
                return report_failure("context initialization", context, status);
            }}

            uint32_t input_values[{_ELEMENT_COUNT}] = {{{inputs}}};
            uint32_t output_values[{_ELEMENT_COUNT}] = {{}};
            const uint32_t specialization_value = {_SPECIALIZATION_VALUE}u;
            const CrossTLNativeLoaderBindingRequest bindings[] = {{
                {{
                    "input_values",
                    "buffer",
                    "StructuredBuffer<uint>",
                    "srv",
                    0u,
                    0u,
                    CROSSTL_NATIVE_LOADER_ACCESS_READ,
                    input_values,
                    sizeof(input_values),
                }},
                {{
                    "output_values",
                    "buffer",
                    "RWStructuredBuffer<uint>",
                    "uav",
                    0u,
                    1u,
                    CROSSTL_NATIVE_LOADER_ACCESS_WRITE,
                    output_values,
                    sizeof(output_values),
                }},
            }};
            const CrossTLNativeLoaderSpecializationRequest specializations[] = {{
                {{
                    1u,
                    {_SPECIALIZATION_ID}u,
                    "{_SPECIALIZATION_NAME}",
                    "uint32",
                    &specialization_value,
                    sizeof(specialization_value),
                }},
            }};
            const CrossTLNativeLoaderExecutionRequest request = {{
                CROSSTL_NATIVE_LOADER_ABI_VERSION,
                "directx",
                2u,
                bindings,
                1u,
                specializations,
                {{
                    {{{_ELEMENT_COUNT // _WORKGROUP_SIZE}u, 1u, 1u}},
                    {{{_WORKGROUP_SIZE}u, 1u, 1u}},
                }},
            }};
            const CrossTLNativeLoaderAdapter adapter =
                crosstl_directx_native_loader_adapter(&context);
            if (verify_live_pipeline_shutdown_is_rejected(context, adapter) != 0) {{
                crosstl_directx_native_loader_context_shutdown(&context);
                return 3;
            }}
            const CrossTLNativeLoaderExecutionResult result =
                {execute_symbol}(&request, &adapter);

            if (!result.succeeded) {{
                std::fprintf(
                    stderr,
                    "execution failed: phase=%d code=%d adapter_status=%d "
                    "cleanup_code=%d\\n",
                    static_cast<int>(result.error.phase),
                    static_cast<int>(result.error.code),
                    result.error.adapter_status,
                    static_cast<int>(result.cleanup_error.code));
                report_failure(
                    "native adapter execution",
                    context,
                    result.error.adapter_status);
                crosstl_directx_native_loader_context_shutdown(&context);
                return 4;
            }}

            const uint32_t expected[] = {{{expected}}};
            for (size_t index = 0u; index < {_ELEMENT_COUNT}u; ++index) {{
                if (output_values[index] != expected[index]) {{
                    std::fprintf(
                        stderr,
                        "output[%zu] mismatch: expected=%u actual=%u\\n",
                        index,
                        expected[index],
                        output_values[index]);
                    crosstl_directx_native_loader_context_shutdown(&context);
                    return 5;
                }}
            }}

            status = crosstl_directx_native_loader_context_shutdown(&context);
            if (status != CROSSTL_DIRECTX_NATIVE_LOADER_OK) {{
                return report_failure("context shutdown", context, status);
            }}
            return 0;
        }}
        """)


def _require_device_test() -> None:
    value = os.environ.get(_RUN_DEVICE_TEST)
    if value is None:
        pytest.skip(f"set {_RUN_DEVICE_TEST}=1 to run the Direct3D 12 device test")
    if value != "1":
        pytest.fail(f"{_RUN_DEVICE_TEST} must be set to 1, got {value!r}")
    assert (
        sys.platform == "win32"
    ), f"{_RUN_DEVICE_TEST}=1 requires Windows, got {sys.platform!r}"


def _required_path(name: str) -> Path:
    value = shutil.which(name)
    assert value is not None, f"{name} must be available on PATH"
    return Path(value)


def _required_directory(environment: str) -> Path:
    value = os.environ.get(environment)
    assert value, f"{environment} must identify an existing directory"
    path = Path(value)
    assert path.is_dir(), f"{environment} does not identify a directory: {path}"
    return path


def _prepare_dxc_runtime(build_root: Path) -> Path:
    include_root = _required_directory(_DXC_INCLUDE_DIR)
    assert (
        include_root / "dxcapi.h"
    ).is_file(), f"{_DXC_INCLUDE_DIR} must contain the official dxcapi.h"

    dll_root = _required_directory(_DXC_DLL_DIR)
    for name in ("dxcompiler.dll", "dxil.dll"):
        source = dll_root / name
        assert source.is_file(), f"{_DXC_DLL_DIR} must contain {name}"
        shutil.copy2(source, build_root / name)
    return include_root


def test_directx_adapter_device_fixture_matches_generated_contract(tmp_path):
    source = _FIXTURE.read_text(encoding="utf-8")
    package_root, build_root, execute_symbol, unit_symbol, artifact = _write_package(
        tmp_path
    )
    descriptor = _descriptor(artifact)

    assert package_root.joinpath(_ARTIFACT_PATH).read_bytes() == artifact
    assert (
        descriptor["artifact"]["hash"]["value"] == hashlib.sha256(artifact).hexdigest()
    )
    assert (
        f"layout(constant_id = {_SPECIALIZATION_ID}) const uint "
        f"{_SPECIALIZATION_NAME} = {_DEFAULT_VALUE}u;"
    ) in source
    generated_hlsl = artifact.decode("utf-8")
    assert (
        f"CrossGL DirectX specialization constant id {_SPECIALIZATION_ID} "
        "fixed to its default."
    ) in generated_hlsl
    assert f"static const uint {_SPECIALIZATION_NAME} = {_DEFAULT_VALUE}u;" in (
        generated_hlsl
    )
    assert "StructuredBuffer<uint> input_values : register(t0);" in generated_hlsl
    assert "RWStructuredBuffer<uint> output_values : register(u1);" in generated_hlsl
    assert "[numthreads(4, 1, 1)]" in generated_hlsl
    assert "void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)" in generated_hlsl
    assert (
        "output_values[index] = (input_values.Load(index) * MULTIPLIER);"
        in generated_hlsl
    )
    assert _EXPECTED_VALUES != _DEFAULT_EXPECTED_VALUES

    execution_header = (build_root / "execution.h").read_text(encoding="utf-8")
    assert _ARTIFACT_PATH.as_posix() in execution_header
    assert descriptor["artifact"]["hash"]["value"] in execution_header
    assert '"input_values",' in execution_header
    assert '"StructuredBuffer<uint>",' in execution_header
    assert '"srv",' in execution_header
    assert '"output_values",' in execution_header
    assert '"RWStructuredBuffer<uint>",' in execution_header
    assert '"uav",' in execution_header
    assert (
        f'{{1u, {_SPECIALIZATION_ID}u, "{_SPECIALIZATION_NAME}", "uint32"}}'
        in execution_header
    )
    harness = _harness_source(execute_symbol, unit_symbol)
    assert execute_symbol in harness
    assert f"&{unit_symbol}" in harness


def test_generated_directx_adapter_executes_specialized_hlsl_on_device(tmp_path):
    _require_device_test()
    compiler = _required_path("cl.exe")
    package_root, build_root, execute_symbol, unit_symbol, _artifact = _write_package(
        tmp_path
    )
    include_root = _prepare_dxc_runtime(build_root)
    source_path = build_root / "directx-adapter-device.cpp"
    source_path.write_text(
        _harness_source(execute_symbol, unit_symbol),
        encoding="utf-8",
    )
    executable = build_root / "directx-adapter-device.exe"

    compile_result = subprocess.run(
        [
            str(compiler),
            "/nologo",
            "/std:c++17",
            "/EHsc",
            "/W4",
            "/WX",
            f"/I{include_root}",
            str(source_path),
            f"/Fe:{executable}",
            "/link",
            "d3d12.lib",
            "dxgi.lib",
            "dxguid.lib",
            "bcrypt.lib",
        ],
        cwd=build_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    assert executable.is_file()

    run_environment = os.environ.copy()
    run_environment["PATH"] = (
        str(build_root) + os.pathsep + run_environment.get("PATH", "")
    )
    execution_result = subprocess.run(
        [str(executable), str(package_root)],
        cwd=build_root,
        env=run_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert execution_result.returncode == 0, (
        execution_result.stdout + execution_result.stderr
    )
