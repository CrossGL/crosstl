from __future__ import annotations

import re
import shutil
import subprocess
import sys
import textwrap

import pytest

from crosstl.project.native_directx_adapter import (
    generate_directx_native_loader_adapter,
)
from crosstl.project.native_loader_abi import (
    NATIVE_LOADER_ABI_KIND,
    NATIVE_LOADER_ABI_VERSION,
    generate_native_loader_execution_abi,
)


def _execution_descriptor():
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
        "unitId": "native-directx-adapter-test",
        "target": "directx",
        "stage": "compute",
        "entryPoint": {
            "name": "CSMain",
            "stage": "compute",
            "executionConfig": {"numthreads": [1, 1, 1]},
            "provenance": {},
        },
        "artifact": {
            "packagePath": "artifacts/directx/test.dxil",
            "format": "DXIL binary",
            "hash": {"algorithm": "sha256", "value": "0" * 64},
            "sizeBytes": 4,
        },
        "source": {
            "path": "kernels/test.metal",
            "artifactPath": "out/directx/test.hlsl",
            "backend": "metal",
            "hash": None,
            "remap": None,
        },
        "bindings": [
            {
                "name": "output",
                "kind": "buffer",
                "type": "RWStructuredBuffer<uint>",
                "namespace": "uav",
                "coordinates": {"set": 0, "binding": 0},
                "access": "write",
                "scalarLayout": scalar_layout,
                "provenance": {},
            }
        ],
        "scalarLayout": {
            "constants": [],
            "bindings": [{"binding": "output", "layout": scalar_layout}],
        },
        "specializationConstants": [],
        "provenance": {},
    }


def test_directx_native_loader_adapter_generation_is_deterministic():
    first = generate_directx_native_loader_adapter()
    second = generate_directx_native_loader_adapter()

    assert first == second
    assert first.endswith("\n")
    assert "#ifndef CROSSTL_DIRECTX_NATIVE_LOADER_ADAPTER_V1_H" in first
    assert 'adapter.target = "directx";' in first
    assert "adapter.abi_version = CROSSTL_NATIVE_LOADER_ABI_VERSION;" in first
    assert (
        '#error "Include a generated CrossTL native loader execution header first"'
        in first
    )
    assert "#if !defined(_WIN32)" in first
    assert "#include <d3d12.h>" in first
    assert "#include <dxgi1_6.h>" in first
    assert "#if __has_include(<dxcapi.h>)" in first
    assert "#include <dxcapi.h>" in first


def test_directx_native_loader_adapter_encodes_native_execution_lifecycle():
    header = generate_directx_native_loader_adapter()

    for operation in (
        "CreateDXGIFactory2(",
        "D3D12CreateDevice(",
        "CreateCommandQueue(",
        "CreateCommandAllocator(",
        "CreateCommandList(",
        "D3D12SerializeRootSignature(",
        "CreateRootSignature(",
        "CreateComputePipelineState(",
        "CreateDescriptorHeap(",
        "CreateCommittedResource(",
        "CreateConstantBufferView(",
        "CreateShaderResourceView(",
        "CreateUnorderedAccessView(",
        "CopyBufferRegion(",
        "SetComputeRootDescriptorTable(",
        "pipeline->command_list->Dispatch(",
        "context->queue->ExecuteCommandLists(",
        "context->queue->Signal(",
        "SetEventOnCompletion(",
        "WaitForSingleObject(",
        "resource->readback->Map(",
        "std::memcpy(request->payload, mapped, resource->size_bytes);",
    ):
        assert operation in header

    dispatch_function = header.rindex(
        "static inline int32_t crosstl_directx_native_loader_dispatch("
    )
    native_dispatch = header.index(
        "pipeline->command_list->Dispatch(", dispatch_function
    )
    command_submission = header.index(
        "context->queue->ExecuteCommandLists(", native_dispatch
    )
    success = header.index(
        "return crosstl_directx_native_loader_succeed(context);",
        command_submission,
    )
    assert (
        "return crosstl_directx_native_loader_succeed(context);"
        not in header[dispatch_function:native_dispatch]
    )
    assert dispatch_function < native_dispatch < command_submission < success


def test_directx_native_loader_adapter_accepts_verified_hlsl_and_dxil_contracts():
    header = generate_directx_native_loader_adapter()
    load_start = header.rindex(
        "static inline int32_t crosstl_directx_native_loader_load_artifact("
    )
    load_end = header.index(
        "static inline int32_t crosstl_directx_native_loader_unload_artifact(",
        load_start,
    )
    load_artifact = header[load_start:load_end]

    assert 'unit->artifact_format, "HLSL source"' in load_artifact
    assert 'unit->artifact_format, "DXIL binary"' in load_artifact
    assert "artifact->packaged_content.resize((size_t)length);" in load_artifact
    hash_verification = load_artifact.index(
        "crosstl_directx_native_loader_verify_sha256("
    )
    dxil_container_check = load_artifact.index("if (is_dxil_binary &&")
    success = load_artifact.index("*artifact_out = artifact.release();")
    assert hash_verification < dxil_container_check < success

    pipeline_start = header.index(
        "static inline int32_t crosstl_directx_native_loader_create_pipeline("
    )
    pipeline_end = header.index(
        "static inline int32_t " "crosstl_directx_native_loader_specialization_value(",
        pipeline_start,
    )
    pipeline = header[pipeline_start:pipeline_end]
    hlsl_branch = pipeline.index("CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_HLSL_SOURCE")
    source_compile = pipeline.index(
        "crosstl_directx_native_loader_compile_hlsl(", hlsl_branch
    )
    dxil_branch = pipeline.index(
        "CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_DXIL_BINARY",
        source_compile,
    )
    pipeline_creation = pipeline.index("CreateComputePipelineState(")
    assert hlsl_branch < source_compile < dxil_branch < pipeline_creation
    assert "shader_bytecode = &artifact->compiled_dxil;" in pipeline
    assert "shader_bytecode = &artifact->packaged_content;" in pipeline
    assert "artifact->compiled_entry_point != unit->entry_point" in pipeline


def test_directx_native_loader_adapter_encodes_dxc_compile_and_diagnostics():
    header = generate_directx_native_loader_adapter()

    for contract in (
        'LoadLibraryExW(\n        L"dxcompiler.dll"',
        'GetProcAddress(\n        context->dxcompiler_module, "DxcCreateInstance")',
        "DxcCreateInstanceProc",
        "ComPtr<IDxcUtils> utilities;",
        "ComPtr<IDxcCompiler3> compiler;",
        "utilities->BuildArguments(",
        "compiler->Compile(",
        "DXC_OUT_ERRORS",
        "diagnostics->GetStringPointer()",
        "diagnostics->GetStringLength()",
        "compile_result->GetStatus(&compilation_status)",
        "DXC_OUT_OBJECT",
        "crosstl_directx_native_loader_last_diagnostic(",
    ):
        assert contract in header

    assert (
        "#define CROSSTL_DIRECTX_NATIVE_LOADER_DXC_COMPUTE_PROFILE " 'L"cs_6_2"'
    ) in header
    assert 'L"-enable-16bit-types"' in header
    assert 'L"-HV"' in header
    assert 'L"2021"' in header
    assert "entry_point.c_str()," in header
    assert "include_root.c_str()," in header

    compile_start = header.index(
        "static inline int32_t crosstl_directx_native_loader_compile_hlsl("
    )
    source_preparation = header.index(
        "crosstl_directx_native_loader_prepare_hlsl_source(", compile_start
    )
    build_arguments = header.index("utilities->BuildArguments(", compile_start)
    compile_call = header.index("compiler->Compile(", build_arguments)
    diagnostic_output = header.index("DXC_OUT_ERRORS", compile_call)
    compilation_status = header.index(
        "compile_result->GetStatus(&compilation_status)", diagnostic_output
    )
    object_output = header.index("DXC_OUT_OBJECT", compilation_status)
    bytecode_copy = header.index("artifact->compiled_dxil.assign(", object_output)
    entry_point_cache = header.index(
        "artifact->compiled_entry_point.assign(unit->entry_point)", bytecode_copy
    )
    assert (
        compile_start
        < source_preparation
        < build_arguments
        < compile_call
        < diagnostic_output
        < compilation_status
        < object_output
        < bytecode_copy
        < entry_point_cache
    )


def test_directx_native_loader_adapter_encodes_named_dxc_definitions():
    header = generate_directx_native_loader_adapter()

    for encoding in (
        'descriptor->type_name, "bool"',
        '*value_out = value ? L"true" : L"false";',
        'descriptor->type_name, "int32"',
        '*value_out = L"(-2147483647 - 1)";',
        'descriptor->type_name, "uint32"',
        '*value_out = std::to_wstring(value) + L"u";',
        'descriptor->type_name, "float32"',
        "std::numeric_limits<float>::max_digits10",
        'encoded += "f";',
        "DxcDefine definition = {};",
        "definition.Name = specialization.name.c_str();",
        "definition.Value = specialization.value.c_str();",
        "definitions.empty() ? NULL : definitions.data()",
        '"CrossGL DirectX specialization constant id "',
        '"__crosstl_materialized_" + specialization.source_name + "_"',
        "source_out->replace(",
    ):
        assert encoding in header

    apply_start = header.index(
        "static inline int32_t " "crosstl_directx_native_loader_apply_specialization("
    )
    apply_end = header.index(
        "static inline size_t crosstl_directx_native_loader_binding_slot(",
        apply_start,
    )
    specialization = header[apply_start:apply_end]
    assert (
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_PRECOMPILED_DXIL"
        in specialization
    )
    assert (
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_NAME_REQUIRED" in specialization
    )
    assert "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_NAME_INVALID" in specialization
    assert "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_DUPLICATE" in specialization
    assert "existing.name == name" in specialization
    assert "existing.id == descriptor->id" in specialization
    assert "artifact->specializations.push_back(" in specialization


def test_directx_native_loader_adapter_rejects_fail_closed_contracts():
    header = generate_directx_native_loader_adapter()

    for status in (
        "CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_FORMAT_UNSUPPORTED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_NOT_DXIL",
        "CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_SIZE_MISMATCH",
        "CROSSTL_DIRECTX_NATIVE_LOADER_ARTIFACT_HASH_MISMATCH",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_NAME_REQUIRED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_NAME_INVALID",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_TYPE_UNSUPPORTED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_PAYLOAD_INVALID",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_DUPLICATE",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_PRECOMPILED_DXIL",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_AFTER_COMPILE",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SPECIALIZATION_SOURCE_REWRITE_FAILED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_DXC_API_UNAVAILABLE",
        "CROSSTL_DIRECTX_NATIVE_LOADER_DXC_LIBRARY_UNAVAILABLE",
        "CROSSTL_DIRECTX_NATIVE_LOADER_DXC_ENTRY_POINT_UNAVAILABLE",
        "CROSSTL_DIRECTX_NATIVE_LOADER_DXC_INSTANCE_CREATION_FAILED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_DXC_ARGUMENT_BUILD_FAILED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_DXC_COMPILE_CALL_FAILED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_DXC_COMPILATION_FAILED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_DXC_OUTPUT_MISSING",
        "CROSSTL_DIRECTX_NATIVE_LOADER_DXC_OUTPUT_INVALID",
        "CROSSTL_DIRECTX_NATIVE_LOADER_STAGE_UNSUPPORTED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_RESOURCE_KIND_UNSUPPORTED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_NAMESPACE_UNSUPPORTED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_BINDING_ACCESS_UNSUPPORTED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_BUFFER_TYPE_UNSUPPORTED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SCALAR_LAYOUT_UNSUPPORTED",
        "CROSSTL_DIRECTX_NATIVE_LOADER_BINDINGS_INCOMPLETE",
        "CROSSTL_DIRECTX_NATIVE_LOADER_SYNCHRONIZE_BEFORE_DISPATCH",
        "CROSSTL_DIRECTX_NATIVE_LOADER_READBACK_BEFORE_SYNCHRONIZE",
    ):
        assert status in header

    assert "BCryptOpenAlgorithmProvider(" in header
    assert "BCRYPT_SHA256_ALGORITHM" in header
    assert (
        "std::memcmp(\n                 artifact->packaged_content.data(), "
        '"DXBC", 4u)'
    ) in header


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="The portable failure path is exercised on non-Windows platforms.",
)
def test_directx_native_loader_adapter_portable_path_fails_closed(tmp_path):
    compiler = next(
        (
            executable
            for name in ("c++", "clang++", "g++")
            if (executable := shutil.which(name)) is not None
        ),
        None,
    )
    if compiler is None:
        pytest.skip("A C++17 compiler is unavailable.")

    execution_header = generate_native_loader_execution_abi(_execution_descriptor())
    execute_match = re.search(
        r"static inline CrossTLNativeLoaderExecutionResult\s+"
        r"([A-Za-z_]\w*_execute)\(",
        execution_header,
    )
    assert execute_match is not None
    execute_function = execute_match.group(1)

    (tmp_path / "execution.h").write_text(execution_header, encoding="utf-8")
    (tmp_path / "directx-adapter.h").write_text(
        generate_directx_native_loader_adapter(),
        encoding="utf-8",
    )
    source = textwrap.dedent(f"""
        #include "execution.h"
        #include "directx-adapter.h"

        int main() {{
            CrossTLDirectXNativeLoaderContext context = {{}};
            int32_t status =
                crosstl_directx_native_loader_context_initialize(&context, ".");
            if (status !=
                CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE) {{
                return 1;
            }}
            if (crosstl_directx_native_loader_is_available() != 0) {{
                return 2;
            }}

            uint32_t output = 0u;
            CrossTLNativeLoaderBindingRequest binding = {{
                "output",
                "buffer",
                "RWStructuredBuffer<uint>",
                "uav",
                0u,
                0u,
                CROSSTL_NATIVE_LOADER_ACCESS_WRITE,
                &output,
                sizeof(output)
            }};
            CrossTLNativeLoaderExecutionRequest request = {{
                CROSSTL_NATIVE_LOADER_ABI_VERSION,
                "directx",
                1u,
                &binding,
                0u,
                NULL,
                {{{{1u, 1u, 1u}}, {{1u, 1u, 1u}}}}
            }};
            CrossTLNativeLoaderAdapter adapter =
                crosstl_directx_native_loader_adapter(&context);
            CrossTLNativeLoaderExecutionResult result =
                {execute_function}(&request, &adapter);
            if (result.succeeded != 0 ||
                result.error.phase !=
                    CROSSTL_NATIVE_LOADER_PHASE_LOAD_ARTIFACT ||
                result.error.code !=
                    CROSSTL_NATIVE_LOADER_CODE_ADAPTER_FAILURE ||
                result.error.adapter_status !=
                    CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE ||
                result.cleanup_error.code !=
                    CROSSTL_NATIVE_LOADER_CODE_OK) {{
                return 3;
            }}

            void *pipeline = (void *)1;
            status = adapter.create_pipeline(
                adapter.context, NULL, NULL, &pipeline);
            if (status !=
                    CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE ||
                pipeline != NULL) {{
                return 4;
            }}
            void *resource = (void *)1;
            status = adapter.bind_resource(
                adapter.context, NULL, NULL, NULL, &resource);
            if (status !=
                    CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE ||
                resource != NULL) {{
                return 5;
            }}
            if (adapter.dispatch(adapter.context, NULL, NULL) !=
                    CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE ||
                adapter.synchronize(adapter.context, NULL) !=
                    CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE ||
                adapter.readback(adapter.context, NULL, NULL, NULL) !=
                    CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE) {{
                return 6;
            }}
            if (crosstl_directx_native_loader_context_shutdown(&context) !=
                CROSSTL_DIRECTX_NATIVE_LOADER_PLATFORM_UNAVAILABLE) {{
                return 7;
            }}
            return 0;
        }}
        """)
    source_path = tmp_path / "portable-failure.cpp"
    executable_path = tmp_path / "portable-failure"
    source_path.write_text(source, encoding="utf-8")

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source_path),
            "-o",
            str(executable_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    run_result = subprocess.run(
        [str(executable_path)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr


@pytest.mark.parametrize("dxc_api", ("enabled", "disabled"))
def test_directx_native_loader_adapter_cross_compiles_for_windows(tmp_path, dxc_api):
    compiler = shutil.which("x86_64-w64-mingw32-g++")
    if compiler is None:
        pytest.skip("The MinGW-w64 C++ cross-compiler is unavailable.")

    (tmp_path / "execution.h").write_text(
        generate_native_loader_execution_abi(_execution_descriptor()),
        encoding="utf-8",
    )
    (tmp_path / "directx-adapter.h").write_text(
        generate_directx_native_loader_adapter(),
        encoding="utf-8",
    )
    source_path = tmp_path / "windows-compile.cpp"
    source_path.write_text(
        textwrap.dedent("""
            #include "execution.h"
            #include "directx-adapter.h"

            int main() {
                CrossTLDirectXNativeLoaderContext context = {};
                CrossTLNativeLoaderAdapter adapter =
                    crosstl_directx_native_loader_adapter(&context);
                return adapter.abi_version ==
                    CROSSTL_NATIVE_LOADER_ABI_VERSION ? 0 : 1;
            }
            """),
        encoding="utf-8",
    )
    command = [
        compiler,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unknown-pragmas",
    ]
    if dxc_api == "disabled":
        command.append("-DCROSSTL_DIRECTX_NATIVE_LOADER_DISABLE_DXC_API=1")
    command.extend(
        [
            "-c",
            str(source_path),
            "-o",
            str(tmp_path / f"windows-compile-{dxc_api}.o"),
        ]
    )
    result = subprocess.run(
        command,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
