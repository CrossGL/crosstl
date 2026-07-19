import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from crosstl.project.native_loader_abi import (
    NATIVE_LOADER_ABI_KIND,
    NATIVE_LOADER_ABI_VERSION,
    NativeLoaderABIError,
    build_native_loader_abi_descriptor,
    generate_native_loader_declarations,
    generate_native_loader_execution_abi,
)
from crosstl.project.pipeline import (
    REPORT_SCHEMA_VERSION,
    RUNTIME_LOADER_MANIFEST_KIND,
    RUNTIME_LOADER_MANIFEST_LOAD_UNIT_FIELDS,
)


def _load_unit(target="directx"):
    entry_point = "CSMain" if target == "directx" else "main"
    artifact_format = "HLSL source" if target == "directx" else "GLSL source"
    extension = "hlsl" if target == "directx" else "comp"
    unit = {
        "id": f"copy:{target}",
        "target": target,
        "adapterKind": f"{target}-compute-adapter",
        "artifactFormat": artifact_format,
        "packagePath": f"artifacts/{target}/copy.{extension}",
        "source": "kernels/copy.metal",
        "sourcePath": f"out/{target}/copy.{extension}",
        "sourceBackend": "metal",
        "stage": "compute",
        "variant": "float32",
        "defines": {"BLOCK_SIZE": "64"},
        "entryPoint": {
            "source": "copy_float32",
            "target": entry_point,
            "stage": "compute",
        },
        "templateMaterialization": None,
        "specializationMaterialization": None,
        "specializationConstants": [
            {
                "id": 3,
                "name": "vector_width",
                "dtype": "uint32",
                "value": 4,
                "provenance": {"source": "dispatch-contract"},
            }
        ],
        "provenance": {
            "pipeline": "metal-to-crossgl",
            "source": "kernels/copy.metal",
            "target": target,
        },
        "sourceHash": {"algorithm": "sha256", "value": "1" * 64},
        "sourceSizeBytes": 512,
        "hash": {"algorithm": "sha256", "value": "2" * 64},
        "sizeBytes": 1024,
        "sourceRemap": {
            "sourcePath": "out/copy.source-remap.json",
            "packagePath": "source-remaps/copy.source-remap.json",
            "hash": {"algorithm": "sha256", "value": "3" * 64},
            "sizeBytes": 128,
        },
        "hostInterface": {
            "status": "ready",
            "source": "package-artifact",
            "parser": f"{target}-reflection",
            "artifactFormat": artifact_format,
            "entryPointCount": 1,
            "resourceCount": 2,
            "constantCount": 1,
            "specializationConstantCount": 1,
            "entryPoints": [
                {
                    "name": entry_point,
                    "stage": "compute",
                    "executionConfig": {"workgroupSize": [64, 1, 1]},
                }
            ],
            "resources": [
                {
                    "name": "output_values",
                    "kind": "buffer",
                    "type": "float[]",
                    "set": 0,
                    "binding": 1,
                    "access": "write",
                    "scalarLayout": {
                        "elementType": "float32",
                        "elementSizeBytes": 4,
                        "elementStrideBytes": 4,
                    },
                    "provenance": {"parameter": 1},
                },
                {
                    "name": "input_values",
                    "kind": "buffer",
                    "type": "float[]",
                    "set": 0,
                    "binding": 0,
                    "access": "read",
                    "scalarLayout": {
                        "elementType": "float32",
                        "elementSizeBytes": 4,
                        "elementStrideBytes": 4,
                    },
                    "provenance": {"parameter": 0},
                },
            ],
            "constants": [
                {
                    "name": "element_count",
                    "kind": "scalar-parameter",
                    "dtype": "uint32",
                    "byteOffset": 0,
                    "byteSize": 4,
                    "alignment": 4,
                }
            ],
            "specializationConstants": [
                {
                    "id": 3,
                    "name": "vector_width",
                    "dtype": "uint32",
                    "required": True,
                }
            ],
            "diagnostics": [],
            "diagnosticRecords": [],
        },
        "requiredTools": ["dxc" if target == "directx" else "glslangValidator"],
        "hostResponsibilities": ["Provide input and output buffer storage."],
        "loadSteps": [
            {
                "kind": "load-package-artifact",
                "message": "Load the translated compute artifact.",
                "target": target,
                "packagePath": f"artifacts/{target}/copy.{extension}",
                "tools": [],
                "command": None,
                "hostInterfaceStatus": None,
                "metadata": {},
            }
        ],
        "blockers": [],
        "validation": {"hostInterface": "ready", "loadReady": True},
    }
    assert set(unit) == RUNTIME_LOADER_MANIFEST_LOAD_UNIT_FIELDS
    return unit


def _loader_manifest(*units):
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "kind": RUNTIME_LOADER_MANIFEST_KIND,
        "success": True,
        "loadUnits": list(units),
    }


def test_native_loader_abi_output_is_deterministic_from_loader_manifest():
    unit = _load_unit()
    manifest = _loader_manifest(unit)
    reordered_unit = {
        key: copy.deepcopy(unit[key]) for key in reversed(tuple(unit.keys()))
    }
    reordered_unit["hostInterface"]["resources"].reverse()
    reordered_unit["defines"] = {
        key: reordered_unit["defines"][key]
        for key in reversed(tuple(reordered_unit["defines"]))
    }

    first = build_native_loader_abi_descriptor(manifest)
    second = build_native_loader_abi_descriptor(_loader_manifest(reordered_unit))
    first_header = generate_native_loader_declarations(first)
    second_header = generate_native_loader_declarations(second)
    first_execution_abi = generate_native_loader_execution_abi(first)
    second_execution_abi = generate_native_loader_execution_abi(second)

    assert first == second
    assert first_header == second_header
    assert first_execution_abi == second_execution_abi
    assert first["kind"] == NATIVE_LOADER_ABI_KIND
    assert first["abiVersion"] == NATIVE_LOADER_ABI_VERSION
    assert first["artifact"] == {
        "packagePath": "artifacts/directx/copy.hlsl",
        "format": "HLSL source",
        "hash": {"algorithm": "sha256", "value": "2" * 64},
        "sizeBytes": 1024,
    }
    assert [binding["name"] for binding in first["bindings"]] == [
        "input_values",
        "output_values",
    ]
    assert first["scalarLayout"]["constants"][0]["byteOffset"] == 0
    assert first["scalarLayout"]["bindings"][0] == {
        "binding": "input_values",
        "layout": {
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "elementType": "float32",
        },
    }
    assert first["provenance"] == unit["provenance"]
    assert "typedef struct CrossTLNativeLoaderUnitDescriptor" in first_header
    assert "#ifndef CROSSTL_NATIVE_LOADER_ABI_V1_TYPES" in first_header
    assert '"directx"' in first_header
    assert '"CSMain"' in first_header
    assert f'"{"2" * 64}"' in first_header
    assert "const char *source_artifact_path;" in first_header
    assert "const char *source_hash_algorithm;" in first_header
    assert "const char *source_hash_value;" in first_header
    assert "const char *source_remap_json;" in first_header
    assert '"out/directx/copy.hlsl"' in first_header
    assert f'"{"1" * 64}"' in first_header
    assert r"\"packagePath\":\"source-remaps/copy.source-remap.json\"" in first_header
    assert first_header.endswith("\n")
    assert first_execution_abi.startswith(first_header)
    assert "typedef struct CrossTLNativeLoaderAdapter" in first_execution_abi
    assert "CROSSTL_NATIVE_LOADER_CODE_TARGET_MISMATCH = 3" in first_execution_abi
    assert "CROSSTL_NATIVE_LOADER_PHASE_CLEANUP_ARTIFACT = 11" in first_execution_abi
    assert "malloc(" not in first_execution_abi
    assert "calloc(" not in first_execution_abi
    assert "realloc(" not in first_execution_abi
    assert "free(" not in first_execution_abi


def test_directx_and_opengl_descriptors_share_one_host_binding_contract():
    directx = build_native_loader_abi_descriptor(_load_unit("directx"))
    opengl = build_native_loader_abi_descriptor(_load_unit("opengl"))

    assert directx["target"] == "directx"
    assert directx["entryPoint"]["name"] == "CSMain"
    assert opengl["target"] == "opengl"
    assert opengl["entryPoint"]["name"] == "main"
    for index in range(2):
        directx_binding = directx["bindings"][index]
        opengl_binding = opengl["bindings"][index]
        for field in ("name", "kind", "type", "coordinates", "access", "scalarLayout"):
            assert directx_binding[field] == opengl_binding[field]
    assert [binding["namespace"] for binding in directx["bindings"]] == ["srv", "uav"]
    assert [binding["namespace"] for binding in opengl["bindings"]] == [
        "storage-buffer",
        "storage-buffer",
    ]
    assert directx["specializationConstants"][0]["id"] == 3
    assert opengl["specializationConstants"][0]["id"] == 3
    assert '"directx"' in generate_native_loader_declarations(directx)
    assert '"opengl"' in generate_native_loader_declarations(opengl)


def test_generated_unit_headers_share_one_guarded_abi_type_contract():
    directx_header = generate_native_loader_declarations(
        build_native_loader_abi_descriptor(_load_unit("directx"))
    )
    opengl_header = generate_native_loader_declarations(
        build_native_loader_abi_descriptor(_load_unit("opengl"))
    )

    assert directx_header.count("typedef enum CrossTLNativeLoaderAccess") == 1
    assert opengl_header.count("typedef enum CrossTLNativeLoaderAccess") == 1
    assert "CROSSTL_NATIVE_LOADER_ABI_V1_TYPES" in directx_header
    assert "CROSSTL_NATIVE_LOADER_ABI_V1_TYPES" in opengl_header


def test_inspected_source_remap_identity_is_preserved():
    unit = _load_unit()
    unit["sourceRemap"] = {
        "sourcePath": "translated/directx/copy.source-remap.json",
        "packagePath": "source-remaps/translated/directx/copy.source-remap.json",
        "status": "ready",
    }

    descriptor = build_native_loader_abi_descriptor(unit)
    header = generate_native_loader_declarations(descriptor)

    assert descriptor["source"]["remap"] == unit["sourceRemap"]
    assert r"\"status\":\"ready\"" in header


def _native_compiler(language):
    environment_name = "CC" if language == "c" else "CXX"
    fallback_names = (
        ("clang", "gcc", "cc", "cl", "clang-cl")
        if language == "c"
        else ("clang++", "g++", "c++", "cl", "clang-cl")
    )
    candidates = []
    configured = os.environ.get(environment_name)
    if configured:
        candidates.append(shlex.split(configured, posix=os.name != "nt"))
    candidates.extend([[name] for name in fallback_names])
    for command in candidates:
        if command and shutil.which(command[0]):
            command[0] = shutil.which(command[0])
            return command
    return None


def _native_compile_command(
    compiler, language, source_path, output_path, *, link=False
):
    executable_names = {
        Path(part).name.lower() for part in compiler if not part.startswith("-")
    }
    msvc_style = bool(executable_names & {"cl", "cl.exe", "clang-cl", "clang-cl.exe"})
    if msvc_style:
        standard = "/std:c11" if language == "c" else "/std:c++17"
        command = compiler + ["/nologo", standard, "/W4", "/WX"]
        if not link:
            command.append("/c")
        command.append(str(source_path))
        command.append(f"/Fe{output_path}" if link else f"/Fo{output_path}")
        return command
    standard = "-std=c11" if language == "c" else "-std=c++17"
    command = compiler + [
        standard,
        "-pedantic-errors",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(source_path),
        "-o",
        str(output_path),
    ]
    if not link:
        command.insert(-3, "-c")
    return command


@pytest.mark.parametrize("language", ("c", "c++"), ids=("c11", "cxx17"))
def test_generated_source_identity_fields_compile(tmp_path, language):
    compiler = _native_compiler(language)
    if compiler is None:
        pytest.skip(f"No native {language} compiler is available")

    header = generate_native_loader_declarations(
        build_native_loader_abi_descriptor(_load_unit())
    )
    symbol_match = re.search(
        r"static const CrossTLNativeLoaderUnitDescriptor ([A-Za-z_]\w*) = \{",
        header,
    )
    assert symbol_match is not None
    symbol = symbol_match.group(1)
    header_path = tmp_path / "native_loader.h"
    header_path.write_text(header, encoding="utf-8")
    suffix = ".c" if language == "c" else ".cpp"
    source_path = tmp_path / f"source_identity{suffix}"
    source_path.write_text(
        textwrap.dedent(f"""
            #include "native_loader.h"

            int main(void) {{
                const CrossTLNativeLoaderUnitDescriptor *descriptor = &{symbol};
                return descriptor->source_artifact_path == NULL ||
                       descriptor->source_hash_algorithm == NULL ||
                       descriptor->source_hash_value == NULL ||
                       descriptor->source_remap_json == NULL;
            }}
            """).strip() + "\n",
        encoding="utf-8",
    )

    object_suffix = ".obj" if os.name == "nt" else ".o"
    object_path = tmp_path / f"source_identity{object_suffix}"
    command = _native_compile_command(compiler, language, source_path, object_path)

    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert object_path.is_file()


@pytest.mark.parametrize("language", ("c", "c++"), ids=("c11", "cxx17"))
def test_generated_execution_abi_compiles(tmp_path, language):
    compiler = _native_compiler(language)
    if compiler is None:
        pytest.skip(f"No native {language} compiler is available")

    execution_abi = generate_native_loader_execution_abi(
        build_native_loader_abi_descriptor(_load_unit())
    )
    symbol_match = re.search(
        r"static inline CrossTLNativeLoaderExecutionResult\s+"
        r"([A-Za-z_]\w*_execute)\(",
        execution_abi,
    )
    assert symbol_match is not None
    execute_symbol = symbol_match.group(1)
    header_path = tmp_path / "native_loader_execution.h"
    header_path.write_text(execution_abi, encoding="utf-8")
    suffix = ".c" if language == "c" else ".cpp"
    source_path = tmp_path / f"execution_abi{suffix}"
    source_path.write_text(
        textwrap.dedent(f"""
            #include "native_loader_execution.h"

            int main(void) {{
                CrossTLNativeLoaderExecutionResult result =
                    {execute_symbol}(NULL, NULL);
                return result.error.code ==
                    CROSSTL_NATIVE_LOADER_CODE_INVALID_ARGUMENT ? 0 : 1;
            }}
            """).strip() + "\n",
        encoding="utf-8",
    )
    object_suffix = ".obj" if os.name == "nt" else ".o"
    object_path = tmp_path / f"execution_abi{object_suffix}"
    command = _native_compile_command(compiler, language, source_path, object_path)

    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert object_path.is_file()


def test_generated_execution_abi_runs_fake_adapter_and_cleans_up(tmp_path):
    compiler = _native_compiler("c")
    if compiler is None:
        pytest.skip("No native C compiler is available")

    execution_abi = generate_native_loader_execution_abi(
        build_native_loader_abi_descriptor(_load_unit())
    )
    symbol_match = re.search(
        r"static inline CrossTLNativeLoaderExecutionResult\s+"
        r"([A-Za-z_]\w*_execute)\(",
        execution_abi,
    )
    assert symbol_match is not None
    execute_symbol = symbol_match.group(1)
    header_path = tmp_path / "native_loader_execution.h"
    header_path.write_text(execution_abi, encoding="utf-8")
    source_path = tmp_path / "fake_adapter.c"
    source_path.write_text(
        textwrap.dedent(f"""
            #include "native_loader_execution.h"

            typedef struct FakeContext {{
                int events[32];
                size_t event_count;
                int fail_binding;
                int fail_release;
                int artifact;
                int pipeline;
                int resources[2];
            }} FakeContext;

            static void record_event(FakeContext *context, int event) {{
                context->events[context->event_count++] = event;
            }}

            static int32_t fake_load_artifact(
                void *opaque,
                const CrossTLNativeLoaderUnitDescriptor *unit,
                void **artifact_out) {{
                FakeContext *context = (FakeContext *)opaque;
                record_event(context, 10);
                if (unit == NULL || artifact_out == NULL ||
                    !crosstl_native_loader_strings_equal(unit->target, "directx")) {{
                    return 11;
                }}
                *artifact_out = &context->artifact;
                return 0;
            }}

            static int32_t fake_unload_artifact(void *opaque, void *artifact) {{
                FakeContext *context = (FakeContext *)opaque;
                record_event(context, 100);
                return artifact == &context->artifact ? 0 : 12;
            }}

            static int32_t fake_create_pipeline(
                void *opaque,
                void *artifact,
                const CrossTLNativeLoaderUnitDescriptor *unit,
                void **pipeline_out) {{
                FakeContext *context = (FakeContext *)opaque;
                record_event(context, 20);
                if (artifact != &context->artifact || unit == NULL ||
                    pipeline_out == NULL) {{
                    return 21;
                }}
                *pipeline_out = &context->pipeline;
                return 0;
            }}

            static int32_t fake_destroy_pipeline(void *opaque, void *pipeline) {{
                FakeContext *context = (FakeContext *)opaque;
                record_event(context, 90);
                return pipeline == &context->pipeline ? 0 : 22;
            }}

            static int32_t fake_apply_specialization(
                void *opaque,
                void *artifact,
                const CrossTLNativeLoaderSpecializationDescriptor *descriptor,
                const CrossTLNativeLoaderSpecializationRequest *request) {{
                FakeContext *context = (FakeContext *)opaque;
                record_event(context, 30 + (int)descriptor->id);
                if (artifact != &context->artifact || descriptor->has_id != 1u ||
                    descriptor->id != 3u || request->payload_size_bytes != 4u) {{
                    return 31;
                }}
                return 0;
            }}

            static int32_t fake_bind_resource(
                void *opaque,
                void *pipeline,
                const CrossTLNativeLoaderBindingDescriptor *descriptor,
                const CrossTLNativeLoaderBindingRequest *request,
                void **resource_out) {{
                FakeContext *context = (FakeContext *)opaque;
                size_t index = descriptor->binding_index;
                record_event(context, 40 + (int)index);
                if (pipeline != &context->pipeline || index >= 2u ||
                    request->payload == NULL || resource_out == NULL) {{
                    return 41;
                }}
                *resource_out = &context->resources[index];
                if ((int)index == context->fail_binding) {{
                    return 73;
                }}
                return 0;
            }}

            static int32_t fake_release_resource(
                void *opaque,
                void *resource,
                const CrossTLNativeLoaderBindingDescriptor *descriptor) {{
                FakeContext *context = (FakeContext *)opaque;
                size_t index = descriptor->binding_index;
                record_event(context, 80 + (int)index);
                if (index >= 2u || resource != &context->resources[index]) {{
                    return 81;
                }}
                return (int)index == context->fail_release ? 91 : 0;
            }}

            static int32_t fake_dispatch(
                void *opaque,
                void *pipeline,
                const CrossTLNativeLoaderDispatchGeometry *geometry) {{
                FakeContext *context = (FakeContext *)opaque;
                record_event(context, 50);
                if (pipeline != &context->pipeline ||
                    geometry->workgroup_count[0] != 1u ||
                    geometry->workgroup_size[0] != 64u) {{
                    return 51;
                }}
                return 0;
            }}

            static int32_t fake_synchronize(void *opaque, void *pipeline) {{
                FakeContext *context = (FakeContext *)opaque;
                record_event(context, 60);
                return pipeline == &context->pipeline ? 0 : 61;
            }}

            static int32_t fake_readback(
                void *opaque,
                void *resource,
                const CrossTLNativeLoaderBindingDescriptor *descriptor,
                const CrossTLNativeLoaderBindingRequest *request) {{
                FakeContext *context = (FakeContext *)opaque;
                size_t index = descriptor->binding_index;
                record_event(context, 70 + (int)index);
                if (index != 1u || resource != &context->resources[index] ||
                    request->payload_size_bytes < sizeof(float)) {{
                    return 71;
                }}
                *((float *)request->payload) = 50.0f;
                return 0;
            }}

            static int check_events(
                const FakeContext *context,
                const int *expected,
                size_t expected_count) {{
                size_t index;
                if (context->event_count != expected_count) {{
                    return 0;
                }}
                for (index = 0u; index < expected_count; ++index) {{
                    if (context->events[index] != expected[index]) {{
                        return 0;
                    }}
                }}
                return 1;
            }}

            int main(void) {{
                float input_values[4] = {{0.0f, 1.0f, 7.0f, 42.0f}};
                float output_values[1] = {{0.0f}};
                uint32_t vector_width = 4u;
                CrossTLNativeLoaderBindingRequest bindings[2] = {{
                    {{
                        "input_values", "buffer", "float[]", "srv", 0u, 0u,
                        CROSSTL_NATIVE_LOADER_ACCESS_READ,
                        input_values, sizeof(input_values)
                    }},
                    {{
                        "output_values", "buffer", "float[]", "uav", 0u, 1u,
                        CROSSTL_NATIVE_LOADER_ACCESS_WRITE,
                        output_values, sizeof(output_values)
                    }}
                }};
                CrossTLNativeLoaderSpecializationRequest specializations[1] = {{
                    {{
                        1u, 3u, "vector_width", "uint32",
                        &vector_width, sizeof(vector_width)
                    }}
                }};
                CrossTLNativeLoaderExecutionRequest request = {{
                    CROSSTL_NATIVE_LOADER_ABI_VERSION,
                    "directx",
                    2u,
                    bindings,
                    1u,
                    specializations,
                    {{{{1u, 1u, 1u}}, {{64u, 1u, 1u}}}}
                }};
                FakeContext context = {{{{0}}, 0u, -1, -1, 0, 0, {{0, 0}}}};
                CrossTLNativeLoaderAdapter adapter = {{
                    CROSSTL_NATIVE_LOADER_ABI_VERSION,
                    "directx",
                    &context,
                    fake_load_artifact,
                    fake_unload_artifact,
                    fake_create_pipeline,
                    fake_destroy_pipeline,
                    fake_apply_specialization,
                    fake_bind_resource,
                    fake_release_resource,
                    fake_dispatch,
                    fake_synchronize,
                    fake_readback
                }};
                CrossTLNativeLoaderExecutionResult result;
                const int success_events[] = {{
                    10, 33, 20, 40, 41, 50, 60, 71, 81, 80, 90, 100
                }};
                const int failure_events[] = {{
                    10, 33, 20, 40, 41, 81, 80, 90, 100
                }};

                request.abi_version = 2u;
                result = {execute_symbol}(&request, &adapter);
                if (result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_ABI_VERSION_MISMATCH ||
                    context.event_count != 0u) {{
                    return 1;
                }}
                request.abi_version = CROSSTL_NATIVE_LOADER_ABI_VERSION;
                request.target = "opengl";
                result = {execute_symbol}(&request, &adapter);
                if (result.error.code != CROSSTL_NATIVE_LOADER_CODE_TARGET_MISMATCH ||
                    context.event_count != 0u) {{
                    return 2;
                }}
                request.target = "directx";
                bindings[0].access = CROSSTL_NATIVE_LOADER_ACCESS_WRITE;
                result = {execute_symbol}(&request, &adapter);
                if (result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_BINDING_ACCESS_MISMATCH ||
                    result.error.binding_index != 0u || context.event_count != 0u) {{
                    return 3;
                }}
                bindings[0].access = CROSSTL_NATIVE_LOADER_ACCESS_READ;
                specializations[0].type_name = "int32";
                result = {execute_symbol}(&request, &adapter);
                if (result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_SPECIALIZATION_TYPE_MISMATCH ||
                    result.error.specialization_index != 0u ||
                    context.event_count != 0u) {{
                    return 4;
                }}
                specializations[0].type_name = "uint32";
                request.dispatch.workgroup_size[0] = 32u;
                result = {execute_symbol}(&request, &adapter);
                if (result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_WORKGROUP_SIZE_MISMATCH ||
                    context.event_count != 0u) {{
                    return 5;
                }}
                request.dispatch.workgroup_size[0] = 64u;

                result = {execute_symbol}(&request, &adapter);
                if (!result.succeeded ||
                    result.error.code != CROSSTL_NATIVE_LOADER_CODE_OK ||
                    result.cleanup_error.code != CROSSTL_NATIVE_LOADER_CODE_OK ||
                    output_values[0] != 50.0f ||
                    !check_events(
                        &context, success_events,
                        sizeof(success_events) / sizeof(success_events[0]))) {{
                    return 6;
                }}

                context.event_count = 0u;
                context.fail_binding = 1;
                output_values[0] = 0.0f;
                result = {execute_symbol}(&request, &adapter);
                if (result.succeeded ||
                    result.error.phase != CROSSTL_NATIVE_LOADER_PHASE_BIND_RESOURCE ||
                    result.error.code != CROSSTL_NATIVE_LOADER_CODE_ADAPTER_FAILURE ||
                    result.error.binding_index != 1u ||
                    result.error.adapter_status != 73 ||
                    result.cleanup_error.code != CROSSTL_NATIVE_LOADER_CODE_OK ||
                    output_values[0] != 0.0f ||
                    !check_events(
                        &context, failure_events,
                        sizeof(failure_events) / sizeof(failure_events[0]))) {{
                    return 7;
                }}

                context.event_count = 0u;
                context.fail_binding = -1;
                context.fail_release = 1;
                result = {execute_symbol}(&request, &adapter);
                if (result.succeeded ||
                    result.error.phase !=
                        CROSSTL_NATIVE_LOADER_PHASE_CLEANUP_RESOURCE ||
                    result.error.binding_index != 1u ||
                    result.error.adapter_status != 91 ||
                    result.cleanup_error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_ADAPTER_FAILURE ||
                    !check_events(
                        &context, success_events,
                        sizeof(success_events) / sizeof(success_events[0]))) {{
                    return 8;
                }}
                return 0;
            }}
            """).strip() + "\n",
        encoding="utf-8",
    )
    executable_name = "fake_adapter.exe" if os.name == "nt" else "fake_adapter"
    executable_path = tmp_path / executable_name
    command = _native_compile_command(
        compiler, "c", source_path, executable_path, link=True
    )
    compile_result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    assert executable_path.is_file()

    run_result = subprocess.run(
        [str(executable_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr


@pytest.mark.parametrize(
    ("mutate", "code", "path"),
    [
        (
            lambda descriptor: descriptor["artifact"]["hash"].pop("value"),
            "project.native-loader-abi.descriptor-artifact-hash-invalid",
            "$.artifact.hash",
        ),
        (
            lambda descriptor: descriptor["artifact"].update(sizeBytes=1 << 64),
            "project.native-loader-abi.descriptor-artifact-size-invalid",
            "$.artifact.sizeBytes",
        ),
        (
            lambda descriptor: descriptor["source"]["hash"].update(value="invalid"),
            "project.native-loader-abi.descriptor-source-hash-invalid",
            "$.source.hash",
        ),
        (
            lambda descriptor: descriptor["source"]["remap"]["hash"].update(
                algorithm="sha1"
            ),
            "project.native-loader-abi.descriptor-source-remap-hash-invalid",
            "$.source.remap.hash",
        ),
        (
            lambda descriptor: descriptor["source"]["remap"].update(sizeBytes=True),
            "project.native-loader-abi.descriptor-source-remap-size-invalid",
            "$.source.remap.sizeBytes",
        ),
        (
            lambda descriptor: descriptor["source"].update(
                remap={
                    "sourcePath": "out/copy.source-remap.json",
                    "packagePath": "source-remaps/copy.source-remap.json",
                    "status": "unknown",
                }
            ),
            "project.native-loader-abi.descriptor-source-remap-status-invalid",
            "$.source.remap.status",
        ),
        (
            lambda descriptor: descriptor["bindings"][0].update(coordinates={"set": 0}),
            "project.native-loader-abi.descriptor-binding-coordinate-invalid",
            "$.bindings[0].coordinates",
        ),
        (
            lambda descriptor: descriptor["bindings"][0].update(access="execute"),
            "project.native-loader-abi.descriptor-binding-access-invalid",
            "$.bindings[0].access",
        ),
        (
            lambda descriptor: descriptor["bindings"][0].update(scalarLayout=[]),
            "project.native-loader-abi.json-object-invalid",
            "$.bindings[0].scalarLayout",
        ),
        (
            lambda descriptor: descriptor["scalarLayout"]["bindings"][0].update(
                layout=[]
            ),
            "project.native-loader-abi.json-object-invalid",
            "$.scalarLayout.bindings[0].layout",
        ),
        (
            lambda descriptor: descriptor.update(specializationConstants=[None]),
            "project.native-loader-abi.json-object-invalid",
            "$.specializationConstants[0]",
        ),
    ],
)
def test_declaration_generation_rejects_malformed_nested_descriptors(
    mutate, code, path
):
    descriptor = build_native_loader_abi_descriptor(_load_unit())
    mutate(descriptor)

    with pytest.raises(NativeLoaderABIError) as exc_info:
        generate_native_loader_declarations(descriptor)

    assert exc_info.value.code == code
    assert exc_info.value.path == path


@pytest.mark.parametrize(
    ("mutate", "code", "path"),
    [
        (
            lambda unit: unit["blockers"].append(
                {"kind": "resolve-host-interface-metadata"}
            ),
            "project.native-loader-abi.load-unit-blocked",
            "$.blockers",
        ),
        (
            lambda unit: unit["validation"].update(loadReady=False),
            "project.native-loader-abi.load-unit-not-ready",
            "$.validation.loadReady",
        ),
        (
            lambda unit: unit["hostInterface"]["resources"][0].update(binding=None),
            "project.native-loader-abi.binding-coordinate-invalid",
            "$.hostInterface.resources[0].binding",
        ),
        (
            lambda unit: unit.update(hash=None),
            "project.native-loader-abi.artifact-hash-invalid",
            "$.hash",
        ),
        (
            lambda unit: unit["sourceRemap"]["hash"].update(value="invalid"),
            "project.native-loader-abi.source-remap-hash-invalid",
            "$.sourceRemap.hash",
        ),
        (
            lambda unit: unit.pop("provenance"),
            "project.native-loader-abi.load-unit-schema-invalid",
            "$",
        ),
    ],
)
def test_native_loader_abi_rejects_blocked_or_incomplete_units(mutate, code, path):
    unit = _load_unit()
    mutate(unit)

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_descriptor(unit)

    assert exc_info.value.to_json()["severity"] == "error"
    assert exc_info.value.code == code
    assert exc_info.value.path == path


def test_native_loader_abi_manifest_requires_explicit_multi_unit_selection():
    directx = _load_unit("directx")
    opengl = _load_unit("opengl")
    manifest = _loader_manifest(opengl, directx)

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_descriptor(manifest)

    assert exc_info.value.code == (
        "project.native-loader-abi.load-unit-selection-required"
    )
    assert exc_info.value.details == {
        "availableUnitIds": ["copy:opengl", "copy:directx"]
    }

    selected = build_native_loader_abi_descriptor(manifest, load_unit_id="copy:directx")
    assert selected["target"] == "directx"
    assert json.loads(json.dumps(selected, sort_keys=True)) == selected
