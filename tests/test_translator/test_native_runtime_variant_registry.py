import copy
import hashlib
import json
import shutil
import subprocess
import textwrap

import pytest

from crosstl.project import pipeline as project_pipeline
from crosstl.project.native_loader_abi import (
    NATIVE_LOADER_ABI_KIND,
    NATIVE_LOADER_ABI_VERSION,
    generate_native_loader_execution_abi,
)
from crosstl.project.native_runtime_variant_registry import (
    NativeRuntimeVariantRegistryError,
    generate_native_runtime_variant_registry,
)
from crosstl.project.pipeline import build_runtime_variant_registry


def _artifact_format(target):
    return {
        "cuda": "CUDA source",
        "directx": "HLSL source",
        "hip": "HIP source",
        "metal": "Metal source",
        "mojo": "Mojo source",
        "opengl": "GLSL source",
        "rust": "Rust GPU source",
        "slang": "Slang source",
        "vulkan": "Vulkan-targeted shader source",
        "webgl": "GLSL ES source",
        "wgsl": "WGSL source",
    }[target]


def _host_interface(target, entry_point, width):
    return {
        "status": "ready",
        "source": "test-contract",
        "parser": "test",
        "artifactFormat": _artifact_format(target),
        "entryPointCount": 1,
        "resourceCount": 0,
        "constantCount": 0,
        "specializationConstantCount": 0,
        "entryPoints": [
            {
                "name": entry_point,
                "stage": "compute",
                "executionConfig": {"workgroupSize": [width, 1, 1]},
            }
        ],
        "resources": [],
        "constants": [],
        "specializationConstants": [],
        "diagnostics": [],
        "diagnosticRecords": [],
    }


def _write_registry_input(
    tmp_path,
    *,
    targets=("directx", "opengl"),
    unit_id_suffix="",
    include_opengl_specializations=False,
    directx_specialization_id=7,
    directx_specialization_name="mode",
    specialization_dtype="uint",
    descriptor_specialization_dtype="uint32",
    specialization_value=None,
    require_ready=True,
):
    artifacts = []
    metadata = []
    for variant, width, default_mode in (("n4", 4, 11), ("n8", 8, 13)):
        mode = default_mode if specialization_value is None else specialization_value
        for target in targets:
            has_specialization = target == "directx" or include_opengl_specializations
            entry_point = f"copy_{variant}" if target == "directx" else "main"
            extension = "hlsl" if target == "directx" else "glsl"
            artifact_bytes = (
                f"[numthreads({width}, 1, 1)] void {entry_point}() {{}}\n".encode()
                if target == "directx"
                else (
                    "#version 430\n"
                    f"layout(local_size_x = {width}) in;\n"
                    "void main() {}\n"
                ).encode()
            )
            package_path = f"artifacts/{target}/{variant}.{extension}"
            artifact_path = tmp_path / package_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(artifact_bytes)
            artifact_hash = {
                "algorithm": "sha256",
                "value": hashlib.sha256(artifact_bytes).hexdigest(),
            }
            unit_id = f"{target}|{variant}{unit_id_suffix}"
            host_interface = _host_interface(target, entry_point, width)
            specialization_constants = (
                [
                    {
                        "name": directx_specialization_name,
                        "id": directx_specialization_id,
                        "kind": "specialization-constant",
                        "dtype": specialization_dtype,
                        "concreteValue": mode,
                        "required": False,
                        "source": "project.variant",
                    }
                ]
                if has_specialization
                else []
            )
            artifact = {
                "id": unit_id,
                "status": "packaged",
                "source": "kernels/copy.metal",
                "sourcePath": f"translated/{target}/{variant}.{extension}",
                "packagePath": package_path,
                "target": target,
                "sourceBackend": "metal",
                "stage": "compute",
                "variant": variant,
                "defines": {},
                "entryPoint": {
                    "source": "copy_values",
                    "target": entry_point,
                    "stage": "compute",
                },
                "templateMaterialization": {
                    "status": "materialized",
                    "specializations": [
                        {
                            "name": "copy_values",
                            "hostName": entry_point,
                            "materializedName": entry_point,
                            "parameters": {"N": str(width)},
                            "parameterSources": {"N": "project.variant"},
                            "source": "project.variant",
                        }
                    ],
                },
                "specializationMaterialization": {
                    "status": "materialized",
                    "values": (
                        {
                            str(
                                directx_specialization_id
                                if directx_specialization_id is not None
                                else directx_specialization_name
                            ): mode
                        }
                        if has_specialization
                        else {}
                    ),
                },
                "specializationConstants": specialization_constants,
                "provenance": {"pipeline": "test"},
                "sourceHash": artifact_hash,
                "sourceSizeBytes": len(artifact_bytes),
                "hash": artifact_hash,
                "sizeBytes": len(artifact_bytes),
                "sourceRemap": None,
                "hostInterface": host_interface,
            }
            artifacts.append(artifact)
            descriptor = {
                "schemaVersion": NATIVE_LOADER_ABI_VERSION,
                "kind": NATIVE_LOADER_ABI_KIND,
                "abiVersion": NATIVE_LOADER_ABI_VERSION,
                "unitId": unit_id,
                "target": target,
                "stage": "compute",
                "entryPoint": {
                    "name": entry_point,
                    "stage": "compute",
                    "executionConfig": {"workgroupSize": [width, 1, 1]},
                    "provenance": {},
                },
                "artifact": {
                    "packagePath": package_path,
                    "format": _artifact_format(target),
                    "hash": artifact_hash,
                    "sizeBytes": len(artifact_bytes),
                },
                "source": {
                    "path": "kernels/copy.metal",
                    "artifactPath": f"translated/{target}/{variant}.{extension}",
                    "backend": "metal",
                    "hash": None,
                    "remap": None,
                },
                "bindings": [],
                "scalarLayout": {"constants": [], "bindings": []},
                "specializationConstants": (
                    [
                        {
                            "id": directx_specialization_id,
                            "name": directx_specialization_name,
                            "dtype": descriptor_specialization_dtype,
                            "value": mode,
                        }
                    ]
                    if has_specialization
                    else []
                ),
                "provenance": {"pipeline": "test"},
            }
            metadata.append((descriptor, f"units/{target}-{variant}.execution.h"))
    package = {
        "schemaVersion": 1,
        "kind": project_pipeline.RUNTIME_PACKAGE_KIND,
        "success": True,
        "packageRoot": str(tmp_path),
        "project": {"targets": list(targets)},
        "artifacts": artifacts,
        "runtimePlan": {"runtimeReferenceCount": 0},
    }
    package_path = tmp_path / "runtime-package.json"
    package_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    registry = build_runtime_variant_registry(package_path)
    if require_ready:
        assert registry["status"] == "ready"
    units = [
        {"descriptor": descriptor, "executionHeader": execution_header}
        for descriptor, execution_header in metadata
    ]
    return registry, units


def _refresh_registry_hash(registry):
    registry["registryHash"] = project_pipeline._runtime_variant_payload_hash(
        {
            "schemaVersion": registry["schemaVersion"],
            "keySchema": registry["keySchema"],
            "variants": registry["variants"],
        }
    )


def _variant_key(registry, target, variant="n4"):
    return next(
        key
        for key, record in registry["variants"].items()
        if record["target"]["backend"] == target and record["variant"] == variant
    )


def _compile_and_run(tmp_path, source):
    compiler = next(
        (
            candidate
            for candidate in ("clang++", "g++", "c++")
            if shutil.which(candidate)
        ),
        None,
    )
    if compiler is None:
        pytest.skip("A C++17 compiler is required")
    source_path = tmp_path / "registry-test.cpp"
    executable = tmp_path / "registry-test"
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
            str(executable),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    result = subprocess.run(
        [str(executable)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_generated_native_runtime_variant_registry_selects_and_executes_exact_key(
    tmp_path,
):
    registry, units = _write_registry_input(tmp_path)
    first = generate_native_runtime_variant_registry(registry, units)
    second = generate_native_runtime_variant_registry(
        copy.deepcopy(registry), list(reversed(copy.deepcopy(units)))
    )
    assert first == second

    for unit in units:
        path = tmp_path / unit["executionHeader"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            generate_native_loader_execution_abi(unit["descriptor"]),
            encoding="utf-8",
        )
    header_path = tmp_path / "native-runtime-variants.hpp"
    header_path.write_text(first, encoding="utf-8")
    directx_key = _variant_key(registry, "directx")
    _compile_and_run(
        tmp_path,
        textwrap.dedent(f"""
            #include <stdint.h>
            #include "native-runtime-variants.hpp"

            static uint32_t applied_value = 0u;
            static uint32_t dispatch_count = 0u;
            static uint32_t *mutable_specialization_value = NULL;

            static int32_t load_artifact(
                void *,
                const CrossTLNativeLoaderUnitDescriptor *unit,
                void **artifact_out) {{
                if (unit == NULL || artifact_out == NULL) return 1;
                if (mutable_specialization_value != NULL) {{
                    *mutable_specialization_value = 12u;
                }}
                *artifact_out = reinterpret_cast<void *>(1);
                return 0;
            }}
            static int32_t unload_artifact(void *, void *) {{ return 0; }}
            static int32_t create_pipeline(
                void *,
                void *,
                const CrossTLNativeLoaderUnitDescriptor *,
                void **pipeline_out) {{
                *pipeline_out = reinterpret_cast<void *>(2);
                return 0;
            }}
            static int32_t destroy_pipeline(void *, void *) {{ return 0; }}
            static int32_t apply_specialization(
                void *,
                void *,
                const CrossTLNativeLoaderSpecializationDescriptor *,
                const CrossTLNativeLoaderSpecializationRequest *request) {{
                if (request == NULL ||
                    request->payload_size_bytes != sizeof(uint32_t)) return 1;
                applied_value =
                    *static_cast<const uint32_t *>(request->payload);
                return 0;
            }}
            static int32_t bind_resource(
                void *,
                void *,
                const CrossTLNativeLoaderBindingDescriptor *,
                const CrossTLNativeLoaderBindingRequest *,
                void **) {{ return 1; }}
            static int32_t release_resource(
                void *,
                void *,
                const CrossTLNativeLoaderBindingDescriptor *) {{ return 1; }}
            static int32_t dispatch(
                void *,
                void *,
                const CrossTLNativeLoaderDispatchGeometry *geometry) {{
                if (geometry == NULL ||
                    geometry->workgroup_count[0] != 2u ||
                    geometry->workgroup_size[0] != 4u) return 1;
                ++dispatch_count;
                return 0;
            }}
            static int32_t synchronize(void *, void *) {{ return 0; }}
            static int32_t readback(
                void *,
                void *,
                const CrossTLNativeLoaderBindingDescriptor *,
                const CrossTLNativeLoaderBindingRequest *) {{ return 1; }}

            int main() {{
                const CrossTLNativeRuntimeVariantEntry *missing =
                    crosstl_native_runtime_variant_lookup("missing");
                if (missing != NULL) return 1;
                const CrossTLNativeRuntimeVariantEntry *variant =
                    crosstl_native_runtime_variant_lookup(
                        {json.dumps(directx_key)});
                if (variant == NULL ||
                    variant->workgroup_size[0] != 4u ||
                    variant->specialization_count != 1u) return 2;
                const uint32_t counts[3] = {{2u, 1u, 1u}};
                CrossTLNativeLoaderExecutionRequest request =
                    crosstl_native_runtime_variant_make_request(
                        variant, 0u, NULL, counts);
                CrossTLNativeLoaderAdapter adapter = {{
                    CROSSTL_NATIVE_LOADER_ABI_VERSION,
                    "directx",
                    NULL,
                    load_artifact,
                    unload_artifact,
                    create_pipeline,
                    destroy_pipeline,
                    apply_specialization,
                    bind_resource,
                    release_resource,
                    dispatch,
                    synchronize,
                    readback,
                }};
                CrossTLNativeLoaderExecutionResult result =
                    crosstl_native_runtime_variant_execute(
                        variant, &request, &adapter);
                if (!result.succeeded ||
                    applied_value != 11u ||
                    dispatch_count != 1u) return 3;

                uint32_t wrong_value = 12u;
                CrossTLNativeLoaderSpecializationRequest wrong =
                    request.specializations[0];
                wrong.payload = &wrong_value;
                request.specializations = &wrong;
                result = crosstl_native_runtime_variant_execute(
                    variant, &request, &adapter);
                if (result.succeeded ||
                    result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_SPECIALIZATION_VALUE_MISMATCH ||
                    dispatch_count != 1u) return 4;

                CrossTLNativeRuntimeVariantEntry copied_variant = *variant;
                CrossTLNativeLoaderExecutionRequest copied_request =
                    crosstl_native_runtime_variant_make_request(
                        &copied_variant, 0u, NULL, counts);
                if (copied_request.target != NULL) return 5;
                result = crosstl_native_runtime_variant_execute(
                    &copied_variant, &request, &adapter);
                if (result.succeeded ||
                    result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_INVALID_ARGUMENT ||
                    dispatch_count != 1u) return 6;

                request = crosstl_native_runtime_variant_make_request(
                    variant, 0u, NULL, counts);
                request.abi_version = 0u;
                result = crosstl_native_runtime_variant_execute(
                    variant, &request, &adapter);
                if (result.succeeded ||
                    result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_ABI_VERSION_MISMATCH ||
                    dispatch_count != 1u) return 7;

                request = crosstl_native_runtime_variant_make_request(
                    variant, 0u, NULL, counts);
                adapter.abi_version = 0u;
                result = crosstl_native_runtime_variant_execute(
                    variant, &request, &adapter);
                if (result.succeeded ||
                    result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_ABI_VERSION_MISMATCH ||
                    dispatch_count != 1u) return 8;
                adapter.abi_version = CROSSTL_NATIVE_LOADER_ABI_VERSION;

                adapter.target = "opengl";
                result = crosstl_native_runtime_variant_execute(
                    variant, &request, &adapter);
                if (result.succeeded ||
                    result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_TARGET_MISMATCH ||
                    dispatch_count != 1u) return 9;
                adapter.target = "directx";

                CrossTLNativeLoaderSpecializationRequest wrong_type =
                    request.specializations[0];
                wrong_type.type_name = "int32";
                request.specializations = &wrong_type;
                result = crosstl_native_runtime_variant_execute(
                    variant, &request, &adapter);
                if (result.succeeded ||
                    result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_SPECIALIZATION_TYPE_MISMATCH ||
                    dispatch_count != 1u) return 10;

                request = crosstl_native_runtime_variant_make_request(
                    variant, 0u, NULL, counts);
                request.specializations = NULL;
                result = crosstl_native_runtime_variant_execute(
                    variant, &request, &adapter);
                if (result.succeeded ||
                    result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_INVALID_ARGUMENT ||
                    result.error.specialization_index != 0u ||
                    dispatch_count != 1u) return 11;

                CrossTLNativeLoaderSpecializationRequest missing_payload =
                    variant->specializations[0];
                missing_payload.payload = NULL;
                request.specializations = &missing_payload;
                result = crosstl_native_runtime_variant_execute(
                    variant, &request, &adapter);
                if (result.succeeded ||
                    result.error.code !=
                        CROSSTL_NATIVE_LOADER_CODE_SPECIALIZATION_PAYLOAD_MISSING ||
                    result.error.specialization_index != 0u ||
                    dispatch_count != 1u) return 12;

                uint32_t mutable_value = 11u;
                CrossTLNativeLoaderSpecializationRequest mutable_specialization =
                    variant->specializations[0];
                mutable_specialization.payload = &mutable_value;
                request = crosstl_native_runtime_variant_make_request(
                    variant, 0u, NULL, counts);
                request.specializations = &mutable_specialization;
                mutable_specialization_value = &mutable_value;
                result = crosstl_native_runtime_variant_execute(
                    variant, &request, &adapter);
                mutable_specialization_value = NULL;
                if (!result.succeeded ||
                    mutable_value != 12u ||
                    applied_value != 11u ||
                    dispatch_count != 2u) return 13;
                return 0;
            }}
            """) + "\n",
    )


def test_native_runtime_variant_registry_rejects_unit_and_path_mismatches(tmp_path):
    registry, units = _write_registry_input(tmp_path)

    missing = copy.deepcopy(units)
    missing.pop()
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="does not match a native loader unit",
    ):
        generate_native_runtime_variant_registry(registry, missing)

    unsafe = copy.deepcopy(units)
    unsafe[0]["executionHeader"] = "../escape.h"
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="normalized package-relative header path",
    ):
        generate_native_runtime_variant_registry(registry, unsafe)

    unsafe_quote = copy.deepcopy(units)
    unsafe_quote[0]["executionHeader"] = 'units/bad"name.h'
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="normalized package-relative header path",
    ):
        generate_native_runtime_variant_registry(registry, unsafe_quote)

    mismatched = copy.deepcopy(units)
    mismatched[0]["descriptor"]["entryPoint"]["name"] = "wrong_entry"
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="does not match its native loader descriptor",
    ):
        generate_native_runtime_variant_registry(registry, mismatched)


def test_native_runtime_variant_registry_rejects_specialization_mismatch(tmp_path):
    registry, units = _write_registry_input(tmp_path)
    mismatched = copy.deepcopy(units)
    mismatched[0]["descriptor"]["specializationConstants"][0]["id"] = 99

    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="does not exist in the loader unit",
    ):
        generate_native_runtime_variant_registry(registry, mismatched)


@pytest.mark.parametrize(
    ("unsupported_target", "expected_code"),
    (
        ("cuda", "project.native-runtime-variant-registry.target-unsupported"),
        ("hip", "project.native-runtime-variant-registry.target-unsupported"),
        ("metal", "project.native-runtime-variant-registry.target-unsupported"),
        ("mojo", "project.native-runtime-variant-registry.target-unsupported"),
        ("rust", "project.native-runtime-variant-registry.target-unsupported"),
        ("slang", "project.native-runtime-variant-registry.target-unsupported"),
        ("vulkan", "project.native-runtime-variant-registry.registry-invalid"),
        ("webgl", "project.native-runtime-variant-registry.registry-invalid"),
        ("wgsl", "project.native-runtime-variant-registry.target-unsupported"),
    ),
)
def test_native_runtime_variant_registry_rejects_adapter_incompatible_units(
    tmp_path,
    unsupported_target,
    expected_code,
):
    registry, units = _write_registry_input(tmp_path)
    directx_key = _variant_key(registry, "directx")
    unit_id = registry["variants"][directx_key]["artifact"]["id"]

    stage_registry = copy.deepcopy(registry)
    stage_units = copy.deepcopy(units)
    stage_record = stage_registry["variants"][directx_key]
    stage_record["target"]["stage"] = "fragment"
    stage_record["bindingInterface"]["entryPoint"]["stage"] = "fragment"
    stage_unit = next(
        unit for unit in stage_units if unit["descriptor"]["unitId"] == unit_id
    )
    stage_unit["descriptor"]["stage"] = "fragment"
    stage_unit["descriptor"]["entryPoint"]["stage"] = "fragment"
    _refresh_registry_hash(stage_registry)
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="compute-stage units only",
    ):
        generate_native_runtime_variant_registry(stage_registry, stage_units)

    format_registry = copy.deepcopy(registry)
    format_units = copy.deepcopy(units)
    format_registry["variants"][directx_key]["artifact"]["format"] = "PTX"
    format_unit = next(
        unit for unit in format_units if unit["descriptor"]["unitId"] == unit_id
    )
    format_unit["descriptor"]["artifact"]["format"] = "PTX"
    _refresh_registry_hash(format_registry)
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="artifact format is unsupported",
    ):
        generate_native_runtime_variant_registry(format_registry, format_units)

    entry_registry = copy.deepcopy(registry)
    entry_units = copy.deepcopy(units)
    entry_record = entry_registry["variants"][directx_key]
    entry_record["target"]["entryPoint"] = "invalid-entry"
    entry_record["bindingInterface"]["entryPoint"]["name"] = "invalid-entry"
    entry_unit = next(
        unit for unit in entry_units if unit["descriptor"]["unitId"] == unit_id
    )
    entry_unit["descriptor"]["entryPoint"]["name"] = "invalid-entry"
    _refresh_registry_hash(entry_registry)
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="portable shader identifier",
    ):
        generate_native_runtime_variant_registry(entry_registry, entry_units)

    path_registry = copy.deepcopy(registry)
    path_units = copy.deepcopy(units)
    invalid_path = "artifacts/directx/n4.hlsl\x00shadow"
    path_registry["variants"][directx_key]["artifact"]["path"] = invalid_path
    path_unit = next(
        unit for unit in path_units if unit["descriptor"]["unitId"] == unit_id
    )
    path_unit["descriptor"]["artifact"]["packagePath"] = invalid_path
    _refresh_registry_hash(path_registry)
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="embedded null bytes",
    ):
        generate_native_runtime_variant_registry(path_registry, path_units)

    unsupported_registry, unsupported_units = _write_registry_input(
        tmp_path / "unsupported-target",
        targets=(unsupported_target,),
        require_ready=False,
    )
    ready_keys = sorted(unsupported_registry["variants"])
    for record in unsupported_registry["variants"].values():
        record["status"] = "ready"
        record["execution"] = {"workgroupSize": None, "subgroupWidth": None}
        record["bindingInterface"]["status"] = "ready"
        record["bindingInterface"]["entryPoint"] = {
            "name": record["target"]["entryPoint"],
            "stage": record["target"]["stage"],
            "executionConfig": {},
        }
        record["lookup"]["eligible"] = True
        record["blockers"] = []
    unsupported_registry["success"] = True
    unsupported_registry["status"] = "ready"
    unsupported_registry["summary"]["readyVariantCount"] = len(ready_keys)
    unsupported_registry["summary"]["blockedVariantCount"] = 0
    unsupported_registry["targets"][0]["readyVariantCount"] = len(ready_keys)
    unsupported_registry["targets"][0]["blockedVariantCount"] = 0
    unsupported_registry["lookup"]["readyKeys"] = ready_keys
    unsupported_registry["lookup"]["blockedKeys"] = []
    unsupported_registry["diagnosticCounts"] = {
        "note": 0,
        "warning": 0,
        "error": 0,
    }
    unsupported_registry["diagnostics"] = []
    _refresh_registry_hash(unsupported_registry)
    with pytest.raises(NativeRuntimeVariantRegistryError) as exc_info:
        generate_native_runtime_variant_registry(
            unsupported_registry,
            unsupported_units,
        )
    assert exc_info.value.code == expected_code


def test_native_runtime_variant_registry_rejects_unexecutable_specializations(
    tmp_path,
):
    registry, units = _write_registry_input(tmp_path / "base")

    noncanonical = copy.deepcopy(units)
    noncanonical[0]["descriptor"]["specializationConstants"][0]["dtype"] = "uint"
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="canonical ABI names",
    ):
        generate_native_runtime_variant_registry(registry, noncanonical)

    duplicate = copy.deepcopy(units)
    duplicate[0]["descriptor"]["specializationConstants"].append(
        copy.deepcopy(duplicate[0]["descriptor"]["specializationConstants"][0])
    )
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="identities must be unique",
    ):
        generate_native_runtime_variant_registry(registry, duplicate)

    duplicate_name = copy.deepcopy(units)
    second_constant = copy.deepcopy(
        duplicate_name[0]["descriptor"]["specializationConstants"][0]
    )
    second_constant["id"] += 1
    duplicate_name[0]["descriptor"]["specializationConstants"].append(second_constant)
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="specialization names must be unique",
    ):
        generate_native_runtime_variant_registry(registry, duplicate_name)

    glsl_registry, glsl_units = _write_registry_input(
        tmp_path / "glsl",
        include_opengl_specializations=True,
    )
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="cannot apply specializations to this artifact format",
    ):
        generate_native_runtime_variant_registry(glsl_registry, glsl_units)

    missing_id_registry, missing_id_units = _write_registry_input(
        tmp_path / "missing-id",
        directx_specialization_id=None,
    )
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="requires a numeric constant ID",
    ):
        generate_native_runtime_variant_registry(
            missing_id_registry,
            missing_id_units,
        )

    invalid_name_registry, invalid_name_units = _write_registry_input(
        tmp_path / "invalid-name",
        directx_specialization_name="invalid-name",
    )
    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="portable identifier name",
    ):
        generate_native_runtime_variant_registry(
            invalid_name_registry,
            invalid_name_units,
        )


def test_native_runtime_variant_registry_reports_float32_overflow(tmp_path):
    registry, units = _write_registry_input(
        tmp_path,
        specialization_dtype="float32",
        descriptor_specialization_dtype="float32",
        specialization_value=10**400,
    )

    with pytest.raises(
        NativeRuntimeVariantRegistryError,
        match="outside the representable range",
    ):
        generate_native_runtime_variant_registry(registry, units)


def test_native_runtime_variant_registry_emits_portable_utf8_literals(tmp_path):
    registry, units = _write_registry_input(
        tmp_path,
        unit_id_suffix="\U0001f600",
    )
    header = generate_native_runtime_variant_registry(registry, units)
    assert r"\360\237\230\200" in header

    for unit in units:
        path = tmp_path / unit["executionHeader"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            generate_native_loader_execution_abi(unit["descriptor"]),
            encoding="utf-8",
        )
    header_path = tmp_path / "native-runtime-variants.hpp"
    header_path.write_text(header, encoding="utf-8")
    key = _variant_key(registry, "directx")
    _compile_and_run(
        tmp_path,
        textwrap.dedent(f"""
            #include "native-runtime-variants.hpp"

            int main() {{
                return crosstl_native_runtime_variant_lookup(
                    {json.dumps(key)}) == NULL;
            }}
            """) + "\n",
    )
