from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from crosstl.project import (
    NATIVE_LOADER_ABI_PACKAGE_KIND,
    NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
    NATIVE_LOADER_ABI_PACKAGE_VERSION,
    NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH,
    NATIVE_RUNTIME_VARIANT_REGISTRY_PATH,
    NativeLoaderABIError,
    NativeLoaderTargetAdapterError,
    build_native_loader_abi_package,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    build_runtime_variant_dispatch_request,
    generate_native_loader_target_adapter,
    native_loader_abi_package,
    translate_project,
)
from crosstl.project.native_loader_abi import (
    build_native_loader_abi_descriptor,
    generate_native_loader_declarations,
    generate_native_loader_execution_abi,
)

REDUCED_METAL_COMPUTE = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    [[kernel]] void copy_values(
        const device float* source_values [[buffer(0)]],
        device float* destination_values [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        destination_values[gid] = source_values[gid];
    }
    """).strip()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def reduced_runtime_package(tmp_path_factory):
    repo = tmp_path_factory.mktemp("native-loader-abi") / "repo"
    kernel_dir = repo / "kernels"
    kernel_dir.mkdir(parents=True)
    source_path = kernel_dir / "copy.metal"
    source_path.write_text(REDUCED_METAL_COMPUTE + "\n", encoding="utf-8")

    report = translate_project(
        repo,
        targets=["directx", "opengl"],
        output_dir="translated",
        format_output=False,
    )
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    artifact_manifest_path = repo / "translated" / "runtime-artifacts.json"
    artifact_manifest_path.write_text(
        json.dumps(
            build_runtime_artifact_manifest(report_path), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    package_dir = repo / "runtime-package"
    package = build_runtime_package(artifact_manifest_path, package_dir)
    loader_manifest = build_runtime_loader_manifest(
        package_dir / "runtime-package.json"
    )
    loader_manifest_path = package_dir / "runtime-loader-manifest.json"
    loader_manifest_path.write_text(
        json.dumps(loader_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert report.to_json()["summary"]["translatedCount"] == 2
    assert package["success"] is True
    assert loader_manifest["success"] is True
    assert loader_manifest["summary"]["readyLoadUnitCount"] == 2
    assert loader_manifest["summary"]["blockedLoadUnitCount"] == 0

    units = {unit["target"]: unit for unit in loader_manifest["loadUnits"]}
    descriptors = {
        target: build_native_loader_abi_descriptor(
            loader_manifest, load_unit_id=unit["id"]
        )
        for target, unit in units.items()
    }
    headers = {
        target: generate_native_loader_declarations(descriptor)
        for target, descriptor in descriptors.items()
    }
    execution_headers = {
        target: generate_native_loader_execution_abi(descriptor)
        for target, descriptor in descriptors.items()
    }
    target_adapters = {
        target: generate_native_loader_target_adapter(target)
        for target in sorted(descriptors)
    }
    return {
        "source_path": source_path,
        "package_dir": package_dir,
        "loader_manifest": loader_manifest,
        "loader_manifest_path": loader_manifest_path,
        "units": units,
        "descriptors": descriptors,
        "headers": headers,
        "execution_headers": execution_headers,
        "target_adapters": target_adapters,
    }


def test_runtime_packages_preserve_reflected_native_loader_contract(
    reduced_runtime_package,
):
    source_path = reduced_runtime_package["source_path"]
    package_dir = reduced_runtime_package["package_dir"]
    units = reduced_runtime_package["units"]
    descriptors = reduced_runtime_package["descriptors"]
    headers = reduced_runtime_package["headers"]

    assert set(units) == {"directx", "opengl"}
    assert set(descriptors) == {"directx", "opengl"}
    assert {
        target: descriptor["entryPoint"]["name"]
        for target, descriptor in descriptors.items()
    } == {"directx": "CSMain", "opengl": "main"}

    expected_bindings = {
        "directx": [
            (
                "source_values",
                "srv",
                {"set": 0, "binding": 0},
                "read",
            ),
            (
                "destination_values",
                "uav",
                {"set": 0, "binding": 1},
                "read_write",
            ),
        ],
        "opengl": [
            (
                "source_valuesBuffer",
                "storage-buffer",
                {"set": 0, "binding": 0},
                "read",
            ),
            (
                "destination_valuesBuffer",
                "storage-buffer",
                {"set": 0, "binding": 1},
                "read_write",
            ),
        ],
    }
    for target, descriptor in descriptors.items():
        unit = units[target]
        reflected_entry_point = unit["hostInterface"]["entryPoints"][0]
        assert {
            field: descriptor["entryPoint"][field]
            for field in ("name", "stage", "executionConfig")
        } == reflected_entry_point
        assert descriptor["entryPoint"]["provenance"] == {}
        assert [
            (
                binding["name"],
                binding["namespace"],
                binding["coordinates"],
                binding["access"],
            )
            for binding in descriptor["bindings"]
        ] == expected_bindings[target]
        for binding in descriptor["bindings"]:
            layout = binding["scalarLayout"]
            assert layout["physicalType"] == "float"
            assert layout["elementType"] == "float32"
            assert layout["elementSizeBytes"] == 4
            assert layout["elementStrideBytes"] == 4
            assert layout["alignmentBytes"] == 4
            assert layout["memberOffsetBytes"] == 0
            assert layout["runtimeSized"] is True
            assert (
                layout["storageLayout"]
                == {
                    "directx": "hlsl-structured-buffer",
                    "opengl": "std430",
                }[target]
            )

        artifact_path = package_dir / descriptor["artifact"]["packagePath"]
        assert descriptor["artifact"]["hash"] == {
            "algorithm": "sha256",
            "value": _sha256(artifact_path),
        }
        assert descriptor["source"]["path"] == "kernels/copy.metal"
        assert descriptor["source"]["backend"] == "metal"
        assert descriptor["source"]["hash"] == {
            "algorithm": "sha256",
            "value": _sha256(source_path),
        }
        assert descriptor["source"]["remap"] == unit["sourceRemap"]
        remap_path = package_dir / descriptor["source"]["remap"]["packagePath"]
        assert remap_path.is_file()
        assert descriptor["provenance"] == unit["provenance"]
        assert descriptor["provenance"] == {
            "intermediate": "crossgl",
            "pipeline": "single-file-translate",
        }

        rebuilt = build_native_loader_abi_descriptor(unit)
        assert rebuilt == descriptor
        assert generate_native_loader_declarations(rebuilt) == headers[target]


def _compiler_command(language):
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


def _compile_headers(tmp_path, language, headers):
    compiler = _compiler_command(language)
    if compiler is None:
        pytest.skip(f"No native {language} compiler is available")

    directx_header = tmp_path / "directx_loader.h"
    opengl_header = tmp_path / "opengl_loader.h"
    directx_header.write_text(headers["directx"], encoding="utf-8")
    opengl_header.write_text(headers["opengl"], encoding="utf-8")
    symbols = {}
    for target, header in headers.items():
        match = re.search(
            r"static const CrossTLNativeLoaderUnitDescriptor ([A-Za-z_]\w*) = \{",
            header,
        )
        assert match is not None
        symbols[target] = match.group(1)
    suffix = ".c" if language == "c" else ".cpp"
    source_path = tmp_path / f"loader_contract{suffix}"
    source_path.write_text(
        textwrap.dedent(f"""
            #include "directx_loader.h"
            #include "opengl_loader.h"

            int main(void) {{
                return (int)({symbols["directx"]}.binding_count +
                             {symbols["opengl"]}.binding_count == 0u);
            }}
            """).strip() + "\n",
        encoding="utf-8",
    )

    executable_names = {
        Path(part).name.lower() for part in compiler if not part.startswith("-")
    }
    msvc_style = bool(executable_names & {"cl", "cl.exe", "clang-cl", "clang-cl.exe"})
    if msvc_style:
        object_path = tmp_path / "loader_contract.obj"
        standard = "/std:c11" if language == "c" else "/std:c++17"
        command = compiler + [
            "/nologo",
            standard,
            "/W4",
            "/WX",
            "/c",
            str(source_path),
            f"/Fo{object_path}",
        ]
    else:
        object_path = tmp_path / "loader_contract.o"
        standard = "-std=c11" if language == "c" else "-std=c++17"
        command = compiler + [
            standard,
            "-pedantic-errors",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ]

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
def test_generated_runtime_loader_headers_compile(
    tmp_path, reduced_runtime_package, language
):
    _compile_headers(tmp_path, language, reduced_runtime_package["headers"])


def _relative_file_contents(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_runtime_loader_manifest_builds_deterministic_multi_target_abi_package(
    tmp_path, reduced_runtime_package
):
    manifest_path = reduced_runtime_package["loader_manifest_path"]
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = build_native_loader_abi_package(manifest_path, first_root)
    second = build_native_loader_abi_package(manifest_path, second_root)

    assert first == second
    assert first["kind"] == NATIVE_LOADER_ABI_PACKAGE_KIND
    assert first["schemaVersion"] == NATIVE_LOADER_ABI_PACKAGE_VERSION
    assert first["schemaVersion"] == 3
    assert first["success"] is True
    assert first["summary"] == {
        "unitCount": 2,
        "targetCount": 2,
        "targetAdapterCount": 2,
        "unavailableTargetAdapterCount": 0,
        "runtimeVariantCount": 2,
        "generatedFileCount": 13,
    }
    assert len(first["generatedFiles"]) == first["summary"]["generatedFileCount"]
    assert first["generatedFiles"][0] == {
        "path": NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
        "kind": "native-loader-abi-package-manifest",
    }
    assert [unit["target"] for unit in first["units"]] == ["directx", "opengl"]
    assert [adapter["target"] for adapter in first["targetAdapters"]] == [
        "directx",
        "opengl",
    ]
    assert _relative_file_contents(first_root) == _relative_file_contents(second_root)
    assert (
        json.loads(
            (first_root / NATIVE_LOADER_ABI_PACKAGE_MANIFEST).read_text(
                encoding="utf-8"
            )
        )
        == first
    )

    for unit in first["units"]:
        target = unit["target"]
        descriptor_path = first_root / unit["descriptorPath"]
        declarations_path = first_root / unit["declarationsPath"]
        execution_abi_path = first_root / unit["executionABIPath"]
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        assert descriptor == reduced_runtime_package["descriptors"][target]
        assert (
            declarations_path.read_text(encoding="utf-8")
            == reduced_runtime_package["headers"][target]
        )
        assert (
            execution_abi_path.read_text(encoding="utf-8")
            == reduced_runtime_package["execution_headers"][target]
        )
        assert unit["executionABIPath"].endswith(".native-loader-execution.h")
        assert unit["descriptorHash"] == {
            "algorithm": "sha256",
            "value": _sha256(descriptor_path),
        }
        assert unit["descriptorSizeBytes"] == descriptor_path.stat().st_size
        assert unit["declarationsHash"] == {
            "algorithm": "sha256",
            "value": _sha256(declarations_path),
        }
        assert unit["executionABIHash"] == {
            "algorithm": "sha256",
            "value": _sha256(execution_abi_path),
        }

    generated_file_kinds = {
        generated["path"]: generated["kind"] for generated in first["generatedFiles"]
    }
    for adapter in first["targetAdapters"]:
        assert adapter["available"] is True
        adapter_path = first_root / adapter["path"]
        assert (
            adapter_path.read_text(encoding="utf-8")
            == reduced_runtime_package["target_adapters"][adapter["target"]]
        )
        assert adapter["hash"] == {
            "algorithm": "sha256",
            "value": _sha256(adapter_path),
        }
        assert generated_file_kinds[adapter["path"]] == ("native-loader-target-adapter")
    for unit in first["units"]:
        assert generated_file_kinds[unit["descriptorPath"]] == (
            "native-loader-abi-descriptor"
        )
        assert generated_file_kinds[unit["declarationsPath"]] == (
            "native-loader-c-declarations"
        )
        assert generated_file_kinds[unit["executionABIPath"]] == (
            "native-loader-execution-abi"
        )
    runtime_registry = first["runtimeVariantRegistry"]
    registry_path = first_root / NATIVE_RUNTIME_VARIANT_REGISTRY_PATH
    native_header_path = first_root / NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["kind"] == "crosstl-runtime-variant-registry"
    assert registry["status"] == "ready"
    assert runtime_registry == {
        "available": True,
        "path": NATIVE_RUNTIME_VARIANT_REGISTRY_PATH,
        "hash": {
            "algorithm": "sha256",
            "value": _sha256(registry_path),
        },
        "registryHash": registry["registryHash"],
        "variantCount": 2,
        "nativeHeader": {
            "available": True,
            "path": NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH,
            "hash": {
                "algorithm": "sha256",
                "value": _sha256(native_header_path),
            },
        },
    }
    assert generated_file_kinds[NATIVE_RUNTIME_VARIANT_REGISTRY_PATH] == (
        "runtime-variant-registry"
    )
    assert generated_file_kinds[NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH] == (
        "native-runtime-variant-registry"
    )
    for record in registry["variants"].values():
        artifact_path = record["artifact"]["path"]
        assert generated_file_kinds[artifact_path] == "runtime-target-artifact"
        assert (first_root / artifact_path).is_file()


def test_runtime_loader_abi_package_routes_glsl_specialization_through_deferred_compilation(
    tmp_path,
):
    from crosstl.project import (
        build_runtime_variant_deferred_compilation_request,
        load_project_config,
    )

    repo = tmp_path / "specialized-opengl-repo"
    kernel_dir = repo / "kernels"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "copy.metal").write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;

            constant bool enabled [[function_constant(3)]];

            [[kernel]] void copy_values(
                const device float* source_values [[buffer(0)]],
                device float* destination_values [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
                destination_values[gid] = enabled ? source_values[gid] : 0.0f;
            }
            """).strip() + "\n",
        encoding="utf-8",
    )
    config_path = repo / "crosstl.toml"
    config_path.write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["kernels"]
            include = ["kernels/*.metal"]
            targets = ["opengl"]
            output_dir = "translated"

            [project.sources]
            "**/*.metal" = "metal"

            [project.specialization_constants]
            "3" = true
            """).strip() + "\n",
        encoding="utf-8",
    )
    report = translate_project(
        load_project_config(repo, config_path),
        targets=("opengl",),
        output_dir="translated",
        format_output=False,
    )
    report_path = repo / "translated" / "portability-report.json"
    report.write_json(report_path)
    artifact_manifest_path = repo / "translated" / "runtime-artifacts.json"
    artifact_manifest_path.write_text(
        json.dumps(
            build_runtime_artifact_manifest(report_path), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_package_dir = repo / "runtime-package"
    runtime_package = build_runtime_package(
        artifact_manifest_path,
        runtime_package_dir,
    )
    loader_manifest = build_runtime_loader_manifest(
        runtime_package_dir / "runtime-package.json"
    )
    loader_manifest_path = runtime_package_dir / "runtime-loader-manifest.json"
    loader_manifest_path.write_text(
        json.dumps(loader_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    package_root = repo / "native-loader-package"
    package = build_native_loader_abi_package(loader_manifest_path, package_root)

    assert package["success"] is True
    assert package["runtimeVariantRegistry"]["available"] is True
    assert package["runtimeVariantRegistry"]["nativeHeader"] == {
        "available": False,
        "reason": "specialization-requires-deferred-compilation",
    }
    assert not (package_root / NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH).exists()
    registry = json.loads(
        (package_root / package["runtimeVariantRegistry"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert registry["status"] == "ready"
    assert registry["summary"]["readyVariantCount"] == 1
    key = registry["lookup"]["readyKeys"][0]
    request = build_runtime_variant_deferred_compilation_request(
        registry,
        key,
        package_root / NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
    )
    assert request["source"]["format"] == "GLSL source"
    assert request["target"] == {
        "backend": "opengl",
        "profile": None,
        "stage": "compute",
        "entryPoint": "main",
        "outputFormat": "SPIR-V binary",
    }
    assert request["variant"]["specializationValues"] == [
        {"id": 3, "name": "enabled", "value": True}
    ]


@pytest.mark.parametrize(
    ("code", "details"),
    (
        (
            "specializations-invalid",
            {"target": "opengl", "artifactFormat": "GLSL source"},
        ),
        (
            "specialization-mechanism-unsupported",
            {"target": "directx", "artifactFormat": "GLSL source"},
        ),
        (
            "specialization-mechanism-unsupported",
            {"target": "opengl", "artifactFormat": "SPIR-V binary"},
        ),
    ),
)
def test_runtime_loader_abi_package_does_not_hide_other_registry_failures(
    tmp_path,
    reduced_runtime_package,
    monkeypatch,
    code,
    details,
):
    def fail_generation(_registry, _units):
        raise native_loader_abi_package.NativeRuntimeVariantRegistryError(
            code,
            "Native registry generation failed.",
            path="$.runtimeVariantRegistry",
            details=details,
        )

    monkeypatch.setattr(
        native_loader_abi_package,
        "generate_native_runtime_variant_registry",
        fail_generation,
    )
    package_root = tmp_path / "abi-package"

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(
            reduced_runtime_package["loader_manifest_path"],
            package_root,
        )

    assert exc_info.value.code == (
        "project.native-loader-abi." "native-runtime-variant-registry-generation-failed"
    )
    assert exc_info.value.path == "$.runtimeVariantRegistry"
    assert exc_info.value.details["diagnostic"]["code"].endswith(f".{code}")
    assert exc_info.value.details["diagnostic"]["details"] == details
    assert not package_root.exists()


def test_runtime_loader_abi_package_reports_unavailable_target_adapters(
    tmp_path, reduced_runtime_package, monkeypatch
):
    monkeypatch.setattr(
        native_loader_abi_package,
        "native_loader_target_adapter_targets",
        lambda: ("directx",),
    )

    package_root = tmp_path / "abi-package"
    package = build_native_loader_abi_package(
        reduced_runtime_package["loader_manifest_path"],
        package_root,
    )

    assert package["summary"]["targetAdapterCount"] == 1
    assert package["summary"]["unavailableTargetAdapterCount"] == 1
    assert package["summary"]["runtimeVariantCount"] == 2
    assert package["summary"]["generatedFileCount"] == 11
    assert package["targetAdapters"] == [
        {
            "target": "directx",
            "available": True,
            "path": "targets/directx-7fde9c43d3d7/" "native-loader-target-adapter.hpp",
            "hash": {
                "algorithm": "sha256",
                "value": _sha256(
                    package_root
                    / "targets/directx-7fde9c43d3d7"
                    / "native-loader-target-adapter.hpp"
                ),
            },
        },
        {
            "target": "opengl",
            "available": False,
            "reason": "target-adapter-unavailable",
        },
    ]
    assert package["runtimeVariantRegistry"]["nativeHeader"] == {
        "available": False,
        "reason": "target-adapter-unavailable",
        "unavailableTargets": ["opengl"],
    }
    assert (package_root / package["runtimeVariantRegistry"]["path"]).is_file()
    assert not (package_root / NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH).exists()


def test_runtime_loader_abi_package_rejects_adapter_generation_failure_before_writing(
    tmp_path, reduced_runtime_package, monkeypatch
):
    def fail_generation(target):
        raise NativeLoaderTargetAdapterError(
            "generation-failed",
            "Reference adapter generation failed.",
            details={"target": target},
        )

    monkeypatch.setattr(
        native_loader_abi_package,
        "generate_native_loader_target_adapter",
        fail_generation,
    )
    package_root = tmp_path / "abi-package"

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(
            reduced_runtime_package["loader_manifest_path"],
            package_root,
        )

    assert exc_info.value.code == (
        "project.native-loader-abi.target-adapter-generation-failed"
    )
    assert exc_info.value.details == {
        "target": "directx",
        "diagnostic": {
            "severity": "error",
            "code": "project.native-loader-target-adapter.generation-failed",
            "message": "Reference adapter generation failed.",
            "path": "$",
            "details": {"target": "directx"},
        },
    }
    assert not package_root.exists()


def test_runtime_loader_abi_package_rejects_blocked_units_before_writing(
    tmp_path, reduced_runtime_package
):
    manifest = copy.deepcopy(reduced_runtime_package["loader_manifest"])
    manifest["loadUnits"][0]["blockers"] = [{"kind": "resolve-host-interface-metadata"}]
    manifest_path = tmp_path / "blocked-loader-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    package_root = tmp_path / "abi-package"

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(manifest_path, package_root)

    assert exc_info.value.code == "project.native-loader-abi.load-unit-blocked"
    assert not package_root.exists()


def test_runtime_loader_abi_package_rejects_duplicate_unit_ids(
    tmp_path, reduced_runtime_package
):
    manifest = copy.deepcopy(reduced_runtime_package["loader_manifest"])
    manifest["loadUnits"][1]["id"] = manifest["loadUnits"][0]["id"]
    manifest_path = tmp_path / "duplicate-loader-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    package_root = tmp_path / "abi-package"

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(manifest_path, package_root)

    assert exc_info.value.code == ("project.native-loader-abi.load-unit-id-duplicate")
    assert not package_root.exists()


def _copy_runtime_package_with_artifact_path(
    tmp_path,
    reduced_runtime_package,
    package_path,
):
    package_dir = tmp_path / "runtime-package"
    shutil.copytree(reduced_runtime_package["package_dir"], package_dir)
    manifest_path = package_dir / "runtime-loader-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unit = manifest["loadUnits"][0]
    artifact_bytes = (package_dir / unit["packagePath"]).read_bytes()
    replacement_path = package_dir / package_path
    replacement_path.parent.mkdir(parents=True, exist_ok=True)
    replacement_path.write_bytes(artifact_bytes)
    unit["packagePath"] = package_path
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_runtime_loader_abi_package_rejects_artifact_package_manifest_collision(
    tmp_path, reduced_runtime_package
):
    manifest_path = _copy_runtime_package_with_artifact_path(
        tmp_path,
        reduced_runtime_package,
        NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
    )
    output_root = tmp_path / "abi-package"

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(manifest_path, output_root)

    assert exc_info.value.code == "project.native-loader-abi.output-path-collision"
    assert exc_info.value.details == {
        "firstRelativePath": NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
        "secondRelativePath": NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
    }
    assert not output_root.exists()


def test_runtime_loader_abi_package_rejects_portable_case_collision(
    tmp_path, reduced_runtime_package
):
    artifact_path = NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH.upper()
    manifest_path = _copy_runtime_package_with_artifact_path(
        tmp_path,
        reduced_runtime_package,
        artifact_path,
    )
    output_root = tmp_path / "abi-package"

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(manifest_path, output_root)

    assert exc_info.value.code == "project.native-loader-abi.output-path-collision"
    assert {
        exc_info.value.details["firstRelativePath"].casefold(),
        exc_info.value.details["secondRelativePath"].casefold(),
    } == {NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH.casefold()}
    assert not output_root.exists()


@pytest.mark.parametrize(
    "artifact_path",
    [
        "artifacts/CON.hlsl",
        "artifacts/kernel.hlsl:payload",
        "artifacts/kernel.hlsl.",
        "artifacts/trailing /kernel.hlsl",
    ],
)
def test_runtime_loader_abi_package_rejects_nonportable_artifact_paths(
    tmp_path, reduced_runtime_package, artifact_path
):
    package_dir = tmp_path / "runtime-package"
    shutil.copytree(reduced_runtime_package["package_dir"], package_dir)
    manifest_path = package_dir / "runtime-loader-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["loadUnits"][0]["packagePath"] = artifact_path
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "abi-package"

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(manifest_path, output_root)

    assert exc_info.value.code == "project.native-loader-abi.artifact-path-invalid"
    assert not output_root.exists()


def test_runtime_loader_abi_package_rejects_generated_path_input_collision(
    tmp_path, reduced_runtime_package
):
    manifest_path = tmp_path / NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH
    manifest_path.write_text(
        json.dumps(
            reduced_runtime_package["loader_manifest"],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copytree(
        reduced_runtime_package["package_dir"] / "artifacts",
        tmp_path / "artifacts",
    )

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(manifest_path, tmp_path)

    assert exc_info.value.code == "project.native-loader-abi.path-collision"
    assert exc_info.value.details == {
        "manifestPath": str(manifest_path),
        "relativePath": NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH,
    }
    assert manifest_path.read_text(encoding="utf-8").startswith("{")


def test_runtime_loader_abi_package_rejects_output_symlink_escape(
    tmp_path, reduced_runtime_package
):
    output_root = tmp_path / "abi-package"
    outside_root = tmp_path / "outside"
    output_root.mkdir()
    outside_root.mkdir()
    try:
        (output_root / "artifacts").symlink_to(
            outside_root,
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(
            reduced_runtime_package["loader_manifest_path"],
            output_root,
        )

    assert exc_info.value.code == "project.native-loader-abi.output-path-escape"
    assert exc_info.value.details["relativePath"].startswith("artifacts/")
    assert not list(outside_root.iterdir())
    assert not (output_root / NATIVE_LOADER_ABI_PACKAGE_MANIFEST).exists()


def test_runtime_loader_abi_package_does_not_follow_temporary_output_symlink(
    tmp_path, reduced_runtime_package
):
    output_root = tmp_path / "abi-package"
    artifact_package_path = reduced_runtime_package["loader_manifest"]["loadUnits"][0][
        "packagePath"
    ]
    artifact_output_path = output_root / artifact_package_path
    artifact_output_path.parent.mkdir(parents=True)
    outside_path = tmp_path / "outside.bin"
    outside_path.write_bytes(b"outside")
    temporary_path = artifact_output_path.with_name(f".{artifact_output_path.name}.tmp")
    try:
        temporary_path.symlink_to(outside_path)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    build_native_loader_abi_package(
        reduced_runtime_package["loader_manifest_path"],
        output_root,
    )

    assert outside_path.read_bytes() == b"outside"
    assert temporary_path.is_symlink()
    assert (
        artifact_output_path.read_bytes()
        == (reduced_runtime_package["package_dir"] / artifact_package_path).read_bytes()
    )


def test_runtime_loader_abi_package_invalidates_manifest_before_failed_rebuild(
    tmp_path, reduced_runtime_package, monkeypatch
):
    output_root = tmp_path / "abi-package"
    build_native_loader_abi_package(
        reduced_runtime_package["loader_manifest_path"],
        output_root,
    )
    package_manifest_path = output_root / NATIVE_LOADER_ABI_PACKAGE_MANIFEST
    assert package_manifest_path.is_file()
    original_write_output = native_loader_abi_package._write_output

    def fail_native_registry_write(root, relative_path, content):
        if relative_path == NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH:
            raise NativeLoaderABIError(
                "package-write-failed",
                "Native registry write failed.",
                path="$.outputDirectory",
            )
        original_write_output(root, relative_path, content)

    monkeypatch.setattr(
        native_loader_abi_package,
        "_write_output",
        fail_native_registry_write,
    )

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(
            reduced_runtime_package["loader_manifest_path"],
            output_root,
        )

    assert exc_info.value.code == "project.native-loader-abi.package-write-failed"
    assert not package_manifest_path.exists()


def test_runtime_loader_abi_package_rejects_modified_target_artifact(
    tmp_path, reduced_runtime_package
):
    package_dir = tmp_path / "runtime-package"
    shutil.copytree(reduced_runtime_package["package_dir"], package_dir)
    manifest_path = package_dir / "runtime-loader-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_package_path = manifest["loadUnits"][0]["packagePath"]
    artifact_path = package_dir / artifact_package_path
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")
    output_root = tmp_path / "abi-package"

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(manifest_path, output_root)

    assert exc_info.value.code == "project.native-loader-abi.artifact-size-mismatch"
    assert exc_info.value.details["packagePath"] == artifact_package_path
    assert not output_root.exists()


def test_runtime_loader_abi_package_rejects_same_size_artifact_hash_mismatch(
    tmp_path, reduced_runtime_package
):
    package_dir = tmp_path / "runtime-package"
    shutil.copytree(reduced_runtime_package["package_dir"], package_dir)
    manifest_path = package_dir / "runtime-loader-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_package_path = manifest["loadUnits"][0]["packagePath"]
    artifact_path = package_dir / artifact_package_path
    artifact_bytes = bytearray(artifact_path.read_bytes())
    artifact_bytes[0] ^= 0x01
    artifact_path.write_bytes(artifact_bytes)
    output_root = tmp_path / "abi-package"

    with pytest.raises(NativeLoaderABIError) as exc_info:
        build_native_loader_abi_package(manifest_path, output_root)

    assert exc_info.value.code == "project.native-loader-abi.artifact-hash-mismatch"
    assert exc_info.value.details["packagePath"] == artifact_package_path
    assert not output_root.exists()


@pytest.mark.parametrize("language", ("c", "c++"), ids=("c11", "cxx17"))
def test_packaged_runtime_loader_headers_compile(
    tmp_path, reduced_runtime_package, language
):
    package_root = tmp_path / "abi-package"
    package = build_native_loader_abi_package(
        reduced_runtime_package["loader_manifest_path"],
        package_root,
    )
    headers = {
        unit["target"]: (
            (package_root / unit["declarationsPath"]).read_text(encoding="utf-8")
        )
        for unit in package["units"]
    }

    _compile_headers(tmp_path, language, headers)


def _compile_packaged_execution_headers(language, package_root, package):
    compiler = _compiler_command(language)
    if compiler is None:
        pytest.skip(f"No native {language} compiler is available")

    units = {unit["target"]: unit for unit in package["units"]}
    assert set(units) == {"directx", "opengl"}
    symbols = {}
    for target, unit in units.items():
        execution_abi_path = package_root / unit["executionABIPath"]
        execution_abi = execution_abi_path.read_text(encoding="utf-8")
        match = re.search(
            r"static inline CrossTLNativeLoaderExecutionResult\s+"
            r"([A-Za-z_]\w*_execute)\(",
            execution_abi,
        )
        assert match is not None
        symbols[target] = match.group(1)
    assert len(set(symbols.values())) == len(symbols)

    suffix = ".c" if language == "c" else ".cpp"
    source_path = package_root / f"execution_contract{suffix}"
    source_path.write_text(
        textwrap.dedent(f"""
            #include "{units['directx']['executionABIPath']}"
            #include "{units['opengl']['executionABIPath']}"

            int main(void) {{
                CrossTLNativeLoaderExecutionResult directx_result =
                    {symbols['directx']}(NULL, NULL);
                CrossTLNativeLoaderExecutionResult opengl_result =
                    {symbols['opengl']}(NULL, NULL);
                return directx_result.error.code ==
                           CROSSTL_NATIVE_LOADER_CODE_INVALID_ARGUMENT &&
                       opengl_result.error.code ==
                           CROSSTL_NATIVE_LOADER_CODE_INVALID_ARGUMENT
                    ? 0 : 1;
            }}
            """).strip() + "\n",
        encoding="utf-8",
    )

    executable_names = {
        Path(part).name.lower() for part in compiler if not part.startswith("-")
    }
    msvc_style = bool(executable_names & {"cl", "cl.exe", "clang-cl", "clang-cl.exe"})
    if msvc_style:
        object_path = package_root / "execution_contract.obj"
        standard = "/std:c11" if language == "c" else "/std:c++17"
        command = compiler + [
            "/nologo",
            standard,
            "/W4",
            "/WX",
            "/c",
            str(source_path),
            f"/Fo{object_path}",
        ]
    else:
        object_path = package_root / "execution_contract.o"
        standard = "-std=c11" if language == "c" else "-std=c++17"
        command = compiler + [
            standard,
            "-pedantic-errors",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ]

    result = subprocess.run(
        command,
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert object_path.is_file()


@pytest.mark.parametrize("language", ("c", "c++"), ids=("c11", "cxx17"))
def test_packaged_execution_headers_compile_together(
    tmp_path, reduced_runtime_package, language
):
    package_root = tmp_path / "abi-package"
    package = build_native_loader_abi_package(
        reduced_runtime_package["loader_manifest_path"],
        package_root,
    )

    _compile_packaged_execution_headers(language, package_root, package)


def test_packaged_native_runtime_variant_registry_compiles(
    tmp_path, reduced_runtime_package
):
    compiler = _compiler_command("c++")
    if compiler is None:
        pytest.skip("No native C++ compiler is available")

    package_root = tmp_path / "abi-package"
    package = build_native_loader_abi_package(
        reduced_runtime_package["loader_manifest_path"],
        package_root,
    )
    runtime_registry = package["runtimeVariantRegistry"]
    assert runtime_registry["nativeHeader"]["available"] is True
    registry = json.loads(
        (package_root / runtime_registry["path"]).read_text(encoding="utf-8")
    )
    selected_key = sorted(registry["lookup"]["readyKeys"])[0]
    source_path = package_root / "native_runtime_variant_registry.cpp"
    source_path.write_text(
        textwrap.dedent(f"""
            #include "{runtime_registry['nativeHeader']['path']}"

            int main() {{
                const CrossTLNativeRuntimeVariantEntry *variant =
                    crosstl_native_runtime_variant_lookup(
                        {json.dumps(selected_key)});
                if (variant == nullptr) return 1;
                const uint32_t workgroup_count[3] = {{1u, 1u, 1u}};
                CrossTLNativeLoaderExecutionRequest request =
                    crosstl_native_runtime_variant_make_request(
                        variant, 0u, nullptr, workgroup_count);
                return request.target == nullptr ||
                       request.dispatch.workgroup_size[0] == 0u;
            }}
            """).strip() + "\n",
        encoding="utf-8",
    )

    executable_names = {
        Path(part).name.lower() for part in compiler if not part.startswith("-")
    }
    msvc_style = bool(executable_names & {"cl", "cl.exe", "clang-cl", "clang-cl.exe"})
    if msvc_style:
        object_path = package_root / "native_runtime_variant_registry.obj"
        command = compiler + [
            "/nologo",
            "/std:c++17",
            "/W4",
            "/WX",
            "/c",
            str(source_path),
            f"/Fo{object_path}",
        ]
    else:
        object_path = package_root / "native_runtime_variant_registry.o"
        command = compiler + [
            "-std=c++17",
            "-pedantic-errors",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ]
    result = subprocess.run(
        command,
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert object_path.is_file()


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_packaged_runtime_variant_registry_builds_exact_dispatch_request(
    tmp_path, reduced_runtime_package, target
):
    package_root = tmp_path / "abi-package"
    package = build_native_loader_abi_package(
        reduced_runtime_package["loader_manifest_path"],
        package_root,
    )
    registry = json.loads(
        (package_root / package["runtimeVariantRegistry"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    key, record = next(
        (key, record)
        for key, record in registry["variants"].items()
        if record["target"]["backend"] == target
    )
    descriptor = reduced_runtime_package["descriptors"][target]
    input_binding = next(
        binding for binding in descriptor["bindings"] if binding["access"] == "read"
    )
    output_binding = next(
        binding
        for binding in descriptor["bindings"]
        if binding["access"] == "read_write"
    )

    request = build_runtime_variant_dispatch_request(
        registry,
        key,
        package_root,
        {
            input_binding["name"]: {
                "dtype": "float32",
                "shape": [4],
                "values": [1.0, 2.0, 3.0, 4.0],
            }
        },
        {
            output_binding["name"]: {
                "dtype": "float32",
                "shape": [4],
            }
        },
        {
            "workgroupCount": [4, 1, 1],
            "globalSize": [4, 1, 1],
        },
    )

    assert request.artifact["target"] == target
    assert request.fixture.selector.artifact_id == record["artifact"]["id"]
    assert request.execution_plan.dispatch.workgroup_size == (1, 1, 1)
