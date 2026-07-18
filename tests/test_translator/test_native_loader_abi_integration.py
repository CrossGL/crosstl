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
    NativeLoaderABIError,
    build_native_loader_abi_package,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from crosstl.project.native_loader_abi import (
    build_native_loader_abi_descriptor,
    generate_native_loader_declarations,
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
    return {
        "source_path": source_path,
        "package_dir": package_dir,
        "loader_manifest": loader_manifest,
        "loader_manifest_path": loader_manifest_path,
        "units": units,
        "descriptors": descriptors,
        "headers": headers,
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
    assert first["success"] is True
    assert first["summary"] == {
        "unitCount": 2,
        "targetCount": 2,
        "generatedFileCount": 5,
    }
    assert len(first["generatedFiles"]) == first["summary"]["generatedFileCount"]
    assert first["generatedFiles"][0] == {
        "path": NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
        "kind": "native-loader-abi-package-manifest",
    }
    assert [unit["target"] for unit in first["units"]] == ["directx", "opengl"]
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
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        assert descriptor == reduced_runtime_package["descriptors"][target]
        assert (
            declarations_path.read_text(encoding="utf-8")
            == reduced_runtime_package["headers"][target]
        )
        assert unit["descriptorHash"] == {
            "algorithm": "sha256",
            "value": _sha256(descriptor_path),
        }
        assert unit["declarationsHash"] == {
            "algorithm": "sha256",
            "value": _sha256(declarations_path),
        }


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
