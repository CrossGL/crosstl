from __future__ import annotations

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
