"""Repository-independent package and native compiler test helpers."""

import json
import subprocess

from crosstl.project import (
    build_native_loader_abi_descriptor,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
)
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
from tests.test_translator.test_metal_builtin_ownership import _compile


def run(command, directory, name, *, timeout=120):
    directory.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in command]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (directory / f"{name}.stdout").write_text(result.stdout, encoding="utf-8")
    (directory / f"{name}.stderr").write_text(result.stderr, encoding="utf-8")
    (directory / f"{name}.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timeoutSeconds": timeout,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def compile_metal(artifact, root, directory):
    air, module = directory / "kernel.air", directory / "kernel.metallib"
    run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "metal",
            "-Werror",
            "-I",
            root,
            "-c",
            artifact,
            "-o",
            air,
        ],
        directory,
        "compile",
    )
    run(["xcrun", "--sdk", "macosx", "metallib", air, "-o", module], directory, "link")
    assert module.stat().st_size > 0
    return module


def _validate_generated_artifact(path, work, target):
    if target == "directx":
        module = work / "kernel.dxil"
        commands = [
            (
                "dxc",
                [
                    "dxc",
                    *dxc_compiler_arguments_for_source(
                        path.read_text(encoding="utf-8")
                    ),
                    "-WX",
                    "-T",
                    "cs_6_2",
                    "-E",
                    "CSMain",
                    path,
                    "-Fo",
                    module,
                ],
            )
        ]
    else:
        module = work / "kernel.spv"
        commands = [
            (
                "glslang",
                [
                    "glslangValidator",
                    "--target-env",
                    "opengl",
                    "--target-env",
                    "spirv1.3",
                    "-S",
                    "comp",
                    path,
                    "-o",
                    module,
                ],
            ),
            ("spirv-val", ["spirv-val", "--target-env", "spv1.3", module]),
        ]
    for label, command in commands:
        run(command, work, label, timeout=300)
        log = json.loads((work / f"{label}.json").read_text(encoding="utf-8"))
        assert "warning:" not in (log["stdout"] + log["stderr"]).lower(), log
    assert module.stat().st_size > 0


def _prepare_native_package(report, work):
    report.write_json(work / "report.json")
    manifest = build_runtime_artifact_manifest(work / "report.json")
    assert manifest["success"], manifest
    (work / "artifacts.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    package = work / "package"
    assert build_runtime_package(work / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"], loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    (work / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )
    return descriptor, package


def _validate(path, work, target):
    if target == "metal":
        return _compile(
            path.read_text(), target, work, metal_compile_flags=("-fno-fast-math",)
        )[1]
    _validate_generated_artifact(path, work, target)
    return work / ("kernel.dxil" if target == "directx" else "kernel.spv")
