"""Keep compiler paths usable without moving project artifacts or includes."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

from crosstl.project import (
    directx_toolchain,
    native_deferred_compilation_runtime,
    pipeline,
)


@pytest.mark.parametrize(
    "value",
    [
        r"C:\project\shader.hlsl",
        r"relative\shader.hlsl",
        r"\\server\share\shader.hlsl",
        r"\\?\C:\already\extended.hlsl",
        r"\\.\NUL",
    ],
)
def test_dxc_short_and_explicit_device_paths_are_unchanged(monkeypatch, value):
    monkeypatch.setattr(directx_toolchain.sys, "platform", "win32")
    path = PureWindowsPath(value)
    assert directx_toolchain.dxc_file_path(path) == str(path)


@pytest.mark.parametrize("length", [259, 260, 320])
@pytest.mark.parametrize("root", ["C:\\", "\\\\server\\share\\"])
def test_dxc_long_windows_paths_use_extended_namespace(monkeypatch, length, root):
    value = root + "directory\\" * 15
    value += "x" * (length - len(value) - len(".hlsl")) + ".hlsl"
    monkeypatch.setattr(directx_toolchain.sys, "platform", "win32")
    expected = value
    if length >= 260:
        expected = (
            "\\\\.\\UNC\\" + value[2:] if root.startswith("\\\\") else "\\\\.\\" + value
        )
    actual = directx_toolchain.dxc_file_path(PureWindowsPath(value))
    assert actual == expected
    assert directx_toolchain.dxc_file_path(PureWindowsPath(actual)) == expected


def test_dxc_windows_path_normalizes_parent_components_before_prefix(monkeypatch):
    value = "C:\\project\\" + "directory\\" * 30
    value += "..\\shader.hlsl"
    monkeypatch.setattr(directx_toolchain.sys, "platform", "win32")
    assert directx_toolchain.dxc_file_path(PureWindowsPath(value)) == (
        "\\\\.\\C:\\project\\" + "directory\\" * 29 + "shader.hlsl"
    )


def test_dxc_path_limit_counts_utf16_code_units(monkeypatch):
    value = "C:\\" + "directory\\" * 20 + "\U00010000" * 30 + ".hlsl"
    assert len(value) < 260 <= len(value.encode("utf-16-le")) // 2
    monkeypatch.setattr(directx_toolchain.sys, "platform", "win32")
    assert directx_toolchain.dxc_file_path(PureWindowsPath(value)) == "\\\\.\\" + value


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_dxc_non_windows_paths_are_unchanged(monkeypatch, platform):
    value = "/project/" + "directory/" * 30 + "shader.hlsl"
    monkeypatch.setattr(directx_toolchain.sys, "platform", platform)
    assert directx_toolchain.dxc_file_path(PurePosixPath(value)) == value


@pytest.mark.parametrize(
    "source,profiles",
    [
        ("[numthreads(1, 1, 1)] void CSMain() {}", ["cs_6_0"]),
        ("float4 PSMain() : SV_Target { return 1.0; }", ["ps_6_0"]),
        (
            "float4 VSMain() : SV_Position { return 0.0; }\n"
            "float4 PSMain() : SV_Target { return 1.0; }",
            ["vs_6_0", "ps_6_0"],
        ),
        ('[shader("raygeneration")] void RayGen() {}', ["lib_6_3"]),
    ],
)
def test_all_dxc_validation_commands_format_input_paths(
    tmp_path, monkeypatch, source, profiles
):
    shader = tmp_path / "shader.hlsl"
    shader.write_text(source, encoding="utf-8")
    formatted = []

    def format_path(path):
        formatted.append(path)
        return "formatted-input.hlsl"

    monkeypatch.setattr(pipeline, "dxc_file_path", format_path)
    commands = pipeline._directx_dxc_smoke_commands("dxc", shader)
    assert formatted == [shader] * len(profiles)
    assert [command[command.index("-T") + 1] for command in commands] == profiles
    assert all(command[-3] == "formatted-input.hlsl" for command in commands)


def test_deferred_dxc_formats_source_output_and_include_paths(tmp_path, monkeypatch):
    shader = tmp_path / "shader.hlsl"
    shader.write_text("[numthreads(1, 1, 1)] void CSMain() {}", encoding="utf-8")
    output = tmp_path / "shader.dxil"
    include = tmp_path / "include"
    formatted = []

    def format_path(path):
        formatted.append(path)
        return "formatted:" + str(path)

    monkeypatch.setattr(
        native_deferred_compilation_runtime, "dxc_file_path", format_path
    )
    command = native_deferred_compilation_runtime._compiler_command(
        {
            "variant": {"compileDefines": {"COUNT": 7}},
            "target": {
                "backend": "directx",
                "profile": "cs_6_0",
                "entryPoint": "CSMain",
            },
        },
        {
            "source": {"path": str(shader)},
            "sourceRoot": str(tmp_path),
            "includeDirectories": [str(include)],
        },
        SimpleNamespace(executable="dxc"),
        output_path=output,
    )
    assert formatted == [tmp_path, include, output, shader]
    assert command == (
        "dxc",
        "-T",
        "cs_6_0",
        "-E",
        "CSMain",
        "-DCOUNT=7",
        "-I",
        "formatted:" + str(tmp_path),
        "-I",
        "formatted:" + str(include),
        "-Fo",
        "formatted:" + str(output),
        "formatted:" + str(shader),
    )


def test_windows_path_regression_uses_existing_project_job():
    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text(
            encoding="utf-8"
        )
    )
    jobs = [
        job
        for job in workflow["jobs"].values()
        if any(
            step.get("name") == "Validate DirectX compiler paths"
            for step in job.get("steps", [])
        )
    ]
    assert len(jobs) == 1
    job = jobs[0]
    assert any(
        step.get("name") == "Validate Metal builtin ownership" for step in job["steps"]
    )
    step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Validate DirectX compiler paths"
    )
    assert step["if"] == "runner.os == 'Windows'"
    assert step["env"]["CROSTL_REQUIRE_DXC_PATHS"] == "1"
    assert "--timeout-seconds 90" in step["run"]
    assert "-n auto" in step["run"]
    assert "tests/test_translator/test_directx_toolchain_paths.py" in step["run"]


@pytest.mark.parametrize("invalid", [False, True], ids=["includes", "diagnostic"])
def test_native_dxc_validates_long_project_paths(tmp_path, invalid):
    required = os.environ.get("CROSTL_REQUIRE_DXC_PATHS") == "1"
    if required and sys.platform != "win32":
        pytest.fail("Required Windows path validation cannot run on this platform.")
    if shutil.which("dxc") is None:
        if required:
            pytest.fail("Long-path validation requires Windows and DXC.")
        pytest.skip("DXC is not installed")

    directory = tmp_path
    while len(str(directory)) < 310:
        directory /= "nested project artifacts"
    directory.mkdir(parents=True)
    include = directory / "include"
    include.mkdir()
    (include / "value.hlsli").write_text(
        '#include "../factor.hlsli"\nfloat value() { return FACTOR; }\n',
        encoding="utf-8",
    )
    (directory / "factor.hlsli").write_text("#define FACTOR 7.0\n", encoding="utf-8")
    source = (
        '#include "include/value.hlsli"\n'
        "RWStructuredBuffer<float> result : register(u0);\n"
        "[numthreads(1, 1, 1)] void CSMain() { result[0] = value(); }\n"
    )
    if invalid:
        source += "#error retained_long_path_diagnostic\n"
    shader = directory / "shader.hlsl"
    shader.write_text(source, encoding="utf-8")
    relative = shader.relative_to(tmp_path).as_posix()
    runs = pipeline._run_toolchain_smoke(
        [
            {
                "source": "shader.cgl",
                "target": "directx",
                "path": relative,
                "status": "translated",
            }
        ],
        tmp_path,
    )
    (tmp_path / "validation.json").write_text(
        json.dumps(runs, indent=2) + "\n", encoding="utf-8"
    )
    assert len(runs) == 1
    run = runs[0]
    assert run["path"] == relative
    assert shader.read_text(encoding="utf-8") == source
    if sys.platform == "win32":
        assert run["command"][-3].startswith("\\\\.\\")
    else:
        assert run["command"][-3] == str(shader)
    if invalid:
        assert run["status"] == "failed"
        assert "retained_long_path_diagnostic" in run["stderr"]
        return
    assert run["status"] == "ok", run
    module = directory / "shader.dxil"
    command = [*run["command"][:-1], directx_toolchain.dxc_file_path(module)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert module.read_bytes().startswith(b"DXBC")
    (tmp_path / "module.json").write_text(
        json.dumps(
            {
                "command": command,
                "sourceSha256": hashlib.sha256(shader.read_bytes()).hexdigest(),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                "sourceLength": len(str(shader)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
