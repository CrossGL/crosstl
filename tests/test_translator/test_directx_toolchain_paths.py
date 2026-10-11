"""Keep compiler paths usable without moving project artifacts or includes."""

import ctypes
import hashlib
import json
import os
import posixpath
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

from crosstl.project import (
    directx_toolchain,
    dxc_compiler,
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
            "\\\\?\\UNC\\" + value[2:] if root.startswith("\\\\") else "\\\\?\\" + value
        )
    actual = directx_toolchain.dxc_file_path(PureWindowsPath(value))
    assert actual == expected
    assert directx_toolchain.dxc_file_path(PureWindowsPath(actual)) == expected


def test_dxc_windows_path_normalizes_parent_components_before_prefix(monkeypatch):
    value = "C:\\project\\" + "directory\\" * 30
    value += "..\\shader.hlsl"
    monkeypatch.setattr(directx_toolchain.sys, "platform", "win32")
    assert directx_toolchain.dxc_file_path(PureWindowsPath(value)) == (
        "\\\\?\\C:\\project\\" + "directory\\" * 29 + "shader.hlsl"
    )


def test_dxc_path_limit_counts_utf16_code_units(monkeypatch):
    value = "C:\\" + "directory\\" * 20 + "\U00010000" * 30 + ".hlsl"
    assert len(value) < 260 <= len(value.encode("utf-16-le")) // 2
    monkeypatch.setattr(directx_toolchain.sys, "platform", "win32")
    assert directx_toolchain.dxc_file_path(PureWindowsPath(value)) == "\\\\?\\" + value


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
    assert pipeline._validation_toolchain_run_tool_name(run) == "dxc"
    assert not pipeline._toolchain_run_contract_reasons(0, run, root_path=tmp_path)
    assert shader.read_text(encoding="utf-8") == source
    if sys.platform == "win32":
        assert run["command"][0] == sys.executable
        assert Path(run["command"][1]).name == "dxc_compiler.py"
        assert run["command"][run["command"].index("--source") + 1] == str(shader)
    else:
        assert run["command"][-3] == str(shader)
    if invalid:
        assert run["status"] == "failed"
        assert "retained_long_path_diagnostic" in run["stderr"]
        return
    assert run["status"] == "ok", run
    module = directory / "shader.dxil"
    command = list(run["command"])
    output_flag = "--output" if sys.platform == "win32" else "-Fo"
    command[command.index(output_flag) + 1] = directx_toolchain.dxc_file_path(module)
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


@pytest.mark.parametrize("root", ["C:\\", "\\\\server\\share\\"])
def test_include_handler_normalizes_extended_parent_paths(monkeypatch, root):
    monkeypatch.setattr(dxc_compiler.sys, "platform", "win32")
    value = root + "directory\\" * 30
    expected = directx_toolchain.dxc_file_path(PureWindowsPath(value + "factor.hlsli"))
    assert dxc_compiler._include_path(value + "include\\..\\factor.hlsli") == expected
    assert (
        dxc_compiler._include_path(
            expected.rsplit("\\", 1)[0] + "\\include\\..\\factor.hlsli"
        )
        == expected
    )


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_dxc_api_non_windows_include_paths(monkeypatch, platform):
    monkeypatch.setattr(dxc_compiler.sys, "platform", platform)
    monkeypatch.setattr(dxc_compiler, "os", SimpleNamespace(path=posixpath))
    assert (
        dxc_compiler._include_path("/project/include/../factor.hlsli")
        == "/project/factor.hlsli"
    )


def test_dxc_api_handler_interface_lifetime_and_missing_include():
    calls = []

    @dxc_compiler._CALL(
        dxc_compiler._HRESULT,
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    def load(this, filename, result):
        calls.append(filename)
        result[0] = None
        return -2147024894

    table = (ctypes.c_void_p * 4)(0, 0, 0, ctypes.cast(load, ctypes.c_void_p).value)
    default = ctypes.pointer(ctypes.cast(table, ctypes.POINTER(ctypes.c_void_p)))
    handler = dxc_compiler._IncludeHandler(default)
    pointer = ctypes.byref(handler)
    result = ctypes.c_void_p()
    query = handler.callbacks[0]
    assert (
        query(
            pointer, ctypes.byref(dxc_compiler._INCLUDE_HANDLER), ctypes.byref(result)
        )
        == 0
    )
    assert result.value == ctypes.addressof(handler)
    assert handler.references == 2
    assert handler.callbacks[2](pointer) == 1
    unsupported = dxc_compiler._Guid.parse("ffffffff-ffff-ffff-ffff-ffffffffffff")
    assert (
        query(pointer, ctypes.byref(unsupported), ctypes.byref(result)) == -2147467262
    )
    assert result.value is None and handler.references == 1
    assert (
        handler.callbacks[3](pointer, "missing.hlsli", ctypes.byref(result))
        == -2147024894
    )
    assert result.value is None and len(calls) == 1 and not handler.errors


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_dxc_library_bridge_preserves_compiler_arguments(monkeypatch, platform):
    source = PureWindowsPath("C:\\" + "directory\\" * 30 + "shader.hlsl")
    monkeypatch.setattr(directx_toolchain.sys, "platform", platform)
    monkeypatch.setattr(
        directx_toolchain.shutil, "which", lambda value: "C:/DXC/dxc.exe"
    )
    arguments = [
        "-T",
        "cs_6_2",
        "-E",
        "CSMain",
        "-DVALUE=7",
        "-I",
        "C:/includes",
        "-enable-16bit-types",
        "-WX",
    ]
    command = [
        "dxc",
        *arguments,
        directx_toolchain.dxc_file_path(source),
        "-Fo",
        "output.dxil",
    ]
    bridge = directx_toolchain.dxc_long_path_command(command, source)
    if platform != "win32":
        assert bridge == command
        return
    assert bridge[0] == sys.executable
    assert bridge[bridge.index("--compiler") + 1] == "dxc"
    assert bridge[bridge.index("--source") + 1] == str(source)
    assert bridge[bridge.index("--output") + 1] == "output.dxil"
    assert bridge[bridge.index("--") + 1 :] == arguments
    assert directx_toolchain.dxc_library_command_tool(bridge) == "dxc"


def test_dxc_unavailable_tool_does_not_become_available_through_bridge(monkeypatch):
    source = PureWindowsPath("C:\\" + "directory\\" * 30 + "shader.hlsl")
    monkeypatch.setattr(directx_toolchain.sys, "platform", "win32")
    monkeypatch.setattr(directx_toolchain.shutil, "which", lambda value: None)
    command = [
        "dxc",
        "-T",
        "cs_6_0",
        directx_toolchain.dxc_file_path(source),
        "-Fo",
        "NUL",
    ]
    assert directx_toolchain.dxc_long_path_command(command, source) == command


def test_dxc_long_defines_do_not_select_library_bridge(monkeypatch):
    source = PureWindowsPath(r"C:\project\shader.hlsl")
    monkeypatch.setattr(directx_toolchain.sys, "platform", "win32")
    command = ["dxc", "-DVALUE=" + "1" * 300, str(source), "-Fo", "NUL"]
    assert directx_toolchain.dxc_long_path_command(command, source) == command


@pytest.mark.parametrize("field", ["-I", "-Fo"])
def test_dxc_long_include_and_output_paths_select_library_bridge(monkeypatch, field):
    source = PureWindowsPath(r"C:\project\shader.hlsl")
    long_path = (
        "C:\\" + "directory\\" * 30 + ("out.dxil" if field == "-Fo" else "include")
    )
    monkeypatch.setattr(directx_toolchain.sys, "platform", "win32")
    monkeypatch.setattr(
        directx_toolchain.shutil, "which", lambda value: "C:/DXC/dxc.exe"
    )
    command = [
        "dxc",
        "-I",
        long_path if field == "-I" else r"C:\includes",
        "-Fo",
        long_path if field == "-Fo" else "NUL",
        str(source),
    ]
    bridge = directx_toolchain.dxc_long_path_command(command, source)
    assert directx_toolchain.dxc_library_command_tool(bridge) == "dxc"
    assert bridge[bridge.index("--") + 1 :] == command[1:3]


@pytest.mark.parametrize(
    "index,value",
    [
        (0, "powershell.exe"),
        (1, "C:/unrelated/dxc_compiler.py"),
        (2, "--other"),
        (3, "glslangValidator"),
        (4, "--other"),
        (6, "--other"),
        (8, "--other"),
    ],
)
def test_report_rejects_unrecognized_or_mismatched_dxc_bridge(index, value):
    command = [
        "C:/Python/python.exe",
        "C:/installed/crosstl/project/dxc_compiler.py",
        "--compiler",
        "dxc",
        "--source",
        "shader.hlsl",
        "--output",
        "NUL",
        "--",
        "-T",
        "cs_6_0",
    ]
    run = {
        "source": "shader.cgl",
        "target": "directx",
        "path": "shader.hlsl",
        "status": "ok",
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "command": command,
    }
    assert not pipeline._toolchain_run_contract_reasons(0, run)
    assert pipeline._validation_toolchain_run_status_by_tool([run]) == {
        "dxc": {"runCount": 1, "okCount": 1, "failedCount": 0}
    }
    command[index] = value
    assert any(
        "configured validation tool" in reason
        for reason in pipeline._toolchain_run_contract_reasons(0, run)
    )


def test_dxc_api_callback_exceptions_fail_closed(monkeypatch):
    @dxc_compiler._CALL(
        dxc_compiler._HRESULT,
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    def load(this, filename, result):
        pytest.fail("An invalid path must not reach the default include handler")

    table = (ctypes.c_void_p * 4)(0, 0, 0, ctypes.cast(load, ctypes.c_void_p).value)
    default = ctypes.pointer(ctypes.cast(table, ctypes.POINTER(ctypes.c_void_p)))
    handler = dxc_compiler._IncludeHandler(default)

    def reject(value):
        raise ValueError("invalid include path")

    monkeypatch.setattr(dxc_compiler, "_include_path", reject)
    result = ctypes.c_void_p()
    assert (
        handler.callbacks[3](
            ctypes.byref(handler), "header.hlsli", ctypes.byref(result)
        )
        == -2147467259
    )
    assert handler.errors == ["invalid include path"] and result.value is None
