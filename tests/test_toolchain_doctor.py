import json
import textwrap
from unittest import mock

import pytest

import crosstl._crosstl as crosstl_cli
from crosstl.project import (
    toolchain_doctor_report,
    translate_project,
    validate_project_report,
)
from crosstl.project.doctor import DOCTOR_VERSION_COMMANDS
from crosstl.project.pipeline import TOOLCHAIN_BY_BACKEND, _tool_status

SHADER = textwrap.dedent("""
    shader DoctorShader {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float* values;

            void main() {
                values[0] = 1.0;
            }
        }
    }
    """)


def test_tool_status_names_missing_tools(monkeypatch):
    monkeypatch.setattr("crosstl.project.pipeline.shutil.which", lambda _name: None)
    status = _tool_status("cuda")
    assert status["status"] == "unavailable"
    assert status["tools"] == [{"name": "nvcc", "path": None, "available": False}]


def test_toolchain_doctor_report_lists_every_backend(monkeypatch):
    monkeypatch.setattr("crosstl.project.pipeline.shutil.which", lambda _name: None)
    payload = toolchain_doctor_report()
    targets = {row["target"] for row in payload["tools"]}
    assert targets == set(TOOLCHAIN_BY_BACKEND)
    cuda = next(row for row in payload["tools"] if row["target"] == "cuda")
    assert cuda["status"] == "unavailable"
    assert cuda["tools"][0]["name"] == "nvcc"
    assert cuda["tools"][0]["path"] is None


def test_every_backend_has_a_version_command():
    assert set(DOCTOR_VERSION_COMMANDS) == set(TOOLCHAIN_BY_BACKEND)
    assert DOCTOR_VERSION_COMMANDS["directx"] == ("dxc", "--version")
    assert DOCTOR_VERSION_COMMANDS["slang"] == ("slangc", "-v")


def test_toolchain_doctor_includes_version_when_available(monkeypatch):
    monkeypatch.setattr(
        "crosstl.project.pipeline.shutil.which", lambda _name: "/usr/bin/fake"
    )
    monkeypatch.setattr(
        "crosstl.project.doctor.subprocess.run",
        lambda *args, **kwargs: mock.Mock(
            stdout="nvcc 12.0\n", stderr="", returncode=0
        ),
    )
    payload = toolchain_doctor_report()
    cuda = next(row for row in payload["tools"] if row["target"] == "cuda")
    assert cuda["status"] == "available"
    assert cuda["version"] == "nvcc 12.0"


def test_toolchain_doctor_reports_no_version_when_the_command_cannot_run(
    monkeypatch,
):
    monkeypatch.setattr(
        "crosstl.project.pipeline.shutil.which", lambda _name: "/usr/bin/fake"
    )

    def refuse(*args, **kwargs):
        raise OSError("exec format error")

    monkeypatch.setattr("crosstl.project.doctor.subprocess.run", refuse)
    payload = toolchain_doctor_report()
    cuda = next(row for row in payload["tools"] if row["target"] == "cuda")
    assert cuda["status"] == "available"
    assert cuda["version"] is None


def test_toolchain_doctor_reports_no_version_when_the_command_fails(monkeypatch):
    monkeypatch.setattr(
        "crosstl.project.pipeline.shutil.which", lambda _name: "/usr/bin/fake"
    )
    monkeypatch.setattr(
        "crosstl.project.doctor.subprocess.run",
        lambda *args, **kwargs: mock.Mock(
            stdout="", stderr="error: unknown argument: '--version'\n", returncode=1
        ),
    )
    payload = toolchain_doctor_report()
    slang = next(row for row in payload["tools"] if row["target"] == "slang")
    assert slang["status"] == "available"
    assert slang["versionCommand"] == ["slangc", "-v"]
    assert slang["version"] is None


def test_doctor_command_prints_json_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr("crosstl.project.pipeline.shutil.which", lambda _name: None)

    assert crosstl_cli.main(["doctor", "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert {row["target"] for row in payload["tools"]} == set(TOOLCHAIN_BY_BACKEND)


def test_doctor_command_prints_one_line_per_backend(monkeypatch, capsys):
    monkeypatch.setattr("crosstl.project.pipeline.shutil.which", lambda _name: None)

    assert crosstl_cli.main(["doctor"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Validation toolchains"
    assert "- cuda: unavailable; nvcc=not found" in lines
    assert "- vulkan: unavailable; spirv-val=not found, spirv-as=not found" in lines
    assert len(lines) == 1 + len(TOOLCHAIN_BY_BACKEND)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("opengl", "glslangValidator is not on PATH"),
        ("vulkan", "spirv-val, spirv-as are not on PATH"),
    ],
)
def test_validation_warning_names_every_missing_tool(
    tmp_path, monkeypatch, target, message
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "shader.cgl").write_text(SHADER, encoding="utf-8")
    monkeypatch.setattr("crosstl.project.pipeline.shutil.which", lambda _name: None)

    report = translate_project(repo, targets=[target], output_dir="out")
    report_path = repo / "portability-report.json"
    report.write_json(report_path)
    validation = validate_project_report(report_path)

    [diagnostic] = [
        diagnostic
        for diagnostic in validation["diagnostics"]
        if diagnostic["code"] == "project.validate.toolchain-unavailable"
    ]
    assert diagnostic["target"] == target
    assert diagnostic["message"] == message
