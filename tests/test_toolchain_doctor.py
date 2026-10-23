from unittest import mock

from crosstl.project import toolchain_doctor_report
from crosstl.project.pipeline import TOOLCHAIN_BY_BACKEND, _tool_status


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


def test_toolchain_doctor_includes_version_when_available(monkeypatch):
    monkeypatch.setattr(
        "crosstl.project.pipeline.shutil.which", lambda _name: "/usr/bin/fake"
    )
    monkeypatch.setattr(
        "crosstl.project.pipeline.subprocess.run",
        lambda *args, **kwargs: mock.Mock(
            stdout="nvcc 12.0\n", stderr="", returncode=0
        ),
    )
    payload = toolchain_doctor_report()
    cuda = next(row for row in payload["tools"] if row["target"] == "cuda")
    assert cuda["status"] == "available"
    assert cuda["version"] == "nvcc 12.0"
