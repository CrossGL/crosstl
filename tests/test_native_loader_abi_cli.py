import copy
import json
from pathlib import Path

import crosstl.project as project_api
from crosstl._crosstl import main
from crosstl.project.pipeline import (
    REPORT_SCHEMA_VERSION,
    RUNTIME_LOADER_MANIFEST_KIND,
    RUNTIME_LOADER_MANIFEST_LOAD_UNIT_FIELDS,
)


def _load_unit(target="directx"):
    entry_point = "CSMain" if target == "directx" else "main"
    extension = "hlsl" if target == "directx" else "comp"
    unit = {
        "id": f"copy:{target}",
        "target": target,
        "adapterKind": f"{target}-compute-adapter",
        "artifactFormat": "HLSL source" if target == "directx" else "GLSL source",
        "packagePath": f"artifacts/{target}/copy.{extension}",
        "source": "kernels/copy.metal",
        "sourcePath": f"out/{target}/copy.{extension}",
        "sourceBackend": "metal",
        "stage": "compute",
        "variant": None,
        "defines": {},
        "entryPoint": {
            "source": "copy_values",
            "target": entry_point,
            "stage": "compute",
        },
        "templateMaterialization": None,
        "specializationMaterialization": None,
        "specializationConstants": [],
        "provenance": {"source": "kernels/copy.metal", "target": target},
        "sourceHash": None,
        "sourceSizeBytes": 128,
        "hash": {"algorithm": "sha256", "value": "2" * 64},
        "sizeBytes": 256,
        "sourceRemap": None,
        "hostInterface": {
            "status": "ready",
            "entryPointCount": 1,
            "resourceCount": 1,
            "constantCount": 0,
            "specializationConstantCount": 0,
            "entryPoints": [
                {
                    "name": entry_point,
                    "stage": "compute",
                    "executionConfig": {"workgroupSize": [64, 1, 1]},
                }
            ],
            "resources": [
                {
                    "name": "values",
                    "kind": "buffer",
                    "type": "float[]",
                    "set": 0,
                    "binding": 0,
                    "access": "read_write",
                }
            ],
            "constants": [],
            "specializationConstants": [],
        },
        "requiredTools": [],
        "hostResponsibilities": [],
        "loadSteps": [],
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


def _write_manifest(tmp_path, *units):
    manifest_path = tmp_path / "runtime-loader-manifest.json"
    manifest_path.write_text(
        json.dumps(_loader_manifest(*units), indent=2),
        encoding="utf-8",
    )
    return manifest_path


def test_project_exports_native_loader_abi_contract():
    assert project_api.NATIVE_LOADER_ABI_KIND == (
        "crosstl-native-loader-abi-descriptor"
    )
    assert project_api.NATIVE_LOADER_ABI_VERSION == 1
    assert issubclass(project_api.NativeLoaderABIError, ValueError)
    assert callable(project_api.build_native_loader_abi_descriptor)
    assert callable(project_api.generate_native_loader_declarations)


def test_native_loader_abi_cli_emits_json_descriptor(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, _load_unit("directx"))

    result = main(["native-loader-abi", str(manifest_path)])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == project_api.NATIVE_LOADER_ABI_KIND
    assert payload["unitId"] == "copy:directx"
    assert payload["target"] == "directx"
    assert payload["entryPoint"]["name"] == "CSMain"


def test_native_loader_abi_cli_writes_deterministic_declarations(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, _load_unit("opengl"))
    declarations_path = tmp_path / "include" / "copy_opengl.h"

    result = main(
        [
            "native-loader-abi",
            str(manifest_path),
            "--declarations-output",
            str(declarations_path),
        ]
    )

    assert result == 0
    descriptor = json.loads(capsys.readouterr().out)
    declarations = declarations_path.read_text(encoding="utf-8")
    assert declarations == project_api.generate_native_loader_declarations(descriptor)
    assert "CrossTLNativeLoaderUnitDescriptor" in declarations
    assert '"opengl"' in declarations


def test_native_loader_abi_cli_selects_one_multi_unit_entry(tmp_path, capsys):
    manifest_path = _write_manifest(
        tmp_path,
        _load_unit("directx"),
        _load_unit("opengl"),
    )

    result = main(
        [
            "native-loader-abi",
            str(manifest_path),
            "--load-unit",
            "copy:opengl",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["unitId"] == "copy:opengl"
    assert payload["target"] == "opengl"


def test_native_loader_abi_cli_reports_blocked_unit_as_json(tmp_path, capsys):
    unit = copy.deepcopy(_load_unit("directx"))
    unit["blockers"] = [{"kind": "resolve-host-interface-metadata"}]
    manifest_path = _write_manifest(tmp_path, unit)

    result = main(["native-loader-abi", str(manifest_path)])

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "code": "project.native-loader-abi.load-unit-blocked",
        "details": {"blockers": unit["blockers"]},
        "message": (
            "Blocked runtime loader units cannot produce native ABI declarations."
        ),
        "path": "$.blockers",
        "severity": "error",
    }


def test_native_loader_abi_cli_rejects_resolved_output_path_collision(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, _load_unit())
    output_path = tmp_path / "generated" / "descriptor.json"
    declarations_path = output_path.parent / "nested" / ".." / output_path.name

    result = main(
        [
            "native-loader-abi",
            str(manifest_path),
            "--output",
            str(output_path),
            "--declarations-output",
            str(declarations_path),
        ]
    )

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "code": "project.native-loader-abi.output-path-conflict",
        "details": {
            "declarationsOutput": str(declarations_path),
            "output": str(output_path),
            "resolvedPath": str(output_path.resolve()),
        },
        "message": (
            "Descriptor and declaration outputs must resolve to different paths."
        ),
        "path": "$.declarationsOutput",
        "severity": "error",
    }
    assert not output_path.exists()


def test_native_loader_abi_cli_reports_manifest_filesystem_failure(tmp_path, capsys):
    manifest_path = tmp_path / "missing.json"

    result = main(["native-loader-abi", str(manifest_path)])

    assert result == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "project.native-loader-abi.manifest-read-failed"
    assert payload["path"] == "$.loaderManifest"
    assert payload["details"] == {"source": str(manifest_path)}
    assert captured.err == ""


def test_native_loader_abi_cli_reports_manifest_unicode_failure(tmp_path, capsys):
    manifest_path = tmp_path / "runtime-loader-manifest.json"
    manifest_path.write_bytes(b"\xff\xfe")

    result = main(["native-loader-abi", str(manifest_path)])

    assert result == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "project.native-loader-abi.manifest-read-failed"
    assert payload["path"] == "$.loaderManifest"
    assert payload["details"] == {"source": str(manifest_path)}
    assert captured.err == ""


def test_native_loader_abi_cli_reports_descriptor_filesystem_failure(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, _load_unit())
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    output_path = blocked_parent / "descriptor.json"

    result = main(
        ["native-loader-abi", str(manifest_path), "--output", str(output_path)]
    )

    assert result == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "project.native-loader-abi.output-write-failed"
    assert payload["path"] == "$.output"
    assert payload["details"] == {"destination": str(output_path)}
    assert captured.err == ""


def test_native_loader_abi_cli_reports_descriptor_unicode_failure(
    tmp_path, capsys, monkeypatch
):
    manifest_path = _write_manifest(tmp_path, _load_unit())
    output_path = tmp_path / "descriptor.json"
    _fail_path_write_with_unicode(monkeypatch, output_path)

    result = main(
        ["native-loader-abi", str(manifest_path), "--output", str(output_path)]
    )

    assert result == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "project.native-loader-abi.output-write-failed"
    assert payload["details"] == {"destination": str(output_path)}
    assert captured.err == ""


def test_native_loader_abi_cli_reports_declaration_filesystem_failure(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, _load_unit())
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    declarations_path = blocked_parent / "copy.h"

    result = main(
        [
            "native-loader-abi",
            str(manifest_path),
            "--declarations-output",
            str(declarations_path),
        ]
    )

    assert result == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "project.native-loader-abi.declarations-write-failed"
    assert payload["path"] == "$.declarationsOutput"
    assert payload["details"] == {"destination": str(declarations_path)}
    assert captured.err == ""


def test_native_loader_abi_cli_reports_declaration_unicode_failure(
    tmp_path, capsys, monkeypatch
):
    manifest_path = _write_manifest(tmp_path, _load_unit())
    declarations_path = tmp_path / "copy.h"
    _fail_path_write_with_unicode(monkeypatch, declarations_path)

    result = main(
        [
            "native-loader-abi",
            str(manifest_path),
            "--declarations-output",
            str(declarations_path),
        ]
    )

    assert result == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "project.native-loader-abi.declarations-write-failed"
    assert payload["details"] == {"destination": str(declarations_path)}
    assert captured.err == ""


def test_native_loader_abi_package_cli_writes_deterministic_package(tmp_path, capsys):
    manifest_path = _write_manifest(
        tmp_path,
        _load_unit("opengl"),
        _load_unit("directx"),
    )
    first_root = tmp_path / "first-package"
    second_root = tmp_path / "second-package"

    first_result = main(
        ["native-loader-abi-package", str(manifest_path), str(first_root)]
    )
    first = json.loads(capsys.readouterr().out)
    second_result = main(
        ["native-loader-abi-package", str(manifest_path), str(second_root)]
    )
    second = json.loads(capsys.readouterr().out)

    assert first_result == second_result == 0
    assert first == second
    assert first["kind"] == project_api.NATIVE_LOADER_ABI_PACKAGE_KIND
    assert first["success"] is True
    assert first["summary"] == {
        "generatedFileCount": 5,
        "targetCount": 2,
        "unitCount": 2,
    }
    assert [unit["descriptorPath"] for unit in first["units"]] == [
        (
            "targets/directx-7fde9c43d3d7/"
            "copy-directx-4dcb25d07969.native-loader-abi.json"
        ),
        (
            "targets/opengl-d5211293c7c7/"
            "copy-opengl-bbdf2b883b33.native-loader-abi.json"
        ),
    ]
    expected_files = {
        project_api.NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
        *(entry["path"] for entry in first["generatedFiles"]),
    }
    assert {
        path.relative_to(first_root).as_posix()
        for path in first_root.rglob("*")
        if path.is_file()
    } == expected_files
    assert {
        path.relative_to(second_root).as_posix()
        for path in second_root.rglob("*")
        if path.is_file()
    } == expected_files


def test_native_loader_abi_package_cli_reports_manifest_failure(tmp_path, capsys):
    manifest_path = tmp_path / "missing-loader-manifest.json"
    output_dir = tmp_path / "abi-package"

    result = main(["native-loader-abi-package", str(manifest_path), str(output_dir)])

    assert result == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "project.native-loader-abi.manifest-read-failed"
    assert payload["path"] == "$.loaderManifestPath"
    assert payload["details"] == {"manifestPath": str(manifest_path)}
    assert captured.err == ""
    assert not output_dir.exists()


def test_native_loader_abi_package_cli_reports_output_failure(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, _load_unit())
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    output_dir = blocked_parent / "abi-package"

    result = main(["native-loader-abi-package", str(manifest_path), str(output_dir)])

    assert result == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "project.native-loader-abi.package-write-failed"
    assert payload["path"] == "$.outputDirectory"
    assert payload["details"]["outputDirectory"] == str(output_dir)
    assert payload["details"]["relativePath"].startswith("targets/directx-")
    assert captured.err == ""


def _fail_path_write_with_unicode(monkeypatch, failed_path):
    original_write_text = Path.write_text

    def write_text(path, *args, **kwargs):
        if path == failed_path:
            raise UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogate")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
