import copy
import json

import crosstl.project as project_api
from crosstl._crosstl import main
from tests.test_translator.test_native_loader_abi import _load_unit, _loader_manifest


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
