import json
from dataclasses import dataclass, field

import pytest

import crosstl._crosstl as cli
import crosstl.project as project


@dataclass(frozen=True)
class _ProjectConfig:
    source_roots: tuple[str, ...] = (".",)
    include_dirs: tuple[str, ...] = ()
    defines: dict[str, str] = field(default_factory=dict)
    source_overrides: dict[str, str] = field(default_factory=dict)
    dispatch_contracts: tuple[str, ...] = ()


def _project_args(command, *options):
    return cli._build_parser().parse_args([command, "repository", *options])


def _write_dispatch_contract(path):
    payload = {
        "kind": project.DISPATCH_CONTRACT_KIND,
        "schemaVersion": project.DISPATCH_CONTRACT_SCHEMA_VERSION,
        "inputs": [{"name": "elementCount", "role": "shape", "type": "integer"}],
        "workloads": [{"id": "count-64", "values": {"elementCount": 64}}],
        "capabilities": [],
        "devices": [{"id": "portable", "values": {}}],
        "contracts": [
            {
                "id": "copy-dispatch",
                "source": "kernels/copy.metal",
                "entryPoint": "copy_float32",
                "branches": [
                    {
                        "id": "default",
                        "when": True,
                        "workgroupSize": [64, 1, 1],
                        "dispatch": {"workgroupCount": [1, 1, 1]},
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("command", ("scan", "report", "translate-project"))
def test_project_commands_accept_repeatable_dispatch_contracts(command):
    args = _project_args(
        command,
        "--dispatch-contract",
        "contracts/base.json",
        "--dispatch-contract",
        "contracts/specialized.json",
    )

    assert args.dispatch_contract == [
        "contracts/base.json",
        "contracts/specialized.json",
    ]


def test_project_config_is_preserved_without_dispatch_contract_overrides(monkeypatch):
    configured = _ProjectConfig(dispatch_contracts=("contracts/configured.json",))
    monkeypatch.setattr(project, "load_project_config", lambda root, path: configured)

    loaded = cli._load_project_config_from_args(_project_args("scan"))

    assert loaded is configured
    assert loaded.dispatch_contracts == ("contracts/configured.json",)


def test_dispatch_contract_overrides_append_normalize_and_deduplicate(monkeypatch):
    configured = _ProjectConfig(
        dispatch_contracts=(
            "contracts/configured.json",
            "contracts/shared.json",
        )
    )
    monkeypatch.setattr(project, "load_project_config", lambda root, path: configured)
    args = _project_args(
        "translate-project",
        "--dispatch-contract",
        r"contracts\shared.json",
        "--dispatch-contract",
        r"contracts\windows\specialized.json",
        "--dispatch-contract",
        "contracts/configured.json",
        "--dispatch-contract",
        "contracts/windows/specialized.json",
    )

    loaded = cli._load_project_config_from_args(args)

    assert loaded.dispatch_contracts == (
        "contracts/configured.json",
        "contracts/shared.json",
        "contracts/windows/specialized.json",
    )
    assert configured.dispatch_contracts == (
        "contracts/configured.json",
        "contracts/shared.json",
    )


def test_unrelated_project_overrides_preserve_dispatch_contracts(monkeypatch):
    configured = _ProjectConfig(dispatch_contracts=("contracts/configured.json",))
    monkeypatch.setattr(project, "load_project_config", lambda root, path: configured)
    args = _project_args("report", "--include-dir", "includes")

    loaded = cli._load_project_config_from_args(args)

    assert loaded is not configured
    assert loaded.include_dirs == ("includes",)
    assert loaded.dispatch_contracts == configured.dispatch_contracts


@pytest.mark.parametrize("command", ("scan", "report", "translate-project"))
def test_project_commands_reject_empty_dispatch_contracts(command, capsys):
    with pytest.raises(SystemExit, match="2"):
        _project_args(command, "--dispatch-contract", " ")

    assert (
        "argument --dispatch-contract: " "--dispatch-contract entries must be non-empty"
    ) in capsys.readouterr().err


def test_dispatch_contract_parser_rejects_empty_values():
    with pytest.raises(
        ValueError, match="--dispatch-contract entries must be non-empty"
    ):
        cli._parse_project_dispatch_contracts([" "])


def test_single_file_translate_does_not_accept_dispatch_contracts(capsys):
    with pytest.raises(SystemExit, match="2"):
        cli._build_parser().parse_args(
            [
                "translate",
                "shader.metal",
                "--dispatch-contract",
                "contracts/base.json",
            ]
        )

    assert "unrecognized arguments: --dispatch-contract" in capsys.readouterr().err


def test_scan_command_imports_dispatch_contract_into_report(tmp_path, capsys):
    repository = tmp_path / "repository"
    repository.mkdir()
    contract_path = tmp_path / "copy.dispatch.json"
    _write_dispatch_contract(contract_path)

    exit_code = cli.main(
        [
            "scan",
            str(repository),
            "--target",
            "cgl",
            "--dispatch-contract",
            str(contract_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["project"]["dispatchContractFiles"] == [str(contract_path)]
    assert payload["project"]["dispatchContractCount"] == 1
    assert payload["project"]["dispatchVariantCount"] == 1
    assert (
        payload["project"]["dispatchContracts"][0]["evaluation"]["variants"][0][
            "entryPoint"
        ]
        == "copy_float32"
    )
