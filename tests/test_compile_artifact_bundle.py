import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import compile_artifact_bundle as bundles


def _fixture(tmp_path):
    sources = {}
    entries = []
    for name in ("first", "second"):
        source = tmp_path / f"{name}.metal"
        source.write_text(f"kernel void {name}() {{}}\n", encoding="utf-8")
        sources[name] = source
        entries.append(
            {
                "entryPoint": name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "sizeBytes": source.stat().st_size,
            }
        )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "target": "metal",
                "artifactContract": {"artifactCount": len(entries)},
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "bundle"
    for index, (name, source) in enumerate(sources.items()):
        bundles.write_bundle_entry(source, contract, name, root / f"shard-{index}")
    return contract, root, sources


def _command(
    body="Path(sys.argv[2]).write_bytes(b'compiled:' + Path(sys.argv[1]).read_bytes())",
):
    return [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; " + body,
        "{artifact}",
        "{output}",
    ]


def test_bundle_compiles_complete_shards_and_retains_evidence(tmp_path):
    contract, root, _ = _fixture(tmp_path)
    result = bundles.compile_bundle(root, contract, tmp_path / "compiled", _command())
    assert result["status"] == "passed"
    assert result["expectedCount"] == len(result["records"]) == 2
    assert result["failures"] == []
    assert result["numericalExecution"] is False
    assert result["fullUpstreamSuite"] is False
    assert result == json.loads((tmp_path / "compiled/report.json").read_text())
    for record in result["records"]:
        output = Path(record["compiledPath"])
        assert output.read_bytes() == b"compiled:" + Path(record["path"]).read_bytes()
        assert (
            hashlib.sha256(output.read_bytes()).hexdigest() == record["compiledSha256"]
        )
        assert json.loads((output.parent / "evidence.json").read_text()) == record


@pytest.mark.parametrize(
    "damage",
    (
        "missing",
        "duplicate",
        "unexpected",
        "contract",
        "hash",
        "size",
        "bytes",
        "truncated",
        "escape",
        "symlink",
        "symlink_directory",
        "extra",
        "unknown_field",
        "boolean_schema",
        "duplicate_json",
    ),
)
def test_bundle_rejects_invalid_handoff_before_compilation(
    tmp_path, monkeypatch, damage
):
    contract, root, _ = _fixture(tmp_path)
    receipt = sorted(root.rglob("entry.json"))[0]
    record = json.loads(receipt.read_text())
    artifact = receipt.parent / record["artifact"]
    if damage == "missing":
        shutil.rmtree(receipt.parent)
    elif damage == "duplicate":
        shutil.copytree(receipt.parent, root / "duplicate")
    elif damage == "unexpected":
        record["entryPoint"] = "unexpected"
    elif damage == "contract":
        record["contractSha256"] = "f" * 64
    elif damage == "hash":
        record["sha256"] = "f" * 64
    elif damage == "size":
        record["sizeBytes"] += 1
    elif damage == "bytes":
        artifact.write_bytes(b"!" * artifact.stat().st_size)
    elif damage == "truncated":
        artifact.write_bytes(b"")
    elif damage == "escape":
        record["artifact"] = "../../outside.metal"
    elif damage in {"symlink", "symlink_directory"}:
        link = root / "link" if damage == "symlink_directory" else artifact
        if link == artifact:
            link.unlink()
        try:
            link.symlink_to(
                receipt.parent if damage == "symlink_directory" else contract,
                target_is_directory=damage == "symlink_directory",
            )
        except OSError:
            pytest.skip("Symlinks unavailable")
    elif damage == "extra":
        (root / "unexpected.txt").write_text("unexpected")
    elif damage == "unknown_field":
        record["compilerCommand"] = ["ignored"]
    elif damage == "boolean_schema":
        record["schemaVersion"] = True
    if damage not in {"missing", "duplicate"}:
        receipt.write_text(json.dumps(record))
    if damage == "duplicate_json":
        receipt.write_text(
            receipt.read_text().replace(
                '"schemaVersion": 1', '"schemaVersion": 1, "schemaVersion": 1'
            )
        )
    monkeypatch.setattr(
        bundles,
        "_compile",
        lambda *args: pytest.fail("Invalid bundle reached compiler"),
    )
    result = bundles.compile_bundle(root, contract, tmp_path / "compiled", _command())
    assert result["status"] == "failed"
    assert result["records"] == []
    assert result["error"]
    assert (
        json.loads((tmp_path / "compiled/report.json").read_text())["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "mode", ("failure", "empty", "missing", "mutation", "timeout", "unavailable")
)
def test_bundle_retains_native_compiler_failures(tmp_path, mode):
    contract, root, _ = _fixture(tmp_path)
    body = {
        "failure": "sys.stderr.write('compiler diagnostic'); sys.exit(7)",
        "empty": "Path(sys.argv[2]).write_bytes(b'')",
        "missing": "pass",
        "mutation": (
            "Path(sys.argv[1]).write_bytes(b'changed'); Path(sys.argv[2]).write_bytes(b'output')"
        ),
        "timeout": "import time; time.sleep(5)",
        "unavailable": "pass",
    }[mode]
    command = _command(body)
    if mode == "unavailable":
        command[0] = str(tmp_path / "missing-compiler")
    output = tmp_path / "compiled"
    result = bundles.compile_bundle(
        root, contract, output, command, timeout=0.05 if mode == "timeout" else 120
    )
    assert result["status"] == "failed"
    assert result["failures"] == ["first", "second"]
    assert all(record["status"] == "failed" for record in result["records"])
    assert len(list(output.rglob("evidence.json"))) == 2
    if mode == "failure":
        assert all(
            record["returncode"] == 7 and record["stderr"] == "compiler diagnostic"
            for record in result["records"]
        )


def test_bundle_cannot_accept_stale_compiler_output(tmp_path):
    contract, root, _ = _fixture(tmp_path)
    output = tmp_path / "compiled"
    assert (
        bundles.compile_bundle(root, contract, output, _command())["status"] == "passed"
    )
    result = bundles.compile_bundle(root, contract, output, _command("pass"))
    assert result["status"] == "failed"
    assert result["failures"] == ["first", "second"]


def test_bundle_writer_rejects_mismatch_and_duplicate_entries(tmp_path):
    contract, root, sources = _fixture(tmp_path)
    with pytest.raises(FileExistsError):
        bundles.write_bundle_entry(
            sources["first"], contract, "first", root / "shard-0"
        )
    with pytest.raises(ValueError, match="Unexpected artifact"):
        bundles.write_bundle_entry(sources["first"], contract, "unknown", root)
    sources["first"].write_text("changed")
    with pytest.raises(ValueError, match="Artifact size differs"):
        bundles.write_bundle_entry(sources["first"], contract, "first", root)


@pytest.mark.parametrize(
    "damage", ("empty", "duplicate", "hash", "size", "count", "requirements")
)
def test_bundle_rejects_invalid_contract(tmp_path, damage):
    contract, root, _ = _fixture(tmp_path)
    data = json.loads(contract.read_text())
    if damage == "empty":
        data["entries"] = []
    elif damage == "duplicate":
        data["entries"].append(data["entries"][0])
    elif damage == "hash":
        data["entries"][0]["sha256"] = "invalid"
    elif damage == "size":
        data["entries"][0]["sizeBytes"] = True
    elif damage == "count":
        data["artifactContract"]["artifactCount"] += 1
    else:
        data["artifactContract"] = []
    contract.write_text(json.dumps(data))
    result = bundles.compile_bundle(root, contract, tmp_path / "compiled", _command())
    assert result["status"] == "failed"
    assert not result["records"]


def test_bundle_cli_propagates_compiler_exit_failure(tmp_path):
    contract, root, _ = _fixture(tmp_path)
    process = subprocess.run(
        [
            sys.executable,
            str(Path(bundles.__file__)),
            "--bundle-root",
            str(root),
            "--contract",
            str(contract),
            "--output-dir",
            str(tmp_path / "compiled"),
            "--compiler-command",
            json.dumps(_command("sys.exit(9)")),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    assert json.loads(process.stdout)["status"] == "failed"


@pytest.mark.parametrize("case", ("jobs", "timeout", "command", "output"))
def test_bundle_rejects_unsafe_invocation(tmp_path, case):
    contract, root, _ = _fixture(tmp_path)
    args = (
        {"jobs": 0}
        if case == "jobs"
        else {"timeout": float("inf")} if case == "timeout" else {}
    )
    with pytest.raises(ValueError):
        bundles.compile_bundle(
            root,
            contract,
            root / "out" if case == "output" else tmp_path / "compiled",
            [sys.executable] if case == "command" else _command(),
            **args,
        )
