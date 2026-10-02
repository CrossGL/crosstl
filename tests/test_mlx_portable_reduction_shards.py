"""Complete row-package collections preserve artifact ownership across CI shards."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from demos.integrations.mlx.portable_host import reduction_shards as shards
from demos.integrations.mlx.portable_host import row_workloads, verify_rows
from demos.integrations.mlx.portable_host.reduction_packages import (
    ROW_ENTRIES,
    ROW_WIDTHS,
)


def test_partition_keeps_all_entries_and_widths_without_overlap():
    parts = [shards.widths_for_shard(number) for number in range(shards.SHARD_COUNT)]
    assert sorted(width for part in parts for width in part) == list(ROW_WIDTHS)
    assert max(map(len, parts)) == 4
    assert sum(len(part) * len(ROW_ENTRIES) for part in parts) == 1456
    cases = list(row_workloads.cases(ROW_WIDTHS))
    assert len(cases) == 1870
    assert len({(case["entry"], case["width"]) for case in cases}) == 896


@pytest.mark.parametrize("value", [-1, 8, True, None, "0", 0.0])
def test_invalid_shard_identifiers_fail(value):
    with pytest.raises(ValueError, match="Row shard"):
        shards.widths_for_shard(value)


@pytest.fixture(params=["metal", "opengl", "directx"])
def collection(tmp_path, monkeypatch, request):
    target = request.param
    monkeypatch.setattr(shards, "ROW_WIDTHS", ROW_WIDTHS[:8])
    monkeypatch.setattr(shards, "ROW_ENTRIES", dict(list(ROW_ENTRIES.items())[:2]))
    monkeypatch.setattr(shards, "translator_revision", lambda: "a" * 40)
    root = tmp_path / "collection"
    for number in range(shards.SHARD_COUNT):
        directory = root / "shards" / f"artifact-{number}" / "packages"
        package = directory / "package"
        package.mkdir(parents=True)
        (package / "shader").write_text("synthetic package bytes")
        artifact = {
            "packagePath": "shader",
            "sizeBytes": (package / "shader").stat().st_size,
            "hash": {"algorithm": "sha256", "value": shards.digest(package / "shader")},
        }
        widths = shards.widths_for_shard(number)
        index = {
            "target": target,
            "family": "row",
            "widths": list(widths),
            "entries": list(shards.ROW_ENTRIES),
            "descriptors": {
                f"w{width}/{entry}": {
                    "target": target,
                    "artifact": artifact,
                    "entryPoint": {"name": entry},
                    "source": {
                        "path": shards.SOURCE,
                        "backend": "metal",
                        "hash": {"algorithm": "sha256", "value": "b" * 64},
                    },
                }
                for width in widths
                for entry in shards.ROW_ENTRIES
            },
        }
        shards.write_json(directory / "index.json", index)
        shards.write_json(
            directory / "shard.json",
            {
                "schemaVersion": shards.SCHEMA_VERSION,
                "kind": "mlx-row-package-shard",
                "shard": number,
                "commit": shards.COMMIT,
                "translatorRevision": "a" * 40,
                "target": target,
                "widths": list(widths),
                "indexSha256": shards.digest(directory / "index.json"),
                "sourceSha256": "b" * 64,
            },
        )
    return root, target


def update_index(directory, modify):
    index = json.loads((directory / "index.json").read_text())
    modify(index)
    shards.write_json(directory / "index.json", index)
    record = json.loads((directory / "shard.json").read_text())
    record["indexSha256"] = shards.digest(directory / "index.json")
    shards.write_json(directory / "shard.json", record)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "extra",
        "duplicate",
        "record-type",
        "record-version",
        "record-pin",
        "record-revision",
        "record-target",
        "record-widths",
        "record-index",
        "record-source",
        "mixed-source",
        "wrong-family",
        "missing-entry",
        "missing-width",
        "descriptor-source",
        "descriptor-target",
        "artifact-missing",
        "artifact-hash",
        "artifact-size",
        "escape",
        "absolute-path",
        "backslash",
    ],
)
def test_collection_rejects_incomplete_or_changed_shards(collection, fault):
    root, target = collection
    directory = root / "shards/artifact-0/packages"
    record_path = directory / "shard.json"
    record = json.loads(record_path.read_text())
    if fault == "missing":
        shutil.rmtree(directory.parent)
    elif fault == "extra":
        (root / "shards/extra").mkdir()
    elif fault == "duplicate":
        record["shard"] = 1
        shards.write_json(record_path, record)
    elif fault and fault.startswith("record-"):
        key = {
            "record-version": "schemaVersion",
            "record-pin": "commit",
            "record-revision": "translatorRevision",
            "record-target": "target",
            "record-widths": "widths",
            "record-index": "indexSha256",
            "record-source": "sourceSha256",
        }.get(fault)
        if key:
            record[key] = True if key == "schemaVersion" else "wrong"
        else:
            record = []
        shards.write_json(record_path, record)
    elif fault == "mixed-source":
        record["sourceSha256"] = "c" * 64
        shards.write_json(record_path, record)
        update_index(
            directory,
            lambda index: [
                item["source"]["hash"].update(value="c" * 64)
                for item in index["descriptors"].values()
            ],
        )
    elif fault in {
        "wrong-family",
        "missing-entry",
        "missing-width",
        "descriptor-source",
        "descriptor-target",
    }:

        def mutate(index):
            if fault == "wrong-family":
                index["family"] = "all"
            elif fault == "missing-entry":
                index["entries"].pop()
            elif fault == "missing-width":
                index["widths"] = []
            else:
                item = next(iter(index["descriptors"].values()))
                if fault == "descriptor-source":
                    item["source"]["path"] = "other.metal"
                else:
                    item["target"] = "wrong"

        update_index(directory, mutate)
    elif fault == "artifact-missing":
        (directory / "package/shader").unlink()
    elif fault == "artifact-hash":
        (directory / "package/shader").write_text("changed package bytes")
    elif fault in {"artifact-size", "escape", "absolute-path", "backslash"}:

        def mutate(index):
            for item in index["descriptors"].values():
                if fault == "artifact-size":
                    item["artifact"]["sizeBytes"] += 1
                else:
                    item["artifact"]["packagePath"] = {
                        "escape": "../../outside",
                        "absolute-path": str(root / "outside"),
                        "backslash": "..\\outside",
                    }[fault]

        update_index(directory, mutate)
    if fault:
        with pytest.raises((ValueError, FileNotFoundError)):
            shards.collect_shards(root, target)
        assert not (root / "collection.json").exists()
    else:
        evidence = shards.collect_shards(root, target)
        index, directories = shards.load_row_packages(
            root, target, require_all_widths=True
        )
        assert evidence["artifactCount"] == len(index["descriptors"]) == 16
        assert index["widths"] == list(shards.ROW_WIDTHS) and len(directories) == 8
        assert all(directory.is_relative_to(root) for directory in directories)
        with pytest.raises(ValueError, match="already exists"):
            shards.collect_shards(root, target)


@pytest.mark.parametrize(
    "fault", ["manifest", "index", "artifact", "ambiguous", "revision"]
)
def test_loading_rechecks_collection_identity(collection, monkeypatch, fault):
    root, target = collection
    shards.collect_shards(root, target)
    if fault == "manifest":
        record = json.loads((root / "collection.json").read_text())
        record["shards"].reverse()
        shards.write_json(root / "collection.json", record)
    elif fault == "index":
        path = root / "shards/artifact-0/packages/index.json"
        path.write_text(path.read_text() + "\n")
    elif fault == "artifact":
        (root / "shards/artifact-0/packages/package/shader").write_bytes(b"different")
    elif fault == "ambiguous":
        (root / "index.json").write_text("{}")
    else:
        monkeypatch.setattr(shards, "translator_revision", lambda: "d" * 40)
    with pytest.raises(ValueError):
        shards.load_row_packages(root, target, require_all_widths=True)


@pytest.mark.parametrize(
    "path",
    [
        "shards",
        "shards/artifact-0",
        "shards/artifact-0/packages",
        "shards/artifact-0/packages/index.json",
        "shards/artifact-0/packages/shard.json",
        "shards/artifact-0/packages/package",
        "shards/artifact-0/packages/package/shader",
        "collection.json",
    ],
)
def test_collections_reject_external_symlinks(collection, path):
    root, target = collection
    shards.collect_shards(root, target)
    original = root / path
    destination = root.parent / "external"
    original.rename(destination)
    try:
        original.symlink_to(destination, target_is_directory=destination.is_dir())
    except OSError as error:
        pytest.skip(f"Creating symlinks is unavailable: {error}")
    with pytest.raises(ValueError):
        shards.load_row_packages(root, target, require_all_widths=True)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "compiler",
        "timeout",
        "unavailable",
        "empty-module",
        "source-changed",
        "validation",
    ],
)
def test_native_compilation_requires_every_variant(
    collection, monkeypatch, tmp_path, fault
):
    root, target = collection
    shards.collect_shards(root, target)
    monkeypatch.setattr(
        shards.sys,
        "platform",
        {"metal": "darwin", "directx": "win32", "opengl": "linux"}[target],
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] is False and kwargs["timeout"] in {30, 120}
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        if fault == "unavailable":
            raise FileNotFoundError("compiler")
        if command[0] == "spirv-val":
            return SimpleNamespace(
                returncode=int(fault == "validation"), stdout="", stderr="diagnostic"
            )
        flag = "-Fo" if target == "directx" else "-o"
        Path(command[command.index(flag) + 1]).write_bytes(
            b"" if fault == "empty-module" else b"compiled"
        )
        if fault == "source-changed":
            source = (
                command[-1]
                if target == "directx"
                else (
                    command[command.index("-c") + 1]
                    if target == "metal"
                    else command[command.index("-S") + 2]
                )
            )
            Path(source).write_text("changed")
        return SimpleNamespace(
            returncode=int(fault == "compiler"), stdout="", stderr="diagnostic"
        )

    monkeypatch.setattr(shards.subprocess, "run", run)
    output = tmp_path / "compiled"
    if fault and not (fault == "validation" and target != "opengl"):
        with pytest.raises(
            (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError)
        ):
            shards.compile_packages(root, output, target)
        assert not (output / "evidence.json").exists()
        assert (
            json.loads((output / "results.json").read_text())[-1]["status"] == "failed"
        )
    else:
        evidence = shards.compile_packages(root, output, target)
        assert evidence == {
            "target": target,
            "compiledArtifacts": 16,
            "status": "passed",
        }
        assert len(commands) == (32 if target == "opengl" else 16)
        for record in json.loads((output / "results.json").read_text()):
            assert record["moduleSha256"] == shards.digest(output / record["module"])


def test_native_compilation_rejects_cross_platform_execution(
    collection, monkeypatch, tmp_path
):
    root, target = collection
    monkeypatch.setattr(shards.sys, "platform", "unsupported")
    with pytest.raises(ValueError, match="native CI platform"):
        shards.compile_packages(root, tmp_path / "compiled", target)


def test_native_compilation_rechecks_package_metadata(
    collection, monkeypatch, tmp_path
):
    root, target = collection
    shards.collect_shards(root, target)
    monkeypatch.setattr(
        shards.sys,
        "platform",
        {"metal": "darwin", "directx": "win32", "opengl": "linux"}[target],
    )
    path = root / "collection.json"

    def run(command, **kwargs):
        if command[0] != "spirv-val":
            flag = "-Fo" if target == "directx" else "-o"
            Path(command[command.index(flag) + 1]).write_bytes(b"compiled")
        record = json.loads(path.read_text())
        record["translatorRevision"] = "changed"
        shards.write_json(path, record)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shards.subprocess, "run", run)
    output = tmp_path / "compiled"
    with pytest.raises(ValueError, match="collection changed"):
        shards.compile_packages(root, output, target)
    assert not (output / "evidence.json").exists()
    assert len(json.loads((output / "results.json").read_text())) == 16


@pytest.mark.parametrize("failure", [None, "translation", "revision"])
def test_shard_marker_requires_successful_translation(tmp_path, monkeypatch, failure):
    revisions = iter(["a" * 40, "b" * 40 if failure == "revision" else "a" * 40])
    monkeypatch.setattr(shards, "translator_revision", lambda: next(revisions))
    (tmp_path / shards.SOURCE).parent.mkdir(parents=True)
    (tmp_path / shards.SOURCE).write_text("source")
    output = tmp_path / "output"
    calls = []

    def build(root, path, target, **kwargs):
        calls.append((root, target, kwargs))
        path.mkdir()
        (path / "index.json").write_text("{}")
        if failure == "translation":
            raise RuntimeError("translation failed")

    monkeypatch.setattr(shards, "build_packages", build)
    if failure:
        with pytest.raises(
            (RuntimeError, ValueError), match="translation failed|revision changed"
        ):
            shards.build_shard(tmp_path, output, "metal", 2)
        assert not (output / "shard.json").exists()
    else:
        record = shards.build_shard(tmp_path, output, "metal", 2)
        assert record["indexSha256"] == shards.digest(output / "index.json")
        assert record["sourceSha256"] == shards.digest(tmp_path / shards.SOURCE)
    assert calls == [
        (
            tmp_path,
            "metal",
            {"family": "row", "widths": shards.widths_for_shard(2), "jobs": 2},
        )
    ]


def test_ci_requires_shards_and_complete_native_gates():
    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    build = workflow["jobs"]["row-packages"]
    assert build["runs-on"] == "ubuntu-24.04" and "needs" not in build
    assert build["strategy"]["matrix"] == {
        "target": ["metal", "opengl", "directx"],
        "shard": list(range(shards.SHARD_COUNT)),
    }
    assert build["strategy"]["fail-fast"] is False
    steps = {step.get("name"): step for step in build["steps"]}
    translate = steps["Translate complete row shard"]
    assert "--shard" in translate["run"] and "--jobs 2" in translate["run"]
    assert "--width" not in translate["run"] and "--entry" not in translate["run"]
    assert "continue-on-error" not in translate and "if" not in translate
    assert steps["Retain row translation evidence"]["if"] == "always()"
    assert (
        steps["Retain row translation evidence"]["with"]["include-hidden-files"] is True
    )
    checkout = steps["Check out pinned MLX kernels"]["run"]
    assert checkout.index("config core.autocrlf false") < checkout.index(
        "checkout --detach"
    )
    job = workflow["jobs"]["reductions"]
    assert job["needs"] == ["portable-host", "row-packages"]
    native = {step.get("name"): step for step in job["steps"]}
    for name in (
        "Select same-run row shards",
        "Download row shards",
        "Verify complete row package collection",
        "Compile every row variant natively",
    ):
        assert (
            native[name]["if"] == "matrix.family == 'row'"
            and "continue-on-error" not in native[name]
        )
    assert (
        "run_id: context.runId"
        in native["Select same-run row shards"]["with"]["script"]
    )
    assert "shard < 8" in native["Select same-run row shards"]["with"]["script"]
    assert native["Download row shards"]["with"]["merge-multiple"] is False
    for variant in job["strategy"]["matrix"]["include"]:
        if variant["family"] == "row":
            assert "--require-all-widths" in variant["companion_args"]
            assert variant["verification_timeout"] > 2 * 3600
    for event in ("push", "pull_request"):
        assert (
            "tests/test_mlx_portable_reduction_shards.py"
            in workflow.get("on", workflow.get(True))[event]["paths"]
        )


def test_single_package_remains_supported_but_full_ci_rejects_partial_widths(
    collection,
):
    root, target = collection
    directory = root / "shards/artifact-0/packages"
    first = json.loads((directory / "index.json").read_text())
    second = json.loads((root / "shards/artifact-1/packages/index.json").read_text())
    first["widths"] = [32, 128]
    first["descriptors"].update(second["descriptors"])
    shards.write_json(directory / "index.json", first)
    index, directories = shards.load_row_packages(directory, target)
    assert directories == [directory] and index["widths"] == [32, 128]
    with pytest.raises(ValueError, match="every supported width"):
        shards.load_row_packages(directory, target, require_all_widths=True)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "worker",
        "unknown-variant",
        "changed-index",
        "changed-directory",
        "changed-source",
    ],
)
def test_row_parent_retains_package_ownership_and_complete_width_requirement(
    tmp_path, monkeypatch, fault
):
    base = tmp_path / "base"
    base.mkdir()
    shards.write_json(base / "index.json", {"target": "metal"})
    directories = [tmp_path / "first", tmp_path / "second"]
    entries = list(ROW_ENTRIES)[:2]
    records = [{"actual": [1]}, {"actual": [2]}]
    trace = [
        {"entry": entry, "target": "metal", "workgroupSize": [width, 1, 1]}
        for width, entry in zip((32, 128), entries)
    ]
    indexes = {
        directory: {
            "target": "metal",
            "descriptors": {f'w{event["workgroupSize"][0]}/{event["entry"]}': {}},
        }
        for directory, event in zip(directories, trace)
    }
    merged = {
        "widths": list(ROW_WIDTHS),
        "entries": list(ROW_ENTRIES),
        "descriptors": {
            key: value
            for index in indexes.values()
            for key, value in index["descriptors"].items()
        },
    }
    loaded, validated, verified, commands = [], [], [], []

    def load(directory, target, *, require_all_widths):
        loaded.append((directory, target, require_all_widths))
        index = copy.deepcopy(merged)
        paths = list(directories)
        if len(loaded) > 1:
            if fault == "changed-index":
                index["widths"] = [32, 128]
            elif fault == "changed-directory":
                paths.reverse()
        return index, paths

    source_calls = []

    def prepared(root):
        source_calls.append(root)
        return {
            "hash": (
                "changed"
                if fault == "changed-source" and len(source_calls) > 1
                else "original"
            )
        }

    monkeypatch.setattr(verify_rows, "load_row_packages", load)
    monkeypatch.setattr(
        verify_rows, "load_index", lambda directory, target: indexes[directory]
    )
    monkeypatch.setattr(verify_rows, "verify_prepared", prepared)
    monkeypatch.setattr(
        verify_rows.row_workloads,
        "validate",
        lambda *args, **kwargs: validated.append((args, kwargs)),
    )
    monkeypatch.setattr(
        verify_rows,
        "verify_artifacts",
        lambda events, directory, index: verified.append((events, directory, index)),
    )

    def run(command, **kwargs):
        commands.append(command)
        if fault == "worker":
            return SimpleNamespace(returncode=1)
        directory = Path(command[command.index("--output-dir") + 1])
        directory.mkdir()
        shards.write_json(directory / "result.json", records)
        events = copy.deepcopy(trace)
        if fault == "unknown-variant":
            events[0]["entry"] = "unknown"
        (directory / "dispatch.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(verify_rows.subprocess, "run", run)
    args = SimpleNamespace(
        packages=base,
        reductions=tmp_path / "collection",
        all_reductions=None,
        mlx_root=tmp_path / "mlx",
        output_dir=tmp_path / "proof",
        require_all_widths=True,
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            verify_rows.verify(args)
        assert not (args.output_dir / "evidence.json").exists()
    else:
        result = verify_rows.verify(args)
        assert result["packageCount"] == 2 and result["artifactCount"] == 2
        assert len(validated) == 2 and len(loaded) == 2
        assert verified == [
            ([trace[0]], directories[0], indexes[directories[0]]),
            ([trace[1]], directories[1], indexes[directories[1]]),
        ]
        assert all("--require-all-widths" in command for command in commands)
        assert all(require_all for _, _, require_all in loaded)


@pytest.mark.parametrize("available", [False, True])
def test_row_worker_passes_every_owned_package_to_runtime(
    tmp_path, monkeypatch, available
):
    base = tmp_path / "base"
    base.mkdir()
    shards.write_json(base / "index.json", {"target": "opengl"})
    directories = [tmp_path / "one", tmp_path / "two"]
    monkeypatch.setattr(
        verify_rows,
        "load_row_packages",
        lambda *args, **kwargs: ({"widths": [32, 128]}, directories),
    )
    registered, devices = [], []

    def runtime(*args, **kwargs):
        registered.append(kwargs["reductions"])
        return SimpleNamespace(install=lambda: None, dispatch_count=0)

    core = SimpleNamespace(
        gpu="gpu",
        cpu="cpu",
        is_available=lambda device: available,
        set_default_device=devices.append,
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(verify_rows, "HostRuntime", runtime)
    for module in (verify_rows.row_workloads, verify_rows.mixed_reduction_workloads):
        monkeypatch.setattr(module, "collect", lambda *args, **kwargs: [])
        monkeypatch.setattr(module, "validate", lambda *args, **kwargs: None)
    args = SimpleNamespace(
        packages=base,
        reductions=tmp_path / "collection",
        all_reductions=tmp_path / "all",
        output_dir=tmp_path / "native",
        worker="native",
        require_all_widths=True,
    )
    if available:
        with pytest.raises(RuntimeError, match="already available"):
            verify_rows.worker(args)
        assert registered == devices == []
        return
    verify_rows.worker(args)
    assert registered == [[*directories, args.all_reductions]] and devices == ["gpu"]
