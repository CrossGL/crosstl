"""Contract checks for unchanged upstream block-format execution evidence."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from demos.integrations.mlx.portable_host import verify_upstream_blocks as proof


def test_ci_requires_unchanged_block_tests_on_all_three_backends():
    root = Path(__file__).resolve().parents[5]
    job = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )["jobs"]["half-host"]
    assert {
        (item["os"], item["target"]) for item in job["strategy"]["matrix"]["include"]
    } == {
        ("ubuntu-24.04", "opengl"),
        ("windows-2025", "directx"),
        ("macos-26", "metal"),
    }
    steps = {step.get("name"): step for step in job["steps"]}
    assert "test_upstream_blocks.py" in steps["Validate half host contracts"]["run"]
    translation = steps["Translate upstream block assertion packages"]
    execution = steps["Execute unchanged upstream block tests"]
    for step in (translation, execution):
        assert "if" not in step and not step.get("continue-on-error")
        assert "run_bounded_command.py" in step["run"]
    assert "--timeout-seconds 900" in translation["run"]
    assert "--width 32 --width 128 --width 256" in translation["run"]
    assert (
        "--entry all_reduce_andbool_ --entry all_reduce_maxfloat32" in translation[
            "run"
        ]
    )
    assert "--timeout-seconds 2100" in execution["run"]
    assert "portable_host.verify_upstream_blocks" in execution["run"]
    for family in ("bfloat-reductions", "block-assertion-reductions"):
        assert f"--reductions .mlx-portable-half/{family}" in execution["run"]
    for option, path in (
        ("packages", "base/packages"),
        ("random", "random-packages"),
        ("bfloat", "bfloat-packages"),
    ):
        assert f"--{option} .mlx-portable-half/{path}" in execution["run"]
    assert execution["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    upload = steps["Retain half execution evidence"]
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == ".mlx-portable-half"
    assert upload["with"]["if-no-files-found"] == "error"


def records(native):
    return [
        {
            "test": name,
            "testsRun": 1,
            "success": True,
            "errors": [],
            "failures": [],
            "skipped": [],
            "expectedFailures": [],
            "unexpectedSuccesses": [],
            "dispatchStart": index if native else 0,
            "dispatchCount": 1 if native else 0,
        }
        for index, name in enumerate(proof.UPSTREAM_TESTS)
    ]


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "worker",
        "skip",
        "empty-trace",
        "missing-trace",
        "extra-trace",
        "coverage",
        "source",
    ],
)
def test_runner_preserves_failed_evidence_and_requires_complete_trace(
    tmp_path, monkeypatch, fault
):
    args = SimpleNamespace(
        mlx_root=tmp_path / "mlx",
        packages=tmp_path / "packages",
        random=tmp_path / "random",
        bfloat=tmp_path / "bfloat",
        reductions=[tmp_path / "assertions", tmp_path / "extrema"],
        output_dir=tmp_path / "evidence",
    )
    monkeypatch.setattr(
        proof, "make_host", lambda *args: SimpleNamespace(target="opengl")
    )
    snapshots = iter(
        [
            {"commit": proof.COMMIT},
            {"commit": "changed" if fault == "source" else proof.COMMIT},
        ]
    )
    monkeypatch.setattr(proof, "verify_prepared", lambda root: next(snapshots))
    monkeypatch.setattr(
        proof, "upstream_test_sources", lambda *args: {"test_quantized.py": "hash"}
    )
    monkeypatch.setattr(proof, "event_packages", lambda *args: [])
    audited = []
    monkeypatch.setattr(
        proof, "audit_event", lambda event, *args, **kwargs: audited.append(event)
    )
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        mode = command[command.index("--worker") + 1]
        output = Path(command[-1])
        output.mkdir(parents=True)
        result = records(mode == "native")
        if mode == "native":
            result[-1]["dispatchCount"] = 2
            if fault == "skip":
                result[0]["skipped"] = ["not executed"]
            trace = [
                {"entry": entry}
                for entry in (
                    "all_reduce_maxbfloat16",
                    "all_reduce_maxfloat32",
                    "all_reduce_andbool_",
                    "row_reduce_small_1_reduce_maxbfloat16",
                    "row_reduce_small_1_reduce_maxfloat32",
                )
            ]
            if fault == "empty-trace":
                trace.clear()
            elif fault == "missing-trace":
                trace.pop()
            elif fault == "extra-trace":
                trace.append(trace[0])
            elif fault == "coverage":
                trace[-1] = trace[0]
            (output / "trace.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in trace)
            )
        (output / "results.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=int(fault == "worker" and mode == "native"))

    monkeypatch.setattr(proof.subprocess, "run", execute)
    if fault:
        with pytest.raises((RuntimeError, ValueError)):
            proof.verify(args)
    else:
        proof.verify(args)
    evidence = json.loads((args.output_dir / "summary.json").read_text())
    assert evidence["passed"] is (fault is None)
    assert evidence["fullUpstreamSuite"] is False
    assert evidence["sourceUnchanged"] is (fault != "source")
    assert len(commands) == 2
    for mode, command in zip(("cpu", "native"), commands):
        receipt = json.loads((args.output_dir / f"{mode}.command.json").read_text())
        assert receipt["command"] == command
        assert command.count("--reductions") == 2
        assert command[command.index("--timeout-seconds") + 1] == (
            "1800" if mode == "native" else "180"
        )
    if fault is None:
        assert evidence["nativeDispatches"] == len(audited) == 5


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "duplicate",
        "test",
        "testsRun",
        "success",
        "errors",
        "failures",
        "skipped",
        "expectedFailures",
        "unexpectedSuccesses",
        "dispatchStart",
        "dispatchCount",
        "boolean-count",
        "boolean-start",
    ],
)
def test_every_upstream_method_must_execute_without_skips(native, fault):
    result = records(native)
    if fault == "missing":
        result.pop()
    elif fault == "duplicate":
        result.append(copy.deepcopy(result[0]))
    elif fault == "test":
        result[0][fault] = "another.test"
    elif fault in ("testsRun", "dispatchStart"):
        result[0][fault] += 1
    elif fault == "dispatchCount":
        result[0][fault] = 0 if native else 1
    elif fault == "success":
        result[0][fault] = False
    elif fault == "boolean-count":
        result[0]["dispatchCount"] = native
    elif fault == "boolean-start":
        result[0]["dispatchStart"] = False
    elif fault:
        result[0][fault] = ["unexpected outcome"]
    if fault:
        with pytest.raises(ValueError):
            proof.validate_results(result, native=native)
    else:
        assert proof.validate_results(result, native=native) == (4 if native else 0)


def event_fixture(tmp_path, target, region):
    root = tmp_path / "package"
    root.mkdir()
    (root / "source").write_bytes(b"shader source")
    artifact = {
        "packagePath": "source",
        "sizeBytes": 13,
        "hash": {
            "algorithm": "sha256",
            "value": hashlib.sha256(b"shader source").hexdigest(),
        },
    }
    descriptor = {"target": target, "artifact": artifact}
    module_root = tmp_path / "native/native-modules"
    module_root.mkdir(parents=True)
    modules = []
    for extension in {
        "metal": (".air", ".metallib"),
        "directx": (".dxil",),
        "opengl": (".glsl",),
    }[target]:
        path = module_root / ("module" + extension)
        path.write_bytes(b"unit-test module")
        modules.append(
            {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    entry = "row_reduce_small_1_reduce_maxbfloat16"
    launch = {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 1, 1]}
    request = {
        "target": target,
        "dispatch": copy.deepcopy(launch),
        "entryPoint": {"metal": entry, "directx": "CSMain", "opengl": "main"}[target],
    }
    identity = {key: artifact[key] for key in ("sizeBytes", "hash")}
    details = {
        "request": request,
        "nativeRuntimeDispatch": copy.deepcopy(request),
        "module": modules[0],
        "validationModules": modules[1:],
        "adapterSteps": [
            {"action": name, "status": "passed"}
            for name in {
                "metal": [
                    "compile-metal-for-native-runtime",
                    "link-metal-for-native-runtime",
                ],
                "directx": ["compile-hlsl-for-directx-runtime"],
                "opengl": ["validate-glsl-for-opengl-runtime"],
            }[target]
        ],
        "artifactIdentityVerification": {
            "verificationStatus": "verified",
            "target": target,
            "expectedIdentity": identity,
            "observedIdentity": copy.deepcopy(identity),
        },
    }
    if region:
        descriptor["provenance"] = {"dispatchRegion": copy.deepcopy(launch)}
        details = {
            "regions": [
                {
                    "artifact": artifact,
                    "packageRoot": str(root),
                    "provenance": descriptor["provenance"],
                    **copy.deepcopy(launch),
                    "moduleFile": modules[0]["file"],
                    "moduleHash": modules[0]["sha256"],
                }
            ]
        }
    event = {
        "target": target,
        "dispatchVersion": 3,
        "entry": entry,
        **launch,
        "artifact": copy.deepcopy(artifact),
        "packageRoot": str(root),
        "details": details,
    }
    return event, [(descriptor, root)]


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "target",
        "version",
        "artifact",
        "source",
        "escape",
        "root",
        "module",
        "module-escape",
        "launch",
        "entry",
        "compiler",
        "identity",
    ],
)
def test_native_receipts_and_sources_are_required(tmp_path, target, fault):
    event, packages = event_fixture(tmp_path, target, False)
    details = event["details"]
    if fault == "target":
        event["target"] = "another"
    elif fault == "version":
        event["dispatchVersion"] = 1
    elif fault == "artifact":
        event["artifact"]["sizeBytes"] += 1
    elif fault == "source":
        (packages[0][1] / "source").write_bytes(b"modified")
    elif fault == "escape":
        packages[0][0]["artifact"]["packagePath"] = "../source"
        event["artifact"]["packagePath"] = "../source"
    elif fault == "root":
        event["packageRoot"] = str(tmp_path)
    elif fault == "module":
        Path(details["module"]["file"]).write_bytes(b"modified")
    elif fault == "module-escape":
        details["module"]["file"] = str(tmp_path / "outside")
    elif fault == "launch":
        details["request"]["dispatch"]["workgroupSize"][0] += 1
    elif fault == "entry":
        details["request"]["entryPoint"] = "another"
    elif fault == "compiler":
        details["adapterSteps"][0]["status"] = "failed"
    elif fault == "identity":
        details["artifactIdentityVerification"]["verificationStatus"] = "failed"
    if fault:
        with pytest.raises(ValueError):
            proof.audit_event(event, packages, tmp_path, target=target)
    else:
        proof.audit_event(event, packages, tmp_path, target=target)


@pytest.mark.parametrize("target", ["opengl", "directx"])
@pytest.mark.parametrize(
    "fault", [None, "missing", "extra", "geometry", "provenance", "module", "root"]
)
def test_every_region_is_audited_and_strictly_compiled(
    tmp_path, monkeypatch, target, fault
):
    event, packages = event_fixture(tmp_path, target, True)
    regions = event["details"]["regions"]
    if fault == "missing":
        regions.clear()
    elif fault == "extra":
        regions.append(copy.deepcopy(regions[0]))
    elif fault == "geometry":
        regions[0]["workgroupSize"][0] += 1
    elif fault == "provenance":
        regions[0]["provenance"] = {}
    elif fault == "module":
        regions[0]["moduleHash"] = "0" * 64
    elif fault == "root":
        regions[0]["packageRoot"] = str(tmp_path)
    compiled = []
    monkeypatch.setattr(proof, "compile_entry", lambda *args: compiled.append(args))
    if fault:
        with pytest.raises(ValueError):
            proof.audit_event(event, packages, tmp_path, target=target)
        assert not compiled
    else:
        proof.audit_event(event, packages, tmp_path, target=target)
        assert compiled == [
            (target, event["entry"] + "-0", *packages[0], tmp_path / "compilation")
        ]
