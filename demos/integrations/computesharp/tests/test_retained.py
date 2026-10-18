"""Retain the original Windows-specific and CPU-only upstream test projects."""

import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import (  # noqa: E402
    HOST_CONTRACT,
    ROOT,
    export,
    load_module,
    vstest,
)

RETAINED = load_module("computesharp_retained", ROOT / "retained.py")
QUALIFY = load_module("computesharp_qualify_retained", ROOT / "qualify.py")
SUMMARY = load_module(
    "computesharp_results_summary_retained", ROOT / "results_summary.py"
)
SUITES = sys.modules["suites"]
BASELINE = sys.modules["baseline"]
COVERAGE = sys.modules["coverage_contract"]
DEVICE_LOST = "ComputeSharp.Tests.DeviceLost"
SOURCE_GENERATORS = "ComputeSharp.Tests.SourceGenerators"


def inventory():
    return SUITES.load_inventory()


def write_inventory(tmp_path, data):
    path = tmp_path / "test_projects.json"
    path.write_text(json.dumps(data))
    return path


def test_windows_specific_and_cpu_only_projects_are_retained_or_explained():
    data = inventory()
    retained = SUITES.retained_suites(data)
    assert list(retained) == [DEVICE_LOST, SOURCE_GENERATORS]
    assert retained[DEVICE_LOST].scope == "windows-specific"
    assert retained[SOURCE_GENERATORS].scope == "cpu-only"
    assert retained[SOURCE_GENERATORS].retention["hostOnly"] is True
    resolver = data["projects"]["ComputeSharp.Tests.NativeLibrariesResolver"]
    assert (
        "retained" not in resolver
        and "verify-packages" in resolver["notRetainedReason"]
    )
    assert SUITES.check_contracts(data) == []
    # Every upstream step of DeviceLost is reproduced verbatim or is the
    # allocator variant; the accounted runs are its three test classes.
    entry = data["projects"][DEVICE_LOST]
    runs = entry["retained"]["runs"]
    reproduced = {run["upstreamStep"] for run in runs if run.get("upstreamStep")}
    assert reproduced | set(entry["variants"]) == set(entry["upstreamSteps"])
    assert [run["filter"] for run in runs if run["accounted"]] == [
        f"ClassName={DEVICE_LOST}.{name}Tests"
        for name in ("DeviceDisposal", "DeviceLost", "GetDefaultDevice")
    ]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda e: e["retained"].update(runs=[]), "must list its retained runs"),
        (
            lambda e: [run.update(accounted=False) for run in e["retained"]["runs"]],
            "must reproduce a step or say why",
        ),
        (
            lambda e: e["retained"]["runs"][0].update(upstreamStep="run-tests: other"),
            "reproduces an unknown step",
        ),
        (lambda e: e.pop("variants"), "neither reproduces nor explains"),
        (lambda e: e["retained"].update(configuration="Debug "), "configuration"),
        (lambda e: e["retained"].pop("hostOnly"), "host-only"),
        (lambda e: e.update(notRetainedReason="x" * 40), "must not say why"),
        (lambda e: e.pop("retained"), "must be retained or say why"),
        (
            lambda e: e["retained"]["runs"].append(dict(e["retained"]["runs"][0])),
            "invalid or repeated run name",
        ),
    ],
)
def test_incomplete_retention_is_rejected(tmp_path, edit, message):
    data = copy.deepcopy(inventory())
    edit(data["projects"][DEVICE_LOST])
    with pytest.raises(ValueError, match=message):
        SUITES.load_inventory(write_inventory(tmp_path, data))


def test_only_windows_specific_and_cpu_only_projects_are_retained(tmp_path):
    data = copy.deepcopy(inventory())
    data["projects"]["ComputeSharp.D2D1.Tests"]["notRetainedReason"] = "x" * 40
    with pytest.raises(ValueError, match="not Windows-specific or CPU-only"):
        SUITES.load_inventory(write_inventory(tmp_path, data))


def test_retained_contracts_hold_one_category_on_the_original_profile(
    tmp_path, monkeypatch
):
    data = inventory()
    item = SUITES.retained_suites(data)[DEVICE_LOST]
    contract = json.loads(item.contract.read_text())
    main = json.loads((ROOT / "coverage.json").read_text())
    assert contract["profiles"] == {"directx-warp": main["profiles"]["directx-warp"]}

    def problems(changed):
        path = tmp_path / "coverage.json"
        path.write_text(json.dumps(changed))
        moved = SUITES.Suite(**{**item.__dict__, "contract": path})
        monkeypatch.setattr(
            SUITES, "retained_suites", lambda inv=None: {item.name: moved}
        )
        return SUITES.check_contracts(data)

    widened = copy.deepcopy(contract)
    widened["profiles"]["metal-apple-gpu"] = main["profiles"]["metal-apple-gpu"]
    assert f"{item.name}: contract profiles differs" in " ".join(problems(widened))
    mixed = copy.deepcopy(contract)
    next(iter(mixed["classes"].values()))["category"] = "required-gpu"
    assert any("must be windows-specific" in p for p in problems(mixed))
    # Any edit of a frozen contract also needs a new lock.
    assert any("not frozen from the current contract" in p for p in problems(mixed))


def test_the_coverage_command_freezes_retained_projects():
    assert SUITES.suite(SOURCE_GENERATORS, retained=True).scope == "cpu-only"
    with pytest.raises(ValueError, match="not a ported upstream test project"):
        SUITES.suite(SOURCE_GENERATORS)


def _host_discovery(host_tools, root):
    discovery = root / "discovery"
    result = export(host_tools, discovery, options=["--host-only"])
    assert result.returncode == 0, result.stderr
    return discovery


def test_host_only_export_names_rows_as_mstest_does(host_tools, tmp_path):
    discovery = _host_discovery(host_tools, tmp_path)
    report = json.loads((discovery / "discovery.json").read_text())
    assert report["hostOnly"] is True and report["shaderCount"] == 0
    assert {case["displayName"] for case in report["tests"]} == {
        "Plain",
        'Named ("checked")',
        'Named ("two words")',
        'Mixed ("first",1)',
        'Mixed ("second",2)',
        "Displayed custom",
    }
    trx = vstest(host_tools, tmp_path, "all")
    accounting = BASELINE.verify_results(discovery, trx, "directx-warp", HOST_CONTRACT)
    assert accounting["requiredCasesPassed"] == 6
    assert accounting["requiredGpuCasesPassed"] == 0
    lock = COVERAGE.build_lock(
        COVERAGE.load_contract(HOST_CONTRACT),
        BASELINE.load_discovery(discovery),
        HOST_CONTRACT.read_bytes(),
    )
    assert lock["hostOnly"] is True


def test_an_export_without_shaders_must_be_host_only(
    host_tools, discovery_tools, tmp_path
):
    plain = export(host_tools, tmp_path / "plain")
    assert plain.returncode != 0
    assert "No generated HLSL descriptors discovered" in plain.stderr
    shaders = export(discovery_tools, tmp_path / "shaders", options=["--host-only"])
    assert shaders.returncode != 0
    assert "host-only test project contains generated HLSL" in shaders.stderr
    usage = export(discovery_tools, tmp_path / "usage", options=["--other"])
    assert usage.returncode == 2


def test_discovery_rejects_inconsistent_host_only_reports(baseline_fixture, tmp_path):
    discovery, _ = baseline_fixture
    report = json.loads((discovery / "discovery.json").read_text())

    def load(changed):
        directory = tmp_path / str(len(list(tmp_path.iterdir())))
        directory.mkdir()
        (directory / "discovery.json").write_text(json.dumps(changed))
        return BASELINE.load_discovery(directory)

    with pytest.raises(ValueError, match="shader inventory"):
        load({**report, "hostOnly": True})
    with pytest.raises(ValueError, match="host-only marker"):
        load({**report, "hostOnly": False})
    with pytest.raises(ValueError, match="shader inventory"):
        load({**report, "shaders": [], "shaderCount": 0})


def test_split_runs_merge_into_one_accounted_result(host_tools, tmp_path):
    discovery = _host_discovery(host_tools, tmp_path)
    first = vstest(host_tools, tmp_path, "first", "Name=Plain")
    rest = vstest(host_tools, tmp_path, "rest", "Name!=Plain")
    merged = tmp_path / "merged" / "baseline.trx"
    RETAINED.merge_trx([first, rest], merged)
    accounting = BASELINE.verify_results(
        discovery, merged, "directx-warp", HOST_CONTRACT
    )
    assert accounting["requiredCasesPassed"] == 6
    root = ET.parse(merged).getroot()
    counters = root.find("t:ResultSummary/t:Counters", RETAINED.TRX)
    assert counters.get("total") == "6" and counters.get("passed") == "6"
    # A case missing from every run, or reported by two, is not accounted.
    RETAINED.merge_trx([rest], tmp_path / "partial.trx")
    with pytest.raises(
        BASELINE.AccountingError, match="expected 1 results but found 0"
    ):
        BASELINE.verify_results(
            discovery, tmp_path / "partial.trx", "directx-warp", HOST_CONTRACT
        )
    RETAINED.merge_trx([first, rest, first], tmp_path / "twice.trx")
    with pytest.raises(ValueError, match="duplicate TRX test definition"):
        BASELINE.verify_results(
            discovery, tmp_path / "twice.trx", "directx-warp", HOST_CONTRACT
        )


def test_the_merged_outcome_keeps_failures_and_incomplete_runs(tmp_path):
    def trx(name, outcome):
        path = tmp_path / f"{name}.trx"
        path.write_text(
            f'<TestRun xmlns="{RETAINED.TRX_NAMESPACE}"><Results/><TestDefinitions/>'
            f'<ResultSummary outcome="{outcome}"><Counters total="1" failed="0"/>'
            "</ResultSummary></TestRun>"
        )
        return path

    def outcome(*paths):
        RETAINED.merge_trx(list(paths), tmp_path / "merged.trx")
        root = ET.parse(tmp_path / "merged.trx").getroot()
        return root.find("t:ResultSummary", RETAINED.TRX).get("outcome")

    assert outcome(trx("a", "Completed"), trx("b", "Passed")) == "Completed"
    assert outcome(trx("c", "Completed"), trx("d", "Failed")) == "Failed"
    assert outcome(trx("e", "Failed"), trx("f", "Aborted")) == "Aborted"
    with pytest.raises(ValueError, match="No TRX files"):
        RETAINED.merge_trx([], tmp_path / "none.trx")


def test_the_merged_trx_has_the_same_bytes_on_every_platform(tmp_path, monkeypatch):
    """The Linux qualification re-merges the Windows runs and compares bytes."""
    runs = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.trx"
        document = (
            f'<TestRun xmlns="{RETAINED.TRX_NAMESPACE}"><Results><UnitTestResult>'
            "<Output><StdOut>first\r\nsecond</StdOut></Output></UnitTestResult>"
            '</Results><ResultSummary outcome="Completed"><Counters total="1"/>'
            "</ResultSummary></TestRun>"
        )
        path.write_bytes(document.encode("utf-8"))
        runs.append(path)
    RETAINED.merge_trx(runs, tmp_path / "native.trx")

    # Text-mode files translate line endings on Windows.
    def windows_open(file, mode="r", *args, **kwargs):
        if "b" not in mode:
            kwargs["newline"] = "\r\n"
        return open(file, mode, *args, **kwargs)

    monkeypatch.setattr(ET, "open", windows_open, raising=False)
    RETAINED.merge_trx(runs, tmp_path / "windows.trx")

    merged = (tmp_path / "windows.trx").read_bytes()
    assert merged == (tmp_path / "native.trx").read_bytes()
    assert b"\r" not in merged
    assert merged.startswith(b"<?xml version='1.0' encoding='utf-8'?>\n")


class FakeDotnet:
    """Record commands and write the TRX files and logs dotnet would."""

    def __init__(self, tmp_path, results, exits=None):
        self.tmp_path = tmp_path
        self.results = results
        self.exits = exits or {}
        self.commands = []

    def __call__(self, command, log, *, deadline, check=True):
        self.commands.append(command)
        stdout = ""
        returncode = 0
        if command[1] == "msbuild":
            assembly = self.tmp_path / "bin" / (Path(command[2]).stem + ".dll")
            assembly.parent.mkdir(parents=True, exist_ok=True)
            assembly.write_bytes(b"assembly")
            stdout = str(assembly) + "\n"
        if command[1] == "test":
            name = command[command.index("--logger") + 1].split("=")[1][:-4]
            directory = Path(command[command.index("--results-directory") + 1])
            outcomes = self.results.get(name)
            returncode = self.exits.get(name, 0)
            if outcomes is None:
                stdout = RETAINED.NO_MATCH + ".\n"
            else:
                directory.mkdir(parents=True, exist_ok=True)
                cases = "".join(
                    f'<UnitTestResult testId="{i}" testName="{name}{i}" '
                    f'outcome="{o}"/>'
                    for i, o in enumerate(outcomes)
                )
                (directory / f"{name}.trx").write_text(
                    f'<TestRun xmlns="{RETAINED.TRX_NAMESPACE}"><Results>{cases}'
                    '</Results><ResultSummary outcome="Completed"/></TestRun>'
                )
        return type(
            "Completed", (), {"stdout": stdout, "stderr": "", "returncode": returncode}
        )()


def _patch_accounting(monkeypatch, tmp_path, calls):
    monkeypatch.setattr(
        RETAINED.companions.coverage_contract,
        "check",
        lambda discovery, lock, contract: calls.append(("lock", lock))
        or {"caseCount": 1, "totals": {}},
    )
    monkeypatch.setattr(
        RETAINED.baseline,
        "load_discovery",
        lambda directory: {
            "assembly": {"sha256": RETAINED._sha256(next((tmp_path / "bin").iterdir()))}
        },
    )

    def verify(discovery, trx, profile, contract):
        calls.append(("verify", trx, profile, contract))
        return {
            "expected": {"pass": 1},
            "requiredCasesPassed": 1,
            "requiredGpuCasesPassed": 0,
        }

    monkeypatch.setattr(RETAINED.baseline, "verify_results", verify)


def test_device_lost_reproduces_the_upstream_steps_and_runs_each_class(
    tmp_path, monkeypatch
):
    calls = []
    _patch_accounting(monkeypatch, tmp_path, calls)
    item = SUITES.retained_suites()[DEVICE_LOST]
    classes = ("DeviceDisposalTests", "DeviceLostTests", "GetDefaultDeviceTests")
    fake = FakeDotnet(tmp_path, {name: ["Passed"] for name in classes})
    directory = tmp_path / "out"
    report = RETAINED.run_suite(item, tmp_path / "upstream", directory, runner=fake)
    build, target, discovery, *tests = fake.commands
    assert build[1:5] == [
        "build",
        str(tmp_path / "upstream" / item.project),
        "-c",
        "Release",
    ]
    assert "-p:Platform=x64" not in build
    assert target[-1] == "-p:Configuration=Release"
    assert "--host-only" not in discovery
    assert [t[t.index("--filter") + 1] for t in tests] == [
        run["filter"] for run in item.retention["runs"]
    ]
    assert all("--no-build" in t and t[t.index("-c") + 1] == "Release" for t in tests)
    upstream = [run for run in report["runs"] if not run["accounted"]]
    assert [run["selectedCases"] for run in upstream] == [0, 0, 0]
    assert all(run["trx"] is None for run in upstream)
    assert [run["trx"] for run in report["runs"] if run["accounted"]] == [
        f"{name}.trx" for name in classes
    ]
    merged = directory / "trx" / "baseline.trx"
    assert calls[-1] == ("verify", merged, "directx-warp", item.contract)
    results = (
        ET.parse(merged).getroot().findall("t:Results/t:UnitTestResult", RETAINED.TRX)
    )
    assert [r.get("testName") for r in results] == [f"{name}0" for name in classes]
    assert (
        json.loads((directory / "baseline.json").read_text())["requiredCasesPassed"]
        == 1
    )


def test_source_generators_export_host_only_and_run_as_upstream(tmp_path, monkeypatch):
    calls = []
    _patch_accounting(monkeypatch, tmp_path, calls)
    item = SUITES.retained_suites()[SOURCE_GENERATORS]
    fake = FakeDotnet(tmp_path, {"SourceGenerators": ["Passed"]})
    RETAINED.run_suite(item, tmp_path, tmp_path / "out", runner=fake)
    build, target, discovery, test = fake.commands
    assert build[build.index("-c") + 1] == "Debug"
    assert discovery[-1] == "--host-only"
    assert "--filter" not in test


@pytest.mark.parametrize(
    ("results", "exits", "message"),
    [
        ({"DeviceDisposalTests": ["Passed"]}, {}, "DeviceLostTests wrote no TRX"),
        (
            {"upstream-DeviceLost": ["Failed"], "DeviceDisposalTests": ["Passed"]},
            {"upstream-DeviceLost": 1},
            "Upstream step run upstream-DeviceLost failed",
        ),
    ],
)
def test_failing_or_missing_runs_fail_the_project(
    tmp_path, monkeypatch, results, exits, message
):
    _patch_accounting(monkeypatch, tmp_path, [])
    every = {"DeviceDisposalTests", "DeviceLostTests", "GetDefaultDeviceTests"}
    if "DeviceLostTests wrote" not in message:
        results = {**{name: ["Passed"] for name in every}, **results}
    item = SUITES.retained_suites()[DEVICE_LOST]
    with pytest.raises(RETAINED.companions.StepError, match=message):
        RETAINED.run_suite(
            item,
            tmp_path,
            tmp_path / "out",
            runner=FakeDotnet(tmp_path, results, exits),
        )


def _retained_artifacts(tmp_path, item):
    """Windows artifacts of one retained project whose runs all passed."""
    directory = tmp_path / "directx" / "retained" / item.name
    runs_dir = directory / "trx" / "runs"
    runs_dir.mkdir(parents=True)
    runs, accounted = [], []
    for index, run in enumerate(item.retention["runs"]):
        entry = {
            key: run.get(key) for key in ("name", "upstreamStep", "filter", "accounted")
        }
        entry.update(exitCode=0, failedCases=0, selectedCases=0, trx=None)
        if run["accounted"]:
            trx = runs_dir / f"{run['name']}.trx"
            trx.write_text(
                f'<TestRun xmlns="{RETAINED.TRX_NAMESPACE}"><Results>'
                f'<UnitTestResult testId="{index}" outcome="Passed"/></Results>'
                '<ResultSummary outcome="Completed"/></TestRun>'
            )
            entry.update(trx=trx.name, selectedCases=1, trxSHA256=RETAINED._sha256(trx))
            accounted.append(trx)
        runs.append(entry)
    RETAINED.merge_trx(accounted, directory / "trx" / "baseline.trx")
    (directory.parent / "retained.json").write_text(
        json.dumps({"suites": {item.name: {"status": "passed", "runs": runs}}})
    )
    return directory


def test_retained_projects_qualify_on_the_original_profile_only(tmp_path, monkeypatch):
    item = SUITES.retained_suites()[DEVICE_LOST]
    directory = _retained_artifacts(tmp_path, item)
    seen = []
    monkeypatch.setattr(
        QUALIFY,
        "directx_accounting",
        lambda d, *, contract_path, lock_path: seen.append(
            (d, contract_path, lock_path)
        )
        or {"profile": "directx-warp", "accounting": {"status": "passed"}},
    )
    report = QUALIFY.qualify_retained(item, directx=tmp_path / "directx")
    assert seen == [(directory, item.contract, item.lock)]
    assert report["status"] == "passed"
    assert report["countedAsTranslatedExecution"] is False
    assert list(report["profiles"]) == ["directx-warp"]
    runs = report["profiles"]["directx-warp"]["runs"]
    assert runs["selectedCases"]["upstream-DeviceLost"] == 0
    assert runs["selectedCases"]["DeviceLostTests"] == 1
    summary = SUMMARY.summarize_retained(report)
    assert summary["scope"] == "windows-specific"
    assert summary["profiles"]["directx-warp"]["selectedCases"] == runs["selectedCases"]

    # A changed run TRX, or a merged TRX that is not the merge of the runs, fails.
    trx = directory / "trx" / "runs" / "DeviceLostTests.trx"
    original = trx.read_text()
    trx.write_text(original.replace("Passed", "Failed"))
    failed = QUALIFY.qualify_retained(item, directx=tmp_path / "directx")
    assert failed["status"] == "failed"
    assert failed["profiles"]["directx-warp"]["runs"]["errors"] == [
        "run DeviceLostTests has no TRX matching its record"
    ]
    trx.write_text(original)
    merged = directory / "trx" / "baseline.trx"
    merged.write_bytes(merged.read_bytes() + b"\n")
    failed = QUALIFY.qualify_retained(item, directx=tmp_path / "directx")
    assert failed["profiles"]["directx-warp"]["runs"]["errors"] == [
        "baseline.trx is not the merge of the accounted runs"
    ]


def test_a_failed_retained_project_fails_the_qualification(tmp_path, monkeypatch):
    item = SUITES.retained_suites()[SOURCE_GENERATORS]
    directory = _retained_artifacts(tmp_path, item)
    record = json.loads((directory.parent / "retained.json").read_text())
    record["suites"][item.name] = {
        "status": "failed",
        "error": "6 accounting mismatch(es)",
    }
    (directory.parent / "retained.json").write_text(json.dumps(record))
    monkeypatch.setattr(
        QUALIFY, "directx_accounting", lambda d, **kwargs: {"profile": "directx-warp"}
    )
    monkeypatch.setattr(QUALIFY, "qualify_directx", lambda d, **kwargs: {})
    report = QUALIFY.qualify(
        directx=tmp_path / "directx",
        targets={},
        output=tmp_path / "qualification",
        retained_suites={item.name: item},
    )
    assert report["status"] == "failed"
    retained = report["retained"][item.name]
    assert retained["profiles"]["directx-warp"]["runs"]["errors"] == [
        "the Windows job reported 'failed': 6 accounting mismatch(es)"
    ]
    assert f"  retained cpu-only {item.name}: failed" in QUALIFY.summary_lines(report)
    published = json.loads((tmp_path / "qualification" / "results.json").read_text())
    assert published["retained"][item.name]["countedAsTranslatedExecution"] is False


def test_the_qualification_command_requires_every_retained_project(
    tmp_path, monkeypatch
):
    captured = {}

    def qualify(**kwargs):
        captured.update(kwargs)
        return {"status": "passed", "profiles": {}, "suites": {}}

    monkeypatch.setattr(QUALIFY, "qualify", qualify)
    assert QUALIFY.main(["--directx", str(tmp_path), "--output", str(tmp_path)]) == 0
    assert set(captured["retained_suites"]) == {DEVICE_LOST, SOURCE_GENERATORS}


def test_the_retained_command_rejects_unknown_projects(capsys):
    with pytest.raises(SystemExit):
        RETAINED.main(
            ["run", "--source", ".", "--output", ".", "--suite", "ComputeSharp.Tests"]
        )
    assert "not a retained project" in capsys.readouterr().err
