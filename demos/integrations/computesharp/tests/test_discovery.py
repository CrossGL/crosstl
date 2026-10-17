"""Exercise discovery through real MSTest data sources and descriptor fixtures."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "computesharp_baseline", ROOT / "baseline.py"
)
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)
FIXTURE_CONTRACT = ROOT / "tests" / "fixture" / "coverage.json"
FIXTURE_SOURCE = "demos/integrations/computesharp/tests/fixture/Fixture.cs"


@pytest.fixture(scope="session")
def discovery_tools(tmp_path_factory):
    dotnet = shutil.which("dotnet")
    if dotnet is None:
        if os.environ.get("CROSSTL_REQUIRE_DOTNET") == "1":
            pytest.fail("The discovery contract requires the .NET 10 SDK")
        pytest.skip("The discovery contract requires the .NET 10 SDK")
    output = tmp_path_factory.mktemp("discovery-build")
    for name, project in (
        ("tool", ROOT / "discovery" / "Discovery.csproj"),
        ("fixture", ROOT / "tests" / "fixture" / "Fixture.csproj"),
    ):
        subprocess.run(
            [
                dotnet,
                "build",
                str(project),
                "-c",
                "Release",
                "-o",
                str(output / name),
                "--artifacts-path",
                str(output / (name + "-artifacts")),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    return [
        dotnet,
        str(output / "tool" / "Discovery.dll"),
        str(output / "fixture" / "Fixture.dll"),
    ]


def export(tools, output, mode=None):
    environment = os.environ.copy()
    environment.pop("DISCOVERY_FIXTURE_MODE", None)
    if mode:
        environment["DISCOVERY_FIXTURE_MODE"] = mode
    return subprocess.run(
        [*tools, str(output)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_exports_expanded_cases_and_actual_descriptor_source(discovery_tools, tmp_path):
    output = tmp_path / "export"
    result = export(discovery_tools, output)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "discovery.json").read_text())
    assert report["testMethodCount"] == 5
    assert report["caseCount"] == 9
    assert len({case["id"] for case in report["tests"]}) == 9
    sources = {entry["method"]: entry["source"] for entry in report["methods"]}
    assert set(sources) == {case["method"] for case in report["tests"]}
    assert {source["path"] for source in sources.values()} == {FIXTURE_SOURCE}
    fixture_lines = (ROOT / "tests" / "fixture" / "Fixture.cs").read_text().splitlines()
    signature_line = next(
        index for index, line in enumerate(fixture_lines, 1) if "void Gated" in line
    )
    # The first sequence point is the method body (brace or first statement).
    assert 0 < sources["DiscoveryFixture.Cases.Gated"]["line"] - signature_line <= 4
    values = [case for case in report["tests"] if case["method"].endswith(".Values")]
    assert {case["arguments"][0]["bits"] for case in values} == {"80000000", "3fc00000"}
    assert {case["arguments"][1]["name"] for case in values} == {
        "System.Int32",
        "System.UInt32",
    }
    repeated = [case for case in values if case["arguments"][0]["bits"] == "3fc00000"]
    assert sorted(case["occurrence"] for case in repeated) == [1, 2]
    assert len({case["displayName"] for case in repeated}) == 1
    assert len({case["id"] for case in repeated}) == 2
    assert report["shaderCount"] == 1
    shader = report["shaders"][0]
    source = (output / shader["path"]).read_bytes()
    assert b"output[id.x] = id.x" in source
    assert shader["sha256"] == hashlib.sha256(source).hexdigest()
    assert shader["threads"] == [32, 1, 1]
    assert shader["constantBufferSize"] == 16
    assert shader["resourceDescriptorRanges"] == [{"type": 1, "register": 2}]
    assert not report["coverageFrozen"]
    assert not report["nativeExecutionQualified"]
    other = tmp_path / "second-export"
    assert export(discovery_tools, other).returncode == 0
    assert (other / "discovery.json").read_bytes() == (
        output / "discovery.json"
    ).read_bytes()


def test_repeated_custom_data_rows_are_numbered(discovery_tools, tmp_path):
    output = tmp_path / "export"
    result = export(discovery_tools, output, "duplicate")
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "discovery.json").read_text())
    expanded = [
        case for case in report["tests"] if case["method"].endswith(".Expanded")
    ]
    assert [case["occurrence"] for case in expanded] == [1, 2]
    assert expanded[0]["arguments"] == expanded[1]["arguments"]
    assert expanded[0]["id"] != expanded[1]["id"]


@pytest.mark.parametrize(
    "mode, message",
    [
        ("empty", "Empty data source"),
        ("throw", "Fixture discovery failure"),
        ("arity", "Test data arity differs"),
        ("unsupported", "Unsupported test argument type"),
        ("blank", "Empty HLSL source"),
    ],
)
def test_failed_discovery_cannot_publish_a_partial_inventory(
    discovery_tools, tmp_path, mode, message
):
    output = tmp_path / "export"
    result = export(discovery_tools, output, mode)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial-*"))


def test_existing_export_is_preserved(discovery_tools, tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    (output / "keep.txt").write_text("previous evidence")
    result = export(discovery_tools, output)
    assert result.returncode != 0
    assert (output / "keep.txt").read_text() == "previous evidence"
    assert not (output / "discovery.json").exists()


@pytest.fixture(scope="session")
def baseline_fixture(discovery_tools, tmp_path_factory):
    root = tmp_path_factory.mktemp("baseline-proof")
    discovery = root / "discovery"
    result = export(discovery_tools, discovery)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            discovery_tools[0],
            "vstest",
            discovery_tools[2],
            "--Logger:trx;LogFileName=baseline.trx",
            "--ResultsDirectory:" + str(root),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return discovery, root / "baseline.trx"


def test_original_result_accounting_uses_real_mstest_output(baseline_fixture):
    _, trx = baseline_fixture
    names = [
        item.get("testName")
        for item in (
            ET.parse(trx).getroot().iter(f"{{{BASELINE.TRX['t']}}}UnitTestResult")
        )
    ]
    # MSTest executes the repeated row twice under one display name.
    assert len(names) == 9
    assert len(set(names)) == 8
    report = BASELINE.verify_baseline(*baseline_fixture, FIXTURE_CONTRACT)
    assert report["caseCount"] == 9
    assert report["expected"] == {
        "pass": 7,
        "inconclusive": 1,
        "optional": 1,
        "excluded": 0,
    }
    assert report["requiredCasesPassed"] == 7
    assert report["requiredGpuCasesPassed"] == 6
    assert report["warpCasesPassed"] == 1
    assert len(report["capabilityInconclusive"]) == 1
    assert len(report["optionalHardwareSkipped"]) == 1
    assert not report["nativeTargetExecutionQualified"]
    assert report["coverageFrozen"]


def test_translated_profile_rejects_baseline_only_outcomes(baseline_fixture):
    # The fixture run executes an excluded platform case, leaves a Discrete row
    # inconclusive and reports an available capability as inconclusive.
    with pytest.raises(ValueError):
        BASELINE.verify_results(*baseline_fixture, "translated", FIXTURE_CONTRACT)


def _mutated_trx(original, tmp_path, mutation):
    tree = ET.parse(original)
    namespace = BASELINE.TRX
    results = tree.getroot().find("t:Results", namespace)
    required = next(item for item in results if "Warp" in item.get("testName", ""))
    names = [item.get("testName") for item in results]
    repeated = next(item for item in results if names.count(item.get("testName")) > 1)
    if mutation == "missing":
        results.remove(required)
    elif mutation == "missing-repeat":
        results.remove(repeated)
    elif mutation == "extra-repeat":
        results.append(copy.deepcopy(repeated))
    elif mutation == "failed":
        required.set("outcome", "Failed")
    elif mutation == "failed-repeat":
        repeated.set("outcome", "Failed")
    elif mutation == "skipped":
        required.set("outcome", "NotExecuted")
    elif mutation == "duplicate":
        results.append(copy.deepcopy(required))
    elif mutation == "unknown":
        required.set("testId", "unknown")
    elif mutation == "gate-missing-device":
        gated = next(
            item
            for item in results
            if item.get("outcome") == "NotExecuted"
            and "Gated" in item.get("testName", "")
        )
        trace = gated.find("t:Output/t:ErrorInfo/t:StackTrace", namespace)
        trace.text = (
            "   at ComputeSharp.Tests.Extensions.DeviceExtensions.Get(Device type)\n"
            + (trace.text or "")
        )
    elif mutation == "gate-failed":
        gated = next(
            item
            for item in results
            if item.get("outcome") == "NotExecuted"
            and "Gated" in item.get("testName", "")
        )
        gated.set("outcome", "Failed")
    elif mutation == "optional-failed":
        optional = next(
            item for item in results if "Discrete" in item.get("testName", "")
        )
        optional.set("outcome", "Failed")
    else:
        tree.getroot().find("t:ResultSummary", namespace).set("outcome", "Aborted")
    changed = tmp_path / "changed.trx"
    tree.write(changed)
    return changed


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "missing-repeat",
        "extra-repeat",
        "failed",
        "failed-repeat",
        "skipped",
        "duplicate",
        "unknown",
        "gate-missing-device",
        "gate-failed",
        "optional-failed",
        "aborted",
    ],
)
def test_baseline_rejects_incomplete_or_failed_results(
    baseline_fixture, tmp_path, mutation
):
    discovery, original = baseline_fixture
    changed = _mutated_trx(original, tmp_path, mutation)
    with pytest.raises(ValueError):
        BASELINE.verify_baseline(discovery, changed, FIXTURE_CONTRACT)


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "escape",
        "absolute",
        "duplicate-case",
        "count",
        "ignored",
        "occurrence",
        "method-source",
        "method-missing",
    ],
)
def test_baseline_rejects_corrupt_discovery(baseline_fixture, tmp_path, mutation):
    original, trx = baseline_fixture
    discovery = tmp_path / "discovery"
    shutil.copytree(original, discovery)
    report = json.loads((discovery / "discovery.json").read_text())
    if mutation == "source":
        (discovery / report["shaders"][0]["path"]).write_text("different computation")
    elif mutation == "escape":
        report["shaders"][0]["path"] = "../outside.hlsl"
    elif mutation == "absolute":
        report["shaders"][0]["path"] = str(
            (discovery / report["shaders"][0]["path"]).resolve()
        )
    elif mutation == "duplicate-case":
        report["tests"][0]["id"] = report["tests"][1]["id"]
    elif mutation == "count":
        report["caseCount"] -= 1
    elif mutation == "ignored":
        report["tests"][0]["ignored"] = True
    elif mutation == "method-source":
        report["methods"][0]["source"]["path"] = "/absolute/Fixture.cs"
    elif mutation == "method-missing":
        report["methods"].pop()
        report["testMethodCount"] -= 1
    else:
        repeated = next(case for case in report["tests"] if case["occurrence"] == 2)
        repeated["occurrence"] = 3
    (discovery / "discovery.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        BASELINE.verify_baseline(discovery, trx, FIXTURE_CONTRACT)


COMPARE_SPEC = importlib.util.spec_from_file_location(
    "computesharp_compare", ROOT / "compare_discovery.py"
)
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
COMPARE_SPEC.loader.exec_module(COMPARE)


def test_parity_accepts_identical_inventories_from_separate_exports(
    discovery_tools, tmp_path
):
    for name in ("reference", "candidate"):
        result = export(discovery_tools, tmp_path / name)
        assert result.returncode == 0, result.stderr
    report = COMPARE.compare(tmp_path / "reference", tmp_path / "candidate")
    assert report["caseCount"] == 9
    assert report["shaderCount"] == 1


@pytest.mark.parametrize(
    "mutation", ["hlsl", "threads", "case", "ranges", "occurrence", "source-line"]
)
def test_parity_rejects_changed_hlsl_metadata_or_cases(
    discovery_tools, tmp_path, mutation
):
    for name in ("reference", "candidate"):
        result = export(discovery_tools, tmp_path / name)
        assert result.returncode == 0, result.stderr
    candidate = tmp_path / "candidate"
    report = json.loads((candidate / "discovery.json").read_text())
    shader = report["shaders"][0]
    if mutation == "hlsl":
        changed = b"RWStructuredBuffer<uint> output : register(u2);\n"
        (candidate / shader["path"]).write_bytes(changed)
        shader["sha256"] = hashlib.sha256(changed).hexdigest()
    elif mutation == "threads":
        shader["threads"] = [64, 1, 1]
    elif mutation == "case":
        report["tests"][0]["displayName"] += " changed"
    elif mutation == "ranges":
        shader["resourceDescriptorRanges"][0]["register"] = 3
    elif mutation == "source-line":
        report["methods"][0]["source"]["line"] += 1
    else:
        repeated = next(case for case in report["tests"] if case["occurrence"] == 2)
        report["tests"].remove(repeated)
        report["caseCount"] -= 1
    (candidate / "discovery.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        COMPARE.compare(tmp_path / "reference", candidate)


CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "computesharp_coverage_contract", ROOT / "coverage_contract.py"
)
CONTRACT = importlib.util.module_from_spec(CONTRACT_SPEC)
CONTRACT_SPEC.loader.exec_module(CONTRACT)


def test_frozen_lock_matches_the_reviewed_contract():
    lock = json.loads((ROOT / "coverage.lock.json").read_text())
    contract = CONTRACT.load_contract()
    assert (
        lock["contractSHA256"]
        == hashlib.sha256((ROOT / "coverage.json").read_bytes()).hexdigest()
    )
    assert lock["commit"] == contract["commit"]
    assert (
        sum(entry["cases"] for entry in lock["methods"].values()) == lock["caseCount"]
    )
    assert sum(total["cases"] for total in lock["totals"].values()) == lock["caseCount"]
    assert len(lock["methods"]) == lock["testMethodCount"]
    for name, profile in lock["profiles"].items():
        assert sum(profile["expected"].values()) == lock["caseCount"], name
        assert profile["execution"] == contract["profiles"][name]["execution"]
    translated = [
        name
        for name, profile in contract["profiles"].items()
        if profile["role"] == "translated-target"
    ]
    assert sorted(translated) == ["metal-apple-gpu", "opengl-mesa-llvmpipe"]
    for name in translated:
        expected = lock["profiles"][name]["expected"]
        # Nothing on a translated target is optional; only capability gates
        # may be inconclusive and only platform tests are excluded.
        assert expected["optional"] == 0
        assert expected["excluded"] == lock["totals"]["windows-specific"]["cases"]
    for signature, entry in lock["methods"].items():
        method = signature.split("(", 1)[0]
        assert (
            entry["category"]
            == (
                contract["methods"].get(method)
                or contract["classes"][method.rsplit(".", 1)[0]]
            )["category"]
        )
        assert entry["source"]["path"].startswith("tests/ComputeSharp.Tests/")


@pytest.fixture()
def fixture_export(discovery_tools, tmp_path):
    output = tmp_path / "export"
    result = export(discovery_tools, output)
    assert result.returncode == 0, result.stderr
    return output


def _contract_copy(tmp_path, change=None):
    contract = json.loads(FIXTURE_CONTRACT.read_text())
    if change:
        change(contract)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path


def test_lock_freezes_categories_gates_and_profile_expectations(
    fixture_export, tmp_path
):
    lock_path = tmp_path / "lock.json"
    contract = CONTRACT.load_contract(FIXTURE_CONTRACT)
    lock = CONTRACT.build_lock(
        contract, BASELINE.load_discovery(fixture_export), FIXTURE_CONTRACT.read_bytes()
    )
    lock_path.write_text(json.dumps(lock))
    assert lock["totals"]["windows-specific"] == {"methods": 1, "cases": 1}
    assert lock["totals"]["required-gpu"] == {"methods": 4, "cases": 8}
    assert lock["profiles"]["directx-warp"]["expected"] == {
        "pass": 7,
        "inconclusive": 1,
        "optional": 1,
        "excluded": 0,
    }
    assert lock["profiles"]["translated"]["expected"] == {
        "pass": 8,
        "inconclusive": 0,
        "optional": 0,
        "excluded": 1,
    }
    gated = next(entry for key, entry in lock["methods"].items() if ".Gated(" in key)
    assert gated["capabilityGates"] == {"fixture-feature:available": 1}
    CONTRACT.check(fixture_export, lock_path, FIXTURE_CONTRACT)


def test_lock_rejects_changed_inventory(fixture_export, tmp_path):
    lock_path = tmp_path / "lock.json"
    lock = CONTRACT.build_lock(
        CONTRACT.load_contract(FIXTURE_CONTRACT),
        BASELINE.load_discovery(fixture_export),
        FIXTURE_CONTRACT.read_bytes(),
    )
    lock_path.write_text(json.dumps(lock))
    report = json.loads((fixture_export / "discovery.json").read_text())
    dropped = next(case for case in report["tests"] if case["occurrence"] == 2)
    report["tests"].remove(dropped)
    report["caseCount"] -= 1
    (fixture_export / "discovery.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Coverage lock mismatch"):
        CONTRACT.check(fixture_export, lock_path, FIXTURE_CONTRACT)


def _drop_class(contract):
    contract["classes"].clear()


def _stale_method(contract):
    contract["methods"]["DiscoveryFixture.Cases.Removed"] = {
        "category": "cpu-only",
        "reason": "No longer exists.",
    }


def _unmatched_gate(contract):
    contract["capabilities"]["fixture-feature"]["gates"][0]["typeArguments"] = [
        "System.Double"
    ]


def _overlapping_gates(contract):
    gate = dict(contract["capabilities"]["fixture-feature"]["gates"][0])
    contract["capabilities"]["other"] = {"description": "Overlap.", "gates": [gate]}
    for profile in contract["profiles"].values():
        profile["capabilities"]["other"] = True


def _unstated_capability(contract):
    del contract["profiles"]["translated"]["capabilities"]["fixture-feature"]


@pytest.mark.parametrize(
    "change",
    [
        _drop_class,
        _stale_method,
        _unmatched_gate,
        _overlapping_gates,
        _unstated_capability,
    ],
)
def test_contract_rejects_incomplete_or_stale_classification(
    fixture_export, tmp_path, change
):
    path = _contract_copy(tmp_path, change)
    with pytest.raises(ValueError):
        CONTRACT.build_lock(
            CONTRACT.load_contract(path),
            BASELINE.load_discovery(fixture_export),
            path.read_bytes(),
        )
