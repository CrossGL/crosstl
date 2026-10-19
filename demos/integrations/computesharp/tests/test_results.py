"""Check the results summary and the published results of the port.

``results.json`` in the demo directory is the summary of a qualifying workflow
run. It must describe the current frozen contract: a passing qualification of
every profile with the frozen expected outcomes, every frozen negative control
and every shader of the image checks.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT, load_module  # noqa: E402

SUMMARY = load_module("computesharp_results_summary", ROOT / "results_summary.py")
PUBLISHED = ROOT / "results.json"
TRANSLATED = ("metal-apple-gpu", "opengl-mesa-llvmpipe")


def sha256(path, *, normalize_newlines=False):
    data = path.read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def synthetic_report():
    outputs = [
        {
            "shader": "HelloWorld",
            "sha256": "ab" * 32,
            "upstreamDelta": 1e-7,
            "versusDirectX": {
                "meanAbsoluteError": 0.01,
                "maxAbsoluteError": 1,
                "largePixelFraction": 0.0,
            },
            "directxBounds": {"meanAbsoluteError": 0.5, "largePixelFraction": 0.001},
            "difference": "HelloWorld-abababababab.png",
        }
    ]
    target = {
        "profile": "metal-apple-gpu",
        "status": "passed",
        "accounting": {
            "status": "passed",
            "execution": "hardware",
            "expected": {"pass": 2, "inconclusive": 1, "optional": 0, "excluded": 1},
            "requiredCasesPassed": 2,
            "requiredGpuCasesPassed": 2,
            "capabilityInconclusive": ["0" * 64],
            "optionalHardwareSkipped": [],
        },
        "categories": {
            "status": "passed",
            "categories": {
                "required-gpu": {
                    "discovered": 3,
                    "executed": 3,
                    "notRun": 0,
                    "outcomes": {"NotExecuted": 1, "Passed": 2},
                }
            },
        },
        "evidence": {
            "status": "passed",
            "records": {"device": 1, "dispatch": 4, "pipeline": 1, "readback": 2},
            "resources": {"device": 2, "host": 1},
            "pipelines": 1,
            "requiredCaseNamesExecuted": 3,
            "requiredCaseNamesWithEvidence": 2,
            "hostOnlyCaseNames": 1,
            "hostOnlyMethodsWithEvidence": [],
        },
        "negativeControls": {
            "status": "passed",
            "controls": [
                {
                    "fault": "omit-dispatch",
                    "status": "passed",
                    "methods": ["T.M"],
                    "cases": 2,
                    "failed": 2,
                    "sampleFailures": [{"name": "T.M", "message": "differs"}],
                }
            ],
        },
        "run": {
            "status": "passed",
            "errors": [],
            "execution": "hardware",
            "host": {"system": "Darwin", "machine": "arm64"},
            "preflightDevice": {"name": "GPU", "hardware": True, "fp64": False},
            "translator": {"commit": "c" * 40},
            "translation": {"translated": 1},
        },
        "arithmeticExactness": {
            "status": "passed",
            "tool": {"compiler": "test compiler"},
            "exactArtifacts": 1,
            "notCompiled": 0,
            "floatOperations": 12,
            "contractible": 0,
        },
        "intrinsicPrecision": {
            "status": "passed",
            "inputs": 3,
            "device": {"name": "GPU"},
            "referenceDevice": {"name": "WARP"},
            "functions": {
                "sin": {
                    "expression": "sin(x)",
                    "identical": 3,
                    "mismatches": 0,
                    "required": "bit-identical to WARP",
                    "ulpDistance": {"max": 0},
                    "examples": [],
                },
                "sqrt": {
                    "expression": "sqrt(x)",
                    "identical": 2,
                    "ulpDistance": {"max": 1},
                },
            },
        },
        "shaderImages": {
            "status": "passed",
            "errors": [],
            "casesExecuted": 2,
            "casesCompared": 2,
            "references": {
                "HelloWorld": {
                    "size": [8, 4],
                    "upstreamThreshold": 1e-5,
                    "directxUpstreamDelta": 0.0,
                    "directxVsReference": {"meanAbsoluteError": 0.0},
                }
            },
            "outputs": outputs,
            "cases": [{"case": "T.HelloWorld (Warp)", "captures": ["ab" * 32]}],
        },
        "processingImages": {"status": "passed", "errors": [], "methods": {}},
    }
    return {
        "schemaVersion": 1,
        "kind": "computesharp-port-qualification",
        "status": "passed",
        "upstreamCommit": "40dbe40de0b1eafb570e2af43f78bcdbffdcc00a",
        "contractSHA256": "1" * 64,
        "imageChecksSHA256": "2" * 64,
        "negativeControlsSHA256": "3" * 64,
        "provenance": {"workflowRun": "https://example.invalid/runs/1"},
        "profiles": {
            "directx-warp": {
                "profile": "directx-warp",
                "status": "passed",
                "device": {"name": "Microsoft Basic Render Driver", "hardware": False},
                "accounting": {"status": "passed", "execution": "software"},
            },
            "metal-apple-gpu": target,
            "opengl-mesa-llvmpipe": {
                "status": "failed",
                "errors": ["no opengl artifacts were supplied"],
            },
        },
    }


def test_summary_keeps_outcomes_devices_controls_precision_and_images():
    summary = SUMMARY.summarize(synthetic_report())

    assert summary["kind"] == "computesharp-port-results"
    assert summary["status"] == "passed"
    assert summary["provenance"] == {"workflowRun": "https://example.invalid/runs/1"}
    warp = summary["profiles"]["directx-warp"]
    assert warp["device"]["name"] == "Microsoft Basic Render Driver"
    assert warp["execution"] == "software"
    assert warp["checks"] == {"accounting": "passed"}

    metal = summary["profiles"]["metal-apple-gpu"]
    assert metal["device"] == {"name": "GPU", "hardware": True, "fp64": False}
    assert metal["execution"] == "hardware"
    assert metal["cases"]["capabilityGated"] == 1
    assert metal["cases"]["requiredGpuCasesPassed"] == 2
    assert metal["cases"]["categories"]["required-gpu"]["outcomes"]["Passed"] == 2
    assert metal["evidence"]["records"]["dispatch"] == 4
    assert metal["evidence"]["resources"] == {"device": 2, "host": 1}
    assert metal["evidence"]["requiredCaseNamesWithEvidence"] == 2
    assert metal["evidence"]["hostOnlyCaseNames"] == 1
    assert "hostOnlyMethodsWithEvidence" not in metal["evidence"]
    assert metal["negativeControls"]["faults"]["omit-dispatch"] == {
        "status": "passed",
        "methods": ["T.M"],
        "cases": 2,
        "failed": 2,
    }
    functions = metal["intrinsicPrecision"]["functions"]
    assert functions["sin"]["mismatches"] == 0
    assert functions["sqrt"] == {
        "expression": "sqrt(x)",
        "identical": 2,
        "ulpDistance": {"max": 1},
    }
    shader = metal["shaderImages"]["shaders"]["HelloWorld"]
    assert shader["size"] == [8, 4]
    assert shader["outputs"][0]["meanAbsoluteErrorVsDirectX"] == 0.01
    assert shader["outputs"][0]["meanAbsoluteErrorBound"] == 0.5
    assert "checks" in metal and metal["checks"]["shaderImages"] == "passed"
    assert metal["checks"]["arithmeticExactness"] == "passed"
    assert metal["arithmeticExactness"] == {
        "status": "passed",
        "tool": {"compiler": "test compiler"},
        "exactArtifacts": 1,
        "notCompiled": 0,
        "floatOperations": 12,
        "contractible": 0,
    }

    assert summary["profiles"]["opengl-mesa-llvmpipe"] == {
        "status": "failed",
        "errors": ["no opengl artifacts were supplied"],
        "execution": None,
        "checks": {},
        "cases": {
            "categories": None,
            "expected": None,
            "requiredCasesPassed": None,
            "requiredGpuCasesPassed": None,
            "capabilityGated": 0,
        },
    }


def test_command_line_overrides_the_provenance(tmp_path):
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps(synthetic_report()))
    output = tmp_path / "results.json"

    status = SUMMARY.main(
        [
            "--qualification",
            str(qualification),
            "--output",
            str(output),
            "--workflow-run",
            "https://example.invalid/runs/2",
            "--source-commit",
            "d" * 40,
        ]
    )

    assert status == 0
    written = json.loads(output.read_text())
    assert written["provenance"] == {
        "workflowRun": "https://example.invalid/runs/2",
        "sourceCommit": "d" * 40,
    }


def test_published_results_qualify_the_current_contract():
    results = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "coverage.lock.json").read_text(encoding="utf-8"))
    controls = json.loads((ROOT / "negative_controls.json").read_text())
    image_checks = json.loads((ROOT / "image_checks.json").read_text())

    assert results["kind"] == "computesharp-port-results"
    assert results["status"] == "passed"
    assert results["upstreamCommit"] == lock["commit"]
    assert results["contractSHA256"] == sha256(
        ROOT / "coverage.json", normalize_newlines=True
    )
    assert results["contractSHA256"] == lock["contractSHA256"]
    assert results["imageChecksSHA256"] == sha256(ROOT / "image_checks.json")
    assert results["negativeControlsSHA256"] == sha256(ROOT / "negative_controls.json")
    assert results["provenance"]["workflowRun"].startswith(
        "https://github.com/CrossGL/crosstl/actions/runs/"
    )
    assert len(results["provenance"]["sourceCommit"]) == 40

    assert set(results["profiles"]) == set(lock["profiles"])
    for name, frozen in lock["profiles"].items():
        profile = results["profiles"][name]
        assert profile["status"] == "passed", name
        assert set(profile["checks"].values()) == {"passed"}, name
        assert profile["execution"] == frozen["execution"], name
        assert profile["device"]["hardware"] is (frozen["execution"] == "hardware")
        cases = profile["cases"]
        assert cases["expected"] == frozen["expected"], name
        assert cases["requiredGpuCasesPassed"] == frozen["requiredGpuExpected"]["pass"]
        assert cases["capabilityGated"] == frozen["expected"]["inconclusive"], name
        for category, totals in lock["totals"].items():
            assert cases["categories"][category]["discovered"] == totals["cases"]
        required = cases["categories"]["required-gpu"]
        assert required["notRun"] == 0, name

    faults = {control["fault"] for control in controls["controls"]}
    shaders = set(image_checks["shaderImages"]["upstreamThresholds"])
    for name in TRANSLATED:
        profile = results["profiles"][name]
        assert profile["checks"].get("arithmeticExactness") == "passed", name
        exactness = profile.get("arithmeticExactness")
        assert isinstance(exactness, dict), name
        assert exactness.get("status") == "passed", name
        assert exactness.get("contractible") == 0, name
        assert exactness.get("floatOperations", 0) > 0, name
        assert exactness.get("exactArtifacts", 0) > 0, name
        assert exactness["exactArtifacts"] == profile["translation"]["translated"], name
        assert exactness.get("notCompiled") == profile["translation"].get(
            "compile-failed", 0
        ), name
        assert exactness.get("tool"), name
        assert profile["evidence"]["records"]["dispatch"] > 0
        assert set(profile["negativeControls"]["faults"]) == faults
        for fault in profile["negativeControls"]["faults"].values():
            assert fault["status"] == "passed"
            assert fault["cases"] == fault["failed"] > 0
        assert set(profile["shaderImages"]["shaders"]) == shaders
        for shader in profile["shaderImages"]["shaders"].values():
            assert shader["outputs"]
            for output in shader["outputs"]:
                assert (
                    output["meanAbsoluteErrorVsDirectX"]
                    <= output["meanAbsoluteErrorBound"]
                )
                assert (
                    output["largePixelFractionVsDirectX"]
                    <= output["largePixelFractionBound"]
                )
                assert output["upstreamDelta"] <= shader["upstreamThreshold"]
        for function in profile["intrinsicPrecision"]["functions"].values():
            if "required" in function:
                assert function["identical"] > 0


@pytest.mark.parametrize("profile_name", TRANSLATED)
@pytest.mark.parametrize(
    "defect",
    [
        "missing-check",
        "missing-report",
        "contractible",
        "empty",
        "artifacts",
        "not-compiled",
    ],
)
def test_published_results_reject_incomplete_arithmetic_evidence(
    tmp_path, monkeypatch, profile_name, defect
):
    results = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    profile = results["profiles"][profile_name]
    if defect == "missing-check":
        profile["checks"].pop("arithmeticExactness")
    elif defect == "missing-report":
        profile.pop("arithmeticExactness")
    elif defect == "contractible":
        profile["arithmeticExactness"]["contractible"] = 1
    elif defect == "empty":
        profile["arithmeticExactness"]["floatOperations"] = 0
    elif defect == "artifacts":
        profile["arithmeticExactness"]["exactArtifacts"] -= 1
    else:
        profile["arithmeticExactness"]["notCompiled"] += 1
    invalid = tmp_path / "results.json"
    invalid.write_text(json.dumps(results), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "PUBLISHED", invalid)
    with pytest.raises(AssertionError):
        test_published_results_qualify_the_current_contract()


SUITES = load_module("computesharp_results_suites", ROOT / "suites.py")


def _check_frozen_profiles(name, summary, lock):
    """Every frozen profile of ``lock`` passed with its frozen outcomes."""
    assert set(summary["profiles"]) == set(lock["profiles"]), name
    for profile_name, frozen in lock["profiles"].items():
        profile = summary["profiles"][profile_name]
        label = f"{name} {profile_name}"
        assert profile["status"] == "passed", label
        assert profile["checks"] and set(profile["checks"].values()) == {
            "passed"
        }, label
        assert profile["execution"] == frozen["execution"], label
        cases = profile["cases"]
        assert cases["expected"] == frozen["expected"], label
        assert (
            cases["requiredGpuCasesPassed"] == frozen["requiredGpuExpected"]["pass"]
        ), label
        assert cases["capabilityGated"] == frozen["expected"]["inconclusive"], label
        for category, totals in lock["totals"].items():
            if totals["cases"]:
                observed = cases["categories"][category]
                assert observed["discovered"] == totals["cases"], label
                assert observed["notRun"] == 0, label
            else:
                assert category not in cases["categories"], label


def test_published_results_cover_every_ported_companion_project():
    results = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    suites = SUITES.ported_suites()
    companions = {
        name: item for name, item in suites.items() if name != SUITES.MAIN_SUITE
    }
    assert companions
    assert set(results["suites"]) == set(companions)
    for name, item in companions.items():
        summary = results["suites"][name]
        lock = json.loads(item.lock.read_text(encoding="utf-8"))
        assert summary["status"] == "passed", name
        assert summary["kind"] == item.kind, name
        assert summary["contractSHA256"] == lock["contractSHA256"], name
        assert summary["contractSHA256"] == sha256(
            item.contract, normalize_newlines=True
        ), name
        assert summary["evidencePolicySHA256"] == sha256(item.evidence_policy), name
        assert summary["negativeControlsSHA256"] == sha256(item.negative_controls), name
        assert summary.get("noFloatArithmeticReason") == (
            item.no_float_arithmetic_reason
        ), name
        _check_frozen_profiles(name, summary, lock)
        controls = json.loads(item.negative_controls.read_text(encoding="utf-8"))
        faults = {control["fault"] for control in controls["controls"]}
        for profile_name in TRANSLATED:
            profile = summary["profiles"][profile_name]
            label = f"{name} {profile_name}"
            assert profile["device"]["hardware"] is (
                lock["profiles"][profile_name]["execution"] == "hardware"
            ), label
            assert profile["arithmeticExactness"]["status"] == "passed", label
            assert profile["arithmeticExactness"]["contractible"] == 0, label
            if item.no_float_arithmetic_reason is None:
                assert profile["arithmeticExactness"]["floatOperations"] > 0, label
            assert set(profile["negativeControls"]["faults"]) == faults, label
            for fault in profile["negativeControls"]["faults"].values():
                assert fault["status"] == "passed", label
                assert fault["cases"] == fault["failed"] > 0, label


def test_published_results_keep_retained_projects_out_of_translated_execution():
    results = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    retained = SUITES.retained_suites()
    assert retained
    assert set(results["retained"]) == set(retained)
    for name, item in retained.items():
        summary = results["retained"][name]
        lock = json.loads(item.lock.read_text(encoding="utf-8"))
        assert summary["status"] == "passed", name
        assert summary["scope"] == item.scope, name
        assert summary["countedAsTranslatedExecution"] is False, name
        assert summary["contractSHA256"] == lock["contractSHA256"], name
        assert set(lock["profiles"]) == {SUITES.RETAINED_PROFILE}, name
        _check_frozen_profiles(name, summary, lock)
        profile = summary["profiles"][SUITES.RETAINED_PROFILE]
        assert profile["cases"]["requiredGpuCasesPassed"] == 0, name
        assert set(profile["cases"]["categories"]) == {
            SUITES.RETAINED_SCOPES[item.scope]
        }, name
        assert isinstance(profile.get("selectedCases"), dict), name


@pytest.mark.parametrize(
    "defect", ["missing-suite", "failed-profile", "counted", "missing-retained"]
)
def test_published_results_reject_incomplete_project_results(
    tmp_path, monkeypatch, defect
):
    results = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    suite_name = sorted(results["suites"])[0]
    retained_name = sorted(results["retained"])[0]
    if defect == "missing-suite":
        results["suites"].pop(suite_name)
        check = test_published_results_cover_every_ported_companion_project
    elif defect == "failed-profile":
        profile = results["suites"][suite_name]["profiles"]["opengl-mesa-llvmpipe"]
        profile["cases"]["requiredGpuCasesPassed"] -= 1
        check = test_published_results_cover_every_ported_companion_project
    elif defect == "counted":
        results["retained"][retained_name]["countedAsTranslatedExecution"] = True
        check = (
            test_published_results_keep_retained_projects_out_of_translated_execution
        )
    else:
        results["retained"].pop(retained_name)
        check = (
            test_published_results_keep_retained_projects_out_of_translated_execution
        )
    invalid = tmp_path / "results.json"
    invalid.write_text(json.dumps(results), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "PUBLISHED", invalid)
    with pytest.raises(AssertionError):
        check()


def _readme_results_section():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("## Results\n")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def _table_rows(section, header):
    """Return the cells of the rows of the Markdown table under ``header``."""
    lines = section.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[index + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def test_readme_results_describe_the_published_run():
    """The Results section of the README cites the run of ``results.json``."""
    results = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    section = _readme_results_section()
    provenance = results["provenance"]
    run_id = provenance["workflowRun"].split("/runs/")[1].split("/")[0]
    assert f"[{run_id}](https://github.com/CrossGL/crosstl/actions/runs/" in section
    assert f"(commit `{provenance['sourceCommit']}`)" in section

    profiles = results["profiles"]
    for name in TRANSLATED:
        exactness = profiles[name]["arithmeticExactness"]
        sentence = (
            f"{exactness['exactArtifacts']} "
            f"{'Metal' if name.startswith('metal') else 'OpenGL'} artifacts with "
            f"{exactness['floatOperations']:,} floating-point operations"
        )
        assert sentence in " ".join(section.split()), name

    rows = {
        row[0].split()[0].strip("`"): row
        for row in _table_rows(section, "| Profile | Device |")
    }
    assert set(rows) == set(profiles)
    for name, profile in profiles.items():
        cases = profile["cases"]
        row = rows[name]
        assert row[2] == profile["execution"], name
        assert f"{cases['requiredGpuCasesPassed']:,}" in row[3], name
        assert f"({cases['requiredCasesPassed']:,} cases in total)" in row[3], name
        assert row[4] == str(cases["capabilityGated"]), name

    shaders = {
        row[0].strip("`"): row
        for row in _table_rows(section, "| Shader | WARP vs reference |")
    }
    metal = profiles["metal-apple-gpu"]["shaderImages"]["shaders"]
    opengl = profiles["opengl-mesa-llvmpipe"]["shaderImages"]["shaders"]
    assert set(shaders) == set(metal) == set(opengl)
    for shader, row in shaders.items():
        (metal_output,) = metal[shader]["outputs"]
        (opengl_output,) = opengl[shader]["outputs"]
        assert row[1:] == [
            f"{metal[shader]['directxMeanAbsoluteErrorVsReference']:.3f}",
            f"{metal_output['meanAbsoluteErrorVsDirectX']:.4f}",
            f"{opengl_output['meanAbsoluteErrorVsDirectX']:.4f}",
            f"{metal_output['meanAbsoluteErrorBound']:.3f}",
        ], shader


def _project_cell(profile):
    cases = profile["cases"]
    cell = f"{cases['requiredCasesPassed']:,} passed"
    if cases["optionalHardwareSkipped"]:
        cell += f", {cases['optionalHardwareSkipped']:,} optional"
    if cases["capabilityGated"]:
        cell += f", {cases['capabilityGated']:,} capability-gated"
    return cell


def test_readme_results_describe_every_companion_and_retained_project():
    """The project table of the Results section matches ``results.json``."""
    results = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    section = _readme_results_section()
    header = "| Project | Scope | `directx-warp` |"
    rows = {row[0].strip("`"): row for row in _table_rows(section, header)}
    projects = {
        **{name: ("ported", item) for name, item in results["suites"].items()},
        **{name: (item["scope"], item) for name, item in results["retained"].items()},
    }
    assert set(rows) == {name.removeprefix("ComputeSharp.Tests.") for name in projects}
    profiles = ("directx-warp", *TRANSLATED)
    for name, (scope, summary) in projects.items():
        row = rows[name.removeprefix("ComputeSharp.Tests.")]
        assert row[1] == scope, name
        expected = [
            (
                _project_cell(summary["profiles"][profile])
                if profile in summary["profiles"]
                else "not run"
            )
            for profile in profiles
        ]
        assert row[2:] == expected, name


def test_readme_results_state_what_the_upstream_device_lost_steps_selected():
    """The README claims the verbatim upstream DeviceLost steps select no case."""
    results = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    profile = results["retained"]["ComputeSharp.Tests.DeviceLost"]["profiles"][
        SUITES.RETAINED_PROFILE
    ]
    upstream = {
        step: count
        for step, count in profile["selectedCases"].items()
        if step.startswith("upstream-")
    }
    assert len(upstream) == 3
    accounted = sum(
        count
        for step, count in profile["selectedCases"].items()
        if step not in upstream
    )
    assert accounted == profile["cases"]["categories"]["windows-specific"]["discovered"]
    section = " ".join(_readme_results_section().split())
    claim = "the three upstream DeviceLost steps selected no case"
    assert (claim in section) is (set(upstream.values()) == {0})
