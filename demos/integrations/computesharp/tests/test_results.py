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
