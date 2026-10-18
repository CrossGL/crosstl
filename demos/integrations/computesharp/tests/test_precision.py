"""Check the comparison of translated intrinsic results with WARP.

Each test writes two probe result directories in the layout of
``precision/Program.cs`` and introduces one defect the comparison must reject.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT, load_module  # noqa: E402

PRECISION = load_module("computesharp_precision_report", ROOT / "precision_report.py")
INPUTS = np.array(
    [0.0, -0.0, 1.0, -2.5, 3141.0, 43758.547, 1e-38, 1.17549435e-38, np.inf, np.nan],
    dtype=np.float32,
)


def write_probe(directory: Path, values: dict, inputs=INPUTS, hlsl=None, device="WARP"):
    directory.mkdir(parents=True)
    inputs_bytes = inputs.astype("<f4").tobytes()
    (directory / "inputs.bin").write_bytes(inputs_bytes)
    functions = []
    for name, array in values.items():
        data = np.asarray(array, dtype="<f4").tobytes()
        (directory / f"{name}.bin").write_bytes(data)
        functions.append(
            {
                "name": name,
                "expression": f"{name}(x)",
                "shader": f"Probe.{name}",
                "hlslSha256": (hlsl or {}).get(name, "hlsl-" + name),
                "file": f"{name}.bin",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    meta = {
        "schemaVersion": 1,
        "kind": "computesharp.intrinsic-probe",
        "device": {"name": device, "hardware": device != "WARP"},
        "inputs": {
            "file": "inputs.bin",
            "count": int(inputs.size),
            "sha256": hashlib.sha256(inputs_bytes).hexdigest(),
            "sets": [],
        },
        "functions": functions,
    }
    (directory / "probe.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


def reference_values():
    with np.errstate(all="ignore"):
        sine = np.sin(INPUTS.astype(np.float64)).astype(np.float32)
        exp = np.exp(np.minimum(INPUTS, 80).astype(np.float64)).astype(np.float32)
    return {"sin": sine, "exp": exp}


def test_identical_results_pass(tmp_path):
    reference = write_probe(tmp_path / "warp", reference_values())
    candidate = write_probe(tmp_path / "metal", reference_values(), device="GPU")

    report = PRECISION.compare(reference, candidate)

    assert report["status"] == "passed"
    assert report["functions"]["sin"]["mismatches"] == 0
    assert report["functions"]["exp"]["identical"] == INPUTS.size


def test_one_ulp_in_an_exact_function_fails(tmp_path):
    values = reference_values()
    reference = write_probe(tmp_path / "warp", values)
    changed = dict(values)
    changed["sin"] = values["sin"].copy()
    changed["sin"][4] = np.nextafter(changed["sin"][4], np.float32(2))
    candidate = write_probe(tmp_path / "metal", changed, device="GPU")

    report = PRECISION.compare(reference, candidate)

    assert report["status"] == "failed"
    assert report["functions"]["sin"]["mismatches"] == 1
    assert report["functions"]["sin"]["examples"][0]["input"] == pytest.approx(3141.0)


def test_native_functions_are_reported_without_failing(tmp_path):
    values = reference_values()
    reference = write_probe(tmp_path / "warp", values)
    changed = dict(values)
    changed["exp"] = np.nextafter(values["exp"], np.float32(np.inf))
    candidate = write_probe(tmp_path / "metal", changed, device="GPU")

    report = PRECISION.compare(reference, candidate)

    assert report["status"] == "passed"
    assert report["functions"]["exp"]["ulpDistance"]["max"] >= 1


def test_operands_with_a_subnormal_revolution_are_exempt(tmp_path):
    values = reference_values()
    reference = write_probe(tmp_path / "warp", values)
    changed = dict(values)
    changed["sin"] = values["sin"].copy()
    changed["sin"][6] = np.float32(0.0)  # 1e-38 / (2 * pi) is subnormal
    candidate = write_probe(tmp_path / "metal", changed, device="GPU")

    report = PRECISION.compare(reference, candidate)

    assert report["status"] == "passed"
    assert report["functions"]["sin"]["requiredInputs"] == INPUTS.size - 2


def test_different_inputs_or_hlsl_fail(tmp_path):
    reference = write_probe(tmp_path / "warp", reference_values())
    other_inputs = INPUTS.copy()
    other_inputs[2] = np.float32(1.5)
    candidate = write_probe(
        tmp_path / "metal", reference_values(), inputs=other_inputs, device="GPU"
    )
    assert "inputs differ" in PRECISION.compare(reference, candidate)["errors"][0]

    candidate = write_probe(
        tmp_path / "opengl", reference_values(), hlsl={"sin": "other"}, device="GPU"
    )
    report = PRECISION.compare(reference, candidate)
    assert report["status"] == "failed"
    assert "generated HLSL differs" in report["errors"][0]


def test_tampered_result_files_are_rejected(tmp_path):
    reference = write_probe(tmp_path / "warp", reference_values())
    candidate = write_probe(tmp_path / "metal", reference_values(), device="GPU")
    (candidate / "sin.bin").write_bytes(b"\0" * (4 * INPUTS.size))

    with pytest.raises(ValueError, match="does not match probe.json"):
        PRECISION.compare(reference, candidate)
