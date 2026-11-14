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
        model = PRECISION.ARITHMETIC_MODELS.get(name)
        functions.append(
            {
                "name": name,
                "expression": model[0] if model else f"{name}(x)",
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


def reference_values(evaluation=0, inputs=INPUTS):
    """WARP-like results; ``evaluation=1`` keeps IEEE subnormals instead."""
    with np.errstate(all="ignore"):
        operands = inputs.astype(np.float64)
        values = {
            "sin": np.sin(operands).astype(np.float32),
            "cos": np.cos(operands).astype(np.float32),
            "tan": np.tan(operands).astype(np.float32),
            "exp": np.exp(np.minimum(operands, 80)).astype(np.float32),
        }
    for name in PRECISION.ARITHMETIC_MODELS:
        values[name] = PRECISION.arithmetic_model(name, inputs)[evaluation]
    return values


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


def test_ieee_subnormals_in_arithmetic_are_allowed(tmp_path):
    reference = write_probe(tmp_path / "warp", reference_values())
    candidate = write_probe(tmp_path / "opengl", reference_values(1), device="GPU")

    report = PRECISION.compare(reference, candidate)

    assert report["status"] == "passed", report.get("errors")
    dot2 = report["functions"]["dot2"]
    # 1e-38 is subnormal: flushing it changes the WARP result.
    assert dot2["subnormalInputs"] >= 1
    assert dot2["identical"] < INPUTS.size
    assert dot2["mismatches"] == 0


def test_contracted_arithmetic_fails(tmp_path):
    # y cancels the rounded product: x * 27.619 + y is 0 when the product is
    # rounded and the product's rounding error when the multiply-add is fused.
    x = np.float32(1.1)
    inputs = np.array([x, -(x * np.float32(27.619)), 0.0], dtype=np.float32)
    values = reference_values(inputs=inputs)
    assert values["multiply-add"][0] == 0
    reference = write_probe(tmp_path / "warp", values, inputs=inputs)
    changed = dict(values)
    operands = inputs.astype(np.float64)
    fused = operands * np.float64(np.float32(27.619)) + np.roll(operands, -1)
    changed["multiply-add"] = fused.astype(np.float32)
    assert changed["multiply-add"][0] != 0
    candidate = write_probe(tmp_path / "metal", changed, inputs=inputs, device="GPU")

    report = PRECISION.compare(reference, candidate)

    assert report["status"] == "failed"
    assert report["functions"]["multiply-add"]["mismatches"] >= 1
    assert report["functions"]["multiply-add"]["examples"][0]["input"] == x
    assert any("beyond subnormal flushing" in error for error in report["errors"])


def test_one_ulp_in_arithmetic_fails(tmp_path):
    values = reference_values()
    reference = write_probe(tmp_path / "warp", values)
    changed = dict(values)
    changed["dot3"] = values["dot3"].copy()
    changed["dot3"][2] = np.nextafter(changed["dot3"][2], np.float32(np.inf))
    candidate = write_probe(tmp_path / "metal", changed, device="GPU")

    report = PRECISION.compare(reference, candidate)

    assert report["status"] == "failed"
    assert report["functions"]["dot3"]["mismatches"] == 1
    assert report["functions"]["dot3"]["examples"][0]["input"] == pytest.approx(1.0)


def test_warp_results_must_match_the_model(tmp_path):
    values = reference_values()
    values["lerp"] = values["lerp"].copy()
    values["lerp"][2] = np.nextafter(values["lerp"][2], np.float32(np.inf))
    reference = write_probe(tmp_path / "warp", values)
    candidate = write_probe(tmp_path / "metal", values, device="GPU")

    report = PRECISION.compare(reference, candidate)

    assert report["status"] == "failed"
    assert report["functions"]["lerp"]["modelMismatches"] == 1
    assert any("single-precision model" in error for error in report["errors"])


def test_missing_or_different_arithmetic_probes_fail(tmp_path):
    values = reference_values()
    del values["smoothstep"]
    reference = write_probe(tmp_path / "warp", values)
    candidate = write_probe(tmp_path / "metal", values, device="GPU")
    report = PRECISION.compare(reference, candidate)
    assert report["status"] == "failed"
    assert "required probes are missing: ['smoothstep']" in report["errors"][0]

    values = reference_values()
    reference = write_probe(tmp_path / "warp2", values)
    candidate = write_probe(tmp_path / "metal2", values, device="GPU")
    meta_path = candidate / "probe.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for function in meta["functions"]:
        if function["name"] == "mad":
            function["expression"] = "mad(x, 2.0, y)"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    report = PRECISION.compare(reference, candidate)
    assert report["status"] == "failed"
    assert any("the probe evaluates 'mad(x, 2.0, y)'" in e for e in report["errors"])
