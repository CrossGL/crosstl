"""Execute pinned MLX trigonometry through project packaging and native loading."""

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.test_current_unary_fma import (
    _run_current_unary,
)
from tests import workflow_text
from tests.ci_helpers import assert_native_step_prerequisites
from tests.test_translator.test_metal_precise_asin import _inputs as _asin_inputs
from tests.test_translator.test_metal_precise_trig import (
    _bits,
    _float,
    _inputs,
    _oracle,
)

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_TRIG"


def _check_inverse_word(actual, expected):
    magnitude = expected & 0x7FFFFFFF
    if magnitude > 0x7F800000:
        assert actual & 0x7FFFFFFF > 0x7F800000, "NaN classification"
        return 0
    assert actual & 0x7FFFFFFF < 0x7F800000, "finite result"
    assert actual >> 31 == expected >> 31, "result sign"
    if magnitude == 0:
        assert actual == expected, "zero sign"
    error = abs(actual - expected)
    assert error <= 4, (actual, expected, error)
    return error


@pytest.mark.parametrize(
    "actual,expected",
    [
        (0, 0x80000000),
        (0x80000000, 0),
        (0, 0x7FC00000),
        (0x7FC00000, 0x3F800000),
        (0x7F800000, 0x3F800000),
        (0xBF800000, 0x3F800000),
        (0x3F800005, 0x3F800000),
    ],
)
def test_inverse_trig_verifier_rejects_corruption(actual, expected):
    with pytest.raises(AssertionError):
        _check_inverse_word(actual, expected)


def test_inverse_trig_inputs_cover_domain_and_boundaries():
    words = {_bits(value) for value in _asin_inputs()}
    assert {0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000} <= words
    for boundary in (0x39800000, 0x3F000000, 0x3F800000):
        assert {
            word | sign
            for word in range(boundary - 4, boundary + 5)
            for sign in (0, 0x80000000)
        } <= words
    assert {_bits(index / 2048) for index in range(-2048, 2049)} <= words


@pytest.mark.parametrize("operation", ["ArcSin", "ArcCos"])
def test_current_mlx_inverse_trig_native_loader(tmp_path, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned trigonometry execution")
    values = _asin_inputs()
    reference = math.asin if operation == "ArcSin" else math.acos
    expected = [
        _bits(math.nan if math.isnan(value) or abs(value) > 1 else reference(value))
        for value in values
    ]
    actual = _run_current_unary(
        tmp_path,
        operation,
        values,
        [_float(word) for word in expected],
        0,
        1e-6,
        f"metal_precise_{'asin' if operation == 'ArcSin' else 'acos'}_float",
    )
    actual_words = [_bits(value) for value in actual]
    maximum = max(
        _check_inverse_word(got, want)
        for got, want in zip(actual_words, expected, strict=True)
    )
    evidence = tmp_path / "evidence"
    parity = json.loads((evidence / "parity.json").read_text())
    (packaged,) = json.loads((evidence / "package/runtime-package.json").read_text())[
        "artifacts"
    ]
    source = evidence / "package" / packaged["packagePath"]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    for name, key in (("report.json", "generatedHash"), ("artifacts.json", "hash")):
        (artifact,) = json.loads((evidence / name).read_text())["artifacts"]
        assert artifact["target"] == packaged["target"] == parity["target"]
        assert artifact[key]["value"] == packaged["hash"]["value"] == source_hash
    identity = parity["runtime"]["artifactIdentityVerification"]
    assert identity["verificationStatus"] == "verified"
    assert identity["observedIdentity"] == identity["expectedIdentity"]
    assert identity["observedIdentity"]["hash"]["value"] == source_hash
    (evidence / "inverse-trig-audit.json").write_text(
        json.dumps(
            {
                "operation": operation,
                "commit": parity["commit"],
                "target": parity["target"],
                "generatedSourceSHA256": source_hash,
                "values": len(values),
                "inputWords": [_bits(value) for value in values],
                "expectedWords": expected,
                "actualWords": actual_words,
                "maxUlpError": maximum,
                "maximumAllowedUlps": 4,
                "oracle": "host binary64 inverse trigonometry rounded to binary32",
                "nanPayloadsPreserved": False,
                "guardRegionsChecked": False,
                "inputDeviceReadbacksRetained": False,
                "fullBackendParity": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("operation", ["Sin", "Cos"])
def test_current_mlx_precise_trig_native_loader(tmp_path, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned trigonometry execution")
    words = _inputs()
    index = 0 if operation == "Sin" else 1
    expected = [_oracle(word)[index] for word in words]
    actual = _run_current_unary(
        tmp_path,
        operation,
        [_float(word) for word in words],
        [_float(word) for word in expected],
        0,
        1e-6,
        f"metal_precise_{operation.lower()}_float",
    )
    maximum = 0
    for got, want in zip(actual, expected):
        actual_word = _bits(got)
        if want & 0x7FFFFFFF > 0x7F800000:
            assert actual_word & 0x7FFFFFFF > 0x7F800000, "NaN classification"
            continue
        assert actual_word >> 31 == want >> 31, "result sign"
        if want & 0x7FFFFFFF == 0:
            assert actual_word == want, "zero sign"
        error = abs(actual_word - want)
        assert error <= 4, (got, _float(want), error)
        maximum = max(maximum, error)
    (tmp_path / "ulp-audit.json").write_text(
        json.dumps(
            {
                "operation": operation,
                "values": len(words),
                "maxUlpError": maximum,
                "oracle": "160-digit Decimal, correctly rounded binary32",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("job_id", ["portable-host", "mlx-metal-porting"])
def test_ci_requires_pinned_trigonometry(job_id):
    text = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = (
        workflow_text.workflow_job_text(text, job_id)
        .split("      - name: Validate pinned unary arithmetic\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    assert_native_step_prerequisites(step, job=job_id, pinned=True)
    assert "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "demos/integrations/mlx/tests/kernels/test_current_unary_trig.py" in step
    assert "--timeout-seconds 300" in step
    assert "--basetemp=" in step and "--junitxml=" in step
