"""Execute pinned MLX backward entries with upstream dispatch geometry."""

import hashlib
import importlib
import json
import os
import struct
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from demos.integrations.mlx.run_mlx_metal_host import run
from tests.test_translator.test_mlx_gated_delta_metal import _translate_pinned

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_GATED_DELTA_RUNTIME"
ROOT = Path(__file__).resolve().parents[2]
GUARD = struct.pack("<I", 0x6A15BEEF) * 32
RTOL = ATOL = 1e-5
CONFIGURATIONS = (
    (1, 24, 24, 1, 1, "float32", True),
    (1, 24, 24, 1, 5, "float32", False),
    (2, 16, 32, 4, 5, "float32", False),
    (1, 16, 48, 8, 9, "float32", False),
    (1, 16, 64, 16, 17, "float32", False),
    (1, 32, 32, 4, 3, "float32", False),
    (1, 16, 16, 8, 8, "float32", False),
    (1, 16, 32, 4, 5, "float16", False),
)


@pytest.fixture(scope="module")
def reference():
    if os.environ.get(REQUIRE_ENV) == "1":
        importlib.import_module("numpy")
    else:
        pytest.importorskip("numpy")
    return importlib.import_module(
        "tests.fixtures.runtime_verification.mlx_gated_delta_reference"
    )


@pytest.fixture(scope="session")
def metal_runner(tmp_path_factory):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned backward execution")
    assert sys.platform == "darwin", "Native Metal execution requires macOS"
    directory = tmp_path_factory.mktemp("metal-raw-buffer-runner")
    executable = directory / "readback"
    run(
        [
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            executable,
        ],
        directory,
        "compile",
    )
    return executable


def _compile(artifact, root, directory):
    air, module = directory / "kernel.air", directory / "kernel.metallib"
    run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "metal",
            "-Werror",
            "-I",
            root,
            "-c",
            artifact,
            "-o",
            air,
        ],
        directory,
        "compile",
    )
    run(["xcrun", "--sdk", "macosx", "metallib", air, "-o", module], directory, "link")
    assert module.stat().st_size > 0
    return module


def _check_readbacks(np, directory, buffers, gradients, *, exact=False):
    records, outputs = [], []
    for slot, buffer in enumerate(buffers):
        data = (directory / f"buffer-{slot}.bin").read_bytes()
        assert len(data) == buffer.nbytes + len(GUARD), (slot, "size")
        assert data[buffer.nbytes :] == GUARD, (slot, "guard")
        payload = data[: buffer.nbytes]
        record = {
            "slot": slot,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if slot < 9:
            assert payload == buffer.tobytes(), (slot, "input modified")
        else:
            reference = gradients[slot - 9]
            actual = np.frombuffer(payload, dtype="<f4").reshape(reference.shape)
            assert np.isfinite(actual).all(), (slot, "nonfinite output")
            if exact:
                assert payload == reference.astype("<f4").tobytes(), slot
            else:
                np.testing.assert_allclose(
                    actual, reference, rtol=RTOL, atol=ATOL, err_msg=f"buffer {slot}"
                )
            record.update(
                values=actual.size,
                maximumAbsoluteError=float(
                    np.max(np.abs(actual.astype("float64") - reference))
                ),
            )
            outputs.append(actual)
        records.append(record)
    return records, outputs


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_pinned_gated_delta_gradients(tmp_path, metal_runner, reference, configuration):
    np = importlib.import_module("numpy")
    case = reference.Case(*configuration)
    revision, source, generated = _translate_pinned(
        tmp_path, "metal", entry=case.entry, workgroup_size=(32, 4, 1)
    )
    _, _, buffers, gradients, _, _ = reference.dataset(case)
    if case.uniform:
        for gradient, value in zip(
            gradients, (35 / 1024, 5 / 256, 5 / 512, 155 / 64, 15 / 32, 155 / 32768)
        ):
            np.testing.assert_array_equal(gradient, value)
    paths = []
    for slot, buffer in enumerate(buffers):
        path = tmp_path / f"input-{slot}.bin"
        path.write_bytes(buffer.tobytes() + GUARD)
        paths.append(str(path))
    for slot, gradient in enumerate(gradients, 9):
        (tmp_path / f"expected-{slot}.f64").write_bytes(
            gradient.astype("<f8").tobytes()
        )
    request = {
        "buffers": paths,
        "workgroupCount": [1, 32, case.batch * case.value_heads],
        "workgroupSize": [32, 4, 1],
        "simdWidth": 32,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    evidence = {
        "commit": revision,
        "case": asdict(case),
        "entry": case.entry,
        "rtol": 0 if case.uniform else RTOL,
        "atol": 0 if case.uniform else ATOL,
        "fullUpstreamSuite": False,
        "fullTranslatedBackend": False,
        "results": {},
    }
    paired = {}
    for label, artifact in (("original", source), ("generated", generated)):
        directory = tmp_path / label
        module = _compile(artifact, source.parents[4], directory)
        run(
            [metal_runner, module, case.entry, request_path, directory],
            directory,
            "dispatch",
            timeout=60,
        )
        records, paired[label] = _check_readbacks(
            np, directory, buffers, gradients, exact=case.uniform
        )
        evidence["results"][label] = {
            "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "runtime": json.loads((directory / "dispatch.stdout").read_text()),
            "buffers": records,
        }
        (tmp_path / "evidence.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
    for original, translated in zip(paired["original"], paired["generated"]):
        np.testing.assert_allclose(translated, original, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "length,checkpoint,key_heads,value_heads", [(1, 1, 24, 24), (5, 4, 16, 32)]
)
def test_backward_reference_matches_forward_finite_differences(
    reference, length, checkpoint, key_heads, value_heads
):
    np = importlib.import_module("numpy")
    case = reference.Case(1, key_heads, value_heads, checkpoint, length)
    parameters, cotangents, _buffers, gradients, _, _ = reference.dataset(case)
    parameters = [value.astype("float64") for value in parameters]
    rng = np.random.default_rng(817 + length)

    def loss():
        output, final, _, _ = reference.forward(parameters, checkpoint)
        return float((output * cotangents[0]).sum() + (final * cotangents[1]).sum())

    for parameter, gradient in zip(parameters, gradients):
        for index in rng.choice(parameter.size, size=4, replace=False):
            value, epsilon = parameter.flat[index], 1e-5
            parameter.flat[index] = value + epsilon
            above = loss()
            parameter.flat[index] = value - epsilon
            below = loss()
            parameter.flat[index] = value
            observed = (above - below) / (2 * epsilon)
            assert abs(observed - gradient.flat[index]) <= ATOL + RTOL * abs(observed)


@pytest.mark.parametrize("corruption", ["size", "guard", "input", "output", "nan"])
def test_readback_verifier_rejects_corruption(tmp_path, reference, corruption):
    np = importlib.import_module("numpy")
    buffers = [np.zeros(1, dtype="<f4") for _ in range(15)]
    gradients = [np.zeros(1) for _ in range(6)]
    for slot, buffer in enumerate(buffers):
        (tmp_path / f"buffer-{slot}.bin").write_bytes(buffer.tobytes() + GUARD)
    slot = 0 if corruption == "input" else 9
    path = tmp_path / f"buffer-{slot}.bin"
    data = path.read_bytes()
    if corruption == "size":
        data = data[:-4]
    elif corruption == "guard":
        data = data[:-4] + b"\x00" * 4
    else:
        data = (
            struct.pack("<f", float("nan") if corruption == "nan" else 1.0) + data[4:]
        )
    path.write_bytes(data)
    with pytest.raises(AssertionError):
        _check_readbacks(np, tmp_path, buffers, gradients)
