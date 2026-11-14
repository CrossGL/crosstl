"""Original-source Metal parity for complex magnitude and sign layouts."""

import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    load_project_config,
    reflect_target_host_interface,
    translate_project,
)
from demos.integrations.mlx.tests.kernels import test_unary_native_loader as unary
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_logaddexp import (
    _assert_same_words,
    _pack,
    _words,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from tests import workflow_text

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_COMPLEX_MAGNITUDE"
SHAPES = ("v", "v2", "gn1", "gn4large")
GUARDS = (0x35555555, 0xB5555555) * 8
SENTINELS = (0x4EAA5555, 0x4D55AAAA)


def _samples():
    special = (
        0,
        0x80000000,
        1,
        0x7FFFFF,
        0x800000,
        0x33800000,
        0x3F000000,
        0x3F800000,
        0x3F800001,
        0x40000000,
        0x40400000,
        0x5EFFFFFF,
        0x5F000000,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0xFFC00001,
    )
    values = [(a, b) for a in special for b in special]
    generator = random.Random(3400)
    values.extend(
        (generator.getrandbits(32), generator.getrandbits(32)) for _ in range(4096)
    )
    return values


def _workload(shape):
    values = _samples()
    if shape.startswith("gn"):
        indices = [
            z * 90 + y * 2 + x * 11
            for z in range(3)
            for y in range(5)
            for x in range(7)
        ]
        constants = {
            "in_shape": struct.pack("<3i", 3, 5, 7),
            "in_strides": struct.pack("<3q", 90, 2, 11),
            "ndim": struct.pack("<i", 3),
        }
        grid = [7 if shape == "gn1" else 2, 5, 3]
    else:
        indices = list(range(len(values)))
        constants = {"size": struct.pack("<I" if shape == "v" else "<q", len(values))}
        assert len(values) % 13 == 0
        grid = [len(values), 1, 1] if shape == "v" else [13, len(values) // 13, 1]
    return values, indices, constants, grid


def _check_readback(payloads, readbacks, sentinel):
    assert set(readbacks) == set(payloads)
    for name, payload in payloads.items():
        assert len(readbacks[name]) == len(payload)
        if name != "out_" and readbacks[name] != payload:
            # Keep exact byte equality without pytest expanding whole buffers.
            raise AssertionError(f"Read-only buffer changed: {name}")
    words = _words(readbacks["out_"])
    assert words[-len(GUARDS) :] == GUARDS
    actual = words[: -len(GUARDS)]
    assert sentinel not in actual, "Output was not written"
    return actual


def _execute(runner, library, entry, directory, payloads, grid, resources, sentinel):
    directory.mkdir(parents=True)
    assert [row["binding"] for row in resources] == list(range(len(resources)))
    assert {row["name"] for row in resources} == set(payloads)
    buffers = []
    for row in resources:
        path = directory / f"input-{row['binding']}.bin"
        path.write_bytes(payloads[row["name"]])
        buffers.append(str(path))
    request = directory / "request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": buffers,
                "workgroupCount": grid,
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    _run([runner, library, entry, request, directory], directory, "execute")
    readbacks = {
        row["name"]: (directory / f"buffer-{row['binding']}.bin").read_bytes()
        for row in resources
    }
    return _check_readback(payloads, readbacks, sentinel)


@pytest.mark.parametrize("shape", SHAPES)
def test_complex_magnitude_workloads_cover_layouts_and_boundaries(shape):
    values, indices, constants, grid = _workload(shape)
    assert len(values) == 4420 and values == _samples()
    assert len(indices) == (105 if shape.startswith("gn") else 4420)
    assert len(set(indices)) == len(indices)
    assert min(indices) == 0 and max(indices) < len(values)
    assert {
        0,
        0x80000000,
        1,
        0x7FFFFF,
        0x800000,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
    } <= {word for pair in values for word in pair}
    assert not set(SENTINELS) & {word for pair in values for word in pair}
    assert len(GUARDS) == 16
    if shape.startswith("gn"):
        assert struct.unpack("<3q", constants["in_strides"]) == (90, 2, 11)
        assert grid == ([7, 5, 3] if shape == "gn1" else [2, 5, 3])
    else:
        assert grid[0] * grid[1] == len(indices)


@pytest.mark.parametrize("sentinel", SENTINELS)
@pytest.mark.parametrize(
    "mutation", (None, "input", "guard", "truncated", "missing", "unwritten")
)
def test_complex_magnitude_readbacks_reject_corruption(sentinel, mutation):
    payloads = {"in_": _pack([1, 2]), "out_": _pack([3, 4] + list(GUARDS))}
    readbacks = dict(payloads)
    if mutation == "input":
        readbacks["in_"] = _pack([0, 2])
    elif mutation == "guard":
        readbacks["out_"] = readbacks["out_"][:-4] + _pack([0])
    elif mutation == "truncated":
        readbacks["out_"] = readbacks["out_"][:-4]
    elif mutation == "missing":
        del readbacks["in_"]
    elif mutation == "unwritten":
        readbacks["out_"] = _pack([sentinel, 4] + list(GUARDS))
    if mutation:
        with pytest.raises(AssertionError) as error:
            _check_readback(payloads, readbacks, sentinel)
        if mutation == "input":
            assert str(error.value) == "Read-only buffer changed: in_"
    else:
        assert _check_readback(payloads, readbacks, sentinel) == (3, 4)


def test_ci_requires_complex_magnitude_source_parity():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = (
        workflow_text.workflow_job_text(workflow, "mlx-metal-porting")
        .split("      - name: Validate pinned unary arithmetic\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert (
        "demos/integrations/mlx/tests/kernels/test_current_complex_magnitude.py" in step
    )
    assert "--timeout-seconds 300" in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step


@pytest.fixture(scope="module")
def magnitude_source():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for complex magnitude execution")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    assert target in {"metal", "directx", "opengl"}, "Missing or invalid target"
    if target != "metal":
        pytest.skip("Exact original-source comparison requires native Metal")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "mlx/backend/metal/kernels"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert (
        hashlib.sha256((root / unary.MLX_UNARY_SOURCE).read_bytes()).hexdigest()
        == unary.MLX_UNARY_SHA256
    )
    return root


@pytest.mark.parametrize("target", (None, "unknown"))
def test_complex_magnitude_requires_valid_target(monkeypatch, target):
    monkeypatch.setenv(REQUIRE_ENV, "1")
    if target is None:
        monkeypatch.delenv("CROSTL_MLX_CURRENT_TARGET", raising=False)
    else:
        monkeypatch.setenv("CROSTL_MLX_CURRENT_TARGET", target)
    with pytest.raises(AssertionError, match="Missing or invalid target"):
        magnitude_source.__wrapped__()


@pytest.fixture(scope="module")
def magnitude_reference(magnitude_source, tmp_path_factory):
    work = tmp_path_factory.mktemp("complex-magnitude-source")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    original = _metal_library(
        magnitude_source / unary.MLX_UNARY_SOURCE,
        work / "original.metallib",
        magnitude_source,
        upstream=True,
    )
    return runner, original


@pytest.mark.parametrize("operation", ("Abs", "Sign"))
def test_current_complex_magnitude_native_parity(
    magnitude_source, magnitude_reference, tmp_path, operation
):
    root = magnitude_source
    runner, original = magnitude_reference
    entries = [f"{shape}_{operation}complex64complex64" for shape in SHAPES]
    workloads = {row.entry_point: row for row in unary.UNARY_METAL_WORKLOADS}
    with tempfile.TemporaryDirectory(
        prefix=".current-complex-magnitude-", dir=root
    ) as path:
        work = Path(path)
        try:
            config_path = work / "crosstl.toml"
            config_path.write_text(
                unary._project_config("metal", workloads[entries[0]]), encoding="utf-8"
            )
            config = replace(
                load_project_config(root, config_path),
                entry_points={unary.MLX_UNARY_SOURCE: entries},
                entry_workgroup_size_rules={unary.MLX_UNARY_SOURCE: {"*": (1, 1, 1)}},
            )
            report = translate_project(
                config, output_dir=work.name + "/out", format_output=False
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            assert (
                payload["summary"]["translatedCount"] == 4
                and payload["summary"]["failedCount"] == 0
            ), payload["diagnostics"]
            assert not [
                row for row in payload["diagnostics"] if row["severity"] == "error"
            ]
            artifacts = {
                row["entryPoint"]["source"]: row for row in payload["artifacts"]
            }
            assert len(payload["artifacts"]) == 4 and set(artifacts) == set(entries)
            for shape, entry in zip(SHAPES, entries):
                directory = work / entry
                directory.mkdir()
                artifact = artifacts[entry]
                source = root / artifact["path"]
                assert (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    == artifact["generatedHash"]["value"]
                )
                assert artifact["sourceHash"]["value"] == unary.MLX_UNARY_SHA256
                assert artifact["templateMaterialization"][
                    "specializations"
                ] == unary._expected_unary_materializations(workloads[entry])
                interface = reflect_target_host_interface(source, target="metal")
                assert interface["status"] == "ready" and not interface["diagnostics"]
                resources = sorted(
                    interface["resources"], key=lambda row: row["binding"]
                )
                generated = _metal_library(
                    source, directory / "generated.metallib", root
                )
                values, indices, constants, grid = _workload(shape)
                previous = None
                for index, sentinel in enumerate(SENTINELS):
                    payloads = {
                        "in_": _pack(
                            [word for value in values for word in value] + list(GUARDS)
                        ),
                        "out_": _pack([sentinel] * (2 * len(indices)) + list(GUARDS)),
                        **constants,
                    }
                    expected = _execute(
                        runner,
                        original,
                        entry,
                        directory / str(index) / "original",
                        payloads,
                        grid,
                        resources,
                        sentinel,
                    )
                    actual = _execute(
                        runner,
                        generated,
                        artifact["entryPoint"]["target"],
                        directory / str(index) / "translated",
                        payloads,
                        grid,
                        resources,
                        sentinel,
                    )
                    _assert_same_words(actual, expected)
                    if previous is not None:
                        _assert_same_words(actual, previous)
                    previous = actual
                (directory / "evidence.json").write_text(
                    json.dumps(
                        {
                            "commit": MLX_COMMIT,
                            "entryPoint": entry,
                            "complexOutputsPerInitialization": len(indices),
                            "initializations": len(SENTINELS),
                            "guardWordsPerPath": len(GUARDS) * len(SENTINELS),
                            "comparison": "exact-words-except-nan-payloads",
                            "readonlyBuffersUnchanged": True,
                            "generatedSourceSHA256": (
                                hashlib.sha256(source.read_bytes()).hexdigest()
                            ),
                            "generatedLibrarySHA256": (
                                hashlib.sha256(generated.read_bytes()).hexdigest()
                            ),
                            "originalLibrarySHA256": (
                                hashlib.sha256(original.read_bytes()).hexdigest()
                            ),
                            "fullUpstreamSuite": False,
                            "fullTranslatedBackend": False,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
