"""Native logarithm packages and original-source operand-underflow checks."""

import ast
import hashlib
import json
import math
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
from demos.integrations.mlx.tests.kernels import test_current_unary_roots as roots
from demos.integrations.mlx.tests.kernels import test_unary_native_loader as unary
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_logaddexp import (
    _assert_same_words,
    _pack,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_magnitude import (
    GUARDS,
    SENTINELS,
    SHAPES,
    _execute,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from demos.integrations.mlx.tests.kernels.test_current_unary_fma import (
    _run_current_unary,
)
from tests import workflow_text
from tests.test_translator import test_metal_precise_log as logarithm
from tests.test_translator.test_metal_precise_trig import _bits, _float

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_LOGARITHMS"
OPERATIONS = ("Log", "Log2", "Log10")
PROFILE = {"binary32_atan2_profile": "flush-subnormals"}
LOG_PROFILE = {"binary32_log_profile": "flush-subnormals"}
# The pinned logarithm tests use numpy.allclose without overriding these.
RTOL, ATOL = 1e-5, 1e-8


def _samples():
    special = (
        0,
        0x80000000,
        1,
        0x80000001,
        0x7FFFFF,
        0x807FFFFF,
        0x800000,
        0x80800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0xFF800001,
        0x7FC01234,
        0xFFC01234,
        _bits(1),
        _bits(-1),
        _bits(0.5),
        _bits(-0.5),
    )
    values = [(a, b) for a in special for b in special]
    for center in (
        0x800000,
        _bits(0.5),
        _bits(1),
        _bits(2),
        0x5F000000,
        _bits(math.sqrt(0.5)),
    ):
        for word in range(center - 4, center + 5):
            for sign in (0, 0x80000000):
                values.extend((word | sign, imag) for imag in special)
                values.extend((imag, word | sign) for imag in special)
    generator = random.Random(3413)
    values.extend(
        (generator.getrandbits(32), generator.getrandbits(32)) for _ in range(16384)
    )
    values.extend(
        (_bits(generator.uniform(-2, 2)), _bits(generator.uniform(-2, 2)))
        for _ in range(4096)
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
        selected = random.Random(3414).sample(values, 105)
        selected[:40] = values[:40]
        for index, value in zip(indices, selected):
            values[index] = value
        constants = {
            "in_shape": struct.pack("<3i", 3, 5, 7),
            "in_strides": struct.pack("<3q", 90, 2, 11),
            "ndim": struct.pack("<i", 3),
        }
        grid = [7 if shape == "gn1" else 2, 5, 3]
    else:
        indices = list(range(len(values)))
        constants = {"size": struct.pack("<q" if shape == "v2" else "<I", len(values))}
        assert len(values) % 15 == 0
        grid = [15, len(values) // 15, 1] if shape == "v2" else [len(values), 1, 1]
    return values, indices, constants, grid


def _check_values(actual, expected):
    assert len(actual) == len(expected), "Output length changed"
    maximum = 0
    for index, (got, want) in enumerate(zip(actual, expected)):
        a, b = got & 0x7FFFFFFF, want & 0x7FFFFFFF
        if b > 0x7F800000:
            assert a > 0x7F800000, (index, "NaN classification")
        elif b in (0, 0x7F800000):
            assert got == want, (index, "zero sign or infinity")
        else:
            assert 0 < a < 0x7F800000, (index, "finite nonzero result")
            assert got >> 31 == want >> 31, (index, "result sign")
            assert abs(_float(got) - _float(want)) <= ATOL + RTOL * abs(_float(want)), (
                index,
                hex(got),
                hex(want),
            )
            maximum = max(maximum, abs(got - want))
    return maximum


@pytest.mark.parametrize("shape", SHAPES)
def test_complex_logarithm_inputs_cover_layouts_and_boundaries(shape):
    values, indices, constants, grid = _workload(shape)
    assert len(values) == 25200
    assert len(indices) == (105 if shape.startswith("gn") else 25200)
    assert len(set(indices)) == len(indices) and max(indices) < len(values)
    assert _workload(shape) == (values, indices, constants, grid)
    selected = {word for index in indices for word in values[index]}
    assert {
        0,
        0x80000000,
        1,
        0x80000001,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
    } <= selected
    if shape.startswith("gn"):
        assert struct.unpack("<3q", constants["in_strides"]) == (90, 2, 11)
    else:
        assert math.prod(grid) == len(indices)
        assert (_bits(1), _bits(2)) in values


@pytest.mark.parametrize(
    "got,want",
    (
        (0x80000000, 0),
        (0, 1),
        (0x7F800000, 0xFF800000),
        (0x7F800000, 0x7FC00001),
        (0x3F800000, 0xBF800000),
        (0x3F801000, 0x3F800000),
    ),
)
def test_complex_logarithm_comparison_rejects_corruption(got, want):
    with pytest.raises(AssertionError):
        _check_values([got], [want])


def test_complex_logarithm_comparison_preserves_source_tolerance():
    assert _check_values([0x3F800001, 0xFFC00001], [0x3F800000, 0x7FC00002]) == 1
    with pytest.raises(AssertionError, match="Output length"):
        _check_values([], [0])


@pytest.mark.parametrize(
    "path", (unary.UNARY_METAL_CONTRACT_PATH, unary.SCALAR_UNARY_METAL_CONTRACT_PATH)
)
def test_complex_logarithm_contracts_record_entry_policy(path):
    contract = json.loads(path.read_text())
    shapes = SHAPES if path == unary.UNARY_METAL_CONTRACT_PATH else ("v",)
    assert contract["sourceOptionsByEntryPoint"] == {
        f"{shape}_{operation}complex64complex64": PROFILE
        for shape in shapes
        for operation in (*OPERATIONS, "ArcSin", "ArcCos")
    }
    assert unary.COMPLEX_ATAN2_SOURCE_OPTIONS == PROFILE
    assert contract["sourceOptionsByOperator"]["Log"] == LOG_PROFILE
    assert unary.LOG_SOURCE_OPTIONS == LOG_PROFILE


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "operation", (*OPERATIONS, "Log1p", "Sqrt", "ArcSin", "ArcCos", "ArcTan")
)
@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "complex64"))
def test_complex_logarithm_policy_is_scoped_to_metal_complex_entries(
    tmp_path, target, operation, dtype
):
    entry = f"v_{operation}{dtype}{dtype}"
    workload = next(
        row for row in unary.UNARY_METAL_WORKLOADS if row.entry_point == entry
    )
    path = tmp_path / "crosstl.toml"
    path.write_text(unary._project_config(target, workload), encoding="utf-8")
    options = load_project_config(tmp_path, path).source_options["metal"]
    actual = {key: options[key] for key in PROFILE if key in options}
    expected = (
        PROFILE
        if target == "metal"
        and dtype == "complex64"
        and operation in (*OPERATIONS, "ArcSin", "ArcCos")
        else {}
    )
    assert actual == expected
    assert {key: options[key] for key in LOG_PROFILE if key in options} == (
        LOG_PROFILE if target == "metal" and operation == "Log" else {}
    )


def test_ci_requires_complex_logarithm_source_comparison():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    checkout = (
        workflow_text.workflow_job_text(workflow, "mlx-metal-porting")
        .split("      - name: Checkout current MLX kernel tree\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    assert "sparse-checkout set mlx/backend/metal/kernels python/tests" in checkout
    step = (
        workflow_text.workflow_job_text(workflow, "mlx-metal-porting")
        .split("      - name: Validate pinned unary roots and logarithms\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert (
        "demos/integrations/mlx/tests/kernels/test_current_unary_logarithms.py" in step
    )
    assert "--timeout-seconds 300" in step and "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step


@pytest.fixture(scope="module")
def logarithm_source():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned logarithm execution")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    assert target in {"metal", "directx", "opengl"}, "Missing or invalid target"
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
            "python/tests/test_ops.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert (
        hashlib.sha256((root / unary.MLX_UNARY_SOURCE).read_bytes()).hexdigest()
        == unary.MLX_UNARY_SHA256
    )
    module = ast.parse((root / "python/tests/test_ops.py").read_text())
    functions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("test_log", "test_log2", "test_log10"):
        calls = [
            node
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "allclose"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
        ]
        assert len(calls) == 2 and all(
            len(node.args) == 2 and not node.keywords for node in calls
        )
    return root


@pytest.mark.parametrize("target", (None, "unknown"))
def test_complex_logarithm_requires_valid_target(monkeypatch, target):
    monkeypatch.setenv(REQUIRE_ENV, "1")
    if target is None:
        monkeypatch.delenv("CROSTL_MLX_CURRENT_TARGET", raising=False)
    else:
        monkeypatch.setenv("CROSTL_MLX_CURRENT_TARGET", target)
    with pytest.raises(AssertionError, match="Missing or invalid target"):
        logarithm_source.__wrapped__()


@pytest.fixture(scope="module")
def logarithm_reference(logarithm_source, tmp_path_factory):
    if os.environ["CROSTL_MLX_CURRENT_TARGET"] != "metal":
        pytest.skip("Original-source comparison requires native Metal")
    work = tmp_path_factory.mktemp("complex-logarithm-source")
    return _build_complex_reference(logarithm_source, work)


def _build_complex_reference(root, work):
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
        root / unary.MLX_UNARY_SOURCE,
        work / "original.metallib",
        root,
        upstream=True,
    )
    return runner, original


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_logarithm_original_reference_requires_metal(monkeypatch, target):
    monkeypatch.setenv("CROSTL_MLX_CURRENT_TARGET", target)
    with pytest.raises(pytest.skip.Exception, match="requires native Metal"):
        logarithm_reference.__wrapped__(None, None)


def test_logarithm_package_oracle_covers_source_domain():
    inputs = logarithm._inputs()
    assert len(inputs) == 28024
    assert {(word >> 23) & 255 for word in inputs} == set(range(256))
    assert {word ^ 0x80000000 for word in inputs} == set(inputs)
    assert set(range(_bits(1) - 4096, _bits(1) + 4097)) <= set(inputs)
    assert logarithm._oracle(_bits(1), "flush-subnormals") == 0
    for word in (0, 0x80000000, 1, 0x80000001, 0x7FFFFF, 0x807FFFFF):
        assert logarithm._oracle(word, "flush-subnormals") == 0xFF800000


def test_current_logarithm_native_loader(logarithm_source, tmp_path):
    inputs = logarithm._inputs()
    expected = [logarithm._oracle(word, "flush-subnormals") for word in inputs]
    actual = _run_current_unary(
        tmp_path,
        "Log",
        [_float(word) for word in inputs],
        [_float(word) for word in expected],
        RTOL,
        ATOL,
        "metal_precise_log_float",
        source_options=LOG_PROFILE,
    )
    actual_words = [_bits(value) for value in actual]
    assert len(actual_words) == len(expected)
    maximum = max(
        logarithm._check_value(got, want)
        for got, want in zip(actual_words, expected, strict=True)
    )
    evidence = tmp_path / "evidence"
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    for path in (
        evidence / "report.json",
        evidence / "artifacts.json",
        evidence / "package/runtime-package.json",
    ):
        (artifact,) = json.loads(path.read_text())["artifacts"]
        assert artifact["target"] == target
        assert artifact["provenance"]["binary32LogProfile"] == "flush-subnormals"
    report = json.loads((evidence / "report.json").read_text())
    (artifact,) = report["artifacts"]
    parts = Path(artifact["path"]).parts
    source = evidence.joinpath(*parts[parts.index("out") :])
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert artifact["generatedHash"]["value"] == source_hash
    parity = json.loads((evidence / "parity.json").read_text())
    identity = parity["runtime"]["artifactIdentityVerification"]
    assert identity["verificationStatus"] == "verified"
    assert identity["observedIdentity"] == identity["expectedIdentity"]
    assert identity["observedIdentity"]["hash"]["value"] == source_hash
    (evidence / "logarithm-oracle.json").write_text(
        json.dumps(
            {
                "commit": MLX_COMMIT,
                "entryPoint": "v_Logfloat32float32",
                "target": target,
                "sourceOptions": parity["sourceOptions"],
                "generatedSourceSHA256": source_hash,
                "oracle": "120-digit Decimal ln, nearest-even binary32",
                "maximumAllowedUlps": 4,
                "maximumObservedUlps": maximum,
                "requestedInputWords": inputs,
                "expectedWords": expected,
                "actualWords": actual_words,
                "inputEncoding": "float32 values; NaN payloads are not preserved",
                "guardRegionsChecked": False,
                "inputDeviceReadbacksRetained": False,
                "fullUpstreamSuite": False,
                "fullTranslatedBackend": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("operation", OPERATIONS)
def test_current_complex_logarithm_native_comparison(
    logarithm_source, logarithm_reference, tmp_path, operation
):
    _compare_complex_source(logarithm_source, logarithm_reference, tmp_path, operation)


def _compare_complex_source(root, reference, tmp_path, operation):
    runner, original = reference
    entries = [f"{shape}_{operation}complex64complex64" for shape in SHAPES]
    workloads = {row.entry_point: row for row in unary.UNARY_METAL_WORKLOADS}
    with tempfile.TemporaryDirectory(
        prefix=".current-complex-unary-", dir=root
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
            data = report.to_json()
            assert (
                data["summary"]["translatedCount"] == 4
                and data["summary"]["failedCount"] == 0
            ), data["diagnostics"]
            assert not [
                row for row in data["diagnostics"] if row["severity"] == "error"
            ]
            artifacts = {row["entryPoint"]["source"]: row for row in data["artifacts"]}
            assert len(data["artifacts"]) == 4 and set(artifacts) == set(entries)
            for shape, entry in zip(SHAPES, entries):
                directory = work / entry
                directory.mkdir()
                artifact = artifacts[entry]
                assert (
                    artifact["provenance"]["binary32Atan2Profile"] == "flush-subnormals"
                )
                if operation == "Log":
                    assert (
                        artifact["provenance"]["binary32LogProfile"]
                        == "flush-subnormals"
                    )
                assert artifact["sourceHash"]["value"] == unary.MLX_UNARY_SHA256
                assert artifact["templateMaterialization"][
                    "specializations"
                ] == unary._expected_unary_materializations(workloads[entry])
                source = root / artifact["path"]
                assert (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    == artifact["generatedHash"]["value"]
                )
                interface = reflect_target_host_interface(source, target="metal")
                assert interface["status"] == "ready" and not interface["diagnostics"]
                resources = sorted(
                    interface["resources"], key=lambda row: row["binding"]
                )
                generated = _metal_library(
                    source, directory / "generated.metallib", root
                )
                values, indices, constants, grid = _workload(shape)
                previous, previous_original, maximum = None, None, 0
                for trial, sentinel in enumerate(SENTINELS):
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
                        directory / str(trial) / "original",
                        payloads,
                        grid,
                        resources,
                        sentinel,
                    )
                    actual = _execute(
                        runner,
                        generated,
                        artifact["entryPoint"]["target"],
                        directory / str(trial) / "translated",
                        payloads,
                        grid,
                        resources,
                        sentinel,
                    )
                    maximum = max(maximum, _check_values(actual, expected))
                    if previous is not None:
                        _assert_same_words(actual, previous)
                        _assert_same_words(expected, previous_original)
                    previous = actual
                    previous_original = expected
                (directory / "evidence.json").write_text(
                    json.dumps(
                        {
                            "commit": MLX_COMMIT,
                            "entryPoint": entry,
                            "complexOutputsPerInitialization": len(indices),
                            "initializations": len(SENTINELS),
                            "guardWordsPerPath": len(GUARDS) * len(SENTINELS),
                            "sourceOptions": {
                                **PROFILE,
                                **(LOG_PROFILE if operation == "Log" else {}),
                            },
                            "rtol": RTOL,
                            "atol": ATOL,
                            "maximumFiniteUlpDifference": maximum,
                            "exactZeroSignsAndSpecialClassifications": True,
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


def _real_inputs(dtype):
    if dtype in {"float16", "bfloat16"}:
        return list(range(65536))
    assert dtype == "float32"
    words = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x7FFFFF,
        0x807FFFFF,
        0x800000,
        0x80800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0xFF800001,
        0x7FC01234,
        0xFFC01234,
    ]
    for value in (0.927734375, 1, 2, 0.5, 4, 10, 16, 64, 80, 87, 88, 89, 90, 100):
        center = _bits(value)
        words.extend(
            word | sign
            for word in range(center - 16, center + 17)
            for sign in (0, 0x80000000)
        )
    generator = random.Random(3404)
    words.extend(generator.getrandbits(32) for _ in range(65536))
    return words


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
def test_real_logarithm_inputs_cover_storage_domain(dtype):
    words = _real_inputs(dtype)
    assert words == _real_inputs(dtype)
    if dtype == "float32":
        assert len(words) == 66476
        assert {
            0,
            0x80000000,
            1,
            0x80000001,
            0x7FFFFF,
            0x800000,
            _bits(1),
            _bits(100),
            0x7F800000,
            0xFF800000,
            0x7FC01234,
        } <= set(words)
    else:
        assert words == list(range(65536))


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
def test_current_real_logarithm_native_comparison(
    logarithm_source, logarithm_reference, tmp_path, dtype
):
    root = logarithm_source
    runner, original = logarithm_reference
    entry = f"v_Log{dtype}{dtype}"
    workload = next(
        row for row in unary.UNARY_METAL_WORKLOADS if row.entry_point == entry
    )
    with tempfile.TemporaryDirectory(
        prefix=".current-real-logarithm-", dir=root
    ) as path:
        work = Path(path)
        try:
            config_path = work / "crosstl.toml"
            config_path.write_text(
                unary._project_config("metal", workload), encoding="utf-8"
            )
            config = replace(
                load_project_config(root, config_path), output_dir=work / "out"
            )
            report = translate_project(config, format_output=False)
            report.write_json(work / "report.json")
            data = report.to_json()
            assert (
                data["summary"]["translatedCount"] == 1
                and not data["summary"]["failedCount"]
            ), data["diagnostics"]
            assert not [
                row for row in data["diagnostics"] if row["severity"] == "error"
            ]
            (artifact,) = data["artifacts"]
            assert artifact["entryPoint"]["source"] == entry
            assert artifact["sourceHash"]["value"] == unary.MLX_UNARY_SHA256
            assert artifact["templateMaterialization"][
                "specializations"
            ] == unary._expected_unary_materializations(workload)
            assert artifact["provenance"]["binary32LogProfile"] == "flush-subnormals"
            assert "binary32Atan2Profile" not in artifact["provenance"]
            source = root / artifact["path"]
            assert (
                hashlib.sha256(source.read_bytes()).hexdigest()
                == artifact["generatedHash"]["value"]
            )
            interface = reflect_target_host_interface(source, target="metal")
            assert interface["status"] == "ready" and not interface["diagnostics"]
            resources = sorted(interface["resources"], key=lambda row: row["binding"])
            library = _metal_library(source, work / "generated.metallib", root)
            values, guards = _real_inputs(dtype), list(roots._guards(dtype))
            previous = {}
            maximum = 0
            for trial, sentinel in enumerate(roots._fills(dtype)):
                payloads = {
                    "in_": roots._pack(values + guards, dtype),
                    "out_": roots._pack([sentinel] * len(values) + guards, dtype),
                    "size": struct.pack("<I", len(values)),
                }
                outputs = {}
                for label, module, name in (
                    ("original", original, entry),
                    ("translated", library, artifact["entryPoint"]["target"]),
                ):
                    outputs[label] = roots._execute(
                        runner,
                        module,
                        name,
                        work / str(trial) / label,
                        payloads,
                        len(values),
                        resources,
                        dtype,
                    )
                    if label in previous:
                        roots._assert_words(outputs[label], previous[label], dtype)
                if dtype == "float32":
                    maximum = max(
                        maximum,
                        _check_values(outputs["translated"], outputs["original"]),
                    )
                else:
                    roots._assert_words(
                        outputs["translated"], outputs["original"], dtype
                    )
                previous = outputs
            (work / "evidence.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entryPoint": entry,
                        "outputsPerInitialization": len(values),
                        "initializations": 2,
                        "guardWordsPerPath": len(guards) * 2,
                        "sourceOptions": LOG_PROFILE,
                        "comparison": (
                            "source-tolerance"
                            if dtype == "float32"
                            else "exact-words-except-nan-payloads"
                        ),
                        "rtol": RTOL if dtype == "float32" else 0,
                        "atol": ATOL if dtype == "float32" else 0,
                        "maximumFiniteUlpDifference": maximum,
                        "readonlyBuffersUnchanged": True,
                        "generatedSourceSHA256": (
                            hashlib.sha256(source.read_bytes()).hexdigest()
                        ),
                        "generatedLibrarySHA256": (
                            hashlib.sha256(library.read_bytes()).hexdigest()
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
