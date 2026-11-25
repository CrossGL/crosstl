"""Prove fft.metal's 101 entries natively.

``fft.cpp`` plans every 1-D transform of up to 4096 points as one
threadgroup of ``fft``, ``rader_fft`` or ``bluestein_fft`` (Stockham, Rader
or Bluestein), and every larger power of two as two ``four_step_fft``
passes; it freezes the plan's function constants into the pipeline and
names it by its hash name. ``fft_model`` turns each of the 101 entries into
units, one hash name each: the 61 entries ``fft.cpp`` reaches through the
host's own names and launches, the 15 float ones it cannot reach (Rader
past 2048 points and four-step passes outside its splits) and the 25
complex32_t ones (MLX has no complex32 array) through gate-local plans
built by the same rules. Each unit is translated with the host's recipe
(``fft_packages.signature_config``) and pinned per target.

Every unit then runs ``fft.cpp``'s grid: a partial last threadgroup where
transforms share one, the Rader and Bluestein tables, both directions of
complex transforms and Bluestein's first and last convolved length. Every
float output must stay within ``test_fft``'s tolerances of a
double-precision reference. test_fft cannot run complex32_t, so every
complex32_t output must instead stay within Higham's normwise FFT error
bound for binary16 of the double-precision transform of its exact binary16
inputs. Every element past the kernel's footprint must keep its guard and
no input may change. On Metal the translation must also match the
unchanged upstream kernel, run with the same function constants, word for
word, including complex32_t.
"""

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.portable_host import fft_dispatch
from demos.integrations.mlx.portable_host import fft_layout as layout
from demos.integrations.mlx.portable_host import fft_packages
from demos.integrations.mlx.portable_host.element_storage import (
    COMPLEX,
    GUARD_COUNT,
    GUARDS,
    packet,
    unpack,
)
from demos.integrations.mlx.portable_host.matmul_dispatch import encode_binding
from demos.integrations.mlx.tests.kernels import fft_model as model
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_FFT"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_FFT_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_FFT_SHARD_COUNT"
CI_SHARD_COUNT = 5
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/fft.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
SOURCE = model.SOURCE
SOURCE_SHA256 = "c478eb84283bbdf585c0cb34b2bfde5b0fc32d1740c6ad76e8559698a57b8d2e"
HEADERS = dict(
    zip(
        model.HEADERS,
        (
            "fba2fbfd998e437cbe6db4842fcd75ae995ca1eda891057db76caba376367cda",
            "e6d2aaa314822f546d9c5f5230185577642da5b8a18b4ebb094e8e7ce06fb206",
            "0f38957ced3346cf67d7c28207b0bbc629ac7f490ade67b3d193aa8c1938a559",
        ),
    )
)
HOST_SHA256 = "15abb8ae0445410ece0c2ccb60931faaf7a9210cb16953adcaffb1ebed07de48"
UPSTREAM_TEST_SHA256 = (
    "47373d942ae896690bc6b61d5a025987c44b1d6a44c0dc5d3bad8c83182b3320"
)
UNITS = model.UNITS
# Each target's carrier of one binary16 part of complex32 storage: element
# type and bytes.
COMPLEX32_PARTS = {
    "metal": ("float16", 2),
    "directx": ("uint16", 2),
    "opengl": ("float32", 4),
}
BINARY16_STORAGE = {"encoding": "ieee754-binary16", "logicalElementType": "float16"}
# Read by no transform: 1.5 in every float part.
POISON = {**fft_dispatch.POISON, "float16": 0x3E00}
SIMD_WIDTH = 32
# fft.h's function constant indices, the twiddle-table flag included.
CONSTANT_IDS = {**layout.CONSTANT_IDS, layout.TWIDDLE_TABLE_CONSTANT: 22}


def _parameter_type(info, name):
    """The storage dtype of one fft.metal kernel parameter."""
    kind = info["kind"]
    if name in layout.SCALARS[kind]:
        return "int32"
    if name in layout.TABLES[kind]:
        dtype = layout.TABLES[kind][name]
        # complex32_t kernels read their complex tables as half2.
        return (
            info["input"]
            if dtype == COMPLEX and info["input"] == "complex32"
            else dtype
        )
    if name == "in":
        return info["input"]
    return info["output"]


def _config(work, root, target, unit):
    info = UNITS[unit]["info"]
    return fft_packages.signature_config(
        root, str(Path(work).relative_to(root) / "out"), target, info
    )


def _selected_units():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return tuple(UNITS)
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"fft shard {index} is outside [0, {count})")
    return tuple(
        unit for position, unit in enumerate(UNITS) if position % count == index
    )


SELECTED_UNITS = _selected_units()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _dispatch_count(unit):
    return len(model.cases(unit))


def _expected_validation_contract():
    return {
        "units": (
            "fft.cpp hash names <source>_n<size>_inv_<inverse>[_precomputed]: "
            "the host's for the 61 entries fft.cpp reaches, gate-local plans "
            "by the same rules for the other 40"
        ),
        "lengths": (
            "per kernel and threadgroup memory, the longest planned length "
            "whose plan uses the most radices; Bluestein convolves at the "
            "memory size; Rader at 4096 points plans past MAX_RADER_FFT_SIZE"
        ),
        "fourStepSplits": {
            "host": [list(split) for split in model.HOST_SPLITS],
            "gateLocal": [list(split) for split in model.LOCAL_SPLITS],
            "batch": model.FOUR_STEP_BATCH,
        },
        "recipe": "fft_packages.signature_config, constants frozen",
        "cases": (
            "2 * threadgroup batch + 1 transforms, or 3; both directions of "
            "complex transforms; Bluestein's first and last convolved length; "
            "one pass per four-step unit"
        ),
        "inputs": (
            "test_fft's: real and imaginary parts uniform in [0, 1); "
            "complex32_t halves them and rounds them to binary16"
        ),
        "reference": (
            "numpy.fft in double precision; the four-step step 0 temporary "
            "is the column DFTs times exp(-2 pi i k1 c / (n1 n2)), and step 1 "
            "reads its float32 rounding, or for complex32_t its binary16 "
            "rounding; complex32_t inverse passes normalize by 1/n each"
        ),
        "tolerance": (
            "test_fft: atol 1e-4 below 1025 points, else 1e-3; 1e-2 from 2^17 "
            "points; rtol 1e-4, relative to the device output"
        ),
        "complex32Tolerance": (
            "normwise relative error within Higham's Theorem 24.2 bound "
            "t eta / (1 - t eta), eta = u + gamma_4 (sqrt 2 + u), u = 2^-11, "
            "t = fft_model.half_levels"
        ),
        "twiddleTolerance": "fft_dispatch.BLUESTEIN_TWIDDLE_TOLERANCE, 1e-5",
        "guards": (
            f"{GUARD_COUNT} guards past every output's footprint stay untouched; "
            "inputs end in poisoned elements and must stay unchanged"
        ),
        "metalUpstreamComparison": (
            "the unchanged upstream kernel with the same function constants, "
            "word for word over the whole output buffer, complex32_t included"
        ),
    }


def _expected_scope(artifacts):
    host_entries = {
        record["info"]["source"] for record in UNITS.values() if record["hostReachable"]
    }
    return {
        "instantiatedEntryCount": len(model.ENTRIES),
        "coveredEntryCount": len(set(model.unit_entries().values())),
        "hostReachableEntryCount": len(host_entries),
        "unitCount": len(UNITS),
        "hostUnitCount": sum(record["hostReachable"] for record in UNITS.values()),
        "artifactCountByTarget": {
            target: sum(target in pins for pins in artifacts.values())
            for target in TARGETS
        },
        "dispatchCountByTarget": {
            target: sum(
                _dispatch_count(unit) for unit in UNITS if target in artifacts[unit]
            )
            for target in TARGETS
        },
        "ciShardCount": CI_SHARD_COUNT,
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def expected_contract(artifacts=None):
    if artifacts is None:
        artifacts = _contract()["artifacts"]
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-fft-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": dict(HEADERS),
            "hostSource": {"path": model.HOST_SOURCE, "sha256": HOST_SHA256},
            "upstreamTest": {
                "path": model.UPSTREAM_TEST,
                "sha256": UPSTREAM_TEST_SHA256,
            },
        },
        "corpusCensus": {
            "metalUnitCount": 49,
            "entryCount": 17832,
            "discoveryDiagnosticCount": 0,
        },
        "entries": list(model.ENTRIES),
        "artifacts": artifacts,
        "validationContract": _expected_validation_contract(),
        "scope": _expected_scope(artifacts),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_current_mlx_fft_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(UNITS)
    for unit, pins in artifacts.items():
        assert list(pins) == list(TARGETS)
        for target, pin in pins.items():
            record = parse_pin(pin)
            assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            assert record["sizeBytes"] > 0
    scope = contract["scope"]
    assert scope["artifactCountByTarget"] == {
        "metal": 153,
        "opengl": 153,
        "directx": 153,
    }


def test_fft_units_cover_every_instantiated_entry():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fft.current-tree.translation.json" in readme
    entries = model.ENTRIES
    assert len(entries) == 101 == len(set(entries))
    assert entries[:5] == (
        "fft_mem_256_float2_float2",
        "rader_fft_mem_256_float2_float2",
        "bluestein_fft_mem_256_float2_float2",
        "four_step_mem_256_float2_float2_0_false",
        "four_step_mem_256_float2_float2_1_false",
    )
    assert entries[-2:] == (
        "four_step_mem_4096_float2_float_1_true",
        layout.TWIDDLE_ENTRY,
    )
    assert set(model.unit_entries().values()) == set(entries)
    assert len(UNITS) == 153
    host = {r["info"]["source"] for r in UNITS.values() if r["hostReachable"]}
    assert len(host) == 61 and not any(model.HALF in entry for entry in host)
    unreachable = sorted(set(entries) - host - {e for e in entries if model.HALF in e})
    assert unreachable == sorted(
        [
            f"rader_fft_mem_4096_{pair}"
            for pair in ("float2_float2", "float_float2", "float2_float")
        ]
        + [
            f"four_step_mem_{memory}_{args}"
            for memory, steps in ((256, (0,)), (512, (1,)), (1024, (1,)), (2048, (1,)))
            for step in steps
            for args in (
                ("float2_float2_0_false", "float_float2_0_true", "float2_float2_0_true")
                if step == 0
                else (
                    "float2_float2_1_false",
                    "float2_float2_1_true",
                    "float2_float_1_true",
                )
            )
        ]
    )
    assert model.expected_unit_counts() == {
        ("stockham", "float", True): 20,
        ("rader", "float", True): 16,
        ("rader", "float", False): 4,
        ("bluestein", "float", True): 22,
        ("four_step", "float", True): 24,
        ("four_step", "float", False): 16,
        ("twiddles", "float", True): 1,
        ("stockham", "complex32", False): 10,
        ("rader", "complex32", False): 10,
        ("bluestein", "complex32", False): 10,
        ("four_step", "complex32", False): 20,
    }


def test_fft_units_follow_the_host_plans():
    assert model.chosen_lengths() == {
        ("stockham", 256): 240,
        ("stockham", 512): 480,
        ("stockham", 1024): 1008,
        ("stockham", 2048): 2002,
        ("stockham", 4096): 3360,
        ("rader", 256): 254,
        ("rader", 512): 488,
        ("rader", 1024): 1016,
        ("rader", 2048): 2037,
        ("rader", 4096): 3660,
        **{("bluestein", memory): memory for memory in model.MEMORY_SIZES},
    }
    # fft.cpp plans 3660 points as a multi-upload Bluestein transform;
    # without its Rader limit the plan is Rader's 61 points times a
    # 60-point Stockham transform.
    assert layout.plan_kind(layout.plan_fft(3660)) == "four_step"
    plan = model.unbounded_rader_plan(3660)
    assert plan["rader_n"] == 61 and plan["stockham"] == layout.plan_stockham_fft(60)
    for n1, n2 in model.HOST_SPLITS:
        assert layout.four_step_splits()[n1 * n2] == (n1, n2)
    for n1, n2 in model.LOCAL_SPLITS:
        assert layout.four_step_splits()[n1 * n2] != (n1, n2)
    for name, record in UNITS.items():
        info = record["info"]
        assert model.unit_name(info) == name
        if record["hostReachable"]:
            assert layout.signature(name) == info
        else:
            with pytest.raises(ValueError):
                layout.signature(name)
        assert info["memory"] in model.MEMORY_SIZES or info["kind"] == "twiddles"
        assert info["source"] in model.ENTRIES
        if info["kind"] != "twiddles":
            batch, threads = info["threadgroupBatchSize"], info["threadsPerFFT"]
            assert batch * info["n"] <= info["memory"] and batch * threads <= 1024
            assert info["constants"]["elems_per_thread_"] * threads >= info["n"]


def test_fft_four_step_references_compose_to_the_dft():
    """The per-pass references chain into numpy's full transform."""
    for transform in model.TRANSFORM_TYPES:
        for inverse in model.DIRECTIONS[transform]:
            for n1, n2 in ((8, 4), (4, 8), (16, 16)):
                overall = n1 * n2
                rng = np.random.default_rng(overall + inverse)
                real = rng.random((2, overall))
                signal = real + 1j * rng.random((2, overall))
                if transform == "r2c":
                    source, expected = real, np.fft.rfft(real)
                elif transform == "c2r":
                    source = np.fft.rfft(real)
                    expected = np.fft.irfft(source, n=overall)
                else:
                    source = signal
                    expected = (np.fft.ifft if inverse else np.fft.fft)(signal)
                if transform == "c2r":
                    columns = np.conj(model._full_spectrum(source, overall))
                elif inverse:
                    columns = np.conj(source)
                else:
                    columns = source
                rows = model._step1_rows(
                    model.four_step_intermediate(columns, n1, n2), n1, n2
                )
                if transform == "c2c":
                    actual = np.conj(rows) / overall if inverse else rows
                elif transform == "r2c":
                    actual = rows[:, : overall // 2 + 1]
                else:
                    actual = rows.real / overall
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_fft_cases_follow_fft_cpp_launches():
    for unit in list(UNITS)[::7]:
        info = UNITS[unit]["info"]
        for case in model.cases(unit):
            launch = case["launch"]
            assert launch["workgroupSize"] == list(info["workgroupSize"])
            if info["kind"] == "twiddles":
                assert launch["workgroupCount"] == [1, 1, 1]
                continue
            groups = launch["workgroupCount"][0]
            assert 0 < groups <= layout.MAX_WORKGROUPS
            if UNITS[unit]["hostReachable"] and info["kind"] != "bluestein":
                counts = {
                    name: case["spans"][name]
                    for name in layout.PARAMETERS[info["kind"]]
                    if name not in layout.SCALARS[info["kind"]]
                }
                metadata = layout.plan_launch(
                    unit, counts, case["scalars"], case["spans"]["out"], launch
                )
                assert metadata["footprint"] == case["footprint"]
            assert (
                case["input"].shape[0] * case["input"].shape[1] == case["spans"]["in"]
            )


@pytest.mark.parametrize("damage", ("value", "nan", "scale"))
def test_fft_comparison_rejects_defects(damage):
    case = model.cases("fft_mem_256_float2_float2_n240_inv_0")[0]
    actual = case["reference"].astype(np.complex64)
    assert model.compare(actual, case["reference"], case["tolerance"])[0]
    broken = actual.copy()
    if damage == "value":
        broken[1, 7] += 2e-3
    elif damage == "nan":
        broken[0, 0] = np.nan
    else:
        broken *= 1.001
    assert not model.compare(broken, case["reference"], case["tolerance"])[0]


@pytest.mark.parametrize("damage", ("value", "nan", "conjugate", "scale"))
def test_fft_complex32_bound_rejects_defects(damage):
    unit = "fft_mem_256_complex32_t_complex32_t_n240_inv_0"
    (case,) = model.cases(unit)
    assert case["tolerance"] is None
    bound = case["normwiseBound"]
    # Higham's bound for 8 levels in binary16.
    assert bound == pytest.approx(0.026751, abs=1e-6)
    actual = model.binary16(case["reference"])
    assert model.normwise(actual, case["reference"]) <= model.HALF_UNIT_ROUNDOFF
    broken = actual.copy()
    if damage == "value":
        broken[1, 7] += 1.5 * np.linalg.norm(case["reference"]) * bound
    elif damage == "nan":
        broken[0, 0] = np.nan
    elif damage == "conjugate":
        broken = np.conj(broken)
    else:
        broken = broken * (1 + 2 * bound)
    assert model.normwise(broken, case["reference"]) > bound


def test_fft_complex32_cases_are_exact_binary16():
    """complex32_t inputs and four-step temporaries are binary16 values, and
    the inverse four-step references normalize per pass."""
    for unit in (u for u in UNITS if model.is_half(u)):
        info = UNITS[unit]["info"]
        for case in model.cases(unit):
            values = np.asarray(case["input"])
            assert np.array_equal(model.binary16(values), values), unit
            assert case["tolerance"] is None
            levels = model.half_levels(info)
            assert case["normwiseBound"] == model.half_normwise_bound(levels)
            assert 0 < case["normwiseBound"] < 0.1, unit
            assert np.abs(case["reference"]).max() < 65504, unit
    n1, n2 = 64, 128
    step0 = "four_step_mem_256_complex32_t_complex32_t_0_false_n64_inv_1"
    step1 = "four_step_mem_512_complex32_t_complex32_t_1_false_n128_inv_1"
    assert tuple(UNITS[step0]["split"]) == tuple(UNITS[step1]["split"]) == (n1, n2)
    (first,), (second,) = model.cases(step0), model.cases(step1)
    # Step 0 scales the inverse transform's input by 1/n1 ...
    columns = np.conj(first["input"].astype(np.complex128)) / n1
    np.testing.assert_allclose(
        first["reference"], model.four_step_intermediate(columns, n1, n2)
    )
    # ... and step 1 its own input by 1/n2, leaving only the conjugation.
    rows = model._step1_rows(second["input"].astype(np.complex128), n1, n2)
    np.testing.assert_allclose(second["reference"], np.conj(rows) / n2)


def test_fft_shards_partition_the_units(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_units())
    assert sorted(unit for shard in shards for unit in shard) == sorted(UNITS)
    for shard in shards:
        assert {UNITS[unit]["info"]["kind"] for unit in shard} == {
            "stockham",
            "rader",
            "bluestein",
            "four_step",
        } | ({"twiddles"} if layout.TWIDDLE_ENTRY in shard else set())
    monkeypatch.delenv(SHARD_INDEX_ENV)
    with pytest.raises(RuntimeError):
        _selected_units()


@pytest.mark.parametrize("target", TARGETS)
def test_fft_configs_follow_the_host_recipe(tmp_path, target):
    for unit in list(UNITS)[::11]:
        info = UNITS[unit]["info"]
        work = tmp_path / "work"
        config = _config(work, tmp_path, target, unit)
        assert config.entry_points == {SOURCE: (info["source"],)}
        assert config.specialization_constants == info["constants"]
        assert config.freeze_specialization_constants is True
        if UNITS[unit]["hostReachable"]:
            host = fft_packages.project_config(tmp_path, "work/out", target, unit)
            assert config == host
        if info["kind"] != "twiddles":
            (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
            assert tuple(int(value) for value in rule) == info["workgroupSize"]
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "opengl":
            assert layout.REFERENCE_INDEX in expressions
        else:
            assert expressions == [] and config.workgroup_access_assertions == ()


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(REQUIRE_ENV, "fft", TARGETS, MLX_COMMIT)
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    pins = {
        **HEADERS,
        model.HOST_SOURCE: HOST_SHA256,
        model.UPSTREAM_TEST: UPSTREAM_TEST_SHA256,
    }
    for path, digest in pins.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    model.check_instantiation(source.decode("utf-8"))
    model.check_kernels((root / model.HEADERS[0]).read_text(encoding="utf-8"))
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("fft-metal-reference")
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
    return runner, _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )


def _translate(root, work, target, unit, record):
    """Translate one unit, check its pin and prepare its native package."""
    info = UNITS[unit]["info"]
    translation = translate_project(
        _config(work, root, target, unit),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    report = translation.to_json()
    translation.write_json(work / "report.json")
    assert validate_project_report(work / "report.json")["success"] is True
    assert report["diagnostics"] == []
    assert report["summary"]["translatedCount"] == 1
    assert report["summary"]["failedCount"] == 0
    (artifact,) = report["artifacts"]
    assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
    check_generated_artifact(
        artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
    )
    execution = artifact.get("execution")
    if info["kind"] == "twiddles":
        # Metal takes the plain kernel's threadgroup from the dispatch.
        if target == "metal":
            assert execution is None
        else:
            assert execution["sourceEntryPoints"] == [info["source"]]
            assert execution["workgroupSize"] == list(info["workgroupSize"])
    else:
        (point,) = execution["entryPoints"]
        assert point["workgroupSize"] == list(info["workgroupSize"])
    generated = root / artifact["path"]
    assert "unsupported" not in generated.read_text(encoding="utf-8")
    if target == "metal":
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    return _prepare_native_package(translation, work)


def _words(dtype, values):
    """The storage words of logical values."""
    if dtype == COMPLEX:
        return (
            np.ascontiguousarray(values, dtype=np.complex64)
            .view(np.uint64)
            .ravel()
            .tolist()
        )
    if dtype == "complex32":
        parts = np.ascontiguousarray(values, dtype=np.complex64).view(np.float32)
        return parts.astype(np.float16).view(np.uint16).ravel().tolist()
    if dtype == "float32":
        return (
            np.ascontiguousarray(values, dtype=np.float32)
            .view(np.uint32)
            .ravel()
            .tolist()
        )
    if dtype == "int16":
        return [int(value) for value in np.asarray(values, dtype=np.int16).ravel()]
    raise KeyError(dtype)


def _values(dtype, words):
    """The logical values of storage words."""
    if dtype == COMPLEX:
        return np.array(words, dtype=np.uint64).view(np.complex64)
    if dtype == "float32":
        return np.array(words, dtype=np.uint32).view(np.float32)
    raise KeyError(dtype)


def _width(dtype):
    """Storage words per element: complex32 travels as two halves."""
    return 2 if dtype == "complex32" else 1


def _guard(dtype):
    return GUARDS["float16" if dtype == "complex32" else dtype]


def _poison(dtype):
    return POISON["float16" if dtype == "complex32" else dtype]


def _storage(info, case):
    """Every parameter's storage words: inputs and tables with poisoned
    tails past their footprint, the output as guards past its footprint."""
    output = "twiddles" if info["kind"] == "twiddles" else "out"
    storage = {}
    for name in layout.PARAMETERS[info["kind"]]:
        dtype = _parameter_type(info, name)
        if name in layout.SCALARS[info["kind"]]:
            storage[name] = [case["scalars"][name]]
        elif name == output:
            reach = case["footprint"][name] + GUARD_COUNT
            storage[name] = [_guard(dtype)] * (reach * _width(dtype))
        else:
            words = _words(
                dtype, case["input"] if name == "in" else case["tables"][name]
            )
            reach = case["footprint"].get(name, case["spans"].get(name, 0))
            tail = (reach + GUARD_COUNT) * _width(dtype) - len(words)
            assert tail >= GUARD_COUNT * _width(dtype), name
            storage[name] = words + [_poison(dtype)] * tail
    return output, storage


def _encode(dtype, words, target, binding):
    scalar = binding["scalarLayout"]
    if dtype == "int32":
        return encode_binding(dtype, words, target, scalar, binding.get("kind", ""))
    if dtype == "complex32":
        # Every target reflects complex32 storage as two binary16 parts: the
        # complex_t_half record (in/out) or a half2 vector (the tables). Metal
        # keeps half words, Direct3D uint16 words holding binary16 bits and
        # OpenGL float32 carriers of binary16 values.
        element, part = COMPLEX32_PARTS[target]
        components = (
            ("record", scalar["componentCount"])
            if "componentCount" in scalar
            else ("vector", scalar["vectorWidth"])
        )
        assert components in {("vector", 2), ("record", 2)}, scalar
        if components[0] == "record":
            members = [(m["name"], m["offsetBytes"]) for m in scalar["structMembers"]]
            assert members == [("real", 0), ("imag", part)], scalar
        assert (
            scalar["elementType"],
            scalar["elementSizeBytes"],
            scalar["elementStrideBytes"],
        ) == (element, 2 * part, 2 * part), scalar
        if target == "directx":
            assert scalar["storageEncoding"] == BINARY16_STORAGE, scalar
        return packet("float16", target, words)
    return fft_dispatch.encode(dtype, words, target, scalar, binding.get("kind", ""))


def _request(descriptor, package, target, info, case):
    output_name, storage = _storage(info, case)
    inputs, observed, output = {}, [], None
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if provenance.get("kind") == "generated-execution-input":
            continue
        name = provenance["sourceResource"]["parameter"]
        assert name in storage and binding["name"] not in inputs, binding
        inputs[binding["name"]] = _encode(
            _parameter_type(info, name), storage[name], target, binding
        )
        if name == output_name:
            assert binding["access"] == "read_write", binding
            output = binding["name"]
        else:
            observed.append(binding["name"])
    assert output is not None and len(inputs) == len(storage)
    execution = fft_dispatch.native_execution(descriptor, case["launch"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: {k: v for k, v in inputs[output].items() if k != "values"}},
        execution,
        expected_target=target,
        observe_inputs=tuple(observed),
    )
    return request, output, observed, storage


def _dispatch(executor, request, output, observed):
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    assert result.status == "ok", result.details
    # The twiddle kernel reads no input.
    observations = (
        result.details.get("inputObservations", {})
        if not observed
        else (result.details["inputObservations"])
    )
    assert len(observations) == len(observed)
    assert all(item["unchanged"] for item in observations.values())
    return result.outputs[output]["values"]


def _raw(dtype, words):
    formats = {
        COMPLEX: "Q",
        "complex32": "H",
        "float32": "I",
        "int16": "h",
        "int32": "i",
    }
    return struct.pack(f"<{len(words)}{formats[dtype]}", *words)


def _upstream(work, reference, info, label, launch, storage):
    """The unchanged upstream kernel on raw ``[[buffer(i)]]`` storage, with
    the unit's function constants set as fft.cpp sets them."""
    runner, library = reference
    files, payloads = [], {}
    names = layout.PARAMETERS[info["kind"]]
    for index, name in enumerate(names):
        payload = _raw(_parameter_type(info, name), storage[name])
        payloads[name] = payload
        path = work / f"{label}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    constants = info["constants"]
    request = work / f"{label}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": launch["workgroupCount"],
                "workgroupSize": list(info["workgroupSize"]),
                "simdWidth": SIMD_WIDTH,
                "boolConstants": [
                    {"index": CONSTANT_IDS[name], "value": value}
                    for name, value in constants.items()
                    if type(value) is bool
                ],
                "intConstants": [
                    {"index": CONSTANT_IDS[name], "value": value}
                    for name, value in constants.items()
                    if type(value) is int
                ],
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run(
            [runner, library, info["source"], request, readback],
            work,
            f"{label}-upstream",
        )
    )
    output = "twiddles" if info["kind"] == "twiddles" else "out"
    words = None
    for index, name in enumerate(names):
        raw = (readback / f"buffer-{index}.bin").read_bytes()
        if name == output:
            dtype = _parameter_type(info, name)
            fmt = {COMPLEX: "Q", "complex32": "H", "float32": "I"}[dtype]
            words = list(
                struct.unpack(f"<{len(raw) // struct.calcsize(fmt)}{fmt}", raw)
            )
        else:
            assert raw == payloads[name], name
    return device, words


def _run_case(work, target, unit, index, case, executor, translated, reference):
    """Dispatch one case on the translation and, on Metal, upstream."""
    info = UNITS[unit]["info"]
    descriptor, package = translated
    label = f"case-{index}"
    request, output, observed, storage = _request(
        descriptor, package, target, info, case
    )
    name = "twiddles" if info["kind"] == "twiddles" else "out"
    dtype = _parameter_type(info, name)
    values = _dispatch(executor, request, output, observed)
    words = unpack("float16" if dtype == "complex32" else dtype, target, values)
    width = _width(dtype)
    end = case["footprint"][name] * width
    assert len(words) == len(storage[name])
    # Nothing past the kernel's footprint changes.
    assert words[end:] == storage[name][end:]
    upstream = None
    if target == "metal":
        device, upstream_words = _upstream(
            work, reference, info, label, case["launch"], storage
        )
        assert_identical(label, words, upstream_words)
        upstream = {"device": device, "identicalToTranslation": True}
    evidence = {
        "case": index,
        "launch": case["launch"],
        "transforms": case["transforms"],
        "outputHash": hashlib.sha256(_raw(dtype, words)).hexdigest(),
        "upstream": upstream,
    }
    if dtype == "complex32":
        span = case["spans"][name]
        parts = np.array(words[: 2 * span], dtype=np.uint16).view(np.float16)
        parts = parts.astype(np.float64)
        actual = (parts[0::2] + 1j * parts[1::2]).reshape(np.shape(case["reference"]))
        error = model.normwise(actual, case["reference"])
        assert error <= case["normwiseBound"], (unit, index, error)
        return {
            **evidence,
            "normwiseError": error,
            "normwiseBound": case["normwiseBound"],
        }
    span = case["spans"][name]
    actual = _values(dtype, words[:span])
    reference_values = np.asarray(case["reference"])
    actual = actual.reshape(reference_values.shape)
    within, worst = model.compare(actual, reference_values, case["tolerance"])
    assert within, (unit, index, worst)
    return {
        **evidence,
        "worstErrorToTolerance": worst,
        "tolerance": list(case["tolerance"]),
    }


@pytest.mark.parametrize("unit", SELECTED_UNITS)
def test_current_mlx_fft_native_parity(current_mlx, metal_reference, tmp_path, unit):
    root, target = current_mlx
    pin = _contract()["artifacts"][unit][target]
    with tempfile.TemporaryDirectory(prefix=".current-fft-", dir=root) as path:
        work = Path(path)
        try:
            translated = _translate(root, work, target, unit, parse_pin(pin))
            constants = translated[0].get("specializationConstants", [])
            assert all(item.get("frozen") is True for item in constants), constants
            executor = _executor(target)
            try:
                evidence = [
                    _run_case(
                        work,
                        target,
                        unit,
                        index,
                        case,
                        executor,
                        translated,
                        metal_reference,
                    )
                    for index, case in enumerate(model.cases(unit))
                ]
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "unit": unit,
                        "entry": UNITS[unit]["info"]["source"],
                        "hostReachable": UNITS[unit]["hostReachable"],
                        "target": target,
                        "artifact": pin,
                        "cases": evidence,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(
                work,
                tmp_path / "evidence",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "*.air", "*.metallib", "out", "*.bin", "*-readback", "package"
                ),
            )


def test_ci_requires_native_fft_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX FFT native parity",
        family="fft",
        count=CI_SHARD_COUNT,
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        f"{SHARD_INDEX_ENV}: ${{{{ matrix.shard }}}}",
        f'{SHARD_COUNT_ENV}: "{CI_SHARD_COUNT}"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/${{ matrix.group }}",
        "--junitxml=mlx-current-results/${{ matrix.group }}.xml",
        "demos/integrations/mlx/tests/kernels/test_current_fft.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_fft.py"
    )


def test_fft_storage_keeps_guards_and_poison():
    unit = "rader_fft_mem_256_float2_float2_n254_inv_0"
    info = UNITS[unit]["info"]
    case = model.cases(unit)[0]
    output, storage = _storage(info, case)
    assert output == "out"
    assert storage["out"] == [GUARDS[COMPLEX]] * (
        case["footprint"]["out"] + GUARD_COUNT
    )
    assert storage["in"][case["spans"]["in"] :] == [POISON[COMPLEX]] * (
        case["footprint"]["in"] - case["spans"]["in"] + GUARD_COUNT
    )
    assert storage["raders_g_q"][-GUARD_COUNT:] == [POISON["int16"]] * GUARD_COUNT
    assert storage["n"] == [254] and storage["rader_n"] == [127]
    half = "fft_mem_256_complex32_t_complex32_t_n240_inv_0"
    output, storage = _storage(UNITS[half]["info"], model.cases(half)[0])
    assert len(storage["in"]) == 2 * (
        model.cases(half)[0]["footprint"]["in"] + GUARD_COUNT
    )
    assert (
        _values(COMPLEX, _words(COMPLEX, case["input"])).tolist()
        == case["input"].ravel().tolist()
    )
