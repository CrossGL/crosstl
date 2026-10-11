"""Original-source Metal parity for complex LogAddExp and every binary layout."""

import hashlib
import itertools
import json
import math
import random
import shutil
import struct
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    load_project_config,
    reflect_target_host_interface,
    translate_project,
)
from demos.integrations.mlx.tests.kernels import test_current_binary_shapes as shapes
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)

GUARDS = (0x43555555, 0xC322AAAA) * 16
OUTPUT_SENTINEL = 0x4EAA5555
current_binary_source = shapes.current_binary_source


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _pack(words):
    return struct.pack(f"<{len(words)}I", *words)


def _words(payload):
    return struct.unpack(f"<{len(payload) // 4}I", payload)


def _complex_words(values):
    return [(_bits(value.real), _bits(value.imag)) for value in values]


def _stress_pairs():
    generator = random.Random(2200)
    pairs = [
        (
            (_bits(generator.uniform(-100, 100)), _bits(generator.uniform(-50, 50))),
            (_bits(generator.uniform(-100, 100)), _bits(generator.uniform(-50, 50))),
        )
        for _ in range(4096)
    ]
    # exp(-log(2)) reaches the magnitude branch boundary inside complex log1p.
    for delta, angle, base in itertools.product(
        range(-8, 9),
        (0.0, -0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi, 0.25),
        (-100.0, -1.0, 0.0, 1.0, 100.0),
    ):
        gap = struct.unpack("<f", _pack([_bits(math.log(2)) + delta]))[0]
        a, b = (_bits(base), 0), (_bits(base - gap), _bits(angle))
        pairs.extend(((a, b), (b, a)))
    magnitudes = (
        0,
        1,
        0x7FFFFF,
        0x800000,
        _bits(0.25),
        _bits(0.5),
        _bits(1),
        _bits(2),
        _bits(100),
        0x7F7FFFFF,
        0x7F800000,
        0x7FC00001,
    )
    components = [word | sign for word in magnitudes for sign in (0, 0x80000000)]
    controls = (
        (0, 0),
        (_bits(1), 0),
        (0, _bits(math.pi)),
        (_bits(-100), 0),
        (0x7F800000, 0),
        (0xFF800000, 0),
    )
    for a, b in itertools.product(itertools.product(components, repeat=2), controls):
        pairs.extend(((a, b), (b, a)))
    return pairs


def _assert_same_words(actual, expected):
    assert len(actual) == len(expected)
    differences = [
        (index, f"{want:08x}", f"{got:08x}")
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
        and not (got & 0x7FFFFFFF > 0x7F800000 and want & 0x7FFFFFFF > 0x7F800000)
    ]
    assert not differences, differences[:32]


def _check_readback(payloads, readbacks):
    assert set(readbacks) == set(payloads)
    for name, payload in payloads.items():
        assert len(readbacks[name]) == len(payload)
        if name != "c":
            assert readbacks[name] == payload, f"Read-only buffer changed: {name}"
    actual = _words(readbacks["c"])
    assert actual[-len(GUARDS) :] == GUARDS
    assert OUTPUT_SENTINEL not in actual[: -len(GUARDS)], "Output was not written"
    return actual[: -len(GUARDS)]


def _execute(runner, library, entry, directory, payloads, grid, resources):
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
    return _check_readback(payloads, readbacks)


def test_logaddexp_stress_inputs_cover_boundary_and_special_values():
    pairs = _stress_pairs()
    assert len(pairs) == 12198
    assert pairs == _stress_pairs()
    components = {word for pair in pairs for value in pair for word in value}
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
    } <= components
    assert len(shapes.SHAPES) == 15
    assert (
        sum(len(shapes._workload(shape)["indices"]) for shape in shapes.SHAPES) == 291
    )


@pytest.mark.parametrize(
    "actual,expected,accept",
    (
        ([0], [0], True),
        ([0x7FC00001], [0xFFC00002], True),
        ([0x80000000], [0], False),
        ([0x3F800001], [0x3F800000], False),
        ([0x7F800000], [0xFF800000], False),
        ([0x7FC00001], [0x7F800000], False),
        ([0], [0, 0], False),
    ),
)
def test_logaddexp_word_comparison_rejects_numeric_drift(actual, expected, accept):
    if accept:
        _assert_same_words(actual, expected)
    else:
        with pytest.raises(AssertionError):
            _assert_same_words(actual, expected)


@pytest.mark.parametrize(
    "mutation", (None, "input", "guard", "truncated", "missing", "unwritten")
)
def test_logaddexp_readback_checks_full_allocations(mutation):
    payloads = {"a": _pack([1, 2]), "c": _pack([3, 4] + list(GUARDS))}
    readbacks = dict(payloads)
    if mutation == "input":
        readbacks["a"] = _pack([0, 2])
    elif mutation == "guard":
        readbacks["c"] = readbacks["c"][:-4] + _pack([0])
    elif mutation == "truncated":
        readbacks["c"] = readbacks["c"][:-4]
    elif mutation == "missing":
        del readbacks["a"]
    elif mutation == "unwritten":
        readbacks["c"] = _pack([OUTPUT_SENTINEL, 4] + list(GUARDS))
    if mutation:
        with pytest.raises(AssertionError):
            _check_readback(payloads, readbacks)
    else:
        assert _check_readback(payloads, readbacks) == (3, 4)


@pytest.fixture(scope="module")
def current_source(current_binary_source):
    root, target = current_binary_source
    if target != "metal":
        pytest.skip("Complex LogAddExp exact source comparison requires native Metal")
    return root


@pytest.mark.parametrize(
    "batch", shapes.SHAPE_BATCHES, ids=lambda batch: "-".join(batch)
)
def test_current_complex_logaddexp_native_parity(
    current_source, binary_metal_reference, tmp_path, batch
):
    root = current_source
    runner, original = binary_metal_reference(root, "metal")
    entries = [f"{shape}_LogAddExpcomplex64" for shape in batch]
    with tempfile.TemporaryDirectory(
        prefix=".current-complex-logaddexp-", dir=root
    ) as path:
        work = Path(path)
        config = work / "crosstl.toml"
        config.write_text(
            shapes._configuration(work, "metal", entries[0])
            .replace(
                f'"{shapes.SOURCE}" = "{entries[0]}"',
                f'"{shapes.SOURCE}" = {json.dumps(entries)}',
            )
            .replace(f'"{entries[0]}" = [1, 1, 1]', '"*" = [1, 1, 1]'),
            encoding="utf-8",
        )
        try:
            report = translate_project(
                load_project_config(root, config), format_output=False
            )
            report.write_json(work / "report.json")
            data = report.to_json()
            assert data["summary"]["failedCount"] == 0, data["diagnostics"]
            assert data["summary"]["translatedCount"] == len(entries)
            artifacts = {row["entryPoint"]["source"]: row for row in data["artifacts"]}
            assert len(data["artifacts"]) == len(entries) and set(artifacts) == set(
                entries
            )
            for shape, entry in zip(batch, entries):
                directory = work / entry
                directory.mkdir()
                artifact = artifacts[entry]
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
                workload = shapes._workload(shape)
                cases = [
                    (
                        "layout",
                        _complex_words(workload["a"]),
                        _complex_words(workload["b"]),
                        len(workload["indices"]),
                        workload["constants"],
                        workload["grid"],
                    )
                ]
                if shape == "vv":
                    pairs = _stress_pairs()
                    cases.append(
                        (
                            "stress",
                            [a for a, b in pairs],
                            [b for a, b in pairs],
                            len(pairs),
                            {"size": ("uint32", [len(pairs)])},
                            [len(pairs), 1, 1],
                        )
                    )
                for name, a, b, count, constants, grid in cases:
                    payloads = {
                        "a": _pack([word for value in a for word in value]),
                        "b": _pack([word for value in b for word in value]),
                        "c": _pack([OUTPUT_SENTINEL] * (2 * count) + list(GUARDS)),
                    }
                    formats = {"int32": "i", "uint32": "I", "int64": "q"}
                    payloads.update(
                        {
                            key: struct.pack(f"<{len(values)}{formats[dtype]}", *values)
                            for key, (dtype, values) in constants.items()
                        }
                    )
                    expected = _execute(
                        runner,
                        original,
                        entry,
                        directory / name / "original",
                        payloads,
                        grid,
                        resources,
                    )
                    actual = _execute(
                        runner,
                        generated,
                        artifact["entryPoint"]["target"],
                        directory / name / "translated",
                        payloads,
                        grid,
                        resources,
                    )
                    _assert_same_words(actual, expected)
                    (directory / name / "evidence.json").write_text(
                        json.dumps(
                            {
                                "commit": shapes.MLX_COMMIT,
                                "entryPoint": entry,
                                "complexOutputs": count,
                                "guardWords": len(GUARDS),
                                "comparison": "exact-words-except-nan-payloads",
                                "readonlyBuffersUnchanged": True,
                                "generatedSourceSHA256": (
                                    hashlib.sha256(source.read_bytes()).hexdigest()
                                ),
                                "originalLibrarySHA256": (
                                    hashlib.sha256(original.read_bytes()).hexdigest()
                                ),
                                "generatedLibrarySHA256": (
                                    hashlib.sha256(generated.read_bytes()).hexdigest()
                                ),
                                "fullUpstreamSuite": False,
                                "fullTranslatedBackend": False,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
