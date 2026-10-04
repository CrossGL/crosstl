"""Execute pinned float scatter policies through their unchanged JIT wrapper."""

import itertools
import json
import os
import re
import struct
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host.prepare import COMMIT
from tests.test_translator.test_float_storage_encoding import WORDS
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_mlx_atomic_load_runtime import HEADER
from tests.test_translator.test_mlx_current_gather import _validate
from tests.test_translator.test_mlx_general_scatter_runtime import (
    HEADERS,
    JIT,
    _prepare_request,
    _verify_source,
)
from tests.test_translator.test_mlx_general_scatter_runtime import (
    _workload as _integer_workload,
)

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_FLOAT_SCATTER"
OPERATIONS = ("none", "sum", "prod", "min", "max")
CASES = (
    *itertools.product(OPERATIONS, ("strided", "partial", "contention")),
    ("none", "payloads"),
)
GUARD = 0x4B123456


def _word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _source(root, operation, layout):
    count = 2 if layout == "partial" else 1
    contiguous, nwork = (True, 4) if count == 2 else (False, 1)
    template = re.search(
        r'scatter_kernels = R"\((.*?)\)";', (root / JIT).read_text(), re.S
    ).group(1)
    entry = f"scatterfloat32int64_{operation}_{count}_updc_{str(contiguous).lower()}_nwork{nwork}_int"
    policy = "None" if operation == "none" else f"{operation.title()}<float>"
    source = "".join(f'#include "{header}"\n' for header in HEADERS) + template.format(
        f"float32int64_{operation}",
        "float",
        "int64_t",
        policy,
        count,
        "\n".join(
            f"const device int64_t *idx{i} [[buffer({20 + i})]]," for i in range(count)
        ),
        ",".join(f"idx{i}" for i in range(count)),
        str(contiguous).lower(),
        nwork,
        "int",
    )
    return entry, source


def _workload(operation, layout, target="metal"):
    count = 2 if layout == "partial" else 1
    supplied, _, grid = _integer_workload(count, target)
    size = 15 if count == 2 else 5
    length = 7 if count == 2 else 8
    if layout in {"contention", "payloads"}:
        length = 65 if layout == "contention" else len(WORDS)
        if layout == "payloads":
            size = length
            supplied["out_shape"]["values"] = [size]
        supplied["upd_shape"]["values"] = [length, 1]
        supplied["idx_shapes"]["values"] = [length]
        supplied["idx_size"]["values"] = [length]
        indices = [2] * length if layout == "contention" else list(range(length))
        supplied["idx0"] = {
            "dtype": "int64",
            "shape": [2 * length],
            "values": [v for index in indices for v in (index, -999)],
        }
        grid = [1, length, 1]
    columns = (
        supplied["idx1"]["values"] if count == 2 else supplied["idx0"]["values"][::2]
    )
    rows = supplied["idx0"]["values"] if count == 2 else [0] * length
    addresses = [
        (row % 3) * 5 + column % 5 if count == 2 else column % size
        for row, column in zip(rows, columns)
    ]
    initial = [(-1.0 if index % 2 else 1.0) * (index + 1) / 2 for index in range(size)]
    if operation == "none":
        updates = [(destination + 1) / -4 for destination in addresses]
    elif operation == "prod":
        updates = [(-1.0, 0.5, -2.0, 1.0)[index % 4] for index in range(length)]
    else:
        updates = [(-3.25, 1.5, -0.5, 4.75, -2.0)[index % 5] for index in range(length)]
    expected = initial.copy()
    for destination, update in zip(addresses, updates):
        value = expected[destination]
        if operation == "none":
            expected[destination] = update
        elif operation == "sum":
            expected[destination] = value + update
        elif operation == "prod":
            expected[destination] = value * update
        elif operation == "min":
            expected[destination] = min(value, update)
        else:
            expected[destination] = max(value, update)
    initial_words = [_word(value) for value in initial]
    update_words = [_word(value) for value in updates]
    final_words = [_word(value) for value in expected]
    if layout == "payloads":
        update_words = list(WORDS)
        final_words = list(WORDS)
    padded_updates = (
        update_words
        if count == 2
        else [v for bits in update_words for v in (bits, GUARD)]
    )
    supplied["updates"] = {
        "dtype": "float32",
        "shape": [len(padded_updates)],
        "values": padded_updates,
        "encoding": FLOAT32_BITS,
    }
    supplied["out"] = {
        "dtype": "float32",
        "shape": [size + 32, 1],
        "values": [*initial_words, *([GUARD] * 32)],
        "encoding": FLOAT32_BITS,
    }
    expected = {**supplied["out"], "values": [*final_words, *([GUARD] * 32)]}
    return supplied, expected, grid


def _request(root, target, work, operation, layout):
    entry, source = _source(root, operation, layout)
    supplied, expected, grid = _workload(operation, layout, target)
    if target == "opengl":
        for value in (supplied["out"], expected):
            value["dtype"] = "uint32"
            del value["encoding"]
    return _prepare_request(root, target, work, entry, source, supplied, expected, grid)


@pytest.mark.parametrize("operation,layout", CASES)
def test_pinned_float_scatter_executes_natively(tmp_path, operation, layout):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native pinned float scatter")
    assert sys.platform in {"darwin", "win32"}
    target = "metal" if sys.platform == "darwin" else "directx"
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    hashes = _verify_source(root, (*HEADERS, JIT, HEADER))
    try:
        entry, source, request, expected = _request(
            root, target, tmp_path, operation, layout
        )
        (tmp_path / "workload.json").write_text(
            json.dumps(
                {
                    "commit": COMMIT,
                    "headers": hashes,
                    "entry": entry,
                    "operation": operation,
                    "layout": layout,
                    "inputs": _workload(operation, layout, target)[0],
                    "outputs": expected,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _execute(
            request,
            expected,
            tmp_path,
            original_source=source,
            original_entry=entry,
            metal_compile_flags=("-I", str(root)),
            validate=_validate,
        )
    finally:
        assert _verify_source(root, (*HEADERS, JIT, HEADER)) == hashes


@pytest.mark.parametrize("operation,layout", CASES)
def test_float_scatter_workload_preserves_guards_and_metadata(operation, layout):
    inputs, expected, grid = _workload(operation, layout)
    assert inputs["out"]["values"][-32:] == expected["values"][-32:] == [GUARD] * 32
    assert inputs["updates"]["encoding"] == expected["encoding"] == FLOAT32_BITS
    assert grid == [1, 2 if layout == "partial" else inputs["idx_size"]["values"][0], 1]
    if layout != "partial":
        assert (
            inputs["updates"]["values"][1::2]
            == [GUARD] * inputs["idx_size"]["values"][0]
        )
    if layout == "payloads":
        assert expected["values"][:-32] == WORDS


def test_float_scatter_requires_native_windows_and_metal_execution():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/mlx-gather-roundtrip.yml"
    ).read_text()
    for name in (
        "Validate float compare-exchange and scatter",
        "Validate indexed DirectX gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "test_mlx_float_scatter_runtime.py" in step
        assert "--timeout-seconds 1200" in step and "-n auto" in step
        assert "--junitxml=" in step and "--basetemp=" in step
        assert "if:" not in step
    opengl = ci_coverage.workflow_step_section(
        workflow, "Validate indexed OpenGL gather and resource aggregates"
    )
    assert REQUIRE_ENV not in opengl
    assert "continue-on-error" not in workflow


@pytest.mark.parametrize("operation,layout", CASES)
def test_float_scatter_target_flags_preserve_index_metadata(operation, layout):
    metal, metal_expected, metal_grid = _workload(operation, layout)
    directx, directx_expected, directx_grid = _workload(operation, layout, "directx")
    assert metal_expected == directx_expected and metal_grid == directx_grid
    assert metal.pop("idx_contigs") == {
        "dtype": "bool",
        "shape": [2 if layout == "partial" else 1],
        "values": [layout == "partial"] * (2 if layout == "partial" else 1),
    }
    flags = directx.pop("idx_contigs")
    assert flags["dtype"] == "uint32"
    assert flags["values"] == [int(layout == "partial")] * len(flags["values"])
    assert metal == directx


@pytest.mark.parametrize("operation,layout", CASES)
def test_float_scatter_reference_matches_indexed_updates(operation, layout):
    import numpy as np

    inputs, expected, _ = _workload(operation, layout)
    shape = inputs["out_shape"]["values"]
    size = int(np.prod(shape))
    indices = []
    for axis in range(len(shape)):
        stride = inputs["idx_strides"]["values"][axis]
        indices.append(np.asarray(inputs[f"idx{axis}"]["values"][::stride]))
    update_stride = inputs["upd_strides"]["values"][0]
    updates = np.asarray(inputs["updates"]["values"], dtype=np.uint32)[::update_stride]
    actual = np.asarray(inputs["out"]["values"], dtype=np.uint32)[:size].reshape(shape)
    if operation == "none":
        actual[tuple(indices)] = updates
    else:
        function = {
            "sum": np.add,
            "prod": np.multiply,
            "min": np.minimum,
            "max": np.maximum,
        }[operation]
        function.at(actual.view(np.float32), tuple(indices), updates.view(np.float32))
    assert actual.reshape(-1).tolist() == expected["values"][:size]


@pytest.mark.parametrize("operation,layout", CASES)
def test_float_scatter_concurrent_results_are_order_independent(operation, layout):
    if layout == "payloads":
        assert operation == "none"
        assert len(WORDS) == len(
            set(_workload(operation, layout)[0]["idx0"]["values"][::2])
        )
        return
    inputs, _, _ = _workload(operation, layout)
    shape = inputs["out_shape"]["values"]
    count = inputs["idx_size"]["values"][0]
    stride = inputs["upd_strides"]["values"][0]
    groups = {}
    for item in range(count):
        address = tuple(
            inputs[f"idx{axis}"]["values"][item * inputs["idx_strides"]["values"][axis]]
            % extent
            for axis, extent in enumerate(shape)
        )
        value = struct.unpack(
            "<f", struct.pack("<I", inputs["updates"]["values"][item * stride])
        )[0]
        groups.setdefault(address, []).append(Fraction(value))
    for address, values in groups.items():
        offset = sum(
            index * stride
            for index, stride in zip(address, inputs["out_strides"]["values"])
        )
        initial = Fraction(
            struct.unpack("<f", struct.pack("<I", inputs["out"]["values"][offset]))[0]
        )
        if operation == "none":
            assert len(set(values)) == 1
        elif operation == "sum":
            # Every partial sum is a representable multiple of one quarter.
            assert all((value * 4).denominator == 1 for value in (initial, *values))
            assert (abs(initial) + sum(map(abs, values))) * 4 < 2**24
        elif operation == "prod":
            # Powers of two only change the exponent; no ordering can overflow or underflow.
            assert set(map(abs, values)) <= {Fraction(1, 2), Fraction(1), Fraction(2)}
            assert abs(initial) * Fraction(1, 2) ** len(values) >= Fraction(2) ** -126
            assert abs(initial) * 2 ** len(values) < Fraction(2) ** 127
        else:
            assert operation in {"min", "max"}
            assert all(value != 0 for value in (initial, *values))
