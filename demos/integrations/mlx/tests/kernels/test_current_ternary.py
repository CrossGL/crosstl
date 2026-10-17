"""Execute every pinned MLX ternary ``Select`` kernel with bit-exact readbacks."""

import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/ternary.metal"
SOURCE_SHA256 = "e042fe449dbf49761547d244d521bbe0be883eff2e67cffd33f674b95c3a4dc5"
HEADER = "mlx/backend/metal/kernels/ternary.h"
HEADER_SHA256 = "1ad7ebbd7c6bb028b7d9a883ac72c549d218e1696cafa6a97c442f4e91af7e43"
OPS_HEADER = "mlx/backend/metal/kernels/ternary_ops.h"
OPS_HEADER_SHA256 = "afef5c3048963f9d3d4301ff40e8d5322dc07a3b3ec307c175262566091e5c11"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_TERNARY"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/ternary.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
DTYPES = (
    "bool_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "bfloat16",
    "complex64",
)
# ``instantiate_ternary_base`` covers every type; the work-per-thread ``*n``
# variants of ``instantiate_ternary_all`` skip the 8-byte types.
BASE_ONLY_DTYPES = ("uint64", "int64", "complex64")
ALL_KINDS = ("vn", "vsn", "svn")
BASE_KINDS = (
    "v",
    "v2",
    "vs",
    "vs2",
    "sv",
    "sv2",
    "gn2",
    "g1",
    "g2",
    "g3",
    "g1large",
    "g2large",
    "g3large",
    "gn4large",
)
ENTRIES = tuple(
    f"{kind}_Select{dtype}"
    for dtype in DTYPES
    for kind in (() if dtype in BASE_ONLY_DTYPES else ALL_KINDS) + BASE_KINDS
)
ITEM_SIZES = {
    "bool_": 1,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
    "complex64": 8,
}
# (kernel, BSCALAR, CSCALAR, explicit work per thread or None for
# ``WorkPerThread<T>::n``, generic tile width).
KINDS = {
    "v": ("ternary_v", False, False, 1, None),
    "vs": ("ternary_v", False, True, 1, None),
    "sv": ("ternary_v", True, False, 1, None),
    "vn": ("ternary_v", False, False, None, None),
    "vsn": ("ternary_v", False, True, None, None),
    "svn": ("ternary_v", True, False, None, None),
    "v2": ("ternary_v2", False, False, None, None),
    "vs2": ("ternary_v2", False, True, None, None),
    "sv2": ("ternary_v2", True, False, None, None),
    "g1": ("ternary_g_nd1", False, False, 1, None),
    "g1large": ("ternary_g_nd1", False, False, 1, None),
    "g2": ("ternary_g_nd2", False, False, 1, None),
    "g2large": ("ternary_g_nd2", False, False, 1, None),
    "g3": ("ternary_g_nd3", False, False, 1, None),
    "g3large": ("ternary_g_nd3", False, False, 1, None),
    "gn2": ("ternary_g", False, False, 1, 2),
    "gn4large": ("ternary_g", False, False, 1, 4),
}
# Element counts for the flat kernels, (x, y) grids for ``ternary_v2`` and
# shapes with per-operand strides (outermost first) for the strided kernels.
CASES = {
    "ternary_v": {"tail": {"size": 37}, "exact": {"size": 64}},
    "ternary_v2": {"tail": {"grid": [3, 2]}, "wide": {"grid": [4, 3]}},
    "ternary_g_nd1": {
        "strided": {"shape": [19], "strides": [[1], [2], [0]]},
        "permuted": {"shape": [11], "strides": [[3], [1], [2]]},
    },
    "ternary_g_nd2": {
        "transpose": {"shape": [3, 5], "strides": [[5, 1], [1, 3], [0, 1]]},
        "broadcast": {"shape": [4, 2], "strides": [[1, 4], [0, 0], [2, 1]]},
    },
    "ternary_g_nd3": {
        "permute": {"shape": [2, 2, 3], "strides": [[6, 3, 1], [1, 2, 4], [0, 0, 1]]},
        "broadcast": {"shape": [3, 1, 4], "strides": [[4, 4, 1], [0, 0, 1], [1, 0, 3]]},
    },
    "ternary_g": {
        "nd4": {
            "shape": [2, 3, 2, 5],
            "strides": [[30, 10, 5, 1], [0, 10, 1, 2], [15, 5, 0, 1]],
        },
        "nd5-broadcast": {
            "shape": [2, 2, 1, 3, 3],
            "strides": [[18, 9, 9, 3, 1], [0, 3, 0, 1, 6], [9, 0, 0, 0, 3]],
        },
    },
}
GUARD_COUNT = 16
DECOY_COUNT = 4
# OpenGL indexes 32-bit storage arrays. These host preconditions bound every
# 64-bit source index of the ``large``, ``v2`` and generic kernels to signed
# int range; each fixture index stays far below the bound.
OPENGL_INDEX_RANGE_ASSERTIONS = tuple(
    (expression, 0, 2**31 - 1)
    for expression in (
        "offset + i",
        "a_idx",
        "b_idx",
        "c_idx",
        "bidx",
        "cidx",
        "out_idx",
        "out_idx++",
        "idx.x",
        "idx.y",
        "idx.z",
    )
)
METAL_FORMATS = {
    "bool_": "B",
    "uint8": "B",
    "uint16": "H",
    "uint32": "I",
    "uint64": "Q",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "int64": "q",
    "float16": "H",
    "float32": "I",
    "bfloat16": "H",
}
# Reflected physical element types: narrow and 16-bit floating storage widens
# on OpenGL; DirectX keeps 16-bit integers and stores 16-bit floats as bits.
ELEMENT_TYPES = {
    "metal": {
        "bool_": "bool",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "complex64": "float32",
    },
    "directx": {
        "bool_": "uint32",
        "uint8": "uint32",
        "int8": "int32",
        "float16": "uint16",
        "bfloat16": "uint16",
        "complex64": "float32",
    },
    "opengl": {
        "bool_": "uint32",
        "uint8": "uint32",
        "uint16": "uint32",
        "int8": "int32",
        "int16": "int32",
        "float16": "float32",
        "bfloat16": "float32",
        "complex64": "float32",
    },
}
# Logical payloads: every integer extreme and sign boundary, and for the
# floating types signed zeros, infinities, quiet and signaling NaNs with
# payloads, subnormals and the largest finite values. Select must move every
# payload bit unchanged.
POOLS = {
    "bool_": [0, 1],
    "uint8": [0, 1, 0x7F, 0x80, 0xFF, 0x5A, 0xA5, 3],
    "uint16": [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x1234, 0xFEDC, 7],
    "uint32": [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x12345678, 0xFEDCBA98, 9],
    "uint64": [
        0,
        1,
        2**63 - 1,
        2**63,
        2**64 - 1,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        2**32,
    ],
    "int8": [0, 1, -1, 127, -128, 90, -91, 3],
    "int16": [0, 1, -1, 2**15 - 1, -(2**15), 4660, -4661, 7],
    "int32": [0, 1, -1, 2**31 - 1, -(2**31), 305419896, -305419897, 9],
    "int64": [
        0,
        1,
        -1,
        2**63 - 1,
        -(2**63),
        81985529216486895,
        -81985529216486896,
        2**32,
    ],
    "float16": [
        0x0000,
        0x8000,
        0x3C00,
        0xBC00,
        0x7C00,
        0xFC00,
        0x7E00,
        0x7D01,
        0xFE13,
        0x0001,
        0x83FF,
        0x7BFF,
    ],
    "float32": [
        0x00000000,
        0x80000000,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x7FA00001,
        0xFFC12345,
        0x00000001,
        0x807FFFFF,
        0x7F7FFFFF,
    ],
    "bfloat16": [
        0x0000,
        0x8000,
        0x3F80,
        0xBF80,
        0x7F80,
        0xFF80,
        0x7FC0,
        0x7F81,
        0xFFC5,
        0x0001,
        0x807F,
        0x7F7F,
    ],
    "complex64": [
        (0x3F800000, 0x80000000),
        (0x7FC00001, 0x40000000),
        (0x00000001, 0xFF800000),
        (0x7FA00001, 0x7F7FFFFF),
        (0x80000000, 0x00000000),
        (0xC0490FDB, 0x807FFFFF),
        (0x7F800000, 0xFFC12345),
    ],
}
GUARDS = {
    "bool_": 0x0B00B1E5,
    "uint8": 0xC3,
    "uint16": 0xC3C3,
    "uint32": 0xC3C3C3C3,
    "uint64": 0xC3C3C3C3C3C3C3C3,
    "int8": -61,
    "int16": -15421,
    "int32": -1010580541,
    "int64": -4340410370284600381,
    "float16": 0x6A15,
    "float32": 0x6A15BEEF,
    "bfloat16": 0x6A15,
    "complex64": (0x6A15BEEF, 0x6A15BEEF),
}


def _kind(entry):
    return entry.split("_Select", 1)[0]


def _dtype(entry):
    return entry.split("_Select", 1)[1]


def _kernel(entry):
    return KINDS[_kind(entry)][0]


def _work(entry):
    """Return the elements each thread of a flat kernel handles."""
    explicit = KINDS[_kind(entry)][3]
    return explicit if explicit is not None else 8 // ITEM_SIZES[_dtype(entry)]


def _element_type(target, dtype):
    return ELEMENT_TYPES[target].get(dtype, dtype)


def _half_to_float32_word(word):
    """Widen binary16 bits exactly, keeping NaN payloads and signaling bits."""
    sign = (word & 0x8000) << 16
    exponent = (word >> 10) & 0x1F
    mantissa = word & 0x3FF
    if exponent == 0x1F:
        return sign | 0x7F800000 | (mantissa << 13)
    if exponent == 0 and mantissa == 0:
        return sign
    value = struct.unpack("<e", struct.pack("<H", word))[0]
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _values(dtype, count, salt):
    pool = POOLS[dtype]
    return [
        pool[(index * 5 + salt * 3 + index // len(pool)) % len(pool)]
        for index in range(count)
    ]


def _conditions(count):
    return [((index * 7 + 3) % 5) < 2 for index in range(count)]


def _operand_offsets(shape, strides):
    coordinates = [()]
    for extent in shape:
        coordinates = [
            point + (index,) for point in coordinates for index in range(extent)
        ]
    return [sum(i * s for i, s in zip(point, strides)) for point in coordinates]


def _case(entry, name):
    """Return grid, constants, operand storage and expected logical output."""
    kernel, bscalar, cscalar, _, tile = KINDS[_kind(entry)]
    dtype = _dtype(entry)
    spec = CASES[kernel][name]
    work = _work(entry)
    constants = {}
    if kernel in {"ternary_v", "ternary_v2"}:
        if kernel == "ternary_v":
            size = spec["size"]
            grid = [math.ceil(size / work), 1, 1]
        else:
            threads = spec["grid"][0] * spec["grid"][1]
            # A short last tile exercises the tail loop; one element per
            # thread has no tail, so the grid then covers the size exactly.
            size = threads * work - (1 if name == "tail" and work > 1 else 0)
            grid = [spec["grid"][0], spec["grid"][1], 1]
        constants["size"] = size
        offsets = {
            "a": list(range(size)),
            "b": [0] * size if bscalar else list(range(size)),
            "c": [0] * size if cscalar else list(range(size)),
        }
    else:
        shape = spec["shape"]
        strides = dict(zip("abc", spec["strides"]))
        offsets = {name: _operand_offsets(shape, strides[name]) for name in "abc"}
        if kernel == "ternary_g":
            grid = [
                math.ceil(shape[-1] / tile),
                shape[-2],
                math.prod(shape[:-2]),
            ]
            constants["shape"] = list(shape)
            constants["ndim"] = len(shape)
        else:
            grid = list(reversed(shape)) + [1] * (3 - len(shape))
        for name in "abc":
            constants[f"{name}_strides"] = list(strides[name])
    count = len(offsets["a"])
    operands = {}
    for salt, name in enumerate("abc"):
        length = max(offsets[name]) + 1
        if name == "a":
            operands[name] = _conditions(length) + [True, False] * (DECOY_COUNT // 2)
        else:
            operands[name] = (
                _values(dtype, length, salt) + [GUARDS[dtype]] * DECOY_COUNT
            )
    expected = [
        (
            operands["b"][offsets["b"][index]]
            if operands["a"][offsets["a"][index]]
            else operands["c"][offsets["c"][index]]
        )
        for index in range(count)
    ]
    return {
        "grid": grid,
        "constants": constants,
        "operands": operands,
        "count": count,
        "expected": expected,
    }


def _runtime_value(target, dtype, values):
    """Encode logical values for the reflected physical storage of a target."""
    element_type = _element_type(target, dtype)
    count = len(values)
    if dtype == "complex64":
        return {
            "dtype": "float32",
            "shape": [count, 2],
            "encoding": FLOAT32_BITS,
            "values": [part for value in values for part in value],
        }
    if dtype == "float32":
        return {
            "dtype": "float32",
            "shape": [count],
            "encoding": FLOAT32_BITS,
            "values": list(values),
        }
    if dtype in {"float16", "bfloat16"}:
        if element_type == "float32":
            widened = [
                _half_to_float32_word(word) if dtype == "float16" else word << 16
                for word in values
            ]
            return {
                "dtype": "float32",
                "shape": [count],
                "encoding": FLOAT32_BITS,
                "values": widened,
            }
        if target == "directx" and dtype == "float16":
            # DirectX binary16 storage takes logical float16 words.
            return {
                "dtype": "float16",
                "shape": [count],
                "encoding": FLOAT16_BITS,
                "values": list(values),
            }
        if element_type == "uint16":
            return {"dtype": "uint16", "shape": [count], "values": list(values)}
        return {
            "dtype": element_type,
            "shape": [count],
            "encoding": FLOAT16_BITS if dtype == "float16" else BFLOAT16_BITS,
            "values": list(values),
        }
    if dtype == "bool_" and element_type == "bool":
        return {"dtype": "bool", "shape": [count], "values": [bool(v) for v in values]}
    return {"dtype": element_type, "shape": [count], "values": [int(v) for v in values]}


def _condition_value(target, values):
    if target == "metal":
        return {"dtype": "bool", "shape": [len(values)], "values": list(values)}
    return {
        "dtype": "uint32",
        "shape": [len(values)],
        "values": [int(v) for v in values],
    }


def _guard_words(dtype, target=None):
    """Return logical guard values that follow the output region."""
    if dtype == "bool_" and target in {None, "metal"}:
        # One-byte Metal bools hold canonical values; alternate them instead.
        return [index % 2 for index in range(GUARD_COUNT)]
    return [GUARDS[dtype]] * GUARD_COUNT


def _output_initial(target, dtype, case):
    """Fill the output so that every unwritten element differs from its result."""
    if dtype == "bool_" and target == "metal":
        return [1 - int(value) for value in case["expected"]]
    return [GUARDS[dtype]] * case["count"]


def _metal_bytes(dtype, values):
    if dtype == "complex64":
        return b"".join(struct.pack("<II", *value) for value in values)
    if dtype == "bool_":
        # One-byte Metal bools hold canonical values, exactly as the
        # translated dispatch uploads them; word-sized decoys become true.
        values = [bool(value) for value in values]
    return struct.pack(
        "<" + METAL_FORMATS[dtype] * len(values), *(int(value) for value in values)
    )


def _config(output, target, entry):
    assertions = ""
    if target == "opengl":
        for expression, low, high in OPENGL_INDEX_RANGE_ASSERTIONS:
            assertions += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = {low}
maximum = {high}
"""
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [1, 1, 1]
[project.source_options.metal]
max_template_specializations = 512
max_template_materialization_work = 65536
{assertions}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _case_contract():
    return {
        kernel: {name: dict(spec) for name, spec in cases.items()}
        for kernel, cases in CASES.items()
    }


def _dispatch_count():
    return sum(len(CASES[_kernel(entry)]) for entry in ENTRIES)


def test_ternary_entries_match_the_upstream_instantiation():
    assert len(ENTRIES) == 212 and len(set(ENTRIES)) == 212
    assert sum(1 for entry in ENTRIES if _kind(entry) in ALL_KINDS) == 30
    assert {
        _work(f"vn_Select{dtype}") for dtype in DTYPES if dtype not in BASE_ONLY_DTYPES
    } == {
        2,
        4,
        8,
    }
    assert _work("v2_Selectcomplex64") == 1 and _work("v_Selectuint8") == 1


def test_current_mlx_ternary_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-ternary-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "header": HEADER,
        "headerSha256": HEADER_SHA256,
        "opsHeader": OPS_HEADER,
        "opsHeaderSha256": OPS_HEADER_SHA256,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 49,
        "entryCount": 17832,
        "discoveryDiagnosticCount": 0,
    }
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    validation = contract["validationContract"]
    assert validation["workgroupSize"] == [1, 1, 1]
    assert validation["indexRangeAssertions"] == {
        "opengl": [
            {"expression": expression, "minimum": low, "maximum": high}
            for expression, low, high in OPENGL_INDEX_RANGE_ASSERTIONS
        ],
        "metal": [],
        "directx": [],
    }
    assert validation["cases"] == _case_contract()
    assert validation["outputGuardCount"] == GUARD_COUNT
    assert validation["inputDecoyCount"] == DECOY_COUNT
    assert validation["elementTypes"] == {
        target: {dtype: _element_type(target, dtype) for dtype in DTYPES}
        for target in TARGETS
    }
    assert validation["compilerValidationRequiredInCi"] == list(TARGETS)
    assert validation["runtimeRequiredInCi"] == list(TARGETS)
    assert validation["upstreamMetallibParityTargets"] == ["metal"]
    assert contract["scope"] == {
        "coveredEntryCount": 212,
        "runtimeCoveredEntryCount": 212,
        "dispatchCountByTarget": {target: _dispatch_count() for target in TARGETS},
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


@pytest.mark.parametrize("entry", ENTRIES)
def test_ternary_cases_cover_both_branches_and_payloads(entry):
    dtype = _dtype(entry)
    for name in CASES[_kernel(entry)]:
        case = _case(entry, name)
        assert len(case["expected"]) == case["count"] > 0
        conditions = case["operands"]["a"][:-DECOY_COUNT]
        assert True in conditions and False in conditions
        # Decoys follow every operand and never reach a valid output.
        if dtype != "bool_":
            assert GUARDS[dtype] not in case["expected"]
        if _kind(entry) in {"vs", "vsn", "vs2"}:
            assert len(case["operands"]["c"]) == 1 + DECOY_COUNT
        if _kind(entry) in {"sv", "svn", "sv2"}:
            assert len(case["operands"]["b"]) == 1 + DECOY_COUNT
    kernel = _kernel(entry)
    if kernel == "ternary_v" and _work(entry) > 1:
        # The 37-element case ends inside a tile.
        assert 37 % _work(entry) != 0
    if kernel == "ternary_v2" and _work(entry) > 1:
        assert _case(entry, "tail")["count"] % _work(entry) == _work(entry) - 1


@pytest.mark.parametrize("dtype", ("float16", "float32", "bfloat16"))
def test_ternary_floating_pools_hold_every_payload_class(dtype):
    pool = POOLS[dtype]
    bits = {"float16": 16, "float32": 32, "bfloat16": 16}[dtype]
    exponent_bits = {"float16": 5, "float32": 8, "bfloat16": 8}[dtype]
    mantissa_bits = bits - 1 - exponent_bits
    exponent_mask = ((1 << exponent_bits) - 1) << mantissa_bits
    quiet = 1 << (mantissa_bits - 1)
    nans = [
        w
        for w in pool
        if w & exponent_mask == exponent_mask and w & ((1 << mantissa_bits) - 1)
    ]
    assert any(w & quiet for w in nans) and any(not w & quiet for w in nans)
    assert any(w >> (bits - 1) for w in nans)
    assert any(0 < w & ((1 << (bits - 1)) - 1) < (1 << mantissa_bits) for w in pool)
    assert 0 in pool and 1 << (bits - 1) in pool


def test_half_widening_keeps_payloads():
    assert _half_to_float32_word(0x7D01) == 0x7FA02000
    assert _half_to_float32_word(0xFE13) == 0xFFC26000
    assert _half_to_float32_word(0x8000) == 0x80000000
    assert _half_to_float32_word(0x0001) == 0x33800000
    assert _half_to_float32_word(0x7BFF) == 0x477FE000


@pytest.mark.parametrize("target", TARGETS)
def test_ternary_configs_scope_opengl_index_preconditions(tmp_path, target):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, ENTRIES[0]), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert [item.to_json() for item in config.index_range_assertions] == (
        [
            {
                "source": SOURCE,
                "expression": expression,
                "minimum": low,
                "maximum": high,
            }
            for expression, low, high in OPENGL_INDEX_RANGE_ASSERTIONS
        ]
        if target == "opengl"
        else []
    )
    assert list(config.targets) == [target]


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native ternary gate")
    assert root_value and target in TARGETS
    for name in {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    for path, digest in (
        (SOURCE, SOURCE_SHA256),
        (HEADER, HEADER_SHA256),
        (OPS_HEADER, OPS_HEADER_SHA256),
    ):
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("ternary-metal-reference")
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


def _metal_constant_payloads(kernel, constants):
    if kernel == "ternary_v":
        return [struct.pack("<I", constants["size"])]
    if kernel == "ternary_v2":
        return [struct.pack("<q", constants["size"])]
    strides = [
        struct.pack(f"<{len(constants[name])}q", *constants[name])
        for name in ("a_strides", "b_strides", "c_strides")
    ]
    if kernel != "ternary_g":
        return strides
    shape = struct.pack(f"<{len(constants['shape'])}i", *constants["shape"])
    return [shape, *strides, struct.pack("<i", constants["ndim"])]


def _upstream_metal(work, reference, entry, case, name):
    runner, library = reference
    dtype = _dtype(entry)
    payloads = [
        struct.pack(f"<{len(case['operands']['a'])}B", *case["operands"]["a"]),
        _metal_bytes(dtype, case["operands"]["b"]),
        _metal_bytes(dtype, case["operands"]["c"]),
        _metal_bytes(
            dtype, _output_initial("metal", dtype, case) + _guard_words(dtype)
        ),
        *_metal_constant_payloads(_kernel(entry), case["constants"]),
    ]
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{name}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": case["grid"],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{name}-upstream")
    )
    for index, payload in enumerate(payloads):
        if index != 3:
            assert (readback / f"buffer-{index}.bin").read_bytes() == payload
    return device, (readback / "buffer-3.bin").read_bytes()


def _dispatch(descriptor, package, target, entry, case):
    dtype = _dtype(entry)
    kernel = _kernel(entry)
    guards = _guard_words(dtype, target)
    data = {
        "a": _condition_value(target, case["operands"]["a"]),
        "b": _runtime_value(target, dtype, case["operands"]["b"]),
        "c": _runtime_value(target, dtype, case["operands"]["c"]),
        "d": _runtime_value(
            target, dtype, _output_initial(target, dtype, case) + guards
        ),
    }
    constants = case["constants"]
    if kernel == "ternary_v":
        data["size"] = {"dtype": "uint32", "shape": [1], "values": [constants["size"]]}
    elif kernel == "ternary_v2":
        data["size"] = {"dtype": "int64", "shape": [1], "values": [constants["size"]]}
    else:
        for name in ("a_strides", "b_strides", "c_strides"):
            data[name] = {
                "dtype": "int64",
                "shape": [len(constants[name])],
                "values": list(constants[name]),
            }
        if kernel == "ternary_g":
            data["shape"] = {
                "dtype": "int32",
                "shape": [len(constants["shape"])],
                "values": list(constants["shape"]),
            }
            data["ndim"] = {
                "dtype": "int32",
                "shape": [1],
                "values": [constants["ndim"]],
            }
    inputs, observed, output_name = {}, [], None
    # DirectX constant members carry the sanitized entry name as a prefix.
    prefix = re.sub(r"_+", "_", entry + "_")
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        name = layout.get("memberName", binding["name"]).removeprefix(prefix)
        value = data[name]
        storage = layout.get("storageEncoding")
        if storage is None:
            assert layout["elementType"] == value["dtype"], (name, layout)
        else:
            assert layout["elementType"] == "uint16", (name, layout)
            assert storage == {
                "encoding": FLOAT16_BITS,
                "logicalElementType": value["dtype"],
            }
            assert value["encoding"] == FLOAT16_BITS
        inputs[binding["name"]] = value
        if name == "d":
            output_name = binding["name"]
        else:
            observed.append(binding["name"])
    assert output_name is not None and len(inputs) == len(data)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output_name: dict(data["d"])},
        {"workgroupCount": case["grid"], "workgroupSize": [1, 1, 1]},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    wanted = _runtime_value(target, dtype, case["expected"] + guards)["values"]
    return request, output_name, wanted, len(observed)


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_ternary_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    dtype = _dtype(entry)
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(prefix=".current-ternary-", dir=root) as directory:
        work = Path(directory)
        try:
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", target, entry), encoding="utf-8"
            )
            translation = translate_project(
                load_project_config(root, config),
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
            report = translation.to_json()
            translation.write_json(work / "report.json")
            validation = validate_project_report(work / "report.json")
            assert validation["success"] is True, validation
            assert report["diagnostics"] == []
            assert report["summary"]["translatedCount"] == 1
            assert report["summary"]["failedCount"] == 0
            (artifact,) = report["artifacts"]
            assert artifact["sourceHash"] == {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
            assert artifact["generatedHash"] == {
                "algorithm": "sha256",
                "value": record["sha256"],
            }
            assert artifact["generatedSizeBytes"] == record["sizeBytes"]
            (execution,) = artifact["execution"]["entryPoints"]
            assert execution["workgroupSize"] == [1, 1, 1]
            generated = root / artifact["path"]
            text = generated.read_text(encoding="utf-8")
            assert "unsupported" not in text
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name in CASES[_kernel(entry)]:
                    case = _case(entry, name)
                    request, output_name, wanted, observed = _dispatch(
                        descriptor, package, target, entry, case
                    )
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    (work / f"{name}-result.json").write_text(
                        json.dumps(
                            {
                                "status": result.status,
                                "outputs": result.outputs,
                                "details": result.details,
                            },
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    assert result.status == "ok"
                    output = result.outputs[output_name]
                    assert output["values"] == wanted, (name, output["values"][:8])
                    observations = result.details["inputObservations"]
                    assert len(observations) == observed
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, raw = _upstream_metal(
                            work, metal_reference, entry, case, name
                        )
                        expected_raw = _metal_bytes(
                            dtype, case["expected"] + _guard_words(dtype)
                        )
                        assert raw == expected_raw, name
                        upstream = {"device": device, "matched": True}
                    evidence.append(
                        {
                            "case": name,
                            "grid": case["grid"],
                            "constants": case["constants"],
                            "outputCount": case["count"],
                            "outputGuardCount": GUARD_COUNT,
                            "observedInputCount": observed,
                            "upstream": upstream,
                        }
                    )
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "indexRangeAssertions": (
                            [list(item) for item in OPENGL_INDEX_RANGE_ASSERTIONS]
                            if target == "opengl"
                            else []
                        ),
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_ternary_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX ternary native parity", group="elementwise"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/ternary",
        "--junitxml=mlx-current-results/ternary.xml",
        "demos/integrations/mlx/tests/kernels/test_current_ternary.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_ternary.py"
    )
