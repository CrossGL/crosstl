"""Translate and execute every pinned MLX ``arange`` kernel on all three backends."""

import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from fractions import Fraction
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
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_fused_math import _oracle as _fma_bits
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/arange.metal"
SOURCE_SHA256 = "ca29a59005b5ad54dccc369542e32804229490a41eb30d568adcd913d1ee68d1"
HEADER = "mlx/backend/metal/kernels/arange.h"
HEADER_SHA256 = "f0dfe5df2e06e9028a4a9beebdb6b7fa639df43dcddc6291ad1c25ef44a1e61c"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ARANGE"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/arange.current-tree.translation.json"
)
DTYPES = (
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
)
ENTRIES = tuple(f"arange{dtype}" for dtype in DTYPES)
TARGETS = ("metal", "opengl", "directx")
# MLX dispatches one thread per output element; one-thread workgroups reproduce
# that grid exactly, so no generated thread writes past ``out``.
WORKGROUP_SIZE = (1, 1, 1)
# Metal fuses ``start + index * step`` even with -fno-fast-math. The generated
# OpenGL and DirectX artifacts reproduce that single rounding, including the
# Apple GPU binary32 subnormal flush, instead of rounding the product first.
CONTRACTION_PROFILE = "rne-flush"
CONTRACTION_TARGETS = ("opengl", "directx")
GUARD_COUNT = 16
INTEGER_BITS = {
    "uint8": 8,
    "uint16": 16,
    "uint32": 32,
    "uint64": 64,
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
}
FLOATING = ("float16", "float32", "bfloat16")
METAL_FORMATS = {
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
# Reflected physical element types. OpenGL widens narrow storage; DirectX keeps
# 16-bit integers and stores 16-bit floating values as bits.
ELEMENT_TYPES = {
    "metal": {},
    "directx": {
        "uint8": "uint32",
        "int8": "int32",
        "float16": "uint16",
        "bfloat16": "uint16",
    },
    "opengl": {
        "uint8": "uint32",
        "uint16": "uint32",
        "int8": "int32",
        "int16": "int32",
        "float16": "float32",
        "bfloat16": "float32",
    },
}


def _f32_word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _f32_value(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _half_value(word):
    return struct.unpack("<e", struct.pack("<H", word))[0]


def _half_word(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _bfloat_word(word32):
    """Round binary32 bits to bfloat16 with ties to even."""
    if (word32 & 0x7F800000) == 0x7F800000 and word32 & 0x007FFFFF:
        return (word32 >> 16) | 0x40
    return ((word32 + 0x7FFF + ((word32 >> 16) & 1)) >> 16) & 0xFFFF


GUARDS = {
    "uint8": 0xA5,
    "uint16": 0xA5C3,
    "uint32": 0xA5C3E1F7,
    "uint64": 0xA5C3E1F70B2D4F69,
    "int8": -0x5B,
    "int16": -0x5A3D,
    "int32": -0x5A3C1E09,
    "int64": -0x5A3C1E08F4D2B097,
    "float16": 0xC0DE,
    "float32": 0xC0DEC0DE,
    "bfloat16": 0xC0DE,
}


def _int_word(dtype, value):
    bits = INTEGER_BITS[dtype]
    value %= 1 << bits
    if dtype.startswith("int") and value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


# name -> (start, step, count). Integer values are logical integers; floating
# values are the logical bit patterns of the dtype.
CASES = {
    "uint8": {"wrap": (250, 3, 300), "down": (7, 255, 300), "unit": (0, 1, 4099)},
    "uint16": {
        "wrap": (65530, 7, 300),
        "down": (11, 65535 - 2, 300),
        "unit": (0, 1, 4099),
    },
    "uint32": {
        "wrap": (0xFFFFFF00, 0x10001, 4099),
        "down": (1000, 0xFFFFFFF9, 300),
        "unit": (0, 1, 300),
    },
    "uint64": {
        "wide": ((1 << 40) + 11, (1 << 33) + 7, 4099),
        "wrap": ((1 << 64) - 5, 3, 300),
        "down": (1 << 63, (1 << 64) - 0x100000001, 300),
    },
    "int8": {"wrap": (120, 3, 300), "down": (-100, -7, 300), "unit": (0, 1, 4099)},
    "int16": {
        "wrap": (32760, 7, 300),
        "down": (-32700, -301, 300),
        "unit": (-2048, 1, 4099),
    },
    "int32": {
        "wrap": (2**31 - 300, 0x10001, 4099),
        "down": (-(2**31) + 5, -7, 300),
        "unit": (-2048, 1, 300),
    },
    "int64": {
        "wide": (-(2**62), (1 << 50) + 3, 4099),
        "down": (2**62, -((1 << 33) + 7), 4099),
        "unit": (-(2**40), 1, 300),
    },
    # 0.1 + index * 0.3: about a quarter of the separately rounded results
    # differ from the fused results, so the case separates the two models.
    "float32": {
        "contracted": (_f32_word(0.1), _f32_word(0.3), 4096),
        "descending": (_f32_word(1000.5), _f32_word(-0.0137), 4096),
        "overflow": (_f32_word(3.0e38), _f32_word(1.0e35), 4096),
        "flush": (0x00000001, 0x00000003, 64),
        # -2^-126 + 1.5 * 2^-126 is subnormal before rounding and flushes to +0.
        "underflow": (0x80800000, 0x00C00000, 64),
    },
    "float16": {
        "contracted": (_half_word(0.1), _half_word(0.3), 4096),
        "subnormal": (0x0001, 0x0003, 4096),
        "overflow": (_half_word(65000.0), _half_word(0.5), 2048),
        "index": (0x0000, 0x3C00, 65535),
    },
    "bfloat16": {
        "contracted": (
            _bfloat_word(_f32_word(0.1)),
            _bfloat_word(_f32_word(0.3)),
            4096,
        ),
        "flush": (0x0001, 0x0003, 4096),
        "descending": (_bfloat_word(_f32_word(1024.0)), 0xBDCD, 4096),
        "index": (0x0000, 0x3F80, 65535),
    },
}


def _dtype(entry):
    return entry[len("arange") :]


def _element_type(target, dtype):
    return ELEMENT_TYPES[target].get(dtype, dtype)


def _half_from_index(index):
    """Convert an unsigned integer to binary16 with ties to even."""
    return _round_half(Fraction(index), False)


def _round_half(value, negative_zero):
    """Round an exact rational to binary16 bits with gradual subnormals."""
    if value == 0:
        return 0x8000 if negative_zero else 0
    sign = 0x8000 if value < 0 else 0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    quantum = max(exponent, -14) - 10
    scaled = magnitude / Fraction(2) ** quantum
    rounded, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (twice == scaled.denominator and rounded & 1):
        rounded += 1
    result = Fraction(rounded) * Fraction(2) ** quantum
    if result >= 65536:
        return sign | 0x7C00
    if result == 0:
        return sign
    return sign | _half_word(float(result))


def _half_expected(start, step, count):
    start_value = Fraction(_half_value(start))
    step_value = _half_value(step)
    values = []
    for index in range(count):
        converted = _half_from_index(index)
        if converted == 0x7C00:
            assert step_value > 0 and math.isfinite(_half_value(start))
            values.append(0x7C00)
            continue
        product = Fraction(_half_value(converted)) * Fraction(step_value)
        product_negative = step >> 15 == 1
        negative_zero = (
            product == 0 and start_value == 0 and product_negative and start >> 15 == 1
        )
        values.append(_round_half(product + start_value, negative_zero))
    return values


def _float32_expected(start, step, count):
    # Metal fuses the multiply-add and flushes binary32 subnormals.
    return [
        _fma_bits(_f32_word(float(index)), step, start, True) for index in range(count)
    ]


def _bfloat_expected(start, step, count):
    # Metal fuses bfloat arithmetic in binary32 (subnormals flushed), then
    # rounds the binary32 result to bfloat16 once.
    values = []
    for index in range(count):
        converted = _bfloat_word(_f32_word(float(index)))
        fused = _fma_bits(converted << 16, step << 16, start << 16, True)
        values.append(_bfloat_word(fused))
    return values


def _expected(dtype, case):
    start, step, count = CASES[dtype][case]
    if dtype in INTEGER_BITS:
        return [_int_word(dtype, start + index * step) for index in range(count)]
    return {
        "float16": _half_expected,
        "float32": _float32_expected,
        "bfloat16": _bfloat_expected,
    }[dtype](start, step, count)


def _runtime_value(target, dtype, values, storage_encoding=None):
    """Encode logical integers or floating bit patterns for physical storage."""
    element_type = _element_type(target, dtype)
    if dtype in INTEGER_BITS:
        assert storage_encoding is None
        return {"dtype": element_type, "shape": [len(values)], "values": list(values)}
    if element_type == "float32":
        assert storage_encoding is None
        if dtype == "float16":
            words = [_f32_word(_half_value(word)) for word in values]
        elif dtype == "bfloat16":
            words = [word << 16 for word in values]
        else:
            words = list(values)
        return {
            "dtype": "float32",
            "shape": [len(values)],
            "encoding": FLOAT32_BITS,
            "values": words,
        }
    if storage_encoding is not None or (
        dtype == "float16" and element_type == "float16"
    ):
        # Metal and DirectX binary16 storage take logical float16 words.
        assert dtype == "float16"
        if storage_encoding is not None:
            assert element_type == "uint16"
            assert storage_encoding == {
                "encoding": FLOAT16_BITS,
                "logicalElementType": "float16",
            }
        return {
            "dtype": "float16",
            "shape": [len(values)],
            "encoding": FLOAT16_BITS,
            "values": list(values),
        }
    if element_type == "uint16":
        return {"dtype": "uint16", "shape": [len(values)], "values": list(values)}
    assert element_type == "bfloat16" == dtype, (element_type, dtype)
    return {
        "dtype": "bfloat16",
        "shape": [len(values)],
        "encoding": BFLOAT16_BITS,
        "values": list(values),
    }


def _decode(target, dtype, words):
    """Return logical integers or floating bit patterns from physical words."""
    if _element_type(target, dtype) != "float32" or dtype == "float32":
        return list(words)
    if dtype == "float16":
        decoded = [_half_word(_f32_value(word)) for word in words]
        # Widened storage must hold exactly representable binary16 values.
        assert [_f32_word(_half_value(word)) for word in decoded] == list(words)
        return decoded
    assert all(word & 0xFFFF == 0 for word in words), words[:8]
    return [word >> 16 for word in words]


def _metal_bytes(dtype, values):
    return struct.pack("<" + METAL_FORMATS[dtype] * len(values), *values)


def _config(output, target, entry):
    options = ""
    if target in CONTRACTION_TARGETS:
        options = f"""
[project.source_options.metal.target_options.{target}]
floating_contraction_profile = "{CONTRACTION_PROFILE}"
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
"{entry}" = [{", ".join(str(item) for item in WORKGROUP_SIZE)}]
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _expected_validation_contract():
    return {
        "workgroupSize": list(WORKGROUP_SIZE),
        "floatingContractionProfile": {
            target: CONTRACTION_PROFILE if target in CONTRACTION_TARGETS else None
            for target in TARGETS
        },
        "cases": {
            dtype: {
                name: {"start": start, "step": step, "count": count}
                for name, (start, step, count) in CASES[dtype].items()
            }
            for dtype in DTYPES
        },
        "floatingValueEncoding": "logical bit patterns",
        "outputGuardCount": GUARD_COUNT,
        "observedInputCount": 2,
        "elementTypes": {
            target: {dtype: _element_type(target, dtype) for dtype in DTYPES}
            for target in TARGETS
        },
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def test_current_mlx_arange_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-arange-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "header": HEADER,
        "headerSha256": HEADER_SHA256,
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
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == {
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "caseCount": sum(len(CASES[dtype]) for dtype in DTYPES),
        "valueCountByTarget": {
            target: sum(case[2] for dtype in DTYPES for case in CASES[dtype].values())
            for target in TARGETS
        },
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


@pytest.mark.parametrize("dtype", DTYPES)
def test_arange_cases_cover_wrapping_and_rounding(dtype):
    for name, (start, step, count) in CASES[dtype].items():
        expected = _expected(dtype, name)
        assert len(expected) == count <= 65535
        if dtype in INTEGER_BITS:
            bits = INTEGER_BITS[dtype]
            low = -(1 << (bits - 1)) if dtype.startswith("int") else 0
            assert all(low <= value < low + (1 << bits) for value in expected)
            assert _int_word(dtype, start) == start and _int_word(dtype, step) == step
        else:
            assert all(
                0 <= value < (1 << (32 if dtype == "float32" else 16))
                for value in expected
            )
    if dtype in INTEGER_BITS:
        bits = INTEGER_BITS[dtype]
        wraps = any(
            any(
                start + index * step != _int_word(dtype, start + index * step)
                for index in range(count)
            )
            for start, step, count in CASES[dtype].values()
        )
        # Narrow and unsigned arithmetic wraps modulo the width in MLX's kernel.
        # int64 arithmetic is signed in Metal, so its cases avoid overflow.
        assert wraps is (dtype != "int64")
        if bits == 64:
            # The 64-bit product must not be formed in 32 bits.
            start, step, count = next(iter(CASES[dtype].values()))
            assert any(index * step >= 1 << 32 for index in range(count))


def _separately_rounded(dtype, case):
    """Round the product before the addition, which Metal does not do."""
    start, step, count = CASES[dtype][case]
    values = []
    for index in range(count):
        if dtype == "float32":
            product = _fma_bits(_f32_word(float(index)), step, 0, True)
            values.append(_fma_bits(start, 0x3F800000, product, True))
        elif dtype == "float16":
            converted = _half_from_index(index)
            product = _round_half(
                Fraction(_half_value(converted)) * Fraction(_half_value(step)), False
            )
            values.append(
                _round_half(
                    Fraction(_half_value(product)) + Fraction(_half_value(start)), False
                )
            )
        else:
            converted = _bfloat_word(_f32_word(float(index)))
            product = _bfloat_word(_fma_bits(converted << 16, step << 16, 0, True))
            values.append(
                _bfloat_word(_fma_bits(start << 16, 0x3F800000, product << 16, True))
            )
    return values


@pytest.mark.parametrize("dtype", FLOATING)
def test_arange_floating_cases_separate_fused_rounding(dtype):
    expected = _expected(dtype, "contracted")
    separate = _separately_rounded(dtype, "contracted")
    differing = sum(left != right for left, right in zip(expected, separate))
    # Measured against the pinned Metal library: 1269, 906 and 227 of 4096.
    assert differing == {"float32": 1269, "float16": 906, "bfloat16": 227}[dtype]


def test_arange_floating_edge_expectations():
    assert set(_expected("float32", "flush")) == {0}
    overflow = _expected("float32", "overflow")
    assert overflow[0] == _f32_word(3.0e38) and overflow[-1] == 0x7F800000
    assert _expected("float16", "subnormal")[:3] == [0x0001, 0x0004, 0x0007]
    index = _expected("float16", "index")
    assert index[2049] == _half_word(2048.0) and index[2051] == _half_word(2052.0)
    assert index[65519] == 0x7BFF and index[65520:] == [0x7C00] * 15
    assert _expected("float16", "overflow")[-1] == 0x7C00
    assert set(_expected("bfloat16", "flush")) == {0}
    assert _expected("bfloat16", "index")[257] == _bfloat_word(_f32_word(256.0))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_arange_runtime_values_match_physical_storage(target, dtype):
    start, step, count = next(iter(CASES[dtype].values()))
    encoding = (
        {"encoding": FLOAT16_BITS, "logicalElementType": "float16"}
        if target == "directx" and dtype == "float16"
        else None
    )
    value = _runtime_value(target, dtype, [start, step], encoding)
    if dtype == "float16" and target != "opengl":
        assert value["dtype"] == "float16" and value["encoding"] == FLOAT16_BITS
    else:
        assert value["dtype"] == _element_type(target, dtype)
    words = value["values"]
    assert _decode(target, dtype, words) == [start, step]
    if target == "metal":
        size = struct.calcsize("<" + METAL_FORMATS[dtype])
        assert len(_metal_bytes(dtype, [start, step])) == 2 * size


@pytest.mark.parametrize("target", TARGETS)
def test_arange_configs_scope_contraction_profile(tmp_path, target):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, ENTRIES[0]), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert list(config.targets) == [target]
    options = config.source_options.get("metal", {})
    assert options.get("floating_contraction_profile") is None
    assert options.get("target_options", {}).get(target, {}).get(
        "floating_contraction_profile"
    ) == (CONTRACTION_PROFILE if target in CONTRACTION_TARGETS else None)
    assert [item.to_json() for item in config.index_range_assertions] == []


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native arange gate")
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
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    assert hashlib.sha256((root / HEADER).read_bytes()).hexdigest() == HEADER_SHA256
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("arange-metal-reference")
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


def _upstream_metal(work, reference, entry, dtype, case, name):
    runner, library = reference
    start, step, count = CASES[dtype][case]
    payloads = [
        _metal_bytes(dtype, [start]),
        _metal_bytes(dtype, [step]),
        _metal_bytes(dtype, [GUARDS[dtype]] * (count + GUARD_COUNT)),
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
                "workgroupCount": [count, 1, 1],
                "workgroupSize": list(WORKGROUP_SIZE),
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, output], work, f"{name}-upstream")
    )
    raw = (output / "buffer-2.bin").read_bytes()
    for index in (0, 1):
        assert (output / f"buffer-{index}.bin").read_bytes() == payloads[index]
    size = struct.calcsize("<" + METAL_FORMATS[dtype])
    return device, list(
        struct.unpack("<" + METAL_FORMATS[dtype] * (len(raw) // size), raw)
    )


def _dispatch(descriptor, package, target, entry, dtype, case):
    start, step, count = CASES[dtype][case]
    data = {
        "start": [start],
        "step": [step],
        "out_": [GUARDS[dtype]] * (count + GUARD_COUNT),
    }
    inputs, observed, output_name = {}, [], None
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        # DirectX constant members carry the sanitized entry name as a prefix.
        name = layout.get("memberName", binding["name"]).removeprefix(
            re.sub(r"_+", "_", entry) + "_"
        )
        storage = layout.get("storageEncoding")
        value = _runtime_value(target, dtype, data[name], storage)
        if storage is None:
            assert layout["elementType"] == value["dtype"], (name, layout)
        else:
            assert layout["elementType"] == "uint16", (name, layout)
            assert value["encoding"] == FLOAT16_BITS
        inputs[binding["name"]] = value
        if name == "out_":
            output_name = binding["name"]
        else:
            observed.append(binding["name"])
    assert output_name is not None and len(inputs) == 3 and len(observed) == 2
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output_name: dict(inputs[output_name])},
        {"workgroupCount": [count, 1, 1], "workgroupSize": list(WORKGROUP_SIZE)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output_name, inputs[output_name]["values"][count:]


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_arange_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    dtype = _dtype(entry)
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(prefix=".current-arange-", dir=root) as directory:
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
            check_generated_artifact(
                artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
            )
            assert artifact["provenance"].get("floatingContractionProfile") == (
                CONTRACTION_PROFILE if target in CONTRACTION_TARGETS else None
            )
            (execution,) = artifact["execution"]["entryPoints"]
            assert execution["workgroupSize"] == list(WORKGROUP_SIZE)
            generated = root / artifact["path"]
            text = generated.read_text(encoding="utf-8")
            assert "unsupported" not in text
            if target in CONTRACTION_TARGETS:
                # GLSL drops the reserved leading double underscore.
                assert ("crossgl_metal_fmuladd_float" in text) is (
                    dtype in {"float32", "bfloat16"}
                )
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name in CASES[dtype]:
                    expected = _expected(dtype, name)
                    request, output_name, guards = _dispatch(
                        descriptor, package, target, entry, dtype, name
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
                    count = len(expected)
                    assert output["values"][count:] == guards
                    actual = _decode(target, dtype, output["values"][:count])
                    mismatches = [
                        (index, got, want)
                        for index, (got, want) in enumerate(zip(actual, expected))
                        if got != want
                    ]
                    assert not mismatches, mismatches[:16]
                    observations = result.details["inputObservations"]
                    assert len(observations) == 2
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, values = _upstream_metal(
                            work, metal_reference, entry, dtype, name, name
                        )
                        assert values[count:] == [GUARDS[dtype]] * GUARD_COUNT
                        assert values[:count] == expected
                        upstream = {"device": device, "matched": True}
                    evidence.append(
                        {
                            "case": name,
                            "start": CASES[dtype][name][0],
                            "step": CASES[dtype][name][1],
                            "count": count,
                            "outputGuardCount": GUARD_COUNT,
                            "observedInputCount": 2,
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
                        "floatingContractionProfile": (
                            CONTRACTION_PROFILE
                            if target in CONTRACTION_TARGETS
                            else None
                        ),
                        "cases": evidence,
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_arange_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Prove current MLX arange native parity"
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "--timeout-seconds 900",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/arange",
        "--junitxml=mlx-current-results/arange.xml",
        "demos/integrations/mlx/tests/kernels/test_current_arange.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_arange.py"
    )
    upload = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX arange native parity") < (
        workflow.index("Upload current MLX runtime diagnostics")
    )
