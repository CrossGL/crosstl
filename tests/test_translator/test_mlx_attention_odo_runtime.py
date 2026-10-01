"""Native row-dot execution for pinned MLX attention backward entries."""

import hashlib
import importlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    _complete_directx_register_layout,
    _prepare_directx_buffers,
    _prepare_directx_constants,
    _validate_directx_register_layout,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeConstantBinding,
    NativeRuntimeDispatchRequest,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
    RuntimeSpecializationConstant,
)
from demos.integrations.mlx.run_mlx_metal_host import run
from tests.test_translator.test_directx_float_atomics import (
    _compile as _compile_directx,
)
from tests.test_translator.test_mlx_gated_delta_metal import _translate_pinned
from tests.test_translator.test_mlx_gated_delta_runtime import (
    GUARD,
)
from tests.test_translator.test_mlx_gated_delta_runtime import (
    _compile as _compile_metal,
)

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_ATTENTION_RUNTIME"
COMPILE_ENV = "CROSTL_REQUIRE_MLX_ATTENTION_COMPILE"
TARGET_ENV = "CROSTL_MLX_ATTENTION_TARGET"
SOURCE = "mlx/backend/metal/kernels/scaled_dot_product_attention_vjp.metal"
SOURCE_SHA256 = "e4939faab6c21b0e733d60bb828338696ecafe737db4bbe63e819bd69f186661"
DIMENSIONS = (64, 72, 80, 96, 128, 192, 256, 512)
TYPES = {"float": "<f4", "float16_t": "<f2", "bfloat16_t": "<u2"}
CONFIGURATIONS = tuple(
    (dimension, dtype, "dyadic", 1 + index % 3 * 2)
    for index, dimension in enumerate(DIMENSIONS)
    for dtype in TYPES
) + (
    (80, "float", "fractional", 5),
    (512, "float16_t", "fractional", 3),
    (128, "bfloat16_t", "fractional", 5),
    (64, "float", "dyadic", 0),
)
HEADS = 6
ROOT = Path(__file__).resolve().parents[2]


def _dataset(np, dimension, dtype, pattern, length):
    indices = np.arange(HEADS * length * dimension, dtype=np.int64)
    buffers, values = [], []
    for multiplier, modulus, divisor in ((17, 63, 37), (23, 61, 41)):
        numerators = indices * multiplier % modulus - modulus // 2
        source = (numerators / (32 if pattern == "dyadic" else divisor)).astype("<f4")
        if dtype == "bfloat16_t":
            bits = source.view("<u4")
            encoded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
            decoded = (encoded.astype("<u4") << 16).view("<f4")
        else:
            encoded = source.astype(TYPES[dtype])
            decoded = encoded.astype("<f4")
        if pattern == "dyadic":
            np.testing.assert_array_equal(decoded, source)
        buffers.append(encoded)
        values.append(decoded.astype("<f8").reshape(HEADS, length, dimension))
    expected = np.sum(values[0] * values[1], axis=-1, dtype=np.float64)
    buffers.extend(
        (
            np.full(HEADS * length, np.nan, dtype="<f4"),
            np.asarray([length], dtype="<i4"),
        )
    )
    return buffers, expected


def _check_output(np, data, expected, *, exact):
    assert len(data) == expected.size * 4 + len(GUARD), "size"
    assert data[-len(GUARD) :] == GUARD, "guard"
    payload = data[: -len(GUARD)]
    actual = np.frombuffer(payload, dtype="<f4").reshape(expected.shape)
    assert np.isfinite(actual).all(), "nonfinite"
    if exact:
        assert payload == expected.astype("<f4").tobytes(), "values"
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    return {
        "values": actual.size,
        "sha256": hashlib.sha256(data).hexdigest(),
        "maximumAbsoluteError": (
            float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        ),
    }


@pytest.fixture(scope="session")
def attention_metal_runner(tmp_path_factory):
    assert sys.platform == "darwin", "Attention Metal execution requires macOS"
    directory = tmp_path_factory.mktemp("attention-metal-runner")
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


def _directx_request(np, buffers, artifact, module, dtype, length):
    source = artifact.read_text(encoding="utf-8")
    assert "cbuffer " in source and ": register(b3)" in source
    bindings = {}
    for slot, (name, buffer) in enumerate(zip(("o", "cot_o", "odo"), buffers)):
        element = (
            "float"
            if slot == 2
            else {"float": "float", "float16_t": "float16_t", "bfloat16_t": "uint16_t"}[
                dtype
            ]
        )
        type_name = f"{'RW' if slot == 2 else ''}StructuredBuffer<{element}>"
        assert (
            f"{type_name} {name} : register({'u' if slot == 2 else 't'}{slot});"
            in source
        )
        words = np.frombuffer(buffer.tobytes() + GUARD, dtype="<u4").tolist()
        bindings[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=slot,
                type_name=type_name,
                access="read_write" if slot == 2 else "read",
                metadata={"byteStride": buffer.itemsize},
            ),
            dtype="uint32",
            shape=(len(words),),
            value=words,
            source="expectedOutput" if slot == 2 else "input",
        )
    return NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=module.read_bytes(),
        buffers=bindings,
        constants={
            "qL": NativeRuntimeConstantBinding(
                name="qL",
                value=length,
                constant=RuntimeSpecializationConstant(
                    name="qL",
                    dtype="int32",
                    kind="constant-buffer",
                    metadata={"binding": 3, "byteOffset": 0},
                ),
            )
        },
        entry_point="CSMain",
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_size=(32, 1, 1),
            workgroup_count=(1, length + 2, HEADS),
        ),
    )


@pytest.fixture(scope="module", params=CONFIGURATIONS)
def compiled_attention(request, tmp_path_factory):
    if not any(os.environ.get(name) == "1" for name in (REQUIRE_ENV, COMPILE_ENV)):
        pytest.skip(f"set {REQUIRE_ENV}=1 to require pinned attention execution")
    target = os.environ.get(TARGET_ENV)
    assert target in {"directx", "metal"}, f"Set {TARGET_ENV} to directx or metal"
    if target == "directx":
        assert shutil.which("dxc"), "Attention DirectX validation requires DXC"
    else:
        assert sys.platform == "darwin", "Attention Metal validation requires macOS"
    np = importlib.import_module("numpy")
    dimension, dtype, pattern, length = request.param
    entry = f"sdpa_vjp_odo_{dtype}_{dimension}"
    directory = tmp_path_factory.mktemp("attention-odo")
    revision, original, generated = _translate_pinned(
        directory,
        target,
        entry=entry,
        source_path=SOURCE,
        source_sha256=SOURCE_SHA256,
        target_options=(
            {
                "software_subgroup_width": 32,
                "relative_wave_shuffle_out_of_range": "self",
            }
            if target == "directx"
            else {}
        ),
    )
    artifacts = {}
    if target == "directx":
        source = generated.read_text(encoding="utf-8")
        assert (
            "WaveActive" not in source
            and "GroupMemoryBarrierWithGroupSync();" in source
        )
        assert "[numthreads(32, 1, 1)]" in source
        artifacts["generated"] = _compile_directx(
            source, directory, profile="cs_6_2", flags=("-enable-16bit-types",)
        )
    else:
        for label, artifact in (("original", original), ("generated", generated)):
            artifacts[label] = (
                artifact,
                _compile_metal(artifact, original.parents[4], directory / label),
            )
    buffers, expected = _dataset(np, dimension, dtype, pattern, length)
    for slot, buffer in enumerate(buffers):
        (directory / f"input-{slot}.bin").write_bytes(buffer.tobytes() + GUARD)
    (directory / "expected.f64").write_bytes(expected.astype("<f8").tobytes())
    evidence = {
        "commit": revision,
        "sourceSha256": SOURCE_SHA256,
        "entry": entry,
        "target": target,
        "dimension": dimension,
        "dtype": dtype,
        "pattern": pattern,
        "length": length,
        "heads": HEADS,
        "workgroupSize": [32, 1, 1],
        "workgroupCount": [1, length + 2, HEADS],
        "rtol": 0 if pattern == "dyadic" else 1e-5,
        "atol": 0 if pattern == "dyadic" else 1e-5,
        "execution": "not-tested",
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
        "inputReadbacks": target == "metal",
        "results": {},
        "artifacts": {
            label: {
                "sourceSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                "source": str(artifact),
                "module": str(module),
            }
            for label, (artifact, module) in artifacts.items()
        },
    }
    native_request = None
    if target == "directx":
        artifact, module = artifacts["generated"]
        native_request = _directx_request(np, buffers, artifact, module, dtype, length)
        prepared = _validate_directx_register_layout(
            _complete_directx_register_layout(
                (
                    *_prepare_directx_buffers(native_request.buffers),
                    *_prepare_directx_constants(native_request.constants),
                )
            )
        )
        actual = {
            value.name: value for value in prepared if value.source != "descriptor-gap"
        }
        assert len(actual) == 4
        assert actual["constants[qL]"].payload[:4] == struct.pack("<i", length)
        for slot, name in enumerate(("o", "cot_o", "odo")):
            assert actual[name].payload == buffers[slot].tobytes() + GUARD
            assert actual[name].stride == buffers[slot].itemsize
            assert actual[name].binding_index == slot
        evidence["bindings"] = [
            value.to_json() for value in native_request.buffers.values()
        ]
        evidence["constants"] = [
            value.to_json() for value in native_request.constants.values()
        ]
    (directory / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    return np, directory, buffers, expected, artifacts, native_request, evidence


def test_pinned_attention_compiles(compiled_attention):
    for _artifact, module in compiled_attention[4].values():
        assert module.stat().st_size > 0


def test_pinned_attention_executes(compiled_attention, request):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native attention execution")
    np, directory, buffers, expected, artifacts, native_request, evidence = (
        compiled_attention
    )
    if native_request is not None:
        assert sys.platform == "win32", "DirectX attention execution requires Windows"
        state = SimpleNamespace(details={})
        outputs = DirectXComputeRuntime().dispatch(None, state, native_request)
        (directory / "readback.json").write_text(json.dumps(outputs), encoding="utf-8")
        assert set(outputs) == {"odo"}
        data = np.asarray(outputs["odo"]["values"], dtype="<u4").tobytes()
        (directory / "buffer-2.bin").write_bytes(data)
        evidence["results"]["generated"] = _check_output(
            np, data, expected, exact=evidence["pattern"] == "dyadic"
        )
        evidence["runtime"] = state.details
    else:
        runner = request.getfixturevalue("attention_metal_runner")
        payload = {
            "buffers": [str(directory / f"input-{slot}.bin") for slot in range(4)],
            "workgroupCount": evidence["workgroupCount"],
            "workgroupSize": [32, 1, 1],
            "simdWidth": 32,
        }
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for label, (_artifact, module) in artifacts.items():
            output = directory / label
            run(
                [runner, module, evidence["entry"], request_path, output],
                output,
                "dispatch",
                timeout=60,
            )
            for slot in (0, 1, 3):
                assert (output / f"buffer-{slot}.bin").read_bytes() == buffers[
                    slot
                ].tobytes() + GUARD, (label, slot)
            data = (output / "buffer-2.bin").read_bytes()
            evidence["results"][label] = _check_output(
                np, data, expected, exact=evidence["pattern"] == "dyadic"
            )
    evidence["execution"] = "passed"
    (directory / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )


@pytest.mark.parametrize("corruption", ["size", "guard", "value", "nan"])
def test_attention_output_verifier_rejects_corruption(corruption):
    np = pytest.importorskip("numpy")
    data = np.zeros(3, dtype="<f4").tobytes() + GUARD
    if corruption == "size":
        data = data[4:]
    elif corruption == "guard":
        data = data[:-4] + b"\x00" * 4
    else:
        data = (
            np.asarray(
                [np.nan if corruption == "nan" else 1.0, 0, 0], dtype="<f4"
            ).tobytes()
            + GUARD
        )
    with pytest.raises(AssertionError):
        _check_output(np, data, np.zeros((1, 3)), exact=False)
