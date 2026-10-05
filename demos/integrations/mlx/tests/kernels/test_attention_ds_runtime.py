"""Native parity for pinned attention derivatives with mixed parameters and aliasing."""

import hashlib
import importlib
import json
import os
import shutil
import struct
import subprocess
import sys

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
    RuntimeAllocationView,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
    RuntimeSpecializationConstant,
)
from crosstl.translator.resource_storage import (
    BINARY16_STORAGE,
    parse_resource_storage_header,
)
from demos.integrations.mlx.run_metal_host import run
from demos.integrations.mlx.tests.kernels.test_attention_odo_runtime import (
    SOURCE,
    SOURCE_SHA256,
    attention_metal_runner,
)
from demos.integrations.mlx.tests.kernels.test_gated_delta_metal import (
    _translate_pinned,
)
from demos.integrations.mlx.tests.kernels.test_gated_delta_runtime import (
    GUARD,
)
from demos.integrations.mlx.tests.kernels.test_gated_delta_runtime import (
    _compile as compile_metal,
)
from tests.test_translator.test_directx_float_atomics import _compile as compile_directx

attention_metal_runner = attention_metal_runner
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_ATTENTION_DS_RUNTIME"
COMPILE_ENV = "CROSTL_REQUIRE_MLX_ATTENTION_DS_COMPILE"
TARGET_ENV = "CROSTL_MLX_ATTENTION_DS_TARGET"
PARAMETERS = ("bq", "bk", "qL", "i0", "j0", "scale", "diag_off", "causal")
NAMES = ("S", "dP", "lse", "odo", "dS", "P")
TYPES = {"float": "<f4", "float16_t": "<f2", "bfloat16_t": "<u2"}
HEADS, BQ, BK, QL, I0, J0, SCALE, DIAG = 3, 9, 35, 17, 4, 3, 0.5, -1


def _encode(np, values, dtype):
    values = np.asarray(values, dtype="<f4")
    if dtype == "bfloat16_t":
        words = values.view("<u4")
        return ((words + 0x7FFF + ((words >> 16) & 1)) >> 16).astype("<u2")
    return values.astype(TYPES[dtype])


def _decode(np, values, dtype):
    if dtype == "bfloat16_t":
        return (values.astype("<u4") << 16).view("<f4")
    return values.astype("<f4")


def _dataset(np, dtype, pattern, causal, alias):
    indices = np.arange(HEADS * BQ * BK).reshape(HEADS, BQ, BK)
    rows = np.arange(HEADS * BQ).reshape(HEADS, BQ)
    scores = np.broadcast_to(((rows % 5 - 2) / 4)[:, :, None], indices.shape)
    if pattern == "fractional":
        scores = (indices % 29 - 14) / 31
    scores = _encode(np, scores, dtype)
    dp = _encode(np, (indices % 13 - 6) / 8, dtype)
    lse = np.full((HEADS, QL), -0.875, dtype="<f4")
    odo = np.asarray(
        (np.arange(HEADS * QL).reshape(HEADS, QL) % 7 - 3) / 4, dtype="<f4"
    )
    lse[:, I0 : I0 + BQ] = _decode(np, scores, dtype)[:, :, 0] * SCALE
    lse[:, I0 : I0 + BQ][rows % 4 == 0] = -np.inf
    dead = np.isneginf(lse[:, I0 : I0 + BQ])[:, :, None]
    masked = dead | (
        (np.arange(BK)[None, None, :] + J0 > I0 + np.arange(BQ)[None, :, None] + DIAG)
        & bool(causal)
    )
    logits = _decode(np, scores, dtype).astype("<f8") * SCALE - np.where(
        dead, 0, lse[:, I0 : I0 + BQ, None]
    ).astype("<f8")
    probability = np.where(masked, 0, np.exp(logits))
    derivative = (
        probability
        * (
            _decode(np, dp, dtype).astype("<f8")
            - odo[:, I0 : I0 + BQ, None].astype("<f8")
        )
        * SCALE
    )
    expected = [_encode(np, value, dtype) for value in (derivative, probability)]
    buffers = [
        scores,
        dp,
        lse,
        odo,
        scores.copy() if alias else _encode(np, np.full(indices.shape, -123), dtype),
        _encode(np, np.full(indices.shape, -456), dtype),
    ]
    parameters = dict(zip(PARAMETERS, (BQ, BK, QL, I0, J0, SCALE, DIAG, causal)))
    return buffers, expected, parameters


def _check_output(np, data, reference, dtype, *, exact):
    size = reference.nbytes
    padding = b"\x00" * (-(size + len(GUARD)) % 4)
    assert len(data) == size + len(GUARD) + len(padding), "size"
    assert data[size:] == GUARD + padding, "guard"
    payload = data[:size]
    actual = np.frombuffer(payload, dtype=reference.dtype).reshape(reference.shape)
    actual = _decode(np, actual, dtype).astype("<f8")
    expected = _decode(np, reference, dtype).astype("<f8")
    assert np.isfinite(actual).all(), "nonfinite"
    if exact:
        assert payload == reference.tobytes(), "values"
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    return {
        "values": actual.size,
        "sha256": hashlib.sha256(data).hexdigest(),
        "maximumAbsoluteError": float(np.max(np.abs(actual - expected))),
    }


def _payload(data):
    payload = data + GUARD
    return payload + b"\x00" * (-len(payload) % 4)


def _directx_request(np, artifact, module, buffers, parameters, dtype, alias):
    source = artifact.read_text(encoding="utf-8")
    assert parse_resource_storage_header(source) == (
        {name: BINARY16_STORAGE for name in ("S", "dP", "dS", "P")}
        if dtype == "float16_t"
        else {}
    )
    assert "ConstantBuffer<SDPAVJPTileParams> p : register(b6);" in source
    bindings = {}
    for slot, (name, buffer) in enumerate(zip(NAMES, buffers)):
        element = (
            "uint16_t"
            if dtype in {"float16_t", "bfloat16_t"} and slot not in (2, 3)
            else "float"
        )
        type_name = f"{'RW' if slot >= 4 else ''}StructuredBuffer<{element}>"
        assert (
            f"{type_name} {name} : register({'u' if slot >= 4 else 't'}{slot});"
            in source
        )
        # Word transport does not change the shader's 16-bit structured stride.
        # Pad after the guard so odd element counts keep their complete payload.
        payload = _payload(buffer.tobytes())
        words = np.frombuffer(payload, dtype="<u4").tolist()
        bindings[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=slot,
                type_name=type_name,
                access="read_write" if slot >= 4 else "read",
                metadata={"byteStride": buffer.itemsize},
            ),
            dtype="uint32",
            shape=(len(words),),
            value=words,
            source="expectedOutput" if slot >= 4 else "input",
            allocation=(
                RuntimeAllocationView(
                    allocation_id="scores",
                    byte_length=len(payload),
                    allocation_byte_length=len(payload),
                )
                if alias and slot in (0, 4)
                else None
            ),
        )
    constants = {
        name: NativeRuntimeConstantBinding(
            name=name,
            value=value,
            constant=RuntimeSpecializationConstant(
                name=name,
                dtype="float32" if name == "scale" else "int32",
                kind="constant-buffer",
                metadata={"binding": 6, "byteOffset": index * 4},
            ),
        )
        for index, (name, value) in enumerate(parameters.items())
    }
    prepared = _complete_directx_register_layout(
        (*_prepare_directx_buffers(bindings), *_prepare_directx_constants(constants))
    )
    _validate_directx_register_layout(prepared)
    parameter = next(
        value
        for value in prepared
        if value.namespace == "cbv" and value.binding_index == 6
    )
    assert parameter.payload == struct.pack("<5if2i", *parameters.values())
    for name, buffer in zip(NAMES, buffers):
        value = next(item for item in prepared if item.name == name)
        assert value.stride == buffer.itemsize
        assert value.payload == _payload(buffer.tobytes())
    return NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=module.read_bytes(),
        buffers=bindings,
        constants=constants,
        entry_point="CSMain",
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_size=(32, 8, 1),
            workgroup_count=(2, 2, HEADS),
        ),
    )


@pytest.fixture(scope="module", params=tuple(TYPES))
def compiled_derivative(request, tmp_path_factory):
    if not any(os.environ.get(name) == "1" for name in (REQUIRE_ENV, COMPILE_ENV)):
        pytest.skip(f"set {REQUIRE_ENV}=1 to require pinned derivative execution")
    target = os.environ.get(TARGET_ENV)
    assert target in {"metal", "directx"}, f"Set {TARGET_ENV} to metal or directx"
    assert sys.platform == "darwin" if target == "metal" else shutil.which("dxc")
    dtype = request.param
    entry = f"sdpa_vjp_ds_{dtype}"
    directory = tmp_path_factory.mktemp("attention-ds-compiled")
    revision, original, generated = _translate_pinned(
        directory,
        target,
        entry=entry,
        source_path=SOURCE,
        source_sha256=SOURCE_SHA256,
        workgroup_size=(32, 8, 1),
    )
    if target == "directx":
        artifact, module = compile_directx(
            generated.read_text(),
            directory,
            profile="cs_6_2",
            flags=("-enable-16bit-types",),
        )
        artifacts = {"generated": (artifact, module)}
    else:
        artifacts = {
            label: (
                artifact,
                compile_metal(artifact, original.parents[4], directory / label),
            )
            for label, artifact in (("original", original), ("generated", generated))
        }
    return target, dtype, entry, revision, artifacts


def test_pinned_attention_derivative_compiles(compiled_derivative):
    for artifact, module in compiled_derivative[-1].values():
        assert artifact.is_file() and module.stat().st_size > 0


@pytest.fixture
def derivative_case(compiled_derivative, request, tmp_path):
    np = importlib.import_module("numpy")
    target, dtype, entry, revision, artifacts = compiled_derivative
    pattern, causal, alias = request.param
    buffers, expected, parameters = _dataset(np, dtype, pattern, causal, alias)
    payloads = [buffer.tobytes() for buffer in buffers] + [
        struct.pack("<5if2i", *parameters.values())
    ]
    for slot, payload in enumerate(payloads):
        (tmp_path / f"input-{slot}.bin").write_bytes(_payload(payload))
    for slot, reference in zip((4, 5), expected):
        (tmp_path / f"expected-{slot}.bin").write_bytes(reference.tobytes())
    evidence = {
        "commit": revision,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "entry": entry,
        "target": target,
        "dtype": dtype,
        "pattern": pattern,
        "causal": causal,
        "alias": alias,
        "parameters": parameters,
        "heads": HEADS,
        "workgroupSize": [32, 8, 1],
        "workgroupCount": [2, 2, HEADS],
        "rtol": 0 if pattern == "dyadic" else 1e-5,
        "atol": 0 if pattern == "dyadic" else 1e-5,
        "inputReadbacks": target == "metal",
        "execution": "not-tested",
        "results": {},
        "transportWordBytes": 4,
        "payloadBytes": [len(value) for value in payloads],
        "tailPaddingBytes": [-len(value) % 4 for value in payloads],
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
        "artifacts": {
            label: {
                "source": str(artifact),
                "module": str(module),
                "sourceSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            }
            for label, (artifact, module) in artifacts.items()
        },
    }
    native = None
    if target == "directx":
        native = _directx_request(
            np, *artifacts["generated"], buffers, parameters, dtype, alias
        )
        evidence["bindings"] = [value.to_json() for value in native.buffers.values()]
        evidence["constants"] = [value.to_json() for value in native.constants.values()]
    (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
    return np, tmp_path, evidence, artifacts, native, payloads, expected


@pytest.mark.parametrize(
    "derivative_case",
    [
        (pattern, causal, alias)
        for pattern in ("dyadic", "fractional")
        for causal in (0, 1)
        for alias in (False, True)
    ],
    indirect=True,
)
def test_pinned_attention_derivative(derivative_case, request):
    np, directory, evidence, artifacts, native, inputs, expected = derivative_case
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native derivative execution")
    if native is not None:
        assert sys.platform == "win32", "DirectX execution requires Windows"
        outputs = DirectXComputeRuntime().dispatch(None, None, native)
        assert set(outputs) == {"dS", "P"}
        (directory / "readback.json").write_text(json.dumps(outputs))
        for slot, name in ((4, "dS"), (5, "P")):
            output = outputs[name]
            assert output["dtype"] == "uint32"
            (directory / f"buffer-{slot}.bin").write_bytes(
                np.asarray(output["values"], dtype="<u4").tobytes()
            )
    else:
        dispatch = {
            "buffers": [str(directory / f"input-{slot}.bin") for slot in range(7)],
            "allocationIds": [0, 1, 2, 3, 0 if evidence["alias"] else 4, 5, 6],
            "workgroupSize": evidence["workgroupSize"],
            "workgroupCount": evidence["workgroupCount"],
            "simdWidth": 32,
        }
        path = directory / "request.json"
        path.write_text(json.dumps(dispatch, indent=2))
        runner = request.getfixturevalue("attention_metal_runner")
        for label, (_, module) in artifacts.items():
            output = directory / label
            run([runner, module, evidence["entry"], path, output], output, "dispatch")
            runtime = json.loads((output / "dispatch.stdout").read_text())
            assert runtime["allocationIds"] == dispatch["allocationIds"]
            assert runtime["uniqueAllocations"] == (6 if evidence["alias"] else 7)
            for slot, payload in enumerate(inputs):
                if slot not in (4, 5) and not (slot == 0 and evidence["alias"]):
                    assert (output / f"buffer-{slot}.bin").read_bytes() == _payload(
                        payload
                    )
            if evidence["alias"]:
                assert (output / "buffer-0.bin").read_bytes() == (
                    output / "buffer-4.bin"
                ).read_bytes()
    for label in artifacts:
        output = directory if native is not None else directory / label
        evidence["results"][label] = [
            _check_output(
                np,
                (output / f"buffer-{slot}.bin").read_bytes(),
                reference,
                evidence["dtype"],
                exact=evidence["pattern"] == "dyadic",
            )
            for slot, reference in zip((4, 5), expected)
        ]
    evidence["execution"] = "passed"
    (directory / "evidence.json").write_text(json.dumps(evidence, indent=2))


@pytest.mark.parametrize("corruption", ["guard", "value", "nonfinite", "size"])
def test_derivative_verifier_rejects_corruption(corruption):
    np = pytest.importorskip("numpy")
    reference = np.zeros((1,), dtype="<f4")
    data = reference.tobytes() + GUARD
    if corruption == "guard":
        data = data[:-1] + bytes([data[-1] ^ 1])
    elif corruption == "value":
        data = struct.pack("<f", 3) + GUARD
    elif corruption == "nonfinite":
        data = struct.pack("<f", float("nan")) + GUARD
    else:
        data = data[:-4]
    with pytest.raises(AssertionError):
        _check_output(np, data, reference, "float", exact=False)


@pytest.mark.parametrize("dtype", ["float16_t", "bfloat16_t"])
def test_derivative_transport_preserves_odd_native_element_count(dtype):
    np = pytest.importorskip("numpy")
    reference = _encode(np, [0.5, -1.0, 2.0], dtype)
    payload = _payload(reference.tobytes())
    assert len(payload) % 4 == 0 and payload[-2:] == b"\x00\x00"
    assert payload[:6] == reference.tobytes()
    _check_output(np, payload, reference, dtype, exact=True)
    with pytest.raises(AssertionError, match="guard"):
        _check_output(np, payload[:-1] + b"\x01", reference, dtype, exact=True)


@pytest.mark.parametrize("corruption", ["identities", "payload"])
def test_metal_runner_rejects_inconsistent_aliases(
    compiled_derivative, request, tmp_path, corruption
):
    target, _, entry, _, artifacts = compiled_derivative
    if target != "metal" or os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip("requires native Metal allocation controls")
    first, second = tmp_path / "first.bin", tmp_path / "second.bin"
    first.write_bytes(struct.pack("<f", 1))
    second.write_bytes(struct.pack("<f", 2))
    payload = {
        "buffers": [str(first), str(second)],
        "allocationIds": [0] if corruption == "identities" else [0, 0],
        "workgroupSize": [32, 8, 1],
        "workgroupCount": [1, 1, 1],
        "simdWidth": 32,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload))
    runner = request.getfixturevalue("attention_metal_runner")
    result = subprocess.run(
        [
            str(runner),
            str(artifacts["original"][1]),
            entry,
            str(path),
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    expected = (
        "Allocation identity count"
        if corruption == "identities"
        else "Conflicting shared-allocation payloads"
    )
    assert expected in result.stderr
    assert not (tmp_path / "output").exists()
