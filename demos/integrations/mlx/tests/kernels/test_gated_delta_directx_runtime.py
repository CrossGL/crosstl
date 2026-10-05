"""Native DirectX execution of pinned MLX gated-delta backward kernels."""

import hashlib
import importlib
import json
import os
import shutil
import struct
import sys
from dataclasses import asdict
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
from crosstl.translator.resource_storage import (
    BINARY16_STORAGE,
    parse_resource_storage_header,
    resource_storage_header,
)
from demos.integrations.mlx.tests.kernels.test_gated_delta_metal import (
    _translate_pinned,
)
from demos.integrations.mlx.tests.kernels.test_gated_delta_runtime import (
    ATOL,
    CONFIGURATIONS,
    GUARD,
    RTOL,
)
from tests.test_translator.test_directx_float_atomics import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_GATED_DELTA_DIRECTX_RUNTIME"
COMPILE_ENV = "CROSTL_REQUIRE_MLX_GATED_DELTA_DIRECTX"
BUFFER_NAMES = (
    "q",
    "k",
    "v",
    "g",
    "b",
    "cot_o",
    "cot_h",
    "state_cache",
    "T",
    "dq",
    "dk",
    "dv",
    "dg",
    "db",
    "dh",
)


def _request(np, case, buffers, artifact, module):
    source = artifact.read_text(encoding="utf-8")
    assert len(buffers) == len(BUFFER_NAMES)
    assert parse_resource_storage_header(source) == {
        name: BINARY16_STORAGE
        for name, buffer in zip(BUFFER_NAMES, buffers)
        if buffer.dtype == np.dtype("<f2")
    }
    bindings = {}
    for slot, (name, buffer) in enumerate(zip(BUFFER_NAMES, buffers)):
        if slot == 8:
            assert buffer.dtype == np.dtype("<i4") and buffer.tolist() == [case.length]
            assert f"cbuffer {case.entry}_T_Constants : register(b8)" in source
            continue
        atomic = slot in {9, 10, 12, 13}
        element = (
            "mlx_atomic_float_void"
            if atomic
            else "uint16_t" if buffer.dtype == np.dtype("<f2") else "float"
        )
        type_name = f"{'RW' if slot >= 9 else ''}StructuredBuffer<{element}>"
        register = f"{'u' if slot >= 9 else 't'}{slot}"
        assert f"{type_name} {name} : register({register});" in source
        if atomic:
            assert "struct mlx_atomic_float_void {\n    float val;\n};" in source
        payload = buffer.tobytes() + GUARD
        words = np.frombuffer(payload, dtype="<u4").tolist()
        bindings[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=slot,
                type_name=type_name,
                access="read_write" if slot >= 9 else "read",
                metadata={"byteStride": buffer.itemsize},
            ),
            dtype="uint32",
            shape=(len(words),),
            value=words,
            source="expectedOutput" if slot >= 9 else "input",
        )
    return NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=module.read_bytes(),
        buffers=bindings,
        constants={
            "T": NativeRuntimeConstantBinding(
                name="T",
                value=case.length,
                constant=RuntimeSpecializationConstant(
                    name="T",
                    dtype="int32",
                    kind="constant-buffer",
                    metadata={"binding": 8, "byteOffset": 0},
                ),
            )
        },
        entry_point="CSMain",
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_size=(32, 4, 1),
            workgroup_count=(1, 32, case.batch * case.value_heads),
        ),
    )


@pytest.fixture(params=("float16", "float32"))
def request_inputs(request, tmp_path):
    np = pytest.importorskip("numpy")
    case = SimpleNamespace(entry="gated_delta", length=1, batch=1, value_heads=16)
    buffers = []
    declarations = ["struct mlx_atomic_float_void {\n    float val;\n};"]
    encodings = {}
    for slot, name in enumerate(BUFFER_NAMES):
        if slot == 8:
            buffers.append(np.array([case.length], dtype="<i4"))
            declarations.append(
                f"cbuffer {case.entry}_T_Constants : register(b8) {{ int T; }};"
            )
            continue
        if slot < 6 and request.param == "float16":
            buffer = np.array([0, 0x8000, 1, 0x7C01, 0x7FFF, 0xFC01], dtype="<u2").view(
                "<f2"
            )
            element = "uint16_t"
            encodings[name] = BINARY16_STORAGE
        else:
            buffer = np.array([0, 0x80000000, 1, 0x7F812345], dtype="<u4").view("<f4")
            element = "mlx_atomic_float_void" if slot in {9, 10, 12, 13} else "float"
        buffers.append(buffer)
        declarations.append(
            f"{'RW' if slot >= 9 else ''}StructuredBuffer<{element}> {name} : register({'u' if slot >= 9 else 't'}{slot});"
        )
    source = (resource_storage_header(encodings) if encodings else "") + "\n".join(
        declarations
    )
    artifact, module = tmp_path / "bindings.hlsl", tmp_path / "bindings.dxil"
    artifact.write_text(source, encoding="utf-8")
    module.write_bytes(b"binding-test-module")
    return np, case, buffers, artifact, module


def test_directx_backward_request_preserves_storage_bytes(request_inputs):
    np, case, buffers, artifact, module = request_inputs
    request = _request(np, case, buffers, artifact, module)
    prepared = _validate_directx_register_layout(
        _complete_directx_register_layout(
            (
                *_prepare_directx_buffers(request.buffers),
                *_prepare_directx_constants(request.constants),
            )
        )
    )
    actual = {item.name: item for item in prepared if item.source != "descriptor-gap"}
    assert len(actual) == 15
    for slot, (name, buffer) in enumerate(zip(BUFFER_NAMES, buffers)):
        if slot == 8:
            assert actual["constants[T]"].payload[:4] == struct.pack("<i", case.length)
        else:
            assert actual[name].payload == buffer.tobytes() + GUARD
            assert actual[name].stride == buffer.itemsize
            assert actual[name].binding_index == slot


@pytest.mark.parametrize(
    "damage",
    ("missing", "extra", "wrong-declaration", "wrong-register", "missing-buffer"),
)
def test_directx_backward_request_rejects_storage_mismatch(request_inputs, damage):
    np, case, buffers, artifact, module = request_inputs
    source = artifact.read_text(encoding="utf-8")
    encodings = parse_resource_storage_header(source)
    if encodings:
        source = source.partition("\n")[2]
    if damage == "missing":
        if encodings:
            encodings.pop("q")
        else:
            encodings["q"] = BINARY16_STORAGE
    elif damage == "extra":
        encodings["missing"] = BINARY16_STORAGE
    elif damage == "wrong-declaration":
        source = source.replace(
            "StructuredBuffer<uint16_t> q", "StructuredBuffer<float16_t> q"
        ).replace("StructuredBuffer<float> q", "StructuredBuffer<uint> q")
    elif damage == "wrong-register":
        source = source.replace("register(t0)", "register(t15)")
    else:
        buffers.pop()
    source = (resource_storage_header(encodings) if encodings else "") + source
    artifact.write_text(source, encoding="utf-8")
    with pytest.raises(AssertionError):
        _request(np, case, buffers, artifact, module)


@pytest.fixture(scope="module", params=CONFIGURATIONS)
def compiled_case(request, tmp_path_factory):
    if not any(os.environ.get(name) == "1" for name in (REQUIRE_ENV, COMPILE_ENV)):
        pytest.skip(f"set {COMPILE_ENV}=1 for pinned software-subgroup compilation")
    assert shutil.which("dxc"), "Pinned DirectX backward validation requires DXC"
    np = importlib.import_module("numpy")
    reference = importlib.import_module(
        "demos.integrations.mlx.tests.fixtures.gated_delta_reference"
    )
    case = reference.Case(*request.param)
    directory = tmp_path_factory.mktemp("gated-delta-directx")
    revision, original, generated = _translate_pinned(
        directory,
        "directx",
        entry=case.entry,
        workgroup_size=(32, 4, 1),
        target_options={
            "software_subgroup_width": 32,
            "relative_wave_shuffle_out_of_range": "self",
        },
    )
    source = generated.read_text(encoding="utf-8")
    assert "WaveActive" not in source
    assert "GroupMemoryBarrierWithGroupSync();" in source
    assert "[numthreads(32, 4, 1)]" in source
    artifact, module = _compile(
        source, directory, profile="cs_6_2", flags=("-enable-16bit-types",)
    )
    _, _, buffers, gradients, _, _ = reference.dataset(case)
    native_request = _request(np, case, buffers, artifact, module)
    prepared = _validate_directx_register_layout(
        _complete_directx_register_layout(
            (
                *_prepare_directx_buffers(native_request.buffers),
                *_prepare_directx_constants(native_request.constants),
            )
        )
    )
    actual = {item.name: item for item in prepared if item.source != "descriptor-gap"}
    assert len(actual) == 15
    for slot, (name, buffer) in enumerate(zip(BUFFER_NAMES, buffers)):
        (directory / f"input-{slot}.bin").write_bytes(buffer.tobytes() + GUARD)
        if slot == 8:
            assert actual["constants[T]"].payload[:4] == struct.pack("<i", case.length)
        else:
            assert actual[name].payload == buffer.tobytes() + GUARD
            assert actual[name].stride == buffer.itemsize
            assert actual[name].binding_index == slot
    for slot, gradient in enumerate(gradients, 9):
        (directory / f"expected-{slot}.f64").write_bytes(
            gradient.astype("<f8").tobytes()
        )
    evidence = {
        "commit": revision,
        "case": asdict(case),
        "entry": case.entry,
        "targetEntry": "CSMain",
        "execution": "not-tested",
        "rtol": 0 if case.uniform else RTOL,
        "atol": 0 if case.uniform else ATOL,
        "fullUpstreamSuite": False,
        "fullTranslatedBackend": False,
        "originalSha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "dispatch": native_request.dispatch.to_json(),
        "bindings": [value.to_json() for value in native_request.buffers.values()],
        "constants": [value.to_json() for value in native_request.constants.values()],
        "readbackSlots": list(range(9, 15)),
        "inputReadbacks": "not-requested-read-only-bindings",
    }
    (directory / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    return np, case, directory, gradients, native_request, evidence


def test_pinned_software_backward_compiles(compiled_case):
    _np, _case, _directory, _gradients, request, _evidence = compiled_case
    assert request.module_path.stat().st_size > 0


def _check_gradients(np, directory, gradients, outputs, *, exact=False):
    assert set(outputs) == set(BUFFER_NAMES[9:])
    records = []
    for slot, gradient in enumerate(gradients, 9):
        name = BUFFER_NAMES[slot]
        words = outputs[name]["values"]
        assert len(words) == gradient.size + len(GUARD) // 4, (name, "size")
        data = np.asarray(words, dtype="<u4").tobytes()
        (directory / f"buffer-{slot}.bin").write_bytes(data)
        assert data[-len(GUARD) :] == GUARD, (name, "guard")
        payload = data[: -len(GUARD)]
        values = np.frombuffer(payload, dtype="<f4").reshape(gradient.shape)
        assert np.isfinite(values).all(), (name, "nonfinite")
        if exact:
            assert payload == gradient.astype("<f4").tobytes(), name
        else:
            np.testing.assert_allclose(
                values, gradient, rtol=RTOL, atol=ATOL, err_msg=name
            )
        records.append(
            {
                "slot": slot,
                "name": name,
                "values": gradient.size,
                "sha256": hashlib.sha256(data).hexdigest(),
                "maximumAbsoluteError": float(
                    np.max(np.abs(values.astype("float64") - gradient))
                ),
            }
        )
    return records


def test_pinned_software_backward_executes(compiled_case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for DirectX backward execution")
    assert sys.platform == "win32", "DirectX backward execution requires Windows"
    np, case, directory, gradients, request, evidence = compiled_case
    state = SimpleNamespace(details={})
    outputs = DirectXComputeRuntime().dispatch(None, state, request)
    (directory / "readback.json").write_text(json.dumps(outputs), encoding="utf-8")
    records = _check_gradients(np, directory, gradients, outputs, exact=case.uniform)
    evidence.update(execution="passed", runtime=state.details, outputs=records)
    (directory / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )


@pytest.mark.parametrize("corruption", ["missing", "size", "guard", "value", "nan"])
def test_directx_gradient_verifier_rejects_corruption(tmp_path, corruption):
    np = pytest.importorskip("numpy")
    gradients = [np.zeros(1) for _ in range(6)]
    outputs = {name: {"values": [0] + [0x6A15BEEF] * 32} for name in BUFFER_NAMES[9:]}
    if corruption == "missing":
        outputs.pop("dh")
    elif corruption == "size":
        outputs["dq"]["values"].pop()
    elif corruption == "guard":
        outputs["dq"]["values"][-1] = 0
    else:
        outputs["dq"]["values"][0] = 0x7FC00000 if corruption == "nan" else 0x3F800000
    with pytest.raises(AssertionError):
        _check_gradients(np, tmp_path, gradients, outputs)
