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
            else "float16_t" if buffer.itemsize == 2 else "float"
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
