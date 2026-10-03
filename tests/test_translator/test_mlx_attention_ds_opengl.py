"""OpenGL execution of unchanged MLX attention tile derivatives."""

import hashlib
import importlib
import json
import os
import shutil
import struct

import pytest

from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_runtime_drivers import (
    OpenGLComputeRuntime,
    _prepare_opengl_buffers,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeAllocationView,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
)
from demos.integrations.mlx.run_mlx_metal_host import run
from tests.test_translator.test_mlx_attention_ds_runtime import (
    BK,
    BQ,
    HEADS,
    I0,
    NAMES,
    PARAMETERS,
    QL,
    SOURCE,
    SOURCE_SHA256,
    _dataset,
    _decode,
)
from tests.test_translator.test_mlx_gated_delta_metal import _translate_pinned
from tests.test_translator.test_mlx_gated_delta_runtime import GUARD

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_ATTENTION_DS_OPENGL"


def _index_assertions():
    return tuple(
        {"source": SOURCE, "expression": expression, "minimum": 0, "maximum": maximum}
        for expression, maximum in (
            ("sbase + col", HEADS * BQ * BK - 1),
            ("qi", (HEADS - 1) * QL + I0 + BQ - 1),
        )
    )


@pytest.fixture(scope="module", params=("float", "float16_t"))
def derivative_module(request, tmp_path_factory):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native OpenGL derivatives")
    compiler = shutil.which("glslangValidator")
    assert compiler, "OpenGL derivatives require glslangValidator"
    dtype = request.param
    directory = tmp_path_factory.mktemp("attention-ds-opengl-module")
    revision, _, artifact = _translate_pinned(
        directory,
        "opengl",
        entry=f"sdpa_vjp_ds_{dtype}",
        source_path=SOURCE,
        source_sha256=SOURCE_SHA256,
        workgroup_size=(32, 8, 1),
        index_range_assertions=_index_assertions(),
    )
    module = directory / "kernel.spv"
    run([compiler, "-G", "-S", "comp", artifact, "-o", module], directory, "compile")
    assert module.stat().st_size > 0
    reflection = reflect_target_host_interface(
        artifact, target="opengl", stage="compute"
    )
    assert reflection["status"] == "ready", reflection
    resources = sorted(reflection["resources"], key=lambda item: item["binding"])
    assert [item["binding"] for item in resources] == list(range(7))
    layout = resources[6]["scalarLayout"]
    assert layout["blockSizeBytes"] == 32
    assert [member["name"] for member in layout["blockMembers"]] == list(PARAMETERS)
    assert [member["offsetBytes"] for member in layout["blockMembers"]] == list(
        range(0, 32, 4)
    )
    assert [member["physicalType"] for member in layout["blockMembers"]] == [
        "int"
    ] * 5 + ["float", "int", "int"]
    for item in resources[:6]:
        assert item["scalarLayout"]["elementType"] == "float32"
        assert item["scalarLayout"]["elementStrideBytes"] == 4
    (directory / "reflection.json").write_text(json.dumps(reflection, indent=2))
    return dtype, revision, artifact, module, resources


def _check_output(np, data, reference, dtype, exact):
    decoded = _decode(np, reference, dtype)
    assert len(data) == decoded.nbytes + len(GUARD), "size"
    assert data[decoded.nbytes :] == GUARD, "guard"
    actual = np.frombuffer(data[: decoded.nbytes], dtype="<f4").reshape(reference.shape)
    assert np.isfinite(actual).all(), "nonfinite"
    if exact:
        assert actual.tobytes() == decoded.tobytes(), "values"
    else:
        np.testing.assert_allclose(actual, decoded, rtol=1e-5, atol=1e-5)
    return {
        "values": actual.size,
        "maximumAbsoluteError": float(np.max(np.abs(actual - decoded))),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


@pytest.mark.parametrize(
    "pattern,causal,alias",
    [
        (p, c, a)
        for p in ("dyadic", "fractional")
        for c in (0, 1)
        for a in (False, True)
    ],
)
def test_pinned_opengl_attention_derivative(
    derivative_module, tmp_path, pattern, causal, alias
):
    np = importlib.import_module("numpy")
    dtype, revision, artifact, module, resources = derivative_module
    buffers, expected, parameters = _dataset(np, dtype, pattern, causal, alias)
    bindings = {}
    for slot, (name, buffer, resource) in enumerate(zip(NAMES, buffers, resources)):
        original = buffer.tobytes()
        decoded = _decode(np, buffer, dtype) if slot not in (2, 3) else buffer
        payload = decoded.tobytes() + GUARD
        (tmp_path / f"input-{slot}.bin").write_bytes(original)
        (tmp_path / f"upload-{slot}.bin").write_bytes(payload)
        words = np.frombuffer(payload, dtype="<u4").tolist()
        bindings[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=slot,
                type_name=resource["type"],
                access=resource["access"],
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
    payload = struct.pack("<5if2i", *parameters.values())
    (tmp_path / "upload-6.bin").write_bytes(payload)
    bindings["parameters"] = NativeRuntimeBufferBinding(
        name="parameters",
        binding=RuntimeResourceBinding(
            name="parameters",
            kind="constant-buffer",
            set=0,
            binding=6,
            access="read",
            metadata={"scalarLayout": resources[6]["scalarLayout"]},
        ),
        dtype="uint32",
        shape=(8,),
        value=list(struct.unpack("<8I", payload)),
        source="input",
    )
    prepared = {item.name: item for item in _prepare_opengl_buffers(bindings)}
    for slot, name in enumerate((*NAMES, "parameters")):
        assert prepared[name].payload == (tmp_path / f"upload-{slot}.bin").read_bytes()
    assert prepared["parameters"].byte_length == 32
    assert (prepared["S"].allocation_id == prepared["dS"].allocation_id) == alias
    request = NativeRuntimeDispatchRequest(
        target="opengl",
        artifact={"target": "opengl"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=artifact.read_text(),
        buffers=bindings,
        constants={},
        entry_point="main",
        dispatch=RuntimeDispatchGeometry(
            entry_point="main",
            workgroup_size=(32, 8, 1),
            workgroup_count=(2, 2, HEADS),
        ),
    )
    evidence = {
        "commit": revision,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "entry": f"sdpa_vjp_ds_{dtype}",
        "target": "opengl",
        "dtype": dtype,
        "pattern": pattern,
        "causal": causal,
        "alias": alias,
        "parameters": parameters,
        "workgroupSize": [32, 8, 1],
        "workgroupCount": [2, 2, HEADS],
        "sourceElementBytes": buffers[0].itemsize,
        "targetElementBytes": 4,
        "indexRangeAssertions": list(_index_assertions()),
        "bindings": [value.to_json() for value in bindings.values()],
        "artifact": str(artifact),
        "module": str(module),
        "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "execution": "not-tested",
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
    outputs = OpenGLComputeRuntime(context_backends=("egl",)).dispatch(
        None, None, request
    )
    assert set(outputs) == {"dS", "P"}
    evidence["results"] = {}
    for slot, name, reference in zip((4, 5), ("dS", "P"), expected):
        assert outputs[name]["dtype"] == "uint32"
        data = np.asarray(outputs[name]["values"], dtype="<u4").tobytes()
        (tmp_path / f"buffer-{slot}.bin").write_bytes(data)
        (tmp_path / f"expected-{slot}.bin").write_bytes(reference.tobytes())
        evidence["results"][name] = _check_output(
            np, data, reference, dtype, pattern == "dyadic"
        )
    evidence["execution"] = "passed"
    (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))


def test_opengl_derivative_bounds_cover_the_dispatched_tile():
    assert I0 + BQ <= QL
    bounds = {item["expression"]: item["maximum"] for item in _index_assertions()}
    assert (
        max(
            (head * BQ + row) * BK + col
            for head in range(HEADS)
            for row in range(BQ)
            for col in range(BK)
        )
        == bounds["sbase + col"]
    )
    assert (
        max(head * QL + I0 + row for head in range(HEADS) for row in range(BQ))
        == bounds["qi"]
    )


@pytest.mark.parametrize(
    "corruption", ["size", "guard", "value", "nonfinite", "unrounded"]
)
def test_opengl_derivative_verifier_rejects_corruption(corruption):
    np = importlib.import_module("numpy")
    reference = np.array([1.0], dtype="<f2")
    payload = np.array([1.0], dtype="<f4").tobytes() + GUARD
    if corruption == "size":
        payload = payload[:-1]
    elif corruption == "guard":
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
    else:
        payload = (
            np.array(
                [{"value": 2.0, "nonfinite": np.nan, "unrounded": 1.0004}[corruption]],
                dtype="<f4",
            ).tobytes()
            + GUARD
        )
    with pytest.raises(AssertionError):
        _check_output(np, payload, reference, "float16_t", False)
