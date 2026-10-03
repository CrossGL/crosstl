"""Resident execution of pinned MLX attention reduction sequences."""

import hashlib
import importlib
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
    _complete_directx_register_layout,
    _prepare_directx_buffers,
    _prepare_directx_constants,
    _prepare_opengl_buffers,
    _prepare_sequence_allocations,
    _validate_directx_register_layout,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeConstantBinding,
    NativeRuntimeDispatchRequest,
    RuntimeAdapterSetupError,
    RuntimeAllocationView,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
    RuntimeSpecializationConstant,
)
from demos.integrations.mlx.run_mlx_metal_host import run
from tests.test_translator.test_directx_float_atomics import _compile as compile_directx
from tests.test_translator.test_mlx_attention_ds_runtime import TYPES, _decode, _encode
from tests.test_translator.test_mlx_attention_odo_runtime import SOURCE, SOURCE_SHA256
from tests.test_translator.test_mlx_gated_delta_metal import _translate_pinned
from tests.test_translator.test_mlx_gated_delta_runtime import GUARD
from tests.test_translator.test_mlx_gated_delta_runtime import _compile as compile_metal

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_ATTENTION_REDUCE_RUNTIME"
COMPILE_ENV = "CROSTL_REQUIRE_MLX_ATTENTION_REDUCE_COMPILE"
TARGET_ENV = "CROSTL_MLX_ATTENTION_REDUCE_TARGET"
SCALARS = dict(zip(("rows", "dim", "group", "acc_rows", "row_off"), (9, 37, 3, 15, 4)))
HEADS = 2
MODES = ("set", "add", "add", "set")


def _index_assertions():
    return tuple(
        {"source": SOURCE, "expression": name, "minimum": 0, "maximum": maximum}
        for name, maximum in (
            ("si", HEADS * SCALARS["group"] * SCALARS["rows"] * SCALARS["dim"] - 1),
            (
                "ai",
                (
                    (HEADS - 1) * SCALARS["acc_rows"]
                    + SCALARS["row_off"]
                    + SCALARS["rows"]
                )
                * SCALARS["dim"]
                - 1,
            ),
        )
    )


def _dataset(np, dtype, pattern, count):
    shape = (HEADS, SCALARS["acc_rows"], SCALARS["dim"])
    initial_acc = (
        ((np.arange(np.prod(shape)) % 13 - 6) / 32).astype("<f4").reshape(shape)
    )
    initial_out = _encode(np, np.full(shape, -16), dtype)
    acc, out = initial_acc.copy(), initial_out.copy()
    indices = np.arange(
        HEADS * SCALARS["group"] * SCALARS["rows"] * SCALARS["dim"]
    ).reshape(HEADS, SCALARS["group"], SCALARS["rows"], SCALARS["dim"])
    sources = []
    region = slice(SCALARS["row_off"], SCALARS["row_off"] + SCALARS["rows"])
    for stage, mode in enumerate(MODES[:count]):
        source = _encode(
            np,
            ((indices * (7 + stage) + stage) % 31 - 15)
            / (31 if pattern == "fractional" else 32),
            dtype,
        )
        sources.append(source)
        total = np.zeros((HEADS, SCALARS["rows"], SCALARS["dim"]), dtype="<f4")
        for group in range(SCALARS["group"]):
            total = (total + _decode(np, source[:, group], dtype)).astype("<f4")
        if mode == "add":
            total = (acc[:, region] + total).astype("<f4")
        acc[:, region] = total
        out[:, region] = _encode(np, total, dtype)
    return sources, (initial_acc, initial_out), (acc, out)


def _physical(np, value, dtype, target):
    return _decode(np, value, dtype) if target == "opengl" else value


def _check_output(data, expected):
    assert len(data) == len(expected) + len(GUARD), "size"
    assert data[len(expected) :] == GUARD, "guard"
    assert data[: len(expected)] == expected, "values"
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@pytest.fixture(scope="module")
def sequence_runner(tmp_path_factory):
    directory = tmp_path_factory.mktemp("metal-sequence-runner")
    executable = directory / "sequence"
    run(
        [
            "swiftc",
            Path(__file__).parents[1]
            / "fixtures/runtime_verification/metal_dispatch_sequence.swift",
            "-o",
            executable,
        ],
        directory,
        "compile",
    )
    return executable


@pytest.fixture(
    scope="module",
    params=(
        ("float", "float16_t")
        if os.environ.get(TARGET_ENV) == "opengl"
        else tuple(TYPES)
    ),
)
def reduction_modules(request, tmp_path_factory):
    if not any(os.environ.get(name) == "1" for name in (REQUIRE_ENV, COMPILE_ENV)):
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned resident reductions")
    target = os.environ.get(TARGET_ENV)
    assert target in {"metal", "directx", "opengl"}
    assert (
        (sys.platform == "darwin")
        if target == "metal"
        else shutil.which("dxc" if target == "directx" else "glslangValidator")
    )
    dtype = request.param
    directory = tmp_path_factory.mktemp("attention-reduction-modules")
    result = {"target": target, "dtype": dtype, "modules": {}}
    for mode in ("set", "add"):
        work = directory / mode
        work.mkdir()
        entry = f"sdpa_vjp_reduce_{mode}_{dtype}"
        revision, original, artifact = _translate_pinned(
            work,
            target,
            entry=entry,
            workgroup_size=(32, 8, 1),
            source_path=SOURCE,
            source_sha256=SOURCE_SHA256,
            index_range_assertions=_index_assertions() if target == "opengl" else (),
        )
        result["commit"] = revision
        if target == "metal":
            module = compile_metal(artifact, original.parents[4], work / "generated")
            if "original" not in result:
                result["original"] = compile_metal(
                    original, original.parents[4], work / "original"
                )
        elif target == "directx":
            artifact, module = compile_directx(
                artifact.read_text(),
                work,
                profile="cs_6_2",
                flags=("-enable-16bit-types",),
            )
        else:
            module = work / "kernel.spv"
            run(
                [
                    shutil.which("glslangValidator"),
                    "-G",
                    "-S",
                    "comp",
                    artifact,
                    "-o",
                    module,
                ],
                work,
                "compile",
            )
        reflection = reflect_target_host_interface(
            artifact, target=target, stage="compute"
        )
        assert reflection["status"] == "ready", reflection
        (work / "reflection.json").write_text(json.dumps(reflection, indent=2))
        result["modules"][mode] = {
            "artifact": artifact,
            "module": module,
            "entry": entry,
            "reflection": reflection,
        }
    return result


def _requests(np, modules, sources, initial):
    target, dtype = modules["target"], modules["dtype"]
    entry = "main" if target == "opengl" else "CSMain"
    requests = []
    for stage, mode in enumerate(MODES[: len(sources)]):
        compiled = modules["modules"][mode]
        source_text = compiled["artifact"].read_text()
        bindings, constants = {}, {}
        for slot, (name, value) in enumerate(
            zip(("src", "acc", "out"), (sources[stage], *initial))
        ):
            element_dtype = "float" if slot == 1 else dtype
            physical = _physical(np, value, element_dtype, target)
            payload = physical.tobytes() + GUARD
            assert len(payload) % 4 == 0
            words = np.frombuffer(payload, dtype="<u4").tolist()
            element = {
                "float": "float",
                "float16_t": "float16_t",
                "bfloat16_t": "uint16_t",
            }[element_dtype]
            type_name = (
                f"{'RW' if slot else ''}StructuredBuffer<{element}>"
                if target == "directx"
                else None
            )
            if target == "directx":
                shader_name = "out_" if name == "out" else name
                assert (
                    f"{type_name} {shader_name} : register({'u' if slot else 't'}{slot});"
                    in source_text
                )
            bindings[name] = NativeRuntimeBufferBinding(
                name=name,
                binding=RuntimeResourceBinding(
                    name=name,
                    kind="buffer",
                    set=0,
                    binding=slot,
                    type_name=type_name,
                    access="read_write" if slot else "read",
                    metadata=(
                        {"byteStride": physical.itemsize} if target == "directx" else {}
                    ),
                ),
                dtype="uint32",
                shape=(len(words),),
                value=words if slot == 0 or stage == 0 else None,
                source=(
                    "expectedOutput" if slot and stage == len(sources) - 1 else "input"
                ),
                allocation=RuntimeAllocationView(
                    allocation_id=f"src-{stage}" if slot == 0 else name,
                    byte_length=len(payload),
                    allocation_byte_length=len(payload),
                ),
            )
        resources = {
            item["binding"]: item for item in compiled["reflection"]["resources"]
        }
        for slot, (name, value) in enumerate(SCALARS.items(), 3):
            if target == "directx":
                assert f": register(b{slot})" in source_text
                constants[name] = NativeRuntimeConstantBinding(
                    name=name,
                    value=value,
                    constant=RuntimeSpecializationConstant(
                        name=name,
                        dtype="int32",
                        kind="constant-buffer",
                        metadata={"binding": slot, "byteOffset": 0},
                    ),
                )
            else:
                layout = resources[slot]["scalarLayout"]
                assert (
                    layout["elementType"] == "int32" and layout["blockSizeBytes"] == 16
                )
                bindings[name] = NativeRuntimeBufferBinding(
                    name=name,
                    binding=RuntimeResourceBinding(
                        name=name,
                        kind="uniform",
                        set=0,
                        binding=slot,
                        access="read",
                        metadata={"scalarLayout": layout},
                    ),
                    dtype="int32",
                    shape=(1,),
                    value=[value] if stage == 0 else None,
                    source="input",
                    allocation=RuntimeAllocationView(
                        allocation_id=name, byte_length=16, allocation_byte_length=16
                    ),
                )
        prepare = (
            _prepare_directx_buffers if target == "directx" else _prepare_opengl_buffers
        )
        prepared = {item.name: item for item in prepare(bindings)}
        for name in ("acc", "out"):
            assert prepared[name].upload == (stage == 0)
        requests.append(
            NativeRuntimeDispatchRequest(
                target=target,
                artifact={"target": target},
                artifact_path=compiled["artifact"],
                module_path=compiled["module"],
                loaded_artifact=(
                    compiled["module"].read_bytes()
                    if target == "directx"
                    else source_text
                ),
                buffers=bindings,
                constants=constants,
                entry_point=entry,
                dispatch=RuntimeDispatchGeometry(
                    entry_point=entry,
                    workgroup_size=(32, 8, 1),
                    workgroup_count=(2, 2, HEADS),
                ),
            )
        )
    allocations = _allocation_plan(requests, target)
    assert len(allocations) == (
        len(requests) + 7 if target == "opengl" else len(requests) * 10 + 2
    )
    for name in ("acc", "out"):
        allocation = next(item for item in allocations if item.allocation_id == name)
        assert len(allocation.views) == len(requests)
        assert sum(item.upload for item in allocation.views) == 1
        assert sum(item.readback for item in allocation.views) == 1
    return requests


def _allocation_plan(requests, target):
    prepared = []
    for node in requests:
        if target == "directx":
            prepared.append(
                _validate_directx_register_layout(
                    _complete_directx_register_layout(
                        (
                            *_prepare_directx_buffers(node.buffers),
                            *_prepare_directx_constants(node.constants),
                        )
                    )
                )
            )
        else:
            prepared.append(_prepare_opengl_buffers(node.buffers))
    return _prepare_sequence_allocations(prepared, target=target)[0]


def _portable_execute(modules, requests):
    base = (
        DirectXComputeRuntime
        if modules["target"] == "directx"
        else OpenGLComputeRuntime
    )

    class ObservedRuntime(base):
        readbacks = 0
        allocations = 0

        def _read_outputs(self, *args):
            self.readbacks += 1
            return super()._read_outputs(*args)

        def _create_allocation_buffer(self, *args):
            self.allocations += 1
            return super()._create_allocation_buffer(*args)

        def _create_buffer_resource(self, *args):
            self.allocations += 1
            return super()._create_buffer_resource(*args)

    runtime = ObservedRuntime(
        **({"context_backends": ("egl",)} if modules["target"] == "opengl" else {})
    )
    outputs = runtime.dispatch_sequence(None, None, requests)
    assert runtime.readbacks == 1
    # DirectX keeps five CBVs and four sparse-register placeholders per dispatch.
    assert runtime.allocations == (
        len(requests) + 7 if modules["target"] == "opengl" else len(requests) * 10 + 2
    )
    return outputs, {
        "readbackPasses": runtime.readbacks,
        "allocations": runtime.allocations,
    }


@pytest.mark.parametrize("count", (1, 2, 3, 4))
@pytest.mark.parametrize("pattern", ("dyadic", "fractional"))
def test_pinned_resident_attention_reductions(
    reduction_modules, request, tmp_path, count, pattern
):
    np = importlib.import_module("numpy")
    modules = reduction_modules
    target, dtype = modules["target"], modules["dtype"]
    sources, initial, expected = _dataset(np, dtype, pattern, count)
    inputs = []
    for slot, value in enumerate([*sources, *initial]):
        source_dtype = "float" if slot == count else dtype
        (tmp_path / f"source-{slot}.bin").write_bytes(value.tobytes())
        path = tmp_path / f"upload-{slot}.bin"
        path.write_bytes(_physical(np, value, source_dtype, target).tobytes() + GUARD)
        inputs.append(path)
    for slot, value in enumerate(SCALARS.values(), count + 2):
        path = tmp_path / f"upload-{slot}.bin"
        path.write_bytes(struct.pack("<i", value))
        inputs.append(path)
    info = {
        "commit": modules["commit"],
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "target": target,
        "dtype": dtype,
        "pattern": pattern,
        "modes": MODES[:count],
        "parameters": SCALARS,
        "heads": HEADS,
        "workgroupSize": [32, 8, 1],
        "workgroupCount": [2, 2, HEADS],
        "indexRangeAssertions": _index_assertions() if target == "opengl" else [],
        "modules": {
            mode: {
                "entry": data["entry"],
                "artifactSha256": (
                    hashlib.sha256(data["artifact"].read_bytes()).hexdigest()
                ),
                "moduleSha256": hashlib.sha256(data["module"].read_bytes()).hexdigest(),
            }
            for mode, data in modules["modules"].items()
        },
        "execution": "not-tested",
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    if target == "metal":
        info["originalModuleSha256"] = hashlib.sha256(
            modules["original"].read_bytes()
        ).hexdigest()
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(info, indent=2))
    requests = None if target == "metal" else _requests(np, modules, sources, initial)
    if requests:
        (tmp_path / "bindings.json").write_text(
            json.dumps(
                [
                    [item.to_json() for item in node.buffers.values()]
                    for node in requests
                ],
                indent=2,
            )
        )
    if os.environ.get(REQUIRE_ENV) != "1":
        assert target == "directx" and os.environ.get(COMPILE_ENV) == "1"
        return
    actual = {}
    if target == "metal":
        runner = request.getfixturevalue("sequence_runner")
        for label in ("original", "generated"):
            directory = tmp_path / label
            directory.mkdir()
            nodes = [
                {
                    "library": str(
                        modules["original"]
                        if label == "original"
                        else modules["modules"][mode]["module"]
                    ),
                    "entry": modules["modules"][mode]["entry"],
                    "bindings": [stage, *range(count, count + 7)],
                    "workgroupSize": [32, 8, 1],
                    "workgroupCount": [2, 2, HEADS],
                    "simdWidth": 32,
                }
                for stage, mode in enumerate(MODES[:count])
            ]
            manifest = directory / "request.json"
            manifest.write_text(
                json.dumps(
                    {"allocations": list(map(str, inputs)), "dispatches": nodes},
                    indent=2,
                )
            )
            run([runner, manifest, directory], directory, "dispatch")
            stats = json.loads((directory / "dispatch.stdout").read_text())
            assert (
                stats["dispatches"] == count
                and stats["readbackPasses"] == 1
                and stats["uniqueAllocations"] == count + 7
            )
            assert stats["bindings"] == [node["bindings"] for node in nodes]
            for slot in (*range(count), *range(count + 2, count + 7)):
                assert (directory / f"allocation-{slot}.bin").read_bytes() == inputs[
                    slot
                ].read_bytes()
            actual[label] = [
                (directory / f"allocation-{slot}.bin").read_bytes()
                for slot in (count, count + 1)
            ]
    else:
        outputs, stats = _portable_execute(modules, requests)
        assert set(outputs) == {"acc", "out"}
        actual["generated"] = []
        for name in ("acc", "out"):
            assert outputs[name]["dtype"] == "uint32"
            data = np.asarray(outputs[name]["values"], dtype="<u4").tobytes()
            (tmp_path / f"{name}-readback.bin").write_bytes(data)
            actual["generated"].append(data)
    info["runtime"] = stats
    info["results"] = {}
    for label, values in actual.items():
        checks = []
        for name, data, reference in zip(("acc", "out"), values, expected):
            logical_dtype = "float" if name == "acc" else dtype
            encoded = _physical(np, reference, logical_dtype, target).tobytes()
            (tmp_path / f"expected-{name}.bin").write_bytes(encoded)
            checks.append(_check_output(data, encoded))
        info["results"][label] = checks
    if target == "metal":
        assert actual["original"] == actual["generated"]
    info["execution"] = "passed"
    evidence.write_text(json.dumps(info, indent=2))


@pytest.mark.parametrize("corruption", ("size", "guard", "value"))
def test_resident_reduction_verifier_rejects_corruption(corruption):
    expected = struct.pack("<f", 1.0)
    data = expected + GUARD
    if corruption == "size":
        data = data[:-1]
    elif corruption == "guard":
        data = expected + bytes(len(GUARD))
    else:
        data = struct.pack("<f", 1.0004) + GUARD
    with pytest.raises(
        AssertionError, match=corruption if corruption != "value" else "values"
    ):
        _check_output(data, expected)


def test_resident_reduction_rejects_conflicting_accumulator_upload(reduction_modules):
    if reduction_modules["target"] == "metal":
        pytest.skip("Portable allocation planner control")
    np = importlib.import_module("numpy")
    sources, initial, _ = _dataset(np, reduction_modules["dtype"], "fractional", 3)
    requests = _requests(np, reduction_modules, sources, initial)
    buffers = dict(requests[1].buffers)
    buffers["acc"] = replace(buffers["acc"], value=[0] * buffers["acc"].shape[0])
    requests[1] = replace(requests[1], buffers=buffers)
    with pytest.raises(RuntimeAdapterSetupError) as error:
        _allocation_plan(requests, reduction_modules["target"])
    assert error.value.details["reasonKind"] == "allocation-upload-conflict"


def test_resident_reduction_bounds_and_reset_reference():
    np = importlib.import_module("numpy")
    bounds = {item["expression"]: item["maximum"] for item in _index_assertions()}
    assert bounds == {"si": 1997, "ai": 1035}
    assert SCALARS["row_off"] + SCALARS["rows"] <= SCALARS["acc_rows"]
    sources, initial, outputs = _dataset(np, "float16_t", "fractional", 4)
    region = slice(SCALARS["row_off"], SCALARS["row_off"] + SCALARS["rows"])
    total = np.zeros((HEADS, SCALARS["rows"], SCALARS["dim"]), dtype="<f4")
    for group in range(SCALARS["group"]):
        total += sources[-1][:, group].astype("<f4")
    np.testing.assert_array_equal(outputs[0][:, region], total)
    for before, after in zip(initial, outputs):
        np.testing.assert_array_equal(
            before[:, : region.start], after[:, : region.start]
        )
        np.testing.assert_array_equal(before[:, region.stop :], after[:, region.stop :])


@pytest.mark.parametrize(
    "invalid", ("empty", "allocation", "geometry", "entry", "width")
)
def test_metal_resident_sequence_rejects_invalid_request(
    reduction_modules, request, tmp_path, invalid
):
    if reduction_modules["target"] != "metal" or os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip("Metal sequence rejection controls require native Metal")
    runner = request.getfixturevalue("sequence_runner")
    compiled = reduction_modules["modules"]["set"]
    node = {
        "library": str(compiled["module"]),
        "entry": compiled["entry"],
        "bindings": [0] * 8,
        "workgroupSize": [32, 8, 1],
        "workgroupCount": [1, 1, 1],
        "simdWidth": 32,
    }
    payload = {
        "allocations": [str(tmp_path / "must-not-be-read.bin")],
        "dispatches": [node],
    }
    if invalid == "empty":
        payload["dispatches"] = []
    elif invalid == "allocation":
        node["bindings"][0] = 1
    elif invalid == "geometry":
        node["workgroupSize"] = [32, 0, 1]
    elif invalid == "entry":
        node["entry"] = "missing_entry"
    else:
        node["simdWidth"] = 16
    manifest = tmp_path / "request.json"
    manifest.write_text(json.dumps(payload))
    result = subprocess.run(
        [runner, manifest, tmp_path / "output"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "rejection.json").write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        )
    )
    assert result.returncode != 0
    assert {
        "empty": "Empty dispatch sequence",
        "allocation": "Invalid geometry or allocation binding",
        "geometry": "Invalid geometry or allocation binding",
        "entry": "Missing entry point",
        "width": "Unsupported SIMD width or workgroup size",
    }[invalid] in result.stderr
    assert not (tmp_path / "output").exists()
