"""Affine matrix-vector host routing, launch contracts and guarded output commits."""

import ctypes
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import matmul_dispatch as dispatch
from demos.integrations.mlx.portable_host import matmul_layout as layout
from demos.integrations.mlx.portable_host import matmul_packages, prepare, runtime
from demos.integrations.mlx.portable_host import verify_quantized_matmul as verify

HOST = Path(layout.__file__).parent
CTYPES = {
    "uint32": ctypes.c_uint32,
    "float32": ctypes.c_uint32,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}


def contiguous(shape):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.insert(0, step)
        step *= extent
    return strides


def span(shape, strides):
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))


def native_binding(record):
    """What backend.cpp binds for one verification case: counts and parameters."""
    k, n, m, batches = record["K"], record["N"], record["M"], record["batches"]
    words, groups = k * record["bits"] // 32, k // record["groupSize"]
    x_shape = list(record["xShape"])
    x_strides = contiguous(x_shape)
    if record["layout"] == "strided":
        x_strides[0] *= 2
    w_batch = record["wBatchShape"]
    out_batch = x_shape[:-2] if w_batch else []
    w_shape = [*out_batch, n, words]
    strides = {}
    for name, columns in (("w", words), ("s", groups), ("b", groups)):
        dense = contiguous([*w_batch, n, columns])
        strides[name] = [
            0 if extent == 1 and target > 1 else stride
            for extent, target, stride in zip(w_batch, out_batch, dense)
        ] + dense[len(w_batch) :]
    counts = {
        "x": span(x_shape, x_strides),
        "w": math.prod(w_batch) * n * words,
        "scales": math.prod(w_batch) * n * groups,
        "biases": math.prod(w_batch) * n * groups,
        "y": batches * m * n,
    }
    parameters = {"in_vec_size": k, "out_vec_size": n}
    if batches > 1:
        parameters.update(
            x_batch_ndims=len(x_shape) - 2,
            x_shape=x_shape,
            x_strides=x_strides,
            w_batch_ndims=len(w_shape) - 2,
            w_shape=w_shape,
            w_strides=strides["w"],
            s_strides=strides["s"],
            b_strides=strides["b"],
        )
    rows = layout.tile_rows(record["entry"])
    execution = {
        "workgroupCount": [m, -(-n // rows), batches],
        "workgroupSize": list(layout.workgroup_size(record["entry"])),
    }
    return counts, parameters, execution


RECORDS = [verify.case_record(case) for case in verify.CASES]


@pytest.mark.parametrize("dtype", ("float", "float16_t", "bfloat16_t"))
@pytest.mark.parametrize("group", (32, 64, 128))
@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6, 8))
def test_signature_accepts_exactly_the_dispatched_specializations(dtype, group, bits):
    stem = f"{dtype}_gs_{group}_b_{bits}"
    for batch in (0, 1):
        for kernel in ("qmv", "qmv_fast"):
            info = layout.signature(f"affine_{kernel}_{stem}_batch_{batch}")
            assert (info["kernel"], info["batched"], info["D"]) == (
                kernel,
                bool(batch),
                None,
            )
        for d in (64, 128):
            entry = f"affine_qmv_quad_{stem}_d_{d}_batch_{batch}"
            if bits in (2, 4, 8) and d % group == 0:
                assert layout.signature(entry)["D"] == d
            else:
                with pytest.raises(ValueError):
                    layout.signature(entry)
    for entry in (
        f"affine_qmv_wide_{stem}_batch_0",
        f"affine_qmv_{stem}_d_64_batch_0",
        f"affine_qmv_quad_{stem}_batch_0",
        f"affine_gather_qmv_{stem}",
        f"affine_qmv_{stem}_batch_0_hgs",
    ):
        with pytest.raises(ValueError):
            layout.signature(entry)


@pytest.mark.parametrize("group", (128, 64, 32))
@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6, 8))
def test_selection_follows_dispatch_qmv_for_upstream_qmv_shapes(group, bits):
    # test_quantized.py test_qmv: (M, N) weights times (B, 1, N) vectors.
    for outputs in (256, 512, 67):
        for k in (64, 256):
            if group > k:
                continue
            for batches in (1, 3, 8):
                entry = layout.select("float32", group, bits, 1, outputs, k, batches)
                info = layout.signature(entry)
                assert info["batched"] == (batches > 1)
                if k == 64 and bits in (2, 4, 8):
                    assert info["kernel"] == "qmv_quad" and info["D"] == 64
                elif outputs % 8 == 0 and k % layout.fast_alignment(bits) == 0:
                    assert info["kernel"] == "qmv_fast" and bits in (6, 8)
                else:
                    assert info["kernel"] == "qmv"
    assert [layout.fast_alignment(bits) for bits in (2, 3, 4, 5, 6, 8)] == [
        512,
        512,
        512,
        512,
        256,
        256,
    ]


def test_verification_cases_cover_every_route_and_batch_layout():
    infos = [layout.signature(record["entry"]) for record in RECORDS]
    assert {(info["kernel"], info["batched"]) for info in infos} == {
        (kernel, batched)
        for kernel in ("qmv_quad", "qmv_fast", "qmv")
        for batched in (False, True)
    }
    assert {info["dtype"] for info in infos} == {"float32", "float16", "bfloat16"}
    assert {record["layout"] for record in RECORDS} == {"dense", "strided"}
    assert any(len(record["wBatchShape"]) == 2 for record in RECORDS)
    assert any(1 in record["wBatchShape"] for record in RECORDS)
    assert all(record["M"] < 18 for record in RECORDS)


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_plan_accepts_each_verification_launch(record):
    counts, parameters, execution = native_binding(record)
    metadata = layout.plan(record["entry"], counts, parameters, counts["y"], execution)
    assert (metadata["M"], metadata["N"], metadata["K"], metadata["batches"]) == (
        record["M"],
        record["N"],
        record["K"],
        record["batches"],
    )
    # The translation reflects every parameter, batched or not.
    assert set(metadata["parameters"]) == {
        *layout.SCALAR_PARAMETERS,
        *layout.BATCH_PARAMETERS,
    }
    padding = metadata["overreadPadding"]
    if record["name"] == "quad-d64-partial-tile":
        # One 64-row tile over 72 rows of one group: 56 rows past N.
        assert padding == {"scales": 56, "biases": 56}
    elif layout.signature(record["entry"])["kernel"] != "qmv_quad":
        assert padding == {"scales": 0, "biases": 0}


def mutate(fault, counts, parameters, execution):
    if fault == "tiles":
        execution["workgroupCount"][1] += 1
    elif fault == "workgroup":
        execution["workgroupSize"] = [32, 1, 1]
    elif fault == "threads":
        counts["y"] += 1
    elif fault == "x-short":
        counts["x"] -= 1
    elif fault == "scales-short":
        counts["scales"] -= 1
    elif fault == "negative-stride":
        parameters["x_strides"][0] = -parameters["x_strides"][0]
    elif fault == "column-stride":
        parameters["w_strides"][-1] = 2
    elif fault == "row-stride":
        parameters["s_strides"][-2] += 1
    elif fault == "rank":
        parameters["x_batch_ndims"] += 1
    elif fault == "extent":
        parameters["x_shape"][0] += 1
    elif fault == "missing":
        parameters.pop("b_strides")
    elif fault == "bound":
        counts["w"] = layout.MAX_ELEMENTS + 1
    elif fault == "dimension":
        parameters["in_vec_size"] += 32


@pytest.mark.parametrize(
    "fault",
    (
        "tiles",
        "workgroup",
        "threads",
        "x-short",
        "scales-short",
        "negative-stride",
        "column-stride",
        "row-stride",
        "rank",
        "extent",
        "missing",
        "bound",
        "dimension",
    ),
)
def test_plan_rejects_launches_that_differ_from_mlx(fault):
    record = next(item for item in RECORDS if item["name"] == "qmv-two-batch-axes")
    counts, parameters, execution = native_binding(record)
    mutate(fault, counts, parameters, execution)
    with pytest.raises(ValueError):
        layout.plan(record["entry"], counts, parameters, counts["y"], execution)


def test_plan_rejects_entries_mlx_would_not_select():
    record = next(item for item in RECORDS if item["name"] == "fast-rows")
    counts, parameters, execution = native_binding(record)
    slow = record["entry"].replace("qmv_fast", "qmv")
    with pytest.raises(ValueError, match="routing"):
        layout.plan(slow, counts, parameters, counts["y"], execution)
    unbatched = next(item for item in RECORDS if item["name"] == "quad-d128-rows")
    counts, parameters, execution = native_binding(unbatched)
    parameters["x_batch_ndims"] = 0
    with pytest.raises(ValueError, match="batched"):
        layout.plan(unbatched["entry"], counts, parameters, counts["y"], execution)


def native_buffers(record, data=None):
    """ctypes buffers in backend.cpp's binding order, with backing storage."""
    counts, parameters, execution = native_binding(record)
    entry, memory, buffers = record["entry"], {}, []
    for name in layout.parameter_names(entry):
        dtype = layout.parameter_type(entry, name)
        if name in counts:
            values = (data or {}).get(name, [0] * counts[name])
        else:
            value = parameters[name]
            values = [value] if isinstance(value, int) else value
        memory[name] = (CTYPES[dtype] * len(values))(*values)
        buffers.append(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(memory[name]),
                len(values),
                int(name == "y"),
            )
        )
    return buffers, memory, execution


@pytest.mark.parametrize("fault", ("dtype", "direction", "empty", "scalar", "names"))
def test_storage_rejects_native_identity_faults(fault):
    record = RECORDS[1]
    buffers, memory, _ = native_buffers(record)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    if fault == "dtype":
        supplied["x"].dtype = b"float32"
    elif fault == "direction":
        supplied["x"].output = 1
    elif fault == "empty":
        supplied["scales"].count = 0
    elif fault == "scalar":
        supplied["in_vec_size"].count = 2
    else:
        supplied.pop("biases")
    with pytest.raises(ValueError):
        layout.read_storage(record["entry"], supplied)
    assert memory


def test_storage_rejects_outputs_that_alias_inputs():
    record = RECORDS[0]
    buffers, memory, _ = native_buffers(record)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    supplied["y"].data = supplied["x"].data
    supplied["y"].count = 1
    with pytest.raises(ValueError, match="overlaps"):
        layout.read_storage(record["entry"], supplied)
    assert memory


def physical_layout(target, dtype):
    if dtype in ("float16", "bfloat16") and target != "metal":
        physical = "uint16" if target == "directx" else "float32"
    else:
        physical = {"float32": "float32"}.get(dtype, dtype)
    scalar = {"elementType": physical, "elementStrideBytes": dispatch.STRIDES[physical]}
    if dtype == "float16" and target == "directx":
        scalar.update(
            storageLayout="hlsl-structured-buffer",
            runtimeSized=True,
            storageEncoding={
                "encoding": "ieee754-binary16",
                "logicalElementType": "float16",
            },
        )
    return scalar


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault", (None, "guard", "missing", "shape", "trace", "unbound", "readonly")
)
@pytest.mark.parametrize("name", ("quad-d128-stacked", "fast-stacked", "fast-rows"))
def test_dispatch_commits_the_output_only_after_validation(
    tmp_path, monkeypatch, target, fault, name
):
    record = next(item for item in RECORDS if item["name"] == name)
    entry, dtype = record["entry"], record["dtype"]
    buffers, memory, execution = native_buffers(record)
    descriptor = {"artifact": {}, "bindings": []}
    for parameter in (
        *layout.MATRIX_PARAMETERS,
        *layout.SCALAR_PARAMETERS,
        *layout.BATCH_PARAMETERS,
    ):
        if fault == "unbound" and parameter == "w_strides":
            continue
        kind = layout.parameter_type(entry, parameter)
        descriptor["bindings"].append(
            {
                "name": f"binding_{parameter}",
                "kind": "buffer",
                "access": (
                    "read_write" if parameter == "y" and fault != "readonly" else "read"
                ),
                "provenance": {"sourceResource": {"parameter": parameter}},
                "scalarLayout": physical_layout(target, kind),
            }
        )
    host = SimpleNamespace(
        target=target,
        quantized_matmuls=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    requests = []

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        requests.append((inputs, kwargs))
        return outputs

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)
    one = {"float32": 0x3F800000, "float16": 0x3C00, "bfloat16": 0x3F80}[dtype]

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        value = outputs["binding_y"]
        count = record["batches"] * record["M"] * record["N"]
        value["values"][:count] = dispatch.encode([one] * count, dtype, value["dtype"])[
            "values"
        ]
        if fault == "guard":
            value["values"][-1] ^= 1
        elif fault == "missing":
            outputs.clear()
        elif fault == "shape":
            value["shape"] = [1]
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "execute", execute)
    if fault == "trace":
        host.trace.mkdir()
    supplied = (runtime.Buffer * len(buffers))(*buffers)
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    call = lambda: dispatch.dispatch(
        host, entry, supplied, len(buffers), memory["y"]._length_, launch
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            call()
        assert host.dispatch_count == 0
        assert not any(memory["y"])
        return
    call()
    assert list(memory["y"]) == [one] * len(memory["y"])
    assert host.dispatch_count == 1
    ((inputs, kwargs),) = requests
    assert kwargs["expected_target"] == target
    assert set(kwargs["observe_inputs"]) == set(inputs) - {"binding_y"}
    padding = 0
    if layout.signature(entry)["kernel"] == "qmv_quad":
        groups = record["K"] // record["groupSize"]
        padding = (
            max(0, 64 * -(-record["N"] // 64) - record["N"]) * groups
            if record["batches"] == 1
            else (64 - record["N"] % 64) % 64 * groups
        )
    for parameter in ("scales", "biases"):
        assert len(inputs[f"binding_{parameter}"]["values"]) == (
            len(memory[parameter]) + dispatch.GUARD_COUNT + padding
        )
    if not layout.signature(entry)["batched"]:
        assert inputs["binding_x_shape"]["values"] == [record["M"], record["K"]]
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["entry"] == entry and event["matmulMetadata"]["N"] == record["N"]


def test_package_configs_scope_target_options_and_assertions(tmp_path):
    batched = "affine_qmv_fast_float_gs_64_b_4_batch_1"
    unbatched = "affine_qmv_quad_float_gs_64_b_4_d_64_batch_0"
    gl = matmul_packages.project_config(tmp_path, "out", "opengl", batched)
    assert [item.expression for item in gl.index_range_assertions] == list(
        matmul_packages.BATCH_INDEX_EXPRESSIONS
    )
    assert {item.value_range.maximum for item in gl.index_range_assertions} == {
        layout.MAX_ELEMENTS
    }
    for target in ("opengl", "directx", "metal"):
        config = matmul_packages.project_config(tmp_path, "out", target, unbatched)
        if target != "opengl":
            assert not matmul_packages.project_config(
                tmp_path, "out", target, batched
            ).index_range_assertions
        assert not config.index_range_assertions
        options = config.source_options["metal"]
        assert options["preserve_resource_origins"] is True
        assert options["target_options"]["directx"] == {
            "software_subgroup_width": 32,
            "relative_wave_shuffle_out_of_range": "self",
        }
        assert config.entry_points == {matmul_packages.SOURCE: (unbatched,)}


def test_runtime_routes_matmul_entries_before_quantization(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = object()
    host.quantized_matmuls = None
    entry = "affine_qmv_float_gs_32_b_3_batch_0"
    assert host._entry_available(entry.encode()) == 0
    host.quantized_matmuls = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"affine_qmv_wide_float_gs_32_b_3_batch_0") == 0
    assert host._entry_available(b"affine_quantize_float_gs_32_b_3") == 1
    calls = []
    monkeypatch.setattr(
        runtime.matmul_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    monkeypatch.setattr(
        runtime.quantization_dispatch, "dispatch", lambda *args: calls.append("q")
    )
    host.dispatch(entry, [], 7, 1, launch=None)
    host.dispatch("affine_quantize_float_gs_32_b_3", [], 4, 32, launch=None)
    assert calls == [entry, "q"]


def test_preparation_replaces_the_quantized_matmul_stub(monkeypatch):
    replaced = []

    def record(text, before, after):
        replaced.append(before)
        return text

    monkeypatch.setattr(prepare, "replace_once", record)
    prepare.adapted_sources("", "", "")
    assert "NO_GPU(QuantizedMatmul)" in replaced
    assert "NO_GPU(GatherQMM)" not in replaced and "NO_GPU(Matmul)" not in replaced
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert "void QuantizedMatmul::eval_gpu(" in backend
    for text in (
        '"affine_qmv_quad_"',
        '"affine_qmv_fast_"',
        "(K == 64 || K == 128) && power_of_two",
        "pack_factor * (bits_ == 2 ? 1 : 2) * 32",
        "{32, quad ? 1u : 2u, 1}",
    ):
        assert text in backend


def test_reference_rounding_and_exact_case_data():
    import numpy as np

    values = np.array([1.0 + 2**-8, 1.0 + 3 * 2**-8, 1.0 + 2**-9, -(2**-133)])
    assert verify.rounded(values, "bfloat16").tolist() == [
        1.0,
        1.0 + 2**-6,
        1.0,
        -(2**-133),
    ]
    with pytest.raises(ValueError, match="exact"):
        verify.rounded(np.array([1.0 + 2**-30]), "float32")
    record = next(item for item in RECORDS if item["name"] == "qmv-shared-weights")
    data = verify.case_data(record, 7)
    k, group, bits = record["K"], record["groupSize"], record["bits"]
    words = data["w"].reshape(record["N"], -1)
    codes = [
        [
            (int.from_bytes(row.astype("<u4").tobytes(), "little") >> (bits * i))
            & ((1 << bits) - 1)
            for i in range(k)
        ]
        for row in words
    ]
    weights = np.array(codes) * np.repeat(data["scales"], group, axis=-1) + np.repeat(
        data["biases"], group, axis=-1
    )
    expected = np.einsum("bmk,nk->bmn", data["x"][::2], weights)
    assert data["expected"].shape == (3, 1, record["N"])
    assert np.array_equal(data["expected"], verify.rounded(expected, "float16"))


def test_trace_validation_requires_the_routed_entries():
    trace = [
        {
            "entry": record["entry"],
            "target": "metal",
            "matmulMetadata": {key: record[key] for key in ("M", "N", "K", "batches")},
        }
        for record in RECORDS
    ]
    verify.validate_trace(trace, RECORDS, "metal")
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace[1:], RECORDS, "metal")
    trace[0]["matmulMetadata"]["M"] += 1
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace, RECORDS, "metal")


def test_ci_runs_the_routing_on_every_half_host_platform():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["half-host"]
    assert {item["target"] for item in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "opengl",
        "directx",
    }
    names = [item.get("name") for item in job["steps"]]
    name = "Execute affine quantized matrix-vector products through MLX"
    step = job["steps"][names.index(name)]
    assert "if" not in step and "continue-on-error" not in step
    assert "portable_host.verify_quantized_matmul" in step["run"]
    assert "test_portable_quantized_matmul.py" in step["run"]
    assert "--packages .mlx-portable-half/base/packages" in step["run"]
    assert "tee .mlx-portable-half/quantized-matmul.log" in step["run"]
    assert step["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    assert names.index("Build adapted upstream MLX") < names.index(name)
    assert names.index(name) < names.index("Retain half execution evidence")
