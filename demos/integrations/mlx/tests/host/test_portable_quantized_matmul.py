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
    # w holds (N, K) codes when transposed and (K, N) otherwise.
    rows, columns = (n, k) if record["transpose"] else (k, n)
    words, groups = columns * record["bits"] // 32, columns // record["groupSize"]
    x_shape = list(record["xShape"])
    x_strides = contiguous(x_shape)
    if record["layout"] == "strided":
        x_strides[0] *= 2
    w_batch = record["wBatchShape"]
    out_batch = x_shape[:-2] if w_batch else []
    w_shape = [*out_batch, rows, words]
    strides = {}
    for name, width in (("w", words), ("s", groups), ("b", groups)):
        dense = contiguous([*w_batch, rows, width])
        strides[name] = [
            0 if extent == 1 and target > 1 else stride
            for extent, target, stride in zip(w_batch, out_batch, dense)
        ] + dense[len(w_batch) :]
    counts = {
        "x": span(x_shape, x_strides),
        "w": math.prod(w_batch) * rows * words,
        "scales": math.prod(w_batch) * rows * groups,
        "biases": math.prod(w_batch) * rows * groups,
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
        for kernel in ("qmv", "qmv_fast", "qvm"):
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
        f"affine_qvm_{stem}_d_64_batch_0",
        f"affine_qvm_split_k_{stem}_spk_8",
        f"affine_qvm_{stem}_batch_1_hgs",
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


@pytest.mark.parametrize("group", (128, 64, 32))
@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6, 8))
def test_selection_follows_quantized_matmul_for_upstream_qvm_shapes(group, bits):
    # test_quantized.py test_qvm: (B, 1, K) vectors times (B, K, N) weights,
    # with MLX's names M for the outputs and N for the reduction.
    for outputs in (32, 128, 256):
        if outputs < group:
            continue
        for k in (128, 256, 67):
            for batches in (0, 1, 3, 8):
                entry = layout.select(
                    "float32", group, bits, 1, outputs, k, max(batches, 1), False
                )
                assert entry == (
                    f"affine_qvm_float_gs_{group}_b_{bits}" f"_batch_{int(batches > 1)}"
                )
                assert not layout.transposed(entry)
                assert layout.tile_rows(entry) == 64
                assert layout.workgroup_size(entry) == (32, 2, 1)
    # Longer reductions go to qvm_split_k, and four vectors to qmm_n.
    with pytest.raises(ValueError, match="Split-K"):
        layout.select("float32", group, bits, 1, 128, 1024, 1, False)
    with pytest.raises(ValueError, match="matrix-matrix"):
        layout.select("float32", group, bits, 4, 128, 256, 1, False)
    assert layout.select("float32", group, 4, 3, 128, 1023, 1, False).startswith(
        "affine_qvm_"
    )
    # Transposed weights keep get_qmv_batch_limit's bound.
    assert [layout.vector_limit(k, k, True) for k in (2048, 4096, 4097)] == [18, 12, 10]
    with pytest.raises(ValueError, match="matrix-matrix"):
        layout.select("float32", group, bits, 18, 128, 256, 1)


def test_verification_cases_cover_every_route_and_batch_layout():
    infos = [layout.signature(record["entry"]) for record in RECORDS]
    assert {(info["kernel"], info["batched"]) for info in infos} == {
        (kernel, batched)
        for kernel in ("qmv_quad", "qmv_fast", "qmv", "qvm")
        for batched in (False, True)
    }
    qvm = [record for record in RECORDS if not record["transpose"]]
    assert {record["bits"] for record in qvm} == {2, 3, 4, 5, 6, 8}
    assert {record["dtype"] for record in qvm} == {"float32", "float16", "bfloat16"}
    # A K tail, a K below one 32-row block, and a half-empty last tile.
    assert any(record["K"] % 32 for record in qvm)
    assert any(record["K"] < 32 for record in qvm)
    assert any(record["N"] % 64 for record in qvm)
    assert all(record["M"] < 4 and record["K"] < 1024 for record in qvm)
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
@pytest.mark.parametrize("name", ("qmv-two-batch-axes", "qvm-two-batch-axes"))
def test_plan_rejects_launches_that_differ_from_mlx(fault, name):
    record = next(item for item in RECORDS if item["name"] == name)
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


def test_plan_rejects_vector_matrix_launches_mlx_routes_elsewhere():
    record = next(item for item in RECORDS if item["name"] == "qvm-tail-rows")
    counts, parameters, execution = native_binding(record)
    # Four vectors make a qmm_n product.
    execution["workgroupCount"][0] = 4
    counts["y"] = 4 * record["N"]
    counts["x"] = 4 * record["K"]
    with pytest.raises(ValueError, match="matrix-matrix"):
        layout.plan(record["entry"], counts, parameters, counts["y"], execution)
    # The transposed kernels read w as (N, K): the same buffers do not fit.
    counts, parameters, execution = native_binding(record)
    transposed = record["entry"].replace("affine_qvm_", "affine_qmv_")
    with pytest.raises(ValueError):
        layout.plan(transposed, counts, parameters, counts["y"], execution)
    # qvm groups run along N, so N must hold whole groups.
    counts, parameters, execution = native_binding(record)
    parameters["out_vec_size"] += 1
    with pytest.raises(ValueError, match="dimensions"):
        layout.plan(record["entry"], counts, parameters, counts["y"], execution)


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
@pytest.mark.parametrize(
    "name",
    ("quad-d128-stacked", "fast-stacked", "fast-rows", "qvm-stacked", "qvm-tail-rows"),
)
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


def test_package_builds_cover_the_verification_entries(tmp_path, monkeypatch):
    import concurrent.futures

    entries = matmul_packages.verification_entries()
    assert entries == [record["entry"] for record in RECORDS]
    calls = []

    def build(root, directory, target, entry):
        calls.append((Path(directory), target, entry))
        return {"entry": entry}

    # Threads stand in for processes so the stub is visible to the workers.
    monkeypatch.setattr(
        matmul_packages, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(matmul_packages, "_build_one", build)
    records = matmul_packages.build_entries(
        tmp_path, tmp_path / "packages", "opengl", [*entries, entries[0]], 3
    )
    assert [record["entry"] for record in records] == sorted(set(entries))
    assert {call[:2] for call in calls} == {
        ((tmp_path / "packages" / matmul_packages.CACHE_DIRECTORY).resolve(), "opengl")
    }
    for entries, jobs in (([], 2), (entries, 0)):
        with pytest.raises(ValueError, match="entries and jobs"):
            matmul_packages.build_entries(tmp_path, tmp_path, "opengl", entries, jobs)
    with pytest.raises(ValueError):
        matmul_packages.build_entries(
            tmp_path,
            tmp_path,
            "opengl",
            ["affine_qvm_split_k_float_gs_32_b_4_spk_8"],
            1,
        )
    calls.clear()
    assert (
        matmul_packages.main(
            [
                "--mlx-root",
                str(tmp_path),
                "--packages",
                str(tmp_path),
                "--target",
                "metal",
            ]
            + ["--entry", entries[0], "--jobs", "1"]
        )
        == 0
    )
    assert [call[1:] for call in calls] == [("metal", entries[0])]


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
    assert "NO_GPU(GatherQMM)" not in replaced
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert "void QuantizedMatmul::eval_gpu(" in backend
    for text in (
        '"affine_qmv_quad_"',
        '"affine_qmv_fast_"',
        "(K == 64 || K == 128) && power_of_two",
        "pack_factor * (bits_ == 2 ? 1 : 2) * 32",
        "{32, quad ? 1u : 2u, 1}",
        '"affine_qvm_"',
        "!transpose_ ? 4",
        "!transpose_ && K >= 1024",
        "!transpose_ ? 2 * std::min(group_size_, 32)",
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


@pytest.mark.parametrize("name", ("qvm-shared-weights", "qvm-two-batch-axes"))
def test_untransposed_case_data_packs_groups_along_the_outputs(name):
    import numpy as np

    record = next(item for item in RECORDS if item["name"] == name)
    data = verify.case_data(record, 3)
    k, n, group, bits = record["K"], record["N"], record["groupSize"], record["bits"]
    w_batch = tuple(record["wBatchShape"])
    assert data["w"].shape == (*w_batch, k, n * bits // 32)
    assert data["scales"].shape == data["biases"].shape == (*w_batch, k, n // group)
    codes = np.array(
        [
            [
                (int.from_bytes(row.astype("<u4").tobytes(), "little") >> (bits * i))
                & ((1 << bits) - 1)
                for i in range(n)
            ]
            for row in data["w"].reshape(-1, n * bits // 32)
        ]
    ).reshape(*w_batch, k, n)
    weights = codes * np.repeat(data["scales"], group, axis=-1) + np.repeat(
        data["biases"], group, axis=-1
    )
    x = data["x"][::2] if record["layout"] == "strided" else data["x"]
    expected = np.matmul(x, weights)
    assert data["expected"].shape == (*x.shape[:-1], n)
    assert np.array_equal(data["expected"], verify.rounded(expected, record["dtype"]))


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


def test_ci_runs_the_routing_on_every_quantized_host_platform():
    import re

    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["quantized-host"]
    assert {
        item["target"]: item["os"] for item in job["strategy"]["matrix"]["include"]
    } == {"metal": "macos-26", "directx": "windows-2025", "opengl": "ubuntu-24.04"}
    assert job["if"] == "needs.ci-plan.outputs.heavy == 'true'"
    assert job["needs"] == ["ci-plan", "portable-host"]
    assert "continue-on-error" not in job
    assert "quantized-host" in workflow["jobs"]["result"]["needs"]
    names = [item.get("name") for item in job["steps"]]
    steps = dict(zip(names, job["steps"]))
    translate = steps["Translate quantized matrix-vector packages"]
    name = "Execute affine quantized matrix-vector products through MLX"
    step = steps[name]
    for item in (translate, step):
        assert "if" not in item and "continue-on-error" not in item
        assert "set -euo pipefail" in item["run"]
        assert "--packages .mlx-portable-quantized/base/packages" in item["run"]
    assert "portable_host.matmul_packages" in translate["run"]
    assert "--verification" in translate["run"] and "--jobs 3" in translate["run"]
    assert "portable_host.verify_quantized_matmul" in step["run"]
    assert "tee .mlx-portable-quantized/quantized-matmul.log" in step["run"]
    assert step["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    assert (
        "test_portable_quantized_matmul.py"
        in steps["Validate quantized host contracts"]["run"]
    )
    order = [
        "Build adapted upstream MLX",
        "Translate quantized matrix-vector packages",
        name,
        "Retain quantized execution evidence",
    ]
    assert [names.index(item) for item in order] == sorted(
        names.index(item) for item in order
    )
    retain = steps["Retain quantized execution evidence"]
    assert retain["if"] == "always()"
    assert retain["with"]["if-no-files-found"] == "error"
    deadlines = [
        int(value)
        for item in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", item.get("run", ""))
    ]
    assert deadlines == [1800, 3600, 1800, 1200, 2400, 1200]
    assert sum(deadlines) + 900 < job["timeout-minutes"] * 60
    # The half host job no longer runs the products.
    half = [item.get("name") for item in workflow["jobs"]["half-host"]["steps"]]
    assert name not in half
