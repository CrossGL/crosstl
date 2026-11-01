"""AddMM host routing, epilogue launch contracts and their verification cases."""

import copy
import ctypes
import math
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import addmm_layout, gemm_layout
from demos.integrations.mlx.portable_host import gemm_splitk_layout as splitk_layout
from demos.integrations.mlx.portable_host import gemv_layout, prepare, runtime
from demos.integrations.mlx.portable_host import verify_addmm as verify
from demos.integrations.mlx.portable_host import verify_matmul

HOST = Path(addmm_layout.__file__).parent
RECORDS = [verify.case_record(case) for case in verify.CASES]
BY_NAME = {record["name"]: record for record in RECORDS}
CTYPES = {
    "float32": ctypes.c_float,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}


def family(name):
    return [record for record in RECORDS if record["family"] == name]


def span(shape, strides):
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))


def bound(record, name):
    """The (shape, strides) backend.cpp binds: copies are row-major."""
    shape, strides = record[name]["shape"], record[name]["strides"]
    if record["route"]["copies"][name]:
        strides = addmm_layout._row_strides(shape)
    return shape, strides


def gemv_binding(record):
    route = record["route"]
    mat, vec = ("b", "a") if route["N"] != 1 else ("a", "b")
    counts = {
        "mat": span(*bound(record, mat)),
        "in_vec": span(*bound(record, vec)),
        "out_vec": route["batches"] * route["outVecSize"],
    }
    parameters = {
        "in_vec_size": route["K"],
        "out_vec_size": route["outVecSize"],
        "marix_ld": route["matrixLd"],
        "batch_ndim": len(route["batchShape"]),
        "batch_shape": list(route["batchShape"]),
        "vector_batch_stride": list(route["vectorBatchStride"]),
        "matrix_batch_stride": list(route["matrixBatchStride"]),
    }
    if "axpby" in route:
        counts["bias"] = span(*bound(record, "c"))
        parameters.update(
            alpha=route["axpby"]["alpha"],
            beta=route["axpby"]["beta"],
            bias_stride=route["axpby"]["biasStride"],
            bias_batch_stride=list(route["biasBatchStride"]),
        )
    execution = {
        "workgroupCount": [
            -(-route["outVecSize"] // gemv_layout.outputs_per_group(record["entry"])),
            1,
            route["batches"],
        ],
        "workgroupSize": list(gemv_layout.workgroup_size(record["entry"])),
    }
    return counts, parameters, execution


def gemm_binding(record):
    route = record["route"]
    counts = {
        "A": span(*bound(record, "a")),
        "B": span(*bound(record, "b")),
        "C": span(*bound(record, "c")),
        "D": route["batches"] * route["M"] * route["N"],
    }
    parameters = {
        "params": dict(route["params"]),
        "batch_shape": list(route["batchShape"]),
        "batch_strides": list(route["batchStrides"]),
        "addmm_params": dict(route["addmm"]),
    }
    execution = {
        "workgroupCount": [
            route["params"]["tiles_n"],
            route["params"]["tiles_m"],
            route["batches"],
        ],
        "workgroupSize": [32, 2, 2],
    }
    return counts, parameters, execution


def accum_binding(record):
    route = record["route"]
    m, n = route["M"], route["N"]
    counts = {"C_split": route["partitions"] * m * n, "D": m * n}
    if splitk_layout.signature(route["accum"])["kind"] == "axbpy":
        counts["C"] = span(*bound(record, "c"))
    execution = {"workgroupCount": [n, m, 1], "workgroupSize": [1, 1, 1]}
    return counts, dict(route["accumParams"]), execution


def test_cases_cover_every_addmm_route():
    assert {record["family"] for record in RECORDS} == set(verify.KERNELS)
    for name, _, matches, covered in verify.FAMILIES:
        assert covered(family(name)), name
    assert verify._scale_covered(family("scale"))
    import numpy as np

    for seed, record in enumerate(RECORDS):
        data = verify.case_data(np, record, seed)
        assert list(data["expected"].shape) == record["c"]["shape"]
        if record["family"] not in ("empty", "scale"):
            assert np.any(data["expected"] != 0)


def test_routes_mirror_addmm_eval_gpu():
    # matmul.cpp folds a contiguous batch into M only if C's batches follow.
    folded = BY_NAME["gemm-folded"]["route"]
    assert (folded["M"], folded["batches"], folded["addmm"]["batch_stride_c"]) == (
        48,
        1,
        0,
    )
    batched = BY_NAME["gemm-has-batch"]["route"]
    assert batched["batchShape"] == [2, 3]
    assert batched["batchStrides"] == [2304, 768, 0, 1920, 1920, 640]
    transposed = BY_NAME["gemm-transposed"]["route"]["addmm"]
    assert (transposed["ldc"], transposed["fdc"]) == (1, 33)
    assert BY_NAME["gemm-bias"]["route"]["addmm"]["ldc"] == 0
    assert BY_NAME["gemm-copied"]["route"]["copies"] == {
        "a": True,
        "b": False,
        "c": True,
    }
    # gemv_axbpy binds C's last stride whichever operand is the matrix.
    assert BY_NAME["matrix-vector"]["route"]["axpby"]["biasStride"] == 1
    assert BY_NAME["vector-batched"]["route"]["biasBatchStride"] == [64]
    assert "axpby" not in BY_NAME["vector-plain"]["route"]
    assert BY_NAME["vector-plain"]["entry"].endswith("_axpby0")
    accum = BY_NAME["splitk-plain"]["route"]
    assert accum["accum"].endswith("_float32") and set(accum["accumParams"]) == set(
        splitk_layout.SCALARS
    )
    assert BY_NAME["splitk-transposed"]["route"]["accumParams"]["fdc"] == 32
    assert BY_NAME["empty-k-copy"]["route"]["copy"] is True
    assert BY_NAME["empty-k-scale"]["route"]["copy"] is False
    # The multiplication copies the broadcast beta first.
    assert BY_NAME["empty-k-scale"]["copies"] == 1


@pytest.mark.parametrize(
    "alpha, beta, constants",
    (
        (1.0, 1.0, (True, False)),
        (0.5, 1.0, (True, True)),
        (1.0, 0.0, (True, True)),
        (0.0, 2.0, (True, True)),
        # steel_matmul_regular_axpby reads no C for alpha 0 and beta 1.
        (0.0, 1.0, (False, False)),
    ),
)
def test_epilogue_constants_follow_steel_matmul_regular_axpby(alpha, beta, constants):
    assert gemm_layout.epilogue(alpha, beta) == constants
    assert gemm_layout.epilogue() == (False, False)
    use_out_source, do_axpby = constants
    entry = gemm_layout.select("float32", False, False, 64, 64, 32, 1, alpha, beta)
    info = gemm_layout.signature(entry)["constants"]
    assert (info["use_out_source"], info["do_axpby"]) == constants
    assert addmm_layout._axpby(alpha, beta) == (alpha != 1.0 or beta != 0.0)


def test_addmm_without_epilogue_takes_matmuls_entries():
    for record in RECORDS:
        if record["family"] not in ("gemv", "gemm", "splitk"):
            continue
        a, b = record["a"], record["b"]
        plain = addmm_layout.route(
            record["dtype"],
            (a["shape"], a["strides"]),
            (b["shape"], b["strides"]),
            (record["c"]["shape"], record["c"]["strides"]),
            1.0,
            0.0,
        )[1]
        if record["family"] == "gemv":
            assert plain["entry"] == record["entry"].replace("_axpby1", "_axpby0")
        if record["family"] == "splitk":
            assert plain["accum"] == record["route"]["accum"].removesuffix("_axbpy")
        if not any(record["route"]["copies"].values()) and len(a["shape"]) == 2:
            _, matmul = gemm_layout.matmul_route(
                record["dtype"], a["shape"], a["strides"], b["shape"], b["strides"]
            )
            assert matmul.get("accum", "") == plain.get("accum", "")
            if record["family"] == "gemv":
                assert matmul["entry"] == plain["entry"]


@pytest.mark.parametrize("record", family("gemv"), ids=lambda record: record["name"])
def test_gemv_plan_accepts_each_addmm_launch(record):
    counts, parameters, execution = gemv_binding(record)
    metadata = gemv_layout.plan(
        record["entry"], counts, parameters, counts["out_vec"], execution
    )
    assert metadata.get("axpby") == record["route"].get("axpby")
    assert ("axpby" in metadata) is (metadata["inert"] == {})


@pytest.mark.parametrize(
    "fault",
    (
        "bias-short",
        "bias-stride",
        "bias-batch",
        "plain-scales",
        "alpha-type",
        "missing",
    ),
)
def test_gemv_plan_rejects_epilogues_that_differ_from_mlx(fault):
    record = BY_NAME["vector-batched"]
    counts, parameters, execution = gemv_binding(record)
    if fault == "bias-short":
        counts["bias"] -= 1
    elif fault == "bias-stride":
        parameters["bias_stride"] = -1
    elif fault == "bias-batch":
        parameters["bias_batch_stride"] = [64, 0]
    elif fault == "plain-scales":
        parameters.update(alpha=1.0, beta=0.0)
    elif fault == "alpha-type":
        parameters["alpha"] = 1
    else:
        del parameters["beta"]
    with pytest.raises(ValueError):
        gemv_layout.plan(
            record["entry"], counts, parameters, counts["out_vec"], execution
        )


def native_buffers(names, types, counts, parameters, output, records=()):
    memory, buffers = {}, []
    for name in names:
        dtype = types(name)
        if name in records:
            value = records[name]
            memory[name] = gemm_layout.GEMMAddMMParams(**value)
            data, count = ctypes.addressof(memory[name]), ctypes.sizeof(memory[name])
        else:
            value = [0] * counts[name] if name in counts else parameters[name]
            values = value if isinstance(value, list) else [value]
            memory[name] = (CTYPES[dtype] * len(values))(*values)
            data, count = ctypes.addressof(memory[name]), len(values)
        buffers.append(
            runtime.Buffer(
                name.encode(), dtype.encode(), data, count, int(name == output)
            )
        )
    return {buffer.name.decode(): buffer for buffer in buffers}, memory


def test_gemv_storage_reads_the_axpby_bindings():
    record = BY_NAME["vector-two-batch-axes"]
    entry = record["entry"]
    counts, parameters, _ = gemv_binding(record)
    names = gemv_layout.host_parameters(entry)
    assert names[2] == "bias" and names[-1] == "bias_stride"
    supplied, memory = native_buffers(
        names,
        lambda name: gemv_layout.parameter_type(entry, name),
        counts,
        parameters,
        "out_vec",
    )
    read_counts, values = gemv_layout.read_storage(entry, supplied)
    assert read_counts == counts and values == parameters
    supplied.pop("bias")
    with pytest.raises(ValueError, match="do not match"):
        gemv_layout.read_storage(entry, supplied)
    with pytest.raises(ValueError, match="do not match"):
        gemv_layout.read_storage(entry.replace("_axpby1", "_axpby0"), {**supplied})


@pytest.mark.parametrize("record", family("gemm"), ids=lambda record: record["name"])
def test_gemm_plan_accepts_each_addmm_launch(record):
    counts, parameters, execution = gemm_binding(record)
    metadata = gemm_layout.plan(
        record["entry"], counts, parameters, counts["D"], execution
    )
    assert metadata["addmm"] == record["route"]["addmm"] and "inert" not in metadata


@pytest.mark.parametrize(
    "fault",
    ("c-short", "batch-stride-c", "c-strides", "scales", "ldc", "missing", "matmul"),
)
def test_gemm_plan_rejects_epilogues_that_differ_from_mlx(fault):
    record = BY_NAME["gemm-has-batch"]
    entry = record["entry"]
    counts, parameters, execution = gemm_binding(record)
    addmm = parameters["addmm_params"]
    if fault == "c-short":
        counts["C"] -= 1
    elif fault == "batch-stride-c":
        addmm["batch_stride_c"] += 1
    elif fault == "c-strides":
        parameters["batch_strides"] = parameters["batch_strides"][:4]
    elif fault == "scales":
        # do_axpby would be set for these.
        addmm.update(alpha=0.5)
    elif fault == "ldc":
        addmm["ldc"] = -1
    elif fault == "missing":
        del parameters["addmm_params"]
    else:
        entry = entry.replace("use_out_source_t", "use_out_source_n")
        del parameters["addmm_params"]
        parameters["batch_strides"] = parameters["batch_strides"][:4]
        counts.pop("C")
        assert gemm_layout.plan(entry, counts, parameters, counts["D"], execution)
        entry = entry.replace("do_axpby_n", "do_axpby_t")
    with pytest.raises(ValueError):
        gemm_layout.plan(entry, counts, parameters, counts["D"], execution)


def test_gemm_storage_reads_gemm_addmm_params():
    record = BY_NAME["gemm-transposed"]
    entry = record["entry"]
    counts, parameters, _ = gemm_binding(record)
    names = gemm_layout.host_parameters(entry)
    assert names[-2:] == gemm_layout.EPILOGUE_PARAMETERS
    flat = {
        **parameters,
        "params": [parameters["params"][name] for name in gemm_layout.PARAMS_FIELDS],
    }
    supplied, memory = native_buffers(
        names,
        lambda name: gemm_layout.parameter_type(entry, name),
        counts,
        flat,
        "D",
        records={"addmm_params": parameters["addmm_params"]},
    )
    assert ctypes.sizeof(gemm_layout.GEMMAddMMParams) == 24
    read_counts, values = gemm_layout.read_storage(entry, supplied)
    assert read_counts == counts and values == parameters
    supplied["addmm_params"].count = 16
    with pytest.raises(ValueError):
        gemm_layout.read_storage(entry, supplied)


@pytest.mark.parametrize("record", family("splitk"), ids=lambda record: record["name"])
def test_splitk_accumulation_plan_accepts_each_addmm_launch(record):
    entry = record["route"]["accum"]
    counts, parameters, execution = accum_binding(record)
    metadata = splitk_layout.plan(entry, counts, parameters, counts["D"], execution)
    axbpy = splitk_layout.signature(entry)["kind"] == "axbpy"
    assert ("axpby" in metadata) is axbpy
    if axbpy:
        assert metadata["axpby"] == {
            key: parameters[key] for key in ("alpha", "beta", "ldc", "fdc")
        }


@pytest.mark.parametrize("fault", ("c-short", "fdc", "plain", "type"))
def test_splitk_accumulation_rejects_epilogues_that_differ_from_mlx(fault):
    record = BY_NAME["splitk-transposed"]
    entry = record["route"]["accum"]
    counts, parameters, execution = accum_binding(record)
    if fault == "c-short":
        counts["C"] -= 1
    elif fault == "fdc":
        parameters["fdc"] = -1
    elif fault == "plain":
        parameters.update(alpha=1.0, beta=0.0)
    else:
        parameters["beta"] = 1
    with pytest.raises(ValueError):
        splitk_layout.plan(entry, counts, parameters, counts["D"], execution)


def test_splitk_storage_reads_the_axbpy_scalars():
    record = BY_NAME["splitk-axpby"]
    entry = record["route"]["accum"]
    counts, parameters, _ = accum_binding(record)
    names = splitk_layout.PARAMETERS["axbpy"]
    supplied, memory = native_buffers(
        names,
        lambda name: splitk_layout.parameter_type(entry, name),
        counts,
        parameters,
        "D",
    )
    assert splitk_layout.parameter_type(entry, "C") == "float16"
    read_counts, values = splitk_layout.read_storage(entry, supplied)
    assert read_counts == counts and values == parameters


def trace_events(record, target):
    """The product events one case's launch checks read."""
    route = record["route"]
    if record["family"] == "gemv":
        counts, parameters, _ = gemv_binding(record)
        metadata = {
            "K": route["K"],
            "outVecSize": route["outVecSize"],
            "matrixLd": route["matrixLd"],
            "batches": route["batches"],
            **({"axpby": route["axpby"]} if "axpby" in route else {}),
        }
        return [
            {
                "entry": record["entry"],
                "target": target,
                "gemvMetadata": metadata,
                "parameters": parameters,
            }
        ]
    if record["family"] == "gemm":
        _, parameters, _ = gemm_binding(record)
        metadata = {key: route[key] for key in ("M", "N", "K", "batches")}
        return [
            {
                "entry": record["entry"],
                "target": target,
                "gemmMetadata": {**metadata, "addmm": route["addmm"]},
                "parameters": parameters,
            }
        ]
    if record["family"] == "splitk":
        metadata = {key: route[key] for key in ("M", "N", "K", "partitions")}
        return [
            {
                "entry": record["entry"],
                "target": target,
                "splitkMetadata": metadata,
                "parameters": {"params": route["params"]},
                "outputHash": record["name"],
            },
            {
                "entry": route["accum"],
                "target": target,
                "splitkMetadata": {"partitions": route["partitions"]},
                "parameters": route["accumParams"],
                "inputHashes": {"C_split": record["name"]},
            },
        ]
    if record["family"] == "scale":
        last = (
            "ggn1_dynamic_copyuint32uint32"
            if route["copy"]
            else f"vv_Multiply{record['dtype']}"
        )
        copies = [{"entry": "v_copyuint16uint16", "target": target}] * record["copies"]
        return copies + [{"entry": last, "target": target}]
    return []


def traced(records, target="opengl"):
    """Records with their dispatch windows, and the trace they produce."""
    events, placed = [], []
    for record in records:
        produced = trace_events(record, target)
        placed.append(
            {**record, "dispatchStart": len(events), "dispatchCount": len(produced)}
        )
        events += produced
    return events, placed


def test_trace_validation_requires_the_routed_epilogues():
    events, records = traced(RECORDS)
    verify.validate_trace(events, records, "opengl")
    with pytest.raises(RuntimeError, match="routed entries"):
        verify.validate_trace(events, records, "directx")
    for name, change in (
        ("vector-bias", lambda event: event["gemvMetadata"]["axpby"].update(beta=2.0)),
        (
            "vector-batched",
            lambda event: event["parameters"].update(bias_batch_stride=[0]),
        ),
        (
            "gemm-axpby",
            lambda event: event["parameters"]["addmm_params"].update(fdc=2),
        ),
    ):
        changed = copy.deepcopy(events)
        change(changed[next(r for r in records if r["name"] == name)["dispatchStart"]])
        with pytest.raises(RuntimeError, match="differs"):
            verify.validate_trace(changed, records, "opengl")
    # K == 0 scales C by beta with a multiplication, after its copies.
    scaled = next(r for r in records if r["name"] == "empty-k-scale")
    changed = copy.deepcopy(events)
    changed[scaled["dispatchStart"] + scaled["dispatchCount"] - 1][
        "entry"
    ] = "vv_Addfloat16"
    with pytest.raises(RuntimeError, match="scaling"):
        verify.validate_trace(changed, records, "opengl")
    events, kept = traced([r for r in RECORDS if r["name"] != "vector-plain"])
    with pytest.raises(RuntimeError, match="cover"):
        verify.validate_trace(events, kept, "opengl")


def test_backend_ports_addmm_eval_gpu():
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    start = backend.index("void AddMM::eval_gpu(")
    body = backend[start : backend.index("\n}\n", start)]
    for text in (
        "if (out.size() == 0) {",
        "if (inputs[0].shape(-1) == 0) {",
        "if (beta_ == 1.0f) {",
        'dispatch_binary({c, scale}, out, "Multiply");',
        "collapse_batches(a, b, c);",
        "C_batch_stride.back() == M * c.strides()[c.ndim() - 2]",
        "const AddmmEpilogue epilogue{c, C_batch_stride, alpha_, beta_};",
        "if (std::min(M, N) == 1) {",
    ):
        assert text in body, text
    for text in (
        "epilogue && (epilogue->alpha != 1.0f || epilogue->beta != 0.0f);",
        '(do_axpby ? "_axbpy" : "")',
        '"_axpby" + (do_axpby ? "1" : "0")',
        "static_assert(sizeof(GemmAddmmParams) == 24",
        "epilogue->batch_stride.back(),",
    ):
        assert text in backend, text
    for name in gemv_layout.AXPBY_PARAMETERS + gemm_layout.EPILOGUE_PARAMETERS:
        assert f'{{"{name}", ' in backend, name
    for name in splitk_layout.AXBPY_SCALARS + ("C",):
        assert f'push_back({{"{name}", ' in backend or f'{{"{name}", ' in backend, name


def test_preparation_replaces_the_addmm_stub(monkeypatch):
    replaced = []

    def record(text, before, after):
        replaced.append(before)
        return text

    monkeypatch.setattr(prepare, "replace_once", record)
    prepare.adapted_sources("", "", "")
    assert "NO_GPU(AddMM)" in replaced


def test_ci_runs_the_addmm_route_on_every_quantized_host_platform():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    steps = workflow["jobs"]["quantized-host"]["steps"]
    names = [item.get("name") for item in steps]
    execute = steps[names.index("Execute AddMM products through MLX")]
    assert (
        names.index("Execute floating-point matrix products through MLX")
        < names.index("Execute AddMM products through MLX")
        < names.index("Retain quantized execution evidence")
    )
    assert "if" not in execute and "continue-on-error" not in execute
    assert "set -euo pipefail" in execute["run"]
    assert "portable_host.verify_addmm" in execute["run"]
    assert "--packages .mlx-portable-quantized/base/packages" in execute["run"]
    assert "tee .mlx-portable-quantized/addmm.log" in execute["run"]
    assert execute["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    validate = steps[names.index("Validate quantized host contracts")]["run"]
    assert "test_portable_addmm.py" in validate


def test_verify_matmul_and_addmm_share_view_layouts():
    for view in ("dense", "swap", "step", ("broadcast", (3, 4, 5))):
        stored = (1, 4, 5) if view != "dense" else (3, 4, 5)
        assert verify.view_layout(stored, view) == verify_matmul.view_layout(
            stored, view
        )
    assert verify.view_layout((4, 5), "flip") == ([4, 5], [-5, 1])
    assert verify.view_layout((5,), ("bias", (2, 4, 5))) == ([2, 4, 5], [0, 0, 1])
    assert math.prod(verify.view_layout((1,), ("bias", (3, 1)))[0]) == 3
