"""Floating-point quantized matrix-vector routing, launch contracts and guarded commits."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import fp_matmul_layout as layout
from demos.integrations.mlx.portable_host import fp_matmul_packages
from demos.integrations.mlx.portable_host import matmul_dispatch as dispatch
from demos.integrations.mlx.portable_host import matmul_layout, runtime

HOST = Path(layout.__file__).parent
CTYPES = {
    "uint8": ctypes.c_uint8,
    "uint32": ctypes.c_uint32,
    "float32": ctypes.c_uint32,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}
# (name, mode, dtype, M, N, K, batches, transpose)
CASES = (
    ("quad-rows", "mxfp4", "float32", 1, 5, 64, 1, True),
    ("quad-stacked", "mxfp8", "float16", 1, 67, 128, 3, True),
    ("wide-rows", "nvfp4", "float32", 3, 67, 256, 1, True),
    ("wide-stacked", "mxfp4", "bfloat16", 7, 256, 512, 2, True),
    ("fast-rows", "mxfp8", "float32", 1, 256, 256, 1, True),
    ("qmv-stacked", "nvfp4", "float32", 1, 7, 48, 8, True),
    ("qvm-rows", "mxfp4", "float32", 1, 67 * 32, 128, 1, False),
    ("qvm-stacked", "nvfp4", "float16", 3, 48, 256, 3, False),
    # qvm_split_k: 32 partitions above K = 8192, else 8 with a shorter last.
    ("split-rows", "mxfp4", "float32", 1, 128, 16384, 1, False),
    ("split-stacked", "nvfp4", "float16", 2, 48, 2001, 3, False),
    ("split-tail", "mxfp8", "bfloat16", 3, 64, 1030, 1, False),
)


def contiguous(shape):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.insert(0, step)
        step *= extent
    return strides


def case(name):
    _, mode, dtype, m, n, k, batches, transpose = next(
        item for item in CASES if item[0] == name
    )
    return mode, dtype, m, n, k, batches, transpose


def native_binding(name):
    """What backend.cpp binds for one case: entry, counts, parameters and launch."""
    mode, dtype, m, n, k, batches, transpose = case(name)
    group, bits = layout.MODES[mode]
    entry = layout.select(mode, dtype, m, n, k, batches, transpose)
    rows, columns = (n, k) if transpose else (k, n)
    words, groups = columns * bits // 32, columns // group
    lead = [batches] if batches > 1 else []
    counts = {
        "x": batches * m * k,
        "w": batches * rows * words,
        "scales": batches * rows * groups,
        "y": batches * m * n,
    }
    parameters = {"in_vec_size": k, "out_vec_size": n}
    info = layout.signature(entry)
    if info["kernel"] == "qmv_wide":
        parameters["M"] = m
    if batches > 1:
        parameters.update(
            x_batch_ndims=1,
            x_shape=[*lead, m, k],
            x_strides=contiguous([*lead, m, k]),
            w_batch_ndims=1,
            w_shape=[*lead, rows, words],
            w_strides=contiguous([*lead, rows, words]),
            s_strides=contiguous([*lead, rows, groups]),
        )
    columns = -(-m // info["vectors"]) if info["vectors"] else m
    execution = {
        "workgroupCount": [columns, -(-n // layout.tile_rows(entry)), batches],
        "workgroupSize": list(layout.workgroup_size(entry)),
    }
    split = info.get("splitK")
    if split:
        # qvm_split_k always binds the views, with the partition axis added
        # and no bias strides.
        split_d = -(-k // split)
        x_shape = [*lead, m, k]
        w_shape = [*lead, rows, words]
        parameters = {
            "in_vec_size": split_d,
            "out_vec_size": n,
            "final_block_size": k - (split - 1) * split_d,
            **matmul_layout.split_views(
                x_shape,
                contiguous(x_shape),
                w_shape,
                contiguous(w_shape),
                contiguous([*lead, rows, groups]),
                None,
                split,
                k,
                groups,
            ),
        }
        counts["y"] *= split
        execution["workgroupCount"][2] *= split
    return entry, counts, parameters, execution


def native_buffers(name, data=None):
    """ctypes buffers in backend.cpp's binding order, with backing storage."""
    entry, counts, parameters, execution = native_binding(name)
    memory, buffers = {}, []
    for parameter in layout.parameter_names(entry):
        dtype = layout.parameter_type(entry, parameter)
        if parameter in counts:
            values = (data or {}).get(parameter, [0] * counts[parameter])
        else:
            value = parameters[parameter]
            values = [value] if isinstance(value, int) else value
        memory[parameter] = (CTYPES[dtype] * len(values))(*values)
        buffers.append(
            runtime.Buffer(
                parameter.encode(),
                dtype.encode(),
                ctypes.addressof(memory[parameter]),
                len(values),
                int(parameter == "y"),
            )
        )
    return entry, buffers, memory, execution


@pytest.mark.parametrize(
    "entry",
    (
        "mxfp4_qmv_float_gs_32_b_4_batch_0",
        "nvfp4_qmv_fast_float16_t_gs_16_b_4_batch_1",
        "mxfp8_qmv_quad_bfloat16_t_gs_32_b_8_d_128_batch_0",
        "nvfp4_qmv_wide_float_gs_16_b_4_nv_5_kl_16_batch_1",
        "mxfp8_qvm_float_gs_32_b_8_batch_1",
        "mxfp4_qvm_split_k_float_gs_32_b_4_spk_8",
        "nvfp4_qvm_split_k_float16_t_gs_16_b_4_spk_32",
        "mxfp8_qvm_split_k_bfloat16_t_gs_32_b_8_spk_8",
    ),
)
def test_signature_accepts_the_instantiated_specializations(entry):
    info = layout.signature(entry)
    assert (info["groupSize"], info["bits"]) == layout.MODES[info["mode"]]
    assert entry.startswith(layout.ENTRY_PREFIXES)


@pytest.mark.parametrize(
    "entry",
    (
        "mxfp4_qmv_float_gs_16_b_4_batch_0",
        "nvfp4_qmv_float_gs_32_b_4_batch_0",
        "mxfp8_qmv_float_gs_32_b_4_batch_0",
        "mxfp4_qmv_quad_float_gs_32_b_4_batch_0",
        "mxfp4_qmv_float_gs_32_b_4_d_64_batch_0",
        "mxfp4_qmv_wide_float_gs_32_b_4_batch_0",
        "mxfp4_qmv_wide_float_gs_32_b_4_nv_6_kl_16_batch_0",
        "mxfp4_qmv_wide_float_gs_32_b_4_nv_2_kl_8_batch_0",
        "mxfp4_qmv_float_gs_32_b_4_batch_0_hgs",
        "mxfp4_qmv_fast_float_gs_32_b_4_r_2_batch_0",
        "mxfp4_qvm_split_k_float_gs_32_b_4_spk_16",
        "mxfp4_qvm_split_k_float_gs_32_b_4_spk_8_batch_0",
        "nvfp4_qvm_split_k_float_gs_32_b_4_spk_8",
        "mxfp8_qvm_split_k_float_gs_32_b_4_spk_32",
        "affine_qvm_split_k_float_gs_32_b_4_spk_8",
        "affine_qmv_float_gs_32_b_4_batch_0",
    ),
)
def test_signature_refuses_entries_quantized_cpp_never_selects_here(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_wide_vector_tiles_follow_quantized_cpp():
    for m in range(2, 18):
        tiles = (m + 4) // 5
        assert layout.wide_vectors(m) == (m + tiles - 1) // tiles
    assert [layout.wide_vectors(m) for m in (2, 5, 6, 7, 10, 11, 17)] == [
        2,
        5,
        3,
        4,
        5,
        4,
        5,
    ]


def test_selection_follows_dispatch_qmv_and_qvm():
    select = layout.select
    assert select("mxfp4", "float32", 1, 7, 64, 1) == (
        "mxfp4_qmv_quad_float_gs_32_b_4_d_64_batch_0"
    )
    # qmv_quad precedes qmv_wide, which every floating-point mode uses.
    assert select("mxfp8", "float32", 3, 7, 128, 2) == (
        "mxfp8_qmv_quad_float_gs_32_b_8_d_128_batch_1"
    )
    assert select("nvfp4", "float16", 3, 64, 256, 1) == (
        "nvfp4_qmv_wide_float16_t_gs_16_b_4_nv_3_kl_16_batch_0"
    )
    assert select("mxfp8", "float32", 1, 256, 256, 3) == (
        "mxfp8_qmv_fast_float_gs_32_b_8_batch_1"
    )
    # 4-bit fast kernels need K in multiples of 512 and N in multiples of 8.
    assert select("mxfp4", "float32", 1, 256, 256, 1).startswith("mxfp4_qmv_float")
    assert select("nvfp4", "float32", 1, 4096, 512, 1).startswith("nvfp4_qmv_fast_")
    assert select("nvfp4", "float32", 1, 4092, 512, 1).startswith("nvfp4_qmv_float")
    assert select("mxfp4", "float32", 3, 128, 64, 1, transpose=False) == (
        "mxfp4_qvm_float_gs_32_b_4_batch_0"
    )
    with pytest.raises(ValueError, match="matrix-matrix"):
        select("mxfp4", "float32", 18, 64, 256, 1)
    with pytest.raises(ValueError, match="matrix-matrix"):
        select("mxfp4", "float32", 4, 64, 256, 1, transpose=False)
    # qvm_split_k from K = 1024, in 32 partitions above 8192.
    assert select("mxfp4", "float32", 1, 64, 1024, 1, transpose=False) == (
        "mxfp4_qvm_split_k_float_gs_32_b_4_spk_8"
    )
    assert select("nvfp4", "float16", 3, 48, 8192, 2, transpose=False) == (
        "nvfp4_qvm_split_k_float16_t_gs_16_b_4_spk_8"
    )
    assert select("mxfp8", "float32", 1, 128, 8193, 1, transpose=False) == (
        "mxfp8_qvm_split_k_float_gs_32_b_8_spk_32"
    )
    assert select("mxfp4", "float32", 1, 64, 1023, 1, transpose=False) == (
        "mxfp4_qvm_float_gs_32_b_4_batch_0"
    )
    assert select("mxfp4", "float32", 17, 64, 256, 1).endswith("_nv_5_kl_16_batch_0")


@pytest.mark.parametrize("name", [item[0] for item in CASES])
def test_plan_accepts_backend_launches_and_adds_inert_values(name):
    entry, counts, parameters, execution = native_binding(name)
    mode, dtype, m, n, k, batches, transpose = case(name)
    metadata = layout.plan(entry, counts, parameters, counts["y"], execution)
    assert (metadata["M"], metadata["N"], metadata["K"]) == (m, n, k)
    assert metadata["batches"] == batches and metadata["mode"] == mode
    values = metadata["parameters"]
    assert ("global_scale" in values) == layout.declares_global_scale(entry)
    if "global_scale" in values:
        assert values["global_scale"] == [layout.INACTIVE_GLOBAL_SCALE]
    assert "b_strides" not in values
    assert set(layout.BATCH_PARAMETERS) <= set(values)
    split = metadata.get("splitK")
    assert split == layout.signature(entry).get("splitK")
    if split:
        assert values == parameters and metadata["overreadPadding"] == {"scales": 0}
        assert values["x_shape"][values["x_batch_ndims"] - 1] == split
    elif batches == 1:
        rows = n if transpose else k
        assert values["x_shape"] == [m, k] and values["w_batch_ndims"] == 0
        assert values["w_shape"][0] == rows
    padding = metadata["overreadPadding"]["scales"]
    if metadata["kernel"] == "qmv_quad":
        groups = k // layout.MODES[mode][0]
        assert padding == (64 * -(-n // 64) - n) * groups
    else:
        assert padding == 0


@pytest.mark.parametrize(
    "fault",
    (
        "workgroup",
        "tiles",
        "vectors",
        "rows",
        "threads",
        "groups",
        "batch-missing",
        "batch-shape",
        "unbatched-views",
        "short-weights",
    ),
)
@pytest.mark.parametrize("name", ("wide-stacked", "qmv-stacked", "quad-rows"))
def test_plan_rejects_launches_that_differ_from_mlx(fault, name):
    entry, counts, parameters, execution = native_binding(name)
    info = layout.signature(entry)
    threads = counts["y"]
    if fault == "workgroup":
        execution["workgroupSize"] = [32, 1 if info["kernel"] != "qmv_quad" else 2, 1]
    elif fault == "tiles":
        execution["workgroupCount"][1] += 1
    elif fault == "vectors":
        execution["workgroupCount"][0] += 1
    elif fault == "rows":
        if "M" in parameters:
            parameters["M"] += 1
        else:
            execution["workgroupCount"][0] += 1
    elif fault == "threads":
        threads += 1
    elif fault == "groups":
        parameters["in_vec_size"] += 1
    elif fault == "batch-missing":
        if not info["batched"]:
            parameters["x_batch_ndims"] = 0
        else:
            parameters.pop("s_strides")
    elif fault == "batch-shape":
        if info["batched"]:
            parameters["w_shape"][-2] += 1
        else:
            execution["workgroupCount"][2] = 2
    elif fault == "unbatched-views":
        parameters["b_strides"] = [1]
    else:
        counts["w"] -= 1
    with pytest.raises(ValueError):
        layout.plan(entry, counts, parameters, threads, execution)


@pytest.mark.parametrize("fault", ("dtype", "direction", "empty", "names", "biases"))
def test_storage_rejects_native_identity_faults(fault):
    entry, buffers, memory, _ = native_buffers("qmv-stacked")
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    if fault == "dtype":
        supplied["scales"].dtype = b"float32"
    elif fault == "direction":
        supplied["x"].output = 1
    elif fault == "empty":
        supplied["scales"].count = 0
    elif fault == "names":
        supplied.pop("s_strides")
    else:
        supplied["biases"] = supplied["scales"]
    with pytest.raises(ValueError):
        layout.read_storage(entry, supplied)
    assert memory


def physical_layout(target, dtype):
    if dtype in ("float16", "bfloat16") and target != "metal":
        physical = "uint16" if target == "directx" else "float32"
    elif dtype == "uint8" and target == "opengl":
        physical = "uint32"
    else:
        physical = dtype
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
@pytest.mark.parametrize("fault", (None, "guard", "unbound", "readonly", "extra"))
@pytest.mark.parametrize(
    "name", ("quad-stacked", "wide-rows", "qmv-stacked", "split-stacked")
)
def test_dispatch_commits_the_output_only_after_validation(
    tmp_path, monkeypatch, target, fault, name
):
    entry, buffers, memory, execution = native_buffers(name)
    mode, dtype, m, n, k, batches, transpose = case(name)
    names = [*layout.parameter_names(entry)]
    if not layout.signature(entry)["batched"]:
        names += layout.BATCH_PARAMETERS
    if layout.declares_global_scale(entry):
        names.append("global_scale")
    if fault == "unbound":
        names.remove("w_strides")
    if fault == "extra":
        names.append("b_strides" if "b_strides" not in names else "global_scale")
    descriptor = {"artifact": {}, "bindings": []}
    for parameter in names:
        kind = (
            "int64"
            if parameter == "b_strides"
            else layout.parameter_type(entry, parameter)
        )
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
    cache = SimpleNamespace(get=lambda name: (descriptor, tmp_path))
    host = SimpleNamespace(
        target=target, trace=tmp_path / "trace.jsonl", dispatch_count=0
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
        count = len(memory["y"])
        value["values"][:count] = dispatch.encode([one] * count, dtype, value["dtype"])[
            "values"
        ]
        if fault == "guard":
            value["values"][-1] ^= 1
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "execute", execute)
    supplied = (runtime.Buffer * len(buffers))(*buffers)
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    call = lambda: dispatch.dispatch_with(
        host, layout, cache, entry, supplied, len(buffers), memory["y"]._length_, launch
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            call()
        assert host.dispatch_count == 0
        assert not any(memory["y"])
        return
    call()
    assert list(memory["y"]) == [one] * len(memory["y"])
    assert host.dispatch_count == 1
    ((inputs, kwargs),) = requests
    assert set(kwargs["observe_inputs"]) == set(inputs) - {"binding_y"}
    scales = inputs["binding_scales"]
    # qmv_quad reads one scale per row of its last tile past N.
    padding = 0
    if layout.signature(entry)["kernel"] == "qmv_quad":
        padding = (64 * -(-n // 64) - n) * (k // layout.MODES[mode][0])
    assert scales["dtype"] == physical_layout(target, "uint8")["elementType"]
    assert scales["values"] == [0] * len(memory["scales"]) + [
        dispatch.POISON["uint8"]
    ] * (dispatch.GUARD_COUNT + padding)
    if layout.declares_global_scale(entry):
        assert inputs["binding_global_scale"]["values"] == [0]
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["entry"] == entry and set(event["inputHashes"]) == {
        "w",
        "scales",
        "x",
    }


def test_package_configs_translate_the_block_source_with_scoped_assertions(tmp_path):
    batched = "nvfp4_qmv_float_gs_16_b_4_batch_1"
    unbatched = "mxfp4_qmv_quad_float_gs_32_b_4_d_64_batch_0"
    gl = fp_matmul_packages.project_config(tmp_path, "out", "opengl", batched)
    assert list(gl.include_patterns) == [layout.SOURCE]
    assert [item.expression for item in gl.index_range_assertions] == list(
        fp_matmul_packages.BATCH_INDEX_EXPRESSIONS
    )
    assert {item.value_range.maximum for item in gl.index_range_assertions} == {
        layout.MAX_ELEMENTS
    }
    for target in ("opengl", "directx", "metal"):
        config = fp_matmul_packages.project_config(tmp_path, "out", target, unbatched)
        assert not config.index_range_assertions
        rules = config.entry_workgroup_size_rules[layout.SOURCE]
        assert [int(size) for size in rules[unbatched]] == [32, 1, 1]


def test_runtime_routes_block_products_apart_from_block_quantization(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization, host.quantized_matmuls = object(), None
    host.fp_quantized_matmuls = None
    entry = "mxfp8_qmv_wide_float_gs_32_b_8_nv_3_kl_16_batch_0"
    assert host._entry_available(entry.encode()) == 0
    host.fp_quantized_matmuls = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"mxfp8_qmv_wide_float_gs_32_b_8_nv_6_kl_16") == 0
    assert host._entry_available(b"nvfp4_quantize_float_gs_16_b_4_hgs_false") == 1
    calls = []
    monkeypatch.setattr(
        runtime.matmul_dispatch,
        "dispatch_with",
        lambda host, layout, cache, entry, *args: calls.append((layout, entry)),
    )
    monkeypatch.setattr(
        runtime.quantization_dispatch, "dispatch", lambda *args: calls.append("q")
    )
    host.dispatch(entry, [], 8, 1, launch=None)
    host.dispatch("nvfp4_quantize_float_gs_16_b_4_hgs_false", [], 3, 16, launch=None)
    assert calls == [(layout, entry), "q"]


def test_backend_routes_block_modes_through_the_floating_point_kernels():
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    start = backend.index("void dispatch_fp_quantized_matvec(")
    body = backend[start : backend.index("void QuantizedMatmul::eval_gpu", start)]
    for fragment in (
        '"_qmv_quad_"',
        '"_qmv_wide_"',
        '"_qmv_fast_"',
        '"_qvm_"',
        '"_kl_16"',
        '{"scales", "uint8"',
        '{"M", "int32"',
        "QuantizedBatchViews views(x, w, scales);",
        "dispatch_qvm_split_k(x, w, scales, std::nullopt, out, mode,",
    ):
        assert fragment in body
    assert '"biases"' not in body and "vector_limit" not in body
    routing = backend[backend.index("void QuantizedMatmul::eval_gpu") :]
    # Products of at least the vector limit's rows take the qmm routes first.
    vectors = routing.split("array biases")[0]
    assert vectors.index("if (M >= vector_limit) {") < vectors.index(
        "dispatch_fp_quantized_matvec("
    )
    # The split-K kernels of every mode share one host route, which binds
    # biases and their strides only for affine weights.
    start = backend.index("void dispatch_qvm_split_k(")
    split = backend[start : backend.index("void dispatch_fp_quantized_matvec(", start)]
    for fragment in (
        "const std::optional<array>& biases,",
        'std::string(mode) + "_qvm_split_k_" +',
        '{"scales", biases ? storage : "uint8"',
        'buffers.push_back(vector("b_strides", "int64", b_strides));',
    ):
        assert fragment in split
    assert '"affine", group_size_, bits_, M, N, K);' in routing


@pytest.mark.parametrize(
    "fault",
    (
        "final",
        "partition-axis",
        "partition-stride",
        "partitions",
        "k-view",
        "bias-strides",
        "short-scales",
    ),
)
@pytest.mark.parametrize("name", ("split-rows", "split-stacked", "split-tail"))
def test_split_plan_rejects_partitions_quantized_cpp_would_not_make(fault, name):
    entry, counts, parameters, execution = native_binding(name)
    if fault == "final":
        parameters["final_block_size"] += 1
    elif fault == "partition-axis":
        parameters["x_shape"][parameters["x_batch_ndims"] - 1] += 1
    elif fault == "partition-stride":
        parameters["s_strides"][parameters["w_batch_ndims"] - 1] += 1
    elif fault == "partitions":
        execution["workgroupCount"][2] += 1
    elif fault == "k-view":
        parameters["w_shape"][-2] += 1
    elif fault == "bias-strides":
        parameters["b_strides"] = list(parameters["s_strides"])
    else:
        counts["scales"] -= 1
    with pytest.raises(ValueError):
        layout.plan(entry, counts, parameters, counts["y"], execution)
