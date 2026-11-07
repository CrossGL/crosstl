"""GatherQMM: gather_qmv, gather_qvm and gather_qmm routing and launch contracts."""

import ctypes
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import gather_qmm_layout as matrix
from demos.integrations.mlx.portable_host import gather_qmm_packages
from demos.integrations.mlx.portable_host import gather_qmv_layout as vector
from demos.integrations.mlx.portable_host import (
    gather_qmv_packages,
    lazy_package_evidence,
)
from demos.integrations.mlx.portable_host import matmul_dispatch as dispatch
from demos.integrations.mlx.portable_host import qmm_layout, runtime
from demos.integrations.mlx.tests.host.test_portable_fp_quantized_matmul import (
    CTYPES,
    physical_layout,
)

HOST = Path(vector.__file__).parent


def gathered(
    layout=vector,
    mode="affine",
    *,
    transpose=True,
    x_batches=(5,),
    w_batches=(3,),
    lhs=(0, 2),
    rhs=(2, 1),
    m=1,
    n=32,
    k=256,
    group=None,
    bits=None,
    global_scales=None,
    route=b"gather_qmm",
    dtype="float32",
):
    """gather_qmm.inc's binding of one GatherQMM product with 1-D indices."""
    if group is None:
        group, bits = (64, 4) if mode == "affine" else vector.MODES[mode]
    affine = mode == "affine"
    rows, columns = (n, k) if transpose else (k, n)
    words, groups = columns * bits // 32, columns // group

    def strides(batches, *inner):
        size = math.prod(inner)
        outer = [size * math.prod(batches[i + 1 :]) for i in range(len(batches))]
        return [*outer, inner[1], 1]

    batches = len(lhs)
    data = {
        "w": ("uint32", [0] * (math.prod(w_batches) * rows * words), 0),
        "scales": (
            dtype if affine else "uint8",
            [0] * (math.prod(w_batches) * rows * groups),
            0,
        ),
        "x": (dtype, [0] * (math.prod(x_batches) * m * k), 0),
        "lhs_indices": ("uint32", list(lhs), 0),
        "rhs_indices": ("uint32", list(rhs), 0),
        "y": (dtype, [0] * (batches * m * n), 1),
    }
    if affine:
        data["biases"] = (dtype, [0] * len(data["scales"][1]), 0)
    if layout is vector:
        data["in_vec_size"] = ("int32", [k], 0)
        data["out_vec_size"] = ("int32", [n], 0)
    else:
        data.update(K=("int32", [k], 0), N=("int32", [n], 0), M=("int32", [m], 0))
    data.update(
        {
            "x_batch_ndims": ("int32", [len(x_batches)], 0),
            "x_shape": ("int32", [*x_batches, m, k], 0),
            "x_strides": ("int64", strides(x_batches, m, k), 0),
            "w_batch_ndims": ("int32", [len(w_batches)], 0),
            "w_shape": ("int32", [*w_batches, rows, words], 0),
            "w_strides": ("int64", strides(w_batches, rows, words), 0),
            "s_strides": ("int64", strides(w_batches, rows, groups), 0),
            "batch_ndims": ("int32", [1], 0),
            "batch_shape": ("int32", [batches], 0),
            "lhs_strides": ("int64", [1], 0),
            "rhs_strides": ("int64", [1], 0),
        }
    )
    if affine:
        data["b_strides"] = ("int64", strides(w_batches, rows, groups), 0)
    if global_scales is not None:
        data["global_scale"] = ("float32", [0x3F800000] * global_scales, 0)
    scaled = global_scales is not None
    if layout is vector:
        entry = vector.select(mode, dtype, n, k, scaled, transpose, group, bits)
        count = [m, -(-n // vector.tile_rows(entry)), batches]
        size = [32, 2, 1]
    else:
        entry = matrix.select(mode, dtype, m, n, k, scaled, transpose, group, bits)
        count = [-(-n // 32), -(-m // 32), batches]
        size = [32, 2, 2]
    memory, buffers = {}, {}
    for name, (dtype, values, output) in data.items():
        memory[name] = (CTYPES[dtype] * len(values))(*values)
        buffers[name] = runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            len(values),
            output,
        )
    if route is not None:
        memory["route"] = ctypes.create_string_buffer(route)
        buffers["route"] = runtime.Buffer(
            b"route", b"utf8", ctypes.addressof(memory["route"]), len(route), 0
        )
    execution = {"workgroupCount": count, "workgroupSize": size}
    return entry, buffers, memory, execution


def planned(layout, entry, buffers, execution):
    buffers = dict(buffers)
    route = dispatch._route(layout, buffers.pop("route", None), layout.LABEL)
    counts, parameters = layout.read_storage(entry, buffers)
    return layout.plan(entry, counts, parameters, counts["y"], execution, route)


@pytest.mark.parametrize(
    "entry,source,kernel",
    (
        ("affine_gather_qmv_fast_float_gs_64_b_4", vector.AFFINE_SOURCE, "qmv_fast"),
        ("affine_gather_qvm_bfloat16_t_gs_128_b_3", vector.AFFINE_SOURCE, "qvm"),
        ("mxfp4_gather_qvm_float16_t_gs_32_b_4", vector.SOURCE, "qvm"),
        ("nvfp4_gather_qvm_float_gs_16_b_4_hgs", vector.SOURCE, "qvm"),
    ),
)
def test_vector_signature_reads_affine_and_qvm_entries(entry, source, kernel):
    assert vector.signature(entry)["kernel"] == kernel
    assert vector.source(entry) == source
    assert entry.startswith(vector.ENTRY_PREFIXES)
    assert lazy_package_evidence.family(entry) == "gather-quantized-mv"
    affine = entry.startswith("affine")
    assert ("biases" in vector.matrix_parameters(entry)) == affine
    assert ("b_strides" in vector.parameter_names(entry)) == affine
    assert vector.parameter_type(entry, "scales") == (
        vector.signature(entry)["dtype"] if affine else "uint8"
    )


def test_vector_selection_follows_quantized_cpp_gather_qmv_and_qvm():
    # Affine qmv_fast needs K in whole fast blocks: 3 bits pack 8 values.
    assert vector.select("affine", "float32", 32, 512, False, True, 64, 3) == (
        "affine_gather_qmv_fast_float_gs_64_b_3"
    )
    assert vector.select("affine", "float32", 32, 256, False, True, 64, 3) == (
        "affine_gather_qmv_float_gs_64_b_3"
    )
    assert vector.select("mxfp8", "float32", 256, 32, False, False) == (
        "mxfp8_gather_qvm_float_gs_32_b_8"
    )
    # gather_qvm covers two simdgroups of min(group_size, 32) columns.
    assert vector.tile_rows("affine_gather_qvm_float_gs_128_b_4") == 64
    assert vector.tile_rows("nvfp4_gather_qvm_float_gs_16_b_4") == 32
    with pytest.raises(ValueError, match="group and bits"):
        vector.select("affine", "float32", 32, 256)
    for entry in gather_qmv_packages.verification_entries():
        vector.signature(entry)
    assert gather_qmv_packages.verification_entries() == sorted(
        gather_qmv_packages.verification_entries()
    )


@pytest.mark.parametrize(
    "kwargs,kernel",
    (
        # 4-bit qmv_fast needs K in whole blocks of 512 values.
        ({}, "qmv"),
        ({"k": 512}, "qmv_fast"),
        ({"transpose": False, "n": 256, "k": 32}, "qvm"),
        ({"mode": "mxfp4", "transpose": False, "n": 256, "k": 32}, "qvm"),
        ({"mode": "nvfp4", "global_scales": 3, "k": 128}, "qmv"),
    ),
)
def test_vector_plan_bounds_the_gathered_weights_biases_and_scales(kwargs, kernel):
    entry, buffers, _memory, execution = gathered(**kwargs)
    metadata = planned(vector, entry, buffers, execution)
    assert (metadata["kernel"], metadata["route"]) == (kernel, "gather_qmm")
    assert metadata["lhsIndices"] == [0, 2] and metadata["rhsIndices"] == [2, 1]
    values = metadata["parameters"]
    affine = kwargs.get("mode", "affine") == "affine"
    assert ("b_strides" in values) == affine
    scaled = "global_scales" in kwargs
    assert ("global_scale" in values) == (not affine and not scaled)


@pytest.mark.parametrize(
    "fault",
    (
        "biases",
        "bias-contiguity",
        "qqmm-affine",
        "qqmm-qvm",
        "vector-limit",
        "qvm-groups",
    ),
)
def test_vector_plan_refuses_launches_gather_qmm_never_makes(fault):
    kwargs = {}
    if fault in ("qqmm-qvm", "qvm-groups"):
        kwargs = {"mode": "mxfp8", "transpose": False, "n": 256, "k": 32}
    entry, buffers, memory, execution = gathered(**kwargs)
    if fault == "biases":
        # The third matrix's biases reach past the buffer.
        buffers["biases"] = runtime.Buffer(
            b"biases", b"float32", buffers["biases"].data, 3 * 32 * 4 - 1, 0
        )
    elif fault == "bias-contiguity":
        memory["b_strides"][1] = 5
    elif fault in ("qqmm-affine", "qqmm-qvm"):
        memory["route"] = ctypes.create_string_buffer(b"gather_qqmm")
        buffers["route"] = runtime.Buffer(
            b"route", b"utf8", ctypes.addressof(memory["route"]), 11, 0
        )
    elif fault == "vector-limit":
        entry, buffers, memory, execution = gathered(m=18, x_batches=(5,))
    elif fault == "qvm-groups":
        memory["out_vec_size"][0] = 255
        execution["workgroupCount"][1] = -(-255 // 64)
    with pytest.raises(ValueError):
        planned(vector, entry, buffers, execution)


@pytest.mark.parametrize(
    "entry,info",
    (
        (
            "affine_gather_qmm_t_float_gs_64_b_4_alN_true",
            ("affine", "qmm_t", "float32", 64, 4, True, False),
        ),
        (
            "affine_gather_qmm_n_bfloat16_t_gs_32_b_6",
            ("affine", "qmm_n", "bfloat16", 32, 6, None, False),
        ),
        (
            "nvfp4_gather_qmm_t_float16_t_gs_16_b_4_alN_false_hgs",
            ("nvfp4", "qmm_t", "float16", 16, 4, False, True),
        ),
        (
            "mxfp8_gather_qmm_n_float_gs_32_b_8",
            ("mxfp8", "qmm_n", "float32", 32, 8, None, False),
        ),
    ),
)
def test_matrix_signature_reads_the_instantiated_gather_qmm_kernels(entry, info):
    keys = ("mode", "kernel", "dtype", "groupSize", "bits", "aligned", "globalScale")
    assert matrix.signature(entry) == dict(zip(keys, info))
    assert matrix.workgroup_size(entry) == (32, 2, 2)
    assert entry.startswith(matrix.ENTRY_PREFIXES)
    assert not entry.startswith(vector.ENTRY_PREFIXES)
    assert lazy_package_evidence.family(entry) == "gather-quantized-mm"
    source = vector.AFFINE_SOURCE if info[0] == "affine" else vector.SOURCE
    assert matrix.source(entry) == source


@pytest.mark.parametrize(
    "entry",
    (
        "affine_gather_qmm_n_float_gs_64_b_4_alN_true",
        "affine_gather_qmm_t_float_gs_64_b_4",
        "affine_gather_qmm_t_float_gs_64_b_4_alN_true_hgs",
        "mxfp8_gather_qmm_t_float_gs_32_b_8_alN_true_hgs",
        "mxfp4_gather_qmm_n_float_gs_16_b_4",
        "affine_gather_qmm_t_splitk_float_gs_64_b_4_alN_true",
    ),
)
def test_matrix_signature_refuses_entries_gather_qmm_never_routes(entry):
    with pytest.raises(ValueError):
        matrix.signature(entry)


def test_matrix_selection_starts_at_the_vector_limit():
    assert matrix.select("affine", "float32", 18, 32, 256, False, True, 64, 4) == (
        "affine_gather_qmm_t_float_gs_64_b_4_alN_true"
    )
    assert matrix.select("nvfp4", "bfloat16", 18, 40, 256, True) == (
        "nvfp4_gather_qmm_t_bfloat16_t_gs_16_b_4_alN_false_hgs"
    )
    assert matrix.select("mxfp4", "float32", 4, 256, 32, False, False) == (
        "mxfp4_gather_qmm_n_float_gs_32_b_4"
    )
    for args in (
        ("affine", "float32", 17, 32, 256, False, True, 64, 4),
        ("mxfp4", "float32", 3, 256, 32, False, False),
    ):
        with pytest.raises(ValueError, match="vector products"):
            matrix.select(*args)
    for entry in gather_qmm_packages.verification_entries():
        matrix.signature(entry)


@pytest.mark.parametrize(
    "kwargs,kernel",
    (
        ({"m": 32}, "qmm_t"),
        ({"m": 32, "transpose": False, "n": 256, "k": 32}, "qmm_n"),
        ({"m": 32, "mode": "mxfp4", "transpose": False, "n": 512, "k": 32}, "qmm_n"),
        ({"m": 33, "mode": "nvfp4", "global_scales": 3, "n": 40}, "qmm_t"),
    ),
)
def test_matrix_plan_reads_each_batch_index(kwargs, kernel):
    entry, buffers, _memory, execution = gathered(matrix, **kwargs)
    metadata = planned(matrix, entry, buffers, execution)
    assert (metadata["kernel"], metadata["route"]) == (kernel, "gather_qmm")
    assert metadata["lhsIndices"] == [0, 2] and metadata["rhsIndices"] == [2, 1]
    assert metadata["M"] == kwargs["m"] and metadata["batches"] == 2
    values = metadata["parameters"]
    assert not set(values) & set(vector.INDEX_PARAMETERS)
    affine = kwargs.get("mode", "affine") == "affine"
    scaled = "global_scales" in kwargs
    assert ("global_scale" in values) == (not affine and not scaled)


@pytest.mark.parametrize(
    "kwargs,accepted",
    (
        ({"dtype": "bfloat16", "global_scales": 3}, True),
        # GatherQQMM's matrix kernels read the weights' global scales.
        ({"dtype": "bfloat16"}, False),
        ({"dtype": "float32", "global_scales": 3}, False),
        ({"dtype": "bfloat16", "global_scales": 3, "transpose": False, "k": 32}, False),
        ({"dtype": "bfloat16", "global_scales": 3, "k": 272}, False),
    ),
)
def test_matrix_plan_holds_gather_qqmm_launches_to_its_routing(kwargs, accepted):
    entry, buffers, _memory, execution = gathered(
        matrix, "nvfp4", m=32, route=b"gather_qqmm", **kwargs
    )
    if not accepted:
        with pytest.raises(ValueError, match="GatherQQMM"):
            planned(matrix, entry, buffers, execution)
        return
    metadata = planned(matrix, entry, buffers, execution)
    assert (metadata["route"], metadata["kernel"]) == ("gather_qqmm", "qmm_t")


@pytest.mark.parametrize(
    "fault",
    (
        "route",
        "qqmm-route",
        "vector-rows",
        "grid",
        "workgroup",
        "lhs-range",
        "rhs-range",
        "input-batch",
        "global-scale",
        "entry",
        "k-groups",
    ),
)
def test_matrix_plan_refuses_launches_that_differ_from_mlx(fault):
    kwargs = {"m": 32}
    if fault == "global-scale":
        kwargs.update(mode="nvfp4", global_scales=2)
    entry, buffers, memory, execution = gathered(matrix, **kwargs)
    if fault == "route":
        buffers.pop("route")
    elif fault == "qqmm-route":
        memory["route"] = ctypes.create_string_buffer(b"gather_qqmm")
        buffers["route"] = runtime.Buffer(
            b"route", b"utf8", ctypes.addressof(memory["route"]), 11, 0
        )
    elif fault == "vector-rows":
        memory["M"][0] = 17
        memory["x_shape"][1] = 17
        execution["workgroupCount"][1] = 1
    elif fault == "grid":
        execution["workgroupCount"][0] += 1
    elif fault == "workgroup":
        execution["workgroupSize"] = [32, 2, 1]
    elif fault == "lhs-range":
        memory["lhs_indices"][1] = 5
    elif fault == "rhs-range":
        memory["rhs_indices"][0] = 3
    elif fault == "input-batch":
        buffers["x"] = runtime.Buffer(
            b"x", b"float32", buffers["x"].data, 3 * 32 * 256 - 1, 0
        )
    elif fault == "entry":
        entry = "affine_gather_qmm_t_float_gs_64_b_4_alN_false"
    elif fault == "k-groups":
        memory["K"][0] = 250
    with pytest.raises(ValueError):
        planned(matrix, entry, buffers, execution)


def test_dispatch_commits_gathered_matrix_products_after_validation(
    tmp_path, monkeypatch
):
    entry, buffers, memory, execution = gathered(matrix, m=32)
    descriptor = {"artifact": {}, "bindings": []}
    for parameter in matrix.parameter_names(entry):
        kind = matrix.parameter_type(entry, parameter)
        descriptor["bindings"].append(
            {
                "name": f"binding_{parameter}",
                "kind": "buffer",
                "access": "read_write" if parameter == "y" else "read",
                "provenance": {"sourceResource": {"parameter": parameter}},
                "scalarLayout": physical_layout("opengl", kind),
            }
        )
    cache = SimpleNamespace(get=lambda name: (descriptor, tmp_path))
    host = SimpleNamespace(
        target="opengl", trace=tmp_path / "trace.jsonl", dispatch_count=0
    )
    requests = []

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        requests.append(inputs)
        return outputs

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        count = len(memory["y"])
        outputs["binding_y"]["values"][:count] = [0x40000000] * count
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)
    monkeypatch.setattr(dispatch, "execute", execute)
    supplied = (runtime.Buffer * len(buffers))(*buffers.values())
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    dispatch.dispatch_with(
        host, matrix, cache, entry, supplied, len(buffers), len(memory["y"]), launch
    )
    assert list(memory["y"]) == [0x40000000] * len(memory["y"])
    ((inputs,),) = [requests]
    assert inputs["binding_lhs_indices"]["values"][:2] == [0, 2]
    assert inputs["binding_M"]["values"] == [32]
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["matmulMetadata"]["route"] == "gather_qmm"
    assert set(event["inputHashes"]) == {
        "w",
        "scales",
        "biases",
        "x",
        "lhs_indices",
        "rhs_indices",
    }


def test_package_configs_follow_each_entrys_source_and_offsets(tmp_path):
    entry = "affine_gather_qvm_float_gs_64_b_4"
    gl = gather_qmv_packages.project_config(tmp_path, "out", "opengl", entry)
    assert list(gl.include_patterns) == [vector.AFFINE_SOURCE]
    assert [item.expression for item in gl.index_range_assertions] == list(
        vector.AFFINE_INDEX_EXPRESSIONS
    )
    cache = gather_qmv_packages.GatherQuantizedMatmulPackageCache(
        tmp_path, tmp_path / "cache", "opengl"
    )
    assert cache.source_path(entry) == vector.AFFINE_SOURCE
    assert cache.source_path("mxfp4_gather_qvm_float_gs_32_b_4") == vector.SOURCE
    entry = "nvfp4_gather_qmm_t_float_gs_16_b_4_alN_true_hgs"
    gl = gather_qmm_packages.project_config(tmp_path, "out", "opengl", entry)
    # The qmm tiles' offsets, plus the uint32 index reads the gathered
    # kernels add; adjust_matrix_offsets moves resource references in 64 bits.
    scopes = {
        (item.expression, item.function, item.indexed)
        for item in gl.index_range_assertions
    }
    assert ("reference.offset + index", "crosstl_resource_load_uint", None) in scopes
    assert ("reference.offset + index", "crosstl_resource_load_uint8", None) in scopes
    assert ("reference.offset + index", None, "crosstl_workgroup_Ws") in scopes
    assert not {expression for expression, _, _ in scopes} & set(
        vector.INDEX_EXPRESSIONS
    )
    assert [item.parameter for item in gl.metadata_preconditions] == ["K"]
    loaders = {item.function for item in gl.workgroup_access_assertions}
    assert loaders == {
        qmm_layout.access_assertions(
            entry, {"kernel": "qmm_t", "dtype": "float32"}, vector.SOURCE
        )[0]["function"]
    }
    for target in ("directx", "metal"):
        config = gather_qmm_packages.project_config(tmp_path, "out", target, entry)
        assert not config.index_range_assertions
        rules = config.entry_workgroup_size_rules[vector.SOURCE]
        assert [int(size) for size in rules[entry]] == [32, 2, 2]
    metal = gather_qmm_packages.project_config(tmp_path, "out", "metal", entry)
    assert not metal.workgroup_access_assertions


def test_runtime_routes_gathered_matrix_products_through_their_cache(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = host.quantized_matmuls = host.fp_quantized_matmuls = None
    host.quantized_qmms = host.fp_quantized_qmms = None
    host.gather_quantized_matmuls = host.gather_quantized_qmms = None
    entry = "affine_gather_qmm_n_float_gs_64_b_4"
    assert host._entry_available(entry.encode()) == 0
    host.gather_quantized_qmms = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"affine_gather_qmm_n_float_gs_64_b_4_hgs") == 0
    calls = []
    monkeypatch.setattr(
        runtime.matmul_dispatch,
        "dispatch_with",
        lambda host, layout, cache, entry, *args: calls.append((layout, cache, entry)),
    )
    host.dispatch(entry, [], 22, 64, launch=None)
    assert calls == [(matrix, host.gather_quantized_qmms, entry)]


def test_backend_ports_gather_qmm_as_quantized_cpp_does():
    port = (HOST / "gather_qmm.inc").read_text(encoding="utf-8")
    start = port.index("void GatherQMM::eval_gpu(")
    body = port[start : port.index("\n}\n", start)]
    order = [
        "biases = matmul_operand(inputs[3]);",
        "global_scale = dense_input(inputs[3]);",
        "gather_indices(inputs[inputs.size() - 2]);",
        # quantized.cpp's routes, in its order.
        "if (M == 1 && B >= 16 && right_sorted_ && B / E >= 4) {",
        "dispatch_gather_qmm_rhs(",
        "const int vector_limit = transpose_ ? qq_vector_limit(K, N) : 4;",
        "if (M >= vector_limit) {",
        "dispatch_gather_qmm(",
        "dispatch_gather_qmv(",
    ]
    positions = [body.index(text) for text in order]
    assert positions == sorted(positions)
    assert body.count('"gather_qmm");') == 3
    assert "does not yet route" not in port
    for text in (
        "const CrosstlMlxLaunch launch{{tiles_n, tiles_m, uint32_t(B)}, {32, 2, 2}};",
        "const int rows = transpose ? 8 : 2 * std::min(group_size, 32);",
        '(transpose ? (N % 32 == 0 ? "_alN_true" : "_alN_false") : "")',
        '{"scales", biases ? storage : "uint8", data(scales), matmul_span(scales), 0},',
        'buffers.push_back(vector("b_strides", "int64", b_strides));',
        # gather_qmm_rhs: one input row per index, the offsets of each
        # matrix's rows, then (32, 2, 1) threadgroups over 16-row tiles.
        "x_.size() / x_.shape(-2) / x_.shape(-1) != indices.size()",
        "array offsets = dispatch_gather_mm_offsets(indices, E, M);",
        '"_bm_16_bn_32_bk_32_wm_1_wn_2" + (gs ? "_hgs" : "") + "_align_N_"',
        "std::min<int64_t>(M, (int64_t(M) + bm - 1) / bm + E - 1);",
        "{{uint32_t(tiles_n), uint32_t(tiles_m), 1}, {32, 2, 1}};",
    ):
        assert text in port, text
    adapter = (HOST / "prepare.py").read_text(encoding="utf-8")
    assert '        "GatherQQMM",\n        "GatherQMM",\n' in adapter
