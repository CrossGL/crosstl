"""GatherQQMM: gather_qmv's routing, gathered launch contracts and guarded commits."""

import ctypes
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import gather_qmv_layout as layout
from demos.integrations.mlx.portable_host import (
    gather_qmv_packages,
    lazy_package_evidence,
)
from demos.integrations.mlx.portable_host import matmul_dispatch as dispatch
from demos.integrations.mlx.portable_host import runtime
from demos.integrations.mlx.tests.host.test_portable_fp_quantized_matmul import (
    CTYPES,
    physical_layout,
)

HOST = Path(layout.__file__).parent


@pytest.mark.parametrize(
    "entry,info",
    (
        (
            "mxfp8_gather_qmv_fast_float_gs_32_b_8",
            ("mxfp8", "qmv_fast", "float32", 32, 8, False),
        ),
        (
            "nvfp4_gather_qmv_bfloat16_t_gs_16_b_4_hgs",
            ("nvfp4", "qmv", "bfloat16", 16, 4, True),
        ),
        (
            "mxfp4_gather_qmv_float16_t_gs_32_b_4",
            ("mxfp4", "qmv", "float16", 32, 4, False),
        ),
    ),
)
def test_signature_reads_the_instantiated_gather_kernels(entry, info):
    keys = ("mode", "kernel", "dtype", "groupSize", "bits", "globalScale")
    assert layout.signature(entry) == dict(zip(keys, info))
    assert layout.workgroup_size(entry) == (32, 2, 1)
    assert entry.startswith(layout.ENTRY_PREFIXES)
    assert lazy_package_evidence.family(entry) == "gather-quantized-mv"


@pytest.mark.parametrize(
    "entry",
    (
        # Only nvfp4 weights carry global scales.
        "mxfp8_gather_qmv_float_gs_32_b_8_hgs",
        "nvfp4_gather_qmv_float_gs_32_b_4",
        "mxfp8_gather_qvm_float_gs_32_b_8",
        "affine_gather_qmv_float_gs_64_b_4",
        "nvfp4_gather_qmv_float_gs_16_b_4_batch_0",
    ),
)
def test_signature_refuses_entries_gather_qqmm_never_routes(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_selection_follows_quantized_cpp_gather_qmv():
    # qmv_fast needs N in whole tiles and K in whole fast blocks: 256 values
    # of 8 bits, 512 of 4.
    assert layout.select("mxfp8", "float32", 32, 256) == (
        "mxfp8_gather_qmv_fast_float_gs_32_b_8"
    )
    assert layout.select("mxfp8", "float32", 36, 256) == (
        "mxfp8_gather_qmv_float_gs_32_b_8"
    )
    assert layout.select("nvfp4", "float32", 256, 256, True) == (
        "nvfp4_gather_qmv_float_gs_16_b_4_hgs"
    )
    assert layout.select("nvfp4", "bfloat16", 256, 512, True) == (
        "nvfp4_gather_qmv_fast_bfloat16_t_gs_16_b_4_hgs"
    )
    with pytest.raises(ValueError, match="global scales"):
        layout.select("mxfp8", "float32", 32, 256, True)
    assert gather_qmv_packages.verification_entries() == sorted(
        gather_qmv_packages.verification_entries()
    )
    for entry in gather_qmv_packages.verification_entries():
        layout.signature(entry)


def _float_word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def gathered_buffers(
    mode="mxfp8",
    *,
    x_batches=(5,),
    w_batches=(3,),
    lhs=(0, 2),
    rhs=(2, 1),
    m=1,
    n=32,
    k=256,
    global_scales=None,
    route=b"gather_qqmm",
):
    """gather_qmm.inc's binding of one gathered product with 1-D indices."""
    group, bits = layout.MODES[mode]
    words, groups = k * bits // 32, k // group
    x_count = m * k
    for extent in x_batches:
        x_count *= extent
    w_count, s_count = n * words, n * groups
    for extent in w_batches:
        w_count, s_count = w_count * extent, s_count * extent
    x_shape = [*x_batches, m, k]
    w_shape = [*w_batches, n, words]
    x_strides = [m * k * _prod(x_batches[i + 1 :]) for i in range(len(x_batches))]
    w_strides = [n * words * _prod(w_batches[i + 1 :]) for i in range(len(w_batches))]
    s_strides = [n * groups * _prod(w_batches[i + 1 :]) for i in range(len(w_batches))]
    batches = len(lhs)
    data = {
        "w": ("uint32", [0] * w_count, 0),
        "scales": ("uint8", [0] * s_count, 0),
        "x": ("float32", [0] * x_count, 0),
        "lhs_indices": ("uint32", list(lhs), 0),
        "rhs_indices": ("uint32", list(rhs), 0),
        "y": ("float32", [0] * (batches * m * n), 1),
        "in_vec_size": ("int32", [k], 0),
        "out_vec_size": ("int32", [n], 0),
        "x_batch_ndims": ("int32", [len(x_batches)], 0),
        "x_shape": ("int32", x_shape, 0),
        "x_strides": ("int64", [*x_strides, k, 1], 0),
        "w_batch_ndims": ("int32", [len(w_batches)], 0),
        "w_shape": ("int32", w_shape, 0),
        "w_strides": ("int64", [*w_strides, words, 1], 0),
        "s_strides": ("int64", [*s_strides, groups, 1], 0),
        "batch_ndims": ("int32", [1], 0),
        "batch_shape": ("int32", [batches], 0),
        "lhs_strides": ("int64", [1], 0),
        "rhs_strides": ("int64", [1], 0),
    }
    if global_scales is not None:
        data["global_scale"] = (
            "float32",
            [_float_word(value) for value in global_scales],
            0,
        )
    entry = layout.select(mode, "float32", n, k, global_scales is not None)
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
    execution = {
        "workgroupCount": [m, -(-n // layout.TILE_ROWS), batches],
        "workgroupSize": [32, 2, 1],
    }
    return entry, buffers, memory, execution


def _prod(values):
    result = 1
    for value in values:
        result *= value
    return result


def planned(entry, buffers, execution):
    buffers = dict(buffers)
    route = dispatch._route(layout, buffers.pop("route", None), layout.LABEL)
    counts, parameters = layout.read_storage(entry, buffers)
    return layout.plan(entry, counts, parameters, counts["y"], execution, route)


def test_plan_reads_each_batch_index_and_adds_the_inert_global_scale():
    entry, buffers, _memory, execution = gathered_buffers()
    metadata = planned(entry, buffers, execution)
    assert metadata["route"] == "gather_qqmm" and metadata["kernel"] == "qmv_fast"
    assert metadata["lhsIndices"] == [0, 2] and metadata["rhsIndices"] == [2, 1]
    assert metadata["batches"] == 2 and (metadata["M"], metadata["N"]) == (1, 32)
    values = metadata["parameters"]
    assert values["global_scale"] == [layout.INACTIVE_GLOBAL_SCALE]
    assert not set(values) & set(layout.INDEX_PARAMETERS)
    assert set(values) == {*layout.SCALAR_PARAMETERS, *layout.VECTOR_TYPES} | {
        "global_scale"
    }


def test_plan_binds_one_global_scale_per_weight_matrix():
    entry, buffers, _memory, execution = gathered_buffers(
        "nvfp4", global_scales=(1.5, 2.0, 3.0)
    )
    assert entry == "nvfp4_gather_qmv_float_gs_16_b_4_hgs"
    assert "global_scale" in layout.matrix_parameters(entry)
    metadata = planned(entry, buffers, execution)
    assert "global_scale" not in metadata["parameters"]
    # Weight matrix 2 reads the third scale; two scales do not cover it.
    entry, buffers, _memory, execution = gathered_buffers(
        "nvfp4", global_scales=(1.5, 2.0)
    )
    with pytest.raises(ValueError, match="global scale"):
        planned(entry, buffers, execution)


def test_plan_accepts_scalar_indices_and_two_dimensional_inputs():
    entry, buffers, memory, execution = gathered_buffers(
        x_batches=(), w_batches=(3,), lhs=(0,), rhs=(1,), m=3, n=40, k=64
    )
    for name, values in (("batch_ndims", [0]), ("lhs_strides", [0])):
        memory[name][:] = values
    memory["rhs_strides"][:] = [0]
    metadata = planned(entry, buffers, execution)
    assert metadata["kernel"] == "qmv" and metadata["rhsIndices"] == [1]


@pytest.mark.parametrize(
    "fault",
    (
        "route",
        "unknown-route",
        "lhs-range",
        "rhs-range",
        "index-read",
        "grid",
        "workgroup",
        "rows",
        "contiguity",
        "scalar-views",
        "entry",
        "overlap",
        "views",
    ),
)
def test_plan_refuses_launches_that_differ_from_mlx(fault):
    entry, buffers, memory, execution = gathered_buffers()
    if fault == "route":
        buffers.pop("route")
    elif fault == "unknown-route":
        route = ctypes.create_string_buffer(b"qqmm")
        memory["route"] = route
        buffers["route"] = runtime.Buffer(
            b"route", b"utf8", ctypes.addressof(route), 4, 0
        )
    elif fault == "lhs-range":
        memory["lhs_indices"][1] = 5
    elif fault == "rhs-range":
        memory["rhs_indices"][0] = 3
    elif fault == "index-read":
        memory["lhs_strides"][0] = 2
    elif fault == "grid":
        execution["workgroupCount"][1] += 1
    elif fault == "workgroup":
        execution["workgroupSize"] = [32, 1, 1]
    elif fault == "rows":
        memory["x_shape"][1] = 2
    elif fault == "contiguity":
        memory["w_strides"][1] = 65
    elif fault == "scalar-views":
        memory["batch_ndims"][0] = 0
    elif fault == "entry":
        entry = "mxfp8_gather_qmv_float_gs_32_b_8"
    elif fault == "overlap":
        buffers["y"] = runtime.Buffer(b"y", b"float32", buffers["x"].data, 64, 1)
    elif fault == "views":
        buffers["batch_shape"] = runtime.Buffer(
            b"batch_shape", b"int32", buffers["x"].data, 17, 0
        )
    with pytest.raises(ValueError):
        planned(entry, buffers, execution)


def test_dispatch_commits_gathered_products_only_after_validation(
    tmp_path, monkeypatch
):
    entry, buffers, memory, execution = gathered_buffers(
        "nvfp4", global_scales=(1.0, 1.0, 1.0)
    )
    descriptor = {"artifact": {}, "bindings": []}
    for parameter in layout.parameter_names(entry):
        kind = layout.parameter_type(entry, parameter)
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
        outputs["binding_y"]["values"][:count] = [0x3F800000] * count
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)
    monkeypatch.setattr(dispatch, "execute", execute)
    supplied = (runtime.Buffer * len(buffers))(*buffers.values())
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    dispatch.dispatch_with(
        host, layout, cache, entry, supplied, len(buffers), len(memory["y"]), launch
    )
    assert list(memory["y"]) == [0x3F800000] * len(memory["y"])
    ((inputs,),) = [requests]
    # The index buffers and global scales are copied with guards like every
    # operand bound by address.
    assert inputs["binding_rhs_indices"]["values"][:2] == [2, 1]
    assert inputs["binding_global_scale"]["values"][:3] == [_float_word(1.0)] * 3
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["matmulMetadata"]["rhsIndices"] == [2, 1]
    assert set(event["inputHashes"]) == {
        "w",
        "scales",
        "x",
        "lhs_indices",
        "rhs_indices",
        "global_scale",
    }


def test_package_configs_assert_the_gathered_offsets_on_opengl(tmp_path):
    entry = "nvfp4_gather_qmv_float_gs_16_b_4_hgs"
    gl = gather_qmv_packages.project_config(tmp_path, "out", "opengl", entry)
    assert [item.expression for item in gl.index_range_assertions] == list(
        layout.INDEX_EXPRESSIONS
    )
    for target in ("directx", "metal"):
        config = gather_qmv_packages.project_config(tmp_path, "out", target, entry)
        assert not config.index_range_assertions
        rules = config.entry_workgroup_size_rules[layout.SOURCE]
        assert [int(size) for size in rules[entry]] == [32, 2, 1]


def test_runtime_routes_gathered_products_through_their_cache(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = host.quantized_matmuls = host.fp_quantized_matmuls = None
    host.quantized_qmms = host.fp_quantized_qmms = None
    host.gather_quantized_matmuls = None
    entry = "nvfp4_gather_qmv_float_gs_16_b_4_hgs"
    assert host._entry_available(entry.encode()) == 0
    host.gather_quantized_matmuls = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"mxfp8_gather_qmv_float_gs_32_b_8_hgs") == 0
    calls = []
    monkeypatch.setattr(
        runtime.matmul_dispatch,
        "dispatch_with",
        lambda host, layout, cache, entry, *args: calls.append((layout, cache, entry)),
    )
    host.dispatch(entry, [], 22, 64, launch=None)
    assert calls == [(layout, host.gather_quantized_matmuls, entry)]


def test_backend_ports_gather_qqmm_as_quantized_cpp_does():
    port = (HOST / "gather_qmm.inc").read_text(encoding="utf-8")
    start = port.index("void GatherQQMM::eval_gpu(")
    body = port[start : port.index("\n}\n", start)]
    order = [
        # quantize_input and quantize_dequantize_input, as QQMatmul's port.
        "qq_weights(stream(), inputs[1], scales, global_w, mode_, group_size_, bits_);",
        "qq_rounded_input(inputs[0], global_x, mode, group_size_, bits_);",
        # The weights' global scale is broadcast to one per matrix.
        "broadcast(*global_w, per_matrix);",
        # Globally scaled bfloat16 matrix products are refused.
        "does not yet route globally",
        '"gather_qqmm");',
    ]
    positions = [body.index(text) for text in order]
    assert positions == sorted(positions)
    for text in (
        'buffers.push_back(\n        {"global_scale", "float32", data(*global_scale), '
        "matmul_span(*global_scale), 0});",
        "collapse_contiguous_dims(\n      lhs_indices.shape(), "
        "{lhs_indices.strides(), rhs_indices.strides()});",
        "const CrosstlMlxLaunch launch{{uint32_t(M), tiles, uint32_t(B)}, {32, 2, 1}};",
    ):
        assert text in port, text
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert backend.rstrip().endswith(
        '#include "mlx/backend/no_gpu/crosstl_gather_qmm.inc"'
    )
    adapter = (HOST / "prepare.py").read_text(encoding="utf-8")
    assert '        "QQMatmul",\n        "GatherQQMM",\n' in adapter
