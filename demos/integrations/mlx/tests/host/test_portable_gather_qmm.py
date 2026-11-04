import ctypes
import math
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import gather_qmm_layout as layout
from demos.integrations.mlx.portable_host import (
    gather_qmm_packages,
    matmul_layout,
    qmm_layout,
    runtime,
    verify_upstream_gather_matmul_grad,
)
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES
from demos.integrations.mlx.tests.host.test_portable_fp_quantized_matmul import (
    CTYPES,
    contiguous,
)

HOST = Path(layout.__file__).parent
MODES = verify_upstream_gather_matmul_grad.MODES


def collapse(shape, strides):
    """common/utils.cpp collapse_contiguous_dims over several stride vectors."""
    groups = []
    if shape:
        groups = [[0]] if shape[0] != 1 else [[]]
        for axis in range(1, len(shape)):
            if any(st[axis] * shape[axis] != st[axis - 1] for st in strides):
                groups.append([])
            if shape[axis] != 1:
                groups[-1].append(axis)
    groups = [group for group in groups if group]
    out_shape = [math.prod(shape[axis] for axis in group) for group in groups]
    out_strides = [[st[group[-1]] for group in groups] for st in strides]
    if shape and not out_shape:
        return [1], [[0] for _ in strides]
    return out_shape, out_strides


def broadcast(values, shape):
    """Row-major values of an index array and its strides once broadcast."""
    lead = len(shape) - len(values["shape"])
    own = [1] * lead + list(values["shape"])
    strides = [0] * lead + contiguous(values["shape"])
    return [
        0 if extent == 1 and target != 1 else stride
        for extent, target, stride in zip(own, shape, strides)
    ]


def native_binding(
    mode,
    dtype,
    m,
    n,
    k,
    transpose,
    x_batch=(1,),
    w_batch=(3,),
    lhs=None,
    rhs=None,
    global_scale=False,
):
    """What gather_qmm.inc binds for one product: entry, counts, values, launch.

    ``lhs`` and ``rhs`` are ``{"shape", "values"}`` index arrays; a missing
    one is ops.cpp's indices_or_default over its operand's batch axes.
    """
    group, bits = MODES[mode]
    for name, batch in (("lhs", x_batch), ("rhs", w_batch)):
        if locals()[name] is None:
            size = math.prod(batch)
            default = {"shape": list(batch), "values": list(range(size))}
            if name == "lhs":
                lhs = default
            else:
                rhs = default
    batch_shape = list(
        max(a, b)
        for a, b in zip(
            [1] * (len(rhs["shape"]) - len(lhs["shape"])) + list(lhs["shape"]),
            [1] * (len(lhs["shape"]) - len(rhs["shape"])) + list(rhs["shape"]),
        )
    )
    batches = math.prod(batch_shape)
    rows, columns = (n, k) if transpose else (k, n)
    words, groups = columns * bits // 32, columns // group
    entry = layout.select(mode, dtype, group, bits, m, n, k, transpose, global_scale)
    experts = math.prod(w_batch)
    counts = {
        "x": math.prod(x_batch) * m * k,
        "w": experts * rows * words,
        "scales": experts * rows * groups,
        "y": batches * m * n,
        "lhs_indices": len(lhs["values"]),
        "rhs_indices": len(rhs["values"]),
    }
    if mode == "affine":
        counts["biases"] = counts["scales"]
    if global_scale:
        counts["global_scale"] = experts
    shape, (lhs_strides, rhs_strides) = collapse(
        batch_shape, [broadcast(lhs, batch_shape), broadcast(rhs, batch_shape)]
    )
    parameters = {
        "x_batch_ndims": len(x_batch),
        "x_shape": [*x_batch, m, k],
        "x_strides": contiguous([*x_batch, m, k]),
        "w_batch_ndims": len(w_batch),
        "w_shape": [*w_batch, rows, words],
        "w_strides": contiguous([*w_batch, rows, words]),
        "s_strides": contiguous([*w_batch, rows, groups]),
        "batch_ndims": len(shape),
        "batch_shape": shape,
        "lhs_strides": lhs_strides,
        "rhs_strides": rhs_strides,
        "lhs_indices": list(lhs["values"]),
        "rhs_indices": list(rhs["values"]),
    }
    if mode == "affine":
        parameters["b_strides"] = parameters["s_strides"]
    if layout.signature(entry)["kernel"] in layout.VECTOR_KERNELS:
        parameters.update(in_vec_size=k, out_vec_size=n)
        grid = [m, -(-n // layout.tile_columns(entry)), batches]
    else:
        parameters.update(K=k, N=n, M=m)
        grid = [-(-n // 32), -(-m // 32), batches]
    execution = {
        "workgroupCount": grid,
        "workgroupSize": list(layout.workgroup_size(entry)),
    }
    return entry, counts, parameters, execution


def native_buffers(*args, **kwargs):
    """ctypes buffers holding one binding, by name, with their storage."""
    entry, counts, parameters, execution = native_binding(*args, **kwargs)
    memory, buffers = {}, {}
    for name in layout.parameter_names(entry):
        dtype = layout.parameter_type(entry, name)
        if name in counts and name not in layout.INDEX_PARAMETERS:
            values = [0] * counts[name]
        else:
            value = parameters[name]
            values = [value] if isinstance(value, int) else value
        memory[name] = (CTYPES[dtype] * len(values))(*values)
        buffers[name] = runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            len(values),
            int(name == "y"),
        )
    return entry, buffers, memory, execution


# test_gather_qmm's index layouts with explicit left indices and two batch
# axes, and one broadcast weight.
EXPERTS = {
    "lhs": {"shape": [1], "values": [0]},
    "rhs": {"shape": [2], "values": [2, 1]},
}
TWO_AXES = {
    "x_batch": (4, 2),
    "w_batch": (4, 1),
    "lhs": {"shape": [3, 2], "values": [7, 6, 5, 4, 1, 2]},
    "rhs": {"shape": [3, 1], "values": [2, 0, 1]},
}
CASES = {
    "qmv-affine": (("affine", "float32", 1, 32, 128, True), EXPERTS),
    "qmv-fast-two-axes": (("affine", "float32", 1, 32, 512, True), TWO_AXES),
    "qvm-implied": (
        ("affine", "float32", 1, 256, 32, False),
        {"x_batch": (2,), "rhs": {"shape": [2], "values": [2, 1]}},
    ),
    "qmm-t-mxfp8": (("mxfp8", "float32", 32, 32, 256, True), EXPERTS),
    "qmm-n-two-axes": (("affine", "float32", 32, 256, 32, False), TWO_AXES),
    "qmv-nvfp4-hgs": (
        ("nvfp4", "bfloat16", 3, 40, 64, True),
        {**EXPERTS, "global_scale": True},
    ),
    "qmm-t-unaligned": (("mxfp4", "float16", 40, 40, 64, True), EXPERTS),
}


def binding(name):
    args, kwargs = CASES[name]
    return native_binding(*args, **kwargs)


@pytest.mark.parametrize("dtype", tuple(SOURCE_TYPES.values()))
def test_signature_accepts_exactly_the_instantiated_gather_entries(dtype):
    """quantized.metal's and fp_quantized.metal's gather instantiations."""
    accepted = []
    for mode in ("affine", "nvfp4", "mxfp4", "mxfp8"):
        configurations = (
            [(group, bits) for group in (32, 64, 128) for bits in (2, 3, 4, 5, 6, 8)]
            if mode == "affine"
            else [MODES[mode]]
        )
        for group, bits in configurations:
            stem = f"{dtype}_gs_{group}_b_{bits}"
            names = [
                f"{mode}_gather_{kernel}_{stem}"
                for kernel in ("qmv_fast", "qmv", "qvm", "qmm_n")
            ]
            names += [
                f"{mode}_gather_qmm_t_{stem}_alN_{aligned}"
                for aligned in ("true", "false")
            ]
            if mode != "affine":
                names += [name + "_hgs" for name in names]
            accepted += names
    for entry in accepted:
        info = layout.signature(entry)
        assert entry.startswith(layout.ENTRY_PREFIXES)
        assert layout.layout_for(entry).SOURCE == layout.source(entry)
        assert layout.source(entry).endswith(
            "/quantized.metal" if info["mode"] == "affine" else "/fp_quantized.metal"
        )
        expected = (32, 2, 1) if info["kernel"] in layout.VECTOR_KERNELS else (32, 2, 2)
        assert layout.workgroup_size(entry) == expected
    assert len(set(accepted)) == len(accepted) == 18 * 6 + 3 * 12


@pytest.mark.parametrize(
    "entry",
    (
        "affine_gather_qmm_rhs_nt_float_gs_64_b_4_bm_16_bn_32_bk_32_wm_1_wn_2",
        "affine_gather_qmv_float_gs_64_b_4_hgs",
        "affine_gather_qmv_float_gs_16_b_4",
        "affine_gather_qmm_t_float_gs_64_b_4",
        "affine_gather_qmm_n_float_gs_64_b_4_alN_true",
        "nvfp4_gather_qmv_float_gs_32_b_4",
        "mxfp8_gather_qvm_float_gs_32_b_4",
        "mxfp4_gather_qmm_t_float_gs_32_b_4_alN_true_batch_0",
        "affine_qmv_float_gs_64_b_4_batch_0",
    ),
)
def test_signature_refuses_entries_gather_qmm_never_selects(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_selection_follows_gather_qmm_eval_gpu():
    select = layout.select
    assert select("affine", "float32", 64, 4, 1, 32, 128, True) == (
        "affine_gather_qmv_float_gs_64_b_4"
    )
    # K = 512 is a multiple of qmv_fast_k_alignment(4) = 8 * 2 * 32.
    assert select("affine", "float32", 64, 4, 1, 32, 512, True) == (
        "affine_gather_qmv_fast_float_gs_64_b_4"
    )
    assert matmul_layout.fast_alignment(8) == 256
    assert select("mxfp8", "float32", 32, 8, 1, 32, 256, True) == (
        "mxfp8_gather_qmv_fast_float_gs_32_b_8"
    )
    # The transposed vector limit is 18 below 2048 columns and depth.
    assert select("affine", "float16", 64, 4, 17, 64, 512, True).startswith(
        "affine_gather_qmv_fast_"
    )
    assert select("affine", "float16", 64, 4, 18, 64, 512, True) == (
        "affine_gather_qmm_t_float16_t_gs_64_b_4_alN_true"
    )
    assert select("mxfp4", "bfloat16", 32, 4, 18, 40, 512, True) == (
        "mxfp4_gather_qmm_t_bfloat16_t_gs_32_b_4_alN_false"
    )
    assert select("nvfp4", "float32", 16, 4, 3, 32, 32, False) == (
        "nvfp4_gather_qvm_float_gs_16_b_4"
    )
    assert select("nvfp4", "float32", 16, 4, 4, 32, 32, False, True) == (
        "nvfp4_gather_qmm_n_float_gs_16_b_4_hgs"
    )


@pytest.mark.parametrize("name", tuple(CASES))
def test_plan_reads_the_indices_and_accepts_backend_launches(name):
    entry, counts, parameters, execution = binding(name)
    info = layout.signature(entry)
    metadata = layout.plan(entry, counts, dict(parameters), counts["y"], execution)
    args, kwargs = CASES[name]
    _, _, m, n, k, _ = args
    assert (metadata["M"], metadata["N"], metadata["K"]) == (m, n, k)
    assert metadata["batches"] == execution["workgroupCount"][2]
    values = metadata["parameters"]
    assert values["lhs_indices"] == parameters["lhs_indices"]
    if info["family"] == "fp" and not info["globalScale"]:
        assert values["global_scale"] == [layout.INACTIVE_GLOBAL_SCALE]
    else:
        assert "global_scale" not in values
    assert ("b_strides" in values) == (info["family"] == "affine")
    if name.endswith("two-axes"):
        # The lhs (3, 2) and broadcast rhs (3, 1) indices stay two axes.
        assert values["batch_shape"] == [3, 2]
        assert values["rhs_strides"] == [1, 0]
        assert metadata["gathered"] == [(7, 2), (6, 2), (5, 0), (4, 0), (1, 1), (2, 1)]


def test_layouts_bind_each_familys_matrices():
    affine = layout.layout_for("affine_gather_qmv_float_gs_64_b_4")
    plain = layout.layout_for("nvfp4_gather_qmv_float_gs_16_b_4")
    scaled = layout.layout_for("nvfp4_gather_qmv_float_gs_16_b_4_hgs")
    assert affine.MATRIX_PARAMETERS == ("w", "scales", "biases", "x", "y")
    assert plain.MATRIX_PARAMETERS == ("w", "scales", "x", "y")
    assert scaled.MATRIX_PARAMETERS == ("w", "scales", "global_scale", "x", "y")
    assert "b_strides" not in plain.BATCH_PARAMETERS
    with pytest.raises(ValueError, match="another layout"):
        plain.signature("nvfp4_gather_qmv_float_gs_16_b_4_hgs")
    names = layout.parameter_names("affine_gather_qmv_float_gs_64_b_4")
    from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
        GATHER_PARAMETERS,
    )

    assert names == tuple(name for name, _ in GATHER_PARAMETERS)


@pytest.mark.parametrize(
    "fault",
    (
        "workgroup",
        "grid",
        "depth",
        "threads",
        "groups",
        "route",
        "lhs-index",
        "rhs-index",
        "index-read",
        "index-views",
        "x-rows",
        "w-shape",
        "global-scale",
    ),
)
# Single batch axes: elem_to_loc wraps the indices of several axes, as the
# kernels do, so only one axis moves past its buffer.
@pytest.mark.parametrize("name", ("qmv-affine", "qmm-t-mxfp8", "qmv-nvfp4-hgs"))
def test_plan_rejects_launches_that_differ_from_mlx(fault, name):
    entry, counts, parameters, execution = binding(name)
    info = layout.signature(entry)
    threads = counts["y"]
    if fault == "workgroup":
        execution["workgroupSize"] = [32, 1, 1]
    elif fault == "grid":
        execution["workgroupCount"][1] += 1
    elif fault == "depth":
        per = threads // execution["workgroupCount"][2]
        execution["workgroupCount"][2] += 1
        threads = counts["y"] = per * execution["workgroupCount"][2]
    elif fault == "threads":
        threads += 1
    elif fault == "groups":
        parameters["in_vec_size" if "in_vec_size" in parameters else "K"] += 8
    elif fault == "route":
        parameters["x_shape"][parameters["x_batch_ndims"]] = 20
        if "M" in parameters:
            parameters["M"] = 3
    elif fault == "lhs-index":
        parameters["lhs_indices"][-1] = 9
    elif fault == "rhs-index":
        parameters["rhs_indices"][0] = 5
    elif fault == "index-read":
        parameters["lhs_strides"][0] = 4
    elif fault == "index-views":
        parameters["batch_shape"].append(1)
    elif fault == "x-rows":
        parameters["x_strides"][-1] = 2
    elif fault == "w-shape":
        parameters["w_shape"][-1] += 1
    elif info["globalScale"]:
        counts["global_scale"] = 2
    else:
        counts["scales"] -= 1
    with pytest.raises(ValueError):
        layout.plan(entry, counts, parameters, threads, execution)


@pytest.mark.parametrize("fault", ("dtype", "direction", "names", "overlap", "none"))
def test_storage_reads_index_arrays_by_value(fault):
    entry, buffers, memory, execution = native_buffers(
        *CASES["qmm-n-two-axes"][0], **TWO_AXES
    )
    if fault == "none":
        counts, parameters = layout.read_storage(entry, buffers)
        assert parameters["lhs_indices"] == TWO_AXES["lhs"]["values"]
        assert counts["lhs_indices"] == 6 and counts["rhs_indices"] == 3
        assert parameters["batch_shape"] == [3, 2]
        layout.plan(entry, counts, parameters, counts["y"], execution)
        return
    if fault == "dtype":
        buffers["rhs_indices"].dtype = b"int32"
    elif fault == "direction":
        buffers["lhs_indices"].output = 1
    elif fault == "names":
        buffers.pop("batch_ndims")
    else:
        buffers["lhs_indices"].data = buffers["y"].data
    with pytest.raises(ValueError):
        layout.read_storage(entry, buffers)


def test_assertions_match_the_translation_gates():
    from demos.integrations.mlx.tests.kernels import (
        test_current_quantized_gather_qmm as gate,
    )
    from demos.integrations.mlx.tests.kernels import test_current_quantized_qmv as qmv

    for entry in gate.ENTRIES:
        assert layout.index_assertions(entry) == tuple(
            gate._opengl_index_assertions(entry)
        )
        assert layout.access_assertions(entry) == tuple(
            gate._workgroup_access_assertions(entry)
        )
        assert layout.bounded_parameters(entry) == ("K",)
    for entry in (
        "affine_gather_qmv_float_gs_64_b_4",
        "affine_gather_qvm_bfloat16_t_gs_32_b_8",
    ):
        assertions = layout.index_assertions(entry)
        assert [item["expression"] for item in assertions] == list(
            qmv.OPENGL_INDEX_RANGE_ASSERTIONS["gather"]
        )
        assert {item["maximum"] for item in assertions} == {layout.MAX_ELEMENTS}
        assert layout.access_assertions(entry) == ()
    block = layout.index_assertions("nvfp4_gather_qmv_float_gs_16_b_4")
    assert [item["expression"] for item in block] == list(
        layout.GATHER_INDEX_EXPRESSIONS
    )
    assert {item["source"] for item in block} == {layout.SOURCES["fp"]}


def test_package_configs_take_each_gates_options(tmp_path):
    for entry in (
        "affine_gather_qmv_fast_float_gs_64_b_4",
        "mxfp4_gather_qvm_float16_t_gs_32_b_4",
        "affine_gather_qmm_t_float_gs_64_b_4_alN_true",
        "nvfp4_gather_qmm_n_bfloat16_t_gs_16_b_4_hgs",
    ):
        matrix = layout.signature(entry)["kernel"] not in layout.VECTOR_KERNELS
        path = layout.source(entry)
        for target in ("metal", "directx", "opengl"):
            config = gather_qmm_packages.project_config(tmp_path, "out", target, entry)
            assert list(config.include_patterns) == [path]
            rules = config.entry_workgroup_size_rules[path]
            assert tuple(int(size) for size in rules[entry]) == layout.workgroup_size(
                entry
            )
            options = config.source_options["metal"]
            lowering = options["target_options"]["opengl"].get(
                "cooperative_matrix_software_lowering", False
            )
            assert lowering == matrix
            assert ("cooperative_matrix_fragment_mapping" in options) == matrix
            assert options["target_options"]["directx"]["software_subgroup_width"] == 32
            assert bool(config.workgroup_access_assertions) == (
                matrix and target != "metal"
            )
            assert bool(config.index_range_assertions) == (target == "opengl")
            assert [item.parameter for item in config.metadata_preconditions] == (
                ["K"] if matrix else []
            )
    cache = gather_qmm_packages.GatherQuantizedPackageCache(
        tmp_path, tmp_path, "opengl"
    )
    assert cache.source_path("mxfp8_gather_qmv_float_gs_32_b_8") == layout.SOURCES["fp"]
    with pytest.raises(ValueError):
        cache.signature("affine_gather_qmm_rhs_nt_float_gs_64_b_4")


def test_runtime_routes_gathered_products_through_their_cache(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    for name in (
        "scans",
        "gathers",
        "quantization",
        "quantized_matmuls",
        "fp_quantized_matmuls",
        "quantized_qmms",
        "fp_quantized_qmms",
        "gathered_quantized",
    ):
        setattr(host, name, None)
    host.descriptors = {}
    entry = "affine_gather_qmm_t_float_gs_64_b_4_alN_true"
    assert host._entry_available(entry.encode()) == 0
    host.gathered_quantized = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"nvfp4_gather_qvm_float_gs_16_b_4_hgs") == 1
    assert host._entry_available(b"affine_gather_qmm_rhs_nt_float_gs_64_b_4") == 0
    calls = []
    monkeypatch.setattr(
        runtime.matmul_dispatch,
        "dispatch_with",
        lambda host, layout_, cache, entry, *args: calls.append(
            (layout_, cache, entry)
        ),
    )
    host.dispatch(entry, [], 21, 1, launch=None)
    assert calls == [(layout.layout_for(entry), host.gathered_quantized, entry)]


def test_backend_ports_gather_qmm_eval_gpu():
    port = (HOST / "gather_qmm.inc").read_text(encoding="utf-8")
    routing = port[port.index("void GatherQMM::eval_gpu") :]
    for fragment in (
        "M == 1 && B >= 16 && right_sorted_ && B / E >= 4",
        "const bool matrix = M >= vector_limit;",
        "N % 8 == 0 && K % gather_qmv_fast_alignment(bits_) == 0",
        '(matrix && transpose_ ? (N % 32 == 0 ? "_alN_true" : "_alN_false") : "")',
        '(global ? "_hgs" : "")',
        "views.bind(buffers);",
        "index_views.bind(buffers);",
        "{{uint32_t(tiles), uint32_t(rows), uint32_t(B)}, {32, 2, 2}}",
        "{{uint32_t(rows), uint32_t(tiles), uint32_t(B)}, {32, 2, 1}}",
    ):
        assert fragment in routing, fragment
    for bits in (2, 3, 4, 5, 6, 8):
        pack = 8 if bits in (3, 5) else 4 if bits == 6 else 32 // bits
        assert pack * (1 if bits == 2 else 2) * 32 == matmul_layout.fast_alignment(bits)
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert '#include "mlx/backend/no_gpu/crosstl_gather_qmm.inc"' in backend
    assert backend.index("crosstl_gather_mm.inc") < backend.index(
        "crosstl_gather_qmm.inc"
    )
    prepare = (HOST / "prepare.py").read_text(encoding="utf-8")
    assert '"GatherQMM",' in prepare
    assert (
        '"crosstl_gather_qmm.inc": (HERE / "gather_qmm.inc").read_bytes(),' in prepare
    )
    # The tile constants the host binds are the layout's.
    assert qmm_layout.BM == qmm_layout.BN == 32 and layout.QMV_ROWS == 8
