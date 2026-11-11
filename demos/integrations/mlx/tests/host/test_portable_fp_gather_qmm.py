"""Block-scaled GatherQMM host routing, launch contracts and package configs."""

import ctypes
import math
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import fp_gather_qmm_layout as layout
from demos.integrations.mlx.portable_host import (
    fp_gather_qmm_packages,
    fp_matmul_packages,
    fp_qmm_layout,
    gather_qmm_layout,
    lazy_package_evidence,
    matmul_dispatch,
    qmm_packages,
    runtime,
)
from demos.integrations.mlx.portable_host import verify_fp_gather_qmm as verify
from demos.integrations.mlx.tests.host.test_portable_gather_qmm import (
    broadcast_strides,
    collapse,
    flat,
)
from demos.integrations.mlx.tests.host.test_portable_quantized_matmul import (
    CTYPES,
    contiguous,
)

STORAGE = {**CTYPES, "uint8": ctypes.c_uint8}

HOST = Path(layout.__file__).parent
RECORDS = [verify.case_record(case) for case in verify.CASES]
TYPES = ("float", "float16_t", "bfloat16_t")
VECTOR_KERNELS = ("qmv_fast", "qmv", "qvm")


def instantiated():
    """fp_quantized.metal's gathered entries, outside gather_qmm_rhs.

    instantiate_quantized_all_single and instantiate_quantized_all_aligned
    name each kernel with and without ``_hgs``, for every mode and type.
    """
    entries = []
    for mode, (group, bits) in layout.MODES.items():
        for source in TYPES:
            stem = f"{source}_gs_{group}_b_{bits}"
            for kernel in (*VECTOR_KERNELS, "qmm_n"):
                entries.append(f"{mode}_gather_{kernel}_{stem}")
            for aligned in ("true", "false"):
                entries.append(f"{mode}_gather_qmm_t_{stem}_alN_{aligned}")
    return entries


def native_binding(record, **faults):
    """What gather_qmm.inc binds for one verification case."""
    k, n, m = record["K"], record["N"], record["M"]
    rows, columns = (n, k) if record["transpose"] else (k, n)
    words, groups = columns * record["bits"] // 32, columns // record["groupSize"]
    x_shape = [*record["xBatchShape"], m, k]
    w_batch = record["wBatchShape"]
    w_shape = [*w_batch, rows, words]
    index_shape = record["indexShape"]
    lhs, rhs = record["lhsIndices"], record["rhsIndices"]
    shape, (lhs_strides, rhs_strides) = collapse(
        index_shape,
        [
            broadcast_strides(verify.shape_of(lhs), index_shape),
            broadcast_strides(verify.shape_of(rhs), index_shape),
        ],
    )
    values = {
        "lhs_indices": faults.get("lhs", flat(lhs)),
        "rhs_indices": faults.get("rhs", flat(rhs)),
    }
    counts = {
        "x": math.prod(x_shape),
        "w": math.prod(w_shape),
        "scales": math.prod(w_batch) * rows * groups,
        "lhs_indices": len(values["lhs_indices"]),
        "rhs_indices": len(values["rhs_indices"]),
        "y": record["batches"] * m * n,
    }
    if record["globalScaleExponents"] is not None:
        counts["global_scale"] = faults.get("experts", math.prod(w_batch))
    if layout.is_matrix(record["entry"]):
        parameters = {"K": k, "N": n, "M": m}
        grid = [-(-n // 32), -(-m // 32), record["batches"]]
    else:
        parameters = {"in_vec_size": k, "out_vec_size": n}
        grid = [m, -(-n // layout.tile_columns(record["entry"])), record["batches"]]
    parameters.update(
        x_batch_ndims=len(x_shape) - 2,
        x_shape=x_shape,
        x_strides=contiguous(x_shape),
        w_batch_ndims=len(w_shape) - 2,
        w_shape=w_shape,
        w_strides=contiguous(w_shape),
        s_strides=contiguous([*w_batch, rows, groups]),
        batch_ndims=len(shape),
        batch_shape=shape,
        lhs_strides=lhs_strides,
        rhs_strides=rhs_strides,
    )
    parameters.update(faults.get("parameters", {}))
    execution = {
        "workgroupCount": faults.get("grid", grid),
        "workgroupSize": list(layout.workgroup_size(record["entry"])),
    }
    return counts, values, parameters, execution


def native_buffers(record, **faults):
    """ctypes buffers in gather_qmm.inc's binding order, with backing storage."""
    counts, values, parameters, execution = native_binding(record, **faults)
    entry, memory, buffers = record["entry"], {}, []
    for name in layout.parameter_names(entry):
        dtype = layout.parameter_type(entry, name)
        if name in values:
            data = values[name]
        elif name in counts:
            data = [0] * counts[name]
        else:
            value = parameters[name]
            data = [value] if isinstance(value, int) else value
        memory[name] = (STORAGE[dtype] * len(data))(*data)
        buffers.append(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(memory[name]),
                len(data),
                int(name == "y"),
            )
        )
    return buffers, memory, execution


def planned(record, **faults):
    buffers, memory, execution = native_buffers(record, **faults)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    counts, parameters = layout.read_storage(record["entry"], supplied)
    threads = record["batches"] * record["M"] * record["N"]
    return layout.plan(record["entry"], counts, parameters, threads, execution), memory


def _record(name):
    return next(record for record in RECORDS if record["name"] == name)


def test_signature_accepts_every_instantiated_entry():
    entries = instantiated()
    # 3 modes and 3 types: 4 kernels, and qmm_t over N's alignment.
    assert len(entries) == 54
    for entry in entries:
        info = layout.signature(entry)
        assert entry.startswith(layout.ENTRY_PREFIXES)
        assert not info["globalScale"]
        expected = (32, 2, 2) if info["kernel"] in ("qmm_t", "qmm_n") else (32, 2, 1)
        assert layout.workgroup_size(entry) == expected
        # ops.cpp passes a global scale with nvfp4 weights alone.
        if info["mode"] == "nvfp4":
            assert layout.signature(entry + "_hgs")["globalScale"]
        else:
            with pytest.raises(ValueError, match="never routed"):
                layout.signature(entry + "_hgs")
    assert sum(entry.startswith("nvfp4_") for entry in entries) == 18


@pytest.mark.parametrize(
    "entry",
    (
        "mxfp4_gather_qmv_float_gs_32_b_4_hgs",
        "mxfp8_gather_qmm_t_float_gs_32_b_8_alN_true_hgs",
        "mxfp4_gather_qmm_t_float_gs_32_b_4",
        "mxfp4_gather_qmm_n_float_gs_32_b_4_alN_true",
        "nvfp4_gather_qmv_float_gs_32_b_4",
        "mxfp8_gather_qmv_float_gs_32_b_4",
        "mxfp4_gather_qmm_rhs_nt_float_gs_32_b_4_bm_16_bn_32_bk_32_wm_1_wn_2",
        "nvfp4_gather_qmv_float_gs_16_b_4_hgs_true",
        "mxfp4_qmv_float_gs_32_b_4_batch_0",
        "affine_gather_qmv_float_gs_32_b_4",
        None,
    ),
)
def test_signature_rejects_other_entries(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_prefixes_do_not_claim_other_families():
    for entry in (
        "affine_gather_qmv_float_gs_32_b_4",
        "mxfp4_qmv_float_gs_32_b_4_batch_0",
        "nvfp4_qmm_t_float_gs_16_b_4_alN_true_batch_0",
    ):
        assert not entry.startswith(layout.ENTRY_PREFIXES)
    assert not any(
        prefix.startswith(gather_qmm_layout.ENTRY_PREFIXES)
        for prefix in layout.ENTRY_PREFIXES
    )


def test_selection_follows_gather_qmm_for_upstream_shapes():
    # test_gather_qmm's block-scaled shapes, at each mode's group size and width.
    expected = {
        (1, 32, 128, True): "qmv",
        (32, 32, 256, True): "qmm_t_{}_alN_true",
        (1, 32, 512, True): "qmv_fast",
        (32, 256, 32, False): "qmm_n",
        (1, 256, 32, False): "qvm",
    }
    for mode, (group, bits) in layout.MODES.items():
        stem = f"float_gs_{group}_b_{bits}"
        for (m, n, k, transpose), kernel in expected.items():
            name = kernel.format(stem) if "{}" in kernel else f"{kernel}_{stem}"
            assert layout.select(mode, "float32", m, n, k, transpose) == (
                f"{mode}_gather_{name}"
            )
    # mxfp8 packs 4 codes a word, so K of 256 is qmv_fast's alignment.
    assert "qmv_fast" in layout.select("mxfp8", "float32", 1, 32, 256)
    assert "qmv_fast" not in layout.select("mxfp4", "float32", 1, 32, 256)
    # test_gather_qmm_matrix_path and test_gather_qmm_global_scale.
    assert layout.select("mxfp4", "bfloat16", 64, 512, 512) == (
        "mxfp4_gather_qmm_t_bfloat16_t_gs_32_b_4_alN_true"
    )
    assert layout.select("nvfp4", "float16", 32, 128, 256, False, True) == (
        "nvfp4_gather_qmm_n_float16_t_gs_16_b_4_hgs"
    )
    assert layout.select("nvfp4", "bfloat16", 1, 128, 2048, True, True) == (
        "nvfp4_gather_qmv_fast_bfloat16_t_gs_16_b_4_hgs"
    )
    assert layout.select("nvfp4", "float32", 1, 128, 256, False, True) == (
        "nvfp4_gather_qvm_float_gs_16_b_4_hgs"
    )
    with pytest.raises(ValueError, match="nvfp4"):
        layout.select("mxfp4", "float32", 1, 32, 128, True, True)


def test_verification_cases_cover_every_route():
    entries = [record["entry"] for record in RECORDS]
    assert len(set(entries)) == len(entries)
    infos = [layout.signature(entry) for entry in entries]
    assert {info["kernel"] for info in infos} == verify.KERNELS
    assert {info["mode"] for info in infos} == set(layout.MODES)
    assert {info["globalScale"] for info in infos} == {False, True}
    assert {info["aligned"] for info in infos} == {None, False, True}
    # Every kernel runs with and without a global scale.
    for kernel in verify.KERNELS - {"qmv_fast", "qmm_t"}:
        assert {info["globalScale"] for info in infos if info["kernel"] == kernel} == {
            False,
            True,
        }
    for record in RECORDS:
        assert record["dispatches"] == [record["entry"]]
    assert fp_gather_qmm_packages.verification_entries() == sorted(entries)


def test_case_records_require_one_global_scale_per_expert():
    case = list(verify.CASES[1])
    case[-1] = (0, 1)
    with pytest.raises(ValueError, match="one value per expert"):
        verify.case_record(tuple(case))


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_case_data_is_exact_in_every_element_type(record):
    import numpy as np

    data = verify.case_data(record, 3)
    assert data["expected"].shape == (*record["indexShape"], record["M"], record["N"])
    assert data["pairs"] == verify.matrix_pairs(np, record)
    # Every dequantized weight and input holds in the element type.
    for name in ("weights", "x"):
        values = np.asarray(data[name], dtype=np.float64)
        assert np.array_equal(verify.rounded(values, record["dtype"]), values), name
    assert data["scales"].dtype.kind in "iu" and data["scales"].max() < 256
    if record["globalScaleExponents"] is None:
        assert data["globalScale"] is None
    else:
        assert data["globalScale"].shape == tuple(record["wBatchShape"])
        ratios = data["globalScale"] / verify.ENCODED_RANGE
        assert np.array_equal(np.exp2(np.log2(ratios).round()), ratios)


def test_reference_codes_decode_as_the_pinned_formats():
    assert verify.fp8_value(0x38) == 1.0
    assert verify.fp8_value(0x2F) == 0.25 * (1 + 7 / 8)
    assert verify.fp8_value(0xCF) == -7.5
    assert min(abs(verify.fp8_value(code)) for code in verify.FP8_CODES) == 0.25
    assert verify.scale_code("nvfp4", -2) == 40 and verify.scale_code("nvfp4", 1) == 64
    assert verify.scale_code("mxfp4", 0) == 127
    assert verify.FP4_VALUES[7] == 6 and verify.FP4_VALUES[15] == -6


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_plan_replays_the_gathered_offsets(record):
    import numpy as np

    metadata, memory = planned(record)
    assert memory
    info = layout.signature(record["entry"])
    assert (
        metadata["M"],
        metadata["N"],
        metadata["K"],
        metadata["batches"],
        metadata["kernel"],
        metadata["mode"],
        metadata["globalScale"],
    ) == (
        record["M"],
        record["N"],
        record["K"],
        record["batches"],
        info["kernel"],
        record["mode"],
        record["globalScaleExponents"] is not None,
    )
    assert metadata["matrixIndices"] == verify.matrix_pairs(np, record)
    names = set(layout.parameter_names(record["entry"]))
    assert "b_strides" not in names and "biases" not in names
    bound = set(layout.matrix_parameters(record["entry"]))
    if info["globalScale"]:
        # The bound global scale reaches the kernel as an array.
        assert set(metadata["parameters"]) == names - bound
    else:
        assert metadata["parameters"]["global_scale"] == [layout.INACTIVE_GLOBAL_SCALE]
        assert set(metadata["parameters"]) == names - bound | {"global_scale"}


def test_global_scale_binds_at_the_biases_slot():
    entry = _record("nvfp4-qmm-t-global-unaligned")["entry"]
    names = layout.parameter_names(entry)
    assert names[:4] == ("w", "scales", "global_scale", "x")
    assert layout.parameter_type(entry, "global_scale") == "float32"
    assert layout.parameter_type(entry, "scales") == "uint8"
    plain = _record("mxfp4-qmm-n-experts")["entry"]
    assert "global_scale" not in layout.parameter_names(plain)


def test_cases_take_every_offset_path():
    parameters = [planned(record)[0]["parameters"] for record in RECORDS]
    assert {value["batch_ndims"] for value in parameters} == {1, 2}
    assert {value["x_batch_ndims"] for value in parameters} == {1, 2}
    assert {value["w_batch_ndims"] for value in parameters} == {0, 1, 2}
    assert any(0 in value["rhs_strides"] for value in parameters)
    assert any(0 in value["lhs_strides"] for value in parameters)


@pytest.mark.parametrize(
    ("name", "faults", "message"),
    (
        ("mxfp4-qmv-fast-experts", {"rhs": [2, 0, 3, 1]}, "index names no matrix"),
        ("nvfp4-qmv-global-broadcast", {"rhs": [3, 4, 2]}, "index names no matrix"),
        (
            "nvfp4-qmv-global-broadcast",
            {"experts": 3},
            "global scale does not hold one value per expert",
        ),
        ("nvfp4-qmm-n-global-shared", {"experts": 2}, "one value per expert"),
        (
            "mxfp4-qmv-fast-experts",
            {"parameters": {"lhs_strides": [2]}},
            "index view exceeds its buffer",
        ),
        (
            "nvfp4-qmm-t-global-unaligned",
            {"parameters": {"rhs_strides": [-1, 1]}},
            "index batch does not match",
        ),
        ("mxfp8-qmm-t-limit", {"grid": [1, 2, 2]}, "launch does not match"),
        ("mxfp4-qvm-experts", {"grid": [3, 3, 3]}, "launch does not match"),
        (
            "nvfp4-qvm-global-broadcast",
            {"grid": [1, 2, 2]},
            "launch does not match",
        ),
        (
            "mxfp4-qmm-n-experts",
            {"parameters": {"w_strides": [520, 13, 1]}},
            "row contiguous",
        ),
        (
            "mxfp8-qmv-shared-weights",
            {"parameters": {"x_strides": [2000, 512, 256, 1]}},
            "x matrix exceeds its buffer",
        ),
        (
            "mxfp8-qmm-t-limit",
            {"parameters": {"M": 17}, "grid": [1, 1, 2]},
            "differs from MLX's routing",
        ),
    ),
)
def test_plan_rejects_launches_that_differ_from_mlx(name, faults, message):
    with pytest.raises(ValueError, match=message):
        planned(_record(name), **faults)


def test_storage_rejects_native_identity_faults():
    record = _record("nvfp4-qmm-n-global-shared")
    buffers, memory, _ = native_buffers(record)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    layout.read_storage(record["entry"], supplied)
    for name, field, value in (
        ("global_scale", "dtype", b"float16"),
        ("scales", "dtype", b"float32"),
        ("global_scale", "output", 1),
    ):
        faulty = dict(supplied)
        buffer = supplied[name]
        values = {
            key: getattr(buffer, key)
            for key in ("name", "dtype", "data", "count", "output")
        }
        values[field] = value
        faulty[name] = runtime.Buffer(*values.values())
        with pytest.raises(ValueError):
            layout.read_storage(record["entry"], faulty)
    for missing in ("global_scale", "rhs_indices"):
        with pytest.raises(ValueError):
            layout.read_storage(
                record["entry"],
                {key: item for key, item in supplied.items() if key != missing},
            )
    # An unscaled entry binds no global scale.
    plain = _record("mxfp4-qmm-n-experts")
    with pytest.raises(ValueError):
        layout.read_storage(plain["entry"], supplied)
    assert memory


def test_dispatch_routes_block_scaled_entries_through_their_cache():
    record = _record("nvfp4-qmm-t-global-unaligned")
    buffers, memory, execution = native_buffers(record)
    calls = []

    class Cache:
        def get(self, entry):
            calls.append(entry)
            raise RuntimeError("stop")

    host = object.__new__(runtime.HostRuntime)
    host.fp_gather_qmms = Cache()
    launch = type("Launch", (), {"execution": lambda self: execution})()
    array = (runtime.Buffer * len(buffers))(*buffers)
    threads = record["batches"] * record["M"] * record["N"]
    with pytest.raises(RuntimeError, match="stop"):
        host.dispatch(record["entry"], array, len(buffers), threads, launch=launch)
    assert calls == [record["entry"]]
    with pytest.raises(ValueError, match="dispatch requires"):
        matmul_dispatch.dispatch_with(
            host, layout, None, record["entry"], array, len(buffers), threads, launch
        )
    assert memory


def test_runtime_reports_block_scaled_entries_available():
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = host.quantized_matmuls = host.quantized_qmms = None
    host.fp_quantized_matmuls = host.fp_quantized_qmms = None
    host.gather_qmms = host.fp_gather_qmms = None
    entry = b"nvfp4_gather_qmm_t_float_gs_16_b_4_alN_true_hgs"
    assert host._entry_available(entry) == 0
    host.fp_gather_qmms = object()
    assert host._entry_available(entry) == 1
    assert host._entry_available(b"mxfp8_gather_qvm_bfloat16_t_gs_32_b_8") == 1
    assert host._entry_available(b"mxfp4_gather_qvm_float_gs_32_b_4_hgs") == 0
    assert (
        host._entry_available(
            b"mxfp4_gather_qmm_rhs_nt_float_gs_32_b_4_bm_16_bn_32_bk_32_wm_1_wn_2"
        )
        == 0
    )


def test_lazy_evidence_audits_block_scaled_dispatches(tmp_path):
    assert (
        lazy_package_evidence.family("nvfp4_gather_qmv_fast_float_gs_16_b_4_hgs")
        == "fp-gather-quantized"
    )
    assert lazy_package_evidence.family("affine_gather_qmv_float_gs_64_b_4") == (
        "gather-quantized"
    )
    lazy = lazy_package_evidence.LazyPackages(tmp_path, tmp_path / "p", "opengl")
    cache = lazy.caches["fp-gather-quantized"]
    assert isinstance(cache, fp_gather_qmm_packages.FpGatherQmmPackageCache)
    assert cache.directory == (tmp_path / "p" / fp_gather_qmm_packages.CACHE_DIRECTORY)


@pytest.mark.parametrize("kernel", ("qmm_t", "qmm_n"))
def test_matrix_package_configs_take_the_block_scaled_qmm_tiles(tmp_path, kernel):
    entries = [entry for entry in instantiated() if f"_gather_{kernel}_" in entry]
    entries += [entry + "_hgs" for entry in entries if entry.startswith("nvfp4_")]
    for entry in entries[:: max(1, len(entries) // 7)]:
        sibling = layout.sibling(entry)
        assert fp_qmm_layout.signature(sibling)["batched"]
        for target in ("metal", "directx", "opengl"):
            config = fp_gather_qmm_packages.project_config(
                tmp_path, "work/out", target, entry
            )
            base = qmm_packages.project_config(
                tmp_path, "work/out", target, sibling, fp_qmm_layout
            )
            assert config.source_options == base.source_options
            assert config.entry_points == {layout.SOURCE: (entry,)}
            assert config.entry_workgroup_size_rules == {
                layout.SOURCE: {
                    entry: base.entry_workgroup_size_rules[layout.SOURCE][sibling]
                }
            }
            assert bool(config.index_range_assertions) == (target == "opengl")
            assert bool(config.workgroup_access_assertions) == (target != "metal")
            if target == "opengl":
                expressions = {
                    (item.expression, item.function)
                    for item in config.index_range_assertions
                }
                assert ("reference.offset + index", "crosstl_resource_load_uint") in (
                    expressions
                )
                scaled = layout.signature(entry)["globalScale"]
                assert (
                    "reference.offset + index",
                    "crosstl_resource_load_float",
                ) in expressions or not scaled


@pytest.mark.parametrize("kernel", VECTOR_KERNELS)
def test_vector_package_configs_take_the_block_scaled_qmv_options(tmp_path, kernel):
    entries = [entry for entry in instantiated() if f"_gather_{kernel}_" in entry]
    entries += [entry + "_hgs" for entry in entries if entry.startswith("nvfp4_")]
    for entry in entries[:: max(1, len(entries) // 5)]:
        for target in ("metal", "directx", "opengl"):
            config = fp_gather_qmm_packages.project_config(
                tmp_path, "work/out", target, entry
            )
            base = fp_matmul_packages.project_config(
                tmp_path,
                "work/out",
                target,
                "mxfp4_qmv_float_gs_32_b_4_batch_1",
            )
            assert config.source_options == base.source_options
            assert config.entry_workgroup_size_rules == {
                layout.SOURCE: {
                    entry: base.entry_workgroup_size_rules[layout.SOURCE][
                        "mxfp4_qmv_float_gs_32_b_4_batch_1"
                    ]
                }
            }
            assert not config.workgroup_access_assertions
            assert [item.expression for item in config.index_range_assertions] == (
                list(layout.GATHER_INDEX_EXPRESSIONS) if target == "opengl" else []
            )
            assert {
                item.value_range.maximum for item in config.index_range_assertions
            } <= {layout.MAX_ELEMENTS}
    cache = fp_gather_qmm_packages.FpGatherQmmPackageCache(
        tmp_path, tmp_path / "c", "metal"
    )
    assert cache.SOURCE == layout.SOURCE
    with pytest.raises(ValueError, match="product target"):
        fp_gather_qmm_packages.FpGatherQmmPackageCache(tmp_path, tmp_path, "vulkan")


def test_package_builds_cover_the_verification_entries(tmp_path, monkeypatch):
    import concurrent.futures

    from demos.integrations.mlx.portable_host import matmul_packages

    entries = fp_gather_qmm_packages.verification_entries()
    calls = []

    def build(root, directory, target, entry):
        calls.append((Path(directory), target, entry))
        return {"entry": entry}

    monkeypatch.setattr(
        matmul_packages, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(fp_gather_qmm_packages, "_build_one", build)
    records = fp_gather_qmm_packages.build_entries(
        tmp_path, tmp_path / "packages", "opengl", entries, 3
    )
    assert [record["entry"] for record in records] == entries
    assert {call[:2] for call in calls} == {
        (
            (tmp_path / "packages" / fp_gather_qmm_packages.CACHE_DIRECTORY).resolve(),
            "opengl",
        )
    }
    with pytest.raises(ValueError):
        fp_gather_qmm_packages.build_entries(
            tmp_path, tmp_path, "opengl", ["mxfp4_gather_qmv_float_gs_32_b_4_hgs"], 1
        )


def test_backend_routes_block_scaled_products_as_quantized_cpp():
    source = (HOST / "gather_qmm.inc").read_text(encoding="utf-8")
    for text in (
        "const bool global = !affine && inputs.size() == 6;",
        "(global && mode_ != QuantizationMode::Nvfp4) ||",
        "group_size_ != (mode_ == QuantizationMode::Nvfp4 ? 16 : 32) ||",
        "bits_ != (mode_ == QuantizationMode::Mxfp8 ? 8 : 4)) {",
        "inputs[2].dtype() != (affine ? dtype : uint8)",
        "(global && inputs[3].dtype() != float32)",
        "const array& lhs_input = inputs[inputs.size() - 2];",
        "const array& rhs_input = inputs[inputs.size() - 1];",
        "inputs[3].flags().row_contiguous ? inputs[3] : contiguous_copy(inputs[3]);",
        "if (global_scale && global_scale->size() != E) {",
        'const std::string mode = affine ? "affine" : fp_mode;',
        'const std::string suffix = global_scale ? "_hgs" : "";',
        '{"scales", affine ? storage : "uint8", data(scales), matmul_span(scales), 0},',
        '{"global_scale", "float32", data(*global_scale), matmul_span(*global_scale), 0});',
        "biases ? QuantizedBatchViews(x, w, scales, *biases) : QuantizedBatchViews(x, w, scales);",
    ):
        assert text in source, text
    order = [
        source.index(f'{{"{name}", ')
        for name in ("w", "scales", "biases", "global_scale", "x", "lhs_indices", "y")
    ]
    assert order == sorted(order)
    assert "translates affine weights only" not in source


def test_trace_validation_requires_the_routed_entries():
    import numpy as np

    records = [
        {**record, "pairs": verify.matrix_pairs(np, record)} for record in RECORDS
    ]
    refusals = [
        {"name": "sorted-rhs", "dispatches": [verify.DEFAULT_INDEX_ENTRY]},
        {"name": "rhs-out-of-range", "dispatches": []},
    ]
    trace = [
        {
            "entry": record["entry"],
            "target": "directx",
            "matmulMetadata": {
                "M": record["M"],
                "N": record["N"],
                "K": record["K"],
                "batches": record["batches"],
                "matrixIndices": record["pairs"],
                "globalScale": record["globalScaleExponents"] is not None,
            },
            "parameters": planned(record)[0]["parameters"],
        }
        for record in records
    ]
    trace.append({"entry": verify.DEFAULT_INDEX_ENTRY, "target": "directx"})
    verify.validate_trace(trace, records, refusals, "directx")
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace, records, refusals, "metal")
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace[:-1], records, refusals, "directx")
    trace[1]["matmulMetadata"]["globalScale"] = False
    with pytest.raises(RuntimeError, match="launch differs"):
        verify.validate_trace(trace, records, refusals, "directx")
    trace[1]["matmulMetadata"]["globalScale"] = True
    kept = [item for item in records if item["globalScaleExponents"] is None]
    with pytest.raises(RuntimeError, match="cover"):
        verify.validate_trace(
            [event for event in trace if not event["entry"].endswith("_hgs")],
            kept,
            refusals,
            "directx",
        )


def test_ci_translates_and_executes_the_block_scaled_products():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["quantized-host"]
    names = [item.get("name") for item in job["steps"]]
    steps = dict(zip(names, job["steps"]))
    translate = steps["Translate block-scaled gathered quantized packages"]
    name = "Execute block-scaled gathered quantized products through MLX"
    step = steps[name]
    for item in (translate, step):
        assert "if" not in item and "continue-on-error" not in item
        assert "set -euo pipefail" in item["run"]
        assert "--packages .mlx-portable-quantized/base/packages" in item["run"]
    assert "portable_host.fp_gather_qmm_packages" in translate["run"]
    assert "--verification" in translate["run"] and "--jobs 3" in translate["run"]
    assert "portable_host.verify_fp_gather_qmm" in step["run"]
    assert "tee .mlx-portable-quantized/fp-gather-qmm.log" in step["run"]
    assert step["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    assert (
        "test_portable_fp_gather_qmm.py" in steps["Validate quantized host contracts"][
            "run"
        ]
    )
    assert (
        names.index("Execute affine gathered quantized products through MLX")
        < names.index("Translate block-scaled gathered quantized packages")
        < names.index(name)
        < names.index("Retain quantized execution evidence")
    )
