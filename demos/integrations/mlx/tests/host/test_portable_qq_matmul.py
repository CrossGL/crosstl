"""QQMatmul: block quantize-dequantize launches and dispatch_qmv's QQ routes."""

import ctypes
import struct
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import fp_matmul_layout as layout
from demos.integrations.mlx.portable_host import matmul_dispatch, prepare
from demos.integrations.mlx.portable_host import quantization_layout as quantization
from demos.integrations.mlx.portable_host import (
    runtime,
    upstream_module_groups,
    verify_upstream_ops,
)
from demos.integrations.mlx.tests.host.test_portable_fp_quantized_matmul import (
    CTYPES,
)

HOST = Path(layout.__file__).parent
FORMATS = (("mxfp4", 32, 4), ("mxfp8", 32, 8), ("nvfp4", 16, 4))


def qdq_entry(mode, dtype, global_scale):
    group, bits = layout.MODES[mode]
    source = quantization.SOURCE_TYPES[dtype]
    hgs = "true" if global_scale else "false"
    return f"{mode}_quantize_dequantize_{source}_gs_{group}_b_{bits}_hgs_{hgs}"


@pytest.mark.parametrize("mode,group,bits", FORMATS)
@pytest.mark.parametrize("dtype", quantization.SOURCE_TYPES)
@pytest.mark.parametrize("global_scale", (False, True))
def test_quantize_dequantize_layout_matches_every_instantiation(
    mode, group, bits, dtype, global_scale
):
    entry = qdq_entry(mode, dtype, global_scale)
    if global_scale and mode != "nvfp4":
        # instantiate_quantize_dequantize reads a global scale only for nvfp4.
        with pytest.raises(ValueError, match="block quantization specialization"):
            quantization.signature(entry)
        return
    assert quantization.signature(entry) == ("quantize_dequantize", dtype, group, bits)
    # One workgroup per group: simd_max spans the whole group.
    assert quantization.workgroup_size(entry) == group
    contract = quantization.buffer_contract(entry, 3)
    expected = {"w": (dtype, 3 * group, 0), "out": (dtype, 3 * group, 1)}
    if global_scale:
        expected["global_scale"] = ("float32", 1, 0)
    assert contract == expected


def qdq_buffers(entry, groups, *, drop=None, overlap=False):
    contract = quantization.buffer_contract(entry, groups)
    memory, buffers = {}, {}
    for name, (dtype, count, output) in contract.items():
        if name == drop:
            continue
        memory[name] = (ctypes.c_uint8 * (count * quantization.ITEM_SIZES[dtype]))()
        address = ctypes.addressof(memory[name])
        if overlap and name == "out":
            address = buffers["w"].data
        buffers[name] = runtime.Buffer(
            name.encode(), dtype.encode(), address, count, output
        )
    return buffers, memory


@pytest.mark.parametrize("fault", (None, "missing", "overlap", "geometry"))
def test_quantize_dequantize_validation_refuses_invalid_launches(fault):
    entry = qdq_entry("nvfp4", "float32", True)
    buffers, _memory = qdq_buffers(
        entry,
        4,
        drop="global_scale" if fault == "missing" else None,
        overlap=fault == "overlap",
    )
    execution = {"workgroupCount": [4, 1, 1], "workgroupSize": [16, 1, 1]}
    if fault == "geometry":
        execution["workgroupSize"] = [32, 1, 1]
    if fault is None:
        assert quantization.validate(entry, buffers, 64, execution) == {
            "groupCount": 4,
            "elementCount": 64,
        }
        return
    with pytest.raises(ValueError):
        quantization.validate(entry, buffers, 64, execution)


def test_global_scale_kernels_are_nvfp4_qmv_and_qmv_fast_only():
    for entry in (
        "nvfp4_qmv_float_gs_16_b_4_batch_0_hgs",
        "nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_1_hgs",
        "nvfp4_qvm_float16_t_gs_16_b_4_batch_0_hgs",
    ):
        assert layout.signature(entry)["globalScale"] is True
        assert "global_scale" in layout.parameter_names(entry)
    assert layout.signature("nvfp4_qmv_float_gs_16_b_4_batch_0")["globalScale"] is False
    for entry in (
        "mxfp8_qmv_float_gs_32_b_8_batch_0_hgs",
        "nvfp4_qmv_quad_float_gs_16_b_4_d_64_batch_0_hgs",
        "nvfp4_qmv_wide_float_gs_16_b_4_nv_2_kl_16_batch_0_hgs",
    ):
        with pytest.raises(ValueError):
            layout.signature(entry)


def test_qq_selection_follows_quantized_cpp_dispatch_qmv():
    # Without a global scale QQMatmul takes qmv_quad and qmv_wide like
    # QuantizedMatmul, but for any vector count.
    assert (
        layout.select("mxfp8", "float32", 64, 256, 128, 1, qq=True)
        == "mxfp8_qmv_quad_float_gs_32_b_8_d_128_batch_0"
    )
    assert (
        layout.select("mxfp8", "float32", 256, 512, 512, 1, qq=True)
        == "mxfp8_qmv_wide_float_gs_32_b_8_nv_5_kl_16_batch_0"
    )
    with pytest.raises(ValueError, match="matrix-matrix"):
        layout.select("mxfp8", "float32", 256, 512, 512, 1)
    # An nvfp4 global scale skips qmv_quad and qmv_wide.
    assert (
        layout.select("nvfp4", "float32", 67, 256, 64, 1, qq=True, global_scale=True)
        == "nvfp4_qmv_float_gs_16_b_4_batch_0_hgs"
    )
    assert (
        layout.select("nvfp4", "float32", 1, 512, 512, 1, qq=True, global_scale=True)
        == "nvfp4_qmv_fast_float_gs_16_b_4_batch_0_hgs"
    )
    for mode, qq in (("nvfp4", False), ("mxfp8", True)):
        with pytest.raises(ValueError, match="global scales"):
            layout.select(mode, "float32", 1, 512, 512, 1, qq=qq, global_scale=True)


def qq_buffers(m, n, k, *, global_scale=1.5, route=b"qqmm"):
    """backend.cpp's QQMatmul binding of one nvfp4 product with a global scale."""
    entry = layout.select("nvfp4", "float32", m, n, k, 1, qq=True, global_scale=True)
    group, bits = layout.MODES["nvfp4"]
    words, groups = k * bits // 32, k // group
    data = {
        "w": ("uint32", [0] * (n * words), 0),
        "scales": ("uint8", [0] * (n * groups), 0),
        "x": ("float32", [0] * (m * k), 0),
        "y": ("float32", [0] * (m * n), 1),
        "in_vec_size": ("int32", [k], 0),
        "out_vec_size": ("int32", [n], 0),
        "global_scale": (
            "float32",
            [struct.unpack("<I", struct.pack("<f", global_scale))[0]],
            0,
        ),
    }
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
        "workgroupCount": [m, -(-n // layout.tile_rows(entry)), 1],
        "workgroupSize": list(layout.workgroup_size(entry)),
    }
    return entry, buffers, memory, execution


def test_qq_plan_binds_the_weights_global_scale():
    entry, buffers, _memory, execution = qq_buffers(67, 256, 64)
    route = matmul_dispatch._route(layout, buffers.pop("route"), layout.LABEL)
    assert route == "qqmm"
    counts, parameters = layout.read_storage(entry, buffers)
    bits = struct.unpack("<I", struct.pack("<f", 1.5))[0]
    assert parameters["global_scale"] == [bits]
    metadata = layout.plan(entry, counts, parameters, counts["y"], execution, route)
    assert metadata["route"] == "qqmm" and metadata["kernel"] == "qmv"
    assert metadata["parameters"]["global_scale"] == [bits]
    # QuantizedMatmul never selects a globally scaled kernel.
    with pytest.raises(ValueError, match="global scales"):
        layout.plan(entry, counts, parameters, counts["y"], execution)


@pytest.mark.parametrize("fault", ("dtype", "count", "missing"))
def test_qq_storage_refuses_an_invalid_global_scale(fault):
    entry, buffers, memory, _execution = qq_buffers(1, 64, 512)
    buffers.pop("route")
    scale = buffers["global_scale"]
    if fault == "missing":
        del buffers["global_scale"]
    else:
        buffers["global_scale"] = runtime.Buffer(
            b"global_scale",
            b"uint32" if fault == "dtype" else b"float32",
            scale.data,
            2 if fault == "count" else 1,
            0,
        )
    with pytest.raises(ValueError):
        layout.read_storage(entry, buffers)


@pytest.mark.parametrize("fault", ("unknown", "dtype", "unrouted"))
def test_route_buffers_are_read_only_where_a_layout_takes_them(fault):
    from demos.integrations.mlx.portable_host import qmm_layout

    text = b"gemm" if fault == "unknown" else b"qqmm"
    memory = ctypes.create_string_buffer(text)
    buffer = runtime.Buffer(
        b"route",
        b"uint8" if fault == "dtype" else b"utf8",
        ctypes.addressof(memory),
        len(text),
        0,
    )
    target = qmm_layout if fault == "unrouted" else layout
    with pytest.raises(ValueError, match="route"):
        matmul_dispatch._route(target, buffer, target.LABEL)
    assert matmul_dispatch._route(target, None, target.LABEL) is None


def test_backend_ports_qqmatmul_as_quantized_cpp_does():
    port = (HOST / "qq_matmul.inc").read_text(encoding="utf-8")
    for text in (
        "void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {",
        # quantize_input: the host's Quantize port quantizes float weights.
        "fast::Quantize(stream, {}, group_size, bits, mode, false)",
        "qq_weights(stream(), inputs[1], scales, global_w, mode_, group_size_, bits_);",
        # quantize_dequantize_input rounds the input, donating it if it can.
        "bool donate = x_pre.is_donatable();",
        '"_quantize_dequantize_" +',
        '(global_scale ? "_hgs_true" : "_hgs_false")',
        # dispatch_qmv with the weights' global scale and QQMatmul's route.
        "x, w_q, scales_w, out, mode, true, group_size_, bits_, M, N, K, global_w, true);",
        # qmm with the weights' global scale for globally scaled bfloat16
        # products of one quantized matrix from the vector limit's rows on.
        "if (global && w_quantized && non_batched && dtype == bfloat16 && K % 32 == 0 &&\n"
        "      M >= qq_vector_limit(K, N)) {",
        "x, w_q, scales_w, std::nullopt, out, mode, true, group_size_, bits_, M, N, K,\n"
        "        global_w, true);",
    ):
        assert text in port, text
    # Every globally scaled product now runs; none is refused.
    assert "does not yet route" not in port
    start = port.index("void QQMatmul::eval_gpu(")
    body = port[start : port.index("\n}\n", start)]
    assert body.index("dispatch_qmm(") < body.index("dispatch_fp_quantized_matvec(")
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert (
        "} // namespace mlx::core::fast\n\n"
        "// The port of QQMatmul, which quantizes both operands of its product with the\n"
        "// quantization, matrix-vector and matrix-matrix dispatches above.\n"
        '#include "mlx/backend/no_gpu/crosstl_qq_matmul.inc"\n'
    ) in backend
    for text in (
        "const bool quad = transpose && (K == 64 || K == 128) && !global_scale;",
        "const bool wide = transpose && !quad && M >= 2 && !global_scale;",
        '(B > 1 ? "_batch_1" : "_batch_0") + (global_scale ? "_hgs" : "");',
        'buffers.push_back({"global_scale", "float32", data(*global_scale), 1, 0});',
        'buffers.push_back({"route", "utf8", const_cast<char*>(qq_route), 4, 0});',
        # dispatch_qmm takes QQMatmul's global scale to the _hgs kernels.
        '(B > 1 ? "_batch_1" : "_batch_0") + (global_scale ? "_hgs" : "");\n'
        "  require_entry(entry);\n"
        "  const uint32_t tiles_n",
        "if (global_scale && (!transpose || biases)) {",
    ):
        assert text in backend, text
    assert (
        backend.count(
            'buffers.push_back({"route", "utf8", const_cast<char*>(qq_route), 4, 0});'
        )
        == 2
    )
    # prepare replaces no_gpu's NO_GPU(QQMatmul) and installs the port;
    # test_portable_host checks the adapted tree.
    adapter = Path(prepare.__file__).read_text(encoding="utf-8")
    assert '        "QuantizedMatmul",\n        "QQMatmul",\n' in adapter
    assert '"crosstl_qq_matmul.inc": (HERE / "qq_matmul.inc").read_bytes(),' in adapter


def test_qq_products_run_in_their_own_upstream_group():
    assert upstream_module_groups.GROUPS["quantized-products"] == (
        "test_quantized.TestQuantized.test_qqmm",
        "test_quantized.TestQuantized.test_qqmv",
        "test_quantized.TestQuantized.test_qqmm_global_scale_matrix_kernels",
        "test_quantized.TestQuantized.test_gather_qqmm",
        "test_quantized.TestQuantized.test_gather_qmm",
        "test_quantized.TestQuantized.test_gather_matmul_grad",
        "test_quantized.TestQuantized.test_gather_qmm_matrix_path",
        "test_quantized.TestQuantized.test_gather_qmm_sorted_sliced_weight",
        "test_quantized.TestQuantized.test_gather_qmm_sorted_unaligned_k",
        "test_quantized.TestQuantized.test_gather_qmm_grad",
        "test_quantized.TestQuantized.test_gather_qqmm_global_scale_matrix_paths",
    )
    # gather_qmm_rhs's longest products run in groups of their own.
    assert upstream_module_groups.GROUPS["quantized-sorted"] == (
        "test_quantized.TestQuantized.test_gather_qmm_sorted",
        "test_quantized.TestQuantized.test_gather_qmm_sorted_nax_large_m",
    )
    assert upstream_module_groups.GROUPS["quantized-fp-tails"] == (
        "test_quantized.TestQuantized.test_fp_gather_qmm_non_multiple_of_32",
    )
    # test_qqmm and the GatherQQMM tests skip on MLX's CPU, which has neither
    # QQMatmul nor GatherQQMM, so the CPU worker must skip them and the
    # native worker must run them; the large sorted product and QQMatmul's
    # globally scaled matrix products run only where mx.metal.is_available().
    # test_qqmv and MLX's CPU GatherQMM tests run on both.
    for name in (
        "test_gather_qmm_sorted_nax_large_m",
        "test_qqmm_global_scale_matrix_kernels",
    ):
        assert (
            f"test_quantized.TestQuantized.{name}"
            in verify_upstream_ops.METAL_PATH_TESTS
        )
    for name in (
        "test_qqmm",
        "test_qqmm_global_scale_matrix_kernels",
        "test_gather_qqmm",
        "test_gather_qqmm_global_scale_matrix_paths",
        "test_gather_qmm_sorted_nax_large_m",
    ):
        assert (
            f"test_quantized.TestQuantized.{name}" in verify_upstream_ops.GPU_ONLY_TESTS
        )
    for name in (
        "test_qqmv",
        "test_gather_qmm",
        "test_gather_matmul_grad",
        "test_gather_qmm_matrix_path",
        "test_fp_gather_qmm_non_multiple_of_32",
        "test_gather_qmm_sorted",
        "test_gather_qmm_sorted_sliced_weight",
        "test_gather_qmm_sorted_unaligned_k",
        "test_gather_qmm_grad",
    ):
        assert f"test_quantized.TestQuantized.{name}" not in (
            verify_upstream_ops.GPU_ONLY_TESTS
        )
