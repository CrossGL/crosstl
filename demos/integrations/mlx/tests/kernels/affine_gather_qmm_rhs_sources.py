"""Checks that the pinned MLX sources still read as the affine rhs gate models them.

``test_current_quantized_gather_qmm_rhs.py`` calls these on the pinned
checkout before any native run: ``quantized.metal``'s instantiation macros,
the kernel's buffers, tile, loaders and branches, the affine
``QuantizedBlockLoader`` and ``gather_qmm_rhs``'s affine bindings. The
schedule, gemm loop, dispatch and offset checks are the fp rhs gate's, since
both kernels share them.
"""

import re

from demos.integrations.mlx.tests.kernels.fp_gather_qmm_rhs_model import (
    BK,
    BM,
    BN,
    WM,
    WN,
)
from demos.integrations.mlx.tests.kernels.fp_gather_qmm_rhs_sources import (
    CONSTANT_IDS,
    check_host,
    check_loops,
    check_offsets,
    check_schedule,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    BITS,
    DTYPES,
    GROUP_SIZES,
    _macro,
)

__all__ = (
    "CONSTANT_IDS",
    "KERNEL",
    "check_affine_host",
    "check_host",
    "check_instantiation",
    "check_kernel",
    "check_loops",
    "check_offsets",
    "check_schedule",
)

KERNEL = "affine_gather_qmm_rhs"


def check_instantiation(text):
    """ENTRIES follow quantized.metal's own instantiation macros."""
    groups = re.findall(
        r"^  instantiate_quantized_groups\((\d+)\)",
        _macro(text, "instantiate_quantized_all"),
        re.MULTILINE,
    )
    assert groups == [str(bits) for bits in BITS]
    sizes = re.findall(
        r"instantiate_quantized_types\((\d+), bits\)",
        _macro(text, "instantiate_quantized_groups"),
    )
    assert sizes == [str(size) for size in GROUP_SIZES]
    types = re.findall(
        r"instantiate_quantized_funcs\((\w+), group_size, bits\)",
        _macro(text, "instantiate_quantized_types"),
    )
    assert types == [itype for itype, _ in DTYPES]
    order = re.findall(
        r"(instantiate_quantized_all_\w+)\(",
        _macro(text, "instantiate_quantized_funcs"),
    )
    assert order[-1] == "instantiate_quantized_all_rhs"
    rhs = re.findall(
        r"instantiate_gather_qmm_rhs\(affine_gather_qmm_rhs, affine_gather_qmm_rhs_(\w+), "
        r"type, group_size, bits, (\d+), (\d+), (\d+), (\d+), (\d+), (true|false)\)",
        _macro(text, "instantiate_quantized_all_rhs"),
    )
    tile = (str(BM), str(BN), str(BK), str(WM), str(WN))
    assert rhs == [("nt", *tile, "true"), ("nn", *tile, "false")]
    body = _macro(text, "instantiate_gather_qmm_rhs")
    assert body.count("instantiate_kernel(") == 1
    assert '"_wm_" #wm "_wn_" #wn,' in body


def check_kernel(header, parameters):
    """The kernel's buffers, tile, loaders, schedule and branches."""
    match = re.search(rf"\]\]\s*void\s+{KERNEL}\s*\((.*?)\)\s*\{{", header, re.DOTALL)
    assert match, KERNEL
    found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)\]\]", match.group(1))
    assert tuple((name, int(index)) for name, index in found) == parameters
    for name, identifier in CONSTANT_IDS.items():
        assert f"constant bool {name} [[function_constant({identifier})]];" in header
    body = header[header.index(f"[[kernel]] void {KERNEL}(") :]
    body = body[: body.index("\n}\n")]
    for required in (
        "constexpr int pack_factor = get_pack_factor<bits, 8>();",
        "constexpr int bytes_per_pack = get_bytes_per_pack<bits>();",
        "constexpr int BK_padded = (BK + 16 / sizeof(T));",
        "constexpr int BN_padded = (BN + 16 / sizeof(T));",
        "mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;",
        "      transpose ? BN : BK,\n      transpose ? BK : BN,\n"
        "      transpose ? BK_padded : BN_padded,\n      transpose,\n"
        "      WM * WN * SIMD_SIZE,\n      group_size,\n      bits>;",
        "const int K_w = K * bytes_per_pack / pack_factor;",
        "const int N_w = N * bytes_per_pack / pack_factor;",
        "const size_t stride_w = transpose ? N * K_w : K * N_w;",
        "const size_t stride_s = transpose ? N * K_g : K * N_g;",
        "if (!schedule_row_tile<BM>(\n"
        "          offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {",
        "const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));",
        "const int k_remain = K - K_it * BK;",
        "transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);",
        "wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;",
        "scales += transpose ? y_col_long * K_g : y_col / group_size;",
        "biases += transpose ? y_col_long * K_g : y_col / group_size;",
        "      wl + group * stride_w,\n      scales + group * stride_s,\n"
        "      biases + group * stride_s,\n      transpose ? K : N,",
        "if (tgp_bm == BM && (align_N || tgp_bn == BN)) {",
        "mma_op.store_result(y, N);",
        "gemm_loop_unaligned<false, true, transpose>(",
        "mma_op.store_result_safe(y, N, short2(BN, tgp_bm));",
        "gemm_loop_unaligned<true, false, transpose>(",
        "mma_op.store_result_safe(y, N, short2(tgp_bn, BM));",
        "gemm_loop_unaligned<false, false, transpose>(",
        "mma_op.store_result_safe(y, N, short2(tgp_bn, tgp_bm));",
    ):
        assert required in body, required
    assert body.count("if (!align_K) {") == 4
    loader = header[header.index("struct QuantizedBlockLoader {") :]
    loader = loader[: loader.index("\n};\n")]
    for required in (
        "BCOLS <= group_size,",
        "group_size % BCOLS == 0,",
        "MLX_MTL_CONST short group_steps = group_size / BCOLS;",
        "src(src_ + bi * src_ld * bytes_per_pack / pack_factor +\n"
        "            bj * bytes_per_pack),",
        "scales(scales_ + bi * src_ld / group_size),",
        "biases(biases_ + bi * src_ld / group_size) {}",
        "    if (bi >= src_tile_dim.y) {",
        "      scales += group_stride;",
        "          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);",
    ):
        assert required in loader, required
    dequantize = header[
        header.index("inline void dequantize(const device uint8_t* w") :
    ]
    dequantize = dequantize[: dequantize.index("\n}\n")]
    for required in (
        "const float s = float(scale);",
        "const float b = float(bias);",
        "w_local[i] = static_cast<U>(s * w[i] + b);",
    ):
        assert required in dequantize, required


def check_affine_host(host):
    """gather_qmm_rhs's shared dispatch, with the biases in slot 3."""
    check_host(host)
    dispatch = host[host.index("void gather_qmm_rhs(") :]
    dispatch = dispatch[: dispatch.index("\n}\n")]
    assert (
        "  if (biases) {\n    compute_encoder.set_input_array(*biases, 3);" in dispatch
    )
    route = host[host.index("void GatherQMM::eval_gpu(") :]
    route = route[: route.index("\n}\n")]
    assert "if (mode_ == QuantizationMode::Affine) {" in route
    assert "biases = ensure_row_contiguous_matrix(inputs[3], d, s);" in route
