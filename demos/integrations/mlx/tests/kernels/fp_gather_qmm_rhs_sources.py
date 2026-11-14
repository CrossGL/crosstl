"""Checks that the pinned MLX sources still read as the rhs gate models them.

``test_current_fp_quantized_gather_qmm_rhs.py`` calls these on the pinned
checkout before any native run: the instantiation macros, the kernel's
buffers, tile, loaders and branches, ``schedule_row_tile``, the gemm loop
helpers, ``gather_qmm_rhs``'s dispatch and routing, and
``gather_mm_offsets``.
"""

import re

from demos.integrations.mlx.tests.kernels.current_gate import declared_buffers
from demos.integrations.mlx.tests.kernels.fp_gather_qmm_rhs_model import (
    BK,
    BM,
    BN,
    WM,
    WN,
)
from demos.integrations.mlx.tests.kernels.fp_qmv_model import MODES
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import DTYPES
from demos.integrations.mlx.tests.kernels.test_current_quantized import _macro

KERNEL = "fp_gather_qmm_rhs"
CONSTANT_IDS = {"align_N": 201, "align_K": 202}


def check_instantiation(text):
    """ENTRIES follow fp_quantized.metal's own instantiation macros."""
    types = re.findall(r"^instantiate_quantized_types\((\w+)\)", text, re.MULTILINE)
    assert types == [itype for itype, _ in DTYPES]
    modes = re.findall(
        r"instantiate_quantized_modes\(type, (\w+), (\d+), (\d+)\)",
        _macro(text, "instantiate_quantized_types"),
    )
    assert modes == [(mode, str(size), str(bits)) for mode, size, bits in MODES]
    order = re.findall(
        r"(instantiate_quantized_all_\w+)\(",
        _macro(text, "instantiate_quantized_modes"),
    )
    assert order[-1] == "instantiate_quantized_all_rhs"
    rhs = re.findall(
        r"instantiate_gather_qmm_rhs\(fp_gather_qmm_rhs, gather_qmm_rhs_(\w+), type, "
        r"(\d+), (\d+), (\d+), (\d+), (\d+), (true|false), mode, group_size, bits\)",
        _macro(text, "instantiate_quantized_all_rhs"),
    )
    tile = (str(BM), str(BN), str(BK), str(WM), str(WN))
    assert rhs == [("nt", *tile, "true"), ("nn", *tile, "false")]
    body = _macro(text, "instantiate_gather_qmm_rhs")
    # Each entry is followed by its _hgs twin with has_global_scale = true.
    assert body.count("instantiate_kernel(") == 2
    assert '#wn "_hgs",' in body and body.rstrip().endswith(
        "transpose, \\\n      true)"
    )


def check_kernel(header, parameters):
    """The kernel's buffers, tile, loaders, schedule and branches."""
    assert declared_buffers(header, KERNEL) == parameters
    for name, identifier in CONSTANT_IDS.items():
        assert f"constant bool {name} [[function_constant({identifier})]];" in header
    body = header[header.index(f"[[kernel]] void {KERNEL}(") :]
    body = body[: body.index("\n}\n")]
    for required in (
        "constexpr int BK_padded = (BK + 16 / sizeof(T));",
        "constexpr int BN_padded = (BN + 16 / sizeof(T));",
        "mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;",
        "      transpose ? BN : BK,\n      transpose ? BK : BN,\n"
        "      transpose ? BK_padded : BN_padded,\n      transpose,\n"
        "      WM * WN * SIMD_SIZE,",
        "if (!schedule_row_tile<BM>(\n"
        "          offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {",
        "const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));",
        "const int k_remain = K - K_it * BK;",
        "transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);",
        "wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;",
        "scales += transpose ? y_col_long * K_g : y_col / group_size;",
        "      wl + group * stride_w,\n      scales + group * stride_s,\n"
        "      transpose ? K : N,",
        "      global_scale + group);",
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
        "group_size < BCOLS ? 1 : group_size / BCOLS;",
        "group_size < BCOLS ? BCOLS / group_size : 1;",
        "inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);",
        "    if (bi >= src_tile_dim.y) {",
        "      scales += group_stride;",
    ):
        assert required in loader, required


def check_schedule(utils):
    body = utils[utils.index("METAL_FUNC bool schedule_row_tile(") :]
    body = body[: body.index("\n}\n")]
    for required in (
        "for (int e = 0; e < num_groups; e += 32) {",
        "const int g = e + simd_lane_id;",
        "end = g + 1 < num_groups ? offsets[g + 1] : M;",
        "const int n = (end - start + BM - 1) / BM;",
        "const int tiles_through = tiles_before + simd_prefix_inclusive_sum(n);",
        "ushort(simd_sum(int(tiles_through <= tile)));",
        "(tile - simd_shuffle(tiles_through - n, owner_lane)) * BM;",
        "rows = short(min(BM, simd_shuffle(end, owner_lane) - row));",
        "tiles_before = simd_shuffle(tiles_through, 31);",
    ):
        assert required in body, required


def check_loops(loops):
    for required in (
        "      loader_a.load_safe(short2(tgp_bk, tgp_bm));",
        "          transpose ? short2(tgp_bk, tgp_bn) : short2(tgp_bn, tgp_bk));",
        "  loader_a.load_safe(tile_a);\n  loader_b.load_safe(tile_b);",
    ):
        assert required in loops, required


def check_host(host):
    dispatch = host[host.index("void gather_qmm_rhs(") :]
    dispatch = dispatch[: dispatch.index("\n}\n")]
    for required in (
        f"int bm = {BM}, bn = {BN}, bk = {BK};",
        f"int wm = {WM}, wn = {WN};",
        "const bool align_N = (N % bn) == 0;",
        "const bool align_K = (K % bk) == 0;",
        'mode + (transpose ? "_gather_qmm_rhs_nt_" : "_gather_qmm_rhs_nn_"),',
        'global_scale ? "_hgs" : "");',
        "{&align_N, MTL::DataType::DataTypeBool, 201},",
        "{&align_K, MTL::DataType::DataTypeBool, 202},",
        "\"_align_N_\",\n      align_N ? 't' : 'n',\n      \"_align_K_\",",
        "array offsets = gather_mm_offsets(indices, E, M, d, s);",
        "MTL::Size group_dims(32, wn, wm);",
        "(N + bn - 1) / bn, std::min(M, (M + bm - 1) / bm + E - 1), 1);",
        "  } else if (gs) {\n    compute_encoder.set_input_array(*gs, 3);\n  }",
        "int c = 4;",
        "compute_encoder.set_input_array(offsets, c++);",
        "compute_encoder.set_output_array(out, c++);",
        "compute_encoder.set_bytes(E, c++);",
    ):
        assert required in dispatch, required
    assert "metal::is_nax_available() && transpose" in dispatch
    route = host[host.index("void GatherQMM::eval_gpu(") :]
    route = route[: route.index("\n}\n")]
    assert "if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {" in route
    assert "        x.size() / K,\n        N,\n        K," in route


def check_offsets(source):
    for required in (
        "if (indices[mid] < g) {",
        "offsets[g] = lo;",
    ):
        assert required in source, required
