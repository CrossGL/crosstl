"""Translate the matrix kernels an upstream op group reaches ahead of its run.

The upstream op host translates GEMV, dot-product, steel GEMM, split-K,
GatherMM and SegmentedMM kernels on first use, one at a time. The BLAS tests
reach about eighty of them, which would leave the native worker too little of
its deadline to run the tests. Run as a module, this translates the entries a
group's tests dispatch into the same caches the host and the audit derive
(``lazy_package_evidence.LazyPackages``), several processes at a time. It only
warms the caches: the native worker still derives every package again and the
audit checks each dispatch against its cache. An entry this step leaves out,
or does not reach within its budget, is translated on demand instead.

``ENTRIES`` lists the matrix entries the native Metal run of each group
dispatched, sorted. The host reports the same device to MLX on every target,
so MLX selects the same kernels on each.
"""

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from demos.integrations.mlx.portable_host import lazy_package_evidence
from demos.integrations.mlx.portable_host.matmul_packages import build_record

# The lazily translated matrix families, slowest first, so the longest
# translations start before the budget can run out.
ORDER = (
    "gemm",
    "gemm-splitk",
    "gather-mm",
    "segmented-mm",
    "gather-mm-offsets",
    "dot",
    "gemv",
    "gather-mv",
)
_CACHES = {}

ENTRIES = {
    "blas": (
        "dot_product_float16_it32_tg512_sg16",
        "dot_product_float32_it32_tg512_sg16",
        "gather_mm_offsets",
        "gemv_float16_bm1_bn1_sm8_sn4_tm1_tn4_nc0_axpby0",
        "gemv_float16_bm1_bn1_sm8_sn4_tm4_tn4_nc0_axpby0",
        "gemv_float16_bm1_bn1_sm8_sn4_tm4_tn4_nc1_axpby0",
        "gemv_float32_bm1_bn1_sm8_sn4_tm1_tn4_nc0_axpby0",
        "gemv_float32_bm1_bn1_sm8_sn4_tm4_tn4_nc0_axpby0",
        "gemv_float32_bm1_bn1_sm8_sn4_tm4_tn4_nc1_axpby0",
        "gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0",
        "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
        "gemv_gather_float32_bm4_bn1_sm1_sn32_tm4_tn4",
        "gemv_t_bfloat16_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0",
        "gemv_t_float16_bm1_bn2_sm8_sn4_tm4_tn1_nc0_axpby0",
        "gemv_t_float16_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0",
        "gemv_t_float16_bm1_bn2_sm8_sn4_tm4_tn4_nc1_axpby0",
        "gemv_t_float32_bm1_bn16_sm8_sn4_tm4_tn4_nc0_axpby0",
        "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn1_nc0_axpby0",
        "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0",
        "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc1_axpby0",
        "gemv_t_float32_bm1_bn4_sm8_sn4_tm4_tn4_nc0_axpby0",
        "steel_gather_mm_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_align_M_n_align_N_n_align_K_t",
        "steel_gather_mm_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_t_align_M_n_align_N_n_align_K_t",
        "steel_gather_mm_nt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_t_align_M_n_align_N_n_align_K_t",
        "steel_gather_mm_rhs_nn_float32_float32_bm16_bn64_bk16_wm1_wn2_align_N_n_align_K_n",
        "steel_gather_mm_rhs_nt_float32_float32_bm16_bn64_bk16_wm1_wn2_align_N_n_align_K_n",
        "steel_gather_mm_tn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_t_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_n",
        "steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_t_align_K_t",
        "steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t",
        "steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_t_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_n",
        "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_t_align_K_t",
        "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_n_align_K_t",
        "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t",
        "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_t_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_nt_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_n",
        "steel_gemm_fused_nt_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t",
        "steel_gemm_fused_nt_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_t_do_axpby_n_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_nt_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_t_use_out_source_n_do_axpby_n_align_M_n_align_N_t_align_K_t",
        "steel_gemm_fused_nt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_n",
        "steel_gemm_fused_nt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_nt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t",
        "steel_gemm_fused_nt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_t_do_axpby_n_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_nt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_t_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_nt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_t_use_out_source_n_do_axpby_n_align_M_n_align_N_t_align_K_t",
        "steel_gemm_fused_tn_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_n",
        "steel_gemm_fused_tn_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t",
        "steel_gemm_fused_tn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_n",
        "steel_gemm_fused_tn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_t",
        "steel_gemm_fused_tn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t",
        "steel_gemm_fused_tt_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_n",
        "steel_gemm_fused_tt_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t",
        "steel_gemm_fused_tt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_n",
        "steel_gemm_fused_tt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n_use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t",
        "steel_gemm_splitk_accum_bfloat16_float32",
        "steel_gemm_splitk_accum_bfloat16_float32_axbpy",
        "steel_gemm_splitk_accum_float16_float32",
        "steel_gemm_splitk_accum_float16_float32_axbpy",
        "steel_gemm_splitk_accum_float32_float32",
        "steel_gemm_splitk_accum_float32_float32_axbpy",
        "steel_gemm_splitk_nn_bfloat16_float32_bm16_bn32_bk16_wm2_wn2_MN_naligned_K_taligned",
        "steel_gemm_splitk_nn_float16_float32_bm32_bn32_bk16_wm2_wn2_MN_naligned_K_naligned",
        "steel_gemm_splitk_nn_float16_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_taligned",
        "steel_gemm_splitk_nn_float32_float32_bm32_bn32_bk16_wm2_wn2_MN_naligned_K_naligned",
        "steel_gemm_splitk_nn_float32_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_taligned",
        "steel_gemm_splitk_nt_bfloat16_float32_bm16_bn32_bk16_wm2_wn2_MN_naligned_K_taligned",
        "steel_gemm_splitk_nt_float16_float32_bm16_bn32_bk16_wm2_wn2_MN_naligned_K_taligned",
        "steel_gemm_splitk_nt_float16_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_taligned",
        "steel_gemm_splitk_nt_float32_float32_bm16_bn32_bk16_wm2_wn2_MN_naligned_K_taligned",
        "steel_gemm_splitk_nt_float32_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_taligned",
        "steel_gemm_splitk_tn_float16_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_taligned",
        "steel_gemm_splitk_tn_float32_float32_bm16_bn16_bk16_wm2_wn2_MN_taligned_K_taligned",
        "steel_gemm_splitk_tn_float32_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_taligned",
        "steel_gemm_splitk_tt_float16_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_taligned",
        "steel_gemm_splitk_tt_float32_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_taligned",
        "steel_segmented_mm_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_segments_contiguous_t_align_M_n_align_N_n",
    ),
}


def ordered(entries):
    """``entries`` in translation order; raises for a non-matrix entry."""
    keyed = []
    for entry in entries:
        name = lazy_package_evidence.family(entry)
        if name not in ORDER:
            raise ValueError(f"Not a lazily translated matrix entry: {entry}")
        keyed.append((ORDER.index(name), entry))
    if len(set(keyed)) != len(keyed):
        raise ValueError("Matrix entries must be unique")
    return [entry for _, entry in sorted(keyed)]


def caches(root, packages, target):
    """This process's lazy caches of one package directory and target."""
    key = (str(root), str(packages), target)
    if key not in _CACHES:
        _CACHES[key] = lazy_package_evidence.LazyPackages(root, packages, target)
    return _CACHES[key].caches


def build_one(root, packages, target, entry):
    cache = caches(root, packages, target)[lazy_package_evidence.family(entry)]
    return build_record(cache, entry)


def build_group(root, packages, target, group, jobs, budget):
    """Translate ``group``'s entries, starting none after ``budget`` seconds.

    Returns the translated records and the entries left to translate on
    demand.
    """
    if group not in ENTRIES:
        raise ValueError(f"No matrix entries are recorded for group {group}")
    if type(jobs) is not int or jobs < 1 or not budget > 0:
        raise ValueError("Matrix warming needs jobs and a positive budget")
    entries = ordered(ENTRIES[group])
    lazy = caches(root, Path(packages).resolve(), target)
    for entry in entries:
        lazy[lazy_package_evidence.family(entry)].signature(entry)
    deadline = time.monotonic() + budget
    records, pending = [], list(entries)
    with ProcessPoolExecutor(max_workers=min(jobs, len(entries))) as pool:
        running = set()
        while pending or running:
            while pending and len(running) < jobs and time.monotonic() < deadline:
                running.add(
                    pool.submit(
                        build_one,
                        str(root),
                        str(Path(packages).resolve()),
                        target,
                        pending.pop(0),
                    )
                )
            if not running:
                break
            done, running = wait(running, return_when=FIRST_COMPLETED)
            records.extend(future.result() for future in done)
    return sorted(records, key=lambda record: record["entry"]), pending


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("metal", "directx", "opengl"), required=True
    )
    parser.add_argument("--group", choices=tuple(ENTRIES), required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--budget-seconds", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    records, deferred = build_group(
        args.mlx_root,
        args.packages,
        args.target,
        args.group,
        args.jobs,
        args.budget_seconds,
    )
    for record in records:
        print(json.dumps(record), flush=True)
    print(json.dumps({"translated": len(records), "deferred": deferred}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
