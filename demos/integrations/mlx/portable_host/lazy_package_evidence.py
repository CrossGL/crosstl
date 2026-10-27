"""Tie lazily translated scan, reduction and matrix dispatches to their caches.

Scans, small row reductions, small column reductions, GEMV, dot-product,
steel GEMM, split-K GEMM, GatherMM, SegmentedMM, BlockMaskedMM, convolution,
Stockham FFT and affine and floating-point quantized matrix-vector and matrix-matrix
kernels are translated from the pinned sources on first use, so no static index lists
their packages. Each dispatch is checked by deriving its
package again from the same cache, from the event's entry, launch and
metadata, and by comparing the recorded artifact with the unchanged cached
source. Scans and matrix kernels run through the source-specialized executor,
so their retained native modules are checked as for indexing kernels. Small
rows on Direct3D and OpenGL run as dispatch regions, each of which must name
its cached region package and a retained native module.
"""

import hashlib
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    block_masked_mm_layout,
    column_small_layout,
    conv_layout,
    dot_layout,
    fft_layout,
    fp_matmul_layout,
    fp_qmm_layout,
    gather_mm_layout,
    gather_mv_layout,
    gemm_layout,
    gemm_splitk_layout,
    gemv_layout,
    matmul_layout,
    qmm_layout,
    scan_layout,
    sdpa_layout,
    sdpa_vjp_layout,
    segmented_mm_layout,
    steel_attention_layout,
)
from demos.integrations.mlx.portable_host.block_masked_mm_packages import (
    MASKED_MM_CACHE,
    MASKED_MV_CACHE,
    MaskedMmPackageCache,
    MaskedMvPackageCache,
)
from demos.integrations.mlx.portable_host.column_small_packages import (
    CACHE_DIRECTORY as SMALL_COLUMN_CACHE,
)
from demos.integrations.mlx.portable_host.column_small_packages import (
    SmallColumnPackageCache,
)
from demos.integrations.mlx.portable_host.conv_packages import (
    CACHE_DIRECTORY as CONV_CACHE,
)
from demos.integrations.mlx.portable_host.conv_packages import ConvPackageCache
from demos.integrations.mlx.portable_host.dot_packages import (
    CACHE_DIRECTORY as DOT_CACHE,
)
from demos.integrations.mlx.portable_host.dot_packages import DotPackageCache
from demos.integrations.mlx.portable_host.fft_packages import (
    CACHE_DIRECTORY as FFT_CACHE,
)
from demos.integrations.mlx.portable_host.fft_packages import FftPackageCache
from demos.integrations.mlx.portable_host.fp_matmul_packages import (
    CACHE_DIRECTORY as FP_QUANTIZED_MV_CACHE,
)
from demos.integrations.mlx.portable_host.fp_matmul_packages import (
    FpQuantizedMatmulPackageCache,
)
from demos.integrations.mlx.portable_host.fp_qmm_packages import (
    CACHE_DIRECTORY as FP_QUANTIZED_MM_CACHE,
)
from demos.integrations.mlx.portable_host.fp_qmm_packages import (
    FpQuantizedQmmPackageCache,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.gather_mm_packages import (
    GATHER_MM_CACHE,
    GATHER_MV_CACHE,
    GATHER_OFFSETS_CACHE,
    GatherMmPackageCache,
    GatherMvPackageCache,
    GatherOffsetsPackageCache,
)
from demos.integrations.mlx.portable_host.gemm_packages import (
    CACHE_DIRECTORY as GEMM_CACHE,
)
from demos.integrations.mlx.portable_host.gemm_packages import GemmPackageCache
from demos.integrations.mlx.portable_host.gemm_splitk_packages import (
    CACHE_DIRECTORY as SPLITK_CACHE,
)
from demos.integrations.mlx.portable_host.gemm_splitk_packages import SplitKPackageCache
from demos.integrations.mlx.portable_host.gemv_packages import (
    CACHE_DIRECTORY as GEMV_CACHE,
)
from demos.integrations.mlx.portable_host.gemv_packages import GemvPackageCache
from demos.integrations.mlx.portable_host.matmul_packages import (
    CACHE_DIRECTORY as QUANTIZED_MV_CACHE,
)
from demos.integrations.mlx.portable_host.matmul_packages import (
    QuantizedMatmulPackageCache,
)
from demos.integrations.mlx.portable_host.qmm_packages import (
    CACHE_DIRECTORY as QUANTIZED_MM_CACHE,
)
from demos.integrations.mlx.portable_host.qmm_packages import QuantizedQmmPackageCache
from demos.integrations.mlx.portable_host.scan_packages import ScanPackageCache
from demos.integrations.mlx.portable_host.sdpa_packages import (
    CACHE_DIRECTORY as SDPA_CACHE,
)
from demos.integrations.mlx.portable_host.sdpa_packages import SdpaPackageCache
from demos.integrations.mlx.portable_host.sdpa_vjp_packages import (
    CACHE_DIRECTORY as SDPA_VJP_CACHE,
)
from demos.integrations.mlx.portable_host.sdpa_vjp_packages import SdpaVjpPackageCache
from demos.integrations.mlx.portable_host.segmented_mm_packages import (
    SEGMENTED_MM_CACHE,
    SegmentedMmPackageCache,
)
from demos.integrations.mlx.portable_host.small_row_packages import (
    ALL_SMALL_ROW_ENTRIES,
    SMALL_ROW_ENTRIES,
    SmallRowPackageCache,
)
from demos.integrations.mlx.portable_host.steel_attention_packages import (
    CACHE_DIRECTORY as STEEL_ATTENTION_CACHE,
)
from demos.integrations.mlx.portable_host.steel_attention_packages import (
    SteelAttentionPackageCache,
)

# Matrix kernels: one package per entry, launched by the source-specialized
# executor like scans. The first matching prefix names the family, so the
# gathered and masked gemv prefixes precede GEMV's.
MATRIX_FAMILIES = {
    "masked-mv": block_masked_mm_layout.GEMV_PREFIXES,
    "masked-mm": block_masked_mm_layout.GEMM_PREFIXES,
    "gather-mv": gather_mv_layout.ENTRY_PREFIXES,
    "gather-mm": gather_mm_layout.ENTRY_PREFIXES,
    "gather-mm-offsets": (gather_mm_layout.OFFSETS_ENTRY,),
    "segmented-mm": segmented_mm_layout.ENTRY_PREFIXES,
    "gemv": gemv_layout.ENTRY_PREFIXES,
    "dot": dot_layout.ENTRY_PREFIXES,
    "gemm": gemm_layout.ENTRY_PREFIXES,
    "gemm-splitk": gemm_splitk_layout.ENTRY_PREFIXES,
    "conv": conv_layout.ENTRY_PREFIXES,
    "fft": fft_layout.ENTRY_PREFIXES,
    "sdpa": sdpa_layout.ENTRY_PREFIXES,
    "steel-attention": steel_attention_layout.ENTRY_PREFIXES,
    "sdpa-vjp": sdpa_vjp_layout.ENTRY_PREFIXES,
}
# Quantized products run through the same executor. Their own upstream
# verifiers translate their packages ahead; any other test that reaches one
# translates it on demand.
QUANTIZED_FAMILIES = {
    "quantized-mv": matmul_layout.ENTRY_PREFIXES,
    "fp-quantized-mv": fp_matmul_layout.ENTRY_PREFIXES,
    "quantized-mm": qmm_layout.ENTRY_PREFIXES,
    "fp-quantized-mm": fp_qmm_layout.ENTRY_PREFIXES,
}
EXECUTED_FAMILIES = {**MATRIX_FAMILIES, **QUANTIZED_FAMILIES}
FAMILIES = ("scan", "small-row", "small-column", *EXECUTED_FAMILIES)
GEOMETRY = ("workgroupCount", "workgroupSize")


def family(entry):
    """The lazily translated family of ``entry``, or None."""
    if not isinstance(entry, str):
        return None
    if entry.startswith(scan_layout.ENTRY_PREFIXES):
        return "scan"
    if entry in SMALL_ROW_ENTRIES:
        return "small-row"
    if entry in column_small_layout.ALL_SMALL_COLUMN_ENTRIES:
        return "small-column"
    for name, prefixes in EXECUTED_FAMILIES.items():
        if name == "gather-mm-offsets":
            if entry in prefixes:
                return name
        elif entry.startswith(prefixes):
            return name
    return None


class LazyPackages:
    """The host's lazy caches for one package directory and target."""

    def __init__(self, mlx_root, packages, target):
        packages = Path(packages)
        self.target = target
        self.caches = {
            "gather-mv": GatherMvPackageCache(
                mlx_root, packages / GATHER_MV_CACHE, target
            ),
            "gather-mm": GatherMmPackageCache(
                mlx_root, packages / GATHER_MM_CACHE, target
            ),
            "gather-mm-offsets": GatherOffsetsPackageCache(
                mlx_root, packages / GATHER_OFFSETS_CACHE, target
            ),
            "segmented-mm": SegmentedMmPackageCache(
                mlx_root, packages / SEGMENTED_MM_CACHE, target
            ),
            "masked-mv": MaskedMvPackageCache(
                mlx_root, packages / MASKED_MV_CACHE, target
            ),
            "masked-mm": MaskedMmPackageCache(
                mlx_root, packages / MASKED_MM_CACHE, target
            ),
            "scan": ScanPackageCache(mlx_root, packages / "scans", target),
            "small-row": SmallRowPackageCache(
                mlx_root, packages / "small-rows", target
            ),
            "small-column": SmallColumnPackageCache(
                mlx_root, packages / SMALL_COLUMN_CACHE, target
            ),
            "gemv": GemvPackageCache(mlx_root, packages / GEMV_CACHE, target),
            "dot": DotPackageCache(mlx_root, packages / DOT_CACHE, target),
            "gemm": GemmPackageCache(mlx_root, packages / GEMM_CACHE, target),
            "gemm-splitk": SplitKPackageCache(
                mlx_root, packages / SPLITK_CACHE, target
            ),
            "conv": ConvPackageCache(mlx_root, packages / CONV_CACHE, target),
            "fft": FftPackageCache(mlx_root, packages / FFT_CACHE, target),
            "sdpa": SdpaPackageCache(mlx_root, packages / SDPA_CACHE, target),
            "steel-attention": SteelAttentionPackageCache(
                mlx_root, packages / STEEL_ATTENTION_CACHE, target
            ),
            "sdpa-vjp": SdpaVjpPackageCache(
                mlx_root, packages / SDPA_VJP_CACHE, target
            ),
            "quantized-mv": QuantizedMatmulPackageCache(
                mlx_root, packages / QUANTIZED_MV_CACHE, target
            ),
            "fp-quantized-mv": FpQuantizedMatmulPackageCache(
                mlx_root, packages / FP_QUANTIZED_MV_CACHE, target
            ),
            "quantized-mm": QuantizedQmmPackageCache(
                mlx_root, packages / QUANTIZED_MM_CACHE, target
            ),
            "fp-quantized-mm": FpQuantizedQmmPackageCache(
                mlx_root, packages / FP_QUANTIZED_MM_CACHE, target
            ),
        }

    def packages(self, event):
        """The ``(descriptor, directory)`` pairs ``event`` must have used."""
        entry, name = event["entry"], family(event["entry"])
        cache = self.caches.get(name)
        require(cache is not None, "Dispatch is not lazily translated")
        if name == "scan":
            execution = {key: event[key] for key in GEOMETRY}
            return [cache.get(entry, event["scanMetadata"], execution)]
        if name == "small-row":
            return list(
                cache.get(
                    entry,
                    thread_grid_size=event["threadGridSize"],
                    workgroup_size=event["workgroupSize"],
                )
            )
        if name in EXECUTED_FAMILIES:
            return [cache.get(entry)]
        key = column_small_layout.variant(entry, tuple(event["workgroupSize"]))
        return [cache.get(key)]


def require_unchanged(descriptor, directory):
    artifact = descriptor["artifact"]
    root = Path(directory).resolve()
    path = (root / artifact["packagePath"]).resolve()
    require(path.is_relative_to(root), "Lazy package artifact escapes its package")
    data = path.read_bytes()
    require(
        len(data) == artifact["sizeBytes"]
        and artifact["hash"] == {
            "algorithm": "sha256",
            "value": hashlib.sha256(data).hexdigest(),
        },
        "Lazy package artifact changed after translation",
    )


def audit_regions(event, packages, modules):
    """Each Direct3D or OpenGL region names its package and native module."""
    regions = event["details"]["regions"]
    require(
        event["entry"] in ALL_SMALL_ROW_ENTRIES and len(regions) == len(packages),
        "Lazy dispatch region coverage differs",
    )
    for region, (descriptor, directory) in zip(regions, packages):
        geometry = descriptor["provenance"]["dispatchRegion"]
        module = Path(region["moduleFile"]).resolve()
        require(
            region["artifact"] == descriptor["artifact"]
            and region["provenance"] == descriptor["provenance"]
            and Path(region["packageRoot"]).resolve() == Path(directory).resolve()
            and all(region[key] == geometry[key] for key in GEOMETRY)
            and module.is_relative_to(modules)
            and hashlib.sha256(module.read_bytes()).hexdigest() == region["moduleHash"],
            "Lazy dispatch region identity differs",
        )


def audit_event(event, lazy, modules):
    """Check one lazily translated dispatch against its cached package.

    Returns True when this check covered the native execution, so the
    caller's per-dispatch native identity check must not run again; small
    reductions on a single dispatch keep that check.
    """
    name = family(event["entry"])
    require(event["target"] == lazy.target, "Lazy dispatch targets another backend")
    packages = lazy.packages(event)
    require(packages, "Lazy dispatch has no cached package")
    for descriptor, directory in packages:
        require_unchanged(descriptor, directory)
    require(
        event["artifact"] == packages[0][0]["artifact"]
        and event["dispatchVersion"] == 3,
        "Lazy dispatch does not name its cached artifact",
    )
    if "regions" in event["details"]:
        audit_regions(event, packages, Path(modules).resolve())
        return True
    require(len(packages) == 1, "Lazy dispatch skipped its regions")
    executed = name == "scan" or name in EXECUTED_FAMILIES
    require(
        ("packageRoot" in event or not executed)
        and Path(event.get("packageRoot", packages[0][1])).resolve()
        == Path(packages[0][1]).resolve(),
        "Lazy dispatch names another package",
    )
    if not executed:
        return False
    request = event["details"]["request"]
    entry_point = (
        packages[0][0]["entryPoint"]["name"]
        if name in EXECUTED_FAMILIES
        else {"metal": event["entry"], "opengl": "main", "directx": "CSMain"}[
            lazy.target
        ]
    )
    require(
        request["target"] == lazy.target
        and request["entryPoint"] == entry_point
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Lazy native launch differs",
    )
    audit_native_execution(event)
    return True
