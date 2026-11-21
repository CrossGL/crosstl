"""Cache translated MLX BlockMaskedMM entries: masked gemv and masked steel GEMM.

``gemv_masked`` and ``gemv_t_masked`` come from the pinned gemv_masked.metal,
``block_masked_gemm`` from steel_gemm_masked.metal, each with the options of
its current-tree kernel gate (``test_current_gemv_masked`` and
``test_current_gemm_masked``). Run as a module, it translates entries into a
host package directory's caches ahead of execution, several processes at a
time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import block_masked_mm_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directories under a host package root that HostRuntime caches entries in.
MASKED_MV_CACHE = "masked-mv"
MASKED_MM_CACHE = "masked-mm"
SIMD_SIZE = layout.SIMD_SIZE
# The steel kernel's mask parameters: eval_gpu binds the masks as inputs, and
# nothing writes them while the kernel runs, which lets the subgroup proofs
# read them as uniform.
GEMM_MASK_PARAMETERS = ("out_mask", "lhs_mask", "rhs_mask")


def masked_mv_config(root, output_dir, target, entry):
    """One masked gemv entry, with the current-tree masked gemv gate's options."""
    layout.signature(entry)
    source = layout.GEMV_SOURCE
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (entry,)},
        entry_workgroup_size_rules={source: {entry: layout.workgroup_size(entry)}},
        # block_masked_mm_layout.plan bounds every bound buffer at
        # MAX_ELEMENTS elements, so each asserted offset stays below 2^31 - 1.
        index_range_assertions=(
            list(layout.gemv_index_assertions(entry)) if target == "opengl" else []
        ),
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 16384,
                "target_options": {
                    "directx": {
                        "software_subgroup_width": SIMD_SIZE,
                        "relative_wave_shuffle_out_of_range": "self",
                    },
                    "opengl": {"software_subgroup_width": SIMD_SIZE},
                },
            }
        },
    )


def masked_mm_config(root, output_dir, target, entry):
    """One masked steel entry, with the current-tree masked GEMM gate's options."""
    layout.signature(entry)
    source = layout.GEMM_SOURCE
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (entry,)},
        workgroup_size_rules={source: layout.workgroup_size(entry)},
        immutable_storage=[
            {"source": source, "entryPoint": entry, "parameter": name}
            for name in GEMM_MASK_PARAMETERS
        ],
        workgroup_access_assertions=(
            layout.loader_access_assertions(entry) if target != "metal" else ()
        ),
        index_range_assertions=(
            layout.gemm_index_assertions(entry) if target == "opengl" else ()
        ),
        source_options={
            "metal": {
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "preserve_resource_origins": True,
                "target_options": {
                    "directx": {
                        "cooperative_matrix_software_lowering": True,
                        "software_subgroup_width": SIMD_SIZE,
                        "relative_wave_shuffle_out_of_range": "self",
                    },
                    "opengl": {
                        "cooperative_matrix_software_lowering": True,
                        "software_subgroup_width": SIMD_SIZE,
                    },
                },
            }
        },
    )


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("block_masked_mm_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


class _MaskedCache(TranslatedEntryCache):
    KIND = None

    def signature(self, entry):
        info = layout.signature(entry)
        if info["kind"] != self.KIND:
            raise ValueError(f"Not a {self.LABEL.lower()} entry")
        return info

    def workgroup_size(self, entry):
        return layout.workgroup_size(entry)

    def recipe_hashes(self):
        return _recipe_hashes()


class MaskedMvPackageCache(_MaskedCache):
    LABEL = "Block masked matrix-vector"
    SOURCE = layout.GEMV_SOURCE
    PREFIX = "masked-mv"
    KIND = "masked-mv"

    def project_config(self, output_dir, entry):
        return masked_mv_config(self.root, output_dir, self.target, entry)


class MaskedMmPackageCache(_MaskedCache):
    LABEL = "Block masked steel GEMM"
    SOURCE = layout.GEMM_SOURCE
    PREFIX = "masked-mm"
    KIND = "masked-mm"

    def project_config(self, output_dir, entry):
        return masked_mm_config(self.root, output_dir, self.target, entry)


CACHES = {
    "masked-mv": (MASKED_MV_CACHE, MaskedMvPackageCache),
    "masked-mm": (MASKED_MM_CACHE, MaskedMmPackageCache),
}


def cache_for(root, packages, target, entry):
    """The cache ``entry`` translates into under one host package root."""
    directory, cache = CACHES[layout.kind_of(entry)]
    return cache(root, Path(packages) / directory, target)


def _build_one(root, packages, target, entry):
    return build_record(cache_for(root, packages, target, entry), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' caches, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        layout.signature,
        "BlockMaskedMM",
        root,
        Path(packages).resolve(),
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The entries verify_block_masked_mm's cases dispatch."""
    from demos.integrations.mlx.portable_host import verify_block_masked_mm

    entries = [
        entry
        for case in verify_block_masked_mm.CASES
        for entry in verify_block_masked_mm.case_record(case)["entries"]
    ]
    return list(dict.fromkeys(entries))


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
