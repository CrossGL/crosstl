"""Cache translated MLX SegmentedMM entries of steel_gemm_segmented.metal.

Each ``steel_segmented_mm`` entry is translated from the pinned source with
its frozen function constants and the options of its current-tree kernel
gate (``test_current_gemm_segmented``). Run as a module, it translates
entries into a host package directory's cache ahead of execution, several
processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import segmented_mm_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directory under a host package root that HostRuntime caches entries in.
SEGMENTED_MM_CACHE = "segmented-mm"


def segmented_mm_config(root, output_dir, target, entry):
    """One segmented entry, with the current-tree segmented GEMM gate's options."""
    info = layout.signature(entry)
    source = layout.SOURCE
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (info["source"],)},
        specialization_constants=dict(info["constants"]),
        freeze_specialization_constants=True,
        workgroup_size_rules={source: layout.workgroup_size(entry)},
        # segmented_mm binds the segments as an input; nothing writes them
        # while the kernel runs, which lets the subgroup proofs read the K
        # range, and so the K loop's trip count, as uniform.
        immutable_storage=[
            {"source": source, "entryPoint": info["source"], "parameter": "segments"}
        ],
        workgroup_access_assertions=(
            layout.loader_access_assertions(entry) if target != "metal" else ()
        ),
        index_range_assertions=(
            layout.index_assertions(entry) if target == "opengl" else ()
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
                        "software_subgroup_width": layout.SIMD_SIZE,
                        "relative_wave_shuffle_out_of_range": "self",
                    },
                    "opengl": {
                        "cooperative_matrix_software_lowering": True,
                        "software_subgroup_width": layout.SIMD_SIZE,
                    },
                },
            }
        },
    )


class SegmentedMmPackageCache(TranslatedEntryCache):
    LABEL = "Segmented steel GEMM"
    SOURCE = layout.SOURCE
    PREFIX = "segmented-mm"

    def signature(self, entry):
        return layout.signature(entry)

    def source_entry(self, entry):
        return layout.signature(entry)["source"]

    def workgroup_size(self, entry):
        return layout.workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return segmented_mm_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        here = Path(__file__)
        return {
            "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
            "layoutHash": (
                hashlib.sha256(
                    here.with_name("segmented_mm_layout.py").read_bytes()
                ).hexdigest()
            ),
        }


def _check(entry):
    layout.signature(entry)


def _build_one(root, directory, target, entry):
    cache = SegmentedMmPackageCache(root, Path(directory) / SEGMENTED_MM_CACHE, target)
    return build_record(cache, entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        _check,
        "SegmentedMM",
        root,
        Path(packages).resolve(),
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The entries verify_segmented_mm's cases dispatch."""
    from demos.integrations.mlx.portable_host import verify_segmented_mm

    entries = [
        entry
        for case in verify_segmented_mm.CASES
        for entry in verify_segmented_mm.case_record(case)["entries"]
    ]
    return list(dict.fromkeys(entries))


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
