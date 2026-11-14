"""Cache translated split-K entries of the pinned steel_gemm_splitk.metal.

Each host entry is one ``gemm_splitk`` instantiation or the accumulation
kernel that sums its partitions. Run as a module, it translates entries into
a host package directory's cache ahead of execution, several processes at a
time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.gemm_splitk_layout import (
    SIMD_SIZE,
    SOURCE,
    index_assertions,
    loader_access_assertions,
    signature,
    workgroup_size,
)
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "gemm-splitk"
TARGET_OPTIONS = {
    "directx": {
        "cooperative_matrix_software_lowering": True,
        "software_subgroup_width": SIMD_SIZE,
        "relative_wave_shuffle_out_of_range": "self",
    },
    "opengl": {
        "cooperative_matrix_software_lowering": True,
        "software_subgroup_width": SIMD_SIZE,
    },
}


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("gemm_splitk_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """One entry's translation, with the current-tree split-K gate's options."""
    splitk = signature(entry)["kind"] == "splitk"
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (entry,)},
        workgroup_size_rules={SOURCE: workgroup_size(entry)},
        workgroup_access_assertions=(
            loader_access_assertions(entry) if target != "metal" else ()
        ),
        # gemm_splitk_layout.plan bounds every bound buffer at MAX_ELEMENTS
        # elements, so each asserted offset stays below 2^31 - 1.
        index_range_assertions=(index_assertions(entry) if target == "opengl" else ()),
        source_options={
            "metal": {
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "preserve_resource_origins": True,
                # The accumulation kernels use no simdgroup operations.
                "target_options": TARGET_OPTIONS if splitk else {},
            }
        },
    )


class SplitKPackageCache(TranslatedEntryCache):
    LABEL = "Split-K GEMM"
    SOURCE = SOURCE
    PREFIX = "gemm-splitk"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(SplitKPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        "Split-K GEMM",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    from demos.integrations.mlx.portable_host import verify_matmul

    records = (verify_matmul.case_record(case) for case in verify_matmul.CASES)
    return [
        name
        for record in records
        if record["family"] == "splitk"
        for name in (record["entry"], record["route"]["accum"])
    ]


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
