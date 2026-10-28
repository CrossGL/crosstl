"""Cache translated steel GEMM entries of the pinned steel_gemm_fused.metal.

Each host entry names a source entry point and the function constants
matmul.cpp specializes it with; the translation freezes those constants. Run
as a module, it translates entries into a host package directory's cache ahead
of execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.gemm_layout import (
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
CACHE_DIRECTORY = "gemm"


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("gemm_layout.py").read_bytes()).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """One entry's translation, with the current-tree GEMM gate's options."""
    info = signature(entry)
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (info["source"],)},
        specialization_constants=dict(info["constants"]),
        freeze_specialization_constants=True,
        workgroup_size_rules={SOURCE: workgroup_size(entry)},
        workgroup_access_assertions=(
            loader_access_assertions(entry) if target != "metal" else ()
        ),
        # gemm_layout.plan bounds every bound buffer at MAX_ELEMENTS elements,
        # so each asserted offset stays below 2^31 - 1.
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


class GemmPackageCache(TranslatedEntryCache):
    LABEL = "Steel GEMM"
    SOURCE = SOURCE
    PREFIX = "gemm"

    def signature(self, entry):
        return signature(entry)

    def source_entry(self, entry):
        return signature(entry)["source"]

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(GemmPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        "Steel GEMM",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    from demos.integrations.mlx.portable_host import verify_matmul

    records = (verify_matmul.case_record(case) for case in verify_matmul.CASES)
    return [record["entry"] for record in records if record["family"] == "gemm"]


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
