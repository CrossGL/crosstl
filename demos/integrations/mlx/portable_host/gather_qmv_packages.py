"""Cache translated gathered quantized matrix-vector entries.

Each entry of the unchanged pinned fp_quantized.metal is translated with the
options of the floating-point qmv entries (fp_matmul_packages). Run as a
module, it translates entries into a host package directory's cache ahead of
execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.gather_qmv_layout import (
    INDEX_EXPRESSIONS,
    MAX_ELEMENTS,
    SIMD_SIZE,
    SOURCE,
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
CACHE_DIRECTORY = "gather-quantized-matmul"


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("gather_qmv_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """The translation of one entry, with the floating-point qmv entries' options.

    Every gathered kernel computes adjust_matrix_offsets' index locations and
    batch offsets, so OpenGL asserts their range for each entry.
    """
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
        },
        "opengl": {"software_subgroup_width": SIMD_SIZE},
    }
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: workgroup_size(entry)}},
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=MAX_ELEMENTS)
            for expression in INDEX_EXPRESSIONS
            if target == "opengl"
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                "target_options": options,
            }
        },
    )


class GatherQuantizedMatmulPackageCache(TranslatedEntryCache):
    LABEL = "Gathered quantized matrix-vector"
    SOURCE = SOURCE
    PREFIX = "gather-quantized-matmul"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(
        GatherQuantizedMatmulPackageCache(root, directory, target), entry
    )


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        "Gathered quantized matrix-vector",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The entries test_gather_qqmm routes, sorted.

    Its K of 32 and 256 aligns qmv_fast only for mxfp8's 8 bits, so nvfp4,
    whose weights carry global scales, takes the _hgs qmv kernel alone.
    """
    return [
        "mxfp8_gather_qmv_fast_float_gs_32_b_8",
        "mxfp8_gather_qmv_float_gs_32_b_8",
        "nvfp4_gather_qmv_float_gs_16_b_4_hgs",
    ]


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
