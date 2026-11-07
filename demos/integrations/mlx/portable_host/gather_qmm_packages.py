"""Cache translated gathered quantized matrix-matrix entries.

Each entry of the unchanged pinned quantized.metal (affine weights) or
fp_quantized.metal (mxfp4, mxfp8 and nvfp4 weights) is translated with the
options of the ungathered qmm entries (qmm_packages): the gathered kernels
run the same qmm_t and qmm_n tiles. Run as a module, it translates entries
into a host package directory's cache ahead of execution, several processes
at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import gather_qmm_layout
from demos.integrations.mlx.portable_host.gather_qmm_layout import (
    SIMD_SIZE,
    select,
    signature,
    source,
    workgroup_size,
)
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)
from demos.integrations.mlx.portable_host.qmm_layout import K_LIMIT

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "gather-quantized-qmm"


def _recipe_hashes():
    here = Path(__file__)
    layouts = ("gather_qmm_layout.py", "gather_qmv_layout.py", "qmm_layout.py")
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                b"".join(here.with_name(name).read_bytes() for name in layouts)
            ).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """One entry's translation, with the ungathered qmm entries' options.

    BlockMMA has no simd_sum: software subgroups give DirectX and OpenGL
    MLX's 32 lanes, and the cooperative-matrix lowering MLX's fragments.
    OpenGL asserts the qmm offsets and the gathered index locations.
    """
    path = source(entry)
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
            "cooperative_matrix_software_lowering": True,
        },
        "opengl": {
            "software_subgroup_width": SIMD_SIZE,
            "cooperative_matrix_software_lowering": True,
        },
    }
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(path,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={path: (entry,)},
        entry_workgroup_size_rules={path: {entry: workgroup_size(entry)}},
        workgroup_access_assertions=(
            gather_qmm_layout.access_assertions(entry) if target != "metal" else ()
        ),
        # The layout's plan bounds every bound buffer at MAX_ELEMENTS elements.
        index_range_assertions=(
            gather_qmm_layout.index_assertions(entry) if target == "opengl" else ()
        ),
        metadata_preconditions=[
            dict(
                source=path,
                entryPoint=entry,
                parameter="K",
                minimum=0,
                maximum=K_LIMIT,
            )
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "target_options": options,
            }
        },
    )


class GatherQuantizedQmmPackageCache(TranslatedEntryCache):
    LABEL = "Gathered quantized matrix-matrix"
    SOURCE = gather_qmm_layout.SOURCE
    PREFIX = "gather-quantized-qmm"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def source_path(self, entry):
        return source(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(GatherQuantizedQmmPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        "Gathered quantized matrix-matrix",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The entries test_gather_qmm and test_gather_matmul_grad route, sorted.

    test_gather_qmm's 32-row products take gather_qmm_t for K of 256 and 512
    and gather_qmm_n for K of 32, with affine weights of 4 bits in groups of
    64 and every block-scaled mode; the gradient of test_gather_matmul_grad
    multiplies by the untransposed weights. test_gather_qmm_matrix_path's
    bfloat16 products of 64 rows take gather_qmm_t for affine and mxfp4
    weights.
    """
    entries = set()
    for mode, group, bits in (
        ("affine", 64, 4),
        ("mxfp4", 32, 4),
        ("mxfp8", 32, 8),
        ("nvfp4", 16, 4),
    ):
        for k in (256, 512):
            entries.add(select(mode, "float32", 32, 32, k, False, True, group, bits))
        for n in (256, 512):
            entries.add(select(mode, "float32", 32, n, 32, False, False, group, bits))
    for mode, group, bits in (("affine", 64, 4), ("mxfp4", 32, 4)):
        entries.add(select(mode, "bfloat16", 64, 512, 512, False, True, group, bits))
    return sorted(entries)


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
