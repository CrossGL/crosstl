"""Cache translated sorted gathered quantized matrix-matrix entries.

Each entry specializes one ``affine_gather_qmm_rhs`` instantiation of the
unchanged pinned quantized.metal, or one ``fp_gather_qmm_rhs`` instantiation
of fp_quantized.metal, with the ``align_N`` and ``align_K`` constants its name
carries, frozen. The kernels run the qmm entries' BlockMMA tiles, so they
translate with the qmm entries' options (gather_qmm_packages). Run as a
module, it translates entries into a host package directory's cache ahead of
execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import gather_qmm_rhs_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)
from demos.integrations.mlx.portable_host.qmm_layout import K_LIMIT

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "gather-quantized-qmm-rhs"


def _recipe_hashes():
    here = Path(__file__)
    layouts = (
        "gather_qmm_rhs_layout.py",
        "gather_qmm_layout.py",
        "gather_qmv_layout.py",
        "qmm_layout.py",
    )
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                b"".join(here.with_name(name).read_bytes() for name in layouts)
            ).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """One entry's translation, with the qmm entries' options.

    BlockMMA has no simd_sum: software subgroups give DirectX and OpenGL
    MLX's 32 lanes, and the cooperative-matrix lowering MLX's fragments.
    """
    info = layout.signature(entry)
    path = layout.source(entry)
    kernel = info["source"]
    options = {
        "directx": {
            "software_subgroup_width": layout.SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
            "cooperative_matrix_software_lowering": True,
        },
        "opengl": {
            "software_subgroup_width": layout.SIMD_SIZE,
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
        entry_points={path: (kernel,)},
        entry_workgroup_size_rules={path: {kernel: layout.workgroup_size(entry)}},
        specialization_constants=dict(info["constants"]),
        freeze_specialization_constants=True,
        workgroup_access_assertions=(
            layout.access_assertions(entry) if target != "metal" else ()
        ),
        metadata_preconditions=[
            dict(
                source=path,
                entryPoint=kernel,
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


class GatherQuantizedQmmRhsPackageCache(TranslatedEntryCache):
    LABEL = layout.LABEL
    SOURCE = layout.SOURCE
    PREFIX = "gather-quantized-qmm-rhs"

    def signature(self, entry):
        return layout.signature(entry)

    def workgroup_size(self, entry):
        return layout.workgroup_size(entry)

    def source_entry(self, entry):
        return layout.source_entry(entry)

    def source_path(self, entry):
        return layout.source(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(
        GatherQuantizedQmmRhsPackageCache(root, directory, target), entry
    )


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        layout.signature,
        layout.LABEL,
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The entries test_gather_qmm_sorted and its variants route, sorted.

    test_gather_qmm_sorted multiplies float16 affine weights of 4 bits in
    groups of 64, transposed with N of 512 and 555 and untransposed with K
    of 512, 544 and 555, and bfloat16 mxfp4, mxfp8 and nvfp4 weights with
    544 rows or columns; test_fp_gather_qmm_non_multiple_of_32 multiplies
    nvfp4 weights whose K or N leaves half a tile, and
    test_gather_qqmm_global_scale_matrix_paths globally scaled bfloat16
    nvfp4 weights of 64 rows.
    """
    entries = set()
    for n in (512, 555):
        entries.add(layout.select("affine", "float16", n, 512, False, True, 64, 4))
    for k in (512, 544, 555):
        entries.add(layout.select("affine", "float16", 512, k, False, False, 64, 4))
    for mode in layout.MODES:
        entries.add(layout.select(mode, "bfloat16", 544, 512, False, True))
        entries.add(layout.select(mode, "bfloat16", 512, 544, False, False))
    for dtype in ("float32", "float16", "bfloat16"):
        entries.add(layout.select("nvfp4", dtype, 128, 1040, False, True))
        entries.add(layout.select("nvfp4", dtype, 1040, 512, False, False))
    entries.add(layout.select("nvfp4", "bfloat16", 64, 256, True, True))
    return sorted(entries)


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
