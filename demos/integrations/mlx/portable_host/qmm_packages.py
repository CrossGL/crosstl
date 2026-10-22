"""Cache translated affine matrix-matrix entries of the unchanged pinned source.

Each entry is translated with the current-tree qmm_t, qmm_n and qmm_t_splitk
gates' options. Run as a module, it translates entries into a host package
directory's cache ahead of execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import qmm_layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)
from demos.integrations.mlx.portable_host.qmm_layout import (
    K_LIMIT,
    SIMD_SIZE,
    SOURCE,
    signature,
    workgroup_size,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "quantized-qmm"


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("qmm_layout.py").read_bytes()).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry, layout=qmm_layout):
    """One entry's translation, with the current-tree qmm gates' options.

    BlockMMA has no simd_sum: software subgroups give DirectX and OpenGL
    MLX's 32 lanes, and the cooperative-matrix lowering MLX's fragments.
    ``layout`` is ``qmm_layout`` or ``fp_qmm_layout``, whose kernels share
    these tiles: it names the source, workgroup and assertions.
    """
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
        include_patterns=(layout.SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={layout.SOURCE: (entry,)},
        entry_workgroup_size_rules={
            layout.SOURCE: {entry: layout.workgroup_size(entry)}
        },
        workgroup_access_assertions=(
            layout.access_assertions(entry) if target != "metal" else ()
        ),
        # The layout's plan bounds every bound buffer at MAX_ELEMENTS elements.
        index_range_assertions=(
            layout.index_assertions(entry) if target == "opengl" else ()
        ),
        metadata_preconditions=[
            dict(
                source=layout.SOURCE,
                entryPoint=entry,
                parameter=parameter,
                minimum=0,
                maximum=K_LIMIT,
            )
            for parameter in layout.bounded_parameters(entry)
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


class QuantizedQmmPackageCache(TranslatedEntryCache):
    LABEL = "Quantized matrix-matrix"
    SOURCE = SOURCE
    PREFIX = "quantized-qmm"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(QuantizedQmmPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        "Quantized matrix-matrix",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    from demos.integrations.mlx.portable_host import verify_quantized_qmm

    return sorted(
        {
            verify_quantized_qmm.case_record(case)["entry"]
            for case in verify_quantized_qmm.CASES
        }
    )


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
