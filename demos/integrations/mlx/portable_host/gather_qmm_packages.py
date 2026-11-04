"""Cache translated gathered quantized product entries of the unchanged pinned sources.

Each quantized.metal or fp_quantized.metal ``gather_qmv_fast``,
``gather_qmv``, ``gather_qvm``, ``gather_qmm_t`` and ``gather_qmm_n`` entry
is translated with its gate's options: the qmv and qvm gates' software
subgroups for the vector kernels, and the qmm gates' subgroups and
cooperative-matrix lowering for the matrix kernels, with the assertions
``gather_qmm_layout`` names. Run as a module, it translates entries into a
host package directory's cache ahead of execution, several processes at a
time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import gather_qmm_layout
from demos.integrations.mlx.portable_host.gather_qmm_layout import (
    LABEL,
    VECTOR_KERNELS,
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
from demos.integrations.mlx.portable_host.qmm_layout import K_LIMIT, SIMD_SIZE

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "gather-quantized"


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("gather_qmm_layout.py").read_bytes()
            ).hexdigest()
        ),
        # The shared tile and loader assertions.
        "baseLayoutHash": (
            hashlib.sha256(here.with_name("qmm_layout.py").read_bytes()).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """One entry's translation, with its gate's options and assertions."""
    matrix = signature(entry)["kernel"] not in VECTOR_KERNELS
    path = source(entry)
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
        },
        "opengl": {"software_subgroup_width": SIMD_SIZE},
    }
    metal = {
        "preserve_resource_origins": True,
        "max_template_specializations": 512,
        "max_template_materialization_work": 65536,
    }
    if matrix:
        # BlockMMA's fragments: the cooperative-matrix lowering gives
        # DirectX and OpenGL MLX's simdgroup matrices.
        for name in options:
            options[name]["cooperative_matrix_software_lowering"] = True
        metal["cooperative_matrix_fragment_mapping"] = "tile_4x4_row_pair"
        metal["cooperative_matrix_fragment_mapping_provenance"] = (
            "mlx_steel_BaseMMAFrag_get_coord"
        )
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
        index_range_assertions=(
            gather_qmm_layout.index_assertions(entry) if target == "opengl" else ()
        ),
        metadata_preconditions=[
            dict(
                source=path,
                entryPoint=entry,
                parameter=parameter,
                minimum=0,
                maximum=K_LIMIT,
            )
            for parameter in gather_qmm_layout.bounded_parameters(entry)
        ],
        source_options={"metal": {**metal, "target_options": options}},
    )


class GatherQuantizedPackageCache(TranslatedEntryCache):
    LABEL = LABEL
    PREFIX = "gather-quantized"

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
    return build_record(GatherQuantizedPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        LABEL,
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    from demos.integrations.mlx.portable_host import (
        verify_upstream_gather_matmul_grad,
    )

    return sorted(verify_upstream_gather_matmul_grad.entries())


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
