"""Cache translated affine gathered quantized entries of the unchanged pinned source.

The matrix-matrix entries are translated with the qmm entries' options
(``qmm_packages``) and their batched siblings' tile assertions, as the
current-tree gather_qmm gate translates them. The matrix-vector entries take
the qmv gates' options with ``gather_qmm_layout``'s OpenGL offset ranges. Run
as a module, it translates entries into a host package directory's cache
ahead of execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import gather_qmm_layout, qmm_packages
from demos.integrations.mlx.portable_host.gather_qmm_layout import (
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
CACHE_DIRECTORY = "gather-quantized"
LABEL = "Gathered quantized product"
# The modules whose options and assertions each translation depends on.
RECIPE_MODULES = (
    "gather_qmm_packages.py",
    "gather_qmm_layout.py",
    "qmm_packages.py",
    "qmm_layout.py",
    "matmul_layout.py",
)


def _recipe_hashes():
    here = Path(__file__)
    return {
        name.removesuffix(".py"): (
            hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
        )
        for name in RECIPE_MODULES
    }


def project_config(root, output_dir, target, entry, layout=gather_qmm_layout):
    """One entry's translation, with the current-tree gather gates' options.

    ``layout`` is ``gather_qmm_layout`` or ``fp_gather_qmm_layout``, whose
    kernels share these options: it names the source, workgroup and
    assertions.
    """
    if layout.is_matrix(entry):
        return qmm_packages.project_config(root, output_dir, target, entry, layout)
    # simd_sum: software subgroups give DirectX and OpenGL MLX's 32 lanes.
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
        include_patterns=(layout.SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={layout.SOURCE: (entry,)},
        entry_workgroup_size_rules={
            layout.SOURCE: {entry: layout.workgroup_size(entry)}
        },
        index_range_assertions=(
            list(layout.index_assertions(entry)) if target == "opengl" else []
        ),
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                "target_options": options,
            }
        },
    )


class GatherQmmPackageCache(TranslatedEntryCache):
    LABEL = LABEL
    SOURCE = SOURCE
    PREFIX = "gather-quantized"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(GatherQmmPackageCache(root, directory, target), entry)


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
    from demos.integrations.mlx.portable_host import verify_gather_qmm

    return sorted(
        {
            verify_gather_qmm.case_record(case)["entry"]
            for case in verify_gather_qmm.CASES
        }
    )


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
