"""Cache translated floating-point matrix-vector entries of the pinned gemv.metal.

Run as a module, it translates entries into a host package directory's cache
ahead of execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.gemv_layout import (
    SIGNED_MAX,
    SIMD_SIZE,
    SOURCE,
    index_assertions,
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
CACHE_DIRECTORY = "gemv"


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("gemv_layout.py").read_bytes()).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """The translation of one entry, with the current-tree gemv gate's options."""
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: workgroup_size(entry)}},
        # gemv_layout.plan bounds every bound buffer at MAX_ELEMENTS elements,
        # so each asserted offset stays below 2^31 - 1.
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=SIGNED_MAX)
            for expression in (index_assertions(entry) if target == "opengl" else ())
        ],
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


class GemvPackageCache(TranslatedEntryCache):
    LABEL = "Matrix-vector"
    SOURCE = SOURCE
    PREFIX = "gemv"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(GemvPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        "Matrix-vector",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    from demos.integrations.mlx.portable_host import verify_matmul

    records = (verify_matmul.case_record(case) for case in verify_matmul.CASES)
    return [record["entry"] for record in records if record["family"] == "gemv"]


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
