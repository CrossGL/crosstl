"""Cache translated ``col_reduce_small`` entries of the pinned reduce.metal.

Each host variant is one entry at one threadgroup size, keyed
``<entry>@<lanes>x<rows>``. Run as a module, it translates variants into a
host package directory's cache ahead of execution, several processes at a
time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.column_small_layout import split_variant
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)
from demos.integrations.mlx.portable_host.reduction_packages import SOURCE

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "column-small"


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("column_small_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


def project_config(root, output_dir, target, key):
    """One variant's translation; the kernel uses no SIMD-group operations."""
    entry, workgroup = split_variant(key)
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: workgroup}},
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 128,
                "max_template_materialization_work": 8192,
            }
        },
    )


class SmallColumnPackageCache(TranslatedEntryCache):
    LABEL = "Small column reduction"
    SOURCE = SOURCE
    PREFIX = "column-small"

    def signature(self, key):
        return split_variant(key)

    def workgroup_size(self, key):
        return split_variant(key)[1]

    def source_entry(self, key):
        return split_variant(key)[0]

    def project_config(self, output_dir, key):
        return project_config(self.root, output_dir, self.target, key)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, key):
    return build_record(SmallColumnPackageCache(root, directory, target), key)


def build_entries(root, packages, target, keys, jobs):
    """Translate variant ``keys`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        split_variant,
        "Small column reduction",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        keys,
        jobs,
    )


def verification_entries():
    """The variants the column and quantized product verifications launch."""
    from demos.integrations.mlx.portable_host import (
        column_workloads,
        verify_quantized_matmul,
        verify_quantized_qmm,
    )

    return sorted(
        column_workloads.small_variants()
        | verify_quantized_matmul.small_column_variants()
        | verify_quantized_qmm.small_column_variants()
    )


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
