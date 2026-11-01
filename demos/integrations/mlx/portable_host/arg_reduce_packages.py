"""Cache translated ``arg_reduce_general`` entries of the pinned arg_reduce.metal.

Each host variant is one ``argmin_<type>`` or ``argmax_<type>`` entry at one
threadgroup width, keyed ``<entry>@<width>``. The kernel reduces each simdgroup
with ``simd_shuffle_down`` and then returns every simdgroup but the first
before its final shuffles. Direct3D runs it in exact 32-lane waves
(``WaveSize(32)``), which need no workgroup barrier for that exit; OpenGL runs
32-lane software subgroups, which synchronize the whole workgroup, so its
host refuses widths above one simdgroup (``arg_reduce_dispatch``). Run as a
module, it translates variants into a host package directory's cache ahead
of execution, several processes at a time.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.arg_reduce_layout import (
    MAX_WIDTH,
    SIGNED_MAX,
    SIMD_SIZE,
    SOURCE,
    index_assertions,
    split_variant,
)
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "arg-reduce"
TARGET_OPTIONS = {
    "directx": {"relative_wave_shuffle_out_of_range": "self"},
    "opengl": {"software_subgroup_width": SIMD_SIZE},
}


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("arg_reduce_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


def project_config(root, output_dir, target, key):
    """One variant's translation at the threadgroup width its key names."""
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
        subgroup_width_rules={SOURCE: SIMD_SIZE} if target == "directx" else {},
        # arg_reduce_layout.plan refuses a launch whose furthest source
        # offset, padded reads included, exceeds 2^31 - 1.
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=SIGNED_MAX)
            for expression in (index_assertions() if target == "opengl" else ())
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 128,
                "max_template_materialization_work": 8192,
                "target_options": TARGET_OPTIONS,
            }
        },
    )


class ArgReducePackageCache(TranslatedEntryCache):
    LABEL = "Arg reduction"
    SOURCE = SOURCE
    PREFIX = "arg-reduce"

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
    return build_record(ArgReducePackageCache(root, directory, target), key)


def build_entries(root, packages, target, keys, jobs):
    """Translate variant ``keys`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        split_variant,
        "Arg reduction",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        keys,
        jobs,
    )


def verification_entries(target=None):
    """The variants the unchanged upstream arg reduction tests launch.

    ``test_reduce.test_arg_reduce`` and ``test_ops.test_argmin_argmax`` reduce
    each axis of a ``(10, 12, 13)`` array (32 lanes) and the flattened array
    of 1560 elements (416 lanes) for every real dtype but Boolean;
    ``test_bf16`` adds bfloat16. A target with a width limit translates only
    the variants its host runs.
    """
    from demos.integrations.mlx.portable_host.arg_reduce_layout import (
        ENTRIES,
        TARGET_WIDTH_LIMITS,
        variant,
        width,
    )

    limit = TARGET_WIDTH_LIMITS.get(target, MAX_WIDTH)
    widths = sorted(
        {width(size) for size in (10, 12, 13, 1560)} & set(range(limit + 1))
    )
    return [
        variant(entry, size)
        for entry, (_, dtype) in ENTRIES.items()
        if dtype != "bool_"
        for size in widths
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target")
    target = parser.parse_known_args(argv)[0].target
    return cli(argv, __doc__, build_entries, lambda: verification_entries(target))


if __name__ == "__main__":
    sys.exit(main())
