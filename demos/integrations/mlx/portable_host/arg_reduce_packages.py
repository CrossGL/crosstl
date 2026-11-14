"""Cache translated ``arg_reduce_general`` entries of the pinned arg_reduce.metal.

Each host variant is one ``argmin_<type>`` or ``argmax_<type>`` entry at one
threadgroup width, keyed ``<entry>@<width>``. The kernel reduces each simdgroup
with ``simd_shuffle_down`` and then returns every simdgroup but the first
before its final shuffles. Direct3D and OpenGL run it in 32-lane software
subgroups, whose early exits converge before the final shuffles, so the
surviving subgroup holds thread 0 as on Metal. Direct3D does not use hardware
waves here: WARP, which runs the Windows legs, corrupts the low lanes of the
``WaveReadLaneAt`` shuffles an exact 32-lane wave lowers them to
(``tests/test_translator/test_multiwave_index_ties.py``). Run as a module, it
translates variants into a host package directory's cache ahead of execution,
several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.arg_reduce_layout import (
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
    "directx": {
        "software_subgroup_width": SIMD_SIZE,
        "relative_wave_shuffle_out_of_range": "self",
    },
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


def verification_entries():
    """The variants the unchanged upstream arg reduction tests launch.

    ``test_reduce.test_arg_reduce`` and ``test_ops.test_argmin_argmax`` reduce
    each axis of a ``(10, 12, 13)`` array (32 lanes) and the flattened array
    of 1560 elements (416 lanes) for every real dtype but Boolean;
    ``test_bf16`` adds bfloat16. Every target runs both widths.
    """
    from demos.integrations.mlx.portable_host.arg_reduce_layout import (
        ENTRIES,
        variant,
        width,
    )

    widths = sorted({width(size) for size in (10, 12, 13, 1560)})
    return [
        variant(entry, size)
        for entry, (_, dtype) in ENTRIES.items()
        if dtype != "bool_"
        for size in widths
    ]


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
