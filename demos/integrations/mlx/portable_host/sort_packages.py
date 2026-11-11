"""Cache translated merge-sort kernels of the pinned sort.metal.

Each host variant is one single-block or multi-block stage entry at the
threadgroup width ``multi_block_sort`` or ``single_block_sort`` launches it
with, keyed ``<entry>@<width>``. The kernels use no simdgroup operations, so
every target runs them at that width. OpenGL is translated with the host's
preconditions: the strided kernels' 64-bit offsets stay in signed 32-bit
indices, and each threadgroup view a merge helper receives stays inside its
``bn * 4`` tile. Run as a module, it translates variants into a host package
directory's cache ahead of execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)
from demos.integrations.mlx.portable_host.sort_layout import (
    N_PER_THREAD,
    SIGNED_MAX,
    SOURCE,
    index_assertions,
    signature,
    split_variant,
    workgroup_views,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "sort"


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("sort_layout.py").read_bytes()).hexdigest()
        ),
    }


def opengl_assertions(entry):
    """The workgroup-view and index-range preconditions OpenGL is built with."""
    tile = signature(entry)[4] * N_PER_THREAD
    views = [
        dict(
            source=SOURCE,
            entry_point=entry,
            function=function,
            parameter=parameter,
            minimum=0,
            maximum=tile - 1,
        )
        for function, parameters in workgroup_views(entry)
        for parameter in parameters
    ]
    # sort_layout.plan bounds every strided span by 2^28 - 1 elements.
    indices = [
        dict(source=SOURCE, expression=expression, minimum=0, maximum=SIGNED_MAX)
        for expression in index_assertions(entry)
    ]
    return views, indices


def project_config(root, output_dir, target, key):
    """One variant's translation at the threadgroup width its key names."""
    entry, workgroup = split_variant(key)
    views, indices = opengl_assertions(entry) if target == "opengl" else ((), ())
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: workgroup}},
        workgroup_access_assertions=list(views),
        index_range_assertions=list(indices),
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 16384,
            }
        },
    )


class SortPackageCache(TranslatedEntryCache):
    LABEL = "Sort"
    SOURCE = SOURCE
    PREFIX = "sort"

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
    return build_record(SortPackageCache(root, directory, target), key)


def build_entries(root, packages, target, keys, jobs):
    """Translate variant ``keys`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        split_variant,
        "Sort",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        keys,
        jobs,
    )


# The (value type, sorted axis length) pairs the unchanged upstream sort,
# argsort, partition and argpartition tests send to the primitives:
# ``test_sort`` sorts every axis and the flattened array of a (6, 4, 10)
# array and of its (3, 4, 5) strided view, 32769-element rows and their
# 10923-element strided views, broadcast (16, 8) rows, a 2^22-element row,
# a (1, 4, 8, 1) array, 512 rows of 128 and reversed (3, 4) and (3, 32769)
# views; ``test_partition`` and ``test_argpartition`` take (3, 4, 5) and
# (2, 3) arrays; ``test_sort_nan`` and ``test_argsort_nan`` four values.
VERIFICATION_AXES = (
    ("int32", (240, 60, 6, 4, 10, 3, 5, 2, 8, 32769)),
    ("float32", (240, 60, 6, 4, 10, 3, 5, 8, 128, 32769, 10923, 2**22)),
    ("complex64", (240, 60, 6, 4, 10, 3, 5)),
    ("int64", (16, 8)),
    ("float16", (4,)),
    ("bfloat16", (4,)),
)


def verification_entries():
    """The variants the unchanged upstream sort tests launch."""
    from demos.integrations.mlx.portable_host.sort_layout import (
        STAGES,
        entry_name,
        launch,
        variant,
        width,
    )

    keys = set()
    for dtype, sizes in VERIFICATION_AXES:
        for size in sizes:
            _, n_blocks = launch(size, dtype)
            if n_blocks == 1:
                entries = [
                    entry_name("block", dtype, size, argsort, contiguous)
                    for argsort in (False, True)
                    for contiguous in (False, True)
                ]
            else:
                entries = [entry_name(stage, dtype, size) for stage in STAGES]
            keys.update(variant(entry, width(entry, n_blocks)) for entry in entries)
    return sorted(keys)


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
