"""Cache translated softmax and logsumexp row kernels of the pinned sources.

Each host variant is one ``block_*`` or ``looped_*`` entry of softmax.metal or
logsumexp.metal at one threadgroup width, keyed ``<entry>@<width>``. The
kernels combine each simdgroup with ``simd_max`` and ``simd_sum`` and every
simdgroup reaches every threadgroup barrier, so Direct3D runs them in exact
32-lane waves (``WaveSize(32)``) and OpenGL in 32-lane software subgroups at
every width. Run as a module, it translates variants into a host package
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
from demos.integrations.mlx.portable_host.softmax_layout import (
    SIGNED_MAX,
    SIMD_SIZE,
    SOURCES,
    index_assertions,
    signature,
    split_variant,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "softmax"
TARGET_OPTIONS = {"opengl": {"software_subgroup_width": SIMD_SIZE}}


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("softmax_layout.py").read_bytes()).hexdigest()
        ),
    }


def project_config(root, output_dir, target, key):
    """One variant's translation at the threadgroup width its key names."""
    entry, workgroup = split_variant(key)
    source = SOURCES[signature(entry)[0]]
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (entry,)},
        entry_workgroup_size_rules={source: {entry: workgroup}},
        subgroup_width_rules={source: SIMD_SIZE} if target == "directx" else {},
        # softmax_layout.plan refuses a launch whose furthest row offset,
        # padded reads included, exceeds 2^31 - 1.
        index_range_assertions=[
            dict(source=source, expression=expression, minimum=0, maximum=SIGNED_MAX)
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


class _RowKernelCache(TranslatedEntryCache):
    OPERATION = None

    def signature(self, key):
        entry, workgroup = split_variant(key)
        if signature(entry)[0] != self.OPERATION:
            raise ValueError(f"{self.LABEL} cache received another kernel")
        return entry, workgroup

    def workgroup_size(self, key):
        return split_variant(key)[1]

    def source_entry(self, key):
        return split_variant(key)[0]

    def project_config(self, output_dir, key):
        return project_config(self.root, output_dir, self.target, key)

    def recipe_hashes(self):
        return _recipe_hashes()


class _SoftmaxCache(_RowKernelCache):
    LABEL = "Softmax"
    SOURCE = SOURCES["softmax"]
    PREFIX = "softmax"
    OPERATION = "softmax"


class _LogSumExpCache(_RowKernelCache):
    LABEL = "Logsumexp"
    SOURCE = SOURCES["logsumexp"]
    PREFIX = "logsumexp"
    OPERATION = "logsumexp"


class SoftmaxPackageCache:
    """Both sources' variants in one directory; identities name the source."""

    def __init__(self, root, directory, target):
        self.caches = {
            cache.OPERATION: cache(root, directory, target)
            for cache in (_SoftmaxCache, _LogSumExpCache)
        }
        self.target = target

    def get(self, key):
        entry, _ = split_variant(key)
        return self.caches[signature(entry)[0]].get(key)


def _build_one(root, directory, target, key):
    return build_record(SoftmaxPackageCache(root, directory, target), key)


def build_entries(root, packages, target, keys, jobs):
    """Translate variant ``keys`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        split_variant,
        "Softmax and logsumexp",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        keys,
        jobs,
    )


def verification_entries():
    """The variants the unchanged upstream softmax and logsumexp tests launch."""
    from demos.integrations.mlx.portable_host.softmax_layout import (
        entry_name,
        variant,
        width,
    )

    keys = set()
    for operation, dtype, precise, sizes in VERIFICATION_ROWS:
        for size in sizes:
            entry = entry_name(operation, dtype, size, precise)
            keys.add(variant(entry, width(entry, size)))
    return sorted(keys)


# Row lengths the unchanged upstream tests send to the primitives; the other
# axis groups take the composed max, exp and sum path. ``test_softmax`` reduces
# the last axis of a (16, 8, 32) array, the long -inf rows, a sliced (8, 2)
# view, the precise 16-bit rows and their float32 reference, and the all -inf
# rows; ``test_logsumexp`` and ``test_logsumexp_shape`` reduce a 1025-element
# row and arrays whose leading axes are all one.
VERIFICATION_ROWS = (
    ("softmax", "float32", False, (32, 100, 2049, 4097, 8193, 2, 1024, 127, 128, 129)),
    ("softmax", "float16", False, (32,)),
    ("softmax", "float16", True, (1024,)),
    ("softmax", "bfloat16", True, (1024,)),
    ("logsumexp", "float32", False, (1025, 1, 2, 4, 5, 8)),
)


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
