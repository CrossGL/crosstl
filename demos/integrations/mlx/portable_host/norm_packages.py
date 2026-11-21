"""Cache translated RMS and layer normalization kernels of the pinned sources.

Each host variant is one forward or gradient entry of rms_norm.metal or
layer_norm.metal at one threadgroup width, keyed ``<entry>@<width>``; a
gradient entry also freezes the ``has_w`` function constant its hash name
selects. The kernels combine each simdgroup with ``simd_sum`` and every
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
from demos.integrations.mlx.portable_host.norm_layout import (
    SIGNED_MAX,
    SIMD_SIZE,
    SOURCES,
    constants,
    entry_name,
    index_assertions,
    signature,
    source_entry,
    split_variant,
    variant,
    width,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "normalization"
TARGET_OPTIONS = {"opengl": {"software_subgroup_width": SIMD_SIZE}}


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("norm_layout.py").read_bytes()).hexdigest()
        ),
    }


def project_config(root, output_dir, target, key):
    """One variant's translation at the threadgroup width its key names."""
    entry, workgroup = split_variant(key)
    source = SOURCES[signature(entry)[0]]
    kernel = source_entry(entry)
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (kernel,)},
        entry_workgroup_size_rules={source: {kernel: workgroup}},
        specialization_constants=constants(entry),
        freeze_specialization_constants=bool(constants(entry)),
        subgroup_width_rules={source: SIMD_SIZE} if target == "directx" else {},
        # norm_layout.plan refuses a launch whose furthest row, weight or bias
        # offset, padded reads included, exceeds 2^31 - 1.
        index_range_assertions=[
            dict(source=source, expression=expression, minimum=0, maximum=SIGNED_MAX)
            for expression in (index_assertions(entry) if target == "opengl" else ())
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


class _NormKernelCache(TranslatedEntryCache):
    OPERATION = None

    def signature(self, key):
        entry, workgroup = split_variant(key)
        if signature(entry)[0] != self.OPERATION:
            raise ValueError(f"{self.LABEL} cache received another kernel")
        return entry, workgroup

    def workgroup_size(self, key):
        return split_variant(key)[1]

    def source_entry(self, key):
        return source_entry(split_variant(key)[0])

    def project_config(self, output_dir, key):
        return project_config(self.root, output_dir, self.target, key)

    def recipe_hashes(self):
        return _recipe_hashes()


class _RMSNormCache(_NormKernelCache):
    LABEL = "RMS normalization"
    SOURCE = SOURCES["rms"]
    PREFIX = "rms"
    OPERATION = "rms"


class _LayerNormCache(_NormKernelCache):
    LABEL = "Layer normalization"
    SOURCE = SOURCES["layer_norm"]
    PREFIX = "layer-norm"
    OPERATION = "layer_norm"


class NormPackageCache:
    """Both sources' variants in one directory; identities name the source."""

    def __init__(self, root, directory, target):
        self.caches = {
            cache.OPERATION: cache(root, directory, target)
            for cache in (_RMSNormCache, _LayerNormCache)
        }
        self.target = target

    def get(self, key):
        entry, _ = split_variant(key)
        return self.caches[signature(entry)[0]].get(key)


def _build_one(root, directory, target, key):
    return build_record(NormPackageCache(root, directory, target), key)


def build_entries(root, packages, target, keys, jobs):
    """Translate variant ``keys`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        split_variant,
        "Normalization",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        keys,
        jobs,
    )


def verification_entries():
    """The variants the unchanged upstream normalization tests launch."""
    keys = set()
    for operation, gradient, dtypes, sizes in VERIFICATION_ROWS:
        for dtype in dtypes:
            for size in sizes:
                for has_w in (True, False) if gradient else (None,):
                    entry = entry_name(operation, gradient, dtype, size, has_w)
                    keys.add(variant(entry, width(entry, size)))
    return sorted(keys)


# Row lengths the unchanged upstream tests send to the primitives.
# ``test_rms_norm`` normalizes rows of 31, 32, 33, 256, 512 and 4099 values
# and 32-value rows of every dtype; ``test_rms_norm_grad`` differentiates rows
# of 32, 256 and 8192 values with and without a weight. ``test_layer_norm``
# normalizes rows of 31, 32, 33 and 4099 values and 32-value rows of every
# dtype, ``test_slice_into_layer_norm`` 128-value rows, and the gradient tests
# differentiate rows of 8, 32 and 8192 values with and without a weight.
VERIFICATION_ROWS = (
    ("rms", False, ("float32",), (31, 32, 33, 256, 512, 4099, 8192)),
    ("rms", False, ("float16", "bfloat16"), (32,)),
    ("rms", True, ("float32",), (32, 256, 8192)),
    ("layer_norm", False, ("float32",), (8, 31, 32, 33, 128, 4099, 8192)),
    ("layer_norm", False, ("float16", "bfloat16"), (32,)),
    ("layer_norm", True, ("float32",), (8, 32, 8192)),
)


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
