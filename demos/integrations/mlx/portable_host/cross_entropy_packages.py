"""Cache translated fused cross-entropy kernels of the pinned source.

Each host variant is one ``cross_entropy_<type>`` or ``cross_entropy_vjp_<type>``
entry of cross_entropy.metal at one threadgroup width, keyed
``<entry>@<width>``. The forward kernel combines each simdgroup with
``simd_max`` and ``simd_sum`` and every simdgroup reaches every threadgroup
barrier, so Direct3D runs it in exact 32-lane waves (``WaveSize(32)``) and
OpenGL in 32-lane software subgroups at every width; the gradient kernel uses
no subgroup operation. Run as a module, it translates variants into a host
package directory's cache ahead of execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.cross_entropy_layout import (
    SIGNED_MAX,
    SIMD_SIZE,
    SOURCE,
    entry_name,
    index_assertions,
    signature,
    split_variant,
    variant,
    width,
)
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "cross-entropy"
TARGET_OPTIONS = {"opengl": {"software_subgroup_width": SIMD_SIZE}}


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("cross_entropy_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


def project_config(root, output_dir, target, key):
    """One variant's translation at the threadgroup width its key names."""
    entry, workgroup = split_variant(key)
    gradient, _ = signature(entry)
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: workgroup}},
        subgroup_width_rules=(
            {SOURCE: SIMD_SIZE} if target == "directx" and not gradient else {}
        ),
        # cross_entropy_layout.plan refuses a launch whose furthest row
        # offset, padded reads included, exceeds 2^31 - 1, and the dispatch
        # refuses class indices outside their rows.
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=SIGNED_MAX)
            for expression in (index_assertions(entry) if target == "opengl" else ())
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 128,
                "max_template_materialization_work": 8192,
                "target_options": {} if gradient else TARGET_OPTIONS,
            }
        },
    )


class CrossEntropyPackageCache(TranslatedEntryCache):
    LABEL = "Cross-entropy"
    SOURCE = SOURCE
    PREFIX = "cross-entropy"

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
    return build_record(CrossEntropyPackageCache(root, directory, target), key)


def build_entries(root, packages, target, keys, jobs):
    """Translate variant ``keys`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        split_variant,
        "Cross-entropy",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        keys,
        jobs,
    )


def verification_entries():
    """The variants the unchanged upstream cross-entropy tests launch."""
    return sorted(
        {
            variant(entry_name(gradient, dtype), width(size))
            for gradient, dtypes, sizes in VERIFICATION_ROWS
            for dtype in dtypes
            for size in sizes
        }
    )


# Row lengths the unchanged upstream tests send to the primitives.
# ``test_fast.test_cross_entropy`` reduces rows of 7 to 8192 logits of every
# dtype, ``test_cross_entropy_grad`` differentiates float32 rows of 7, 128,
# 512, 1000 and 4096 logits, and ``test_losses.test_cross_entropy`` reduces
# float32 rows of two logits.
VERIFICATION_ROWS = (
    (
        False,
        ("float32", "float16", "bfloat16"),
        (7, 32, 128, 255, 256, 1000, 4096, 8192),
    ),
    (False, ("float32",), (2, 512)),
    (True, ("float32",), (7, 128, 512, 1000, 4096)),
)


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
