"""Cache translated floating-point quantized matrix-vector entries.

Each entry of the unchanged pinned fp_quantized.metal is translated with the
options of the affine qmv entries (matmul_packages). Run as a module, it
translates entries into a host package directory's cache ahead of execution,
several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.fp_matmul_layout import (
    MAX_ELEMENTS,
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
CACHE_DIRECTORY = "fp-quantized-matmul"
# Expressions OpenGL may evaluate in 32 bits: fp_quantized.h
# adjust_matrix_offsets' batch offsets. fp_matmul_layout.plan proves each
# names an element of a buffer of at most MAX_ELEMENTS elements; unbatched
# kernels never compute them.
BATCH_INDEX_EXPRESSIONS = (
    "x_idx * x_strides[0]",
    "w_idx * w_strides[0]",
    "w_idx * s_strides[0]",
    "idx.x",
    "idx.y",
)


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("fp_matmul_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """The translation of one entry, with the affine qmv entries' options."""
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
        },
        "opengl": {"software_subgroup_width": SIMD_SIZE},
    }
    batched = signature(entry)["batched"]
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: workgroup_size(entry)}},
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=MAX_ELEMENTS)
            for expression in BATCH_INDEX_EXPRESSIONS
            if target == "opengl" and batched
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                "target_options": options,
            }
        },
    )


class FpQuantizedMatmulPackageCache(TranslatedEntryCache):
    LABEL = "Floating-point quantized matrix-vector"
    SOURCE = SOURCE
    PREFIX = "fp-quantized-matmul"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(FpQuantizedMatmulPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        "Floating-point quantized matrix-vector",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The block-scaled entries the upstream runners route, sorted, once each.

    test_fp_qmv, test_fp_qmv_large_output and test_fp_qvm route only
    block-scaled products; test_qmv_small_non_multiples routes its nvfp4 ones
    beside affine products.
    """
    from demos.integrations.mlx.portable_host import (
        verify_upstream_fp_qmv,
        verify_upstream_fp_qmv_large_output,
        verify_upstream_fp_qvm,
        verify_upstream_qmv_small_non_multiples,
    )

    return sorted(
        {
            *verify_upstream_fp_qmv.package_entries()["quantized"],
            *verify_upstream_fp_qmv_large_output.package_entries()["quantized"],
            *verify_upstream_fp_qvm.package_entries()["quantized"],
            *verify_upstream_qmv_small_non_multiples.package_entries()["fp"],
        }
    )


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
