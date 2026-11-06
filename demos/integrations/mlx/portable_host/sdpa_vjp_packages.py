"""Cache translated MLX fused attention VJP entries.

Each host entry is one pinned ``scaled_dot_product_attention_vjp.metal``
kernel, ``sdpa_vjp_odo``, ``sdpa_vjp_ds`` or ``sdpa_vjp_reduce_add`` /
``sdpa_vjp_reduce_set``, named as ``scaled_dot_product_attention.cpp`` names
its pipeline, at that file's threadgroups (``sdpa_vjp_layout``).
``sdpa_vjp_odo`` sums each row with ``simd_sum``, so Direct3D runs it in
exact 32-lane waves and OpenGL in 32-lane software subgroups. Run as a
module, it translates entries into a host package directory's cache ahead of
execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import sdpa_vjp_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "sdpa-vjp"
TARGET_OPTIONS = {"opengl": {"software_subgroup_width": layout.SIMD_SIZE}}


def project_config(root, output_dir, target, entry):
    """One attention VJP entry with the host's proven offsets."""
    info = layout.signature(entry)
    source = info["source"]
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(layout.SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={layout.SOURCE: (source,)},
        entry_workgroup_size_rules={
            layout.SOURCE: {source: list(info["workgroupSize"])}
        },
        subgroup_width_rules=(
            {layout.SOURCE: layout.SIMD_SIZE} if target == "directx" else {}
        ),
        index_range_assertions=(
            layout.opengl_index_assertions(entry) if target == "opengl" else ()
        ),
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                # Only the row dot product has subgroup operations.
                "target_options": TARGET_OPTIONS if info["kind"] == "odo" else {},
            }
        },
    )


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("sdpa_vjp_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


class SdpaVjpPackageCache(TranslatedEntryCache):
    LABEL = "SDPA VJP"
    SOURCE = layout.SOURCE
    PREFIX = "sdpa-vjp"

    def signature(self, entry):
        return layout.signature(entry)

    def source_entry(self, entry):
        return layout.source_entry(entry)

    def workgroup_size(self, entry):
        return layout.workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, packages, target, entry):
    cache = SdpaVjpPackageCache(root, Path(packages) / CACHE_DIRECTORY, target)
    return build_record(cache, entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        layout.signature,
        "SDPA VJP",
        root,
        Path(packages).resolve(),
        target,
        entries,
        jobs,
    )


def verification_entries():
    """Every row dot product head dim once and every type of the others."""
    entries = [layout.odo_entry("float32", dv) for dv in sorted(layout.ODO_DIMS)]
    entries += [layout.odo_entry(dtype, 64) for dtype in ("float16", "bfloat16")]
    entries += [layout.ds_entry(dtype) for dtype in layout.TYPE_NAMES]
    entries += [
        layout.reduce_entry(dtype, accumulate)
        for dtype in layout.TYPE_NAMES
        for accumulate in (False, True)
    ]
    return entries


def main(argv=None):
    """Translate attention VJP entries into a host package directory's cache."""
    argv = list(sys.argv[1:] if argv is None else argv)
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
