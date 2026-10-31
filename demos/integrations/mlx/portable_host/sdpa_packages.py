"""Cache translated MLX vector scaled-dot-product-attention entries.

Each host entry is one pinned ``scaled_dot_product_attention.metal`` vector
kernel, ``sdpa_vector``, ``sdpa_vector_2pass_1`` (or its grouped-query
variant) or ``sdpa_vector_2pass_2``, named by
``scaled_dot_product_attention.cpp``'s hash name, with the mask, transposed
query, causal, sink and block function constants that name implies frozen,
at that file's threadgroups (``sdpa_layout``). Every kernel combines each
simdgroup with ``simd_sum`` or ``simd_max``, so Direct3D runs them in exact
32-lane waves and OpenGL in 32-lane software subgroups. OpenGL translation
receives the key and value offset ranges and the key count range the
host's launch bounds prove. Run as a module, it translates entries into a
host package directory's cache ahead of execution, several processes at a
time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import sdpa_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "sdpa"
TARGET_OPTIONS = {"opengl": {"software_subgroup_width": layout.SIMD_SIZE}}


def project_config(root, output_dir, target, entry):
    """One frozen vector attention entry with the host's proven ranges."""
    info = layout.signature(entry)
    source = info["source"]
    opengl = target == "opengl"
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
        specialization_constants=dict(info["constants"]),
        freeze_specialization_constants=bool(info["constants"]),
        subgroup_width_rules=(
            {layout.SOURCE: layout.SIMD_SIZE} if target == "directx" else {}
        ),
        index_range_assertions=layout.opengl_index_assertions(entry) if opengl else (),
        metadata_preconditions=(
            [
                {
                    "source": layout.SOURCE,
                    "entryPoint": source,
                    "parameter": name,
                    "minimum": minimum,
                    "maximum": maximum,
                }
                for name, (minimum, maximum) in layout.parameter_ranges(entry).items()
            ]
            if opengl
            else []
        ),
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "target_options": TARGET_OPTIONS,
            }
        },
    )


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("sdpa_layout.py").read_bytes()).hexdigest()
        ),
    }


class SdpaPackageCache(TranslatedEntryCache):
    LABEL = "SDPA"
    SOURCE = layout.SOURCE
    PREFIX = "sdpa"

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
    cache = SdpaPackageCache(root, Path(packages) / CACHE_DIRECTORY, target)
    return build_record(cache, entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        layout.signature,
        "SDPA",
        root,
        Path(packages).resolve(),
        target,
        entries,
        jobs,
    )


def verification_entries():
    """Every vector kernel shape and function-constant family once: each
    head-dimension pair of ``sdpa_vector`` with each mask kind, transposed
    and causal queries and sinks; the first pass with both mask kinds, sinks
    and causal queries; every grouped-query first pass; and every second
    pass."""
    flags = {
        "has_mask": False,
        "bool_mask": False,
        "float_mask": False,
        "query_transposed": False,
        "do_causal": False,
        "has_sinks": False,
    }
    boolean = {**flags, **layout.MASKS["boolmask"], "do_causal": True}
    additive = {**flags, **layout.MASKS["floatmask"], "query_transposed": True}
    sinks = {**flags, "has_sinks": True}
    entries = [
        layout.vector_entry("float32", d, v, flags)
        for d, v in sorted(layout.VECTOR_DIMS)
    ]
    entries += [
        layout.vector_entry(dtype, 128, 128, constants)
        for dtype in layout.TYPE_NAMES
        for constants in (boolean, additive, sinks)
    ]
    entries += [
        layout.pass1_entry("float16", 128, 128, flags, 64, 4, 1),
        layout.pass1_entry("float32", 64, 64, boolean, 32, 4, 2),
        layout.pass1_entry(
            "bfloat16", 128, 128, {**additive, "has_sinks": True}, 32, 2, 3
        ),
    ]
    entries += [
        layout.pass1_entry("float32", d, d, flags, 32, gqa, 1, gqa_kernel=True)
        for gqa, d in sorted(layout.GQA_HEADS)
    ]
    entries += [
        layout.pass2_entry("float32", v) for v in sorted(layout.AGGREGATION_DIMS)
    ]
    return entries


def main(argv=None):
    """Translate vector attention entries into a host package directory's cache."""
    argv = list(sys.argv[1:] if argv is None else argv)
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
