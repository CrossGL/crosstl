"""Cache translated MLX steel full-attention entries.

Each host entry is the pinned ``steel_attention.metal`` ``attention`` kernel
for one ``sdpa_full_self_attention_metal`` hash name, with the alignment,
mask, causal, sink and logsumexp function constants that name implies frozen,
at the ``(32, 4, WN)`` threadgroup it launches (``steel_attention_layout``).
The kernel multiplies its tiles through ``BaseMMAFrag``'s 8x8 simdgroup
matrices, whose lanes ``get_coord`` maps as the steel GEMM's do, so Direct3D
and OpenGL lower them in software over exact 32-lane subgroups. OpenGL
translation receives the threadgroup array extents and the device offset
bound ``plan_launch`` proves. Run as a module, it translates entries into a
host package directory's cache ahead of execution, several processes at a
time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import steel_attention_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "steel-attention"
SOFTWARE_MATRICES = {
    "cooperative_matrix_software_lowering": True,
    "software_subgroup_width": layout.SIMD_SIZE,
}
TARGET_OPTIONS = {
    "directx": {**SOFTWARE_MATRICES, "relative_wave_shuffle_out_of_range": "self"},
    "opengl": SOFTWARE_MATRICES,
}


def project_config(root, output_dir, target, entry):
    """One frozen steel attention entry with the host's proven ranges."""
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
        specialization_constants=dict(info["constants"]),
        freeze_specialization_constants=True,
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
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "target_options": TARGET_OPTIONS,
            }
        },
    )


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("steel_attention_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


class SteelAttentionPackageCache(TranslatedEntryCache):
    LABEL = "Steel attention"
    SOURCE = layout.SOURCE
    PREFIX = "steel-attention"

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
    cache = SteelAttentionPackageCache(root, Path(packages) / CACHE_DIRECTORY, target)
    return build_record(cache, entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        layout.signature,
        "Steel attention",
        root,
        Path(packages).resolve(),
        target,
        entries,
        jobs,
    )


def verification_entries():
    """Every head-dim tile the host selects once with no flags, in float32
    except head dim 192, whose float32 tiles need 40,448 bytes of threadgroup
    memory, past the 32 KiB that Direct3D 12 and Apple GPUs provide
    (scaled_dot_product_attention.cpp's use_fallback fuses head dim 192 only
    when forced); the bfloat16 tile of head dim 128; and each function
    constant, with both mask kinds, on head dim 64."""
    none = dict.fromkeys(layout.FLAGS, False)
    every = dict.fromkeys(layout.FLAGS, True)
    tiles = ((64, 64), (72, 72), (80, 80), (96, 96), (96, 64), (128, 128), (256, 256))
    entries = [
        layout.entry_name("float32", bd, bv, "float32", none) for bd, bv in tiles
    ]
    entries += [
        layout.entry_name(dtype, bd, bd, dtype, none)
        for dtype, bd in (("float16", 192), ("bfloat16", 128))
    ]
    entries += [
        layout.entry_name("float32", 64, 64, "bool_", every),
        layout.entry_name(
            "float16", 64, 64, "float16", {**every, "align_Q": False, "save_lse": False}
        ),
        layout.entry_name(
            "bfloat16", 64, 64, "bool_", {**none, "align_K": True, "has_mask": True}
        ),
    ]
    return entries


def main(argv=None):
    """Translate steel attention entries into a host package directory's cache."""
    argv = list(sys.argv[1:] if argv is None else argv)
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
