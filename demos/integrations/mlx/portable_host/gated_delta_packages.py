"""Cache translated MLX gated-delta kernels of the pinned sources.

Each host entry is one pipeline ``gated_delta_update.cpp`` builds: the
sequential forward with ``save_state`` cleared or, as ``<name>_save``, set,
the eight-step chunk kernel and the sequential backward
(``gated_delta_layout``). Each translates with the options of its
current-tree kernel gate (``test_current_gated_delta`` and
``test_current_gated_delta_vjp``) and with function constant 200 frozen to
the pipeline's value. OpenGL and DirectX run the SIMD-group sums in 32-lane
software subgroups and the chunk kernel's 8 x 8 tiles in software, through
MLX's own lane coordinates. Run as a module, it translates entries into a
host package directory's cache ahead of execution, several processes at a
time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import gated_delta_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "gated-delta"
SOFTWARE_SUBGROUP_TARGETS = frozenset({"opengl", "directx"})
FRAGMENT_MAPPING = "tile_4x4_row_pair"
FRAGMENT_MAPPING_PROVENANCE = "mlx_gated_delta_lane_coordinates"


def _target_options(target, kind):
    if target not in SOFTWARE_SUBGROUP_TARGETS:
        return {}
    options = {"software_subgroup_width": layout.SIMD_SIZE}
    if kind == "fused":
        options["cooperative_matrix_software_lowering"] = True
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    return {target: options}


def project_config(root, output_dir, target, entry):
    """One pipeline with its kernel gate's options and frozen constants."""
    info = layout.signature(entry)
    source, kernel = info["source"], info["kernel"]
    metal = {
        "preserve_resource_origins": True,
        "max_template_specializations": 256,
        "max_template_materialization_work": 16384,
        "target_options": _target_options(target, info["kind"]),
    }
    if info["kind"] == "fused":
        # The chunk kernel's tiles move between buffers and lanes through
        # MLX's own (fm, fn) lane coordinates.
        metal["cooperative_matrix_fragment_mapping"] = FRAGMENT_MAPPING
        metal["cooperative_matrix_fragment_mapping_provenance"] = (
            FRAGMENT_MAPPING_PROVENANCE
        )
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (kernel,)},
        specialization_constants=layout.constants(entry),
        freeze_specialization_constants=True,
        entry_workgroup_size_rules={source: {kernel: list(layout.WORKGROUP_SIZE)}},
        metadata_preconditions=list(layout.metadata_preconditions(entry, target)),
        index_range_assertions=[
            dict(
                source=source,
                expression=expression,
                minimum=0,
                maximum=layout.SIGNED_MAX,
            )
            for expression in (
                layout.opengl_index_assertions(entry) if target == "opengl" else ()
            )
        ],
        source_options={"metal": metal},
    )


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("gated_delta_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


class GatedDeltaPackageCache(TranslatedEntryCache):
    LABEL = "Gated delta"
    SOURCE = layout.SOURCES["forward"]
    PREFIX = "gated-delta"

    def signature(self, entry):
        return layout.signature(entry)

    def source_entry(self, entry):
        return layout.source_entry(entry)

    def source_path(self, entry):
        return layout.source(entry)

    def workgroup_size(self, entry):
        return layout.workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, packages, target, entry):
    cache = GatedDeltaPackageCache(root, Path(packages) / CACHE_DIRECTORY, target)
    return build_record(cache, entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        layout.signature,
        "Gated delta",
        root,
        Path(packages).resolve(),
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The pipelines the unchanged upstream gated-delta tests launch.

    The host has no NAX, so gated_delta_ckpt() returns 1 for every launch:
    the forward runs the sequential kernel up to eight steps and the chunk
    kernel beyond, and the backward the save pass and the sequential
    backward. ``test_gated_delta_dtypes`` runs float32 and bfloat16 queries
    of 32 heads; ``test_gated_delta_vjp`` differentiates float32 sequences
    of 32 heads and of 16 key heads sharing 32 or 48 value heads.
    """
    entries = [
        layout.entry_name("seq", dtype, 128, 128, 32, 32, 1)
        for dtype in ("float32", "bfloat16")
    ]
    for hk, hv in ((32, 32), (16, 32), (16, 48)):
        entries += [
            layout.entry_name("seq", "float32", 128, 128, hk, hv, 1),
            layout.entry_name("fused", "float32", 128, 128, hk, hv, layout.CHUNK),
            layout.entry_name("seq", "float32", 128, 128, hk, hv, 1, save=True),
            layout.entry_name("vjp", "float32", 128, 128, hk, hv, 1),
        ]
    return sorted(set(entries))


def main(argv=None):
    """Translate gated-delta entries into a host package directory's cache."""
    argv = list(sys.argv[1:] if argv is None else argv)
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
