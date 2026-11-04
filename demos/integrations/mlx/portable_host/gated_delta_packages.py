"""Cache translated MLX gated-delta recurrence entries.

Each host entry is one pinned ``gated_delta_update.metal`` or
``gated_delta_update_vjp.metal`` kernel named by
``gated_delta_update.cpp``'s hash name, with the ``save_state`` function
constant that name implies frozen, at the ``(32, 4, 1)`` threadgroups
``gated_delta_layout`` checks. Every kernel combines each state row with
``simd_sum``, so Direct3D and OpenGL run them in 32-lane software subgroups;
the fused chunk kernel's ``simdgroup_float8x8`` tiles lower to those
subgroups through MLX's own lane coordinates. This is the recipe the pinned
kernel tests (``test_current_gated_delta`` and its VJP) prove. Run as a
module, it translates entries into a host package directory's cache ahead of
execution, several processes at a time.
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
SOFTWARE_SUBGROUP_TARGETS = ("opengl", "directx")


def target_options(target, fused):
    options = {}
    if target in SOFTWARE_SUBGROUP_TARGETS:
        options["software_subgroup_width"] = layout.SIMD_SIZE
        if fused:
            options["cooperative_matrix_software_lowering"] = True
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    return {target: options} if options else {}


def project_config(root, output_dir, target, entry):
    """One gated-delta entry with its frozen constants and proven ranges."""
    info = layout.signature(entry)
    source, name = info["source"], info["sourceEntry"]
    fused = info["kind"] == "fused"
    metal = {
        # The host binds each reflected resource by its source parameter.
        "preserve_resource_origins": True,
        "max_template_specializations": 256,
        "max_template_materialization_work": 16384,
        "target_options": target_options(target, fused),
    }
    if fused:
        metal.update(
            cooperative_matrix_fragment_mapping=layout.FRAGMENT_MAPPING,
            cooperative_matrix_fragment_mapping_provenance=(
                layout.FRAGMENT_MAPPING_PROVENANCE
            ),
        )
    constants = layout.specialization_constants(entry)
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (name,)},
        entry_workgroup_size_rules={source: {name: list(info["workgroupSize"])}},
        specialization_constants=constants,
        freeze_specialization_constants=bool(constants),
        index_range_assertions=(
            layout.opengl_index_assertions(entry) if target == "opengl" else ()
        ),
        metadata_preconditions=layout.metadata_preconditions(entry, target),
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
    PREFIX = "gated-delta"

    def signature(self, entry):
        return layout.signature(entry)

    def source_entry(self, entry):
        return layout.source_entry(entry)

    def source_path(self, entry):
        return layout.signature(entry)["source"]

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
    """Each kernel the host can dispatch once per storage type and head shape
    family: inference, the save pass and the backward at interval 1, and the
    fused chunk."""
    entries = []
    for dtype in layout.TYPE_NAMES:
        for hk, hv in ((32, 32), (16, 48)):
            entries += [
                layout.entry_name("sequential", dtype, hk, hv),
                layout.entry_name("sequential", dtype, hk, hv, save=True),
                layout.entry_name("fused", dtype, hk, hv, layout.CHUNK),
                layout.entry_name("vjp", dtype, hk, hv),
            ]
    return entries


def main(argv=None):
    """Translate gated-delta entries into a host package directory's cache."""
    argv = list(sys.argv[1:] if argv is None else argv)
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
