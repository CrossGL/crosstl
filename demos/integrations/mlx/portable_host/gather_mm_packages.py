"""Cache translated MLX GatherMM entries: gathered gemv, steel gather and offsets.

``gemv_gather`` and ``gemv_t_gather`` come from the pinned gemv.metal,
``steel_gather_mm`` and ``steel_gather_mm_rhs`` from steel_gemm_gather.metal
and ``gather_mm_offsets`` from gather_mm_offsets.metal, each with the options
of its current-tree kernel gate. Run as a module, it translates entries into a
host package directory's caches ahead of execution, several processes at a
time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import gather_mm_layout, gather_mv_layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directories under a host package root that HostRuntime caches entries in.
GATHER_MV_CACHE = "gather-mv"
GATHER_MM_CACHE = "gather-mm"
GATHER_OFFSETS_CACHE = "gather-mm-offsets"
SIMD_SIZE = gather_mm_layout.SIMD_SIZE


def _recipe_hashes(layout):
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name(f"{layout}.py").read_bytes()).hexdigest()
        ),
    }


def gather_mv_config(root, output_dir, target, entry):
    """One gathered gemv entry, with the current-tree gemv gate's options."""
    source = gather_mv_layout.SOURCE
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (entry,)},
        entry_workgroup_size_rules={
            source: {entry: gather_mv_layout.workgroup_size(entry)}
        },
        # gather_mv_layout.plan bounds every bound buffer at MAX_ELEMENTS
        # elements and every gathered offset inside its buffer.
        index_range_assertions=[
            dict(
                source=source,
                expression=expression,
                minimum=0,
                maximum=gather_mv_layout.SIGNED_MAX,
            )
            for expression in (
                gather_mv_layout.index_assertions(entry) if target == "opengl" else ()
            )
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 16384,
                "target_options": {
                    "directx": {
                        "software_subgroup_width": SIMD_SIZE,
                        "relative_wave_shuffle_out_of_range": "self",
                    },
                    "opengl": {"software_subgroup_width": SIMD_SIZE},
                },
            }
        },
    )


def gather_mm_config(root, output_dir, target, entry):
    """One steel gather entry, with the current-tree gather GEMM gate's options."""
    info = gather_mm_layout.signature(entry)
    source = gather_mm_layout.SOURCE
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (info["source"],)},
        specialization_constants=dict(info["constants"]),
        freeze_specialization_constants=True,
        workgroup_size_rules={source: gather_mm_layout.workgroup_size(entry)},
        workgroup_access_assertions=(
            gather_mm_layout.loader_access_assertions(entry)
            if target != "metal"
            else ()
        ),
        index_range_assertions=(
            gather_mm_layout.index_assertions(entry) if target == "opengl" else ()
        ),
        source_options={
            "metal": {
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "preserve_resource_origins": True,
                "target_options": {
                    "directx": {
                        "cooperative_matrix_software_lowering": True,
                        "software_subgroup_width": SIMD_SIZE,
                        "relative_wave_shuffle_out_of_range": "self",
                    },
                    "opengl": {
                        "cooperative_matrix_software_lowering": True,
                        "software_subgroup_width": SIMD_SIZE,
                    },
                },
            }
        },
    )


# How each target's package records gather_mm_offsets' one-thread workgroup;
# Metal takes it from the dispatch.
EXECUTION_CONFIGS = {
    "metal": {},
    "directx": {"numthreads": [1, 1, 1]},
    "opengl": {"local_size_x": 1, "local_size_y": 1, "local_size_z": 1},
}


def gather_offsets_config(root, output_dir, target, entry):
    """gather_mm_offsets with one-thread workgroups, as its kernel gate runs it."""
    gather_mm_layout.signature(entry)
    source = gather_mm_layout.OFFSETS_SOURCE
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={source: (entry,)},
        # A plain kernel takes the project's workgroup size; per-entry rules
        # need template materializations.
        workgroup_size=gather_mm_layout.workgroup_size(entry),
        source_options={"metal": {"preserve_resource_origins": True}},
    )


class GatherMvPackageCache(TranslatedEntryCache):
    LABEL = "Gathered matrix-vector"
    SOURCE = gather_mv_layout.SOURCE
    PREFIX = "gather-mv"

    def signature(self, entry):
        return gather_mv_layout.signature(entry)

    def workgroup_size(self, entry):
        return gather_mv_layout.workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return gather_mv_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes("gather_mv_layout")


class GatherMmPackageCache(TranslatedEntryCache):
    LABEL = "Gathered steel GEMM"
    SOURCE = gather_mm_layout.SOURCE
    PREFIX = "gather-mm"

    def signature(self, entry):
        info = gather_mm_layout.signature(entry)
        if info["kind"] == "offsets":
            raise ValueError("Group offsets have their own cache")
        return info

    def source_entry(self, entry):
        return gather_mm_layout.signature(entry)["source"]

    def workgroup_size(self, entry):
        return gather_mm_layout.workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return gather_mm_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes("gather_mm_layout")


class GatherOffsetsPackageCache(TranslatedEntryCache):
    LABEL = "Gathered group offsets"
    SOURCE = gather_mm_layout.OFFSETS_SOURCE
    PREFIX = "gather-mm-offsets"

    def signature(self, entry):
        info = gather_mm_layout.signature(entry)
        if info["kind"] != "offsets":
            raise ValueError("Unsupported gathered group offsets entry")
        return info

    def workgroup_size(self, entry):
        return gather_mm_layout.workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return gather_offsets_config(self.root, output_dir, self.target, entry)

    def translated_workgroup_size(self, entry, artifact):
        # A plain kernel records the project's workgroup size, and its Metal
        # artifact none, since Metal takes the size from the dispatch; _load
        # checks every target's packaged execution configuration.
        execution = artifact.get("execution")
        if execution is None:
            return list(self.workgroup_size(entry))
        if execution["sourceEntryPoints"] != [entry]:
            raise ValueError(
                "Gathered group offsets translation selected another entry"
            )
        return execution["workgroupSize"]

    def _load(self, directory, identity):
        descriptor, package = super()._load(directory, identity)
        if (
            descriptor["entryPoint"].get("executionConfig")
            != EXECUTION_CONFIGS[self.target]
        ):
            raise ValueError(
                "Gathered group offsets package changed the workgroup size"
            )
        return descriptor, package

    def recipe_hashes(self):
        return _recipe_hashes("gather_mm_layout")


def cache_for(entry):
    """``(cache class, cache directory)`` of one GatherMM entry."""
    if entry == gather_mm_layout.OFFSETS_ENTRY:
        return GatherOffsetsPackageCache, GATHER_OFFSETS_CACHE
    if isinstance(entry, str) and entry.startswith(gather_mv_layout.ENTRY_PREFIXES):
        return GatherMvPackageCache, GATHER_MV_CACHE
    return GatherMmPackageCache, GATHER_MM_CACHE


def _check(entry):
    cache, _ = cache_for(entry)
    cache.signature(None, entry)


def _build_one(root, directory, target, entry):
    cache, name = cache_for(entry)
    return build_record(cache(root, Path(directory) / name, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' caches, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        _check,
        "GatherMM",
        root,
        Path(packages).resolve(),
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The entries verify_gather_mm's cases dispatch."""
    from demos.integrations.mlx.portable_host import verify_gather_mm

    entries = [
        entry
        for case in verify_gather_mm.CASES
        for entry in verify_gather_mm.case_record(case)["entries"]
    ]
    return list(dict.fromkeys(entries))


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
