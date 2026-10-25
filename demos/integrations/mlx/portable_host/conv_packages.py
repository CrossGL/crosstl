"""Cache translated MLX convolution entries: unfold, separable and implicit GEMM.

``naive_unfold_nd``, ``naive_unfold_transpose_nd`` and ``depthwise_conv_1d``
come from the pinned conv.metal with the options of its current-tree kernel
gate (``test_current_conv``), at the one-thread workgroups conv_layout
launches. ``depthwise_conv_2d`` comes from the same source at conv.cpp's
threadgroups, one entry per conv.cpp pipeline with its function constants
frozen. ``implicit_gemm_conv_2d`` and ``implicit_gemm_conv_3d`` come from
steel_conv.metal and steel_conv_3d.metal with the options of
``test_current_steel_conv``. Run as a module, it translates entries into a
host package directory's cache ahead of execution, several processes at a
time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import conv_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)
from demos.integrations.mlx.tests.kernels import steel_conv_model as steel

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "conv"
# depthwise_conv_2d indexes its threadgroup tile with simd_gid * 32 + simd_lid,
# the linear thread index of 32-lane SIMD groups.
SIMD_SIZE = 32
# test_current_conv: the depthwise kernels accumulate into `float o = 0.`,
# which GLSL and HLSL compilers may fold away; the software binary32 addition
# keeps Metal's +0.0.
ADDITIVE_PROFILE = "rne-flush"


def implicit_project_config(root, output_dir, target, entry):
    """One steel_conv entry with the current-tree steel_conv gate's options."""
    model = layout.signature(entry)["model"]
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(model.source,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={model.source: (entry,)},
        entry_workgroup_size_rules={model.source: {entry: model.workgroup_size}},
        index_range_assertions=(
            steel.index_assertions(model) if target == "opengl" else ()
        ),
        source_options={"metal": steel.metal_source_options()},
    )


def project_config(root, output_dir, target, entry):
    """One conv.metal entry with the current-tree conv gate's options."""
    info = layout.signature(entry)
    if info["kind"] == "implicit":
        return implicit_project_config(root, output_dir, target, entry)
    target_options = {}
    if info["kind"] in {"dw1", "dw2"} and target in {"opengl", "directx"}:
        target_options[target] = {"binary32_additive_profile": ADDITIVE_PROFILE}
    if info["kind"] == "dw2" and target in {"opengl", "directx"}:
        target_options[target]["software_subgroup_width"] = SIMD_SIZE
    source_entry = layout.source_entry(entry)
    assertions = []
    if target == "opengl":
        assertions = [
            {
                "source": layout.SOURCE,
                "expression": expression,
                "minimum": 0,
                "maximum": layout.SIGNED_MAX,
            }
            for expression in layout.OPENGL_INDEX_RANGE_ASSERTIONS[
                layout.assertion_kind(entry)
            ]
        ]
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(layout.SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={layout.SOURCE: (source_entry,)},
        entry_workgroup_size_rules={
            layout.SOURCE: {source_entry: layout.workgroup_size(entry)}
        },
        specialization_constants=dict(info.get("constants", {})),
        freeze_specialization_constants=info["kind"] == "dw2",
        index_range_assertions=assertions,
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 16384,
                **({"target_options": target_options} if target_options else {}),
            }
        },
    )


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("conv_layout.py").read_bytes()).hexdigest()
        ),
    }


class ConvPackageCache(TranslatedEntryCache):
    LABEL = "Convolution"
    SOURCE = layout.SOURCE
    PREFIX = "conv"

    def signature(self, entry):
        return layout.signature(entry)

    def source_path(self, entry):
        return layout.source(entry)

    def source_entry(self, entry):
        return layout.source_entry(entry)

    def workgroup_size(self, entry):
        return layout.workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, packages, target, entry):
    cache = ConvPackageCache(root, Path(packages) / CACHE_DIRECTORY, target)
    return build_record(cache, entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        layout.signature,
        "Convolution",
        root,
        Path(packages).resolve(),
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The entries verify_conv's cases dispatch."""
    from demos.integrations.mlx.portable_host import verify_conv

    entries = [
        entry
        for case in verify_conv.CASES
        for entry in verify_conv.case_record(case)["convEntries"]
    ]
    return list(dict.fromkeys(entries))


def build_products(root, packages, target, entries, jobs):
    """Translate the explicit routes' steel GEMM and split-K entries."""
    from demos.integrations.mlx.portable_host import gemm_packages, gemm_splitk_packages

    regular = [entry for entry in entries if entry.startswith("steel_gemm_fused_")]
    split = [entry for entry in entries if entry not in regular]
    return [
        *(
            gemm_packages.build_entries(root, packages, target, regular, jobs)
            if regular
            else []
        ),
        *(
            gemm_splitk_packages.build_entries(root, packages, target, split, jobs)
            if split
            else []
        ),
    ]


def product_entries():
    """The steel GEMM and split-K entries verify_conv's cases dispatch."""
    from demos.integrations.mlx.portable_host import verify_conv

    entries = [
        entry
        for case in verify_conv.CASES
        for entry in verify_conv.case_record(case)["productEntries"]
    ]
    return list(dict.fromkeys(entries))


def main(argv=None):
    """Translate conv.metal entries, or with ``products`` first the GEMM entries."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["products"]:
        return cli(argv[1:], __doc__, build_products, product_entries)
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
