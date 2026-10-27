"""Cache translated MLX Stockham, Rader, Bluestein and four-step FFT entries, one per fft.cpp pipeline.

Each entry is the pinned ``fft.metal`` ``fft``, ``rader_fft``, ``bluestein_fft`` or ``four_step_fft``
kernel at one threadgroup memory size and input/output type pair (and, for
``four_step_fft``, one pass), with the function constants
fft.cpp sets for one transform size and direction frozen into it, at
fft.cpp's ``(1, threadgroup_batch_size, threads_per_fft)`` threadgroups.
The host names it by fft.cpp's pipeline hash name (``fft_layout``). OpenGL
translation receives the subscript ranges the host's launch bounds prove.
Run as a module, it translates entries into a host package directory's
cache ahead of execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import fft_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "fft"


def project_config(root, output_dir, target, entry):
    """One frozen fft.metal entry with the host's proven subscript ranges."""
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
        freeze_specialization_constants=True,
        index_range_assertions=layout.opengl_index_assertions(entry) if opengl else (),
        workgroup_access_assertions=(
            layout.workgroup_access_assertions(entry) if opengl else ()
        ),
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 4096,
                "max_template_materialization_work": 2097152,
            }
        },
    )


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("fft_layout.py").read_bytes()).hexdigest()
        ),
    }


class FftPackageCache(TranslatedEntryCache):
    LABEL = "FFT"
    SOURCE = layout.SOURCE
    PREFIX = "fft"

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
    cache = FftPackageCache(root, Path(packages) / CACHE_DIRECTORY, target)
    return build_record(cache, entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        layout.signature,
        "FFT",
        root,
        Path(packages).resolve(),
        target,
        entries,
        jobs,
    )


# Both directions of each transform: (inverse, input type, output type).
DIRECTIONS = (
    (False, "float2", "float2"),
    (True, "float2", "float2"),
    (False, "float", "float2"),
    (True, "float2", "float"),
)


def verification_entries():
    """Both directions of every transform kind at two Stockham sizes, one
    Rader size and one Bluestein size, and at a Stockham size whose
    threadgroups pass the 64-thread z limit of Direct3D and OpenGL; and both
    passes of the four-step transforms of 8,192 points and of 2**23 points,
    whose 4,096- and 2,048-point passes also pass that limit."""
    single = [
        layout.entry_name(n, *direction)
        for n in (8, 100, 17, 47, 1024)
        for direction in DIRECTIONS
    ]
    four_step = [
        name
        for n in (1 << 13, 1 << 23)
        for direction in DIRECTIONS
        for name in layout.four_step_entry_names(n, *direction)
    ]
    return single + four_step


def main(argv=None):
    """Translate fft.metal entries into a host package directory's cache."""
    argv = list(sys.argv[1:] if argv is None else argv)
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
