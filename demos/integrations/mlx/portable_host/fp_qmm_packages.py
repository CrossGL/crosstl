"""Cache translated block-scaled matrix-matrix entries of the unchanged pinned source.

Each fp_quantized.metal ``qmm_t``, ``qmm_n`` and ``qmm_t_splitk`` entry is
translated with the affine qmm entries' options (``qmm_packages``): the
kernels share the steel tiles, loaders and fragments. Run as a module, it
translates entries into a host package directory's cache ahead of
execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host import fp_qmm_layout, qmm_packages
from demos.integrations.mlx.portable_host.fp_qmm_layout import (
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
CACHE_DIRECTORY = "fp-quantized-qmm"
LABEL = "Floating-point quantized matrix-matrix"


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("fp_qmm_layout.py").read_bytes()).hexdigest()
        ),
        # The shared options and tile assertions.
        "baseRecipeHash": (
            hashlib.sha256(here.with_name("qmm_packages.py").read_bytes()).hexdigest()
        ),
        "baseLayoutHash": (
            hashlib.sha256(here.with_name("qmm_layout.py").read_bytes()).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """One entry's translation, with the affine qmm entries' options."""
    return qmm_packages.project_config(root, output_dir, target, entry, fp_qmm_layout)


class FpQuantizedQmmPackageCache(TranslatedEntryCache):
    LABEL = LABEL
    SOURCE = SOURCE
    PREFIX = "fp-quantized-qmm"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(FpQuantizedQmmPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        LABEL,
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    """The entries test_fp_qmm_non_multiple_of_32 routes, sorted once each."""
    from demos.integrations.mlx.portable_host import verify_upstream_fp_qmm

    return verify_upstream_fp_qmm.package_entries()["qmm"]


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
