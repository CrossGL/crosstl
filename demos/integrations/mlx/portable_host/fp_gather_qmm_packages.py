"""Cache translated block-scaled gathered quantized entries of the unchanged pinned source.

Each fp_quantized.metal ``gather_qmv_fast``, ``gather_qmv``, ``gather_qvm``,
``gather_qmm_t`` and ``gather_qmm_n`` entry, with or without a global scale,
is translated with the affine gathered entries' options
(``gather_qmm_packages``) and ``fp_gather_qmm_layout``'s assertions: the
matrix-matrix entries take their batched block-scaled siblings' tile
assertions. Run as a module, it translates entries into a host package
directory's cache ahead of execution, several processes at a time.
"""

import hashlib
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    fp_gather_qmm_layout,
    gather_qmm_packages,
)
from demos.integrations.mlx.portable_host.fp_gather_qmm_layout import (
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
CACHE_DIRECTORY = "fp-gather-quantized"
LABEL = fp_gather_qmm_layout.LABEL
# The modules whose options and assertions each translation depends on.
RECIPE_MODULES = (
    "fp_gather_qmm_packages.py",
    "fp_gather_qmm_layout.py",
    "gather_qmm_packages.py",
    "gather_qmm_layout.py",
    "fp_qmm_layout.py",
    "qmm_packages.py",
    "qmm_layout.py",
    "matmul_layout.py",
)


def _recipe_hashes():
    here = Path(__file__)
    return {
        name.removesuffix(".py"): (
            hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
        )
        for name in RECIPE_MODULES
    }


def project_config(root, output_dir, target, entry):
    """One entry's translation, with the affine gathered entries' options."""
    return gather_qmm_packages.project_config(
        root, output_dir, target, entry, fp_gather_qmm_layout
    )


class FpGatherQmmPackageCache(TranslatedEntryCache):
    LABEL = LABEL
    SOURCE = SOURCE
    PREFIX = "fp-gather-quantized"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


def _build_one(root, directory, target, entry):
    return build_record(FpGatherQmmPackageCache(root, directory, target), entry)


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
    from demos.integrations.mlx.portable_host import verify_fp_gather_qmm

    return sorted(
        {
            verify_fp_gather_qmm.case_record(case)["entry"]
            for case in verify_fp_gather_qmm.CASES
        }
    )


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
