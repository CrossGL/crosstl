"""Cache translated Hadamard JIT entries of the pinned hadamard.cpp recipe.

``hadamard_mn_contiguous`` compiles one library per transform size and type:
``metal::utils()``, the codelet ``gen_hadamard_codelet(m)`` generates from the
pinned Hadamard matrices, ``metal::hadamard()`` and one template definition per
stage. ``source`` renders the same text for one stage, with the two JIT
headers included from the unchanged pinned tree. Run as a module, it translates
entries into a host package directory's cache ahead of execution, several
processes at a time.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host import hadamard_layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    TranslatedEntryCache,
    build_record,
    cli,
    run_builds,
)
from demos.integrations.mlx.portable_host.prepare import require_revision

# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "hadamard"
MATRICES = "mlx/backend/common/hadamard.h"
KERNEL = "mlx/backend/metal/kernels/hadamard.h"
RECIPE = "mlx/backend/metal/hadamard.cpp"
WRAPPER = "hadamard.metal"


def matrices(root):
    """The raw ``h<m>`` strings ``hadamard_matrices()`` returns, by ``m``."""
    text = (Path(root) / MATRICES).read_text(encoding="utf-8")
    found = {}
    for m in hadamard_layout.FACTORS:
        match = re.search(
            r'constexpr std::string_view h%d = R"\((.*?)\)";' % m, text, re.S
        )
        if match is None:
            raise ValueError(f"Pinned Hadamard matrix h{m} is missing")
        found[m] = match.group(1)
    return found


def codelet(root, m):
    """``gen_hadamard_codelet(m)``, statement for statement."""
    lines = ["METAL_FUNC void hadamard_radix_m(thread float *x) {"]
    if m == 1:
        return "\n".join(lines + ["}"]) + "\n"
    matrix = matrices(root)[m]
    lines.append(f"  float tmp[{m}];")
    start, index = 1, 0
    end = matrix.find("\n", start)
    while end != -1:
        row = matrix[start:end]
        lines.append(
            f"  tmp[{index}] = "
            + "".join(f" {sign} x[{i}]" for i, sign in enumerate(row))
            + ";"
        )
        start = end + 1
        end = matrix.find("\n", start)
        index += 1
    lines.append(f"  for (int i = 0; i < {m}; i++) {{ x[i] = tmp[i]; }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def template_definition(name, function, arguments):
    """``get_template_definition``'s text."""
    rendered = f"{function}<{', '.join(str(argument) for argument in arguments)}>"
    return (
        f'\ntemplate [[host_name("{name}")]] [[kernel]] '
        f"decltype({rendered}) {rendered};\n"
    )


def source(root, entry):
    info = hadamard_layout.signature(entry)
    return (
        '#include "mlx/backend/metal/kernels/utils.h"\n'
        + codelet(root, info["m"])
        + f'#include "{KERNEL}"\n'
        + template_definition(entry, info["function"], info["templateArguments"])
    )


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("hadamard_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


class HadamardPackageCache(TranslatedEntryCache):
    LABEL = "Hadamard"
    SOURCE = KERNEL
    PREFIX = "hadamard"

    def signature(self, entry):
        return hadamard_layout.signature(entry)

    def workgroup_size(self, entry):
        return hadamard_layout.signature(entry)["workgroupSize"]

    def recipe_hashes(self):
        return _recipe_hashes()

    def _require_source(self):
        require_revision(self.root)
        dirty = subprocess.check_output(
            ["git", "-C", str(self.root), "status", "--porcelain", "--"]
            + ["mlx/backend/metal/kernels", MATRICES, RECIPE],
            text=True,
            timeout=30,
        )
        if dirty.strip():
            raise ValueError(
                "Hadamard translation requires unchanged pinned kernels and matrices"
            )

    def _identity(self, entry):
        identity = super()._identity(entry)
        # The translated source is the generated wrapper, not the header.
        identity["source"] = WRAPPER
        identity["sourceHash"] = hashlib.sha256(
            source(self.root, entry).encode("utf-8")
        ).hexdigest()
        return identity

    def project_config(self, output_dir, entry):
        work = Path(output_dir).parent
        wrapper = self.root / work / WRAPPER
        wrapper.write_bytes(source(self.root, entry).encode("utf-8"))
        relative = wrapper.relative_to(self.root).as_posix()
        return ProjectConfig(
            root=self.root,
            source_roots=(work.as_posix(),),
            include_patterns=(relative,),
            include_dirs=(".",),
            targets=(self.target,),
            output_dir=output_dir,
            entry_points={relative: (entry,)},
            workgroup_size_rules={relative: self.workgroup_size(entry)},
            source_options={"metal": {"preserve_resource_origins": True}},
        )


def _build_one(root, directory, target, entry):
    return build_record(HadamardPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        hadamard_layout.signature,
        "Hadamard",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    from demos.integrations.mlx.portable_host import verify_upstream_hadamard

    return verify_upstream_hadamard.package_entries()


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
