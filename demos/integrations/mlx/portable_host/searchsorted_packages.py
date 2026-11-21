"""Cache translated searchsorted.metal entries at their exact pass size.

Each host variant is one entry launched over one pass of values, keyed
``<entry>@<count>``. Direct3D and OpenGL packages carry an artifact-owned
dispatch region for that pass, so the kernel's ``threads_per_grid`` is the
pass's value count. OpenGL is translated with the host precondition that every
sequence index stays below ``MAX_SPAN``.
"""

import hashlib
from pathlib import Path

from crosstl.project import ProjectConfig
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from demos.integrations.mlx.portable_host.matmul_packages import TranslatedEntryCache
from demos.integrations.mlx.portable_host.searchsorted_layout import (
    MAX_SPAN,
    SOURCE,
    execution,
    split_variant,
)

CACHE_DIRECTORY = "searchsorted"
# The sequence offset each search forms (the GLSL spelling of
# ``int64_t(mid) * a_stride``).
INDEX_EXPRESSION = "int64(mid) * a_stride"


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(
                here.with_name("searchsorted_layout.py").read_bytes()
            ).hexdigest()
        ),
    }


def _region(count):
    launch = execution(count)
    grid = [n * w for n, w in zip(launch["workgroupCount"], launch["workgroupSize"])]
    (region,) = plan_dispatch_regions(grid, launch["workgroupSize"])
    return region.to_json()


def project_config(root, output_dir, target, key):
    """One variant's translation for one pass of ``count`` values."""
    entry, count = split_variant(key)
    target_options = {} if target == "metal" else {"dispatch_region": _region(count)}
    indices = (
        [
            dict(
                source=SOURCE,
                expression=INDEX_EXPRESSION,
                minimum=0,
                maximum=MAX_SPAN - 1,
            )
        ]
        if target == "opengl"
        else []
    )
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: (1, 1, 1)}},
        index_range_assertions=indices,
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 16384,
                "target_options": {target: target_options},
            }
        },
    )


class SearchSortedPackageCache(TranslatedEntryCache):
    LABEL = "Searchsorted"
    SOURCE = SOURCE
    PREFIX = "searchsorted"

    def signature(self, key):
        return split_variant(key)

    def workgroup_size(self, key):
        split_variant(key)
        return (1, 1, 1)

    def source_entry(self, key):
        return split_variant(key)[0]

    def project_config(self, output_dir, key):
        return project_config(self.root, output_dir, self.target, key)

    def recipe_hashes(self):
        return _recipe_hashes()
