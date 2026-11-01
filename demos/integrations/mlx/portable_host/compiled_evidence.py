"""Tie compiled-kernel dispatches to the packages their traced sources name.

A compiled dispatch records the SHA-256 of its generated source. The audit
finds the cached package of that source, checks that the cached source, its
translated artifact and the retained native modules are unchanged, and that
the launch named the package's entry point and geometry. The upstream tests
compare the values with MLX's own composed operations.
"""

from pathlib import Path

from demos.integrations.mlx.portable_host import compiled_layout
from demos.integrations.mlx.portable_host.compiled_packages import (
    CACHE_DIRECTORY,
    CompiledPackageCache,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    GEOMETRY,
    require_unchanged,
)


def is_compiled(event):
    return isinstance(event.get("entry"), str) and event["entry"].startswith(
        compiled_layout.ENTRY_PREFIX
    )


def cache(mlx_root, packages, target):
    return CompiledPackageCache(mlx_root, Path(packages) / CACHE_DIRECTORY, target)


def audit_event(event, compiled):
    """Check one compiled dispatch against the package of its traced source."""
    entry = event["entry"]
    compiled_layout.variant(entry)
    require(
        event["target"] == compiled.target, "Compiled dispatch targets another backend"
    )
    (descriptor, directory), source = compiled.find(entry, event["sourceHash"])
    require_unchanged(descriptor, directory)
    parameters = [list(item) for item in compiled_layout.parse(entry, source)]
    require(
        event["artifact"] == descriptor["artifact"]
        and event["dispatchVersion"] == 3
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve()
        and event["parameters"] == parameters
        and event["compiledMetadata"]["variant"] == compiled_layout.variant(entry)
        and event["compiledMetadata"]["workPerThread"]
        == compiled_layout.work_per_thread(source),
        "Compiled dispatch does not name its cached artifact",
    )
    request = event["details"]["request"]
    require(
        request["target"] == compiled.target
        and request["entryPoint"] == descriptor["entryPoint"]["name"]
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Compiled native launch differs",
    )
    audit_native_execution(event)
