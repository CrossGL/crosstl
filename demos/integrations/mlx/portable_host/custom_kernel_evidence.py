"""Tie custom-kernel dispatches to the packages their traced sources name.

A custom-kernel dispatch records the SHA-256 of the source metal_kernel.cpp
built, its math mode and its exact grid. The audit finds the cached packages
of that source and launch, checks that the cached source, the translated
artifacts and the retained native modules are unchanged, and that the launch
named each package's entry point and geometry: Metal's single package, or
every Direct3D or OpenGL dispatch region in order. The upstream tests compare
the values with MLX's own operations.
"""

import hashlib
from pathlib import Path

from demos.integrations.mlx.portable_host import custom_kernel_layout as layout
from demos.integrations.mlx.portable_host.custom_kernel_packages import (
    CACHE_DIRECTORY,
    CustomKernelPackageCache,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    GEOMETRY,
    require_unchanged,
)

MODES = {name: value for value, name in layout.MATH_MODES.items()}


def is_custom_kernel(event):
    return layout.is_entry(event.get("entry"))


def cache(mlx_root, packages, target):
    return CustomKernelPackageCache(mlx_root, Path(packages) / CACHE_DIRECTORY, target)


def audit_regions(event, packages, modules):
    """Each Direct3D or OpenGL region names its package and native module."""
    regions = event["details"].get("regions")
    require(
        isinstance(regions, list) and len(regions) == len(packages),
        "Custom kernel region coverage differs",
    )
    for region, (descriptor, directory) in zip(regions, packages):
        geometry = descriptor["provenance"]["dispatchRegion"]
        module = Path(region["moduleFile"]).resolve()
        require(
            region["artifact"] == descriptor["artifact"]
            and region["provenance"] == descriptor["provenance"]
            and Path(region["packageRoot"]).resolve() == Path(directory).resolve()
            and all(region[key] == geometry[key] for key in GEOMETRY)
            and module.is_relative_to(modules)
            and hashlib.sha256(module.read_bytes()).hexdigest() == region["moduleHash"],
            "Custom kernel region identity differs",
        )


def audit_event(event, kernels, modules):
    """Check one custom-kernel dispatch against the packages of its source."""
    entry = event["entry"]
    require(
        event["target"] == kernels.target,
        "Custom kernel dispatch targets another backend",
    )
    metadata = event["customKernel"]
    packages, source = kernels.find(
        entry,
        event["sourceHash"],
        MODES[metadata["mathMode"]],
        thread_grid_size=metadata["threadGridSize"],
        workgroup_size=metadata["workgroupSize"],
    )
    for descriptor, directory in packages:
        require_unchanged(descriptor, directory)
    signature = layout.parse(entry, source)
    descriptor, directory = packages[0]
    require(
        event["artifact"] == descriptor["artifact"]
        and event["dispatchVersion"] == 3
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve()
        and event["parameters"] == [item.to_json() for item in signature.parameters]
        and metadata["threadGridSize"] == event.get("threadGridSize")
        and metadata["workgroupSize"] == event["workgroupSize"]
        and set(event["boundParameters"]) >= {item.name for item in signature.outputs},
        "Custom kernel dispatch does not name its cached artifact",
    )
    if kernels.target != "metal":
        audit_regions(event, packages, Path(modules).resolve())
        return
    request = event["details"]["request"]
    require(
        len(packages) == 1
        and request["target"] == kernels.target
        and request["entryPoint"] == descriptor["entryPoint"]["name"]
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Custom kernel native launch differs",
    )
    audit_native_execution(event)
