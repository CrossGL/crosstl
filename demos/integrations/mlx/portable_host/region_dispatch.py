"""Run one launch as the Direct3D or OpenGL dispatch regions it splits into.

Small-row kernels read their exact thread grid, which Direct3D and OpenGL do
not expose, so each region of a launch is its own translated package. The
regions run in order in one native sequence, and every region's compiled
module is retained by content hash beside the trace.
"""

from crosstl.project import prepare_native_loader_dispatch_regions
from crosstl.project.runtime_verification import RuntimeExecutorResult
from demos.integrations.mlx.portable_host.retained_modules import retain


def run_regions(host, region_packages, inputs, outputs, execution):
    """Dispatch ``region_packages`` over one bound launch and record them."""
    adapter = host.executor.runtime_adapter
    with prepare_native_loader_dispatch_regions(
        region_packages,
        inputs,
        outputs,
        thread_grid_size=execution["threadGridSize"],
        source_workgroup_size=execution["workgroupSize"],
        adapter=adapter,
    ) as requests:
        records = []
        for (region_descriptor, root), request in zip(region_packages, requests):
            module = retain(
                host.trace.parent / "native-modules",
                request.module_path.read_bytes(),
                request.module_path.suffix,
            )
            records.append(
                {
                    "artifact": region_descriptor["artifact"],
                    "packageRoot": str(root),
                    "provenance": region_descriptor["provenance"],
                    "moduleHash": module["sha256"],
                    "moduleFile": module["file"],
                    "workgroupCount": list(request.dispatch.workgroup_count),
                    "workgroupSize": list(request.dispatch.workgroup_size),
                }
            )
        native_outputs = adapter.runtime.dispatch_sequence(None, None, requests)
        return RuntimeExecutorResult(
            outputs=native_outputs,
            details={"runtime": adapter.runtime.name, "regions": records},
        )
