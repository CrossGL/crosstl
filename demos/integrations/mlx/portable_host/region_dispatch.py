"""Run one launch as the Direct3D or OpenGL dispatch regions it splits into.

Small-row kernels read their exact thread grid, which Direct3D and OpenGL do
not expose, so each region of a launch is its own translated package. The
regions run in order in one native sequence, and every region's compiled
module is retained by content hash beside the trace.
"""

import hashlib

from crosstl.project import prepare_native_loader_dispatch_regions
from crosstl.project.runtime_verification import RuntimeExecutorResult


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
            module_bytes = request.module_path.read_bytes()
            module_hash = hashlib.sha256(module_bytes).hexdigest()
            module = (
                host.trace.parent
                / "native-modules"
                / (module_hash + request.module_path.suffix)
            )
            module.parent.mkdir(parents=True, exist_ok=True)
            module.write_bytes(module_bytes)
            records.append(
                {
                    "artifact": region_descriptor["artifact"],
                    "packageRoot": str(root),
                    "provenance": region_descriptor["provenance"],
                    "moduleHash": module_hash,
                    "moduleFile": str(module),
                    "workgroupCount": list(request.dispatch.workgroup_count),
                    "workgroupSize": list(request.dispatch.workgroup_size),
                }
            )
        native_outputs = adapter.runtime.dispatch_sequence(None, None, requests)
        return RuntimeExecutorResult(
            outputs=native_outputs,
            details={"runtime": adapter.runtime.name, "regions": records},
        )
