"""Prepare a complete exact-grid region plan with shared native allocations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from crosstl.translator.dispatch_regions import DispatchRegion, plan_dispatch_regions

from .native_loader_abi import _validate_descriptor
from .native_loader_dispatch import (
    NativeLoaderDispatchError,
    build_native_loader_dispatch_request,
)
from .runtime_verification import (
    NativeRuntimeDispatchRequest,
    RuntimeAllocationView,
    RuntimeExecutionState,
)


def select_native_loader_dispatch_regions(
    packages: Sequence[tuple[Mapping[str, Any], Path]],
    *,
    thread_grid_size: Sequence[int],
    source_workgroup_size: Sequence[int],
) -> tuple[tuple[Mapping[str, Any], Path], ...]:
    """Order an exact region set, rejecting missing, duplicate or mixed programs.

    Each pair contains an ABI descriptor and its runtime package root. Selection
    never guesses a source kernel from its generated entry name or file path.
    """
    required = plan_dispatch_regions(thread_grid_size, source_workgroup_size)
    if len(packages) != len(required):
        raise NativeLoaderDispatchError(
            "dispatch-region-coverage-invalid",
            "Supply exactly one package per planned region.",
        )
    selected = {}
    identity = None
    for descriptor, root in packages:
        descriptor = _validate_descriptor(descriptor)
        provenance = descriptor["provenance"]
        if (
            "dispatchRegionProgram" not in provenance
            or "dispatchRegion" not in provenance
        ):
            raise NativeLoaderDispatchError(
                "dispatch-region-program-missing",
                "Region selection requires a recorded source program identity.",
            )
        region = DispatchRegion.from_json(provenance["dispatchRegion"])
        if region not in required or region in selected:
            raise NativeLoaderDispatchError(
                "dispatch-region-coverage-invalid",
                "Region packages must cover the requested canonical plan exactly once.",
            )
        current = (
            provenance["dispatchRegionProgram"],
            descriptor["target"],
            descriptor["source"]["backend"],
            descriptor["source"]["hash"],
            sorted(descriptor["bindings"], key=lambda item: item["name"]),
            descriptor["specializationConstants"],
            descriptor["scalarLayout"],
        )
        if identity is not None and current != identity:
            raise NativeLoaderDispatchError(
                "dispatch-region-program-mismatch",
                "Region packages must share the source program, translation settings and binding contract.",
            )
        identity = current
        selected[region] = (descriptor, Path(root))
    return tuple(selected[region] for region in required)


@contextmanager
def prepare_native_loader_dispatch_regions(
    packages: Sequence[tuple[Mapping[str, Any], Path]],
    input_values: Mapping[str, Any] | Sequence[Any],
    output_values: Mapping[str, Any] | Sequence[Any],
    *,
    thread_grid_size: Sequence[int],
    source_workgroup_size: Sequence[int],
    adapter: Any,
    specialization_values: Mapping[int, Any] | None = None,
) -> Iterator[tuple[NativeRuntimeDispatchRequest, ...]]:
    """Validate and compile all regions before yielding a single native sequence.

    Source buffers share allocations and are uploaded only once. Derived launch
    uniforms stay region-local. Native modules remain alive for the context and
    are cleaned up even if compilation or dispatch fails. No dispatch is issued
    by this function; pass the yielded tuple to the adapter's dispatch_sequence.
    """
    ordered = select_native_loader_dispatch_regions(
        packages,
        thread_grid_size=thread_grid_size,
        source_workgroup_size=source_workgroup_size,
    )
    requests = []
    for descriptor, root in ordered:
        region = DispatchRegion.from_json(descriptor["provenance"]["dispatchRegion"])
        request = build_native_loader_dispatch_request(
            descriptor,
            root,
            input_values,
            output_values,
            {
                "workgroupCount": list(region.workgroup_count),
                "workgroupSize": list(region.workgroup_size),
            },
            specialization_values,
            expected_target=adapter.target,
        )
        if any(
            item.get("severity") == "error"
            for item in request.execution_plan.diagnostics
        ):
            raise NativeLoaderDispatchError(
                "dispatch-region-preflight-failed",
                "Region execution plan contains setup errors.",
                details={"diagnostics": list(request.execution_plan.diagnostics)},
            )
        requests.append(request)
    states, prepared = [], []
    try:
        for index, (request, (descriptor, _)) in enumerate(zip(requests, ordered)):
            state = RuntimeExecutionState(request=request, plan=request.execution_plan)
            states.append(state)
            native = adapter.prepare_buffers(state)
            local = {
                item["name"]
                for item in descriptor["bindings"]
                if "executionInput" in item.get("provenance", {})
            }
            buffers = {
                name: replace(
                    buffer,
                    value=buffer.value if index == 0 or name in local else None,
                    source=buffer.source if index == 0 or name in local else None,
                    allocation=(
                        replace(
                            buffer.allocation
                            or RuntimeAllocationView(allocation_id=name),
                            allocation_id=f"region:{index}:{name}",
                        )
                        if name in local
                        else buffer.allocation
                        or RuntimeAllocationView(allocation_id=f"source:{name}")
                    ),
                )
                for name, buffer in native.buffers.items()
            }
            prepared.append(replace(native, buffers=buffers))
        yield tuple(prepared)
    finally:
        for state in reversed(states):
            for directory in reversed(state.temporary_directories):
                directory.cleanup()
