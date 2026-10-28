"""Enforcement of immutable storage contracts at the native dispatch boundary.

An artifact that relies on a contract treats the named read-only storage as
immutable in its software-subgroup uniformity proofs. A dispatch is therefore
valid only when no writable binding can reach that storage. Allocation views of
unknown length within one allocation count as overlapping.
"""

from __future__ import annotations

from crosstl.translator.artifact_contracts import IMMUTABLE_STORAGE_PREFIX
from crosstl.translator.immutable_storage import (
    binding_immutable_storage,
    parse_immutable_storage_header,
)


def artifact_immutable_bindings(artifact_path, target, bindings):
    """Reconcile descriptor bindings with independent artifact reflection.

    Returns the names of the descriptor bindings the artifact's contract names.
    """
    from .host_reflection import reflect_target_host_interface

    declared = binding_immutable_storage(bindings)
    content = artifact_path.read_bytes()
    if (
        not declared
        and IMMUTABLE_STORAGE_PREFIX.rstrip().encode("ascii") not in content
    ):
        return []
    if target not in {"metal", "directx", "opengl"}:
        raise ValueError("Immutable storage is not supported by this runtime target.")
    records = parse_immutable_storage_header(content.decode("utf-8"))
    if records != declared:
        raise ValueError("Descriptor immutable storage differs from the artifact.")
    reflected = reflect_target_host_interface(
        artifact_path, target=target, stage="compute"
    )
    if reflected["status"] != "ready":
        raise ValueError("Immutable storage requires independently reflected bindings.")
    names = []
    for resource in reflected["resources"]:
        provenance = resource.get("metadata", {}).get("provenance", {})
        if "immutableStorage" not in provenance:
            continue
        matches = [item for item in bindings if item.get("name") == resource["name"]]
        if len(matches) != 1:
            raise ValueError(
                "Immutable storage binding is missing or ambiguous in the descriptor."
            )
        binding = matches[0]
        if (
            any(
                binding.get(key) != resource.get(key)
                for key in ("kind", "type", "access")
            )
            or binding.get("access") != "read"
            or binding.get("coordinates") != {
                "set": resource["set"],
                "binding": resource["binding"],
            }
            or binding.get("provenance", {}).get("immutableStorage")
            != provenance["immutableStorage"]
        ):
            raise ValueError(
                "Immutable storage binding differs from its reflected source resource."
            )
        names.append(resource["name"])
    if len(names) != len(records):
        raise ValueError(
            "Not every immutable storage contract has a reflected binding."
        )
    return sorted(names)


def _views_overlap(left, right):
    """Same allocation, and byte ranges that intersect or are not known."""
    if left[0] != right[0]:
        return False
    if left[2] is None or right[2] is None:
        return True
    return left[1] < right[1] + right[2] and right[1] < left[1] + left[2]


def _value_view(name, value, default_ids):
    allocation = getattr(value, "allocation", None)
    if allocation is None:
        return (default_ids[name], 0, None)
    return (allocation.allocation_id, allocation.byte_offset, allocation.byte_length)


def check_immutable_inputs(names, inputs, outputs, bindings):
    """Each named binding is an uploaded input no writable binding can reach."""
    from .runtime_verification import _runtime_default_allocation_id

    if not names:
        return
    default_ids = {item.name: _runtime_default_allocation_id(item) for item in bindings}
    access = {item.name: item.access for item in bindings}
    writable = {
        name
        for name in [*inputs, *outputs]
        if name in outputs or access.get(name) != "read"
    }
    for name in names:
        if name in outputs or access.get(name) != "read":
            raise ValueError("Immutable storage requires a read-only binding.")
        value = inputs.get(name)
        if value is None:
            raise ValueError("Immutable storage requires an uploaded input value.")
        view = _value_view(name, value, default_ids)
        for other in sorted(writable):
            other_value = outputs.get(other, inputs.get(other))
            if _views_overlap(view, _value_view(other, other_value, default_ids)):
                raise ValueError(
                    f"Immutable storage binding '{name}' shares an allocation range "
                    f"with writable binding '{other}'."
                )


def check_native_immutable_snapshots(artifact_path, target, buffers):
    """Check the native buffers the driver will bind before any dispatch."""
    bindings = [
        dict(
            name=native.binding.name,
            kind=native.binding.kind,
            type=native.binding.type_name,
            access=native.binding.access,
            provenance=native.binding.metadata.get("provenance", {}),
            coordinates={"set": native.binding.set, "binding": native.binding.binding},
        )
        for native in buffers.values()
    ]
    names = artifact_immutable_bindings(artifact_path, target, bindings)

    def view(native):
        allocation = native.allocation
        if allocation is None:
            return (f"binding:{native.binding.name}", 0, None)
        return (
            allocation.allocation_id,
            allocation.byte_offset,
            allocation.byte_length,
        )

    receipts = []
    for name in names:
        matches = [item for item in buffers.values() if item.binding.name == name]
        if len(matches) != 1:
            raise ValueError("Immutable storage does not have one native binding.")
        native = matches[0]
        if (
            native.upload_snapshot is None
            or native.expected_output is not None
            or native.source == "expectedOutput"
        ):
            raise ValueError("Immutable storage requires an uploaded input snapshot.")
        for other in buffers.values():
            if other is native:
                continue
            if (
                other.binding.access != "read" or other.expected_output is not None
            ) and _views_overlap(view(native), view(other)):
                raise ValueError(
                    f"Immutable storage binding '{name}' shares an allocation range "
                    f"with writable binding '{other.binding.name}'."
                )
        receipts.append(
            {
                "binding": name,
                "requirement": native.binding.metadata["provenance"][
                    "immutableStorage"
                ],
                "uploadSnapshot": native.upload_snapshot.to_json(),
                **(
                    {"allocation": native.allocation.to_json()}
                    if native.allocation is not None
                    else {}
                ),
            }
        )
    return receipts
