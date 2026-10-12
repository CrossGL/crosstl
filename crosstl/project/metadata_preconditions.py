"""Enforcement of source metadata obligations at the native dispatch boundary."""

from crosstl.translator.metadata_preconditions import (
    INTEGER_METADATA_BYTES,
    METADATA_PRECONDITION_PREFIX,
    binding_metadata_preconditions,
    parse_metadata_precondition_header,
)


def artifact_metadata_bindings(artifact_path, target, bindings):
    """Reconcile descriptor resources with independent artifact reflection."""
    from .host_reflection import reflect_target_host_interface

    declared = binding_metadata_preconditions(bindings)
    content = artifact_path.read_bytes()
    if (
        not declared
        and METADATA_PRECONDITION_PREFIX.rstrip().encode("ascii") not in content
    ):
        return {}
    if target not in {"metal", "directx", "opengl"}:
        raise ValueError(
            "Metadata preconditions are not supported by this runtime target."
        )
    source = content.decode("utf-8")
    records = parse_metadata_precondition_header(source)
    if records != declared:
        raise ValueError("Descriptor metadata preconditions differ from the artifact.")
    if not records:
        return {}
    reflected = reflect_target_host_interface(
        artifact_path, target=target, stage="compute"
    )
    if reflected["status"] != "ready":
        raise ValueError(
            "Metadata preconditions require independently reflected bindings."
        )
    result = {}
    for resource in reflected["resources"]:
        provenance = resource.get("metadata", {}).get("provenance", {})
        requirement = provenance.get("metadataPrecondition")
        if requirement is None:
            continue
        matches = [item for item in bindings if item.get("name") == resource["name"]]
        if len(matches) != 1:
            raise ValueError(
                "Metadata binding is missing or ambiguous in the descriptor."
            )
        binding = matches[0]
        if (
            any(
                binding.get(key) != resource.get(key)
                for key in ("kind", "type", "access", "scalarLayout")
            )
            or binding.get("coordinates") != {
                "set": resource["set"],
                "binding": resource["binding"],
            }
            or binding.get("provenance", {}).get("sourceResource")
            != provenance["sourceResource"]
            or binding.get("provenance", {}).get("metadataPrecondition") != requirement
        ):
            raise ValueError(
                "Metadata binding differs from its reflected source resource."
            )
        result[resource["name"]] = requirement
    if len(result) != len(records):
        raise ValueError("Not every metadata precondition has a reflected binding.")
    return result


def check_metadata_value(requirement, value, dtype):
    """Reject instead of converting, clamping or silently skipping a dispatch."""
    if dtype != requirement["dtype"] or type(value) is not int:
        raise ValueError("Metadata input requires its exact reflected integer type.")
    if not requirement["minimum"] <= value <= requirement["maximum"]:
        origin = requirement["sourceResource"]
        raise ValueError(
            f"Metadata {origin['entryPoint']}.{origin['parameter']} value {value} "
            f"is outside [{requirement['minimum']}, {requirement['maximum']}]."
        )


def check_metadata_allocation(name, value, inputs, outputs, default_ids):
    """A host-checked scalar cannot share an allocation with another binding.

    This deliberately rejects even disjoint views until cross-dispatch ownership
    and GPU-produced metadata have a separate validation protocol.
    """
    if name in outputs:
        raise ValueError("Metadata preconditions require read-only uploaded inputs.")
    allocation_id = (
        value.allocation.allocation_id
        if value.allocation is not None
        else default_ids[name]
    )
    for other_name, other in [*inputs.items(), *outputs.items()]:
        other_id = (
            other.allocation.allocation_id
            if other.allocation is not None
            else default_ids[other_name]
        )
        if other_name != name and other_id == allocation_id:
            raise ValueError("Checked metadata cannot share a native allocation.")


def check_metadata_inputs(requirements, inputs, outputs, bindings):
    from .runtime_verification import _runtime_default_allocation_id

    default_ids = {item.name: _runtime_default_allocation_id(item) for item in bindings}
    for name, requirement in requirements.items():
        value = inputs.get(name)
        if value is None:
            raise ValueError("Checked metadata requires an uploaded scalar input.")
        words = value.values
        if (
            value.encoding is not None
            or not isinstance(words, (list, tuple))
            or not words
        ):
            raise ValueError(
                "Checked metadata requires a nonempty typed scalar payload."
            )
        check_metadata_value(requirement, words[0], value.dtype)
        check_metadata_allocation(name, value, inputs, outputs, default_ids)


def check_native_metadata_snapshots(artifact_path, target, buffers):
    """Check the immutable payloads that the native driver will actually upload."""
    bindings = []
    for native in buffers.values():
        resource = native.binding
        bindings.append(
            dict(
                name=resource.name,
                kind=resource.kind,
                type=resource.type_name,
                access=resource.access,
                scalarLayout=resource.metadata.get("scalarLayout"),
                provenance=resource.metadata.get("provenance", {}),
                coordinates={"set": resource.set, "binding": resource.binding},
            )
        )
    requirements = artifact_metadata_bindings(artifact_path, target, bindings)
    receipts = []
    for name, requirement in requirements.items():
        matches = [item for item in buffers.values() if item.binding.name == name]
        if len(matches) != 1:
            raise ValueError("Checked metadata does not have one native binding.")
        native = matches[0]
        snapshot = native.upload_snapshot
        width = INTEGER_METADATA_BYTES[requirement["dtype"]]
        if (
            snapshot is None
            or type(snapshot.payload) is not bytes
            or snapshot.dtype != requirement["dtype"]
            or snapshot.encoding is not None
            or len(snapshot.payload) < width
            or native.expected_output is not None
            or native.source == "expectedOutput"
        ):
            raise ValueError(
                "Checked metadata requires an immutable typed upload snapshot."
            )

        def allocation_id(binding):
            return (
                binding.allocation.allocation_id
                if binding.allocation is not None
                else f"binding:{binding.name}"
            )

        if any(
            other is not native and allocation_id(other) == allocation_id(native)
            for other in buffers.values()
        ):
            raise ValueError("Checked metadata cannot share a native allocation.")
        value = int.from_bytes(
            snapshot.payload[:width],
            "little",
            signed=requirement["dtype"].startswith("int"),
        )
        check_metadata_value(requirement, value, snapshot.dtype)
        receipts.append(
            {
                "binding": name,
                "requirement": requirement,
                "value": value,
                "uploadSnapshot": snapshot.to_json(),
                **(
                    {"allocation": native.allocation.to_json()}
                    if native.allocation is not None
                    else {}
                ),
            }
        )
    return receipts
