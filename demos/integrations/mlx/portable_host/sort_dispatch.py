"""Dispatch lazily translated MLX merge-sort stages.

The native host passes every buffer ``single_block_sort`` or
``multi_block_sort`` binds for one stage: the values (and, past the first
multi-block stage, their indices and the tile partitions), the stage's
outputs and its ``int`` and rank-array metadata. ``sort_layout`` checks them
and the launch against ``gpu_merge_sort`` before any package is translated.
Storage travels as MLX's exact words through ``element_storage``; every
output carries a trailing guard that must survive and is decoded exactly.

A launch that sorts more than ``RETAINED_LIMIT`` values keeps only the length
and digest of its long value lists in the trace.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import sort_layout
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    buffer_words,
    guard_words,
    packet,
    reflected_layout_matches,
    supplied_buffers,
    unpack,
)
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.trace_records import (
    compact_trace_values,
    write_trace,
)
from demos.integrations.mlx.portable_host.wide_column_dispatch import (
    refuse_subnormal_bfloat,
)

ENTRIES = sort_layout.ENTRIES
ELEMENTS, ARRAYS = sort_layout.ELEMENTS, sort_layout.ARRAYS
# Entry prefixes whose trace records keep their values for the audit.
PREFIXES = (
    "c_block_sort_",
    "carg_block_sort_",
    "nc_block_sort_",
    "ncarg_block_sort_",
    *(f"{stage}_mbsort_" for stage in sort_layout.STAGES),
)
RETAINED_LIMIT = 1 << 16
LAYOUT_KEYS = ("dtype", "encoding", "shape")


def inputs_of(entry):
    """The element parameters ``entry`` reads without writing."""
    written = sort_layout.outputs(entry)
    return tuple(
        name
        for name in sort_layout.operand_types(entry)
        if name in ELEMENTS and name not in written
    )


def uploads(entry, target, values, counts, metadata):
    """The packets one launch binds.

    ``values`` holds every read element buffer's words, ``counts`` every
    output's extent; outputs start as guards with a trailing guard.
    """
    written = sort_layout.outputs(entry)
    packets = {}
    for name, kind in sort_layout.operand_types(entry).items():
        if name in written:
            words = guard_words(kind, counts[name]) + guard_words(kind)
        elif name in ELEMENTS:
            words = values[name]
        elif name in ARRAYS:
            words = metadata[name]
        else:
            words = [metadata[name]]
        packets[name] = packet(kind, target, words)
    return packets


def bind(entry, descriptor, supplied, packets, types):
    """Map reflected bindings to the kernel parameters and their packets."""
    # Direct3D prefixes constants with the entry; "out" is a keyword.
    prefix = entry.rstrip("_") + "_"
    written = sort_layout.outputs(entry)
    inputs, outputs, observed, matched = {}, {}, [], {}
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        scalar = binding["scalarLayout"]
        name = scalar.get("memberName", binding["name"]).removeprefix(prefix)
        name = {"out_": "out"}.get(name, name)
        if name not in supplied or name in matched or binding["name"] in inputs:
            raise ValueError("Reflected sort binding does not match its source")
        matched[name] = binding["name"]
        value = packets[name]
        if not reflected_layout_matches(scalar, value, types[name]):
            raise ValueError("Reflected sort storage differs from the native buffer")
        inputs[binding["name"]] = value
        if name in written:
            outputs[binding["name"]] = {k: v for k, v in value.items() if k != "values"}
        else:
            observed.append(binding["name"])
    if set(matched) != set(supplied):
        raise ValueError("Sort reflection does not cover the complete source contract")
    return inputs, outputs, observed, matched


def output_hash(entry, results):
    """The digest of every output's exact storage, in parameter order."""
    types = sort_layout.operand_types(entry)
    digest = hashlib.sha256()
    for name in sort_layout.outputs(entry):
        words = results[name]
        digest.update(bytes((CTYPES[types[name]] * len(words))(*words)))
    return digest.hexdigest()


def _readback(result, native_outputs, matched, types, counts, target):
    if result.status != "ok" or set(result.outputs) != set(native_outputs):
        raise RuntimeError("Native sort did not return its outputs")
    results = {}
    for name, count in counts.items():
        returned = result.outputs[matched[name]]
        expected = native_outputs[matched[name]]
        if any(returned.get(key) != expected.get(key) for key in LAYOUT_KEYS):
            raise RuntimeError("Native sort output layout differs")
        words = returned.get("values")
        if not isinstance(words, list):
            raise RuntimeError("Native sort output extent differs")
        words = unpack(types[name], target, words)
        if len(words) != count + GUARD_COUNT:
            raise RuntimeError("Native sort output extent differs")
        if words[count:] != guard_words(types[name]):
            raise RuntimeError("Native sort changed its output guard")
        results[name] = words[:count]
    return results


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    types = sort_layout.operand_types(entry)
    if host.sorts is None or launch is None or not buffers or count != len(types):
        raise ValueError(
            "Sort dispatch requires complete buffers, pinned sources and launch "
            "geometry"
        )
    supplied = supplied_buffers(buffers, count, "Sort")
    execution = launch.execution()
    metadata = sort_layout.validate(entry, supplied, threads, execution)
    target = host.target
    values = {
        name: buffer_words(supplied[name], types[name]) for name in inputs_of(entry)
    }
    for name, words in values.items():
        refuse_subnormal_bfloat(types[name], target, words)
    counts = {name: supplied[name].count for name in sort_layout.outputs(entry)}
    packets = uploads(entry, target, values, counts, metadata)
    descriptor, directory = host.sorts.get(
        sort_layout.variant(entry, execution["workgroupSize"][0])
    )
    inputs, native_outputs, observed, matched = bind(
        entry, descriptor, supplied, packets, types
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        inputs,
        native_outputs,
        execution,
        expected_target=target,
        observe_inputs=tuple(observed),
    )
    result = execute(host, request)
    results = _readback(result, native_outputs, matched, types, counts, target)
    event = {
        "entry": entry,
        "target": target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "sortMetadata": metadata,
        "inputs": inputs,
        "operandValues": values,
        "outputValues": results,
        "outputHash": output_hash(entry, results),
    }
    if threads > RETAINED_LIMIT:
        event = compact_trace_values(event)
    write_trace(host, event)
    for name, words in results.items():
        payload = bytes((CTYPES[types[name]] * len(words))(*words))
        ctypes.memmove(supplied[name].data, payload, len(payload))
    host.dispatch_count += 1
