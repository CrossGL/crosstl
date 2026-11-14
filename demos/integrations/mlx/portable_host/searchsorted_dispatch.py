"""Dispatch lazily translated searchsorted.metal passes.

The native host passes one pass of ``SearchSorted::eval_gpu``: the sorted
sequence span, at most ``MAX_GROUPS`` row-contiguous values, their uint32
output and the ``n`` and ``a_stride`` metadata. ``searchsorted_layout`` checks
them and the launch before any package is translated. Storage travels as
MLX's exact words through ``element_storage``; the output carries a trailing
guard that must survive and is decoded exactly.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import searchsorted_layout
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

ENTRIES = searchsorted_layout.ENTRIES
PREFIX = searchsorted_layout.PREFIX
INPUTS = ("a", "v")
OUTPUT = "out"
LAYOUT_KEYS = ("dtype", "encoding", "shape")
# A pass that binds more sequence and value elements than this keeps only the
# length and digest of its long lists in the trace.
RETAINED_LIMIT = 1 << 16


def retained(span, count):
    """Whether a pass's trace record keeps its values for the audit."""
    return span + count <= RETAINED_LIMIT


def uploads(entry, target, values, count, metadata):
    """The packets one pass binds; the output starts as guards."""
    types = searchsorted_layout.operand_types(entry)
    return {
        "a": packet(types["a"], target, values["a"]),
        "v": packet(types["v"], target, values["v"]),
        OUTPUT: packet(
            searchsorted_layout.OUTPUT_TYPE,
            target,
            guard_words(searchsorted_layout.OUTPUT_TYPE, count)
            + guard_words(searchsorted_layout.OUTPUT_TYPE),
        ),
        "n": packet("uint32", target, [metadata["n"]]),
        "a_stride": packet("int64", target, [metadata["aStride"]]),
    }


def bind(entry, descriptor, packets):
    """Map reflected bindings to the kernel parameters and their packets."""
    types = searchsorted_layout.operand_types(entry)
    # Direct3D prefixes constants with the entry; "out" is a keyword.
    prefix = entry + "_"
    inputs, outputs, observed, matched = {}, {}, [], {}
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        scalar = binding["scalarLayout"]
        name = scalar.get("memberName", binding["name"]).removeprefix(prefix)
        name = {"out_": OUTPUT}.get(name, name)
        if name not in types or name in matched or binding["name"] in inputs:
            raise ValueError("Reflected searchsorted binding does not match its source")
        matched[name] = binding["name"]
        value = packets[name]
        if not reflected_layout_matches(scalar, value, types[name]):
            raise ValueError(
                "Reflected searchsorted storage differs from the native buffer"
            )
        inputs[binding["name"]] = value
        if name == OUTPUT:
            outputs[binding["name"]] = {k: v for k, v in value.items() if k != "values"}
        else:
            observed.append(binding["name"])
    if set(matched) != set(types):
        raise ValueError(
            "Searchsorted reflection does not cover the complete source contract"
        )
    return inputs, outputs, observed, matched


def output_hash(words):
    """The digest of the output's exact uint32 storage."""
    return hashlib.sha256(bytes((ctypes.c_uint32 * len(words))(*words))).hexdigest()


def _readback(result, native_outputs, binding, count, target):
    if result.status != "ok" or set(result.outputs) != set(native_outputs):
        raise RuntimeError("Native searchsorted did not return its output")
    returned, expected = result.outputs[binding], native_outputs[binding]
    if any(returned.get(key) != expected.get(key) for key in LAYOUT_KEYS):
        raise RuntimeError("Native searchsorted output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native searchsorted output extent differs")
    words = unpack(searchsorted_layout.OUTPUT_TYPE, target, words)
    if len(words) != count + GUARD_COUNT:
        raise RuntimeError("Native searchsorted output extent differs")
    if words[count:] != guard_words(searchsorted_layout.OUTPUT_TYPE):
        raise RuntimeError("Native searchsorted changed its output guard")
    return words[:count]


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    types = searchsorted_layout.operand_types(entry)
    if (
        host.searchsorted is None
        or launch is None
        or not buffers
        or count != len(types)
    ):
        raise ValueError(
            "Searchsorted dispatch requires complete buffers, pinned sources and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Searchsorted")
    execution = launch.execution()
    metadata = searchsorted_layout.validate(entry, supplied, threads, execution)
    target = host.target
    values = {name: buffer_words(supplied[name], types[name]) for name in INPUTS}
    for name, words in values.items():
        refuse_subnormal_bfloat(types[name], target, words)
    packets = uploads(entry, target, values, threads, metadata)
    descriptor, directory = host.searchsorted.get(
        searchsorted_layout.variant(entry, threads)
    )
    inputs, native_outputs, observed, matched = bind(entry, descriptor, packets)
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
    words = _readback(result, native_outputs, matched[OUTPUT], threads, target)
    event = {
        "entry": entry,
        "target": target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "searchMetadata": metadata,
        "inputs": inputs,
        "operandValues": values,
        "outputValues": words,
        "outputHash": output_hash(words),
    }
    if not retained(len(values["a"]), threads):
        event = compact_trace_values(event)
    write_trace(host, event)
    payload = bytes((CTYPES[searchsorted_layout.OUTPUT_TYPE] * threads)(*words))
    ctypes.memmove(supplied[OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
