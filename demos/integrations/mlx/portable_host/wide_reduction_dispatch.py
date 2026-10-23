"""Dispatch lazily translated wide whole-array reductions.

The native host passes one pass of upstream's ``all_reduce`` plan: ``in``
holds ``threads`` elements, ``out`` one element per row, and ``in_size`` and
``row_size`` are the plan's uint64 metadata. Two-pass reductions call this
once per pass. Storage travels as MLX's exact words through
``element_storage``; the output carries a trailing guard and is decoded
exactly.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import reduction_layout
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    buffer_words,
    guard_words,
    packet,
    supplied_buffers,
    unpack,
)
from demos.integrations.mlx.portable_host.elementwise_dispatch import bind
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.trace_records import write_trace
from demos.integrations.mlx.portable_host.wide_reduction_packages import (
    dispatch_contract,
    signature,
)

METADATA = ("in_size", "row_size")
INPUT, OUTPUT = "in", "out"


def operand_types(entry):
    """Every source parameter's element type."""
    _, dtype = signature(entry)
    return {INPUT: dtype, OUTPUT: dtype, **{name: "uint64" for name in METADATA}}


def validate(entry, buffers, threads, execution):
    """Check one pass against upstream's plan for ``threads`` input elements."""
    operation, dtype = signature(entry)
    if set(buffers) != {INPUT, OUTPUT, *METADATA}:
        raise ValueError("Wide reduction buffers do not match their kernel")
    contract = dispatch_contract(entry, threads, execution)
    rows = contract["execution"]["workgroupCount"][1]
    plan = {"rowSize": contract["rowSize"], **contract["execution"]}
    if reduction_layout.max_rebased_offset(plan) > reduction_layout.MAX_ELEMENTS:
        raise ValueError("Wide reduction offsets exceed the signed index bound")
    types, extents = operand_types(entry), {INPUT: threads, OUTPUT: rows}
    for name, buffer in buffers.items():
        size = ctypes.sizeof(CTYPES[types[name]])
        if (
            buffer.dtype != types[name].encode("ascii")
            or not buffer.data
            or buffer.data % size
            or buffer.count != extents.get(name, 1)
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError(
                "Wide reduction buffer type, extent, alignment or direction differs"
            )
    metadata = {name: buffer_words(buffers[name], "uint64")[0] for name in METADATA}
    if metadata != {"in_size": threads, "row_size": contract["rowSize"]}:
        raise ValueError("Wide reduction metadata does not match the upstream plan")
    return {"operation": operation, "dtype": dtype, "rows": rows, **metadata}


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.wide_reductions is None or launch is None or not buffers or count != 4:
        raise ValueError(
            "Wide reduction dispatch requires complete buffers, pinned sources and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Wide reduction")
    execution = launch.execution()
    metadata = validate(entry, supplied, threads, execution)
    dtype, rows, target = metadata["dtype"], metadata["rows"], host.target
    values = buffer_words(supplied[INPUT], dtype)
    if (
        dtype == "bfloat16"
        and target != "metal"
        and any(0 < (word & 0x7FFF) < 0x80 for word in values)
    ):
        raise ValueError(
            f"Subnormal bfloat16 reduction parity is not established for {target}; "
            "see CrossGL/crosstl#2000"
        )
    # A fresh output is filled with guards, so the readback is never stale.
    initial = guard_words(dtype, rows)
    tail = guard_words(dtype)
    uploads = {
        INPUT: packet(dtype, target, values),
        OUTPUT: packet(dtype, target, initial + tail),
        **{
            name: {"dtype": "uint64", "shape": [1], "values": [metadata[name]]}
            for name in METADATA
        },
    }
    descriptor, directory = host.wide_reductions.get(entry, threads, execution)
    inputs, outputs, observed, matched = bind(
        entry, descriptor, supplied, uploads, OUTPUT, operand_types(entry)
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        inputs,
        outputs,
        execution,
        expected_target=target,
        observe_inputs=tuple(observed),
    )
    result = execute(host, request)
    binding = matched[OUTPUT]
    if result.status != "ok" or set(result.outputs) != {binding}:
        raise RuntimeError("Native wide reduction did not return its output")
    returned, expected = result.outputs[binding], outputs[binding]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native wide reduction output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native wide reduction output extent differs")
    words = unpack(dtype, target, words)
    if len(words) != rows + GUARD_COUNT:
        raise RuntimeError("Native wide reduction output extent differs")
    if words[rows:] != tail:
        raise RuntimeError("Native wide reduction changed its output guard")
    words = words[:rows]
    payload = bytes((CTYPES[dtype] * rows)(*words))
    write_trace(
        host,
        {
            "entry": entry,
            "target": target,
            "threads": threads,
            **execution,
            "dispatchVersion": DISPATCH_VERSION,
            "artifact": descriptor["artifact"],
            "packageRoot": str(directory),
            "details": result.details,
            "reductionMetadata": metadata,
            "inputs": inputs,
            "operandValues": values,
            "initialValues": initial,
            "outputValues": words,
            "outputHash": hashlib.sha256(payload).hexdigest(),
        },
    )
    ctypes.memmove(supplied[OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
