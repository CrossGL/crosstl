"""Dispatch lazily translated small-column reductions of the wide types.

``col_reduce_small`` folds each output column in the accumulator type
``remap_reduce_types`` keeps for int64, uint64 and complex64 numeric
reductions and 16-bit float sums and products, the element type itself;
16-bit float extrema select an input element, and the 8- and 16-bit integers
accumulate as ``wide_reduction_packages.accumulator`` names. The native host passes
upstream's eleven buffers. ``column_small_layout`` checks the column
metadata, source span and launch against ``strided_reduce_small``'s plan
before any package is translated. The kernel reads its threadgroup size, so
each geometry is its own package, and every target runs the launch as one
dispatch. Storage travels as MLX's exact words through ``element_storage``;
the output carries a trailing guard and is decoded exactly.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import column_small_layout
from demos.integrations.mlx.portable_host.column_reduction_layout import LOOPED
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
from demos.integrations.mlx.portable_host.wide_reduction_packages import accumulator

ENTRIES = column_small_layout.WIDE_SMALL_COLUMN_ENTRIES
INPUT, OUTPUT = "in", "out"
METADATA = {name: dtype for name, dtype in LOOPED.items() if dtype}
PREFIX = column_small_layout.ENTRY_PREFIX
BUFFER_COUNT = len(LOOPED)


def signature(entry):
    """The ``(operation, element type)`` of a wide small-column entry."""
    try:
        dtype = ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown wide small-column reduction entry") from error
    operation = entry.removeprefix(PREFIX).split("_reduce_", 1)[1]
    return operation.removesuffix(dtype), dtype


def operand_types(entry):
    """Every source parameter's element type."""
    operation, dtype = signature(entry)
    return {INPUT: dtype, OUTPUT: accumulator(operation, dtype), **METADATA}


def uploads(dtype, target, values, outputs, metadata, output_type=None):
    """The packets one launch binds: inputs, a guarded output and metadata.

    ``output_type`` is the kernel's accumulator type, by default ``dtype``.
    """
    output_type = output_type or dtype
    return {
        INPUT: packet(dtype, target, values),
        OUTPUT: packet(
            output_type,
            target,
            guard_words(output_type, outputs) + guard_words(output_type),
        ),
        **{
            name: packet(kind, target, metadata[name])
            for name, kind in METADATA.items()
        },
    }


def validate(entry, buffers, threads, execution):
    """Check the buffers' types and directions, then upstream's column plan."""
    operation, dtype = signature(entry)
    if set(buffers) != set(operand_types(entry)):
        raise ValueError("Wide small-column buffers do not match their kernel")
    for name, kind in operand_types(entry).items():
        buffer = buffers[name]
        if (
            buffer.dtype != kind.encode("ascii")
            or not buffer.data
            or buffer.data % ctypes.sizeof(CTYPES[kind])
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError(
                "Wide small-column buffer type, alignment or direction differs"
            )
    metadata = column_small_layout.validate(entry, buffers, threads, execution)
    return operation, dtype, metadata


def refuse_subnormal_bfloat(dtype, target, values):
    """Non-Metal bfloat16 parity excludes subnormal inputs (#2000)."""
    if (
        dtype == "bfloat16"
        and target != "metal"
        and any(0 < (word & 0x7FFF) < 0x80 for word in values)
    ):
        raise ValueError(
            f"Subnormal bfloat16 reduction parity is not established for {target}; "
            "see CrossGL/crosstl#2000"
        )


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if (
        host.small_columns is None
        or launch is None
        or not buffers
        or count != BUFFER_COUNT
    ):
        raise ValueError(
            "Wide small-column dispatch requires complete buffers, pinned sources "
            "and launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Wide small-column")
    execution = launch.execution()
    operation, dtype, metadata = validate(entry, supplied, threads, execution)
    target = host.target
    values = buffer_words(supplied[INPUT], dtype)
    outputs = supplied[OUTPUT].count
    refuse_subnormal_bfloat(dtype, target, values)
    output_type = accumulator(operation, dtype)
    packets = uploads(dtype, target, values, outputs, metadata, output_type)
    descriptor, directory = host.small_columns.get(
        column_small_layout.variant(entry, tuple(execution["workgroupSize"]))
    )
    inputs, native_outputs, observed, matched = bind(
        entry, descriptor, supplied, packets, OUTPUT, operand_types(entry)
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
    binding = matched[OUTPUT]
    if result.status != "ok" or set(result.outputs) != {binding}:
        raise RuntimeError(
            "Native wide small-column reduction did not return its output"
        )
    returned, expected = result.outputs[binding], native_outputs[binding]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native wide small-column output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native wide small-column output extent differs")
    words = unpack(output_type, target, words)
    if len(words) != outputs + GUARD_COUNT:
        raise RuntimeError("Native wide small-column output extent differs")
    if words[outputs:] != guard_words(output_type):
        raise RuntimeError(
            "Native wide small-column reduction changed its output guard"
        )
    words = words[:outputs]
    payload = bytes((CTYPES[output_type] * outputs)(*words))
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
            "columnMetadata": metadata,
            "inputs": inputs,
            "operandValues": values,
            "initialValues": guard_words(output_type, outputs),
            "outputValues": words,
            "outputHash": hashlib.sha256(payload).hexdigest(),
        },
    )
    ctypes.memmove(supplied[OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
