"""Dispatch lazily translated small-row reductions of the wide types.

``row_reduce_small`` folds each output's rows in the accumulator type
``remap_reduce_types`` keeps for int64, uint64 and complex64 numeric
reductions and 16-bit float sums and products, the element type itself, and
for the 8- and 16-bit integers: their own type for extrema and 32 bits for
sums and products (``wide_reduction_packages.accumulator``). The native host
passes upstream's ten buffers. ``row_reduction_layout`` checks
the row metadata, source span and launch against upstream's plan before any
package is translated. Storage travels as MLX's exact words through
``element_storage``; the output carries a trailing guard and is decoded
exactly. Direct3D and OpenGL run the launch as its dispatch regions.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import region_dispatch, row_reduction_layout
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
from demos.integrations.mlx.portable_host.small_row_packages import (
    WIDE_SMALL_ROW_ENTRIES as ENTRIES,
)
from demos.integrations.mlx.portable_host.trace_records import write_trace
from demos.integrations.mlx.portable_host.wide_reduction_packages import accumulator

INPUT, OUTPUT = "in", "out"
METADATA = {name: dtype for name, dtype in row_reduction_layout.LOOPED.items() if dtype}
PREFIX = "row_reduce_small_"


def signature(entry):
    """The ``(operation, element type)`` of a wide small-row entry."""
    try:
        dtype = ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown wide small-row reduction entry") from error
    operation = entry.removeprefix(PREFIX).split("_reduce_", 1)[1]
    return operation.removesuffix(dtype), dtype


def operand_types(entry):
    """Every source parameter's element type."""
    operation, dtype = signature(entry)
    return {INPUT: dtype, OUTPUT: accumulator(operation, dtype), **METADATA}


def uploads(dtype, target, values, rows, metadata, output_type=None):
    """The packets one launch binds: inputs, a guarded output and metadata.

    ``output_type`` is the kernel's accumulator type, by default ``dtype``.
    """
    output_type = output_type or dtype
    return {
        INPUT: packet(dtype, target, values),
        OUTPUT: packet(
            output_type,
            target,
            guard_words(output_type, rows) + guard_words(output_type),
        ),
        **{
            name: packet(kind, target, metadata[name])
            for name, kind in METADATA.items()
        },
    }


def validate(entry, buffers, threads, execution):
    """Check the buffers' types and directions, then upstream's row plan."""
    operation, dtype = signature(entry)
    if set(buffers) != set(operand_types(entry)):
        raise ValueError("Wide small-row buffers do not match their kernel")
    for name, kind in operand_types(entry).items():
        buffer = buffers[name]
        if (
            buffer.dtype != kind.encode("ascii")
            or not buffer.data
            or buffer.data % ctypes.sizeof(CTYPES[kind])
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError(
                "Wide small-row buffer type, alignment or direction differs"
            )
    metadata = row_reduction_layout.validate(entry, buffers, threads, execution)
    return operation, dtype, metadata


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.small_rows is None or launch is None or not buffers or count != 10:
        raise ValueError(
            "Wide small-row dispatch requires complete buffers, pinned sources "
            "and launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Wide small-row")
    execution = launch.execution()
    operation, dtype, metadata = validate(entry, supplied, threads, execution)
    target = host.target
    values = buffer_words(supplied[INPUT], dtype)
    rows = supplied[OUTPUT].count
    if (
        dtype == "bfloat16"
        and target != "metal"
        and any(0 < (word & 0x7FFF) < 0x80 for word in values)
    ):
        raise ValueError(
            f"Subnormal bfloat16 reduction parity is not established for {target}; "
            "see CrossGL/crosstl#2000"
        )
    output_type = accumulator(operation, dtype)
    packets = uploads(dtype, target, values, rows, metadata, output_type)
    regions = host.small_rows.get(
        entry,
        thread_grid_size=execution["threadGridSize"],
        workgroup_size=execution["workgroupSize"],
    )
    descriptor, directory = regions[0]
    inputs, outputs, observed, matched = bind(
        entry, descriptor, supplied, packets, OUTPUT, operand_types(entry)
    )
    if target == "metal":
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
    else:
        result = region_dispatch.run_regions(host, regions, inputs, outputs, execution)
    binding = matched[OUTPUT]
    if result.status != "ok" or set(result.outputs) != {binding}:
        raise RuntimeError("Native wide small-row reduction did not return its output")
    returned, expected = result.outputs[binding], outputs[binding]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native wide small-row output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native wide small-row output extent differs")
    words = unpack(output_type, target, words)
    if len(words) != rows + GUARD_COUNT:
        raise RuntimeError("Native wide small-row output extent differs")
    if words[rows:] != guard_words(output_type):
        raise RuntimeError("Native wide small-row reduction changed its output guard")
    words = words[:rows]
    payload = bytes((CTYPES[output_type] * rows)(*words))
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
            "rowMetadata": metadata,
            "inputs": inputs,
            "operandValues": values,
            "initialValues": guard_words(output_type, rows),
            "outputValues": words,
            "outputHash": hashlib.sha256(payload).hexdigest(),
        },
    )
    ctypes.memmove(supplied[OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
