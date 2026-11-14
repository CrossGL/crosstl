"""Dispatch the translated ``masked_assign`` kernels of masked scatters.

``MaskedScatter`` binds the flattened mask, its exclusive per-batch prefix
counts, the source, the output, the source's shape, strides and rank and the
source and mask batch sizes, as ``mlx/backend/metal/indexing.cpp`` binds them,
in one-thread workgroups over one thread per mask element. The output uploads
the destination copy it already holds, because unset elements keep it, and
carries a trailing guard. ``masked_scatter_evidence`` audits each dispatch.
"""

import ctypes
import hashlib
import math

from crosstl.project import build_native_loader_dispatch_request
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
from demos.integrations.mlx.portable_host.masked_scatter_packages import (
    ENTRIES,
    SIGNED_MAX,
    signature,
)
from demos.integrations.mlx.portable_host.trace_records import write_trace

__all__ = (
    "ENTRIES",
    "MAX_ELEMENTS",
    "MAX_RANK",
    "dispatch",
    "launch_for",
    "operand_types",
    "source_location",
    "validate",
)

OUTPUT = "out"
# backend.cpp's max_masked_scatter_elements and the source rank it binds.
MAX_ELEMENTS = 65535
MAX_RANK = 64
# The contiguous kernel's int64 source index, which OpenGL lowers to a signed
# int (masked_scatter_packages.OPENGL_INDEX_ASSERTIONS), stays below the span.
MAX_SOURCE_ELEMENTS = SIGNED_MAX
SCALARS = ("src_ndim", "src_batch_size", "mask_batch_size")


def operand_types(entry):
    """Every source parameter's element type."""
    dtype, _ = signature(entry)
    return {
        "mask": "bool_",
        "scatter_offsets": "uint32",
        "src": dtype,
        OUTPUT: dtype,
        "src_shapes": "int32",
        "src_strides": "int64",
        "src_ndim": "int32",
        "src_batch_size": "int64",
        "mask_batch_size": "int64",
    }


def launch_for(threads):
    """indexing.cpp's grid of one thread per mask element, as one-thread groups."""
    return {"workgroupCount": [threads, 1, 1], "workgroupSize": [1, 1, 1]}


def row_contiguous(shape, strides):
    """MLX's row contiguity, which ignores the strides of size-one axes."""
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if size != 1 and stride != expected:
            return False
        expected *= size
    return True


def source_location(element, shape, strides, contiguous):
    """The source element ``masked_assign_impl`` reads for flat ``element``.

    ``elem_to_loc<uint>`` evaluates in 32-bit unsigned arithmetic; the
    validated span keeps every location below 2**31.
    """
    if contiguous:
        return element
    location = 0
    for size, stride in zip(reversed(shape), reversed(strides)):
        if element <= 0:
            break
        location = (location + (element % size) * stride) % (1 << 32)
        element //= size
    return location


def validate(entry, buffers, threads, execution):
    """Check one launch and return the operands it reads."""
    dtype, contiguous = signature(entry)
    types = operand_types(entry)
    if set(buffers) != set(types):
        raise ValueError("Masked scatter buffers do not match their kernel")
    for name, buffer in buffers.items():
        size = ctypes.sizeof(CTYPES[types[name]])
        if (
            buffer.dtype != types[name].encode("ascii")
            or not buffer.data
            or buffer.data % size
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError(
                "Masked scatter buffer type, alignment or direction differs"
            )
    if (
        type(threads) is not int
        or not 0 < threads <= MAX_ELEMENTS
        or execution != launch_for(threads)
        or any(buffers[name].count != threads for name in ("mask", "scatter_offsets"))
        or buffers[OUTPUT].count != threads
        or any(buffers[name].count != 1 for name in SCALARS)
    ):
        raise ValueError("Masked scatter launch differs from one thread per element")
    (rank,) = buffer_words(buffers["src_ndim"], "int32")
    (src_batch,) = buffer_words(buffers["src_batch_size"], "int64")
    (mask_batch,) = buffer_words(buffers["mask_batch_size"], "int64")
    if (
        not 0 < rank <= MAX_RANK
        or buffers["src_shapes"].count != rank
        or buffers["src_strides"].count != rank
    ):
        raise ValueError("Masked scatter source rank differs from its layout")
    shape = buffer_words(buffers["src_shapes"], "int32")
    strides = buffer_words(buffers["src_strides"], "int64")
    span = 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    if (
        any(size < 1 for size in shape)
        or any(stride < 0 for stride in strides)
        or not 0 < span <= MAX_SOURCE_ELEMENTS
        or buffers["src"].count != span
        or src_batch * shape[0] != math.prod(shape)
        or mask_batch * shape[0] != threads
        or contiguous != row_contiguous(shape, strides)
    ):
        raise ValueError("Masked scatter source layout or batches are invalid")
    mask = buffer_words(buffers["mask"], "bool_")
    offsets = buffer_words(buffers["scatter_offsets"], "uint32")
    end = buffers[OUTPUT].data + threads * ctypes.sizeof(CTYPES[dtype])
    for name, buffer in buffers.items():
        start = buffer.data
        stop = start + buffer.count * ctypes.sizeof(CTYPES[types[name]])
        if name != OUTPUT and max(start, buffers[OUTPUT].data) < min(stop, end):
            raise ValueError("Masked scatter output overlaps an input allocation")
    return {
        "dtype": dtype,
        "sourceContiguous": contiguous,
        "mask": mask,
        "scatterOffsets": offsets,
        "sourceShape": shape,
        "sourceStrides": strides,
        "sourceBatchSize": src_batch,
        "maskBatchSize": mask_batch,
        "sourceSpan": span,
    }


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.masked_scatters is None or launch is None or not buffers or count != 9:
        raise ValueError(
            "Masked scatter dispatch requires complete buffers, pinned sources and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Masked scatter")
    execution = launch.execution()
    metadata = validate(entry, supplied, threads, execution)
    dtype, target = metadata["dtype"], host.target
    types = operand_types(entry)
    source = buffer_words(supplied["src"], dtype)
    initial, tail = buffer_words(supplied[OUTPUT], dtype), guard_words(dtype)
    words = {
        "mask": metadata["mask"],
        "scatter_offsets": metadata["scatterOffsets"],
        "src": source,
        "src_shapes": metadata["sourceShape"],
        "src_strides": metadata["sourceStrides"],
        "src_ndim": [len(metadata["sourceShape"])],
        "src_batch_size": [metadata["sourceBatchSize"]],
        "mask_batch_size": [metadata["maskBatchSize"]],
    }
    uploads = {
        **{name: packet(types[name], target, value) for name, value in words.items()},
        OUTPUT: packet(dtype, target, initial + tail),
    }
    descriptor, directory = host.masked_scatters.get(entry)
    inputs, outputs, observed, matched = bind(
        entry, descriptor, supplied, uploads, OUTPUT, types
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
        raise RuntimeError("Native masked scatter kernel did not return its output")
    returned, expected = result.outputs[binding], outputs[binding]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ) or not isinstance(returned.get("values"), list):
        raise RuntimeError("Native masked scatter output layout differs")
    try:
        values = unpack(dtype, target, returned["values"])
    except ValueError as error:
        raise RuntimeError(
            "Native masked scatter output is outside its storage type"
        ) from error
    if len(values) != threads + GUARD_COUNT or values[threads:] != tail:
        raise RuntimeError("Native masked scatter kernel changed its output guard")
    payload = bytes((CTYPES[dtype] * threads)(*values[:threads]))
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
            "maskedScatterMetadata": metadata,
            "inputs": inputs,
            "sourceValues": source,
            "initialValues": initial,
            "outputValues": values[:threads],
            "outputHash": hashlib.sha256(payload).hexdigest(),
        },
    )
    ctypes.memmove(supplied[OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
