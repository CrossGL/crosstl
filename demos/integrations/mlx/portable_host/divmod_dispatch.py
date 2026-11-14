"""Dispatch checked DivMod buffers through translated ``binary_two`` kernels.

The native host passes the two operands, both outputs and the launch geometry
that ``binary_op_gpu`` would use. Every buffer is checked against its kernel's
source contract before upload, each output carries a trailing guard, and the
readback is decoded exactly into the logical MLX storage.
"""

import ctypes
import hashlib
import math

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
)
from demos.integrations.mlx.portable_host import bfloat_storage, half_storage
from demos.integrations.mlx.portable_host.divmod_packages import KERNELS, signature
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.trace_records import write_trace

ENTRY_PREFIXES = tuple(f"{kernel}_DivMod" for kernel in KERNELS)
# Each output element and input span stays below the OpenGL signed index bound.
MAX_ELEMENTS = 1 << 20
GUARD_COUNT = 32
CTYPES = {
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "float32": ctypes.c_uint32,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}
METADATA_CTYPES = {
    "uint32": ctypes.c_uint32,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}
# Narrow integers widen to 32-bit storage words where the target lacks them.
WIDENED = {
    "metal": {},
    "directx": {"uint8": "uint32", "int8": "int32"},
    "opengl": {
        "uint8": "uint32",
        "uint16": "uint32",
        "int8": "int32",
        "int16": "int32",
    },
}
PHYSICAL_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "uint16": 2,
    "int16": 2,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "uint64": 8,
    "int64": 8,
}
GUARDS = {
    "float16": 0x3555,
    "bfloat16": 0x3EAB,
    "float32": 0x6A15BEEF,
    "uint8": 0xA5,
    "uint16": 0xA5A5,
    "uint32": 0x6A15BEEF,
    "uint64": 0x6A15BEEF6A15BEEF,
    "int8": -91,
    "int16": 0x3555,
    "int32": 0x6A15BEEF,
    "int64": -0x6A15BEEF,
}


def metadata_types(kernel, ndim):
    """Return each scalar parameter's dtype and element count for ``kernel``."""
    if kernel == "ss":
        return {}
    if kernel in {"sv", "vs", "vv"}:
        return {"size": ("uint32", 1)}
    if kernel == "g1":
        return {"a_stride": ("int64", 1), "b_stride": ("int64", 1)}
    if kernel in {"g2", "g3"}:
        rank = int(kernel[1])
        return {"a_strides": ("int64", rank), "b_strides": ("int64", rank)}
    return {
        "shape": ("int32", ndim),
        "a_strides": ("int64", ndim),
        "b_strides": ("int64", ndim),
        "ndim": ("int32", 1),
    }


def buffer_values(buffer, ctype):
    return list(ctypes.cast(buffer.data, ctypes.POINTER(ctype * buffer.count)).contents)


def general_launch(kernel, shape):
    """``binary_op_gpu``'s general grid: the last two axes and the rest."""
    dim0 = shape[-1]
    dim1 = shape[-2] if len(shape) > 1 else 1
    rest = math.prod(shape) // (dim0 * dim1)
    if kernel == "gn2":
        dim0 = (dim0 + 1) // 2
    return [dim0, dim1, rest]


VECTOR_STRIDES = {"ss": (0, 0), "sv": (0, 1), "vs": (1, 0), "vv": (1, 1)}


def layout(kernel, buffers, threads, execution):
    """Return the logical shape and operand strides the kernel will read."""
    if kernel in VECTOR_STRIDES:
        a_stride, b_stride = VECTOR_STRIDES[kernel]
        return [threads], [a_stride], [b_stride]
    rank = buffers["shape"].count if kernel == "gn2" else 0
    values = {
        name: buffer_values(buffers[name], METADATA_CTYPES[dtype])
        for name, (dtype, _) in metadata_types(kernel, rank).items()
    }
    if kernel == "g1":
        return [threads], values["a_stride"], values["b_stride"]
    count = execution["workgroupCount"]
    if kernel in {"g2", "g3"}:
        # The 2-D and 3-D kernels take their shape from the grid, last axis first.
        shape = count[: int(kernel[1])][::-1]
        return shape, values["a_strides"], values["b_strides"]
    shape = values["shape"]
    if values["ndim"] != [len(shape)] or len(shape) < 4:
        raise ValueError("DivMod general rank does not match its shape")
    return shape, values["a_strides"], values["b_strides"]


def validate(entry, buffers, threads, execution):
    kernel, dtype = signature(entry)
    if type(threads) is not int or not 0 < threads <= MAX_ELEMENTS:
        raise ValueError("DivMod output size exceeds the host contract")
    rank = buffers["shape"].count if "shape" in buffers else 0
    types = {name: (dtype, None) for name in "abcd"}
    types.update(metadata_types(kernel, rank))
    if set(buffers) != set(types):
        raise ValueError("DivMod buffers do not match their kernel")
    pointer_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    ranges = {}
    for name, buffer in buffers.items():
        kind, count = types[name]
        size = ctypes.sizeof(CTYPES.get(kind) or METADATA_CTYPES[kind])
        if (
            buffer.dtype != kind.encode("ascii")
            or not buffer.data
            or buffer.data % size
            or not 0 < buffer.count <= MAX_ELEMENTS
            or buffer.data + buffer.count * size > pointer_limit
            or (count is not None and buffer.count != count)
            or buffer.output != int(name in {"c", "d"})
        ):
            raise ValueError(
                "DivMod buffer type, extent, alignment or direction differs"
            )
        ranges[name] = (buffer.data, buffer.data + buffer.count * size)
    shape, a_strides, b_strides = layout(kernel, buffers, threads, execution)
    grid = (
        [threads, 1, 1]
        if kernel in {*VECTOR_STRIDES, "g1"}
        else general_launch(kernel, shape)
    )
    if (
        execution != {"workgroupCount": grid, "workgroupSize": [1, 1, 1]}
        or any(size < 1 for size in shape)
        or math.prod(shape) != threads
        or len(a_strides) != len(shape)
        or len(b_strides) != len(shape)
    ):
        raise ValueError("DivMod launch does not match its output layout")
    for name, strides in (("a", a_strides), ("b", b_strides)):
        if any(stride < 0 for stride in strides):
            raise ValueError("DivMod general strides must be nonnegative")
        span = 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
        if span != buffers[name].count:
            raise ValueError("DivMod operand span does not match its allocation")
    if buffers["c"].count != threads or buffers["d"].count != threads:
        raise ValueError("DivMod output extent differs from its launch")
    for output in "cd":
        low, high = ranges[output]
        for name, (start, end) in ranges.items():
            if name != output and max(low, start) < min(high, end):
                raise ValueError("DivMod outputs overlap another buffer")
    return {
        "kernel": kernel,
        "dtype": dtype,
        "shape": shape,
        "aStrides": a_strides,
        "bStrides": b_strides,
    }


def logical_operands(metadata, a, b):
    """Gather each output element's operand pair from the retained storage."""
    shape, pairs = metadata["shape"], []
    for index in range(math.prod(shape)):
        a_index = b_index = 0
        remaining = index
        for axis in range(len(shape) - 1, -1, -1):
            position = remaining % shape[axis]
            remaining //= shape[axis]
            a_index += position * metadata["aStrides"][axis]
            b_index += position * metadata["bStrides"][axis]
        pairs.append((a[a_index], b[b_index]))
    return pairs


def check_integer_operands(dtype, pairs):
    """Refuse division by zero and signed overflow, which no target defines."""
    if dtype in {"float16", "bfloat16", "float32"}:
        return
    minimum = -(1 << (8 * ctypes.sizeof(CTYPES[dtype]) - 1))
    for x, y in pairs:
        if y == 0 or (dtype.startswith("int") and y == -1 and x == minimum):
            raise ValueError(
                "DivMod integer operands require nonzero divisors without overflow"
            )


def packet(dtype, target, words):
    if dtype == "float32":
        return {
            "dtype": "float32",
            "shape": [len(words)],
            "encoding": FLOAT32_BITS,
            "values": list(words),
        }
    if dtype == "float16":
        if target == "opengl":
            words, encoding, physical = (
                half_storage.pack(words, target),
                FLOAT32_BITS,
                "float32",
            )
        else:
            words, encoding, physical = (
                half_storage.pack(words, target),
                FLOAT16_BITS,
                "float16",
            )
        return {
            "dtype": physical,
            "shape": [len(words)],
            "encoding": encoding,
            "values": words,
        }
    if dtype == "bfloat16":
        words = bfloat_storage.pack(words, target)
        if target == "directx":
            return {"dtype": "uint16", "shape": [len(words)], "values": words}
        encoding = BFLOAT16_BITS if target == "metal" else FLOAT32_BITS
        physical = "bfloat16" if target == "metal" else "float32"
        return {
            "dtype": physical,
            "shape": [len(words)],
            "encoding": encoding,
            "values": words,
        }
    return {
        "dtype": WIDENED[target].get(dtype, dtype),
        "shape": [len(words)],
        "values": [int(word) for word in words],
    }


def unpack(dtype, target, words):
    if dtype == "float16":
        return half_storage.unpack(words, target)
    if dtype == "bfloat16":
        return bfloat_storage.unpack(words, target)
    ctype = CTYPES[dtype]
    if any(type(word) is not int or ctype(word).value != word for word in words):
        raise ValueError("Native DivMod returned invalid scalar storage")
    return list(words)


def reflected_layout_matches(scalar, value):
    """The reflected element type must carry the packet's exact storage."""
    storage = scalar.get("storageEncoding")
    if storage is not None:
        return (
            value["dtype"] == "float16"
            and scalar["elementType"] == "uint16"
            and scalar["elementStrideBytes"] == 2
            and storage == {"encoding": FLOAT16_BITS, "logicalElementType": "float16"}
        )
    return scalar["elementType"] == value["dtype"] and scalar[
        "elementStrideBytes"
    ] == PHYSICAL_SIZES.get(value["dtype"])


def supplied_buffers(buffers, count):
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError("DivMod buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("DivMod buffer names must be unique")
        supplied[name] = buffer
    return supplied


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    kernel, dtype = signature(entry)
    if host.divmods is None or launch is None or not buffers or not 0 < count <= 8:
        raise ValueError(
            "DivMod dispatch requires complete buffers, pinned sources and launch geometry"
        )
    supplied = supplied_buffers(buffers, count)
    execution = launch.execution()
    metadata = validate(entry, supplied, threads, execution)
    logical = {name: buffer_values(supplied[name], CTYPES[dtype]) for name in "ab"}
    check_integer_operands(
        dtype, logical_operands(metadata, logical["a"], logical["b"])
    )
    descriptor, directory = host.divmods.get(entry)
    prefix = entry + "_"
    inputs, outputs, observed, matched = {}, {}, [], {}
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        scalar = binding["scalarLayout"]
        name = scalar.get("memberName", binding["name"]).removeprefix(prefix)
        if name not in supplied or name in matched or binding["name"] in inputs:
            raise ValueError("Reflected DivMod binding does not match its source")
        matched[name] = binding["name"]
        buffer = supplied[name]
        if name in "ab":
            value = packet(
                dtype, host.target, logical[name] + [GUARDS[dtype]] * GUARD_COUNT
            )
        elif name in "cd":
            value = packet(
                dtype, host.target, [GUARDS[dtype]] * (threads + GUARD_COUNT)
            )
        else:
            kind = buffer.dtype.decode("ascii")
            value = {
                "dtype": kind,
                "shape": [buffer.count],
                "values": buffer_values(buffer, METADATA_CTYPES[kind]),
            }
        if not reflected_layout_matches(scalar, value):
            raise ValueError("Reflected DivMod storage differs from the native buffer")
        inputs[binding["name"]] = value
        if name in "cd":
            outputs[binding["name"]] = {k: v for k, v in value.items() if k != "values"}
        else:
            observed.append(binding["name"])
    if set(matched) != set(supplied):
        raise ValueError(
            "DivMod reflection does not cover the complete source contract"
        )
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        inputs,
        outputs,
        execution,
        expected_target=host.target,
        observe_inputs=tuple(observed),
    )
    result = execute(host, request)
    if result.status != "ok" or set(result.outputs) != set(outputs):
        raise RuntimeError("Native DivMod did not return both outputs")
    values, guards, payloads = {}, {}, []
    for name in "cd":
        output = result.outputs[matched[name]]
        expected = outputs[matched[name]]
        if any(
            output.get(key) != expected.get(key)
            for key in ("dtype", "encoding", "shape")
        ):
            raise RuntimeError("Native DivMod output layout differs")
        words = output.get("values")
        if not isinstance(words, list) or len(words) != threads + GUARD_COUNT:
            raise RuntimeError("Native DivMod output extent differs")
        words = unpack(dtype, host.target, words)
        if words[threads:] != [GUARDS[dtype]] * GUARD_COUNT:
            raise RuntimeError("Native DivMod changed its output guard")
        values[name], guards[name] = words[:threads], words[threads:]
        payloads.append(bytes((CTYPES[dtype] * threads)(*values[name])))
    event = {
        "entry": entry,
        "target": host.target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "divmodMetadata": metadata,
        "inputs": inputs,
        "operandValues": logical,
        "outputValues": values,
        "outputGuards": guards,
        "outputHash": hashlib.sha256(b"".join(payloads)).hexdigest(),
    }
    write_trace(host, event)
    for name, payload in zip("cd", payloads):
        ctypes.memmove(supplied[name].data, payload, len(payload))
    host.dispatch_count += 1
