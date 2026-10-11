"""Validate random key storage and copy native random bytes into MLX arrays."""

import ctypes
import hashlib
import json
import math

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.random_layout import (
    GUARD_COUNT,
    MAX_KEY_ELEMENTS,
    MAX_NATIVE_BYTES,
    RandomOutputLayout,
)
from demos.integrations.mlx.portable_host.random_packages import ENTRIES

GUARD = [91] * GUARD_COUNT
TYPES = {
    "keys": ("uint32", ctypes.c_uint32),
    "out": ("int8", ctypes.c_int8),
    "bytes_per_key": ("uint64", ctypes.c_uint64),
    "ndim": ("int32", ctypes.c_int32),
    "key_shape": ("int32", ctypes.c_int32),
    "key_strides": ("int64", ctypes.c_int64),
}


def values(name, buffer):
    return list(
        ctypes.cast(buffer.data, ctypes.POINTER(TYPES[name][1] * buffer.count)).contents
    )


def validate(entry, buffers, byte_count, execution):
    if (
        entry not in ENTRIES
        or set(buffers) != set(TYPES)
        or type(byte_count) is not int
        or not 0 < byte_count <= MAX_NATIVE_BYTES
    ):
        raise ValueError("Random buffers or output size do not match the entry")
    for name, buffer in buffers.items():
        itemsize = ctypes.sizeof(TYPES[name][1])
        maximum = MAX_NATIVE_BYTES if name == "out" else MAX_KEY_ELEMENTS
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.data + buffer.count * itemsize > 1 << (
                8 * ctypes.sizeof(ctypes.c_void_p)
            )
            or buffer.dtype != TYPES[name][0].encode("ascii")
            or not 0 < buffer.count <= maximum
            or buffer.output != int(name == "out")
        ):
            raise ValueError("Random buffer layout or direction is invalid")
    rank = buffers["key_shape"].count
    lengths = {
        "out": byte_count,
        "bytes_per_key": 1,
        "ndim": 1,
        "key_shape": rank,
        "key_strides": rank,
    }
    if not 1 <= rank <= 64 or any(
        buffers[name].count != count for name, count in lengths.items()
    ):
        raise ValueError("Random metadata lengths do not match")
    if values("ndim", buffers["ndim"]) != [rank]:
        raise ValueError("Random key rank does not match its shape")
    shape = values("key_shape", buffers["key_shape"])
    strides = values("key_strides", buffers["key_strides"])
    if (
        shape[-1] != 2
        or any(not 1 <= size <= MAX_KEY_ELEMENTS for size in shape)
        or any(not 0 <= stride <= MAX_KEY_ELEMENTS for stride in strides)
    ):
        raise ValueError("Random key shape or strides are invalid")
    key_count = math.prod(shape) // 2
    per_key = values("bytes_per_key", buffers["bytes_per_key"])[0]
    layout = RandomOutputLayout(key_count, per_key)
    if (
        not per_key
        or layout.logical_byte_count != byte_count
        or layout.native_byte_count > MAX_NATIVE_BYTES
    ):
        raise ValueError("Random native allocation exceeds its validated bounds")
    span = 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    if span != buffers["keys"].count:
        raise ValueError("Random key span does not match its allocation")
    if entry == "rbitsc" and any(
        size > 1 and stride != math.prod(shape[axis + 1 :])
        for axis, (size, stride) in enumerate(zip(shape, strides))
    ):
        raise ValueError("Contiguous random entry requires row-contiguous keys")
    word_count = (per_key + 3) // 4
    if execution != {
        "workgroupCount": layout.workgroup_count,
        "workgroupSize": [1, 1, 1],
    }:
        raise ValueError("Random launch does not match its output layout")
    output = buffers["out"]
    for name, buffer in buffers.items():
        end = buffer.data + buffer.count * ctypes.sizeof(TYPES[name][1])
        if name != "out" and max(buffer.data, output.data) < min(
            end, output.data + byte_count
        ):
            raise ValueError("Random output overlaps an input allocation")
    return layout, {
        "keyCount": key_count,
        "keyShape": shape,
        "keyStrides": strides,
        "logicalBytesPerKey": per_key,
        "nativeBytesPerKey": layout.native_bytes_per_key,
        "wordCount": word_count,
    }


def dispatch(host, entry, buffers, count, byte_count, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.random_directory is None or entry not in host.descriptors:
        raise ValueError("No translated random package is configured")
    if count != len(TYPES) or not buffers or launch is None:
        raise ValueError(
            "Random dispatch requires complete buffers and launch geometry"
        )
    supplied = {}
    for i in range(count):
        buffer = buffers[i]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Random buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Random buffer names must be unique")
        supplied[name] = buffer
    execution = launch.execution()
    layout, metadata = validate(entry, supplied, byte_count, execution)
    source_values = {
        name: values(name, buffer) for name, buffer in supplied.items() if name != "out"
    }
    source_values["bytes_per_key"] = [layout.native_bytes_per_key]
    source_values["odd"] = [metadata["wordCount"] % 2]
    descriptor = host.descriptors[entry]
    inputs, outputs, matched = {}, {}, set()
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        scalar = binding["scalarLayout"]
        name = scalar.get("memberName", binding["name"]).removeprefix(entry + "_")
        name = "out" if name == "out_" else name
        if name in matched or binding["name"] in inputs:
            raise ValueError("Random reflected bindings must be unique")
        matched.add(name)
        if name == "out":
            dtype, size = ("int8", 1) if host.target == "metal" else ("int32", 4)
            data = [91] * (layout.native_byte_count + len(GUARD))
        elif name == "odd":
            dtype, size = ("bool", 1) if host.target == "metal" else ("uint32", 4)
            data = (
                [bool(source_values[name][0])]
                if dtype == "bool"
                else source_values[name]
            )
        elif name in TYPES:
            dtype, ctype = TYPES[name]
            size, data = ctypes.sizeof(ctype), source_values[name]
        else:
            raise ValueError("Unexpected random reflected binding")
        if scalar["elementType"] != dtype or scalar["elementStrideBytes"] != size:
            raise ValueError("Native and reflected random layouts disagree")
        value = {"dtype": dtype, "shape": [len(data)], "values": data}
        inputs[binding["name"]] = value
        if name == "out":
            outputs[binding["name"]] = value
    required = {"keys", "out", "odd", "bytes_per_key"}
    if entry == "rbits":
        required |= {"ndim", "key_shape", "key_strides"}
    if matched != required:
        raise ValueError("Reflected random bindings do not cover the source contract")
    directory = host.random_directory / "package"
    request = build_native_loader_dispatch_request(
        descriptor, directory, inputs, outputs, execution, expected_target=host.target
    )
    result = execute(host, request)
    if result.status != "ok" or set(result.outputs) != set(outputs):
        raise RuntimeError("Native random execution did not return its output")
    (output_name,) = outputs
    output = result.outputs[output_name]
    if (
        output.get("dtype") != outputs[output_name]["dtype"]
        or output.get("shape") != outputs[output_name]["shape"]
        or output.get("encoding") is not None
    ):
        raise RuntimeError("Native random readback layout differs")
    readback = output.get("values", [])
    if (
        not isinstance(readback, list)
        or len(readback) != layout.native_byte_count + len(GUARD)
        or any(type(value) is not int for value in readback)
        or readback[layout.native_byte_count :] != GUARD
    ):
        raise RuntimeError("Native random readback or its guard differs")
    logical = layout.unpack(readback[: layout.native_byte_count])
    event = {
        "entry": entry,
        "target": host.target,
        "threads": byte_count,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "randomMetadata": metadata,
        "inputs": inputs,
        "randomValues": readback[: layout.native_byte_count],
        "randomGuardValues": readback[layout.native_byte_count :],
        "outputHash": hashlib.sha256(logical).hexdigest(),
    }
    with host.trace.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    ctypes.memmove(supplied["out"].data, logical, len(logical))
    host.dispatch_count += 1
