"""Audit random uploads, complete native storage and logical byte readbacks."""

import hashlib
import math

from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_input_bindings,
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.random_workloads import words


def audit_event(np, event):
    target, entry = event["target"], event["entry"]
    require(
        target in {"metal", "opengl", "directx"} and entry in {"rbits", "rbitsc"},
        "Unknown random execution target or entry",
    )
    request = event["details"]["request"]
    inputs = {}
    for name, value in event["inputs"].items():
        binding = request["buffers"][name]
        layout = binding["binding"]["metadata"]["scalarLayout"]
        member = layout.get("memberName", name).removeprefix(entry + "_")
        member = "out" if member == "out_" else member
        require(member not in inputs, "Duplicate random upload")
        require(
            binding["dtype"] == value["dtype"] == layout["elementType"]
            and binding["shape"] == value["shape"] == [len(value["values"])]
            and layout["elementStrideBytes"] == np.dtype(value["dtype"]).itemsize
            and value.get("encoding") is None,
            "Random upload layout differs",
        )
        inputs[member] = value
    required = {"keys", "out", "bytes_per_key", "odd"}
    if entry == "rbits":
        required |= {"key_shape", "key_strides", "ndim"}
    require(set(inputs) == required, "Random upload inventory differs")
    dtypes = {
        "keys": "uint32",
        "out": "int8" if target == "metal" else "int32",
        "bytes_per_key": "uint64",
        "odd": "bool" if target == "metal" else "uint32",
        "key_shape": "int32",
        "key_strides": "int64",
        "ndim": "int32",
    }
    require(
        all(value["dtype"] == dtypes[name] for name, value in inputs.items()),
        "Random source storage widths differ",
    )
    metadata = event["randomMetadata"]
    shape, strides = metadata["keyShape"], metadata["keyStrides"]
    per_key, native_size = metadata["logicalBytesPerKey"], metadata["nativeBytesPerKey"]
    count = math.prod(shape) // 2
    require(
        len(shape) == len(strides)
        and 1 <= len(shape) <= 64
        and shape[-1] == 2
        and all(type(n) is int and 0 < n <= 65535 for n in shape)
        and all(type(n) is int and 0 <= n <= 65535 for n in strides),
        "Random key layout differs",
    )
    require(
        inputs["keys"]["dtype"] == "uint32"
        and all(
            type(value) is int and 0 <= value < 2**32
            for value in inputs["keys"]["values"]
        ),
        "Random key word differs",
    )
    require(
        len(inputs["keys"]["values"])
        == 1 + sum((size - 1) * step for size, step in zip(shape, strides)),
        "Random key allocation differs",
    )
    storage = np.array(inputs["keys"]["values"], dtype=np.uint32)
    keys = np.ndarray(
        shape,
        dtype=np.uint32,
        buffer=storage,
        strides=tuple(step * 4 for step in strides),
    ).reshape(-1, 2)
    if entry == "rbits":
        require(
            inputs["key_shape"]["values"] == shape
            and inputs["key_strides"]["values"] == strides
            and inputs["ndim"]["values"] == [len(shape)],
            "Random uploaded metadata differs",
        )
    else:
        require(
            all(
                size == 1 or strides[axis] == math.prod(shape[axis + 1 :])
                for axis, size in enumerate(shape)
            ),
            "Random contiguous entry has strided keys",
        )
    word_count = (per_key + 3) // 4
    require(
        per_key > 0
        and native_size == max(4, per_key)
        and native_size * count + 17 <= 65535
        and count == metadata["keyCount"]
        and word_count == metadata["wordCount"],
        "Random logical storage metadata differs",
    )
    require(
        inputs["bytes_per_key"]["values"] == [native_size]
        and inputs["odd"]["values"] == [word_count % 2],
        "Random source counters differ",
    )
    expected, logical = [], b""
    for key in keys:
        raw = np.array(words(key, word_count), dtype="<u4").tobytes()
        expected.extend(
            value if value < 128 else value - 256 for value in raw[:native_size]
        )
        logical += raw[:per_key]
    require(
        event["randomValues"] == expected
        and all(type(value) is int for value in event["randomValues"]),
        "Random native values differ from the reference",
    )
    require(
        event["randomGuardValues"] == [91] * 17
        and inputs["out"]["values"] == [91] * (len(expected) + 17),
        "Random guard or output initialization differs",
    )
    require(
        event["threads"] == len(logical)
        and event["outputHash"] == hashlib.sha256(logical).hexdigest()
        and event["dispatchVersion"] == 3,
        "Random host byte readback differs",
    )
    grid = [count, (word_count + 1) // 2, 1]
    require(
        event["workgroupCount"] == request["dispatch"]["workgroupCount"] == grid
        and event["workgroupSize"] == request["dispatch"]["workgroupSize"] == [1, 1, 1],
        "Random native launch differs",
    )
    require(
        request["target"] == target
        and request["entryPoint"]
        == {"opengl": "main", "directx": "CSMain"}.get(target, entry),
        "Random native entry differs",
    )
    audit_input_bindings(event)
    audit_native_execution(event)
    return logical
