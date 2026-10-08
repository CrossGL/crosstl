"""Storage and launch contracts for pinned affine quantization kernels."""

import ctypes
import re

MAX_GROUPS = 65535
SOURCE_TYPES = {"float32": "float", "float16": "float16_t", "bfloat16": "bfloat16_t"}
ITEM_SIZES = {"float32": 4, "float16": 2, "bfloat16": 2, "uint8": 1}


def signature(entry):
    match = re.fullmatch(
        r"affine_(quantize|dequantize)_(float|float16_t|bfloat16_t)_gs_(32|64|128)_b_(2|3|4|5|6|8)",
        entry,
    )
    if match is None:
        raise ValueError("Unsupported affine quantization entry")
    operation, source, group, bits = match.groups()
    dtype = next(key for key, value in SOURCE_TYPES.items() if value == source)
    return operation, dtype, int(group), int(bits)


def pack_factor(bits):
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError("Unsupported affine packing width")
    return 8 if bits in (3, 5) else 4 if bits == 6 else 8 // bits


def workgroup_size(entry):
    operation, _, group, bits = signature(entry)
    return 32 if operation == "quantize" else group // pack_factor(bits)


def buffer_contract(entry, groups):
    operation, dtype, group, bits = signature(entry)
    if type(groups) is not int or not 1 <= groups <= MAX_GROUPS:
        raise ValueError("Affine dispatch requires 1 to 65535 complete groups")
    values, packed = groups * group, groups * group * bits // 8
    quantize = operation == "quantize"
    return {
        "w": (dtype if quantize else "uint8", values if quantize else packed, 0),
        "out": ("uint8" if quantize else dtype, packed if quantize else values, 1),
        "scales": (dtype, groups, int(quantize)),
        "biases": (dtype, groups, int(quantize)),
    }


def validate(entry, buffers, elements, execution):
    _, _, group, _ = signature(entry)
    if type(elements) is not int or elements <= 0 or elements % group:
        raise ValueError("Affine value count must contain complete groups")
    groups = elements // group
    contract = buffer_contract(entry, groups)
    if set(buffers) != set(contract):
        raise ValueError("Affine buffers do not match the entry")
    if execution != {
        "workgroupCount": [groups, 1, 1],
        "workgroupSize": [workgroup_size(entry), 1, 1],
    }:
        raise ValueError("Affine launch does not match complete quantization groups")
    spans = {}
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    for name, (dtype, count, output) in contract.items():
        buffer = buffers[name]
        itemsize = ITEM_SIZES[dtype]
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.count != count
            or buffer.output != output
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError("Affine buffer storage or direction does not match")
        spans[name] = (buffer.data, buffer.data + count * itemsize)
    for name, (start, end) in spans.items():
        if not contract[name][2]:
            continue
        if any(
            other != name and max(start, low) < min(end, high)
            for other, (low, high) in spans.items()
        ):
            raise ValueError("Affine output overlaps another buffer")
    return {"groupCount": groups, "elementCount": elements}
