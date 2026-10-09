"""Storage and launch contracts for pinned quantization kernels."""

import ctypes
import re

MAX_GROUPS = 65535
SOURCE_TYPES = {"float32": "float", "float16": "float16_t", "bfloat16": "bfloat16_t"}
ITEM_SIZES = {"float32": 4, "float16": 2, "bfloat16": 2, "uint8": 1}
ENTRY_PREFIXES = ("affine_", "mxfp4_", "mxfp8_", "nvfp4_")


def signature(entry):
    match = re.fullmatch(
        r"affine_(quantize|dequantize)_(float|float16_t|bfloat16_t)_gs_(32|64|128)_b_(2|3|4|5|6|8)",
        entry,
    )
    if match is not None:
        operation, source, group, bits = match.groups()
    else:
        match = re.fullmatch(
            r"(mxfp4|mxfp8|nvfp4)_(quantize|dequantize)_(float|float16_t|bfloat16_t)_gs_(16|32)_b_(4|8)_hgs_(false|true)",
            entry,
        )
        if match is None:
            raise ValueError("Unsupported quantization entry")
        mode, operation, source, group, bits, global_scale = match.groups()
        if (int(group), int(bits)) != {
            "mxfp4": (32, 4),
            "mxfp8": (32, 8),
            "nvfp4": (16, 4),
        }[mode] or (global_scale == "true" and mode != "nvfp4"):
            raise ValueError("Unsupported block quantization specialization")
    dtype = next(key for key, value in SOURCE_TYPES.items() if value == source)
    return operation, dtype, int(group), int(bits)


def pack_factor(bits):
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError("Unsupported quantization packing width")
    return 8 if bits in (3, 5) else 4 if bits == 6 else 8 // bits


def workgroup_size(entry):
    operation, _, group, bits = signature(entry)
    return min(group, 32) if operation == "quantize" else group // pack_factor(bits)


def buffer_contract(entry, groups):
    operation, dtype, group, bits = signature(entry)
    if type(groups) is not int or not 1 <= groups <= MAX_GROUPS:
        raise ValueError("Quantization dispatch requires 1 to 65535 complete groups")
    values, packed = groups * group, groups * group * bits // 8
    quantize = operation == "quantize"
    block = not entry.startswith("affine_")
    contract = {
        "w": (dtype if quantize else "uint8", values if quantize else packed, 0),
        "out": ("uint8" if quantize else dtype, packed if quantize else values, 1),
        "scales": ("uint8" if block else dtype, groups, int(quantize)),
    }
    if not block:
        contract["biases"] = (dtype, groups, int(quantize))
    elif entry.endswith("_hgs_true"):
        contract["global_scale"] = ("float32", 1, 0)
    return contract


def validate(entry, buffers, elements, execution):
    _, _, group, _ = signature(entry)
    if type(elements) is not int or elements <= 0 or elements % group:
        raise ValueError("Quantization value count must contain complete groups")
    groups = elements // group
    contract = buffer_contract(entry, groups)
    if set(buffers) != set(contract):
        raise ValueError("Quantization buffers do not match the entry")
    if execution != {
        "workgroupCount": [groups, 1, 1],
        "workgroupSize": [workgroup_size(entry), 1, 1],
    }:
        raise ValueError(
            "Quantization launch does not match complete quantization groups"
        )
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
            raise ValueError("Quantization buffer storage or direction does not match")
        spans[name] = (buffer.data, buffer.data + count * itemsize)
    for name, (start, end) in spans.items():
        if not contract[name][2]:
            continue
        if any(
            other != name and max(start, low) < min(end, high)
            for other, (low, high) in spans.items()
        ):
            raise ValueError("Quantization output overlaps another buffer")
    return {"groupCount": groups, "elementCount": elements}
