"""Execute translated MLX BlockMaskedMM packages and commit guarded MLX outputs.

One dispatcher serves the masked gemv kernels and the steel block-masked GEMM
kernels. Each launch is checked against ``block_masked_mm_layout``'s plan, run
through its translated package with guarded outputs, and recorded in the
trace before the MLX output changes. Boolean masks are bound as the target's
boolean storage, masks of the element type as the operands are; a mask
matmul.cpp leaves unbound gets inert words the kernel never reads.
"""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request, pack_storage_records
from demos.integrations.mlx.portable_host import block_masked_mm_layout as layout
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.matmul_dispatch import (
    GUARD_COUNT,
    POISON,
    _words,
    encode_binding,
)
from demos.integrations.mlx.portable_host.quantization_dispatch import (
    GUARDS,
    WORD_FORMATS,
    decode,
)
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES
from demos.integrations.mlx.portable_host.trace_records import write_trace

ENTRY_PREFIXES = layout.ENTRY_PREFIXES
LABEL = "BlockMaskedMM"
# The bindings each kernel reads whatever its masks.
REQUIRED = {
    "masked-mv": {"mat", "in_vec", "out_vec", "in_vec_size", "out_vec_size"},
    "masked-mm": {"A", "B", "D", "params"},
}
# The reflected GEMMParams record's stride.
PARAMS_STRIDE = 72
# An unbound mask's storage: the one-byte _NoMask on Metal, one 32-bit word
# per element elsewhere. The gates bind four of them.
INERT_TYPES = {"metal": ("int8", 1), "directx": ("int32", 4), "opengl": ("int32", 4)}
INERT_COUNT = 4
# Boolean masks: one byte on Metal, a uint32 word elsewhere.
BOOL_TYPES = {"metal": ("bool", 1), "directx": ("uint32", 4), "opengl": ("uint32", 4)}


def is_entry(entry):
    try:
        layout.signature(entry)
    except ValueError:
        return False
    return True


def cache_for(host, kind):
    """The host's package cache for one kind of BlockMaskedMM entry."""
    return host.masked_mvs if kind == "masked-mv" else host.masked_mms


def _itemsize(dtype):
    return 1 if dtype == "bool_" else ITEM_SIZES.get(dtype, 4)


def _bool_words(buffer):
    words = list((ctypes.c_uint8 * buffer.count).from_address(buffer.data))
    if any(word > 1 for word in words):
        raise ValueError(f"{LABEL} boolean mask is not canonical")
    return words


def _storage(entry, supplied, counts, parameters):
    """Every bound parameter's words, with guards after the array buffers."""
    info = layout.signature(entry)
    output = layout.OUTPUTS[info["kind"]]
    storage = {output: [GUARDS[info["dtype"]]] * (counts[output] + GUARD_COUNT)}
    for name, buffer in supplied.items():
        if name == output:
            continue
        dtype = layout.parameter_type(entry, name)
        if dtype == "bool_":
            # Unread positions hold true, which no read past a block can use.
            storage[name] = _bool_words(buffer) + [1] * GUARD_COUNT
        elif name in counts:
            storage[name] = _words(buffer, dtype) + [POISON[dtype]] * GUARD_COUNT
        else:
            value = parameters[name]
            storage[name] = [value] if isinstance(value, int) else value
    return storage


def _value(host, entry, name, storage, binding, absent):
    """The runtime value of one reflected binding."""
    scalar = binding["scalarLayout"]
    if name in absent or layout.parameter_type(entry, name) == "bool_":
        dtype, stride = (INERT_TYPES if name in absent else BOOL_TYPES)[host.target]
        if (
            scalar.get("elementType") != dtype
            or scalar.get("elementStrideBytes") != stride
        ):
            raise ValueError(f"{LABEL} reflected mask layout differs")
        words = [0] * INERT_COUNT if name in absent else storage[name]
        values = [bool(word) for word in words] if dtype == "bool" else list(words)
        return {"dtype": dtype, "shape": [len(values)], "values": values}
    if name == "params":
        if scalar.get("elementStrideBytes") != PARAMS_STRIDE:
            raise ValueError(f"{LABEL} reflected record stride differs")
        words = pack_storage_records(scalar, [dict(storage[name])])
        return {"dtype": "uint32", "shape": [len(words)], "values": words}
    return encode_binding(
        layout.parameter_type(entry, name),
        storage[name],
        host.target,
        scalar,
        binding.get("kind", ""),
    )


def _inputs(host, entry, descriptor, storage):
    """Each reflected binding's value, the output binding and the bound names."""
    kind = layout.kind_of(entry)
    absent = set(layout.absent_masks(entry))
    inputs, observed, output, bound = {}, [], None, set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        name = provenance.get("sourceResource", {}).get("parameter")
        if (
            "executionInput" in provenance
            or (name not in storage and name not in absent)
            or name in bound
            or binding["name"] in inputs
        ):
            raise ValueError(f"Reflected {LABEL} binding differs from its source")
        bound.add(name)
        inputs[binding["name"]] = _value(host, entry, name, storage, binding, absent)
        if name == layout.OUTPUTS[kind]:
            if binding.get("access") != "read_write":
                raise ValueError(f"{LABEL} output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    if not REQUIRED[kind] <= bound or output is None:
        raise ValueError(f"{LABEL} reflection is incomplete")
    return inputs, observed, output, bound


def _supplied(buffers, count, names):
    if count != len(names) or not buffers:
        raise ValueError(f"{LABEL} dispatch requires pinned packages and buffers")
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError(f"{LABEL} buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError(f"{LABEL} buffer names must be unique")
        supplied[name] = buffer
    return supplied


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    info = layout.signature(entry)
    kind, dtype = info["kind"], info["dtype"]
    cache = cache_for(host, kind)
    if cache is None or launch is None:
        raise ValueError(f"{LABEL} dispatch requires pinned packages and geometry")
    supplied = _supplied(buffers, count, layout.host_parameters(entry))
    counts, parameters, views = layout.read_storage(entry, supplied)
    execution = launch.execution()
    metadata = layout.plan(entry, counts, parameters, views, threads, execution)
    storage = _storage(entry, supplied, counts, parameters)
    descriptor, directory = cache.get(entry)
    inputs, observed, output, bound = _inputs(host, entry, descriptor, storage)
    outputs = {output: inputs[output]}
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
    if result.status != "ok" or set(result.outputs) != {output}:
        raise RuntimeError(f"Native {LABEL} execution returned no output")
    initial, value = outputs[output], result.outputs[output]
    if {key: item for key, item in value.items() if key != "values"} != {
        key: item for key, item in initial.items() if key != "values"
    }:
        raise RuntimeError(f"Native {LABEL} output layout differs")
    out_name = layout.OUTPUTS[kind]
    size = counts[out_name]
    words = decode(value, dtype, initial["dtype"])
    guard = GUARDS[dtype]
    if len(words) != size + GUARD_COUNT or words[size:] != [guard] * GUARD_COUNT:
        raise RuntimeError(f"Native {LABEL} output guard differs")
    itemsize = ITEM_SIZES[dtype]
    payload = struct.pack("<" + str(size) + WORD_FORMATS[itemsize], *words[:size])
    arrays = sorted(name for name in supplied if name in counts and name != out_name)
    event = {
        "entry": entry,
        "target": host.target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "maskedMetadata": metadata,
        "parameters": parameters,
        "boundParameters": sorted(bound),
        "inputHashes": {
            name: (
                hashlib.sha256(
                    ctypes.string_at(
                        supplied[name].data,
                        supplied[name].count
                        * _itemsize(layout.parameter_type(entry, name)),
                    )
                ).hexdigest()
            )
            for name in arrays
        },
        "outputValues": words[:size] if size <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied[out_name].data, payload, len(payload))
    host.dispatch_count += 1
