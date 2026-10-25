"""Execute translated MLX GatherMM and SegmentedMM packages and commit guarded
MLX outputs.

One dispatcher serves the gathered matrix-vector kernels, the steel gather
kernels, ``gather_mm_offsets`` and the segmented steel GEMM kernels. Each launch is checked against its layout's
plan, run through its translated package with guarded outputs, and recorded
in the trace before the MLX output changes.
"""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request, pack_storage_records
from demos.integrations.mlx.portable_host import (
    gather_mm_layout,
    gather_mv_layout,
    segmented_mm_layout,
)
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

ENTRY_PREFIXES = (
    *gather_mv_layout.ENTRY_PREFIXES,
    *gather_mm_layout.ENTRY_PREFIXES,
    gather_mm_layout.OFFSETS_ENTRY,
    *segmented_mm_layout.ENTRY_PREFIXES,
)
# Every group offset starts as this int32 value, which no row in [0, 2^28)
# can equal; int32 inputs end in the signed form of the uint32 poison.
OFFSET_GUARD = -0x2468ACE
INT32_POISON = POISON["uint32"] - (1 << 32)
# The bindings each kernel reads; the gathered gemv never reads its bias.
REQUIRED = {
    "mv": {"mat", "in_vec", "out_vec", "vec_indices", "mat_indices"},
    "mm": {"A", "B", "C", "lhs_indices", "rhs_indices", "params"},
    "rhs": {"A", "B", "C", "offsets", "params"},
    "offsets": {"indices", "offsets"},
    "segmented": {"A", "B", "C", "segments", "params"},
}
OUTPUTS = {
    "mv": "out_vec",
    "mm": "C",
    "rhs": "C",
    "offsets": "offsets",
    "segmented": "C",
}
# The reflected GEMMParams record's stride.
PARAMS_STRIDE = 72
INERT_TYPES = {"alpha": "float32", "beta": "float32"}


def label_of(kind):
    """The MLX primitive a kind of entry implements, as errors name it."""
    return "SegmentedMM" if kind == "segmented" else "GatherMM"


def family(entry):
    """``mv``, ``mm``, ``rhs``, ``offsets`` or ``segmented``; raises otherwise."""
    if isinstance(entry, str) and entry.startswith(gather_mv_layout.ENTRY_PREFIXES):
        gather_mv_layout.signature(entry)
        return "mv"
    if isinstance(entry, str) and entry.startswith(segmented_mm_layout.ENTRY_PREFIXES):
        return segmented_mm_layout.signature(entry)["kind"]
    return gather_mm_layout.signature(entry)["kind"]


def is_entry(entry):
    try:
        family(entry)
    except ValueError:
        return False
    return True


def _float_word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _layout(kind):
    if kind == "mv":
        return gather_mv_layout
    if kind == "segmented":
        return segmented_mm_layout
    return gather_mm_layout


def _dtype(entry, kind):
    return "int32" if kind == "offsets" else _layout(kind).signature(entry)["dtype"]


def _type(entry, kind, name):
    if name in INERT_TYPES:
        return INERT_TYPES[name]
    if name == "bias":
        return _dtype(entry, kind)
    return _layout(kind).parameter_type(entry, name)


def _storage(entry, kind, supplied, counts, parameters, metadata):
    """Every bound parameter's words, with guards after the array buffers."""
    dtype = _dtype(entry, kind)
    output = OUTPUTS[kind]
    guard = OFFSET_GUARD if kind == "offsets" else GUARDS[dtype]
    storage = {output: [guard] * (counts[output] + GUARD_COUNT)}
    for name, buffer in supplied.items():
        if name == output:
            continue
        kind_name = _type(entry, kind, name)
        if name in counts:
            if kind_name == "int32":
                words = list((ctypes.c_int32 * buffer.count).from_address(buffer.data))
                storage[name] = words + [INT32_POISON] * GUARD_COUNT
            else:
                words = _words(buffer, kind_name)
                storage[name] = words + [POISON[kind_name]] * GUARD_COUNT
        elif name == "params":
            storage[name] = parameters[name]
        else:
            value = parameters[name]
            storage[name] = [value] if isinstance(value, int) else list(value)
    if kind == "mv":
        # gather_mv binds no bias; GEMVKernel<..., false> never reads it.
        storage["bias"] = [POISON[dtype]] * GUARD_COUNT
        for name, value in metadata["inert"].items():
            storage[name] = [_float_word(value)]
    return storage


def _inputs(host, entry, kind, descriptor, storage):
    """Each reflected binding's value, the output binding and the bound names."""
    label = label_of(kind)
    inputs, observed, output, bound = {}, [], None, set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        name = provenance.get("sourceResource", {}).get("parameter")
        if (
            "executionInput" in provenance
            or name not in storage
            or name in bound
            or binding["name"] in inputs
        ):
            raise ValueError(f"Reflected {label} binding differs from its source")
        bound.add(name)
        if name == "params":
            layout = binding["scalarLayout"]
            if layout.get("elementStrideBytes") != PARAMS_STRIDE:
                raise ValueError(f"{label} reflected record stride differs")
            words = pack_storage_records(layout, [dict(storage[name])])
            inputs[binding["name"]] = {
                "dtype": "uint32",
                "shape": [len(words)],
                "values": words,
            }
        else:
            inputs[binding["name"]] = encode_binding(
                _type(entry, kind, name),
                storage[name],
                host.target,
                binding["scalarLayout"],
                binding.get("kind", ""),
            )
        if name == OUTPUTS[kind]:
            if binding.get("access") != "read_write":
                raise ValueError(f"{label} output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    if not REQUIRED[kind] <= bound or output is None:
        raise ValueError(f"{label} reflection is incomplete")
    return inputs, observed, output, bound


def cache_for(host, kind):
    """The host's package cache for one kind of GatherMM or SegmentedMM entry."""
    if kind == "mv":
        return host.gather_mvs
    if kind == "offsets":
        return host.gather_offsets
    if kind == "segmented":
        return host.segmented_mms
    return host.gather_mms


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    kind = family(entry)
    label = label_of(kind)
    layout = _layout(kind)
    names = (
        gather_mv_layout.HOST_PARAMETERS
        if kind == "mv"
        else layout.host_parameters(entry)
    )
    cache = cache_for(host, kind)
    if cache is None or count != len(names) or not buffers or launch is None:
        raise ValueError(
            f"{label} dispatch requires pinned packages, buffers and geometry"
        )
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError(f"{label} buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError(f"{label} buffer names must be unique")
        supplied[name] = buffer
    counts, parameters, views = layout.read_storage(entry, supplied)
    execution = launch.execution()
    metadata = layout.plan(entry, counts, parameters, views, threads, execution)
    dtype = _dtype(entry, kind)
    storage = _storage(entry, kind, supplied, counts, parameters, metadata)
    descriptor, directory = cache.get(entry)
    inputs, observed, output, bound = _inputs(host, entry, kind, descriptor, storage)
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
        raise RuntimeError(f"Native {label} execution returned no output")
    initial, value = outputs[output], result.outputs[output]
    if {key: item for key, item in value.items() if key != "values"} != {
        key: item for key, item in initial.items() if key != "values"
    }:
        raise RuntimeError(f"Native {label} output layout differs")
    size = counts[OUTPUTS[kind]]
    if kind == "offsets":
        words = list(value["values"])
        guard = OFFSET_GUARD
    else:
        words = decode(value, dtype, initial["dtype"])
        guard = GUARDS[dtype]
        itemsize = ITEM_SIZES[dtype]
    if len(words) != size + GUARD_COUNT or words[size:] != [guard] * GUARD_COUNT:
        raise RuntimeError(f"Native {label} output guard differs")
    if kind == "offsets":
        indices = list(views["indices"])[: parameters["M"]]
        if words[:size] != gather_mm_layout.offsets_reference(indices, size):
            raise RuntimeError(f"Native {label} group offsets differ from the search")
    payload = (
        struct.pack(f"<{size}i", *words[:size])
        if kind == "offsets"
        else struct.pack("<" + str(size) + WORD_FORMATS[itemsize], *words[:size])
    )
    matrices = [name for name in supplied if name in counts and name != OUTPUTS[kind]]
    event = {
        "entry": entry,
        "target": host.target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "gatherMetadata": {
            key: item for key, item in metadata.items() if key != "inert"
        },
        "parameters": parameters,
        "boundParameters": sorted(bound),
        "inputHashes": {
            name: (
                hashlib.sha256(
                    ctypes.string_at(
                        supplied[name].data,
                        supplied[name].count
                        * (ITEM_SIZES.get(_type(entry, kind, name)) or 4),
                    )
                ).hexdigest()
            )
            for name in sorted(matrices)
        },
        "outputValues": words[:size] if size <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied[OUTPUTS[kind]].data, payload, len(payload))
    host.dispatch_count += 1
