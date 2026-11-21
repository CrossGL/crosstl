"""Execute translated MLX vector attention packages and commit guarded outputs.

``sdpa_vector`` binds the queries, keys, values and output, the GQA factor,
the key count, the key and value strides and the scale, as
``mlx/backend/metal/scaled_dot_product_attention.cpp`` binds them, and the
Boolean or additive mask with its strides and the sinks when their function
constants are set. ``sdpa_vector_2pass_1`` writes the per-block partial
outputs, exponent sums and maxima instead, and ``sdpa_vector_2pass_2``
combines them. Each launch is checked against ``sdpa_layout``'s plan for its
hash name, which derives every buffer's exact span from the launch and the
strides. Inputs are followed by poisoned elements no kernel reads, each
output starts as guard elements and keeps a guard tail, and a binding whose
function constant is off, which the kernel never reads, holds only poison.
Every output is checked before any reaches MLX, and the dispatch is recorded
in the trace before the MLX outputs change. ``lazy_package_evidence``
audits each dispatch.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import sdpa_layout as layout
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    GUARDS,
    buffer_words,
    packet,
    reflected_layout_matches,
    supplied_buffers,
    unpack,
)
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.matmul_dispatch import encode_binding
from demos.integrations.mlx.portable_host.trace_records import write_trace

LABEL = "SDPA"
ENTRY_PREFIXES = layout.ENTRY_PREFIXES
# Read by no kernel: 1.5 in every floating type; False for Booleans.
POISON = {"float32": 0x3FC00000, "float16": 0x3E00, "bfloat16": 0x3FC0, "bool_": 0}
# Every parameter the pinned kernels declare, active or not, and its type
# relative to the entry's dtype (``None``) when no plan binds it.
DECLARED = {
    "bmask": "bool_",
    "fmask": None,
    "sinks": None,
    "mask_kv_seq_stride": "int32",
    "mask_q_seq_stride": "int32",
    "mask_head_stride": "int32",
    "num_q_heads": "int32",
}


def is_entry(entry):
    try:
        layout.signature(entry)
    except ValueError:
        return False
    return True


def _declared_type(entry, name):
    """The type of a declared parameter the launch does not bind."""
    if name not in DECLARED:
        raise ValueError(f"Reflected {LABEL} binding differs from its source")
    return DECLARED[name] or layout.signature(entry)["dtype"]


def encode(dtype, words, target, scalar, kind, is_scalar):
    """One reflected binding's runtime value for ``dtype`` storage words."""
    if dtype == "uint64":
        value = {"dtype": "uint64", "shape": [len(words)], "values": list(words)}
        if (
            scalar.get("elementType") != "uint64"
            or scalar.get("elementStrideBytes") != 8
        ):
            raise ValueError(f"{LABEL} reflected uint64 storage differs")
        return value
    if is_scalar:
        return encode_binding(dtype, words, target, scalar, kind)
    value = packet(dtype, target, words)
    if not reflected_layout_matches(scalar, value):
        raise ValueError(f"{LABEL} reflected {dtype} storage differs")
    return value


def validate(entry, supplied, threads, execution):
    """Check the buffers and launch against the entry and return metadata."""
    parameters = layout.parameters(entry)
    scalars = layout.scalars(entry)
    outputs = layout.outputs(entry)
    if set(supplied) != set(parameters):
        raise ValueError(f"{LABEL} buffers do not match the attention kernel")
    for name, buffer in supplied.items():
        dtype = layout.parameter_type(entry, name)
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % ctypes.sizeof(CTYPES[dtype])
            or buffer.output != int(name in outputs)
            or (name in scalars and buffer.count != 1)
        ):
            raise ValueError(f"{LABEL} buffer type, alignment or direction differs")
    values = {}
    for name in scalars:
        (word,) = buffer_words(supplied[name], layout.parameter_type(entry, name))
        if name == "scale":
            word = ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word)).value
        values[name] = word
    arrays = [name for name in parameters if name not in scalars]
    counts = {name: supplied[name].count for name in arrays}
    metadata = layout.plan_launch(entry, counts, values, threads, execution)
    spans = {}
    for name in arrays:
        size = ctypes.sizeof(CTYPES[layout.parameter_type(entry, name)])
        spans[name] = (supplied[name].data, supplied[name].data + counts[name] * size)
    for name in outputs:
        for other, span in spans.items():
            if other != name and max(span[0], spans[name][0]) < min(
                span[1], spans[name][1]
            ):
                raise ValueError(f"{LABEL} output {name} overlaps {other}")
    return metadata, values


def _storage(entry, supplied):
    """Every bound parameter's words, with poison or guard tails."""
    storage = {}
    scalars = layout.scalars(entry)
    for name in layout.parameters(entry):
        dtype = layout.parameter_type(entry, name)
        if name in layout.outputs(entry):
            storage[name] = [GUARDS[dtype]] * (supplied[name].count + GUARD_COUNT)
        elif name in scalars:
            storage[name] = buffer_words(supplied[name], dtype)
        else:
            words = buffer_words(supplied[name], dtype)
            storage[name] = words + [POISON[dtype]] * GUARD_COUNT
    return storage


def _inputs(host, entry, descriptor, storage):
    """Each reflected binding's value, the output bindings and observed inputs."""
    inputs, observed, outputs, bound = {}, [], {}, set()
    scalars = layout.scalars(entry)
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if provenance.get("kind") == "generated-execution-input":
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if "executionInput" in provenance or name in bound or binding["name"] in inputs:
            raise ValueError(f"Reflected {LABEL} binding differs from its source")
        bound.add(name)
        if name in storage:
            dtype = layout.parameter_type(entry, name)
            words, is_scalar = storage[name], name in scalars
        else:
            # A parameter whose function constant is off: the kernel never
            # reads it, so it holds poison only.
            dtype = _declared_type(entry, name)
            is_scalar = dtype == "int32"
            words = [0] if is_scalar else [POISON[dtype]] * GUARD_COUNT
        inputs[binding["name"]] = encode(
            dtype,
            words,
            host.target,
            binding["scalarLayout"],
            binding.get("kind", ""),
            is_scalar,
        )
        if name in layout.outputs(entry):
            if binding.get("access") != "read_write":
                raise ValueError(f"{LABEL} output {name} is not writable")
            outputs[name] = binding["name"]
        else:
            observed.append(binding["name"])
    if set(outputs) != set(layout.outputs(entry)) or not set(storage) <= bound:
        raise ValueError(f"{LABEL} reflection is incomplete")
    return inputs, observed, outputs, bound


def _readback(entry, result, requested, outputs, supplied, target):
    """Each output's logical words, after checking every guard."""
    if result.status != "ok" or set(result.outputs) != set(requested):
        raise RuntimeError(f"Native {LABEL} execution returned no output")
    payloads = {}
    for name, binding in outputs.items():
        returned = result.outputs[binding]
        if any(
            returned.get(key) != requested[binding].get(key)
            for key in ("dtype", "encoding", "shape")
        ) or not isinstance(returned.get("values"), list):
            raise RuntimeError(f"Native {LABEL} output layout differs")
        dtype = layout.parameter_type(entry, name)
        try:
            words = unpack(dtype, target, returned["values"])
        except ValueError as error:
            raise RuntimeError(
                f"Native {LABEL} output is outside its storage"
            ) from error
        size = supplied[name].count
        if (
            len(words) != size + GUARD_COUNT
            or words[size:] != [GUARDS[dtype]] * GUARD_COUNT
        ):
            raise RuntimeError(f"Native {LABEL} kernel changed its {name} guard")
        payloads[name] = (words[:size], bytes((CTYPES[dtype] * size)(*words[:size])))
    return payloads


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    parameters = layout.parameters(entry)
    if host.sdpas is None or launch is None or not buffers or count != len(parameters):
        raise ValueError(f"{LABEL} dispatch requires pinned packages and geometry")
    supplied = supplied_buffers(buffers, count, LABEL)
    execution = launch.execution()
    metadata, values = validate(entry, supplied, threads, execution)
    storage = _storage(entry, supplied)
    descriptor, directory = host.sdpas.get(entry)
    inputs, observed, outputs, bound = _inputs(host, entry, descriptor, storage)
    requested = {
        binding: {key: item for key, item in inputs[binding].items() if key != "values"}
        for binding in outputs.values()
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        inputs,
        requested,
        execution,
        expected_target=host.target,
        observe_inputs=tuple(observed),
    )
    result = execute(host, request)
    payloads = _readback(entry, result, requested, outputs, supplied, host.target)
    input_hashes = {}
    for name in parameters:
        if name in outputs or name in layout.scalars(entry):
            continue
        item = ctypes.sizeof(CTYPES[layout.parameter_type(entry, name)])
        input_hashes[name] = hashlib.sha256(
            ctypes.string_at(supplied[name].data, supplied[name].count * item)
        ).hexdigest()
    write_trace(
        host,
        {
            "entry": entry,
            "target": host.target,
            "threads": threads,
            **execution,
            "dispatchVersion": DISPATCH_VERSION,
            "artifact": descriptor["artifact"],
            "packageRoot": str(directory),
            "details": result.details,
            "sdpaMetadata": metadata,
            "scalars": values,
            "boundParameters": sorted(name for name in bound if name),
            "inputHashes": input_hashes,
            "outputValues": {
                name: words if len(words) <= 4096 else None
                for name, (words, _) in payloads.items()
            },
            "outputHashes": {
                name: hashlib.sha256(payload).hexdigest()
                for name, (_, payload) in payloads.items()
            },
        },
    )
    for name, (_, payload) in payloads.items():
        ctypes.memmove(supplied[name].data, payload, len(payload))
    host.dispatch_count += 1
