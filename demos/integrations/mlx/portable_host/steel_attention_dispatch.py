"""Execute translated MLX steel attention packages and commit guarded outputs.

``attention`` binds the queries, keys, values and output and the
``AttnParams`` record, as ``sdpa_full_self_attention_metal`` binds them, and
the ``AttnMaskParams`` record and mask, the sinks and the logsumexp output
when their function constants are set. The host passes each record as int64
words in field order, with the scale as its binary32 bits, and
``steel_attention_layout.plan_launch`` checks every field, the grid and each
buffer's exact span before the words are packed into the reflected record
layout. Inputs are followed by poisoned elements no kernel reads, each output
starts as guard elements and keeps a guard tail, and a binding whose function
constant is off, which the kernel never reads, holds only poison. The loader
binds a writable one, the logsumexp without ``save_lse``, as an output, so it
starts as guards only and must return them unchanged. Every
output is checked before any reaches MLX, and the dispatch is recorded in the
trace before the MLX outputs change. ``lazy_package_evidence`` audits each
dispatch.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request, pack_storage_records
from demos.integrations.mlx.portable_host import steel_attention_layout as layout
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
from demos.integrations.mlx.portable_host.sdpa_dispatch import POISON
from demos.integrations.mlx.portable_host.trace_records import write_trace

LABEL = "Steel attention"
ENTRY_PREFIXES = layout.ENTRY_PREFIXES
# params.h's AttnParams and AttnMaskParams strides.
RECORD_STRIDES = {"params": 152, "mask_params": 24}
RECORD_COUNTS = {"params": layout.PARAMS_COUNT, "mask_params": layout.MASK_PARAMS_COUNT}
# Every array parameter the kernel declares behind a function constant.
OPTIONAL = ("mask", "sinks", "lse")


def is_entry(entry):
    try:
        layout.signature(entry)
    except ValueError:
        return False
    return True


def _declared_type(entry, name):
    """The type of a declared array the launch does not bind."""
    info = layout.signature(entry)
    if name not in OPTIONAL:
        raise ValueError(f"Reflected {LABEL} binding differs from its source")
    return {"mask": info["maskType"], "lse": "float32"}.get(name, info["dtype"])


def validate(entry, supplied, threads, execution):
    """Check the buffers and launch against the entry; return the plan and words."""
    parameters = layout.parameters(entry)
    outputs = layout.outputs(entry)
    records = layout.records(entry)
    if set(supplied) != set(parameters):
        raise ValueError(f"{LABEL} buffers do not match the attention kernel")
    for name, buffer in supplied.items():
        dtype = layout.parameter_type(entry, name)
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % ctypes.sizeof(CTYPES[dtype])
            or buffer.output != int(name in outputs)
            or (name in records and buffer.count != RECORD_COUNTS[name])
        ):
            raise ValueError(f"{LABEL} buffer type, alignment or direction differs")
    words = {name: buffer_words(supplied[name], "int64") for name in records}
    arrays = [name for name in parameters if name not in records]
    counts = {name: supplied[name].count for name in arrays}
    metadata = layout.plan_launch(entry, counts, words, threads, execution)
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
    return metadata, words


def _record_values(name, words):
    """One record's named fields for ``pack_storage_records``."""
    if name == "mask_params":
        return {"M_strides": list(words)}
    params = layout.decode_params(words)
    return {key: params[key] for key in (*layout.PARAMS_FIELDS, *layout.STRIDE_FIELDS)}


def _record(binding, name, values):
    """One reflected struct record as uint32 words."""
    scalar = binding["scalarLayout"]
    if scalar.get("elementStrideBytes") != RECORD_STRIDES[name]:
        raise ValueError(f"{LABEL} reflected {name} record stride differs")
    if values is None:
        # AttnMaskParams without has_mask: the kernel never reads it.
        values = {
            member["name"]: (
                [0] * member["arrayLength"] if member.get("arrayLength") else 0
            )
            for member in scalar["structMembers"]
        }
    words = pack_storage_records(scalar, [values])
    return {"dtype": "uint32", "shape": [len(words)], "values": words}


def _storage(entry, supplied):
    """Every bound array's words, with poison or guard tails."""
    storage = {}
    for name in layout.parameters(entry):
        if name in layout.RECORDS:
            continue
        dtype = layout.parameter_type(entry, name)
        if name in layout.outputs(entry):
            storage[name] = [GUARDS[dtype]] * (supplied[name].count + GUARD_COUNT)
        else:
            words = buffer_words(supplied[name], dtype)
            storage[name] = words + [POISON[dtype]] * GUARD_COUNT
    return storage


def _inputs(host, entry, descriptor, storage, words):
    """Each reflected binding's value, the output bindings, the writable
    bindings whose function constant is off, and the observed inputs."""
    inputs, observed, outputs, unused, bound = {}, [], {}, {}, set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if provenance.get("kind") == "generated-execution-input":
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if "executionInput" in provenance or name in bound or binding["name"] in inputs:
            raise ValueError(f"Reflected {LABEL} binding differs from its source")
        bound.add(name)
        writable = binding.get("access") == "read_write"
        if name in layout.RECORDS:
            values = _record_values(name, words[name]) if name in words else None
            inputs[binding["name"]] = _record(binding, name, values)
        else:
            if name in storage:
                dtype, values = layout.parameter_type(entry, name), storage[name]
            else:
                # The loader binds a writable parameter whose function
                # constant is off as an output: it starts as guards only.
                dtype = _declared_type(entry, name)
                fill = GUARDS[dtype] if writable else POISON[dtype]
                values = [fill] * GUARD_COUNT
            value = packet(dtype, host.target, values)
            if not reflected_layout_matches(binding["scalarLayout"], value):
                raise ValueError(f"{LABEL} reflected {dtype} storage differs")
            inputs[binding["name"]] = value
        if name in layout.outputs(entry):
            if not writable:
                raise ValueError(f"{LABEL} output {name} is not writable")
            outputs[name] = binding["name"]
        elif writable and name not in storage and name not in layout.RECORDS:
            unused[name] = binding["name"]
        else:
            observed.append(binding["name"])
    if (
        set(outputs) != set(layout.outputs(entry))
        or not set(storage) | set(words) <= bound
    ):
        raise ValueError(f"{LABEL} reflection is incomplete")
    return inputs, observed, outputs, unused, bound


def _readback(entry, result, requested, outputs, unused, supplied, target):
    """Each output's logical words, after checking every guard."""
    if result.status != "ok" or set(result.outputs) != set(requested):
        raise RuntimeError(f"Native {LABEL} execution returned no output")
    for name, binding in unused.items():
        # The kernel must not write a parameter whose function constant is
        # off; it is never committed to MLX.
        dtype = _declared_type(entry, name)
        returned = result.outputs[binding]
        try:
            words = unpack(dtype, target, returned.get("values"))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Native {LABEL} output is outside its storage"
            ) from error
        if words != [GUARDS[dtype]] * GUARD_COUNT:
            raise RuntimeError(f"Native {LABEL} kernel wrote its unbound {name}")
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
    cache = host.steel_attentions
    if cache is None or launch is None or not buffers or count != len(parameters):
        raise ValueError(f"{LABEL} dispatch requires pinned packages and geometry")
    supplied = supplied_buffers(buffers, count, LABEL)
    execution = launch.execution()
    metadata, words = validate(entry, supplied, threads, execution)
    storage = _storage(entry, supplied)
    descriptor, directory = cache.get(entry)
    inputs, observed, outputs, unused, bound = _inputs(
        host, entry, descriptor, storage, words
    )
    requested = {
        binding: {key: item for key, item in inputs[binding].items() if key != "values"}
        for binding in (*outputs.values(), *unused.values())
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
    payloads = _readback(
        entry, result, requested, outputs, unused, supplied, host.target
    )
    input_hashes = {}
    for name in parameters:
        if name in outputs or name in layout.RECORDS:
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
            "steelAttentionMetadata": metadata,
            "records": words,
            "boundParameters": sorted(name for name in bound if name),
            "inputHashes": input_hashes,
            "outputValues": {
                name: values if len(values) <= 4096 else None
                for name, (values, _) in payloads.items()
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
