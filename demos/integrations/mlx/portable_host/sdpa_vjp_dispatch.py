"""Execute translated MLX attention VJP packages and commit guarded outputs.

``sdpa_vjp_odo`` binds the forward output, its cotangent, the per-row
products and the query length; ``sdpa_vjp_ds`` the block's scores, output
gradients, logsumexp and row products, its score gradients and
probabilities, and the ``SDPAVJPTileParams`` record; and
``sdpa_vjp_reduce_add`` / ``sdpa_vjp_reduce_set`` the block gradients, the
float32 accumulator, the gradient output and their extents, as
``mlx/backend/metal/scaled_dot_product_attention.cpp`` binds them. The host
passes the record as int64 words in field order, with the scale as its
binary32 bits. Each launch is checked against ``sdpa_vjp_layout``'s plan.
Inputs are followed by poisoned elements no kernel reads, and each fresh
output starts as guard elements and keeps a guard tail. The reduction's
accumulator and gradient output start as the host's storage, since each
launch writes one window of rows; every element outside that window must
come back unchanged. Every output is checked before any reaches MLX, and
the dispatch is recorded in the trace before the MLX outputs change.
``lazy_package_evidence`` audits each dispatch.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request, pack_storage_records
from demos.integrations.mlx.portable_host import sdpa_vjp_layout as layout
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
from demos.integrations.mlx.portable_host.sdpa_dispatch import POISON
from demos.integrations.mlx.portable_host.trace_records import write_trace

LABEL = "SDPA VJP"
ENTRY_PREFIXES = layout.ENTRY_PREFIXES
# SDPAVJPTileParams: seven ints and a float.
TILE_STRIDE = 32


def is_entry(entry):
    try:
        layout.signature(entry)
    except ValueError:
        return False
    return True


def validate(entry, supplied, threads, execution):
    """Check the buffers and launch; return the plan, scalars and words."""
    parameters = layout.parameters(entry)
    directions = layout.directions(entry)
    scalars = layout.scalars(entry)
    records = layout.records(entry)
    if set(supplied) != set(parameters):
        raise ValueError(f"{LABEL} buffers do not match the attention kernel")
    for name, buffer in supplied.items():
        dtype = layout.parameter_type(entry, name)
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % ctypes.sizeof(CTYPES[dtype])
            or buffer.output != directions[name]
            or (name in scalars and buffer.count != 1)
            or (name in records and buffer.count != len(layout.TILE_FIELDS))
        ):
            raise ValueError(f"{LABEL} buffer type, alignment or direction differs")
    values = {name: buffer_words(supplied[name], "int32")[0] for name in scalars}
    words = {name: buffer_words(supplied[name], "int64") for name in records}
    arrays = [name for name in parameters if name not in scalars + records]
    counts = {name: supplied[name].count for name in arrays}
    metadata = layout.plan_launch(entry, counts, values, words, threads, execution)
    spans = {}
    for name in arrays:
        size = ctypes.sizeof(CTYPES[layout.parameter_type(entry, name)])
        spans[name] = (supplied[name].data, supplied[name].data + counts[name] * size)
    for name in layout.outputs(entry):
        for other, span in spans.items():
            if other != name and max(span[0], spans[name][0]) < min(
                span[1], spans[name][1]
            ):
                raise ValueError(f"{LABEL} output {name} overlaps {other}")
    return metadata, values, words


def _storage(entry, supplied):
    """Every bound array's words, with poison or guard tails."""
    storage = {}
    directions = layout.directions(entry)
    for name in layout.parameters(entry):
        if name in layout.scalars(entry) + layout.records(entry):
            continue
        dtype = layout.parameter_type(entry, name)
        if directions[name] == 1:
            storage[name] = [GUARDS[dtype]] * (supplied[name].count + GUARD_COUNT)
        elif directions[name] == layout.INOUT:
            words = buffer_words(supplied[name], dtype)
            storage[name] = words + [GUARDS[dtype]] * GUARD_COUNT
        else:
            words = buffer_words(supplied[name], dtype)
            storage[name] = words + [POISON[dtype]] * GUARD_COUNT
    return storage


def _record(binding, metadata):
    """``SDPAVJPTileParams`` as the reflected record's uint32 words."""
    scalar = binding["scalarLayout"]
    if scalar.get("elementStrideBytes") != TILE_STRIDE:
        raise ValueError(f"{LABEL} reflected tile record stride differs")
    fields = {name: metadata[name] for name in layout.TILE_FIELDS}
    words = pack_storage_records(scalar, [fields])
    return {"dtype": "uint32", "shape": [len(words)], "values": words}


def _inputs(host, entry, descriptor, storage, values, metadata):
    """Each reflected binding's value, the output bindings and observed inputs."""
    inputs, observed, outputs, bound = {}, [], {}, set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if provenance.get("kind") == "generated-execution-input":
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if (
            "executionInput" in provenance
            or name in bound
            or name not in layout.parameters(entry)
            or binding["name"] in inputs
        ):
            raise ValueError(f"Reflected {LABEL} binding differs from its source")
        bound.add(name)
        if name in layout.records(entry):
            inputs[binding["name"]] = _record(binding, metadata)
        elif name in layout.scalars(entry):
            inputs[binding["name"]] = encode_binding(
                "int32",
                [values[name]],
                host.target,
                binding["scalarLayout"],
                binding.get("kind", ""),
            )
        else:
            value = packet(
                layout.parameter_type(entry, name), host.target, storage[name]
            )
            if not reflected_layout_matches(binding["scalarLayout"], value):
                raise ValueError(f"{LABEL} reflected storage differs")
            inputs[binding["name"]] = value
        if name in layout.outputs(entry):
            if binding.get("access") != "read_write":
                raise ValueError(f"{LABEL} output {name} is not writable")
            outputs[name] = binding["name"]
        else:
            observed.append(binding["name"])
    if bound != set(layout.parameters(entry)):
        raise ValueError(f"{LABEL} reflection is incomplete")
    return inputs, observed, outputs, bound


def _windows(entry, metadata, size):
    """The element ranges one launch may write in each output."""
    if layout.signature(entry)["kind"] != "reduce":
        return [(0, size)]
    dim, acc_rows = metadata["dim"], metadata["accRows"]
    start, rows = metadata["rowOffset"], metadata["rows"]
    return [
        ((head * acc_rows + start) * dim, (head * acc_rows + start + rows) * dim)
        for head in range(metadata["heads"])
    ]


def _readback(entry, result, requested, outputs, storage, metadata, target):
    """Each output's logical words, after checking every guard and every
    element outside the launch's windows."""
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
        initial = storage[name]
        size = len(initial) - GUARD_COUNT
        if len(words) != len(initial) or words[size:] != initial[size:]:
            raise RuntimeError(f"Native {LABEL} kernel changed its {name} guard")
        previous = 0
        for start, end in [*_windows(entry, metadata, size), (size, size)]:
            if words[previous:start] != initial[previous:start]:
                raise RuntimeError(
                    f"Native {LABEL} kernel wrote {name} outside its window"
                )
            previous = end
        payloads[name] = (words[:size], bytes((CTYPES[dtype] * size)(*words[:size])))
    return payloads


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    parameters = layout.parameters(entry)
    cache = host.sdpa_vjps
    if cache is None or launch is None or not buffers or count != len(parameters):
        raise ValueError(f"{LABEL} dispatch requires pinned packages and geometry")
    supplied = supplied_buffers(buffers, count, LABEL)
    execution = launch.execution()
    metadata, values, words = validate(entry, supplied, threads, execution)
    storage = _storage(entry, supplied)
    descriptor, directory = cache.get(entry)
    inputs, observed, outputs, bound = _inputs(
        host, entry, descriptor, storage, values, metadata
    )
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
    payloads = _readback(
        entry, result, requested, outputs, storage, metadata, host.target
    )
    input_hashes = {}
    for name in parameters:
        if name in outputs or name in layout.scalars(entry) + layout.records(entry):
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
            "sdpaVjpMetadata": metadata,
            "scalars": values,
            "records": words,
            "boundParameters": sorted(name for name in bound if name),
            "inputHashes": input_hashes,
            "outputValues": {
                name: items if len(items) <= 4096 else None
                for name, (items, _) in payloads.items()
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
