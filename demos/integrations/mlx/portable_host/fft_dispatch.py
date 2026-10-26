"""Execute translated MLX Stockham FFT packages and commit guarded outputs.

``FFT`` binds the input and output transform batches, the transform size and
the batch count, as ``mlx/backend/metal/fft.cpp`` binds them to the ``fft``
kernel, in fft.cpp's threadgroups. Each launch is checked against
``fft_layout``'s plan for its hash name. ``complex64`` operands travel as the
``float2`` vectors fft.metal declares, each holding the real and imaginary
float32 bits; ``float32`` operands as their bits. Inputs are followed by
poisoned elements no transform reads, the output starts as guard elements
and keeps a guard tail, and the dispatch is recorded in the trace before the
MLX output changes. ``lazy_package_evidence`` audits each dispatch.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import fft_layout as layout
from demos.integrations.mlx.portable_host.element_storage import (
    COMPLEX,
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

LABEL = "FFT"
ENTRY_PREFIXES = layout.ENTRY_PREFIXES
# Read by no transform: 1.5 in every float32 part.
POISON = {"float32": 0x3FC00000, COMPLEX: 0x3FC000003FC00000}
# How every target reflects fft.metal's float2 storage element.
VECTOR_LAYOUT = {
    "physicalType": "float2",
    "elementType": "float32",
    "elementSizeBytes": 8,
    "elementStrideBytes": 8,
    "vectorWidth": 2,
}


def is_entry(entry):
    try:
        layout.signature(entry)
    except ValueError:
        return False
    return True


def encode(dtype, words, target, scalar, kind):
    """One reflected binding's runtime value for ``dtype`` storage words."""
    if dtype == "int32":
        return encode_binding(dtype, words, target, scalar, kind)
    value = packet(dtype, target, words)
    matches = (
        all(scalar.get(key) == item for key, item in VECTOR_LAYOUT.items())
        if dtype == COMPLEX
        else reflected_layout_matches(scalar, value)
    )
    if not matches:
        raise ValueError(f"{LABEL} reflected {dtype} storage differs")
    return value


def validate(entry, supplied, threads, execution):
    """Check the buffers and launch against the entry and return metadata."""
    if set(supplied) != set(layout.PARAMETERS):
        raise ValueError(f"{LABEL} buffers do not match the fft kernel")
    for name, buffer in supplied.items():
        dtype = layout.parameter_type(entry, name)
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % ctypes.sizeof(CTYPES[dtype])
            or buffer.output != int(name == layout.OUTPUT)
            or (name in layout.SCALARS and buffer.count != 1)
        ):
            raise ValueError(f"{LABEL} buffer type, alignment or direction differs")
    (n,) = buffer_words(supplied["n"], "int32")
    (total,) = buffer_words(supplied["batch_size"], "int32")
    counts = {name: supplied[name].count for name in ("in", "out")}
    metadata = layout.plan_launch(entry, counts, n, total, threads, execution)
    spans = {}
    for name in ("in", "out"):
        size = ctypes.sizeof(CTYPES[layout.parameter_type(entry, name)])
        spans[name] = (supplied[name].data, supplied[name].data + counts[name] * size)
    if max(spans["in"][0], spans["out"][0]) < min(spans["in"][1], spans["out"][1]):
        raise ValueError(f"{LABEL} output overlaps its input")
    return metadata


def _inputs(host, entry, descriptor, storage):
    """Each reflected binding's value, the output binding and observed inputs."""
    inputs, observed, output = {}, [], None
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if provenance.get("kind") == "generated-execution-input":
            # threads_per_grid: the request derives it from the launch.
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if (
            "executionInput" in provenance
            or name not in storage
            or binding["name"] in inputs
            or any(item == name for item in (output, *observed))
        ):
            raise ValueError(f"Reflected {LABEL} binding differs from its source")
        inputs[binding["name"]] = encode(
            layout.parameter_type(entry, name),
            storage[name],
            host.target,
            binding["scalarLayout"],
            binding.get("kind", ""),
        )
        if name == layout.OUTPUT:
            if binding.get("access") != "read_write":
                raise ValueError(f"{LABEL} output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    if output is None or len(inputs) != len(storage):
        raise ValueError(f"{LABEL} reflection is incomplete")
    return inputs, observed, output


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.ffts is None or launch is None or not buffers or count != 4:
        raise ValueError(f"{LABEL} dispatch requires pinned packages and geometry")
    supplied = supplied_buffers(buffers, count, LABEL)
    execution = launch.execution()
    metadata = validate(entry, supplied, threads, execution)
    source_type = layout.parameter_type(entry, "in")
    output_type = layout.parameter_type(entry, layout.OUTPUT)
    source = buffer_words(supplied["in"], source_type)
    size = supplied[layout.OUTPUT].count
    storage = {
        "in": source + [POISON[source_type]] * GUARD_COUNT,
        layout.OUTPUT: [GUARDS[output_type]] * (size + GUARD_COUNT),
        "n": [metadata["n"]],
        "batch_size": [metadata["batchSize"]],
    }
    descriptor, directory = host.ffts.get(entry)
    inputs, observed, output = _inputs(host, entry, descriptor, storage)
    outputs = {output: {k: v for k, v in inputs[output].items() if k != "values"}}
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
    returned = result.outputs[output]
    if any(
        returned.get(key) != outputs[output].get(key)
        for key in ("dtype", "encoding", "shape")
    ) or not isinstance(returned.get("values"), list):
        raise RuntimeError(f"Native {LABEL} output layout differs")
    try:
        words = unpack(output_type, host.target, returned["values"])
    except ValueError as error:
        raise RuntimeError(f"Native {LABEL} output is outside its storage") from error
    guard = [GUARDS[output_type]] * GUARD_COUNT
    if len(words) != size + GUARD_COUNT or words[size:] != guard:
        raise RuntimeError(f"Native {LABEL} kernel changed its output guard")
    payload = bytes((CTYPES[output_type] * size)(*words[:size]))
    item = ctypes.sizeof(CTYPES[source_type])
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
            "fftMetadata": metadata,
            "inputHash": (
                hashlib.sha256(
                    ctypes.string_at(supplied["in"].data, supplied["in"].count * item)
                ).hexdigest()
            ),
            "outputValues": words[:size] if size <= 4096 else None,
            "outputHash": hashlib.sha256(payload).hexdigest(),
        },
    )
    ctypes.memmove(supplied[layout.OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
