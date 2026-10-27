"""Execute translated MLX Stockham, Rader, Bluestein and four-step FFT packages and commit guarded outputs.

``FFT`` binds the input and output transform batches, the transform size and
the batch count, as ``mlx/backend/metal/fft.cpp`` binds them to the ``fft``
kernel, in fft.cpp's threadgroups; Rader plans bind the ``rader_fft``
kernel's tables and Rader factor too, and Bluestein plans the
``bluestein_fft`` kernel's chirp ``w_k``, its spectrum ``w_q``, the
transform length and the convolution size. Each launch is checked against
``fft_layout``'s plan for its hash name, each Rader table against the
plan's generator powers and kernel spectrum, and each Bluestein table
against its double-precision value; four-step passes bind ``n1``,
``n2`` and their row or column count, checked against fft.cpp's split.
``complex64`` operands travel as
the ``float2`` vectors fft.metal declares, each holding the real and
imaginary float32 bits; ``float32`` and ``int16`` operands as their bits.
Inputs are followed by poisoned elements no transform reads, the output
starts as guard elements and keeps a guard tail past the elements the
kernel's cooperative copy may touch, only the transforms' elements reach
MLX, and the dispatch is recorded in the trace before the MLX output
changes.
``lazy_package_evidence`` audits each dispatch.
"""

import ctypes
import hashlib
import math
import struct

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
POISON = {"float32": 0x3FC00000, COMPLEX: 0x3FC000003FC00000, "int16": 0x3FC0}
# The absolute error, per table element, that pocketfft's float32 FFT of the
# Rader kernel may differ from its double-precision DFT, per transform size.
RADER_SPECTRUM_TOLERANCE = 1e-5
# The absolute error per element of the float32 Bluestein chirp, rounded
# from double precision, and per element and convolution size of its
# pocketfft float32 spectrum.
BLUESTEIN_CHIRP_TOLERANCE = 1e-6
BLUESTEIN_SPECTRUM_TOLERANCE = 1e-5
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


def _complex(word):
    real, imag = struct.unpack("<ff", struct.pack("<Q", word))
    return complex(real, imag)


def _check_bluestein_tables(info, supplied):
    length = buffer_words(supplied["length"], "int32")[0]
    chirp, spectrum = layout.bluestein_tables(length, info["n"])
    for name, expected, tolerance in (
        ("w_k", chirp, BLUESTEIN_CHIRP_TOLERANCE),
        ("w_q", spectrum, BLUESTEIN_SPECTRUM_TOLERANCE * info["n"]),
    ):
        words = buffer_words(supplied[name], COMPLEX)
        if len(words) != len(expected) or any(
            abs(_complex(word) - value) > tolerance
            for word, value in zip(words, expected)
        ):
            raise ValueError(
                f"{LABEL} Bluestein {name} differs from compute_bluestein_constants"
            )


def check_tables(entry, supplied):
    """Refuse Rader and Bluestein tables other than fft.cpp's for the plan."""
    info = layout.signature(entry)
    if info["kind"] == "bluestein":
        _check_bluestein_tables(info, supplied)
        return
    if info["kind"] != "rader":
        return
    g_q, g_minus_q = layout.rader_generator_tables(info["raderN"])
    if (
        buffer_words(supplied["raders_g_q"], "int16") != g_q
        or buffer_words(supplied["raders_g_minus_q"], "int16") != g_minus_q
    ):
        raise ValueError(f"{LABEL} Rader generator tables differ from fft.cpp's")
    expected = layout.rader_kernel_spectrum(info["raderN"])
    tolerance = RADER_SPECTRUM_TOLERANCE * info["raderN"]
    words = buffer_words(supplied["raders_b_q"], COMPLEX)
    if any(
        abs(_complex(word) - value) > tolerance for word, value in zip(words, expected)
    ):
        raise ValueError(f"{LABEL} Rader kernel spectrum differs from fft.cpp's")


def validate(entry, supplied, threads, execution):
    """Check the buffers and launch against the entry and return metadata."""
    parameters = layout.parameters(entry)
    scalars = layout.scalars(entry)
    if set(supplied) != set(parameters):
        raise ValueError(f"{LABEL} buffers do not match the fft kernel")
    for name, buffer in supplied.items():
        dtype = layout.parameter_type(entry, name)
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % ctypes.sizeof(CTYPES[dtype])
            or buffer.output != int(name == layout.OUTPUT)
            or (name in scalars and buffer.count != 1)
        ):
            raise ValueError(f"{LABEL} buffer type, alignment or direction differs")
    values = {name: buffer_words(supplied[name], "int32")[0] for name in scalars}
    arrays = [name for name in parameters if name not in scalars]
    counts = {name: supplied[name].count for name in arrays}
    metadata = layout.plan_launch(entry, counts, values, threads, execution)
    spans = {}
    for name in arrays:
        size = ctypes.sizeof(CTYPES[layout.parameter_type(entry, name)])
        spans[name] = (supplied[name].data, supplied[name].data + counts[name] * size)
    output = spans[layout.OUTPUT]
    for name, span in spans.items():
        if name != layout.OUTPUT and max(span[0], output[0]) < min(span[1], output[1]):
            raise ValueError(f"{LABEL} output overlaps its input")
    check_tables(entry, supplied)
    return metadata


def native_execution(descriptor, execution):
    """The launch at the workgroup size the translated entry declares.

    Direct3D 12 and OpenGL cap a workgroup's z dimension at 64 threads, so
    fft.cpp's ``(1, batch, threads_per_fft)`` threadgroups past 64 threads per
    transform are declared linearized, ``(batch * threads_per_fft, 1, 1)``,
    with the thread positions rebuilt inside the entry. The workgroup count
    stays fft.cpp's.
    """
    config = descriptor["entryPoint"].get("executionConfig", {})
    declared = config.get("numthreads", config.get("local_size"))
    size = list(execution["workgroupSize"])
    if declared is None or list(declared) == size:
        return execution
    if list(declared) != [math.prod(size), 1, 1]:
        raise ValueError(f"{LABEL} entry declares another workgroup")
    return {**execution, "workgroupSize": list(declared)}


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

    parameters = layout.parameters(entry)
    if host.ffts is None or launch is None or not buffers or count != len(parameters):
        raise ValueError(f"{LABEL} dispatch requires pinned packages and geometry")
    supplied = supplied_buffers(buffers, count, LABEL)
    execution = launch.execution()
    metadata = validate(entry, supplied, threads, execution)
    source_type = layout.parameter_type(entry, "in")
    output_type = layout.parameter_type(entry, layout.OUTPUT)
    size = supplied[layout.OUTPUT].count
    # The elements the kernel may touch: a partial last threadgroup's
    # cooperative copy runs past the transforms (fft_layout.footprint).
    reach = metadata["footprint"]
    storage = {}
    for name in parameters:
        dtype = layout.parameter_type(entry, name)
        if name == layout.OUTPUT:
            storage[name] = [GUARDS[output_type]] * (reach[name] + GUARD_COUNT)
        elif name in layout.scalars(entry):
            storage[name] = buffer_words(supplied[name], "int32")
        else:
            words = buffer_words(supplied[name], dtype)
            tail = reach.get(name, len(words)) - len(words) + GUARD_COUNT
            storage[name] = words + [POISON[dtype]] * tail
    descriptor, directory = host.ffts.get(entry)
    launch_size = execution["workgroupSize"]
    execution = native_execution(descriptor, execution)
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
    end = reach[layout.OUTPUT]
    if len(words) != end + GUARD_COUNT or words[end:] != guard:
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
            **(
                {"launchWorkgroupSize": list(launch_size)}
                if list(launch_size) != execution["workgroupSize"]
                else {}
            ),
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
