"""Execute translated MLX gated-delta packages and commit guarded outputs.

The forward kernels bind the queries, keys, values, initial state, decays,
betas, output and final state and ``T`` as ``gated_delta_update.cpp`` binds
them, and the save pass also its state cache; the backward binds the
forward's inputs, both cotangents and the cache, ``T`` and the six
gradients. Each launch is checked against ``gated_delta_layout``'s plan,
which derives every buffer's exact span from the batch and ``T``. Inputs are
followed by poisoned elements no kernel reads, and Metal's inference state
cache, declared under the off ``save_state`` constant, holds poison only; when
reflection marks it writable it is read back and must still hold that poison.
Each output keeps a guard
tail; one that upstream zero-fills before the launch starts from those
zeros, so the backward's atomic sums add to them, and every other output
starts as guard elements. Every output is checked before any reaches MLX,
and the dispatch is recorded in the trace before the MLX outputs change.
"""

import array
import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host import gated_delta_layout as layout
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    GUARDS,
    packet,
    reflected_layout_matches,
    supplied_buffers,
    unpack,
)
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.matmul_dispatch import encode_binding
from demos.integrations.mlx.portable_host.scatter_layout import (
    ATOMIC_TYPES,
    atomic_storage_dtype,
)
from demos.integrations.mlx.portable_host.trace_records import write_trace

LABEL = "Gated delta"
ENTRY_PREFIXES = layout.ENTRY_PREFIXES
# Read by no kernel: 1.5 in every floating type.
POISON = {"float32": 0x3FC00000, "float16": 0x3E00, "bfloat16": 0x3FC0}
INOUT = 2
# A parameter the kernel declares under an off function constant: Metal keeps
# the inference kernel's state cache, which it never reads or writes.
DECLARED = {"state_cache": "float32"}
# array typecodes of the unsigned storage words.
TYPECODES = {"float32": "I", "float16": "H", "bfloat16": "H", "int32": "i"}


def is_entry(entry):
    try:
        layout.signature(entry)
    except ValueError:
        return False
    return True


def _words(buffer, dtype):
    """A native buffer's storage words, read through one memoryview."""
    ctype = CTYPES[dtype]
    storage = (ctype * buffer.count).from_address(buffer.data)
    return memoryview(storage).cast("B").cast(TYPECODES[dtype]).tolist()


def _direction(entry, name):
    if name in layout.initialized(entry):
        return INOUT
    return int(name in layout.outputs(entry))


def validate(entry, supplied, threads, execution):
    """Check the buffers and launch against the entry and return metadata."""
    parameters = layout.parameters(entry)
    if set(supplied) != set(parameters):
        raise ValueError(f"{LABEL} buffers do not match the kernel")
    for name, buffer in supplied.items():
        dtype = layout.parameter_type(entry, name)
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % ctypes.sizeof(CTYPES[dtype])
            or buffer.output != _direction(entry, name)
            or (name in layout.scalars(entry) and buffer.count != 1)
        ):
            raise ValueError(f"{LABEL} buffer type, alignment or direction differs")
    (steps,) = _words(supplied["T"], "int32")
    arrays = [name for name in parameters if name not in layout.scalars(entry)]
    counts = {name: supplied[name].count for name in arrays}
    metadata = layout.plan_launch(entry, counts, {"T": steps}, threads, execution)
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
    return metadata, steps


def _storage(entry, supplied):
    """Every bound array's words, with poison or guard tails."""
    storage = {}
    for name in layout.parameters(entry):
        if name in layout.scalars(entry):
            continue
        dtype = layout.parameter_type(entry, name)
        direction = _direction(entry, name)
        if direction == 1:
            storage[name] = [GUARDS[dtype]] * (supplied[name].count + GUARD_COUNT)
        elif direction == INOUT:
            storage[name] = (
                _words(supplied[name], dtype) + [GUARDS[dtype]] * GUARD_COUNT
            )
        else:
            storage[name] = (
                _words(supplied[name], dtype) + [POISON[dtype]] * GUARD_COUNT
            )
    return storage


def _atomic_packet(target, words):
    """An ``mlx_atomic<float>`` gradient's words as one-member struct elements.

    Metal and DirectX reflect the member as float, OpenGL as the uint word its
    compare-and-swap loop adds through; either carries the float32 bits.
    """
    storage = atomic_storage_dtype("float32", target)
    value = {"dtype": storage, "shape": [len(words), 1], "values": list(words)}
    if storage == "float32":
        value["encoding"] = FLOAT32_BITS
    return value


def _atomic_layout_matches(scalar, value):
    return (
        scalar.get("elementType") == value["dtype"]
        and scalar.get("elementStrideBytes") == 4
        and scalar.get("componentCount") == 1
        and scalar.get("structMembers")
        == [
            {
                "name": "val",
                "offsetBytes": 0,
                "physicalType": ATOMIC_TYPES[value["dtype"]],
            }
        ]
    )


def _inputs(host, entry, descriptor, storage, steps):
    """Each reflected binding's value, the outputs, observed and unused bindings.

    A declared parameter is reflected writable when the kernel writes it under
    its off constant; it is then read back and must still hold its poison.
    """
    inputs, observed, outputs, bound, unused = {}, [], {}, set(), set()
    untouched = {}
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if provenance.get("kind") == "generated-execution-input":
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        declared = name in DECLARED and name not in layout.parameters(entry)
        if (
            "executionInput" in provenance
            or name in bound
            or (name not in layout.parameters(entry) and not declared)
            or binding["name"] in inputs
        ):
            raise ValueError(f"Reflected {LABEL} binding differs from its source")
        bound.add(name)
        if declared:
            unused.add(name)
            value = packet(
                DECLARED[name], host.target, [POISON[DECLARED[name]]] * GUARD_COUNT
            )
            if not reflected_layout_matches(binding["scalarLayout"], value):
                raise ValueError(f"{LABEL} reflected {name} storage differs")
            inputs[binding["name"]] = value
            if binding.get("access") == "read_write":
                untouched[name] = binding["name"]
            else:
                observed.append(binding["name"])
            continue
        if name in layout.scalars(entry):
            inputs[binding["name"]] = encode_binding(
                "int32",
                [steps],
                host.target,
                binding["scalarLayout"],
                binding.get("kind", ""),
            )
        else:
            dtype = layout.parameter_type(entry, name)
            if name in layout.atomics(entry):
                value = _atomic_packet(host.target, storage[name])
                matches = _atomic_layout_matches(binding["scalarLayout"], value)
            else:
                value = packet(dtype, host.target, storage[name])
                matches = reflected_layout_matches(binding["scalarLayout"], value)
            if not matches:
                raise ValueError(f"{LABEL} reflected {name} storage differs")
            inputs[binding["name"]] = value
        if name in layout.outputs(entry):
            if binding.get("access") != "read_write":
                raise ValueError(f"{LABEL} output {name} is not writable")
            outputs[name] = binding["name"]
        else:
            observed.append(binding["name"])
    if set(outputs) != set(layout.outputs(entry)) or bound - unused != set(
        layout.parameters(entry)
    ):
        raise ValueError(f"{LABEL} reflection is incomplete")
    return inputs, observed, outputs, untouched


def _check_untouched(result, requested, untouched, target):
    """Fail unless every declared writable binding still holds its poison."""
    for name, binding in untouched.items():
        returned = result.outputs[binding]
        if any(
            returned.get(key) != requested[binding].get(key)
            for key in ("dtype", "encoding", "shape")
        ) or not isinstance(returned.get("values"), list):
            raise RuntimeError(f"Native {LABEL} {name} layout differs")
        dtype = DECLARED[name]
        try:
            words = unpack(dtype, target, returned["values"])
        except ValueError as error:
            raise RuntimeError(
                f"Native {LABEL} {name} is outside its storage"
            ) from error
        if words != [POISON[dtype]] * GUARD_COUNT:
            raise RuntimeError(f"Native {LABEL} kernel wrote its unused {name}")


def _readback(entry, result, requested, outputs, untouched, supplied, target):
    """Each output's logical words, after checking every guard."""
    if result.status != "ok" or set(result.outputs) != set(requested):
        raise RuntimeError(f"Native {LABEL} execution returned no output")
    _check_untouched(result, requested, untouched, target)
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
        payload = array.array(TYPECODES[dtype], words[:size]).tobytes()
        payloads[name] = (words[:size], payload)
    return payloads


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    parameters = layout.parameters(entry)
    if (
        host.gated_deltas is None
        or launch is None
        or not buffers
        or count != len(parameters)
    ):
        raise ValueError(f"{LABEL} dispatch requires pinned packages and geometry")
    supplied = supplied_buffers(buffers, count, LABEL)
    execution = launch.execution()
    metadata, steps = validate(entry, supplied, threads, execution)
    storage = _storage(entry, supplied)
    descriptor, directory = host.gated_deltas.get(entry)
    inputs, observed, outputs, untouched = _inputs(
        host, entry, descriptor, storage, steps
    )
    requested = {
        binding: {key: item for key, item in inputs[binding].items() if key != "values"}
        for binding in (*outputs.values(), *untouched.values())
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
        entry, result, requested, outputs, untouched, supplied, host.target
    )
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
            "gatedDeltaMetadata": metadata,
            "steps": steps,
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
