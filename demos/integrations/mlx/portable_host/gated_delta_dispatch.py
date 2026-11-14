"""Execute translated MLX gated-delta packages and commit guarded outputs.

``GatedDeltaUpdate::eval_gpu`` binds the row-contiguous queries, keys,
values, initial state, decays and betas, the output, the final state and
``int T``; its backward's save pass also binds the checkpoint cache, and the
sequential backward the cotangents, the cache and the six gradients
(``gated_delta_layout``). Each launch is checked against
``gated_delta_layout.plan`` before its package is translated. Inputs are
followed by poisoned elements no kernel reads; each fresh output starts as
guard elements, and each initialized output as the host's storage, the
zeros eval_gpu filled it with, and both keep a guard tail. Every output is
checked before any reaches MLX, and the dispatch is recorded in the trace
before the MLX outputs change. ``gated_delta_evidence`` audits each record.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import gated_delta_layout as layout
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

LABEL = "Gated delta"
ENTRIES = layout.ENTRIES
LAYOUT_KEYS = ("dtype", "encoding", "shape")


def is_entry(entry):
    return layout.is_entry(entry)


def validate(entry, supplied, threads, execution):
    """Check the buffers and launch; return the plan and ``T``."""
    parameters = layout.parameters(entry)
    directions = layout.directions(entry)
    if set(supplied) != set(parameters):
        raise ValueError(f"{LABEL} buffers do not match the kernel")
    for name, buffer in supplied.items():
        dtype = layout.parameter_type(entry, name)
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % ctypes.sizeof(CTYPES[dtype])
            or buffer.output != directions[name]
            or (name in layout.SCALARS and buffer.count != 1)
        ):
            raise ValueError(f"{LABEL} buffer type, alignment or direction differs")
    steps = buffer_words(supplied["T"], "int32")[0]
    arrays = [name for name in parameters if name not in layout.SCALARS]
    counts = {name: supplied[name].count for name in arrays}
    metadata = layout.plan(entry, counts, steps, threads, execution)
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


def storage(entry, supplied):
    """Every bound array's words, with poison or guard tails."""
    result = {}
    directions = layout.directions(entry)
    for name in layout.parameters(entry):
        if name in layout.SCALARS:
            continue
        dtype = layout.parameter_type(entry, name)
        if directions[name] == 1:
            result[name] = [GUARDS[dtype]] * (supplied[name].count + GUARD_COUNT)
        elif directions[name] == layout.INOUT:
            words = buffer_words(supplied[name], dtype)
            result[name] = words + [GUARDS[dtype]] * GUARD_COUNT
        else:
            words = buffer_words(supplied[name], dtype)
            result[name] = words + [POISON[dtype]] * GUARD_COUNT
    return result


def atomic_packet(scalar, target, words):
    """The packet of a gradient the backward adds through ``mlx_atomic<float>``.

    Each element is a one-member ``{float val}`` record; OpenGL backs it with
    a uint word that holds the float bits.
    """
    word_type = "uint32" if target == "opengl" else "float32"
    if (
        scalar.get("componentCount") != 1
        or [
            (item.get("name"), item.get("offsetBytes"))
            for item in scalar.get("structMembers", ())
        ]
        != [("val", 0)]
        or scalar.get("elementStrideBytes") != 4
        or scalar.get("elementType") != word_type
    ):
        raise ValueError(f"{LABEL} reflected atomic gradient differs")
    if word_type == "uint32":
        return {"dtype": "uint32", "shape": [len(words)], "values": list(words)}
    return packet("float32", target, words)


def bind(host, entry, descriptor, words, steps):
    """Each reflected binding's value, the output bindings and observed inputs."""
    parameters = layout.parameters(entry)
    inactive = layout.inactive(entry)
    declared = parameters + inactive
    inputs, observed, outputs, bound, untouched = {}, [], {}, set(), {}
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if provenance.get("kind") == "generated-execution-input":
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if (
            "executionInput" in provenance
            or name in bound
            or name not in declared
            or binding["name"] in inputs
        ):
            raise ValueError(f"Reflected {LABEL} binding differs from its source")
        bound.add(name)
        if name in layout.SCALARS:
            inputs[binding["name"]] = encode_binding(
                "int32",
                [steps],
                host.target,
                binding["scalarLayout"],
                binding.get("kind", ""),
            )
        elif name in layout.ATOMIC_GRADIENTS:
            inputs[binding["name"]] = atomic_packet(
                binding["scalarLayout"], host.target, words[name]
            )
        else:
            items = words.get(name)
            if name in inactive:
                # A parameter whose function constant is off holds poison
                # only; a writable one is read back to prove the kernel left
                # every element unchanged (untouched).
                items = [POISON["float32"]] * GUARD_COUNT
                if binding.get("access") == "read_write":
                    untouched[binding["name"]] = items
            value = packet(layout.parameter_type(entry, name), host.target, items)
            if not reflected_layout_matches(binding["scalarLayout"], value):
                raise ValueError(f"{LABEL} reflected storage differs")
            inputs[binding["name"]] = value
        if name in inactive:
            continue
        if name in layout.outputs(entry):
            if binding.get("access") != "read_write":
                raise ValueError(f"{LABEL} output {name} is not writable")
            outputs[name] = binding["name"]
        else:
            observed.append(binding["name"])
    if not set(parameters) <= bound or set(outputs) != set(layout.outputs(entry)):
        raise ValueError(f"{LABEL} reflection is incomplete")
    return inputs, observed, outputs, untouched


def require_untouched(entry, result, untouched, target):
    """Each inactive writable binding must read back as its poison, exactly."""
    for binding, items in untouched.items():
        returned = result.outputs.get(binding, {})
        try:
            values = unpack("float32", target, returned.get("values"))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Native {LABEL} inactive binding is invalid") from error
        if values != items:
            raise RuntimeError(f"Native {LABEL} kernel wrote its inactive {binding}")


def readback(entry, result, requested, outputs, words, target):
    """Each output's exact storage after checking its guard tail."""
    if result.status != "ok" or set(result.outputs) != set(requested):
        raise RuntimeError(f"Native {LABEL} execution returned no output")
    payloads = {}
    for name, binding in outputs.items():
        returned = result.outputs[binding]
        if any(
            returned.get(key) != requested[binding].get(key) for key in LAYOUT_KEYS
        ) or not isinstance(returned.get("values"), list):
            raise RuntimeError(f"Native {LABEL} output layout differs")
        dtype = layout.parameter_type(entry, name)
        try:
            if returned.get("dtype") == "uint32" and name in layout.ATOMIC_GRADIENTS:
                values = unpack("uint32", target, returned["values"])
            else:
                values = unpack(dtype, target, returned["values"])
        except ValueError as error:
            raise RuntimeError(
                f"Native {LABEL} output is outside its storage"
            ) from error
        initial = words[name]
        size = len(initial) - GUARD_COUNT
        if len(values) != len(initial) or values[size:] != initial[size:]:
            raise RuntimeError(f"Native {LABEL} kernel changed its {name} guard")
        payloads[name] = bytes((CTYPES[dtype] * size)(*values[:size]))
    return payloads


def input_hashes(entry, supplied):
    """The digest of every bound input array's exact storage."""
    hashes = {}
    for name in layout.parameters(entry):
        if name in layout.outputs(entry) or name in layout.SCALARS:
            continue
        item = ctypes.sizeof(CTYPES[layout.parameter_type(entry, name)])
        hashes[name] = hashlib.sha256(
            ctypes.string_at(supplied[name].data, supplied[name].count * item)
        ).hexdigest()
    return hashes


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    parameters = layout.parameters(entry)
    cache = host.gated_deltas
    if cache is None or launch is None or not buffers or count != len(parameters):
        raise ValueError(f"{LABEL} dispatch requires pinned packages and geometry")
    supplied = supplied_buffers(buffers, count, LABEL)
    execution = launch.execution()
    metadata, steps = validate(entry, supplied, threads, execution)
    words = storage(entry, supplied)
    descriptor, directory = cache.get(entry)
    inputs, observed, outputs, untouched = bind(host, entry, descriptor, words, steps)
    requested = {
        binding: {key: item for key, item in inputs[binding].items() if key != "values"}
        for binding in (*outputs.values(), *untouched)
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
    payloads = readback(entry, result, requested, outputs, words, host.target)
    require_untouched(entry, result, untouched, host.target)
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
            "counts": {
                name: supplied[name].count
                for name in parameters
                if name not in layout.SCALARS
            },
            "inputHashes": input_hashes(entry, supplied),
            "outputHashes": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in payloads.items()
            },
        },
    )
    for name, payload in payloads.items():
        ctypes.memmove(supplied[name].data, payload, len(payload))
    host.dispatch_count += 1
