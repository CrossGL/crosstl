"""Audit lazily translated ``arange`` dispatches.

Every dispatch must name the package its cache derives again from the pinned
source, and its native request and modules must match the event.
``out[index] = start + index * step`` promotes 8- and 16-bit integers to 32
bits and wraps modulo the storage width, so integer events are recomputed
exactly; float16 and bfloat16 events are checked by package and native
identity like the prebuilt float families, and their values are compared with
the CPU results by the upstream tests themselves.
"""

import hashlib

from demos.integrations.mlx.portable_host.arange_dispatch import SCALARS
from demos.integrations.mlx.portable_host.arange_packages import signature
from demos.integrations.mlx.portable_host.copy_evidence import integer_bounds, wrap
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    INEXACT,
    guard_words,
    packet,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)

__all__ = ("audit_event", "expected_words", "recomputable")


def recomputable(entry):
    """Whether the audit recomputes ``entry``'s results from its scalars."""
    return signature(entry) not in INEXACT


def expected_words(dtype, start, step, threads):
    """Every output word of an integer arange: ``start + index * step``, wrapped."""
    require(dtype not in INEXACT, "Arange recompute requires an integer type")
    low, high = integer_bounds(dtype)
    require(
        all(type(word) is int and low <= word <= high for word in (start, step)),
        "Arange scalar outside its storage range",
    )
    return [wrap(start + index * step, dtype) for index in range(threads)]


def audit_event(event, cache=None, *, native=True):
    """Check one lazy arange dispatch; returns the outputs checked.

    ``cache`` derives the event's package again from the pinned source;
    ``native`` also verifies the retained native modules and the request.
    """
    entry, target, threads = event["entry"], event["target"], event["threads"]
    dtype = signature(entry)
    require(
        event["arangeMetadata"] == {"dtype": dtype},
        "Arange event metadata does not match its entry",
    )
    if recomputable(entry):
        values, outputs = event["operandValues"], event["outputValues"]
        require(
            set(values) == set(SCALARS)
            and all(len(values[name]) == 1 for name in SCALARS)
            and len(outputs) == threads
            and event["initialValues"] == guard_words(dtype, threads),
            "Arange readback extent differs",
        )
        uploads = [*event["inputs"].values()]
        for value in (
            *(packet(dtype, target, values[name]) for name in SCALARS),
            packet(dtype, target, guard_words(dtype, threads) + guard_words(dtype)),
        ):
            require(value in uploads, "Arange upload differs from its values")
        expected = expected_words(dtype, values["start"][0], values["step"][0], threads)
        for index, (actual, wanted) in enumerate(zip(outputs, expected)):
            require(
                actual == wanted,
                f"Arange result {index} of {entry} differs from its scalars",
            )
        payload = bytes((CTYPES[dtype] * threads)(*outputs))
        require(
            hashlib.sha256(payload).hexdigest() == event.get("outputHash"),
            "Arange host output hash changed",
        )
    if cache is not None:
        descriptor, directory = cache.get(entry)
        require_unchanged(descriptor, directory)
        require(
            descriptor["artifact"] == event["artifact"]
            and str(directory) == event["packageRoot"],
            "Arange dispatch does not name its cached package",
        )
    if native:
        request = event["details"]["request"]
        require(
            request["target"] == target
            and request["entryPoint"]
            == {"opengl": "main", "directx": "CSMain"}.get(target, entry)
            and request["dispatch"]["workgroupCount"]
            == event["workgroupCount"]
            == [threads, 1, 1]
            and request["dispatch"]["workgroupSize"]
            == event["workgroupSize"]
            == [1, 1, 1]
            and event["dispatchVersion"] == 3,
            "Arange native request changed",
        )
        audit_native_execution(event)
    return threads
