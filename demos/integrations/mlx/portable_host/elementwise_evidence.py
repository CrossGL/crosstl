"""Audit lazily translated unary and binary dispatches.

Every dispatch must name the package its cache derives again from the pinned
source, and its native request and modules must match the event. Integer and
Boolean operands are then recomputed exactly from the pinned ``unary_ops.h``
and ``binary_ops.h``: C++ promotes narrow operands to ``int``, so results
wrap modulo their storage width and Booleans keep only whether the promoted
result is nonzero. Floating operands are checked by package and native
identity like the prebuilt float families; their values are compared with
the CPU results by the upstream tests themselves.
"""

import ctypes
import hashlib

from demos.integrations.mlx.portable_host.copy_evidence import integer_bounds, wrap
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    FLOATING,
    guard_words,
    packet,
)
from demos.integrations.mlx.portable_host.elementwise_dispatch import operands
from demos.integrations.mlx.portable_host.elementwise_packages import signature
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)


def _divide(x, y):
    """C++ integer division, which truncates toward zero."""
    quotient = abs(x) // abs(y)
    return quotient if (x < 0) == (y < 0) else -quotient


def _remainder(x, y, dtype):
    # Unsigned operands use x % y; signed ones move a nonzero remainder to
    # the divisor's sign, which is Python's floored ``%``.
    return x % y if dtype.startswith("int") else x - y * _divide(x, y)


def _power(base, exponent, dtype):
    if exponent < 0:
        return 0
    bits = 1 if dtype == "bool_" else 8 * ctypes.sizeof(CTYPES[dtype])
    return pow(base, exponent, 1 << bits)


UNARY = {
    "Abs": abs,
    "Ceil": lambda x: x,
    "Floor": lambda x: x,
    "Negative": lambda x: -x,
    "Sign": lambda x: (x > 0) - (x < 0),
    "Square": lambda x: x * x,
    "BitwiseInvert": lambda x: ~x,
    "LogicalNot": lambda x: not x,
}
BINARY = {
    "Add": lambda x, y, _: x + y,
    "Subtract": lambda x, y, _: x - y,
    "Multiply": lambda x, y, _: x * y,
    "Divide": lambda x, y, _: _divide(x, y),
    "Remainder": _remainder,
    "Maximum": lambda x, y, _: max(x, y),
    "Minimum": lambda x, y, _: min(x, y),
    "Power": _power,
    "Equal": lambda x, y, _: x == y,
    "NotEqual": lambda x, y, _: x != y,
    "Less": lambda x, y, _: x < y,
    "LessEqual": lambda x, y, _: x <= y,
    "Greater": lambda x, y, _: x > y,
    "GreaterEqual": lambda x, y, _: x >= y,
    "LogicalAnd": lambda x, y, _: bool(x and y),
    "LogicalOr": lambda x, y, _: bool(x or y),
    "BitwiseAnd": lambda x, y, _: x & y,
    "BitwiseOr": lambda x, y, _: x | y,
    "BitwiseXor": lambda x, y, _: x ^ y,
    "LeftShift": lambda x, y, _: x << y,
    "RightShift": lambda x, y, _: x >> y,
}


def stored(value, dtype):
    """The ``dtype`` storage word of a promoted integer result."""
    if dtype == "bool_":
        return int(value != 0)
    return wrap(int(value), dtype)


def expected_words(kernel, operation, source, destination, values):
    """Every output word of an integer or Boolean unary or binary vector."""
    require(source not in FLOATING, "Elementwise recompute requires integer operands")
    low, high = integer_bounds(source) if source != "bool_" else (0, 1)
    for words in values.values():
        require(
            all(type(word) is int and low <= word <= high for word in words),
            "Elementwise operand outside its storage range",
        )
    if kernel == "v_":
        function = UNARY[operation]
        return [stored(function(word), destination) for word in values["in"]]
    function = BINARY[operation]
    return [
        stored(function(x, y, source), destination)
        for x, y in zip(values["a"], values["b"], strict=True)
    ]


def recomputable(entry):
    """Whether the audit recomputes ``entry``'s results from its operands."""
    return signature(entry)[2] not in FLOATING


def audit_event(event, cache=None, *, native=True):
    """Check one lazy unary or binary dispatch; returns the outputs checked.

    ``cache`` derives the event's package again from the pinned source;
    ``native`` also verifies the retained native modules and the request.
    """
    entry, target = event["entry"], event["target"]
    kernel, operation, source, destination = signature(entry)
    metadata = event["elementwiseMetadata"]
    require(
        metadata == {
            "kernel": kernel,
            "operation": operation,
            "source": source,
            "destination": destination,
        },
        "Elementwise event metadata does not match its entry",
    )
    threads = event["threads"]
    if recomputable(entry):
        names, _ = operands(kernel)
        values, outputs = event["operandValues"], event["outputValues"]
        require(
            set(values) == set(names)
            and all(len(values[name]) == threads for name in names)
            and len(outputs) == threads
            and event["initialValues"] == guard_words(destination, threads),
            "Elementwise readback extent differs",
        )
        uploads = [*event["inputs"].values()]
        for value in (
            *(packet(source, target, values[name]) for name in names),
            packet(
                destination,
                target,
                guard_words(destination, threads) + guard_words(destination),
            ),
        ):
            require(value in uploads, "Elementwise upload differs from its values")
        expected = expected_words(kernel, operation, source, destination, values)
        for index, (actual, wanted) in enumerate(zip(outputs, expected)):
            require(
                actual == wanted,
                f"Elementwise result {index} of {entry} differs from its operands",
            )
        payload = bytes((CTYPES[destination] * threads)(*outputs))
        require(
            hashlib.sha256(payload).hexdigest() == event.get("outputHash"),
            "Elementwise host output hash changed",
        )
    if cache is not None:
        descriptor, directory = cache.get(entry)
        require_unchanged(descriptor, directory)
        require(
            descriptor["artifact"] == event["artifact"]
            and str(directory) == event["packageRoot"],
            "Elementwise dispatch does not name its cached package",
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
            "Elementwise native request changed",
        )
        audit_native_execution(event)
    return threads
