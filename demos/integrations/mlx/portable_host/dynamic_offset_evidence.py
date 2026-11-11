"""Audit translated ``compute_dynamic_offset`` dispatches.

Every dispatch must name the package its cache derives again from the pinned
template, upload exactly its retained operands and keep its output guard, and
its native request and modules must match the event. The offset is
recomputed as ``sum(indices[i] * strides[axes[i]])``, which C++ evaluates in
64 bits and stores as an int64.
"""

import ctypes
import hashlib

from demos.integrations.mlx.portable_host.copy_evidence import wrap
from demos.integrations.mlx.portable_host.dynamic_offset_dispatch import (
    LAUNCH,
    MAX_AXES,
    operand_types,
)
from demos.integrations.mlx.portable_host.dynamic_offset_packages import signature
from demos.integrations.mlx.portable_host.element_storage import guard_words, packet
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)

__all__ = ("audit_event", "expected_offset")


def expected_offset(indices, strides, axes):
    """``sum(indices[i] * strides[axes[i]])`` as the int64 MLX stores."""
    return wrap(
        sum(index * strides[axis] for index, axis in zip(indices, axes)), "int64"
    )


def audit_event(event, cache=None, *, native=True):
    """Recompute one dynamic offset dispatch from its retained operands.

    ``cache`` derives the event's package again from the pinned template;
    ``native`` also verifies the retained native modules and the request.
    """
    entry, target = event["entry"], event["target"]
    metadata = event["dynamicOffsetMetadata"]
    dtype = signature(entry)
    indices, strides, axes = (
        metadata.get(key) for key in ("indices", "strides", "axes")
    )
    require(
        metadata.get("dtype") == dtype
        and set(metadata) == {"dtype", "indices", "strides", "axes"}
        and all(isinstance(item, list) for item in (indices, strides, axes))
        and 0 < len(axes) == len(indices) <= MAX_AXES
        and all(type(axis) is int and 0 <= axis < len(strides) for axis in axes),
        "Dynamic offset event metadata does not match its entry",
    )
    types = operand_types(entry)
    uploads = [*event["inputs"].values()]
    for name, words in (
        ("indices", indices),
        ("strides", strides),
        ("axes", axes),
        ("n_axes", [len(axes)]),
    ):
        require(
            packet(types[name], target, words) in uploads,
            "Dynamic offset upload differs from its operands",
        )
    require(
        event["initialValues"] == guard_words("int64", 1)
        and packet("int64", target, guard_words("int64", 1) + guard_words("int64"))
        in uploads,
        "Dynamic offset output guard differs",
    )
    outputs = event["outputValues"]
    require(
        outputs == [expected_offset(indices, strides, axes)],
        f"Dynamic offset of {entry} differs from its operands",
    )
    require(
        hashlib.sha256(bytes((ctypes.c_int64 * 1)(*outputs))).hexdigest()
        == event.get("outputHash"),
        "Dynamic offset host output hash changed",
    )
    if cache is not None:
        descriptor, directory = cache.get(entry)
        require_unchanged(descriptor, directory)
        require(
            descriptor["artifact"] == event["artifact"]
            and str(directory) == event["packageRoot"],
            "Dynamic offset dispatch does not name its cached package",
        )
    if native:
        request = event["details"]["request"]
        require(
            request["target"] == target
            and request["entryPoint"]
            == {"opengl": "main", "directx": "CSMain"}.get(target, entry)
            and request["dispatch"]["workgroupCount"]
            == event["workgroupCount"]
            == LAUNCH["workgroupCount"]
            and request["dispatch"]["workgroupSize"]
            == event["workgroupSize"]
            == LAUNCH["workgroupSize"]
            and event["dispatchVersion"] == 3,
            "Dynamic offset native request changed",
        )
        audit_native_execution(event)
    return 1
