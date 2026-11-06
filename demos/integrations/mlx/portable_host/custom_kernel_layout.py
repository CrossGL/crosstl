"""The buffers of the kernels ``mx.fast.metal_kernel`` builds.

metal_kernel.cpp's ``write_signature`` declares a kernel's parameters in
buffer order: each input, then the shape, strides and dimension count of it
that the source names, then every output, and last the thread attributes the
source names. The host binds every slot's buffer under the slot's name
(custom_kernel.inc); ``parse`` reads the parameters back from the source, and
``validate`` checks each supplied buffer and the launch against them.
"""

import ctypes
import math
import re
from dataclasses import dataclass

from demos.integrations.mlx.portable_host.compiled_layout import TYPES

# metal::utils(), which MLX prepends to every custom kernel's source.
PREAMBLE = '#include "mlx/backend/metal/kernels/utils.h"\n'
ENTRY_PREFIX = "custom_kernel_"
SOURCE_BUFFER = "kernel_source"
OPTIONS_BUFFER = "compile_options"
# CompileOptions::serialize: the MathMode enumerators.
MATH_MODES = {0: "safe", 1: "relaxed", 2: "fast"}
MAX_ELEMENTS = 2**31 - 1
MAX_GROUPS = 65535
MAX_THREADGROUP = 1024
# metal_kernel's metal_attributes: each one's type.
ATTRIBUTES = {
    "dispatch_quadgroups_per_threadgroup": "uint",
    "dispatch_simdgroups_per_threadgroup": "uint",
    "dispatch_threads_per_threadgroup": "uint3",
    "grid_origin": "uint3",
    "grid_size": "uint3",
    "quadgroup_index_in_threadgroup": "uint",
    "quadgroups_per_threadgroup": "uint",
    "simdgroup_index_in_threadgroup": "uint",
    "simdgroups_per_threadgroup": "uint",
    "thread_execution_width": "uint",
    "thread_index_in_quadgroup": "uint",
    "thread_index_in_simdgroup": "uint",
    "thread_index_in_threadgroup": "uint",
    "thread_position_in_grid": "uint3",
    "thread_position_in_threadgroup": "uint3",
    "threadgroup_position_in_grid": "uint3",
    "threadgroups_per_grid": "uint3",
    "threads_per_grid": "uint3",
    "threads_per_simdgroup": "uint",
    "threads_per_threadgroup": "uint3",
}

_TYPE = "|".join(re.escape(name) for name in sorted(TYPES, key=len, reverse=True))
_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_SLOT = r" \[\[buffer\((?P<slot>0|[1-9][0-9]*)\)\]\]"
_INPUT = re.compile(
    rf"  const (?P<space>constant|device) (?P<type>{_TYPE})(?P<ref>[*&]) "
    rf"(?P<name>{_NAME}){_SLOT},"
)
# The shape, strides and dimension count of the input before them.
_METADATA = (
    ("shape", re.compile(rf"  const constant int\* (?P<name>{_NAME})_shape{_SLOT},")),
    (
        "strides",
        re.compile(rf"  const constant int64_t\* (?P<name>{_NAME})_strides{_SLOT},"),
    ),
    ("ndim", re.compile(rf"  const constant int& (?P<name>{_NAME})_ndim{_SLOT},")),
)
METADATA_TYPES = {"shape": "int32", "strides": "int64", "ndim": "int32"}
_OUTPUT = re.compile(
    rf"  device (?:(?P<atomic>atomic<)(?P<atomic_type>{_TYPE})>|(?P<type>{_TYPE}))\* "
    rf"(?P<name>{_NAME}){_SLOT}(?P<end>,|\) \{{)"
)
_ATTRIBUTE = re.compile(
    rf"  (?P<type>uint3|uint) (?P<name>{_NAME}) \[\[(?P<attribute>{_NAME})\]\]"
    r"(?P<end>,|\) \{)"
)


@dataclass(frozen=True)
class Parameter:
    """One buffer slot: an input, an input's metadata, or an output."""

    name: str
    dtype: str
    kind: str
    slot: int
    owner: str = ""
    reference: bool = False
    atomic: bool = False

    @property
    def buffer(self):
        return f"buffer{self.slot}"

    def to_json(self):
        return [self.name, self.dtype, self.kind, self.slot]


@dataclass(frozen=True)
class Signature:
    parameters: tuple
    attributes: tuple
    template: bool

    @property
    def outputs(self):
        return tuple(item for item in self.parameters if item.kind == "output")


def is_entry(entry):
    return isinstance(entry, str) and entry.startswith(ENTRY_PREFIX)


def _instantiation(entry):
    name = re.escape(entry)
    return re.compile(
        rf"\ntemplate \[\[host_name\(\"{name}\"\)\]\] \[\[kernel\]\] "
        rf"decltype\({name}(?P<arguments><[^;\n]*>)\) {name}(?P=arguments);\n\Z"
    )


def _signature(entry, source):
    """The parameter lines after ``entry``'s declaration, and whether it is
    a template.

    write_signature appends the declaration to the kernel's header as is, so
    the declaration's line starts with the header's last line.
    """
    marker = f"[[kernel]] void {entry}(\n"
    if source.count(marker) != 1:
        raise ValueError("Custom kernel source does not declare exactly its entry")
    prefix, rest = source.split(marker)
    last = prefix.rsplit("\n", 2)[-2] if prefix.endswith("\n") else ""
    template = prefix.endswith(">\n") and "template <" in last
    if template != bool(_instantiation(entry).search(source)):
        raise ValueError("Custom kernel template is not instantiated as MLX does")
    return rest.split("\n"), template


def _inputs(lines, index, parameters):
    """Read the inputs and their metadata from ``lines[index]``."""
    while index < len(lines) and (match := _INPUT.fullmatch(lines[index])):
        owner = match["name"]
        reference = match["ref"] == "&"
        parameters.append(
            Parameter(
                owner,
                TYPES[match["type"]],
                "input",
                int(match["slot"]),
                reference=reference,
            )
        )
        index += 1
        for kind, pattern in _METADATA:
            metadata = pattern.fullmatch(lines[index]) if index < len(lines) else None
            if metadata is None or metadata["name"] != owner:
                continue
            if reference:
                raise ValueError("A scalar custom kernel input has no shape")
            parameters.append(
                Parameter(
                    f"{owner}_{kind}",
                    METADATA_TYPES[kind],
                    kind,
                    int(metadata["slot"]),
                    owner=owner,
                )
            )
            index += 1
    return index


def parse(entry, source):
    """The parameters of ``entry``'s kernel in ``source``, in buffer order."""
    if not is_entry(entry) or not isinstance(source, str):
        raise ValueError("Custom kernel entry or source is invalid")
    lines, template = _signature(entry, source)
    parameters = []
    index = _inputs(lines, 0, parameters)
    closed = False
    while not closed and index < len(lines):
        match = _OUTPUT.fullmatch(lines[index])
        if match is None:
            break
        atomic = match["atomic"] is not None
        dtype = TYPES[match["atomic_type"] if atomic else match["type"]]
        parameters.append(
            Parameter(match["name"], dtype, "output", int(match["slot"]), atomic=atomic)
        )
        closed = match["end"] != ","
        index += 1
    if not any(item.kind == "output" for item in parameters):
        raise ValueError("Custom kernel source writes no output")
    attributes = []
    while not closed and index < len(lines):
        match = _ATTRIBUTE.fullmatch(lines[index])
        if (
            match is None
            or match["name"] != match["attribute"]
            or ATTRIBUTES.get(match["name"]) != match["type"]
        ):
            break
        attributes.append(match["name"])
        closed = match["end"] != ","
        index += 1
    if not closed:
        raise ValueError("Custom kernel parameters differ from write_signature's")
    names = [item.name for item in parameters] + attributes
    if len(set(names)) != len(names):
        raise ValueError("Custom kernel parameter names must be unique")
    if [item.slot for item in parameters] != list(range(len(parameters))):
        raise ValueError("Custom kernel buffer slots are not consecutive")
    return Signature(tuple(parameters), tuple(attributes), template)


def math_mode(supplied):
    """Remove and decode the ``compile_options`` buffer the host appends."""
    buffer = supplied.pop(OPTIONS_BUFFER, None)
    if (
        buffer is None
        or buffer.dtype != b"int32"
        or buffer.count != 1
        or not buffer.data
        or buffer.output
    ):
        raise ValueError("Custom kernel dispatch requires its compile options")
    value = _words(buffer, ctypes.c_int32, 1)[0]
    if value not in MATH_MODES:
        raise ValueError("Custom kernel math mode is unknown")
    return value


def _words(buffer, ctype, count):
    return list(ctypes.cast(buffer.data, ctypes.POINTER(ctype * count)).contents)


def _require_metadata(signature, supplied):
    """Each input's bound shape, strides and dimension count agree with it."""
    bound = {}
    for item in signature.parameters:
        if item.kind in METADATA_TYPES:
            ctype = ctypes.c_int64 if item.kind == "strides" else ctypes.c_int32
            buffer = supplied[item.buffer]
            bound.setdefault(item.owner, {})[item.kind] = _words(
                buffer, ctype, buffer.count
            )
    for item in signature.parameters:
        values = bound.get(item.name)
        if item.kind != "input" or values is None:
            continue
        ranks = {len(values[kind]) for kind in ("shape", "strides") if kind in values}
        ranks |= set(values.get("ndim", ()))
        if len(ranks) != 1 or min(ranks) < 1:
            raise ValueError("Custom kernel input metadata disagree on its rank")
        shape, strides = values.get("shape"), values.get("strides")
        if shape is not None and min(shape) < 1:
            raise ValueError("Custom kernel input shapes must be positive")
        if strides is not None and min(strides) < 0:
            raise ValueError("Custom kernel input strides must be non-negative")
        if shape is not None and strides is not None:
            span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
            if span != supplied[item.buffer].count:
                raise ValueError("Custom kernel input extent differs from its view")


def validate(entry, signature, supplied, threads, execution, mode):
    """Check the slot buffers and the exact grid; return the launch metadata."""
    if set(supplied) != {item.buffer for item in signature.parameters}:
        raise ValueError("Custom kernel buffers differ from its source's slots")
    for item in signature.parameters:
        buffer = supplied[item.buffer]
        direction = buffer.output
        if (
            not buffer.data
            or buffer.dtype.decode("ascii") != item.dtype
            or not 1 <= buffer.count <= MAX_ELEMENTS
            or (item.kind == "output") != (direction in (1, 2))
            or direction not in (0, 1, 2)
            or (item.reference and buffer.count != 1)
            or (item.kind == "ndim" and buffer.count != 1)
        ):
            raise ValueError(
                f"Custom kernel buffer {item.name} differs from its source"
            )
    _require_metadata(signature, supplied)
    size = list(execution.get("workgroupSize", ()))
    grid = list(execution.get("threadGridSize", ()))
    if (
        len(size) != 3
        or len(grid) != 3
        or math.prod(size) > MAX_THREADGROUP
        or threads != math.prod(grid)
        or any(width > extent for width, extent in zip(size, grid))
        or any(count > MAX_GROUPS for count in execution["workgroupCount"])
    ):
        raise ValueError("Custom kernel launch differs from its exact grid")
    return {
        "mathMode": MATH_MODES[mode],
        "threadGridSize": grid,
        "workgroupSize": size,
        "template": signature.template,
        "attributes": list(signature.attributes),
        "initialized": sorted(
            item.name for item in signature.outputs if supplied[item.buffer].output == 2
        ),
    }
