"""Textures, samplers and the buffers besides its inputs and output that the D3D12 harness binds for an HLSL kernel.

A compute probe may declare resources besides the storage buffers of the harness's contract (:mod:`.interface`): a
``Texture2D`` and the ``SamplerState`` a method samples it through, a ``Buffer<float4>``, an ``AppendStructuredBuffer``
or a ``RWByteAddressBuffer`` whose methods it calls. :func:`declarations` reads every resource a kernel declares at
an explicit register, :func:`split` separates the contract's buffers from the others, and :func:`records` describes
each of those for the job. :func:`views` turns the records into the allocations and views of a request to the D3D12
worker (``crosstl/project/directx_runtime_worker.cpp``), which creates the textures, typed views, UAV counters and
static samplers.

The k-th resource of a register class (``t``, ``u`` or ``s``), in register order and counting each element of an
array, gets the same contents whatever register it takes, as the k-th input buffer does: a probe and its
translation then read the same values when they declare the same resources in the same order, even where the
translation numbers the registers from 0 and the probe does not. Every element and texel holds a fixed function of
that ordinal and its index, written in the view's format, so two resources of one kernel hold different values.

Every sampler has the same state: point filtering, wrapping addresses and every mip level; a comparison sampler
passes when the reference is at most the texel. A texture that a comparison sampler may sample is a single-channel
32-bit float texture, the format Direct3D can compare, and every texture has one mip level. Textures and typed
buffers have 64 elements along the first axis of a one-dimensional shape and 4 along each axis of the others, with 2
array layers. A consume buffer holds one value in every element and starts with a counter of 64, so the elements it
hands out do not depend on the order the threads consume them in.

Resources the harness cannot give a value both runs agree on raise :class:`InterfaceChanged`: a counter the threads
increment or decrement, whose values they take in no fixed order; an interlocked operation on one of these resources
that returns the original value, for the same reason; multisampled, feedback and rasterizer-ordered textures,
acceleration structures, unsized arrays, sampler arrays and element types without a format here.
"""

from __future__ import annotations

import re
import struct

from .interface import InterfaceChanged

# Every resource declaration at a register: its type, template argument, name, array size and register.
_DECLARATION = re.compile(
    r"(?:^|[;}\n])\s*(?:globallycoherent\s+)?(?P<type>[A-Za-z_]\w*)\s*"
    r"(?:<\s*(?P<element>[^<>;]*?)\s*>)?\s+(?P<name>[A-Za-z_]\w*)\s*"
    r"(?:\[\s*(?P<count>\d*)\s*\])?\s*:\s*register\(\s*(?P<register>[a-z])(?P<slot>\d+)\s*"
    r"(?:,\s*space(?P<space>\d+)\s*)?\)\s*;"
)
# The types of the harness's own storage buffers, which :func:`split` assigns to the contract's slots first.
BUFFERS = frozenset(
    {
        "StructuredBuffer",
        "RWStructuredBuffer",
        "ByteAddressBuffer",
        "RWByteAddressBuffer",
    }
)
# Texture types: (dimension, width, height, layers) as the worker's request takes them. A cube's layers count its
# faces.
_WIDE, _SIDE, _LAYERS = 64, 4, 2
_TEXTURES = {
    "Texture1D": (1, _WIDE, 1, 1),
    "Texture1DArray": (2, _WIDE, 1, _LAYERS),
    "Texture2D": (3, _SIDE, _SIDE, 1),
    "Texture2DArray": (4, _SIDE, _SIDE, _LAYERS),
    "Texture3D": (5, _SIDE, _SIDE, _SIDE),
    "TextureCube": (6, _SIDE, _SIDE, 6),
    "TextureCubeArray": (7, _SIDE, _SIDE, 6 * _LAYERS),
}
_STORAGE_TEXTURES = {
    f"RW{name}": shape
    for name, shape in _TEXTURES.items()
    if not name.startswith("TextureCube")
}
_SAMPLERS = {"SamplerState": 0, "SamplerComparisonState": 1}
# Structured and raw buffers besides the contract's: (register class, counter use).
_OTHER_BUFFERS = {
    "StructuredBuffer": ("t", None),
    "RWStructuredBuffer": ("u", "counter"),
    "AppendStructuredBuffer": ("u", "append"),
    "ConsumeStructuredBuffer": ("u", "consume"),
    "ByteAddressBuffer": ("t", None),
    "RWByteAddressBuffer": ("u", None),
}
_TYPED_BUFFERS = {"Buffer": "t", "RWBuffer": "u"}
_ELEMENTS = 64
# DXGI formats by component kind and count, with the bytes of one component and its struct code.
_FORMATS = {
    "float": ({1: 41, 2: 16, 3: 6, 4: 2}, 4, "f"),
    "uint": ({1: 42, 2: 17, 3: 7, 4: 3}, 4, "I"),
    "int": ({1: 43, 2: 18, 3: 8, 4: 4}, 4, "i"),
    "unorm": ({1: 61, 2: 49, 4: 28}, 1, "B"),
    "snorm": ({1: 63, 2: 51, 4: 31}, 1, "b"),
}
# DXGI_FORMAT_R32_FLOAT, the format of a texture a comparison sampler may sample.
_COMPARABLE = 41
_ELEMENT = re.compile(r"(?:(unorm|snorm)\s+)?(float|uint|int|dword)([1-4])?")
# Interlocked operations and the argument count at which the last argument receives the original value; a
# compare-and-store returns none.
_ORIGINAL_AT = {
    **{
        f"Interlocked{name}{width}": 3
        for name in (
            "Add",
            "And",
            "Or",
            "Xor",
            "Min",
            "Max",
            "UMin",
            "UMax",
            "Exchange",
        )
        for width in ("", "64")
    },
    "InterlockedExchangeFloat": 3,
    "InterlockedCompareExchange": 4,
    "InterlockedCompareExchange64": 4,
    "InterlockedCompareExchangeFloatBitwise": 4,
}
_INTERLOCKED = re.compile(
    r"\b(?:(\w+)\s*(?:\[[^\]]*\])?\s*\.\s*)?(Interlocked\w+)\s*\("
)


def declarations(text: str) -> list[dict]:
    """Every resource ``text`` declares at a ``t``, ``u`` or ``s`` register, in declaration order."""
    found = []
    for match in _DECLARATION.finditer(text):
        if match["register"] == "b" or match["type"] in {"cbuffer", "tbuffer"}:
            continue
        found.append(
            {
                "type": match["type"],
                "element": match["element"],
                "name": match["name"],
                "count": match["count"],
                "register": match["register"],
                "slot": int(match["slot"]),
                "space": int(match["space"] or 0),
            }
        )
    return found


def split(declared: list[dict], input_count: int) -> tuple[dict, list[dict]]:
    """The contract's buffers by slot and the other declarations.

    Input ``k`` is the buffer at ``u<k>``, or at ``t<k>`` when no buffer takes ``u<k>``, and the output the buffer at
    ``u<input_count>``; a slot no buffer takes stays empty, which :mod:`.interface` reports.
    """
    buffers = {}
    for slot in range(input_count + 1):
        for register in ("u", "t"):
            match = next(
                (
                    record
                    for record in declared
                    if record["type"] in BUFFERS
                    and not record["count"]
                    and record["register"] == register
                    and record["slot"] == slot
                ),
                None,
            )
            if match is not None:
                buffers[slot] = match
                break
    taken = {id(record) for record in buffers.values()}
    return buffers, [record for record in declared if id(record) not in taken]


def _element(spelled: str | None, what: str) -> tuple[str, int]:
    """The component kind and count of a texture or typed buffer's element type ``spelled``."""
    found = _ELEMENT.fullmatch(" ".join((spelled or "float4").split()))
    if found is None:
        raise InterfaceChanged(f"The harness has no format for {what}<{spelled}>")
    norm, scalar, count = found.groups()
    kind = norm or {"dword": "uint"}.get(scalar, scalar)
    if norm and scalar != "float":
        raise InterfaceChanged(f"The harness has no format for {what}<{spelled}>")
    return kind, int(count or 1)


def _format(kind: str, count: int, what: str, writable: bool) -> tuple[int, int]:
    """The DXGI format and texel bytes of ``count`` components of ``kind``."""
    formats, size, _ = _FORMATS[kind]
    # Direct3D has no unordered-access view of a three-component format.
    if count not in formats or (writable and count == 3):
        raise InterfaceChanged(f"The harness has no format for {what}")
    return formats[count], size * count


def _stride(spelled: str | None, name: str) -> int:
    from .interface import _hlsl_stride

    stride = _hlsl_stride(spelled, "") if spelled else None
    if not stride:
        raise InterfaceChanged(f"Unknown structured-buffer stride for {name}")
    return stride


def _order_dependent(text: str, names: set[str]) -> str:
    """Why the threads of ``text`` see values of the resources ``names`` that depend on the order they run in."""
    for name in sorted(names):
        if re.search(
            rf"\b{re.escape(name)}\s*\.\s*(IncrementCounter|DecrementCounter)\s*\(",
            text,
        ):
            return f"the threads take the counter values of {name} in no fixed order"
    for found in _INTERLOCKED.finditer(text):
        owner, operation = found.groups()
        # A method's first argument is the destination's offset, as an intrinsic's is the destination, so the two
        # take the original value at the same argument count.
        arguments = _arguments(text, found.end())
        target = owner
        if owner is None and arguments:
            leading = re.match(r"\s*(\w+)", arguments[0])
            target = leading and leading.group(1)
        if target in names and len(arguments) >= _ORIGINAL_AT.get(operation, 99):
            return f"{operation} returns the original value of {target}, which depends on the thread order"
    return ""


def _arguments(text: str, start: int) -> list[str]:
    """The top-level arguments of the call whose argument list starts at ``start``."""
    depth, index, current, found = 1, start, [], []
    while index < len(text) and depth:
        char = text[index]
        depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(char, 0)
        if depth == 1 and char == ",":
            found.append("".join(current))
            current = []
        elif depth:
            current.append(char)
        index += 1
    if "".join(current).strip():
        found.append("".join(current))
    return found


def records(extras: list[dict], text: str) -> list[dict]:
    """The job records of the declarations ``extras`` of the kernel ``text``, one per register.

    Raises :class:`InterfaceChanged` for a resource the harness cannot bind or give repeatable contents.
    """
    reason = _order_dependent(text, {record["name"] for record in extras})
    if reason:
        raise InterfaceChanged(f"No second run repeats the kernel: {reason}")
    comparable = any(record["type"] == "SamplerComparisonState" for record in extras)
    found = []
    for record in sorted(extras, key=lambda item: (item["register"], item["slot"])):
        if record["count"] == "":
            raise InterfaceChanged(
                f"The harness binds no unsized array: {record['name']}"
            )
        count = int(record["count"] or 1)
        if count < 1 or count > 8:
            raise InterfaceChanged(
                f"The harness binds arrays of 1 to 8 resources: {record['name']}"
            )
        described = _describe(record, comparable)
        if described["register"] != record["register"]:
            raise InterfaceChanged(
                f"Register class {record['register']} does not match {record['type']} {record['name']}"
            )
        if described["kind"] == "sampler" and count != 1:
            raise InterfaceChanged(
                f"The harness binds no sampler array: {record['name']}"
            )
        for element in range(count):
            found.append(
                dict(
                    described,
                    name=record["name"],
                    register=f"{record['register']}{record['slot'] + element}",
                    element=element,
                )
            )
    ordinals: dict[str, int] = {}
    for record in found:
        record["ordinal"] = ordinals.get(record["register"][0], 0)
        ordinals[record["register"][0]] = record["ordinal"] + 1
    return found


def _describe(record: dict, comparable: bool) -> dict:
    """The kind, register class and layout of one declaration."""
    kind, element = record["type"], record["element"]
    if kind in _SAMPLERS:
        return {"kind": "sampler", "register": "s", "comparison": _SAMPLERS[kind]}
    if kind in _TEXTURES or kind in _STORAGE_TEXTURES:
        writable = kind in _STORAGE_TEXTURES
        dimension, width, height, layers = (
            _STORAGE_TEXTURES if writable else _TEXTURES
        )[kind]
        component, components = _element(element, kind)
        if comparable and not writable and component == "float" and components != 3:
            texture_format, texel, components = _COMPARABLE, 4, 1
        else:
            texture_format, texel = _format(
                component, components, f"{kind}<{element}>", writable
            )
        return {
            "kind": "texture",
            "register": "u" if writable else "t",
            "format": texture_format,
            "texel": texel,
            "component": component,
            "components": components,
            "dimension": dimension,
            "extent": (width, height, layers),
        }
    if kind in _TYPED_BUFFERS:
        writable = _TYPED_BUFFERS[kind] == "u"
        component, components = _element(element, kind)
        buffer_format, texel = _format(
            component, components, f"{kind}<{element}>", writable
        )
        return {
            "kind": "typed",
            "register": _TYPED_BUFFERS[kind],
            "format": buffer_format,
            "texel": texel,
            "component": component,
            "components": components,
        }
    if kind in _OTHER_BUFFERS:
        register, counter = _OTHER_BUFFERS[kind]
        raw = kind.endswith("ByteAddressBuffer")
        return {
            "kind": "raw" if raw else "structured",
            "register": register,
            "stride": 0 if raw else _stride(element, record["name"]),
            "counter": counter,
        }
    raise InterfaceChanged(f"The harness binds no {kind} ({record['name']})")


def _value(seed: int, component: str):
    """The value at ``seed`` in a resource of ``component`` kind: a small exact number, nonzero for every seed."""
    step = seed % 251
    if component == "float":
        return (step + 1) / 8
    if component == "snorm":
        return step - 125
    return step + 1


def contents(record: dict) -> bytes:
    """The bytes of the resource ``record`` describes, in its format and with no row padding."""
    base = record["ordinal"] * 1021
    if record["kind"] in {"texture", "typed"}:
        width, height, layers = record.get("extent", (_ELEMENTS, 1, 1))
        _, size, code = _FORMATS[record["component"]]
        count = width * height * layers
        values = [
            _value(base + texel * 4 + component, record["component"])
            for texel in range(count)
            for component in range(record["components"])
        ]
        return struct.pack(f"<{len(values)}{code}", *values)
    if record["kind"] == "raw":
        return struct.pack(
            f"<{_ELEMENTS}I",
            *(_value(base + word, "uint") for word in range(_ELEMENTS)),
        )
    # A structured buffer's words hold small exact floats, which read as nonzero integers too. Every element of a
    # consume buffer is the same.
    words = record["stride"] // 4
    if record["stride"] % 4:
        raise InterfaceChanged(
            f"Structured-buffer stride {record['stride']} is not a whole number of words"
        )
    consume = record["counter"] == "consume"
    return b"".join(
        struct.pack(
            f"<{words}f",
            *(
                _value(base + (0 if consume else index * words + word), "float")
                for word in range(words)
            ),
        )
        for index in range(_ELEMENTS)
    )


# The worker's view kinds and flags, and its texture dimensions (``directx_runtime_worker.cpp``).
_SRV, _UAV, _SAMPLER, _RAW, _EXTENDED, _CONTINUES = 1, 2, 3, 4, 8, 16


def views(extras: list[dict], first_allocation: int) -> tuple[list[bytes], list[bytes]]:
    """The allocations and encoded views of the request for the job records ``extras``.

    The allocations follow the ``first_allocation`` ones the request already holds: each resource's contents, and
    after a structured buffer with a counter the 4-byte counter, which starts at the buffer's element count for a
    consume buffer and at 0 otherwise.
    """
    allocations, encoded = [], []
    previous = None
    for record in extras:
        register = record["register"]
        slot = int(register[1:])
        if record["kind"] == "sampler":
            kind, fields, trailer = (
                _SAMPLER | _EXTENDED,
                (0, 0, 0, 0),
                (record["comparison"], 0, 0, 0, 0, 0, 0),
            )
        else:
            data = contents(record)
            allocation = first_allocation + len(allocations)
            allocations.append(data)
            kind = (_SRV if register[0] == "t" else _UAV) | _EXTENDED
            if record["kind"] == "raw":
                kind |= _RAW
            continued = (
                record["element"]
                and previous is not None
                and previous["name"] == record["name"]
            )
            if continued:
                kind |= _CONTINUES
            counter = 0
            if record["kind"] == "structured" and record["counter"] is not None:
                start = _ELEMENTS if record["counter"] == "consume" else 0
                allocations.append(struct.pack("<I", start))
                counter = first_allocation + len(allocations)
            stride = record.get("stride", 0)
            fields = (allocation, stride, 0, len(data))
            width, height, layers = record.get("extent", (0, 0, 0))
            trailer = (
                record.get("format", 0),
                record.get("texel", 0),
                record.get("dimension", 0),
                width,
                height,
                layers,
                counter,
            )
        allocation_index, stride, offset, length = fields
        encoded.append(
            struct.pack("<4I2Q", kind, slot, allocation_index, stride, offset, length)
            + struct.pack("<8I", *trailer, 0)
        )
        previous = record
    return allocations, encoded
