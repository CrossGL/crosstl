"""SPIR-V 1.4 entry-point interfaces and ray tracing storage classes.

From SPIR-V 1.4 the interface of ``OpEntryPoint`` lists every global ``OpVariable`` the entry point statically uses,
whatever its storage class (SPIR-V 1.6 section 3.32.6, ``OpEntryPoint``; Vulkan's ``VUID-StandaloneSpirv-None-04633``
family). The emitter tracked uses of ``Input``, ``Output`` and ``Workgroup`` variables only, so a 1.4 module (any
ray tracing or mesh shader) that touched a ``Private``, ray payload or descriptor variable failed ``spirv-val``
("Interface variable id is used by entry point but is not listed as an interface").
:func:`complete_entry_point_interfaces` derives each entry point's interface from the finished module instead: the
global variables its function, and the functions it calls, reference.

:data:`RAY_STORAGE_CLASSES` maps the GL_EXT_ray_tracing declaration qualifiers the hub carries as attributes
(``@rayPayloadEXT``, ``@rayPayloadInEXT``, ``@callableDataEXT``, ``@callableDataInEXT``, ``@hitAttributeEXT``) to
the SPV_KHR_ray_tracing storage classes. The emitter had given such variables ``Private`` storage, so a miss
shader's payload write never reached the caller.
"""

from __future__ import annotations

import re
from typing import Iterable

__all__ = [
    "RAY_STORAGE_CLASSES",
    "complete_entry_point_interfaces",
    "ray_storage_class",
]

RAY_STORAGE_CLASSES = {
    "raypayloadext": "RayPayloadKHR",
    "raypayloadnv": "RayPayloadKHR",
    "raypayloadinext": "IncomingRayPayloadKHR",
    "raypayloadinnv": "IncomingRayPayloadKHR",
    "callabledataext": "CallableDataKHR",
    "callabledatanv": "CallableDataKHR",
    "callabledatainext": "IncomingCallableDataKHR",
    "callabledatainnv": "IncomingCallableDataKHR",
    "hitattributeext": "HitAttributeKHR",
    "hitattributenv": "HitAttributeKHR",
}

_GLOBAL = re.compile(r"%(\w+) = OpVariable %\w+ (\w+)")
_FUNCTION = re.compile(r"%(\w+) = OpFunction\b")
_CALL = re.compile(r"OpFunctionCall %\w+ %(\w+)")
_ID = re.compile(r"%(\w+)")
_ENTRY = re.compile(r"(OpEntryPoint\s+\w+\s+%(\w+)\s+\"[^\"]*\")(.*)")


def ray_storage_class(names: Iterable[str]):
    """The ray tracing storage class a declaration's lower-case qualifier or attribute names select, or ``None``."""
    classes = {
        RAY_STORAGE_CLASSES[name] for name in names if name in RAY_STORAGE_CLASSES
    }
    if len(classes) > 1:
        raise ValueError(
            "SPIR-V variable has conflicting ray tracing storage qualifiers: "
            + ", ".join(sorted(classes))
        )
    return next(iter(classes), None)


def _function_uses(body: list[str]):
    """``{function id: (referenced ids, called function ids)}`` of the function definitions in ``body``."""
    uses: dict[str, set[str]] = {}
    calls: dict[str, set[str]] = {}
    current = None
    for line in body:
        match = _FUNCTION.search(line)
        if match:
            current = match.group(1)
            uses.setdefault(current, set())
            calls.setdefault(current, set())
            continue
        if current is None:
            continue
        if line.strip() == "OpFunctionEnd":
            current = None
            continue
        uses[current].update(_ID.findall(line))
        call = _CALL.search(line)
        if call:
            calls[current].add(call.group(1))
    return uses, calls


def complete_entry_point_interfaces(
    entry_points: list[str], global_variables: list[str], body: list[str]
) -> list[str]:
    """Append to each ``OpEntryPoint`` line every global variable its call tree references and it does not list."""
    globals_in_order = [
        match.group(1)
        for line in global_variables
        if (match := _GLOBAL.search(line)) and match.group(2) != "Function"
    ]
    if not globals_in_order:
        return entry_points
    known = set(globals_in_order)
    uses, calls = _function_uses(body)
    completed = []
    for line in entry_points:
        match = _ENTRY.match(line.strip())
        if match is None:
            completed.append(line)
            continue
        head, function_id, tail = match.groups()
        listed = _ID.findall(tail)
        referenced, pending, seen = set(), [function_id], set()
        while pending:
            function = pending.pop()
            if function in seen:
                continue
            seen.add(function)
            referenced |= uses.get(function, set()) & known
            pending.extend(sorted(calls.get(function, set())))
        missing = [
            name
            for name in globals_in_order
            if name in referenced and name not in listed
        ]
        completed.append(
            line
            if not missing
            else f"{line} {' '.join(f'%{name}' for name in missing)}"
        )
    return completed
