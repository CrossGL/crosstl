"""Drop struct types that hold a runtime array but are neither a storage block nor used.

Vulkan requires every ``OpTypeStruct`` that contains an ``OpTypeRuntimeArray`` to be decorated ``Block`` or
``BufferBlock`` (VUID-StandaloneSpirv-OpTypeRuntimeArray-04680), whether or not anything uses it. The SPIR-V
emitter declares each CrossGL struct once, laid out in std430, and gives a buffer block that uses another layout
(std140 or scalar) a re-laid-out copy. A struct ending in a runtime array that is only used through such a copy
was therefore left in the module undecorated, and spirv-val rejected the module.

A struct type is dropped together with its debug names and member decorations only when nothing else in the
module refers to it, so this cannot change what a module computes.
"""

from __future__ import annotations

import re

__all__ = ["drop_unused_runtime_array_structs"]

_DEFINITION = re.compile(r"(%\d+) = (OpType\w+)(.*)")
_ID = re.compile(r"%\d+\b")
_BLOCK = re.compile(r"OpDecorate (%\d+) (?:Block|BufferBlock)\b")


def _subject(line):
    """The id a debug-name or decoration line describes."""
    parts = line.split()
    return parts[1] if len(parts) > 1 else None


def drop_unused_runtime_array_structs(debug_names, annotations, declarations, others):
    """Return ``(debug_names, annotations, declarations)`` without unused, undecorated runtime-array structs.

    ``others`` are the remaining module lines (entry points, global variables and function bodies), which may
    refer to a type.
    """
    runtime_arrays = set()
    structs = {}
    for line in declarations:
        match = _DEFINITION.match(line)
        if match is None:
            continue
        result, opcode, operands = match.groups()
        if opcode == "OpTypeRuntimeArray":
            runtime_arrays.add(result)
        elif opcode == "OpTypeStruct" and runtime_arrays.intersection(
            _ID.findall(operands)
        ):
            structs[result] = line
    blocks = {match.group(1) for line in annotations if (match := _BLOCK.match(line))}
    candidates = set(structs) - blocks
    if not candidates:
        return debug_names, annotations, declarations
    referenced = set()
    for line in [*declarations, *others]:
        match = _DEFINITION.match(line)
        ids = _ID.findall(match.group(3) if match else line)
        referenced.update(ids)
    for line in annotations:
        # A decoration that names another id, such as OpDecorateId, refers to it.
        referenced.update(_ID.findall(line)[1:])
    unused = candidates - referenced
    if not unused:
        return debug_names, annotations, declarations
    return (
        [line for line in debug_names if _subject(line) not in unused],
        [line for line in annotations if _subject(line) not in unused],
        [line for line in declarations if line.split(" ", 1)[0] not in unused],
    )
