"""The host interface of a Metal library's entry points, read from its AIR, and how two libraries' interfaces differ.

A graphics probe is compiled but never run, so a translation that moves a buffer to another argument-table index,
drops a vertex attribute or writes another render target still compiles. The Metal compiler records the interface of
every entry point in the metadata of the AIR it writes (``metal -S -emit-llvm``), the data a pipeline's reflection
(``MTLRenderPipelineReflection``, ``MTLComputePipelineReflection``) is built from: ``!air.vertex``, ``!air.fragment``,
``!air.kernel``, ``!air.mesh``, ``!air.object`` and ``!air.intersection`` list the entry points, and each lists the
metadata of its results and of its arguments. An argument's metadata names its kind (``air.buffer``,
``air.indirect_buffer`` for an argument buffer, ``air.texture``, ``air.sampler``, ``air.vertex_input``,
``air.fragment_input``, a built-in such as ``air.vertex_id``), its ``air.location_index`` (the argument-table index
and the array length), its access and address space, the size of its type and, for a structure, the offset, size and
type of each member (``air.struct_type_info``), with ``air.indirect_argument`` for each member of an argument buffer.

:func:`interfaces` reads that metadata into one :class:`Interface` per entry point, without the names a translation
may change (of arguments, members, structures and functions), and :func:`differences` says how a translation's
interfaces differ from those of the library it should match. The built-in values an entry point reads are not part of
its host interface: a translation may read another built-in, or one more, and compute the same results, which
running the probe or its decoration anchors check (``tools/coverage/decorations.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Interface", "differences", "interfaces", "round_trip_difference"]

# The named metadata that lists a library's entry points of each kind; ``!air.vertex`` also lists post-tessellation
# vertex functions, and ``!air.object`` the object (task) functions of a mesh pipeline.
ENTRY_LISTS = (
    "vertex",
    "fragment",
    "kernel",
    "mesh",
    "object",
    "intersection",
    "visible",
)

_NODE = re.compile(r"^!(\d+) = (?:distinct )?!\{(.*)\}\s*$", re.M)
_ENTRIES = re.compile(r"^!air\.(\w+) = !\{(.*)\}\s*$", re.M)
_TOKEN = re.compile(r'!"((?:[^"\\]|\\.)*)"|!(\d+)|\bi\d+ (-?\d+)|@("[^"]*"|[\w.$-]+)')

# Arguments a host binds or a pipeline links: resources by argument-table index, and the per-vertex, per-fragment,
# per-patch and per-mesh data of a stage.
_HOST_KINDS = frozenset(
    {
        "air.buffer",
        "air.indirect_buffer",
        "air.texture",
        "air.sampler",
        "air.vertex_input",
        "air.fragment_input",
        "air.patch_control_point_input",
        "air.patch_input",
        "air.stage_in",
        "air.mesh",
        "air.payload",
        "air.object_data",
        "air.primitive_data",
        "air.indirect_constant",
        "air.visible_function_table",
        "air.intersection_function_table",
        "air.instance_acceleration_structure",
        "air.primitive_acceleration_structure",
        "air.imageblock_data",
    }
)
# Properties that name rather than describe an argument.
_NAMES = frozenset({"air.arg_name", "air.arg_type_name"})
_MARKERS = frozenset({"air.arg_unused"})
# Type names a translation keeps: scalars, vectors and matrices, and the Metal types whose spelling is their shape.
_BUILTIN_TYPE = re.compile(
    r"^(?:bool|char|uchar|short|ushort|int|uint|long|ulong|half|bfloat|float|double)"
    r"(?:[2-4](?:x[2-4])?)?$"
)
_SHAPED_TYPE = re.compile(
    r"^(?:(?:metal::)?(?:texture|depth)\w*<|sampler$|array<(?:texture|depth|sampler)|(?:metal::)?_?atomic)"
)


@dataclass(frozen=True)
class Interface:
    """The host interface of one entry point: its kind and the name-free descriptions of its results and of the
    arguments a host binds or a pipeline links, each a sorted tuple."""

    kind: str
    results: tuple
    arguments: tuple

    def describe(self) -> str:
        return f"{self.kind} entry"


def _tokens(body: str) -> list[tuple[str, object]]:
    found = []
    for match in _TOKEN.finditer(body):
        text, node, number, function = match.groups()
        if text is not None:
            found.append(("s", text))
        elif node is not None:
            found.append(("n", int(node)))
        elif number is not None:
            found.append(("i", int(number)))
        else:
            found.append(("f", function))
    return found


def _type_name(name: str) -> str:
    """``name`` when it spells a shape, or ``struct`` for a name the program chose."""
    return name if _BUILTIN_TYPE.match(name) or _SHAPED_TYPE.match(name) else "struct"


def _location(name: str) -> str:
    """A ``[[user(name)]]`` stage value keeps its name; one the compiler named after the member is ``generated``."""
    return "generated" if name.startswith("generated(") else name


class _Metadata:
    def __init__(self, air: str):
        self.nodes = {int(m.group(1)): m.group(2) for m in _NODE.finditer(air)}
        self.lists = {m.group(1): m.group(2) for m in _ENTRIES.finditer(air)}

    def tokens(self, node: int) -> list[tuple[str, object]]:
        return _tokens(self.nodes.get(node, ""))

    def describe(self, node: int, seen: frozenset = frozenset()) -> tuple:
        """The name-free description of the metadata ``node``: its strings and integers, with nested nodes described
        in turn and the names of arguments, members, structures and functions left out.
        """
        if node in seen:
            return ("cycle",)
        seen = seen | {node}
        tokens = self.tokens(node)
        if tokens and tokens[0] == ("s", "air.struct_type_info"):
            tokens = tokens[1:]
        described: list = []
        index = 0
        while index < len(tokens):
            kind, value = tokens[index]
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if kind == "s" and value in _NAMES:
                if value == "air.arg_type_name" and following and following[0] == "s":
                    described.append(("type", _type_name(following[1])))
                index += 2
                continue
            if kind == "s" and value in _MARKERS or kind == "f":
                index += 1
                continue
            if kind == "s" and (
                value.startswith("user(") or value.startswith("generated(")
            ):
                described.append(("location", _location(value)))
            elif kind == "n":
                described.append(self.describe(value, seen))
            else:
                described.append(value)
            index += 1
        return _atomic_value(tuple(_struct_members(described)))

    def entries(self) -> list[Interface]:
        found = []
        for kind in ENTRY_LISTS:
            for token_kind, node in _tokens(self.lists.get(kind, "")):
                if token_kind != "n":
                    continue
                parts = [value for part, value in self.tokens(node) if part == "n"]
                results = self._items(parts[0]) if parts else ()
                arguments = self._items(parts[1]) if len(parts) > 1 else ()
                found.append(
                    Interface(
                        kind,
                        tuple(sorted(map(_clip_distances, results), key=repr)),
                        tuple(
                            sorted(
                                (
                                    argument[1:]
                                    for argument in arguments
                                    if len(argument) > 1 and argument[1] in _HOST_KINDS
                                ),
                                key=repr,
                            )
                        ),
                    )
                )
        return found

    def _items(self, node: int) -> list[tuple]:
        return [
            self.describe(value) for part, value in self.tokens(node) if part == "n"
        ]


def _atomic_value(described: tuple) -> tuple:
    """A buffer of ``atomic_int`` is a buffer of ``int`` to its host: AIR describes the atomic as a structure of one
    member of the value type, which a translation that reads the values atomically through a cast does not have.
    """
    if (
        "type",
        "metal::_atomic",
    ) not in described or "air.struct_type_info" not in described:
        return described
    position = described.index("air.struct_type_info")
    members = described[position + 1] if position + 1 < len(described) else ()
    if not (isinstance(members, tuple) and len(members) == 1 and len(members[0]) == 4):
        return described
    value_type = members[0][3]
    return tuple(
        ("type", value_type) if item == ("type", "metal::_atomic") else item
        for index, item in enumerate(described)
        if index not in (position, position + 1)
    )


def _clip_distances(result: tuple) -> tuple:
    """A ``[[clip_distance]]`` result of type ``float`` is an array of one distance, which AIR leaves unsized."""
    if (
        result[:1] == ("air.clip_distance",)
        and "air.clip_distance_array_size" not in result
    ):
        return (result[0], "air.clip_distance_array_size", 1, *result[1:])
    return result


def _struct_members(described: list) -> list:
    """``air.struct_type_info`` lists each member as its offset, size, array length, type name and member name; leave
    the member's name out and the type name only when it spells a shape."""
    if len(described) < 5 or not all(isinstance(value, int) for value in described[:3]):
        return described
    members, index = [], 0
    while index + 3 < len(described):
        offset, size, count = described[index : index + 3]
        if not all(isinstance(value, int) for value in (offset, size, count)):
            return described
        type_name = described[index + 3]
        if not isinstance(type_name, str):
            return described
        index += 4
        if index < len(described) and isinstance(described[index], str):
            index += 1  # the member's name
        member = [offset, size, count, _type_name(type_name)]
        if index < len(described) and described[index] == "air.indirect_argument":
            member.append(described[index + 1] if index + 1 < len(described) else ())
            index += 2
        members.append(tuple(member))
    return members if index == len(described) else described


def interfaces(air: str) -> list[Interface]:
    """The host interface of every entry point of the textual AIR ``air``, in the order of :data:`ENTRY_LISTS`."""
    return _Metadata(air).entries()


def _spell(item: tuple) -> str:
    """A short spelling of a described result or argument for a diagnostic."""
    kind = item[0] if item else "?"
    rest = [value for value in item[1:] if not isinstance(value, tuple)]
    location = next(
        (
            value[1]
            for value in item
            if isinstance(value, tuple) and value[:1] == ("location",)
        ),
        None,
    )
    index = None
    if "air.location_index" in rest:
        position = rest.index("air.location_index")
        index = rest[position + 1] if position + 1 < len(rest) else None
    shown = str(kind).removeprefix("air.")
    if index is not None:
        shown += f"({index})"
    elif location:
        shown += f" {location}"
    elif kind == "air.render_target" and len(rest) > 0:
        shown += f"({rest[0]})"
    type_name = next(
        (
            value[1]
            for value in item
            if isinstance(value, tuple) and value[:1] == ("type",)
        ),
        None,
    )
    return f"{shown} {type_name}" if type_name else shown


def _compare(label: str, wanted: tuple, found: tuple) -> list[str]:
    missing = [item for item in wanted if item not in found]
    extra = [item for item in found if item not in wanted]
    if not missing and not extra:
        return []
    parts = []
    if missing:
        parts.append("lacks " + ", ".join(_spell(item) for item in missing))
    if extra:
        parts.append("adds " + ", ".join(_spell(item) for item in extra))
    return [f"{label} {' and '.join(parts)}"]


def differences(expected: str, translated: str) -> list[str]:
    """How the interfaces of the textual AIR ``translated`` differ from those of ``expected``; empty when every entry
    point of each has one of the same kind and interface in the other."""
    wanted, found = interfaces(expected), interfaces(translated)
    problems = []
    for kind in ENTRY_LISTS:
        ours = [entry for entry in wanted if entry.kind == kind]
        theirs = [entry for entry in found if entry.kind == kind]
        if len(ours) != len(theirs):
            problems.append(
                f"{len(theirs)} {kind} entry points where {len(ours)} are expected"
            )
            continue
        remaining = list(theirs)
        for entry in ours:
            if entry in remaining:
                remaining.remove(entry)
                continue
            # With one entry of the kind, say what differs; with several, pair them in order.
            other = remaining.pop(0)
            problems += _compare(
                f"the {kind} entry's results", entry.results, other.results
            )
            problems += _compare(
                f"the {kind} entry's arguments", entry.arguments, other.arguments
            )
    return problems


def round_trip_difference(native: str, translated: str) -> str:
    """How the interfaces of ``translated``, the Metal that CrossTL writes back from the Metal source ``native``,
    differ from those of ``native``: a host that drives ``native`` must drive its translation in the same way.

    Empty when they agree, or when the Metal compiler is not installed here or does not compile either source.
    """
    from . import validators

    expected = validators.air(native)
    found = validators.air(translated) if expected else ""
    if not found:
        return ""
    changed = differences(expected, found)
    return f"the host interface changes: {'; '.join(changed)}" if changed else ""
