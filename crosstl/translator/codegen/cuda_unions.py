"""C++ unions for the CrossGL structs that stand for a union in CUDA and HIP output.

CrossGL has no union type. A struct marked ``@union_layout(size, alignment, byte_order, abi)``, whose members are each
marked ``@union_member_layout(offset, size, alignment)``, stands for one: its members share storage, each starting at
byte ``offset``. The CUDA importer writes this contract for a CUDA union and the Metal importer for an MSL union, and
:mod:`crosstl.translator.codegen.union_layout` stores it in 32-bit words for the targets that have no union. CUDA and
HIP compile C++, whose union places every member at its first byte (C++ [class.union]/2). The CUDA and HIP generators
wrote the struct as a ``struct`` whose members had separate storage, so a program that wrote one member and read
another read a value that was never stored.

:class:`NativeUnions` writes the struct as a C++ ``union`` when that union has the recorded layout:

* every member is at offset 0;
* the size and alignment of each member, by C++ rules and the sizes and alignments of CUDA's built-in vector types
  (CUDA C++ Programming Guide, "Built-in Vector Types", which HIP's vector types keep), are the recorded ones;
* the union's own size and alignment are the recorded ones, an alignment above the members' being written
  ``alignas``; and
* the byte order is little-endian, the byte order of every CUDA and HIP device.

Any other union is refused. The source ABI is not compared by name: a Metal union whose members have the same layout
in CUDA is the same union. A member whose type has a constructor, such as the generators' ``cgl_half3``,
``cgl_half4`` and matrix helper structs, is refused as well, because a C++ union with such a member has no default
constructor, and so is a four-component 64-bit vector, which is 16-byte aligned as CUDA's ``double4`` and 32-byte
aligned as CUDA 13's ``double4_32a``. After the union the generator writes a ``static_assert`` of its size and
alignment, so nvcc and hipcc confirm the layout this module computed.

A CrossGL union constructor with no argument zero-initializes the storage, one argument of the union's own type copies
it, and one other argument initializes the first member, which is what the word-based targets do with it. These are
C++ value-initialization ([dcl.init]/8), copy and aggregate initialization ([dcl.init.aggr]/16) of the union. Any
other constructor is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..ast import StructNode
from .array_utils import evaluate_literal_int_expression, split_array_type_suffix

__all__ = ["NativeUnion", "NativeUnions"]

# (size, alignment) in bytes of each scalar spelling the CUDA and HIP generators write.
_SCALARS = {
    "bool": (1, 1),
    "char": (1, 1),
    "signed char": (1, 1),
    "unsigned char": (1, 1),
    "int8_t": (1, 1),
    "uint8_t": (1, 1),
    "short": (2, 2),
    "unsigned short": (2, 2),
    "int16_t": (2, 2),
    "uint16_t": (2, 2),
    "half": (2, 2),
    "__half": (2, 2),
    "int": (4, 4),
    "uint": (4, 4),
    "unsigned int": (4, 4),
    "int32_t": (4, 4),
    "uint32_t": (4, 4),
    "float": (4, 4),
    "long long": (8, 8),
    "unsigned long long": (8, 8),
    "int64_t": (8, 8),
    "uint64_t": (8, 8),
    "double": (8, 8),
}
# Bytes per lane of each built-in vector type family.
_VECTOR_LANES = {
    "char": 1,
    "uchar": 1,
    "short": 2,
    "ushort": 2,
    "int": 4,
    "uint": 4,
    "float": 4,
    "longlong": 8,
    "ulonglong": 8,
    "double": 8,
}
_VECTOR = re.compile(r"^(" + "|".join(_VECTOR_LANES) + r")([1-4])$")
_ARRAY_DIMENSION = re.compile(r"\[\s*(\d+)\s*\]")


class _Unmodelled(Exception):
    """A member type whose layout as a C++ union member this module does not describe."""


@dataclass(frozen=True)
class NativeUnion:
    """A CrossGL union-layout struct that the generator writes as a C++ union."""

    name: str
    size: int
    alignment: int
    natural_alignment: int
    first_member: str
    first_member_type: str
    first_member_is_array: bool


def _round(value, alignment):
    return -(-value // alignment) * alignment


class NativeUnions:
    """The C++ union of each CrossGL union-layout struct that one CUDA or HIP program declares."""

    def __init__(self, generator, refusal, label):
        self.generator = generator
        self.refusal = refusal
        self.label = label
        self.unions = {}

    @staticmethod
    def attribute(node, name):
        return next(
            (
                attribute
                for attribute in getattr(node, "attributes", []) or []
                if str(getattr(attribute, "name", "")).lower() == name
            ),
            None,
        )

    def is_union(self, struct_name):
        node = (getattr(self.generator, "structs_by_name", {}) or {}).get(struct_name)
        return node is not None and self.attribute(node, "union_layout") is not None

    def union(self, node):
        """The :class:`NativeUnion` for struct ``node``, ``None`` for a plain struct, or a refusal."""
        if not isinstance(node, StructNode):
            return None
        attribute = self.attribute(node, "union_layout")
        if attribute is None:
            return None
        cached = self.unions.get(node.name)
        if cached is not None:
            return cached
        union = self._plan(node, attribute)
        self.unions[node.name] = union
        return union

    def struct_keyword(self, node):
        """``union`` with any ``alignas`` it needs for a union-layout struct, otherwise ``struct``."""
        union = self.union(node)
        if union is None:
            return "struct"
        if union.alignment > union.natural_alignment:
            return f"union alignas({union.alignment})"
        return "union"

    def layout_assertion(self, node):
        """The ``static_assert`` of a union-layout struct's size and alignment, or ``None`` for a plain struct."""
        union = self.union(node)
        if union is None:
            return None
        return (
            f"static_assert(sizeof({union.name}) == {union.size} && "
            f"alignof({union.name}) == {union.alignment}, "
            f'"{union.name} keeps its {union.size}-byte union layout");'
        )

    def constructor(self, struct_name, positional, named):
        """The C++ expression of a constructor of union ``struct_name``, or ``None`` for a plain struct.

        ``positional`` and ``named`` are the constructor's argument nodes.
        """
        if not self.is_union(struct_name):
            return None
        union = self.union(self.generator.structs_by_name[struct_name])
        generator = self.generator
        type_name = generator.map_type(struct_name)
        arguments = list(positional or [])
        named = dict(named or {})
        if not arguments and not named:
            return f"{type_name}{{}}"
        if len(arguments) + len(named) != 1 or (
            named and next(iter(named)) != union.first_member
        ):
            raise self._refuse(
                union.name,
                "a union constructor initializes its first member "
                f"'{union.first_member}' from one argument",
            )
        value = arguments[0] if arguments else named[union.first_member]
        value_type = generator.type_name_string(generator.get_expression_type(value))
        if value_type == struct_name:
            return generator.visit(value)
        if union.first_member_is_array:
            raise self._refuse(
                union.name,
                f"its first member '{union.first_member}' is an array, which one "
                "constructor argument cannot initialize",
            )
        rendered = generator.generate_expression_with_expected(
            value, union.first_member_type
        )
        member_type = generator.map_type(union.first_member_type)
        if member_type in _SCALARS:
            rendered = f"static_cast<{member_type}>({rendered})"
        return f"{type_name}{{{rendered}}}"

    def _refuse(self, name, reason):
        return self.refusal(
            f"{self.label} output cannot write CrossGL union '{name}' as a C++ union: {reason}"
        )

    def _layout_int(self, attribute, index):
        arguments = list(getattr(attribute, "arguments", []) or [])
        if index >= len(arguments):
            return None
        return evaluate_literal_int_expression(arguments[index], {})

    @staticmethod
    def _layout_name(attribute, index):
        arguments = list(getattr(attribute, "arguments", []) or [])
        if index >= len(arguments):
            return ""
        argument = arguments[index]
        value = getattr(argument, "name", None) or getattr(argument, "value", None)
        return str(value or "").strip().lower()

    def _member_type(self, member):
        for field_name in ("member_type", "vtype", "var_type"):
            member_type = getattr(member, field_name, None)
            if member_type is not None:
                return str(self.generator.type_name_string(member_type))
        return ""

    def _plan(self, node, attribute):
        name = node.name
        if len(list(getattr(attribute, "arguments", []) or [])) != 4:
            raise self._refuse(
                name,
                "@union_layout needs a size, an alignment, a byte order and an ABI",
            )
        size = self._layout_int(attribute, 0)
        alignment = self._layout_int(attribute, 1)
        if not size or not alignment or alignment & (alignment - 1):
            raise self._refuse(
                name,
                "its @union_layout size and alignment are not literal powers of two",
            )
        byte_order = self._layout_name(attribute, 2)
        if byte_order != "little_endian":
            raise self._refuse(
                name,
                f"its byte order '{byte_order or '<missing>'}' is not the little-endian "
                "order of CUDA and HIP devices",
            )
        members = [
            member
            for member in getattr(node, "members", []) or []
            if getattr(member, "name", None)
        ]
        if not members:
            raise self._refuse(name, "it has no named member")
        natural_size, natural_alignment = 0, 1
        for member in members:
            member_size, member_alignment = self._member_layout(name, member)
            natural_size = max(natural_size, member_size)
            natural_alignment = max(natural_alignment, member_alignment)
        if alignment < natural_alignment:
            raise self._refuse(
                name,
                f"its recorded alignment {alignment} is below the {natural_alignment}-byte "
                "alignment of its members",
            )
        if size != _round(natural_size, alignment):
            raise self._refuse(
                name,
                f"its recorded size {size} is not the {_round(natural_size, alignment)} bytes "
                "its members take",
            )
        first_type = self._member_type(members[0])
        return NativeUnion(
            name=name,
            size=size,
            alignment=alignment,
            natural_alignment=natural_alignment,
            first_member=members[0].name,
            first_member_type=first_type,
            first_member_is_array=bool(split_array_type_suffix(first_type)[1]),
        )

    def _member_layout(self, name, member):
        member_attribute = self.attribute(member, "union_member_layout")
        if (
            member_attribute is None
            or len(list(getattr(member_attribute, "arguments", []) or [])) != 3
        ):
            raise self._refuse(
                name,
                f"member '{member.name}' has no @union_member_layout(offset, size, alignment)",
            )
        offset = self._layout_int(member_attribute, 0)
        if offset != 0:
            raise self._refuse(
                name,
                f"member '{member.name}' starts at byte {offset!r}, and a C++ union places "
                "every member at byte 0",
            )
        member_type = self._member_type(member)
        try:
            layout = self.type_layout(member_type, ())
        except _Unmodelled as error:
            raise self._refuse(name, f"member '{member.name}' {error}") from None
        recorded = (
            self._layout_int(member_attribute, 1),
            self._layout_int(member_attribute, 2),
        )
        if recorded != layout:
            raise self._refuse(
                name,
                f"member '{member.name}' of type '{member_type}' takes {layout[0]} bytes at "
                f"{layout[1]}-byte alignment in {self.label}, not the recorded {recorded[0]!r} "
                f"bytes at {recorded[1]!r}-byte alignment",
            )
        return layout

    def type_layout(self, type_name, resolving):
        """``(size, alignment)`` of CrossGL type ``type_name`` as a C++ union member, or :class:`_Unmodelled`."""
        base, suffix = split_array_type_suffix(str(type_name or "").strip())
        base = base.strip()
        count = 1
        if suffix:
            dimensions = _ARRAY_DIMENSION.findall(suffix)
            if not dimensions or _ARRAY_DIMENSION.sub("", suffix).strip():
                raise _Unmodelled(f"is an array without a literal size ('{type_name}')")
            for dimension in dimensions:
                count *= int(dimension)
            if not count:
                raise _Unmodelled(f"is an empty array ('{type_name}')")
        size, alignment = self._element_layout(base, resolving)
        return size * count, alignment

    def _element_layout(self, base, resolving):
        structs = getattr(self.generator, "structs_by_name", {}) or {}
        if base in structs:
            return self._struct_layout(structs[base], resolving)
        spelling = str(self.generator.map_type(base) or "").strip()
        if spelling in _SCALARS:
            return _SCALARS[spelling]
        if spelling in ("half2", "__half2"):
            return 4, 4
        match = _VECTOR.match(spelling)
        if match is None:
            raise _Unmodelled(
                f"has type '{base}', whose layout as a C++ union member is not modelled"
            )
        lane, lanes = _VECTOR_LANES[match.group(1)], int(match.group(2))
        if lane == 8 and lanes == 4:
            raise _Unmodelled(
                f"has type '{base}', whose alignment is 16 bytes for CUDA's {spelling} and 32 "
                f"for CUDA 13's {spelling}_32a"
            )
        if lanes == 3:
            return lane * 3, lane
        return lane * lanes, min(lane * lanes, 16)

    def _struct_layout(self, node, resolving):
        if node.name in resolving:
            raise _Unmodelled(f"contains struct '{node.name}' within itself")
        if self.attribute(node, "union_layout") is not None:
            union = self.union(node)
            return union.size, union.alignment
        if getattr(node, "is_cbuffer", False) or getattr(node, "attributes", None):
            raise _Unmodelled(
                f"has struct type '{node.name}', whose attributes may change its layout"
            )
        offset, alignment = 0, 1
        for member in getattr(node, "members", []) or []:
            member_size, member_alignment = self.type_layout(
                self._member_type(member), (*resolving, node.name)
            )
            offset = _round(offset, member_alignment) + member_size
            alignment = max(alignment, member_alignment)
        if not offset:
            raise _Unmodelled(f"has the empty struct type '{node.name}'")
        return _round(offset, alignment), alignment
