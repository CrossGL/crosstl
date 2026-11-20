"""The CrossGL union layout contract of each CUDA union the importer writes.

The members of a C++ union share storage that starts at the union's first byte (C++ [class.union]), and CUDA code
writes one member and reads another to reinterpret the stored bytes, which nvcc, GCC and Clang define. CrossGL has no
union type: a ``struct`` marked ``@union_layout(size, alignment, byte_order, abi)`` whose members are each marked
``@union_member_layout(offset, size, alignment)`` stands for one, and the word-based targets store it in 32-bit words
that every member reads and writes (:mod:`crosstl.translator.codegen.union_layout`). The Metal importer writes this
contract for MSL unions. Without it a CUDA union became a struct whose members had separate storage, so reading a
member other than the one written returned a value that was never stored.

:class:`UnionLayoutContracts` records each union the importer writes, the output line of its declaration and of
each member, and, once the program is rendered and every record and alias is known, marks those lines with the
union's CUDA layout: little-endian, every member at offset 0, sizes and alignments by the rules
:class:`~crosstl.backend.CUDA.buffer_layout_import.KernelBufferLayouts` applies to kernel buffers. A union is refused
instead when the importer cannot describe that layout:

* a member whose type has no modelled CUDA layout (a pointer, a matrix, a library type), or whose layout depends on
  the alignment of ``double4``;
* a member declared with an alignment specifier, a ``packed`` or ``aligned`` attribute, or a bit-field width, which
  the parser does not keep;
* members of an anonymous struct, which follow each other inside the union's storage rather than overlap;
* a union whose own alignment attribute is not a literal.

A struct with anonymous union members is refused too, because the importer writes those members one after another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .buffer_layout_import import KernelBufferLayouts, record_alignment

__all__ = ["SOURCE_ABI", "UnionLayoutContracts", "anonymous_union_members"]

# The ABI name the contract records; the generators that rebuild a union natively accept only their own ABI.
SOURCE_ABI = "cuda"


def _member_issue(member):
    """Why the place of union member ``member`` in the union's storage is not described, or ``""``."""
    if "static" in (getattr(member, "qualifiers", None) or ()):
        return "is a static data member, which has storage of its own"
    if getattr(member, "anonymous_aggregate", None) == "struct":
        return (
            "belongs to an anonymous struct, whose members follow each other in the union's storage rather than "
            "overlap"
        )
    specifiers = getattr(member, "layout_specifiers", None)
    if specifiers:
        return f"is declared with {', '.join(specifiers)}, which changes its alignment or size"
    return ""


def anonymous_union_members(node):
    """The names of the members of record ``node`` that come from an anonymous union, in declaration order."""
    return [
        str(getattr(member, "name", ""))
        for member in getattr(node, "members", None) or ()
        if getattr(member, "anonymous_aggregate", None) == "union"
    ]


@dataclass
class _Union:
    name: str
    attributes: tuple
    index: int  # output line of ``struct <name> {``
    members: list = field(
        default_factory=list
    )  # (CrossGL type text, name, output line, issue)

    def member(self, member_type, member_name, index, node):
        """Record a member line; ``node`` is the source member, checked for layout the parser dropped."""
        self.members.append((member_type, member_name, index, _member_issue(node)))


class UnionLayoutContracts:
    """The unions an importer wrote, marked with their layout contract once the program is rendered."""

    def __init__(self, label="CUDA"):
        self.label = label
        self.unions = []

    def union(self, name, attributes, index):
        """Record union ``name`` declared at output line ``index``; returns the record its members go to."""
        record = _Union(str(name), tuple(attributes or ()), index)
        self.unions.append(record)
        return record

    def apply(self, output, structs, aliases, refuse):
        """Mark the union lines of ``output`` in place, or call ``refuse(reason)`` for a union it cannot describe.

        ``structs`` and ``aliases`` are the records and type aliases the importer emitted, as
        :class:`~crosstl.backend.CUDA.buffer_layout_import.EmittedBufferLayouts` holds them.
        """
        if not self.unions:
            return
        planner = KernelBufferLayouts(
            {
                name: (tuple(members), attributes, union)
                for name, (members, attributes, union) in structs.items()
            },
            dict(aliases),
            self.label,
        )
        for record in self.unions:
            if not record.members:
                continue  # a declaration without members has no storage to share
            contract = self._contract(record, planner)
            if isinstance(contract, str):
                refuse(contract)
                continue
            size, alignment, member_layouts = contract
            output[record.index] = _prefix(
                output[record.index],
                f"@union_layout({size}, {alignment}, little_endian, {SOURCE_ABI})",
            )
            for (_type, _name, index, _issue), (member_size, member_alignment) in zip(
                record.members, member_layouts
            ):
                output[index] = _prefix(
                    output[index],
                    f"@union_member_layout(0, {member_size}, {member_alignment})",
                )

    def _contract(self, record, planner):
        """``(size, alignment, member layouts)`` of ``record`` in CUDA's layout, or the reason it has none."""
        subject = f"{self.label} union '{record.name}'"
        alignment, unmodelled = record_alignment(record.attributes)
        if unmodelled:
            return f"{subject}: its layout attribute {unmodelled} is not modelled, so CrossGL cannot describe its storage"
        size, layouts = 0, []
        for member_type, member_name, _index, issue in record.members:
            if issue:
                return f"{subject}: member '{member_name}' {issue}; CrossGL cannot describe its storage"
            layout, reason = planner.cuda_layout(member_type)
            if layout is None:
                return f"{subject}: member '{member_name}': {reason}, so CrossGL cannot describe its storage"
            member_size, member_alignment = layout
            layouts.append(layout)
            size = max(size, member_size)
            alignment = max(alignment, member_alignment)
        size = -(-size // alignment) * alignment
        return size, alignment, layouts


def _prefix(line, attribute):
    body = line.lstrip()
    return f"{line[: len(line) - len(body)]}{attribute} {body}"
