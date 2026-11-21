"""The CrossGL union layout contract of each named HIP union the importer writes.

The members of a C++ union share storage that starts at the union's first byte (C++ [class.union]), and HIP code
writes one member and reads another to reinterpret the stored bytes, which hipcc and Clang define. CrossGL has no
union type: a ``struct`` marked ``@union_layout(size, alignment, byte_order, abi)`` whose members are each marked
``@union_member_layout(offset, size, alignment)`` stands for one (:mod:`crosstl.backend.CUDA.union_layout_import`).
The HIP importer wrote a union as a plain struct and refused every program that read a member other than the one it
last wrote, or that copied, passed or reached a union through a pointer, because the struct gave each member storage
of its own.

:class:`HipUnionLayoutMixin` marks each named union with the contract, with ``hip`` as its ABI and the layout that
CUDA's importer computes, since hipcc lays out a union by the same C++ rules and HIP's vector types have CUDA's sizes
and alignments. A named union is then a value whose members share storage in every target, so the importer stops
tracking which member is active for it. A union is refused instead when the importer cannot describe its layout:

* a member with a bit-field width or a reference type, which the importer does not write;
* a member declared with an alignment specifier or an ``aligned`` or ``packed`` attribute;
* members of an anonymous struct, which follow each other inside the union's storage rather than overlap;
* any member :class:`~crosstl.backend.CUDA.union_layout_import.UnionLayoutContracts` cannot lay out.

An anonymous union has no CrossGL struct of its own: its members are written into the enclosing record one after
another, so the importer keeps tracking its active member and refuses a read of any other member.

The OpenCL C importer is built on the HIP converter but lays out vectors by other rules (a three-component OpenCL C
vector is as wide as a four-component one, OpenCL C 3.0 section 6.3.5), so it writes no contract and keeps tracking
the active member of every union until it describes its own layout.
"""

from __future__ import annotations

import re

from ...CUDA.union_layout_import import UnionLayoutContracts
from ..HipAst import StructNode, VariableNode

__all__ = ["HIP_UNION_ABI", "HipUnionLayoutMixin"]

# The ABI name the HIP importer records in ``@union_layout``.
HIP_UNION_ABI = "hip"

# The importers built on the HIP converter that write the contract, by ``SOURCE_LANGUAGE_NAME``, and their ABI name.
UNION_LAYOUT_ABIS = {"HIP": HIP_UNION_ABI}

_LAYOUT_ATTRIBUTE = re.compile(r"align|packed", re.IGNORECASE)


class HipUnionLayoutMixin:
    """Union layout members of ``HipToCrossGLConverter``."""

    def begin_union_layout_contracts(self):
        language = self.SOURCE_LANGUAGE_NAME
        abi = UNION_LAYOUT_ABIS.get(language)
        self.union_layout_contracts = (
            UnionLayoutContracts(language, abi) if abi else None
        )

    def apply_union_layout_contracts(self):
        """Mark the union lines of the output, or refuse the unions whose layout cannot be described."""
        if self.union_layout_contracts is None:
            return
        layouts = self.emitted_buffer_layouts
        self.union_layout_contracts.apply(
            self.output, layouts.structs, layouts.aliases, self.refuse_import
        )

    def union_tracks_active_member(self, node):
        """Whether a read of a member of union ``node`` must be shown to read the member last written.

        A named union is written as a struct with the union layout contract, whose members share storage in every
        target. An anonymous union's members are written into the enclosing record, each with storage of its own.
        """
        if not getattr(node, "is_union", False):
            return False
        if getattr(self, "union_layout_contracts", None) is None:
            return True
        return not (getattr(node, "name", None) and getattr(node, "has_body", True))

    def hip_union_layout_record(self, node, output_name):
        """The record that marks union ``node``, written as struct ``output_name``, with its layout contract."""
        contracts = getattr(self, "union_layout_contracts", None)
        if contracts is None or not getattr(node, "is_union", False):
            return None
        record = contracts.union(
            output_name, getattr(node, "attributes", None), len(self.output)
        )
        record.issue = self.hip_union_anonymous_struct_issue(node)
        return record

    def hip_union_member_issue(self, owner, member):
        """Why the place of ``member`` in the storage of union ``owner`` is not described, or ``""``."""
        if hasattr(member, "bitfield_width"):
            return "has a bit-field width, which the importer does not write"
        if self.record_member_layout_issue(owner, member) is not None:
            return "is a reference, which has no storage CrossGL can describe"
        specifiers = [
            str(attribute)
            for attribute in getattr(member, "attributes", None) or ()
            if _LAYOUT_ATTRIBUTE.search(str(attribute))
        ]
        if specifiers:
            return f"is declared with {', '.join(specifiers)}, which changes its alignment or size"
        return ""

    def hip_union_anonymous_struct_issue(self, node):
        """The issue of a union with anonymous struct members, which follow each other, or ``""``."""
        members = list(getattr(node, "members", None) or ())
        for index, member in enumerate(members):
            if (
                not isinstance(member, StructNode)
                or getattr(member, "name", None)
                or getattr(member, "is_union", False)
            ):
                continue
            following = members[index + 1] if index + 1 < len(members) else None
            if isinstance(
                following, VariableNode
            ) and self.nested_record_declarator_matches(
                following, self.nested_record_declarator_prefix(member)
            ):
                continue  # ``struct { ... } name;`` is one member of its own struct type
            names = ", ".join(
                repr(str(getattr(field, "name", "")))
                for field in getattr(member, "members", None) or ()
                if isinstance(field, VariableNode)
            )
            return (
                f"has anonymous struct members {names}, which follow each other in the union's storage rather "
                "than overlap"
            )
        return ""
