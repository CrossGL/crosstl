"""Rust locals that hold a mutable reference, ``let p = &mut place;``, imported as names for the place.

CrossGL has no reference locals, and copying the place into a local would drop every write through ``p``. While
``p`` lives, Rust's borrow rules forbid any other access to the place (The Rust Reference, "References"), so the
importer writes ``place`` wherever the function uses ``p`` or ``*p``. An index inside the place is evaluated once,
when the reference is taken, so the importer copies each index that is not a literal into a new local at that point;
a later change to the variables it reads does not move the reference. A place that is not a variable, its fields and
its elements, such as a function's result, is refused.
"""

from __future__ import annotations

import re

from ..common_ast import ArrayAccessNode, MemberAccessNode
from .RustAst import DereferenceNode, ReferenceNode

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INTEGER_LITERAL = re.compile(r"^\d(?:_?\d)*(?:[iu](?:8|16|32|64|size))?$")


class RustReferenceLocalMixin:
    """Members of ``RustToCrossGLConverter`` for ``let`` bindings of ``&mut`` places."""

    def is_mutable_reference_let(self, stmt):
        value = getattr(stmt, "value", None)
        return (
            isinstance(getattr(stmt, "name", None), str)
            and _IDENTIFIER.match(stmt.name) is not None
            and isinstance(value, ReferenceNode)
            and value.is_mutable
        )

    def generate_mutable_reference_let(self, stmt, indent):
        """Bind ``stmt.name`` to the place it borrows and return the copies of the place's indices."""
        copies = []
        place = self._reference_place(stmt.value.expression, indent, copies)
        if place is None:
            return self.refuse_import(
                f"'let {stmt.name} = &mut ...' borrows a value that is not a variable, a field or an element; "
                "CrossGL has no reference locals"
            )
        self.current_name_alias_scope()[stmt.name] = place
        return "".join(copies)

    def _reference_place(self, node, indent, copies):
        """The CrossGL spelling of the place ``node`` names, with its non-literal indices copied into ``copies``."""
        if isinstance(node, DereferenceNode):
            return self._reference_place(node.expression, indent, copies)
        if isinstance(node, str):
            if _IDENTIFIER.match(node) is None:
                return None
            return self.generate_expression(node)
        if isinstance(node, MemberAccessNode):
            if getattr(node, "operator", ".") not in {".", None}:
                return None
            base = self._reference_place(node.object, indent, copies)
            if base is None or not isinstance(node.member, str):
                return None
            return f"{base}.{node.member}"
        if isinstance(node, ArrayAccessNode):
            base = self._reference_place(node.array, indent, copies)
            if base is None:
                return None
            index = node.index
            if isinstance(index, str) and _INTEGER_LITERAL.match(index):
                return f"{base}[{self.generate_expression(index)}]"
            self.reference_index_counter = (
                getattr(self, "reference_index_counter", 0) + 1
            )
            name = f"crosstl_ref_index{self.reference_index_counter}"
            copies.append(
                f"{'    ' * indent}let {name} = {self.generate_expression(index)};\n"
            )
            return f"{base}[{name}]"
        return None
