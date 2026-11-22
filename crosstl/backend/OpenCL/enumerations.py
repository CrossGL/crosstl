"""OpenCL C enumerations imported as CrossGL enumerations whose constants are qualified with their type.

OpenCL C enumerations are C's: unscoped, so a constant such as ``Low`` is an ordinary identifier of the scope that
declares the enumeration (C11 6.2.1p7). CrossGL scopes a constant to its enumeration and reads it as ``Mode::Low``.
A named enumeration (``enum Mode {...}`` or ``typedef enum {...} Mode``) therefore stays a CrossGL ``enum`` whose
variables have that type, and each constant is entered in the current name scope under its qualified name, so a
later reference reads ``Mode::Low`` while a local variable or parameter of the same name still shadows it.

An enumeration with neither a tag nor a typedef name has no CrossGL type to qualify its constants with; its
constants become ``const i32`` declarations, the ``int`` constants C gives them (C11 6.4.4.3p2).

Every CrossGL generator writes an enumeration and its constants as ``int``. Clang gives a C enumeration whose
constants are all non-negative the type ``unsigned int``, so a value of such a type converted to a floating-point
type, divided, shifted right or compared after it has wrapped below zero reads differently; constants themselves
are ``int`` in C as in CrossGL.
"""

from __future__ import annotations

__all__ = ["OpenCLEnumerationMixin"]


def _member_parts(member):
    if isinstance(member, tuple):
        return member[0], member[1]
    return getattr(member, "name", str(member)), getattr(member, "value", None)


class OpenCLEnumerationMixin:
    """``visit_EnumNode`` for the OpenCL C importer."""

    def visit_EnumNode(self, node):
        members = getattr(node, "members", None) or getattr(node, "variants", [])
        name = getattr(node, "name", None)
        if name:
            self.register_enumeration_constants(name, members)
            return super().visit_EnumNode(node)

        next_value = 0
        for member in members:
            member_name, member_value = _member_parts(member)
            if member_value is None:
                value = str(next_value if next_value is not None else 0)
            else:
                value = self.visit(member_value)

            self.emit(f"const i32 {member_name} = {value};")
            next_value = self.next_anonymous_enum_value(value)

    def register_enumeration_constants(self, enum_name, members):
        """Enter each constant of enumeration ``enum_name`` in the current scope as ``Enum::constant``.

        The constants are entered before the enumeration's own initializers are written, so a constant defined
        from an earlier one (``B = A + 1``) reads ``Mode::A`` there too, as CrossGL requires.
        """
        if not self.identifier_name_scopes:
            self.identifier_name_scopes.append({})
        scope = self.identifier_name_scopes[-1]
        qualified_enum = self.sanitize_identifier_name(enum_name)
        for member in members:
            member_name, _value = _member_parts(member)
            if not isinstance(member_name, str) or not member_name:
                continue
            constant = self.sanitize_enum_member_name(member_name)
            scope[member_name] = f"{qualified_enum}::{constant}"

    def next_anonymous_enum_value(self, value):
        try:
            return int(str(value), 0) + 1
        except ValueError:
            return None
