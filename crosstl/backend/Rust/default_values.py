"""``T::default()`` for the value types the Rust importer knows, written as an explicit CrossGL zero value.

``#[derive(Default)]`` gives every field of a struct its own default, and glam vectors, matrices and the scalar
primitives default to zero (``false`` for ``bool``). The importer writes that value out component by component:
``Pair::default()`` becomes ``Pair(0.0, 0)``, ``Vec3::default()`` becomes ``vec3(0.0, 0.0, 0.0)`` and an array
field becomes an initializer list, which every generator reads as an ordinary constructor. A type with a
hand-written ``impl Default`` keeps its own function: the importer resolves that call as an associated impl
function before it reaches this mixin. Generic structs and member types without a known zero are left alone.
"""

from __future__ import annotations

import re

# The CrossGL zero literal of each scalar a Rust scalar primitive maps to.
_SCALAR_ZEROS = {
    "bool": "false",
    "half": "0.0",
    "float": "0.0",
    "double": "0.0",
    "int": "0",
    "uint": "0u",
    "int8_t": "0",
    "int16_t": "0",
    "int64_t": "0",
    "uint8_t": "0u",
    "uint16_t": "0u",
    "uint64_t": "0u",
}
# The scalar of each CrossGL vector or matrix family prefix.
_FAMILY_SCALARS = {
    "": "float",
    "d": "double",
    "h": "half",
    "i": "int",
    "u": "uint",
    "b": "bool",
}
_VECTOR_RE = re.compile(r"^(?P<family>[dhiub]?)vec(?P<count>[234])$")
_MATRIX_RE = re.compile(r"^(?P<family>[dh]?)mat(?P<cols>[234])(?:x(?P<rows>[234]))?$")
_ARRAY_RE = re.compile(r"^\[(?P<count>\d+)\]$")
# Deeper nesting than this is not a value type a kernel declares; such a call is left as written.
_MAX_DEPTH = 16


class RustDefaultValueMixin:
    """Resolve and write ``T::default()`` calls on structs, glam types and scalars."""

    def default_value_type(self, function_name, args):
        """The Rust type whose default ``function_name(args)`` produces, or ``None`` for any other call."""
        if args or not isinstance(function_name, str):
            return None
        function_name = self.normalize_associated_type_path(function_name)
        if "::" not in function_name:
            return None
        type_name, method_name = function_name.rsplit("::", 1)
        if method_name != "default" or not type_name or "<" in type_name:
            return None
        if self.crossgl_zero_value(type_name) is None:
            return None
        return type_name

    def format_default_value_call(self, function_name, args):
        """The CrossGL zero value of a ``T::default()`` call :meth:`default_value_type` resolves, else ``None``."""
        type_name = self.default_value_type(function_name, args)
        if type_name is None:
            return None
        value = self.crossgl_zero_value(type_name)
        self.add_value_type(value, type_name)
        return value

    def crossgl_zero_value(self, type_name, depth=0):
        """The CrossGL expression of ``type_name``'s default value, or ``None`` when it has no known zero."""
        if not type_name or depth > _MAX_DEPTH:
            return None
        type_name = self.normalize_receiver_type(type_name)
        resolved = self.resolve_struct_type(type_name)
        if resolved is not None:
            return self._struct_zero_value(type_name, resolved, depth)
        mapped = self.map_type(type_name)
        array_parts = self.split_array_type(mapped)
        if array_parts is not None:
            return self._array_zero_value(array_parts, depth)
        return self._value_zero(mapped)

    def _struct_zero_value(self, type_name, resolved, depth):
        struct_name, substitutions = resolved
        members = self.struct_member_types.get(struct_name)
        if substitutions or not members:
            return None
        values = []
        for member_type in members.values():
            value = self.crossgl_zero_value(member_type, depth + 1)
            if value is None:
                return None
            values.append(value)
        return f"{self.map_type(type_name)}({', '.join(values)})"

    def _array_zero_value(self, array_parts, depth):
        element_type, suffix = array_parts
        match = _ARRAY_RE.match(suffix.replace(" ", ""))
        if match is None:
            return None
        element = self.crossgl_zero_value(element_type, depth + 1)
        if element is None:
            return None
        return "{" + ", ".join([element] * int(match.group("count"))) + "}"

    @staticmethod
    def _value_zero(mapped):
        scalar = _SCALAR_ZEROS.get(mapped)
        if scalar is not None:
            return scalar
        vector = _VECTOR_RE.match(mapped or "")
        if vector is not None:
            zero = _SCALAR_ZEROS[_FAMILY_SCALARS[vector.group("family")]]
            return f"{mapped}({', '.join([zero] * int(vector.group('count')))})"
        matrix = _MATRIX_RE.match(mapped or "")
        if matrix is not None:
            zero = _SCALAR_ZEROS[_FAMILY_SCALARS[matrix.group("family")]]
            columns = int(matrix.group("cols"))
            rows = int(matrix.group("rows") or columns)
            return f"{mapped}({', '.join([zero] * (columns * rows))})"
        return None
