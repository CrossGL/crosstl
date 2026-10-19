"""Value and reference semantics of local declarations initialised from storage.

C++ (and so MSL, CUDA and HIP) and HLSL give a local declaration of a value type
copy semantics: ``int s = a[i];`` reads the element once, and later writes to
``a[i]`` or to ``s`` do not affect the other (C++ [dcl.init]/17, HLSL's value
semantics for non-resource locals). A reference declaration ``device int& r =
a[i];`` (C++ [dcl.init.ref], MSL section 2.10) binds the element instead, so every
read and write through ``r`` reaches the buffer.

SPIR-V has no reference type. This mixin lowers a reference to a storage element as
a name bound to the element's access-chain pointer, computed once at the point of
declaration. It is not a pointer value: compound assignment through the reference
updates the element, never the address.

The mixin relies on the host class's ``local_variables``, ``local_variable_types``
and ``struct_types`` maps and its type-name helpers (``normalize_reference_type_name``,
``is_resource_type_name``, ``array_base_type_name``, ``normalize_primitive_name``,
``normalize_generic_vector_type``, ``normalize_hlsl_matrix_type`` and
``vector_component_type_and_count``).
"""

from __future__ import annotations

import re

VALUE_SCALAR_TYPE_NAMES = frozenset(
    {"float", "double", "int", "uint", "i64", "u64", "bool"}
)
MATRIX_TYPE_NAME = re.compile(r"(d)?mat([234])(?:x([234]))?")


class SpirvLocalReferenceMixin:
    """Bind local references to storage elements; copy value-typed locals."""

    @staticmethod
    def declared_type_name_is_reference(type_name) -> bool:
        """Return whether a declared type uses CrossGL reference spelling."""
        if type_name is None:
            return False
        type_str = str(type_name).strip()
        return type_str.startswith("&") or type_str.endswith("&")

    def declared_type_name_is_value(self, type_name) -> bool:
        """Return whether a declared type is a scalar, vector, matrix or struct.

        Arrays of those are values too. Pointer, reference, resource and
        unrecognised spellings return False, so they keep alias inference.
        """
        if type_name is None or self.declared_type_name_is_reference(type_name):
            return False
        type_str = re.sub(r"\s+", "", str(type_name))
        if not type_str or "*" in type_str:
            return False
        base = self.array_base_type_name(type_str)
        base = self.normalize_hlsl_matrix_type(self.normalize_generic_vector_type(base))
        if self.normalize_primitive_name(base) in VALUE_SCALAR_TYPE_NAMES | {
            self.BFLOAT16_TYPE_NAME
        }:
            return True
        if self.vector_component_type_and_count(base):
            return True
        if MATRIX_TYPE_NAME.fullmatch(base):
            return True
        return base in self.struct_types

    def referenced_value_type_name(self, type_name) -> str | None:
        """Return the referenced type of a reference to a non-resource value."""
        if not self.declared_type_name_is_reference(type_name):
            return None
        referenced = self.normalize_reference_type_name(type_name)
        if referenced is None or self.is_resource_type_name(referenced):
            return None
        return referenced

    def bind_local_storage_element_reference(
        self, name: str, element_pointer, referenced_type_name: str
    ) -> None:
        """Make ``name`` an lvalue alias of ``element_pointer``.

        The binding does not register pointer-offset state, so ``r += 1`` adds to
        the element rather than advancing an address, and it is not a descriptor
        alias, so plain assignment stores through the element pointer.
        """
        self.local_variables[name] = type(element_pointer)(
            element_pointer.id, element_pointer.type, name
        )
        self.local_variable_types[name] = referenced_type_name
