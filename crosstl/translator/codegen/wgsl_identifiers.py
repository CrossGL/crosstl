"""Identifiers in the WGSL generator: function naming and overload resolution, and WGSL identifier escaping."""

import re

from ..ast import (
    ArrayType,
    GenericType,
    IdentifierNode,
    MatrixType,
    MemberAccessNode,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
    VectorType,
)
from .wgsl_builtin_functions import WGSL_BUILTIN_FUNCTIONS
from .wgsl_numeric_types import unsupported_wgsl_numeric_family

TEXTURE_COORDINATE_COMPONENTS = str.maketrans("stpq", "xyzw")


class WGSLIdentifierMixin:
    """Identifier members of ``WGSLCodeGen``."""

    def texture_sampler_name(self, texture_name):
        return f"{texture_name}_sampler"

    def module_identifier_name(self, name):
        return self._module_identifier_names.get(name, self.safe_wgsl_identifier(name))

    def function_declaration_identifier_name(self, function):
        signature_key = self.function_signature_key(function)
        if signature_key in self._function_signature_identifier_names:
            return self._function_signature_identifier_names[signature_key]
        return self.function_identifier_name(getattr(function, "name", ""))

    def function_identifier_name(self, name):
        if name in self._function_identifier_names:
            return self._function_identifier_names[name]
        overloads = self._function_overloads_by_name.get(name, ())
        if overloads:
            return self.function_declaration_identifier_name(overloads[0])
        return self.safe_wgsl_identifier(name)

    def function_groups_by_name(self, functions):
        groups = {}
        for function in functions:
            function_name = getattr(function, "name", "")
            if not function_name:
                continue
            groups.setdefault(function_name, []).append(function)
        return groups

    def unique_function_module_name(self, base_name, used_names):
        candidate = self.safe_wgsl_identifier(base_name)
        if (
            candidate in used_names
            or self.requires_wgsl_identifier_escape(candidate)
            or candidate in WGSL_BUILTIN_FUNCTIONS
        ):
            candidate = self.unique_wgsl_identifier(
                self.escaped_wgsl_identifier_base(candidate), used_names
            )
        used_names.add(candidate)
        return candidate

    def function_signature_key(self, function):
        return (
            getattr(function, "name", ""),
            tuple(
                self.function_type_signature(getattr(parameter, "param_type", None))
                for parameter in getattr(function, "parameters", []) or []
            ),
        )

    def function_overload_suffix(self, function):
        parts = [
            self.function_type_suffix(getattr(parameter, "param_type", None))
            for parameter in getattr(function, "parameters", []) or []
        ]
        return "_".join(part for part in parts if part) or "void"

    def function_type_suffix(self, vtype):
        signature = self.function_type_signature(vtype)
        suffix = re.sub(r"[^0-9A-Za-z]+", "_", signature).strip("_")
        return suffix or "value"

    def function_type_signature(self, vtype):
        if vtype is None:
            return "void"
        try:
            mapped_type = self.type_name_string(vtype, allow_storage_resources=True)
        except ValueError:
            mapped_type = None
        if mapped_type is not None:
            return re.sub(r"[^0-9A-Za-z]+", "_", mapped_type).strip("_") or "value"
        if isinstance(vtype, PrimitiveType):
            # type_name_string spells every scalar PRIMITIVE_TYPE_MAP holds, so this scalar has no WGSL spelling and
            # its declaration is refused; the overload suffix only has to tell it apart, so it names the source type.
            return vtype.name.lower()
        if isinstance(vtype, VectorType):
            element = self.function_type_signature(vtype.element_type)
            return f"vec{vtype.size}_{element}"
        if isinstance(vtype, MatrixType):
            element = self.function_type_signature(vtype.element_type)
            return f"mat{vtype.cols}x{vtype.rows}_{element}"
        if isinstance(vtype, ArrayType):
            size = "unsized" if vtype.size is None else str(vtype.size)
            return f"array_{size}_{self.function_type_signature(vtype.element_type)}"
        if isinstance(vtype, ReferenceType):
            return self.function_type_signature(vtype.referenced_type)
        if isinstance(vtype, PointerType):
            return f"ptr_{self.function_type_signature(vtype.pointee_type)}"
        if isinstance(vtype, NamedType):
            base = self.semantic_key(str(vtype.name)) or str(vtype.name)
            if not vtype.generic_args:
                return base
            args = "_".join(
                self.function_type_signature(arg) for arg in vtype.generic_args
            )
            return f"{base}_{args}"
        if isinstance(vtype, GenericType):
            return self.semantic_key(str(vtype.name)) or str(vtype.name)
        if isinstance(vtype, str):
            return re.sub(r"[^0-9A-Za-z]+", "_", vtype).strip("_") or "value"
        return re.sub(r"[^0-9A-Za-z]+", "_", str(vtype)).strip("_") or "value"

    def resolve_function_overload(self, function_name, arguments, expected_type=None):
        candidates = [
            function
            for function in self._function_overloads_by_name.get(function_name, ())
            if len(getattr(function, "parameters", []) or []) == len(arguments)
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        argument_types = [self.expression_type(argument) for argument in arguments]
        scored = []
        for function in candidates:
            parameters = getattr(function, "parameters", []) or []
            score = 0
            compatible = True
            known_argument_count = 0
            for argument_type, parameter in zip(argument_types, parameters):
                if argument_type is None:
                    continue
                known_argument_count += 1
                if not self.function_types_compatible(
                    argument_type, getattr(parameter, "param_type", None)
                ):
                    compatible = False
                    break
                score += 4
            if not compatible:
                continue
            if expected_type is not None:
                if self.function_types_compatible(
                    getattr(function, "return_type", None), expected_type
                ):
                    score += 2
                elif known_argument_count == 0:
                    continue
            scored.append((score, function))
        if not scored:
            return candidates[0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def function_types_compatible(self, actual_type, expected_type):
        if actual_type is None or expected_type is None:
            return False
        return self.function_type_match_name(
            actual_type
        ) == self.function_type_match_name(expected_type)

    def function_type_match_name(self, vtype):
        try:
            return self.type_name_string(vtype, allow_storage_resources=True)
        except ValueError:
            return self.function_type_signature(vtype)

    def type_identifier_name(self, name):
        return self._type_identifier_names.get(name, self.safe_wgsl_identifier(name))

    def struct_member_identifier_name(self, struct_name, member_name):
        return self._struct_member_identifier_names.get(
            (struct_name, member_name), self.safe_wgsl_identifier(member_name)
        )

    def member_access_identifier_name(self, expr):
        object_type = self.array_element_type(self.expression_type(expr.object_expr))
        if self.vector_component_type(
            object_type
        ) is not None and self.is_vector_member(expr.member):
            # WGSL names components xyzw or rgba; GLSL's texture-coordinate names stpq are xyzw.
            return expr.member.translate(TEXTURE_COORDINATE_COMPONENTS)
        struct_name = self.struct_type_name(object_type)
        if not struct_name:
            return self.safe_wgsl_identifier(expr.member)
        return self.struct_member_identifier_name(struct_name, expr.member)

    def identifier_name(self, name):
        for aliases in reversed(self._identifier_alias_scopes):
            if name in aliases:
                return aliases[name]
        if name in self._module_identifier_names:
            return self._module_identifier_names[name]
        return self.safe_wgsl_identifier(name)

    def declare_local_identifier(self, name):
        if not self._identifier_alias_scopes:
            return self.safe_wgsl_identifier(name)
        aliases = self._identifier_alias_scopes[-1]
        if name in aliases:
            return aliases[name]
        used_names = set(self._identifier_scopes[-1])
        used_names.update(aliases.values())
        emitted_name = self.safe_wgsl_identifier(name)
        if emitted_name in used_names or self.requires_wgsl_identifier_escape(name):
            emitted_name = self.unique_wgsl_identifier(
                self.escaped_wgsl_identifier_base(name), used_names
            )
        aliases[name] = emitted_name
        self._identifier_scopes[-1].add(name)
        return emitted_name

    def wgsl_identifier_map(self, names):
        unique_names = []
        seen = set()
        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            unique_names.append(name)

        used_names = {
            name
            for name in unique_names
            if not self.requires_wgsl_identifier_escape(name)
        }
        identifier_map = {}
        for name in unique_names:
            if not self.requires_wgsl_identifier_escape(name):
                identifier_map[name] = name
                continue
            emitted_name = self.unique_wgsl_identifier(
                self.escaped_wgsl_identifier_base(name), used_names
            )
            identifier_map[name] = emitted_name
            used_names.add(emitted_name)
        return identifier_map

    def safe_wgsl_identifier(self, name):
        if self.requires_wgsl_identifier_escape(name):
            return self.escaped_wgsl_identifier_base(name)
        return name

    def escaped_wgsl_identifier_base(self, name):
        name = str(name)
        if name == "_":
            return "identifier_"
        if name.startswith("__"):
            stripped = name.lstrip("_")
            if not stripped or stripped[0].isdigit():
                # WGSL identifiers start with a letter or "_" (WGSL section 3.5); SPIR-V's unnamed blocks are "__0".
                return f"identifier_{stripped}" if stripped else "identifier_"
            return f"{stripped}_"
        return f"{name}_"

    def unique_wgsl_identifier(self, base_name, used_names):
        candidate = base_name
        suffix = 2
        while candidate in used_names or self.requires_wgsl_identifier_escape(
            candidate
        ):
            separator = "" if base_name.endswith("_") else "_"
            candidate = f"{base_name}{separator}{suffix}"
            suffix += 1
        return candidate

    def requires_wgsl_identifier_escape(self, name):
        name = str(name)
        return (
            name in self.WGSL_RESERVED_IDENTIFIERS
            or name in self._used_access_mode_identifiers
            or name == "_"
            or name.startswith("__")
        )

    def is_type_constructor_name(self, name):
        lower = str(name).lower()
        return (
            lower in self.PRIMITIVE_TYPE_MAP
            or self.TYPE_CONSTRUCTOR_RE.match(lower)
            or re.fullmatch(r"vec[234]<[^>]+>", lower)
            # A conversion to a type WGSL lacks is refused when its type is mapped.
            or unsupported_wgsl_numeric_family(name) is not None
        )

    def expression_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, MemberAccessNode):
            return f"{self.expression_name(expr.object_expr)}.{expr.member}"
        return self.generate_expression(expr)
