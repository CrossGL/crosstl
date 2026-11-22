"""Identifier scopes and sanitizing, static struct member references, sizeof and Metal layouts."""

import re

from ....translator.codegen.array_utils import evaluate_literal_int_expression
from ..MetalAst import (
    EnumNode,
    FunctionCallNode,
    FunctionNode,
    StaticAssertNode,
    StructNode,
    TypeAliasNode,
    VariableNode,
)
from ..narrow_float_constants import (
    narrow_float_constant,
    render_narrow_float_constant,
)
from ..partial_struct_specializations import UndecidedPartialSpecialization
from ..scalar_type_identity import canonical_scalar_type_spelling
from ..type_layout import metal_type_layout
from .support import (
    MetalSizeofResolutionError,
    MetalStaticConstantResolutionError,
)


class MetalIdentifierMixin:
    """Members of ``MetalToCrossGLConverter``: identifier scopes and sanitizing, static struct member references, sizeof and Metal layouts."""

    def texture_method_descriptor(self, method):
        function = self.texture_method_functions.get(method)
        if function is None:
            return None
        return {
            "method": method,
            "function": function,
            "storage_operation": self.texture_method_storage_operations.get(method),
            "sampled_texture": method in self.sampled_texture_methods,
        }

    def resource_method_descriptor(self, method):
        descriptor = self.texture_method_descriptor(method)
        if descriptor is None:
            return None
        descriptor = dict(descriptor)
        descriptor["resource"] = (
            "texture_or_image" if descriptor["storage_operation"] else "texture"
        )
        descriptor["operation"] = self.texture_method_operations.get(method)
        return descriptor

    def push_identifier_scope(self):
        self.identifier_maps.append({})
        self.used_identifier_names.append(set())

    def pop_identifier_scope(self):
        self.identifier_maps.pop()
        self.used_identifier_names.pop()

    def sanitize_identifier(self, name):
        if not name:
            return name
        if (
            self.crossgl_identifier_pattern.match(name)
            and name not in self.crossgl_reserved_identifiers
        ):
            return name

        # A name's sanitized form depends on the name and, for a template name, on the spelling table, whose
        # entries never change once set; so it is computed once per table.
        spellings = self.__dict__.setdefault("template_name_spellings", {})
        memo = self.__dict__.get("_sanitized_identifier_memo")
        if memo is None or memo[0] is not spellings:
            memo = (spellings, {})
            self.__dict__["_sanitized_identifier_memo"] = memo
        key = str(name)
        sanitized = memo[1].get(key)
        if sanitized is None:
            sanitized = memo[1][key] = self._sanitize_identifier_text(key, spellings)
        return sanitized

    def _sanitize_identifier_text(self, name, spellings):
        if "<" in name:
            # ``f<uint>`` and ``f<uint32_t>`` name one specialization; the first spelling met names it.
            name = spellings.setdefault(canonical_scalar_type_spelling(name), name)
        parts = []
        for index, char in enumerate(str(name)):
            valid = (
                char == "_"
                or ("A" <= char <= "Z")
                or ("a" <= char <= "z")
                or (index > 0 and "0" <= char <= "9")
            )
            parts.append(char if valid else f"_u{ord(char):x}")
        candidate = "".join(parts)
        if not candidate or not re.match(r"^[A-Za-z_]", candidate):
            candidate = f"_{candidate}"
        if candidate in self.crossgl_reserved_identifiers:
            candidate = f"{candidate}_"
        return candidate

    def declare_identifier(self, name):
        if not name:
            return name
        if self.template_binding_shadow_scopes and any(
            name in bindings for bindings in self.template_value_bindings
        ):
            self.template_binding_shadow_scopes[-1].add(name)
        current_map = self.identifier_maps[-1]
        if name in current_map:
            return current_map[name]

        base = self.sanitize_identifier(name)
        candidate = base
        used = self.used_identifier_names[-1]
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        current_map[name] = candidate
        return candidate

    def render_identifier(self, name):
        template_value = self.template_value_binding(name)
        if template_value is not None:
            return template_value
        name = self.substitute_integral_constant_members(name)
        constructor_reference = self.render_constructor_member_identifier(name)
        if constructor_reference is not None:
            return constructor_reference
        for scope in reversed(self.identifier_maps):
            if name in scope:
                return scope[name]
        local_static_member = self.render_local_static_struct_member_identifier(name)
        if local_static_member is not None:
            return local_static_member
        static_member = self.render_static_struct_member_identifier(name)
        if static_member is not None:
            return static_member
        enum_owner = (getattr(self, "unscoped_enumerator_owners", None) or {}).get(name)
        if enum_owner is not None and not self.unscoped_enumerator_is_shadowed(name):
            return f"{enum_owner}::{self.sanitize_identifier(name)}"
        scoped = str(name).rsplit("::", 1)
        if len(scoped) == 2 and tuple(scoped) in (
            getattr(self, "enumerator_names", None) or ()
        ):
            # E::A of a declared enum is CrossGL's own spelling.
            return f"{scoped[0]}::{self.sanitize_identifier(scoped[1])}"
        narrow_constant = narrow_float_constant(name)
        if narrow_constant is not None and not self.unscoped_enumerator_is_shadowed(
            name
        ):
            return render_narrow_float_constant(self, narrow_constant)
        return self.sanitize_identifier(name)

    def unscoped_enumerator_is_shadowed(self, name):
        """Whether a variable of the program declares ``name``, which then hides the enumerator."""
        return name in (
            getattr(self, "current_variable_types", None) or {}
        ) or name in (getattr(self, "global_variable_types", None) or {})

    def constructor_identifier_is_shadowed(self, name):
        if self.current_constructor_scope_index is None:
            return False
        return any(
            name in scope
            for scope in self.identifier_maps[self.current_constructor_scope_index :]
        )

    def render_constructor_member_identifier(self, name):
        function = self.current_function
        if not getattr(function, "is_metal_constructor_factory", False):
            return None
        result_name = getattr(function, "constructor_result_name", None)
        if not result_name:
            return None
        if name == "this":
            return self.render_identifier(result_name)
        if self.constructor_identifier_is_shadowed(name):
            return None
        owner = getattr(function, "constructor_owner", None)
        static_key = (self.map_struct_name(owner), name) if owner else None
        if static_key in self.struct_static_constant_members:
            return self.render_resolved_static_constant(static_key)
        if self.current_constructor_member_type(name) is None:
            return None
        return (
            f"{self.render_identifier(result_name)}."
            f"{self.sanitize_identifier(name)}"
        )

    def substitute_integral_constant_members(self, name):
        rendered = str(name)
        substituted = set()
        for bindings in reversed(self.integral_constant_bindings):
            for parameter_name, value in bindings.items():
                if parameter_name in substituted:
                    continue
                rendered = re.sub(
                    rf"\b{re.escape(parameter_name)}\s*\.\s*value\b",
                    "true" if value else "false",
                    rendered,
                )
                substituted.add(parameter_name)
        return rendered

    def render_local_static_struct_member_identifier(self, name):
        owner = self.current_struct_static_constant_owner
        if owner is None or not isinstance(name, str) or "::" in name:
            return None
        key = (owner, name)
        if key not in self.struct_static_constant_members:
            return None
        self.propagate_struct_static_constexpr_dependency(key)
        return self.render_resolved_static_constant(key)

    def static_struct_member_owner(self, name):
        if not isinstance(name, str) or "::" not in name:
            return None
        struct_name, member_name = name.rsplit("::", 1)
        alias_target = self.local_struct_type_aliases.get(struct_name)
        resolved_struct = self.resolve_local_type_aliases(alias_target or struct_name)
        if resolved_struct not in self.struct_name_map:
            try:
                match = self.struct_templates_for_dependent_owner(resolved_struct)
            except UndecidedPartialSpecialization:
                match = None
            if match is not None and match[2]:
                candidates = match[0]
                if len(candidates) != 1:
                    raise MetalStaticConstantResolutionError(
                        resolved_struct,
                        member_name,
                        "multiple visible explicit specializations match the owner",
                    )
                resolved_struct = candidates[0].name
        if resolved_struct not in self.struct_name_map and "::" in resolved_struct:
            unqualified_struct = resolved_struct.rsplit("::", 1)[-1]
            if unqualified_struct in self.struct_name_map:
                resolved_struct = unqualified_struct
            else:
                resolved_struct = (
                    self.struct_member_alias_owner(resolved_struct) or resolved_struct
                )
        return resolved_struct, member_name

    def struct_member_alias_owner(self, qualified_name):
        # `Tile::Frag_t::kElems` names a static member of the struct that the
        # member alias `Tile::Frag_t` denotes ([class.qual]); follow the alias
        # chain from a visible concrete struct, or give up with None.
        owner, aliases = qualified_name.split("::", 1)
        if owner not in self.struct_name_map:
            return None
        target = self.resolve_concrete_struct_member_aliases(owner, f"::{aliases}")
        if target is None:
            return None
        target = self.normalized_metal_type(self.resolve_type_alias(target))
        return target if target in self.struct_name_map else None

    def static_struct_member_type(self, name):
        owner = self.static_struct_member_owner(name)
        if owner is None:
            return None
        struct_name, member_name = owner
        if struct_name in self.ambiguous_struct_names:
            # Reuse the value resolver's ambiguity checks before choosing a type.
            self.render_equivalent_struct_static_constant(struct_name, member_name)
            struct_name = self.struct_static_constant_owner_candidates[struct_name][
                0
            ].name
        if struct_name not in self.struct_name_map:
            return None
        member = self.struct_static_constant_members.get(
            (self.map_struct_name(struct_name), member_name)
        )
        if member is None:
            return None
        return self.normalized_metal_type(self.resolve_type_alias(member.vtype))

    def render_static_struct_member_identifier(self, name, require_constant=False):
        owner = self.static_struct_member_owner(name)
        if owner is None:
            return None
        resolved_struct, member_name = owner
        if resolved_struct in self.ambiguous_struct_names:
            return self.render_equivalent_struct_static_constant(
                resolved_struct,
                member_name,
            )
        if resolved_struct not in self.struct_name_map:
            if require_constant:
                raise MetalStaticConstantResolutionError(
                    resolved_struct,
                    member_name,
                    "the inferred expression type does not name a visible struct",
                )
            return None
        mapped_struct = self.map_struct_name(resolved_struct)
        key = (mapped_struct, member_name)
        if key in self.struct_static_constant_members:
            self.propagate_struct_static_constexpr_dependency(key)
            return self.render_resolved_static_constant(key)
        if require_constant:
            raise MetalStaticConstantResolutionError(
                mapped_struct,
                member_name,
                "the selected declaration has no compile-time static member",
            )
        return self.sanitize_identifier(f"{mapped_struct}::{member_name}")

    def propagate_struct_static_constexpr_dependency(self, dependency_key):
        if (
            dependency_key in self.struct_static_constexpr_member_keys
            and self.struct_static_constant_resolution_stack
        ):
            self.struct_static_constexpr_member_keys.add(
                self.struct_static_constant_resolution_stack[-1]
            )

    def render_decltype_static_struct_member(self, expr):
        owner = getattr(expr, "object", None)
        if not isinstance(owner, FunctionCallNode):
            return None
        if str(getattr(owner, "name", "")) not in {"decltype", "metal::decltype"}:
            return None
        arguments = getattr(owner, "args", None) or []
        member = str(getattr(expr, "member", ""))
        if len(arguments) != 1:
            raise MetalStaticConstantResolutionError(
                "decltype(expression)",
                member,
                "decltype requires exactly one expression",
            )

        owner_name = f"decltype({self.generate_expression(arguments[0], False)})"
        inferred_owner = self.expression_metal_type(arguments[0])
        if inferred_owner is None:
            raise MetalStaticConstantResolutionError(
                owner_name,
                member,
                "the expression type could not be inferred",
            )
        resolved_owner = self.normalized_metal_type(
            self.resolve_type_alias(inferred_owner)
        )
        rendered = self.render_static_struct_member_identifier(
            f"{resolved_owner}::{member}",
            require_constant=True,
        )
        if rendered is not None:
            self.record_elided_parameter_uses(arguments[0])
        return rendered

    def record_elided_parameter_uses(self, expression):
        parameter_scope = getattr(self, "current_parameter_scope", None)
        if parameter_scope is None:
            return
        names = set()
        self.collect_identifier_references(expression, names)
        for name in names:
            binding = next(
                (scope for scope in reversed(self.identifier_maps) if name in scope),
                None,
            )
            if binding is parameter_scope:
                self.current_elided_parameter_uses.add(name)

    def render_metal_sizeof_expression(self, expr):
        if str(getattr(expr, "name", "")) != "sizeof":
            return None
        arguments = getattr(expr, "args", None) or []
        if len(arguments) != 1:
            raise MetalSizeofResolutionError(
                "expression",
                "sizeof requires exactly one operand",
                getattr(expr, "source_location", None),
            )

        operand = arguments[0]
        operand_type = self.metal_sizeof_operand_type(operand)
        if operand_type is None:
            return None

        # A structure-local alias names its target; any other operand type is resolved as written.
        local_alias = operand_type
        if operand_type in self.local_struct_type_aliases:
            local_alias = self.local_struct_type_aliases[operand_type]
        resolved_type = self.resolve_type_alias(local_alias)
        layout = self.metal_concrete_type_layout(resolved_type)
        if layout is not None:
            self.record_elided_parameter_uses(operand)
            return str(layout[0])

        normalized_type = self.normalized_metal_type(resolved_type)
        if normalized_type in self.struct_name_map:
            raise MetalSizeofResolutionError(
                resolved_type,
                "aggregate object layout is not available",
                getattr(expr, "source_location", None),
            )
        if "*" in str(resolved_type) or "&" in str(resolved_type):
            raise MetalSizeofResolutionError(
                resolved_type,
                "pointer and reference object sizes are not portable",
                getattr(expr, "source_location", None),
            )
        return None

    def metal_concrete_type_layout(self, metal_type, resolving=None):
        resolved_type = self.resolve_type_alias(metal_type)
        vector = self.metal_vector_type_parts(resolved_type)
        if vector is not None:
            element, width = vector
            element = self.resolve_type_alias(self.resolve_local_type_aliases(element))
            resolved_type = f"vec<{element}, {width}>"
        layout = metal_type_layout(resolved_type)
        if layout is not None:
            return layout

        array_base, _ = self.generic_type_parts(resolved_type)
        array = (
            self.metal_array_type_parts(resolved_type)
            if array_base in {"array", "metal::array"}
            else None
        )
        if array is not None:
            element, extent_text = array
            extent = self.evaluate_concrete_array_extent(extent_text)
            if not isinstance(extent, int) or extent < 0:
                return None
            if resolved_type in (resolving or ()):
                return None
            element_layout = self.metal_concrete_type_layout(
                element, {*(resolving or ()), resolved_type}
            )
            if element_layout is None:
                return None
            size, alignment = element_layout
            stride = ((size + alignment - 1) // alignment) * alignment
            total = stride * max(1, extent)
            return (total, alignment) if total < (1 << 63) else None

        struct_name = self.normalized_metal_type(resolved_type)
        if struct_name in self.ambiguous_struct_names:
            return None
        struct_node = self.struct_declarations.get(struct_name)
        if struct_node is None:
            return None
        alignas_alignment = self.metal_alignment_specifier_value(
            getattr(struct_node, "alignas", None)
        )
        if (
            alignas_alignment is None
            or getattr(struct_node, "attributes", None)
            or getattr(struct_node, "template_parameters", None)
            or getattr(struct_node, "generics", None)
            or getattr(struct_node, "bases", None)
            or getattr(struct_node, "base_classes", None)
            or getattr(struct_node, "base_types", None)
        ):
            return None

        resolving = set(resolving or ())
        if struct_name in resolving:
            return None
        resolving.add(struct_name)

        offset = 0
        aggregate_alignment = 1
        is_union = getattr(struct_node, "aggregate_kind", None) == "union"
        for member in getattr(struct_node, "members", []) or []:
            if not isinstance(member, VariableNode):
                if isinstance(
                    member,
                    (
                        EnumNode,
                        FunctionNode,
                        StaticAssertNode,
                        StructNode,
                        TypeAliasNode,
                    ),
                ):
                    continue
                return None
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(member, "qualifiers", []) or []
            }
            if "static" in qualifiers:
                continue
            member_layout = self.metal_struct_member_layout(member, resolving)
            if member_layout is None:
                return None
            member_size, member_alignment = member_layout
            if is_union:
                offset = max(offset, member_size)
            else:
                offset = self.align_metal_layout_offset(offset, member_alignment)
                offset += member_size
            aggregate_alignment = max(aggregate_alignment, member_alignment)

        aggregate_alignment = max(aggregate_alignment, alignas_alignment)
        aggregate_size = self.align_metal_layout_offset(offset, aggregate_alignment)
        if aggregate_size == 0:
            aggregate_size = 1
        if aggregate_size > (1 << 63) - 1:
            return None
        return aggregate_size, aggregate_alignment

    def metal_struct_member_layout(self, member, resolving):
        if (
            getattr(member, "alignas", None)
            or getattr(member, "bitfield_width", None) is not None
        ):
            return None
        member_type = getattr(member, "vtype", None)
        declarator_suffix = str(getattr(member, "declarator_type_suffix", "") or "")
        if (
            not member_type
            or "*" in str(member_type)
            or "&" in str(member_type)
            or "*" in declarator_suffix
            or "&" in declarator_suffix
        ):
            return None

        layout = self.metal_concrete_type_layout(member_type, resolving)
        if layout is None:
            return None
        member_size, member_alignment = layout
        for extent_expression in getattr(member, "array_sizes", []) or []:
            extent = evaluate_literal_int_expression(extent_expression)
            if extent is None and extent_expression is not None:
                # ``sizeof`` and named constants fold only once rendered.
                extent = self.evaluate_concrete_array_extent(
                    self.format_array_extent(extent_expression)
                )
            if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
                return None
            stride = self.align_metal_layout_offset(member_size, member_alignment)
            if stride > ((1 << 63) - 1) // extent:
                return None
            member_size = stride * extent
        return member_size, member_alignment

    def metal_union_layout_contract(self, struct_node):
        """Return source ABI metadata retained by pointer-free target lowering."""
        union_name = getattr(struct_node, "name", None)
        aggregate_layout = self.metal_concrete_type_layout(union_name)
        aggregate_size, aggregate_alignment = aggregate_layout or (0, 0)
        member_layouts = {}
        for member in getattr(struct_node, "members", []) or []:
            if not isinstance(member, VariableNode):
                continue
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(member, "qualifiers", []) or []
            }
            if "static" in qualifiers:
                continue
            member_layouts[id(member)] = self.metal_struct_member_layout(
                member,
                {union_name} if union_name else set(),
            ) or (0, 0)
        return {
            "size": aggregate_size,
            "alignment": aggregate_alignment,
            "members": member_layouts,
        }

    @staticmethod
    def align_metal_layout_offset(offset, alignment):
        return ((offset + alignment - 1) // alignment) * alignment

    def metal_sizeof_operand_type(self, operand):
        if isinstance(operand, str):
            variable_type = self.current_variable_types.get(
                operand,
                self.global_variable_types.get(operand),
            )
            return variable_type or operand
        if (
            isinstance(operand, VariableNode)
            and not getattr(operand, "vtype", None)
            and isinstance(operand.name, str)
        ):
            # The parser reads a type the lexer has no keyword for (``bfloat4``) as a variable reference.
            variable_type = self.expression_metal_type(operand)
            return variable_type or self.metal_sizeof_operand_type(operand.name)
        return self.expression_metal_type(operand)

    def render_resolved_static_constant(self, key):
        constant = self.resolve_struct_static_constant(key)
        if constant is None:
            owner, member = key
            raise MetalStaticConstantResolutionError(
                owner,
                member,
                "the selected declaration has no constant initializer",
            )
        return self.render_static_constant_value(constant)

    @staticmethod
    def render_static_constant_value(constant):
        if re.fullmatch(
            r"(?:true|false|[-+]?\d+(?:[uU])?|"
            r"0[xX][0-9a-fA-F]+[uU]?|0[bB][01]+[uU]?)",
            constant,
        ):
            return constant
        return f"({constant})"

    def reserve_generated_identifier(self, base):
        used = self.used_identifier_names[-1]
        sanitized_base = self.sanitize_identifier(base)
        candidate = sanitized_base
        suffix = 2
        while candidate in used:
            candidate = f"{sanitized_base}_{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate
