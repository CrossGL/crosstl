"""Function-local structs, unions, enums and type aliases."""

import re

from ..generic_local_aliases import bind_generic_local_alias
from ..local_aggregate_declarators import (
    union_layout,
    unnamed_aggregate_display_name,
)
from ..MetalAst import VariableNode
from ..scoped_member_types import (
    record_block_local_struct_alias,
    resolve_concrete_scoped_member_type,
)
from .support import MetalFunctionLocalTypeResolutionError


class MetalLocalAggregateMixin:
    """Members of ``MetalToCrossGLConverter``: function-local structs, unions, enums and type aliases."""

    def local_aggregate_error(
        self,
        struct_node,
        reason,
        *,
        source_location=None,
        unresolved_dependencies=(),
    ):
        return MetalFunctionLocalTypeResolutionError(
            self.current_function_specialization
            or self.current_function_name
            or "<unknown>",
            (
                f"<unnamed {struct_node.unnamed_aggregate_kind}>"
                if getattr(struct_node, "unnamed_aggregate_kind", None)
                else None
            )
            or getattr(struct_node, "name", None)
            or getattr(struct_node, "typedef_tag", None)
            or "<anonymous>",
            reason,
            source_location=(
                source_location
                or getattr(struct_node, "declaration_source_location", None)
                or getattr(struct_node, "source_location", None)
            ),
            unresolved_dependencies=unresolved_dependencies,
        )

    def validate_local_aggregate_shape(self, struct_node):
        aggregate_kind = getattr(struct_node, "aggregate_kind", None) or "struct"
        if aggregate_kind not in {"struct", "union"}:
            raise self.local_aggregate_error(
                struct_node,
                f"{aggregate_kind} storage semantics cannot be represented by "
                "struct hoisting",
            )
        if getattr(struct_node, "base_types", None):
            raise self.local_aggregate_error(
                struct_node,
                "base classes cannot be represented by aggregate hoisting",
            )
        if getattr(struct_node, "alignas", None) or getattr(
            struct_node, "attributes", None
        ):
            raise self.local_aggregate_error(
                struct_node,
                "aggregate alignment or attributes would be lost during hoisting",
            )
        if getattr(struct_node, "type_aliases", None):
            raise self.local_aggregate_error(
                struct_node,
                "aggregate-scoped type aliases are not materialized",
            )
        if getattr(struct_node, "constructors", None) or getattr(
            struct_node, "call_operator_declarations", None
        ):
            raise self.local_aggregate_error(
                struct_node,
                "constructors or call operators cannot be represented by aggregate "
                "hoisting",
            )
        if getattr(struct_node, "trailing_declarations", None):
            raise self.local_aggregate_error(
                struct_node,
                "a trailing object declaration cannot be separated from its local "
                "type",
            )
        if getattr(struct_node, "typedef_array_sizes", None) or getattr(
            struct_node, "typedef_declarator_type_suffix", ""
        ):
            raise self.local_aggregate_error(
                struct_node,
                "the typedef declarator changes the aggregate type",
            )
        if not (getattr(struct_node, "members", None) or []):
            raise self.local_aggregate_error(
                struct_node,
                "empty aggregates have target-dependent size and layout",
            )

    def reserve_local_aggregate_name(self, source_name):
        function_name = self.sanitize_identifier(
            self.current_function_specialization
            or self.current_function_name
            or "Function"
        )
        type_name = self.sanitize_identifier(source_name or "Aggregate")
        base = f"MetalLocal_{function_name}_{type_name}"
        candidate = base
        suffix = 2
        while candidate in self.local_aggregate_reserved_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.local_aggregate_reserved_names.add(candidate)
        return candidate

    def local_aggregate_alias_names(self, struct_node):
        names = []
        for name in (
            getattr(struct_node, "name", None),
            getattr(struct_node, "typedef_tag", None),
        ):
            if name and name not in names:
                names.append(name)
        tag_name = getattr(struct_node, "typedef_tag", None)
        if tag_name:
            names.append(f"struct {tag_name}")
        elif not getattr(struct_node, "is_typedef_aggregate", False):
            source_name = getattr(struct_node, "name", None)
            if source_name:
                names.append(f"struct {source_name}")
        return names

    def bind_local_aggregate_aliases(self, struct_node, canonical_name):
        for name in self.local_aggregate_alias_names(struct_node):
            self.type_aliases[name] = canonical_name
            self.type_alias_qualifiers[name] = []
            self.local_type_alias_names.add(name)
            self.local_struct_type_aliases[name] = canonical_name

    def local_aggregate_type_dependencies(self, metal_type, mapped_type):
        function_parameters = {
            name
            for kind, name in (
                getattr(self.current_function, "template_parameters", None) or []
            )
            if name and kind in {"typename", "class", "value"}
        }
        identifiers = set(
            re.findall(r"\b[A-Za-z_]\w*\b", f"{metal_type} {mapped_type}")
        )
        dependencies = identifiers & function_parameters
        dependencies.update(
            name
            for name in identifiers
            if name in self.local_struct_type_aliases
            and name not in self.local_type_alias_names
        )
        return tuple(sorted(dependencies))

    def local_aggregate_mapped_type_is_concrete(self, metal_type, mapped_type):
        mapped_type = str(mapped_type or "").strip()
        if not mapped_type or any(token in mapped_type for token in ("*", "&")):
            return False
        if self.crossgl_typedef_source_type(mapped_type) is not None:
            return True
        if mapped_type in self.struct_name_map.values():
            return True
        if mapped_type in {
            declaration["name"] for declaration in self.local_aggregate_declarations
        }:
            return True
        if mapped_type in {
            info["type_name"] for info in self.wide_vector_types.values()
        }:
            return True
        if re.fullmatch(
            r"(?:[biu]?vec[2-4]|dvec[2-4]|f16vec[2-4]|"
            r"mat[2-4](?:x[2-4])?|dmat[2-4](?:x[2-4])?)",
            mapped_type,
        ):
            return True
        base_name, arguments = self.generic_type_parts(mapped_type)
        if base_name in {"vector", "matrix", "vec"} and arguments:
            return all(
                argument.strip().isdigit()
                or self.local_aggregate_mapped_type_is_concrete(argument, argument)
                for argument in arguments
            )
        normalized_source = self.normalized_metal_type(metal_type)
        return normalized_source in self.struct_name_map

    def resolve_local_aggregate_member_type(self, struct_node, member):
        source_type = str(getattr(member, "vtype", None) or "").strip()
        if not source_type:
            raise self.local_aggregate_error(
                struct_node,
                f"member '{getattr(member, 'name', '<unnamed>')}' has no type",
                source_location=getattr(member, "source_location", None),
            )
        source_type = self.substitute_template_type_text(source_type)
        source_type = self.resolve_local_type_aliases(source_type)
        source_type = self.resolve_type_alias(source_type)
        mapped_type = self.map_type(source_type)
        if not self.local_aggregate_mapped_type_is_concrete(source_type, mapped_type):
            dependencies = self.local_aggregate_type_dependencies(
                source_type, mapped_type
            )
            raise self.local_aggregate_error(
                struct_node,
                f"member '{member.name}' has an unresolved or unsupported type "
                f"'{source_type}'",
                source_location=getattr(member, "source_location", None),
                unresolved_dependencies=dependencies,
            )
        return source_type, mapped_type

    def substitute_local_integral_constant_text(self, text):
        result = str(text)
        for _iteration in range(len(self.local_integral_constant_bindings) + 1):
            previous = result
            for name in sorted(
                self.local_integral_constant_bindings,
                key=lambda value: (-len(value), value),
            ):
                result = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                    f"({self.local_integral_constant_bindings[name]})",
                    result,
                )
            if result == previous:
                break
        return result

    def resolve_local_aggregate_extent(self, struct_node, member, extent):
        if extent is None:
            raise self.local_aggregate_error(
                struct_node,
                f"member '{member.name}' has an unsized array extent",
                source_location=getattr(member, "source_location", None),
            )
        rendered = self.format_array_extent(extent)
        rendered = self.substitute_local_integral_constant_text(rendered)
        value = self.evaluate_value_template_constant_expression(rendered)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            dependencies = tuple(sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", rendered))))
            raise self.local_aggregate_error(
                struct_node,
                f"member '{member.name}' array extent '{rendered}' is not a "
                "positive integral constant",
                source_location=getattr(member, "source_location", None),
                unresolved_dependencies=dependencies,
            )
        return str(value)

    def register_local_integral_constant(self, declaration, rendered_value):
        name = getattr(declaration, "name", None)
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(declaration, "qualifiers", None) or []
        }
        declared_type = getattr(declaration, "vtype", None)
        # ``constexpr auto`` deduces an integral type exactly when the folded
        # initializer is an integer; floating initializers do not fold below.
        if (
            not name
            or "constexpr" not in qualifiers
            or not (
                str(declared_type or "").strip() == "auto"
                or self.is_integral_constexpr_type(declared_type)
            )
        ):
            return
        rendered_value = self.substitute_local_integral_constant_text(rendered_value)
        value = self.evaluate_value_template_constant_expression(
            rendered_value, constants={"true": 1, "false": 0}
        )
        if isinstance(value, int) and not isinstance(value, bool):
            self.local_integral_constant_bindings[name] = str(value)

    def register_local_aggregate(self, struct_node):
        specialization = (
            self.current_function_specialization
            or self.current_function_name
            or "<unknown>"
        )
        key = (id(struct_node), specialization)
        canonical_name = self.local_aggregate_declaration_keys.get(key)
        if canonical_name is not None:
            self.bind_local_aggregate_aliases(struct_node, canonical_name)
            return canonical_name

        self.validate_local_aggregate_shape(struct_node)
        unnamed_kind = getattr(struct_node, "unnamed_aggregate_kind", None)
        canonical_name = self.reserve_local_aggregate_name(
            unnamed_aggregate_display_name(unnamed_kind)
            if unnamed_kind
            else getattr(struct_node, "name", None)
            or getattr(struct_node, "typedef_tag", None)
        )
        self.bind_local_aggregate_aliases(struct_node, canonical_name)
        rendered_members = []
        member_types = {}
        member_names = {}
        used_member_names = set()
        member_layouts = []
        requires_native_storage_provenance = False
        for member in getattr(struct_node, "members", None) or []:
            if not isinstance(member, VariableNode) or not getattr(
                member, "name", None
            ):
                raise self.local_aggregate_error(
                    struct_node,
                    "only concrete data members can be hoisted",
                    source_location=getattr(member, "source_location", None),
                )
            if getattr(member, "default_value", None) is not None:
                raise self.local_aggregate_error(
                    struct_node,
                    f"member '{member.name}' has a default initializer",
                    source_location=getattr(member, "source_location", None),
                )
            if getattr(member, "qualifiers", None):
                qualifiers = ", ".join(
                    str(qualifier) for qualifier in member.qualifiers
                )
                raise self.local_aggregate_error(
                    struct_node,
                    f"member '{member.name}' has unsupported qualifiers "
                    f"'{qualifiers}'",
                    source_location=getattr(member, "source_location", None),
                )
            if getattr(member, "alignas", None) or getattr(member, "attributes", None):
                raise self.local_aggregate_error(
                    struct_node,
                    f"member '{member.name}' has alignment or attributes that "
                    "would be lost during hoisting",
                    source_location=getattr(member, "source_location", None),
                )
            source_type, mapped_type = self.resolve_local_aggregate_member_type(
                struct_node, member
            )
            extents = [
                self.resolve_local_aggregate_extent(struct_node, member, extent)
                for extent in getattr(member, "array_sizes", None) or []
            ]
            suffix = "".join(f"[{extent}]" for extent in extents)
            member_layouts.append(
                (
                    member,
                    f"{source_type}{suffix}",
                    self.metal_concrete_type_layout(f"{source_type}{suffix}"),
                )
            )
            requires_native_storage_provenance = requires_native_storage_provenance or (
                mapped_type not in self.crossgl_typedef_source_types()
                and self.crossgl_typedef_source_type(mapped_type) is not None
            )
            member_name_base = self.sanitize_identifier(member.name)
            member_name = member_name_base
            suffix_index = 2
            while member_name in used_member_names:
                member_name = f"{member_name_base}_{suffix_index}"
                suffix_index += 1
            used_member_names.add(member_name)
            rendered_members.append(f"{mapped_type}{suffix} {member_name};")
            member_types[member.name] = f"{source_type}{suffix}"
            member_names[member.name] = member_name

        native_storage_alignment = None
        if requires_native_storage_provenance:
            unresolved_layouts = [
                source_type
                for _member, source_type, layout in member_layouts
                if layout is None
            ]
            if unresolved_layouts:
                raise self.local_aggregate_error(
                    struct_node,
                    "native member storage alignment cannot be proven for "
                    + ", ".join(sorted(unresolved_layouts)),
                    unresolved_dependencies=tuple(sorted(unresolved_layouts)),
                )
            native_storage_alignment = max(
                layout[1] for _member, _source_type, layout in member_layouts
            )

        layout_contract = None
        if getattr(struct_node, "aggregate_kind", None) == "union":
            rendered_members, layout_contract = self.local_union_layout_contract(
                struct_node, rendered_members, member_layouts
            )

        self.local_aggregate_declaration_keys[key] = canonical_name
        self.local_aggregate_declarations.append(
            {
                "name": canonical_name,
                "members": tuple(rendered_members),
                "source_location": getattr(
                    struct_node, "declaration_source_location", None
                ),
                "native_storage_alignment": native_storage_alignment,
                "union_layout": layout_contract,
            }
        )
        self.struct_member_types[canonical_name] = member_types
        self.struct_member_name_maps[canonical_name] = member_names
        return canonical_name

    def local_union_layout_contract(
        self, struct_node, rendered_members, member_layouts
    ):
        """Members annotated with their overlapping storage, and the union's size and alignment.

        Hoisting keeps a block-scope union's storage semantics through the same layout metadata that a
        program-scope union carries, so every member layout must be known.
        """
        unresolved = sorted(
            source_type
            for _member, source_type, layout in member_layouts
            if layout is None
        )
        if unresolved:
            raise self.local_aggregate_error(
                struct_node,
                "union storage layout cannot be proven for " + ", ".join(unresolved),
                unresolved_dependencies=tuple(unresolved),
            )
        layouts = [layout for _member, _source_type, layout in member_layouts]
        annotated = tuple(
            f"@union_member_layout(0, {size}, {alignment}) {member}"
            for member, (size, alignment) in zip(rendered_members, layouts)
        )
        return annotated, union_layout(layouts)

    def render_struct_member_identifier(self, object_expression, member):
        try:
            owner = self.expression_mapped_type(object_expression)
        except (TypeError, ValueError):
            owner = None
        owner = str(owner or "").strip()
        while owner.endswith(("*", "&")):
            owner = owner[:-1].rstrip()
        mapped_name = self.struct_member_name_maps.get(owner, {}).get(member)
        return mapped_name or self.sanitize_identifier(member)

    def generate_local_aggregate_declarations(self, indent=0):
        if not self.local_aggregate_declarations:
            return ""
        pad = "    " * indent
        member_pad = "    " * (indent + 1)
        code = f"{pad}// Materialized function-local structs\n"
        for declaration in self.local_aggregate_declarations:
            native_storage_alignment = declaration.get("native_storage_alignment")
            if native_storage_alignment is not None:
                code += f"{pad}@metal_alignas({native_storage_alignment})\n"
            if declaration.get("union_layout") is not None:
                size, alignment = declaration["union_layout"]
                code += (
                    f"{pad}// Metal union {declaration['name']} retains overlapping "
                    "storage through layout metadata\n"
                    f"{pad}@union_layout({size}, {alignment}, little_endian, metal)\n"
                )
            code += f"{pad}struct {declaration['name']} {{\n"
            for member in declaration["members"]:
                code += f"{member_pad}{member}\n"
            code += f"{pad}}}\n\n"
        return code

    def register_local_type_alias(self, alias):
        """Register a function-body alias for subsequent lexical uses."""
        name = getattr(alias, "name", None)
        alias_type = getattr(alias, "alias_type", None)
        if not name or not alias_type:
            return
        self.unresolved_local_type_aliases.pop(name, None)
        alias_type = self.resolve_metal_decltype_type(alias_type) or alias_type
        alias_type = resolve_concrete_scoped_member_type(self, alias_type) or alias_type
        alias_qualifiers = list(getattr(alias, "qualifiers", None) or [])
        binding = self.scalar_alias_binding(alias_type)
        if binding is not None:
            alias_type, inherited = binding
            alias_qualifiers = list(dict.fromkeys([*alias_qualifiers, *inherited]))
        elif alias_type in self.local_type_alias_names:
            alias_qualifiers = list(
                dict.fromkeys(
                    [*alias_qualifiers, *self.type_alias_qualifiers.get(alias_type, ())]
                )
            )
        self.type_alias_pointee_qualifiers[name] = list(
            getattr(alias, "pointee_qualifiers", alias_qualifiers) or []
        )
        # Struct aliases remain uninlined, but scoped static-member references
        # need their concrete owner to resolve constants and backing globals.
        self.local_struct_type_aliases[name] = alias_type
        record_block_local_struct_alias(self, name)
        if (
            self.wide_vector_type_info(
                alias_type, getattr(alias, "source_location", None)
            )
            is not None
        ):
            self.type_aliases[name] = alias_type
            self.type_alias_qualifiers[name] = alias_qualifiers
            self.local_type_alias_names.add(name)
            return
        if alias_qualifiers and (
            self.pointer_element_type(alias_type) is not None
            or self.reference_element_type(alias_type) is not None
            or self.is_metal_resource_type(alias_type)
        ):
            self.type_aliases[name] = alias_type
            self.type_alias_qualifiers[name] = alias_qualifiers
            self.local_type_alias_names.add(name)
            return
        # Inline body-local aliases that resolve to scalar/vector primitives or
        # to a concrete struct emitted in this module. Both forms would otherwise
        # become dangling CrossGL type names and default to float downstream.
        # Unresolved user-template aliases remain untouched until their template
        # arguments have been materialized by the Metal frontend.
        mapped_alias_type = self.map_type(alias_type)
        local_aggregate_type_names = {
            declaration["name"] for declaration in self.local_aggregate_declarations
        }
        concrete_struct_alias = (
            mapped_alias_type in self.struct_name_map.values()
            or mapped_alias_type in local_aggregate_type_names
        )
        if (
            set(alias_qualifiers) - {"const", "volatile"}
            or getattr(alias, "array_sizes", None)
            or getattr(alias, "declarator_type_suffix", "")
            or (
                self.crossgl_typedef_source_type(mapped_alias_type) is None
                and not concrete_struct_alias
            )
        ):
            if not (
                set(alias_qualifiers) - {"const", "volatile"}
                or getattr(alias, "array_sizes", None)
                or getattr(alias, "declarator_type_suffix", "")
            ):
                bind_generic_local_alias(self, alias, alias_type, alias_qualifiers)
            return
        # Bind primitive dependencies now, before later declarations shadow them.
        self.type_aliases[name] = (
            self.resolve_type_alias(alias_type)
            if self.crossgl_typedef_source_type(mapped_alias_type) is not None
            else alias_type
        )
        self.type_alias_qualifiers[name] = alias_qualifiers
        self.local_type_alias_names.add(name)

    def resolve_local_type_aliases(self, metal_type):
        """Resolve concrete body-local aliases inside a Metal type expression."""
        if metal_type is None:
            return metal_type

        original = str(metal_type).strip()
        base = original
        suffix = ""
        while base.endswith("*") or base.endswith("&"):
            suffix = base[-1] + suffix
            base = base[:-1].strip()

        if base in self.local_type_alias_names:
            return f"{self.resolve_type_alias(base)}{suffix}"

        generic_base, generic_args = self.generic_type_parts(base)
        if not generic_base or not generic_args:
            return original
        resolved_args = [self.resolve_local_type_aliases(arg) for arg in generic_args]
        if resolved_args == generic_args:
            return original
        return f"{generic_base}<{', '.join(resolved_args)}>{suffix}"

    def register_metal_enum_arithmetic_contract(self, enum):
        if getattr(enum, "is_scoped", False):
            return
        arithmetic_type = getattr(enum, "underlying_type", None) or "int"
        enum_name = getattr(enum, "name", None)
        if enum_name:
            self.metal_enum_arithmetic_types[self.normalized_metal_type(enum_name)] = (
                arithmetic_type
            )
        for member_name, _member_value in getattr(enum, "members", []) or []:
            self.metal_enum_member_types[str(member_name)] = arithmetic_type

    def generate_local_enum(self, enum, indent, is_main):
        enum_name = enum.name or "MetalAnonymousEnum"
        self.register_metal_enum_arithmetic_contract(enum)
        code = f"enum {enum_name} {{\n"
        for member_name, member_value in enum.members:
            code += "    " * (indent + 1) + member_name
            if member_value is not None:
                code += f" = {self.generate_expression(member_value, is_main)}"
            code += ",\n"
        code += "    " * indent + "};\n"
        return code
