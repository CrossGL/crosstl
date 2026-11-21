"""Top-level functions, constant structs, struct member types, struct static constants and nested structs."""

import re

from ....translator.codegen.array_utils import evaluate_literal_int_expression
from ....translator.lexer import KEYWORDS as CROSSGL_KEYWORDS
from ..constant_constructor_folding import constant_initialization
from ..MetalAst import (
    ConstantBufferNode,
    StructNode,
)
from .support import (
    MetalStaticConstantResolutionError,
    MetalTemplateArgumentResolutionError,
)


class MetalStructDeclarationMixin:
    """Members of ``MetalToCrossGLConverter``: top-level functions, constant structs, struct member types, struct static constants and nested structs."""

    def generate_top_level_function(
        self,
        function,
        *,
        template_value_bindings=None,
        template_type_bindings=None,
        output_name=None,
        specialization_key=None,
    ):
        qualifier = getattr(function, "qualifier", None)
        if qualifier in {"vertex", "fragment", "kernel", "mesh", "object"} and (
            self.unbound_template_parameter_names(
                function, template_value_bindings, template_type_bindings
            )
        ):
            # A stage function template is only an entry point through an
            # instantiation, and every instantiation the source names has
            # already become a concrete function. One that is left has no
            # entry point in the Metal library, so it emits nothing.
            return ""
        if qualifier == "vertex":
            return (
                "    // Vertex Shader\n"
                "    vertex {\n"
                + self.generate_function(
                    function,
                    stage_entry=function.name != "main",
                    template_value_bindings=template_value_bindings,
                    template_type_bindings=template_type_bindings,
                    output_name=output_name,
                    specialization_key=specialization_key,
                )
                + "    }\n\n"
            )
        if qualifier == "fragment":
            return (
                "    // Fragment Shader\n"
                "    fragment {\n"
                + self.generate_fragment_execution_layouts(function)
                + self.generate_function(
                    function,
                    stage_entry=function.name != "main",
                    template_value_bindings=template_value_bindings,
                    template_type_bindings=template_type_bindings,
                    output_name=output_name,
                    specialization_key=specialization_key,
                )
                + "    }\n\n"
            )
        if qualifier == "kernel":
            return (
                "    // Compute Shader\n"
                "    compute {\n"
                + self.kernel_workgroup_size_attribute()
                + self.generate_function(
                    function,
                    stage_entry=self.should_emit_kernel_stage_entry(function),
                    template_value_bindings=template_value_bindings,
                    template_type_bindings=template_type_bindings,
                    output_name=output_name,
                    specialization_key=specialization_key,
                )
                + "    }\n\n"
            )
        if qualifier == "mesh":
            return self.generate_metal_mesh_stage(
                function,
                template_value_bindings=template_value_bindings,
                template_type_bindings=template_type_bindings,
                output_name=output_name,
                specialization_key=specialization_key,
            )
        if qualifier == "object":
            return self.generate_metal_object_stage(
                function,
                template_value_bindings=template_value_bindings,
                template_type_bindings=template_type_bindings,
                output_name=output_name,
                specialization_key=specialization_key,
            )
        ray_attribute = next(
            (
                attribute.name
                for attribute in getattr(function, "attributes", None) or []
                if getattr(attribute, "name", None)
                in self.rt_qualifiers - {"mesh", "object"}
            ),
            None,
        )
        if qualifier is None and ray_attribute is not None:
            # [[intersection(bounding_box, ...)]] keeps the stage as an attribute rather than a qualifier.
            qualifier = ray_attribute
        if qualifier == "intersection" and ray_attribute == "intersection":
            # [[intersection(primitive_type, tags...)]]; MSL has no intersection function qualifier keyword.
            return self.generate_metal_intersection_stage(
                function,
                template_value_bindings=template_value_bindings,
                template_type_bindings=template_type_bindings,
                output_name=output_name,
                specialization_key=specialization_key,
            )
        if qualifier in self.rt_qualifiers:
            # Printed as a plain helper, the function would lose the qualifier that lets the host place it in an
            # intersection function table, and the targets would get no ray-tracing stage.
            return self.refuse_import(
                f"Metal {qualifier} function '{function.name}' has no CrossGL import; its ray-tracing "
                "parameters and results are not yet mapped to a CrossGL ray stage"
            )
        return self.generate_function(
            function,
            template_value_bindings=template_value_bindings,
            template_type_bindings=template_type_bindings,
            output_name=output_name,
            specialization_key=specialization_key,
        )

    def process_constant_struct(self, node):
        constants = (
            getattr(node, "constant", []) or getattr(node, "constants", []) or []
        )
        structs = getattr(node, "structs", []) or getattr(node, "struct", []) or []
        for constant in constants:
            if isinstance(constant, ConstantBufferNode):
                self.constant_struct_name.extend(
                    struct.name for struct in structs if struct.name == constant.name
                )

    def collect_struct_member_types(self, structs):
        member_types = {}
        self.resolved_struct_member_types = {}
        self.resolved_struct_member_qualifiers = {}
        for struct_node in structs or []:
            struct_name = getattr(struct_node, "name", None)
            if not struct_name:
                continue
            members = {}
            for member in getattr(struct_node, "members", []) or []:
                member_name = getattr(member, "name", None)
                member_type = self.resolve_struct_member_declaration_type(
                    struct_node, member
                )
                if member_name and member_type:
                    members[member_name] = member_type
            member_types[struct_name] = members
        return member_types

    def resolve_struct_member_declaration_type(self, struct_node, member):
        declared_type = self.resolve_struct_member_base_type(struct_node, member)
        if declared_type is None:
            return None
        self.resolved_struct_member_types[id(member)] = declared_type
        array_suffix = "".join(
            "[]" if size is None else f"[{self.format_array_extent(size)}]"
            for size in getattr(member, "array_sizes", []) or []
        )
        return f"{declared_type}{array_suffix}"

    def resolve_struct_member_base_type(self, struct_node, member):
        declared_type = str(getattr(member, "vtype", None) or "").strip()
        if not declared_type:
            return None

        candidate = re.sub(r"^typename\s+", "", declared_type).strip()
        suffix = ""
        while candidate.endswith(("*", "&")):
            suffix = candidate[-1] + suffix
            candidate = candidate[:-1].strip()

        alias_names = {
            getattr(alias, "name", None)
            for alias in getattr(struct_node, "type_aliases", None) or []
            if getattr(alias, "name", None)
        }
        if candidate in alias_names:
            if getattr(struct_node, "template_parameters", None):
                return declared_type
            owner = self.normalize_qualified_type_name(
                getattr(struct_node, "qualified_name", None) or struct_node.name
            )
            previous_context = self.current_type_resolution_context
            self.current_type_resolution_context = member
            try:
                resolved = self.resolve_dependent_alias_contract(
                    owner, candidate, required=True
                )
            finally:
                self.current_type_resolution_context = previous_context
            if resolved is None:
                raise self.struct_member_alias_error(
                    owner,
                    candidate,
                    "the owning declaration does not define a concrete alias",
                    [struct_node],
                )
            target, qualifiers = resolved
            self.resolved_struct_member_qualifiers[id(member)] = qualifiers
            declared_type = f"{target}{suffix}"

        return declared_type

    def collect_struct_static_constants(self, structs):
        members = {}
        template_parameters = {}
        owner_candidates = {}
        for struct_node in structs or []:
            struct_name = getattr(struct_node, "name", None)
            if not struct_name:
                continue
            mapped_name = self.map_struct_name(struct_name)
            for owner_name in self.struct_static_constant_declaration_names(
                struct_node
            ):
                candidates = owner_candidates.setdefault(owner_name, [])
                if not any(candidate is struct_node for candidate in candidates):
                    candidates.append(struct_node)
            template_parameters[mapped_name] = {
                name: kind
                for kind, name in (
                    getattr(struct_node, "template_parameters", None) or []
                )
                if name
            }
            for member in getattr(struct_node, "members", []) or []:
                if not self.is_compile_time_static_member(member):
                    continue
                member_name = getattr(member, "name", None)
                if member_name:
                    members[(mapped_name, member_name)] = member

        self.struct_static_constants = {}
        self.struct_static_constant_members = members
        self.struct_static_constant_owner_candidates = owner_candidates
        self.equivalent_struct_static_constants = {}
        self.struct_static_constant_resolution_stack = []
        self.struct_static_constexpr_member_keys = set()
        self.struct_template_parameters = template_parameters
        for key in members:
            self.resolve_struct_static_constant(key)

    def struct_static_constant_declaration_names(self, struct_node):
        raw_name = self.normalize_qualified_type_name(
            getattr(struct_node, "name", None)
        )
        qualified_name = self.normalize_qualified_type_name(
            getattr(struct_node, "qualified_name", None)
        )
        mapped_name = self.normalize_qualified_type_name(self.map_struct_name(raw_name))
        return {name for name in (raw_name, qualified_name, mapped_name) if name}

    def struct_static_constant_owner_identity(self, struct_node):
        return self.normalize_qualified_type_name(
            getattr(struct_node, "qualified_name", None)
            or getattr(struct_node, "name", None)
        )

    def struct_compile_time_static_members(self, struct_node):
        members = {}
        duplicates = set()
        for member in getattr(struct_node, "members", None) or []:
            if not self.is_compile_time_static_member(member):
                continue
            member_name = getattr(member, "name", None)
            if not member_name:
                continue
            if member_name in members:
                duplicates.add(member_name)
            members[member_name] = member
        return members, duplicates

    def resolve_struct_static_constant_candidate(self, struct_node, member_name):
        candidate_members, duplicate_members = self.struct_compile_time_static_members(
            struct_node
        )
        if member_name in duplicate_members:
            return None, None, "the declaration repeats the compile-time member"
        if member_name not in candidate_members:
            return None, None, "the declaration has no compile-time static member"

        values = {}
        resolved = {}
        pending = dict(candidate_members)
        owner_names = self.struct_static_constant_declaration_names(struct_node)
        while pending:
            progress = False
            for name, member in list(pending.items()):
                member_type = self.normalized_metal_type(
                    self.resolve_type_alias(getattr(member, "vtype", None))
                )
                if member_type not in self.constexpr_integral_type_names():
                    continue
                value = evaluate_literal_int_expression(
                    getattr(member, "default_value", None),
                    values,
                )
                if value is None:
                    continue
                value = int(value)
                values[name] = value
                for owner_name in owner_names:
                    values[f"{owner_name}::{name}"] = value
                resolved[name] = (
                    (member_type, value),
                    self.render_static_integral_value(member, value),
                )
                del pending[name]
                progress = True
            if not progress:
                break

        identity_and_value = resolved.get(member_name)
        if identity_and_value is None:
            return (
                None,
                None,
                "the member value is not a resolved integral constant",
            )
        identity, rendered = identity_and_value
        return identity, rendered, None

    def render_equivalent_struct_static_constant(self, owner, member_name):
        cache_key = (owner, member_name)
        if cache_key in self.equivalent_struct_static_constants:
            return self.render_static_constant_value(
                self.equivalent_struct_static_constants[cache_key]
            )

        candidates = self.struct_static_constant_owner_candidates.get(owner, [])
        owner_identities = {
            self.struct_static_constant_owner_identity(candidate)
            for candidate in candidates
        }
        if len(candidates) < 2 or len(owner_identities) != 1:
            raise MetalStaticConstantResolutionError(
                owner,
                member_name,
                "multiple visible struct declarations match the qualified owner",
            )

        resolved_candidates = []
        for candidate in candidates:
            identity, rendered, reason = self.resolve_struct_static_constant_candidate(
                candidate,
                member_name,
            )
            if reason is not None:
                raise MetalStaticConstantResolutionError(
                    owner,
                    member_name,
                    "multiple visible struct declarations remain ambiguous: " + reason,
                )
            resolved_candidates.append((identity, rendered))

        identities = {identity for identity, _rendered in resolved_candidates}
        if len(identities) != 1:
            raise MetalStaticConstantResolutionError(
                owner,
                member_name,
                "multiple visible struct declarations define conflicting "
                "compile-time values",
            )

        rendered = resolved_candidates[0][1]
        self.equivalent_struct_static_constants[cache_key] = rendered
        mapped_owner = self.map_struct_name(candidates[0].name)
        self.propagate_struct_static_constexpr_dependency((mapped_owner, member_name))
        return self.render_static_constant_value(rendered)

    def is_compile_time_static_member(self, member):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(member, "qualifiers", []) or []
        }
        return "constexpr" in qualifiers or (
            "static" in qualifiers
            and bool(qualifiers.intersection({"const", "constant"}))
        )

    def resolve_struct_static_constant(self, key):
        if key in self.struct_static_constants:
            return self.struct_static_constants[key]
        member = self.struct_static_constant_members.get(key)
        if member is None:
            return None
        default_value = getattr(member, "default_value", None)
        if default_value is None:
            return None
        if key in self.struct_static_constant_resolution_stack:
            cycle_start = self.struct_static_constant_resolution_stack.index(key)
            cycle = self.struct_static_constant_resolution_stack[cycle_start:] + [key]
            path = " -> ".join(f"{owner}::{name}" for owner, name in cycle)
            owner, member_name = key
            raise MetalStaticConstantResolutionError(
                owner,
                member_name,
                f"the initializer dependency chain is cyclic ({path})",
            )

        owner, _member_name = key
        self.struct_static_constant_resolution_stack.append(key)
        previous_owner = self.current_struct_static_constant_owner
        self.current_struct_static_constant_owner = owner
        try:
            local_integer_values = self.struct_static_integer_values(owner)
            integer_value = evaluate_literal_int_expression(
                default_value,
                local_integer_values,
            )
            if integer_value is not None:
                rendered = self.render_static_integral_value(member, integer_value)
            else:
                with constant_initialization(self):
                    rendered = self.generate_expression(default_value, False)
                folded_value = self.evaluate_value_template_constant_expression(
                    rendered
                )
                if folded_value is not None:
                    rendered = self.render_static_integral_value(member, folded_value)
        except MetalTemplateArgumentResolutionError as error:
            raise self.contextualize_static_template_resolution_error(
                error,
                key,
                member,
            ) from error
        finally:
            self.current_struct_static_constant_owner = previous_owner
            self.struct_static_constant_resolution_stack.pop()
        if not rendered:
            return None
        self.struct_static_constants[key] = rendered
        return rendered

    def render_static_integral_value(self, member, value):
        member_type = self.normalized_metal_type(getattr(member, "vtype", None))
        if member_type == "bool":
            return "true" if value else "false"
        return str(value)

    def contextualize_static_template_resolution_error(self, error, key, member):
        owner, member_name = key
        return MetalTemplateArgumentResolutionError(
            error.function_name,
            error.parameter_name,
            error.argument_expression,
            error.selected_call,
            error.reason,
            error.argument_kind,
            getattr(member, "source_location", None) or error.source_location,
            owner=owner,
            member=member_name,
            requested_specialization=(
                getattr(error, "requested_specialization", None) or error.selected_call
            ),
        )

    def struct_static_integer_values(self, owner):
        values = {}
        for (
            candidate_owner,
            member_name,
        ), value in self.struct_static_constants.items():
            if candidate_owner != owner or not re.fullmatch(r"[-+]?\d+", value):
                continue
            values[member_name] = int(value)
            values[f"{owner}::{member_name}"] = int(value)
        return values

    def flatten_concrete_nested_structs(self, structs):
        flattened = []
        self.nested_struct_types_by_owner = {}

        def append_struct(struct_node, owner_name=None):
            raw_name = getattr(struct_node, "name", None)
            if not raw_name:
                return
            original_name = getattr(struct_node, "nested_original_name", raw_name)
            if owner_name is not None:
                qualified_name = f"{owner_name}::{original_name}"
                struct_node.nested_original_name = original_name
                struct_node.nested_owner_name = owner_name
                struct_node.name = qualified_name
                struct_node.qualified_name = qualified_name
                self.nested_struct_types_by_owner.setdefault(owner_name, {})[
                    original_name
                ] = qualified_name
                raw_name = qualified_name
                for constructor in getattr(struct_node, "constructors", ()) or ():
                    constructor.owner_name = qualified_name
                    constructor.owner_qualified_name = qualified_name
                for alias in getattr(struct_node, "type_aliases", ()) or ():
                    alias.owner_name = qualified_name
                    alias.owner_qualified_name = qualified_name
            for nested in getattr(struct_node, "nested_structs", ()) or ():
                append_struct(nested, raw_name)
            flattened.append(struct_node)

        for struct_node in structs or ():
            if isinstance(struct_node, StructNode):
                append_struct(struct_node)
        return flattened

    def current_nested_struct_owner(self):
        context = self.current_type_resolution_context
        if isinstance(context, StructNode):
            context_name = getattr(context, "name", None)
            if context_name in self.nested_struct_types_by_owner:
                return context_name

        function = self.current_function
        if getattr(function, "is_metal_constructor_factory", False):
            owner = getattr(function, "constructor_owner", None)
            if owner in self.nested_struct_types_by_owner:
                return owner
        parameters = list(getattr(function, "params", ()) or ())
        if parameters and getattr(parameters[0], "name", None) == "self":
            receiver_type = self.metal_declaration_expression_type(parameters[0])
            owner = self.reference_element_type(receiver_type) or receiver_type
            owner = self.normalized_metal_type(self.resolve_type_alias(owner))
            if owner in self.nested_struct_types_by_owner:
                return owner

        function_name = str(getattr(function, "name", ""))
        for owner in sorted(self.nested_struct_types_by_owner, key=len, reverse=True):
            mapped_owner = self.map_struct_name(owner)
            if function_name.startswith(f"{owner}__") or function_name.startswith(
                f"{mapped_owner}__"
            ):
                return owner
        return None

    def resolve_contextual_nested_struct_type(self, type_name):
        candidate = self.normalize_qualified_type_name(type_name).lstrip(":")
        owner = self.current_nested_struct_owner()
        if owner is not None:
            nested_type = self.nested_struct_types_by_owner.get(owner, {}).get(
                candidate
            )
            if nested_type is not None:
                return nested_type
        if candidate in self.struct_name_map:
            return candidate
        return candidate

    def build_struct_name_map(self, structs):
        mapped_names = {}
        used_names = set()
        self.ambiguous_struct_names = set()
        named = [
            struct_node
            for struct_node in structs or []
            if isinstance(struct_node, StructNode)
            and getattr(struct_node, "name", None)
        ]
        # The CrossGL parser reads a structure name such as ``let`` or ``vec2``, which a Metal program may declare,
        # as its keyword, so such a structure takes a suffixed name that no other structure keeps.
        kept_names = {
            self.sanitize_identifier(struct_node.name) for struct_node in named
        } - set(CROSSGL_KEYWORDS)
        for struct_node in named:
            raw_name = struct_node.name
            if raw_name in mapped_names:
                self.ambiguous_struct_names.add(raw_name)
            base_name = self.sanitize_identifier(raw_name)
            taken = used_names
            if base_name in CROSSGL_KEYWORDS:
                base_name = f"{base_name}_"
                taken = used_names | kept_names
            candidate = base_name
            suffix = 2
            while candidate in taken:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(candidate)
            mapped_names[raw_name] = candidate
        return mapped_names

    def format_generic_prefix(self, node, bound_parameter_names=None):
        defaults = getattr(node, "template_parameter_defaults", {}) or {}
        template_parameters = getattr(node, "template_parameters", None)
        bound_parameter_names = set(bound_parameter_names or ())
        if template_parameters:
            generics = []
            for kind, name in template_parameters:
                if not name or name in bound_parameter_names:
                    continue
                rendered = self.sanitize_identifier(name)
                default_type = defaults.get(name)
                if default_type and kind in {"typename", "class"}:
                    rendered = f"{rendered} = {self.map_type(default_type)}"
                generics.append(rendered)
        else:
            generics = [
                self.sanitize_identifier(name)
                for name in getattr(node, "generics", []) or []
                if name
            ]
        if not generics:
            return ""
        return f"generic<{', '.join(generics)}> "

    def map_struct_name(self, name):
        if not name:
            return name
        return self.struct_name_map.get(name, self.sanitize_identifier(name))
