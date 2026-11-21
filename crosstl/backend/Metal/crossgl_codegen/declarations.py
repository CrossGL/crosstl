"""Texture usage, global integral constants, array extents, variable, parameter and global declarations, and address-space qualifiers."""

import re

from ....translator.codegen.array_utils import (
    evaluate_literal_int_expression,
    split_array_type_suffix,
)
from ....translator.host_workgroup_size import host_workgroup_size_attribute_text
from ..array_designators import (
    has_array_designators,
    positional_array_initializer,
)
from ..comparison_samplers import collect_comparison_sampler_ids
from ..constant_constructor_folding import constant_initialization
from ..MetalAst import (
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    FunctionNode,
    InitializerListNode,
    StructNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)
from ..MetalLexer import MetalLexer
from ..MetalParser import MetalParser
from ..preprocessor import DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT
from .support import (
    MetalScalarAliasResolutionError,
    MetalStaticConstantResolutionError,
)


class MetalDeclarationMixin:
    """Members of ``MetalToCrossGLConverter``: texture usage, global integral constants, array extents, variable, parameter and global declarations, and address-space qualifiers."""

    def collect_storage_texture_declaration_ids(self, root):
        storage_ids = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(
                node, VariableNode
            ) and self.is_access_qualified_storage_texture_type(node.vtype):
                storage_ids.add(id(node))
            for child in self.iter_ast_children(node):
                visit(child)

        visit(root)
        return storage_ids

    def collect_declared_identifier_names(self, root):
        names = set()
        seen = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if not isinstance(node, (dict, list, tuple, set)):
                node_id = id(node)
                if node_id in seen:
                    return
                seen.add(node_id)
            if isinstance(
                node, (VariableNode, FunctionNode, StructNode, TypeAliasNode)
            ):
                name = getattr(node, "name", None)
                if name:
                    names.add(self.sanitize_identifier(name))
            for child in self.iter_ast_children(node):
                visit(child)

        visit(root)
        return names

    def prepare_texture_usage(self, ast):
        self.global_variable_types = {}
        self.current_variable_types = {}
        self.global_variable_type_qualifiers = {}
        self.current_variable_type_qualifiers = {}
        self.metal_source_overload_groups = {}
        self.metal_source_overload_output_names = {}
        self.global_storage_texture_names = set()
        self.current_storage_texture_names = set()
        self.global_structured_buffer_names = set()
        self.current_structured_buffer_names = set()
        self.current_stage_entry_resource_parameter_ids = set()
        self.global_sampler_names = set()
        self.user_function_names = set()
        self.user_function_overloads_by_name = {}
        self.out_of_line_call_operator_replacements = {}
        self.suppressed_out_of_line_call_operator_ids = set()
        self.identifier_maps = [{}]
        self.used_identifier_names = [set()]
        self.template_name_spellings = {}
        self.storage_texture_declaration_ids = (
            self.collect_storage_texture_declaration_ids(ast)
        )
        self.comparison_sampler_ids = collect_comparison_sampler_ids(ast)
        self.struct_member_types = {}
        self.resolved_struct_member_types = {}
        self.resolved_struct_member_qualifiers = {}
        self.struct_declarations = {}
        self.struct_name_map = {}
        self.ambiguous_struct_names = set()
        self.nested_struct_types_by_owner = {}
        self.struct_static_constants = {}
        self.struct_static_constant_members = {}
        self.struct_static_constant_owner_candidates = {}
        self.equivalent_struct_static_constants = {}
        self.struct_static_constant_resolution_stack = []
        self.struct_static_constexpr_member_keys = set()
        self.current_struct_static_constant_owner = None
        self.struct_template_parameters = {}
        self.local_type_alias_names = set()
        self.unresolved_local_type_aliases = {}
        self.local_struct_type_aliases = {}
        self.local_integral_constant_bindings = {}
        self.global_integral_constant_bindings = {}
        self.local_aggregate_declarations = []
        self.local_aggregate_declaration_keys = {}
        self.local_aggregate_reserved_names = set()
        self.integral_constant_bindings = []
        self.template_value_bindings = []
        self.template_type_bindings = []
        self.template_binding_shadow_scopes = []
        self.value_template_functions = {}
        self.constexpr_value_template_functions = {}
        self.constexpr_helper_values = {}
        self.constexpr_helper_resolution_stack = []
        self.default_value_template_bindings = {}
        self.value_template_specializations = {}
        self.pending_value_template_specializations = []
        self.value_template_specialization_dependencies = {}
        self.current_function_specialization_key = None
        self.max_template_specializations = (
            DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT
        )
        self.template_specialization_limit_source = "max_template_specializations"
        self.materialized_template_specialization_count = 0
        self.preserve_unmaterialized_template_calls = False
        self.suppressed_value_template_function_ids = set()
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_function = None
        self.current_function_specialization = None
        self.current_function_materialization_bindings = {}
        self.materialized_constexpr_expression_contexts = []
        self.constructor_contracts_by_owner = {}
        self.constructor_factory_names = {}
        self.pending_constructor_factories = []
        self.current_constructor_scope_index = None

    def prepare_global_integral_constants(self, ast):
        """Collect uniquely visible program-scope integral constexpr values.

        Constructor-bearing local arrays must be expanded into explicit element
        assignments, which requires a finite loop bound.  Metal accepts a proven
        program-scope ``constexpr`` identifier as that bound (MLX reduction uses
        ``static constant constexpr const uint8_t simd_size = 32``), so retain
        those exact values without treating mutable or merely constant-address-
        space data as compile-time expressions.
        """
        self.global_integral_constant_bindings = {}
        candidates = []
        for entry in getattr(ast, "global_variables", []) or getattr(
            ast, "global_vars", []
        ):
            if (
                not isinstance(entry, AssignmentNode)
                or getattr(entry, "operator", None) != "="
                or not isinstance(getattr(entry, "left", None), VariableNode)
            ):
                continue
            declaration = entry.left
            name = getattr(declaration, "name", None)
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(declaration, "qualifiers", []) or []
            }
            if (
                not name
                or "constexpr" not in qualifiers
                or not self.is_integral_constexpr_type(
                    getattr(declaration, "vtype", None)
                )
            ):
                continue
            qualified_name = re.sub(
                r"\s*::\s*",
                "::",
                str(getattr(declaration, "qualified_name", None) or name),
            ).lstrip(":")
            candidates.append((declaration, entry.right, name, qualified_name))

        unqualified_counts = {}
        qualified_counts = {}
        for _declaration, _initializer, name, qualified_name in candidates:
            unqualified_counts[name] = unqualified_counts.get(name, 0) + 1
            qualified_counts[qualified_name] = (
                qualified_counts.get(qualified_name, 0) + 1
            )

        values = {}
        for _declaration, initializer, name, qualified_name in candidates:
            value = evaluate_literal_int_expression(initializer, values)
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            identities = []
            if qualified_counts[qualified_name] == 1:
                identities.append(qualified_name)
            if unqualified_counts[name] == 1:
                identities.append(name)
            for identity in dict.fromkeys(identities):
                values[identity] = value
        self.global_integral_constant_bindings = values

    def visible_global_integral_constant_bindings(self):
        """Return global constexpr values not hidden by a lexical declaration."""
        visible = {}
        local_scopes = self.identifier_maps[1:]
        for name, value in self.global_integral_constant_bindings.items():
            if "::" not in name and any(name in scope for scope in local_scopes):
                continue
            visible[name] = value
        return visible

    def evaluate_concrete_array_extent(self, expression):
        rendered = self.substitute_local_integral_constant_text(expression)
        try:
            lexer = MetalLexer(rendered, preprocess=False)
            parser = MetalParser(lexer.tokenize())
            node = parser.parse_expression()
            if parser.current_token[0] != "EOF":
                return None
        except (SyntaxError, ValueError, TypeError):
            return None
        return evaluate_literal_int_expression(
            node,
            self.visible_global_integral_constant_bindings(),
        )

    def effective_metal_variable_type(self, var):
        metal_type = self.resolved_struct_member_types.get(
            id(var), getattr(var, "vtype", None)
        )
        if not self.is_metal_auto_declaration_type(metal_type):
            return metal_type
        inferred_type = self.current_variable_types.get(
            getattr(var, "name", None),
            self.global_variable_types.get(getattr(var, "name", None)),
        )
        if inferred_type is None or self.is_metal_auto_declaration_type(inferred_type):
            return metal_type
        return inferred_type

    def format_array_suffix(self, var, include_declarator_arrays=True):
        array_type = self.metal_array_type_parts(
            self.effective_metal_variable_type(var)
        )
        if self.zero_extent_metal_array_type_parts(
            self.effective_metal_variable_type(var)
        ):
            array_type = None
        suffix = f"[{self.format_array_extent(array_type[1])}]" if array_type else ""
        if not include_declarator_arrays:
            return suffix
        # ``array<T, N> name[M]`` is M arrays of N: the declarator's dimensions are the outer ones.
        return self.format_declarator_array_suffix(var) + suffix

    def format_declarator_array_suffix(self, var):
        if not hasattr(var, "array_sizes") or not var.array_sizes:
            return ""
        suffix = ""
        for size in var.array_sizes:
            if size is None:
                suffix += "[]"
            else:
                suffix += f"[{self.format_array_extent(size)}]"
        return suffix

    def format_array_extent(self, size):
        if not isinstance(size, str):
            return self.generate_expression(size, False)

        extent = self.substitute_template_value_text(size).strip()
        if self.is_scoped_identifier(extent.lstrip(":")):
            extent = extent.lstrip(":")
            return self.sanitize_identifier(extent)
        return extent

    def use_name_array_suffix(self, mapped_type, var):
        if not getattr(var, "array_sizes", None):
            return False
        return str(mapped_type).rstrip().endswith(("*", "&"))

    def variable_has_array_initializer_shape(self, var):
        return bool(getattr(var, "array_sizes", None)) or bool(
            self.metal_array_type_parts(self.effective_metal_variable_type(var))
        )

    def array_initializer_without_designators(self, declaration, initializer):
        """``initializer`` of ``declaration`` with its C array designators resolved to positions.

        See :mod:`array_designators`; any other initializer is returned unchanged.
        """
        if not isinstance(declaration, VariableNode) or not has_array_designators(
            initializer
        ):
            return initializer
        element_type = getattr(declaration, "vtype", None)
        extents = list(getattr(declaration, "array_sizes", None) or [])
        if not extents:
            parts = self.metal_array_type_parts(self.resolve_type_alias(element_type))
            while parts is not None:
                element_type, extent = parts
                extents.append(extent)
                parts = self.metal_array_type_parts(
                    self.resolve_type_alias(element_type)
                )
        if not extents:
            return self.refuse_import(
                f"Metal designated array initializer of '{getattr(declaration, 'name', '')}', "
                "which is not an array",
                InitializerListNode([]),
            )
        return positional_array_initializer(self, element_type, extents, initializer)

    def map_variable_type(self, var):
        mapped_type = self.map_variable_type_by_declaration(var)
        if mapped_type == "sampler" and id(var) in self.comparison_sampler_ids:
            return "comparison_sampler"
        return mapped_type

    def map_variable_type_by_declaration(self, var):
        raw_type = self.effective_metal_variable_type(var)
        structured_buffer_type = self.structured_buffer_pointer_type(var)
        if structured_buffer_type:
            return structured_buffer_type
        array_type = self.metal_array_type_parts(raw_type)
        if self.zero_extent_metal_array_type_parts(raw_type):
            array_type = None
        type_to_map = array_type[0] if array_type else raw_type
        if (
            id(var) in self.storage_texture_declaration_ids
            or getattr(var, "name", None) in self.current_storage_texture_names
            or getattr(var, "name", None) in self.global_storage_texture_names
        ):
            storage_type = self.map_storage_texture_type(type_to_map)
            if storage_type:
                return storage_type
        resolved_type = self.resolve_type_alias(type_to_map)
        resolved_pointer = self.pointer_element_type(resolved_type)
        resolved_reference = self.reference_element_type(resolved_type)
        if resolved_type != type_to_map and (
            (
                resolved_pointer is not None
                and self.pointer_element_type(type_to_map) is None
            )
            or (
                resolved_reference is not None
                and self.reference_element_type(type_to_map) is None
            )
            or self.is_metal_resource_type(resolved_type)
        ):
            # Resolve aliases that introduce an ABI-relevant type category, but
            # preserve logical pointee aliases on an already-explicit pointer or
            # reference (for example ``threadgroup bfloat16_t*``). Erasing only
            # the pointee spelling would make it disagree with an adjacent
            # ``threadgroup bfloat16_t[N]`` backing declaration.
            return self.map_type(resolved_type)
        if resolved_type != type_to_map and type_to_map in getattr(
            self, "local_type_alias_names", set()
        ):
            # Body-local aliases (e.g. `using OutType = conditional_t<...>;`) have
            # no emitted CrossGL typedef, so map the resolved concrete type inline
            # rather than the dangling alias name (which would default to float).
            return self.map_type(resolved_type)
        return self.map_type(type_to_map)

    def map_declared_variable_type(self, var):
        return self.map_variable_type(var)

    def address_space_qualifier_prefix(self, var):
        if self.structured_buffer_pointer_type(var):
            return ""

        qualifiers = self.effective_declaration_qualifiers(var)
        address_spaces = []
        for qualifier in (
            "threadgroup_imageblock",
            "threadgroup",
            "thread",
            "device",
            "constant",
        ):
            if qualifier in qualifiers and qualifier not in address_spaces:
                address_spaces.append(qualifier)
        return f"{' '.join(address_spaces)} " if address_spaces else ""

    def resolved_declaration_qualifiers(self, var):
        """Return direct qualifiers plus those carried by a resolved type alias."""
        qualifiers = [
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        ]
        qualifiers.extend(self.resolved_struct_member_qualifiers.get(id(var), ()))
        metal_type = str(getattr(var, "vtype", "") or "").strip()
        while metal_type.endswith(("*", "&")):
            metal_type = metal_type[:-1].strip()

        binding = self.scalar_alias_binding(
            metal_type, context=self.parameter_type_contexts.get(id(var))
        )
        if binding is not None:
            qualifiers.extend(binding[1])
            return list(dict.fromkeys(qualifiers))

        seen = set()
        while metal_type in self.type_aliases and metal_type not in seen:
            seen.add(metal_type)
            qualifiers.extend(
                str(qualifier).lower()
                for qualifier in self.type_alias_qualifiers.get(metal_type, [])
            )
            metal_type = str(self.type_aliases[metal_type]).strip()
            while metal_type.endswith(("*", "&")):
                metal_type = metal_type[:-1].strip()

        return list(dict.fromkeys(qualifiers))

    def metal_auto_pointer_object(self, var):
        """Whether ``var`` is a plain ``auto`` local whose initializer deduced a pointer type."""
        if self.metal_auto_pointer_depth(getattr(var, "vtype", None)):
            return False
        if not self.is_metal_auto_declaration_type(getattr(var, "vtype", None)):
            return False
        name = getattr(var, "name", None)
        inferred_type = self.current_variable_types.get(
            name, self.global_variable_types.get(name)
        )
        return self.metal_pointer_pointee_type_once(inferred_type) is not None

    def effective_declaration_qualifiers(self, var):
        qualifiers = self.resolved_declaration_qualifiers(var)
        if self.is_metal_auto_declaration_type(getattr(var, "vtype", None)):
            name = getattr(var, "name", None)
            inferred = self.current_variable_type_qualifiers.get(
                name, self.global_variable_type_qualifiers.get(name, ())
            )
            if self.metal_auto_pointer_object(var):
                # ``const auto p = ptr`` is ``T* const``: the const is the pointer object's own, not the pointee's.
                qualifiers = [
                    qualifier
                    for qualifier in qualifiers
                    if str(qualifier).lower() not in {"const", "volatile"}
                ]
            qualifiers.extend(inferred)
        return list(dict.fromkeys(qualifiers))

    def alias_pointer_pointee_qualifiers(self, var):
        """Keep alias pointee qualifiers separate from a const pointer object."""
        name = str(getattr(var, "vtype", "") or "").strip()
        seen = set()
        while name in self.type_aliases and name not in seen:
            seen.add(name)
            target = str(self.type_aliases[name]).strip()
            if self.pointer_element_type(target) is not None:
                return self.type_alias_pointee_qualifiers.get(name, ())
            name = target
        return ()

    def resource_memory_qualifiers(self, var):
        """Return ordered Metal resource-memory qualifiers for a declaration."""
        if not self.declaration_has_resource_storage(var):
            return []
        qualifiers = []
        for qualifier in self.effective_declaration_qualifiers(var):
            if qualifier == "volatile" or re.fullmatch(
                r"coherent(?:\([A-Za-z_][A-Za-z_0-9]*\))?", qualifier
            ):
                qualifiers.append(qualifier)
        return qualifiers

    def declaration_has_resource_storage(self, var):
        resolved_type = self.resolve_type_alias(self.effective_metal_variable_type(var))
        return bool(
            self.pointer_element_type(resolved_type)
            or self.reference_element_type(resolved_type)
            or self.is_metal_resource_type(resolved_type)
        )

    def resource_memory_qualifier_prefix(self, var):
        qualifiers = self.resource_memory_qualifiers(var)
        # ``T* __restrict p`` promises that no other pointer reaches the buffer: CrossGL's ``restrict``.
        if self.declaration_has_resource_storage(var) and {
            "restrict",
            "__restrict",
            "__restrict__",
        } & set(self.effective_declaration_qualifiers(var)):
            qualifiers = [*qualifiers, "restrict"]
        return f"{' '.join(qualifiers)} " if qualifiers else ""

    def map_resource_pointer_element_type(self, var, element_type):
        """Keep atomic pointer identity when resource qualifiers make it contractual."""
        normalized = self.normalized_metal_type(element_type)
        if self.resource_memory_qualifiers(var) and self.atomic_element_type(
            normalized
        ):
            if normalized.startswith("atomic<") and normalized.endswith(">"):
                inner = normalized[len("atomic<") : -1]
                return f"atomic<{self.map_type(inner)}>"
            if normalized.startswith("metal::"):
                return normalized[len("metal::") :]
            return normalized
        return self.map_type(element_type)

    def address_space_qualifier_annotations(self, var):
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        }
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(var, "attributes", []) or []
        }
        annotations = []
        for qualifier in ("ray_data", "object_data"):
            compact = qualifier.replace("_", "")
            if (
                qualifier in qualifiers
                and qualifier not in attributes
                and compact not in attributes
            ):
                annotations.append(f"@{qualifier}")
        resolved_type = self.resolve_type_alias(getattr(var, "vtype", None))
        if self.uniform_value_payload_type(resolved_type) is not None:
            annotations.append("@uniform_value")
        return " ".join(annotations)

    def resource_address_space_qualifier_annotations(self, var):
        """Retain source-only address-space identity after pointer resource lowering.

        Stage pointer parameters become portable StructuredBuffer declarations in
        CrossGL.  That abstraction preserves read/write access, but it otherwise
        cannot distinguish Metal's read-only ``constant`` address space from a
        ``const device`` pointer.  Keep the narrower source contract as metadata so
        a Metal round trip can pass the resource to a constant-pointer helper
        without silently substituting device storage.  Other targets treat this as
        Metal declaration metadata and continue to use the portable resource type.
        """
        if self.structured_buffer_pointer_type(var) is None:
            return ""
        qualifiers = set(self.effective_declaration_qualifiers(var))
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(var, "attributes", []) or []
        }
        if "constant" in qualifiers and "constant" not in attributes:
            return "@constant"
        return ""

    def is_sampler_variable(self, var):
        return self.is_sampler_type(getattr(var, "vtype", None))

    def is_sampler_type(self, metal_type):
        normalized = self.normalized_metal_type(metal_type)
        if normalized == "sampler":
            return True
        array_type = self.metal_array_type_parts(normalized)
        return bool(
            array_type and self.normalized_metal_type(array_type[0]) == "sampler"
        )

    def format_decl(
        self, var, include_semantic=False, declare_name=True, semantic_context=None
    ):
        self.reject_abi_visible_wide_vector_declaration(var)
        alignas_prefix = ""
        if hasattr(var, "alignas") and var.alignas:
            parts = []
            for item in var.alignas:
                if isinstance(item, tuple) and item[0] == "type":
                    parts.append(f"alignas({self.map_type(item[1])})")
                else:
                    parts.append(f"alignas({self.generate_expression(item, False)})")
            alignas_prefix = " ".join(parts) + " "
        mapped_type = self.map_declared_variable_type(var)
        name_array_suffix = ""
        include_declarator_arrays = True
        grouped_type_suffix = (
            getattr(var, "declarator_type_suffix", "")
            if getattr(var, "declarator_type_suffix_grouped", False)
            else ""
        )
        decayed_stage_entry_array = bool(
            getattr(var, "array_sizes", None)
            and self.structured_buffer_pointer_type(var) is not None
            and self.pointer_element_type(getattr(var, "vtype", None)) is None
        )
        if decayed_stage_entry_array:
            type_str = mapped_type
        elif grouped_type_suffix and getattr(var, "array_sizes", None):
            include_declarator_arrays = False
            base_type = mapped_type
            if base_type.endswith(grouped_type_suffix):
                base_type = base_type[: -len(grouped_type_suffix)].rstrip()
            type_array_suffix = self.format_declarator_array_suffix(var)
            type_str = f"{base_type}{type_array_suffix}{grouped_type_suffix}"
        else:
            if self.use_name_array_suffix(mapped_type, var):
                include_declarator_arrays = False
                name_array_suffix = self.format_declarator_array_suffix(var)
            type_array_suffix = self.format_array_suffix(var, include_declarator_arrays)
            # The element of ``array<array<T, 3>, 2>`` maps to ``T[3]``; the outer dimensions come first.
            element_base, element_suffix = (
                split_array_type_suffix(mapped_type)
                if type_array_suffix and str(mapped_type).rstrip().endswith("]")
                else (mapped_type, "")
            )
            type_str = f"{element_base}{type_array_suffix}{element_suffix}"
        address_space = self.address_space_qualifier_prefix(var)
        qualifiers = set(self.effective_declaration_qualifiers(var))
        alias_name = str(getattr(var, "vtype", ""))
        if (
            "volatile" in qualifiers
            and not self.declaration_has_resource_storage(var)
            and (
                self.scalar_alias_binding(alias_name) is not None
                or alias_name in self.local_type_alias_names
            )
        ):
            raise MetalScalarAliasResolutionError(
                alias_name,
                "volatile-qualified scalar aliases require resource storage",
                source_location=getattr(var, "source_location", None)
                or getattr(
                    self.current_type_resolution_context, "source_location", None
                ),
            )
        lowered_buffer_type = self.structured_buffer_pointer_type(var)
        resolved_effective_type = self.resolve_type_alias(
            self.effective_metal_variable_type(var)
        )
        pointee_qualifiers = getattr(var, "pointee_qualifiers", None)
        if self.metal_auto_pointer_object(var):
            # The initializer's pointer type alone qualifies the pointee of ``const auto p = ptr``.
            pointee_qualifiers = None
        pointee_qualifier_names = {
            str(qualifier).lower()
            for qualifier in (
                qualifiers if pointee_qualifiers is None else pointee_qualifiers
            )
        }
        pointee_qualifier_names.update(
            self.resolved_struct_member_qualifiers.get(id(var), ())
        )
        pointee_qualifier_names.update(self.alias_pointer_pointee_qualifiers(var))
        if self.is_metal_auto_declaration_type(getattr(var, "vtype", None)):
            pointee_qualifier_names.update(qualifiers)
        const_pointer_pointee = bool(
            self.preserve_pointer_pointee_const
            and "const" in pointee_qualifier_names
            and self.pointer_element_type(resolved_effective_type) is not None
        )
        # A leading `const` on a threadgroup pointer local qualifies its pointee
        # (`const threadgroup T* view = tile + offset;`). Parameters drop that
        # qualifier when pointee constness is not preserved, so the local must
        # follow the same policy: otherwise the view cannot be passed to the
        # helper parameter it was declared for (MLX's merge sort partitions).
        threadgroup_pointee_const_local = bool(
            not self.preserve_pointer_pointee_const
            and getattr(var, "is_const", False)
            and "threadgroup" in qualifiers
            and pointee_qualifiers is not None
            and "const" in pointee_qualifier_names
            and "const" not in {
                str(qualifier).lower()
                for qualifier in getattr(var, "indirection_qualifiers", None) or ()
            }
            and self.pointer_element_type(resolved_effective_type) is not None
        )
        const_device_indirection = bool(
            "const" in qualifiers
            and "device" in qualifiers
            and (
                self.pointer_element_type(resolved_effective_type) is not None
                or self.reference_element_type(resolved_effective_type) is not None
            )
            and (pointee_qualifiers is None or "const" in pointee_qualifier_names)
        )
        const_str = (
            "const "
            if (
                (
                    getattr(var, "is_const", False)
                    and not threadgroup_pointee_const_local
                    and not self.metal_auto_pointer_object(var)
                )
                or (
                    "const" in qualifiers
                    and (
                        self.scalar_alias_binding(alias_name) is not None
                        or alias_name in self.local_type_alias_names
                    )
                )
                or const_pointer_pointee
                or const_device_indirection
            )
            and lowered_buffer_type is None
            and address_space.strip() != "constant"
            else ""
        )
        semantic = (
            self.map_semantic(
                getattr(var, "attributes", None), context=semantic_context
            )
            if include_semantic
            else ""
        )
        address_space_annotations = self.address_space_qualifier_annotations(var)
        resource_address_space_annotations = (
            self.resource_address_space_qualifier_annotations(var)
        )
        semantic = " ".join(
            part
            for part in [
                address_space_annotations,
                semantic,
                resource_address_space_annotations,
            ]
            if part
        )
        access = self.storage_texture_access_attribute(var)
        storage_format = self.storage_texture_format_attributes(var)
        name = (
            self.declare_identifier(var.name)
            if declare_name
            else self.sanitize_identifier(var.name)
        )
        if name_array_suffix:
            name = f"{name}{name_array_suffix}"
        memory_qualifiers = self.resource_memory_qualifier_prefix(var)
        if "volatile" in qualifiers and not self.declaration_has_resource_storage(var):
            memory_qualifiers += "volatile "
        parts = [
            alignas_prefix + memory_qualifiers + const_str + address_space + type_str,
            name,
        ]
        if semantic:
            parts.append(semantic)
        if access:
            parts.append(access)
        if storage_format:
            parts.append(storage_format)
        return " ".join(part for part in parts if part)

    def format_struct_member_decl(
        self, member, owner=None, static_member_types=None, semantic_context=None
    ):
        """Render a struct field while preserving compile-time member constants.

        ``static_member_types`` types the struct's own static members while an
        unresolved default such as ``OUT_TILE_SIZE + FILTER_SIZE - 1`` of an
        explicit specialization renders, so arithmetic between them keeps its
        declared type. ``semantic_context`` is ``{"kind": "fetched_member"}``
        for a structure a fragment function reads attachments through.
        """
        declaration = self.format_decl(
            member, include_semantic=True, semantic_context=semantic_context
        )
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(member, "qualifiers", []) or []
        }
        if not qualifiers.intersection({"static", "constexpr"}):
            return declaration

        declaration = f"static {declaration}"
        default_value = getattr(member, "default_value", None)
        if default_value is not None and not getattr(member, "array_sizes", None):
            key = (owner, getattr(member, "name", None))
            if owner is not None and key in self.struct_static_constexpr_member_keys:
                rendered_default = self.render_resolved_static_constant(key)
            elif static_member_types and self.metal_arithmetic_profiles_active():
                saved_types = self.current_variable_types
                self.current_variable_types = {**static_member_types, **saved_types}
                try:
                    with constant_initialization(self):
                        rendered_default = self.generate_expression(
                            default_value, False
                        )
                finally:
                    self.current_variable_types = saved_types
            else:
                with constant_initialization(self):
                    rendered_default = self.generate_expression(default_value, False)
            declaration += f" = {rendered_default}"
        return declaration

    METAL_NUMERIC_LITERAL_RE = re.compile(
        r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFhHuUlL]*|0[xX][0-9a-fA-F]+[uUlL]*"
    )

    def out_of_class_static_member_initializer(self, declaration):
        """In-class initializer of an out-of-class static member redeclaration.

        C++17 makes ``constant constexpr const float S<6, 3, 8>::table[8][8];``
        a redundant redeclaration of a ``static constexpr`` member whose value
        is the in-class initializer. Only literal-valued initializers move to
        program scope: one naming a sibling member would need its scope, so it
        fails closed rather than leaving an uninitialized program constant,
        which OpenGL would otherwise bind as a host-supplied uniform.
        """
        name = getattr(declaration, "name", None)
        if (
            not isinstance(name, str)
            or "::" not in name
            or getattr(declaration, "value", None) is not None
            or getattr(declaration, "default_value", None) is not None
        ):
            return None
        owner, member_name = (part.strip() for part in name.rsplit("::", 1))
        struct_node = self.struct_declarations.get(owner)
        if struct_node is None:
            mapped_owner = self.map_struct_name(owner)
            struct_node = next(
                (
                    node
                    for raw_name, node in self.struct_declarations.items()
                    if self.map_struct_name(raw_name) == mapped_owner
                ),
                None,
            )
        if struct_node is None:
            return None
        for member in getattr(struct_node, "members", []) or []:
            if not isinstance(member, VariableNode) or member.name != member_name:
                continue
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(member, "qualifiers", []) or []
            }
            if "static" not in qualifiers or "constexpr" not in qualifiers:
                return None
            initializer = getattr(member, "default_value", None)
            if initializer is None or not self.is_literal_initializer(initializer):
                raise MetalStaticConstantResolutionError(
                    owner,
                    member_name,
                    "an out-of-class redeclaration needs a literal in-class "
                    "initializer",
                )
            return initializer
        return None

    def is_literal_initializer(self, node):
        """Whether ``node`` only combines numeric literals and brace lists."""
        if isinstance(node, str):
            return bool(self.METAL_NUMERIC_LITERAL_RE.fullmatch(node.strip()))
        if isinstance(node, InitializerListNode):
            return all(self.is_literal_initializer(item) for item in node.elements)
        if isinstance(node, BinaryOpNode):
            return node.op in {"+", "-", "*", "/"} and all(
                self.is_literal_initializer(item) for item in (node.left, node.right)
            )
        if isinstance(node, UnaryOpNode):
            return node.op in {"+", "-"} and self.is_literal_initializer(node.operand)
        return False

    def metal_arithmetic_profiles_active(self):
        """Whether arithmetic rendering needs each operation's resolved type."""
        return self.floating_contraction_profile is not None or any(
            self.metal_arithmetic_profile(operator)[0] is not None
            for operator in ("+", "-", "*", "/")
        )

    def format_parameter_decl(self, var, index, semantic_context=None):
        # An empty declaration drops the parameter, so only None falls through.
        for stage_parameter_declaration in (
            self.metal_mesh_parameter_declaration,
            self.metal_object_parameter_declaration,
            self.metal_post_tessellation_parameter_declaration,
        ):
            declaration = stage_parameter_declaration(var)
            if declaration is not None:
                return declaration
        if getattr(var, "name", None):
            declaration = self.format_decl(
                var,
                include_semantic=True,
                semantic_context=semantic_context,
            )
            declaration = self.lower_c_array_parameter_reference(var, declaration)
            return self.with_parameter_direction_qualifier(
                var, declaration, semantic_context=semantic_context
            )

        generated_name = self.reserve_generated_identifier(f"_unnamed_param_{index}")
        original_name = var.name
        var.name = generated_name
        try:
            declaration = self.format_decl(
                var,
                include_semantic=True,
                declare_name=False,
                semantic_context=semantic_context,
            )
            declaration = self.lower_c_array_parameter_reference(var, declaration)
            declaration = self.with_parameter_direction_qualifier(
                var, declaration, semantic_context=semantic_context
            )
            return f"{declaration} @maybe_unused"
        finally:
            var.name = original_name

    def with_parameter_direction_qualifier(
        self, var, declaration, semantic_context=None
    ):
        qualifiers = [
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        ]
        is_reference = self.reference_parameter(var)
        is_readonly_reference = self.readonly_parameter(var, qualifiers)
        if (
            is_reference
            and self.pointer_element_type(self.effective_metal_variable_type(var))
            is not None
        ):
            is_readonly_reference = self.readonly_pointer_reference(var)
        for qualifier in self.parameter_direction_qualifiers:
            if qualifier in qualifiers:
                if is_reference:
                    declaration = re.sub(r"(?<=\S)&(?=\s)", "", declaration, count=1)
                return f"{qualifier} {declaration}"
        if (
            is_reference
            and id(var) in self.current_stage_entry_resource_parameter_ids
            and self.is_stage_entry_buffer_resource_parameter(var)
        ):
            return declaration
        if is_reference and not is_readonly_reference:
            declaration = re.sub(r"(?<=\S)&(?=\s)", "", declaration, count=1)
            return f"inout {declaration}"
        if is_reference and is_readonly_reference:
            declaration = re.sub(r"(?<=\S)&(?=\s)", "", declaration, count=1)
            return f"in {declaration}"
        if self.writable_c_array_parameter(var, semantic_context):
            return f"inout {declaration}"
        return declaration

    @staticmethod
    def reference_parameter(var):
        raw_type = str(getattr(var, "vtype", "")).strip()
        return raw_type.endswith("&") or (
            getattr(var, "declarator_type_suffix_grouped", False)
            and getattr(var, "declarator_type_suffix", "") == "&"
        )

    @staticmethod
    def readonly_parameter(var, qualifiers=None):
        qualifier_set = set(qualifiers or ())
        return bool(
            qualifier_set & {"const", "constant", "readonly", "in"}
            or getattr(var, "is_const", False)
        )

    @staticmethod
    def readonly_pointer_reference(var):
        raw_type = str(getattr(var, "vtype", "")).strip()
        if not raw_type.endswith("&"):
            return False
        referenced_type = raw_type[:-1].rstrip()
        return re.search(r"\*\s*const$", referenced_type) is not None

    def lower_c_array_parameter_reference(self, var, declaration):
        if not getattr(var, "array_sizes", None):
            return declaration
        if not getattr(var, "declarator_type_suffix_grouped", False):
            return declaration
        if getattr(var, "declarator_type_suffix", "") != "&":
            return declaration
        return re.sub(r"(?<=\])&(?=\s)", "", declaration, count=1)

    def writable_c_array_parameter(self, var, semantic_context=None):
        """Return whether a Metal C-style array parameter aliases writable storage."""
        if not getattr(var, "array_sizes", None):
            return False

        function_qualifier = str(
            (semantic_context or {}).get("function_qualifier") or ""
        ).lower()
        if function_qualifier in {"vertex", "fragment", "kernel"} | self.rt_qualifiers:
            return False

        raw_type = str(getattr(var, "vtype", "")).strip()
        if raw_type.endswith("*"):
            return False

        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        }
        if qualifiers & {"const", "constant", "readonly", "in"}:
            return False
        return not bool(getattr(var, "is_const", False))

    def format_global_decl(self, var, include_semantic=False):
        declaration = self.format_decl(var, include_semantic=include_semantic)
        attributes = getattr(var, "attributes", []) or []
        if any(
            isinstance(attr, AttributeNode) and attr.name == "argument_buffer"
            for attr in attributes
        ):
            declaration = re.sub(r"(?<=\S)&\s+(\w+)", r" \1", declaration, count=1)
        return declaration

    def kernel_workgroup_size_attribute(self):
        """The mark of a kernel whose host chooses its threadgroup size, with that size when it is configured."""
        return (
            f"        @ {host_workgroup_size_attribute_text(self.threadgroup_size)}\n"
        )
