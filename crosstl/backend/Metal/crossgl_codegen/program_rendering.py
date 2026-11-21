"""Phases of ``MetalToCrossGLConverter.render_program``: per-program state, declaration preparation, the shader header, types, globals, functions and the final support-code substitution."""

from ..color_attachments import enumerator_index_values, framebuffer_fetch_struct_names
from ..floating_point import contraction_directive_requires_precise
from ..local_enums import hoist_local_enums
from ..member_aggregates import anonymous_member_paths
from ..MetalParser import (
    AssignmentNode,
    ConstantBufferNode,
    EnumNode,
    FunctionNode,
    PreprocessorNode,
    StaticAssertNode,
    StructNode,
    TypeAliasNode,
    VariableNode,
)
from ..unscoped_enums import unscoped_enumerator_owners
from ..unused_generic_definitions import (
    raise_referenced_deferred_definitions,
    render_generic_definition,
)
from ..vote_masks import generate_vote_mask_support_code


class MetalProgramRenderingMixin:
    """Members of ``MetalToCrossGLConverter``: the phases ``render_program`` runs in order."""

    def reset_program_render_state(
        self,
    ):
        """Clear the helper, alias-template and wide-vector state a previous program left behind."""
        self.small_vector_index_types = {}
        self.small_vector_index_operations = set()
        self.small_vector_resource_index_operations = {}
        self.metal_enum_arithmetic_types = {}
        self.metal_enum_member_types = {}
        self.wide_vector_types = {}
        self.wide_vector_binary_helpers = set()
        self.wide_vector_compound_helpers = set()
        self.wide_vector_reserved_names = set()
        self.metal_precise_math_helper_names = {}
        self.required_metal_vote_masks = set()
        self.required_metal_precise_acos_widths = set()
        self.required_metal_precise_asin_widths = set()
        self.required_metal_precise_acosh_widths = set()
        self.required_metal_precise_atan_widths = set()
        self.required_metal_precise_atan2_widths = set()
        self.required_metal_power_widths = set()
        self.required_metal_precise_power_widths = set()
        self.required_metal_round_widths = set()
        self.required_metal_precise_exp_widths = set()
        self.required_metal_precise_log_widths = set()
        self.required_metal_log2_widths = set()
        self.required_metal_precise_sqrt_widths = set()
        self.required_metal_precise_rsqrt_widths = set()
        self.required_metal_precise_trig_widths = {"sin": set(), "cos": set()}
        self.required_metal_fma_widths = set()
        self.required_metal_fmuladd_widths = set()
        self.required_metal_division_widths = set()
        self.required_metal_half_remainder_widths = set()
        self.required_metal_half_widening_widths = set()
        self.required_metal_comparisons = set()
        self.required_metal_remainder_widths = set()
        self.required_metal_additive_operations = set()
        self.metal_additive_assignments = {}
        self.required_metal_multiplication_widths = set()
        self.metal_multiplication_assignments = {}
        self.metal_division_assignments = {}
        self.cooperative_matrix_fragment_helpers = {}
        self.storage_wrapper_read_helpers = {}
        self.cooperative_matrix_fragment_helper_names = set()
        self.cooperative_matrix_fragment_type_contracts = {}
        self.cooperative_matrix_fragment_type_replacements = {}
        self.alias_template_declarations = {}
        self.alias_template_plain_declarations = {}
        self.scalar_alias_bindings = {}
        self.scalar_alias_names = set()
        self.parameter_type_contexts = {}
        self.type_alias_enum_declarations = {}
        self.alias_template_cache = {}
        self.alias_template_resolution_stack = []
        self.alias_template_structs = []
        self.alias_template_structs_by_qualified_name = {}
        self.alias_template_using_namespaces = []
        self.alias_template_using_declarations = []
        self.current_type_resolution_context = None

    def prepare_program_declarations(
        self,
        ast,
    ):
        """Collect type aliases, user functions, value-template materializations, alias templates and transport constants before any code is emitted."""
        typedefs = getattr(ast, "typedefs", []) or []
        self.collect_callable_type_aliases(typedefs)
        self.validate_runtime_callable_alias_usage(ast)
        self.type_aliases = {
            alias.name: alias.alias_type
            for alias in typedefs
            if isinstance(alias, TypeAliasNode)
            and alias.name not in self.callable_type_aliases
            and not self.is_template_alias_declaration(alias)
        }
        self.type_alias_qualifiers = {
            alias.name: list(getattr(alias, "qualifiers", []) or [])
            for alias in typedefs
            if isinstance(alias, TypeAliasNode)
            and alias.name not in self.callable_type_aliases
            and not self.is_template_alias_declaration(alias)
        }
        self.type_alias_pointee_qualifiers = {
            alias.name: list(
                getattr(alias, "pointee_qualifiers", alias.qualifiers) or []
            )
            for alias in typedefs
            if isinstance(alias, TypeAliasNode) and alias.name in self.type_aliases
        }
        # Body-local ``using`` and ``typedef`` aliases discovered while emitting
        # function bodies; these are inlined at their use sites rather than
        # emitted as typedefs.
        self.local_type_alias_names = set()
        self.unresolved_local_type_aliases = {}
        self.local_struct_type_aliases = {}
        self.local_integral_constant_bindings = {}
        self.global_integral_constant_bindings = {}
        self.local_aggregate_declarations = []
        self.local_aggregate_declaration_keys = {}
        self.local_aggregate_reserved_names = set()
        self.struct_member_name_maps = {}
        functions = getattr(ast, "functions", []) or []
        self.prepare_texture_usage(ast)
        self.prepare_global_integral_constants(ast)
        self.prepare_out_of_line_call_operator_bindings(functions)
        effective_functions = [
            function
            for function in functions
            if id(function) not in self.suppressed_out_of_line_call_operator_ids
        ]
        self.wide_vector_reserved_names.update(
            self.collect_declared_identifier_names(ast)
        )
        self.user_function_names = {
            function.name
            for function in effective_functions
            if isinstance(function, FunctionNode) and function.name
        }
        self.user_function_overloads_by_name = {}
        for function in effective_functions:
            if isinstance(function, FunctionNode) and function.name:
                self.user_function_overloads_by_name.setdefault(
                    function.name, []
                ).append(function)
        self.prepare_value_template_materializations(ast, effective_functions)
        self.prepare_alias_template_resolution(ast, typedefs)
        self.prepare_metal_source_overload_transport(ast)
        self.prepare_metal_atomic_fence_transport_constants(ast)
        return typedefs, functions

    def render_program_header(
        self,
        ast,
        wide_vector_support_marker,
    ):
        """Emit the include comments and open the ``shader main`` block."""
        code = ""
        # Contraction and reassociation directives are carried as @precise functions (floating_point.py).
        self.no_contraction_scope = bool(getattr(ast, "no_contraction", False))
        includes = [
            inc
            for inc in getattr(ast, "includes", []) or []
            if not (
                isinstance(inc, PreprocessorNode)
                and str(inc.directive).strip() == "#pragma"
                and contraction_directive_requires_precise(str(inc.content)) is not None
            )
        ]
        for inc in includes:
            if isinstance(inc, PreprocessorNode):
                line = f"{inc.directive} {inc.content}".strip()
            else:
                line = str(inc).strip()
            if line:
                code += f"{line}\n"
        if includes:
            code += "\n"
        code += "shader main {\n"
        code += "\n"
        code += wide_vector_support_marker
        return code

    def render_program_type_declarations(
        self,
        ast,
        code,
        functions,
        typedefs,
    ):
        """Prepare constants, structs, enums and constructor contracts, and emit the typedefs."""
        self.constant_struct_name = []

        # Get constants - support both 'constant' and 'constants' attributes
        constants = getattr(ast, "constant", []) or getattr(ast, "constants", []) or []
        for constant in constants:
            if isinstance(constant, ConstantBufferNode):
                self.process_constant_struct(ast)

        # Get structs - support both 'struct' and 'structs' attributes
        structs = getattr(ast, "structs", []) or getattr(ast, "struct", []) or []
        structs = self.flatten_concrete_nested_structs(structs)
        self.struct_name_map = self.build_struct_name_map(structs)
        self.struct_declarations = {
            struct_node.name: struct_node
            for struct_node in structs
            if isinstance(struct_node, StructNode)
            and getattr(struct_node, "name", None)
        }
        self.anonymous_member_paths = {
            name: paths
            for name, struct_node in self.struct_declarations.items()
            if (paths := anonymous_member_paths(struct_node, self.struct_declarations))
        }
        self.struct_member_types = self.collect_struct_member_types(structs)
        # Static constants resolve eagerly; constructor calls in their
        # initializers need the constructor contracts.
        self.prepare_metal_constructor_contracts(structs)
        self.collect_struct_static_constants(structs)
        self.wide_vector_reserved_names.update(
            self.map_struct_name(struct_node.name)
            for struct_node in structs
            if isinstance(struct_node, StructNode)
            and getattr(struct_node, "name", None)
        )
        self.wide_vector_reserved_names.update(
            self.sanitize_identifier(alias.name)
            for alias in typedefs
            if isinstance(alias, TypeAliasNode) and getattr(alias, "name", None)
        )
        self.local_aggregate_reserved_names.update(self.struct_name_map.values())
        self.local_aggregate_reserved_names.update(self.wide_vector_reserved_names)
        hoist_local_enums(ast, self.refuse_import)
        enums = getattr(ast, "enums", []) or []
        for enum in enums:
            if isinstance(enum, EnumNode):
                self.register_metal_enum_arithmetic_contract(enum)
        self.metal_enum_index_values = enumerator_index_values(enums)
        self.metal_fetched_struct_names = framebuffer_fetch_struct_names(
            functions,
            [
                struct_node.name
                for struct_node in structs
                if isinstance(struct_node, StructNode)
            ],
        )
        emitted_typedefs = []
        for alias in typedefs:
            if not isinstance(alias, TypeAliasNode):
                continue
            previous_type_resolution_context = self.current_type_resolution_context
            self.current_type_resolution_context = alias
            declaration = self.format_type_alias_declaration(alias)
            self.current_type_resolution_context = previous_type_resolution_context
            if declaration is not None:
                emitted_typedefs.append(declaration)
        if emitted_typedefs:
            code += "    // Typedefs\n"
            for declaration in emitted_typedefs:
                code += f"    {declaration}\n"
            code += "\n"
        return structs, enums, code

    def render_program_enums_and_structs(
        self,
        code,
        enums,
        structs,
    ):
        """Emit the enums and the struct declarations."""
        if enums:
            code += "    // Enums\n"
            used_enum_names = {
                enum.name for enum in enums if isinstance(enum, EnumNode) and enum.name
            }
            anonymous_enum_index = 0
            emitted_enums = []
            for enum in enums:
                if isinstance(enum, EnumNode):
                    enum_name = enum.name
                    if not enum_name:
                        while True:
                            candidate = f"MetalAnonymousEnum{anonymous_enum_index}"
                            anonymous_enum_index += 1
                            if candidate not in used_enum_names:
                                enum_name = candidate
                                used_enum_names.add(candidate)
                                break
                    emitted_enums.append((enum, enum_name))
            # Unscoped enumerators are names of the enclosing scope; CrossGL spells them E::A.
            self.unscoped_enumerator_owners = unscoped_enumerator_owners(emitted_enums)
            self.enumerator_names = {
                (enum_name, member)
                for enum, enum_name in emitted_enums
                for member, _value in enum.members
            }
            for enum, enum_name in emitted_enums:
                code += f"    enum {enum_name} {{\n"
                # An enumerator is in scope from its own definition on, also in a scoped enum's later
                # initializers (C++17 [basic.scope.pdecl]/3): ``enum class W { P = 2, Q = P + 3 }``.
                declared_members = {}
                self.identifier_maps.append(declared_members)
                try:
                    for member_name, member_value in enum.members:
                        if member_value is not None:
                            value = self.generate_expression(member_value, False)
                            code += f"        {member_name} = {value},\n"
                        else:
                            code += f"        {member_name},\n"
                        declared_members[member_name] = (
                            f"{enum_name}::{self.sanitize_identifier(member_name)}"
                        )
                finally:
                    self.identifier_maps.pop()
                code += "    };\n\n"
        for struct_node in structs:
            if isinstance(struct_node, StructNode):
                previous_type_resolution_context = self.current_type_resolution_context
                self.current_type_resolution_context = struct_node
                is_union = getattr(struct_node, "aggregate_kind", None) == "union"
                union_layout = None
                if is_union:
                    union_name = self.map_struct_name(struct_node.name or "anonymous")
                    union_layout = self.metal_union_layout_contract(struct_node)
                    code += (
                        f"    // Metal union {union_name} retains overlapping "
                        "storage through layout metadata\n"
                        f"    @union_layout({union_layout['size']}, "
                        f"{union_layout['alignment']}, little_endian, metal)\n"
                    )
                if struct_node.name in self.constant_struct_name:
                    code += "    // cbuffers\n"
                    code += f"    cbuffer {self.map_struct_name(struct_node.name)} {{\n"
                else:
                    code += "    // Structs\n"
                    code += self.metal_struct_alignment_attributes(struct_node)
                    code += (
                        f"    {self.format_generic_prefix(struct_node)}"
                        "struct "
                        f"{self.map_struct_name(struct_node.name)} {{\n"
                    )
                static_member_types = {
                    member.name: self.metal_declaration_expression_type(member)
                    for member in struct_node.members
                    if isinstance(member, VariableNode)
                    and getattr(member, "name", None)
                    and getattr(member, "vtype", None)
                    and {
                        str(qualifier).lower()
                        for qualifier in getattr(member, "qualifiers", []) or []
                    }.intersection({"static", "constexpr"})
                }
                for member in struct_node.members:
                    if isinstance(member, StaticAssertNode):
                        cond = self.generate_expression(member.condition, False)
                        if member.message is not None:
                            msg = (
                                member.message
                                if isinstance(member.message, str)
                                else self.generate_expression(member.message, False)
                            )
                            code += f"        static_assert({cond}, {msg});\n"
                        else:
                            code += f"        static_assert({cond});\n"
                        continue
                    decl = self.format_struct_member_decl(
                        member,
                        owner=self.map_struct_name(struct_node.name),
                        static_member_types=static_member_types,
                        semantic_context=(
                            {"kind": "fetched_member"}
                            if struct_node.name
                            in getattr(self, "metal_fetched_struct_names", ())
                            else None
                        ),
                    )
                    if self.metal_packed_member(member) and not decl.startswith(
                        "static "
                    ):
                        decl = f"@packed {decl}"
                    if isinstance(member, VariableNode) and getattr(
                        member, "name", None
                    ):
                        owner_name = self.map_struct_name(struct_node.name)
                        self.struct_member_name_maps.setdefault(owner_name, {})[
                            member.name
                        ] = self.identifier_maps[-1].get(
                            member.name, self.sanitize_identifier(member.name)
                        )
                    if is_union and isinstance(member, VariableNode):
                        member_size, member_alignment = union_layout["members"].get(
                            id(member), (0, 0)
                        )
                        decl = (
                            "@union_member_layout(0, "
                            f"{member_size}, {member_alignment}) {decl}"
                        )
                    code += f"        {decl};\n"
                code += "    }\n\n"
                self.current_type_resolution_context = previous_type_resolution_context
        return code

    def render_program_globals(
        self,
        ast,
        code,
        cooperative_matrix_fragment_support_marker,
        local_aggregate_support_marker,
        precise_math_support_marker,
        small_vector_index_support_marker,
    ):
        """Place the support-code markers and emit the global declarations."""

        code += local_aggregate_support_marker
        code += small_vector_index_support_marker
        code += cooperative_matrix_fragment_support_marker
        code += precise_math_support_marker

        globals_list = [
            glob
            for glob in getattr(ast, "global_variables", [])
            or getattr(ast, "global_vars", [])
            if id(glob) not in self.metal_atomic_fence_transport_declaration_ids
        ]
        if globals_list:
            code += "    // Globals\n"
            for glob in globals_list:
                if isinstance(glob, VariableNode):
                    initializer = self.out_of_class_static_member_initializer(glob)
                    if initializer is not None:
                        glob = AssignmentNode(glob, initializer, "=")
                if isinstance(glob, StaticAssertNode):
                    cond = self.generate_expression(glob.condition, False)
                    if glob.message is not None:
                        msg = (
                            glob.message
                            if isinstance(glob.message, str)
                            else self.generate_expression(glob.message, False)
                        )
                        code += f"    static_assert({cond}, {msg});\n"
                    else:
                        code += f"    static_assert({cond});\n"
                    continue
                declaration_context = (
                    glob.left
                    if isinstance(glob, AssignmentNode)
                    and isinstance(glob.left, VariableNode)
                    else glob
                )
                previous_type_resolution_context = self.current_type_resolution_context
                self.current_type_resolution_context = declaration_context
                if isinstance(declaration_context, VariableNode):
                    self.validate_global_constructor_initialization(
                        declaration_context,
                        initialized=isinstance(glob, AssignmentNode),
                    )
                if isinstance(glob, AssignmentNode):
                    if isinstance(glob.left, VariableNode):
                        if self.is_sampler_variable(glob.left):
                            # A program-scope constexpr sampler is fixed sampler
                            # state; it stays a CrossGL constant so every target
                            # either carries that state or refuses it.
                            code += self.program_scope_sampler_constant(
                                glob.left, glob.right
                            )
                            self.current_type_resolution_context = (
                                previous_type_resolution_context
                            )
                            continue
                        # ``constant constexpr auto n = 32;`` takes its type from the
                        # initializer, as a block-scope ``auto`` does ([dcl.type.auto.deduct]).
                        inferred_type = self.inferred_metal_declaration_type(
                            glob.left, glob.right
                        )
                        self.global_variable_types[glob.left.name] = inferred_type
                        self.global_variable_type_qualifiers[glob.left.name] = (
                            self.inferred_metal_declaration_type_qualifiers(
                                glob.left, glob.right, inferred_type
                            )
                        )
                    left = (
                        self.format_decl(glob.left, include_semantic=True)
                        if isinstance(glob.left, VariableNode)
                        else self.generate_expression(glob.left, False)
                    )
                    right = self.generate_initializer_value(
                        self.array_initializer_without_designators(
                            glob.left, glob.right
                        ),
                        False,
                        (
                            getattr(glob.left, "vtype", None)
                            if isinstance(glob.left, VariableNode)
                            else None
                        ),
                        (
                            self.variable_has_array_initializer_shape(glob.left)
                            if isinstance(glob.left, VariableNode)
                            else False
                        ),
                        (
                            self.declaration_constructor_receiver_address_space(
                                glob.left
                            )
                            if isinstance(glob.left, VariableNode)
                            else None
                        ),
                        copy_initialize_lvalue=True,
                    )
                    code += f"    {left} {glob.operator} {right};\n"
                elif isinstance(glob, VariableNode):
                    if self.is_sampler_variable(glob):
                        code += self.program_scope_sampler_constant(glob, None)
                        self.current_type_resolution_context = (
                            previous_type_resolution_context
                        )
                        continue
                    self.global_variable_types[glob.name] = (
                        self.metal_declaration_expression_type(glob)
                    )
                    self.global_variable_type_qualifiers[glob.name] = (
                        self.metal_declaration_type_qualifiers(glob)
                    )
                    if id(glob) in self.storage_texture_declaration_ids:
                        self.global_storage_texture_names.add(glob.name)
                    if self.structured_buffer_pointer_type(glob):
                        self.global_structured_buffer_names.add(glob.name)
                    decl = self.format_global_decl(glob, include_semantic=True)
                    code += f"    {decl};\n"
                self.current_type_resolution_context = previous_type_resolution_context
            code += "\n"
        return code

    def render_program_functions(
        self,
        code,
        functions,
    ):
        """Emit the ordinary functions, the value-template specializations and the deferred template functions."""
        ordinary_function_code = []
        deferred_template_functions = []
        self.deferred_generic_definition_errors = []
        for f in functions:
            if id(f) in self.suppressed_out_of_line_call_operator_ids:
                continue
            if self.is_materialized_metal_stdlib_wrapper(f):
                continue
            default_bindings = self.default_value_template_bindings.get(id(f))
            if self.value_template_parameter_names(f) and default_bindings is None:
                deferred_template_functions.append(f)
                continue
            if (
                id(f) in self.suppressed_value_template_function_ids
                and default_bindings is None
            ):
                continue
            ordinary_function_code.append(
                render_generic_definition(
                    self,
                    f,
                    lambda f=f, default_bindings=default_bindings: (
                        self.generate_top_level_function(
                            f,
                            template_value_bindings=default_bindings,
                        )
                    ),
                )
            )

        specialization_code = {}
        specialization_entries = {}
        specialization_index = 0
        while specialization_index < len(self.pending_value_template_specializations):
            function, value_bindings, type_bindings, specialized_name, key = (
                self.pending_value_template_specializations[specialization_index]
            )
            specialization_index += 1
            specialization_entries[key] = (
                function,
                value_bindings,
                type_bindings,
                specialized_name,
            )
            specialization_code[key] = self.generate_top_level_function(
                function,
                template_value_bindings=value_bindings,
                template_type_bindings=type_bindings,
                output_name=specialized_name,
                specialization_key=key,
            )

        deferred_template_code = []
        self.preserve_unmaterialized_template_calls = True
        try:
            for function in deferred_template_functions:
                if id(function) in self.suppressed_value_template_function_ids:
                    continue
                deferred_template_code.append(
                    render_generic_definition(
                        self,
                        function,
                        lambda function=function: self.generate_top_level_function(
                            function
                        ),
                    )
                )
        finally:
            self.preserve_unmaterialized_template_calls = False

        code += self.generate_pending_constructor_factories()
        code += self.generate_storage_wrapper_read_helpers()
        code += "".join(deferred_template_code)
        for key in self.ordered_value_template_specialization_keys(
            specialization_entries
        ):
            code += specialization_code[key]
        code += "".join(ordinary_function_code)
        return code

    def finish_program_render(
        self,
        code,
        cooperative_matrix_fragment_support_marker,
        local_aggregate_support_marker,
        precise_math_support_marker,
        small_vector_index_support_marker,
        wide_vector_support_marker,
    ):
        """Close the shader block, replace the support-code markers with the helpers the program needs, and check deferred definitions."""
        code += "}\n"
        code = code.replace(
            wide_vector_support_marker,
            self.generate_wide_vector_support_code(indent=1),
            1,
        )
        code = code.replace(
            small_vector_index_support_marker,
            self.generate_small_vector_index_support_code(indent=1),
            1,
        )
        code = code.replace(
            cooperative_matrix_fragment_support_marker,
            self.generate_cooperative_matrix_fragment_support_code(indent=1),
            1,
        )
        code = code.replace(
            precise_math_support_marker,
            self.generate_metal_division_support_code(indent=1)
            + self.generate_metal_fma_support_code(indent=1)
            + self.generate_metal_half_remainder_support_code(indent=1)
            + self.generate_metal_half_widening_support_code(indent=1)
            + self.generate_metal_comparison_support_code(indent=1)
            + self.generate_metal_remainder_support_code(indent=1)
            + self.generate_metal_additive_support_code(indent=1)
            + self.generate_metal_multiplication_support_code(indent=1)
            + self.generate_metal_precise_math_support_code(indent=1)
            + generate_vote_mask_support_code(self, indent=1),
            1,
        )
        code = code.replace(
            local_aggregate_support_marker,
            self.generate_local_aggregate_declarations(indent=1),
            1,
        )
        code = self.propagate_cooperative_matrix_fragment_type_contracts(code)
        raise_referenced_deferred_definitions(self, code)
        return code
