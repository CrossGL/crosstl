"""Program entries and declarations of the Metal generator: entry scoping and pruning, checked program rendering, atomic fence and resource memory contracts, unused declarations, function constants, ABI and interpolation attributes, struct alignment and union layouts, argument buffers, HLSL program constant buffers and struct declarations."""

import re

from ..ast import (
    ArrayLiteralNode,
    ArrayNode,
    CooperativeMatrixOpNode,
    CooperativeMatrixType,
    FunctionCallNode,
    IdentifierNode,
    PointerType,
    PreprocessorNode,
    ReferenceType,
    StructNode,
    VariableNode,
)
from ..explicit_specializations import functions_by_called_name
from ..refusals import MetalRefusal
from ..semantics.byte_address_buffers import lower_byte_address_buffers
from ..semantics.constant_buffers import lower_constant_buffer_resources
from ..semantics.declared_types import declared_type_names
from ..semantics.narrowing_constructors import select_narrowing_constructor_components
from ..semantics.structured_buffer_counters import (
    materialize_structured_buffer_counters,
)
from ..semantics.unsized_arrays import UnsizedArrayUnsupportedError
from ..validation import (
    collect_cbuffer_declaration_name_conflicts,
    collect_cbuffer_member_global_conflicts,
    collect_duplicate_cbuffer_names,
)
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .barrier_peephole import merge_adjacent_metal_barriers
from .call_graph import validate_recursion_free
from .entry_selection import entry_type_references
from .enum_utils import (
    enum_struct_fields,
    generic_enum_specialized_fields,
)
from .generic_pointer_spaces import (
    assign_generic_pointer_spaces,
    refuse_written_generic_pointers,
)
from .generic_struct_utils import generic_struct_specialized_fields
from .metal_combined_samplers import bind_combined_samplers
from .metal_double_refusal import refuse_metal_double_values
from .metal_draw_base_indices import offset_metal_draw_base_indices
from .metal_invariant_position import metal_invariant_attribute_suffix
from .metal_keyword_identifiers import (
    rename_metal_keyword_members,
    rename_metal_keyword_parameters,
)
from .metal_library_functions import rename_msl_library_functions
from .metal_namespace_names import rename_metal_namespace_names
from .metal_reported_hits import lower_metal_reported_hits
from .metal_safe_math import metal_safe_math_requested
from .metal_specialization_constants import (
    metal_function_constant_type_refusal,
    render_metal_specialization_constant,
    render_metal_specialization_macro,
)
from .metal_stage_input_builtins import (
    lower_metal_helper_invocation_reads,
    split_metal_stage_input_builtins,
)
from .metal_vertex_input_builtins import refuse_unbound_metal_vertex_inputs
from .pointer_reinterpret import (
    crossgl_literal_positive_integer,
    validate_pointer_reinterpretation_target,
)
from .relational_builtins import lower_relational_builtins
from .sampler_pairs import canonicalize_separate_textures
from .stage_utils import requested_stage_name
from .workgroup_size_extents import fold_workgroup_size_extents
from .written_function_names import (
    METAL_CALLABLE_FUNCTION_NAMES,
    refuse_written_foreign_calls,
)
from .written_type_names import refuse_written_foreign_types


class MetalProgramEntryMixin:
    """Program entry members of ``MetalCodeGen``."""

    def generate(self, ast):
        """Generate complete Metal Shading Language source for a CrossGL AST."""
        return self.generate_program(ast)

    def generate_entry(self, ast, entry_point):
        """Generate one independently loadable Metal compute entry artifact."""
        scoped_ast = self.entry_scoped_ast(ast, entry_point)
        return self.generate_program(scoped_ast)

    def entry_scoped_functions(self, ast, entry_function):
        functions = list(getattr(ast, "functions", []) or [])
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            functions.extend(getattr(stage, "local_functions", []) or [])
        functions.append(entry_function)
        return functions

    def prune_entry_scoped_declarations(self, ast, entry_function):
        """Retain only module declarations reachable from one compute entry."""
        functions = self.entry_scoped_functions(ast, entry_function)
        function_names = {
            getattr(function, "name", None)
            for function in functions
            if getattr(function, "name", None)
        }
        referenced_names = set()
        referenced_types = set()
        for function in functions:
            local_names = {
                getattr(parameter, "name", None)
                for parameter in (
                    getattr(
                        function,
                        "parameters",
                        getattr(function, "params", []),
                    )
                    or []
                )
                if getattr(parameter, "name", None)
            }
            local_names.update(
                getattr(node, "name", None)
                for node in self.entry_ast_nodes(getattr(function, "body", None))
                if isinstance(node, VariableNode) and getattr(node, "name", None)
            )
            referenced_names.update(
                self.entry_identifier_references(function) - local_names
            )
            referenced_types.update(
                entry_type_references(function, self.entry_ast_nodes)
            )
        referenced_names.difference_update(function_names)
        referenced_names.update(referenced_types)

        declaration_groups = [
            (ast, "global_variables", False),
            (ast, "constants", False),
            (ast, "cbuffers", True),
        ]
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            declaration_groups.extend(
                (
                    (stage, "local_variables", False),
                    (stage, "local_cbuffers", True),
                )
            )

        # Struct fields can depend on aliases even when the entry names only the struct.
        dependency_groups = declaration_groups + [(ast, "structs", False)]
        dependency_groups.extend(
            (stage, "local_structs", False)
            for stage in getattr(ast, "stages", {}).values()
        )
        declarations = []
        for owner, attribute, include_members in dependency_groups:
            for declaration in getattr(owner, attribute, []) or []:
                names = self.entry_declaration_names(
                    declaration,
                    include_members=include_members,
                )
                declarations.append((declaration, names))

        retained_ids = set()
        changed = True
        while changed:
            changed = False
            for declaration, names in declarations:
                declaration_id = id(declaration)
                if declaration_id in retained_ids or not names & referenced_names:
                    continue
                retained_ids.add(declaration_id)
                referenced_names.update(
                    self.entry_identifier_references(declaration) - names
                )
                referenced_names.update(
                    entry_type_references(declaration, self.entry_ast_nodes)
                )
                changed = True

        for owner, attribute, _include_members in declaration_groups:
            setattr(
                owner,
                attribute,
                [
                    declaration
                    for declaration in getattr(owner, attribute, []) or []
                    if id(declaration) in retained_ids
                ],
            )

    def prune_entry_scoped_structs(self, ast, entry_function):
        """Retain struct families referenced by the selected entry closure."""
        declaration_groups = [(ast, "structs")]
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            declaration_groups.append((stage, "local_structs"))
        declarations = [
            declaration
            for owner, attribute in declaration_groups
            for declaration in getattr(owner, attribute, []) or []
        ]
        struct_names = {
            str(declaration.name)
            for declaration in declarations
            if getattr(declaration, "name", None)
        }
        if not struct_names:
            return

        roots = self.entry_scoped_functions(ast, entry_function)
        roots.extend(getattr(ast, "global_variables", []) or [])
        roots.extend(getattr(ast, "constants", []) or [])
        roots.extend(getattr(ast, "cbuffers", []) or [])
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            roots.extend(getattr(stage, "local_variables", []) or [])
            roots.extend(getattr(stage, "local_cbuffers", []) or [])

        retained_names = set()
        pending_names = set()
        for root in roots:
            pending_names.update(self.entry_struct_references(root, struct_names))
        while pending_names:
            name = pending_names.pop()
            if name in retained_names:
                continue
            retained_names.add(name)
            for declaration in declarations:
                if getattr(declaration, "name", None) == name:
                    pending_names.update(
                        self.entry_struct_references(declaration, struct_names)
                        - retained_names
                    )

        for owner, attribute in declaration_groups:
            setattr(
                owner,
                attribute,
                [
                    declaration
                    for declaration in getattr(owner, attribute, []) or []
                    if getattr(declaration, "name", None) in retained_names
                ],
            )

    def entry_struct_references(self, root, struct_names):
        references = set()
        type_attributes = (
            "return_type",
            "param_type",
            "var_type",
            "vtype",
            "member_type",
            "const_type",
            "constructor_type",
            "element_type",
            "referenced_type",
            "pointee_type",
        )
        for node in self.entry_ast_nodes(root):
            if isinstance(node, IdentifierNode):
                name = getattr(node, "name", None)
                if name in struct_names:
                    references.add(str(name))
            for attribute in type_attributes:
                type_value = getattr(node, attribute, None)
                if type_value is None:
                    continue
                try:
                    type_text = self.type_name_string(type_value)
                except (AttributeError, TypeError, ValueError):
                    type_text = str(type_value)
                for name in struct_names:
                    if re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                        str(type_text),
                    ):
                        references.add(name)
            for inherited in getattr(node, "inheritance", []) or []:
                inherited_text = str(inherited)
                for name in struct_names:
                    if re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                        inherited_text,
                    ):
                        references.add(name)
        return references

    def entry_ast_nodes(self, root):
        if root is None:
            return ()
        walk = getattr(root, "walk", None)
        return walk() if callable(walk) else self.iter_ast_nodes(root)

    def entry_identifier_references(self, root):
        return {
            str(node.name)
            for node in self.entry_ast_nodes(root)
            if isinstance(node, IdentifierNode) and getattr(node, "name", None)
        }

    def entry_declaration_names(self, declaration, *, include_members):
        name = getattr(
            declaration,
            "name",
            getattr(declaration, "variable_name", None),
        )
        names = {str(name)} if name else set()
        if include_members:
            names.update(
                str(member.name)
                for member in getattr(declaration, "members", []) or []
                if getattr(member, "name", None)
            )
        return names

    def entry_reachable_function_names(self, ast, entry_function):
        functions = list(getattr(ast, "functions", []) or [])
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            functions.extend(getattr(stage, "local_functions", []) or [])
        functions_by_name = functions_by_called_name(functions)

        reachable_names = set()
        pending = [entry_function]
        visited = set()
        while pending:
            function = pending.pop()
            if id(function) in visited:
                continue
            visited.add(id(function))
            for node in self.metal_render_walks.body_nodes_of_type(
                function, FunctionCallNode
            ):
                name = self.function_call_name(node)
                if not name or name in reachable_names:
                    continue
                candidates = functions_by_name.get(name, ())
                if not candidates:
                    continue
                reachable_names.add(name)
                reachable_names.update(
                    candidate.name for candidate in candidates if candidate.name
                )
                pending.extend(candidates)
        return reachable_names

    def generate_stage(self, ast, shader_type):
        """Generate Metal source for a single requested shader stage."""
        return self.generate_program(ast, target_stage=shader_type)

    def generate_program(self, ast, target_stage=None):
        """Render an AST to Metal, optionally filtering stage entry points.

        Raises :class:`MetalLoweringRefused` listing every construct that has no faithful Metal lowering.
        """
        lower_relational_builtins(ast)
        select_narrowing_constructor_components(ast)
        return merge_adjacent_metal_barriers(
            self.refusals.run(self.render_checked_program, ast, target_stage)
        )

    def render_checked_program(self, ast, target_stage=None):
        """Render the program and refuse the types the Metal language does not have."""
        ast = assign_generic_pointer_spaces(ast, self.refuse_lowering)
        self.program_type_names = declared_type_names(ast)
        self.written_foreign_type_names = set()
        self.metal_safe_math = metal_safe_math_requested(ast)
        try:
            output = self.render_program(ast, target_stage)
            refuse_written_foreign_types(
                output, self.written_foreign_type_names, "Metal"
            )
            refuse_written_foreign_calls(output, METAL_CALLABLE_FUNCTION_NAMES, "Metal")
            refuse_written_generic_pointers(output, self.refuse_lowering, "Metal")
        finally:
            self.written_foreign_type_names = None
        refuse_metal_double_values(output, self.refuse_lowering)
        return output

    def refuse_lowering(self, reason, fallback=""):
        """Record a construct with no faithful Metal lowering and return ``fallback`` in its place.

        The fallback only lets generation reach the other refusals in the program; no program is returned while a
        refusal is pending.
        """
        self.refusals.record(reason)
        return fallback

    def rewrite_rendered_program(self, ast, target_stage):
        """The program ``render_program`` renders, with the Metal rewrites applied, and its stage."""
        target_stage = requested_stage_name(target_stage)
        ast = fold_workgroup_size_extents(ast, "Metal", self.refuse_lowering)
        ast = lower_constant_buffer_resources(
            ast,
            lambda name: self.refuse_lowering(
                f"array of constant buffers '{name}': Metal binds an array of buffers only through an "
                "argument buffer"
            ),
        )
        ast = lower_byte_address_buffers(ast, self.refuse_lowering)
        ast = materialize_structured_buffer_counters(ast, self.refuse_lowering)
        ast = canonicalize_separate_textures(ast, "Metal", depth_for_compares=True)
        ast = bind_combined_samplers(ast)
        validate_pointer_reinterpretation_target(ast, "metal")
        validate_recursion_free(ast, "metal")
        self.validate_supported_stage_types(ast, target_stage)
        ast = rename_metal_keyword_members(ast)
        ast = self.rename_metal_reserved_program_identifiers(ast)
        ast = rename_msl_library_functions(ast)
        ast = rename_metal_namespace_names(ast)
        ast = rename_metal_keyword_parameters(ast)
        refuse_unbound_metal_vertex_inputs(ast, self)
        ast = split_metal_stage_input_builtins(ast, self)
        ast = lower_metal_helper_invocation_reads(ast, self.refuse_lowering)
        ast = offset_metal_draw_base_indices(ast, self)
        ast = lower_metal_reported_hits(ast, self.refuse_lowering)
        return ast, target_stage

    def collect_metal_atomic_fence_calls(self, ast):
        if "atomicThreadFence" in self.user_function_names:
            return []
        walk = getattr(ast, "walk", None)
        nodes = walk() if callable(walk) else self.iter_ast_nodes(ast)
        return [
            node
            for node in nodes
            if isinstance(node, FunctionCallNode)
            and self.function_call_name(node) == "atomicThreadFence"
        ]

    def collect_metal_resource_memory_contracts(self, ast):
        walk = getattr(ast, "walk", None)
        nodes = walk() if callable(walk) else self.iter_ast_nodes(ast)
        contracts = []
        for node in nodes:
            raw_type = getattr(node, "param_type", getattr(node, "var_type", None))
            if not self.metal_resource_memory_contract_is_emittable(node, raw_type):
                continue
            for contract in self.resource_memory_qualifier_contracts(node, raw_type):
                if contract not in contracts:
                    contracts.append(contract)
        return contracts

    def metal_resource_memory_contract_is_emittable(self, node, raw_type):
        if self.glsl_buffer_block_attribute(node) is not None:
            return True
        contract_type = raw_type
        while self.is_array_type_node(contract_type):
            contract_type = contract_type.element_type
        if isinstance(contract_type, (PointerType, ReferenceType)):
            return True
        if self.is_structured_buffer_type(raw_type):
            return True

        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        qualifiers.update(
            str(qualifier).lower()
            for qualifier in getattr(node, "resource_qualifiers", []) or []
        )
        qualifiers.update(
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        )
        return bool(
            qualifiers
            & {
                "constant",
                "device",
                "global",
                "local",
                "private",
                "storage",
                "thread",
                "threadgroup",
                "workgroup",
            }
        )

    @staticmethod
    def generate_metal_system_thread_scope_support():
        return (
            "#ifndef __METAL_MEMORY_SCOPE_SYSTEM__\n"
            "#define __METAL_MEMORY_SCOPE_SYSTEM__ 3\n"
            "#endif\n"
            "namespace metal {\n"
            "constexpr constant thread_scope thread_scope_system =\n"
            "    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);\n"
            "}\n\n"
        )

    def uses_cooperative_matrix(self, ast):
        """Return whether a canonical cooperative-matrix node is present."""
        walk = getattr(ast, "walk", None)
        if not callable(walk):
            return False
        return any(
            isinstance(node, (CooperativeMatrixType, CooperativeMatrixOpNode))
            for node in walk()
        )

    def generate_preprocessor_directive(self, directive):
        if isinstance(directive, PreprocessorNode):
            directive_name = (directive.directive or "").strip()
            if directive_name.lower() in {"version", "extension", "precision"}:
                return None
            return f"#{directive_name} {directive.content}".strip()

        line = str(directive).strip()
        lowered = line.lower()
        if (
            lowered.startswith("#version")
            or lowered.startswith("#extension")
            or lowered.startswith("precision ")
        ):
            return None
        return line

    def generate_constants(self, ast, constants=None, include_function_constants=True):
        code = ""
        for node in (
            getattr(ast, "constants", []) or [] if constants is None else constants
        ):
            name = getattr(node, "name", None)
            if not name:
                continue

            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            value = getattr(node, "value", None)
            value_code = self.generate_constant_expression(
                value,
                expected_type=(
                    const_type if isinstance(value, ArrayLiteralNode) else None
                ),
            )
            mapped_type = self.map_type(const_type)
            specialization_id = self.metal_specialization_constant_id(node)
            declaration = format_c_style_array_declaration(mapped_type, name)
            if specialization_id is not None:
                refusal = metal_function_constant_type_refusal(
                    name,
                    self.type_name_string(const_type),
                    mapped_type,
                    self.is_metal_function_constant_value_type,
                )
                if (
                    refusal is None
                    and name in self.metal_specialization_macro_constants
                ):
                    code += render_metal_specialization_macro(
                        self.metal_unused_declaration_qualifier("constant"),
                        mapped_type,
                        declaration,
                        specialization_id,
                        value_code,
                    )
                    continue
                if refusal is None:
                    code += render_metal_specialization_constant(
                        self.metal_unused_declaration_qualifier("constant"),
                        mapped_type,
                        declaration,
                        name,
                        specialization_id,
                        value_code,
                    )
                    continue
                self.refuse_lowering(refusal)
            if mapped_type == "sampler":
                # Program-scope fixed sampler state; texture calls take it as
                # their explicit sampler argument.
                self.program_scope_sampler_constants.add(name)
            code += (
                f"{self.metal_unused_declaration_qualifier('constant')} "
                f"{declaration} = {value_code};\n"
            )

        if not include_function_constants:
            return f"{code}\n" if code else ""

        used_function_constant_ids = {}
        for node in getattr(ast, "global_variables", []) or []:
            function_constant_id = self.metal_function_constant_id(node)
            if function_constant_id is None:
                continue

            name = getattr(node, "name", getattr(node, "variable_name", None))
            previous_name = used_function_constant_ids.get(function_constant_id)
            if previous_name is not None:
                raise MetalRefusal(
                    "Duplicate Metal function constant id "
                    f"{function_constant_id} for '{name}' and '{previous_name}'"
                )
            used_function_constant_ids[function_constant_id] = name

            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            self.validate_metal_function_constant_type(node, var_type)
            mapped_type = self.map_type(var_type)
            initial_value = getattr(node, "initial_value", None)
            if initial_value is not None:
                self.refuse_lowering(
                    f"function constant default: '{name}' initializers are not "
                    "allowed by MSL"
                )
            code += (
                f"{self.metal_unused_declaration_qualifier('constant')} "
                f"{mapped_type} {name} "
                f"[[function_constant({function_constant_id})]];\n"
            )

        return f"{code}\n" if code else ""

    def metal_unused_declaration_qualifier(self, qualifier):
        return f"__attribute__((unused)) {qualifier}"

    def format_unused_metal_declaration(self, declaration):
        declaration = str(declaration).strip()
        if declaration.startswith("__attribute__((unused)) "):
            return declaration
        return f"__attribute__((unused)) {declaration}"

    def maybe_format_unused_local_declaration(self, declaration, name):
        if name in self.current_unused_local_declaration_names:
            return self.format_unused_metal_declaration(declaration)
        return declaration

    def strip_unused_metal_declaration_attribute(self, declaration):
        declaration = str(declaration).strip()
        prefix = "__attribute__((unused)) "
        if declaration.startswith(prefix):
            return declaration[len(prefix) :]
        return declaration

    def format_match_binding_declaration(self, declaration, _binding_name):
        return self.format_unused_metal_declaration(declaration)

    def metal_function_constant_attributes(self, node):
        attributes = []
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower()
            if normalized.startswith("metal_") or normalized.startswith("msl_"):
                normalized = normalized.split("_", 1)[1]
            if normalized in {"function_constant", "constant_id"}:
                attributes.append(attr)
        return attributes

    def metal_specialization_constant_id(self, node):
        constant_id = None
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower().replace("-", "_")
            if normalized.startswith("metal_") or normalized.startswith("msl_"):
                normalized = normalized.split("_", 1)[1]
            if normalized != "constant_id":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            value = self.attribute_value_to_string(arguments[0])
            if constant_id is not None and constant_id != value:
                name = getattr(
                    node, "name", getattr(node, "variable_name", "<unnamed>")
                )
                raise MetalRefusal(
                    "Conflicting Metal specialization constant ids for "
                    f"'{name}': {constant_id} differs from {value}"
                )
            constant_id = value
        return constant_id

    def is_metal_function_constant_variable(self, node):
        return bool(self.metal_function_constant_attributes(node))

    def normalized_metal_abi_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None
        normalized = str(attr_name).lower()
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]
        return normalized

    def metal_function_constant_id(self, node):
        attributes = self.metal_function_constant_attributes(node)
        if not attributes:
            return None

        name = getattr(node, "name", getattr(node, "variable_name", None))
        if len(attributes) > 1:
            raise MetalRefusal(
                f"Metal function constant '{name}' has multiple function constant "
                "attributes"
            )

        arguments = getattr(attributes[0], "arguments", []) or []
        function_constant_id = (
            self.binding_index_value(arguments[0]) if len(arguments) == 1 else None
        )
        if function_constant_id is None:
            raise MetalRefusal(
                f"Metal function constant '{name}' requires an integer id"
            )
        return function_constant_id

    def validate_metal_function_constant_type(self, node, var_type):
        name = getattr(node, "name", getattr(node, "variable_name", None))
        type_name = self.type_name_string(var_type)
        mapped_type = self.map_type(var_type)
        invalid = (
            self.is_resource_parameter_type(type_name)
            or self.is_array_type_node(var_type)
            or "[" in mapped_type
            or "]" in mapped_type
            or "*" in mapped_type
            or "&" in mapped_type
        )
        if invalid:
            raise MetalRefusal(
                f"Metal function constant '{name}' cannot use type '{type_name}'"
            )

        if not self.is_metal_function_constant_value_type(mapped_type):
            raise MetalRefusal(
                f"Metal function constant '{name}' cannot use type '{type_name}'"
            )

    def is_metal_function_constant_value_type(self, mapped_type):
        scalar_types = {
            "bool",
            "int",
            "uint",
            "int64_t",
            "uint64_t",
            "float",
            "half",
        }
        if mapped_type in scalar_types:
            return True

        for base_type in scalar_types:
            if mapped_type in {
                f"{base_type}2",
                f"{base_type}3",
                f"{base_type}4",
            }:
                return True
        return False

    def is_metal_declaration_qualifier_attribute(self, attr):
        return self.normalized_metal_abi_attribute_name(attr) in {
            "const",
            "constexpr",
            "inline",
            "maybe_unused",
            "source_resource",
            "mutable",
            "static",
            "volatile",
        }

    def metal_function_linkage_prefix(self, func):
        qualifiers = set(getattr(func, "qualifiers", ()) or ())
        if getattr(func, "linkage", None) == "internal":
            qualifiers.add("static")
        if getattr(func, "is_inline", False):
            qualifiers.add("inline")
        qualifiers.update(
            self.normalized_metal_abi_attribute_name(attribute)
            for attribute in getattr(func, "attributes", ()) or ()
        )
        return "".join(
            qualifier + " "
            for qualifier in ("static", "inline")
            if qualifier in qualifiers
        )

    def is_metal_struct_member_abi_attribute(self, attr):
        return self.normalized_metal_abi_attribute_name(attr) == "id"

    def metal_struct_alignment(self, node):
        attributes = [
            attr
            for attr in getattr(node, "attributes", ()) or ()
            if self.normalized_metal_abi_attribute_name(attr) == "alignas"
        ]
        if not attributes:
            return None
        if len(attributes) != 1:
            self.metal_struct_alignment_error(
                node,
                "duplicate-alignment",
                "exactly one @metal_alignas attribute is required",
            )
        arguments = list(getattr(attributes[0], "arguments", ()) or ())
        if len(arguments) != 1:
            self.metal_struct_alignment_error(
                node,
                "malformed-alignment",
                "@metal_alignas requires one positive power-of-two integer",
            )
        alignment = crossgl_literal_positive_integer(arguments[0])
        if alignment is None or alignment & (alignment - 1):
            self.metal_struct_alignment_error(
                node,
                "invalid-alignment",
                "@metal_alignas must be a positive power-of-two integer",
            )
        return alignment

    def metal_union_layout_attribute(self, node, name):
        return [
            attr
            for attr in getattr(node, "attributes", []) or []
            if self.normalized_metal_abi_attribute_name(attr) == name
        ]

    def metal_union_layout_integer(self, node, attribute, index, member=None):
        arguments = list(getattr(attribute, "arguments", []) or [])
        if index >= len(arguments):
            return None
        value = self.binding_index_value(arguments[index])
        if value is None:
            raise self.metal_union_layout_error(
                node,
                "non-literal-layout",
                "layout sizes, alignments, and offsets must be integer literals",
                member,
            )
        return value

    def metal_union_layout_contract(self, node):
        if not isinstance(node, StructNode):
            return None
        attributes = self.metal_union_layout_attribute(node, "union_layout")
        if not attributes:
            return None
        if len(attributes) != 1:
            raise self.metal_union_layout_error(
                node,
                "duplicate-layout",
                "exactly one @union_layout attribute is required",
            )
        attribute = attributes[0]
        arguments = list(getattr(attribute, "arguments", []) or [])
        if len(arguments) != 4:
            raise self.metal_union_layout_error(
                node,
                "malformed-layout",
                "@union_layout requires size, alignment, byte order, and source ABI",
            )
        size = self.metal_union_layout_integer(node, attribute, 0)
        alignment = self.metal_union_layout_integer(node, attribute, 1)
        byte_order = self.attribute_value_to_string(arguments[2]).strip().lower()
        source_abi = self.attribute_value_to_string(arguments[3]).strip().lower()
        if size <= 0 or alignment <= 0:
            raise self.metal_union_layout_error(
                node,
                "invalid-layout-size",
                "aggregate size and alignment must be positive",
            )
        if byte_order != "little_endian":
            raise self.metal_union_layout_error(
                node,
                "byte-order-unsupported",
                f"byte order '{byte_order or '<missing>'}' is not a Metal ABI contract",
            )
        if source_abi not in {"metal", "msl"}:
            raise self.metal_union_layout_error(
                node,
                "source-abi-unsupported",
                f"source ABI '{source_abi or '<missing>'}' cannot be reconstructed natively",
            )

        member_layouts = {}
        for member in getattr(node, "members", []) or []:
            member_attributes = self.metal_union_layout_attribute(
                member, "union_member_layout"
            )
            if len(member_attributes) != 1:
                raise self.metal_union_layout_error(
                    node,
                    "member-layout-missing",
                    "exactly one @union_member_layout attribute is required",
                    member,
                )
            member_attribute = member_attributes[0]
            member_arguments = list(getattr(member_attribute, "arguments", []) or [])
            if len(member_arguments) != 3:
                raise self.metal_union_layout_error(
                    node,
                    "malformed-member-layout",
                    "@union_member_layout requires offset, size, and alignment",
                    member,
                )
            offset = self.metal_union_layout_integer(node, member_attribute, 0, member)
            member_size = self.metal_union_layout_integer(
                node, member_attribute, 1, member
            )
            member_alignment = self.metal_union_layout_integer(
                node, member_attribute, 2, member
            )
            if offset != 0:
                raise self.metal_union_layout_error(
                    node,
                    "member-offset-unsupported",
                    f"member offset {offset!r} does not alias byte zero",
                    member,
                )
            if (
                member_size <= 0
                or member_alignment <= 0
                or member_size > size
                or member_alignment > alignment
            ):
                raise self.metal_union_layout_error(
                    node,
                    "member-layout-out-of-range",
                    "member size or alignment exceeds the aggregate layout",
                    member,
                )
            member_layouts[getattr(member, "name", "")] = {
                "offset": offset,
                "size": member_size,
                "alignment": member_alignment,
            }
        if not member_layouts:
            raise self.metal_union_layout_error(
                node,
                "empty-union-unsupported",
                "at least one instance member is required",
            )
        return {
            "size": size,
            "alignment": alignment,
            "byte_order": byte_order,
            "source_abi": source_abi,
            "members": member_layouts,
        }

    def is_metal_argument_buffer_global(self, node):
        attributes = getattr(node, "attributes", []) or []
        return any(
            self.normalized_metal_abi_attribute_name(attr) == "argument_buffer"
            for attr in attributes
        )

    def format_metal_argument_buffer_global(self, node, mapped_type, name):
        if not name:
            return None

        attributes = []
        seen_attributes = set()
        for attr in getattr(node, "attributes", []) or []:
            attr_name = self.normalized_metal_abi_attribute_name(attr)
            if attr_name not in {"buffer", "id", "argument_buffer"}:
                continue
            if attr_name in seen_attributes:
                raise MetalRefusal(
                    f"Metal argument buffer global '{name}' has multiple "
                    f"{attr_name} attributes"
                )
            seen_attributes.add(attr_name)

            arguments = getattr(attr, "arguments", []) or []
            if attr_name in {"buffer", "id"}:
                attr_value = (
                    self.binding_index_value(arguments[0])
                    if len(arguments) == 1
                    else None
                )
                if attr_value is None:
                    raise MetalRefusal(
                        f"Metal argument buffer global '{name}' requires an "
                        f"integer {attr_name} attribute"
                    )
                attributes.append(f"[[{attr_name}({attr_value})]]")
            else:
                attributes.append("[[argument_buffer]]")

        if "buffer" not in seen_attributes:
            raise MetalRefusal(
                f"Metal argument buffer global '{name}' requires a buffer attribute"
            )

        base_type = str(mapped_type).strip().rstrip("&").strip()
        abi_attributes = " ".join(attributes)
        qualifier = self.metal_unused_declaration_qualifier("constant")
        return f"{qualifier} {base_type}& {name} {abi_attributes};\n"

    def metal_argument_buffer_binding(self, node):
        if not self.is_metal_argument_buffer_global(node):
            return None
        return self.explicit_resource_binding_index(node, {"buffer"}, ("b",))

    def format_metal_struct_member_abi_attributes(self, member):
        attributes = []
        seen_attributes = set()
        member_name = getattr(member, "name", None)
        for attr in getattr(member, "attributes", []) or []:
            if not self.is_metal_struct_member_abi_attribute(attr):
                continue

            arguments = getattr(attr, "arguments", []) or []
            attr_name = self.normalized_metal_abi_attribute_name(attr)

            if attr_name in seen_attributes:
                raise MetalRefusal(
                    f"Metal struct member '{member_name}' has multiple {attr_name} "
                    "attributes"
                )
            seen_attributes.add(attr_name)

            if attr_name == "id":
                member_id = (
                    self.binding_index_value(arguments[0])
                    if len(arguments) == 1
                    else None
                )
                if member_id is None:
                    raise MetalRefusal(
                        f"Metal struct member '{member_name}' requires an integer id"
                    )
                attributes.append(f"[[id({member_id})]]")

        return f" {' '.join(attributes)}" if attributes else ""

    def metal_interpolation_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None
        normalized = str(attr_name).lower()
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]
        return self.METAL_INTERPOLATION_ATTRIBUTES.get(normalized)

    def metal_interpolation_attribute_suffix(self, node):
        # The stage-value attributes that follow the semantic: interpolation, then invariance.
        return self.metal_interpolation_only_suffix(
            node
        ) + metal_invariant_attribute_suffix(self, node)

    def metal_interpolation_only_suffix(self, node):
        for attr in getattr(node, "attributes", []) or []:
            interpolation = self.metal_interpolation_attribute_name(attr)
            if interpolation is not None:
                return f" [[{interpolation}]]"
        implied = self.metal_semantic_implied_interpolation_attribute(node)
        if implied is not None:
            return f" [[{implied}]]"
        return ""

    def metal_semantic_implied_interpolation_attribute(self, node):
        semantic = self.semantic_from_node(node)
        if semantic is None:
            return None
        if str(semantic).lower() in {"gl_barycoordnoperspext", "gl_barycoordnoperspnv"}:
            return "center_no_perspective"
        return None

    def format_struct_resource_array_member(self, member):
        name = getattr(member, "name", None)
        if not name:
            return None

        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", None)
            )
            if element_type is None:
                return None
            base_type = (
                self.convert_type_node_to_string(element_type)
                if hasattr(element_type, "name")
                or hasattr(element_type, "element_type")
                else str(element_type)
            )
            if not self.is_resource_parameter_type(base_type):
                return None
            resource_type = self.map_resource_type_with_format(base_type, member)
            if member.size:
                array_size = self.safe_expression_to_string(member.size)
                return f"array<{resource_type}, {array_size}> {name}"
            return f"array<{resource_type}> {name}"

        raw_member_type = getattr(member, "member_type", None)
        if raw_member_type is None or not self.is_array_type_node(raw_member_type):
            return None

        array_type = self.resource_array_parameter(raw_member_type, member)
        if array_type is None:
            return None
        resource_type, array_size = array_type
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{resource_type}, {array_size}> {name}"
        return f"array<{resource_type}> {name}"

    def generate_constant_expression(self, expr, expected_type=None):
        if expected_type is None:
            value_code = self.generate_expression(expr)
        else:
            # An initializer list gets its declared array type, as a local's does.
            value_code = self.generate_expression_with_expected(expr, expected_type)
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

    def synthetic_hlsl_program_constants_cbuffer(self, ast):
        members = []
        self.hlsl_program_constant_global_ids = set()
        for node in getattr(ast, "global_variables", []) or []:
            if not self.is_hlsl_program_constant_global(node):
                continue
            members.append(node)
            self.hlsl_program_constant_global_ids.add(id(node))

        if not members:
            return None

        cbuffer = StructNode(
            self.unique_hlsl_program_constants_cbuffer_name(ast),
            members,
        )
        cbuffer.is_cbuffer = True
        cbuffer.buffer_kind = "cbuffer"
        cbuffer.is_synthetic_hlsl_program_constants = True
        return cbuffer

    def is_hlsl_program_constant_global(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() == "hlsl_program_constant":
                return True
        return False

    def unique_hlsl_program_constants_cbuffer_name(self, ast):
        reserved_names = {
            getattr(node, "name", None)
            for node in (
                list(getattr(ast, "structs", []) or [])
                + list(getattr(ast, "cbuffers", []) or [])
                + list(getattr(ast, "global_variables", []) or [])
            )
            if getattr(node, "name", None)
        }
        return self.unique_metal_generated_name("HlslProgramConstants", reserved_names)

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = self.cbuffer_variables or getattr(ast, "cbuffers", [])
        duplicate_names = collect_duplicate_cbuffer_names(cbuffers)
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise MetalRefusal(f"Duplicate cbuffer name(s) in Metal output: {names}")

        declaration_conflicts = collect_cbuffer_declaration_name_conflicts(ast)
        if declaration_conflicts:
            names = ", ".join(sorted(declaration_conflicts))
            raise MetalRefusal(
                f"Cbuffer name(s) conflict with existing Metal declaration(s): {names}"
            )

        global_member_conflicts = collect_cbuffer_member_global_conflicts(ast)
        if global_member_conflicts:
            names = ", ".join(sorted(global_member_conflicts))
            raise MetalRefusal(
                "Cbuffer member name(s) conflict with Metal global declaration(s): "
                f"{names}"
            )

        for node in cbuffers:
            layout_plan = self.metal_hlsl_cbuffer_plans.get(id(node))
            if layout_plan is not None:
                code += self.metal_hlsl_layout_struct_code(layout_plan)
                continue
            if isinstance(node, StructNode):
                code += f"struct {node.name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Metal requires a concrete element count for value arrays.
                            raise UnsizedArrayUnsupportedError(
                                "metal",
                                f"{node.name}.{member.name}",
                                "is a cbuffer member",
                            )
                    else:
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        member_type = self.metal_packed_storage_type(
                            member, member_type
                        )
                        declaration = format_c_style_array_declaration(
                            member_type, member.name
                        )
                        code += f"    {declaration};\n"
                code += "};\n"
            elif hasattr(node, "name") and hasattr(
                node, "members"
            ):  # CbufferNode handling
                code += f"struct {node.name} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        if member.size:
                            code += f"    {self.map_type(element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Metal requires a concrete element count for value arrays.
                            raise UnsizedArrayUnsupportedError(
                                "metal",
                                f"{node.name}.{member.name}",
                                "is a cbuffer member",
                            )
                    else:
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        member_type = self.metal_packed_storage_type(
                            member, member_type
                        )
                        declaration = format_c_style_array_declaration(
                            member_type, member.name
                        )
                        code += f"    {declaration};\n"
                code += "};\n"

        return code

    def generate_metal_struct_declarations(self, structs):
        definitions = []

        for specialization in (self.generic_struct_specializations or {}).values():
            fields = generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            )
            definitions.append(
                self.metal_field_struct_definition(
                    specialization["struct_name"],
                    fields,
                )
            )

        for enum in self.struct_payload_enums or []:
            fields = [("variant", "int")]
            fields.extend(enum_struct_fields(enum) or [])
            definitions.append(self.metal_field_struct_definition(enum.name, fields))

        for specialization in (self.generic_enum_specializations or {}).values():
            fields = [("variant", "int")]
            fields.extend(generic_enum_specialized_fields(self, specialization))
            definitions.append(
                self.metal_field_struct_definition(
                    specialization["struct_name"],
                    fields,
                )
            )

        for node in structs or []:
            definition = self.metal_plain_struct_definition(node)
            if definition is not None:
                definitions.append(definition)

        # A generic struct specialized here and a concrete struct the importer
        # already materialized for the same template-id share a name. Emit an
        # identical definition once; a differing one still reaches the compiler.
        emitted_code = {}
        unique_definitions = []
        for definition in definitions:
            name = definition.get("name")
            code = " ".join(definition["code"].split())
            if name and emitted_code.get(name) == code:
                continue
            if name:
                emitted_code.setdefault(name, code)
            unique_definitions.append(definition)
        definitions = unique_definitions

        self.metal_generated_struct_names = {
            definition["name"] for definition in definitions if definition.get("name")
        }
        return "".join(
            definition["code"]
            for definition in self.order_metal_struct_definitions(definitions)
        )

    def metal_field_struct_definition(self, name, fields):
        code = f"struct {name} {{\n"
        dependencies = set()
        for field_name, field_type in fields:
            mapped_type = self.metal_struct_storage_type(field_type)
            declaration = format_c_style_array_declaration(mapped_type, field_name)
            code += f"    {declaration};\n"
            dependencies.update(self.metal_struct_type_dependencies(mapped_type))
        code += "};\n\n"
        dependencies.discard(name)
        return {"name": name, "dependencies": dependencies, "code": code}

    def metal_struct_storage_type(self, value_type):
        """Keep declared integer widths in aggregate storage, not arithmetic."""
        base, suffix = split_array_type_suffix(self.type_name_string(value_type))
        native = self.metal_native_narrow_bitcast_storage_type(base)
        return native + suffix if native is not None else self.map_type(value_type)

    METAL_CONVERSION_RECEIVER_RE = re.compile(
        r"(?P<struct>[A-Za-z_][A-Za-z0-9_]*?)__operator_bool"
        r"(?:__metal_receiver_(?P<cv>const_volatile|const|volatile|mutable)"
        r"_(?P<spaces>[a-z_]+)_(?P<reference>unqualified|lvalue|rvalue))?"
    )
    METAL_RECEIVER_ADDRESS_SPACES = ("thread", "threadgroup", "device", "constant")
