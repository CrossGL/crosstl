"""Reverse code generator that emits CrossGL from Metal AST nodes."""

import re

from ...translator.precise_power import binary32_power_support
from ..import_refusals import ImportRefusalMixin, MetalImportRefused
from .alignment_specifiers import MetalAlignmentSpecifiers
from .buffer_layout_import import MetalBufferLayoutImport
from .builtin_operator_operands import operands_have_no_class_type
from .color_attachments import (
    color_attachment_semantic,
)
from .const_cast_buffer_writes import mark_const_cast_written_buffers
from .crossgl_codegen.alias_templates import MetalAliasTemplateMixin
from .crossgl_codegen.arithmetic_profiles import MetalArithmeticProfileMixin
from .crossgl_codegen.builtin_tables import MetalBuiltinTableMixin
from .crossgl_codegen.constexpr_evaluation import MetalConstexprMixin
from .crossgl_codegen.constructors import MetalConstructorMixin
from .crossgl_codegen.converter_setup import MetalConverterSetupMixin
from .crossgl_codegen.cooperative_matrices import MetalCooperativeMatrixMixin
from .crossgl_codegen.declaration_types import MetalDeclarationTypeMixin
from .crossgl_codegen.declarations import MetalDeclarationMixin
from .crossgl_codegen.expression_types import MetalExpressionTypeMixin
from .crossgl_codegen.expressions import MetalExpressionMixin
from .crossgl_codegen.function_definitions import MetalFunctionDefinitionMixin
from .crossgl_codegen.functions import MetalFunctionMixin
from .crossgl_codegen.identifiers import MetalIdentifierMixin
from .crossgl_codegen.import_tables import (
    FRAGMENT_EXECUTION_ATTRIBUTE_NAMES,
    FUNCTION_METADATA_ATTRIBUTE_NAMES,
    METAL_ATTRIBUTE_SEMANTICS,
    METAL_RAY_TRACING_STAGE_QUALIFIERS,
    METAL_TO_CROSSGL_TYPES,
    PARAMETER_DIRECTION_QUALIFIERS,
    SAMPLED_TEXTURE_METHODS,
    TEXTURE_GATHER_COMPONENTS,
    TEXTURE_METHOD_FUNCTIONS,
    TEXTURE_METHOD_OPERATIONS,
    TEXTURE_METHOD_STORAGE_OPERATIONS,
    TEXTURE_SIZE_QUERY_METHODS,
)
from .crossgl_codegen.local_aggregates import MetalLocalAggregateMixin
from .crossgl_codegen.math_builtins import MetalMathBuiltinMixin
from .crossgl_codegen.operands import MetalOperandMixin
from .crossgl_codegen.overload_resolution import MetalOverloadResolutionMixin
from .crossgl_codegen.precise_math import MetalPreciseMathMixin
from .crossgl_codegen.program_rendering import MetalProgramRenderingMixin
from .crossgl_codegen.source_overloads import MetalSourceOverloadMixin
from .crossgl_codegen.statements import MetalStatementMixin
from .crossgl_codegen.struct_declarations import MetalStructDeclarationMixin
from .crossgl_codegen.struct_methods import MetalStructMethodMixin
from .crossgl_codegen.structured_buffers import MetalStructuredBufferMixin
from .crossgl_codegen.support import (
    MetalAdditiveProfileError,
    MetalAddressProvenanceError,
    MetalAliasTemplateResolutionError,
    MetalArithmeticTypeResolutionError,
    MetalAtomicCompareExchangeLoweringError,
    MetalAtomicFenceLoweringError,
    MetalAtomicFetchSubLoweringError,
    MetalAtomicLoadLoweringError,
    MetalAtomicStoreLoweringError,
    MetalAutoTypeInferenceError,
    MetalBuiltinOverloadResolutionError,
    MetalBuiltinResultTypeResolutionError,
    MetalCallableAliasLoweringError,
    MetalCallableLoweringError,
    MetalComparisonProfileError,
    MetalConstexprBranchLoweringError,
    MetalConstructorContractError,
    MetalContractionProfileError,
    MetalCooperativeMatrixFragmentLoweringError,
    MetalDivisionProfileError,
    MetalFunctionLocalTypeResolutionError,
    MetalHalfRemainderProfileError,
    MetalHalfWideningProfileError,
    MetalIndexedComponentTypeResolutionError,
    MetalMultiplicationProfileError,
    MetalOutOfLineCallOperatorLoweringError,
    MetalPreciseMathLoweringError,
    MetalRemainderProfileError,
    MetalRoundLoweringError,
    MetalScalarAliasResolutionError,
    MetalSizeofResolutionError,
    MetalSourceOverloadResolutionError,
    MetalStageEntryArrayResourceError,
    MetalStandardLibraryWrapperLoweringError,
    MetalStaticConstantResolutionError,
    MetalStructAliasResolutionError,
    MetalStructMethodCallResolutionError,
    MetalTemplateArgumentResolutionError,
    MetalWideVectorLoweringError,
)
from .crossgl_codegen.synchronization import MetalSynchronizationMixin
from .crossgl_codegen.template_specialization import MetalTemplateSpecializationMixin
from .crossgl_codegen.texture_access import MetalTextureAccessMixin
from .crossgl_codegen.type_mapping import MetalTypeMappingMixin
from .crossgl_codegen.value_templates import MetalValueTemplateMixin
from .crossgl_codegen.vector_types import MetalVectorTypeMixin
from .explicit_specialization_annotations import explicit_specialization_annotation
from .external_functions import refuse_external_function_calls
from .function_constants import import_function_constant_defaults
from .library_structs import declare_referenced_library_structs
from .intersection_import import MetalIntersectionImport
from .mesh_import import MetalMeshImport
from .MetalAst import *
from .MetalLexer import *
from .MetalParser import *
from .namespace_scope_variables import resolve_namespace_scope_variables
from .post_tessellation_import import MetalPostTessellationImport
from .simd_aliases import SIMD_TYPE_ALIASES
from .sparse_color import (
    refuse_sparse_color_reference,
)

__all__ = [
    "MetalAtomicCompareExchangeLoweringError",
    "MetalAtomicStoreLoweringError",
    "MetalAdditiveProfileError",
    "MetalAddressProvenanceError",
    "MetalAliasTemplateResolutionError",
    "MetalArithmeticTypeResolutionError",
    "MetalAtomicFenceLoweringError",
    "MetalAtomicFetchSubLoweringError",
    "MetalAtomicLoadLoweringError",
    "MetalAutoTypeInferenceError",
    "MetalBuiltinOverloadResolutionError",
    "MetalBuiltinResultTypeResolutionError",
    "MetalCallableAliasLoweringError",
    "MetalCallableLoweringError",
    "MetalComparisonProfileError",
    "MetalConstexprBranchLoweringError",
    "MetalConstructorContractError",
    "MetalContractionProfileError",
    "MetalCooperativeMatrixFragmentLoweringError",
    "MetalDivisionProfileError",
    "MetalFunctionLocalTypeResolutionError",
    "MetalHalfRemainderProfileError",
    "MetalHalfWideningProfileError",
    "MetalImportRefused",
    "MetalIndexedComponentTypeResolutionError",
    "MetalMultiplicationProfileError",
    "MetalOutOfLineCallOperatorLoweringError",
    "MetalPreciseMathLoweringError",
    "MetalRemainderProfileError",
    "MetalRoundLoweringError",
    "MetalScalarAliasResolutionError",
    "MetalSizeofResolutionError",
    "MetalSourceOverloadResolutionError",
    "MetalStageEntryArrayResourceError",
    "MetalStandardLibraryWrapperLoweringError",
    "MetalStaticConstantResolutionError",
    "MetalStructAliasResolutionError",
    "MetalStructMethodCallResolutionError",
    "MetalTemplateArgumentResolutionError",
    "MetalToCrossGLConverter",
    "MetalWideVectorLoweringError",
    "parse_metal_structs",
]


def parse_metal_structs(source):
    """Parse Metal ``source`` and return its structure declarations."""
    return MetalParser(MetalLexer(source).tokenize()).parse().structs


class MetalToCrossGLConverter(
    MetalBuiltinTableMixin,
    MetalIdentifierMixin,
    MetalTextureAccessMixin,
    MetalStructDeclarationMixin,
    MetalConstructorMixin,
    MetalAliasTemplateMixin,
    MetalValueTemplateMixin,
    MetalConstexprMixin,
    MetalTemplateSpecializationMixin,
    MetalDeclarationMixin,
    MetalFunctionMixin,
    MetalLocalAggregateMixin,
    MetalExpressionMixin,
    MetalStatementMixin,
    MetalSynchronizationMixin,
    MetalStructMethodMixin,
    MetalSourceOverloadMixin,
    MetalOverloadResolutionMixin,
    MetalPreciseMathMixin,
    MetalArithmeticProfileMixin,
    MetalMathBuiltinMixin,
    MetalOperandMixin,
    MetalTypeMappingMixin,
    MetalCooperativeMatrixMixin,
    MetalVectorTypeMixin,
    MetalDeclarationTypeMixin,
    MetalExpressionTypeMixin,
    MetalStructuredBufferMixin,
    ImportRefusalMixin,
    MetalAlignmentSpecifiers,
    MetalBufferLayoutImport,
    MetalMeshImport,
    MetalIntersectionImport,
    MetalPostTessellationImport,
    MetalProgramRenderingMixin,
    MetalConverterSetupMixin,
    MetalFunctionDefinitionMixin,
):
    """Serialize Metal backend AST nodes back into CrossGL source."""

    IMPORT_REFUSAL_TYPE = MetalImportRefused

    crossgl_identifier_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    cast_literal_operand_pattern = re.compile(
        r"^(?:0[xX][0-9a-fA-F]+u?|0[bB][01]+u?|"
        r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fF]?|\d+u?)$"
    )
    metal_free_operator_helper_labels = {
        "+": "add",
        "-": "subtract",
        "*": "multiply",
        "/": "divide",
        "%": "modulo",
        "==": "equal",
        "!=": "not_equal",
        "<": "less",
        ">": "greater",
        "<=": "less_equal",
        ">=": "greater_equal",
        "+=": "add_assign",
        "-=": "subtract_assign",
        "*=": "multiply_assign",
        "/=": "divide_assign",
        "%=": "modulo_assign",
    }
    metal_free_operator_helper_prefix = "crosstl_metal_operator_"
    binary_precedence = {
        "||": 1,
        "&&": 2,
        "|": 3,
        "^": 4,
        "&": 5,
        "==": 6,
        "!=": 6,
        "<": 7,
        ">": 7,
        "<=": 7,
        ">=": 7,
        "<<": 8,
        ">>": 8,
        "+": 9,
        "-": 9,
        "*": 10,
        "/": 10,
        "%": 10,
    }
    assignment_precedence = -1
    conditional_precedence = 0
    postfix_precedence = max(binary_precedence.values()) + 1

    def __init__(
        self,
        cooperative_matrix_fragment_mapping=None,
        cooperative_matrix_fragment_mapping_provenance=None,
        preserve_pointer_pointee_const=True,
        resolve_standard_remove_cv_aliases=True,
        binary32_fma_profile=None,
        binary32_division_profile=None,
        binary16_remainder_profile=None,
        binary16_widening_profile=None,
        binary32_comparison_profile=None,
        binary32_remainder_profile=None,
        binary32_additive_profile=None,
        binary32_multiplication_profile=None,
        binary32_atan2_profile=None,
        binary32_log_profile=None,
        binary32_log2_operand_profile=None,
        binary32_log2_accuracy_profile=None,
        binary32_sqrt_profile=None,
        binary32_rsqrt_profile=None,
        binary32_power_operand_profile=None,
        binary32_power_accuracy_profile=None,
        floating_contraction_profile=None,
        preserve_resource_origins=False,
        threadgroup_size=None,
    ):
        self.configure_dispatch_and_profile_options(
            binary32_fma_profile,
            binary32_division_profile,
            binary16_remainder_profile,
            binary16_widening_profile,
            binary32_comparison_profile,
            binary32_atan2_profile,
            binary32_log_profile,
            binary32_power_operand_profile,
            binary32_power_accuracy_profile,
            floating_contraction_profile,
            preserve_resource_origins,
            threadgroup_size,
        )
        self.configure_log2_root_and_arithmetic_profiles(
            binary32_remainder_profile,
            binary32_additive_profile,
            binary32_multiplication_profile,
            binary32_log2_operand_profile,
            binary32_log2_accuracy_profile,
            binary32_sqrt_profile,
            binary32_rsqrt_profile,
        )
        self.configure_pointer_and_cooperative_matrix_options(
            cooperative_matrix_fragment_mapping,
            cooperative_matrix_fragment_mapping_provenance,
            preserve_pointer_pointee_const,
            resolve_standard_remove_cv_aliases,
        )
        self.rt_qualifiers = set(METAL_RAY_TRACING_STAGE_QUALIFIERS)
        self.type_map = dict(METAL_TO_CROSSGL_TYPES)
        # <simd/simd.h> aliases map like the Metal types they name.
        self.type_map.update(
            {
                alias: self.type_map[metal_type]
                for alias, metal_type in SIMD_TYPE_ALIASES.items()
                if metal_type in self.type_map
            }
        )
        self.initialize_declaration_state()
        self.initialize_helper_state()
        self.texture_method_functions = dict(TEXTURE_METHOD_FUNCTIONS)
        self.texture_method_storage_operations = dict(TEXTURE_METHOD_STORAGE_OPERATIONS)
        self.sampled_texture_methods = set(SAMPLED_TEXTURE_METHODS)
        self.texture_method_operations = dict(TEXTURE_METHOD_OPERATIONS)
        self.texture_gather_components = dict(TEXTURE_GATHER_COMPONENTS)
        self.texture_size_query_methods = set(TEXTURE_SIZE_QUERY_METHODS)
        self.parameter_direction_qualifiers = PARAMETER_DIRECTION_QUALIFIERS
        self.fragment_execution_attribute_names = set(
            FRAGMENT_EXECUTION_ATTRIBUTE_NAMES
        )
        self.function_metadata_attribute_names = set(FUNCTION_METADATA_ATTRIBUTE_NAMES)

        # [[attribute(n)]] is the vertex input at location n: see map_semantic.
        self.map_semantics = dict(METAL_ATTRIBUTE_SEMANTICS)

    def current_constructor_member_type(self, name):
        function = self.current_function
        if not getattr(function, "is_metal_constructor_factory", False):
            return None
        if name == "this":
            return getattr(function, "constructor_owner", None)
        member_types = getattr(function, "constructor_member_types", {}) or {}
        if name not in member_types or self.constructor_identifier_is_shadowed(name):
            return None
        return member_types[name]

    def prepare_metal_atomic_fence_transport_constants(self, ast):
        """Identify enum constants that exist only to name a fence operand."""
        self.metal_atomic_fence_transport_declaration_ids = set()
        globals_list = getattr(ast, "global_variables", []) or getattr(
            ast, "global_vars", []
        )
        values_by_type = {
            "mem_flags": self.metal_atomic_fence_memory_flags,
            "memory_order": self.metal_atomic_fence_memory_orders,
            "thread_scope": self.metal_atomic_fence_thread_scopes,
        }
        candidates = {}
        for declaration in globals_list:
            if not isinstance(declaration, AssignmentNode) or not isinstance(
                declaration.left, VariableNode
            ):
                continue
            variable = declaration.left
            enum_type = self.normalized_metal_type(variable.vtype)
            name = self.metal_atomic_fence_operand_identifier(variable)
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(variable, "qualifiers", []) or []
            }
            if (
                enum_type not in values_by_type
                or name not in values_by_type[enum_type]
                or "constexpr" not in qualifiers
            ):
                continue
            candidates.setdefault(name, []).append(declaration)

        if not candidates:
            return

        uses = {name: set() for name in candidates}
        candidate_names = set(candidates)
        for function in getattr(ast, "functions", []) or []:
            self.record_metal_atomic_fence_constant_uses(
                getattr(function, "body", []), candidate_names, uses
            )
        for declaration in globals_list:
            if isinstance(declaration, AssignmentNode):
                self.record_metal_atomic_fence_constant_uses(
                    declaration.right, candidate_names, uses
                )
            elif isinstance(declaration, VariableNode):
                self.record_metal_atomic_fence_constant_uses(
                    getattr(declaration, "value", None), candidate_names, uses
                )
        for declarations in (
            getattr(ast, "structs", []) or [],
            getattr(ast, "enums", []) or [],
        ):
            self.record_metal_atomic_fence_constant_uses(
                declarations, candidate_names, uses
            )

        self.metal_atomic_fence_transport_declaration_ids = {
            id(declarations[0])
            for name, declarations in candidates.items()
            if len(declarations) == 1 and uses[name] == {"fence"}
        }

    def record_metal_atomic_fence_constant_uses(
        self, value, candidate_names, uses, fence_operand=False
    ):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, dict):
            for item in value.values():
                self.record_metal_atomic_fence_constant_uses(
                    item, candidate_names, uses, fence_operand
                )
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self.record_metal_atomic_fence_constant_uses(
                    item, candidate_names, uses, fence_operand
                )
            return
        if isinstance(value, FunctionCallNode) and self.is_metal_atomic_fence_call(
            value.name
        ):
            for argument in value.args:
                self.record_metal_atomic_fence_constant_uses(
                    argument, candidate_names, uses, True
                )
            return
        if isinstance(value, VariableNode):
            name = self.metal_atomic_fence_operand_identifier(value)
            if name in candidate_names:
                use = (
                    "fence"
                    if fence_operand and not getattr(value, "vtype", None)
                    else "other"
                )
                uses[name].add(use)
            return
        if not hasattr(value, "__dict__"):
            return
        for key, child in vars(value).items():
            if key in {"parent", "annotations", "source_location"}:
                continue
            self.record_metal_atomic_fence_constant_uses(
                child, candidate_names, uses, fence_operand
            )

    def generate(self, ast):
        """Render ``ast`` as a CrossGL program.

        Raises :class:`MetalImportRefused` listing every Metal construct CrossGL cannot represent faithfully, and
        :class:`.external_functions.MetalExternalFunctionError` for a call to a function the unit only declares, and
        :class:`.namespace_scope_variables.MetalNamespaceLookupError` for an unqualified use of a variable name that
        several namespaces declare and lookup cannot bind to one of them.
        """
        resolve_namespace_scope_variables(ast)
        refuse_external_function_calls(ast)
        declare_referenced_library_structs(ast, parse_metal_structs)
        mark_const_cast_written_buffers(ast)
        import_function_constant_defaults(ast)
        return self.refusals.run(self.render_program, ast)

    def render_program(self, ast):
        wide_vector_support_marker = "    // __crossgl_metal_wide_vector_support__\n"
        precise_math_support_marker = "    // __crossgl_metal_precise_math_support__\n"
        small_vector_index_support_marker = (
            "    // __crossgl_metal_small_vector_index_support__\n"
        )
        cooperative_matrix_fragment_support_marker = (
            "    // __crossgl_metal_cooperative_matrix_fragment_support__\n"
        )
        local_aggregate_support_marker = (
            "    // __crossgl_metal_function_local_aggregates__\n"
        )
        self.reset_program_render_state()
        self.plan_metal_post_tessellation_functions(ast)
        typedefs, functions = self.prepare_program_declarations(ast)
        code = self.render_program_header(ast, wide_vector_support_marker)
        structs, enums, code = self.render_program_type_declarations(
            ast, code, functions, typedefs
        )

        code = self.render_program_enums_and_structs(code, enums, structs)
        code = self.render_program_globals(
            ast,
            code,
            cooperative_matrix_fragment_support_marker,
            local_aggregate_support_marker,
            precise_math_support_marker,
            small_vector_index_support_marker,
        )

        code = self.render_program_functions(code, functions)

        return self.finish_program_render(
            code,
            cooperative_matrix_fragment_support_marker,
            local_aggregate_support_marker,
            precise_math_support_marker,
            small_vector_index_support_marker,
            wide_vector_support_marker,
        )

    def constructor_member_array_element(
        self,
        element_type,
        declared_element_type,
        expression,
    ):
        if isinstance(expression, InitializerListNode):
            return expression

        expression_owner = None
        expression_arguments = None
        if isinstance(expression, VectorConstructorNode):
            expression_owner = expression.type_name
            expression_arguments = list(expression.args)
        elif isinstance(expression, FunctionCallNode):
            expression_owner = expression.name
            expression_arguments = list(expression.args)
        elif isinstance(expression, CastNode):
            expression_owner = expression.target_type
            expression_arguments = [expression.expression]
        if expression_owner is not None:
            expression_identity = self.metal_source_overload_type_identity(
                expression_owner
            )
            element_identity = self.metal_source_overload_type_identity(element_type)
            declared_identity = self.metal_source_overload_type_identity(
                declared_element_type
            )
            if expression_identity == element_identity:
                return expression
            if expression_identity == declared_identity:
                value = VectorConstructorNode(element_type, expression_arguments)
                value.source_location = getattr(expression, "source_location", None)
                return value

        expression_type = self.metal_source_overload_value_type(
            self.expression_metal_type(expression)
        )
        if expression_type is not None and (
            self.metal_source_overload_type_identity(expression_type)
            == self.metal_source_overload_type_identity(element_type)
        ):
            return expression
        if (
            self.metal_scalar_arithmetic_type_info(element_type) is not None
            or self.metal_constructor_contract(element_type) is not None
            or self.metal_vector_component_parts(element_type) is not None
            or self.metal_matrix_type_parts(element_type) is not None
            or self.wide_vector_type_info(
                element_type, getattr(expression, "source_location", None)
            )
            is not None
        ):
            value = VectorConstructorNode(element_type, [expression])
            value.source_location = getattr(expression, "source_location", None)
            return value
        return expression

    def generate_function(
        self,
        func,
        indent=2,
        stage_entry=False,
        template_value_bindings=None,
        template_type_bindings=None,
        output_name=None,
        specialization_key=None,
    ):
        """Render one Metal function node as a CrossGL function block."""
        code = ""
        if stage_entry:
            code += "    " * indent
            code += "@ stage_entry\n"
        linkage_annotations = []
        if not getattr(func, "qualifier", None):
            qualifiers = set(getattr(func, "declaration_qualifiers", ()) or ())
            if getattr(func, "internal_linkage", False) or getattr(
                func, "is_metal_constructor_factory", False
            ):
                qualifiers.add("static")
            if (
                "constexpr" in qualifiers
                or getattr(func, "template_parameters", None)
                or getattr(func, "generics", None)
            ):
                qualifiers.add("inline")
            for qualifier in ("static", "inline"):
                if qualifier in qualifiers:
                    linkage_annotations.append(f"@metal_{qualifier}")
            specialization = explicit_specialization_annotation(self, func)
            if specialization is not None:
                linkage_annotations.append(specialization)
        if getattr(self, "no_contraction_scope", False):
            linkage_annotations.append("@precise")
        code += "    " * indent
        implicit_buffer_bindings = (
            self.apply_implicit_stage_entry_buffer_bindings(func) if stage_entry else []
        )
        previous_variable_types = self.current_variable_types
        self.current_variable_types = dict(self.global_variable_types)
        previous_variable_type_qualifiers = self.current_variable_type_qualifiers
        self.current_variable_type_qualifiers = dict(
            self.global_variable_type_qualifiers
        )
        previous_type_aliases = dict(self.type_aliases)
        previous_type_alias_qualifiers = dict(self.type_alias_qualifiers)
        previous_type_alias_pointee_qualifiers = dict(
            self.type_alias_pointee_qualifiers
        )
        previous_local_type_alias_names = set(self.local_type_alias_names)
        previous_unresolved_local_type_aliases = dict(
            self.unresolved_local_type_aliases
        )
        previous_local_struct_type_aliases = dict(self.local_struct_type_aliases)
        previous_local_integral_constant_bindings = (
            self.local_integral_constant_bindings
        )
        self.local_integral_constant_bindings = {}
        previous_metal_enum_arithmetic_types = self.metal_enum_arithmetic_types
        self.metal_enum_arithmetic_types = dict(previous_metal_enum_arithmetic_types)
        previous_metal_enum_member_types = self.metal_enum_member_types
        self.metal_enum_member_types = dict(previous_metal_enum_member_types)
        previous_storage_texture_names = self.current_storage_texture_names
        self.current_storage_texture_names = set(self.global_storage_texture_names)
        previous_structured_buffer_names = self.current_structured_buffer_names
        self.current_structured_buffer_names = set(self.global_structured_buffer_names)
        previous_stage_entry_resource_parameter_ids = (
            self.current_stage_entry_resource_parameter_ids
        )
        previous_function_name = self.current_function_name
        previous_function_return_type = self.current_function_return_type
        previous_function = self.current_function
        previous_type_resolution_context = self.current_type_resolution_context
        previous_function_specialization = self.current_function_specialization
        previous_function_materialization_bindings = (
            self.current_function_materialization_bindings
        )
        previous_function_specialization_key = self.current_function_specialization_key
        previous_constructor_scope_index = self.current_constructor_scope_index
        out_of_line_replacement = self.out_of_line_call_operator_replacements.get(
            id(func)
        )
        function_body = (
            getattr(out_of_line_replacement["definition"], "body", None)
            if out_of_line_replacement is not None
            else func.body
        )
        self.current_function_name = func.name
        self.current_function_return_type = func.return_type
        self.current_function = func
        self.current_type_resolution_context = func
        self.current_stage_entry_resource_parameter_ids = (
            {id(param) for param in func.params} if stage_entry else set()
        )
        active_value_bindings = dict(template_value_bindings or {})
        active_type_bindings = dict(template_type_bindings or {})
        self.type_aliases.update(active_type_bindings)
        if active_value_bindings:
            self.template_value_bindings.append(active_value_bindings)
        if active_type_bindings:
            self.template_type_bindings.append(active_type_bindings)
        self.template_binding_shadow_scopes.append(set())
        self.push_identifier_scope()
        previous_parameter_scope = getattr(self, "current_parameter_scope", None)
        previous_elided_parameter_uses = getattr(
            self, "current_elided_parameter_uses", set()
        )
        self.current_parameter_scope = self.identifier_maps[-1]
        self.current_elided_parameter_uses = set()
        self.current_constructor_scope_index = (
            len(self.identifier_maps) - 1
            if getattr(func, "is_metal_constructor_factory", False)
            else None
        )
        try:
            code = self.render_function_definition(
                func,
                indent,
                stage_entry,
                output_name,
                specialization_key,
                active_type_bindings,
                active_value_bindings,
                code,
                function_body,
                linkage_annotations,
                out_of_line_replacement,
            )
        finally:
            self.current_parameter_scope = previous_parameter_scope
            self.current_elided_parameter_uses = previous_elided_parameter_uses
            for param, attributes in implicit_buffer_bindings:
                param.attributes = attributes
            self.pop_identifier_scope()
            self.template_binding_shadow_scopes.pop()
            if active_type_bindings:
                self.template_type_bindings.pop()
            if active_value_bindings:
                self.template_value_bindings.pop()
            self.current_variable_types = previous_variable_types
            self.current_variable_type_qualifiers = previous_variable_type_qualifiers
            self.type_aliases = previous_type_aliases
            self.type_alias_qualifiers = previous_type_alias_qualifiers
            self.type_alias_pointee_qualifiers = previous_type_alias_pointee_qualifiers
            self.local_type_alias_names = previous_local_type_alias_names
            self.unresolved_local_type_aliases = previous_unresolved_local_type_aliases
            self.local_struct_type_aliases = previous_local_struct_type_aliases
            self.local_integral_constant_bindings = (
                previous_local_integral_constant_bindings
            )
            self.metal_enum_arithmetic_types = previous_metal_enum_arithmetic_types
            self.metal_enum_member_types = previous_metal_enum_member_types
            self.current_storage_texture_names = previous_storage_texture_names
            self.current_structured_buffer_names = previous_structured_buffer_names
            self.current_stage_entry_resource_parameter_ids = (
                previous_stage_entry_resource_parameter_ids
            )
            self.current_function_name = previous_function_name
            self.current_function_return_type = previous_function_return_type
            self.current_function = previous_function
            self.current_type_resolution_context = previous_type_resolution_context
            self.current_function_specialization = previous_function_specialization
            self.current_function_materialization_bindings = (
                previous_function_materialization_bindings
            )
            self.current_function_specialization_key = (
                previous_function_specialization_key
            )
            self.current_constructor_scope_index = previous_constructor_scope_index
        return code

    def generate_expression(self, expr, is_main=False):
        """Render a Metal backend expression node as CrossGL syntax."""
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return self.normalize_literal_string(expr)
        elif isinstance(expr, VariableNode):
            if expr.vtype:
                return self.format_decl(expr, include_semantic=False)
            else:
                sparse_reference = refuse_sparse_color_reference(self, expr)
                if sparse_reference is not None:
                    return sparse_reference
                constant = self.render_integral_constant_binding(expr.name)
                if constant is not None:
                    return constant
                return self.render_identifier(expr.name)
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr, is_main)
        elif isinstance(expr, BinaryOpNode):
            return self.generate_binary_expression(expr, is_main)
        elif isinstance(expr, FunctionCallNode):
            return self.generate_function_call_expression(expr, is_main)
        elif isinstance(expr, LambdaNode):
            return self.generate_lambda_expression(expr, is_main)
        elif isinstance(expr, CallNode):
            return self.generate_call_expression(expr, is_main)
        elif isinstance(expr, MethodCallNode):
            return self.generate_method_call_expression(expr, is_main)
        elif isinstance(expr, MemberAccessNode):
            return self.generate_member_access_expression(expr, is_main)
        elif isinstance(expr, ArrayAccessNode):
            return self.generate_array_access_expression(expr, is_main)
        elif isinstance(expr, UnaryOpNode):
            return self.generate_unary_expression(expr, is_main)
        elif isinstance(expr, PostfixOpNode):
            return self.generate_postfix_expression(expr, is_main)
        elif isinstance(expr, TernaryOpNode):
            return self.generate_ternary_expression(expr, is_main)
        elif isinstance(expr, CastNode):
            return self.generate_cast_expression(expr, is_main)
        elif isinstance(expr, VectorConstructorNode):
            return self.generate_vector_constructor_expression(expr, is_main)
        elif isinstance(expr, InitializerListNode):
            return self.generate_initializer_list(
                expr,
                is_main,
                getattr(expr, "_metal_source_overload_expected_type", None),
            )
        elif isinstance(expr, DesignatedInitializerNode):
            return self.generate_designated_initializer(expr, is_main)
        elif isinstance(expr, TextureSampleNode):
            return self.generate_texture_sample_expression(expr, is_main)
        elif isinstance(expr, float) or isinstance(expr, int) or isinstance(expr, bool):
            return str(expr)
        else:
            return self.refuse_import(
                f"Metal expression of kind {type(expr).__name__} has no CrossGL expression form",
                "0",
            )

    @staticmethod
    def lowered_static_method_is_instance_helper(function):
        params = list(getattr(function, "params", []) or [])
        return bool(
            params
            and getattr(params[0], "name", None) == "self"
            and MetalToCrossGLConverter.reference_parameter(params[0])
        )

    def resolve_metal_free_operator(
        self, operator, arguments, source_location=None, argument_metal_types=None
    ):
        """Select the free operator overload for ``arguments``.

        ``argument_metal_types``, when given, are the ``expression_metal_type`` of
        each argument, so that a caller that needs them too infers each operand
        once. Inferring them twice per binary operator doubled the work at every
        level of a nested expression.
        """
        candidates = self.metal_free_operator_candidates(operator, len(arguments))
        if not candidates:
            return None
        if argument_metal_types is None:
            argument_metal_types = [
                self.expression_metal_type(argument) for argument in arguments
            ]
        argument_types = [
            self.metal_source_overload_value_type(argument_type)
            for argument_type in argument_metal_types
        ]
        if any(argument_type is None for argument_type in argument_types):
            return None
        if (
            len(argument_types) == 2
            and self.metal_builtin_pointer_arithmetic_operand_index(
                operator, argument_types[0], argument_types[1]
            )
            is not None
        ):
            # Pointer +/- integral is a native built-in expression even when the
            # pointee is an aggregate. Unrelated aggregate value overloads are
            # not candidates for this operation and must not intercept it.
            return None
        if operands_have_no_class_type(self, arguments, argument_types):
            # [over.match.oper]/1: the built-in operator (builtin_operator_operands).
            return None
        ranked = []
        for function in candidates:
            ranks = []
            for argument, argument_type, parameter in zip(
                arguments,
                argument_types,
                getattr(function, "params", []) or [],
            ):
                rank = self.metal_source_overload_argument_match_rank(
                    argument,
                    argument_type,
                    parameter,
                )
                if rank is None:
                    break
                ranks.append(rank)
            else:
                ranked.append((tuple(ranks), function))
        if not ranked:
            if (
                self.metal_builtin_lowered_conversion_plan(
                    operator,
                    arguments,
                    argument_types,
                )
                is not None
            ):
                return None
            aggregate_arguments = [
                argument_type
                for argument_type in argument_types
                if self.normalized_metal_type(self.resolve_type_alias(argument_type))
                in self.struct_member_types
            ]
            if aggregate_arguments:
                raise MetalSourceOverloadResolutionError(
                    f"operator {operator}",
                    [argument_type or "<unknown>" for argument_type in argument_types],
                    [
                        self.metal_function_candidate_signature(item)
                        for item in candidates
                    ],
                    "no source-compatible materialized free operator matches the "
                    "inferred argument types",
                    source_location,
                )
            return None

        def dominates(left, right):
            return all(a >= b for a, b in zip(left, right)) and any(
                a > b for a, b in zip(left, right)
            )

        winners = [
            entry
            for entry in ranked
            if not any(
                other is not entry and dominates(other[0], entry[0]) for other in ranked
            )
        ]
        if len(winners) != 1:
            raise MetalSourceOverloadResolutionError(
                f"operator {operator}",
                [argument_type or "<unknown>" for argument_type in argument_types],
                [
                    self.metal_function_candidate_signature(function)
                    for _ranks, function in winners
                ],
                "multiple source-compatible materialized free operators remain "
                "after type matching",
                source_location,
            )
        return winners[0][1]

    def generate_metal_power_call(self, expression, is_main=False):
        mode = self.metal_math_builtin_namespace_mode(expression.name)
        if mode == "fast" or str(expression.name).rsplit("::", 1)[-1] != "pow":
            return None
        # ``precise::pow`` promises 16 ulp (MSL Table 7.2), which no target's own ``pow`` does: GLSL, HLSL and
        # SPIR-V inherit the error of ``exp2(y * log2(x))``. Its finite path therefore always runs the portable
        # binary32 power over the whole normal range, while an unqualified ``pow`` keeps the target's unless the
        # portable-finite profile is selected.
        precise = mode == "precise"
        selected = self.selected_metal_callable(expression)
        bfloat_wrapper = self.is_materialized_metal_stdlib_wrapper(selected) and (
            self.normalized_metal_type(
                self.resolve_type_alias(
                    self.selected_metal_callable_return_type(selected)
                )
            )
            in self.metal_source_bfloat_types
        )
        if bfloat_wrapper:
            result_type = "float"
        elif (
            self.resolve_metal_math_builtin_name(expression.name, expression.args)
            == "pow"
        ):
            result_type = self.metal_math_builtin_result_type(expression)
        else:
            return None
        info = self.metal_math_builtin_type_info(result_type)
        if info is None:
            return None
        element = self.normalized_metal_type(
            self.resolve_type_alias(info["element_type"])
        )
        if element not in {"float", "half"} | self.metal_source_bfloat_types:
            return None
        width = info["width"]
        if width not in {1, 2, 3, 4}:
            return None
        if self.current_function is None:
            raise MetalPreciseMathLoweringError(
                "pow",
                result_type,
                "global initializers cannot call the runtime power helper",
                getattr(expression, "source_location", None),
            )
        (
            self.required_metal_precise_power_widths
            if precise
            else self.required_metal_power_widths
        ).add(width)
        mapped = "float" if width == 1 else f"vec{width}"
        parameters = list(getattr(selected, "params", []) or [])
        arguments = []
        for index, argument in enumerate(expression.args):
            value = self.generate_expression(argument, is_main)
            parameter_type = (
                self.metal_source_overload_parameter_type(parameters[index])
                if bfloat_wrapper
                else result_type
            )
            argument_type = self.expression_metal_type(argument)
            if self.metal_source_overload_type_identity(
                argument_type
            ) != self.metal_source_overload_type_identity(parameter_type):
                value = f"{self.map_type(parameter_type)}({value})"
            arguments.append(f"{mapped}({value})")
        result = (
            f"{self.metal_power_helper_name(width, precise)}({', '.join(arguments)})"
        )
        if element != "float":
            result = f"{self.map_type(result_type)}({result})"
        return result

    def metal_power_helper_name(self, width, precise=False):
        suffix = "" if width == 1 else str(width)
        if precise:
            return self.metal_precise_math_unique_helper_name(
                f"precise-power-float{suffix}",
                f"__crossgl_metal_precise_power_float{suffix}",
            )
        return self.metal_precise_math_unique_helper_name(
            f"power-float{suffix}", f"__crossgl_metal_power_float{suffix}"
        )

    def generate_metal_power_support_code(self, indent=0):
        variants = [
            (precise, widths)
            for precise, widths in (
                (False, self.required_metal_power_widths),
                (True, self.required_metal_precise_power_widths),
            )
            if widths
        ]
        if not variants:
            return ""
        code = ""
        licensed = False
        for precise, widths in variants:
            positive = "pow"
            if precise:
                positive = self.metal_precise_math_unique_helper_name(
                    "precise-power-positive-float",
                    "__crossgl_metal_precise_power_positive_float",
                )
                code += binary32_power_support(
                    positive, boundaries="computed", notice=not licensed
                )
                licensed = True
            elif self.binary32_power_accuracy_profile == "portable-finite":
                positive = self.metal_precise_math_unique_helper_name(
                    "power-positive-float", "__crossgl_metal_power_positive_float"
                )
                code += binary32_power_support(positive, notice=not licensed)
                licensed = True
            code += self.metal_power_wrapper_code(precise, widths, positive)
        pad = "    " * indent
        return "".join(pad + line + "\n" for line in code.strip().splitlines())

    def metal_power_wrapper_code(self, precise, widths, positive):
        """The IEEE special cases of ``pow`` around ``positive``, which computes finite positive powers."""
        scalar = self.metal_power_helper_name(1, precise)
        operand_policy = """    // Keep existing subnormal behavior until a source policy is selected.
    if ((magnitude != 0u && magnitude < 0x00800000u) ||
        (power != 0u && power < 0x00800000u)) {
        return pow(base, exponent);
    }
"""
        if self.binary32_power_operand_profile == "flush-subnormals":
            operand_policy = """    // Apply the selected operand policy after the exact identity shortcut.
    if (magnitude != 0u && magnitude < 0x00800000u) {
        a = a & 0x80000000u;
        base = asfloat(a);
        magnitude = 0u;
    }
    if (power != 0u && power < 0x00800000u) {
        b = b & 0x80000000u;
        exponent = asfloat(b);
        power = 0u;
    }
"""
        code = f"""@metal_static
float {scalar}(float base, float exponent) {{
    uint a = asuint(base);
    uint b = asuint(exponent);
    uint magnitude = a & 0x7fffffffu;
    uint power = b & 0x7fffffffu;
    // Preserve the identity without passing subnormals through arithmetic.
    if (b == 0x3f800000u) {{
        return base;
    }}
{operand_policy}    if (power == 0u || a == 0x3f800000u) {{
        return 1.0;
    }}
    if (magnitude > 0x7f800000u || power > 0x7f800000u) {{
        return asfloat(0x7fc00000u);
    }}
    if (power == 0x7f800000u) {{
        if (magnitude == 0x3f800000u) {{
            return 1.0;
        }}
        bool overflow = (magnitude > 0x3f800000u) != ((b >> 31u) != 0u);
        return asfloat(overflow ? 0x7f800000u : 0u);
    }}
    // At and above 2^24 every finite binary32 integer is even.
    bool integral = power >= 0x4b800000u;
    bool odd = false;
    if (power >= 0x3f800000u && power < 0x4b800000u) {{
        uint shift = 150u - (power >> 23u);
        uint significand = 0x800000u | (power & 0x7fffffu);
        integral = (significand & ((1u << shift) - 1u)) == 0u;
        odd = integral && ((significand >> shift) & 1u) != 0u;
    }}
    uint sign = odd ? (a & 0x80000000u) : 0u;
    if (magnitude == 0u || magnitude == 0x7f800000u) {{
        bool overflow = (magnitude != 0u) != ((b >> 31u) != 0u);
        return asfloat(sign | (overflow ? 0x7f800000u : 0u));
    }}
    if ((a >> 31u) != 0u && !integral) {{
        return asfloat(0x7fc00000u);
    }}
    float result = {positive}(asfloat(magnitude), exponent);
    return asfloat(asuint(result) | sign);
}}
"""
        for width in sorted(widths - {1}):
            arguments = ", ".join(
                f"{scalar}(base.{lane}, exponent.{lane})" for lane in "xyzw"[:width]
            )
            code += (
                f"@metal_static\nvec{width} {self.metal_power_helper_name(width, precise)}"
                f"(vec{width} base, vec{width} exponent) {{\n"
                f"    return vec{width}({arguments});\n}}\n"
            )
        return code

    def map_semantic(self, semantic, *, context=None):
        """Map Metal attributes to CrossGL semantic annotation syntax."""
        if not semantic:
            return ""

        if isinstance(context, dict):
            context_kind = context.get("kind")
            function_qualifier = str(context.get("function_qualifier") or "").lower()
        else:
            context_kind = context
            function_qualifier = ""

        outputs = []
        attr_names = {
            str(getattr(attr, "name", "")).lower()
            for attr in semantic
            if isinstance(attr, AttributeNode)
        }
        has_barycentric_coord = "barycentric_coord" in attr_names
        no_perspective_barycentric = has_barycentric_coord and any(
            "no_perspective" in attr_name for attr_name in attr_names
        )
        for attr in semantic:
            if not isinstance(attr, AttributeNode):
                continue
            name = attr.name
            args = (
                [self.format_metadata_argument(a) for a in attr.args]
                if attr.args
                else []
            )
            key = f"{name}({args[0]})" if args else name
            if (
                context_kind == "parameter"
                and function_qualifier == "fragment"
                and name == "position"
            ):
                out = "gl_FragCoord"
            elif name == "barycentric_coord":
                out = (
                    "gl_BaryCoordNoPerspEXT"
                    if no_perspective_barycentric
                    else "gl_BaryCoordEXT"
                )
            elif has_barycentric_coord and "no_perspective" in str(name).lower():
                continue
            elif context_kind == "parameter" and name == "sample_mask":
                out = "gl_SampleMaskIn"
            elif name == "attribute" and len(args) == 1:
                # A vertex attribute index is the input location every target binds by.
                out = f"location({args[0]})"
            elif (
                context_kind == "parameter"
                and name == "raster_order_group"
                and "color" not in attr_names
            ):
                # A buffer or texture argument in any group orders its accesses between overlapping fragments.
                # CrossGL has one such order (metal_raster_order_groups.py), which orders at least what the Metal
                # groups order. A colour attachment input keeps its group with its imageblock data.
                out = "rasterizer_ordered"
            else:
                out = self.map_semantics.get(key, self.map_semantics.get(name, None))
            if out is None:
                out = self.dynamic_fragment_output_semantic(
                    name, args, context_kind=context_kind
                )
            if out is None:
                if args:
                    out = f"{name}({', '.join(args)})"
                else:
                    out = name
            if out:
                outputs.append(f"@{out}")
        return " ".join(outputs)

    def format_metadata_argument(self, arg):
        """Render a Metal attribute argument as CrossGL metadata syntax."""
        text = str(arg).strip()
        while text.startswith("::"):
            text = text[2:].lstrip()
        return text

    def dynamic_fragment_output_semantic(self, name, args, context_kind=None):
        if name == "color" and args:
            return color_attachment_semantic(
                args[0],
                is_input=context_kind in ("parameter", "fetched_member"),
                enumerator_values=getattr(self, "metal_enum_index_values", {}),
            )
        if name == "depth" and args and args[0].lower() in {"any", "less", "greater"}:
            # MSL 3.2 section 5.2.3.5: depth(greater) and depth(less) promise the direction the shader moves
            # the depth in. CrossGL states that promise as a conservative depth layout next to gl_FragDepth.
            layout = args[0].lower()
            return (
                "gl_FragDepth" if layout == "any" else f"gl_FragDepth @depth_{layout}"
            )
        return None
