"""
CrossGL to HIP Code Generator

This module provides code generation functionality to convert CrossGL AST to HIP source code.
HIP (Heterogeneous-Compute Interface for Portability) is AMD's CUDA-compatible runtime API
for GPU programming.
"""

from ..ast import (
    ASTNode,
    ConstructorNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    MatchNode,
    MemberAccessNode,
    PointerAccessNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ShaderNode,
    VariableNode,
    WaveOpNode,
)
from ..refusals import HipRefusal
from ..semantics.declared_types import declared_type_names
from ..semantics.structured_buffer_counters import (
    materialize_structured_buffer_counters,
)
from ..unhandled import UNHANDLED
from .barrier_peephole import merge_adjacent_cuda_barriers
from .comma_expressions import (
    CommaExpressionVisitorMixin,
    comma_result_operand,
    is_comma_expression,
)
from .cooperative_matrix_refusals import refuse_cooperative_matrices
from .cuda_library_functions import rename_cuda_library_functions
from .enum_utils import infer_enum_constructor_type
from .float_remainder_assignment import float_remainder_assignment
from .generic_function_utils import (
    generic_function_call_name,
    iter_function_nodes,
    numeric_trait_method_result_type,
    prepare_generic_function_specializations,
)
from .generic_global_types import collect_constant_and_cbuffer_member_types
from .generic_struct_utils import infer_struct_constructor_type
from .hip_bindings import HipBindingMixin
from .hip_buffer_calls import HipBufferCallMixin
from .hip_buffer_metadata import HipBufferMetadataMixin
from .hip_declarations import HipDeclarationMixin
from .hip_expression_types import HipExpressionTypeMixin
from .hip_expressions import HipExpressionMixin
from .hip_function_calls import HipFunctionCallMixin
from .hip_function_parts import HipFunctionPartsMixin
from .hip_math_calls import HipMathCallMixin
from .hip_refusals import HipLoweringRefused
from .hip_resource_calls import HipResourceCallMixin
from .hip_semantics import HipSemanticMixin
from .hip_stage_attributes import HipStageAttributeMixin
from .hip_stage_checks import HipStageCheckMixin
from .hip_statements import HipStatementMixin
from .hip_tables import (
    HIP_64BIT_BITCAST_FUNCTIONS,
    HIP_BITCAST_FUNCTION_TARGETS,
    HIP_BUILTIN_MAP,
    HIP_FP16_VECTOR_TYPES,
    HIP_FUNCTION_MAP,
    HIP_INTEGER_BIT_FUNCTION_ALIASES,
    HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES,
    HIP_TYPE_MAP,
    HIP_WARP_SYNC_BUILTIN_ARITIES,
    HIP_WAVE_BOOL_RESULT_OPS,
    HIP_WAVE_OP_ARITIES,
    HIP_WAVE_PREDICATE_ARGUMENT_OPS,
    HIP_WAVE_UINT_RESULT_OPS,
    HIP_WAVE_UVEC4_RESULT_OPS,
)
from .hip_textures import HipTextureMixin
from .hip_types import HipTypeMixin
from .hip_vector_types import HipVectorTypeMixin
from .hip_visitor import HipVisitorMixin
from .hip_wave_ops import HipWaveOpMixin
from .lowering_refusals import LoweringRefusalMixin
from .match_utils import infer_match_expression_result_type
from .precise_evaluation import PreciseEvaluationMixin
from .relational_builtins import lower_relational_builtins
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .sampler_pairs import canonicalize_separate_textures
from .stage_utils import is_fragment_output_parameter
from .value_array_parameters import (
    collect_function_parameters_by_name,
    collect_struct_names,
)
from .vector_arithmetic import VectorArithmeticMixin
from .written_function_names import (
    CUDA_CALLABLE_FUNCTION_NAMES,
    refuse_written_foreign_calls,
)
from .written_type_names import refuse_written_foreign_types

# Most generator members live in the hip_*.py mixin modules next to this one. These names stay importable from here.
__all__ = [
    "HIP_64BIT_BITCAST_FUNCTIONS",
    "HIP_BITCAST_FUNCTION_TARGETS",
    "HIP_FP16_VECTOR_TYPES",
    "HIP_INTEGER_BIT_FUNCTION_ALIASES",
    "HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES",
    "HIP_WARP_SYNC_BUILTIN_ARITIES",
    "HIP_WAVE_BOOL_RESULT_OPS",
    "HIP_WAVE_OP_ARITIES",
    "HIP_WAVE_PREDICATE_ARGUMENT_OPS",
    "HIP_WAVE_UINT_RESULT_OPS",
    "HIP_WAVE_UVEC4_RESULT_OPS",
    "HipCodeGen",
    "UnsupportedHIPAstNodeError",
    "generate_hip_code",
]


def _refuse_buffer_counter(reason):
    raise HipRefusal(f"HIP buffer counter: {reason}")


class UnsupportedHIPAstNodeError(ValueError):
    """Raised when HIP codegen cannot safely lower an AST node."""

    project_diagnostic_code = "project.translate.unsupported-feature"
    missing_capabilities = ("hip.ast-node-lowering",)

    def __init__(self, node):
        self.node_type = type(node).__name__
        self.feature = f"hip.ast-node.{self.node_type}"
        self.source_location = getattr(node, "source_location", None)
        self.suggested_action = (
            f"Implement visit_{self.node_type} with semantics-preserving HIP lowering."
        )
        super().__init__(
            f"HIP code generation does not support AST node {self.node_type}. "
            f"Suggested action: {self.suggested_action}"
        )


class HipCodeGen(
    HipStageCheckMixin,
    HipVisitorMixin,
    HipSemanticMixin,
    HipStageAttributeMixin,
    HipDeclarationMixin,
    HipStatementMixin,
    HipExpressionMixin,
    HipFunctionCallMixin,
    HipWaveOpMixin,
    HipMathCallMixin,
    HipVectorTypeMixin,
    HipTextureMixin,
    HipBufferCallMixin,
    HipBufferMetadataMixin,
    HipTypeMixin,
    HipBindingMixin,
    HipExpressionTypeMixin,
    LoweringRefusalMixin,
    PreciseEvaluationMixin,
    CommaExpressionVisitorMixin,
    VectorArithmeticMixin,
    ResourceQueryMixin,
    ResourceDiagnosticMixin,
    HipResourceCallMixin,
    HipFunctionPartsMixin,
):
    """Emit HIP source from the shared CrossGL translator AST."""

    LOWERING_REFUSAL_TYPE = HipLoweringRefused
    # Clang disables contraction for the compound statement the pragma starts; see precise_evaluation.
    PRECISE_BODY_PRAGMA = "#pragma clang fp contract(off)"
    PRECISE_TARGET_LABEL = "HIP"

    resource_diagnostic_backend = "HIP"
    struct_constructor_uses_braces = True
    synchronization_builtins = {
        "barrier",
        "groupMemoryBarrier",
        "memoryBarrier",
        "memoryBarrierShared",
        "memoryBarrierBuffer",
        "memoryBarrierImage",
        "allMemoryBarrier",
        "deviceMemoryBarrier",
        "workgroupBarrier",
        "workgroupExecutionBarrier",
    }
    sampled_resource_type_aliases = {
        "Texture1D": "sampler1D",
        "Texture1DArray": "sampler1DArray",
        "Texture2D": "sampler2D",
        "Texture2DArray": "sampler2DArray",
        "Texture2DMS": "sampler2DMS",
        "Texture2DMSArray": "sampler2DMSArray",
        "Texture3D": "sampler3D",
        "TextureCube": "samplerCube",
        "TextureCubeArray": "samplerCubeArray",
    }
    storage_resource_type_aliases = {
        "RWTexture1D": "image1D",
        "RWTexture1DArray": "image1DArray",
        "RWTexture2D": "image2D",
        "RWTexture2DArray": "image2DArray",
        "RWTexture2DMS": "image2DMS",
        "RWTexture2DMSArray": "image2DMSArray",
        "RWTexture3D": "image3D",
        "RWTextureCube": "imageCube",
        "RWTextureCubeArray": "imageCubeArray",
    }
    sampler_state_type_aliases = {
        "SamplerState": "sampler",
        "SamplerComparisonState": "sampler",
    }
    # The type names the program declares, and the foreign built-in type names written so far (``None`` outside
    # a rendering); see :func:`written_type_name`.
    program_type_names = frozenset()
    written_foreign_type_names = None

    def __init__(self):
        """Initialize HIP type maps and per-generation visitor state."""
        self.indent_level = 0
        self.code_lines = []
        self.current_function = None
        self.current_function_return_type = None
        self.variable_counter = 0
        self.match_temp_variable_index = 0
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.hip_resource_binding_cursors = {}
        self.hip_used_resource_bindings = {}
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.function_return_types = {}
        self.function_parameters_by_name = {}
        self.value_array_struct_names = frozenset()
        self.helper_functions = {}
        self.query_resource_names = set()
        self.query_metadata_function_params = {}
        self.query_functions_by_name = {}
        self.structured_buffer_length_names = set()
        self.structured_buffer_length_function_params = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_function_name = None
        self.current_stage_name = None
        self.hip_function_capture_params = {}
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0
        # Subscripts visited as the index of another subscript: reads, even on an assignment's left side.
        self.subscript_index_depth = 0
        self.stage_builtin_aliases = {}
        self.stage_builtin_alias_types = {}
        self.current_function_is_kernel_entry = False
        self.current_expression_expected_type = None
        self.structs_by_name = {}
        self.generic_enum_struct_definitions = {}
        self.generic_enum_specializations = {}
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.plain_enums = []
        self.struct_payload_enums = []
        self.enum_type_names = set()
        self.enum_struct_type_names = set()
        self.enum_struct_variant_fields = {}
        self.enum_variant_constructors = {}
        self.enum_variant_constructor_fields = {}
        self.enum_variant_constants = {}

        self.type_map = dict(HIP_TYPE_MAP)

        self.function_map = dict(HIP_FUNCTION_MAP)

        self.builtin_map = dict(HIP_BUILTIN_MAP)

    def generate(self, node):
        """Generate complete HIP source for a CrossGL AST, or raise :class:`HipLoweringRefused`.

        A construct with no faithful HIP lowering is refused; no program is returned while a refusal is
        pending, and a rendered program that still carries a ``/* unsupported ... */`` loss marker is refused.
        """
        refuse_cooperative_matrices(node, label="HIP", target="hip")
        return merge_adjacent_cuda_barriers(
            self.refusals.run(self.render_checked_program, node)
        )

    def render_checked_program(self, node):
        """Render the program and refuse each built-in type and function of another language it still names."""
        self.program_type_names = declared_type_names(node)
        self.written_foreign_type_names = set()
        try:
            output = self.render_program(node)
            refuse_written_foreign_types(output, self.written_foreign_type_names, "HIP")
            refuse_written_foreign_calls(output, CUDA_CALLABLE_FUNCTION_NAMES, "HIP")
        finally:
            self.written_foreign_type_names = None
        return output

    def render_program(self, node: ASTNode) -> str:
        """Render complete HIP source for a CrossGL AST."""
        # A HIP texture object is the image whatever sampler state a separate CrossGL texture is paired with.
        node = canonicalize_separate_textures(node)
        # A user function named like a device library function would redefine it or make its calls ambiguous.
        node = rename_cuda_library_functions(node)
        # GLSL's lessThan, equal, not and the other relational built-ins are the componentwise operators.
        lower_relational_builtins(node)
        # HIP has no hidden buffer counter, so a counted RWStructuredBuffer gets an explicit one.
        node = materialize_structured_buffer_counters(node, _refuse_buffer_counter)
        self.code_lines = []
        self.indent_level = 0
        self.validate_supported_stage_types(node)
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.hip_resource_binding_cursors = {}
        self.hip_used_resource_bindings = {}
        self.current_function_return_type = None
        self.match_temp_variable_index = 0
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.function_return_types = self.collect_function_return_types(node)
        self.function_parameters_by_name = collect_function_parameters_by_name(
            node, self.type_name_string
        )
        self.value_array_struct_names = collect_struct_names(node)
        self.helper_functions = {}
        # File-scope constants already folded, by name (cuda_constant_folding).
        self.folded_constants = {}
        # Workgroup variables declared at file scope because a stage function uses them (cuda_shared_memory).
        self.file_scope_shared_variable_ids = set()
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0
        # Subscripts visited as the index of another subscript: reads, even on an assignment's left side.
        self.subscript_index_depth = 0
        self.stage_builtin_aliases = {}
        self.stage_builtin_alias_types = {}
        self.current_function_is_kernel_entry = False
        self.current_stage_name = None
        self.hip_function_capture_params = {}
        self.current_expression_expected_type = None
        self.setup_enum_and_generic_metadata(node)
        self.generic_function_definitions = {}
        self.generic_function_definition_ordinals = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            list(iter_function_nodes(node)),
            collect_constant_and_cbuffer_member_types(self, node),
            ast=node,
        )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in generic_function_specializations.values()
            }
        )
        self.reject_unsupported_generic_functions(node)
        (
            self.query_resource_names,
            self.query_metadata_function_params,
        ) = self.collect_resource_query_requirements(node)
        (
            self.structured_buffer_length_names,
            self.structured_buffer_length_function_params,
        ) = self.collect_structured_buffer_length_requirements(node)
        self.query_functions_by_name = {
            getattr(func, "name", None): func
            for func in self.query_collect_functions(node)
        }
        self.query_functions_by_name = {
            name: func for name, func in self.query_functions_by_name.items() if name
        }
        self.reserve_explicit_hip_resource_bindings(node)

        self.add_includes()
        self.add_generated_code(self.generate_hip_matrix_type_helpers())
        self.add_line()
        if self.has_geometry_stage(node):
            self.add_generated_code(self.generate_hip_geometry_stream_helpers())
            self.add_line()
        if self.has_tessellation_stage(node):
            self.add_generated_code(self.generate_hip_tessellation_patch_helpers())
            self.add_line()
        if self.has_mesh_task_stage(node):
            self.add_generated_code(self.generate_hip_mesh_task_helpers())
            self.add_line()
        self.visit(node)
        self.insert_helper_functions()

        return "\n".join(self.code_lines)

    def visit(self, node: ASTNode) -> str:
        """Dispatch an AST node to its HIP visitor method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> str:
        """Fail closed when no semantics-preserving HIP visitor exists."""
        raise UnsupportedHIPAstNodeError(node)

    def visit_FunctionNode(self, node: FunctionNode) -> str:
        """Render a CrossGL function or compute entry point as HIP code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_accesses = self.image_resource_accesses.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        saved_glsl_buffer_block_accesses = self.glsl_buffer_block_accesses.copy()
        saved_glsl_buffer_block_layouts = self.glsl_buffer_block_layouts.copy()
        self.current_function = node.name
        saved_current_function_return_type = self.current_function_return_type
        saved_current_function_name = self.current_function_name
        saved_stage_builtin_aliases = self.stage_builtin_aliases
        saved_stage_builtin_alias_types = self.stage_builtin_alias_types
        saved_current_function_is_kernel_entry = self.current_function_is_kernel_entry
        saved_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        saved_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        self.current_function_name = node.name
        self.current_generic_function_substitutions = (
            getattr(node, "_generic_substitutions", {}) or {}
        )
        self.current_structured_buffer_length_parameters = {}
        self.stage_builtin_aliases = {}
        self.stage_builtin_alias_types = {}
        self.current_function_is_kernel_entry = False

        qualifiers = self.hip_function_qualifiers(node)

        stage_name = self.function_stage_name(node)
        is_kernel_entry = "__global__" in qualifiers
        self.current_function_is_kernel_entry = is_kernel_entry

        if hasattr(node, "return_type"):
            self.current_function_return_type = node.return_type
            return_type = self.map_type(node.return_type)
        else:
            self.current_function_return_type = "void"
            return_type = "void"

        param_list = list(getattr(node, "parameters", getattr(node, "params", [])))
        param_list.extend(self.hip_function_capture_params.get(node.name, []))
        return_semantic = self.semantic_from_node(node)
        fragment_output_param = None
        fragment_output_params = [
            param
            for param in param_list
            if is_fragment_output_parameter(
                stage_name,
                param,
                self.semantic_from_node(param),
            )
        ]
        if (
            fragment_output_params
            and return_semantic is None
            and self.map_type(self.current_function_return_type) == "void"
        ):
            fragment_output_param = fragment_output_params[0]
            self.current_function_return_type = self.get_parameter_type(
                fragment_output_param
            )
            return_type = self.map_type(self.current_function_return_type)
            return_semantic = self.semantic_from_node(fragment_output_param)
            param_list = [
                param for param in param_list if param is not fragment_output_param
            ]

        self.validate_hip_return_semantic(
            stage_name, self.current_function_return_type, return_semantic
        )
        self.validate_hip_struct_return_semantics(
            stage_name, self.current_function_return_type
        )
        if is_kernel_entry:
            self.current_function_return_type = "void"
            return_type = "void"

        self.validate_hip_stage_parameter_semantics(stage_name, param_list)
        if stage_name == "geometry":
            self.validate_hip_geometry_stage(node, param_list)
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_hip_tessellation_stage(node, stage_name)
        if stage_name in self.hip_mesh_task_stage_names():
            self.validate_hip_mesh_task_stage(node, stage_name)
        value_array_copy_statements, params = self.hip_function_parameter_declarations(
            node, param_list, stage_name
        )

        qualifier_str = " ".join(qualifiers)
        function_name = getattr(self, "current_stage_entry_function_name", None)
        if not function_name:
            function_name = node.name
        if is_kernel_entry and function_name == "main":
            function_name = "compute_main"
        launch_bounds = self.hip_compute_launch_bounds_attribute(
            node, stage_name, is_kernel_entry
        )
        if launch_bounds is not None:
            signature = (
                f"{qualifier_str} {return_type} {launch_bounds} "
                f"{function_name}({params})"
            )
        else:
            signature = f"{qualifier_str} {return_type} {function_name}({params})"
        body = getattr(node, "body", None)

        self.hip_function_stage_comments(node, param_list, return_semantic, stage_name)
        self.hip_function_body(
            node, body, fragment_output_param, signature, value_array_copy_statements
        )

        self.add_line()
        self.current_function = None
        self.current_function_return_type = saved_current_function_return_type
        self.variable_types = saved_variable_types
        self.image_resource_accesses = saved_image_resource_accesses
        self.buffer_resource_accesses = saved_buffer_resource_accesses
        self.glsl_buffer_block_accesses = saved_glsl_buffer_block_accesses
        self.glsl_buffer_block_layouts = saved_glsl_buffer_block_layouts
        self.current_function_name = saved_current_function_name
        self.stage_builtin_aliases = saved_stage_builtin_aliases
        self.stage_builtin_alias_types = saved_stage_builtin_alias_types
        self.current_function_is_kernel_entry = saved_current_function_is_kernel_entry
        self.current_generic_function_substitutions = (
            saved_generic_function_substitutions
        )
        self.current_structured_buffer_length_parameters = (
            saved_structured_buffer_length_parameters
        )
        return ""

    def format_variable_declaration(self, node: VariableNode) -> str:
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        var_type = self.get_variable_node_type(node)

        if var_type is None and initial_value is not None:
            inferred_type = self.expression_result_type(initial_value)
            self.register_variable_type(node.name, inferred_type, node, initial_value)
            declaration = self.format_typed_declarator(
                inferred_type or "auto", node.name
            )
        else:
            var_type = var_type or "int"
            self.register_variable_type(node.name, var_type, node, initial_value)
            var_type = self.glsl_buffer_block_declaration_type(var_type, node)
            declaration = self.format_typed_declarator(var_type, node.name)

        declaration = self.apply_resource_memory_qualifiers_to_declaration(
            declaration, node.name, node
        )
        if (
            initial_value is not None
            and self.declaration_qualifier_names(node) & {"const", "constant"}
            and "*" not in declaration
            and not declaration.startswith("const ")
        ):
            # A CrossGL constant, or an MSL program-scope ``constant`` value, stays one: file-scope device code
            # can read a const scalar with a constant initializer, but not a mutable host variable.
            declaration = f"const {declaration}"

        qualifiers = self.variable_memory_qualifiers(node, declaration)
        if qualifiers:
            declaration = f"{' '.join(qualifiers)} {declaration}"

        if initial_value is not None:
            declaration += f" = {self.visit(initial_value)}"

        return declaration

    def format_assignment_expression(self, node) -> str:
        operator = getattr(node, "operator", getattr(node, "op", "="))
        diagnostic = self.glsl_buffer_block_write_diagnostic(node.left, "assignment")
        if diagnostic is not None:
            return diagnostic
        if operator == "=":
            diagnostic = self.structured_buffer_element_write_diagnostic(
                node.left, "assignment"
            )
        else:
            diagnostic = self.structured_buffer_element_read_write_diagnostic(
                node.left, "assignment"
            )
        if diagnostic is not None:
            return diagnostic
        # C++ has no floating %=; x = x % y lowers the remainder through fmodf.
        remainder = float_remainder_assignment(
            node.left,
            node.right,
            operator,
            type_of=self.expression_result_type,
            is_floating=self.is_floating_value_type,
            backend="hip",
        )
        if remainder is not None:
            return self.format_assignment_expression(remainder)
        swizzle_assignment = self.format_swizzle_assignment_expression(
            node.left,
            node.right,
            operator,
        )
        if swizzle_assignment is not None:
            return swizzle_assignment
        self.assignment_lhs_depth += 1
        try:
            left = self.visit(node.left)
        finally:
            self.assignment_lhs_depth -= 1
        right = self.visit(node.right)
        compound_binary_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
        }
        if operator in compound_binary_ops:
            lowered_right = self.lower_vector_binary_operation(
                node.left,
                left,
                node.right,
                right,
                compound_binary_ops[operator],
            )
            if lowered_right is not None:
                return f"{left} = {lowered_right}"
            if operator == "%=":
                modulo = self.lower_scalar_modulo_operation(
                    node.left,
                    left,
                    node.right,
                    right,
                )
                if modulo is not None:
                    return f"{left} = {modulo}"
        compound_bitwise_ops = {
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }
        if operator in compound_bitwise_ops:
            lowered_right = self.lower_vector_bitwise_operation(
                node.left,
                left,
                node.right,
                right,
                compound_bitwise_ops[operator],
            )
            if lowered_right is not None:
                return f"{left} = {lowered_right}"
        return f"{left} {operator} {right}"

    def hip_half_constructor_expression(self, constructor_type, raw_args, args):
        """Lower CrossGL FP16 constructors to HIP's documented half intrinsics."""
        if constructor_type in {"f16", "half", "float16"}:
            if not args:
                return "half{}"
            return f"__float2half({args[0]})"

        if constructor_type not in {"vec2<f16>", "half2", "f16vec2"}:
            return None

        if not args:
            return "__float2half2_rn(0.0f)"

        if len(args) == 1:
            arg_type = self.type_name_string(self.expression_result_type(raw_args[0]))
            if arg_type in {"vec2<f16>", "half2", "f16vec2"}:
                return args[0]
            return f"__float2half2_rn({args[0]})"

        return f"__floats2half2_rn({args[0]}, {args[1]})"

    @property
    def local_variable_types(self):
        return self.variable_types

    @local_variable_types.setter
    def local_variable_types(self, value):
        self.variable_types = value

    def is_buffer_resource_type(self, type_name):
        """Return whether a type is a structured or byte-address buffer resource."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def vector_type_info(self, type_name):
        half_vector_info = self.hip_half_vector_type_info(type_name)
        if half_vector_info is not None:
            return half_vector_info
        narrow_info = self.hip_narrow_integer_vector_type_info(type_name)
        if narrow_info is not None:
            return narrow_info
        return super().vector_type_info(type_name)

    def vector_type_for_components(self, component_type, component_count):
        # half has no vecN spelling: f16vec3 and f16vec4 lower to the
        # cgl_half helpers and f16vec2 to the native half2
        if component_type == "half" and 2 <= component_count <= 4:
            return f"f16vec{component_count}"
        return super().vector_type_for_components(component_type, component_count)

    def generate_resource_call(self, func_name, raw_args, args):
        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, raw_args, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, raw_args, args)

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(raw_args)

        raw_args, args = self.normalize_split_sampler_texture_args(
            func_name, raw_args, args
        )

        handled = self.hip_texture_form_call(func_name, raw_args, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.hip_offset_texture_call(func_name, raw_args, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.hip_image_atomic_call(func_name, raw_args, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.hip_sampled_texture_call(func_name, raw_args, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.hip_texel_fetch_call(func_name, raw_args, args)
        if handled is not UNHANDLED:
            return handled

        handled = self.hip_storage_image_call(func_name, raw_args, args)
        if handled is not UNHANDLED:
            return handled

        return None

    def array_access_element_type(self, type_name):
        """Return the element type for HIP arrays and structured buffers."""
        array_element_type = super().array_access_element_type(type_name)
        if array_element_type is not None:
            return array_element_type

        indirect_info = self.hip_indirect_type_info(type_name)
        if indirect_info is not None and indirect_info["kind"] == "pointer":
            return indirect_info["pointee_type"]

        parts = self.structured_buffer_type_parts(type_name)
        if parts is not None:
            return parts[1]
        return None

    def resource_call_result_type(self, func_name, raw_args):
        if func_name in {"texelFetch", "texelFetchOffset"} and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(resource_type):
                return "float"
            if self.is_sampled_texture_family_type(resource_type):
                return self.sampled_texture_value_type(resource_type)

        if func_name in {
            "texture",
            "textureLod",
            "textureGrad",
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "texelFetch",
            "texelFetchOffset",
        }:
            resource_type = None
            if raw_args:
                resource_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
            if self.is_shadow_resource_type(resource_type):
                return "float"
            if self.is_sampled_texture_family_type(resource_type):
                return self.sampled_texture_value_type(resource_type)

        if func_name in {
            "textureProj",
            "textureProjLod",
            "textureProjGrad",
            "texelFetchOffset",
        }:
            resource_type = None
            if raw_args:
                resource_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
            if self.is_shadow_resource_type(resource_type):
                return "float"
            return self.sampled_texture_value_type(resource_type)
        return super().resource_call_result_type(func_name, raw_args)

    def expression_result_type(self, node):
        """Infer expression result types with HIP buffer operations."""
        if is_comma_expression(node):
            return self.expression_result_type(comma_result_operand(node))
        if isinstance(node, (IdentifierNode, VariableNode)):
            name = getattr(node, "name", None)
            if not self.source_identifier_shadows_builtin(name):
                builtin_type = self.hip_compute_builtin_type(name)
                if builtin_type is not None:
                    return builtin_type
        if isinstance(node, WaveOpNode):
            return self.wave_result_type(
                getattr(node, "operation", ""), getattr(node, "arguments", []) or []
            )
        if isinstance(node, RayTracingOpNode):
            if getattr(node, "operation", "") == "ReportHit":
                return "bool"
            return None
        if isinstance(node, RayQueryOpNode):
            operation = getattr(node, "operation", "")
            if operation == "Proceed":
                return "bool"
            if operation in {"CandidateTriangleRayT", "CommittedRayT"}:
                return "float"
            return "uint"
        if isinstance(node, FunctionCallNode):
            function_expr = getattr(node, "function", getattr(node, "name", None))
            func_name = getattr(function_expr, "name", function_expr)
            numeric_result_type = numeric_trait_method_result_type(self, node)
            if numeric_result_type is not None:
                return numeric_result_type
            raw_args = getattr(node, "arguments", getattr(node, "args", [])) or []
            specialized_func_name = generic_function_call_name(
                self,
                func_name,
                raw_args,
            )
            if specialized_func_name in self.function_return_types:
                return self.function_return_types[specialized_func_name]
            if self.is_user_defined_function(func_name):
                return self.function_return_types.get(func_name)
            if func_name in HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES and len(raw_args) <= 1:
                return func_name
            if func_name in HIP_WAVE_OP_ARITIES:
                return self.wave_result_type(
                    func_name,
                    raw_args,
                )
            if func_name == "ReportHit":
                return "bool"
            scalar_boolean_type = self.scalar_boolean_builtin_result_type(
                func_name, raw_args
            )
            if scalar_boolean_type is not None:
                return scalar_boolean_type
            if func_name in self.HIP_SHADER_CLOCK_RESULT_TYPES:
                return self.HIP_SHADER_CLOCK_RESULT_TYPES[func_name]
            if (
                func_name in {"inverseSqrt", "rsqrt"}
                and raw_args
                and not self.is_user_defined_function(func_name)
            ):
                return self.expression_result_type(raw_args[0])
            reflection_result_type = self.reflection_result_type(func_name, raw_args)
            if reflection_result_type is not None:
                return reflection_result_type
            bitcast_result_type = self.hip_bitcast_result_type(func_name, raw_args)
            if bitcast_result_type is not None:
                return bitcast_result_type
            integer_bit_result_type = self.hip_integer_bit_result_type(
                func_name,
                raw_args,
            )
            if integer_bit_result_type is not None:
                return integer_bit_result_type
            if func_name in {"fma", "mad"}:
                cached_type = getattr(node, "expression_type", None) or getattr(
                    node, "vtype", None
                )
                if cached_type is not None:
                    return cached_type
                return self.fused_multiply_add_result_type(raw_args)
            if (
                func_name == "lerp"
                and len(raw_args) == 3
                and not self.is_user_defined_function(func_name)
            ):
                return self.expression_result_type(raw_args[0]) or (
                    self.expression_result_type(raw_args[1])
                    if len(raw_args) > 1
                    else None
                )
            if func_name == "step" and len(raw_args) == 2:
                return self.step_result_type(raw_args)
            buffer_result_type = self.buffer_call_result_type(node)
            if buffer_result_type is not None:
                return buffer_result_type
        if isinstance(node, MemberAccessNode):
            object_node = getattr(
                node,
                "object_expr",
                getattr(node, "object", None),
            )
            object_name = getattr(object_node, "name", None)
            if not self.source_identifier_shadows_builtin(object_name):
                builtin_member_type = self.hip_compute_builtin_member_type(
                    object_name,
                    getattr(node, "member", ""),
                )
                if builtin_member_type is not None:
                    return builtin_member_type
            member_type = self.member_access_member_type(node)
            if member_type is not None:
                return member_type
        if isinstance(node, PointerAccessNode):
            member_type = self.pointer_access_member_type(node)
            if member_type is not None:
                return member_type
        if isinstance(node, ConstructorNode):
            constructor_type = self.type_name_string(
                getattr(node, "constructor_type", getattr(node, "vtype", None))
            )
            return (
                infer_enum_constructor_type(self, node)
                or infer_struct_constructor_type(self, node)
                or (
                    constructor_type
                    if constructor_type in HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES
                    else None
                )
            )
        if isinstance(node, MatchNode):
            return infer_match_expression_result_type(self, node)
        return super().expression_result_type(node)

    def dimension_query_spec(self, type_name):
        """Return HIP-specific resource metadata dimensions before shared specs."""
        base_type = self.resource_base_type(type_name)
        hip_specs = {
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "iimageCube": (("width", "height"), False, False),
            "uimageCube": (("width", "height"), False, False),
            "imageCubeArray": (("width", "height", "elements"), False, False),
            "iimageCubeArray": (("width", "height", "elements"), False, False),
            "uimageCubeArray": (("width", "height", "elements"), False, False),
        }
        spec = hip_specs.get(base_type)
        if spec is None:
            return super().dimension_query_spec(base_type)
        dimensions, mip, samples = spec
        return {"dimensions": dimensions, "mip": mip, "samples": samples}


def generate_hip_code(ast: ShaderNode) -> str:
    """Generate HIP source from a CrossGL shader AST."""
    generator = HipCodeGen()
    return generator.generate(ast)
