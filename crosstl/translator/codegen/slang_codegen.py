"""CrossGL-to-Slang code generator."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    AtomicOpNode,
    BinaryOpNode,
    ConstructorNode,
    EnumNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PointerAccessNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ShaderNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from ..host_workgroup_size import function_entry_execution_config
from ..renames import renamed
from ..semantics.builtin_values import glsl_builtin_name
from .array_utils import (
    collect_literal_int_constants,
    collect_struct_member_types,
    format_c_style_array_declaration,
    get_array_size_from_node,
    split_array_type_suffix,
)
from .assignment_values import assignment_value_expression
from .buffer_length import HlslBufferLength
from .buffer_member_layout import MEMBER_LAYOUT_QUALIFIERS
from .builtin_result_types import (
    C_FAMILY,
    boolean_builtin_result_type,
    builtin_call_result_type,
)
from .c_family_operands import postfix_operand
from .comma_expressions import (
    c_family_comma_expression,
    comma_result_operand,
    is_comma_expression,
)
from .constant_ordering import partition_constants_by_struct_dependency
from .cooperative_matrix_refusals import refuse_cooperative_matrices
from .enum_utils import (
    collect_enum_struct_variant_fields,
    collect_enum_type_names,
    collect_enum_variant_constants,
    collect_enum_variant_constructor_fields,
    collect_enum_variant_constructors,
    collect_generic_enum_specialization_member_types,
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_enum_variant_constants,
    collect_plain_enums,
    collect_struct_payload_enums,
    enum_value_expression,
    generic_enum_specialized_type_name,
)
from .fragment_output_targets import frag_data_attachment_index
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_emission_list,
    generic_function_parameters,
    iter_function_nodes,
    numeric_trait_method_result_type,
    prepare_generic_function_specializations,
)
from .generic_global_types import collect_constant_and_cbuffer_member_types
from .generic_pointer_spaces import (
    assign_generic_pointer_spaces,
    refuse_written_generic_pointers,
)
from .global_specialization_constants import split_global_specialization_constants
from .glsl_buffer_layout import glsl_buffer_block_node_type
from .hlsl_array_value_arguments import HlslArrayValueArguments
from .hlsl_image_formats import texel_shape
from .hlsl_matrix_orientation import HLSLMatrixOrientation
from .lowering_refusals import LoweringRefusalMixin
from .match_utils import infer_match_expression_result_type
from .relational_builtins import lower_relational_builtins
from .sampler_pairs import canonicalize_separate_textures
from .slang_buffer_block_types import SlangBufferBlockTypesMixin
from .slang_byte_address_words import SlangThreeComponentAccess
from .slang_call_expressions import SlangCallExpressionMixin
from .slang_conservative_depth import slang_conservative_depth_semantic
from .slang_control_flow import SlangControlFlowMixin
from .slang_cull_distance import slang_cull_distance_member_type
from .slang_declaration_modifiers import (
    is_slang_declaration_metadata_attribute,
    slang_member_modifier_prefix,
)
from .slang_declarations import SlangDeclarationsMixin
from .slang_expression_nodes import (
    HIT_TRIANGLE_VERTEX_POSITIONS,
    SlangExpressionNodeMixin,
)
from .slang_expression_parts import SlangExpressionPartsMixin
from .slang_function_parts import SlangFunctionPartsMixin
from .slang_geometry_stages import SlangGeometryStagesMixin
from .slang_glsl_buffer_access import SlangGlslBufferAccessMixin
from .slang_glsl_ray_interfaces import lower_slang_glsl_ray_interfaces
from .slang_image_resources import SlangImageResourcesMixin
from .slang_mesh_outputs import SlangMeshOutputsMixin
from .slang_numeric_types import slang_unrepresentable_type_reason
from .slang_pointer_declarations import refuse_slang_pointer_declarations
from .slang_precise_functions import SlangPreciseFunctionMixin
from .slang_program_symbols import SlangProgramSymbolsMixin
from .slang_rasterizer_order import slang_rasterizer_ordered_type
from .slang_ray_tracing import SlangRayTracingMixin
from .slang_refusals import SlangLoweringRefused
from .slang_resource_bindings import SlangResourceBindingsMixin
from .slang_resource_calls import SlangResourceCallsMixin
from .slang_resource_memory import (
    helper_resource_argument_names,
)
from .slang_semantics import SlangSemanticsMixin
from .slang_stage_attributes import SlangStageAttributesMixin
from .slang_stage_interface import SlangStageInterfaceMixin
from .slang_stage_parameters import SlangStageParametersMixin
from .slang_statements import SlangStatementsMixin
from .slang_storage_instances import lower_slang_storage_instances
from .slang_structured_atomics import structured_buffer_atomic_location
from .slang_structured_buffers import SlangStructuredBuffersMixin
from .slang_texture_queries import SlangTextureQueriesMixin
from .slang_texture_sampling import SlangTextureSamplingMixin
from .slang_value_types import SlangValueTypesMixin
from .slang_varying_locations import (
    fragment_output_location_node_ids,
    location_attribute_index,
    struct_member_location_prefix,
    varying_location_semantic,
)
from .slang_wave_mesh_ops import SlangWaveMeshOpsMixin
from .slang_wave_shuffles import (
    SLANG_WAVE_SHUFFLE_ARITIES,
    as_type_target,
)
from .slang_workgroup_atomics import SlangWorkgroupAtomics, workgroup_variable_names
from .slang_workgroup_size import SlangWorkgroupSizeConstants
from .spirv_union_storage import lower_union_storage
from .storage_parameter_texels import StorageParameterTexels


class SlangCodeGen(
    SlangProgramSymbolsMixin,
    SlangBufferBlockTypesMixin,
    SlangStageAttributesMixin,
    SlangGeometryStagesMixin,
    SlangMeshOutputsMixin,
    SlangSemanticsMixin,
    SlangStageInterfaceMixin,
    SlangDeclarationsMixin,
    SlangResourceBindingsMixin,
    SlangStageParametersMixin,
    SlangRayTracingMixin,
    SlangStatementsMixin,
    SlangValueTypesMixin,
    SlangExpressionPartsMixin,
    SlangWaveMeshOpsMixin,
    SlangControlFlowMixin,
    SlangImageResourcesMixin,
    SlangStructuredBuffersMixin,
    SlangGlslBufferAccessMixin,
    SlangResourceCallsMixin,
    SlangTextureSamplingMixin,
    SlangTextureQueriesMixin,
    SlangPreciseFunctionMixin,
    LoweringRefusalMixin,
    SlangCallExpressionMixin,
    SlangExpressionNodeMixin,
    SlangFunctionPartsMixin,
):
    """Emit Slang shader source from the shared CrossGL AST."""

    LOWERING_REFUSAL_TYPE = SlangLoweringRefused

    BINARY_PRECEDENCE = {
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
    ASSOCIATIVE_BINARY_OPS = {"+", "*", "&&", "||", "&", "|", "^"}
    SLANG_WAVE_INTRINSIC_ARITIES = {
        "WaveGetLaneCount": 0,
        "WaveGetLaneIndex": 0,
        "WaveIsFirstLane": 0,
        "WaveActiveSum": 1,
        "WaveActiveProduct": 1,
        "WaveActiveBitAnd": 1,
        "WaveActiveBitOr": 1,
        "WaveActiveBitXor": 1,
        "WaveActiveMin": 1,
        "WaveActiveMax": 1,
        "WaveActiveAllTrue": 1,
        "WaveActiveAnyTrue": 1,
        "WaveActiveBallot": 1,
        "WaveReadLaneAt": 2,
        "WaveReadLaneFirst": 1,
        "WavePrefixSum": 1,
        "WavePrefixProduct": 1,
        "QuadReadAcrossX": 1,
        "QuadReadAcrossY": 1,
        "QuadReadAcrossDiagonal": 1,
        "QuadReadLaneAt": 2,
        "WaveMatch": 1,
        "WaveMultiPrefixSum": 2,
        "WaveMultiPrefixProduct": 2,
        "WaveMultiPrefixBitAnd": 2,
        "WaveMultiPrefixBitOr": 2,
        "WaveMultiPrefixBitXor": 2,
    }
    SLANG_MESH_INTRINSIC_ARITIES = {
        "SetMeshOutputCounts": {2},
        "DispatchMesh": {3, 4},
    }
    SLANG_GEOMETRY_STREAM_TYPES = {
        "PointStream",
        "LineStream",
        "TriangleStream",
    }
    SLANG_GEOMETRY_INPUT_PRIMITIVE_ARITIES = {
        "point": 1,
        "line": 2,
        "triangle": 3,
        "lineadj": 4,
        "triangleadj": 6,
    }
    SLANG_GEOMETRY_STREAM_METHOD_ARITIES = {
        "Append": {1},
        "RestartStrip": {0},
    }
    SLANG_INTERPOLATION_QUALIFIER_MAP = {
        "flat": "nointerpolation",
        "smooth": "linear",
        "noperspective": "noperspective",
        "centroid": "centroid",
        "sample": "sample",
        "nointerpolation": "nointerpolation",
        "linear": "linear",
        "linear_centroid": "linear centroid",
        "linear_noperspective": "linear noperspective",
        "linear_noperspective_centroid": "linear noperspective centroid",
        "linear_sample": "linear sample",
    }

    def __init__(self):
        """Initialize Slang generation state and helper caches."""
        self.indent_level = 0
        self.workgroup_atomics = SlangWorkgroupAtomics(self, ())
        self.indent_str = "    "
        self.variable_types = {}
        self.local_variable_types = self.variable_types
        self.image_resource_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_types = {}
        self.buffer_resource_accesses = {}
        self.helper_functions = {}
        self.hlsl_matrix_orientation = HLSLMatrixOrientation(self)
        self.helper_name_aliases = {}
        self.user_symbol_names = set()
        self.current_function_return_type = None
        self.current_shader_type = None
        self.current_expression_expected_type = None
        self.user_function_names = set()
        self.user_functions_by_name = {}
        self.user_function_return_types = {}
        self.user_function_parameter_types = {}
        self.generic_function_definitions = {}
        self.generic_function_definition_ordinals = {}
        self.generic_function_specializations = {}
        self.generic_function_specialized_names = {}
        self.current_generic_function_substitutions = {}
        self.user_struct_names = set()
        self.user_structs_by_name = {}
        self.structs_by_name = {}
        self.struct_member_types = {}
        self.slang_lowered_struct_resource_members = {}
        self.user_enum_nodes = []
        self.plain_enums = []
        self.struct_payload_enums = []
        self.generic_enum_struct_definitions = {}
        self.generic_enum_specializations = {}
        self.enum_type_names = set()
        self.enum_struct_type_names = set()
        self.enum_struct_variant_fields = {}
        self.enum_variant_constructors = {}
        self.enum_variant_constructor_fields = {}
        self.enum_variant_constants = {}
        self.vertex_entry_input_struct_names = set()
        self.vertex_entry_output_struct_names = set()
        self.fragment_entry_input_struct_names = set()
        self.fragment_entry_output_struct_names = set()
        self.slang_fragment_output_location_ids = set()
        self.literal_int_constants = {}
        self.glsl_buffer_block_struct_names = set()
        self.lowered_glsl_buffer_blocks = {}
        self.glsl_buffer_block_lowering_failures = {}
        self.glsl_buffer_block_struct_lowering_failures = {}
        self.required_glsl_buffer_aggregate_load_helpers = {}
        self.required_byteaddress_atomic_helpers = set()
        self.slang_array_values = HlslArrayValueArguments(self)
        self.slang_three_component_access = SlangThreeComponentAccess()
        self.slang_byteaddress_temp_variable_index = 0
        self.slang_mesh_payload_parameter_types = set()
        self.slang_buffer_length = HlslBufferLength()
        self.slang_ray_payload_parameter_types = set()
        self.slang_callable_data_parameter_types = set()
        self.slang_hit_attribute_parameter_types = set()
        self.stage_entry_name_overrides = {}
        self.explicit_sampler_texture_names = set()
        self.explicit_comparison_sampler_names = set()
        self.identifier_aliases = {}
        self.slang_resource_register_cursors = {}
        self.slang_used_resource_registers = {}
        self.slang_source_sampler_register_assignments = {}
        self.slang_vk_binding_cursors = {}
        self.slang_used_vk_bindings = {}
        self.slang_global_declaration_signatures = {}
        self.function_stage_parameter_dependencies = {}
        self.expression_prelude_stack = []
        self.expression_prelude_result_stack = []
        self.expression_temp_names = set()
        self.atomic_value_context_stack = []
        self.statement_expression_node_stack = []
        self.current_hull_output_rewrite = None
        self._generating = False
        self.semantic_map = {
            "gl_Position": "SV_Position",
            # PSIZE is a user varying for slangc's SPIR-V and GLSL targets; SV_PointSize is the built-in.
            "gl_PointSize": "SV_PointSize",
            "gl_ClipDistance": "SV_ClipDistance",
            "gl_CullDistance": "SV_CullDistance",
            "gl_VertexID": "SV_VertexID",
            "gl_InstanceID": "SV_InstanceID",
            "gl_BaseVertex": "SV_StartVertexLocation",
            "gl_BaseInstance": "SV_StartInstanceLocation",
            "gl_DrawID": "SV_DrawIndex",
            "gl_PrimitiveID": "SV_PrimitiveID",
            "gl_PrimitiveIDIn": "SV_PrimitiveID",
            "gl_CullPrimitiveEXT": "SV_CullPrimitive",
            "gl_TessCoord": "SV_DomainLocation",
            "gl_TessLevelOuter": "SV_TessFactor",
            "gl_TessLevelInner": "SV_InsideTessFactor",
            "gl_Layer": "SV_RenderTargetArrayIndex",
            "gl_ViewportIndex": "SV_ViewportArrayIndex",
            "gl_FragCoord": "SV_Position",
            "gl_PointCoord": "SV_PointCoord",
            "gl_FrontFacing": "SV_IsFrontFace",
            "gl_BaryCoordEXT": "SV_Barycentrics",
            "gl_BaryCoordNoPerspEXT": "SV_Barycentrics",
            "gl_FragDepth": "SV_Depth",
            "gl_FragStencilRefEXT": "SV_StencilRef",
            "gl_FragColor": "SV_Target",
            "gl_SampleID": "SV_SampleIndex",
            "gl_SampleMask": "SV_Coverage",
            "gl_SampleMaskIn": "SV_Coverage",
            "gl_FragColor0": "SV_Target0",
            "gl_FragColor1": "SV_Target1",
            "gl_FragColor2": "SV_Target2",
            "gl_FragColor3": "SV_Target3",
            "gl_FragColor4": "SV_Target4",
            "gl_FragColor5": "SV_Target5",
            "gl_FragColor6": "SV_Target6",
            "gl_FragColor7": "SV_Target7",
            "gl_WorkGroupID": "SV_GroupID",
            "gl_LocalInvocationID": "SV_GroupThreadID",
            "gl_GlobalInvocationID": "SV_DispatchThreadID",
            "gl_LocalInvocationIndex": "SV_GroupIndex",
        }
        self.function_map = {
            "mix": "lerp",
            "fract": "frac",
            "bitCount": "countbits",
            "bitfieldReverse": "reversebits",
            "findLSB": "firstbitlow",
            "findMSB": "firstbithigh",
            "dFdx": "ddx",
            "dFdy": "ddy",
            "dFdxCoarse": "ddx_coarse",
            "dFdxFine": "ddx_fine",
            "dFdyCoarse": "ddy_coarse",
            "dFdyFine": "ddy_fine",
            "fwidthCoarse": "fwidth_coarse",
            "fwidthFine": "fwidth_fine",
            "inversesqrt": "rsqrt",
            "inverseSqrt": "rsqrt",
            # Slang rint rounds halfway cases to the nearest even integer on every target. Slang round
            # leaves them to the target (round on Metal, Round on SPIR-V, roundf on CUDA).
            "roundEven": "rint",
            "floatBitsToInt": "asint",
            "floatBitsToUint": "asuint",
            "intBitsToFloat": "asfloat",
            "uintBitsToFloat": "asfloat",
            "workgroupBarrier": "GroupMemoryBarrierWithGroupSync",
            "workgroupExecutionBarrier": "GroupMemoryBarrierWithGroupSync",
            # GLSL's non-uniform resource index is Slang's NonUniformResourceIndex.
            "nonuniformEXT": "NonUniformResourceIndex",
            # HLSL 6.9's MaybeReorderThread is Slang's ReorderThread; Slang declares the HLSL name for HLSL only.
            "MaybeReorderThread": "ReorderThread",
        }

    def indent(self):
        """Return whitespace for the current indentation level."""
        return self.indent_str * self.indent_level

    def generate(self, ast):
        """Generate Slang source for a CrossGL AST or AST fragment.

        A construct with no Slang lowering is recorded as a refusal while generation continues, and the program is
        then refused as a whole with every reason listed (:class:`SlangLoweringRefused`).
        """
        if self._generating:
            return self.generate_program(ast)
        refuse_cooperative_matrices(ast, label="Slang", target="slang")
        ast = lower_union_storage(ast, "slang")
        # Slang textures are separate objects, as in HLSL: a separate CrossGL texture declares the same Texture2D.
        ast = canonicalize_separate_textures(ast)
        ast = lower_slang_storage_instances(ast)
        lower_slang_glsl_ray_interfaces(ast)
        try:
            return self.refusals.run(self.generate_checked_program, ast)
        finally:
            self._generating = False

    def generate_checked_program(self, ast):
        """Generate the program with its generic pointers specialized, refusing any the output still spells."""
        ast = assign_generic_pointer_spaces(ast, self.refuse_lowering)
        refuse_slang_pointer_declarations(ast, self.refuse_lowering)
        code = self.generate_program(ast)
        refuse_written_generic_pointers(code, self.refuse_lowering, "Slang")
        return code

    def generate_program(self, ast):
        outermost = not self._generating
        if outermost:
            self._generating = True
            lower_relational_builtins(ast)
            self.variable_types = {}
            self.local_variable_types = self.variable_types
            self.image_resource_types = {}
            self.image_resource_accesses = {}
            self.buffer_resource_types = {}
            self.buffer_resource_accesses = {}
            self.helper_functions = {}
            self.hlsl_matrix_orientation = HLSLMatrixOrientation(self)
            self.storage_parameter_texels = StorageParameterTexels(
                self.resource_base_type
            )
            self.storage_parameter_texels.prepare(ast)
            self.workgroup_atomics = SlangWorkgroupAtomics(
                self, workgroup_variable_names(ast)
            )
            self.workgroup_size_constants = SlangWorkgroupSizeConstants(ast)
            self.helper_name_aliases = {}
            self.user_symbol_names = self.collect_user_symbol_names(ast)
            self.current_function_return_type = None
            self.current_shader_type = None
            self.current_expression_expected_type = None
            self.user_function_names = self.collect_user_function_names(ast)
            self.user_functions_by_name = self.collect_user_functions_by_name(ast)
            self.user_function_return_types = self.collect_user_function_return_types(
                ast
            )
            self.user_function_parameter_types = (
                self.collect_user_function_parameter_types(ast)
            )
            self.slang_helper_resource_arguments = helper_resource_argument_names(
                ast, self.user_function_names
            )
            self.generic_function_definitions = {}
            self.generic_function_definition_ordinals = {}
            self.generic_function_specializations = {}
            self.generic_function_specialized_names = {}
            self.current_generic_function_substitutions = {}
            user_structs = self.collect_user_structs(ast)
            self.user_struct_names = {
                struct.name for struct in user_structs if getattr(struct, "name", None)
            }
            self.user_structs_by_name = {
                struct.name: struct
                for struct in user_structs
                if getattr(struct, "name", None)
            }
            self.structs_by_name = dict(self.user_structs_by_name)
            self.vertex_entry_input_struct_names = (
                self.collect_slang_vertex_entry_input_struct_names(ast)
            )
            self.vertex_entry_output_struct_names = (
                self.collect_slang_vertex_entry_output_struct_names(ast)
            )
            self.fragment_entry_input_struct_names = (
                self.collect_slang_fragment_entry_input_struct_names(ast)
            )
            self.fragment_entry_output_struct_names = (
                self.collect_slang_fragment_entry_output_struct_names(ast)
            )
            self.slang_fragment_output_location_ids = fragment_output_location_node_ids(
                self, ast
            )
            self.user_enum_nodes = self.collect_user_enums(ast)
            self.generic_enum_struct_definitions = (
                collect_generic_enum_struct_definitions(user_structs)
            )
            self.generic_enum_specializations = collect_generic_enum_specializations(
                ast,
                self.generic_enum_struct_definitions,
                self.type_name_string,
            )
            self.plain_enums = collect_plain_enums(self.user_enum_nodes)
            self.struct_payload_enums = collect_struct_payload_enums(
                self.user_enum_nodes
            )
            self.enum_type_names = collect_enum_type_names(self.plain_enums)
            self.enum_struct_type_names = (
                collect_enum_type_names(self.struct_payload_enums)
                | set(self.generic_enum_struct_definitions)
                | {
                    specialization["struct_name"]
                    for specialization in self.generic_enum_specializations.values()
                }
            )
            self.enum_struct_variant_fields = collect_enum_struct_variant_fields(
                self.struct_payload_enums
            )
            self.enum_variant_constructors = collect_enum_variant_constructors(
                self.struct_payload_enums
            )
            self.enum_variant_constructor_fields = (
                collect_enum_variant_constructor_fields(self.struct_payload_enums)
            )
            self.enum_variant_constants = {
                **collect_enum_variant_constants(
                    self.plain_enums + self.struct_payload_enums
                ),
                **collect_generic_enum_variant_constants(
                    self.generic_enum_struct_definitions
                ),
            }
            self.struct_member_types = collect_struct_member_types(
                user_structs, self.type_name_string
            )
            self.slang_lowered_struct_resource_members = (
                self.collect_slang_lowered_struct_resource_members(user_structs)
            )
            self.struct_member_types.update(
                self.collect_enum_struct_member_types(self.struct_payload_enums)
            )
            self.struct_member_types.update(
                collect_generic_enum_specialization_member_types(
                    self, self.generic_enum_specializations
                )
            )
            generic_function_specializations = prepare_generic_function_specializations(
                self,
                list(iter_function_nodes(ast)),
                collect_constant_and_cbuffer_member_types(
                    self, ast, type_renderer=self.convert_type_node_to_string
                ),
                ast=ast,
            )
            if generic_function_specializations:
                specialized_functions = list(generic_function_specializations.values())
                self.user_function_names.update(
                    func.name
                    for func in specialized_functions
                    if getattr(func, "name", None)
                )
                self.user_functions_by_name.update(
                    {
                        func.name: func
                        for func in specialized_functions
                        if getattr(func, "name", None)
                    }
                )
                self.user_function_return_types.update(
                    {
                        func.name: self.type_name_string(
                            getattr(func, "return_type", "void")
                        )
                        for func in specialized_functions
                        if getattr(func, "name", None)
                    }
                )
                self.user_function_parameter_types.update(
                    {
                        func.name: self.function_parameter_type_names(func)
                        for func in specialized_functions
                        if getattr(func, "name", None)
                    }
                )
            self.reject_unsupported_generic_functions(ast)
            self.literal_int_constants = collect_literal_int_constants(
                getattr(ast, "constants", [])
            )
            self.collect_slang_glsl_buffer_blocks(ast)
            self.required_glsl_buffer_aggregate_load_helpers = {}
            self.required_byteaddress_atomic_helpers = set()
            self.slang_array_values = HlslArrayValueArguments(self)
            self.slang_three_component_access = SlangThreeComponentAccess()
            self.slang_byteaddress_temp_variable_index = 0
            self.slang_mesh_payload_parameter_types = set()
            self.slang_buffer_length = HlslBufferLength()
            self.slang_ray_payload_parameter_types = set()
            self.slang_callable_data_parameter_types = set()
            self.slang_hit_attribute_parameter_types = set()
            self.stage_entry_name_overrides = {}
            (
                self.explicit_sampler_texture_names,
                self.explicit_comparison_sampler_names,
            ) = self.collect_explicit_sampler_resource_names(ast)
            self.identifier_aliases = {}
            self.slang_resource_register_cursors = {}
            self.slang_used_resource_registers = {}
            self.slang_source_sampler_register_assignments = {}
            self.slang_vk_binding_cursors = {}
            self.slang_used_vk_bindings = {}
            self.slang_global_declaration_signatures = {}
            self.function_stage_parameter_dependencies = (
                self.collect_slang_function_stage_parameter_dependencies(ast)
            )
            self.expression_prelude_stack = []
            self.expression_prelude_result_stack = []
            self.expression_temp_names = set()
            self.atomic_value_context_stack = []
            self.statement_expression_node_stack = []
            self.reserve_explicit_slang_resource_declarations(ast)

        if isinstance(ast, list):
            result = ""
            for node in ast:
                result += self.generate(node) + "\n"
            return self.finish_generation(result, outermost)
        elif isinstance(ast, ShaderNode):
            return self.finish_generation(self.generate_shader(ast), outermost)
        elif isinstance(ast, StructNode):
            return self.finish_generation(self.generate_struct(ast), outermost)
        else:
            result = ""
            result += self.generate_enum_support_code()

            structs = getattr(ast, "structs", [])
            constants, global_vars = split_global_specialization_constants(ast)
            leading_constants, struct_dependent_constants = (
                partition_constants_by_struct_dependency(constants, structs)
            )
            result += self.generate_constants(ast, leading_constants)
            for struct in structs:
                if isinstance(struct, EnumNode):
                    continue
                struct_code = self.generate_struct(struct)
                if struct_code:
                    result += struct_code + "\n\n"
            result += self.generate_constants(ast, struct_dependent_constants)
            result += self.slang_struct_dependent_helper_marker()

            for node in global_vars:
                result += self.generate_global_variable(node)
            result += self.generate_slang_lowered_struct_resource_member_globals()

            cbuffers = getattr(ast, "cbuffers", [])
            for node in cbuffers:
                result += self.generate_cbuffer(node) + "\n\n"

            functions = getattr(ast, "functions", [])
            for function in functions:
                if generic_function_parameters(function):
                    for specialized_func in generic_function_emission_list(
                        self,
                        function,
                    ):
                        result += self.generate_function(specialized_func) + "\n\n"
                    continue
                if hasattr(function, "qualifiers") and function.qualifiers:
                    qualifier = function.qualifiers[0] if function.qualifiers else None
                else:
                    qualifier = getattr(function, "qualifier", None)

                if qualifier == "vertex":
                    result += "// Vertex Shader\n"
                    result += self.generate_function(function) + "\n\n"
                elif qualifier == "fragment":
                    result += "// Fragment Shader\n"
                    result += self.generate_function(function) + "\n\n"
                else:
                    result += self.generate_function(function) + "\n\n"

            if hasattr(ast, "stages") and ast.stages:
                self.validate_slang_tessellation_stage_shapes(ast.stages)
                self.slang_mesh_payload_parameter_types = (
                    self.collect_slang_mesh_payload_parameter_types(ast.stages)
                )
                self.slang_ray_payload_parameter_types = (
                    self.collect_slang_ray_payload_parameter_types(ast.stages)
                )
                self.slang_callable_data_parameter_types = (
                    self.collect_slang_callable_data_parameter_types(ast.stages)
                )
                self.slang_hit_attribute_parameter_types = (
                    self.collect_slang_hit_attribute_parameter_types(ast.stages)
                )
                self.stage_entry_name_overrides = (
                    self.collect_stage_entry_name_overrides(ast.stages)
                )
                for stage_type, stage in ast.stages.items():
                    result += self.generate_stage(stage_type, stage)

            return self.finish_generation(result, outermost)

    def collect_declared_type_names(self, node):
        type_names = {}
        visited = set()

        def collect(current):
            if current is None or isinstance(current, (str, int, float, bool)):
                return
            if isinstance(current, type):
                return
            if isinstance(current, dict):
                for value in current.values():
                    collect(value)
                return
            if isinstance(current, (list, tuple, set)):
                for item in current:
                    collect(item)
                return
            current_id = id(current)
            if current_id in visited:
                return
            visited.add(current_id)

            name = getattr(current, "name", getattr(current, "variable_name", None))
            type_name = None
            for attr in ("var_type", "param_type"):
                type_node = getattr(current, attr, None)
                if type_node is not None:
                    type_name = self.convert_type_node_to_string(type_node)
                    break
            if type_name is None:
                type_name = getattr(current, "vtype", None)
            if type_name and name:
                type_names[name] = type_name

            if hasattr(current, "__dict__"):
                for child in vars(current).values():
                    collect(child)

        collect(node)
        return type_names

    def generate_shader(self, node):
        """Render a full CrossGL shader AST as a Slang translation unit."""
        result = ""
        result += self.generate_enum_support_code()

        structs = getattr(node, "structs", [])
        constants, global_vars = split_global_specialization_constants(node)
        leading_constants, struct_dependent_constants = (
            partition_constants_by_struct_dependency(constants, structs)
        )
        result += self.generate_constants(node, leading_constants)
        for struct in structs:
            if isinstance(struct, EnumNode):
                continue
            struct_code = self.generate_struct(struct)
            if struct_code:
                result += struct_code + "\n\n"
        result += self.generate_constants(node, struct_dependent_constants)
        result += self.slang_struct_dependent_helper_marker()

        for global_var in global_vars:
            result += self.generate_global_variable(global_var)
        result += self.generate_slang_lowered_struct_resource_member_globals()

        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            result += self.generate_cbuffer(cbuffer) + "\n\n"

        functions = getattr(node, "functions", [])
        for function in functions:
            if generic_function_parameters(function):
                for specialized_func in generic_function_emission_list(
                    self,
                    function,
                ):
                    result += self.generate_function(specialized_func) + "\n\n"
                continue
            stage_name = self.get_function_stage(function)
            if stage_name:
                result += f"// {stage_name.title()} Shader\n"
                result += self.generate_function(
                    function,
                    shader_type=stage_name,
                    execution_config=function_entry_execution_config(function),
                )
                result += "\n\n"
            else:
                result += self.generate_function(function) + "\n\n"

        stages = getattr(node, "stages", {})
        self.validate_slang_tessellation_stage_shapes(stages)
        self.slang_mesh_payload_parameter_types = (
            self.collect_slang_mesh_payload_parameter_types(stages)
        )
        self.slang_program_has_mesh_stage = any(
            str(getattr(stage_type, "value", stage_type)).lower() == "mesh"
            for stage_type in stages
        )
        self.slang_ray_payload_parameter_types = (
            self.collect_slang_ray_payload_parameter_types(stages)
        )
        self.slang_callable_data_parameter_types = (
            self.collect_slang_callable_data_parameter_types(stages)
        )
        self.slang_hit_attribute_parameter_types = (
            self.collect_slang_hit_attribute_parameter_types(stages)
        )
        self.stage_entry_name_overrides = self.collect_stage_entry_name_overrides(
            stages
        )
        for stage_type, stage in stages.items():
            result += self.generate_stage(stage_type, stage)

        return result

    def semantic_from_node(self, node, skip_stage_attributes=False):
        depth_semantic = slang_conservative_depth_semantic(node)
        if depth_semantic is not None:
            return depth_semantic
        semantic = getattr(node, "semantic", None)
        if semantic:
            if skip_stage_attributes and self.slang_shader_stage_marker_name(semantic):
                return None
            return semantic

        for attr in getattr(node, "attributes", []) or []:
            if skip_stage_attributes and (
                self.slang_stage_attribute_name(attr)
                or self.slang_shader_stage_marker_name(getattr(attr, "name", None))
            ):
                continue
            if is_slang_declaration_metadata_attribute(attr):
                # A modifier or record of the declaration (slang_declaration_modifiers.py), never a semantic.
                continue
            if str(getattr(attr, "name", "")).lower() in MEMBER_LAYOUT_QUALIFIERS:
                # A block member's placement, which the layout planners read; it never names a semantic.
                continue
            if str(getattr(attr, "name", "")).lower() in (
                "ray_payload_read",
                "ray_payload_write",
            ):
                # HLSL shader model 6.7 payload access qualifiers; Slang declares a ray payload as a plain struct.
                continue
            if (
                self.is_resource_format_attribute(attr)
                or str(getattr(attr, "name", "")).lower() == "maybe_unused"
                or self.is_resource_binding_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.slang_interpolation_qualifier(attr.name)
                or self.slang_mesh_payload_parameter_attribute_name(attr)
                or self.slang_mesh_output_parameter_role(attr)
            ):
                continue
            if str(getattr(attr, "name", "")).lower() == "builtin":
                # ``@builtin(global_invocation_id)`` names the same value as ``@gl_GlobalInvocationID``.
                builtin_name = glsl_builtin_name(attr)
                if builtin_name == "gl_SampleMask":
                    # WGSL's sample_mask is a fragment input and an output, like SV_Coverage; GLSL splits it into
                    # gl_SampleMaskIn and the output-only gl_SampleMask.
                    return "SV_Coverage"
                if builtin_name is not None:
                    return builtin_name
                arguments = getattr(attr, "arguments", None) or [""]
                written = getattr(arguments[0], "name", None) or getattr(
                    arguments[0], "value", arguments[0]
                )
                self.refuse_lowering(
                    f"Slang has no system value for the unknown built-in @builtin({written}) "
                    f"on '{getattr(node, 'name', '')}'"
                )
                return None
            if hasattr(attr, "name"):
                if attr.name == "gl_FragData":
                    index = frag_data_attachment_index(node)
                    if index is not None:
                        return f"gl_FragData[{index}]"
                location = location_attribute_index(attr)
                if location is not None:
                    # slang_varying_locations.py: SV_Target<N> for a fragment colour output, else TEXCOORD<N>.
                    return varying_location_semantic(
                        location, id(node) in self.slang_fragment_output_location_ids
                    )
                return attr.name
        return None

    def slang_static_storage_prefix(self, node, mapped_type=None):
        """``static`` for a module-scope value with one copy per invocation, ``static const`` for a constant.

        As in HLSL, a Slang global without ``static`` is a uniform, so CrossGL's module-scope ``static`` values
        (SPIR-V ``Private`` storage) keep the keyword. Groupshared, uniform and resource declarations do not.
        """
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        constant = bool(qualifiers & {"const", "constant"})
        if not (constant or "static" in qualifiers) or qualifiers & {
            "uniform",
            "groupshared",
            "shared",
            "threadgroup",
            "workgroup",
        }:
            return ""
        if (
            mapped_type is not None
            and self.slang_register_prefix_for_type(mapped_type, node, None) is not None
        ):
            return ""
        # A module-scope constant (``const``, ``static const`` or MSL's ``constant``) is a compile-time value;
        # without ``static`` Slang would make it a uniform parameter and drop its initializer.
        return "static const " if constant else "static "

    def generate_global_variable(self, node):
        direct_buffer_type = self.slang_glsl_buffer_array_resource_type(node)
        if direct_buffer_type is not None:
            self.register_variable_type(node.name, direct_buffer_type, node)
            if self.skip_duplicate_slang_global_variable(node, direct_buffer_type):
                return ""
            declaration = self.slang_resource_memory_qualifier_prefix(
                direct_buffer_type, node
            ) + format_c_style_array_declaration(
                slang_rasterizer_ordered_type(self, direct_buffer_type, node), node.name
            )
            declaration = self.apply_slang_resource_binding_decorations(
                declaration, node, direct_buffer_type, auto_assign=True
            )
            return f"{declaration};\n"

        lowered_block_type = self.slang_glsl_buffer_block_resource_type(node)
        if lowered_block_type is not None:
            original_type = self.type_name_string(glsl_buffer_block_node_type(node))
            self.variable_types[node.name] = original_type
            self.local_variable_types[node.name] = original_type
            if self.skip_duplicate_slang_global_variable(node, lowered_block_type):
                return ""
            declaration = self.slang_resource_memory_qualifier_prefix(
                lowered_block_type, node
            ) + format_c_style_array_declaration(lowered_block_type, node.name)
            declaration = self.apply_slang_resource_binding_decorations(
                declaration, node, lowered_block_type, auto_assign=True
            )
            return f"{declaration};\n"

        storage_buffer_type = self.slang_storage_buffer_resource_type(node)
        if storage_buffer_type is not None:
            self.register_variable_type(node.name, storage_buffer_type, node)
            if self.skip_duplicate_slang_global_variable(node, storage_buffer_type):
                return ""
            declaration = format_c_style_array_declaration(
                slang_rasterizer_ordered_type(self, storage_buffer_type, node),
                node.name,
            )
            declaration = (
                self.slang_resource_memory_qualifier_prefix(storage_buffer_type, node)
                + declaration
            )
            declaration = self.apply_slang_resource_binding_decorations(
                declaration, node, storage_buffer_type, auto_assign=True
            )
            return f"{declaration};\n"

        if self.is_glsl_buffer_block_variable(node, glsl_buffer_block_node_type(node)):
            vtype = self.type_name_string(glsl_buffer_block_node_type(node))
            diagnostic = self.glsl_buffer_block_diagnostic(
                "Slang", vtype, node.name, node
            )
            placeholder_type = self.map_resource_type_with_format(vtype, node)
            placeholder = format_c_style_array_declaration(placeholder_type, node.name)
            # A block the byte-address lowering cannot lay out has no faithful Slang declaration.
            return self.refuse_lowering(
                diagnostic.removeprefix("// ").strip(), f"{placeholder};\n"
            )

        if isinstance(node, ArrayNode):
            self.register_variable_type(node.name, node.element_type)
            element_type = self.convert_type(node.element_type)
            size = get_array_size_from_node(node)
            declaration_type = (
                f"{element_type}[]" if size is None else f"{element_type}[{size}]"
            )
            if self.skip_duplicate_slang_global_variable(node, declaration_type):
                return ""
            prefix = self.slang_static_storage_prefix(
                node
            ) + self.slang_declaration_qualifier_prefix(node)
            if size is None:
                return f"{prefix}{element_type} {node.name}[];\n"
            return f"{prefix}{element_type} {node.name}[{size}];\n"

        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        vtype = self.variable_declaration_type(node, initial_value)
        self.register_variable_type(node.name, vtype, node)
        mapped_type = self.map_resource_type_with_format(vtype, node)
        if self.skip_duplicate_slang_global_variable(node, mapped_type):
            return ""
        declaration = self.format_declaration(vtype, node.name, node)
        declaration = (
            self.slang_storage_image_format_prefix(mapped_type, node)
            + self.slang_static_storage_prefix(node, mapped_type)
            + self.slang_declaration_qualifier_prefix(node)
            + self.slang_resource_memory_qualifier_prefix(mapped_type, node)
            + declaration
        )
        declaration = self.apply_slang_resource_binding_decorations(
            declaration, node, vtype, auto_assign=True
        )
        if initial_value is not None:
            initial_expr = self.generate_expression_with_expected(
                initial_value,
                self.initializer_expected_type(vtype),
            )
            return f"{declaration} = {initial_expr};\n"
        return f"{declaration};\n"

    def generate_struct(self, node):
        if isinstance(node, EnumNode):
            return ""
        if getattr(node, "name", None) in self.generic_enum_struct_definitions:
            return ""
        if getattr(node, "name", None) in self.glsl_buffer_block_struct_names:
            return ""
        result = f"struct {node.name}\n{{\n"
        self.indent_level += 1

        members = getattr(node, "members", [])
        default_member_semantics = self.slang_default_struct_member_semantics(node)
        for member in members:
            if self.slang_should_lower_struct_resource_member(node.name, member.name):
                continue
            if hasattr(member, "member_type"):
                member_type = self.convert_type(
                    self.convert_type_node_to_string(member.member_type)
                )
            elif hasattr(member, "vtype"):
                member_type = self.convert_type(member.vtype)
            else:
                member_type = "float"

            semantic = self.semantic_from_node(member)
            if semantic is None:
                semantic = default_member_semantics.get(member.name)
            interface_type = (
                self.slang_interface_array_type(member) if semantic else None
            )
            if interface_type is not None:
                member_type = self.convert_type(
                    self.convert_type_node_to_string(interface_type)
                )
            self.validate_slang_builtin_semantic_type(
                semantic,
                self.slang_tess_factor_member_type_name(member),
                f"struct member semantic '{node.name}.{member.name}'",
            )
            semantic_str = self.semantic_suffix(semantic)
            member_type = slang_cull_distance_member_type(self, member, member_type)
            declaration = (
                struct_member_location_prefix(
                    self, node, member, default_member_semantics
                )
                + self.slang_interpolation_attribute_prefix(member)
                + slang_member_modifier_prefix(
                    self,
                    member,
                    member_type,
                    f"struct member '{node.name}.{member.name}'",
                )
                + self.format_declaration(member_type, member.name)
            )
            result += f"{self.indent()}{declaration}{semantic_str};\n"

        self.indent_level -= 1
        result += "};"
        return result

    def generate_struct_definition(self, node):
        result = f"{node.name}\n{{\n"

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                member_type = self.convert_type_node_to_string(member.member_type)
            else:
                member_type = getattr(member, "vtype", "float")
            result += f"    {self.format_declaration(member_type, member.name)};\n"

        result += "};"
        return result

    def generate_function(
        self,
        node,
        shader_type=None,
        execution_config=None,
        entry_name=None,
        emit_stage_decorations=True,
        stage_role=None,
        extra_parameters=None,
    ):
        """Render one CrossGL function or shader entry point as Slang code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_types = self.image_resource_types.copy()
        saved_image_resource_accesses = self.image_resource_accesses.copy()
        saved_buffer_resource_types = self.buffer_resource_types.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        saved_function_return_type = self.current_function_return_type
        saved_shader_type = self.current_shader_type
        saved_identifier_aliases = self.identifier_aliases.copy()
        saved_hull_output_rewrite = self.current_hull_output_rewrite
        saved_expression_temp_names = self.expression_temp_names
        saved_generic_function_substitutions = (
            getattr(self, "current_generic_function_substitutions", {}) or {}
        )
        self.current_generic_function_substitutions = (
            getattr(node, "_generic_substitutions", {}) or {}
        )
        self.expression_temp_names = set()
        (
            semantic,
            body_statements,
            param_list,
            fragment_output_rewrite,
            hull_output_rewrite,
            builtin_return_rewrite,
            ret_type_name,
            ret_type,
        ) = self.slang_function_return_plan(
            node, shader_type, stage_role, extra_parameters
        )

        self.current_function_return_type = ret_type_name
        self.current_shader_type = shader_type
        semantic_str = self.semantic_suffix(semantic, shader_type)

        effective_param_list = self.slang_filtered_stage_parameters(
            param_list, hull_output_rewrite
        )
        if shader_type:
            self.validate_slang_ray_stage_parameters(
                node, shader_type, effective_param_list
            )
            self.validate_slang_ray_tracing_calls(
                node, shader_type, effective_param_list
            )
            self.validate_slang_ray_query_calls(node, shader_type, effective_param_list)
            self.validate_slang_mesh_intrinsic_calls(node, shader_type)
            self.validate_slang_mesh_payload_parameter(
                shader_type, effective_param_list
            )
            self.validate_slang_mesh_output_parameters(
                node, shader_type, effective_param_list
            )
            self.validate_slang_geometry_stage(node, shader_type, effective_param_list)
            self.validate_slang_stage_parameter_semantics(
                shader_type, effective_param_list, stage_role=stage_role, function=node
            )
            self.validate_slang_stage_body_builtins(
                body_statements,
                shader_type,
                effective_param_list,
                stage_role=stage_role,
            )
        else:
            self.validate_slang_ray_query_calls(node, "function", effective_param_list)
        params, local_alias_declarations = self.slang_function_parameter_declarations(
            node, shader_type, stage_role, body_statements, effective_param_list
        )

        params_str = ", ".join(params)
        identifier_aliases = self.slang_stage_system_value_aliases(
            body_statements,
            shader_type,
            effective_param_list,
            stage_role=stage_role,
        )
        identifier_aliases.update(
            self.slang_stage_intrinsic_builtin_aliases(
                body_statements, shader_type, effective_param_list
            )
        )
        identifier_aliases.update(
            self.slang_stage_patch_input_aliases(
                body_statements, shader_type, effective_param_list
            )
        )
        identifier_aliases.update(
            self.slang_stage_builtin_return_aliases(builtin_return_rewrite)
        )
        self.identifier_aliases = identifier_aliases
        self.current_hull_output_rewrite = hull_output_rewrite

        result = self.slang_function_definition_text(
            node,
            shader_type,
            execution_config,
            entry_name,
            emit_stage_decorations,
            body_statements,
            builtin_return_rewrite,
            fragment_output_rewrite,
            hull_output_rewrite,
            local_alias_declarations,
            params_str,
            ret_type,
            semantic_str,
        )
        self.variable_types = saved_variable_types
        self.image_resource_types = saved_image_resource_types
        self.image_resource_accesses = saved_image_resource_accesses
        self.buffer_resource_types = saved_buffer_resource_types
        self.buffer_resource_accesses = saved_buffer_resource_accesses
        self.current_function_return_type = saved_function_return_type
        self.current_shader_type = saved_shader_type
        self.identifier_aliases = saved_identifier_aliases
        self.current_hull_output_rewrite = saved_hull_output_rewrite
        self.expression_temp_names = saved_expression_temp_names
        self.current_generic_function_substitutions = (
            saved_generic_function_substitutions
        )
        return result

    def generate_bool_mix_call(self, args):
        if len(args) != 3:
            return None

        condition_type = self.expression_result_type(args[2])
        condition_info = self.vector_value_info(condition_type)
        if condition_info is not None:
            if condition_info["component_type"] != "bool":
                return None
            return self.generate_bool_vector_select_expression(
                args[2], args[1], args[0]
            )

        if self.convert_type(condition_type) != "bool":
            return None

        condition = self.generate_expression(args[2])
        true_value = self.generate_expression(args[1])
        false_value = self.generate_expression(args[0])
        return f"({condition} ? {true_value} : {false_value})"

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if is_comma_expression(expr):
            return self.expression_result_type(comma_result_operand(expr))
        if isinstance(expr, VariableNode):
            return self.local_variable_types.get(
                getattr(expr, "name", None)
            ) or self.variable_types.get(getattr(expr, "name", None))
        if isinstance(expr, IdentifierNode):
            name = getattr(expr, "name", None)
            if isinstance(name, str) and name in self.enum_variant_constants:
                enum_name, _variant_name = name.split("::", 1)
                return enum_name
            return self.local_variable_types.get(name) or self.variable_types.get(name)
        if isinstance(expr, LiteralNode):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
            if isinstance(expr.value, float):
                return "float"
            if isinstance(expr.value, int) and not isinstance(expr.value, bool):
                return "int"
            if isinstance(expr.value, bool):
                return "bool"
        if isinstance(expr, BinaryOpNode):
            return self.binary_expression_result_type(expr)
        if isinstance(expr, UnaryOpNode):
            operand_type = self.expression_result_type(expr.operand)
            if expr.op == "&" and operand_type is not None:
                operand_type = (
                    self.reference_referent_type_name(operand_type) or operand_type
                )
                return f"{operand_type}*"
            if expr.op == "*" and operand_type is not None:
                if self.is_structured_buffer_resource_type(operand_type):
                    return self.structured_buffer_element_type(operand_type)
                return self.pointer_pointee_type_name(operand_type) or operand_type
            return operand_type
        if isinstance(expr, AssignmentNode):
            return self.expression_result_type(getattr(expr, "left", None))
        if isinstance(expr, TernaryOpNode):
            true_type = self.expression_result_type(expr.true_expr)
            false_type = self.expression_result_type(expr.false_expr)
            if true_type is None:
                return None
            same_type = self.convert_type(true_type) == self.convert_type(false_type)
            return true_type if same_type else None
        if isinstance(expr, ConstructorNode):
            constructor_type = self.type_name_string(
                getattr(expr, "constructor_type", None)
            )
            if constructor_type and "::" in constructor_type:
                enum_name, _variant_name = constructor_type.split("::", 1)
                if constructor_type in self.enum_variant_constructor_fields:
                    return enum_name
                expected_type = self.type_name_string(
                    self.current_expression_expected_type
                )
                if generic_enum_specialized_type_name(self, expected_type) is not None:
                    return expected_type
            return constructor_type
        if isinstance(expr, MatchNode):
            try:
                return infer_match_expression_result_type(self, expr)
            except ValueError:
                return None
        if isinstance(expr, ArrayAccessNode):
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if array_type and "[" in array_type and "]" in array_type:
                return self.array_element_type(array_type)
            if self.is_structured_buffer_resource_type(array_type):
                return self.structured_buffer_element_type(array_type)
            pointer_element_type = self.pointer_pointee_type_name(array_type)
            if pointer_element_type is not None:
                return pointer_element_type
            column_type = self.hlsl_matrix_orientation.column_type(array_type)
            if column_type is not None:
                return column_type
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_result_type(expr.object)
            member = str(expr.member)
            if object_type and all(ch in "xyzwrgba" for ch in member):
                component_type = self.vector_component_type(object_type)
                if component_type and len(member) == 1:
                    return component_type
                if component_type:
                    return f"{component_type}{len(member)}"
            member_type = self.struct_member_type(object_type, member)
            if member_type is not None:
                return member_type
            return None
        if isinstance(expr, PointerAccessNode):
            pointer_type = self.expression_result_type(expr.pointer_expr)
            member = str(expr.member)
            member_type = self.struct_member_type(pointer_type, member)
            if member_type is not None:
                return member_type
            return None
        if isinstance(expr, RayQueryOpNode):
            return self.slang_ray_query_method_return_type(expr.operation)
        if isinstance(expr, WaveOpNode):
            return self.slang_wave_result_type(expr)
        if isinstance(expr, AtomicOpNode):
            operation = self.image_atomic_operation_from_atomic_op(expr.operation)
            if operation is None:
                return None
            image_type = self.resource_base_type(self.image_resource_type(expr.target))
            return self.image_atomic_return_type(image_type)
        if isinstance(expr, RayTracingOpNode):
            if expr.operation in self.user_function_names:
                return self.user_function_return_types.get(expr.operation)
            return self.slang_ray_tracing_result_type(expr.operation)
        if isinstance(expr, FunctionCallNode):
            ray_query_call = self.slang_ray_query_call_parts(expr)
            if ray_query_call is not None:
                operation, _query_expr, _args = ray_query_call
                return self.slang_ray_query_method_return_type(operation)
            ray_tracing_call = self.slang_ray_tracing_call_parts(expr)
            if ray_tracing_call is not None:
                operation, _args = ray_tracing_call
                return self.slang_ray_tracing_result_type(operation)
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            bit_cast_type = as_type_target(func_name)
            if bit_cast_type is not None:
                return bit_cast_type
            numeric_result_type = numeric_trait_method_result_type(self, expr)
            if numeric_result_type is not None:
                return numeric_result_type
            raw_args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
            specialized_func_name = generic_function_call_name(
                self,
                func_name,
                raw_args,
            )
            if specialized_func_name in self.user_function_return_types:
                return self.user_function_return_types[specialized_func_name]
            ray_transform_type = self.slang_ray_transform_call_type(func_name, raw_args)
            if ray_transform_type is not None:
                return ray_transform_type
            if isinstance(func_name, str) and "::" in func_name:
                enum_name, _variant_name = func_name.split("::", 1)
                if func_name in self.enum_variant_constructor_fields:
                    return enum_name
                expected_type = self.type_name_string(
                    self.current_expression_expected_type
                )
                if generic_enum_specialized_type_name(self, expected_type) is not None:
                    return expected_type
            if (
                isinstance(func_name, str)
                and func_name in self.structured_buffer_atomic_operations()
                and getattr(expr, "args", None)
            ):
                buffer_type = self.structured_buffer_atomic_target_resource_type(
                    expr.args[0]
                )
                if self.is_structured_buffer_resource_type(buffer_type):
                    _name, element_type = structured_buffer_atomic_location(
                        self, expr.args[0], buffer_type
                    )
                    return element_type if element_type in {"int", "uint"} else "uint"
            if func_name == "imageLoad" and getattr(expr, "args", None):
                element_type = self.image_resource_element_type(
                    self.image_resource_type(expr.args[0])
                )
                shape = texel_shape(element_type)
                if shape is not None and shape[1] < 4:
                    # image_load_expression fills a narrower texel to four.
                    return f"{shape[0]}4"
                return element_type
            if func_name == "buffer_load" and getattr(expr, "args", None):
                buffer_type = self.structured_buffer_resource_type(expr.args[0])
                if self.is_byte_address_buffer_resource_type(buffer_type):
                    return "uint"
                return self.structured_buffer_element_type(buffer_type)
            if func_name == "buffer_consume" and getattr(expr, "args", None):
                return self.structured_buffer_element_type(
                    self.structured_buffer_resource_type(expr.args[0])
                )
            if func_name == "buffer_dimensions":
                return "uint"
            byte_address_result_type = self.byte_address_member_call_result_type(
                func_expr
            )
            if byte_address_result_type is not None:
                return byte_address_result_type
            structured_result_type = self.structured_buffer_member_call_result_type(
                func_expr
            )
            if structured_result_type is not None:
                return structured_result_type
            if isinstance(func_name, str) and func_name in self.user_struct_names:
                return func_name
            if isinstance(func_name, str) and func_name in {
                "float",
                "double",
                "int",
                "uint",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "float2",
                "float3",
                "float4",
                "int2",
                "int3",
                "int4",
                "uint2",
                "uint3",
                "uint4",
                "bool2",
                "bool3",
                "bool4",
            }:
                return str(func_name)
            if isinstance(func_name, str) and self.matrix_value_info(func_name):
                return str(func_name)
            alias_result_type = self.builtin_alias_call_result_type(
                func_name, getattr(expr, "args", [])
            )
            if alias_result_type is not None:
                return alias_result_type
            return self.user_function_return_types.get(func_name)
        return None

    def builtin_alias_call_result_type(self, func_name, args):
        if not isinstance(func_name, str) or func_name in self.user_function_names:
            return None

        if func_name in {
            "fract",
            "bitfieldReverse",
            "reversebits",
            "dFdx",
            "dFdy",
            "dFdxCoarse",
            "dFdxFine",
            "dFdyCoarse",
            "dFdyFine",
            "fwidth",
            "fwidthCoarse",
            "fwidthFine",
            "inversesqrt",
            "inverseSqrt",
        }:
            return self.expression_result_type(args[0]) if args else None

        if func_name in {"mix", "mod"}:
            return self.expression_result_type(args[0]) if args else None

        boolean_type = boolean_builtin_result_type(
            func_name,
            [self.type_name_string(self.expression_result_type(a)) for a in args],
            C_FAMILY,
        )
        if boolean_type is not None:
            return boolean_type

        if func_name in {"transpose", "inverse"}:
            return builtin_call_result_type(
                func_name,
                [self.type_name_string(self.expression_result_type(a)) for a in args],
                C_FAMILY,
            )
        if func_name in {"matrixCompMult", "outerProduct"}:
            return self.hlsl_matrix_orientation.call_type(
                func_name, [self.expression_result_type(arg) for arg in args]
            )

        bit_query_type = self.slang_bit_query_result_type(func_name, args)
        if bit_query_type is not None:
            return bit_query_type

        return self.glsl_bitcast_result_type(func_name, args)

    def generate_binary_expression(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        if self.binary_child_needs_parentheses(node.op, node.left):
            left = f"({left})"
        if self.binary_child_needs_parentheses(node.op, node.right, True):
            right = f"({right})"
        if node.op == "*" and self.binary_requires_slang_mul(
            self.expression_result_type(node.left),
            self.expression_result_type(node.right),
        ):
            return f"mul({left}, {right})"
        return f"{left} {node.op} {right}"

    def generate_expression(self, node):
        """Render a CrossGL expression as Slang expression syntax."""
        if isinstance(node, VariableNode):
            if node.name == "gl_HelperInvocation":
                return self.slang_helper_invocation_expression()
            return node.name
        elif isinstance(node, IdentifierNode):
            if node.name == "gl_HelperInvocation":
                return self.slang_helper_invocation_expression()
            if node.name == HIT_TRIANGLE_VERTEX_POSITIONS:
                # An element is HitTriangleVertexPosition(index) (slang_expression_nodes); the array is no value.
                self.refuse_lowering(
                    f"Slang {node.name}: only an element is readable, as HitTriangleVertexPosition(index)"
                )
            if isinstance(node.name, str) and "::" in node.name:
                enum_value = enum_value_expression(self, node.name)
                if enum_value is not None:
                    return enum_value
            # ``identifier_aliases`` holds the stage system values an entry renames; other names keep theirs.
            return renamed(self.identifier_aliases, node.name)
        elif isinstance(node, LiteralNode):
            return self.generate_literal(node)
        elif isinstance(node, ExpressionStatementNode):
            return self.generate_expression(node.expression)
        elif is_comma_expression(node):
            return c_family_comma_expression(node, self.generate_expression)
        elif isinstance(node, AssignmentNode):
            return assignment_value_expression(self.generate_assignment(node))
        elif isinstance(node, ArrayAccessNode):
            return self.generate_slang_array_access_expression(node)
        elif isinstance(node, ArrayLiteralNode):
            return self.generate_slang_array_literal_expression(node)
        elif isinstance(node, MemberAccessNode):
            return self.generate_slang_member_access_expression(node)
        elif isinstance(node, PointerAccessNode):
            pointer = postfix_operand(self.generate_expression(node.pointer_expr))
            return f"{pointer}->{node.member}"
        elif isinstance(node, BinaryOpNode):
            return self.generate_binary_expression(node)
        elif isinstance(node, WaveOpNode):
            return self.generate_slang_wave_op_expression(node)
        elif isinstance(node, MeshOpNode):
            return self.generate_slang_mesh_op_expression(node)
        elif isinstance(node, AtomicOpNode):
            return self.generate_slang_atomic_op_expression(node)
        elif isinstance(node, RayQueryOpNode):
            return self.generate_slang_ray_query_expression(
                node.operation, node.query_expr, node.arguments
            )
        elif isinstance(node, FunctionCallNode):
            return self.generate_slang_function_call_expression(node)
        elif isinstance(node, ConstructorNode):
            return self.generate_slang_constructor_expression(node)
        elif isinstance(node, RayTracingOpNode):
            return self.generate_slang_ray_tracing_op_expression(node)
        elif isinstance(node, UnaryOpNode):
            return self.generate_slang_unary_expression(node)
        elif isinstance(node, TernaryOpNode):
            return self.generate_slang_ternary_expression(node)
        elif isinstance(node, MatchNode):
            return self.generate_match_expression(node)
        elif isinstance(node, str):
            return node
        else:
            return str(node)

    def slang_synchronization_intrinsic_name(self, callee):
        return {
            "barrier": "GroupMemoryBarrierWithGroupSync",
            "workgroupBarrier": "GroupMemoryBarrierWithGroupSync",
            "workgroupExecutionBarrier": "GroupMemoryBarrierWithGroupSync",
            # groupMemoryBarrier orders buffer, image and shared memory (GLSL 4.60 section 8.16); Slang's
            # GroupMemoryBarrier orders groupshared memory only, as HLSL's does, so it is AllMemoryBarrier.
            "groupMemoryBarrier": "AllMemoryBarrier",
            "memoryBarrierShared": "GroupMemoryBarrier",
            "memoryBarrierBuffer": "DeviceMemoryBarrier",
            "deviceMemoryBarrier": "DeviceMemoryBarrier",
            "memoryBarrierImage": "DeviceMemoryBarrier",
            "memoryBarrier": "AllMemoryBarrier",
            "allMemoryBarrier": "AllMemoryBarrier",
        }.get(callee)

    def generate_slang_wave_op_expression(self, node):
        if node.operation in SLANG_WAVE_SHUFFLE_ARITIES:
            return self.generate_slang_wave_shuffle_expression(node)
        expected_arity = self.SLANG_WAVE_INTRINSIC_ARITIES.get(node.operation)
        if expected_arity is None:
            return self.unsupported_slang_wave_op_expression(
                node.operation, "is not recognized by the Slang backend"
            )

        actual_arity = len(node.arguments)
        rejection_reason = None
        if actual_arity != expected_arity:
            rejection_reason = f"expects {expected_arity} arguments, got {actual_arity}"
        else:
            rejection_reason = self.slang_wave_argument_rejection_reason(
                node.operation, node.arguments
            )
        if rejection_reason is not None:
            return self.unsupported_slang_wave_op_expression(
                node.operation, rejection_reason
            )
        result_type = self.slang_wave_result_type(node)
        conversion = self.slang_wave_result_conversion(result_type)
        if conversion is None:
            target_reason = self.slang_wave_target_type_rejection_reason(result_type)
            if target_reason is not None:
                return self.unsupported_slang_wave_op_expression(
                    node.operation, target_reason
                )

        # Each operand has its own type (a lane index is uint whatever the value is), so the
        # expression the wave op result is assigned to does not type its operands.
        args = ", ".join(
            self.generate_expression_without_expected(arg) for arg in node.arguments
        )
        call = f"{node.operation}({args})"
        return f"{conversion}({call})" if conversion else call

    def convert_type(self, type_name):
        """Map a CrossGL type name or type node to a Slang type string."""
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None

        for suffix in ("*", "&"):
            if type_name.endswith(suffix):
                base_type = type_name[: -len(suffix)].strip()
                return f"{self.convert_type(base_type)}{suffix}"

        base_type, array_suffix = split_array_type_suffix(type_name)
        if array_suffix:
            mapped_base_type = self.convert_type(base_type)
            return f"{mapped_base_type}{array_suffix}"

        generic_enum_type = generic_enum_specialized_type_name(self, type_name)
        if generic_enum_type is not None:
            return generic_enum_type

        if type_name in self.enum_type_names:
            return "int"

        if type_name in self.enum_struct_type_names:
            return type_name

        unrepresentable = slang_unrepresentable_type_reason(type_name)
        if unrepresentable is not None:
            return self.refuse_lowering(unrepresentable, type_name)

        type_map = {
            "f16": "half",
            "float16": "half",
            "f32": "float",
            "f64": "double",
            "i8": "int8_t",
            "u8": "uint8_t",
            "i16": "int16_t",
            "u16": "uint16_t",
            "int16": "int16_t",
            "uint16": "uint16_t",
            "i16vec2": "vector<int16_t, 2>",
            "i16vec3": "vector<int16_t, 3>",
            "i16vec4": "vector<int16_t, 4>",
            "u16vec2": "vector<uint16_t, 2>",
            "u16vec3": "vector<uint16_t, 3>",
            "u16vec4": "vector<uint16_t, 4>",
            "i32": "int",
            "u32": "uint",
            "int32": "int",
            "uint32": "uint",
            "i64": "int64_t",
            "u64": "uint64_t",
            "int8": "int8_t",
            "uint8": "uint8_t",
            "int64": "int64_t",
            "uint64": "uint64_t",
            "i8vec2": "vector<int8_t, 2>",
            "i8vec3": "vector<int8_t, 3>",
            "i8vec4": "vector<int8_t, 4>",
            "u8vec2": "vector<uint8_t, 2>",
            "u8vec3": "vector<uint8_t, 3>",
            "u8vec4": "vector<uint8_t, 4>",
            "i64vec2": "vector<int64_t, 2>",
            "i64vec3": "vector<int64_t, 3>",
            "i64vec4": "vector<int64_t, 4>",
            "u64vec2": "vector<uint64_t, 2>",
            "u64vec3": "vector<uint64_t, 3>",
            "u64vec4": "vector<uint64_t, 4>",
            "vec2<i64>": "vector<int64_t, 2>",
            "vec3<i64>": "vector<int64_t, 3>",
            "vec4<i64>": "vector<int64_t, 4>",
            "vec2<u64>": "vector<uint64_t, 2>",
            "vec3<u64>": "vector<uint64_t, 3>",
            "vec4<u64>": "vector<uint64_t, 4>",
            "half": "half",
            "half2": "half2",
            "half3": "half3",
            "half4": "half4",
            "f16vec2": "half2",
            "f16vec3": "half3",
            "f16vec4": "half4",
            "vec2<f16>": "half2",
            "vec3<f16>": "half3",
            "vec4<f16>": "half4",
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "vec2<i8>": "vector<int8_t, 2>",
            "vec3<i8>": "vector<int8_t, 3>",
            "vec4<i8>": "vector<int8_t, 4>",
            "vec2<u8>": "vector<uint8_t, 2>",
            "vec3<u8>": "vector<uint8_t, 3>",
            "vec4<u8>": "vector<uint8_t, 4>",
            "vec2<i16>": "vector<int16_t, 2>",
            "vec3<i16>": "vector<int16_t, 3>",
            "vec4<i16>": "vector<int16_t, 4>",
            "vec2<u16>": "vector<uint16_t, 2>",
            "vec3<u16>": "vector<uint16_t, 3>",
            "vec4<u16>": "vector<uint16_t, 4>",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "vec2<bool>": "bool2",
            "vec3<bool>": "bool3",
            "vec4<bool>": "bool4",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float3x2",
            "mat2x4": "float4x2",
            "mat3x2": "float2x3",
            "mat3x3": "float3x3",
            "mat3x4": "float4x3",
            "mat4x2": "float2x4",
            "mat4x3": "float3x4",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double3x2",
            "dmat2x4": "double4x2",
            "dmat3x2": "double2x3",
            "dmat3x3": "double3x3",
            "dmat3x4": "double4x3",
            "dmat4x2": "double2x4",
            "dmat4x3": "double3x4",
            "dmat4x4": "double4x4",
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "void": "void",
            "sampler": "SamplerState",
            "comparison_sampler": "SamplerComparisonState",
            "sampler1D": "Sampler1D<float4>",
            "sampler1DArray": "Sampler1DArray<float4>",
            "sampler2D": "Sampler2D<float4>",
            "sampler3D": "Sampler3D<float4>",
            "samplerCube": "SamplerCube<float4>",
            "sampler2DArray": "Sampler2DArray<float4>",
            "samplerCubeArray": "SamplerCubeArray<float4>",
            "sampler2DMS": "Sampler2DMS<float4>",
            "sampler2DMSArray": "Sampler2DMSArray<float4>",
            "isampler1D": "Sampler1D<int4>",
            "isampler1DArray": "Sampler1DArray<int4>",
            "isampler2D": "Sampler2D<int4>",
            "isampler3D": "Sampler3D<int4>",
            "isamplerCube": "SamplerCube<int4>",
            "isampler2DArray": "Sampler2DArray<int4>",
            "isamplerCubeArray": "SamplerCubeArray<int4>",
            "isampler2DMS": "Sampler2DMS<int4>",
            "isampler2DMSArray": "Sampler2DMSArray<int4>",
            "usampler1D": "Sampler1D<uint4>",
            "usampler1DArray": "Sampler1DArray<uint4>",
            "usampler2D": "Sampler2D<uint4>",
            "usampler3D": "Sampler3D<uint4>",
            "usamplerCube": "SamplerCube<uint4>",
            "usampler2DArray": "Sampler2DArray<uint4>",
            "usamplerCubeArray": "SamplerCubeArray<uint4>",
            "usampler2DMS": "Sampler2DMS<uint4>",
            "usampler2DMSArray": "Sampler2DMSArray<uint4>",
            "sampler2DShadow": "Sampler2DShadow",
            "sampler2DArrayShadow": "Sampler2DArrayShadow",
            "samplerCubeShadow": "SamplerCubeShadow",
            "samplerCubeArrayShadow": "SamplerCubeArrayShadow",
            "iimage1D": "RWTexture1D<int4>",
            "iimage1DArray": "RWTexture1DArray<int4>",
            "iimage2D": "RWTexture2D<int4>",
            "iimage3D": "RWTexture3D<int4>",
            "iimageCube": "RWTexture2DArray<int4>",
            "iimage2DArray": "RWTexture2DArray<int4>",
            "iimageCubeArray": "RWTexture2DArray<int4>",
            "iimage2DMS": "RWTexture2DMS<int4>",
            "iimage2DMSArray": "RWTexture2DMSArray<int4>",
            "uimage1D": "RWTexture1D<uint4>",
            "uimage1DArray": "RWTexture1DArray<uint4>",
            "uimage2D": "RWTexture2D<uint4>",
            "uimage3D": "RWTexture3D<uint4>",
            "uimageCube": "RWTexture2DArray<uint4>",
            "uimage2DArray": "RWTexture2DArray<uint4>",
            "uimageCubeArray": "RWTexture2DArray<uint4>",
            "uimage2DMS": "RWTexture2DMS<uint4>",
            "uimage2DMSArray": "RWTexture2DMSArray<uint4>",
            "image1D": "RWTexture1D<float4>",
            "image1DArray": "RWTexture1DArray<float4>",
            "image2D": "RWTexture2D<float4>",
            "image3D": "RWTexture3D<float4>",
            "imageCube": "RWTexture2DArray<float4>",
            "image2DArray": "RWTexture2DArray<float4>",
            "imageCubeArray": "RWTexture2DArray<float4>",
            "image2DMS": "RWTexture2DMS<float4>",
            "image2DMSArray": "RWTexture2DMSArray<float4>",
            "accelerationStructureEXT": "RaytracingAccelerationStructure",
            "AccelerationStructure": "RaytracingAccelerationStructure",
            "acceleration_structure": "RaytracingAccelerationStructure",
            "RaytracingAccelerationStructure": "RaytracingAccelerationStructure",
            "RayDesc": "RayDesc",
            "RayQuery": "RayQuery",
        }

        mapped_type = type_map.get(type_name)
        if mapped_type is not None:
            return mapped_type

        generic_resource_type = self.map_slang_generic_resource_type(type_name)
        if generic_resource_type is not None:
            return generic_resource_type

        return type_name
