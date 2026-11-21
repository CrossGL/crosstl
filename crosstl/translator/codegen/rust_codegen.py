"""CrossGL-to-Rust code generator."""

import re

from ..ast import (
    ArrayNode,
    EnumNode,
    ExpressionStatementNode,
    FunctionNode,
    ReturnNode,
    StructNode,
)
from ..explicit_specializations import explicit_specialization_refusals
from ..refusals import RustRefusal
from ..unhandled import UNHANDLED
from .c_reference_parameters import is_output_parameter
from .cooperative_matrix_refusals import refuse_cooperative_matrices
from .lowering_refusals import LoweringRefusalMixin
from .precise_evaluation import refuse_precise_evaluation
from .rust_builtin_result_types import RustBuiltinResultTypeMixin
from .rust_callables import RustCallableMixin
from .rust_exact_lookups import (
    rust_generic_template_reason,
    rust_operator,
    rust_unmapped_semantic_reason,
)
from .rust_expressions import RustExpressionMixin
from .rust_function_signatures import RustFunctionSignatureMixin
from .rust_global_declarations import RustGlobalDeclarationMixin
from .rust_gpu_constants import rust_integer_constants
from .rust_gpu_entry_buffers import normalize_rust_gpu_buffers
from .rust_gpu_interface import RUST_GPU_PRELUDE, UNIFORM_VALUE_KIND
from .rust_gpu_math_calls import rust_gpu_math_result_type
from .rust_gpu_slice_length import RustGpuSliceLengthMixin
from .rust_gpu_spec_constants import (
    SPEC_CONSTANT_KIND,
    spec_constant_infos,
    spec_constant_local,
)
from .rust_gpu_spelling import glam_spelling
from .rust_gpu_stage_io import GRAPHICS_STAGES, RustGpuStageIoMixin
from .rust_gpu_storage import PADDED_ELEMENTS, RustGpuPaddedSliceMixin
from .rust_gpu_subgroups import rewrite_rust_gpu_lane_queries
from .rust_gpu_subset import refuse_host_only_rust, refuse_outside_rust_gpu_subset
from .rust_match_lowering import RustMatchMixin
from .rust_memory_atomics import rust_memory_atomic_result_type
from .rust_narrow_integers import RUST_EXACT_WIDTH_INTEGER_TYPES
from .rust_operators import RustOperatorMixin
from .rust_output_parameters import RustOutputParameterMixin
from .rust_program_analysis import RustProgramAnalysisMixin
from .rust_refusals import RustLoweringRefused
from .rust_resource_attributes import RustResourceAttributeMixin
from .rust_resource_bindings import referenced_identifier_names
from .rust_resource_calls import RustResourceCallMixin
from .rust_result_types import RustResultTypeMixin
from .rust_return_values import RustReturnValueMixin
from .rust_stage_contracts import RustStageContractMixin
from .rust_statements import RustStatementMixin
from .rust_switch_lowering import RustSwitchLoweringMixin
from .rust_tables import (
    RUST_FUNCTION_MAP,
    RUST_SEMANTIC_MAP,
    RUST_SYNC_MAP,
    RUST_WAVE_OP_MAP,
)
from .rust_type_definitions import RustTypeDefinitionMixin
from .rust_values import RustValueMixin
from .rust_vector_elements import RustVectorElementMixin
from .sampler_pairs import canonicalize_separate_textures
from .stage_utils import is_fragment_output_parameter, normalize_stage_name


def tail_expression_as_return(statements):
    """``statements`` with a trailing tail expression (``value`` without ``;``) written as ``return value;``."""
    statements = list(statements or [])
    if statements and isinstance(statements[-1], ExpressionStatementNode):
        if getattr(statements[-1], "is_tail_expression", False):
            statements[-1] = ReturnNode(statements[-1].expression)
    return statements


class RustCodeGen(
    RustStageContractMixin,
    RustProgramAnalysisMixin,
    RustTypeDefinitionMixin,
    RustGlobalDeclarationMixin,
    RustFunctionSignatureMixin,
    RustStatementMixin,
    RustSwitchLoweringMixin,
    RustMatchMixin,
    RustReturnValueMixin,
    RustOperatorMixin,
    RustExpressionMixin,
    RustResourceCallMixin,
    RustCallableMixin,
    RustValueMixin,
    RustResultTypeMixin,
    RustResourceAttributeMixin,
    RustOutputParameterMixin,
    RustGpuPaddedSliceMixin,
    RustGpuSliceLengthMixin,
    RustGpuStageIoMixin,
    LoweringRefusalMixin,
    RustBuiltinResultTypeMixin,
    RustVectorElementMixin,
):
    """Emit rust-gpu shader crates from the shared CrossGL AST."""

    LOWERING_REFUSAL_TYPE = RustLoweringRefused

    def __init__(self):
        """Initialize Rust type maps and expression-generation state."""
        self.current_shader = None
        self.current_shader_type = None
        self.type_mapping = {
            # Scalar Types
            "void": "()",
            "i8": "i8",
            "u8": "u8",
            "i16": "i16",
            "u16": "u16",
            "int": "i32",
            "short": "i16",
            "long": "i64",
            "int64_t": "i64",
            "uint": "u32",
            "uint64_t": "u64",
            "ushort": "u16",
            "ulong": "u64",
            "float": "f32",
            "double": "f64",
            "half": "f16",
            "f16": "f16",
            "bool": "bool",
            **RUST_EXACT_WIDTH_INTEGER_TYPES,
            "string": "&'static str",
            "str": "&'static str",
            # CrossGL char is a signed 8-bit integer; Rust's char is a Unicode scalar value.
            "char": "i8",
            # Vector Types (using GPU-style vector types)
            "vec2<f16>": "Vec2<f16>",
            "vec3<f16>": "Vec3<f16>",
            "vec4<f16>": "Vec4<f16>",
            "vec2<f32>": "Vec2<f32>",
            "vec3<f32>": "Vec3<f32>",
            "vec4<f32>": "Vec4<f32>",
            "vec2<f64>": "Vec2<f64>",
            "vec3<f64>": "Vec3<f64>",
            "vec4<f64>": "Vec4<f64>",
            "vec2<i8>": "Vec2<i8>",
            "vec3<i8>": "Vec3<i8>",
            "vec4<i8>": "Vec4<i8>",
            "vec2<u8>": "Vec2<u8>",
            "vec3<u8>": "Vec3<u8>",
            "vec4<u8>": "Vec4<u8>",
            "vec2<i16>": "Vec2<i16>",
            "vec3<i16>": "Vec3<i16>",
            "vec4<i16>": "Vec4<i16>",
            "vec2<u16>": "Vec2<u16>",
            "vec3<u16>": "Vec3<u16>",
            "vec4<u16>": "Vec4<u16>",
            "vec2<i32>": "Vec2<i32>",
            "vec3<i32>": "Vec3<i32>",
            "vec4<i32>": "Vec4<i32>",
            "vec2<u32>": "Vec2<u32>",
            "vec3<u32>": "Vec3<u32>",
            "vec4<u32>": "Vec4<u32>",
            "vec2<bool>": "Vec2<bool>",
            "vec3<bool>": "Vec3<bool>",
            "vec4<bool>": "Vec4<bool>",
            "vec2": "Vec2<f32>",
            "vec3": "Vec3<f32>",
            "vec4": "Vec4<f32>",
            "ivec2": "Vec2<i32>",
            "ivec3": "Vec3<i32>",
            "ivec4": "Vec4<i32>",
            "uvec2": "Vec2<u32>",
            "uvec3": "Vec3<u32>",
            "uvec4": "Vec4<u32>",
            "dvec2": "Vec2<f64>",
            "dvec3": "Vec3<f64>",
            "dvec4": "Vec4<f64>",
            "bvec2": "Vec2<bool>",
            "bvec3": "Vec3<bool>",
            "bvec4": "Vec4<bool>",
            "bool2": "Vec2<bool>",
            "bool3": "Vec3<bool>",
            "bool4": "Vec4<bool>",
            "float2": "Vec2<f32>",
            "float3": "Vec3<f32>",
            "float4": "Vec4<f32>",
            "int2": "Vec2<i32>",
            "int3": "Vec3<i32>",
            "int4": "Vec4<i32>",
            "uint2": "Vec2<u32>",
            "uint3": "Vec3<u32>",
            "uint4": "Vec4<u32>",
            # Matrix Types
            "mat2": "Mat2<f32>",
            "mat3": "Mat3<f32>",
            "mat4": "Mat4<f32>",
            "mat2x2": "Mat2<f32>",
            "mat2x3": "Mat2x3<f32>",
            "mat2x4": "Mat2x4<f32>",
            "mat3x2": "Mat3x2<f32>",
            "mat3x3": "Mat3<f32>",
            "mat3x4": "Mat3x4<f32>",
            "mat4x2": "Mat4x2<f32>",
            "mat4x3": "Mat4x3<f32>",
            "mat4x4": "Mat4<f32>",
            "dmat2": "Mat2<f64>",
            "dmat3": "Mat3<f64>",
            "dmat4": "Mat4<f64>",
            "dmat2x2": "Mat2<f64>",
            "dmat2x3": "Mat2x3<f64>",
            "dmat2x4": "Mat2x4<f64>",
            "dmat3x2": "Mat3x2<f64>",
            "dmat3x3": "Mat3<f64>",
            "dmat3x4": "Mat3x4<f64>",
            "dmat4x2": "Mat4x2<f64>",
            "dmat4x3": "Mat4x3<f64>",
            "dmat4x4": "Mat4<f64>",
            # Texture Types
            "sampler1D": "Texture1D<f32>",
            "isampler1D": "Texture1D<i32>",
            "usampler1D": "Texture1D<u32>",
            "sampler1DArray": "Texture1DArray<f32>",
            "isampler1DArray": "Texture1DArray<i32>",
            "usampler1DArray": "Texture1DArray<u32>",
            "sampler2D": "Texture2D<f32>",
            "isampler2D": "Texture2D<i32>",
            "usampler2D": "Texture2D<u32>",
            "sampler2DArray": "Texture2DArray<f32>",
            "isampler2DArray": "Texture2DArray<i32>",
            "usampler2DArray": "Texture2DArray<u32>",
            "sampler3D": "Texture3D<f32>",
            "isampler3D": "Texture3D<i32>",
            "usampler3D": "Texture3D<u32>",
            "samplerCube": "TextureCube<f32>",
            "isamplerCube": "TextureCube<i32>",
            "usamplerCube": "TextureCube<u32>",
            "samplerCubeArray": "TextureCubeArray<f32>",
            "isamplerCubeArray": "TextureCubeArray<i32>",
            "usamplerCubeArray": "TextureCubeArray<u32>",
            "sampler2DMS": "Texture2DMS<f32>",
            "isampler2DMS": "Texture2DMS<i32>",
            "usampler2DMS": "Texture2DMS<u32>",
            "sampler2DMSArray": "Texture2DMSArray<f32>",
            "isampler2DMSArray": "Texture2DMSArray<i32>",
            "usampler2DMSArray": "Texture2DMSArray<u32>",
            "sampler2DShadow": "DepthTexture2D<f32>",
            "sampler2DArrayShadow": "DepthTexture2DArray<f32>",
            "samplerCubeShadow": "DepthTextureCube<f32>",
            "samplerCubeArrayShadow": "DepthTextureCubeArray<f32>",
            "sampler": "Sampler",
            "comparison_sampler": "Sampler",
            "SamplerState": "Sampler",
            "SamplerComparisonState": "Sampler",
            "Texture1D": "Texture1D<f32>",
            "Texture1DArray": "Texture1DArray<f32>",
            "Texture2D": "Texture2D<f32>",
            "Texture2DArray": "Texture2DArray<f32>",
            "Texture3D": "Texture3D<f32>",
            "TextureCube": "TextureCube<f32>",
            "TextureCubeArray": "TextureCubeArray<f32>",
            "Texture2DMS": "Texture2DMS<f32>",
            "Texture2DMSArray": "Texture2DMSArray<f32>",
            "Texture2DShadow": "DepthTexture2D<f32>",
            "Texture2DArrayShadow": "DepthTexture2DArray<f32>",
            "TextureCubeShadow": "DepthTextureCube<f32>",
            "TextureCubeArrayShadow": "DepthTextureCubeArray<f32>",
            "image1D": "Image1D<Vec4<f32>>",
            "image1DArray": "Image1DArray<Vec4<f32>>",
            "image2D": "Image2D<Vec4<f32>>",
            "image3D": "Image3D<Vec4<f32>>",
            "imageCube": "ImageCube<Vec4<f32>>",
            "image2DArray": "Image2DArray<Vec4<f32>>",
            "image2DMS": "Image2DMS<Vec4<f32>>",
            "image2DMSArray": "Image2DMSArray<Vec4<f32>>",
            "iimage1D": "Image1D<Vec4<i32>>",
            "iimage1DArray": "Image1DArray<Vec4<i32>>",
            "iimage2D": "Image2D<Vec4<i32>>",
            "iimage3D": "Image3D<Vec4<i32>>",
            "iimage2DArray": "Image2DArray<Vec4<i32>>",
            "iimage2DMS": "Image2DMS<Vec4<i32>>",
            "iimage2DMSArray": "Image2DMSArray<Vec4<i32>>",
            "uimage1D": "Image1D<Vec4<u32>>",
            "uimage1DArray": "Image1DArray<Vec4<u32>>",
            "uimage2D": "Image2D<Vec4<u32>>",
            "uimage3D": "Image3D<Vec4<u32>>",
            "uimage2DArray": "Image2DArray<Vec4<u32>>",
            "uimage2DMS": "Image2DMS<Vec4<u32>>",
            "uimage2DMSArray": "Image2DMSArray<Vec4<u32>>",
            "StructuredBuffer": "StructuredBuffer",
            "RWStructuredBuffer": "RWStructuredBuffer",
            "AppendStructuredBuffer": "AppendStructuredBuffer",
            "ConsumeStructuredBuffer": "ConsumeStructuredBuffer",
            "ByteAddressBuffer": "ByteAddressBuffer",
            "RWByteAddressBuffer": "RwByteAddressBuffer",
            "PointStream": "CglRustPointStream",
            "LineStream": "CglRustLineStream",
            "TriangleStream": "CglRustTriangleStream",
            "InputPatch": "CglRustInputPatch",
            "OutputPatch": "CglRustOutputPatch",
            "accelerationStructureEXT": "RayTracingAccelerationStructure",
            "AccelerationStructure": "RayTracingAccelerationStructure",
            "acceleration_structure": "RayTracingAccelerationStructure",
            "RaytracingAccelerationStructure": "RayTracingAccelerationStructure",
            "RayTracingAccelerationStructure": "RayTracingAccelerationStructure",
            "RayDesc": "RayDesc",
            "RayQuery": "RayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "BuiltInTriangleIntersectionAttributes"
            ),
        }

        self.semantic_map = dict(RUST_SEMANTIC_MAP)

        self.function_map = dict(RUST_FUNCTION_MAP)
        self.wave_op_map = dict(RUST_WAVE_OP_MAP)
        self.sync_map = dict(RUST_SYNC_MAP)
        self.variable_types = {}
        self.local_variable_names = set()
        self.lazy_static_names = set()
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.rust_resource_binding_cursors = {}
        self.rust_used_resource_bindings = {}
        self.rust_emitted_binding_cursors = {}
        self.rust_used_emitted_bindings = set()
        self.rust_emitted_binding_by_node = {}
        self.rust_resource_binding_infos = {}
        self.rust_entry_resource_params_by_function_id = {}
        self.rust_promoted_resource_node_ids = set()
        self.rust_referenced_names = frozenset()
        self.rust_resource_capture_params = {}
        self.rust_spec_constant_infos = {}
        self.rust_entry_resource_infos = {}
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.struct_generic_params = {}
        self.static_variable_names = set()
        self.static_symbol_names = {}
        self.runtime_type_collisions = set()
        self.current_return_type = None
        self.current_shader_type = None
        self.do_while_contexts = []
        self.for_contexts = []
        self.switch_contexts = []
        self.loop_labels = []
        self.loop_label_counter = 0
        self.loop_depth = 0
        self.do_while_counter = 0
        self.user_function_names = set()
        self.user_function_nodes = {}
        self.user_function_return_types = {}
        self.user_function_param_types = {}
        self.user_function_generic_constraint_cache = {}
        self.active_generic_constraint_functions = set()
        self.rust_function_capture_params = {}
        self.current_generic_param_names = set()
        self.current_function_generic_constraints = {}
        self.trait_methods = {}
        self.enum_variant_names = {}
        self.enum_variant_field_types = {}
        self.enum_variant_payload_type_map = {}
        self.enum_generic_params = {}
        self.enum_nodes_by_name = {}
        self.non_copy_type_names = set()
        self.current_mutated_names = set()
        self.required_generic_math_traits = set()
        self.required_mesh_helpers = set()
        self.swizzle_temp_counter = 0
        self.vector_arg_temp_counter = 0
        self.matrix_arg_temp_counter = 0
        self.lambda_capture_temp_counter = 0
        self.assignment_lhs_depth = 0
        self.member_object_depth = 0
        self.array_object_depth = 0
        self.return_move_blocked_roots = set()
        self.lambda_capture_alias_stack = []
        self.lambda_capture_place_alias_stack = []
        self.force_move_lambda_depth = 0
        self.nested_lambda_capture_preclone_depth = 0

    def generate(self, ast):
        """Generate complete Rust-like shader source for a CrossGL AST.

        A construct with no Rust lowering is recorded as a refusal while generation continues, and the program is
        then refused as a whole with every reason listed (:class:`RustLoweringRefused`).
        """
        # Rust GPU images are separate from their samplers: a separate CrossGL texture is the same image.
        refuse_cooperative_matrices(ast, label="Rust", target="rust")
        refuse_precise_evaluation(ast, label="Rust", target="rust")
        ast = canonicalize_separate_textures(ast)
        rewrite_rust_gpu_lane_queries(ast)
        self.rust_current_program = ast
        return glam_spelling(self.refusals.run(self.generate_program, ast))

    def generate_program(self, ast):
        """Render the Rust program; refusals recorded on the way are raised by :meth:`generate`."""
        self.validate_supported_stage_types(ast)
        normalize_rust_gpu_buffers(self, ast)
        explicit_specialization_refusals(self, ast, "Rust")
        self.rust_referenced_names = referenced_identifier_names(ast)
        self.struct_member_types = {}
        self.collect_rust_gpu_struct_nodes(ast)
        refuse_outside_rust_gpu_subset(self, ast)
        self.variable_types = {}
        self.local_variable_names = set()
        self.lazy_static_names = set()
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.rust_resource_binding_cursors = {}
        self.rust_used_resource_bindings = {}
        self.rust_emitted_binding_cursors = {}
        self.rust_used_emitted_bindings = set()
        self.rust_emitted_binding_by_node = {}
        self.rust_resource_binding_infos = {}
        self.rust_entry_resource_params_by_function_id = {}
        self.rust_promoted_resource_node_ids = set()
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.struct_generic_params = {}
        self.rust_spec_constant_infos = spec_constant_infos(self, ast)
        self.static_variable_names = self.collect_static_variable_names(ast)
        self.static_symbol_names = self.build_static_symbol_names(
            self.static_variable_names
        )
        self.runtime_type_collisions = self.collect_runtime_type_collisions(ast)
        self.current_return_type = None
        self.do_while_contexts = []
        self.for_contexts = []
        self.switch_contexts = []
        self.loop_labels = []
        self.loop_label_counter = 0
        self.loop_depth = 0
        self.do_while_counter = 0
        self.user_function_names = self.collect_user_function_names(ast)
        self.user_function_nodes = self.collect_user_function_nodes(ast)
        self.user_function_return_types = self.collect_user_function_return_types(ast)
        self.user_function_param_types = self.collect_user_function_param_types(ast)
        self.user_function_generic_constraint_cache = {}
        self.active_generic_constraint_functions = set()
        self.rust_function_capture_params = {}
        self.current_generic_param_names = set()
        self.current_function_generic_constraints = {}
        self.trait_methods = self.collect_trait_methods(ast)
        self.enum_variant_names = self.collect_enum_variant_names(ast)
        self.enum_variant_field_types = self.collect_enum_variant_field_types(ast)
        self.enum_variant_payload_type_map = self.collect_enum_variant_payload_type_map(
            ast
        )
        self.enum_generic_params = self.collect_enum_generic_params(ast)
        self.enum_nodes_by_name = self.collect_enum_nodes_by_name(ast)
        self.non_copy_type_names = self.collect_non_copy_type_names(ast)
        self.register_constant_types(ast)
        self.rust_integer_constant_values = rust_integer_constants(ast)
        self.current_mutated_names = set()
        self.required_generic_math_traits = set()
        self.required_mesh_helpers = set()
        self.swizzle_temp_counter = 0
        self.vector_arg_temp_counter = 0
        self.matrix_arg_temp_counter = 0
        self.lambda_capture_temp_counter = 0
        self.assignment_lhs_depth = 0
        self.member_object_depth = 0
        self.array_object_depth = 0
        self.return_move_blocked_roots = set()
        self.lambda_capture_alias_stack = []
        self.lambda_capture_place_alias_stack = []
        self.force_move_lambda_depth = 0
        self.nested_lambda_capture_preclone_depth = 0
        self.rust_referenced_names = referenced_identifier_names(ast)
        self.reserve_explicit_rust_resource_bindings(ast)
        self.rust_entry_resource_params_by_function_id = (
            self.collect_rust_entry_resource_params(ast)
        )
        self.rust_promoted_resource_node_ids = {
            id(info["node"])
            for params in self.rust_entry_resource_params_by_function_id.values()
            for info in params
        } | {
            id(self.rust_entry_resource_infos[param.name]["node"])
            for params in self.rust_resource_capture_params.values()
            for param in params
        }
        self.rust_function_capture_params = {
            name: list(params)
            for name, params in self.rust_resource_capture_params.items()
        }
        code = RUST_GPU_PRELUDE
        if self.has_geometry_stage(ast):
            code += self.generate_rust_geometry_stream_helpers()
        if self.has_tessellation_stage(ast):
            code += self.generate_rust_tessellation_patch_helpers()

        structs = getattr(ast, "structs", [])
        for node in structs:
            if isinstance(node, StructNode):
                if getattr(node, "is_trait", False):
                    code += self.generate_trait(node)
                else:
                    code += self.generate_struct(node)
            elif isinstance(node, EnumNode):
                code += self.generate_enum(node)

        code += self.generate_constant_declarations(ast)

        emitted_static_names = set()
        global_vars = getattr(ast, "global_variables", [])
        for node in global_vars:
            if self.is_rust_gpu_interface_declaration(node):
                self.register_variable_type(
                    node.name, self.rust_variable_source_type(node), scope="static"
                )
                emitted_static_names.add(node.name)
            elif self.is_promoted_rust_entry_resource(node):
                var_type = self.rust_variable_source_type(node)
                self.register_glsl_buffer_block_metadata(node)
                self.register_variable_type(node.name, var_type, scope="static")
                code += self.rust_resource_metadata_comment(node, var_type)
                emitted_static_names.add(node.name)
            elif isinstance(node, ArrayNode):
                code += self.generate_global_array_declaration(node)
                emitted_static_names.add(node.name)
            else:
                code += self.generate_variable_static_declaration(node)
                emitted_static_names.add(node.name)

        cbuffers = self.get_cbuffer_nodes(ast)
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        functions = getattr(ast, "functions", [])
        for func in functions:
            # Handle both old and new AST function structures
            if hasattr(func, "qualifiers") and func.qualifiers:
                qualifier = func.qualifiers[0] if func.qualifiers else None
            else:
                qualifier = getattr(func, "qualifier", None)
            qualifier_name = normalize_stage_name(qualifier)

            if qualifier_name == "vertex":
                code += "// Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif qualifier_name == "fragment":
                code += "// Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif qualifier_name == "compute":
                code += "// Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            elif qualifier_name == "geometry":
                code += "// Geometry Shader\n"
                code += self.generate_function(func, shader_type="geometry")
            elif qualifier_name == "tessellation_control":
                code += "// Tessellation Control Shader\n"
                code += self.generate_function(
                    func,
                    shader_type="tessellation_control",
                    function_name=self.rust_stage_entry_function_name(
                        qualifier_name, func
                    ),
                )
            elif qualifier_name == "tessellation_evaluation":
                code += "// Tessellation Evaluation Shader\n"
                code += self.generate_function(
                    func,
                    shader_type="tessellation_evaluation",
                    function_name=self.rust_stage_entry_function_name(
                        qualifier_name, func
                    ),
                )
            elif qualifier_name in self.rust_mesh_stage_names():
                code += f"// {qualifier_name.title()} Shader\n"
                code += self.generate_function(
                    func,
                    shader_type=qualifier_name,
                    function_name=self.rust_stage_entry_function_name(
                        qualifier_name, func
                    ),
                )
            else:
                code += self.generate_function(
                    func,
                    extra_params=self.rust_function_capture_params.get(
                        getattr(func, "name", None)
                    ),
                )

        # Handle shader stages (new AST structure)
        if hasattr(ast, "stages") and ast.stages:
            stage_entry_name_counts = self.stage_entry_name_counts(ast.stages)
            for stage_type, stage in ast.stages.items():
                for struct in getattr(stage, "local_structs", []) or []:
                    code += self.generate_struct(struct)

                saved_variable_types = self.variable_types.copy()
                for variable in getattr(stage, "local_variables", []) or []:
                    if self.is_rust_shared_variable(variable):
                        continue
                    self.register_variable_type(
                        variable.name,
                        self.get_variable_type(variable),
                        scope="static",
                    )
                    if variable.name in emitted_static_names:
                        continue
                    if self.is_rust_gpu_interface_declaration(variable):
                        pass
                    elif self.is_promoted_rust_entry_resource(variable):
                        var_type = self.rust_variable_source_type(variable)
                        self.register_glsl_buffer_block_metadata(variable)
                        code += self.rust_resource_metadata_comment(variable, var_type)
                    else:
                        code += self.generate_variable_static_declaration(variable)
                    emitted_static_names.add(variable.name)

                local_captures = {}
                if hasattr(stage, "entry_point"):
                    stage_name = str(stage_type).split(".")[-1].lower()
                    function_name = self.stage_entry_function_name(
                        stage_name,
                        stage.entry_point,
                        stage_entry_name_counts,
                    )
                    local_captures = self.collect_stage_local_function_captures(stage)
                    for captured_name, captured in local_captures.items():
                        self.rust_function_capture_params[captured_name] = list(
                            captured
                        ) + self.rust_resource_capture_params.get(captured_name, [])
                    code += f"// {stage_name.title()} Shader\n"
                    code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        function_name=function_name,
                        stage_node=stage,
                    )
                if hasattr(stage, "local_functions"):
                    for func in stage.local_functions:
                        code += self.generate_function(
                            func,
                            extra_params=self.rust_function_capture_params.get(
                                getattr(func, "name", None), []
                            ),
                        )
                for captured_name in local_captures:
                    self.rust_function_capture_params.pop(captured_name, None)
                    if captured_name in self.rust_resource_capture_params:
                        self.rust_function_capture_params[captured_name] = list(
                            self.rust_resource_capture_params[captured_name]
                        )

                self.variable_types = saved_variable_types

        code += self.generate_required_rust_mesh_helpers()
        code += self.generate_required_generic_math_traits()
        refuse_host_only_rust(self, code)
        if re.search(r"\bf16\b", code):
            # f16 is a nightly primitive type, which rust-gpu's toolchain enables per crate.
            code = code.replace("#![no_std]\n", "#![no_std]\n#![feature(f16)]\n", 1)
        return code

    def generate_rust_mesh_stage_comments(
        self, func, parameters, shader_type, stage_node=None
    ):
        lines = [f"// CrossGL mesh/task stage: stage={shader_type}\n"]
        numthreads = self.rust_mesh_stage_numthreads(func, shader_type)
        if numthreads is not None:
            lines.append(f"// CrossGL mesh/task numthreads: {', '.join(numthreads)}\n")
        local_size = self.rust_mesh_stage_layout_local_size(stage_node, shader_type)
        if local_size is not None:
            lines.append(f"// CrossGL mesh/task local size: {', '.join(local_size)}\n")

        if shader_type == "mesh":
            topology = self.rust_mesh_output_topology(func, shader_type, required=True)
            lines.append(f"// CrossGL mesh output topology: {topology}\n")
        max_vertices = self.rust_mesh_stage_positive_attribute(
            func, shader_type, "maxvertices", "max_vertices"
        )
        if max_vertices is not None:
            lines.append(f"// CrossGL mesh max vertices: {max_vertices}\n")
        max_primitives = self.rust_mesh_stage_positive_attribute(
            func, shader_type, "maxprimitives", "max_primitives"
        )
        if max_primitives is not None:
            lines.append(f"// CrossGL mesh max primitives: {max_primitives}\n")

        role_comments = self.generate_rust_mesh_parameter_role_comments(
            self.rust_mesh_param_infos(parameters)
        )
        if role_comments:
            lines.append(role_comments)
        return "".join(lines)

    def generate_rust_mesh_op_expression(self, expr):
        operation = getattr(expr, "operation", "mesh intrinsic")
        arguments = list(getattr(expr, "arguments", []) or [])
        stage_name = self.current_shader_type
        self.validate_rust_mesh_op_stage(operation, stage_name)
        self.validate_rust_mesh_op_arguments(operation, arguments, stage_name)

        args = [self.generate_expression(argument) for argument in arguments]
        if operation == "SetMeshOutputCounts":
            self.required_mesh_helpers.add("set_mesh_output_counts")
            return f"_crossgl_set_mesh_output_counts({', '.join(args)})"
        if operation == "DispatchMesh":
            self.required_mesh_helpers.add("dispatch_mesh")
            if len(args) == 3:
                return f"_crossgl_dispatch_mesh_without_payload({', '.join(args)})"
            return f"_crossgl_dispatch_mesh({', '.join(args)})"

        raise RustRefusal(f"Unsupported Rust mesh intrinsic {operation}")

    def generate_required_rust_mesh_helpers(self):
        if not self.required_mesh_helpers:
            return ""
        code = "// CrossGL mesh/task placeholders\n"
        if "set_mesh_output_counts" in self.required_mesh_helpers:
            code += (
                "pub fn _crossgl_set_mesh_output_counts("
                "vertices: u32, primitives: u32) {\n"
                "    let _ = (vertices, primitives);\n"
                "}\n\n"
            )
        if "dispatch_mesh" in self.required_mesh_helpers:
            code += (
                "pub fn _crossgl_dispatch_mesh<T>("
                "x: u32, y: u32, z: u32, payload: T) {\n"
                "    let _ = (x, y, z, payload);\n"
                "}\n\n"
                "pub fn _crossgl_dispatch_mesh_without_payload("
                "x: u32, y: u32, z: u32) {\n"
                "    let _ = (x, y, z);\n"
                "}\n\n"
            )
        return code

    def rust_stage_attribute_arguments(self, func, attribute_name):
        """Return arguments for a Rust-relevant stage control attribute."""
        expected = str(attribute_name).strip().lower().replace("-", "_")
        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name is None:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            if normalized == expected:
                return self.attribute_arguments(attr)
        return []

    def is_rust_shader_pod_value_type(self, rust_type):
        rust_type = self.unqualify_runtime_type_name(rust_type)
        return str(rust_type).startswith(
            (
                "Vec2<",
                "Vec3<",
                "Vec4<",
                "Mat2<",
                "Mat3<",
                "Mat4<",
                "Mat2x",
                "Mat3x",
                "Mat4x",
            )
        ) and str(rust_type).endswith(">")

    def generate_function(
        self,
        func,
        indent=0,
        shader_type=None,
        function_name=None,
        stage_node=None,
        extra_params=None,
    ):
        """Render one CrossGL function or shader entry point as Rust code."""
        if self.is_function_prototype(func):
            return ""

        code = ""
        code += "  " * indent
        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        saved_glsl_buffer_block_accesses = self.glsl_buffer_block_accesses.copy()
        saved_glsl_buffer_block_layouts = self.glsl_buffer_block_layouts.copy()
        self.local_variable_names = set()
        saved_return_type = self.current_return_type
        saved_shader_type = self.current_shader_type
        saved_generic_param_names = self.current_generic_param_names
        saved_function_generic_constraints = self.current_function_generic_constraints
        saved_mutated_names = self.current_mutated_names
        saved_output_parameters = self.current_rust_output_parameters
        saved_stage_outputs = self.current_rust_stage_outputs
        self.current_rust_output_parameters = []
        self.current_rust_stage_outputs = []
        self.current_generic_param_names = set(self.generic_param_names(func))
        self.current_shader_type = shader_type
        param_list = list(getattr(func, "parameters", getattr(func, "params", [])))
        if extra_params:
            param_list.extend(extra_params)
        promoted_resource_params = (
            self.rust_entry_resource_params_by_function_id.get(id(func), [])
            if shader_type
            else []
        )
        for p in param_list:
            self.register_glsl_buffer_block_metadata(p)
            self.register_variable_type(
                p.name,
                self.function_parameter_type(p),
                scope="local",
            )
        for resource_info in promoted_resource_params:
            self.register_variable_type(
                resource_info["node"].name,
                resource_info["type_name"],
                scope="local",
            )
        self.current_mutated_names = self.collect_mutated_binding_names(
            getattr(func, "body", None)
        )

        inferred_constraints = self.combined_function_generic_constraints(func)
        self.current_function_generic_constraints = inferred_constraints

        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"
        self.current_return_type = return_type
        return_semantic = self.node_semantic(func)
        fragment_output_info = None
        if self.map_type(return_type) == "()" and return_semantic is None:
            for param in param_list:
                semantic = self.node_semantic(param)
                if is_fragment_output_parameter(shader_type, param, semantic):
                    output_type = self.function_parameter_type(param)
                    fragment_output_info = (param, output_type, semantic)
                    return_type = output_type
                    self.current_return_type = return_type
                    return_semantic = semantic
                    break
        if fragment_output_info is not None:
            output_param = fragment_output_info[0]
            param_list = [param for param in param_list if param is not output_param]
        if shader_type == "geometry":
            self.validate_rust_geometry_stage(func, param_list)
        if shader_type in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_rust_tessellation_stage(func, param_list, shader_type)
        if shader_type in self.rust_mesh_stage_names():
            self.validate_rust_mesh_stage(func, param_list, shader_type, stage_node)
        self.validate_rust_return_semantic(shader_type, return_type, return_semantic)
        self.validate_rust_struct_return_semantics(shader_type, return_type)
        self.validate_rust_stage_parameter_semantics(shader_type, param_list)

        stage_interface = None
        if shader_type in GRAPHICS_STAGES:
            stage_interface = self.plan_rust_gpu_stage_interface(
                func, shader_type, param_list, return_type, return_semantic, stage_node
            )
            self.current_rust_stage_outputs = stage_interface.outputs
        elif shader_type == "compute":
            stage_interface = self.plan_rust_gpu_compute_interface(
                func, param_list, return_type
            )
        param_types = [self.function_parameter_type(p) for p in param_list]
        reference_lifetime = self.function_reference_return_lifetime(
            func,
            return_type,
            param_types,
        )

        params = []
        local_alias_declarations = []
        emitted_param_names = set()
        parameter_names = [p.name for p in param_list if getattr(p, "name", None)]
        if stage_interface is not None:
            params.extend(stage_interface.params)
            local_alias_declarations.extend(stage_interface.prologue)
        for p, param_type in zip(
            param_list if stage_interface is None else [], param_types
        ):
            self.register_glsl_buffer_block_metadata(p)
            source_param_name = p.name
            semantic = self.node_semantic(p)
            if shader_type == "compute":
                dispatch_thread_id_bridge = (
                    self.rust_compute_global_invocation_id_parameter_bridge(
                        p, param_type, semantic, emitted_param_names, parameter_names
                    )
                )
                if dispatch_thread_id_bridge is not None:
                    param_name, param_rust_type, local_alias = dispatch_thread_id_bridge
                    self.register_variable_type(param_name, "uint3", scope="local")
                    self.register_variable_type(
                        source_param_name, param_type, scope="local"
                    )
                    params.append(f"{param_name}: {param_rust_type}")
                    local_alias_declarations.append(local_alias)
                    emitted_param_names.add(param_name)
                    continue

            self.register_variable_type(source_param_name, param_type, scope="local")
            param_name = self.rust_identifier(source_param_name)
            if shader_type is None and is_output_parameter(p):
                # GLSL copy-in/copy-out; see rust_output_parameters.
                reference_name = self.rust_output_parameter_reference_name(
                    func, param_name
                )
                mapped_param_type = self.map_function_parameter_type_with_lifetime(
                    param_type, reference_lifetime
                )
                params.append(f"{reference_name}: &mut {mapped_param_type}")
                local_alias_declarations.append(
                    f"let mut {param_name}: {mapped_param_type} = *{reference_name};"
                )
                self.current_rust_output_parameters.append((reference_name, param_name))
                emitted_param_names.add(reference_name)
                continue
            if source_param_name in self.current_mutated_names or (
                self.function_parameter_requires_mut_binding(
                    p,
                    inferred_constraints,
                )
            ):
                param_name = f"mut {param_name}"
            params.append(
                f"{param_name}: "
                f"{self.map_function_parameter_type_with_lifetime(param_type, reference_lifetime)}"
            )
            emitted_param_names.add(param_name)
        for info in promoted_resource_params:
            params.append(self.generate_rust_resource_parameter(info))
            if info.get("kind") == UNIFORM_VALUE_KIND:
                # A uniform is read-only, so the entry point reads it once into a local of the same name.
                uniform_name = self.rust_identifier(info["node"].name)
                local_alias_declarations.append(
                    f"let {uniform_name}: {self.map_type(info['type_name'])} = *{uniform_name};"
                )
            elif info.get("kind") == SPEC_CONSTANT_KIND:
                spec_name = self.rust_identifier(info["node"].name)
                local = spec_constant_local(
                    spec_name, info, self.map_type(info["type_name"])
                )
                if local is not None:
                    local_alias_declarations.append(local)

        params_str = ", ".join(params) if params else ""

        if shader_type in GRAPHICS_STAGES:
            code += self.rust_gpu_stage_attribute(shader_type, stage_interface)
        elif shader_type == "compute":
            code += self.generate_rust_compute_stage_attribute(func, stage_node)
        elif shader_type == "geometry":
            code += self.generate_rust_geometry_stage_comments(func, param_list)
        elif shader_type in {"tessellation_control", "tessellation_evaluation"}:
            code += self.generate_rust_tessellation_stage_comments(
                func,
                param_list,
                shader_type,
            )
        elif shader_type in self.rust_mesh_stage_names():
            code += f"// CrossGL shader stage: {shader_type}\n"
            code += self.generate_rust_mesh_stage_comments(
                func, param_list, shader_type, stage_node
            )
        elif shader_type in self.rust_ray_stage_names():
            code += (
                "// CrossGL ray stage: "
                f"{self.rust_ray_stage_metadata_name(shader_type)}\n"
            )
        if return_semantic and stage_interface is None:
            code += (
                f"// CrossGL return semantic: {self.map_semantic(return_semantic)}\n"
            )
        if shader_type and stage_interface is None:
            for param in param_list:
                param_semantic = self.node_semantic(param)
                if param_semantic:
                    code += (
                        f"// CrossGL parameter semantic: {param.name}: "
                        f"{self.map_semantic(param_semantic)}\n"
                    )

        generic_params = self.format_function_generic_params(
            func,
            extra_constraints=inferred_constraints,
            lifetime=reference_lifetime,
        )
        emitted_name = self.rust_identifier(function_name or func.name)
        emitted_return_type = (
            "()"
            if stage_interface is not None
            else self.map_type_with_lifetime(return_type, reference_lifetime)
        )
        code += (
            f"pub fn {emitted_name}{generic_params}({params_str}) "
            f"-> {emitted_return_type} {{\n"
        )

        if stage_node is not None:
            for variable in getattr(stage_node, "local_variables", []) or []:
                if self.is_rust_shared_variable(
                    variable
                ) and not self.is_promoted_rust_entry_resource(variable):
                    code += self.generate_statement(variable, indent + 1)
        for declaration in local_alias_declarations:
            code += "    " * (indent + 1) + declaration + "\n"

        body = getattr(func, "body", [])
        body_statements = []
        if fragment_output_info is not None:
            output_param, output_type, _semantic = fragment_output_info
            output_name = self.rust_identifier(getattr(output_param, "name", None))
            self.register_variable_type(
                getattr(output_param, "name", None),
                output_type,
                scope="local",
            )
            code += (
                "  " * (indent + 1)
                + f"let mut {output_name}: "
                + f"{self.map_type_with_lifetime(output_type, reference_lifetime)};\n"
            )
        if hasattr(body, "statements"):
            body_statements = body.statements
        elif isinstance(body, list):
            body_statements = body
        if self.current_rust_stage_outputs:
            # The entry's result goes to output parameters, so a trailing value expression is its return.
            body_statements = tail_expression_as_return(body_statements)
        for stmt in body_statements:
            code += self.generate_statement(stmt, indent + 1)
        if self.current_rust_output_parameters and not self.statement_body_terminates(
            body_statements
        ):
            code += self.rust_output_parameter_copy_back("    " * (indent + 1))
        if fragment_output_info is not None and not self.statement_body_terminates(
            body_statements
        ):
            output_param = fragment_output_info[0]
            output_name = self.rust_identifier(getattr(output_param, "name", None))
            if stage_interface is not None:
                code += self.generate_rust_stage_output_return(
                    "  " * (indent + 1), output_name
                )
            else:
                code += "  " * (indent + 1) + f"return {output_name};\n"

        code += "  " * indent + "}\n\n"
        self.variable_types = saved_variable_types
        self.local_variable_names = saved_local_variable_names
        self.glsl_buffer_block_accesses = saved_glsl_buffer_block_accesses
        self.glsl_buffer_block_layouts = saved_glsl_buffer_block_layouts
        self.current_return_type = saved_return_type
        self.current_shader_type = saved_shader_type
        self.current_generic_param_names = saved_generic_param_names
        self.current_function_generic_constraints = saved_function_generic_constraints
        self.current_mutated_names = saved_mutated_names
        self.current_rust_output_parameters = saved_output_parameters
        self.current_rust_stage_outputs = saved_stage_outputs
        return code

    def is_function_prototype(self, func):
        return isinstance(func, FunctionNode) and getattr(func, "body", None) is None

    def builtin_function_result_type(self, func_name, arguments):
        """Infer result types for shader intrinsics emitted as Rust prelude calls."""
        if not isinstance(func_name, str):
            return None

        if not self.is_user_defined_function(func_name):
            atomic_type = rust_memory_atomic_result_type(self, func_name, arguments)
            if atomic_type is not None:
                return atomic_type
            math_type = rust_gpu_math_result_type(self, func_name, arguments)
            if math_type is not None:
                return math_type

        mapped_name = self.mapped_function_name(func_name, len(arguments), arguments)
        arg_types = [self.expression_result_type(arg) for arg in arguments]

        bitcast_alias_component_type = {
            "asfloat": "float",
            "asint": "int",
            "asuint": "uint",
        }.get(func_name)
        if bitcast_alias_component_type is not None and arg_types:
            source_component_type = self.bitcast_argument_component_type(arg_types[0])
            if source_component_type not in {"float", "int", "uint"}:
                return None
            return self.bitcast_value_type(
                arg_types[0],
                bitcast_alias_component_type,
            )

        wave_type = self.wave_helper_result_type(mapped_name, arg_types)
        if wave_type is not None:
            return wave_type

        handled = self.rust_resource_builtin_result_type(
            func_name, arg_types, mapped_name
        )
        if handled is not UNHANDLED:
            return handled

        handled = self.rust_math_builtin_result_type(arg_types, mapped_name)
        if handled is not UNHANDLED:
            return handled

        handled = self.rust_bitwise_numeric_builtin_result_type(arg_types, mapped_name)
        if handled is not UNHANDLED:
            return handled

        return None

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def map_type(self, vtype):
        """Map a CrossGL type name or type node to a Rust type string."""
        if vtype is None:
            return "f32"

        vtype_str = self.type_name_string(vtype)

        reference_parts = self.reference_type_parts(vtype_str)
        if reference_parts is not None:
            is_mutable, referenced_type = reference_parts
            mutability = "mut " if is_mutable else ""
            return f"&{mutability}{self.map_reference_pointee_type(referenced_type)}"

        pointer_parts = self.pointer_type_parts(vtype_str)
        if pointer_parts is not None:
            is_mutable, pointee_type = pointer_parts
            mutability = "mut" if is_mutable else "const"
            return f"*{mutability} {self.map_reference_pointee_type(pointee_type)}"

        function_pointer = self.map_function_pointer_type(vtype_str)
        if function_pointer is not None:
            return function_pointer

        if self.callable_signature_parts(vtype_str) is not None:
            return vtype_str

        patch_type = self.rust_tessellation_patch_mapped_type(vtype_str)
        if patch_type is not None:
            return patch_type

        matrix_alias = self.normalize_hlsl_matrix_type_name(vtype_str)
        if matrix_alias != vtype_str:
            return self.map_type(matrix_alias)

        base_type, sizes = self.c_array_type_parts(vtype_str)
        # Brackets that do not end the name, as in the callee ``pick<float[2]>``,
        # are not an array suffix; the generic mapping below spells its argument.
        if sizes:
            base_mapped = self.map_type(base_type)
            rust_type = self.qualify_colliding_runtime_type(base_mapped)
            for size in reversed(sizes):
                if size is None:
                    rust_type = f"Vec<{rust_type}>"
                else:
                    rust_type = f"[{rust_type}; {size}]"
            return rust_type

        mapped_type = self.type_mapping.get(vtype_str)
        if mapped_type is not None:
            return self.qualify_colliding_runtime_type(mapped_type)

        generic_type = self.map_generic_type_string(vtype_str)
        if generic_type is not None:
            return generic_type

        return self.rust_type_identifier(vtype_str)

    def rust_type_identifier(self, name):
        """A user type name; a Rust keyword (``type``, ``match``) is written as a raw identifier ``r#type``."""
        name = str(name)
        if name in self.rust_reserved_identifiers() and name not in {
            "self",
            "Self",
            "super",
            "crate",
        }:
            return f"r#{name}"
        return name

    def map_type_with_lifetime(self, vtype, lifetime):
        if lifetime is None:
            return self.map_type(vtype)

        vtype_str = self.type_name_string(vtype)
        reference_parts = self.reference_type_parts(vtype_str)
        if reference_parts is None:
            return self.map_type(vtype)

        is_mutable, referenced_type = reference_parts
        mutability = "mut " if is_mutable else ""
        return (
            f"&{lifetime} {mutability}"
            f"{self.map_reference_pointee_type(referenced_type)}"
        )

    def reference_type_parts(self, type_name):
        type_name = str(type_name).strip()
        if type_name.startswith("&mut "):
            return True, type_name[len("&mut ") :].strip()
        if type_name.startswith("&") and not type_name.startswith("&'"):
            return False, type_name[1:].strip()
        return None

    def reference_type_parts_for_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None
        return self.reference_type_parts(type_name)

    def pointer_type_parts(self, type_name):
        type_name = str(type_name).strip()
        if type_name.startswith("*mut "):
            return True, type_name[len("*mut ") :].strip()
        if type_name.startswith("*const "):
            return False, type_name[len("*const ") :].strip()
        return None

    def map_reference_pointee_type(self, type_name):
        type_name = str(type_name).strip()
        if type_name in {"str", "string"}:
            return "str"
        if type_name.endswith("[]") and type_name.count("[") == 1:
            # A reference to an unsized array is a slice: a storage buffer's runtime array, whose 3-vectors take
            # 4-vector slots (rust_gpu_storage).
            element = type_name[:-2].strip()
            padded = PADDED_ELEMENTS.get(element)
            if padded is not None:
                return f"[{padded[0]}]"
            if self.normalize_scalar_type(element) == "bool" or element.startswith(
                "bvec"
            ):
                raise RustRefusal(
                    "resource: a storage buffer of bool has no SPIR-V layout; rust-gpu cannot bind one"
                )
            return f"[{self.map_type(element)}]"
        return self.map_type(type_name)

    def map_function_pointer_type(self, type_name):
        type_name = str(type_name).strip()
        if not type_name.startswith("fn"):
            return None

        open_index = type_name.find("(")
        if type_name[:open_index].strip() != "fn":
            return None
        close_index = self.find_matching_delimiter(type_name, open_index, "(", ")")
        if close_index is None:
            return None

        params_text = type_name[open_index + 1 : close_index].strip()
        suffix = type_name[close_index + 1 :].strip()
        if not suffix.startswith("->"):
            return None

        return_type = suffix[2:].strip()
        params = []
        if params_text:
            params = [
                self.map_type(param_type.strip())
                for param_type in self.split_top_level_list(params_text)
                if param_type.strip()
            ]
        return f"fn({', '.join(params)}) -> {self.map_type(return_type)}"

    def qualify_colliding_runtime_type(self, rust_type):
        """Disambiguate built-in math types when user declarations reuse their names."""
        rust_type = str(rust_type)
        if rust_type.startswith("math::"):
            return rust_type
        base_type = rust_type.split("<", 1)[0]
        if base_type in self.runtime_type_collisions:
            return f"math::{rust_type}"
        return rust_type

    def map_generic_type_string(self, type_name):
        """Map primitive arguments inside a generic type string."""
        if "<" not in type_name or not type_name.endswith(">"):
            return None

        base_type, args_text = type_name.split("<", 1)
        args_text = args_text[:-1]
        args = self.split_top_level_generic_args(args_text)
        mapped_args = [self.map_type(arg.strip()) for arg in args if arg.strip()]
        if not mapped_args:
            return None

        mapped_base = self.type_mapping.get(base_type)
        if mapped_base is None:
            reason = rust_generic_template_reason(base_type)
            if reason is not None:
                return self.refuse_lowering(reason, type_name)
            # A user or Rust template (Option, Result, a declared generic structure) is spelled the same.
            mapped_base = base_type
        if "<" in mapped_base:
            mapped_base = mapped_base.split("<", 1)[0]
        return f"{mapped_base}<{', '.join(mapped_args)}>"

    def split_top_level_generic_args(self, args_text):
        """Split generic arguments without breaking nested generic arguments."""
        args = []
        current = []
        depth = 0

        for char in args_text:
            if char == "<":
                depth += 1
            elif char == ">" and depth > 0:
                depth -= 1
            elif char == "," and depth == 0:
                args.append("".join(current))
                current = []
                continue

            current.append(char)

        if current:
            args.append("".join(current))

        return args

    def rust_constructor_path(self, rust_type):
        """Return a Rust path suitable for associated constructor calls."""
        rust_type = str(rust_type)
        if "<" not in rust_type:
            return rust_type
        return rust_type.replace("<", "::<", 1)

    def map_operator(self, op):
        """``op`` spelled for Rust; an operator CrossGL does not define is refused."""
        return rust_operator(op)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to the Rust backend attribute name."""
        if not semantic:
            return ""
        mapped = self.semantic_map.get(semantic)
        if mapped is not None:
            return mapped
        reason = rust_unmapped_semantic_reason(semantic)
        if reason is not None:
            return self.refuse_lowering(reason, str(semantic))
        return semantic
