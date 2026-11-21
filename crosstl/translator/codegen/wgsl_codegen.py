"""CrossGL-to-WGSL code generator."""

import copy
import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    CastNode,
    ConstructorNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MemberAccessNode,
    NamedType,
    PointerType,
    RangeNode,
    ReturnNode,
    StructMemberNode,
    StructNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from ..refusals import WGSLRefusal
from .c_family_operands import prefix_unary_operand
from .call_graph import validate_recursion_free
from .comma_expressions import is_comma_expression
from .cooperative_matrix_refusals import refuse_cooperative_matrices
from .enum_utils import (
    generic_enum_specialization_name,
)
from .precise_evaluation import refuse_precise_evaluation
from .relational_builtins import lower_relational_builtins, wgsl_logical_not_operand
from .sampler_pairs import canonicalize_separate_textures
from .spirv_union_storage import lower_union_storage
from .stage_utils import normalize_stage_name, requested_stage_name
from .wgsl_array_constructors import (
    wgsl_array_constructor,
    wgsl_array_constructor_type,
    wgsl_array_element,
)
from .wgsl_bindings import WGSLBindingMixin
from .wgsl_boolean_xor import wgsl_boolean_xor
from .wgsl_buffer_atomics import ATOMIC_NAMES, WGSLBufferAtomicsMixin
from .wgsl_buffer_resources import WGSLBufferResourceMixin
from .wgsl_builtin_options import WGSLBuiltinOptionMixin
from .wgsl_call_lowering import WGSLCallLoweringMixin
from .wgsl_composite_conditionals import WGSLCompositeConditionalMixin
from .wgsl_conditional_effects import WGSLConditionalEffectsMixin
from .wgsl_constructors import WGSLConstructorMixin
from .wgsl_cube_texture_offsets import (
    OFFSET_TEXTURE_LOOKUPS,
    WGSLCubeTextureOffsetUnsupportedError,
    is_cube_array_texture_type,
)
from .wgsl_entry_outputs import (
    EntryOutputLoweringError,
    entry_output_parameters,
    lower_entry_output_parameters,
)
from .wgsl_expression_values import WGSLExpressionValueMixin
from .wgsl_float_classification import (
    is_float_classification_call,
    wgsl_float_classification_call,
)
from .wgsl_frexp_modf import (
    is_two_argument_frexp_or_modf,
    wgsl_frexp_modf_call,
)
from .wgsl_functions import WGSLFunctionMixin
from .wgsl_generic_structs import WGSLGenericStructMixin
from .wgsl_global_declarations import WGSLGlobalDeclarationMixin
from .wgsl_identifiers import WGSLIdentifierMixin
from .wgsl_image_atomics import WGSLImageAtomicUnsupportedError, is_image_atomic_name
from .wgsl_integer_builtins import lower_integer_builtin_inputs
from .wgsl_integer_operands import (
    BITWISE_OPERATORS,
    MIXED_INTEGER_OPERATORS,
    SHIFT_OPERATORS,
    mixed_integer_call_is_unsigned,
    mixed_integer_call_type,
    unsigned_bitwise_operand_types,
    wgsl_bitwise_operands,
    wgsl_mixed_integer_call_arguments,
    wgsl_mixed_integer_operands,
    wgsl_shift_amount,
)
from .wgsl_integer_textures import FILTERED_TEXTURE_LOOKUPS
from .wgsl_loop_continuing import WGSLLoopContinuingMixin
from .wgsl_math_calls import WGSLMathCallMixin
from .wgsl_matrix_inverse import (
    is_matrix_inverse_call,
    wgsl_matrix_inverse_call,
)
from .wgsl_matrix_resize import wgsl_matrix_resize
from .wgsl_narrow_storage import narrow_storage_word_type
from .wgsl_parameter_lowering import WGSLParameterLoweringMixin
from .wgsl_program_collection import WGSLProgramCollectionMixin
from .wgsl_program_sections import WGSLProgramSectionMixin
from .wgsl_reference_argument_copies import WGSLReferenceArgumentCopiesMixin
from .wgsl_resource_arrays import WGSLResourceArrayMixin
from .wgsl_specialization_constants import WGSLSpecializationPlan
from .wgsl_specialized_workgroup_size import WorkgroupSizeOverrides
from .wgsl_stage_interfaces import WGSLStageInterfaceMixin
from .wgsl_statements import WGSLStatementMixin
from .wgsl_storage_textures import WGSLStorageTextureMixin
from .wgsl_struct_initializers import (
    wgsl_struct_array_initializer,
    wgsl_struct_initializer,
)
from .wgsl_subgroup_operations import wgsl_wave_operation, wgsl_wave_result_type
from .wgsl_texture_calls import WGSLTextureCallMixin
from .wgsl_type_names import WGSLTypeNameMixin
from .wgsl_uniform_registers import wgsl_uniform_register_load
from .wgsl_unsigned_builtin_members import is_int_builtin_member
from .wgsl_unsigned_negation import wgsl_unsigned_negation
from .wgsl_value_side_effects import WGSLValueSideEffectsMixin, is_increment
from .wgsl_value_types import WGSLValueTypeMixin
from .wgsl_vector_initializers import (
    wgsl_vector_array_initializer,
    wgsl_vector_initializer,
)
from .wgsl_workgroup_locals import hoist_workgroup_locals
from .wgsl_workgroup_size_inputs import lower_workgroup_size_inputs


class WGSLAggregateInitializerError(ValueError):
    """Raised when an aggregate initializer has no faithful WGSL lowering."""

    project_diagnostic_code = "project.translate.wgsl-aggregate-initializer-invalid"
    missing_capabilities = ("wgsl.aggregate-initializer-lowering",)

    def __init__(
        self,
        message,
        *,
        expected_type=None,
        element_count=None,
        expected_element_count=None,
    ):
        super().__init__(message)
        self.expected_type = expected_type
        self.element_count = element_count
        self.expected_element_count = expected_element_count


class WGSLCodeGen(
    WGSLStageInterfaceMixin,
    WGSLFunctionMixin,
    WGSLGenericStructMixin,
    WGSLGlobalDeclarationMixin,
    WGSLStatementMixin,
    WGSLExpressionValueMixin,
    WGSLCallLoweringMixin,
    WGSLMathCallMixin,
    WGSLTextureCallMixin,
    WGSLConstructorMixin,
    WGSLTypeNameMixin,
    WGSLBuiltinOptionMixin,
    WGSLBufferResourceMixin,
    WGSLResourceArrayMixin,
    WGSLStorageTextureMixin,
    WGSLValueTypeMixin,
    WGSLIdentifierMixin,
    WGSLBindingMixin,
    WGSLProgramCollectionMixin,
    WGSLBufferAtomicsMixin,
    WGSLLoopContinuingMixin,
    WGSLValueSideEffectsMixin,
    WGSLConditionalEffectsMixin,
    WGSLReferenceArgumentCopiesMixin,
    WGSLCompositeConditionalMixin,
    WGSLParameterLoweringMixin,
    WGSLProgramSectionMixin,
):
    """Generate WebGPU WGSL output from CrossGL ASTs."""

    SUPPORTED_STAGE_NAMES = {"vertex", "fragment", "compute"}
    UNSUPPORTED_STAGE_NAMES = {
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
        "mesh",
        "task",
        "amplification",
        "object",
        "ray_generation",
        "raygen",
        "ray_intersection",
        "intersection",
        "ray_any_hit",
        "any_hit",
        "ray_closest_hit",
        "closest_hit",
        "ray_miss",
        "miss",
        "ray_callable",
        "callable",
    }
    VECTOR_TYPE_RE = re.compile(r"^(?:vec|float|int|uint|bool|ivec|uvec|bvec)([234])$")
    FLOAT16_VECTOR_TYPE_RE = re.compile(r"^(?:f16vec|half)([234])$")
    MATRIX_TYPE_RE = re.compile(r"^(?:mat|float)([234])(?:x([234]))?$")
    TYPE_CONSTRUCTOR_RE = re.compile(
        r"^(?:f16vec[234]|half[234]|vec[234]|float[234]|int[234]|uint[234]|"
        r"bool[234]|ivec[234]|uvec[234]|bvec[234]|mat[234](?:x[234])?|"
        r"float[234]x[234])$"
    )

    PRIMITIVE_TYPE_MAP = {
        "void": "void",
        "bool": "bool",
        "boolean": "bool",
        "int": "i32",
        "i32": "i32",
        "short": "i32",
        "uint": "u32",
        "u32": "u32",
        "unsigned": "u32",
        "float": "f32",
        "f32": "f32",
        "f16": "f32",
        "float16": "f32",
        "half": "f32",
    }
    BUILTIN_SEMANTICS = {
        "gl_position": "position",
        "gl_fragcoord": "position",
        "position_builtin": "position",
        "sv_position": "position",
        "frag_depth": "frag_depth",
        "gl_fragdepth": "frag_depth",
        "sv_depth": "frag_depth",
        "front_facing": "front_facing",
        "gl_frontfacing": "front_facing",
        "sv_isfrontface": "front_facing",
        "sample_index": "sample_index",
        "gl_sampleid": "sample_index",
        "sv_sampleindex": "sample_index",
        "sample_mask": "sample_mask",
        "gl_samplemask": "sample_mask",
        "gl_samplemaskin": "sample_mask",
        "sv_coverage": "sample_mask",
        "vertex_index": "vertex_index",
        "vertexid": "vertex_index",
        "gl_vertexid": "vertex_index",
        "gl_vertexindex": "vertex_index",
        "sv_vertexid": "vertex_index",
        "instance_index": "instance_index",
        "instanceid": "instance_index",
        "gl_instanceid": "instance_index",
        "gl_instanceindex": "instance_index",
        "sv_instanceid": "instance_index",
        "primitive_index": "primitive_index",
        "primitive_id": "primitive_index",
        "primitiveid": "primitive_index",
        "gl_primitiveid": "primitive_index",
        "sv_primitiveid": "primitive_index",
        "global_invocation_id": "global_invocation_id",
        "gl_globalinvocationid": "global_invocation_id",
        "sv_dispatchthreadid": "global_invocation_id",
        "local_invocation_id": "local_invocation_id",
        "gl_localinvocationid": "local_invocation_id",
        "sv_groupthreadid": "local_invocation_id",
        "local_invocation_index": "local_invocation_index",
        "gl_localinvocationindex": "local_invocation_index",
        "sv_groupindex": "local_invocation_index",
        "workgroup_id": "workgroup_id",
        "gl_workgroupid": "workgroup_id",
        "sv_groupid": "workgroup_id",
        "num_workgroups": "num_workgroups",
        "gl_numworkgroups": "num_workgroups",
        "subgroup_size": "subgroup_size",
        "gl_subgroupsize": "subgroup_size",
        "subgroup_invocation_id": "subgroup_invocation_id",
        "gl_subgroupinvocationid": "subgroup_invocation_id",
        "num_subgroups": "num_subgroups",
        "gl_numsubgroups": "num_subgroups",
        "subgroup_id": "subgroup_id",
        "gl_subgroupid": "subgroup_id",
    }
    BUILTIN_IDENTIFIER_ALIASES = {
        "gl_Position": "position",
        "gl_FragCoord": "position",
        "SV_Position": "position",
        "gl_GlobalInvocationID": "global_invocation_id",
        "SV_DispatchThreadID": "global_invocation_id",
        "gl_LocalInvocationID": "local_invocation_id",
        "SV_GroupThreadID": "local_invocation_id",
        "gl_LocalInvocationIndex": "local_invocation_index",
        "SV_GroupIndex": "local_invocation_index",
        "gl_WorkGroupID": "workgroup_id",
        "SV_GroupID": "workgroup_id",
        "gl_NumWorkGroups": "num_workgroups",
        # The subgroup built-in inputs (WGSL section 13.3.1.1). naga 30 reads them without the
        # `enable subgroups;` directive and rejects that directive as not yet implemented.
        "gl_SubgroupSize": "subgroup_size",
        "gl_SubgroupInvocationID": "subgroup_invocation_id",
        "gl_NumSubgroups": "num_subgroups",
        "gl_SubgroupID": "subgroup_id",
    }
    WORKGROUP_SIZE_IDENTIFIER_ALIASES = {"gl_WorkGroupSize"}
    INPUT_BUILTIN_TYPE_MAP = {
        "position": "vec4<f32>",
        "vertex_index": "u32",
        "instance_index": "u32",
        "global_invocation_id": "vec3<u32>",
        "local_invocation_id": "vec3<u32>",
        "local_invocation_index": "u32",
        "workgroup_id": "vec3<u32>",
        "num_workgroups": "vec3<u32>",
        "subgroup_size": "u32",
        "subgroup_invocation_id": "u32",
        "num_subgroups": "u32",
        "subgroup_id": "u32",
    }
    FUNCTION_NAME_MAP = {
        "atan2": "atan2",
        "bitfieldReverse": "reverseBits",
        "clz": "countLeadingZeros",
        "ctz": "countTrailingZeros",
        "faceforward": "faceForward",
        "packHalf2x16": "pack2x16float",
        "packSnorm2x16": "pack2x16snorm",
        "packUnorm2x16": "pack2x16unorm",
        "packSnorm4x8": "pack4x8snorm",
        "packUnorm4x8": "pack4x8unorm",
        "unpackHalf2x16": "unpack2x16float",
        "unpackSnorm2x16": "unpack2x16snorm",
        "unpackUnorm2x16": "unpack2x16unorm",
        "unpackSnorm4x8": "unpack4x8snorm",
        "unpackUnorm4x8": "unpack4x8unorm",
        "fract": "fract",
        "inversesqrt": "inverseSqrt",
        "inverseSqrt": "inverseSqrt",
        "lerp": "mix",
        "mix": "mix",
        "rsqrt": "inverseSqrt",
        # WGSL round() rounds halfway cases to the nearest even integer.
        "roundEven": "round",
        "saturate": "saturate",
    }
    BIT_CAST_FUNCTIONS = {
        "asuint": "u32",
        "asint": "i32",
        "asfloat": "f32",
        "floatBitsToUint": "u32",
        "floatBitsToInt": "i32",
        "uintBitsToFloat": "f32",
        "intBitsToFloat": "f32",
    }
    DERIVATIVE_FUNCTION_NAME_MAP = {
        "dfdx": "dpdx",
        "ddx": "dpdx",
        "dfdxfine": "dpdxFine",
        "ddxfine": "dpdxFine",
        "ddx_fine": "dpdxFine",
        "dfdxcoarse": "dpdxCoarse",
        "ddxcoarse": "dpdxCoarse",
        "ddx_coarse": "dpdxCoarse",
        "dfdy": "dpdy",
        "ddy": "dpdy",
        "dfdyfine": "dpdyFine",
        "ddyfine": "dpdyFine",
        "ddy_fine": "dpdyFine",
        "dfdycoarse": "dpdyCoarse",
        "ddycoarse": "dpdyCoarse",
        "ddy_coarse": "dpdyCoarse",
        # The GLSL importer writes fwidthFine and fwidthCoarse with underscores.
        "fwidth_fine": "fwidthFine",
        "fwidth_coarse": "fwidthCoarse",
    }
    STAGE_INPUT_BUILTINS = {
        "vertex": {"instance_index", "vertex_index"},
        "fragment": {"position", "subgroup_invocation_id", "subgroup_size"},
        "compute": {
            "global_invocation_id",
            "local_invocation_id",
            "local_invocation_index",
            "num_subgroups",
            "num_workgroups",
            "subgroup_id",
            "subgroup_invocation_id",
            "subgroup_size",
            "workgroup_id",
        },
    }
    RESOURCE_TYPE_NAMES = {
        "image1d",
        "image1darray",
        "image2d",
        "image2darray",
        "image2dms",
        "image2dmsarray",
        "image3d",
        "imagebuffer",
        "imagecube",
        "imagecubearray",
        "iimage1d",
        "iimage1darray",
        "iimage2d",
        "iimage2darray",
        "iimage2dms",
        "iimage2dmsarray",
        "iimage3d",
        "iimagebuffer",
        "iimagecube",
        "iimagecubearray",
        "sampler",
        "sampler1d",
        "sampler1darray",
        "sampler2d",
        "sampler2darray",
        "sampler2darrayshadow",
        "sampler2dms",
        "sampler2dmsarray",
        "sampler2dshadow",
        "sampler3d",
        "samplercube",
        "samplercubearray",
        "samplercubearrayshadow",
        "samplercubeshadow",
        "samplerstate",
        "samplercomparisonstate",
        "texture1d",
        "texture1darray",
        "texture2d",
        "texture2darray",
        "texture2dms",
        "texture2dmsarray",
        "texture3d",
        "texturecube",
        "texturecubearray",
        "uimage1d",
        "uimage1darray",
        "uimage2d",
        "uimage2darray",
        "uimage2dms",
        "uimage2dmsarray",
        "uimage3d",
        "uimagebuffer",
        "uimagecube",
        "uimagecubearray",
        "isampler1d",
        "isampler1darray",
        "isampler2d",
        "isampler2darray",
        "isampler2dms",
        "isampler2dmsarray",
        "isampler2drect",
        "isampler3d",
        "isamplerbuffer",
        "isamplercube",
        "isamplercubearray",
        "sampler2drect",
        "sampler2drectshadow",
        "samplerbuffer",
        "usampler1d",
        "usampler1darray",
        "usampler2d",
        "usampler2darray",
        "usampler2dms",
        "usampler2dmsarray",
        "usampler2drect",
        "usampler3d",
        "usamplerbuffer",
        "usamplercube",
        "usamplercubearray",
    }
    STRUCTURED_BUFFER_TYPE_NAMES = {
        "rwstructuredbuffer",
        "structuredbuffer",
    }
    WRITABLE_STRUCTURED_BUFFER_TYPE_NAMES = {
        "rwstructuredbuffer",
    }
    CONSTANT_BUFFER_TYPE_NAMES = {
        "constantbuffer",
    }
    UNSUPPORTED_STORAGE_BUFFER_TYPE_NAMES = {
        "appendstructuredbuffer",
        "byteaddressbuffer",
        "consumestructuredbuffer",
        "rwbyteaddressbuffer",
    }
    STRUCTURED_BUFFER_FREE_HELPERS = {
        "buffer_load",
        "buffer_store",
    }
    UNSUPPORTED_STORAGE_BUFFER_FREE_HELPERS = {
        "buffer_append",
        "buffer_consume",
        "buffer_dimensions",
    }
    STRUCTURED_BUFFER_MEMBER_HELPERS = {
        "load",
        "store",
    }
    UNSUPPORTED_STRUCTURED_BUFFER_MEMBER_HELPERS = {
        "append",
        "consume",
        "decrementcounter",
        "getdimensions",
        "incrementcounter",
        "load2",
        "load3",
        "load4",
        "store2",
        "store3",
        "store4",
    }
    SUPPORTED_GLSL_BUFFER_BLOCK_LAYOUTS = {"std430"}
    SAMPLED_TEXTURE_TYPE_MAP = {
        "sampler1d": "texture_1d<f32>",
        "sampler2d": "texture_2d<f32>",
        "sampler2darray": "texture_2d_array<f32>",
        "sampler2darrayshadow": "texture_depth_2d_array",
        "sampler2dshadow": "texture_depth_2d",
        "sampler3d": "texture_3d<f32>",
        "samplercube": "texture_cube<f32>",
        "samplercubearray": "texture_cube_array<f32>",
        "samplercubearrayshadow": "texture_depth_cube_array",
        "samplercubeshadow": "texture_depth_cube",
        "texture1d": "texture_1d<f32>",
        "texture2d": "texture_2d<f32>",
        "texture2darray": "texture_2d_array<f32>",
        "texture3d": "texture_3d<f32>",
        "texturecube": "texture_cube<f32>",
        "texturecubearray": "texture_cube_array<f32>",
        "isampler1d": "texture_1d<i32>",
        "isampler2d": "texture_2d<i32>",
        "isampler2darray": "texture_2d_array<i32>",
        "isampler3d": "texture_3d<i32>",
        "isamplercube": "texture_cube<i32>",
        "isamplercubearray": "texture_cube_array<i32>",
        "usampler1d": "texture_1d<u32>",
        "usampler2d": "texture_2d<u32>",
        "usampler2darray": "texture_2d_array<u32>",
        "usampler3d": "texture_3d<u32>",
        "usamplercube": "texture_cube<u32>",
        "usamplercubearray": "texture_cube_array<u32>",
    }
    STORAGE_TEXTURE_DIMENSION_MAP = {
        "image1d": "1d",
        "image2d": "2d",
        "image2darray": "2d_array",
        "image3d": "3d",
        "imagecube": "2d_array",
        "imagecubearray": "2d_array",
        "iimage1d": "1d",
        "iimage2d": "2d",
        "iimage2darray": "2d_array",
        "iimage3d": "3d",
        "iimagecube": "2d_array",
        "iimagecubearray": "2d_array",
        "uimage1d": "1d",
        "uimage2d": "2d",
        "uimage2darray": "2d_array",
        "uimage3d": "3d",
        "uimagecube": "2d_array",
        "uimagecubearray": "2d_array",
    }
    STORAGE_TEXTURE_FORMAT_MAP = {
        "r32f": "r32float",
        "r32i": "r32sint",
        "r32ui": "r32uint",
        "rg32f": "rg32float",
        "rg32i": "rg32sint",
        "rg32ui": "rg32uint",
        "rgba8": "rgba8unorm",
        "rgba8snorm": "rgba8snorm",
        "rgba8i": "rgba8sint",
        "rgba8ui": "rgba8uint",
        "rgba16f": "rgba16float",
        "rgba16i": "rgba16sint",
        "rgba16ui": "rgba16uint",
        "rgba32f": "rgba32float",
        "rgba32i": "rgba32sint",
        "rgba32ui": "rgba32uint",
    }
    SAMPLER_TYPE_NAMES = {
        "sampler",
        "samplerstate",
    }
    COMPARISON_SAMPLER_TYPE_NAMES = {
        "comparison_sampler",
        "samplercomparisonstate",
    }
    TEXTURE_FUNCTION_NAMES = {
        "texture",
        "texturebias",
        "texturecompare",
        "texturecomparegrad",
        "texturecompareoffset",
        "texturecomparelod",
        "texturecomparelodoffset",
        "texturecompareproj",
        "texturecompareprojgrad",
        "texturecompareprojoffset",
        "texturegrad",
        "texturegradoffset",
        "texturegather",
        "texturegathercompare",
        "texturegathercompareoffset",
        "texturegatheroffset",
        "texturegatheroffsets",
        "texturelod",
        "texturelodoffset",
        "textureoffset",
        "textureproj",
        "textureprojgrad",
        "textureprojgradoffset",
        "textureprojlod",
        "textureprojlodoffset",
        "textureprojoffset",
        "texturequerylevels",
        "texturequerylod",
        "texturesize",
        "texelfetch",
        "texelfetchoffset",
        "imagesize",
    }
    BARRIER_FUNCTION_NAMES = {
        "barrier",
        "groupmemorybarrierwithgroupsync",
        "workgroupbarrier",
        "workgroupexecutionbarrier",
        "groupmemorybarrier",
        "memorybarriershared",
        "devicememorybarrier",
        "memorybarrierbuffer",
        "memorybarrierimage",
        "allmemorybarrier",
        "memorybarrier",
    }
    # WGSL has no memory-only barrier: workgroupBarrier, storageBarrier and
    # textureBarrier each also synchronize execution across the workgroup and,
    # like Metal's threadgroup_barrier, must be reached in uniform control flow.
    # A CrossGL memory barrier therefore lowers to the WGSL barriers for every
    # address space it orders; the extra execution synchronization is stronger
    # than the source asks for but never weaker. groupMemoryBarrier orders
    # buffer, image and shared memory (GLSL 4.60 section 8.16), and
    # deviceMemoryBarrier orders buffers and textures (HLSL DeviceMemoryBarrier).
    WGSL_MEMORY_BARRIER_LOWERING = {
        "groupmemorybarrier": ("storageBarrier", "textureBarrier", "workgroupBarrier"),
        "memorybarriershared": ("workgroupBarrier",),
        "devicememorybarrier": ("storageBarrier", "textureBarrier"),
        "memorybarrierbuffer": ("storageBarrier",),
        "memorybarrierimage": ("textureBarrier",),
        "allmemorybarrier": ("storageBarrier", "textureBarrier", "workgroupBarrier"),
        "memorybarrier": ("storageBarrier", "textureBarrier", "workgroupBarrier"),
    }
    WGSL_ACCESS_MODE_IDENTIFIERS = frozenset({"read", "write", "read_write"})
    _used_access_mode_identifiers = frozenset()
    WGSL_RESERVED_IDENTIFIERS = {
        "NULL",
        "Self",
        "abstract",
        "active",
        "alias",
        "alignas",
        "alignof",
        "as",
        "asm",
        "asm_fragment",
        "async",
        "attribute",
        "auto",
        "await",
        "become",
        "break",
        "case",
        "cast",
        "catch",
        "class",
        "co_await",
        "co_return",
        "co_yield",
        "coherent",
        "column_major",
        "common",
        "compile",
        "compile_fragment",
        "concept",
        "const",
        "const_assert",
        "const_cast",
        "consteval",
        "constexpr",
        "constinit",
        "continue",
        "continuing",
        "crate",
        "debugger",
        "decltype",
        "default",
        "delete",
        "demote",
        "demote_to_helper",
        "diagnostic",
        "discard",
        "do",
        "dynamic_cast",
        "else",
        "enable",
        "enum",
        "explicit",
        "export",
        "extends",
        "extern",
        "external",
        "fallthrough",
        "false",
        "filter",
        "final",
        "finally",
        "fn",
        "for",
        "friend",
        "from",
        "fxgroup",
        "get",
        "goto",
        "groupshared",
        "highp",
        "if",
        "impl",
        "implements",
        "import",
        "inline",
        "instanceof",
        "interface",
        "layout",
        "let",
        "loop",
        "lowp",
        "macro",
        "macro_rules",
        "match",
        "mediump",
        "meta",
        "mod",
        "module",
        "move",
        "mut",
        "mutable",
        "namespace",
        "new",
        "nil",
        "noexcept",
        "noinline",
        "nointerpolation",
        "non_coherent",
        "noncoherent",
        "noperspective",
        "null",
        "nullptr",
        "of",
        "operator",
        "override",
        "package",
        "packoffset",
        "partition",
        "pass",
        "patch",
        "pixelfragment",
        "precise",
        "precision",
        "premerge",
        "priv",
        "protected",
        "pub",
        "public",
        "readonly",
        "ref",
        "regardless",
        "register",
        "reinterpret_cast",
        "require",
        "requires",
        "resource",
        "restrict",
        "return",
        "self",
        "set",
        "shared",
        "sizeof",
        "smooth",
        "snorm",
        "static",
        "static_assert",
        "static_cast",
        "std",
        "struct",
        "subroutine",
        "super",
        "switch",
        "target",
        "template",
        "this",
        "thread_local",
        "throw",
        "trait",
        "true",
        "try",
        "type",
        "typedef",
        "typeid",
        "typename",
        "typeof",
        "union",
        "unless",
        "unorm",
        "unsafe",
        "unsized",
        "use",
        "using",
        "var",
        "varying",
        "virtual",
        "volatile",
        "wgsl",
        "where",
        "while",
        "with",
        "writeonly",
        "yield",
    }

    def __init__(self):
        self._current_stage_name = None
        self._helper_stage_names = {}
        self._current_workgroup_size = None
        self._workgroup_size_overrides = WorkgroupSizeOverrides(None)
        self._location_counters = {"in": 0, "out": 0, "generic": 0}
        self._next_implicit_binding_by_group = {}
        self._cbuffer_member_accesses = {}
        self._function_texture_parameters = {}
        self._function_pointer_parameters = {}
        self._function_return_types = {}
        self._function_return_types_by_signature = {}
        self._current_expression_expected_type = None
        self._current_function_return_type = None
        self._identifier_scopes = []
        self._identifier_alias_scopes = []
        self._pointer_identifier_scopes = []
        self._parameter_plan_stack = []
        self._value_type_scopes = []
        self._resource_alias_scopes = []
        self._module_identifier_names = {}
        self._function_identifier_names = {}
        self._function_signature_identifier_names = {}
        self._function_overloads_by_name = {}
        self._type_identifier_names = {}
        self._struct_member_identifier_names = {}
        self._structs_by_name = {}
        self._struct_member_types = {}
        self._generic_struct_definitions = {}
        self._generic_struct_specializations = {}
        self._generic_struct_specialized_names = {}
        self._cbuffer_member_types = {}
        self._struct_resource_paths = {}
        self._glsl_buffer_block_struct_names = set()
        self._uniform_buffer_struct_names = set()
        self._uniform_scalar_array_wrappers = {}
        self._module_variable_types = {}
        self._module_storage_access_modes = {}
        self._used_access_mode_identifiers = set()
        self._read_only_module_names = set()
        self._module_storage_texture_access_modes = {}
        self._inferred_storage_texture_access_modes = {}
        self._module_resource_bindings = {}
        self._sampled_texture_array_bindings = {}
        self._sampled_texture_array_dispatch_helpers = {}
        self._frexp_modf_helpers = {}
        self._matrix_inverse_helpers = {}
        self._matrix_resize_helpers = {}
        self._uniform_registers = None
        self._half_conversion_helpers = {}
        self._constant_values = {}
        self._specialization_plan = WGSLSpecializationPlan({}, set(), set())
        self._specialization_default_values = {}
        self._function_resource_member_parameters = {}
        self._reserved_bindings_by_group = {}
        self._allocated_bindings_by_group = {}
        self._reserved_sampler_bindings = {}
        self._explicit_binding_owners = {}
        self._explicit_binding_register_classes = {}
        self._hlsl_register_binding_allocations = {}
        self._hlsl_register_source_owners = {}
        self._stage_output_lowerings = {}
        self._stage_interface_member_locations = {}
        self._vertex_output_interpolation = {}
        self._stage_interface_struct_names = set()
        self._fragment_output_struct_names = set()
        self._stage_interface_structs_by_name = {}
        self._entry_parameter_locations = {}
        self._stage_interface_builtin_members = {}
        self._builtin_option_specializations = {}
        self._generic_function_specializer = None
        self._uses_primitive_index = False
        self._int_builtin_members = {}
        self._enum_type_names = set()
        self._enum_variant_constants = {}

    def generate(self, ast):
        return self.generate_program(ast)

    def generate_program(self, ast, target_stage=None):
        refuse_cooperative_matrices(ast, label="WGSL", target="wgsl")
        refuse_precise_evaluation(ast, label="WGSL", target="wgsl")
        ast = lower_union_storage(ast, "wgsl")
        target_stage = requested_stage_name(target_stage)
        # WGSL textures are separate objects: a separate CrossGL texture (texture2DShadow, utexture2D,
        # texture2DMS) declares the same texture as its combined spelling, with its explicit samplers kept.
        ast = canonicalize_separate_textures(ast)
        lower_relational_builtins(ast)
        lower_workgroup_size_inputs(ast)
        hoist_workgroup_locals(ast, self._function_stage_name)
        # Atomic built-ins address atomic<T> memory through pointers (wgsl_buffer_atomics). The entry points'
        # workgroup locals are module-scope variables by now, so their atomic slots are module slots.
        ast = self.plan_buffer_atomics(ast)
        lower_integer_builtin_inputs(
            ast, lambda name: self.BUILTIN_SEMANTICS.get(self.semantic_key(name))
        )
        self.validate_wgsl_stage_support(ast, target_stage)
        validate_recursion_free(ast, "wgsl")
        self.reset_wgsl_program_state(ast)

        lines = ["// Generated by CrossGL for WebGPU WGSL"]
        emitted_sections = []

        (
            structs,
            cbuffers,
            global_variable_nodes,
            stage_resource_parameters,
            all_function_nodes,
            generic_functions,
            concrete_helper_function_nodes,
            enums,
            constants,
        ) = self.collect_wgsl_program_tables(ast, target_stage)
        self.collect_wgsl_module_types_and_bindings(
            all_function_nodes,
            cbuffers,
            constants,
            global_variable_nodes,
            stage_resource_parameters,
            structs,
        )
        helper_functions, helper_section_index = self.render_wgsl_module_declarations(
            ast,
            cbuffers,
            concrete_helper_function_nodes,
            constants,
            emitted_sections,
            enums,
            generic_functions,
            global_variable_nodes,
            stage_resource_parameters,
            structs,
        )

        self.render_wgsl_stage_functions_and_helpers(
            ast,
            target_stage,
            emitted_sections,
            generic_functions,
            helper_functions,
            helper_section_index,
            lines,
        )
        return "\n".join(lines).rstrip() + "\n"

    def stage_output_lowerings(self, ast, target_stage, existing_structs):
        used_struct_names = {
            getattr(struct, "name", "")
            for struct in existing_structs
            if getattr(struct, "name", "")
        }
        lowerings = {}
        used_type_names = used_struct_names | self.module_declaration_names(ast)
        structs_by_name = {
            getattr(struct, "name", ""): struct
            for struct in existing_structs
            if getattr(struct, "name", "")
        }
        for stage_node in self._stage_nodes(ast, target_stage):
            output_variable = self.vertex_position_output_variable(stage_node)
            entry_point = getattr(stage_node, "entry_point", None)
            stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
            if entry_output_parameters(entry_point, stage_name):
                if output_variable is not None:
                    raise EntryOutputLoweringError(
                        f"WGSL target cannot combine the output parameters of "
                        f"entry point {entry_point.name} with the stage output "
                        f"variable {output_variable.name}"
                    )
                struct_node, lowered = lower_entry_output_parameters(
                    self, entry_point, stage_name, structs_by_name, used_type_names
                )
                lowerings[id(stage_node)] = {
                    "struct": struct_node,
                    "entry_point": lowered,
                }
                continue
            if output_variable is None or entry_point is None:
                continue
            if not self.is_void_type(getattr(entry_point, "return_type", None)):
                continue
            struct_name = self.unique_stage_output_struct_name(
                "VertexOutput", used_struct_names
            )
            used_struct_names.add(struct_name)
            field_name = "position"
            struct_node = StructNode(
                struct_name,
                [
                    StructMemberNode(
                        field_name,
                        output_variable.var_type,
                        attributes=[AttributeNode("gl_Position")],
                    )
                ],
            )
            lowerings[id(stage_node)] = {
                "source_name": output_variable.name,
                "field_name": field_name,
                "local_name": self.unique_stage_output_local_name(entry_point),
                "struct_name": struct_name,
                "struct": struct_node,
            }
        return lowerings

    def rewrite_stage_output_body(self, body, lowering):
        rewritten = []
        saw_return = False
        for statement in self.block_statements(body):
            rewritten_statement = self.rewrite_stage_output_statement(
                statement, lowering
            )
            if isinstance(rewritten_statement, ReturnNode):
                saw_return = True
            rewritten.append(rewritten_statement)
        if not saw_return:
            rewritten.append(ReturnNode(IdentifierNode(lowering["local_name"])))
        output_declaration = VariableNode(
            lowering["local_name"], NamedType(lowering["struct_name"])
        )
        return BlockNode([output_declaration, *rewritten])

    def rewrite_stage_output_statement(self, statement, lowering):
        if isinstance(statement, ExpressionStatementNode):
            expression = statement.expression
            if isinstance(expression, AssignmentNode):
                rewritten = copy.copy(statement)
                rewritten.expression = self.rewrite_stage_output_assignment(
                    expression, lowering
                )
                return rewritten
            return statement
        if isinstance(statement, AssignmentNode):
            return self.rewrite_stage_output_assignment(statement, lowering)
        if isinstance(statement, ReturnNode) and statement.value is None:
            return ReturnNode(IdentifierNode(lowering["local_name"]))
        if isinstance(statement, BlockNode):
            return BlockNode(
                [
                    self.rewrite_stage_output_statement(child, lowering)
                    for child in getattr(statement, "statements", []) or []
                ]
            )
        if isinstance(statement, IfNode):
            rewritten = copy.copy(statement)
            rewritten.then_branch = self.rewrite_stage_output_statement(
                statement.then_branch, lowering
            )
            if statement.else_branch is not None:
                rewritten.else_branch = self.rewrite_stage_output_statement(
                    statement.else_branch, lowering
                )
            rewritten.if_body = rewritten.then_branch
            rewritten.else_body = rewritten.else_branch
            return rewritten
        if isinstance(statement, (ForNode, WhileNode, DoWhileNode, LoopNode)):
            rewritten = copy.copy(statement)
            rewritten.body = self.rewrite_stage_output_statement(
                statement.body, lowering
            )
            return rewritten
        return statement

    def visit_ast_values(self, value, callback):
        visited = set()

        def visit(current):
            if current is None or isinstance(current, (str, bytes, int, float, bool)):
                return
            if isinstance(current, (list, tuple, set, frozenset)):
                for item in current:
                    visit(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    visit(item)
                return

            current_id = id(current)
            if current_id in visited:
                return
            visited.add(current_id)

            callback(current)
            if not hasattr(current, "__dict__"):
                return
            for child in vars(current).values():
                visit(child)

        visit(value)

    def generate_module_scope_value(self, value, target_type):
        # A brace list takes the declared type, as in a local declaration: a
        # short array list is zero-filled and a structure's, vector's or
        # matrix's list becomes its constructor.
        if isinstance(value, ArrayLiteralNode):
            return self.generate_expression_for_target(value, target_type)
        return self.generate_expression(value)

    def generate_expression_for_target(self, expr, target_type, splat_scalars=True):
        struct_value = wgsl_struct_initializer(self, expr, target_type)
        if struct_value is not None:
            return struct_value
        vector_value = wgsl_vector_initializer(self, expr, target_type)
        if vector_value is not None:
            return vector_value
        padded_array = self.generate_padded_array_literal(expr, target_type)
        if padded_array is not None:
            return padded_array
        matrix_array = self.generate_matrix_array_literal(expr, target_type)
        if matrix_array is not None:
            return matrix_array
        struct_array = wgsl_struct_array_initializer(self, expr, target_type)
        if struct_array is not None:
            return struct_array
        vector_array = wgsl_vector_array_initializer(self, expr, target_type)
        if vector_array is not None:
            return vector_array
        previous_expected_type = self._current_expression_expected_type
        self._current_expression_expected_type = target_type
        try:
            if isinstance(expr, FunctionCallNode):
                expr = self.specialize_generic_function_call(expr)
                rendered = self.generate_function_call(expr, expected_type=target_type)
            else:
                rendered = self.generate_expression(expr)
            source_type = self.expression_type(expr)
            if isinstance(expr, FunctionCallNode):
                resolved_function = self.resolve_function_overload(
                    self.expression_name(expr.function),
                    expr.arguments,
                    expected_type=target_type,
                )
                if resolved_function is not None:
                    source_type = getattr(resolved_function, "return_type", None)
        finally:
            self._current_expression_expected_type = previous_expected_type
        narrowed = self.vector_narrowing_expression(rendered, target_type, source_type)
        if narrowed is not None:
            return narrowed
        if splat_scalars:
            splat = self.vector_splat_expression(
                rendered, target_type, source_type, expr
            )
            if splat is not None:
                return splat
        target_scalar = self.scalar_type_name(target_type)
        source_scalar = self.scalar_type_name(source_type)
        if target_scalar == "i32" and source_scalar == "u32":
            return f"i32({rendered})"
        if target_scalar == "u32" and source_scalar == "i32":
            return f"u32({self.concrete_negative_integer_literal(expr, rendered)})"
        if target_scalar == "f32" and source_scalar in {"i32", "u32"}:
            return f"f32({rendered})"
        return rendered

    def generate_padded_array_literal(self, expr, target_type):
        """``array<T, N>(...)`` for an array literal shorter than its sized array target, else ``None``.

        A CrossGL aggregate such as ``float[3](0.0)`` or ``{1.0}`` leaves the elements it does not give
        zero, as GLSL output writes them. WGSL ``array(...)`` would infer a shorter array, which a
        declared ``array<T, N>`` rejects, so the type is spelled out and each missing element is the
        zero value ``T()`` (WGSL, "Zero Value Expressions").
        """
        if not isinstance(expr, ArrayLiteralNode) or not isinstance(
            target_type, ArrayType
        ):
            return None
        # The size may be a named constant, as in float a[COUNT].
        declared_size = self.resolve_integer_expression(target_type.size)
        if declared_size is not None and len(expr.elements) > declared_size:
            array_type = self.type_name_string(target_type)
            raise WGSLAggregateInitializerError(
                f"WGSL aggregate initializer for '{array_type}' expects at most "
                f"{declared_size} elements, got {len(expr.elements)}",
                expected_type=array_type,
                element_count=len(expr.elements),
                expected_element_count=declared_size,
            )
        size = self.literal_integer_value(target_type.size)
        if size is None or not 0 < len(expr.elements) < size:
            return None
        element_type = target_type.element_type
        element = self.type_name_string(element_type)
        values = [
            wgsl_array_element(self, value, element_type) for value in expr.elements
        ]
        values += [f"{element}()"] * (size - len(values))
        return f"array<{element}, {size}>({', '.join(values)})"

    def generate_expression(self, expr):
        if expr is None:
            return ""
        if is_comma_expression(expr):
            # Statement lowering hoists every comma it can reach (wgsl_value_side_effects).
            raise WGSLRefusal(
                "WGSL target cannot lower a comma expression in this position; WGSL has no "
                "comma operator"
            )
        if isinstance(expr, LiteralNode):
            return self.generate_literal(expr)
        if self._uniform_registers is not None:
            register_value = wgsl_uniform_register_load(self, expr)
            if register_value is not None:
                return register_value
        if isinstance(expr, IdentifierNode):
            enum_constant = self._enum_variant_constants.get(expr.name)
            if enum_constant is not None:
                return enum_constant
            if self.is_builtin_option_none_expression(expr):
                none_value = self.generate_builtin_option_none_value()
                if none_value is not None:
                    return none_value
            if expr.name in self.WORKGROUP_SIZE_IDENTIFIER_ALIASES:
                return self.generate_workgroup_size_literal()
            mapped_builtin = self.BUILTIN_IDENTIFIER_ALIASES.get(expr.name)
            if mapped_builtin:
                return mapped_builtin
            if self.is_reference_parameter_name(expr.name):
                return self.reference_parameter_expression(expr.name)
            if not self.is_local_identifier(expr.name):
                cbuffer_access = self._cbuffer_member_accesses.get(expr.name)
                if cbuffer_access:
                    bool_value = self.uniform_bool_value(
                        cbuffer_access, self._cbuffer_member_types.get(expr.name)
                    )
                    return bool_value or cbuffer_access
            return self.identifier_name(expr.name)
        if isinstance(expr, BinaryOpNode):
            option_none_comparison = self.generate_builtin_option_none_comparison(
                expr.left, expr.operator, expr.right
            )
            if option_none_comparison is not None:
                return option_none_comparison
            matrix_scalar = self.generate_matrix_scalar_binary(expr)
            if matrix_scalar is not None:
                return matrix_scalar
            if expr.operator in SHIFT_OPERATORS:
                left = self.generate_expression(expr.left)
                amount = wgsl_shift_amount(
                    self, expr.right, self.expression_type(expr.left)
                )
                return f"({left} {expr.operator} {amount})"
            if expr.operator == "^":
                boolean_xor = wgsl_boolean_xor(
                    self, expr.left, expr.right, wgsl_bitwise_operands
                )
                if boolean_xor is not None:
                    return boolean_xor
            if expr.operator in BITWISE_OPERATORS:
                left, right = wgsl_bitwise_operands(self, expr.left, expr.right)
                return f"({left} {expr.operator} {right})"
            if expr.operator in MIXED_INTEGER_OPERATORS:
                left, right = wgsl_mixed_integer_operands(self, expr.left, expr.right)
                return f"({left} {expr.operator} {right})"
            return (
                f"({self.generate_expression(expr.left)} {expr.operator} "
                f"{self.generate_expression(expr.right)})"
            )
        if isinstance(expr, UnaryOpNode):
            if is_increment(expr):
                self.reject_unordered_side_effect(expr)
            operand = self.generate_expression(expr.operand)
            if expr.is_postfix:
                return f"{operand}{expr.operator}"
            if expr.operator == "+":
                # WGSL has no unary plus. On a number it is the identity, and the
                # operand text already stands on its own as an expression.
                if self.boolean_expression_width(expr.operand) is not None:
                    raise WGSLRefusal(
                        "WGSL has no unary plus, and unary plus on a boolean "
                        "value has no numeric meaning to preserve"
                    )
                return operand
            if expr.operator == "!":
                operand_type = self.expression_type(expr.operand)
                if operand_type is not None and not isinstance(operand_type, str):
                    operand_type = self.type_name_string(operand_type)
                converted = wgsl_logical_not_operand(operand, operand_type)
                if converted != operand:
                    # A bool conversion is a constructor call, a primary.
                    return f"!{converted}"
            if expr.operator == "-":
                negation = wgsl_unsigned_negation(self, expr.operand, operand)
                if negation is not None:
                    return negation
            if expr.operator == "-" and self.matrix_shape(
                self.expression_type(expr.operand)
            ):
                # WGSL has no unary minus on matrices; multiplying by -1 is exact.
                return f"({operand} * -1.0)"
            return f"{expr.operator}{prefix_unary_operand(expr.operator, operand)}"
        if isinstance(expr, TernaryOpNode):
            self.require_select_operands(expr)
            self.require_unconditional_branches(expr)
            return self.generate_ternary_expression(expr)
        if isinstance(expr, FunctionCallNode):
            return self.generate_function_call(expr)
        if isinstance(expr, ConstructorNode):
            return self.generate_constructor(expr)
        if isinstance(expr, MemberAccessNode):
            resource_binding = self.resource_member_binding_for_access(expr)
            if resource_binding is not None:
                return resource_binding["binding_name"]
            stage_matrix = self.generate_stage_interface_matrix_access(expr)
            if stage_matrix is not None:
                return stage_matrix
            member_name = self.member_access_identifier_name(expr)
            uniform_bool_type = self.uniform_bool_member_type(expr)
            if uniform_bool_type is not None:
                access = f"{self.generate_expression(expr.object_expr)}.{member_name}"
                return self.uniform_bool_value(access, uniform_bool_type)
            if is_int_builtin_member(self, expr):
                access = f"{self.generate_expression(expr.object_expr)}.{member_name}"
                return f"i32({access})"
            if isinstance(
                expr.object_expr, IdentifierNode
            ) and self.is_pointer_identifier(expr.object_expr.name):
                pointer_name = self.identifier_name(expr.object_expr.name)
                return f"(*{pointer_name}).{member_name}"
            return f"{self.generate_expression(expr.object_expr)}." f"{member_name}"
        if isinstance(expr, SwizzleNode):
            return f"{self.generate_expression(expr.vector_expr)}." f"{expr.components}"
        if isinstance(expr, ArrayAccessNode):
            array_binding = self.sampled_texture_array_element_binding(expr)
            if array_binding is not None:
                return array_binding["texture_name"]
            stage_array = self.generate_stage_interface_array_access(expr)
            if stage_array is not None:
                return stage_array
            if isinstance(
                expr.array_expr, IdentifierNode
            ) and self.is_pointer_identifier(expr.array_expr.name):
                pointer_name = self.identifier_name(expr.array_expr.name)
                return (
                    f"(*{pointer_name})"
                    f"[{self.generate_expression(expr.index_expr)}]"
                )
            access = (
                f"{self.generate_expression(expr.array_expr)}"
                f"[{self.generate_expression(expr.index_expr)}]"
            )
            if self.is_uniform_scalar_array_access(expr.array_expr):
                return f"{access}.value"
            return access
        if isinstance(expr, ArrayLiteralNode):
            return (
                "array("
                + ", ".join(
                    self.generate_expression(element) for element in expr.elements
                )
                + ")"
            )
        if isinstance(expr, CastNode):
            narrowed = self.generate_vector_narrowing_conversion(
                expr.target_type, expr.expression
            )
            if narrowed is not None:
                return narrowed
            resized = wgsl_matrix_resize(self, expr.target_type, expr.expression)
            if resized is not None:
                return resized
            return (
                f"{self.type_name_string(expr.target_type)}"
                f"({self.generate_expression(expr.expression)})"
            )
        if isinstance(expr, RangeNode):
            raise WGSLRefusal("WGSL target does not support range expressions")
        if isinstance(expr, WaveOpNode):
            return wgsl_wave_operation(self, expr)
        if isinstance(expr, AssignmentNode):
            self.reject_unordered_side_effect(expr)
        raise WGSLRefusal(
            f"WGSL target does not support expression {type(expr).__name__}"
        )

    def generate_function_call(self, node, expected_type=None):
        node = self.specialize_generic_function_call(node)
        function_name = self.expression_name(node.function)
        if isinstance(node.function, IdentifierNode):
            declared = self.resolve_function_overload(
                function_name, node.arguments, expected_type=expected_type
            )
            if declared is not None:
                # A call resolves to a function the shader declares before any built-in or
                # built-in alias of the same name, such as HLSL lerp for WGSL mix.
                return self.generate_declared_function_call(node, declared)
        array_value = wgsl_array_constructor(self, node)
        if array_value is not None:
            return array_value
        normalized_name = self.semantic_key(function_name)
        option_call = self.generate_builtin_option_call(node, function_name)
        if option_call is not None:
            return option_call
        if normalized_name in self.STRUCTURED_BUFFER_FREE_HELPERS:
            return self.generate_structured_buffer_free_helper_call(
                node, normalized_name
            )
        if normalized_name == "buffer_dimensions" and len(node.arguments) == 1:
            # The element count of a runtime-sized storage array (WGSL section 17.11, arrayLength).
            return f"arrayLength(&{self.generate_expression(node.arguments[0])})"
        if normalized_name == "workgroupuniformload" and len(node.arguments) == 1:
            # WGSL workgroupUniformLoad takes a pointer to workgroup storage (naga: "passed
            # type must be a workgroup pointer"). Emit &arg unless the arg is already a
            # pointer expression.
            return self.generate_workgroup_uniform_load_call(node)
        if normalized_name in self.UNSUPPORTED_STORAGE_BUFFER_FREE_HELPERS:
            raise WGSLRefusal(
                "WGSL target does not support storage buffer helper "
                f"{function_name} yet"
            )
        array_length = self.generate_array_length_call(node)
        if array_length is not None:
            return array_length
        if isinstance(node.function, MemberAccessNode):
            member_name = self.semantic_key(node.function.member)
            if member_name in self.STRUCTURED_BUFFER_MEMBER_HELPERS:
                return self.generate_structured_buffer_member_helper_call(
                    node, member_name
                )
            if member_name in self.UNSUPPORTED_STRUCTURED_BUFFER_MEMBER_HELPERS:
                receiver_type = self.expression_type(node.function.object_expr)
                if self.storage_buffer_like_type_name(receiver_type) is not None:
                    raise WGSLRefusal(
                        "WGSL target does not support storage buffer member helper "
                        f"{node.function.member} yet"
                    )
        if normalized_name in self.TEXTURE_FUNCTION_NAMES:
            return self.generate_texture_function_call(node, function_name)
        if normalized_name == "imageload":
            return self.generate_image_load_call(node, function_name)
        if normalized_name == "imagestore":
            return self.generate_image_store_call(node, function_name)
        if is_image_atomic_name(normalized_name) and node.arguments:
            raise WGSLImageAtomicUnsupportedError(
                function_name, self.generate_expression(node.arguments[0])
            )
        if function_name in ATOMIC_NAMES:
            return self.generate_buffer_atomic_call(node, function_name)
        if normalized_name in self.BARRIER_FUNCTION_NAMES:
            return self.generate_barrier_call(node, function_name)
        derivative_name = self.DERIVATIVE_FUNCTION_NAME_MAP.get(normalized_name)
        if derivative_name is not None and not self.is_user_function_name(
            function_name
        ):
            return self.generate_derivative_call(node, derivative_name, function_name)
        if (
            normalized_name in {"min", "max"}
            and len(node.arguments) == 2
            and not self.is_user_function_name(function_name)
        ):
            args = wgsl_mixed_integer_call_arguments(self, node.arguments)
            if not mixed_integer_call_is_unsigned(self, node.arguments):
                # An unsigned mixed call already has its scalars written as vecN<u32>.
                args = self.convert_integer_call_arguments(
                    normalized_name, node.arguments, args
                )
                args = self.replicate_scalar_call_arguments(
                    node.arguments,
                    args,
                    self.float_call_component(normalized_name, node.arguments),
                )
            return f"{normalized_name}({args[0]}, {args[1]})"
        if normalized_name in {"step", "smoothstep"} and not (
            self.is_user_function_name(function_name)
        ):
            args = [self.generate_expression(arg) for arg in node.arguments]
            args = self.convert_integer_call_arguments(
                normalized_name, node.arguments, args
            )
            args = self.replicate_scalar_call_arguments(
                node.arguments,
                args,
                self.float_call_component(normalized_name, node.arguments),
            )
            return f"{normalized_name}({', '.join(args)})"
        if normalized_name == "clamp":
            return self.generate_clamp_call(node)
        packed_8bit = self.PACKED_8BIT_FUNCTIONS.get(normalized_name)
        if packed_8bit is not None and not self.is_user_function_name(function_name):
            return self.generate_packed_8bit_call(node, function_name, packed_8bit)
        bit_function = self.BIT_FUNCTION_NAME_MAP.get(function_name)
        if bit_function is not None and not self.is_user_function_name(function_name):
            return self.generate_bit_function_call(node, function_name, bit_function)
        if (
            function_name == "faceforward"
            and node.arguments
            and not self.is_user_function_name(function_name)
            and self.scalar_type_name(self.expression_type(node.arguments[0]))
        ):
            # GLSL's faceforward also takes floats; WGSL's faceForward takes vectors only.
            raise WGSLRefusal(
                "WGSL target supports faceforward() on vectors only, as WGSL's "
                "faceForward; write a scalar one as i * nref < 0.0 ? n : -n"
            )
        if (
            function_name == "mix"
            and len(node.arguments) == 3
            and not self.is_user_function_name(function_name)
        ):
            boolean_mix = self.generate_boolean_mix_call(node)
            if boolean_mix is not None:
                return boolean_mix
        bit_cast = self.BIT_CAST_FUNCTIONS.get(function_name)
        if bit_cast is not None and not self.is_user_function_name(function_name):
            return self.generate_bit_cast_call(node, function_name, bit_cast)
        if function_name == "f32tof16" and not self.is_user_function_name(
            function_name
        ):
            return self.generate_f32tof16_call(node)
        if function_name == "f16tof32" and not self.is_user_function_name(
            function_name
        ):
            return self.generate_f16tof32_call(node)
        if function_name == "mod":
            return self.generate_mod_call(node)
        if function_name == "fmod" and not self.is_user_function_name(function_name):
            return self.generate_fmod_call(node)
        if function_name in self.COLUMNWISE_MATRIX_BUILTINS and not (
            self.is_user_function_name(function_name)
        ):
            return self.generate_columnwise_matrix_builtin_call(node, function_name)
        packed_dot = self.PACKED_DOT4ADD_FUNCTIONS.get(normalized_name)
        if packed_dot is not None and not self.is_user_function_name(function_name):
            return self.generate_packed_dot4add_call(node, function_name, packed_dot)
        if is_two_argument_frexp_or_modf(self, node):
            return wgsl_frexp_modf_call(self, node)
        if is_matrix_inverse_call(self, node):
            return wgsl_matrix_inverse_call(self, node)
        if is_float_classification_call(self, node):
            return wgsl_float_classification_call(self, node)

        if self.is_type_constructor_name(function_name):
            if len(node.arguments) == 1:
                narrowed = self.generate_vector_narrowing_conversion(
                    function_name, node.arguments[0]
                )
                if narrowed is not None:
                    return narrowed
                diagonal = self.generate_scalar_matrix_constructor(
                    function_name, node.arguments[0]
                )
                if diagonal is not None:
                    return diagonal
                resized = wgsl_matrix_resize(self, function_name, node.arguments[0])
                if resized is not None:
                    return resized
            args = self.generate_type_constructor_call_arguments(
                function_name, node.arguments
            )
            return f"{self.type_name_string(function_name)}({args})"
        generic_constructor = self.generic_constructor_type_name(function_name)
        if generic_constructor is not None:
            args = self.generate_call_arguments(function_name, node.arguments)
            return f"{generic_constructor}({args})"
        if function_name in self._structs_by_name and not self.is_user_function_name(
            function_name
        ):
            struct = self._structs_by_name[function_name]
            member_types = [
                member.member_type
                for member in getattr(struct, "members", None) or []
                if hasattr(member, "member_type")
            ]
            # Only a call with one argument per member: zip would drop any argument past the last member.
            if member_types and len(node.arguments) == len(member_types):
                args = ", ".join(
                    self.generate_expression_for_target(arg, member_type)
                    for arg, member_type in zip(node.arguments, member_types)
                )
                return f"{self.type_name_string(function_name)}({args})"
        resolved_function = None
        if isinstance(node.function, IdentifierNode):
            resolved_function = self.resolve_function_overload(
                function_name, node.arguments, expected_type=expected_type
            )
            dispatch_call = self.generate_sampled_texture_array_dispatch_call(
                node, resolved_function
            )
            if dispatch_call is not None:
                return dispatch_call
        if (
            resolved_function is None
            and (
                function_name in self.FLOAT_ARGUMENT_BUILTINS
                or function_name in self.FLOAT_ONLY_BUILTINS
            )
            and isinstance(node.function, IdentifierNode)
            and not self.is_user_function_name(function_name)
        ):
            args = [self.generate_expression(arg) for arg in node.arguments]
            args = self.convert_integer_call_arguments(
                function_name, node.arguments, args
            )
            args = ", ".join(
                self.replicate_scalar_call_arguments(
                    node.arguments,
                    args,
                    self.float_call_component(function_name, node.arguments),
                )
            )
        else:
            args = self.generate_call_arguments(
                function_name, node.arguments, function=resolved_function
            )
        mapped_name = self.FUNCTION_NAME_MAP.get(function_name)
        if (
            function_name == "atan"
            and len(node.arguments) == 2
            and isinstance(node.function, IdentifierNode)
            and not self.is_user_function_name(function_name)
        ):
            # GLSL atan(y, x) is the two-argument arctangent; WGSL's atan takes one
            # argument and spells the two-argument form atan2(y, x).
            return f"atan2({args})"
        if (
            function_name == "log10"
            and len(node.arguments) == 1
            and isinstance(node.function, IdentifierNode)
            and not self.is_user_function_name(function_name)
        ):
            # WGSL declares log2 but no log10 (WGSL section 17.5); the
            # identity log10(x) = log2(x) * log10(2) is the polyfill the
            # GLSL target writes for the same call.
            return f"(log2({args}) * 0.3010299956639812)"
        if mapped_name is None and isinstance(node.function, IdentifierNode):
            if resolved_function is not None:
                mapped_name = self.function_declaration_identifier_name(
                    resolved_function
                )
            else:
                mapped_name = self.function_identifier_name(function_name)
                self.refuse_foreign_wgsl_call(function_name, mapped_name)
        elif isinstance(node.function, MemberAccessNode):
            mapped_name = self.generate_expression(node.function)
        elif mapped_name is None:
            # A callee that is neither a name nor a member access is written as the expression names it.
            mapped_name = function_name
            self.refuse_foreign_wgsl_call(function_name, mapped_name)
        return f"{mapped_name}({args})"

    def generate_texture_function_call(self, node, function_name):
        normalized_name = self.semantic_key(function_name)
        args = list(node.arguments)
        if normalized_name in FILTERED_TEXTURE_LOOKUPS and args:
            self.refuse_integer_texture_sample(args[0], function_name)
        if normalized_name in OFFSET_TEXTURE_LOOKUPS and args:
            self.refuse_cube_array_texture_offset(args[0], function_name)
        if normalized_name == "texture":
            return self.generate_texture_sample_call(function_name, args)
        if normalized_name == "texturelod":
            return self.generate_texture_sample_level_call(function_name, args)
        if normalized_name == "texturelodoffset":
            return self.generate_texture_sample_level_offset_call(function_name, args)
        if normalized_name == "texturegrad":
            return self.generate_texture_sample_grad_call(function_name, args)
        if normalized_name == "texturegradoffset":
            return self.generate_texture_sample_grad_offset_call(function_name, args)
        if normalized_name == "textureoffset":
            return self.generate_texture_sample_offset_call(function_name, args)
        if normalized_name == "texturecompare":
            return self.generate_texture_sample_compare_call(function_name, args)
        if normalized_name == "texturecompareoffset":
            return self.generate_texture_sample_compare_offset_call(function_name, args)
        if normalized_name == "texturecomparelod":
            return self.generate_texture_sample_compare_level_call(function_name, args)
        if normalized_name == "texturecomparelodoffset":
            return self.generate_texture_sample_compare_level_offset_call(
                function_name, args
            )
        if normalized_name == "texturesize":
            return self.generate_texture_dimensions_call(args)
        if normalized_name == "imagesize":
            return self.generate_image_dimensions_call(args)
        if normalized_name == "texelfetch":
            return self.generate_texel_fetch_call(function_name, args)
        if normalized_name == "texelfetchoffset":
            return self.generate_texel_fetch_offset_call(function_name, args)
        raise WGSLRefusal(
            "WGSL target does not support CrossGL texture function "
            f"{function_name} yet"
        )

    def refuse_cube_array_texture_offset(self, texture_expr, function_name):
        texture_type = self.sampled_texture_type(self.expression_type(texture_expr))
        if is_cube_array_texture_type(texture_type):
            raise WGSLCubeTextureOffsetUnsupportedError(
                function_name, self.generate_expression(texture_expr)
            )

    def collect_builtin_option_specializations(self, root):
        specializations = {}
        visited = set()

        def visit(value):
            if value is None or isinstance(value, (int, float, bool)):
                return
            if isinstance(value, str):
                payload_type = self.builtin_option_payload_type(value)
                if payload_type is not None:
                    specializations[value] = {
                        "type_name": value,
                        "struct_name": generic_enum_specialization_name(value),
                        "payload_type": payload_type,
                    }
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            payload_type = self.builtin_option_payload_type(value)
            if payload_type is not None:
                type_text = self.raw_type_name_string(value)
                specializations[type_text] = {
                    "type_name": type_text,
                    "struct_name": generic_enum_specialization_name(type_text),
                    "payload_type": payload_type,
                }

            if not hasattr(value, "__dict__"):
                return
            for child in vars(value).values():
                visit(child)

        visit(root)
        return dict(sorted(specializations.items()))

    def expression_type(self, expr):
        if isinstance(expr, WaveOpNode):
            return wgsl_wave_result_type(self, expr)
        if isinstance(expr, IdentifierNode):
            if expr.name in self._enum_variant_constants:
                return "i32"
            if expr.name in self.WORKGROUP_SIZE_IDENTIFIER_ALIASES:
                return "vec3<u32>"
            mapped_builtin = self.BUILTIN_IDENTIFIER_ALIASES.get(expr.name)
            if mapped_builtin:
                return self.INPUT_BUILTIN_TYPE_MAP.get(mapped_builtin)
            value_type = self.value_type(expr.name)
            if value_type is not None:
                return value_type
            module_type = self._module_variable_types.get(expr.name)
            if module_type is not None:
                return module_type
            if not self.is_local_identifier(expr.name):
                return self._cbuffer_member_types.get(expr.name)
            return None
        if isinstance(expr, MemberAccessNode):
            object_type = self.array_element_type(
                self.expression_type(expr.object_expr)
            )
            component_type = self.vector_component_type(object_type)
            if component_type is not None and self.is_vector_member(expr.member):
                if len(expr.member) == 1:
                    return component_type
                return f"vec{len(expr.member)}<{component_type}>"
            struct_name = self.struct_type_name(object_type)
            if not struct_name:
                return None
            return self._struct_member_types.get((struct_name, expr.member))
        if isinstance(expr, BinaryOpNode):
            return self.binary_expression_type(expr)
        if isinstance(expr, TernaryOpNode):
            # WGSL lowers a ternary to select, whose branches share one type;
            # an int branch is converted to the other branch's type.
            return self.ternary_result_type(
                self.expression_type(expr.true_expr),
                self.expression_type(expr.false_expr),
            )
        if (
            isinstance(expr, UnaryOpNode)
            and expr.operator in {"-", "+", "~"}
            and not expr.is_postfix
        ):
            operand_type = self.expression_type(expr.operand)
            if (
                self.matrix_shape(operand_type) is not None
                or self.vector_shape(operand_type) is not None
                or self.scalar_type_name(operand_type) is not None
            ):
                return operand_type
        if (
            isinstance(expr, UnaryOpNode)
            and expr.operator == "!"
            and not expr.is_postfix
        ):
            # `!` keeps the operand's shape: a bvec stays a bvec, and only a
            # scalar bool becomes `bool`.
            operand_type = self.expression_type(expr.operand)
            shape = self.vector_shape(operand_type)
            if shape is not None and shape[0] == "bool":
                return f"vec{shape[1]}<bool>"
            if self.scalar_type_name(operand_type) == "bool":
                return "bool"
        if isinstance(expr, FunctionCallNode):
            expr = self.specialize_generic_function_call(expr)
            function_name = self.expression_name(expr.function)
            option_call_type = self.builtin_option_call_type(expr, function_name)
            if option_call_type is not None:
                return option_call_type
            atomic_call_type = self.buffer_atomic_call_type(expr)
            if atomic_call_type is not None:
                return atomic_call_type
            if self.is_type_constructor_name(function_name):
                return self.type_name_string(function_name)
            if function_name in self._structs_by_name and not (
                self.is_user_function_name(function_name)
            ):
                # A structure constructor such as Pair(a, b) has the structure's type.
                return NamedType(function_name)
            array_type = wgsl_array_constructor_type(self, expr)
            if array_type is not None:
                return array_type
            if function_name == "buffer_dimensions" and len(expr.arguments) == 1:
                return "uint"
            loaded = self.structured_buffer_load_access(expr, function_name)
            if loaded is not None:
                return self.expression_type(loaded)
            packed_8bit = self.PACKED_8BIT_FUNCTIONS.get(
                self.semantic_key(function_name)
            )
            if packed_8bit is not None and not self.is_user_function_name(
                function_name
            ):
                return packed_8bit[2]
            if (
                function_name in {"bitCount", "findLSB", "findMSB"}
                and len(expr.arguments) == 1
                and not self.is_user_function_name(function_name)
            ):
                vector = self.vector_shape(self.expression_type(expr.arguments[0]))
                return "int" if vector is None else f"vec{vector[1]}<i32>"
            if (
                function_name == "f32tof16"
                and len(expr.arguments) == 1
                and not self.is_user_function_name(function_name)
            ):
                vector = self.vector_shape(self.expression_type(expr.arguments[0]))
                return "u32" if vector is None else f"vec{vector[1]}<u32>"
            if (
                function_name == "f16tof32"
                and len(expr.arguments) == 1
                and not self.is_user_function_name(function_name)
            ):
                vector = self.vector_shape(self.expression_type(expr.arguments[0]))
                return "f32" if vector is None else f"vec{vector[1]}<f32>"
            if (
                isinstance(expr.function, MemberAccessNode)
                and expr.function.member == "length"
                and not expr.arguments
            ):
                obj_type = self.expression_type(expr.function.object_expr)
                if isinstance(obj_type, (ArrayType, PointerType)):
                    return "int"
                if (
                    isinstance(obj_type, NamedType)
                    and str(obj_type.name).lower() in self.STRUCTURED_BUFFER_TYPE_NAMES
                ):
                    return "int"
            if function_name in self.COLUMNWISE_MATRIX_BUILTINS and not (
                self.is_user_function_name(function_name)
            ):
                return self.columnwise_matrix_builtin_type(
                    function_name, expr.arguments
                )
            if (
                function_name == "transpose"
                and len(expr.arguments) == 1
                and not self.is_user_function_name(function_name)
            ):
                operand = self.matrix_shape(self.expression_type(expr.arguments[0]))
                if operand is not None:
                    element, columns, rows = operand
                    return f"mat{rows}x{columns}<{element}>"
            if is_matrix_inverse_call(self, expr):
                # inverse(m) has m's type (GLSL 4.60 section 8.6).
                return self.expression_type(expr.arguments[0])
            if (
                function_name in {"min", "max", "clamp"}
                and not self.is_user_function_name(function_name)
                and mixed_integer_call_is_unsigned(self, expr.arguments)
                and self.float_call_component(function_name, expr.arguments) is None
            ):
                # The signed arguments are written as u32 (wgsl_mixed_integer_call_arguments), so the call is u32.
                # A float argument picks the float overload instead (generate_clamp_call).
                return mixed_integer_call_type(self, expr.arguments)
            math_call_type = self.builtin_math_call_type(function_name, expr.arguments)
            if math_call_type is not None:
                return math_call_type
            resolved_function = self.resolve_function_overload(
                function_name, expr.arguments
            )
            if resolved_function is not None:
                return self._function_return_types_by_signature.get(
                    self.function_signature_key(resolved_function),
                    getattr(resolved_function, "return_type", None),
                )
            return self._function_return_types.get(function_name)
        if isinstance(expr, ConstructorNode):
            return expr.constructor_type
        if isinstance(expr, CastNode):
            return expr.target_type
        if isinstance(expr, LiteralNode):
            return getattr(expr, "literal_type", None)
        if isinstance(expr, SwizzleNode):
            vector_type = self.expression_type(expr.vector_expr)
            component_type = self.vector_component_type(vector_type)
            if component_type is None:
                return None
            if len(expr.components) == 1:
                return component_type
            return f"vec{len(expr.components)}<{component_type}>"
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_type(expr.array_expr)
            if isinstance(array_type, ArrayType):
                return array_type.element_type
            storage_element = self.structured_buffer_element_type(array_type)
            if storage_element is None and isinstance(array_type, PointerType):
                storage_element = array_type.pointee_type
            if storage_element is not None:
                # Narrow storage holds one 32-bit word per element (:mod:`.wgsl_narrow_storage`).
                return narrow_storage_word_type(storage_element) or storage_element
            # Indexing a matrix gives a column and indexing a vector a component.
            matrix = self.matrix_shape(array_type)
            if matrix is not None:
                element, _columns, rows = matrix
                return f"vec{rows}<{element}>"
            vector = self.vector_shape(array_type)
            if vector is not None:
                return vector[0]
            return self.array_element_type(array_type)
        return None

    def binary_expression_type(self, expr):
        if expr.operator in BITWISE_OPERATORS:
            # WGSL's bitwise operators need both operands of one type; a scalar is
            # replicated at rendering, so the result is the type the sides share.
            return self._common_value_type(
                *unsigned_bitwise_operand_types(self, expr.left, expr.right)
            )
        if expr.operator not in {"+", "-", "*", "/", "%"}:
            return None
        left_type = self.expression_type(expr.left)
        right_type = self.expression_type(expr.right)
        left_vector = self.vector_shape(left_type)
        right_vector = self.vector_shape(right_type)
        left_matrix = self.matrix_shape(left_type)
        right_matrix = self.matrix_shape(right_type)
        left_scalar = self.scalar_type_name(left_type)
        right_scalar = self.scalar_type_name(right_type)
        if expr.operator == "*":
            if left_matrix is not None and right_vector is not None:
                matrix_element, columns, rows = left_matrix
                vector_element, vector_size = right_vector
                if matrix_element == vector_element and columns == vector_size:
                    return f"vec{rows}<{matrix_element}>"
            if left_vector is not None and right_matrix is not None:
                vector_element, vector_size = left_vector
                matrix_element, columns, rows = right_matrix
                if matrix_element == vector_element and rows == vector_size:
                    return f"vec{columns}<{matrix_element}>"
        if expr.operator == "*" and left_matrix is not None:
            if right_matrix is not None and left_matrix[0] == right_matrix[0]:
                if left_matrix[1] == right_matrix[2]:
                    return f"mat{right_matrix[1]}x{left_matrix[2]}<{left_matrix[0]}>"
        if expr.operator in {"+", "-"} and left_matrix is not None:
            if left_matrix == right_matrix:
                return left_type
        if expr.operator != "%":
            if left_matrix is not None and right_scalar is not None:
                return left_type
            if right_matrix is not None and left_scalar is not None:
                return right_type
        if left_vector is not None and right_vector is not None:
            if left_vector == right_vector:
                return left_type
            # GLSL converts an integer vector operand to the other operand's float vector type (4.1.10).
            if left_vector[1] == right_vector[1]:
                elements = {left_vector[0], right_vector[0]}
                if elements in ({"f32", "i32"}, {"f32", "u32"}):
                    return f"vec{left_vector[1]}<f32>"
            return None
        if left_vector is not None and right_scalar is not None:
            return left_type
        if right_vector is not None and left_scalar is not None:
            return right_type
        if left_scalar is not None and left_scalar == right_scalar:
            return left_scalar
        if {left_scalar, right_scalar} == {"i32", "u32"}:
            return "u32"
        if {left_scalar, right_scalar} in ({"f32", "i32"}, {"f32", "u32"}):
            return "f32"
        if left_type == right_type:
            return left_type
        return None

    def _stage_nodes(self, ast, target_stage):
        nodes = []
        for stage_type, stage_node in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if target_stage is not None and stage_name != target_stage:
                continue
            nodes.append(stage_node)
        if nodes:
            return nodes

        for func in getattr(ast, "functions", []) or []:
            stage_name = self._function_stage_name(func)
            if not stage_name:
                continue
            if target_stage is not None and stage_name != target_stage:
                continue
            nodes.append(
                _FunctionStageNode(
                    stage_name,
                    func,
                    execution_config=self._function_execution_config(func),
                )
            )
        return nodes


class _FunctionStageNode:
    def __init__(self, stage, entry_point, execution_config=None):
        self.stage = stage
        self.entry_point = entry_point
        self.execution_config = execution_config or {}
        self.layout_qualifiers = []
        self.local_structs = []
        self.local_functions = []
