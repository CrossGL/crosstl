"""Parser for Vulkan SPIR-V source AST construction."""

import re  # noqa: F401
import shlex  # noqa: F401
import struct  # noqa: F401

from crosstl.translator.refusals import SpirvSyntaxError

from ..common_ast import InitializerListNode  # noqa: F401
from .access_chains import chain_base_pointee  # noqa: F401
from .access_chains import walk_indices  # noqa: F401
from .assembly_validation import check_defined_ids  # noqa: F401
from .assembly_validation import check_type_declaration_order  # noqa: F401
from .assembly_validation import check_unique_result_ids  # noqa: F401
from .barrier_import import spirv_barrier_statements
from .block_arrays import SpirvBlockArrayUnsupportedError  # noqa: F401
from .block_arrays import block_array_element  # noqa: F401
from .block_layout_import import spirv_block_layout  # noqa: F401
from .buffer_memory_qualifiers import storage_block_memory_qualifiers  # noqa: F401
from .function_structure import SpirvFunctionStructure
from .image_formats import crossgl_image_format  # noqa: F401
from .image_texel_atomics import fold_image_texel_atomic
from .import_exact_lookups import builtin_variable_name  # noqa: F401
from .import_exact_lookups import constant_or_literal  # noqa: F401
from .import_exact_lookups import keyword_safe_identifier  # noqa: F401
from .integer_signedness_import import (
    spirv_integer_binary_expression,
    spirv_integer_conversion_expression,
)
from .local_names import function_local_names
from .parsing.access_chain_expressions import VulkanAccessChainMixin
from .parsing.assembly_module import VulkanAssemblyModuleMixin
from .parsing.atomics import VulkanAtomicMixin
from .parsing.declarations import VulkanDeclarationMixin
from .parsing.expressions import VulkanExpressionMixin
from .parsing.globals import VulkanGlobalMixin
from .parsing.images import VulkanImageMixin
from .parsing.instructions import VulkanInstructionMixin
from .parsing.layouts import VulkanLayoutMixin
from .parsing.operands import VulkanOperandMixin
from .parsing.statements import VulkanStatementMixin
from .parsing.values import VulkanValueMixin
from .ray_tracing_import import (
    BUILTIN_VARIABLE_NAMES,
    SpirvInstructionUnsupportedError,
    ray_statement,
    ray_variable_locations,
    refuse_untranslated_statement,
)
from .sampled_images import sampled_image_arguments  # noqa: F401
from .sampled_images import sampled_image_call, sampled_image_image
from .select_import import spirv_select_expression
from .struct_result_import import (
    spirv_extended_arithmetic_call,
    spirv_struct_result_call,
    spirv_struct_result_member,
)
from .subpass_import import is_subpass_image, subpass_load_expression
from .VulkanAst import *
from .VulkanLexer import *
from .workgroup_size_import import workgroup_size_component  # noqa: F401
from .workgroup_size_import import SpirvWorkgroupSizes


class VulkanParser(
    VulkanAssemblyModuleMixin,
    VulkanAccessChainMixin,
    VulkanImageMixin,
    VulkanValueMixin,
    VulkanAtomicMixin,
    VulkanOperandMixin,
    VulkanInstructionMixin,
    VulkanGlobalMixin,
    VulkanLayoutMixin,
    VulkanDeclarationMixin,
    VulkanStatementMixin,
    VulkanExpressionMixin,
):
    """Parse Vulkan/SPIR-V style tokens into the Vulkan backend AST."""

    MAX_SPIRV_MATERIALIZED_EXPRESSION_COMPLEXITY = 160
    SWIZZLE_COMPONENT_SETS = (set("xyzw"), set("rgba"), set("stpq"))
    GEOMETRY_INPUT_PRIMITIVE_QUALIFIERS = {
        "line",
        "lineadj",
        "point",
        "triangle",
        "triangleadj",
    }
    PARAMETER_QUALIFIER_TOKENS = {"CONST", "IN", "OUT", "INOUT"}
    PRECISION_QUALIFIER_TOKENS = {"HIGHP", "MEDIUMP", "LOWP"}
    RAY_TRACING_STORAGE_QUALIFIERS = {
        "callableDataEXT",
        "callableDataInEXT",
        "hitAttributeEXT",
        "rayPayloadEXT",
        "rayPayloadInEXT",
    }
    MESH_SHADER_STORAGE_QUALIFIERS = {
        "taskPayloadSharedEXT",
        "taskPayloadSharedNV",
    }
    LAYOUT_DECLARATION_QUALIFIERS = {
        "centroid",
        "coherent",
        "flat",
        "highp",
        "invariant",
        "lowp",
        "mediump",
        "noperspective",
        "patch",
        "perprimitiveEXT",
        "pervertexEXT",
        "precise",
        "readonly",
        "restrict",
        "sample",
        "smooth",
        "volatile",
        "writeonly",
        *RAY_TRACING_STORAGE_QUALIFIERS,
    }
    DECLARATION_QUALIFIERS = LAYOUT_DECLARATION_QUALIFIERS | {
        "const",
        "groupshared",
        "in",
        "inout",
        "out",
        *MESH_SHADER_STORAGE_QUALIFIERS,
    }
    ASSIGNMENT_TOKENS = (
        "EQUALS",
        "PLUS_EQUALS",
        "MINUS_EQUALS",
        "MULTIPLY_EQUALS",
        "DIVIDE_EQUALS",
        "ASSIGN_AND",
        "ASSIGN_OR",
        "ASSIGN_XOR",
        "ASSIGN_MOD",
        "ASSIGN_SHIFT_LEFT",
        "ASSIGN_SHIFT_RIGHT",
    )
    SPIRV_INTERFACE_STORAGE_CLASSES = {"Input": "IN", "Output": "OUT"}
    # SPV_KHR_ray_tracing: a shader binding table record, read through no descriptor.
    SPIRV_SHADER_RECORD_STORAGE_CLASSES = frozenset(
        {"ShaderRecordBufferKHR", "ShaderRecordBufferNV"}
    )
    # SPV_KHR_ray_tracing storage classes and the GL_EXT_ray_tracing qualifiers that declare them.
    SPIRV_RAY_STORAGE_QUALIFIERS = {
        "RayPayloadKHR": "rayPayloadEXT",
        "IncomingRayPayloadKHR": "rayPayloadInEXT",
        "CallableDataKHR": "callableDataEXT",
        "IncomingCallableDataKHR": "callableDataInEXT",
        "HitAttributeKHR": "hitAttributeEXT",
        "RayPayloadNV": "rayPayloadEXT",
        "IncomingRayPayloadNV": "rayPayloadInEXT",
        "CallableDataNV": "callableDataEXT",
        "IncomingCallableDataNV": "callableDataInEXT",
        "HitAttributeNV": "hitAttributeEXT",
    }
    SPIRV_INTERFACE_DECORATIONS = {
        "BuiltIn": "builtin",
        "Location": "location",
        "Component": "component",
        "Index": "index",
        # A geometry output's vertex stream (GLSL layout(stream = n)).
        "Stream": "stream",
    }
    SPIRV_DECLARATION_DECORATION_QUALIFIERS = {
        "Centroid": "centroid",
        "Coherent": "coherent",
        "Flat": "flat",
        "Invariant": "invariant",
        "NonReadable": "writeonly",
        "NonWritable": "readonly",
        "NoPerspective": "noperspective",
        "Patch": "patch",
        "PerVertexKHR": "pervertexEXT",
        "PerVertexNV": "pervertexEXT",
        "PerPrimitiveEXT": "perprimitiveEXT",
        "RelaxedPrecision": "mediump",
        "Restrict": "restrict",
        "Sample": "sample",
        "Volatile": "volatile",
    }
    SPIRV_VECTOR_TYPES = {
        ("half", "2"): "half2",
        ("half", "3"): "half3",
        ("half", "4"): "half4",
        ("float", "2"): "vec2",
        ("float", "3"): "vec3",
        ("float", "4"): "vec4",
        ("int", "2"): "ivec2",
        ("int", "3"): "ivec3",
        ("int", "4"): "ivec4",
        ("uint", "2"): "uvec2",
        ("uint", "3"): "uvec3",
        ("uint", "4"): "uvec4",
        ("bool", "2"): "bvec2",
        ("bool", "3"): "bvec3",
        ("bool", "4"): "bvec4",
    }
    SPIRV_BUILTIN_VARIABLE_NAMES = {
        "BaseInstance": "gl_BaseInstance",
        "BaseVertex": "gl_BaseVertex",
        "BaryCoordKHR": "gl_BaryCoordEXT",
        "BaryCoordNV": "gl_BaryCoordEXT",
        "BaryCoordNoPerspKHR": "gl_BaryCoordNoPerspEXT",
        "BaryCoordNoPerspNV": "gl_BaryCoordNoPerspEXT",
        "ClipDistance": "gl_ClipDistance",
        "CullDistance": "gl_CullDistance",
        "FragCoord": "gl_FragCoord",
        "FragDepth": "gl_FragDepth",
        "FrontFacing": "gl_FrontFacing",
        "FragStencilRefEXT": "gl_FragStencilRefEXT",
        "GlobalInvocationId": "gl_GlobalInvocationID",
        "HelperInvocation": "gl_HelperInvocation",
        "InstanceId": "gl_InstanceID",
        "InstanceIndex": "gl_InstanceIndex",
        "InvocationId": "gl_InvocationID",
        "Layer": "gl_Layer",
        "LocalInvocationId": "gl_LocalInvocationID",
        "LocalInvocationIndex": "gl_LocalInvocationIndex",
        "NumSubgroups": "gl_NumSubgroups",
        "NumWorkgroups": "gl_NumWorkGroups",
        "PatchVertices": "gl_PatchVerticesIn",
        "PointCoord": "gl_PointCoord",
        "PointSize": "gl_PointSize",
        "Position": "gl_Position",
        "PrimitiveId": "gl_PrimitiveID",
        "SampleId": "gl_SampleID",
        "SamplePosition": "gl_SamplePosition",
        "SubgroupId": "gl_SubgroupID",
        "SubgroupLocalInvocationId": "gl_SubgroupInvocationID",
        "SubgroupSize": "gl_SubgroupSize",
        "TessCoord": "gl_TessCoord",
        "TessLevelInner": "gl_TessLevelInner",
        "TessLevelOuter": "gl_TessLevelOuter",
        "VertexId": "gl_VertexID",
        "VertexIndex": "gl_VertexID",
        "ViewportIndex": "gl_ViewportIndex",
        "ViewIndex": "gl_ViewIndex",
        "WorkgroupId": "gl_WorkGroupID",
        "WorkgroupSize": "gl_WorkGroupSize",
        **BUILTIN_VARIABLE_NAMES,
    }
    SPIRV_STORAGE_BUILTIN_VARIABLE_NAMES = {
        ("SampleMask", "Input"): "gl_SampleMaskIn",
        ("SampleMask", "Output"): "gl_SampleMask",
    }
    SPIRV_SPEC_CONSTANT_BINARY_OPS = {
        "BitwiseAnd": "&",
        "BitwiseOr": "|",
        "BitwiseXor": "^",
        "FAdd": "+",
        "FDiv": "/",
        "FMod": "%",
        "FMul": "*",
        "FOrdEqual": "==",
        "FOrdGreaterThan": ">",
        "FOrdGreaterThanEqual": ">=",
        "FOrdLessThan": "<",
        "FOrdLessThanEqual": "<=",
        "FOrdNotEqual": "!=",
        "FRem": "%",
        "FSub": "-",
        "FUnordEqual": "==",
        "FUnordGreaterThan": ">",
        "FUnordGreaterThanEqual": ">=",
        "FUnordLessThan": "<",
        "FUnordLessThanEqual": "<=",
        "FUnordNotEqual": "!=",
        "IAdd": "+",
        "IEqual": "==",
        "IMul": "*",
        "INotEqual": "!=",
        "ISub": "-",
        "LogicalAnd": "&&",
        "LogicalEqual": "==",
        "LogicalNotEqual": "!=",
        "LogicalOr": "||",
        "MatrixTimesMatrix": "*",
        "MatrixTimesScalar": "*",
        "MatrixTimesVector": "*",
        "SDiv": "/",
        "SGreaterThan": ">",
        "SGreaterThanEqual": ">=",
        "SLessThan": "<",
        "SLessThanEqual": "<=",
        "SMod": "%",
        "SRem": "%",
        "ShiftLeftLogical": "<<",
        "ShiftRightArithmetic": ">>",
        "ShiftRightLogical": ">>",
        "UDiv": "/",
        "UGreaterThan": ">",
        "UGreaterThanEqual": ">=",
        "ULessThan": "<",
        "ULessThanEqual": "<=",
        "UMod": "%",
        "VectorTimesMatrix": "*",
        "VectorTimesScalar": "*",
    }
    SPIRV_SPEC_CONSTANT_UNARY_OPS = {
        "FNegate": "-",
        "LogicalNot": "!",
        "Not": "~",
        "SNegate": "-",
    }
    SPIRV_DERIVATIVE_FUNCTIONS = {
        "DPdx": "dFdx",
        "DPdy": "dFdy",
        "Fwidth": "fwidth",
        "DPdxCoarse": "dFdxCoarse",
        "DPdyCoarse": "dFdyCoarse",
        "FwidthCoarse": "fwidthCoarse",
        "DPdxFine": "dFdxFine",
        "DPdyFine": "dFdyFine",
        "FwidthFine": "fwidthFine",
    }
    SPIRV_UNARY_FUNCTIONS = {
        "All": "all",
        "Any": "any",
        "IsInf": "isinf",
        "IsNan": "isnan",
    }
    SPIRV_CORE_FUNCTIONS = {
        "BitCount": "bitCount",
        "BitFieldInsert": "bitfieldInsert",
        "BitFieldSExtract": "bitfieldExtract",
        "BitFieldUExtract": "bitfieldExtract",
        "BitReverse": "bitfieldReverse",
        "OuterProduct": "outerProduct",
        "QuantizeToF16": "spirvQuantizeToF16",
        "SatConvertSToU": "spirvSatConvertSToU",
        "SDot": "spirvSDot",
        "SDotKHR": "spirvSDot",
        "SUDot": "spirvSUDot",
        "SUDotKHR": "spirvSUDot",
        "UDot": "spirvUDot",
        "UDotKHR": "spirvUDot",
        "SDotAccSat": "spirvSDotAccSat",
        "SDotAccSatKHR": "spirvSDotAccSat",
        "SUDotAccSat": "spirvSUDotAccSat",
        "SUDotAccSatKHR": "spirvSUDotAccSat",
        "UDotAccSat": "spirvUDotAccSat",
        "UDotAccSatKHR": "spirvUDotAccSat",
    }
    SPIRV_ATOMIC_RMW_FUNCTIONS = {
        "OpAtomicIAdd": "atomicAdd",
        "OpAtomicISub": "atomicAdd",
        "OpAtomicSMin": "atomicMin",
        "OpAtomicUMin": "atomicMin",
        "OpAtomicSMax": "atomicMax",
        "OpAtomicUMax": "atomicMax",
        "OpAtomicAnd": "atomicAnd",
        "OpAtomicOr": "atomicOr",
        "OpAtomicXor": "atomicXor",
        "OpAtomicExchange": "atomicExchange",
    }
    SPIRV_GEOMETRY_EMIT_FUNCTIONS = {
        "OpEmitVertex": "EmitVertex",
        "OpEndPrimitive": "EndPrimitive",
    }
    SPIRV_RAY_QUERY_STATEMENT_FUNCTIONS = {
        "OpRayQueryInitializeKHR": "rayQueryInitializeEXT",
        "OpRayQueryInitializeNV": "rayQueryInitializeEXT",
        "OpRayQueryTerminateKHR": "rayQueryTerminateEXT",
        "OpRayQueryTerminateNV": "rayQueryTerminateEXT",
    }
    SPIRV_RAY_QUERY_EXPRESSION_FUNCTIONS = {
        "OpRayQueryProceedKHR": "rayQueryProceedEXT",
        "OpRayQueryProceedNV": "rayQueryProceedEXT",
        "OpRayQueryGetIntersectionTypeKHR": "rayQueryGetIntersectionTypeEXT",
        "OpRayQueryGetIntersectionTypeNV": "rayQueryGetIntersectionTypeEXT",
        "OpRayQueryGetIntersectionTKHR": "rayQueryGetIntersectionTEXT",
        "OpRayQueryGetIntersectionTNV": "rayQueryGetIntersectionTEXT",
        "OpRayQueryGetIntersectionInstanceCustomIndexKHR": (
            "rayQueryGetIntersectionInstanceCustomIndexEXT"
        ),
        "OpRayQueryGetIntersectionInstanceCustomIndexNV": (
            "rayQueryGetIntersectionInstanceCustomIndexEXT"
        ),
        "OpRayQueryGetIntersectionInstanceIdKHR": (
            "rayQueryGetIntersectionInstanceIdEXT"
        ),
        "OpRayQueryGetIntersectionInstanceIdNV": "rayQueryGetIntersectionInstanceIdEXT",
        "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR": (
            "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT"
        ),
        "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetNV": (
            "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT"
        ),
        "OpRayQueryGetIntersectionGeometryIndexKHR": (
            "rayQueryGetIntersectionGeometryIndexEXT"
        ),
        "OpRayQueryGetIntersectionGeometryIndexNV": (
            "rayQueryGetIntersectionGeometryIndexEXT"
        ),
        "OpRayQueryGetIntersectionPrimitiveIndexKHR": (
            "rayQueryGetIntersectionPrimitiveIndexEXT"
        ),
        "OpRayQueryGetIntersectionPrimitiveIndexNV": (
            "rayQueryGetIntersectionPrimitiveIndexEXT"
        ),
        "OpRayQueryGetIntersectionBarycentricsKHR": (
            "rayQueryGetIntersectionBarycentricsEXT"
        ),
        "OpRayQueryGetIntersectionBarycentricsNV": (
            "rayQueryGetIntersectionBarycentricsEXT"
        ),
        "OpRayQueryGetIntersectionFrontFaceKHR": "rayQueryGetIntersectionFrontFaceEXT",
        "OpRayQueryGetIntersectionFrontFaceNV": "rayQueryGetIntersectionFrontFaceEXT",
        "OpRayQueryGetIntersectionObjectRayDirectionKHR": (
            "rayQueryGetIntersectionObjectRayDirectionEXT"
        ),
        "OpRayQueryGetIntersectionObjectRayDirectionNV": (
            "rayQueryGetIntersectionObjectRayDirectionEXT"
        ),
        "OpRayQueryGetIntersectionObjectRayOriginKHR": (
            "rayQueryGetIntersectionObjectRayOriginEXT"
        ),
        "OpRayQueryGetIntersectionObjectRayOriginNV": (
            "rayQueryGetIntersectionObjectRayOriginEXT"
        ),
        "OpRayQueryGetIntersectionObjectToWorldKHR": (
            "rayQueryGetIntersectionObjectToWorldEXT"
        ),
        "OpRayQueryGetIntersectionObjectToWorldNV": (
            "rayQueryGetIntersectionObjectToWorldEXT"
        ),
        "OpRayQueryGetIntersectionWorldToObjectKHR": (
            "rayQueryGetIntersectionWorldToObjectEXT"
        ),
        "OpRayQueryGetIntersectionWorldToObjectNV": (
            "rayQueryGetIntersectionWorldToObjectEXT"
        ),
        "OpRayQueryGetRayFlagsKHR": "rayQueryGetRayFlagsEXT",
        "OpRayQueryGetRayFlagsNV": "rayQueryGetRayFlagsEXT",
        "OpRayQueryGetRayTMinKHR": "rayQueryGetRayTMinEXT",
        "OpRayQueryGetRayTMinNV": "rayQueryGetRayTMinEXT",
        "OpRayQueryGetWorldRayDirectionKHR": "rayQueryGetWorldRayDirectionEXT",
        "OpRayQueryGetWorldRayDirectionNV": "rayQueryGetWorldRayDirectionEXT",
        "OpRayQueryGetWorldRayOriginKHR": "rayQueryGetWorldRayOriginEXT",
        "OpRayQueryGetWorldRayOriginNV": "rayQueryGetWorldRayOriginEXT",
    }
    SPIRV_GROUP_NON_UNIFORM_REDUCTION_FUNCTIONS = {
        "OpGroupNonUniformBitwiseAnd": "And",
        "OpGroupNonUniformBitwiseOr": "Or",
        "OpGroupNonUniformBitwiseXor": "Xor",
        "OpGroupNonUniformFAdd": "Add",
        "OpGroupNonUniformFMax": "Max",
        "OpGroupNonUniformFMin": "Min",
        "OpGroupNonUniformFMul": "Mul",
        "OpGroupNonUniformIAdd": "Add",
        "OpGroupNonUniformIMul": "Mul",
        "OpGroupNonUniformLogicalAnd": "And",
        "OpGroupNonUniformLogicalOr": "Or",
        "OpGroupNonUniformLogicalXor": "Xor",
        "OpGroupNonUniformSMax": "Max",
        "OpGroupNonUniformSMin": "Min",
        "OpGroupNonUniformUMax": "Max",
        "OpGroupNonUniformUMin": "Min",
    }
    SPIRV_GROUP_NON_UNIFORM_SCAN_PREFIXES = {
        "Reduce": "subgroup",
        "InclusiveScan": "subgroupInclusive",
        "ExclusiveScan": "subgroupExclusive",
        "ClusteredReduce": "subgroupClustered",
    }
    LEGACY_GLSLANG_SPIRV_ID_OPERANDS = {
        "AccessChain": "all",
        "CompositeConstruct": "all",
        "CompositeExtract": {0, 1},
        "Constant": {0},
        "ConstantComposite": "all",
        "ConvertSToF": {0, 1},
        "ConvertUToF": {0, 1},
        "Decorate": {0},
        "EntryPoint": "entry_point",
        "ExecutionMode": {0},
        "ExecutionModeId": "execution_mode_id",
        "ExtInstImport": set(),
        "FAdd": {0, 1, 2},
        "FMul": {0, 1, 2},
        "Function": {0, 2},
        "FunctionCall": "all",
        "FunctionEnd": set(),
        "FunctionParameter": {0},
        "Label": set(),
        "Load": {0, 1},
        "MemberDecorate": {0},
        "MemberName": {0},
        "Name": {0},
        "Return": set(),
        "ReturnValue": {0},
        "Store": {0, 1},
        "TypeArray": {0, 1},
        "TypeBool": set(),
        "TypeFloat": set(),
        "TypeFunction": "all",
        "TypeInt": set(),
        "TypeMatrix": {0},
        "TypePointer": {1},
        "TypeRuntimeArray": {0},
        "TypeStruct": "all",
        "TypeVector": {0},
        "TypeVoid": set(),
        "Variable": {0, 2},
        "VectorTimesScalar": {0, 1, 2},
    }
    LEGACY_GLSLANG_SPIRV_GLOBAL_OPCODES = {
        "Capability",
        "Extension",
        "ExtInstImport",
        "MemoryModel",
        "Source",
        "SourceExtension",
    }
    CROSSGL_RESERVED_IDENTIFIERS = {
        "as",
        "async",
        "await",
        "bool",
        "box",
        "break",
        "buffer",
        "case",
        "cbuffer",
        "char",
        "class",
        "compute",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "fn",
        "for",
        "fragment",
        "from",
        "geometry",
        "global",
        "half",
        "if",
        "impl",
        "import",
        "in",
        "int",
        "interface",
        "let",
        "local",
        "loop",
        "match",
        "mesh",
        "module",
        "move",
        "mut",
        "namespace",
        "object",
        "precision",
        "priv",
        "protected",
        "pub",
        "ref",
        "return",
        "safe",
        "shader",
        "shared",
        "static",
        "string",
        "struct",
        "switch",
        "task",
        "tessellation",
        "tessellation_control",
        "tessellation_evaluation",
        "threadgroup",
        "trait",
        "i8",
        "u8",
        "u16",
        "u32",
        "u64",
        "uint",
        "uniform",
        "unsafe",
        "use",
        "var",
        "vertex",
        "void",
        "while",
        "workgroup",
        "yield",
    }
    SPIRV_VALUE_FALLBACK_RENAMES = {
        "buffer": "buffer_",
        "global": "global_",
        "in": "in_",
        "out": "out_",
        "struct": "struct_",
        "var": "var_",
    }
    SPIRV_GLSL_STD_450_EXT_INST_FUNCTIONS = {
        "Acos": "acos",
        "Acosh": "acosh",
        "Asin": "asin",
        "Asinh": "asinh",
        "Atan": "atan",
        "Atan2": "atan2",
        "Atanh": "atanh",
        "Ceil": "ceil",
        "Cos": "cos",
        "Cosh": "cosh",
        "Cross": "cross",
        "Degrees": "degrees",
        "Determinant": "determinant",
        "Distance": "distance",
        "Exp": "exp",
        "Exp2": "exp2",
        "FAbs": "abs",
        "FClamp": "clamp",
        "FaceForward": "faceforward",
        "FindILsb": "findLSB",
        "FindSMsb": "findMSB",
        "FindUMsb": "findMSB",
        "Floor": "floor",
        "FMax": "max",
        "FMin": "min",
        "FMix": "mix",
        "FSign": "sign",
        "Fma": "fma",
        "Frexp": "frexp",
        "FrexpStruct": "frexp",
        "Fract": "fract",
        "IMix": "mix",
        "InterpolateAtCentroid": "interpolateAtCentroid",
        "InterpolateAtOffset": "interpolateAtOffset",
        "InterpolateAtSample": "interpolateAtSample",
        "InverseSqrt": "inversesqrt",
        "Ldexp": "ldexp",
        "Length": "length",
        "Log": "log",
        "Log2": "log2",
        "MatrixInverse": "inverse",
        "Modf": "modf",
        "ModfStruct": "modf",
        # NMin, NMax and NClamp return the non-NaN operand (GLSL.std.450 sections 79-81),
        # which is the C99 fmin/fmax rule; FMin and FMax leave that case undefined.
        "NClamp": "fmin",
        "Normalize": "normalize",
        "NMax": "fmax",
        "NMin": "fmin",
        "PackDouble2x32": "packDouble2x32",
        "PackHalf2x16": "packHalf2x16",
        "PackSnorm2x16": "packSnorm2x16",
        "PackSnorm4x8": "packSnorm4x8",
        "PackUnorm2x16": "packUnorm2x16",
        "PackUnorm4x8": "packUnorm4x8",
        "Pow": "pow",
        "Radians": "radians",
        "Reflect": "reflect",
        "Refract": "refract",
        "Round": "round",
        "RoundEven": "roundEven",
        "SAbs": "abs",
        "SClamp": "clamp",
        "SMax": "max",
        "SMin": "min",
        "SSign": "sign",
        "Sin": "sin",
        "Sinh": "sinh",
        "SmoothStep": "smoothstep",
        "Sqrt": "sqrt",
        "Step": "step",
        "Tan": "tan",
        "Tanh": "tanh",
        "Trunc": "trunc",
        "UClamp": "clamp",
        "UMax": "max",
        "UMin": "min",
        "UnpackDouble2x32": "unpackDouble2x32",
        "UnpackHalf2x16": "unpackHalf2x16",
        "UnpackSnorm2x16": "unpackSnorm2x16",
        "UnpackSnorm4x8": "unpackSnorm4x8",
        "UnpackUnorm2x16": "unpackUnorm2x16",
        "UnpackUnorm4x8": "unpackUnorm4x8",
    }
    SPIRV_GLSL_STD_450_EXT_INST_IDS = {
        "1": "Round",
        "2": "RoundEven",
        "3": "Trunc",
        "4": "FAbs",
        "5": "SAbs",
        "6": "FSign",
        "7": "SSign",
        "8": "Floor",
        "9": "Ceil",
        "10": "Fract",
        "11": "Radians",
        "12": "Degrees",
        "13": "Sin",
        "14": "Cos",
        "15": "Tan",
        "16": "Asin",
        "17": "Acos",
        "18": "Atan",
        "19": "Sinh",
        "20": "Cosh",
        "21": "Tanh",
        "22": "Asinh",
        "23": "Acosh",
        "24": "Atanh",
        "25": "Atan2",
        "26": "Pow",
        "27": "Exp",
        "28": "Log",
        "29": "Exp2",
        "30": "Log2",
        "31": "Sqrt",
        "32": "InverseSqrt",
        "33": "Determinant",
        "34": "MatrixInverse",
        "35": "Modf",
        "36": "ModfStruct",
        "37": "FMin",
        "38": "UMin",
        "39": "SMin",
        "40": "FMax",
        "41": "UMax",
        "42": "SMax",
        "43": "FClamp",
        "44": "UClamp",
        "45": "SClamp",
        "46": "FMix",
        "47": "IMix",
        "48": "Step",
        "49": "SmoothStep",
        "50": "Fma",
        "51": "Frexp",
        "52": "FrexpStruct",
        "53": "Ldexp",
        "54": "PackSnorm4x8",
        "55": "PackUnorm4x8",
        "56": "PackSnorm2x16",
        "57": "PackUnorm2x16",
        "58": "PackHalf2x16",
        "59": "PackDouble2x32",
        "60": "UnpackSnorm2x16",
        "61": "UnpackUnorm2x16",
        "62": "UnpackHalf2x16",
        "63": "UnpackSnorm4x8",
        "64": "UnpackUnorm4x8",
        "65": "UnpackDouble2x32",
        "66": "Length",
        "67": "Distance",
        "68": "Cross",
        "69": "Normalize",
        "70": "FaceForward",
        "71": "Reflect",
        "72": "Refract",
        "73": "FindILsb",
        "74": "FindSMsb",
        "75": "FindUMsb",
        "76": "InterpolateAtCentroid",
        "77": "InterpolateAtSample",
        "78": "InterpolateAtOffset",
        "79": "NMin",
        "80": "NMax",
        "81": "NClamp",
    }

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.loop_depth = 0
        self.spirv_module_execution_models = []
        self.breakable_depth = 0
        self.skip_comments()

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def peek(self, offset):
        peek_index = self.pos + offset
        if peek_index < len(self.tokens):
            return self.tokens[peek_index][0]
        return None

    def peek_value(self, offset):
        peek_index = self.pos + offset
        if peek_index < len(self.tokens):
            return self.tokens[peek_index][1]
        return None

    def skip_until(self, token_type):
        while self.current_token[0] != token_type and self.current_token[0] != "EOF":
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = ("EOF", None)
        return

    def expect_end_of_input(self):
        if self.current_token[0] != "EOF":
            raise SpirvSyntaxError(f"Expected EOF, got {self.current_token[0]}")

    def eat(self, token_type):
        # The end of input is never consumed: a construct that reads past it is unterminated.
        if token_type == "EOF" and self.current_token[0] == "EOF":
            raise SpirvSyntaxError("Unexpected end of input")
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SpirvSyntaxError(
                f"Expected {token_type}, got {self.current_token[0]}"
            )

    def skip_hlsl_attributes(self):
        while self.current_token[0] == "LBRACKET":
            if self.peek(1) == "LBRACKET":
                self.skip_double_bracket_hlsl_attribute()
            elif self.peek(1) == "IDENTIFIER":
                self.skip_single_bracket_hlsl_attribute()
            else:
                break

    def skip_double_bracket_hlsl_attribute(self):
        self.eat("LBRACKET")
        self.eat("LBRACKET")
        depth = 1
        while depth and self.current_token[0] != "EOF":
            if self.current_token[0] == "LBRACKET" and self.peek(1) == "LBRACKET":
                depth += 1
                self.eat("LBRACKET")
                self.eat("LBRACKET")
            elif self.current_token[0] == "RBRACKET" and self.peek(1) == "RBRACKET":
                depth -= 1
                self.eat("RBRACKET")
                self.eat("RBRACKET")
            else:
                self.eat(self.current_token[0])
        if depth:
            raise SpirvSyntaxError("Unterminated HLSL attribute")

    def skip_single_bracket_hlsl_attribute(self):
        self.eat("LBRACKET")
        depth = 1
        while depth and self.current_token[0] != "EOF":
            if self.current_token[0] == "LBRACKET":
                depth += 1
            elif self.current_token[0] == "RBRACKET":
                depth -= 1
            self.eat(self.current_token[0])
        if depth:
            raise SpirvSyntaxError("Unterminated HLSL attribute")

    def parse_optional_hlsl_semantic(self):
        if self.current_token[0] == "SEMANTIC":
            semantic = self.current_token[1].lstrip(":")
            self.eat("SEMANTIC")
            return semantic

        if self.current_token[0] != "COLON" or self.peek(1) not in {
            "IDENTIFIER",
            "SEMANTIC",
        }:
            return None

        self.eat("COLON")
        semantic = self.current_token[1].lstrip(":")
        self.eat(self.current_token[0])
        if self.current_token[0] == "LPAREN":
            semantic += self.parse_parenthesized_text()
        return semantic

    def parse_parenthesized_text(self):
        self.eat("LPAREN")
        parts = []
        depth = 1
        while depth and self.current_token[0] != "EOF":
            if self.current_token[0] == "LPAREN":
                depth += 1
                parts.append("(")
                self.eat("LPAREN")
            elif self.current_token[0] == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
                parts.append(")")
                self.eat("RPAREN")
            else:
                parts.append(str(self.current_token[1]))
                self.eat(self.current_token[0])
        if depth:
            raise SpirvSyntaxError("Unterminated parenthesized semantic")
        return f"({''.join(parts)})"

    def skip_balanced_brace_tokens(self):
        self.eat("LBRACE")
        depth = 1
        while depth and self.current_token[0] != "EOF":
            if self.current_token[0] == "LBRACE":
                depth += 1
            elif self.current_token[0] == "RBRACE":
                depth -= 1
            self.eat(self.current_token[0])
        if depth:
            raise SpirvSyntaxError("Unterminated brace block")

    def parse(self):
        module = self.parse_module()
        self.expect_end_of_input()
        return module

    def parse_module(self):
        if self.current_token[0] == "SPIRV_ASSEMBLY":
            code = self.current_token[1]
            self.eat("SPIRV_ASSEMBLY")
            return self.parse_spirv_assembly_module(code)

        functions = []
        structs = []
        global_variables = []
        while self.current_token[0] != "EOF":
            self.skip_hlsl_attributes()
            if self.current_token[0] == "EOF":
                break
            if self.current_token[0] == "PRECISION":
                self.parse_precision_declaration()
            elif self.current_token[0] == "LAYOUT":
                global_variables.append(self.parse_layout())
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] == "UNIFORM":
                global_variables.append(self.parse_uniform())
            elif (
                (
                    self.current_token[0]
                    in [
                        "VOID",
                        "FLOAT",
                        "INT",
                        "UINT",
                        "BOOL",
                        "VEC2",
                        "VEC3",
                        "VEC4",
                        "MAT2",
                        "MAT3",
                        "MAT4",
                    ]
                    or self.current_token[1] in VALID_DATA_TYPES
                    or self.current_token[0] == "IDENTIFIER"
                )
                and self.peek(1) == "IDENTIFIER"
                and self.peek(2) == "LPAREN"
            ):
                functions.append(self.parse_function())
            elif (
                (self.current_token[0] == "IDENTIFIER" and self.peek(1) == "IDENTIFIER")
                or self.current_token[1] in self.DECLARATION_QUALIFIERS
                or (
                    self.current_token[1] in VALID_DATA_TYPES
                    and self.peek(1) == "IDENTIFIER"
                )
                or (
                    self.current_token[0] == "CONST"
                    and (
                        self.peek_value(1) in VALID_DATA_TYPES
                        or self.peek(1) == "IDENTIFIER"
                    )
                )
            ):
                global_variables.append(self.parse_assignment_or_function_call())
            else:
                self.eat(self.current_token[0])
        return ShaderNode(
            functions=functions,
            structs=structs,
            global_variables=global_variables,
        )

    def parse_spirv_assembly_module(self, code):
        instructions = self.parse_spirv_assembly_instructions(code)
        names = {}
        decorations = {}
        member_decorations = {}
        member_names = {}
        types = {}
        constants = {}
        constant_types = {}
        spec_constant_ids = []
        extended_instruction_imports = {}
        capabilities = []
        variables = []
        entry_points = []
        execution_modes = {}
        group_decorates = []
        group_member_decorates = []
        workgroup_sizes = SpirvWorkgroupSizes(instructions)
        self.spirv_workgroup_sizes = workgroup_sizes

        for result_id, opcode, operands, _line_number in instructions:
            if result_id in workgroup_sizes.names:
                names[result_id] = workgroup_sizes.names[result_id]
            if opcode == "OpCapability" and operands:
                capabilities.append(operands[0])
            elif opcode == "OpName" and len(operands) >= 2:
                names[operands[0]] = (
                    self.spirv_identifier_name(operands[1], operands[0])
                    if operands[1]
                    else ""
                )
            elif opcode == "OpMemberName" and len(operands) >= 3:
                target, member = operands[0], operands[1]
                name = self.spirv_identifier_name(
                    operands[2],
                    f"{target}_{member}",
                    prefix="member",
                )
                member_names.setdefault(target, {})[member] = name
            elif opcode == "OpDecorate" and len(operands) >= 2:
                target, decoration = operands[0], operands[1]
                decorations.setdefault(target, []).append((decoration, operands[2:]))
            elif opcode == "OpMemberDecorate" and len(operands) >= 3:
                target, member, decoration = operands[0], operands[1], operands[2]
                member_decorations.setdefault(target, []).append(
                    (member, decoration, operands[3:])
                )
            elif opcode == "OpGroupDecorate" and len(operands) >= 2:
                group_decorates.append((operands[0], operands[1:]))
            elif opcode == "OpGroupMemberDecorate" and len(operands) >= 3:
                group_member_decorates.append((operands[0], operands[1:]))
            elif opcode == "OpEntryPoint" and len(operands) >= 3:
                entry_points.append(
                    {
                        "execution_model": operands[0],
                        "id": operands[1],
                        "name": self.spirv_identifier_name(
                            operands[2], operands[1], prefix="entry_point"
                        ),
                        "interface_ids": operands[3:],
                    }
                )
            elif (
                opcode in {"OpExecutionMode", "OpExecutionModeId"}
                and len(operands) >= 2
            ):
                execution_modes.setdefault(operands[0], []).append(
                    {
                        "opcode": opcode,
                        "mode": operands[1],
                        "operands": operands[2:],
                    }
                )
            elif result_id and opcode == "OpTypeVoid":
                types[result_id] = {"kind": "scalar", "name": "void"}
            elif result_id and opcode == "OpTypeBool":
                types[result_id] = {"kind": "scalar", "name": "bool"}
            elif result_id and opcode == "OpTypeFloat" and operands:
                types[result_id] = {
                    "kind": "scalar",
                    "name": self.spirv_float_type_name(operands[0]),
                }
            elif result_id and opcode == "OpTypeInt" and len(operands) >= 2:
                types[result_id] = {
                    "kind": "scalar",
                    "name": self.spirv_int_type_name(operands[0], operands[1]),
                }
            elif result_id and opcode == "OpTypeVector" and len(operands) >= 2:
                component_type = self.spirv_type_name(operands[0], types)
                types[result_id] = {
                    "kind": "vector",
                    "name": self.spirv_vector_type_name(component_type, operands[1]),
                    "component_type": operands[0],
                    "component_count": operands[1],
                }
            elif result_id and opcode == "OpTypeMatrix" and len(operands) >= 2:
                column_type = types.get(operands[0], {})
                component_type = self.spirv_type_name(
                    column_type.get("component_type"), types
                )
                types[result_id] = {
                    "kind": "matrix",
                    "name": self.spirv_matrix_type_name(
                        component_type,
                        column_type.get("component_count"),
                        operands[1],
                    ),
                    "column_type": operands[0],
                    "column_count": operands[1],
                }
            elif (
                result_id
                and opcode in {
                    "OpTypeCooperativeMatrixKHR",
                    "OpTypeCooperativeMatrixNV",
                }
                and len(operands) >= 4
            ):
                types[result_id] = {
                    "kind": "cooperative_matrix",
                    "name": self.spirv_cooperative_matrix_type_name(
                        opcode, operands, types, constants
                    ),
                    "component_type": operands[0],
                    "scope": operands[1],
                    "rows": operands[2],
                    "columns": operands[3],
                    "use": operands[4] if len(operands) >= 5 else None,
                }
            elif result_id and opcode == "OpTypeArray" and len(operands) >= 2:
                types[result_id] = {
                    "kind": "array",
                    "element_type": operands[0],
                    "length_id": operands[1],
                }
            elif result_id and opcode == "OpTypeRuntimeArray" and len(operands) >= 1:
                types[result_id] = {
                    "kind": "runtime_array",
                    "element_type": operands[0],
                }
            elif result_id and opcode == "OpTypeStruct":
                types[result_id] = {"kind": "struct", "member_types": operands}
            elif result_id and opcode == "OpTypeImage" and len(operands) >= 7:
                sampled_type = self.spirv_type_name(operands[0], types)
                types[result_id] = {
                    "kind": "image",
                    "name": self.spirv_image_type_name(
                        sampled_type,
                        operands[1],
                        operands[2],
                        operands[3],
                        operands[4],
                        operands[5],
                    ),
                    "sampled_type": operands[0],
                    "dim": operands[1],
                    "depth": operands[2],
                    "arrayed": operands[3],
                    "multisampled": operands[4],
                    "sampled": operands[5],
                    "format": operands[6],
                    "access_qualifier": operands[7] if len(operands) >= 8 else None,
                }
            elif result_id and opcode == "OpTypeSampledImage" and operands:
                image_type = types.get(operands[0], {})
                types[result_id] = {
                    "kind": "sampled_image",
                    "name": self.spirv_sampled_image_type_name(image_type, types),
                    "image_type": operands[0],
                }
            elif result_id and opcode == "OpTypeSampler":
                types[result_id] = {"kind": "sampler", "name": "sampler"}
            elif result_id and opcode == "OpTypeNamedBarrier":
                types[result_id] = {
                    "kind": "named_barrier",
                    "name": "spirvNamedBarrier",
                }
            elif result_id and opcode in {
                "OpTypeAccelerationStructureKHR",
                "OpTypeAccelerationStructureNV",
            }:
                types[result_id] = {
                    "kind": "acceleration_structure",
                    "name": "accelerationStructureEXT",
                }
            elif result_id and opcode in {"OpTypeRayQueryKHR", "OpTypeRayQueryNV"}:
                types[result_id] = {
                    "kind": "ray_query",
                    "name": "rayQueryEXT",
                }
            elif result_id and opcode == "OpTypePointer" and len(operands) >= 2:
                types[result_id] = {
                    "kind": "pointer",
                    "storage_class": operands[0],
                    "type_id": operands[1],
                }
            elif result_id and opcode == "OpTypeFunction" and operands:
                types[result_id] = {
                    "kind": "function",
                    "return_type": operands[0],
                    "parameter_types": operands[1:],
                }
            elif result_id and opcode in {"OpConstant", "OpSpecConstant"}:
                if len(operands) >= 2:
                    constant_types[result_id] = operands[0]
                    constants[result_id] = self.spirv_constant_literal_expression(
                        operands[0], operands[1], types
                    )
                    if opcode == "OpSpecConstant":
                        spec_constant_ids.append(result_id)
            elif result_id and opcode == "OpString" and operands:
                constants[result_id] = self.spirv_string_literal(operands[0])
            elif result_id and opcode in {
                "OpConstantFalse",
                "OpConstantTrue",
                "OpSpecConstantFalse",
                "OpSpecConstantTrue",
            }:
                if operands:
                    constant_types[result_id] = operands[0]
                    constants[result_id] = (
                        "true" if opcode.endswith("True") else "false"
                    )
                    if opcode.startswith("OpSpecConstant"):
                        spec_constant_ids.append(result_id)
            elif result_id and opcode == "OpUndef" and operands:
                constant_types[result_id] = operands[0]
                constants[result_id] = self.spirv_assembly_undef_expression(
                    operands[0], types
                )
            elif result_id and opcode == "OpConstantNull" and operands:
                constant_types[result_id] = operands[0]
                constants[result_id] = self.spirv_assembly_null_expression(
                    operands[0], types
                )
            elif result_id and opcode in {
                "OpConstantComposite",
                "OpSpecConstantComposite",
            }:
                if operands:
                    constant_types[result_id] = operands[0]
                    constants[result_id] = self.spirv_constant_composite_expression(
                        operands[0], operands[1:], names, types, constants
                    )
                    if opcode == "OpSpecConstantComposite":
                        spec_constant_ids.append(result_id)
            elif result_id and opcode == "OpSpecConstantOp":
                if len(operands) >= 2:
                    constant_types[result_id] = operands[0]
                    constants[result_id] = self.spirv_spec_constant_op_expression(
                        operands[1], operands[2:], names, constants, types
                    )
                    spec_constant_ids.append(result_id)
            elif result_id and opcode == "OpExtInstImport" and operands:
                extended_instruction_imports[result_id] = operands[0]
            elif result_id and opcode == "OpVariable" and len(operands) >= 2:
                variables.append(
                    {
                        "id": result_id,
                        "pointer_type_id": operands[0],
                        "storage_class": operands[1],
                        "initializer": operands[2] if len(operands) >= 3 else None,
                    }
                )

        self.spirv_apply_decoration_groups(
            decorations, member_decorations, group_decorates, group_member_decorates
        )
        self.spirv_geometry_module = any(
            entry_point["execution_model"] == "Geometry" for entry_point in entry_points
        )
        self.spirv_register_struct_type_names(types, names, member_names)
        self.spirv_name_struct_constants(constants, types)
        self.spirv_assign_flattened_resource_block_member_aliases(
            variables,
            names,
            decorations,
            member_names,
            types,
            constants,
        )
        resource_block_type_ids = self.spirv_resource_block_struct_type_ids(
            variables, types, decorations
        )
        builtin_interface_struct_storage_classes = (
            self.spirv_builtin_interface_struct_storage_classes(
                variables, types, member_decorations
            )
        )
        structs = self.spirv_assembly_structs(
            names,
            member_names,
            member_decorations,
            types,
            constants,
            skip_type_ids=resource_block_type_ids,
            builtin_interface_struct_storage_classes=(
                builtin_interface_struct_storage_classes
            ),
        )
        entry_interface_ids = {
            interface_id
            for entry_point in entry_points
            for interface_id in entry_point["interface_ids"]
        }
        global_variables = self.spirv_assembly_interface_variables(
            variables,
            entry_interface_ids,
            names,
            decorations,
            member_decorations,
            member_names,
            types,
            constants,
        )
        global_variables.extend(
            self.spirv_assembly_private_global_variables(
                variables, names, decorations, types, constants
            )
        )
        global_variables.extend(
            self.spirv_assembly_workgroup_global_variables(
                variables, names, decorations, types, constants
            )
        )
        global_variables = (
            self.spirv_assembly_specialization_constants(
                workgroup_sizes.declared_spec_constants(spec_constant_ids),
                names,
                decorations,
                types,
                constants,
                constant_types,
            )
            + global_variables
        )
        functions = self.spirv_assembly_functions(
            instructions,
            names,
            decorations,
            member_decorations,
            member_names,
            types,
            variables,
            entry_points,
            execution_modes,
            constants,
            constant_types,
            extended_instruction_imports,
        )
        if (
            not global_variables
            and not structs
            and not functions
            and not self.spirv_assembly_has_metadata_only_surface(types)
            and not self.spirv_assembly_has_linkage_metadata_surface(
                capabilities, types, extended_instruction_imports
            )
            and not self.spirv_assembly_has_entry_point_metadata_surface(entry_points)
        ):
            raise SpirvSyntaxError(SPIRV_ASSEMBLY_ERROR)

        return ShaderNode(
            functions=functions,
            structs=structs,
            global_variables=global_variables,
            spirv_assembly=True,
            spirv_entry_points=entry_points,
            spirv_execution_modes=execution_modes,
            spirv_names=names,
            spirv_decorations=decorations,
            spirv_member_decorations=member_decorations,
            spirv_member_names=member_names,
            spirv_capabilities=capabilities,
            spirv_types=types,
            spirv_constants=constants,
            spirv_constant_types=constant_types,
            spirv_spec_constant_ids=spec_constant_ids,
            spirv_extended_instruction_imports=extended_instruction_imports,
        )

    def spirv_assembly_functions(
        self,
        instructions,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        variables,
        entry_points,
        execution_modes,
        constants,
        constant_types,
        extended_instruction_imports,
    ):
        functions = []
        entry_points_by_id = {}
        for entry in entry_points:
            entry_points_by_id.setdefault(entry["id"], []).append(entry)
        self.spirv_module_execution_models = sorted(
            {entry["execution_model"] for entry in entry_points}
        )
        entry_instruction = None
        raw_instructions = []
        for result_id, opcode, operands, line_number in instructions:
            if opcode == "OpFunction":
                entry_instruction = (result_id, opcode, operands, line_number)
                raw_instructions = [entry_instruction]
                continue

            if entry_instruction is None:
                continue

            raw_instructions.append((result_id, opcode, operands, line_number))
            if opcode != "OpFunctionEnd":
                continue

            function_id, _opcode, function_operands, _start_line = entry_instruction
            module_names = names
            names = function_local_names(raw_instructions, module_names)
            return_type_id = function_operands[0] if function_operands else None
            function_control = (
                function_operands[1] if len(function_operands) >= 2 else None
            )
            function_type_id = (
                function_operands[2] if len(function_operands) >= 3 else None
            )
            function_type = types.get(function_type_id, {})
            parameter_records = [
                (raw_result_id, raw_operands[0] if raw_operands else None)
                for raw_result_id, raw_opcode, raw_operands, _raw_line_number in (
                    raw_instructions
                )
                if raw_opcode == "OpFunctionParameter"
            ]
            if not parameter_records:
                parameter_records = [
                    (None, parameter_type_id)
                    for parameter_type_id in function_type.get("parameter_types", [])
                ]
            written_pointer_parameter_ids = self.spirv_written_pointer_parameter_ids(
                parameter_records, raw_instructions, types
            )
            params = []
            for index, (parameter_id, parameter_type_id) in enumerate(
                parameter_records
            ):
                param = VariableNode(
                    self.spirv_function_parameter_type_name(
                        parameter_type_id, types, constants
                    ),
                    self.spirv_assembly_value_name(
                        parameter_id,
                        names,
                        decorations,
                        prefix=f"param{index}",
                    ),
                    qualifiers=self.spirv_function_parameter_qualifiers(
                        parameter_type_id,
                        types,
                        parameter_id in written_pointer_parameter_ids,
                    ),
                    spirv_id=parameter_id,
                    spirv_type_id=parameter_type_id,
                )
                params.append(param)
            node = FunctionNode(
                self.spirv_assembly_function_return_type_name(
                    return_type_id, types, constants
                ),
                module_names.get(
                    function_id,
                    self.spirv_function_name_fallback(function_id, entry_points_by_id),
                ),
                params,
                body=self.spirv_assembly_function_body(
                    raw_instructions,
                    names,
                    decorations,
                    member_decorations,
                    member_names,
                    types,
                    variables,
                    constants,
                    constant_types,
                    extended_instruction_imports,
                ),
            )
            function_entry_points = entry_points_by_id.get(function_id, [])
            node.spirv_id = function_id
            node.spirv_return_type_id = return_type_id
            node.spirv_function_control = function_control
            node.spirv_function_type_id = function_type_id
            node.spirv_entry_points = function_entry_points
            node.spirv_entry_point = (
                function_entry_points[0] if function_entry_points else None
            )
            node.spirv_execution_model = (
                function_entry_points[0]["execution_model"]
                if function_entry_points
                else None
            )
            node.spirv_execution_modes = execution_modes.get(function_id, [])
            node.spirv_workgroup_size = self.spirv_entry_workgroup_size(
                function_id, function_entry_points
            )
            node.spirv_names = names
            names = module_names
            node.spirv_constants = constants
            node.spirv_decorations = decorations
            node.spirv_extended_instruction_imports = extended_instruction_imports
            node.spirv_instructions = list(raw_instructions)
            node.spirv_raw_instructions = [
                {
                    "result_id": raw_result_id,
                    "opcode": raw_opcode,
                    "operands": raw_operands,
                    "line_number": raw_line_number,
                }
                for raw_result_id, raw_opcode, raw_operands, raw_line_number in (
                    raw_instructions
                )
            ]
            functions.append(node)
            entry_instruction = None
            raw_instructions = []
        return functions

    SPIRV_WORKGROUP_EXECUTION_MODELS = {
        "GLCompute",
        "MeshNV",
        "MeshEXT",
        "TaskNV",
        "TaskEXT",
    }

    def spirv_assembly_function_body(
        self,
        raw_instructions,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        variables,
        constants,
        constant_types,
        extended_instruction_imports,
    ):
        statements = []
        expressions = {}
        expression_type_ids = {}
        variables_by_id = {variable["id"]: variable for variable in variables}
        used_result_ids = self.spirv_assembly_used_result_ids(raw_instructions)
        structure = SpirvFunctionStructure(
            self, raw_instructions, names, decorations, types, variables, constants
        )
        previous_result_id = None

        def record_expression(result_id, result_type_id, expression):
            expressions[result_id] = self.spirv_maybe_materialize_assembly_expression(
                result_id,
                result_type_id,
                expression,
                statements,
                names,
                decorations,
                types,
                constants,
            )
            expression_type_ids[result_id] = result_type_id

        for result_id, opcode, operands, _line_number in raw_instructions:
            structure.settle(
                previous_result_id, expressions, expression_type_ids, statements
            )
            previous_result_id = result_id
            if result_id and opcode == "OpLabel":
                structure.block_started(result_id, statements)
                continue

            if result_id and opcode == "OpFunctionParameter":
                expressions[result_id] = VariableNode(
                    "",
                    self.spirv_assembly_value_name(result_id, names, decorations),
                    spirv_id=result_id,
                )
                if operands:
                    expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpVariable" and len(operands) >= 2:
                pointer_type = types.get(operands[0], {})
                storage_class = operands[1]
                variable_name = self.spirv_assembly_value_name(
                    result_id, names, decorations
                )
                expressions[result_id] = VariableNode(
                    "", variable_name, spirv_id=result_id
                )
                expression_type_ids[result_id] = operands[0]
                if storage_class == "Function":
                    variable_type, array_suffix = self.spirv_type_name_and_suffix(
                        pointer_type.get("type_id"),
                        types,
                        constants,
                        names=names,
                    )
                    variable_type = variable_type or pointer_type.get("type_id") or ""
                    declaration = VariableNode(
                        variable_type,
                        f"{variable_name}{array_suffix}",
                        spirv_id=result_id,
                        spirv_type_id=pointer_type.get("type_id"),
                    )
                    if len(operands) >= 3:
                        statements.append(
                            AssignmentNode(
                                declaration,
                                self.spirv_assembly_operand_expression(
                                    operands[2],
                                    expressions,
                                    names,
                                    decorations,
                                    constants,
                                ),
                            )
                        )
                    else:
                        statements.append(declaration)
                continue

            if result_id and opcode == "OpLoad" and len(operands) >= 2:
                expressions[result_id] = self.spirv_assembly_operand_expression(
                    operands[1],
                    expressions,
                    names,
                    decorations,
                    constants,
                    variables_by_id=variables_by_id,
                    types=types,
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImage" and len(operands) >= 2:
                record_expression(
                    result_id,
                    operands[0],
                    sampled_image_image(
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ),
                )
                continue

            if result_id and opcode == "OpSampledImage" and len(operands) >= 3:
                record_expression(
                    result_id,
                    operands[0],
                    sampled_image_call(
                        FunctionCallNode,
                        self.spirv_type_name(operands[0], types) or operands[0],
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        self.spirv_assembly_operand_expression(
                            operands[2],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                    ),
                )
                continue

            if result_id and opcode.startswith("OpImageSample") and len(operands) >= 3:
                record_expression(
                    result_id,
                    operands[0],
                    self.spirv_assembly_image_sample_expression(
                        opcode,
                        operands[1],
                        operands[2],
                        operands[3:],
                        expressions,
                        names,
                        decorations,
                        constants,
                        coordinate_components=self.spirv_sample_coordinate_components(
                            opcode,
                            operands[1],
                            operands[2],
                            expression_type_ids,
                            constant_types,
                            types,
                        ),
                    ),
                )
                continue

            if (
                result_id
                and opcode.startswith("OpImageSparseSample")
                and len(operands) >= 3
            ):
                expressions[result_id] = (
                    self.spirv_assembly_image_sparse_sample_expression(
                        opcode,
                        operands[1],
                        operands[2],
                        operands[3:],
                        expressions,
                        names,
                        decorations,
                        constants,
                        coordinate_components=self.spirv_sample_coordinate_components(
                            opcode,
                            operands[1],
                            operands[2],
                            expression_type_ids,
                            constant_types,
                            types,
                        ),
                    )
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImageSparseTexelsResident":
                if len(operands) >= 2:
                    expressions[result_id] = FunctionCallNode(
                        "spirvSparseTexelsResident",
                        [
                            self.spirv_assembly_operand_expression(
                                operands[1],
                                expressions,
                                names,
                                decorations,
                                constants,
                            )
                        ],
                    )
                    expression_type_ids[result_id] = operands[0]
                    continue

            if result_id and opcode == "OpImageSparseRead" and len(operands) >= 3:
                expressions[result_id] = (
                    self.spirv_assembly_image_sparse_read_expression(
                        operands[1],
                        operands[2],
                        operands[3:],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImageSparseFetch" and len(operands) >= 3:
                expressions[result_id] = (
                    self.spirv_assembly_image_sparse_fetch_expression(
                        operands[1],
                        operands[2],
                        operands[3:],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode in {
                "OpImageSparseGather",
                "OpImageSparseDrefGather",
            }:
                if len(operands) >= 4:
                    expressions[result_id] = (
                        self.spirv_assembly_image_sparse_gather_expression(
                            opcode,
                            operands[1],
                            operands[2],
                            operands[3],
                            operands[4:],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    )
                    expression_type_ids[result_id] = operands[0]
                    continue

            if result_id and opcode in {"OpImageGather", "OpImageDrefGather"}:
                if len(operands) >= 4:
                    expressions[result_id] = (
                        self.spirv_assembly_image_gather_expression(
                            opcode,
                            operands[1],
                            operands[2],
                            operands[3],
                            operands[4:],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    )
                    expression_type_ids[result_id] = operands[0]
                    continue

            if result_id and opcode == "OpImageFetch" and len(operands) >= 3:
                expressions[result_id] = self.spirv_assembly_image_fetch_expression(
                    operands[1],
                    operands[2],
                    operands[3:],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImageQuerySizeLod" and len(operands) >= 3:
                expressions[result_id] = FunctionCallNode(
                    "textureSize",
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        self.spirv_assembly_operand_expression(
                            operands[2],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImageQuerySize" and len(operands) >= 2:
                expressions[result_id] = FunctionCallNode(
                    self.spirv_image_query_name(
                        "textureSize",
                        "imageSize",
                        operands[1],
                        expression_type_ids,
                        types,
                    ),
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImageQueryLod" and len(operands) >= 3:
                expressions[result_id] = FunctionCallNode(
                    "textureQueryLod",
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        self.spirv_assembly_operand_expression(
                            operands[2],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImageQueryLevels" and len(operands) >= 2:
                expressions[result_id] = FunctionCallNode(
                    "textureQueryLevels",
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImageQuerySamples" and len(operands) >= 2:
                expressions[result_id] = FunctionCallNode(
                    self.spirv_image_query_name(
                        "textureSamples",
                        "imageSamples",
                        operands[1],
                        expression_type_ids,
                        types,
                    ),
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if (
                result_id
                and opcode in {
                    "OpImageQueryFormat",
                    "OpImageQueryOrder",
                }
                and len(operands) >= 2
            ):
                expressions[result_id] = FunctionCallNode(
                    {
                        "OpImageQueryFormat": "spirvImageQueryFormat",
                        "OpImageQueryOrder": "spirvImageQueryOrder",
                    }[opcode],
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if (
                result_id
                and opcode == "OpImageRead"
                and len(operands) >= 3
                and is_subpass_image(operands[1], expression_type_ids, types)
            ):
                parsed_operands = self.spirv_assembly_image_operands(
                    operands[3:], expressions, names, decorations, constants
                )
                expressions[result_id] = subpass_load_expression(
                    self.spirv_assembly_operand_expression(
                        operands[1], expressions, names, decorations, constants
                    ),
                    operands[2],
                    constants,
                    self.spirv_assembly_image_sample_operand(parsed_operands),
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImageRead" and len(operands) >= 3:
                expressions[result_id] = self.spirv_assembly_image_read_expression(
                    operands[1],
                    operands[2],
                    operands[3:],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpImageTexelPointer" and len(operands) >= 4:
                expressions[result_id] = (
                    self.spirv_assembly_image_texel_pointer_expression(
                        operands[1],
                        operands[2],
                        operands[3],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpAtomicLoad" and len(operands) >= 4:
                expressions[result_id] = self.spirv_assembly_atomic_load_expression(
                    operands[1],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if (
                result_id
                and opcode == "OpNamedBarrierInitialize"
                and len(operands) >= 2
            ):
                expressions[result_id] = FunctionCallNode(
                    "spirvNamedBarrierInitialize",
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if (
                result_id
                and (
                    opcode.startswith("OpConvert")
                    or opcode in {"OpFConvert", "OpSConvert", "OpUConvert"}
                )
                and len(operands) >= 2
            ):
                operand = self.spirv_assembly_operand_expression(
                    operands[1],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                expressions[result_id] = spirv_integer_conversion_expression(
                    opcode[2:],
                    operands[0],
                    operand,
                    expression_type_ids.get(operands[1])
                    or constant_types.get(operands[1]),
                    types,
                    self.spirv_vector_type_name,
                ) or FunctionCallNode(
                    self.spirv_type_name(operands[0], types) or operands[0],
                    [operand],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpBitcast" and len(operands) >= 2:
                expressions[result_id] = self.spirv_assembly_bitcast_expression(
                    operands[0],
                    operands[1],
                    expressions,
                    expression_type_ids,
                    names,
                    decorations,
                    constants,
                    constant_types,
                    types,
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if (
                result_id
                and opcode in {"OpCopyLogical", "OpCopyObject"}
                and len(operands) >= 2
            ):
                expressions[result_id] = self.spirv_assembly_operand_expression(
                    operands[1],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpUndef" and operands:
                expressions[result_id] = self.spirv_assembly_undef_expression(
                    operands[0], types
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpConstantNull" and operands:
                expressions[result_id] = self.spirv_assembly_null_expression(
                    operands[0], types
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpTranspose" and len(operands) >= 2:
                expressions[result_id] = FunctionCallNode(
                    "transpose",
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode in {"OpAccessChain", "OpInBoundsAccessChain"}:
                if len(operands) >= 2:
                    access = self.spirv_assembly_access_chain_expression(
                        operands[1],
                        operands[2:],
                        expressions,
                        names,
                        decorations,
                        member_decorations,
                        member_names,
                        types,
                        variables_by_id,
                        constants,
                        expression_type_ids=expression_type_ids,
                    )
                    expressions[result_id] = access
                    expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode in {
                "OpPtrAccessChain",
                "OpInBoundsPtrAccessChain",
            }:
                if len(operands) >= 3:
                    access = self.spirv_assembly_access_chain_expression(
                        operands[1],
                        operands[2:],
                        expressions,
                        names,
                        decorations,
                        member_decorations,
                        member_names,
                        types,
                        variables_by_id,
                        constants,
                    )
                    expressions[result_id] = access
                    expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpArrayLength" and len(operands) >= 3:
                expressions[result_id] = FunctionCallNode(
                    "spirvArrayLength",
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        operands[2],
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpCompositeConstruct" and operands:
                record_expression(
                    result_id,
                    operands[0],
                    self.spirv_assembly_composite_construct_expression(
                        operands[0],
                        operands[1:],
                        expressions,
                        names,
                        decorations,
                        constants,
                        types,
                    ),
                )
                continue

            if result_id and opcode == "OpCompositeExtract" and len(operands) >= 2:
                record_expression(
                    result_id,
                    operands[0],
                    self.spirv_assembly_composite_extract_expression(
                        operands[1],
                        operands[2:],
                        expressions,
                        expression_type_ids,
                        names,
                        decorations,
                        member_names,
                        types,
                        constants,
                        constant_types,
                    ),
                )
                continue

            if result_id and opcode == "OpVectorExtractDynamic" and len(operands) >= 3:
                expressions[result_id] = ArrayAccessNode(
                    self.spirv_assembly_operand_expression(
                        operands[1],
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                    self.spirv_assembly_operand_expression(
                        operands[2],
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if result_id and opcode == "OpVectorInsertDynamic" and len(operands) >= 4:
                record_expression(
                    result_id,
                    operands[0],
                    self.spirv_assembly_vector_insert_dynamic_expression(
                        operands[0],
                        operands[1],
                        operands[2],
                        operands[3],
                        expressions,
                        names,
                        decorations,
                        constants,
                        types,
                    ),
                )
                continue

            if result_id and opcode == "OpCompositeInsert" and len(operands) >= 3:
                record_expression(
                    result_id,
                    operands[0],
                    self.spirv_assembly_composite_insert_expression(
                        operands[0],
                        operands[1],
                        operands[2],
                        operands[3:],
                        expressions,
                        names,
                        decorations,
                        constants,
                        types,
                    ),
                )
                continue

            if result_id and opcode == "OpVectorShuffle" and len(operands) >= 4:
                record_expression(
                    result_id,
                    operands[0],
                    self.spirv_assembly_vector_shuffle_expression(
                        operands[0],
                        operands[1],
                        operands[2],
                        operands[3:],
                        expressions,
                        expression_type_ids,
                        names,
                        decorations,
                        constants,
                        types,
                    ),
                )
                continue

            if result_id and opcode == "OpExtInst" and len(operands) == 4:
                struct_value = spirv_struct_result_call(
                    self,
                    result_id,
                    operands[0],
                    extended_instruction_imports.get(operands[1]),
                    operands[2],
                    self.spirv_assembly_operand_expression(
                        operands[3], expressions, names, decorations, constants
                    ),
                    statements,
                    names,
                    decorations,
                    types,
                    constants,
                )
                if struct_value is not None:
                    expressions[result_id] = struct_value
                    expression_type_ids[result_id] = operands[0]
                    continue

            if result_id and opcode == "OpExtInst" and len(operands) >= 3:
                call = FunctionCallNode(
                    self.spirv_ext_inst_function_name(
                        extended_instruction_imports.get(operands[1]),
                        operands[2],
                    ),
                    [
                        self.spirv_assembly_operand_expression(
                            operand,
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                        for operand in operands[3:]
                    ],
                )
                if (
                    extended_instruction_imports.get(operands[1]) == "GLSL.std.450"
                    and self.SPIRV_GLSL_STD_450_EXT_INST_IDS.get(
                        str(operands[2]), operands[2]
                    )
                    == "NClamp"
                    and len(call.args) == 3
                ):
                    # NClamp(x, lo, hi) is NMin(NMax(x, lo), hi).
                    call = FunctionCallNode(
                        "fmin",
                        [FunctionCallNode("fmax", call.args[:2]), call.args[2]],
                    )
                if self.spirv_type_name(operands[0], types) == "void":
                    statements.append(call)
                else:
                    record_expression(result_id, operands[0], call)
                continue

            if result_id and opcode in {
                "OpAtomicIIncrement",
                "OpAtomicIDecrement",
            }:
                if len(operands) >= 4:
                    expressions[result_id] = fold_image_texel_atomic(
                        self.spirv_assembly_atomic_increment_expression(
                            opcode,
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        opcode,
                        self.spirv_type_name(operands[0], types),
                    )
                    expression_type_ids[result_id] = operands[0]
                    if result_id not in used_result_ids:
                        statements.append(expressions[result_id])
                    continue

            if result_id and opcode in self.SPIRV_ATOMIC_RMW_FUNCTIONS:
                if len(operands) >= 5:
                    expressions[result_id] = fold_image_texel_atomic(
                        self.spirv_assembly_atomic_expression(
                            opcode,
                            operands[1],
                            operands[4],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        opcode,
                        self.spirv_type_name(operands[0], types),
                    )
                    expression_type_ids[result_id] = operands[0]
                    if result_id not in used_result_ids:
                        statements.append(expressions[result_id])
                    continue

            if result_id and opcode == "OpReportIntersectionKHR":
                if len(operands) < 3:
                    raise SpirvInstructionUnsupportedError(
                        opcode, f"needs 3 operands, got {len(operands)}"
                    )
                expressions[result_id] = FunctionCallNode(
                    "ReportHit",
                    [
                        self.spirv_assembly_operand_expression(
                            operand, expressions, names, decorations, constants
                        )
                        for operand in operands[1:3]
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                if result_id not in used_result_ids:
                    # A discarded result still reports the hit.
                    statements.append(expressions[result_id])
                continue

            if result_id and opcode in {
                "OpAtomicCompareExchange",
                "OpAtomicCompareExchangeWeak",
            }:
                if len(operands) >= 7:
                    expressions[result_id] = fold_image_texel_atomic(
                        self.spirv_assembly_atomic_compare_exchange_expression(
                            operands[1],
                            operands[5],
                            operands[6],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        opcode,
                        self.spirv_type_name(operands[0], types),
                    )
                    expression_type_ids[result_id] = operands[0]
                    if result_id not in used_result_ids:
                        statements.append(expressions[result_id])
                    continue

            if result_id and opcode == "OpGroupNonUniformBroadcastFirst":
                if len(operands) >= 3:
                    expressions[result_id] = FunctionCallNode(
                        "subgroupBroadcastFirst",
                        [
                            self.spirv_assembly_operand_expression(
                                operands[2],
                                expressions,
                                names,
                                decorations,
                                constants,
                            )
                        ],
                    )
                    expression_type_ids[result_id] = operands[0]
                    continue

            if result_id and opcode == "OpGroupNonUniformBroadcast":
                if len(operands) >= 4:
                    expressions[result_id] = FunctionCallNode(
                        "subgroupBroadcast",
                        [
                            self.spirv_assembly_operand_expression(
                                operands[2],
                                expressions,
                                names,
                                decorations,
                                constants,
                            ),
                            self.spirv_assembly_operand_expression(
                                operands[3],
                                expressions,
                                names,
                                decorations,
                                constants,
                            ),
                        ],
                    )
                    expression_type_ids[result_id] = operands[0]
                    continue

            if result_id and opcode in self.SPIRV_GROUP_NON_UNIFORM_REDUCTION_FUNCTIONS:
                if len(operands) >= 4:
                    expressions[result_id] = (
                        self.spirv_assembly_group_non_uniform_reduction_expression(
                            opcode,
                            operands[2],
                            operands[3],
                            operands[4:],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    )
                    expression_type_ids[result_id] = operands[0]
                    continue

            if result_id and opcode == "OpDot" and len(operands) >= 3:
                expressions[result_id] = FunctionCallNode(
                    "dot",
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        self.spirv_assembly_operand_expression(
                            operands[2],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                continue

            if (
                result_id
                and opcode in self.SPIRV_RAY_QUERY_EXPRESSION_FUNCTIONS
                and len(operands) >= 2
            ):
                record_expression(
                    result_id,
                    operands[0],
                    self.spirv_assembly_ray_query_expression(
                        opcode,
                        operands,
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                )
                continue

            if result_id and opcode == "OpSelect" and len(operands) >= 4:
                condition, true_value, false_value = (
                    self.spirv_assembly_operand_expression(
                        operand,
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                    for operand in operands[1:4]
                )
                record_expression(
                    result_id,
                    operands[0],
                    spirv_select_expression(
                        condition,
                        true_value,
                        false_value,
                        expression_type_ids.get(operands[1])
                        or constant_types.get(operands[1]),
                        types,
                    ),
                )
                continue

            if result_id and opcode == "OpPhi" and len(operands) >= 3:
                structure.import_phi(
                    result_id,
                    operands,
                    expressions,
                    expression_type_ids,
                    record_expression,
                )
                continue

            operation = opcode[2:] if opcode.startswith("Op") else opcode
            if result_id and len(operands) >= 3:
                struct_value = spirv_extended_arithmetic_call(
                    self,
                    result_id,
                    operands[0],
                    operation,
                    [
                        self.spirv_assembly_operand_expression(
                            operand,
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                        for operand in operands[1:]
                    ],
                    statements,
                    names,
                    decorations,
                    types,
                )
                if struct_value is not None:
                    expressions[result_id] = struct_value
                    expression_type_ids[result_id] = operands[0]
                    continue

            if (
                result_id
                and operation in self.SPIRV_CORE_FUNCTIONS
                and len(operands) >= 2
            ):
                record_expression(
                    result_id,
                    operands[0],
                    FunctionCallNode(
                        self.SPIRV_CORE_FUNCTIONS[operation],
                        [
                            self.spirv_assembly_operand_expression(
                                operand,
                                expressions,
                                names,
                                decorations,
                                constants,
                            )
                            for operand in operands[1:]
                        ],
                    ),
                )
                continue

            if (
                result_id
                and operation in self.SPIRV_DERIVATIVE_FUNCTIONS
                and len(operands) >= 2
            ):
                record_expression(
                    result_id,
                    operands[0],
                    FunctionCallNode(
                        self.SPIRV_DERIVATIVE_FUNCTIONS[operation],
                        [
                            self.spirv_assembly_operand_expression(
                                operands[1],
                                expressions,
                                names,
                                decorations,
                                constants,
                            )
                        ],
                    ),
                )
                continue

            if (
                result_id
                and operation in self.SPIRV_UNARY_FUNCTIONS
                and len(operands) >= 2
            ):
                record_expression(
                    result_id,
                    operands[0],
                    FunctionCallNode(
                        self.SPIRV_UNARY_FUNCTIONS[operation],
                        [
                            self.spirv_assembly_operand_expression(
                                operands[1],
                                expressions,
                                names,
                                decorations,
                                constants,
                            )
                        ],
                    ),
                )
                continue

            if result_id and operation == "FMod" and len(operands) >= 3:
                # OpFMod takes the sign of the divisor (x - y * floor(x / y)), which is
                # CrossGL mod; the % operator is the truncated remainder of OpFRem.
                record_expression(
                    result_id,
                    operands[0],
                    FunctionCallNode(
                        "mod",
                        [
                            self.spirv_assembly_operand_expression(
                                operand, expressions, names, decorations, constants
                            )
                            for operand in operands[1:3]
                        ],
                    ),
                )
                continue

            if (
                result_id
                and operation in self.SPIRV_SPEC_CONSTANT_BINARY_OPS
                and len(operands) >= 3
            ):
                lhs, rhs = (
                    self.spirv_assembly_operand_expression(
                        operand,
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                    for operand in operands[1:3]
                )
                expression = spirv_integer_binary_expression(
                    operation,
                    operands[0],
                    lhs,
                    expression_type_ids.get(operands[1])
                    or constant_types.get(operands[1]),
                    rhs,
                    expression_type_ids.get(operands[2])
                    or constant_types.get(operands[2]),
                    types,
                    self.spirv_vector_type_name,
                )
                if expression is None:
                    expression = BinaryOpNode(
                        lhs, self.SPIRV_SPEC_CONSTANT_BINARY_OPS[operation], rhs
                    )
                record_expression(result_id, operands[0], expression)
                continue

            if (
                result_id
                and operation in self.SPIRV_SPEC_CONSTANT_UNARY_OPS
                and len(operands) >= 2
            ):
                record_expression(
                    result_id,
                    operands[0],
                    UnaryOpNode(
                        self.SPIRV_SPEC_CONSTANT_UNARY_OPS[operation],
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                    ),
                )
                continue

            if result_id and opcode == "OpFunctionCall" and len(operands) >= 2:
                call = FunctionCallNode(
                    self.spirv_assembly_value_name(
                        operands[1], names, decorations, prefix="function"
                    ),
                    [
                        self.spirv_assembly_operand_expression(
                            operand,
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                        for operand in operands[2:]
                    ],
                )
                if self.spirv_type_name(operands[0], types) == "void":
                    statements.append(call)
                elif result_id not in used_result_ids:
                    # The call's effects happen even when its result is discarded.
                    statements.append(call)
                else:
                    expressions[result_id] = call
                    expression_type_ids[result_id] = operands[0]
                continue

            if opcode == "OpAtomicStore" and len(operands) >= 4:
                statements.append(
                    self.spirv_assembly_atomic_store_statement(
                        operands[0],
                        operands[3],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
                continue

            if opcode == "OpImageWrite" and len(operands) >= 3:
                statements.append(
                    self.spirv_assembly_image_write_statement(
                        operands[0],
                        operands[1],
                        operands[2],
                        operands[3:],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
                continue

            if opcode in self.SPIRV_GEOMETRY_EMIT_FUNCTIONS:
                statements.append(
                    self.spirv_assembly_geometry_emit_statement(
                        opcode,
                        operands,
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
                continue

            if opcode in self.SPIRV_RAY_QUERY_STATEMENT_FUNCTIONS:
                statements.append(
                    self.spirv_assembly_ray_query_statement(
                        opcode,
                        operands,
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
                continue

            if opcode == "OpControlBarrier" and len(operands) >= 3:
                statements.extend(
                    spirv_barrier_statements(
                        opcode,
                        [constants.get(operand) for operand in operands[:3]],
                        self.spirv_module_execution_models,
                    )
                )
                continue

            if opcode == "OpMemoryBarrier" and len(operands) >= 2:
                statements.extend(
                    spirv_barrier_statements(
                        opcode,
                        [constants.get(operand) for operand in operands[:2]],
                        self.spirv_module_execution_models,
                    )
                )
                continue

            if opcode == "OpMemoryNamedBarrier" and len(operands) >= 3:
                statements.append(
                    FunctionCallNode(
                        "spirvMemoryNamedBarrier",
                        [
                            self.spirv_assembly_operand_expression(
                                operand,
                                expressions,
                                names,
                                decorations,
                                constants,
                            )
                            for operand in operands[:3]
                        ],
                    )
                )
                continue

            if opcode == "OpStore" and len(operands) >= 2:
                statements.append(
                    AssignmentNode(
                        self.spirv_assembly_operand_expression(
                            operands[0],
                            expressions,
                            names,
                            decorations,
                            constants,
                            variables_by_id=variables_by_id,
                            types=types,
                        ),
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                            variables_by_id=variables_by_id,
                            types=types,
                        ),
                    )
                )
                continue

            if opcode == "OpCopyMemory" and len(operands) >= 2:
                if operands[0] != operands[1]:
                    statements.append(
                        AssignmentNode(
                            self.spirv_assembly_operand_expression(
                                operands[0],
                                expressions,
                                names,
                                decorations,
                                constants,
                            ),
                            self.spirv_assembly_operand_expression(
                                operands[1],
                                expressions,
                                names,
                                decorations,
                                constants,
                            ),
                        )
                    )
                continue

            if opcode == "OpCopyMemorySized" and len(operands) >= 3:
                statements.append(
                    self.spirv_assembly_copy_memory_sized_statement(
                        operands[0],
                        operands[1],
                        operands[2],
                        operands[3:],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
                continue

            if opcode in {
                "OpKill",
                "OpTerminateInvocation",
                "OpDemoteToHelperInvocation",
            }:
                statements.append(DiscardNode())
                continue

            if opcode == "OpReturnValue" and operands:
                statements.append(
                    ReturnNode(
                        self.spirv_assembly_operand_expression(
                            operands[0],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    )
                )
                continue

            if opcode == "OpReturn":
                statements.append(ReturnNode())
                continue

            ray_call = ray_statement(
                opcode,
                operands,
                lambda operand: self.spirv_assembly_operand_expression(
                    operand, expressions, names, decorations, constants
                ),
                ray_variable_locations(variables, decorations),
            )
            if ray_call is not None:
                statements.append(ray_call)
                continue
            refuse_untranslated_statement(result_id, opcode)

        structure.settle(
            previous_result_id, expressions, expression_type_ids, statements
        )
        return structure.structured_body(statements, expressions)

    def spirv_assembly_composite_extract_expression(
        self,
        composite_operand,
        index_operands,
        expressions,
        expression_type_ids,
        names,
        decorations,
        member_names,
        types,
        constants,
        constant_types,
    ):
        expression = self.spirv_assembly_operand_expression(
            composite_operand,
            expressions,
            names,
            decorations,
            constants,
        )
        current_type_id = expression_type_ids.get(
            composite_operand
        ) or constant_types.get(composite_operand)

        for index_operand in index_operands:
            type_info = types.get(current_type_id, {})
            if type_info.get("kind") == "struct":
                member_index = self.spirv_integer_constant_operand(index_operand, {})
                member_types = type_info.get("member_types", [])
                if member_index is not None and 0 <= member_index < len(member_types):
                    sparse_extract = spirv_struct_result_member(
                        expression, member_index
                    )
                    if sparse_extract is None:
                        sparse_extract = (
                            self.spirv_sparse_image_result_extract_expression(
                                expression, member_index
                            )
                        )
                    if sparse_extract is not None:
                        expression = sparse_extract
                        current_type_id = member_types[member_index]
                        continue

                    member_key = str(member_index)
                    member_name = member_names.get(current_type_id, {}).get(member_key)
                    if member_name is not None:
                        expression = MemberAccessNode(expression, member_name)
                        current_type_id = member_types[member_index]
                        continue

            expression = ArrayAccessNode(expression, index_operand)
            current_type_id = self.spirv_composite_index_type_id(
                current_type_id, index_operand, types
            )

        return expression

    def spirv_function_name_fallback(self, function_id, entry_points_by_id):
        if function_id in entry_points_by_id:
            return entry_points_by_id[function_id][0].get("name")
        if function_id:
            return self.spirv_fallback_identifier(function_id, "function")
        return ""
