"""Parser for Vulkan SPIR-V source AST construction."""

import re
import shlex
import struct

from ..common_ast import InitializerListNode
from .access_chains import chain_base_pointee, walk_indices
from .barrier_import import spirv_barrier_statements
from .block_arrays import SpirvBlockArrayUnsupportedError, block_array_element
from .block_layout_import import spirv_block_layout
from .function_structure import SpirvFunctionStructure
from .integer_signedness_import import (
    spirv_integer_binary_expression,
    spirv_integer_conversion_expression,
)
from .local_names import function_local_names
from .ray_tracing_import import (
    BUILTIN_VARIABLE_NAMES,
    SpirvInstructionUnsupportedError,
    ray_statement,
    ray_variable_locations,
    refuse_untranslated_statement,
)
from .select_import import spirv_select_expression
from .struct_result_import import (
    spirv_extended_arithmetic_call,
    spirv_struct_result_call,
    spirv_struct_result_member,
)
from .VulkanAst import *
from .VulkanLexer import *


class VulkanParser:
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
        "InstanceIndex": "gl_InstanceID",
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

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

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
            raise SyntaxError("Unterminated HLSL attribute")

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
            raise SyntaxError("Unterminated HLSL attribute")

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
            raise SyntaxError("Unterminated parenthesized semantic")
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
            raise SyntaxError("Unterminated brace block")

    def parse(self):
        module = self.parse_module()
        self.eat("EOF")
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

        for result_id, opcode, operands, _line_number in instructions:
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
        self.spirv_register_struct_type_names(types, names, member_names)
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
                spec_constant_ids, names, decorations, types, constants, constant_types
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
            raise SyntaxError(SPIRV_ASSEMBLY_ERROR)

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

    def spirv_apply_decoration_groups(
        self, decorations, member_decorations, group_decorates, group_member_decorates
    ):
        for group_id, targets in group_decorates:
            group_decorations = decorations.get(group_id, [])
            if not group_decorations:
                continue
            for target in targets:
                decorations.setdefault(target, []).extend(
                    (decoration, list(operands))
                    for decoration, operands in group_decorations
                )

        for group_id, target_members in group_member_decorates:
            group_decorations = decorations.get(group_id, [])
            if not group_decorations:
                continue
            for index in range(0, len(target_members) - 1, 2):
                target = target_members[index]
                member = target_members[index + 1]
                member_decorations.setdefault(target, []).extend(
                    (member, decoration, list(operands))
                    for decoration, operands in group_decorations
                )

    def spirv_assembly_has_metadata_only_surface(self, types):
        metadata_type_kinds = {
            "acceleration_structure",
            "array",
            "cooperative_matrix",
            "image",
            "named_barrier",
            "pointer",
            "ray_query",
            "runtime_array",
            "sampled_image",
            "sampler",
        }
        return any(
            type_info.get("kind") in metadata_type_kinds for type_info in types.values()
        )

    def spirv_assembly_has_linkage_metadata_surface(
        self, capabilities, types, extended_instruction_imports
    ):
        if "Linkage" not in capabilities:
            return False
        return bool(types or extended_instruction_imports)

    def spirv_assembly_has_entry_point_metadata_surface(self, entry_points):
        return bool(entry_points)

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
                    self.spirv_assembly_operand_expression(
                        operands[1],
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                )
                continue

            if result_id and opcode == "OpSampledImage" and len(operands) >= 3:
                record_expression(
                    result_id,
                    operands[0],
                    FunctionCallNode(
                        self.spirv_type_name(operands[0], types) or operands[0],
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
                    expressions[result_id] = (
                        self.spirv_assembly_atomic_increment_expression(
                            opcode,
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    )
                    expression_type_ids[result_id] = operands[0]
                    if result_id not in used_result_ids:
                        statements.append(expressions[result_id])
                    continue

            if result_id and opcode in self.SPIRV_ATOMIC_RMW_FUNCTIONS:
                if len(operands) >= 5:
                    expressions[result_id] = self.spirv_assembly_atomic_expression(
                        opcode,
                        operands[1],
                        operands[4],
                        expressions,
                        names,
                        decorations,
                        constants,
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
                    expressions[result_id] = (
                        self.spirv_assembly_atomic_compare_exchange_expression(
                            operands[1],
                            operands[5],
                            operands[6],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
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

    def spirv_assembly_used_result_ids(self, raw_instructions):
        return {
            operand
            for _result_id, _opcode, operands, _line_number in raw_instructions
            for operand in operands
            if isinstance(operand, str) and operand.startswith("%")
        }

    def spirv_maybe_materialize_assembly_expression(
        self,
        result_id,
        result_type_id,
        expression,
        statements,
        names,
        decorations,
        types,
        constants,
    ):
        if not result_id or not result_type_id:
            return expression
        if (
            self.spirv_expression_complexity(
                expression, self.MAX_SPIRV_MATERIALIZED_EXPRESSION_COMPLEXITY
            )
            <= self.MAX_SPIRV_MATERIALIZED_EXPRESSION_COMPLEXITY
        ):
            return expression

        type_info = types.get(result_type_id, {})
        if type_info.get("kind") in {"pointer", "function"}:
            return expression
        variable_type, array_suffix = self.spirv_type_name_and_suffix(
            result_type_id, types, constants, names=names
        )
        if not variable_type or variable_type == "void":
            return expression

        variable_name = self.spirv_assembly_value_name(
            result_id, names, decorations, prefix="value"
        )
        statements.append(
            AssignmentNode(
                VariableNode(
                    variable_type,
                    f"{variable_name}{array_suffix}",
                    spirv_id=result_id,
                    spirv_type_id=result_type_id,
                ),
                expression,
            )
        )
        return VariableNode("", variable_name, spirv_id=result_id)

    def spirv_expression_complexity(self, expression, limit):
        budget = limit + 1

        def count(node):
            nonlocal budget
            if budget <= 0:
                return 0
            budget -= 1
            if node is None or isinstance(node, (str, int, float, VariableNode)):
                return 1
            if isinstance(node, AssignmentNode):
                return 1 + count(node.left) + count(node.right)
            if isinstance(node, BinaryOpNode):
                return 1 + count(node.left) + count(node.right)
            if isinstance(node, UnaryOpNode):
                return 1 + count(node.operand)
            if isinstance(node, TernaryOpNode):
                return (
                    1
                    + count(node.condition)
                    + count(node.true_expr)
                    + count(node.false_expr)
                )
            if isinstance(node, FunctionCallNode):
                return 1 + sum(count(arg) for arg in node.args)
            if isinstance(node, MethodCallNode):
                return 1 + count(node.object) + sum(count(arg) for arg in node.args)
            if isinstance(node, MemberAccessNode):
                return 1 + count(node.object)
            if isinstance(node, ArrayAccessNode):
                return 1 + count(node.array) + count(node.index)
            if isinstance(node, InitializerListNode):
                return 1 + sum(count(element) for element in node.elements)
            return 1

        return count(expression)

    def spirv_assembly_access_chain_expression(
        self,
        base_operand,
        index_operands,
        expressions,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        variables_by_id,
        constants,
        expression_type_ids=None,
    ):
        struct_member = self.spirv_assembly_struct_member_access_expression(
            base_operand,
            index_operands,
            expressions,
            names,
            decorations,
            member_decorations,
            member_names,
            types,
            variables_by_id,
            constants,
        )
        if struct_member is not None:
            return struct_member

        access = self.spirv_assembly_operand_expression(
            base_operand,
            expressions,
            names,
            decorations,
            constants,
            variables_by_id=variables_by_id,
            types=types,
        )
        pointee = (
            chain_base_pointee(
                base_operand, types, variables_by_id, expression_type_ids
            )
            if expression_type_ids is not None
            else None
        )
        return walk_indices(
            self,
            access,
            pointee,
            index_operands,
            lambda operand: self.spirv_assembly_operand_expression(
                operand,
                expressions,
                names,
                decorations,
                constants,
                variables_by_id=variables_by_id,
                types=types,
            ),
            member_names,
            types,
            constants,
        )

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

    def spirv_assembly_composite_construct_expression(
        self,
        type_id,
        constituent_ids,
        expressions,
        names,
        decorations,
        constants,
        types,
    ):
        args = [
            self.spirv_assembly_operand_expression(
                operand,
                expressions,
                names,
                decorations,
                constants,
            )
            for operand in constituent_ids
        ]
        if types.get(type_id, {}).get("kind") in {"array", "runtime_array"}:
            return InitializerListNode(args)
        return FunctionCallNode(self.spirv_type_name(type_id, types) or type_id, args)

    def spirv_sparse_image_result_extract_expression(self, expression, member_index):
        if not (
            isinstance(expression, FunctionCallNode)
            and expression.name.startswith(
                (
                    "spirvImageSparseSample",
                    "spirvImageSparseRead",
                    "spirvImageSparseFetch",
                    "spirvImageSparseGather",
                    "spirvImageSparseDrefGather",
                )
            )
        ):
            return None

        if member_index == 0:
            return FunctionCallNode("spirvSparseResidencyCode", [expression])
        if member_index == 1:
            return FunctionCallNode("spirvSparseTexel", [expression])
        return None

    def spirv_composite_index_type_id(self, type_id, index_operand, types):
        type_info = types.get(type_id, {})
        kind = type_info.get("kind")
        if kind in {"array", "runtime_array"}:
            return type_info.get("element_type")
        if kind == "matrix":
            return type_info.get("column_type")
        if kind == "vector":
            return type_info.get("component_type")
        if kind == "struct":
            member_index = self.spirv_integer_constant_operand(index_operand, {})
            member_types = type_info.get("member_types", [])
            if member_index is not None and 0 <= member_index < len(member_types):
                return member_types[member_index]
        return None

    def spirv_assembly_typed_access_chain(
        self,
        base_operand,
        type_id,
        index_operands,
        expressions,
        names,
        decorations,
        member_names,
        types,
        constants,
    ):
        """``base.member[i].field``: an access chain walked through its pointee type, struct indices as names."""
        access = self.spirv_assembly_operand_expression(
            base_operand, expressions, names, decorations, constants
        )
        return walk_indices(
            self,
            access,
            type_id,
            index_operands,
            lambda operand: self.spirv_assembly_operand_expression(
                operand, expressions, names, decorations, constants
            ),
            member_names,
            types,
            constants,
        )

    def spirv_assembly_member_tail(
        self,
        access,
        struct_type_id,
        index_operands,
        expressions,
        names,
        decorations,
        member_names,
        types,
        constants,
    ):
        """Continue ``access`` (member ``index_operands[0]`` of ``struct_type_id``) through the remaining indices."""
        member_index = self.spirv_integer_constant_operand(index_operands[0], constants)
        member_type_id = types.get(struct_type_id, {}).get("member_types", [])[
            member_index
        ]
        return walk_indices(
            self,
            access,
            member_type_id,
            index_operands[1:],
            lambda operand: self.spirv_assembly_operand_expression(
                operand, expressions, names, decorations, constants
            ),
            member_names,
            types,
            constants,
        )

    def spirv_assembly_storage_block_array_access(
        self,
        base_operand,
        struct_type_id,
        index_operands,
        expressions,
        names,
        decorations,
        member_names,
        types,
        constants,
    ):
        """``v[k][0].m...`` for an access chain into element ``k`` of an array of storage blocks."""
        member_index = self.spirv_integer_constant_operand(index_operands[1], constants)
        member_types = types.get(struct_type_id, {}).get("member_types", [])
        if member_index is None or not 0 <= member_index < len(member_types):
            return None

        def operand(value):
            return self.spirv_assembly_operand_expression(
                value, expressions, names, decorations, constants
            )

        element = ArrayAccessNode(operand(base_operand), operand(index_operands[0]))
        member_name = member_names.get(struct_type_id, {}).get(
            str(member_index), f"member{member_index}"
        )
        access = MemberAccessNode(ArrayAccessNode(element, 0), member_name)
        return self.spirv_assembly_member_tail(
            access,
            struct_type_id,
            index_operands[1:],
            expressions,
            names,
            decorations,
            member_names,
            types,
            constants,
        )

    def spirv_assembly_struct_member_access_expression(
        self,
        base_operand,
        index_operands,
        expressions,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        variables_by_id,
        constants,
    ):
        if not index_operands:
            return None

        variable = variables_by_id.get(base_operand)
        if variable is None:
            return None

        pointer_type = types.get(variable["pointer_type_id"], {})
        if pointer_type.get("kind") != "pointer":
            return None

        storage_class = variable["storage_class"] or pointer_type.get("storage_class")
        struct_type_id = pointer_type.get("type_id")
        struct_type = types.get(struct_type_id, {})
        if (
            storage_class in self.SPIRV_INTERFACE_STORAGE_CLASSES
            and struct_type.get("kind") in {"array", "runtime_array"}
            and types.get(struct_type.get("element_type"), {}).get("kind") == "struct"
            and len(index_operands) >= 2
        ):
            # Per-vertex interface arrays (gl_in, gl_MeshVerticesEXT, mesh outputs): element, then member names.
            return self.spirv_assembly_typed_access_chain(
                base_operand,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )
        array_element = block_array_element(struct_type_id, types)
        if (
            array_element is not None
            and len(index_operands) >= 2
            and self.spirv_is_structured_buffer_block_access(
                storage_class, array_element[0], decorations
            )
        ):
            return self.spirv_assembly_storage_block_array_access(
                base_operand,
                array_element[0],
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )
        if struct_type.get("kind") != "struct":
            return None

        member_index = self.spirv_integer_constant_operand(index_operands[0], constants)
        if member_index is None:
            return None

        member_types = struct_type.get("member_types", [])
        if not 0 <= member_index < len(member_types):
            return None

        member_key = str(member_index)
        if storage_class in self.SPIRV_RAY_STORAGE_QUALIFIERS:
            return self.spirv_assembly_typed_access_chain(
                base_operand,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )
        if storage_class == "Function":
            member_name = member_names.get(struct_type_id, {}).get(
                member_key, f"member_{member_key}"
            )
            access = MemberAccessNode(
                self.spirv_assembly_operand_expression(
                    base_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                member_name,
            )
            return self.spirv_assembly_member_tail(
                access,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )

        if self.spirv_is_flattened_resource_block_access(
            storage_class, struct_type_id, decorations
        ):
            member_name = variable.get("flattened_member_aliases", {}).get(
                member_key
            ) or member_names.get(struct_type_id, {}).get(
                member_key, f"member{member_key}"
            )
            access = VariableNode("", member_name)
            return self.spirv_assembly_member_tail(
                access,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )

        if self.spirv_is_structured_buffer_block_access(
            storage_class, struct_type_id, decorations
        ):
            member_name = member_names.get(struct_type_id, {}).get(
                member_key, f"member{member_key}"
            )
            access = MemberAccessNode(
                ArrayAccessNode(
                    self.spirv_assembly_operand_expression(
                        base_operand,
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                    0,
                ),
                member_name,
            )
            return self.spirv_assembly_member_tail(
                access,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )

        if storage_class not in self.SPIRV_INTERFACE_STORAGE_CLASSES:
            return None

        member_layout_decorations = [
            (decoration, operands)
            for member, decoration, operands in member_decorations.get(
                struct_type_id, []
            )
            if member == member_key
        ]
        qualifiers = self.spirv_layout_qualifiers(member_layout_decorations)
        block_name = self.spirv_assembly_value_name(base_operand, names, decorations={})
        member_name = self.spirv_struct_member_variable_name(
            struct_type_id,
            member_key,
            block_name,
            qualifiers,
            member_names,
            storage_class=storage_class,
        )
        access = VariableNode("", member_name)
        return self.spirv_assembly_member_tail(
            access,
            struct_type_id,
            index_operands,
            expressions,
            names,
            decorations,
            member_names,
            types,
            constants,
        )

    def spirv_sample_coordinate_components(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        expression_type_ids,
        constant_types,
        types,
    ):
        """The coordinate components a sample consumes, when its coordinate operand has more; else ``None``.

        A SPIR-V sample coordinate holds ``(u[, v][, w][, array layer])`` as the image needs, then ``q`` for a
        projective sample; further components are ignored, and glslang passes, for instance, the whole ``vec3`` of
        ``texture(sampler2DShadow, P)`` with its reference. CrossGL coordinates hold exactly the consumed
        components.
        """
        sampled_type = types.get(expression_type_ids.get(image_operand), {})
        image_type = types.get(sampled_type.get("image_type"), {})
        axes = {"1D": 1, "2D": 2, "Rect": 2, "3D": 3, "Cube": 3}.get(
            image_type.get("dim")
        )
        if sampled_type.get("kind") != "sampled_image" or axes is None:
            return None
        needed = axes + (1 if str(image_type.get("arrayed")) == "1" else 0)
        if "Proj" in opcode:
            needed += 1
        coordinate_type = expression_type_ids.get(coordinate_operand)
        if coordinate_type is None:
            coordinate_type = constant_types.get(coordinate_operand)
        available = self.spirv_vector_component_count(coordinate_type, types)
        if available is None or available <= needed:
            return None
        return needed

    def spirv_assembly_image_sample_expression(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
        coordinate_components=None,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        if coordinate_components is not None:
            coordinate = MemberAccessNode(coordinate, "xyzw"[:coordinate_components])
        dref = None
        if "Dref" in opcode and image_operands:
            dref = self.spirv_assembly_operand_expression(
                image_operands[0], expressions, names, decorations, constants
            )
            image_operands = image_operands[1:]
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        base_args = [image, coordinate]
        if dref is not None:
            base_args.append(dref)
        offset = self.spirv_assembly_image_offset_operand(parsed_operands)
        bias = self.spirv_assembly_image_bias_operand(parsed_operands)
        min_lod = self.spirv_assembly_image_min_lod_operand(parsed_operands)
        is_projected = "Proj" in opcode

        if min_lod is not None:
            return self.spirv_assembly_image_min_lod_sample_expression(
                opcode,
                base_args,
                parsed_operands,
                offset,
                bias,
                min_lod,
            )

        if "Grad" in parsed_operands and len(parsed_operands["Grad"]) >= 2:
            if dref is not None and is_projected and offset is not None:
                function_name = "textureCompareProjGradOffset"
            elif dref is not None and is_projected:
                function_name = "textureCompareProjGrad"
            elif dref is not None and offset is not None:
                function_name = "textureCompareGradOffset"
            elif dref is not None:
                function_name = "textureCompareGrad"
            elif is_projected and offset is not None:
                function_name = "textureProjGradOffset"
            elif is_projected:
                function_name = "textureProjGrad"
            elif offset is not None:
                function_name = "textureGradOffset"
            else:
                function_name = "textureGrad"
            args = [*base_args, *parsed_operands["Grad"][:2]]
            if offset is not None:
                args.append(offset)
            return FunctionCallNode(
                function_name,
                args,
            )

        if "Lod" in parsed_operands and parsed_operands["Lod"]:
            if dref is not None and is_projected and offset is not None:
                function_name = "textureCompareProjLodOffset"
            elif dref is not None and is_projected:
                function_name = "textureCompareProjLod"
            elif dref is not None and offset is not None:
                function_name = "textureCompareLodOffset"
            elif dref is not None:
                function_name = "textureCompareLod"
            elif is_projected and offset is not None:
                function_name = "textureProjLodOffset"
            elif is_projected:
                function_name = "textureProjLod"
            elif offset is not None:
                function_name = "textureLodOffset"
            else:
                function_name = "textureLod"
            args = [*base_args, parsed_operands["Lod"][0]]
            if offset is not None:
                args.append(offset)
            return FunctionCallNode(
                function_name,
                args,
            )

        if offset is not None:
            args = [*base_args, offset]
            if bias is not None:
                args.append(bias)
            if dref is not None and bias is None and is_projected:
                function_name = "textureCompareProjOffset"
            elif dref is not None and bias is None:
                function_name = "textureCompareOffset"
            else:
                function_name = "textureProjOffset" if is_projected else "textureOffset"
            return FunctionCallNode(function_name, args)

        if dref is not None and bias is None and is_projected:
            function_name = "textureCompareProj"
        elif dref is not None and bias is None:
            function_name = "textureCompare"
        else:
            function_name = "textureProj" if is_projected else "texture"
        args = list(base_args)
        if bias is not None:
            args.append(bias)
        return FunctionCallNode(function_name, args)

    def spirv_assembly_image_min_lod_sample_expression(
        self,
        opcode,
        base_args,
        parsed_operands,
        offset,
        bias,
        min_lod,
    ):
        """CrossGL ``texture*Clamp`` for a sample with the ``MinLod`` image operand (SPIR-V 3.14).

        ``base_args`` are the sampled image, the coordinate and, for a ``Dref`` sample, the reference.
        """
        is_compare = "Dref" in opcode
        gradient = parsed_operands.get("Grad") or []
        if "Proj" in opcode:
            raise SpirvInstructionUnsupportedError(
                opcode,
                "with a MinLod operand has no CrossGL form: CrossGL has no projective minimum-LOD clamped sample",
            )
        if parsed_operands.get("Lod") or (gradient and len(gradient) < 2):
            raise SpirvInstructionUnsupportedError(
                opcode,
                "combines MinLod with an explicit Lod, which SPIR-V allows only on implicit-LOD or Grad sampling",
            )
        if is_compare and bias is not None:
            raise SpirvInstructionUnsupportedError(
                opcode,
                "with Bias and MinLod has no CrossGL form: CrossGL depth compares take no bias",
            )
        name = "textureCompare" if is_compare else "texture"
        args = list(base_args)
        if gradient:
            name += "Grad"
            args.extend(gradient[:2])
        if offset is not None:
            name += "Offset"
            args.append(offset)
        args.append(min_lod)
        if bias is not None:
            args.append(bias)
        return FunctionCallNode(f"{name}Clamp", args)

    def spirv_assembly_image_sparse_sample_expression(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
        coordinate_components=None,
    ):
        sample = self.spirv_assembly_image_sample_expression(
            opcode,
            image_operand,
            coordinate_operand,
            image_operands,
            expressions,
            names,
            decorations,
            constants,
            coordinate_components=coordinate_components,
        )
        function_name = self.spirv_sparse_image_sample_function_name(
            opcode, sample.name
        )
        return FunctionCallNode(function_name, sample.args)

    def spirv_sparse_image_sample_function_name(self, opcode, sample_function_name):
        if sample_function_name.startswith("spirvTexture"):
            suffix = sample_function_name[len("spirvTexture") :]
        elif "Dref" in opcode and sample_function_name.startswith("textureCompare"):
            suffix = sample_function_name[len("textureCompare") :]
        elif sample_function_name.startswith("texture"):
            suffix = sample_function_name[len("texture") :]
        else:
            suffix = self.spirv_fallback_identifier(sample_function_name, "sample")

        dref = "Dref" if "Dref" in opcode else ""
        return f"spirvImageSparseSample{dref}{suffix}"

    def spirv_assembly_image_sparse_read_expression(
        self,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        read = self.spirv_assembly_image_read_expression(
            image_operand,
            coordinate_operand,
            image_operands,
            expressions,
            names,
            decorations,
            constants,
        )
        function_name = self.spirv_sparse_image_read_function_name(read.name)
        return FunctionCallNode(function_name, read.args)

    def spirv_sparse_image_read_function_name(self, read_function_name):
        if read_function_name == "imageLoad":
            return "spirvImageSparseRead"
        if read_function_name.startswith("spirvImageLoad"):
            suffix = read_function_name[len("spirvImageLoad") :]
            return f"spirvImageSparseRead{suffix}"
        suffix = self.spirv_fallback_identifier(read_function_name, "read")
        return f"spirvImageSparseRead{suffix}"

    def spirv_assembly_image_sparse_fetch_expression(
        self,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        fetch = self.spirv_assembly_image_fetch_expression(
            image_operand,
            coordinate_operand,
            image_operands,
            expressions,
            names,
            decorations,
            constants,
        )
        function_name = self.spirv_sparse_image_fetch_function_name(fetch.name)
        return FunctionCallNode(function_name, fetch.args)

    def spirv_sparse_image_fetch_function_name(self, fetch_function_name):
        if fetch_function_name == "texelFetch":
            return "spirvImageSparseFetch"
        if fetch_function_name.startswith("texelFetch"):
            suffix = fetch_function_name[len("texelFetch") :]
            return f"spirvImageSparseFetch{suffix}"
        suffix = self.spirv_fallback_identifier(fetch_function_name, "fetch")
        return f"spirvImageSparseFetch{suffix}"

    def spirv_assembly_image_sparse_gather_expression(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        gather_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        gather = self.spirv_assembly_image_gather_expression(
            {
                "OpImageSparseGather": "OpImageGather",
                "OpImageSparseDrefGather": "OpImageDrefGather",
            }[opcode],
            image_operand,
            coordinate_operand,
            gather_operand,
            image_operands,
            expressions,
            names,
            decorations,
            constants,
        )
        function_name = self.spirv_sparse_image_gather_function_name(
            opcode,
            gather.name,
        )
        return FunctionCallNode(function_name, gather.args)

    def spirv_sparse_image_gather_function_name(self, opcode, gather_function_name):
        if "Dref" in opcode and gather_function_name.startswith("textureGatherCompare"):
            suffix = gather_function_name[len("textureGatherCompare") :]
            return f"spirvImageSparseDrefGather{suffix}"
        if gather_function_name.startswith("textureGather"):
            suffix = gather_function_name[len("textureGather") :]
            return f"spirvImageSparseGather{suffix}"
        suffix = self.spirv_fallback_identifier(gather_function_name, "gather")
        return f"spirvImageSparseGather{suffix}"

    def spirv_assembly_image_gather_expression(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        gather_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        gathered = self.spirv_assembly_operand_expression(
            gather_operand, expressions, names, decorations, constants
        )
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        offset = self.spirv_assembly_image_offset_operand(parsed_operands)
        offsets = self.spirv_assembly_image_const_offsets_operand(parsed_operands)

        if opcode == "OpImageDrefGather":
            base_args = [image, coordinate, gathered]
            if offsets is not None:
                return FunctionCallNode(
                    "textureGatherCompareOffsets", [*base_args, offsets]
                )
            if offset is not None:
                return FunctionCallNode(
                    "textureGatherCompareOffset", [*base_args, offset]
                )
            return FunctionCallNode("textureGatherCompare", base_args)

        base_args = [image, coordinate]
        if offsets is not None:
            return FunctionCallNode(
                "textureGatherOffsets", [*base_args, offsets, gathered]
            )
        if offset is not None:
            return FunctionCallNode(
                "textureGatherOffset", [*base_args, offset, gathered]
            )
        return FunctionCallNode("textureGather", [*base_args, gathered])

    def spirv_assembly_image_fetch_expression(
        self,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        args = [image, coordinate]
        if "Lod" in parsed_operands and parsed_operands["Lod"]:
            args.append(parsed_operands["Lod"][0])
        sample = self.spirv_assembly_image_sample_operand(parsed_operands)
        if sample is not None:
            args.append(sample)
        offset = self.spirv_assembly_image_offset_operand(parsed_operands)
        if offset is not None:
            args.append(offset)
            return FunctionCallNode("texelFetchOffset", args)
        return FunctionCallNode("texelFetch", args)

    def spirv_assembly_image_read_expression(
        self,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        args = [image, coordinate]
        sample = self.spirv_assembly_image_sample_operand(parsed_operands)
        lod = self.spirv_assembly_image_lod_operand(parsed_operands)
        if lod is not None:
            args.append(lod)
            if sample is not None:
                args.append(sample)
            return FunctionCallNode("spirvImageLoadLod", args)
        if sample is not None:
            args.append(sample)
        return FunctionCallNode("imageLoad", args)

    def spirv_assembly_image_texel_pointer_expression(
        self,
        image_operand,
        coordinate_operand,
        sample_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "spirvImageTexelPointer",
            [
                self.spirv_assembly_operand_expression(
                    image_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    coordinate_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    sample_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
            ],
        )

    def spirv_assembly_image_write_statement(
        self,
        image_operand,
        coordinate_operand,
        texel_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        texel = self.spirv_assembly_operand_expression(
            texel_operand, expressions, names, decorations, constants
        )
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        args = [image, coordinate]
        sample = self.spirv_assembly_image_sample_operand(parsed_operands)
        lod = self.spirv_assembly_image_lod_operand(parsed_operands)
        if lod is not None:
            args.append(lod)
            if sample is not None:
                args.append(sample)
            args.append(texel)
            return FunctionCallNode("spirvImageStoreLod", args)
        if sample is not None:
            args.append(sample)
        args.append(texel)
        return FunctionCallNode("imageStore", args)

    def spirv_assembly_copy_memory_sized_statement(
        self,
        target_operand,
        source_operand,
        size_operand,
        memory_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "spirvCopyMemorySized",
            [
                self.spirv_assembly_operand_expression(
                    operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                for operand in (
                    target_operand,
                    source_operand,
                    size_operand,
                    *memory_operands,
                )
            ],
        )

    def spirv_assembly_image_sample_operand(self, parsed_operands):
        values = parsed_operands.get("Sample")
        if values:
            return values[0]
        return None

    def spirv_assembly_image_lod_operand(self, parsed_operands):
        values = parsed_operands.get("Lod")
        if values:
            return values[0]
        return None

    def spirv_assembly_image_offset_operand(self, parsed_operands):
        for operand_name in ("ConstOffset", "Offset"):
            values = parsed_operands.get(operand_name)
            if values:
                return values[0]
        return None

    def spirv_assembly_image_bias_operand(self, parsed_operands):
        values = parsed_operands.get("Bias")
        if values:
            return values[0]
        return None

    def spirv_assembly_image_min_lod_operand(self, parsed_operands):
        values = parsed_operands.get("MinLod")
        if values:
            return values[0]
        return None

    def spirv_assembly_image_const_offsets_operand(self, parsed_operands):
        values = parsed_operands.get("ConstOffsets")
        if values:
            return values[0]
        return None

    def spirv_assembly_bitcast_expression(
        self,
        result_type_id,
        operand_id,
        expressions,
        expression_type_ids,
        names,
        decorations,
        constants,
        constant_types,
        types,
    ):
        operand = self.spirv_assembly_operand_expression(
            operand_id, expressions, names, decorations, constants
        )
        source_type_id = expression_type_ids.get(operand_id) or constant_types.get(
            operand_id
        )
        source_family = self.spirv_bitcast_component_family(source_type_id, types)
        result_family = self.spirv_bitcast_component_family(result_type_id, types)

        if source_family == result_family and source_family is not None:
            return operand

        function_name = self.spirv_bitcast_function_name(source_family, result_family)
        if function_name is None:
            result_type_name = self.spirv_type_name(result_type_id, types)
            fallback_type = self.spirv_fallback_identifier(
                result_type_name or result_type_id, "type"
            )
            function_name = f"spirvBitcast_{fallback_type}"

        return FunctionCallNode(function_name, [operand])

    def spirv_bitcast_component_family(self, type_id, types):
        type_info = types.get(type_id, {})
        if type_info.get("kind") == "vector":
            return self.spirv_bitcast_component_family(
                type_info.get("component_type"), types
            )

        type_name = self.spirv_type_name(type_id, types)
        if type_name in {"float", "int", "uint"}:
            return type_name
        return None

    def spirv_bitcast_function_name(self, source_family, result_family):
        return {
            ("float", "int"): "floatBitsToInt",
            ("float", "uint"): "floatBitsToUint",
            ("int", "float"): "intBitsToFloat",
            ("int", "uint"): "asuint",
            ("uint", "int"): "asint",
            ("uint", "float"): "uintBitsToFloat",
        }.get((source_family, result_family))

    def spirv_assembly_undef_expression(self, result_type_id, types):
        type_name = self.spirv_type_name(result_type_id, types)
        fallback_type = self.spirv_fallback_identifier(
            type_name or result_type_id, "type"
        )
        return FunctionCallNode(f"spirvUndef_{fallback_type}", [])

    def spirv_is_undef_expression(self, value):
        return (
            isinstance(value, FunctionCallNode)
            and value.name.startswith("spirvUndef_")
            and not value.args
        )

    def spirv_assembly_null_expression(self, result_type_id, types):
        type_name = self.spirv_type_name(result_type_id, types)
        fallback_type = self.spirv_fallback_identifier(
            type_name or result_type_id, "type"
        )
        return FunctionCallNode(f"spirvNull_{fallback_type}", [])

    def spirv_is_null_expression(self, value):
        return (
            isinstance(value, FunctionCallNode)
            and value.name.startswith("spirvNull_")
            and not value.args
        )

    def spirv_is_generated_constant_expression(self, value):
        return self.spirv_is_undef_expression(value) or self.spirv_is_null_expression(
            value
        )

    def spirv_assembly_composite_insert_expression(
        self,
        result_type_id,
        object_operand,
        composite_operand,
        index_operands,
        expressions,
        names,
        decorations,
        constants,
        types,
    ):
        inserted = self.spirv_assembly_operand_expression(
            object_operand, expressions, names, decorations, constants
        )
        composite = self.spirv_assembly_operand_expression(
            composite_operand, expressions, names, decorations, constants
        )
        result_type_name = self.spirv_type_name(result_type_id, types)
        result_component_count = self.spirv_vector_component_count(
            result_type_id, types
        )

        if len(index_operands) == 1 and result_component_count is not None:
            component_index = self.spirv_vector_shuffle_component_index(
                index_operands[0]
            )
            if (
                component_index is not None
                and 0 <= component_index < result_component_count
            ):
                return FunctionCallNode(
                    result_type_name or result_type_id,
                    [
                        (
                            inserted
                            if index == component_index
                            else self.spirv_composite_component_expression(
                                composite,
                                index,
                                result_component_count,
                                result_type_name,
                            )
                        )
                        for index in range(result_component_count)
                    ],
                )

        return FunctionCallNode(
            "spirvCompositeInsert",
            [composite, inserted, *index_operands],
        )

    def spirv_assembly_vector_insert_dynamic_expression(
        self,
        result_type_id,
        vector_operand,
        component_operand,
        index_operand,
        expressions,
        names,
        decorations,
        constants,
        types,
    ):
        vector = self.spirv_assembly_operand_expression(
            vector_operand, expressions, names, decorations, constants
        )
        component = self.spirv_assembly_operand_expression(
            component_operand, expressions, names, decorations, constants
        )
        dynamic_index = self.spirv_assembly_operand_expression(
            index_operand, expressions, names, decorations, constants
        )
        result_type_name = self.spirv_type_name(result_type_id, types)
        result_component_count = self.spirv_vector_component_count(
            result_type_id, types
        )
        component_index = self.spirv_integer_constant_operand(index_operand, constants)

        if (
            component_index is not None
            and result_component_count is not None
            and 0 <= component_index < result_component_count
        ):
            return FunctionCallNode(
                result_type_name or result_type_id,
                [
                    (
                        component
                        if component_position == component_index
                        else self.spirv_composite_component_expression(
                            vector,
                            component_position,
                            result_component_count,
                            result_type_name,
                        )
                    )
                    for component_position in range(result_component_count)
                ],
            )

        return FunctionCallNode(
            "spirvVectorInsertDynamic", [vector, component, dynamic_index]
        )

    def spirv_composite_component_expression(
        self, composite, component, component_count, result_type_name
    ):
        if (
            isinstance(composite, FunctionCallNode)
            and composite.name == result_type_name
            and component < len(composite.args)
        ):
            return composite.args[component]

        if component_count <= 4:
            return MemberAccessNode(
                composite, self.spirv_vector_component_name(component)
            )
        return ArrayAccessNode(composite, str(component))

    def spirv_assembly_image_operands(
        self, operands, expressions, names, decorations, constants
    ):
        parsed = {}
        index = 0
        operand_counts = {
            "Bias": 1,
            "ConstOffset": 1,
            "ConstOffsets": 1,
            "Grad": 2,
            "Lod": 1,
            "MakeTexelAvailable": 1,
            "MakeTexelAvailableKHR": 1,
            "MakeTexelVisible": 1,
            "MakeTexelVisibleKHR": 1,
            "MinLod": 1,
            "Offset": 1,
            "Sample": 1,
        }
        while index < len(operands):
            operand = operands[index]
            operand_names = str(operand).split("|")
            if any(name in operand_counts for name in operand_names):
                index += 1
                for name in operand_names:
                    count = operand_counts.get(name, 0)
                    if count == 0:
                        continue
                    values = [
                        self.spirv_assembly_operand_expression(
                            value,
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                        for value in operands[index : index + count]
                    ]
                    parsed[name] = values
                    index += count
            else:
                index += 1
        return parsed

    def spirv_ext_inst_function_name(self, instruction_set, instruction):
        if instruction_set == "NonSemantic.DebugPrintf" and str(instruction) in {
            "1",
            "DebugPrintf",
        }:
            return "printf"
        if instruction_set == "GLSL.std.450":
            instruction_name = self.SPIRV_GLSL_STD_450_EXT_INST_IDS.get(
                str(instruction), instruction
            )
            mapped_name = self.SPIRV_GLSL_STD_450_EXT_INST_FUNCTIONS.get(
                instruction_name
            )
            if mapped_name is not None:
                return mapped_name
        if not instruction:
            return "spirv_ext_inst"
        sanitized_set = self.spirv_fallback_identifier(instruction_set, "ext_inst")
        sanitized_instruction = self.spirv_fallback_identifier(
            instruction, "instruction"
        )
        return f"spirv_{sanitized_set}_{sanitized_instruction}"

    def spirv_assembly_atomic_expression(
        self,
        opcode,
        pointer_operand,
        value_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        value = self.spirv_assembly_operand_expression(
            value_operand,
            expressions,
            names,
            decorations,
            constants,
        )
        if opcode == "OpAtomicISub":
            value = UnaryOpNode("-", value)

        return FunctionCallNode(
            self.SPIRV_ATOMIC_RMW_FUNCTIONS[opcode],
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                value,
            ],
        )

    def spirv_assembly_atomic_increment_expression(
        self,
        opcode,
        pointer_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        value = "1"
        if opcode == "OpAtomicIDecrement":
            value = UnaryOpNode("-", value)

        return FunctionCallNode(
            "atomicAdd",
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                value,
            ],
        )

    def spirv_assembly_atomic_load_expression(
        self,
        pointer_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "atomicLoad",
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
            ],
        )

    def spirv_assembly_atomic_store_statement(
        self,
        pointer_operand,
        value_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "atomicStore",
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    value_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
            ],
        )

    def spirv_assembly_geometry_emit_statement(
        self,
        opcode,
        operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            self.SPIRV_GEOMETRY_EMIT_FUNCTIONS[opcode],
            [
                self.spirv_assembly_operand_expression(
                    operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                for operand in operands
            ],
        )

    def spirv_assembly_ray_query_statement(
        self,
        opcode,
        operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            self.SPIRV_RAY_QUERY_STATEMENT_FUNCTIONS[opcode],
            [
                self.spirv_assembly_operand_expression(
                    operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                for operand in operands
            ],
        )

    def spirv_assembly_ray_query_expression(
        self,
        opcode,
        operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            self.SPIRV_RAY_QUERY_EXPRESSION_FUNCTIONS[opcode],
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
        )

    def spirv_assembly_atomic_compare_exchange_expression(
        self,
        pointer_operand,
        value_operand,
        comparator_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "atomicCompSwap",
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    comparator_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    value_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
            ],
        )

    def spirv_assembly_group_non_uniform_reduction_expression(
        self,
        opcode,
        group_operation,
        value_operand,
        extra_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        operation_name = self.SPIRV_GROUP_NON_UNIFORM_REDUCTION_FUNCTIONS[opcode]
        prefix = self.SPIRV_GROUP_NON_UNIFORM_SCAN_PREFIXES.get(group_operation)
        if prefix is None:
            sanitized_group_operation = self.spirv_fallback_identifier(
                group_operation, "group_operation"
            )
            prefix = f"spirvGroupNonUniform{sanitized_group_operation}"

        function_name = f"{prefix}{operation_name}"
        args = [
            self.spirv_assembly_operand_expression(
                value_operand,
                expressions,
                names,
                decorations,
                constants,
            )
        ]
        if group_operation == "ClusteredReduce" and extra_operands:
            args.append(
                self.spirv_assembly_operand_expression(
                    extra_operands[0],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
            )
        return FunctionCallNode(function_name, args)

    def spirv_assembly_vector_shuffle_expression(
        self,
        result_type_id,
        vector1_id,
        vector2_id,
        component_operands,
        expressions,
        expression_type_ids,
        names,
        decorations,
        constants,
        types,
    ):
        source1_count = self.spirv_vector_component_count(
            expression_type_ids.get(vector1_id), types
        )
        components = [
            self.spirv_vector_shuffle_component_index(component)
            for component in component_operands
        ]

        if (
            vector1_id == vector2_id
            and source1_count is not None
            and all(
                component is not None and 0 <= component < source1_count
                for component in components
            )
        ):
            swizzle = "".join(
                self.spirv_vector_component_name(component) for component in components
            )
            source = self.spirv_assembly_operand_expression(
                vector1_id, expressions, names, decorations, constants
            )
            return MemberAccessNode(source, swizzle)

        result_type = self.spirv_type_name(result_type_id, types) or result_type_id
        args = [
            self.spirv_vector_shuffle_component_expression(
                vector1_id,
                vector2_id,
                component,
                source1_count,
                expressions,
                expression_type_ids,
                names,
                decorations,
                constants,
                types,
            )
            for component in components
        ]
        return FunctionCallNode(result_type, args)

    def spirv_vector_shuffle_component_expression(
        self,
        vector1_id,
        vector2_id,
        component,
        source1_count,
        expressions,
        expression_type_ids,
        names,
        decorations,
        constants,
        types,
    ):
        if component is None:
            return "0"

        if source1_count is not None and component >= source1_count:
            source_id = vector2_id
            component -= source1_count
        else:
            source_id = vector1_id

        source = self.spirv_assembly_operand_expression(
            source_id, expressions, names, decorations, constants
        )
        source_count = self.spirv_vector_component_count(
            expression_type_ids.get(source_id), types
        )
        if source_count is not None and 0 <= component < min(source_count, 4):
            return MemberAccessNode(source, self.spirv_vector_component_name(component))
        return ArrayAccessNode(source, str(component))

    def spirv_vector_component_count(self, type_id, types):
        type_info = types.get(type_id, {})
        component_count = type_info.get("component_count")
        if component_count is None:
            return None
        try:
            return int(component_count, 0)
        except ValueError:
            return None

    def spirv_vector_shuffle_component_index(self, component):
        try:
            index = int(component, 0)
        except (TypeError, ValueError):
            return None
        if index in {-1, 0xFFFFFFFF}:
            return None
        return index

    def spirv_vector_component_name(self, component):
        return "xyzw"[component]

    def spirv_assembly_operand_expression(
        self,
        operand,
        expressions,
        names,
        decorations,
        constants,
        variables_by_id=None,
        types=None,
    ):
        if operand in expressions:
            return self.spirv_maybe_non_uniform_expression(
                operand, expressions[operand], decorations
            )

        if operand in constants:
            expression = self.spirv_constant_operand_expression(
                operand, names, constants
            )
            return self.spirv_maybe_non_uniform_expression(
                operand, expression, decorations
            )

        if isinstance(operand, str) and operand.startswith("%"):
            expression = VariableNode(
                "",
                self.spirv_assembly_value_name(
                    operand,
                    names,
                    decorations,
                    storage_class=self.spirv_assembly_variable_storage_class(
                        operand, variables_by_id, types
                    ),
                ),
                spirv_id=operand,
            )
            return self.spirv_maybe_non_uniform_expression(
                operand, expression, decorations
            )

        return operand

    def spirv_maybe_non_uniform_expression(self, operand, expression, decorations):
        if not isinstance(operand, str):
            return expression
        if not self.spirv_has_decoration(decorations.get(operand, []), "NonUniform"):
            return expression
        if (
            isinstance(expression, FunctionCallNode)
            and expression.name == "nonuniformEXT"
        ):
            return expression
        return FunctionCallNode("nonuniformEXT", [expression])

    def spirv_assembly_value_name(
        self,
        value_id,
        names,
        decorations,
        prefix="value",
        storage_class=None,
    ):
        qualifiers = self.spirv_layout_qualifiers(decorations.get(value_id, []))
        builtin_name = self.spirv_builtin_variable_name_from_qualifiers(
            qualifiers, storage_class=storage_class
        )
        if builtin_name:
            return builtin_name
        if value_id in names and names[value_id]:
            return names[value_id]
        fallback = self.spirv_fallback_identifier(value_id, prefix)
        return self.SPIRV_VALUE_FALLBACK_RENAMES.get(fallback, fallback)

    def spirv_assembly_variable_storage_class(self, value_id, variables_by_id, types):
        if not variables_by_id or not types:
            return None
        variable = variables_by_id.get(value_id)
        if variable is None:
            return None
        pointer_type = types.get(variable.get("pointer_type_id"), {})
        return variable.get("storage_class") or pointer_type.get("storage_class")

    def spirv_is_flattened_resource_block_access(
        self, storage_class, struct_type_id, decorations
    ):
        struct_decorations = decorations.get(struct_type_id, [])
        if not self.spirv_has_decoration(struct_decorations, "Block"):
            return False
        if storage_class == "PushConstant":
            return True
        if storage_class == "Uniform" and not self.spirv_has_decoration(
            struct_decorations, "BufferBlock"
        ):
            return True
        return False

    def spirv_is_structured_buffer_block_access(
        self, storage_class, struct_type_id, decorations
    ):
        struct_decorations = decorations.get(struct_type_id, [])
        if storage_class == "StorageBuffer":
            return self.spirv_has_decoration(struct_decorations, "Block")
        if storage_class == "Uniform":
            return self.spirv_has_decoration(struct_decorations, "BufferBlock")
        return False

    def spirv_function_name_fallback(self, function_id, entry_points_by_id):
        if function_id in entry_points_by_id:
            return entry_points_by_id[function_id][0].get("name")
        if function_id:
            return self.spirv_fallback_identifier(function_id, "function")
        return ""

    def spirv_assembly_function_return_type_name(self, type_id, types, constants):
        if type_id is None:
            return "void"
        signature_type = self.spirv_function_signature_type_name(
            type_id, types, constants
        )
        if signature_type is not None:
            return signature_type
        return (
            self.spirv_type_name(type_id, types)
            or self.spirv_fallback_identifier(type_id, "return_type")
            or "void"
        )

    def spirv_function_parameter_type_name(self, type_id, types, constants):
        signature_type = self.spirv_function_signature_type_name(
            type_id, types, constants
        )
        if signature_type is not None:
            return signature_type
        return self.spirv_type_name(type_id, types) or self.spirv_fallback_identifier(
            type_id, "param_type"
        )

    def spirv_function_signature_type_name(self, type_id, types, constants):
        type_info = types.get(type_id, {})
        if type_info.get("kind") == "pointer":
            return self.spirv_function_signature_type_name(
                type_info.get("type_id"), types, constants
            )
        base_type, suffix = self.spirv_type_name_and_suffix(type_id, types, constants)
        if base_type is None:
            return None
        return f"{base_type}{suffix}"

    def spirv_written_pointer_parameter_ids(
        self, parameter_records, raw_instructions, types
    ):
        alias_origins = {
            parameter_id: parameter_id
            for parameter_id, parameter_type_id in parameter_records
            if parameter_id
            and types.get(parameter_type_id, {}).get("kind") == "pointer"
        }
        written_parameter_ids = set()

        for result_id, opcode, operands, _line_number in raw_instructions:
            if (
                result_id
                and opcode in {
                    "OpAccessChain",
                    "OpInBoundsAccessChain",
                    "OpPtrAccessChain",
                    "OpInBoundsPtrAccessChain",
                }
                and len(operands) >= 2
                and operands[1] in alias_origins
            ):
                alias_origins[result_id] = alias_origins[operands[1]]
                continue

            if (
                result_id
                and opcode == "OpCopyObject"
                and len(operands) >= 2
                and operands[1] in alias_origins
            ):
                alias_origins[result_id] = alias_origins[operands[1]]
                continue

            if (
                opcode in {
                    "OpStore",
                    "OpCopyMemory",
                    "OpCopyMemorySized",
                    "OpAtomicStore",
                }
                and operands
            ):
                written_parameter_id = alias_origins.get(operands[0])
                if written_parameter_id:
                    written_parameter_ids.add(written_parameter_id)

        return written_parameter_ids

    def spirv_function_parameter_qualifiers(self, type_id, types, is_written):
        type_info = types.get(type_id, {})
        if is_written and type_info.get("kind") == "pointer":
            return ["inout"]
        return []

    def parse_spirv_assembly_instructions(self, code):
        instructions = []
        if self.looks_like_legacy_glslang_spirv_disassembly(code):
            code = self.normalize_legacy_glslang_spirv_disassembly(code)

        tokens = self.spirv_assembly_tokens(code)
        index = 0

        while index < len(tokens):
            instruction_start = index
            operand_start = (
                instruction_start + 3
                if self.is_spirv_assembly_result_instruction_start(
                    tokens, instruction_start
                )
                else instruction_start + 1
            )
            next_index = self.next_spirv_assembly_instruction_index(
                tokens, operand_start
            )
            instruction_tokens = tokens[instruction_start:next_index]
            parts = [text for text, _quoted, _line_number in instruction_tokens]
            instruction_line_number = instruction_tokens[0][2]

            result_id = None
            if len(parts) >= 3 and parts[1] == "=":
                result_id = parts[0]
                opcode = parts[2]
                operands = parts[3:]
            else:
                opcode = parts[0]
                operands = parts[1:]

            if not opcode.startswith("Op"):
                raise SyntaxError(
                    "Expected SPIR-V opcode on line "
                    f"{instruction_line_number}, got {opcode}"
                )
            instructions.append((result_id, opcode, operands, instruction_line_number))
            index = next_index

        return instructions

    def looks_like_legacy_glslang_spirv_disassembly(self, code):
        return bool(
            re.search(
                r"(?m)^\s*(?:"
                r"Capability\s+\w+\b"
                r"|MemoryModel\s+\w+\s+\w+\b"
                r"|EntryPoint\s+\w+\s+\d+\b"
                r"|\d+(?:\([^)]*\))?:\s+(?:Type[A-Za-z0-9_]+|Function|Label|Variable)\b"
                r")",
                code,
            )
        )

    def normalize_legacy_glslang_spirv_disassembly(self, code):
        normalized_lines = []
        module_index = -1
        id_namespace = None
        for raw_line in code.splitlines():
            if re.match(r"\s*//\s*Module Version\b", raw_line):
                module_index += 1
                id_namespace = f"m{module_index}_"
                continue

            line = raw_line.split("//", 1)[0].strip()
            if not line:
                continue
            try:
                parts = shlex.split(line, comments=False, posix=True)
            except ValueError as error:
                if self.can_skip_unparseable_legacy_glslang_spirv_line(line):
                    continue
                if self.legacy_glslang_spirv_line_has_instruction_shape(line):
                    raise SyntaxError(
                        f"Invalid glslang SPIR-V disassembly syntax: {error}"
                    ) from error
                continue

            instruction = self.normalize_legacy_glslang_spirv_instruction(
                parts, id_namespace=id_namespace
            )
            if instruction is not None:
                normalized_lines.append(instruction)

        return "\n".join(normalized_lines)

    def legacy_glslang_spirv_line_has_instruction_shape(self, line):
        opcode_pattern = "|".join(
            sorted(
                set(self.LEGACY_GLSLANG_SPIRV_ID_OPERANDS)
                | self.LEGACY_GLSLANG_SPIRV_GLOBAL_OPCODES
            )
        )
        return bool(
            re.match(r"\d+(?:\([^)]*\))?:\s+", line)
            or re.match(rf"(?:{opcode_pattern})\b", line)
        )

    def can_skip_unparseable_legacy_glslang_spirv_line(self, line):
        return bool(
            re.match(
                r"(?:\d+(?:\([^)]*\))?:\s+)?(?:Source|SourceContinued|String)\b",
                line,
            )
        )

    def normalize_legacy_glslang_spirv_instruction(self, parts, id_namespace=None):
        if not parts:
            return None

        result_id = None
        if self.is_legacy_glslang_spirv_result_token(parts[0]):
            result_id = self.legacy_glslang_spirv_id(
                parts.pop(0), namespace=id_namespace
            )
        if not parts:
            return None

        known_opcodes = (
            set(self.LEGACY_GLSLANG_SPIRV_ID_OPERANDS)
            | self.LEGACY_GLSLANG_SPIRV_GLOBAL_OPCODES
        )
        if result_id is None and parts[0] not in known_opcodes:
            return None

        if result_id is not None and parts[0] in known_opcodes:
            opcode = parts.pop(0)
            operands = [
                self.normalize_legacy_glslang_spirv_operand(
                    opcode, index, operand, id_namespace=id_namespace
                )
                for index, operand in enumerate(parts)
            ]
        elif result_id is not None and len(parts) >= 2 and parts[1] in known_opcodes:
            result_type_token = parts.pop(0)
            result_type = self.legacy_glslang_spirv_id(
                result_type_token, namespace=id_namespace
            )
            opcode = parts.pop(0)
            operands = [
                result_type,
                *[
                    self.normalize_legacy_glslang_spirv_operand(
                        opcode, index + 1, operand, id_namespace=id_namespace
                    )
                    for index, operand in enumerate(parts)
                ],
            ]
            operands = self.normalize_legacy_glslang_float_constant_operands(
                opcode, result_type_token, operands
            )
        else:
            opcode = parts.pop(0)
            operands = [
                self.normalize_legacy_glslang_spirv_operand(
                    opcode, index, operand, id_namespace=id_namespace
                )
                for index, operand in enumerate(parts)
            ]

        if not opcode.startswith("Op"):
            opcode = f"Op{opcode}"
        instruction = f"{result_id} = {opcode}" if result_id is not None else opcode
        if operands:
            instruction = f"{instruction} {' '.join(operands)}"
        return instruction

    def normalize_legacy_glslang_float_constant_operands(
        self, opcode, result_type_token, operands
    ):
        if opcode not in {"Constant", "SpecConstant"} or len(operands) < 2:
            return operands

        type_label = self.legacy_glslang_spirv_type_label(result_type_token)
        literal_widths = {
            "float16_t": 16,
            "float": 32,
            "float32_t": 32,
            "double": 64,
            "float64_t": 64,
        }
        width = literal_widths.get(type_label)
        if width is None:
            return operands

        word_count = 2 if width == 64 else 1
        literal_words = operands[1 : 1 + word_count]
        if len(literal_words) != word_count:
            return operands

        literal = self.legacy_glslang_float_literal_from_words(literal_words, width)
        if literal is None:
            return operands
        return [operands[0], literal, *operands[1 + word_count :]]

    def legacy_glslang_spirv_type_label(self, token):
        match = re.match(r"\d+\(([^)]*)\):?$", str(token))
        return match.group(1) if match else None

    def legacy_glslang_float_literal_from_words(self, words, width):
        try:
            integer_words = [int(str(word), 0) for word in words]
        except (TypeError, ValueError):
            return None

        if width == 16:
            data = (integer_words[0] & 0xFFFF).to_bytes(2, "little")
            value = struct.unpack("<e", data)[0]
        elif width == 32:
            data = (integer_words[0] & 0xFFFFFFFF).to_bytes(4, "little")
            value = struct.unpack("<f", data)[0]
        elif width == 64:
            data = b"".join(
                (word & 0xFFFFFFFF).to_bytes(4, "little") for word in integer_words
            )
            value = struct.unpack("<d", data)[0]
        else:
            return None
        return repr(value)

    def is_legacy_glslang_spirv_result_token(self, token):
        return bool(re.match(r"\d+(?:\([^)]*\))?:$", str(token)))

    def legacy_glslang_spirv_id(self, token, namespace=None):
        namespace = namespace or ""
        match = re.match(r"(\d+)(?:\([^)]*\))?:?$", str(token))
        if match:
            return f"%{namespace}{match.group(1)}"
        if re.match(r"\d+$", str(token)):
            return f"%{namespace}{token}"
        return token

    def normalize_legacy_glslang_spirv_operand(
        self, opcode, operand_index, operand, id_namespace=None
    ):
        if self.is_legacy_glslang_spirv_id_operand(opcode, operand_index):
            return self.legacy_glslang_spirv_id(operand, namespace=id_namespace)
        return operand

    def is_legacy_glslang_spirv_id_operand(self, opcode, operand_index):
        id_operands = self.LEGACY_GLSLANG_SPIRV_ID_OPERANDS.get(opcode)
        if id_operands == "all":
            return True
        if id_operands == "entry_point":
            return operand_index == 1 or operand_index >= 3
        if id_operands == "execution_mode_id":
            return operand_index == 0 or operand_index >= 2
        return operand_index in (id_operands or set())

    def spirv_assembly_tokens(self, code):
        tokens = []
        index = 0
        line_number = 1
        length = len(code)

        while index < length:
            char = code[index]
            if char.isspace():
                if char == "\n":
                    line_number += 1
                index += 1
                continue
            if char == ";":
                while index < length and code[index] != "\n":
                    index += 1
                continue
            if char == '"':
                token, index, line_number = self.spirv_assembly_string_token(
                    code, index, line_number
                )
                tokens.append(token)
                continue

            start = index
            token_line_number = line_number
            while index < length and not code[index].isspace() and code[index] != ";":
                index += 1
            tokens.append((code[start:index], False, token_line_number))

        return tokens

    def spirv_assembly_string_token(self, code, index, line_number):
        token_line_number = line_number
        value = []
        index += 1
        length = len(code)

        while index < length:
            char = code[index]
            if char == '"':
                token = ("".join(value), True, token_line_number)
                return token, index + 1, line_number
            if char == "\\":
                index += 1
                if index >= length:
                    break
                char = code[index]
            value.append(char)
            if char == "\n":
                line_number += 1
            index += 1

        raise SyntaxError(
            "Invalid SPIR-V assembly syntax on line "
            f"{token_line_number}: No closing quotation"
        )

    def next_spirv_assembly_instruction_index(self, tokens, index):
        while index < len(tokens):
            if self.is_spirv_assembly_instruction_start(tokens, index):
                return index
            index += 1
        return len(tokens)

    def is_spirv_assembly_instruction_start(self, tokens, index):
        if self.is_spirv_assembly_result_instruction_start(tokens, index):
            return True
        text, quoted, _line_number = tokens[index]
        if quoted:
            return False
        return text.startswith("Op")

    def is_spirv_assembly_result_instruction_start(self, tokens, index):
        if index + 2 >= len(tokens):
            return False
        text, quoted, _line_number = tokens[index]
        if quoted:
            return False
        equals, equals_quoted, _equals_line = tokens[index + 1]
        opcode, opcode_quoted, _opcode_line = tokens[index + 2]
        return (
            text.startswith("%")
            and equals == "="
            and not equals_quoted
            and opcode.startswith("Op")
            and not opcode_quoted
        )

    def spirv_assembly_interface_variables(
        self,
        variables,
        entry_interface_ids,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
    ):
        layouts = []
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if pointer_type.get("kind") != "pointer":
                continue

            if storage_class == "UniformConstant":
                resource_layout = self.spirv_assembly_uniform_constant_layout(
                    variable, pointer_type, names, decorations, types, constants
                )
                if resource_layout is not None:
                    layouts.append(resource_layout)
                continue

            if storage_class in {
                "PushConstant",
                "StorageBuffer",
                "Uniform",
                *self.SPIRV_SHADER_RECORD_STORAGE_CLASSES,
            }:
                resource_block_layout = self.spirv_assembly_resource_block_layout(
                    variable,
                    pointer_type,
                    storage_class,
                    names,
                    decorations,
                    member_decorations,
                    member_names,
                    types,
                    constants,
                )
                if resource_block_layout is not None:
                    layouts.append(resource_block_layout)
                continue

            ray_qualifier = self.SPIRV_RAY_STORAGE_QUALIFIERS.get(storage_class)
            if ray_qualifier is not None:
                ray_layout = self.spirv_assembly_ray_storage_layout(
                    variable,
                    pointer_type,
                    storage_class,
                    ray_qualifier,
                    names,
                    decorations,
                    types,
                    constants,
                    ray_variable_locations(variables, decorations),
                )
                if ray_layout is not None:
                    layouts.append(ray_layout)
                continue

            layout_type = self.SPIRV_INTERFACE_STORAGE_CLASSES.get(storage_class)
            if layout_type is None:
                continue
            if entry_interface_ids and variable["id"] not in entry_interface_ids:
                continue

            struct_layouts = self.spirv_assembly_struct_interface_layouts(
                variable,
                pointer_type,
                storage_class,
                names,
                decorations.get(variable["id"], []),
                member_decorations,
                member_names,
                types,
                constants,
            )
            if struct_layouts:
                layouts.extend(struct_layouts)
                continue

            variable_decorations = decorations.get(variable["id"], [])
            qualifiers = self.spirv_layout_qualifiers(variable_decorations)
            declaration_qualifiers = self.spirv_declaration_qualifiers(
                variable_decorations
            )
            if not self.spirv_has_interface_qualifier(
                qualifiers, declaration_qualifiers
            ):
                continue

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                pointer_type.get("type_id"), types, constants
            )
            if data_type is None:
                continue

            variable_name = self.spirv_assembly_value_name(
                variable["id"], names, decorations={}
            )
            variable_name = self.spirv_builtin_variable_name_from_qualifiers(
                qualifiers, variable_name, storage_class=storage_class
            )
            variable_name += array_suffix
            layouts.append(
                LayoutNode(
                    qualifiers,
                    layout_type=layout_type,
                    data_type=data_type,
                    variable_name=variable_name,
                    declaration_qualifiers=declaration_qualifiers,
                    spirv_id=variable["id"],
                    spirv_decorations=variable_decorations,
                    spirv_storage_class=storage_class,
                )
            )

        return layouts

    def spirv_assembly_ray_storage_layout(
        self,
        variable,
        pointer_type,
        storage_class,
        ray_qualifier,
        names,
        decorations,
        types,
        constants,
        ray_locations,
    ):
        """A ray payload, callable data or hit attribute variable, as its GL_EXT_ray_tracing declaration."""
        data_type, array_suffix = self.spirv_type_name_and_suffix(
            pointer_type.get("type_id"), types, constants, names=names
        )
        if data_type is None:
            return None
        variable_decorations = decorations.get(variable["id"], [])
        qualifiers = [
            (name, value)
            for name, value in self.spirv_layout_qualifiers(variable_decorations)
            if name == "location"
        ]
        location = ray_locations.get(variable["id"])
        if not qualifiers and location is not None:
            qualifiers = [("location", str(location))]
        variable_name = self.spirv_assembly_value_name(
            variable["id"], names, decorations={}
        )
        return LayoutNode(
            qualifiers,
            data_type=data_type,
            variable_name=f"{variable_name}{array_suffix}",
            declaration_qualifiers=[ray_qualifier],
            spirv_id=variable["id"],
            spirv_decorations=variable_decorations,
            spirv_storage_class=storage_class,
        )

    def spirv_assembly_private_global_variables(
        self, variables, names, decorations, types, constants
    ):
        declarations = []
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if storage_class != "Private" or pointer_type.get("kind") != "pointer":
                continue

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                pointer_type.get("type_id"), types, constants, names=names
            )
            if data_type is None:
                continue

            declaration = VariableNode(
                data_type,
                f"{self.spirv_assembly_value_name(variable['id'], names, decorations)}"
                f"{array_suffix}",
                spirv_id=variable["id"],
                spirv_type_id=pointer_type.get("type_id"),
            )
            initializer = variable.get("initializer")
            if initializer is not None:
                declarations.append(
                    AssignmentNode(
                        declaration,
                        self.spirv_assembly_operand_expression(
                            initializer,
                            {},
                            names,
                            decorations,
                            constants,
                        ),
                    )
                )
            else:
                declarations.append(declaration)

        return declarations

    def spirv_assembly_workgroup_global_variables(
        self, variables, names, decorations, types, constants
    ):
        declarations = []
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if storage_class != "Workgroup" or pointer_type.get("kind") != "pointer":
                continue

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                pointer_type.get("type_id"), types, constants, names=names
            )
            if data_type is None:
                continue

            declarations.append(
                VariableNode(
                    f"groupshared {data_type}",
                    f"{self.spirv_assembly_value_name(variable['id'], names, decorations)}"
                    f"{array_suffix}",
                    spirv_id=variable["id"],
                    spirv_type_id=pointer_type.get("type_id"),
                    spirv_storage_class=storage_class,
                )
            )

        return declarations

    def spirv_assembly_uniform_constant_layout(
        self, variable, pointer_type, names, decorations, types, constants
    ):
        resource_type = types.get(pointer_type.get("type_id"), {})
        resource_element_type = self.spirv_resource_element_type(resource_type, types)
        data_type, array_suffix = self.spirv_type_name_and_suffix(
            pointer_type.get("type_id"), types, constants
        )
        if data_type is None:
            return None

        variable_decorations = decorations.get(variable["id"], [])
        qualifiers = self.spirv_descriptor_qualifiers(variable_decorations)
        image_format = self.spirv_image_format_qualifier(
            resource_element_type.get("format")
        )
        if image_format is not None:
            qualifiers.append((image_format, None))
        qualifier_names = {name for name, _value in qualifiers}
        if not {"set", "binding"}.issubset(qualifier_names):
            return None

        declaration_qualifiers = self.spirv_declaration_qualifiers(variable_decorations)
        access_qualifier = self.spirv_image_access_qualifier(
            resource_element_type.get("access_qualifier")
        )
        if access_qualifier and access_qualifier not in declaration_qualifiers:
            declaration_qualifiers.append(access_qualifier)

        variable_name = self.spirv_assembly_value_name(
            variable["id"], names, decorations
        )
        variable_name += array_suffix
        return LayoutNode(
            qualifiers,
            layout_type="UNIFORM",
            data_type=data_type,
            variable_name=variable_name,
            declaration_qualifiers=declaration_qualifiers,
            spirv_id=variable["id"],
            spirv_decorations=variable_decorations,
            spirv_storage_class="UniformConstant",
        )

    def spirv_resource_element_type(self, resource_type, types):
        while resource_type.get("kind") in {"array", "runtime_array"}:
            resource_type = types.get(resource_type.get("element_type"), {})
        return resource_type

    def spirv_assembly_specialization_constants(
        self, spec_constant_ids, names, decorations, types, constants, constant_types
    ):
        layouts = []
        for result_id in spec_constant_ids:
            constant_decorations = decorations.get(result_id, [])
            constant_id = self.spirv_spec_constant_id(constant_decorations)
            data_type = self.spirv_type_name(constant_types.get(result_id), types)
            if data_type is None:
                continue

            qualifiers = []
            if constant_id is not None:
                qualifiers.append(("constant_id", constant_id))
            qualifiers.extend(self.spirv_layout_qualifiers(constant_decorations))

            variable_name = self.spirv_builtin_variable_name_from_qualifiers(qualifiers)
            if variable_name is None:
                variable_name = names.get(result_id)
            if variable_name is None:
                if constant_id is None:
                    continue
                variable_name = self.spirv_fallback_identifier(
                    constant_id, "spec_constant"
                )

            declaration = AssignmentNode(
                VariableNode(f"const {data_type}", variable_name),
                constants.get(result_id),
            )
            layouts.append(
                LayoutNode(
                    qualifiers,
                    declaration=declaration,
                    layout_type="CONST",
                    spirv_id=result_id,
                    spirv_decorations=constant_decorations,
                )
            )

        return layouts

    def spirv_register_struct_type_names(self, types, names, member_names=None):
        # OpName may give several struct types one name (glslang calls every modf and frexp result ``ResType``).
        # Struct types whose names and members agree are declared once; any other one takes the first free
        # ``name_N``, so no two CrossGL struct declarations conflict.
        member_names = member_names or {}
        taken = {}
        used_names = set(names.values())
        for type_id, type_info in types.items():
            if type_info.get("kind") != "struct":
                continue
            name = names.get(type_id) or self.spirv_fallback_identifier(
                type_id, "struct"
            )
            signature = (
                tuple(type_info.get("member_types", [])),
                tuple(sorted(member_names.get(type_id, {}).items())),
            )
            if taken.get(name, signature) != signature:
                suffix = 1
                while f"{name}_{suffix}" in taken or f"{name}_{suffix}" in used_names:
                    suffix += 1
                name = f"{name}_{suffix}"
            taken.setdefault(name, signature)
            type_info["name"] = name

    def spirv_assembly_structs(
        self,
        names,
        member_names,
        member_decorations,
        types,
        constants,
        skip_type_ids=None,
        builtin_interface_struct_storage_classes=None,
    ):
        skip_type_ids = set(skip_type_ids or [])
        builtin_interface_struct_storage_classes = (
            builtin_interface_struct_storage_classes or {}
        )
        structs = []
        seen_struct_signatures = set()
        for type_id, type_info in types.items():
            if type_info.get("kind") != "struct":
                continue
            if type_id in skip_type_ids:
                continue

            members = []
            for member_index, member_type_id in enumerate(
                type_info.get("member_types", [])
            ):
                data_type, array_suffix = self.spirv_type_name_and_suffix(
                    member_type_id, types, constants, names=names
                )
                if data_type is None:
                    continue

                member_key = str(member_index)
                member_layout_decorations = self.spirv_member_decorations_for_member(
                    member_decorations, type_id, member_key
                )
                member_name = None
                if type_id in builtin_interface_struct_storage_classes:
                    member_name = self.spirv_builtin_variable_name_from_qualifiers(
                        self.spirv_layout_qualifiers(member_layout_decorations),
                        storage_class=(
                            builtin_interface_struct_storage_classes.get(type_id)
                        ),
                    )
                if member_name is None:
                    member_name = member_names.get(type_id, {}).get(
                        member_key, f"member{member_key}"
                    )
                member_node = VariableNode(data_type, f"{member_name}{array_suffix}")
                if type_id in builtin_interface_struct_storage_classes:
                    member_node.declaration_qualifiers = (
                        self.spirv_declaration_qualifiers(member_layout_decorations)
                    )
                members.append(member_node)

            if members:
                signature = (
                    type_info["name"],
                    tuple((member.vtype, member.name) for member in members),
                )
                if signature in seen_struct_signatures:
                    continue
                seen_struct_signatures.add(signature)
                struct = StructNode(type_info["name"], members)
                storage_class = builtin_interface_struct_storage_classes.get(type_id)
                if storage_class is not None:
                    struct.spirv_builtin_interface_storage_class = storage_class
                    struct.spirv_builtin_interface_block_name = (
                        self.spirv_builtin_interface_block_name(
                            type_info["name"], type_id, member_decorations
                        )
                    )
                structs.append(struct)

        return structs

    def spirv_builtin_interface_struct_storage_classes(
        self, variables, types, member_decorations
    ):
        storage_classes = {}
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            if pointer_type.get("kind") != "pointer":
                continue

            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if storage_class not in self.SPIRV_INTERFACE_STORAGE_CLASSES:
                continue

            struct_type_id = pointer_type.get("type_id")
            struct_type = types.get(struct_type_id, {})
            if struct_type.get("kind") != "struct":
                continue
            if not self.spirv_struct_has_builtin_member_decorations(
                struct_type_id, member_decorations
            ):
                continue

            storage_classes.setdefault(struct_type_id, storage_class)
        return storage_classes

    def spirv_struct_has_builtin_member_decorations(
        self, struct_type_id, member_decorations
    ):
        return any(
            decoration == "BuiltIn"
            for _member, decoration, _operands in member_decorations.get(
                struct_type_id, []
            )
        )

    def spirv_member_decorations_for_member(
        self, member_decorations, struct_type_id, member_key
    ):
        return [
            (decoration, operands)
            for member, decoration, operands in member_decorations.get(
                struct_type_id, []
            )
            if member == member_key
        ]

    def spirv_builtin_interface_block_name(
        self, fallback_name, struct_type_id, member_decorations
    ):
        builtin_names = {
            operands[0]
            for _member, decoration, operands in member_decorations.get(
                struct_type_id, []
            )
            if decoration == "BuiltIn" and operands
        }
        per_vertex_builtins = {
            "Position",
            "PointSize",
            "ClipDistance",
            "CullDistance",
        }
        if builtin_names and builtin_names.issubset(per_vertex_builtins):
            return "gl_PerVertex"
        return fallback_name

    def spirv_resource_block_struct_type_ids(self, variables, types, decorations):
        resource_block_type_ids = set()
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            if pointer_type.get("kind") != "pointer":
                continue

            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if storage_class not in {"PushConstant", "StorageBuffer", "Uniform"}:
                continue

            struct_type_id = pointer_type.get("type_id")
            array_element = block_array_element(struct_type_id, types)
            if array_element is not None:
                struct_type_id = array_element[0]
            struct_type = types.get(struct_type_id, {})
            if struct_type.get("kind") != "struct":
                continue

            struct_decorations = decorations.get(struct_type_id, [])
            has_block = self.spirv_has_decoration(struct_decorations, "Block")
            has_buffer_block = self.spirv_has_decoration(
                struct_decorations, "BufferBlock"
            )

            if storage_class == "PushConstant" and has_block:
                resource_block_type_ids.add(struct_type_id)
            elif storage_class == "StorageBuffer" and has_block:
                resource_block_type_ids.add(struct_type_id)
            elif storage_class == "Uniform" and (has_block or has_buffer_block):
                resource_block_type_ids.add(struct_type_id)

        return resource_block_type_ids

    def spirv_spec_constant_id(self, decorations):
        for decoration, operands in decorations:
            if decoration == "SpecId" and operands:
                return operands[0]
        return None

    def spirv_constant_composite_expression(
        self, type_id, constituent_ids, names, types, constants
    ):
        type_name = self.spirv_type_name(type_id, types)
        args = [
            self.spirv_constant_operand_expression(constituent_id, names, constants)
            for constituent_id in constituent_ids
        ]
        return FunctionCallNode(type_name or "spirv_constant_composite", args)

    def spirv_spec_constant_op_expression(
        self, operation, operands, names, constants, types
    ):
        if (
            operation in {
                "CooperativeMatrixLengthKHR",
                "CooperativeMatrixLengthNV",
            }
            and operands
        ):
            length = self.spirv_cooperative_matrix_length(operands[0], types, constants)
            if length is not None:
                return str(length)
            return "1"

        args = [
            self.spirv_constant_operand_expression(operand, names, constants)
            for operand in operands
        ]
        if operation in self.SPIRV_SPEC_CONSTANT_BINARY_OPS and len(args) == 2:
            return BinaryOpNode(
                args[0], self.SPIRV_SPEC_CONSTANT_BINARY_OPS[operation], args[1]
            )
        if operation in self.SPIRV_SPEC_CONSTANT_UNARY_OPS and len(args) == 1:
            return UnaryOpNode(self.SPIRV_SPEC_CONSTANT_UNARY_OPS[operation], args[0])
        if operation == "Select" and len(args) == 3:
            return TernaryOpNode(args[0], args[1], args[2])
        if operation == "CompositeExtract" and len(args) >= 2:
            expression = args[0]
            for index in args[1:]:
                extracted = self.spirv_static_composite_element(expression, index)
                expression = (
                    extracted
                    if extracted is not None
                    else ArrayAccessNode(expression, index)
                )
            return expression
        return FunctionCallNode(f"spirv_{operation}", args)

    def spirv_static_composite_element(self, expression, index):
        try:
            element_index = int(str(index), 0)
        except (TypeError, ValueError):
            return None

        if isinstance(expression, FunctionCallNode):
            elements = expression.args
        elif isinstance(expression, InitializerListNode):
            elements = expression.elements
        else:
            return None

        if 0 <= element_index < len(elements):
            return elements[element_index]
        return None

    def spirv_constant_operand_expression(self, operand, names, constants):
        value = constants.get(operand)
        if self.spirv_is_generated_constant_expression(value):
            return value
        if names and operand in names:
            return names[operand]
        if value is not None:
            return value
        if isinstance(operand, str) and operand.startswith("%"):
            return operand.lstrip("%")
        return operand

    def spirv_constant_literal_expression(self, type_id, literal, types):
        type_info = types.get(type_id, {})
        if type_info.get("name") == "uint" and str(literal).isdigit():
            # CrossGL reads an unsuffixed literal as int; one above INT_MAX would widen to a 64-bit integer.
            return f"{literal}u" if int(str(literal)) > 0x7FFFFFFF else literal
        if type_info.get("name") not in {"half", "float", "double"}:
            return literal

        text = str(literal)
        stripped = text.lstrip("+-")
        if stripped.isdigit():
            # spirv-dis prints integral floating-point values without a fraction; CrossGL reads those as integers.
            return f"{text}.0"
        if not stripped.lower().startswith("0x") or "p" not in stripped.lower():
            return literal

        try:
            float.fromhex(text)
        except OverflowError:
            return "-1e+309" if text.startswith("-") else "1e+309"
        except ValueError:
            return literal
        return literal

    def spirv_integer_constant_operand(self, operand, constants):
        value = constants.get(operand, operand)
        try:
            return int(str(value), 0)
        except (TypeError, ValueError):
            return None

    def spirv_constant_expression_text(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, VariableNode):
            return value.name
        if isinstance(value, BinaryOpNode):
            left = self.spirv_constant_expression_text(value.left)
            right = self.spirv_constant_expression_text(value.right)
            return f"({left} {value.op} {right})"
        if isinstance(value, UnaryOpNode):
            operand = self.spirv_constant_expression_text(value.operand)
            return f"{value.op}{operand}"
        if isinstance(value, TernaryOpNode):
            condition = self.spirv_constant_expression_text(value.condition)
            true_expr = self.spirv_constant_expression_text(value.true_expr)
            false_expr = self.spirv_constant_expression_text(value.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        if isinstance(value, FunctionCallNode):
            args = ", ".join(
                self.spirv_constant_expression_text(arg) for arg in value.args
            )
            return f"{value.name}({args})"
        if isinstance(value, ArrayAccessNode):
            array = self.spirv_constant_expression_text(value.array)
            index = self.spirv_constant_expression_text(value.index)
            return f"{array}[{index}]"
        return str(value)

    def spirv_string_literal(self, value):
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def spirv_fallback_identifier(self, raw_value, prefix):
        identifier = re.sub(r"\W", "_", str(raw_value or "").lstrip("%"))
        if not identifier:
            return self.SPIRV_VALUE_FALLBACK_RENAMES.get(prefix, prefix)
        if identifier[0].isdigit():
            identifier = f"{prefix}_{identifier}"
        if prefix == "part" and identifier.endswith("_"):
            identifier = identifier.rstrip("_") or prefix
        return self.SPIRV_VALUE_FALLBACK_RENAMES.get(identifier, identifier)

    def spirv_identifier_name(self, raw_name, fallback_value=None, prefix="value"):
        name = str(raw_name or "")
        signature_index = name.find("(")
        if signature_index > 0:
            name = name[:signature_index]

        identifier = re.sub(r"[^0-9A-Za-z_]", "_", name)
        if not identifier:
            identifier = self.spirv_fallback_identifier(fallback_value, prefix)
        if identifier and identifier[0].isdigit():
            identifier = f"{prefix}_{identifier}"
        if identifier in self.CROSSGL_RESERVED_IDENTIFIERS:
            identifier = f"{identifier}_"
        return identifier

    def spirv_assembly_resource_block_layout(
        self,
        variable,
        pointer_type,
        storage_class,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
    ):
        struct_type_id = pointer_type.get("type_id")
        instance_suffix = ""
        array_element = block_array_element(struct_type_id, types)
        if array_element is not None:
            struct_type_id, length_id = array_element
            instance_suffix = "[]"
            if length_id is not None:
                length = self.spirv_constant_expression_text(
                    constants.get(length_id, str(length_id).lstrip("%"))
                )
                instance_suffix = f"[{length}]"
        struct_type = types.get(struct_type_id, {})
        if struct_type.get("kind") != "struct":
            return None

        struct_decorations = decorations.get(struct_type_id, [])
        variable_id = variable["id"]
        variable_decorations = decorations.get(variable_id, [])
        has_block = self.spirv_has_decoration(struct_decorations, "Block")
        has_buffer_block = self.spirv_has_decoration(struct_decorations, "BufferBlock")
        if storage_class == "PushConstant":
            if not has_block:
                return None
            layout_type = "UNIFORM"
        elif storage_class == "Uniform":
            if has_buffer_block:
                layout_type = "BUFFER"
            elif has_block:
                layout_type = "UNIFORM"
            else:
                return None
        elif storage_class == "StorageBuffer":
            if not has_block:
                return None
            layout_type = "BUFFER"
        elif storage_class in self.SPIRV_SHADER_RECORD_STORAGE_CLASSES:
            if not has_block:
                return None
            layout_type = "SHADER_RECORD"
        else:
            return None
        if instance_suffix and layout_type != "BUFFER":
            raise SpirvBlockArrayUnsupportedError(
                self.spirv_assembly_value_name(variable_id, names, decorations)
                or variable_id,
                storage_class,
            )

        struct_fields = []
        struct_field_indices = []
        for member_index, member_type_id in enumerate(
            struct_type.get("member_types", [])
        ):
            data_type, array_suffix = self.spirv_type_name_and_suffix(
                member_type_id, types, constants
            )
            if data_type is None:
                continue
            struct_field_indices.append(member_index)

            member_key = str(member_index)
            field_name = variable.get("flattened_member_aliases", {}).get(
                member_key
            ) or member_names.get(struct_type_id, {}).get(
                member_key, f"member{member_key}"
            )
            struct_fields.append((data_type, f"{field_name}{array_suffix}"))

        if not struct_fields:
            return None

        variable_name = self.spirv_assembly_value_name(variable_id, names, decorations)
        block_name = (
            names.get(struct_type_id)
            or variable_name
            or self.spirv_fallback_identifier(struct_type_id, "block")
        )
        qualifiers = []
        if storage_class in {"StorageBuffer", "Uniform"}:
            qualifiers = self.spirv_descriptor_qualifiers(variable_decorations)
            qualifier_names = {name for name, _value in qualifiers}
            if not {"set", "binding"}.issubset(qualifier_names):
                return None
        declaration_qualifiers = self.spirv_declaration_qualifiers(
            struct_decorations + variable_decorations
        )

        node = LayoutNode(
            qualifiers,
            layout_type=layout_type,
            push_constant=storage_class == "PushConstant",
            block_name=block_name,
            variable_name=variable_name,
            struct_fields=struct_fields,
            declaration_qualifiers=declaration_qualifiers,
            spirv_id=variable_id,
            spirv_decorations=struct_decorations + variable_decorations,
            spirv_storage_class=storage_class,
        )
        node.instance_array_suffix = instance_suffix
        if storage_class in {"Uniform", "PushConstant"} and layout_type == "UNIFORM":
            block_layout = spirv_block_layout(
                block_name,
                struct_type_id,
                storage_class == "PushConstant",
                types,
                decorations,
                member_decorations,
                lambda length_id: self.spirv_integer_constant_operand(
                    length_id, constants
                ),
            )
            if block_layout is not None:
                node.buffer_layout = block_layout.layout
                if block_layout.member_offsets is not None:
                    # One explicit offset per emitted field, in struct_fields order.
                    node.member_offsets = tuple(
                        block_layout.member_offsets[index]
                        for index in struct_field_indices
                    )
        return node

    def spirv_has_decoration(self, decorations, target_decoration):
        return any(decoration == target_decoration for decoration, _ in decorations)

    def spirv_assign_flattened_resource_block_member_aliases(
        self,
        variables,
        names,
        decorations,
        member_names,
        types,
        constants,
    ):
        occurrences = []
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            if pointer_type.get("kind") != "pointer":
                continue

            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            struct_type_id = pointer_type.get("type_id")
            struct_type = types.get(struct_type_id, {})
            if struct_type.get("kind") != "struct":
                continue
            if not self.spirv_is_flattened_resource_block_access(
                storage_class, struct_type_id, decorations
            ):
                continue
            if storage_class == "Uniform" and not self.spirv_has_descriptor_binding(
                decorations.get(variable["id"], [])
            ):
                continue

            variable_name = self.spirv_assembly_value_name(
                variable["id"], names, decorations
            )
            for member_index, member_type_id in enumerate(
                struct_type.get("member_types", [])
            ):
                data_type, _array_suffix = self.spirv_type_name_and_suffix(
                    member_type_id, types, constants
                )
                if data_type is None:
                    continue

                member_key = str(member_index)
                member_name = member_names.get(struct_type_id, {}).get(
                    member_key, f"member{member_key}"
                )
                occurrences.append(
                    {
                        "variable": variable,
                        "variable_name": variable_name,
                        "member_key": member_key,
                        "member_name": member_name,
                    }
                )

        member_counts = {}
        for occurrence in occurrences:
            member_counts[occurrence["member_name"]] = (
                member_counts.get(occurrence["member_name"], 0) + 1
            )
        duplicate_members = {
            member_name for member_name, count in member_counts.items() if count > 1
        }
        if not duplicate_members:
            return

        used_names = {
            occurrence["member_name"]
            for occurrence in occurrences
            if occurrence["member_name"] not in duplicate_members
        }
        for occurrence in occurrences:
            if occurrence["member_name"] not in duplicate_members:
                continue

            alias_base = self.spirv_identifier_name(
                f"{occurrence['variable_name']}_{occurrence['member_name']}",
                f"{occurrence['variable']['id']}_{occurrence['member_key']}",
                prefix="member",
            )
            alias = self.spirv_unique_identifier(alias_base, used_names)
            used_names.add(alias)
            occurrence["variable"].setdefault("flattened_member_aliases", {})[
                occurrence["member_key"]
            ] = alias

    def spirv_has_descriptor_binding(self, decorations):
        qualifier_names = {
            name for name, _value in self.spirv_descriptor_qualifiers(decorations)
        }
        return {"set", "binding"}.issubset(qualifier_names)

    def spirv_unique_identifier(self, base_identifier, used_identifiers):
        identifier = base_identifier
        suffix = 1
        while identifier in used_identifiers:
            identifier = f"{base_identifier}_{suffix}"
            suffix += 1
        return identifier

    def spirv_descriptor_qualifiers(self, decorations):
        qualifiers = []
        for decoration, operands in decorations:
            if not operands:
                continue
            if decoration == "DescriptorSet":
                qualifiers.append(("set", operands[0]))
            elif decoration == "Binding":
                qualifiers.append(("binding", operands[0]))
        return qualifiers

    def spirv_assembly_struct_interface_layouts(
        self,
        variable,
        pointer_type,
        storage_class,
        names,
        variable_decorations,
        member_decorations,
        member_names,
        types,
        constants,
    ):
        struct_type_id = pointer_type.get("type_id")
        struct_type = types.get(struct_type_id, {})
        if struct_type.get("kind") != "struct":
            return []

        layouts = []
        block_name = self.spirv_assembly_value_name(
            variable["id"], names, decorations={}
        )
        variable_qualifiers = self.spirv_layout_qualifiers(variable_decorations)
        variable_declaration_qualifiers = self.spirv_declaration_qualifiers(
            variable_decorations
        )
        location_cursor = self.spirv_interface_base_location(variable_qualifiers)
        for member_index, member_type_id in enumerate(
            struct_type.get("member_types", [])
        ):
            member_key = str(member_index)
            member_layout_decorations = [
                (decoration, operands)
                for member, decoration, operands in member_decorations.get(
                    struct_type_id, []
                )
                if member == member_key
            ]
            qualifiers = self.spirv_layout_qualifiers(member_layout_decorations)
            declaration_qualifiers = self.spirv_declaration_qualifiers(
                member_layout_decorations
            )
            if not self.spirv_has_interface_qualifier(
                qualifiers, declaration_qualifiers
            ):
                if location_cursor is None:
                    continue
                qualifiers = self.spirv_interface_member_location_qualifiers(
                    variable_qualifiers, location_cursor
                )
                declaration_qualifiers = list(variable_declaration_qualifiers)

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                member_type_id, types, constants
            )
            if data_type is None:
                continue

            variable_name = self.spirv_struct_member_variable_name(
                struct_type_id,
                member_key,
                block_name,
                qualifiers,
                member_names,
                storage_class=storage_class,
            )
            variable_name += array_suffix
            layouts.append(
                LayoutNode(
                    qualifiers,
                    layout_type=self.SPIRV_INTERFACE_STORAGE_CLASSES[storage_class],
                    data_type=data_type,
                    variable_name=variable_name,
                    declaration_qualifiers=declaration_qualifiers,
                    spirv_id=f"{variable['id']}.{member_key}",
                    spirv_decorations=member_layout_decorations,
                    spirv_storage_class=storage_class,
                )
            )
            if location_cursor is not None:
                location_cursor += self.spirv_interface_location_count(
                    member_type_id, types, constants
                )

        return layouts

    def spirv_interface_base_location(self, qualifiers):
        for name, value in qualifiers:
            if name != "location":
                continue
            try:
                return int(str(value), 0)
            except (TypeError, ValueError):
                return None
        return None

    def spirv_interface_member_location_qualifiers(self, variable_qualifiers, location):
        qualifiers = [
            (name, value) for name, value in variable_qualifiers if name != "location"
        ]
        qualifiers.append(("location", str(location)))
        return qualifiers

    def spirv_interface_location_count(self, type_id, types, constants):
        type_info = types.get(type_id, {})
        kind = type_info.get("kind")

        if kind == "array":
            length = self.spirv_integer_constant_operand(
                type_info.get("length_id"), constants
            )
            if length is None:
                return 1
            return max(
                1,
                length
                * self.spirv_interface_location_count(
                    type_info.get("element_type"), types, constants
                ),
            )

        if kind == "matrix":
            try:
                return max(1, int(str(type_info.get("column_count")), 0))
            except (TypeError, ValueError):
                return 1

        if kind == "struct":
            count = sum(
                self.spirv_interface_location_count(member_type_id, types, constants)
                for member_type_id in type_info.get("member_types", [])
            )
            return max(1, count)

        return 1

    def spirv_layout_qualifiers(self, decorations):
        qualifiers = []
        for decoration, operands in decorations:
            qualifier_name = self.SPIRV_INTERFACE_DECORATIONS.get(decoration)
            if qualifier_name and operands:
                qualifiers.append((qualifier_name, operands[0]))
        return qualifiers

    def spirv_declaration_qualifiers(self, decorations):
        qualifiers = []
        for decoration, _operands in decorations:
            qualifier = self.SPIRV_DECLARATION_DECORATION_QUALIFIERS.get(decoration)
            if qualifier:
                qualifiers.append(qualifier)
        return qualifiers

    def spirv_image_format_qualifier(self, image_format):
        if not image_format or image_format == "Unknown":
            return None
        return str(image_format).replace("Snorm", "_snorm").lower()

    def spirv_image_access_qualifier(self, access_qualifier):
        return {"ReadOnly": "readonly", "WriteOnly": "writeonly"}.get(access_qualifier)

    def spirv_has_interface_qualifier(self, qualifiers, declaration_qualifiers=None):
        if any(name in {"builtin", "location"} for name, _value in qualifiers):
            return True
        declaration_qualifiers = set(declaration_qualifiers or [])
        return bool(
            declaration_qualifiers
            & {
                "centroid",
                "flat",
                "invariant",
                "noperspective",
                "patch",
                "pervertexEXT",
                "sample",
            }
        )

    def spirv_struct_member_variable_name(
        self,
        struct_type_id,
        member_key,
        block_name,
        qualifiers,
        member_names,
        storage_class=None,
    ):
        member_name = member_names.get(struct_type_id, {}).get(member_key)
        if member_name:
            return member_name

        builtin_name = self.spirv_builtin_variable_name_from_qualifiers(
            qualifiers, storage_class=storage_class
        )
        if builtin_name:
            return builtin_name

        if block_name:
            return f"{block_name}_{member_key}"
        return f"member{member_key}"

    def spirv_builtin_variable_name_from_qualifiers(
        self, qualifiers, fallback_name=None, storage_class=None
    ):
        for name, value in qualifiers:
            if name == "builtin":
                storage_builtin_name = self.SPIRV_STORAGE_BUILTIN_VARIABLE_NAMES.get(
                    (value, storage_class)
                )
                if storage_builtin_name:
                    return storage_builtin_name
                return self.SPIRV_BUILTIN_VARIABLE_NAMES.get(value, value)
        return fallback_name

    def spirv_type_name_and_suffix(self, type_id, types, constants, names=None):
        type_info = types.get(type_id)
        if type_info is None:
            return None, ""

        if type_info.get("kind") == "array":
            base_type, suffix = self.spirv_type_name_and_suffix(
                type_info.get("element_type"), types, constants, names=names
            )
            if base_type is None:
                return None, ""
            length_id = type_info.get("length_id")
            length = self.spirv_constant_expression_text(
                constants.get(length_id, str(length_id).lstrip("%"))
            )
            return base_type, f"[{length}]{suffix}"

        if type_info.get("kind") == "runtime_array":
            base_type, suffix = self.spirv_type_name_and_suffix(
                type_info.get("element_type"), types, constants, names=names
            )
            if base_type is None:
                return None, ""
            return base_type, f"[]{suffix}"

        if type_info.get("kind") == "pointer":
            if names and names.get(type_id):
                return names[type_id], ""
            base_type, suffix = self.spirv_type_name_and_suffix(
                type_info.get("type_id"), types, constants, names=names
            )
            if base_type is None:
                return self.spirv_fallback_identifier(type_id, "ptr"), ""
            return f"{base_type}*", suffix

        return type_info.get("name"), ""

    def spirv_type_name(self, type_id, types):
        type_info = types.get(type_id)
        if type_info is None:
            return None
        return type_info.get("name")

    def spirv_sampled_image_type_name(self, image_type, types):
        sampled_type = self.spirv_type_name(image_type.get("sampled_type"), types)
        return self.spirv_image_type_name(
            sampled_type,
            image_type.get("dim"),
            image_type.get("depth"),
            image_type.get("arrayed"),
            image_type.get("multisampled"),
            "1",
        )

    def spirv_image_query_name(
        self, sampled_name, storage_name, image_operand, expression_type_ids, types
    ):
        """GLSL spells a size or sample-count query of a storage image (``Sampled`` 2) ``imageSize``/``imageSamples``.

        ``OpImageQuerySize`` and ``OpImageQuerySamples`` take an image of either kind (SPIR-V 1.6 section 3.49.10);
        ``textureSize`` of an ``image2D`` is not a GLSL overload, and the emitters refuse it.
        """
        image_type = types.get(expression_type_ids.get(image_operand), {})
        if image_type.get("kind") == "image" and str(image_type.get("sampled")) == "2":
            return storage_name
        return sampled_name

    def spirv_image_type_name(
        self, sampled_type, dim, depth, arrayed, multisampled, sampled
    ):
        if dim == "SubpassData":
            return "subpassInputMS" if multisampled == "1" else "subpassInput"

        base_name = self.spirv_image_base_type_name(dim, arrayed, multisampled)
        if base_name is None:
            return None

        sampled_family = self.spirv_image_sampled_type_family(sampled_type)
        if sampled == "2":
            prefix = {"int": "i", "uint": "u"}.get(sampled_family, "")
            return f"{prefix}image{base_name}"

        prefix = {"int": "i", "uint": "u"}.get(sampled_family, "")
        suffix = "Shadow" if depth == "1" and not prefix else ""
        return f"{prefix}sampler{base_name}{suffix}"

    def spirv_image_sampled_type_family(self, sampled_type):
        if sampled_type in {"int", "i8", "i16", "i64"}:
            return "int"
        if sampled_type in {"uint", "u8", "u16", "u64"}:
            return "uint"
        return sampled_type

    def spirv_image_base_type_name(self, dim, arrayed, multisampled):
        if dim == "Buffer":
            return "Buffer"
        if dim == "Cube":
            return "CubeArray" if arrayed == "1" else "Cube"
        if dim == "1D":
            return "1DArray" if arrayed == "1" else "1D"
        if dim == "2D":
            if multisampled == "1":
                return "2DMSArray" if arrayed == "1" else "2DMS"
            return "2DArray" if arrayed == "1" else "2D"
        if dim == "3D":
            return "3D"
        return None

    def spirv_matrix_type_name(self, component_type, row_count, column_count):
        if not component_type or not row_count or not column_count:
            return None

        if component_type == "half":
            return f"half{column_count}x{row_count}"

        prefix = {"double": "dmat", "float": "mat"}.get(component_type)
        if prefix is None:
            return None

        if row_count == column_count:
            return f"{prefix}{column_count}"
        return f"{prefix}{column_count}x{row_count}"

    def spirv_cooperative_matrix_type_name(self, opcode, operands, types, constants):
        component_type = self.spirv_type_name(operands[0], types) or operands[0]
        variant = "khr" if opcode.endswith("KHR") else "nv"
        parts = [
            "spirvCoopMat",
            variant,
            component_type,
            "scope",
            self.spirv_cooperative_matrix_dimension_name(operands[1], constants),
            "rows",
            self.spirv_cooperative_matrix_dimension_name(operands[2], constants),
            "cols",
            self.spirv_cooperative_matrix_dimension_name(operands[3], constants),
        ]
        if len(operands) >= 5:
            parts.extend(
                [
                    "use",
                    self.spirv_cooperative_matrix_dimension_name(
                        operands[4], constants
                    ),
                ]
            )
        return "_".join(self.spirv_fallback_identifier(part, "part") for part in parts)

    def spirv_cooperative_matrix_dimension_name(self, operand, constants):
        value = constants.get(operand, operand)
        text = self.spirv_constant_expression_text(value)
        return self.spirv_fallback_identifier(text, "value")

    def spirv_cooperative_matrix_length(self, type_id, types, constants):
        type_info = types.get(type_id, {})
        if type_info.get("kind") != "cooperative_matrix":
            return None
        rows = self.spirv_cooperative_matrix_static_dimension(
            type_info.get("rows"), constants
        )
        columns = self.spirv_cooperative_matrix_static_dimension(
            type_info.get("columns"), constants
        )
        if rows is None or columns is None:
            return None
        return rows * columns

    def spirv_cooperative_matrix_static_dimension(self, operand, constants):
        value = constants.get(operand, operand)
        try:
            return int(str(value), 0)
        except (TypeError, ValueError):
            return None

    def spirv_float_type_name(self, width):
        if width == "16":
            return "half"
        if width == "64":
            return "double"
        return "float"

    def spirv_int_type_name(self, width, signedness):
        if width == "1":
            return "bool"
        if width in {"8", "16", "64"}:
            prefix = "u" if signedness == "0" else "i"
            return f"{prefix}{width}"
        if signedness == "0":
            return "uint"
        return "int"

    def spirv_vector_type_name(self, component_type, component_count):
        if not component_type or not component_count:
            return None

        mapped_type = self.SPIRV_VECTOR_TYPES.get((component_type, component_count))
        if mapped_type:
            return mapped_type

        if component_type in {"i8", "u8", "i16", "u16", "i64", "u64"}:
            return f"vec{component_count}<{component_type}>"

        return f"{component_type}{component_count}"

    def parse_precision_declaration(self):
        self.eat("PRECISION")
        if self.current_token[0] in self.PRECISION_QUALIFIER_TOKENS:
            self.eat(self.current_token[0])

        if self.current_token[1] not in VALID_DATA_TYPES:
            raise SyntaxError(f"Unexpected precision type: {self.current_token[1]}")
        self.eat(self.current_token[0])
        self.eat("SEMICOLON")

    def parse_layout(self):
        self.eat("LAYOUT")
        self.eat("LPAREN")
        bindings = []
        push_constant = False
        while self.current_token[0] != "RPAREN":
            if self.current_token[0] == "PUSH_CONSTANT":
                push_constant = True
                self.eat("PUSH_CONSTANT")
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                continue

            binding_name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                binding_value = self.parse_layout_qualifier_value()
                bindings.append((binding_name, binding_value))
            else:
                bindings.append((binding_name, None))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RPAREN")

        declaration_qualifiers = self.parse_layout_declaration_qualifiers()
        if (
            self.has_specialization_constant_qualifier(bindings)
            and self.current_token[0] == "CONST"
        ):
            declaration = self.parse_assignment_or_function_call()
            return LayoutNode(
                bindings,
                declaration=declaration,
                layout_type="CONST",
                declaration_qualifiers=declaration_qualifiers,
            )

        layout_type = None
        block_name = None
        if self.current_token[0] in ["IN", "OUT", "UNIFORM", "BUFFER"]:
            layout_type = self.current_token[0]
            self.eat(layout_type)
            declaration_qualifiers.extend(self.parse_layout_declaration_qualifiers())
            if self.current_token[0] == "IDENTIFIER" and self.peek(1) == "LBRACE":
                block_name = self.current_token[1]
                self.eat(self.current_token[0])

        data_type = None
        struct_fields = None
        if layout_type in ["UNIFORM", "BUFFER"]:
            if self.current_token[0] == "LBRACE":
                struct_fields = self.parse_layout_block_fields()
                data_type = "struct"
            elif self.is_data_type_token(allow_identifier=True):
                data_type = self.parse_data_type(allow_identifier=True)
            else:
                raise SyntaxError(
                    "Expected structured data block after 'uniform' or 'buffer'"
                )
        else:
            if layout_type in ["IN", "OUT"] and self.current_token[0] == "SEMICOLON":
                pass
            elif (
                layout_type in ["IN", "OUT"]
                and block_name
                and self.current_token[0] == "LBRACE"
            ):
                struct_fields = self.parse_layout_block_fields()
                data_type = "struct"
            elif self.is_data_type_token(allow_identifier=True):
                data_type = self.parse_data_type(allow_identifier=True)
            else:
                raise SyntaxError(f"Unexpected type: {self.current_token[1]}")

        variable_name = None
        if self.current_token[0] == "IDENTIFIER":
            variable_name = self.current_token[1]
            self.eat("IDENTIFIER")
            variable_name += self.parse_array_suffixes_as_text()

        self.eat("SEMICOLON")
        return LayoutNode(
            bindings,
            push_constant=push_constant,
            layout_type=layout_type,
            data_type=data_type,
            variable_name=variable_name,
            struct_fields=struct_fields,
            block_name=block_name,
            declaration_qualifiers=declaration_qualifiers,
        )

    def parse_layout_qualifier_value(self):
        parts = []
        depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if depth == 0 and token_type in {"COMMA", "RPAREN"}:
                break

            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1

            parts.append(self.current_token[1])
            self.eat(token_type)

        if not parts:
            raise SyntaxError(
                f"Expected layout qualifier value, got {self.current_token[0]}"
            )
        if depth:
            raise SyntaxError("Unterminated layout qualifier value")

        return self.format_layout_qualifier_value(parts)

    def format_layout_qualifier_value(self, parts):
        text = " ".join(str(part) for part in parts)
        return (
            text.replace("( ", "(")
            .replace(" )", ")")
            .replace("[ ", "[")
            .replace(" ]", "]")
            .replace(" . ", ".")
            .replace(" ,", ",")
            .replace(", ", ", ")
        )

    def has_specialization_constant_qualifier(self, qualifiers):
        return any(str(name).lower() == "constant_id" for name, _ in qualifiers)

    def is_data_type_token(self, allow_identifier=False):
        return self.current_token[1] in VALID_DATA_TYPES or (
            allow_identifier and self.current_token[0] == "IDENTIFIER"
        )

    def is_data_type_token_at(self, index, allow_identifier=False):
        if index >= len(self.tokens):
            return False
        token_type, token_value = self.tokens[index]
        return token_value in VALID_DATA_TYPES or (
            allow_identifier and token_type == "IDENTIFIER"
        )

    def parse_data_type(self, allow_identifier=False, error_message=None):
        if not self.is_data_type_token(allow_identifier=allow_identifier):
            raise SyntaxError(
                error_message or f"Unexpected type: {self.current_token[1]}"
            )

        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        type_name += self.parse_type_template_suffix()
        type_name += self.parse_array_suffixes_as_text()
        return type_name

    def parse_type_template_suffix(self):
        if self.current_token[0] != "LESS_THAN":
            return ""

        self.eat("LESS_THAN")
        parts = []
        depth = 1
        while depth:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated type template argument list")

            if self.current_token[0] == "LESS_THAN":
                depth += 1
                parts.append("<")
                self.eat("LESS_THAN")
                continue

            if self.current_token[0] == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    break
                parts.append(">")
                self.eat("GREATER_THAN")
                continue

            parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        return f"<{self.format_type_template_parts(parts)}>"

    def format_type_template_parts(self, parts):
        text = " ".join(str(part) for part in parts)
        return (
            text.replace(" ,", ",")
            .replace(", ", ", ")
            .replace("< ", "<")
            .replace(" >", ">")
        )

    def skip_type_template_suffix_at_pos(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "LESS_THAN":
            return index

        depth = 1
        index += 1
        while index < len(self.tokens) and depth:
            token_type = self.tokens[index][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
            index += 1
        return index if depth == 0 else len(self.tokens)

    def skip_array_suffixes_at_pos(self, index):
        while index < len(self.tokens) and self.tokens[index][0] == "LBRACKET":
            depth = 1
            index += 1
            while index < len(self.tokens) and depth:
                token_type = self.tokens[index][0]
                if token_type == "LBRACKET":
                    depth += 1
                elif token_type == "RBRACKET":
                    depth -= 1
                index += 1
            if depth:
                return len(self.tokens)
        return index

    def skip_type_suffixes_at_pos(self, index):
        index = self.skip_type_template_suffix_at_pos(index)
        return self.skip_array_suffixes_at_pos(index)

    def is_geometry_input_primitive_qualifier_at(self, index):
        if (
            index >= len(self.tokens)
            or self.tokens[index][1] not in self.GEOMETRY_INPUT_PRIMITIVE_QUALIFIERS
        ):
            return False

        type_index = index + 1
        if not self.is_data_type_token_at(type_index, allow_identifier=True):
            return False

        declarator_index = self.skip_type_suffixes_at_pos(type_index + 1)
        return (
            declarator_index < len(self.tokens)
            and self.tokens[declarator_index][0] == "IDENTIFIER"
        )

    def parse_layout_declaration_qualifiers(self):
        qualifiers = []
        while self.current_token[1] in self.LAYOUT_DECLARATION_QUALIFIERS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_declaration_qualifiers(self):
        qualifiers = []
        while self.current_token[1] in self.DECLARATION_QUALIFIERS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_parameter_qualifiers(self):
        qualifiers = []
        while (
            self.current_token[0] in self.PARAMETER_QUALIFIER_TOKENS
            or self.current_token[1] in self.LAYOUT_DECLARATION_QUALIFIERS
            or self.is_geometry_input_primitive_qualifier_at(self.pos)
        ):
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_layout_block_fields(self):
        self.eat("LBRACE")
        struct_fields = []

        while self.current_token[0] != "RBRACE":
            self.skip_hlsl_attributes()
            while self.current_token[0] == "LAYOUT":
                self.skip_layout_annotation()
            self.parse_layout_declaration_qualifiers()
            field_type = self.parse_data_type(
                allow_identifier=True,
                error_message="Expected some data type before an identifier",
            )
            field_name = self.current_token[1]
            self.eat("IDENTIFIER")
            field_name += self.parse_array_suffixes_as_text()
            self.parse_optional_hlsl_semantic()
            self.eat("SEMICOLON")
            struct_fields.append((field_type, field_name))

        self.eat("RBRACE")
        return struct_fields

    def skip_layout_annotation(self):
        self.eat("LAYOUT")
        self.eat("LPAREN")
        depth = 1
        while depth and self.current_token[0] != "EOF":
            if self.current_token[0] == "LPAREN":
                depth += 1
            elif self.current_token[0] == "RPAREN":
                depth -= 1
            self.eat(self.current_token[0])
        if depth:
            raise SyntaxError("Unterminated layout annotation")

    def parse_push_constant(self):
        self.eat("PUSH_CONSTANT")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            members.append(self.parse_variable())
        self.eat("RBRACE")
        return PushConstantNode(members)

    def parse_descriptor_set(self):
        self.eat("DESCRIPTOR_SET")
        set_number = self.current_token[1]
        self.eat("NUMBER")
        self.eat("LBRACE")
        bindings = []
        while self.current_token[0] != "RBRACE":
            bindings.append(self.parse_variable())
        self.eat("RBRACE")
        return DescriptorSetNode(set_number, bindings)

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        members = []

        while self.current_token[0] != "RBRACE":
            self.skip_hlsl_attributes()
            while self.current_token[0] in self.PRECISION_QUALIFIER_TOKENS:
                self.eat(self.current_token[0])
            if self.current_token[0] in [
                "VEC2",
                "VEC3",
                "VEC4",
                "IVEC2",
                "IVEC3",
                "IVEC4",
                "UVEC2",
                "UVEC3",
                "UVEC4",
                "FLOAT",
                "INT",
                "UINT",
                "BOOL",
                "MAT2",
                "MAT3",
                "MAT4",
            ]:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[1] in VALID_DATA_TYPES:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[0] == "IDENTIFIER":
                type_name = self.current_token[1]
                self.eat("IDENTIFIER")
            else:
                raise SyntaxError(
                    f"Unexpected token in struct member: {self.current_token}"
                )
            type_name += self.parse_type_template_suffix()
            type_name += self.parse_array_suffixes_as_text()

            member_name = self.current_token[1]
            self.eat("IDENTIFIER")
            member_name += self.parse_array_suffixes_as_text()
            semantic = self.parse_optional_hlsl_semantic()

            self.eat("SEMICOLON")

            members.append(VariableNode(type_name, member_name, semantic=semantic))

        self.eat("RBRACE")
        while self.current_token[0] == "IDENTIFIER":
            self.eat("IDENTIFIER")
            self.parse_array_suffixes_as_text()
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")
        self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_function(self):
        qualifiers = self.parse_declaration_qualifiers()
        return_type = self.parse_data_type(allow_identifier=True)
        func_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        semantic = self.parse_optional_hlsl_semantic()
        body = self.parse_block()
        return FunctionNode(
            return_type,
            func_name,
            params,
            body,
            qualifiers=qualifiers,
            semantic=semantic,
        )

    def parse_parameters(self):
        params = []
        if self.current_token[0] == "VOID" and self.peek(1) == "RPAREN":
            self.eat("VOID")
            return params

        while self.current_token[0] != "RPAREN":
            self.skip_hlsl_attributes()
            qualifiers = self.parse_parameter_qualifiers()
            vtype = self.parse_data_type(allow_identifier=True)
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            name += self.parse_array_suffixes_as_text()
            semantic = self.parse_optional_hlsl_semantic()
            params.append(
                VariableNode(vtype, name, qualifiers=qualifiers, semantic=semantic)
            )
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        return params

    def parse_block(self):
        self.eat("LBRACE")
        statements = []
        while self.current_token[0] != "RBRACE":
            statement = self.parse_body()
            if isinstance(statement, list):
                statements.extend(statement)
            elif statement is not None:
                statements.append(statement)
        self.eat("RBRACE")
        return statements

    def parse_statement_or_block(self):
        if self.current_token[0] == "LBRACE":
            return self.parse_block()
        statement = self.parse_body()
        if isinstance(statement, list):
            return statement
        if statement is None:
            return []
        return [statement]

    def parse_body(self):
        token_type = self.current_token[0]

        if token_type == "SEMICOLON":
            self.eat("SEMICOLON")
            return []
        if token_type == "LBRACE":
            return self.parse_block()
        if token_type == "CONST":
            return self.parse_assignment_or_function_call()
        if self.current_token[1] in self.DECLARATION_QUALIFIERS:
            return self.parse_assignment_or_function_call()
        if token_type == "IDENTIFIER" and (
            self.peek(1) in ["LPAREN", "LBRACKET"]
            or self.looks_like_member_call_statement()
        ):
            return self.parse_expression_statement()
        if token_type == "IDENTIFIER" or self.current_token[1] in VALID_DATA_TYPES:
            return self.parse_assignment_or_function_call()
        elif token_type == "IF":
            return self.parse_if_statement()
        elif token_type == "FOR":
            return self.parse_for_statement()
        elif token_type == "WHILE":
            return self.parse_while_statement()
        elif token_type == "DO":
            return self.parse_do_while_statement()
        elif token_type == "SWITCH":
            return self.parse_switch_statement()
        elif token_type == "BREAK":
            if self.breakable_depth == 0:
                raise SyntaxError("break used outside loop or switch")
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return BreakNode()
        elif token_type == "CONTINUE":
            if self.loop_depth == 0:
                raise SyntaxError("continue used outside loop")
            self.eat("CONTINUE")
            self.eat("SEMICOLON")
            return ContinueNode()
        elif token_type == "RETURN":
            return self.parse_return_statement()
        elif token_type == "DISCARD":
            self.eat("DISCARD")
            self.eat("SEMICOLON")
            return DiscardNode()
        else:
            return self.parse_expression_statement()

    def parse_return_statement(self):
        self.eat("RETURN")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ReturnNode()

        value = self.parse_expression()
        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_update(self):
        if self.current_token[0] == "IDENTIFIER":
            target = self.parse_update_target()
            if self.current_token[0] == "POST_INCREMENT":
                self.eat("POST_INCREMENT")
                return UnaryOpNode("POST_INCREMENT", target)
            elif self.current_token[0] == "POST_DECREMENT":
                self.eat("POST_DECREMENT")
                return UnaryOpNode("POST_DECREMENT", target)
            elif self.current_token[0] in self.ASSIGNMENT_TOKENS:
                op_name = self.current_token[1]
                self.eat(self.current_token[0])
                value = self.parse_expression()
                return AssignmentNode(target, value, op_name)
            else:
                raise SyntaxError(
                    f"Unexpected token in update: {self.current_token[0]}"
                )
        elif self.current_token[0] == "PRE_INCREMENT":
            self.eat("PRE_INCREMENT")
            return UnaryOpNode("PRE_INCREMENT", self.parse_update_target())
        elif self.current_token[0] == "PRE_DECREMENT":
            self.eat("PRE_DECREMENT")
            return UnaryOpNode("PRE_DECREMENT", self.parse_update_target())
        else:
            raise SyntaxError(f"Unexpected token in update: {self.current_token[0]}")

    def parse_update_target(self):
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(f"Expected update target, got {self.current_token[0]}")

        target = VariableNode("", self.current_token[1])
        self.eat("IDENTIFIER")
        target = self.parse_postfix_suffixes(target)
        if not isinstance(target, (VariableNode, MemberAccessNode, ArrayAccessNode)):
            raise SyntaxError(f"Invalid update target: {type(target).__name__}")
        return target

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        if_condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_statement_or_block()
        else_body = None
        else_if_chain = []
        while self.current_token[0] == "ELSE" and self.peek(1) == "IF":
            self.eat("ELSE")
            self.eat("IF")
            self.eat("LPAREN")
            else_if_condition = self.parse_expression()
            self.eat("RPAREN")
            else_if_chain.append((else_if_condition, self.parse_statement_or_block()))
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement_or_block()
        return IfNode(
            if_condition,
            if_body,
            else_body,
            else_if_chain=else_if_chain,
        )

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")
        initialization = self.parse_for_initializer()
        condition = self.parse_for_condition()
        increment = self.parse_for_update()
        self.eat("RPAREN")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_statement_or_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        return ForNode(initialization, condition, increment, body)

    def parse_for_initializer(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None

        if self.current_token[0] in {"PRE_INCREMENT", "PRE_DECREMENT"}:
            item = self.parse_update()
            self.eat("SEMICOLON")
            return item

        items = [
            self.parse_assignment_or_function_call(
                terminators={"COMMA", "SEMICOLON"},
                consume_terminator=False,
            )
        ]
        declaration_type = self.for_initializer_declaration_type(items[0])
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if declaration_type and self.current_token[0] == "IDENTIFIER":
                items.append(
                    self.parse_variable(
                        declaration_type,
                        terminators={"COMMA", "SEMICOLON"},
                        consume_terminator=False,
                    )
                )
            else:
                items.append(
                    self.parse_assignment_or_function_call(
                        terminators={"COMMA", "SEMICOLON"},
                        consume_terminator=False,
                    )
                )
        self.eat("SEMICOLON")
        return items if len(items) > 1 else items[0]

    def for_initializer_declaration_type(self, item):
        target = item.left if isinstance(item, AssignmentNode) else item
        if isinstance(target, VariableNode) and target.vtype:
            return target.vtype
        return ""

    def parse_for_condition(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None

        condition = self.parse_expression()
        self.eat("SEMICOLON")
        return condition

    def parse_for_update(self):
        if self.current_token[0] == "RPAREN":
            return None
        updates = [self.parse_update()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            updates.append(self.parse_update())
        return updates if len(updates) > 1 else updates[0]

    def parse_variable(
        self,
        type_name="",
        terminators=None,
        consume_terminator=True,
        expected_terminators=None,
    ):
        terminators = terminators or {"SEMICOLON"}
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        if type_name:
            name += self.parse_array_suffixes_as_text()
        target = VariableNode(type_name, name)
        if not type_name:
            target = self.parse_postfix_suffixes(target)
        else:
            target.semantic = self.parse_optional_hlsl_semantic()

        if type_name in {"cbuffer", "tbuffer"} and self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_tokens()
            if consume_terminator and self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return target

        if self.current_token[0] in terminators:
            if consume_terminator:
                self.eat(self.current_token[0])
            return target

        elif self.current_token[0] in self.ASSIGNMENT_TOKENS:
            op_name = self.current_token[1]
            self.eat(self.current_token[0])
            value = self.parse_expression()
            self.consume_terminator(
                terminators,
                consume_terminator,
                expected_terminators=expected_terminators,
            )
            return AssignmentNode(target, value, op_name)

        elif self.current_token[0] in ("BINARY_AND", "BINARY_OR", "BINARY_XOR"):
            op = self.current_token[0]
            op_symbol = (
                "&" if op == "BINARY_AND" else ("|" if op == "BINARY_OR" else "^")
            )
            self.eat(op)
            right = self.parse_expression()
            self.consume_terminator(
                terminators,
                consume_terminator,
                expected_terminators=expected_terminators,
            )
            return BinaryOpNode(target, op_symbol, right)

        elif self.current_token[0] in (
            "EQUAL",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_XOR",
        ):
            op = self.current_token[0]
            op_name = self.current_token[1]
            self.eat(op)
            value = self.parse_expression()
            self.consume_terminator(
                terminators,
                consume_terminator,
                expected_terminators=expected_terminators,
            )
            return BinaryOpNode(target, op_name, value)
        else:
            raise SyntaxError(
                f"Unexpected token after identifier {name}: {self.current_token[0]}"
            )

    def parse_variable_declaration(
        self,
        type_name,
        terminators=None,
        consume_terminator=True,
    ):
        terminators = terminators or {"SEMICOLON"}
        declarator_terminators = {*terminators, "COMMA"}
        declarations = [
            self.parse_variable(
                type_name,
                terminators=declarator_terminators,
                consume_terminator=False,
                expected_terminators=terminators,
            )
        ]
        if (
            isinstance(declarations[0], VariableNode)
            and declarations[0].vtype in {"cbuffer", "tbuffer"}
            and self.current_token[0] not in declarator_terminators
        ):
            return declarations[0]

        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            declarations.append(
                self.parse_variable(
                    type_name,
                    terminators=declarator_terminators,
                    consume_terminator=False,
                    expected_terminators=terminators,
                )
            )

        self.consume_terminator(terminators, consume_terminator)
        return declarations if len(declarations) > 1 else declarations[0]

    def consume_terminator(
        self,
        terminators,
        consume_terminator,
        expected_terminators=None,
    ):
        if self.current_token[0] not in terminators:
            expected = " or ".join(sorted(expected_terminators or terminators))
            raise SyntaxError(f"Expected {expected}, got {self.current_token[0]}")
        if consume_terminator:
            self.eat(self.current_token[0])

    def parse_member_access(self, object):
        self.eat("DOT")
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(
                f"Expected identifier after dot, got {self.current_token[0]}"
            )
        member = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)

    def parse_function_call(self, name):
        args = self.parse_call_arguments()
        return FunctionCallNode(name, args)

    def parse_call_arguments(self):
        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        return args

    def parse_function_call_or_identifier(self):
        func_name = self.current_token[1]
        self.eat(self.current_token[0])

        if self.current_token[0] == "LBRACKET" and self.looks_like_array_constructor():
            func_name += self.parse_array_suffixes_as_text()

        if self.current_token[0] == "LPAREN":
            node = self.parse_function_call(func_name)
        else:
            node = VariableNode("", func_name)
        return self.parse_postfix_suffixes(node)

    def looks_like_array_constructor(self):
        index = self.pos
        if self.tokens[index][0] != "LBRACKET":
            return False

        while index < len(self.tokens) and self.tokens[index][0] == "LBRACKET":
            depth = 1
            index += 1
            while index < len(self.tokens) and depth:
                token_type = self.tokens[index][0]
                if token_type == "LBRACKET":
                    depth += 1
                elif token_type == "RBRACKET":
                    depth -= 1
                index += 1
            if depth:
                return False

        return index < len(self.tokens) and self.tokens[index][0] == "LPAREN"

    def parse_postfix_suffixes(self, node):
        while True:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.current_token[1]
                self.eat("IDENTIFIER")
                if self.current_token[0] == "LPAREN":
                    node = MethodCallNode(node, member, self.parse_call_arguments())
                else:
                    node = MemberAccessNode(node, member)
                continue

            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                node = ArrayAccessNode(node, index)
                continue

            return node

    def is_swizzle_member(self, member):
        return (
            isinstance(member, str)
            and 1 <= len(member) <= 4
            and any(
                set(member) <= component_set
                for component_set in self.SWIZZLE_COMPONENT_SETS
            )
        )

    def parse_numeric_postfix_suffixes(self, value):
        if value.lower().endswith("hf"):
            value = value[:-2]
        elif value[-1:] in {"u", "U", "f", "F"}:
            value = value[:-1]

        if (
            value.endswith(".")
            and self.current_token[0] == "IDENTIFIER"
            and self.is_swizzle_member(self.current_token[1])
        ):
            member = self.current_token[1]
            self.eat("IDENTIFIER")
            node = MemberAccessNode(value[:-1] or "0", member)
            return self.parse_postfix_suffixes(node)

        return self.parse_postfix_suffixes(value)

    def looks_like_member_call_statement(self):
        index = self.pos
        if self.tokens[index][0] != "IDENTIFIER":
            return False

        while index + 2 < len(self.tokens):
            if self.tokens[index + 1][0] != "DOT":
                return False
            if self.tokens[index + 2][0] != "IDENTIFIER":
                return False
            index += 2
            if index + 1 < len(self.tokens) and self.tokens[index + 1][0] == "LPAREN":
                return True

        return False

    def parse_array_suffixes_as_text(self):
        suffix = ""
        while self.current_token[0] == "LBRACKET":
            suffix += "["
            self.eat("LBRACKET")
            while self.current_token[0] != "RBRACKET":
                if self.current_token[0] == "EOF":
                    raise SyntaxError("Unterminated array suffix")
                suffix += str(self.current_token[1])
                self.eat(self.current_token[0])
            self.eat("RBRACKET")
            suffix += "]"
        return suffix

    def parse_primary(self):
        if self.current_token[0] == "MINUS":
            self.eat("MINUS")
            value = self.parse_primary()
            return UnaryOpNode("-", value)

        if (
            self.current_token[0] == "BITWISE_NOT"
            or self.current_token[0] == "BINARY_NOT"
        ):
            self.eat(self.current_token[0])
            value = self.parse_primary()
            return UnaryOpNode("~", value)

        if (
            self.current_token[0] == "IDENTIFIER"
            or self.current_token[1] in VALID_DATA_TYPES
        ):
            return self.parse_function_call_or_identifier()
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return self.parse_numeric_postfix_suffixes(value)
        elif self.current_token[0] == "STRING":
            value = self.current_token[1]
            self.eat("STRING")
            return value
        elif self.current_token[0] == "LPAREN":
            if self.looks_like_c_style_cast():
                return self.parse_c_style_cast()
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return self.parse_postfix_suffixes(expr)
        elif self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        else:
            raise SyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

    def looks_like_c_style_cast(self):
        return (
            self.current_token[0] == "LPAREN"
            and self.peek(1)
            in {
                "IDENTIFIER",
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
            }
            and self.peek(2) == "RPAREN"
            and self.peek(3)
            in {
                "IDENTIFIER",
                "NUMBER",
                "STRING",
                "LPAREN",
                "LBRACE",
                "MINUS",
                "PLUS",
                "NOT",
                "BITWISE_NOT",
                "BINARY_NOT",
            }
        )

    def parse_c_style_cast(self):
        self.eat("LPAREN")
        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        self.eat("RPAREN")
        return self.parse_postfix_suffixes(
            FunctionCallNode(type_name, [self.parse_unary()])
        )

    def parse_initializer_list(self):
        self.eat("LBRACE")
        elements = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated initializer list")
            elements.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if self.current_token[0] == "RBRACE":
                    break
                continue
            break
        self.eat("RBRACE")
        return InitializerListNode(elements)

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            token_type = self.current_token[0]
            op = self.current_token[1]
            self.eat(token_type)
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_assignment_or_function_call(
        self,
        terminators=None,
        consume_terminator=True,
    ):
        self.skip_hlsl_attributes()
        terminators = terminators or {"SEMICOLON"}
        type_name = ""
        qualifiers = self.parse_declaration_qualifiers()

        if qualifiers:
            if (
                self.current_token[0] == "IDENTIFIER"
                or self.current_token[1] in VALID_DATA_TYPES
            ):
                parsed_type = self.parse_data_type(allow_identifier=True)
                type_name = " ".join([*qualifiers, parsed_type])
            else:
                raise SyntaxError(
                    f"Unexpected token after declaration qualifier: {self.current_token[0]}"
                )
        elif self.current_token[0] == "IDENTIFIER" and self.peek(1) in [
            "POST_INCREMENT",
            "POST_DECREMENT",
        ]:
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "POST_INCREMENT":
                self.eat("POST_INCREMENT")
                self.consume_terminator(terminators, consume_terminator)
                return UnaryOpNode("POST_INCREMENT", VariableNode("", name))
            elif self.current_token[0] == "POST_DECREMENT":
                self.eat("POST_DECREMENT")
                self.consume_terminator(terminators, consume_terminator)
                return UnaryOpNode("POST_DECREMENT", VariableNode("", name))
            else:
                raise SyntaxError(
                    f"Unexpected token after identifier: {self.current_token[0]}"
                )
        if self.current_token[0] == "IDENTIFIER" and self.peek(1) == "IDENTIFIER":
            type_name = self.parse_data_type(allow_identifier=True)
        elif self.current_token[1] in VALID_DATA_TYPES:
            type_name = self.parse_data_type()
        if self.current_token[0] == "IDENTIFIER":
            if type_name and "COMMA" not in terminators:
                return self.parse_variable_declaration(
                    type_name,
                    terminators=terminators,
                    consume_terminator=consume_terminator,
                )
            return self.parse_variable(
                type_name,
                terminators=terminators,
                consume_terminator=consume_terminator,
            )

    def parse_expression(self):
        return self.parse_assignment_expression()

    def parse_assignment_expression(self):
        left = self.parse_ternary_expression()
        if self.current_token[0] in self.ASSIGNMENT_TOKENS:
            op_name = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op_name)
        return left

    def parse_ternary_expression(self):
        left = self.parse_logical_or_expression()

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)

        return left

    def parse_logical_or_expression(self):
        left = self.parse_logical_and_expression()
        while self.current_token[0] == "OR":
            op_symbol = self.current_token[1]
            self.eat("OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()
        while self.current_token[0] == "AND":
            op_symbol = self.current_token[1]
            self.eat("AND")
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()
        while self.current_token[0] == "BINARY_OR":
            op_symbol = self.current_token[1]
            self.eat("BINARY_OR")
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()
        while self.current_token[0] == "BINARY_XOR":
            op_symbol = self.current_token[1]
            self.eat("BINARY_XOR")
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()
        while self.current_token[0] == "BINARY_AND":
            op_symbol = self.current_token[1]
            self.eat("BINARY_AND")
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[0]
            op_symbol = self.current_token[1]
            self.eat(op)
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_relational_expression(self):
        left = self.parse_shift_expression()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[0]
            op_symbol = self.current_token[1]
            self.eat(op)
            right = self.parse_shift_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_shift_expression(self):
        left = self.parse_additive()
        while self.current_token[0] in [
            "BITWISE_SHIFT_LEFT",
            "BITWISE_SHIFT_RIGHT",
        ]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_additive()
            op_symbol = "<<" if op == "BITWISE_SHIFT_LEFT" else ">>"
            left = BinaryOpNode(left, op_symbol, right)

        return left

    def parse_expression_statement(self):
        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return expr

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_statement_or_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expr = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        cases = []
        seen_default = False
        self.breakable_depth += 1
        try:
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                if self.current_token[0] == "DEFAULT":
                    if seen_default:
                        raise SyntaxError("duplicate default label in switch")
                    seen_default = True
                cases.append(self.parse_case_statement())
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated switch statement")
        finally:
            self.breakable_depth -= 1
        self.eat("RBRACE")
        return SwitchNode(expr, cases)

    def parse_case_statement(self):
        if self.current_token[0] == "CASE":
            self.eat("CASE")
            value = self.parse_expression()
            self.eat("COLON")
        elif self.current_token[0] == "DEFAULT":
            self.eat("DEFAULT")
            value = None
            self.eat("COLON")
        else:
            raise SyntaxError(
                f"Expected CASE or DEFAULT in switch, got {self.current_token[0]}"
            )
        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            statement = self.parse_body()
            if isinstance(statement, list):
                statements.extend(statement)
            elif statement is not None:
                statements.append(statement)
        if self.current_token[0] == "EOF":
            raise SyntaxError("Unterminated switch case")
        return CaseNode(value, statements)

    def parse_default_statement(self):
        self.eat("DEFAULT")
        self.eat("COLON")
        statements = []
        while self.current_token[0] not in ["CASE", "RBRACE"]:
            statements.append(self.parse_body())
        return DefaultNode(statements)

    def parse_uniform(self):
        self.eat("UNIFORM")
        var_type = self.parse_data_type(allow_identifier=True)
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        name += self.parse_array_suffixes_as_text()
        self.eat("SEMICOLON")
        return UniformNode(var_type, name)

    def parse_unary(self):
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT", "NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        if self.current_token[0] == "PRE_INCREMENT":
            self.eat("PRE_INCREMENT")
            return UnaryOpNode("PRE_INCREMENT", self.parse_unary())
        if self.current_token[0] == "PRE_DECREMENT":
            self.eat("PRE_DECREMENT")
            return UnaryOpNode("PRE_DECREMENT", self.parse_unary())
        return self.parse_primary()
