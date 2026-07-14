"""Reverse code generator that emits CrossGL from Metal AST nodes."""

import re

from ...translator.codegen.array_utils import evaluate_literal_int_expression
from .MetalAst import *
from .MetalLexer import *
from .MetalParser import *
from .preprocessor import (
    DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT,
    MetalTemplateSpecializationError,
)
from .type_layout import metal_type_layout


class MetalCallableAliasLoweringError(ValueError):
    """Raised when a callable alias survives as a runtime value."""

    project_diagnostic_code = "project.translate.metal-callable-alias-unsupported"
    missing_capabilities = ("metal.runtime-callable-alias-lowering",)

    def __init__(
        self,
        alias_name,
        signature,
        usage,
        reason,
        source_location=None,
    ):
        self.alias_name = alias_name
        self.signature = signature
        self.usage = usage
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot lower Metal callable alias '{alias_name}' ({signature}) "
            f"used by {usage}: {reason}"
        )


class MetalStaticConstantResolutionError(ValueError):
    """Raised when a referenced Metal static constant cannot be materialized."""

    project_diagnostic_code = "project.translate.metal-static-constant-unresolved"
    missing_capabilities = ("metal.static-constant-materialization",)

    def __init__(self, owner, member, reason):
        self.owner = owner
        self.member = member
        self.reason = reason
        super().__init__(
            f"Cannot materialize Metal static constant {owner}::{member}: {reason}"
        )


class MetalBuiltinOverloadResolutionError(ValueError):
    """Raised when a Metal call cannot be bound to one user overload."""

    project_diagnostic_code = "project.translate.metal-builtin-overload-ambiguous"
    missing_capabilities = ("metal.builtin-overload-resolution",)

    def __init__(self, function_name, argument_types, candidates):
        self.function_name = function_name
        self.argument_types = tuple(argument_types)
        self.candidates = tuple(candidates)
        signature = ", ".join(self.argument_types)
        super().__init__(
            "Cannot resolve Metal built-in/user overload binding for "
            f"{function_name}({signature}); candidates are "
            f"{', '.join(self.candidates)}"
        )


class MetalStructMethodCallResolutionError(ValueError):
    """Raised when a lowered sibling method call cannot be rebound safely."""

    project_diagnostic_code = "project.translate.metal-struct-method-call-unresolved"
    missing_capabilities = ("metal.struct-method-call-lowering",)

    def __init__(
        self,
        owner,
        method_name,
        argument_types,
        candidates,
        reason,
        source_location=None,
    ):
        self.owner = owner
        self.method_name = method_name
        self.argument_types = tuple(argument_types)
        self.candidates = tuple(candidates)
        self.reason = reason
        self.source_location = source_location
        signature = ", ".join(self.argument_types)
        candidate_text = ", ".join(self.candidates) or "<none>"
        super().__init__(
            "Cannot resolve lowered Metal struct method call "
            f"{owner}::{method_name}({signature}): {reason}; candidates are "
            f"{candidate_text}; qualify the intended call before lowering or "
            "make the overload argument types exact."
        )


class MetalSizeofResolutionError(ValueError):
    """Raised when a concrete Metal object size cannot be represented safely."""

    project_diagnostic_code = "project.translate.metal-sizeof-unresolved"
    missing_capabilities = ("metal.sizeof-materialization",)

    def __init__(self, operand, reason, source_location=None):
        self.operand = operand
        self.reason = reason
        self.source_location = source_location
        super().__init__(f"Cannot materialize Metal sizeof({operand}): {reason}")


class MetalTemplateArgumentResolutionError(ValueError):
    """Raised when a required non-type template argument is not concrete."""

    project_diagnostic_code = "project.translate.metal-template-argument-unresolved"
    missing_capabilities = ("metal.value-template-argument-materialization",)

    def __init__(
        self,
        function_name,
        parameter_name,
        argument_expression,
        selected_call,
        reason,
        argument_kind,
        source_location=None,
        owner=None,
        member=None,
        requested_specialization=None,
        enclosing_function=None,
        enclosing_specialization=None,
        nested_helper=None,
        enclosing_context=None,
    ):
        self.function_name = function_name
        self.parameter_name = parameter_name
        self.argument_expression = argument_expression
        self.argument_kind = argument_kind
        self.default_expression = (
            argument_expression if argument_kind == "default" else None
        )
        self.explicit_argument = (
            argument_expression
            if argument_kind in {"explicit", "explicit_type"}
            else None
        )
        self.selected_call = selected_call
        self.owner = owner
        self.member = member
        self.requested_specialization = requested_specialization or selected_call
        self.enclosing_function = enclosing_function
        self.enclosing_specialization = enclosing_specialization
        self.nested_helper = nested_helper
        self.enclosing_context = enclosing_context
        self.reason = reason
        self.source_location = source_location
        if argument_kind == "missing":
            argument = f"required argument for '{parameter_name}'"
        elif argument_kind == "overload":
            argument = f"overload set for '{function_name}'"
        elif argument_kind == "explicit_type":
            argument = (
                f"explicit type argument for '{parameter_name}' "
                f"({argument_expression})"
            )
        elif argument_kind == "constexpr_body":
            argument = f"constexpr body ({argument_expression})"
        else:
            argument = (
                f"{argument_kind} argument for '{parameter_name}' "
                f"({argument_expression})"
            )
        message = (
            "Cannot materialize Metal function template "
            f"'{function_name}' for call '{selected_call}': {argument} {reason}"
        )
        if owner is not None and member is not None:
            message += f" while resolving {owner}::{member}"
        if enclosing_specialization is not None:
            helper = nested_helper or selected_call
            context = enclosing_context or "constexpr call"
            message += (
                f" while resolving {context} '{helper}' in specialization "
                f"'{enclosing_specialization}'"
            )
        super().__init__(message)


class MetalCallableLoweringError(ValueError):
    """Raised when a Metal callback cannot be lowered without changing semantics."""

    project_diagnostic_code = "project.translate.metal-callable-unsupported"
    missing_capabilities = ("metal.captured-callback-lowering",)

    def __init__(
        self,
        helper,
        reason,
        source_location=None,
        capture=None,
        enclosing_function=None,
        suggested_action=None,
    ):
        self.helper = helper
        self.reason = reason
        self.source_location = source_location
        self.capture = capture
        self.enclosing_function = enclosing_function
        self.suggested_action = suggested_action
        super().__init__(f"Cannot lower Metal callback passed to {helper}: {reason}")


class MetalWideVectorLoweringError(ValueError):
    """Raised when a concrete Metal vector cannot be lowered as an aggregate."""

    project_diagnostic_code = "project.translate.metal-wide-vector-unsupported"
    missing_capabilities = ("metal.wide-vector-aggregate-lowering",)

    def __init__(
        self,
        vector_type,
        reason,
        source_location=None,
        operation=None,
    ):
        self.vector_type = vector_type
        self.reason = reason
        self.source_location = source_location
        self.operation = operation
        detail = f" for operation '{operation}'" if operation else ""
        super().__init__(
            f"Cannot lower Metal wide vector '{vector_type}'{detail}: {reason}"
        )


class MetalStageEntryArrayResourceError(ValueError):
    """Raised when a Metal entry array has no faithful resource contract."""

    project_diagnostic_code = "project.translate.metal-entry-array-resource-invalid"
    missing_capabilities = ("metal.stage-entry-array-resource-lowering",)

    def __init__(
        self,
        parameter_name,
        array_dimensions,
        reason,
        source_location=None,
    ):
        self.parameter_name = parameter_name
        self.array_dimensions = tuple(array_dimensions or ())
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            "Cannot lower Metal stage-entry array resource "
            f"'{parameter_name}': {reason.replace('-', ' ')}"
        )


class MetalAtomicFenceLoweringError(ValueError):
    """Raised when a Metal fence contract cannot be preserved in CrossGL."""

    project_diagnostic_code = "project.translate.metal-atomic-fence-unsupported"
    missing_capabilities = ("metal.atomic-thread-fence-contract-lowering",)

    def __init__(
        self,
        reason,
        *,
        memory_flags=None,
        memory_order=None,
        thread_scope=None,
        source_location=None,
    ):
        self.reason = reason
        self.memory_flags = memory_flags
        self.memory_order = memory_order
        self.thread_scope = thread_scope
        self.source_location = source_location
        contract = (
            f"flags={memory_flags or '<missing>'}, "
            f"order={memory_order or '<missing>'}, "
            f"scope={thread_scope or '<missing>'}"
        )
        super().__init__(
            "Cannot lower Metal atomic_thread_fence without changing its "
            f"semantics ({contract}): {reason.replace('-', ' ')}"
        )


class MetalToCrossGLConverter:
    """Serialize Metal backend AST nodes back into CrossGL source."""

    crossgl_identifier_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    decimal_integer_literal_pattern = re.compile(r"^(?P<body>\d+)(?P<suffix>[uUlL]+)$")
    hex_integer_literal_pattern = re.compile(
        r"^(?P<body>0[xX][0-9a-fA-F]+)(?P<suffix>[uUlL]+)$"
    )
    binary_integer_literal_pattern = re.compile(
        r"^(?P<body>0[bB][01]+)(?P<suffix>[uUlL]+)$"
    )
    cast_literal_operand_pattern = re.compile(
        r"^(?:0[xX][0-9a-fA-F]+u?|0[bB][01]+u?|"
        r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fF]?|\d+u?)$"
    )
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
    crossgl_reserved_identifiers = {
        "break",
        "case",
        "cbuffer",
        "compute",
        "continue",
        "default",
        "do",
        "else",
        "false",
        "for",
        "fragment",
        "if",
        "in",
        "layout",
        "out",
        "return",
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
        "samplercubeshadow",
        "shader",
        "struct",
        "switch",
        "texture",
        "texture1d",
        "texture2d",
        "texture2darray",
        "texture3d",
        "texturecube",
        "true",
        "uniform",
        "vertex",
        "while",
    }
    storage_texture_accesses = {"access::read", "access::write", "access::read_write"}
    unscoped_metal_type_constructors = {
        "long2",
        "long3",
        "long4",
        "ulong2",
        "ulong3",
        "ulong4",
    }
    pixel_data_type_wrappers = {
        "r8unorm",
        "r8snorm",
        "r8uint",
        "r8sint",
        "r16unorm",
        "r16snorm",
        "r16uint",
        "r16sint",
        "r16float",
        "rg8unorm",
        "rg8snorm",
        "rg8uint",
        "rg8sint",
        "rg16unorm",
        "rg16snorm",
        "rg16uint",
        "rg16sint",
        "rg16float",
        "rgba8unorm",
        "rgba8snorm",
        "rgba8uint",
        "rgba8sint",
        "rgba16unorm",
        "rgba16snorm",
        "rgba16uint",
        "rgba16sint",
        "rgba16float",
        "rgba32uint",
        "rgba32sint",
        "rgba32float",
    }
    metal_math_namespace_prefixes = (
        "metal::fast::",
        "metal::precise::",
        "metal::",
        "fast::",
        "precise::",
    )
    metal_math_intrinsics = {
        "abs",
        "acos",
        "acosh",
        "all",
        "any",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "clamp",
        "cos",
        "cosh",
        "cospi",
        "distance",
        "dot",
        "exp",
        "exp2",
        "fabs",
        "floor",
        "fma",
        "fmax",
        "fmin",
        "fmod",
        "fract",
        "isfinite",
        "isinf",
        "isnan",
        "isnormal",
        "isordered",
        "isunordered",
        "length",
        "log",
        "log2",
        "max",
        "min",
        "mix",
        "normalize",
        "pow",
        "reflect",
        "rsqrt",
        "select",
        "sign",
        "sin",
        "sincos",
        "sinh",
        "sinpi",
        "smoothstep",
        "sqrt",
        "step",
        "tan",
        "tanh",
    }
    metal_bit_intrinsics = {
        "popcount": "bitCount",
        "reverse_bits": "bitfieldReverse",
    }
    metal_scalar_arithmetic_types = {
        "bool": ("integer", True, 1),
        "char": ("integer", True, 8),
        "int8_t": ("integer", True, 8),
        "int8": ("integer", True, 8),
        "uchar": ("integer", False, 8),
        "uint8_t": ("integer", False, 8),
        "uint8": ("integer", False, 8),
        "short": ("integer", True, 16),
        "int16_t": ("integer", True, 16),
        "int16": ("integer", True, 16),
        "ushort": ("integer", False, 16),
        "uint16_t": ("integer", False, 16),
        "uint16": ("integer", False, 16),
        "int": ("integer", True, 32),
        "int32_t": ("integer", True, 32),
        "uint": ("integer", False, 32),
        "uint32_t": ("integer", False, 32),
        "long": ("integer", True, 64),
        "int64_t": ("integer", True, 64),
        "int64": ("integer", True, 64),
        "ulong": ("integer", False, 64),
        "uint64_t": ("integer", False, 64),
        "uint64": ("integer", False, 64),
        "size_t": ("integer", False, 64),
        "half": ("floating", True, 16),
        "xhalf": ("floating", True, 16),
        "float16": ("floating", True, 16),
        "float": ("floating", True, 32),
        "double": ("floating", True, 64),
    }
    # Metal SIMD-group (wave) intrinsics -> canonical CrossGL Wave* ops.
    # The inverse mapping lives in crosstl/translator/codegen/metal_codegen.py,
    # and the DirectX and SPIR-V code generators already lower these canonical
    # names. Only same-arity, same-argument-order intrinsics are mapped here.
    # The relative-shuffle family (shuffle up/down/xor) is same-arity
    # (value, delta) and maps to the canonical Wave*Shuffle ops; the
    # inclusive-prefix scans still need argument-aware lowering and stay separate.
    metal_wave_intrinsics = {
        "simd_sum": "WaveActiveSum",
        "simd_product": "WaveActiveProduct",
        "simd_min": "WaveActiveMin",
        "simd_max": "WaveActiveMax",
        "simd_and": "WaveActiveBitAnd",
        "simd_or": "WaveActiveBitOr",
        "simd_xor": "WaveActiveBitXor",
        "simd_all": "WaveActiveAllTrue",
        "simd_any": "WaveActiveAnyTrue",
        "simd_ballot": "WaveActiveBallot",
        "simd_broadcast": "WaveReadLaneAt",
        "simd_broadcast_first": "WaveReadLaneFirst",
        "simd_shuffle": "WaveReadLaneAt",
        "simd_shuffle_down": "WaveShuffleDown",
        "simd_shuffle_up": "WaveShuffleUp",
        "simd_shuffle_and_fill_up": "WaveShuffleAndFillUp",
        "simd_shuffle_xor": "WaveShuffleXor",
        "simd_prefix_exclusive_sum": "WavePrefixSum",
        "simd_prefix_exclusive_product": "WavePrefixProduct",
        "simd_prefix_inclusive_sum": "WavePrefixInclusiveSum",
        "simd_prefix_inclusive_product": "WavePrefixInclusiveProduct",
        # Low-level __metal_simd_* builtins used by bf16_math.h's bfloat16
        # simd wrappers (e.g. simd_max(bfloat16_t) -> __metal_simd_max(float)).
        # They mirror the simd_* intrinsics above and lower to the same canonical
        # wave operations.
        "__metal_simd_sum": "WaveActiveSum",
        "__metal_simd_product": "WaveActiveProduct",
        "__metal_simd_min": "WaveActiveMin",
        "__metal_simd_max": "WaveActiveMax",
        "__metal_simd_and": "WaveActiveBitAnd",
        "__metal_simd_or": "WaveActiveBitOr",
        "__metal_simd_xor": "WaveActiveBitXor",
        "__metal_simd_all": "WaveActiveAllTrue",
        "__metal_simd_any": "WaveActiveAnyTrue",
        "__metal_simd_ballot": "WaveActiveBallot",
        "__metal_simd_broadcast": "WaveReadLaneAt",
        "__metal_simd_broadcast_first": "WaveReadLaneFirst",
        "__metal_simd_shuffle": "WaveReadLaneAt",
        "__metal_simd_shuffle_down": "WaveShuffleDown",
        "__metal_simd_shuffle_up": "WaveShuffleUp",
        "__metal_simd_shuffle_and_fill_up": "WaveShuffleAndFillUp",
        "__metal_simd_shuffle_xor": "WaveShuffleXor",
        "__metal_simd_prefix_exclusive_sum": "WavePrefixSum",
        "__metal_simd_prefix_exclusive_product": "WavePrefixProduct",
        "__metal_simd_prefix_inclusive_sum": "WavePrefixInclusiveSum",
        "__metal_simd_prefix_inclusive_product": "WavePrefixInclusiveProduct",
    }

    # Metal device/threadgroup atomics -> canonical CrossGL atomic intrinsics.
    # Each Metal call carries a trailing memory_order argument that the CrossGL
    # intrinsics (and the DirectX/GLSL/SPIR-V backends) do not take; it is dropped
    # during lowering. Only the read-modify-write operations with a direct CrossGL
    # counterpart are mapped here.
    metal_atomic_intrinsics = {
        "atomic_fetch_add_explicit": "atomicAdd",
        "atomic_fetch_min_explicit": "atomicMin",
        "atomic_fetch_max_explicit": "atomicMax",
        "atomic_fetch_and_explicit": "atomicAnd",
        "atomic_fetch_or_explicit": "atomicOr",
        "atomic_fetch_xor_explicit": "atomicXor",
        "atomic_exchange_explicit": "atomicExchange",
    }

    metal_atomic_fence_memory_flags = frozenset(
        {
            "mem_none",
            "mem_device",
            "mem_threadgroup",
            "mem_texture",
            "mem_threadgroup_imageblock",
            "mem_object_data",
        }
    )
    metal_atomic_fence_memory_orders = frozenset(
        {
            "memory_order_relaxed",
            "memory_order_acquire",
            "memory_order_release",
            "memory_order_acq_rel",
            "memory_order_seq_cst",
        }
    )
    metal_atomic_fence_thread_scopes = frozenset(
        {
            "thread_scope_thread",
            "thread_scope_simdgroup",
            "thread_scope_threadgroup",
            "thread_scope_device",
            "thread_scope_system",
        }
    )

    def __init__(self):
        self.rt_qualifiers = {
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
            "mesh",
            "object",
            "amplification",
        }
        self.type_map = {
            # Scalar Types
            "void": "void",
            "bool": "bool",
            "char": "int8",
            "uchar": "uint8",
            "short": "int16",
            "ushort": "uint16",
            "int8_t": "int8",
            "uint8_t": "uint8",
            "int16_t": "int16",
            "uint16_t": "uint16",
            "int32_t": "int",
            "uint32_t": "uint",
            "int": "int",
            "uint": "uint",
            "long": "int64",
            "ulong": "uint64",
            "int64_t": "int64",
            "uint64_t": "uint64",
            "float": "float",
            "half": "float16",
            "xhalf": "float16",
            "bfloat": "bfloat16",
            "double": "double",
            "size_t": "uint64",
            "ptrdiff_t": "int64",
            "atomic_int": "atomic_int",
            "atomic_uint": "atomic_uint",
            "atomic_bool": "atomic_bool",
            "atomic_ulong": "atomic_ulong",
            "atomic_float": "atomic_float",
            # Vector Types - float
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            # Vector Types - half
            "half2": "f16vec2",
            "half3": "f16vec3",
            "half4": "f16vec4",
            "xhalf2": "f16vec2",
            "xhalf3": "f16vec3",
            "xhalf4": "f16vec4",
            # Vector Types - bfloat
            "bfloat2": "bfloat16vec2",
            "bfloat3": "bfloat16vec3",
            "bfloat4": "bfloat16vec4",
            # Vector Types - int
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            # Vector Types - uint
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            # Vector Types - 64-bit int
            "long2": "i64vec2",
            "long3": "i64vec3",
            "long4": "i64vec4",
            "ulong2": "u64vec2",
            "ulong3": "u64vec3",
            "ulong4": "u64vec4",
            # Vector Types - short
            "short2": "i16vec2",
            "short3": "i16vec3",
            "short4": "i16vec4",
            # Vector Types - ushort
            "ushort2": "u16vec2",
            "ushort3": "u16vec3",
            "ushort4": "u16vec4",
            # Vector Types - char
            "char2": "i8vec2",
            "char3": "i8vec3",
            "char4": "i8vec4",
            # Vector Types - uchar
            "uchar2": "u8vec2",
            "uchar3": "u8vec3",
            "uchar4": "u8vec4",
            # Vector Types - bool
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            # Packed vector types
            "packed_float2": "vec2",
            "packed_float3": "vec3",
            "packed_float4": "vec4",
            "packed_half2": "f16vec2",
            "packed_half3": "f16vec3",
            "packed_half4": "f16vec4",
            "packed_char2": "i8vec2",
            "packed_char3": "i8vec3",
            "packed_char4": "i8vec4",
            "packed_uchar2": "u8vec2",
            "packed_uchar3": "u8vec3",
            "packed_uchar4": "u8vec4",
            "packed_short2": "i16vec2",
            "packed_short3": "i16vec3",
            "packed_short4": "i16vec4",
            "packed_ushort2": "u16vec2",
            "packed_ushort3": "u16vec3",
            "packed_ushort4": "u16vec4",
            "packed_int2": "ivec2",
            "packed_int3": "ivec3",
            "packed_int4": "ivec4",
            "packed_uint2": "uvec2",
            "packed_uint3": "uvec3",
            "packed_uint4": "uvec4",
            # SIMD types
            "simd_float2": "vec2",
            "simd_float3": "vec3",
            "simd_float4": "vec4",
            "simd_float2x2": "mat2",
            "simd_float3x3": "mat3",
            "simd_float4x4": "mat4",
            "simd_int2": "ivec2",
            "simd_int3": "ivec3",
            "simd_int4": "ivec4",
            "simd_uint2": "uvec2",
            "simd_uint3": "uvec3",
            "simd_uint4": "uvec4",
            # simd.h aliases commonly found in shared Metal headers
            "vector_float2": "vec2",
            "vector_float3": "vec3",
            "vector_float4": "vec4",
            "matrix_float3x3": "mat3",
            "matrix_float4x4": "mat4",
            # Matrix Types - float
            "float2x2": "mat2",
            "float2x3": "mat2x3",
            "float2x4": "mat2x4",
            "float3x2": "mat3x2",
            "float3x3": "mat3",
            "float3x4": "mat3x4",
            "float4x2": "mat4x2",
            "float4x3": "mat4x3",
            "float4x4": "mat4",
            # Matrix Types - half
            "half2x2": "f16mat2",
            "half2x3": "f16mat2x3",
            "half2x4": "f16mat2x4",
            "half3x2": "f16mat3x2",
            "half3x3": "f16mat3",
            "half3x4": "f16mat3x4",
            "half4x2": "f16mat4x2",
            "half4x3": "f16mat4x3",
            "half4x4": "f16mat4",
            "xhalf2x2": "f16mat2",
            "xhalf2x3": "f16mat2x3",
            "xhalf2x4": "f16mat2x4",
            "xhalf3x2": "f16mat3x2",
            "xhalf3x3": "f16mat3",
            "xhalf3x4": "f16mat3x4",
            "xhalf4x2": "f16mat4x2",
            "xhalf4x3": "f16mat4x3",
            "xhalf4x4": "f16mat4",
            # Texture Types
            "texture1d": "sampler1D",
            "texture1d<float>": "sampler1D",
            "texture1d<half>": "sampler1D",
            "texture1d<int>": "isampler1D",
            "texture1d<uint>": "usampler1D",
            "texture1d_array": "sampler1DArray",
            "texture1d_array<float>": "sampler1DArray",
            "texture1d_array<half>": "sampler1DArray",
            "texture1d_array<int>": "isampler1DArray",
            "texture1d_array<uint>": "usampler1DArray",
            "texture2d": "sampler2D",
            "texture2d<float>": "sampler2D",
            "texture2d<half>": "sampler2D",
            "texture2d<int>": "isampler2D",
            "texture2d<uint>": "usampler2D",
            "texture2d_ms": "sampler2DMS",
            "texture2d_ms<float>": "sampler2DMS",
            "texture2d_ms<half>": "sampler2DMS",
            "texture2d_ms<int>": "isampler2DMS",
            "texture2d_ms<uint>": "usampler2DMS",
            "texture2d_ms_array": "sampler2DMSArray",
            "texture2d_ms_array<float>": "sampler2DMSArray",
            "texture2d_ms_array<half>": "sampler2DMSArray",
            "texture2d_ms_array<int>": "isampler2DMSArray",
            "texture2d_ms_array<uint>": "usampler2DMSArray",
            "texture3d": "sampler3D",
            "texture3d<float>": "sampler3D",
            "texture3d<half>": "sampler3D",
            "texture3d<int>": "isampler3D",
            "texture3d<uint>": "usampler3D",
            "texturecube": "samplerCube",
            "texturecube<float>": "samplerCube",
            "texturecube<half>": "samplerCube",
            "texturecube<int>": "isamplerCube",
            "texturecube<uint>": "usamplerCube",
            "TextureCube": "samplerCube",
            "texturecube_array": "samplerCubeArray",
            "texturecube_array<float>": "samplerCubeArray",
            "texturecube_array<half>": "samplerCubeArray",
            "texturecube_array<int>": "isamplerCubeArray",
            "texturecube_array<uint>": "usamplerCubeArray",
            "texture2d_array": "sampler2DArray",
            "texture2d_array<float>": "sampler2DArray",
            "texture2d_array<half>": "sampler2DArray",
            "texture2d_array<int>": "isampler2DArray",
            "texture2d_array<uint>": "usampler2DArray",
            "texture_buffer": "samplerBuffer",
            "texture_buffer<float>": "samplerBuffer",
            "texture_buffer<half>": "samplerBuffer",
            "texture_buffer<int>": "isamplerBuffer",
            "texture_buffer<uint>": "usamplerBuffer",
            "depth2d": "sampler2DShadow",
            "depth2d<float>": "sampler2DShadow",
            "depth2d_array": "sampler2DArrayShadow",
            "depth2d_array<float>": "sampler2DArrayShadow",
            "depthcube": "samplerCubeShadow",
            "depthcube<float>": "samplerCubeShadow",
            "depthcube_array": "samplerCubeArrayShadow",
            "depthcube_array<float>": "samplerCubeArrayShadow",
            "depth2d_ms": "sampler2DMS",
            "depth2d_ms<float>": "sampler2DMS",
            "depth2d_ms_array": "sampler2DMSArray",
            "depth2d_ms_array<float>": "sampler2DMSArray",
            # SwiftUI layer-effect shaders expose the rendered layer as a
            # sampler-like object with sample(coord) syntax.
            "SwiftUI::Layer": "sampler2D",
            # RealityKit custom material shaders use framework-scoped opaque
            # parameter types with no direct CrossGL namespace syntax.
            "realitykit::surface_parameters": "surface_parameters",
            "realitykit::geometry_parameters": "geometry_parameters",
            "acceleration_structure": "acceleration_structure",
            "intersection_function_table": "intersection_function_table",
            "visible_function_table": "visible_function_table",
            "indirect_command_buffer": "indirect_command_buffer",
            "ray": "ray",
            "ray_data": "ray_data",
            "intersection_result": "intersection_result",
            "intersection_params": "intersection_params",
            "triangle_intersection_params": "triangle_intersection_params",
            "intersector": "intersector",
            # Sampler type
            "sampler": "sampler",
        }
        self.type_aliases = {}
        self.type_alias_qualifiers = {}
        self.callable_type_aliases = {}
        self.callable_alias_declarations = {}
        self.global_variable_types = {}
        self.current_variable_types = {}
        self.storage_texture_declaration_ids = set()
        self.global_storage_texture_names = set()
        self.current_storage_texture_names = set()
        self.global_structured_buffer_names = set()
        self.current_structured_buffer_names = set()
        self.current_stage_entry_resource_parameter_ids = set()
        self.global_sampler_names = set()
        self.suppress_structured_buffer_index_lowering = False
        self.struct_member_types = {}
        self.struct_declarations = {}
        self.struct_name_map = {}
        self.ambiguous_struct_names = set()
        self.struct_static_constants = {}
        self.struct_static_constant_members = {}
        self.struct_static_constant_resolution_stack = []
        self.struct_static_constexpr_member_keys = set()
        self.current_struct_static_constant_owner = None
        self.struct_template_parameters = {}
        self.local_struct_type_aliases = {}
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
        self.metal_atomic_fence_transport_declaration_ids = set()
        self.current_function_name = None
        self.current_function_return_type = None
        self.current_function = None
        self.current_function_specialization = None
        self.current_function_materialization_bindings = {}
        self.materialized_constexpr_expression_contexts = []
        self.wide_vector_types = {}
        self.wide_vector_binary_helpers = set()
        self.wide_vector_compound_helpers = set()
        self.wide_vector_reserved_names = set()
        self.identifier_maps = [{}]
        self.used_identifier_names = [set()]
        self.texture_method_functions = {
            "read": "textureLoad",
            "write": "textureStore",
            "sample_compare": "textureCompare",
            "sample_compare_level": "textureCompareLod",
            "gather": "textureGather",
            "gather_compare": "textureGatherCompare",
        }
        self.texture_method_storage_operations = {
            "read": "read",
            "write": "write",
        }
        self.sampled_texture_methods = {
            "sample_compare",
            "sample_compare_level",
            "gather",
            "gather_compare",
        }
        self.texture_method_operations = {
            "read": "load",
            "write": "store",
            "sample_compare": "sample_compare",
            "sample_compare_level": "sample_compare_lod",
            "gather": "gather",
            "gather_compare": "gather_compare",
        }
        self.texture_gather_components = {
            "x": "0",
            "y": "1",
            "z": "2",
            "w": "3",
        }
        self.texture_size_query_methods = {
            "get_width",
            "get_height",
            "get_depth",
            "get_array_size",
        }
        self.parameter_direction_qualifiers = ("inout", "out")
        self.fragment_execution_attribute_names = {
            "early_fragment_tests",
        }
        self.function_metadata_attribute_names = {
            "host_name",
        }

        self.map_semantics = {
            # Vertex attributes
            "attribute(0)": "POSITION",
            "attribute(1)": "NORMAL",
            "attribute(2)": "TANGENT",
            "attribute(3)": "BINORMAL",
            "attribute(4)": "TEXCOORD",
            "attribute(5)": "TEXCOORD0",
            "attribute(6)": "TEXCOORD1",
            "attribute(7)": "TEXCOORD2",
            "attribute(8)": "TEXCOORD3",
            "attribute(9)": "COLOR",
            "attribute(10)": "COLOR0",
            "attribute(11)": "COLOR1",
            "vertex_id": "gl_VertexID",
            "instance_id": "gl_InstanceID",
            "base_vertex": "gl_BaseVertex",
            "base_instance": "gl_BaseInstance",
            "position": "gl_Position",
            "point_size": "gl_PointSize",
            "clip_distance": "gl_ClipDistance",
            "front_facing": "gl_FrontFacing",
            "point_coord": "gl_PointCoord",
            "color(0)": "gl_FragColor",
            "color(1)": "gl_FragColor1",
            "color(2)": "gl_FragColor2",
            "color(3)": "gl_FragColor3",
            "color(4)": "gl_FragColor4",
            "depth(any)": "gl_FragDepth",
            "stencil": "gl_FragStencilRefEXT",
            "sample_id": "gl_SampleID",
            "sample_mask": "gl_SampleMask",
            "primitive_id": "gl_PrimitiveID",
            "viewport_array_index": "gl_ViewportIndex",
            "render_target_array_index": "gl_Layer",
            "thread_position_in_grid": "gl_GlobalInvocationID",
            "thread_position_in_threadgroup": "gl_LocalInvocationID",
            "threadgroup_position_in_grid": "gl_WorkGroupID",
            "thread_index_in_threadgroup": "gl_LocalInvocationIndex",
            "thread_index_in_simdgroup": "gl_SubgroupInvocationID",
            "simdgroup_index_in_threadgroup": "gl_SubgroupID",
            "threads_per_threadgroup": "gl_WorkGroupSize",
            "threads_per_simdgroup": "gl_SubgroupSize",
            "threadgroups_per_grid": "gl_NumWorkGroups",
            "stage_in": "",
        }

    def texture_method_descriptor(self, method):
        function = self.texture_method_functions.get(method)
        if function is None:
            return None
        return {
            "method": method,
            "function": function,
            "storage_operation": self.texture_method_storage_operations.get(method),
            "sampled_texture": method in self.sampled_texture_methods,
        }

    def resource_method_descriptor(self, method):
        descriptor = self.texture_method_descriptor(method)
        if descriptor is None:
            return None
        descriptor = dict(descriptor)
        descriptor["resource"] = (
            "texture_or_image" if descriptor["storage_operation"] else "texture"
        )
        descriptor["operation"] = self.texture_method_operations.get(method)
        return descriptor

    def push_identifier_scope(self):
        self.identifier_maps.append({})
        self.used_identifier_names.append(set())

    def pop_identifier_scope(self):
        self.identifier_maps.pop()
        self.used_identifier_names.pop()

    def sanitize_identifier(self, name):
        if not name:
            return name
        if (
            self.crossgl_identifier_pattern.match(name)
            and name not in self.crossgl_reserved_identifiers
        ):
            return name

        parts = []
        for index, char in enumerate(str(name)):
            valid = (
                char == "_"
                or ("A" <= char <= "Z")
                or ("a" <= char <= "z")
                or (index > 0 and "0" <= char <= "9")
            )
            parts.append(char if valid else f"_u{ord(char):x}")
        candidate = "".join(parts)
        if not candidate or not re.match(r"^[A-Za-z_]", candidate):
            candidate = f"_{candidate}"
        if candidate in self.crossgl_reserved_identifiers:
            candidate = f"{candidate}_"
        return candidate

    def declare_identifier(self, name):
        if not name:
            return name
        if self.template_binding_shadow_scopes and any(
            name in bindings for bindings in self.template_value_bindings
        ):
            self.template_binding_shadow_scopes[-1].add(name)
        current_map = self.identifier_maps[-1]
        if name in current_map:
            return current_map[name]

        base = self.sanitize_identifier(name)
        candidate = base
        used = self.used_identifier_names[-1]
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        current_map[name] = candidate
        return candidate

    def render_identifier(self, name):
        template_value = self.template_value_binding(name)
        if template_value is not None:
            return template_value
        name = self.substitute_integral_constant_members(name)
        for scope in reversed(self.identifier_maps):
            if name in scope:
                return scope[name]
        local_static_member = self.render_local_static_struct_member_identifier(name)
        if local_static_member is not None:
            return local_static_member
        static_member = self.render_static_struct_member_identifier(name)
        if static_member is not None:
            return static_member
        return self.sanitize_identifier(name)

    def substitute_integral_constant_members(self, name):
        rendered = str(name)
        substituted = set()
        for bindings in reversed(self.integral_constant_bindings):
            for parameter_name, value in bindings.items():
                if parameter_name in substituted:
                    continue
                rendered = re.sub(
                    rf"\b{re.escape(parameter_name)}\s*\.\s*value\b",
                    "true" if value else "false",
                    rendered,
                )
                substituted.add(parameter_name)
        return rendered

    def render_local_static_struct_member_identifier(self, name):
        owner = self.current_struct_static_constant_owner
        if owner is None or not isinstance(name, str) or "::" in name:
            return None
        key = (owner, name)
        if key not in self.struct_static_constant_members:
            return None
        self.propagate_struct_static_constexpr_dependency(key)
        return self.render_resolved_static_constant(key)

    def render_static_struct_member_identifier(self, name, require_constant=False):
        if not isinstance(name, str) or "::" not in name:
            return None
        struct_name, member_name = name.rsplit("::", 1)
        alias_target = self.local_struct_type_aliases.get(struct_name)
        resolved_struct = self.resolve_local_type_aliases(alias_target or struct_name)
        if resolved_struct not in self.struct_name_map and "::" in resolved_struct:
            unqualified_struct = resolved_struct.rsplit("::", 1)[-1]
            if unqualified_struct in self.struct_name_map:
                resolved_struct = unqualified_struct
        if resolved_struct in self.ambiguous_struct_names:
            raise MetalStaticConstantResolutionError(
                resolved_struct,
                member_name,
                "multiple visible struct declarations match the qualified owner",
            )
        if resolved_struct not in self.struct_name_map:
            if require_constant:
                raise MetalStaticConstantResolutionError(
                    resolved_struct,
                    member_name,
                    "the inferred expression type does not name a visible struct",
                )
            return None
        mapped_struct = self.map_struct_name(resolved_struct)
        key = (mapped_struct, member_name)
        if key in self.struct_static_constant_members:
            self.propagate_struct_static_constexpr_dependency(key)
            return self.render_resolved_static_constant(key)
        if require_constant:
            raise MetalStaticConstantResolutionError(
                mapped_struct,
                member_name,
                "the selected declaration has no compile-time static member",
            )
        return self.sanitize_identifier(f"{mapped_struct}::{member_name}")

    def propagate_struct_static_constexpr_dependency(self, dependency_key):
        if (
            dependency_key in self.struct_static_constexpr_member_keys
            and self.struct_static_constant_resolution_stack
        ):
            self.struct_static_constexpr_member_keys.add(
                self.struct_static_constant_resolution_stack[-1]
            )

    def render_decltype_static_struct_member(self, expr):
        owner = getattr(expr, "object", None)
        if not isinstance(owner, FunctionCallNode):
            return None
        if str(getattr(owner, "name", "")) not in {"decltype", "metal::decltype"}:
            return None
        arguments = getattr(owner, "args", None) or []
        member = str(getattr(expr, "member", ""))
        if len(arguments) != 1:
            raise MetalStaticConstantResolutionError(
                "decltype(expression)",
                member,
                "decltype requires exactly one expression",
            )

        owner_name = f"decltype({self.generate_expression(arguments[0], False)})"
        inferred_owner = self.expression_metal_type(arguments[0])
        if inferred_owner is None:
            raise MetalStaticConstantResolutionError(
                owner_name,
                member,
                "the expression type could not be inferred",
            )
        resolved_owner = self.normalized_metal_type(
            self.resolve_type_alias(inferred_owner)
        )
        rendered = self.render_static_struct_member_identifier(
            f"{resolved_owner}::{member}",
            require_constant=True,
        )
        return rendered

    def render_metal_sizeof_expression(self, expr):
        if str(getattr(expr, "name", "")) != "sizeof":
            return None
        arguments = getattr(expr, "args", None) or []
        if len(arguments) != 1:
            raise MetalSizeofResolutionError(
                "expression",
                "sizeof requires exactly one operand",
                getattr(expr, "source_location", None),
            )

        operand = arguments[0]
        operand_type = self.metal_sizeof_operand_type(operand)
        if operand_type is None:
            return None

        local_alias = self.local_struct_type_aliases.get(operand_type, operand_type)
        resolved_type = self.resolve_type_alias(local_alias)
        layout = self.metal_concrete_type_layout(resolved_type)
        if layout is not None:
            return str(layout[0])

        normalized_type = self.normalized_metal_type(resolved_type)
        if normalized_type in self.struct_name_map:
            raise MetalSizeofResolutionError(
                resolved_type,
                "aggregate object layout is not available",
                getattr(expr, "source_location", None),
            )
        if "*" in str(resolved_type) or "&" in str(resolved_type):
            raise MetalSizeofResolutionError(
                resolved_type,
                "pointer and reference object sizes are not portable",
                getattr(expr, "source_location", None),
            )
        return None

    def metal_concrete_type_layout(self, metal_type, resolving=None):
        resolved_type = self.resolve_type_alias(metal_type)
        layout = metal_type_layout(resolved_type)
        if layout is not None:
            return layout

        struct_name = self.normalized_metal_type(resolved_type)
        if struct_name in self.ambiguous_struct_names:
            return None
        struct_node = self.struct_declarations.get(struct_name)
        if struct_node is None:
            return None
        if (
            getattr(struct_node, "alignas", None)
            or getattr(struct_node, "attributes", None)
            or getattr(struct_node, "template_parameters", None)
            or getattr(struct_node, "generics", None)
            or getattr(struct_node, "bases", None)
            or getattr(struct_node, "base_classes", None)
            or getattr(struct_node, "base_types", None)
        ):
            return None

        resolving = set(resolving or ())
        if struct_name in resolving:
            return None
        resolving.add(struct_name)

        offset = 0
        aggregate_alignment = 1
        is_union = getattr(struct_node, "aggregate_kind", None) == "union"
        for member in getattr(struct_node, "members", []) or []:
            if not isinstance(member, VariableNode):
                if isinstance(
                    member,
                    (
                        EnumNode,
                        FunctionNode,
                        StaticAssertNode,
                        StructNode,
                        TypeAliasNode,
                    ),
                ):
                    continue
                return None
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(member, "qualifiers", []) or []
            }
            if "static" in qualifiers:
                continue
            member_layout = self.metal_struct_member_layout(member, resolving)
            if member_layout is None:
                return None
            member_size, member_alignment = member_layout
            if is_union:
                offset = max(offset, member_size)
            else:
                offset = self.align_metal_layout_offset(offset, member_alignment)
                offset += member_size
            aggregate_alignment = max(aggregate_alignment, member_alignment)

        aggregate_size = self.align_metal_layout_offset(offset, aggregate_alignment)
        if aggregate_size == 0:
            aggregate_size = 1
        if aggregate_size > (1 << 63) - 1:
            return None
        return aggregate_size, aggregate_alignment

    def metal_struct_member_layout(self, member, resolving):
        if (
            getattr(member, "alignas", None)
            or getattr(member, "bitfield_width", None) is not None
        ):
            return None
        member_type = getattr(member, "vtype", None)
        declarator_suffix = str(getattr(member, "declarator_type_suffix", "") or "")
        if (
            not member_type
            or "*" in str(member_type)
            or "&" in str(member_type)
            or "*" in declarator_suffix
            or "&" in declarator_suffix
        ):
            return None

        layout = self.metal_concrete_type_layout(member_type, resolving)
        if layout is None:
            return None
        member_size, member_alignment = layout
        for extent_expression in getattr(member, "array_sizes", []) or []:
            extent = evaluate_literal_int_expression(extent_expression)
            if extent is None or extent <= 0:
                return None
            stride = self.align_metal_layout_offset(member_size, member_alignment)
            if stride > ((1 << 63) - 1) // extent:
                return None
            member_size = stride * extent
        return member_size, member_alignment

    @staticmethod
    def align_metal_layout_offset(offset, alignment):
        return ((offset + alignment - 1) // alignment) * alignment

    def metal_sizeof_operand_type(self, operand):
        if isinstance(operand, str):
            variable_type = self.current_variable_types.get(
                operand,
                self.global_variable_types.get(operand),
            )
            return variable_type or operand
        return self.expression_metal_type(operand)

    def render_resolved_static_constant(self, key):
        constant = self.resolve_struct_static_constant(key)
        if constant is None:
            owner, member = key
            raise MetalStaticConstantResolutionError(
                owner,
                member,
                "the selected declaration has no constant initializer",
            )
        if re.fullmatch(
            r"(?:true|false|[-+]?\d+(?:[uU])?|"
            r"0[xX][0-9a-fA-F]+[uU]?|0[bB][01]+[uU]?)",
            constant,
        ):
            return constant
        return f"({constant})"

    def reserve_generated_identifier(self, base):
        used = self.used_identifier_names[-1]
        sanitized_base = self.sanitize_identifier(base)
        candidate = sanitized_base
        suffix = 2
        while candidate in used:
            candidate = f"{sanitized_base}_{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    def generate_sampler_constructor_arg(self, arg, is_main=False):
        if isinstance(arg, VariableNode) and self.is_scoped_identifier(arg.name):
            return arg.name
        return self.generate_expression(arg, is_main)

    def is_scoped_identifier(self, name):
        return (
            isinstance(name, str)
            and "::" in name
            and all(
                self.crossgl_identifier_pattern.match(part)
                and part not in self.crossgl_reserved_identifiers
                for part in name.split("::")
            )
        )

    def unwrap_texture_option_argument(self, expr, option_name):
        if (
            isinstance(expr, FunctionCallNode)
            and self.unscoped_function_name(expr.name) == option_name
            and len(expr.args) == 1
        ):
            return expr.args[0]
        return expr

    def unscoped_function_name(self, name):
        return str(name).split("::")[-1]

    def texture_sample_options_call(self, options, sample_args, is_main=False):
        if not options:
            return f"texture({', '.join(sample_args)})"
        if len(options) > 2:
            return None

        option = options[0]
        offset = options[1] if len(options) == 2 else None

        if (
            isinstance(option, FunctionCallNode)
            and self.unscoped_function_name(option.name) == "bias"
            and len(option.args) == 1
        ):
            bias = self.generate_expression(option.args[0], is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                args = sample_args + [rendered_offset, bias]
                return f"textureOffset({', '.join(args)})"
            return f"texture({', '.join(sample_args + [bias])})"

        gradient_args = self.texture_gradient_option_arguments(option)
        if gradient_args is not None:
            ddx = self.generate_expression(gradient_args[0], is_main)
            ddy = self.generate_expression(gradient_args[1], is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                return (
                    f"textureGradOffset("
                    f"{', '.join(sample_args + [ddx, ddy, rendered_offset])})"
                )
            return f"textureGrad({', '.join(sample_args + [ddx, ddy])})"

        min_lod_clamp_arg = self.unwrap_texture_option_argument(option, "min_lod_clamp")
        if min_lod_clamp_arg is not option:
            min_lod = self.generate_expression(min_lod_clamp_arg, is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                return (
                    "textureMinLodClampOffset("
                    f"{', '.join(sample_args + [min_lod, rendered_offset])})"
                )
            return f"textureMinLodClamp({', '.join(sample_args + [min_lod])})"

        level_arg = self.unwrap_texture_option_argument(option, "level")
        if level_arg is not option or not self.texture_sample_option_is_offset(option):
            lod = self.generate_expression(level_arg, is_main)
            if offset is not None:
                rendered_offset = self.generate_expression(offset, is_main)
                return (
                    f"textureLodOffset("
                    f"{', '.join(sample_args + [lod, rendered_offset])})"
                )
            return f"textureLod({', '.join(sample_args + [lod])})"

        rendered_offset = self.generate_expression(option, is_main)
        return f"textureOffset({', '.join(sample_args + [rendered_offset])})"

    def texture_sample_option_is_offset(self, option):
        mapped_type = self.expression_mapped_type(option)
        if self.is_integer_vector_type(mapped_type):
            return True
        constructor_type = getattr(option, "name", None) or getattr(
            option, "type_name", None
        )
        if constructor_type is None:
            return False
        return self.is_integer_vector_type(self.map_type(str(constructor_type)))

    def is_integer_vector_type(self, type_name):
        return type_name in {
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "i16vec2",
            "i16vec3",
            "i16vec4",
            "u16vec2",
            "u16vec3",
            "u16vec4",
            "i8vec2",
            "i8vec3",
            "i8vec4",
            "u8vec2",
            "u8vec3",
            "u8vec4",
        }

    def texture_gradient_option_arguments(self, option):
        if (
            isinstance(option, FunctionCallNode)
            and self.unscoped_function_name(option.name)
            in {"gradient2d", "gradient3d", "gradientcube"}
            and len(option.args) == 2
        ):
            return option.args
        return None

    def resource_classification_type(self, mapped_type):
        if not mapped_type:
            return mapped_type
        base = str(mapped_type).strip()
        while base.endswith("*") or base.endswith("&"):
            base = base[:-1].strip()
        return base

    def sampled_array_coordinate_constructor(self, texture_expr):
        mapped_type = self.resource_classification_type(
            self.expression_mapped_type(texture_expr)
        )
        return self.sampled_array_coordinate_constructor_for_type(mapped_type)

    def sampled_array_coordinate_constructor_for_type(self, mapped_type):
        return {
            "sampler1DArray": "vec2",
            "isampler1DArray": "vec2",
            "usampler1DArray": "vec2",
            "sampler2DArray": "vec3",
            "isampler2DArray": "vec3",
            "usampler2DArray": "vec3",
            "sampler2DArrayShadow": "vec3",
            "samplerCubeArray": "vec4",
            "samplerCubeArrayShadow": "vec4",
        }.get(mapped_type)

    def texture_sample_coordinate_and_options(
        self, texture_expr, coords_expr, options, is_main=False
    ):
        coords = self.generate_expression(coords_expr, is_main)
        remaining_options = list(options)
        constructor = self.sampled_array_coordinate_constructor(texture_expr)
        if constructor and remaining_options:
            layer = self.generate_expression(remaining_options.pop(0), is_main)
            coords = f"{constructor}({coords}, {layer})"
        return coords, remaining_options

    def texture_compare_method_base_arguments(
        self, obj_expr, method_args, is_main=False
    ):
        mapped_type = self.resource_classification_type(
            self.expression_mapped_type(obj_expr)
        )
        if mapped_type == "sampler2DArrayShadow" and len(method_args) >= 4:
            sampler = self.generate_expression(method_args[0], is_main)
            coord = self.generate_expression(method_args[1], is_main)
            layer = self.generate_expression(method_args[2], is_main)
            compare = self.generate_expression(method_args[3], is_main)
            return [sampler, f"vec3({coord}, {layer})", compare], 4
        if mapped_type == "samplerCubeArrayShadow" and len(method_args) >= 4:
            sampler = self.generate_expression(method_args[0], is_main)
            coord = self.generate_expression(method_args[1], is_main)
            layer = self.generate_expression(method_args[2], is_main)
            compare = self.generate_expression(method_args[3], is_main)
            return [sampler, f"vec4({coord}, {layer})", compare], 4
        return [self.generate_expression(arg, is_main) for arg in method_args[:3]], 3

    def texture_compare_option_method_call(
        self, obj, obj_expr, method_args, is_main=False
    ):
        compare_args, compare_arg_count = self.texture_compare_method_base_arguments(
            obj_expr, method_args, is_main
        )
        if len(method_args) == compare_arg_count and compare_arg_count == 4:
            return f"textureCompare({obj}, {', '.join(compare_args)})"
        if len(method_args) not in {compare_arg_count + 1, compare_arg_count + 2}:
            return None

        option = method_args[compare_arg_count]
        unsupported_option = self.unsupported_texture_compare_lod_option(obj, option)
        if unsupported_option is not None:
            return unsupported_option

        level_arg = self.unwrap_texture_option_argument(option, "level")
        if level_arg is not option:
            lod = self.generate_expression(level_arg, is_main)
            if len(method_args) == compare_arg_count + 1:
                return f"textureCompareLod({obj}, {', '.join(compare_args + [lod])})"
            offset = self.generate_expression(
                method_args[compare_arg_count + 1], is_main
            )
            return (
                f"textureCompareLodOffset("
                f"{obj}, {', '.join(compare_args + [lod, offset])})"
            )

        gradient_args = self.texture_gradient_option_arguments(option)
        if gradient_args is not None:
            ddx = self.generate_expression(gradient_args[0], is_main)
            ddy = self.generate_expression(gradient_args[1], is_main)
            if len(method_args) == compare_arg_count + 1:
                return (
                    f"textureCompareGrad({obj}, {', '.join(compare_args + [ddx, ddy])})"
                )
            offset = self.generate_expression(
                method_args[compare_arg_count + 1], is_main
            )
            return (
                f"textureCompareGradOffset("
                f"{obj}, {', '.join(compare_args + [ddx, ddy, offset])})"
            )

        if len(method_args) == compare_arg_count + 1 and not isinstance(
            option, FunctionCallNode
        ):
            offset = self.generate_expression(option, is_main)
            return f"textureCompareOffset({obj}, {', '.join(compare_args + [offset])})"

        return None

    def unsupported_texture_compare_lod_option(self, obj, option):
        if not isinstance(option, FunctionCallNode):
            return None
        option_name = self.unscoped_function_name(option.name)
        if option_name not in {"bias", "min_lod_clamp"}:
            return None
        return (
            "0.0 /* unsupported Metal depth compare lod option: "
            f"{option_name} on {obj} */"
        )

    def texture_gather_call(self, obj, texture_expr, method_args, is_main=False):
        args, offset = self.texture_gather_arguments(texture_expr, method_args, is_main)
        function = "textureGatherOffset" if offset else "textureGather"
        return f"{function}({obj}, {', '.join(args)})"

    def texture_gather_arguments(self, texture_expr, method_args, is_main=False):
        if len(method_args) < 2:
            return [
                self.texture_gather_argument(arg, is_main) for arg in method_args
            ], False

        sampler = self.generate_expression(method_args[0], is_main)
        coords = self.generate_expression(method_args[1], is_main)
        tail = list(method_args[2:])
        constructor = self.sampled_array_coordinate_constructor(texture_expr)
        if constructor and tail:
            layer = self.generate_expression(tail.pop(0), is_main)
            coords = f"{constructor}({coords}, {layer})"

        args = [sampler, coords]
        has_offset = False
        if tail:
            if self.texture_gather_component_index(tail[0]) is None:
                args.append(self.generate_expression(tail.pop(0), is_main))
                has_offset = True
            if tail:
                args.append(self.texture_gather_argument(tail.pop(0), is_main))

        args.extend(self.texture_gather_argument(arg, is_main) for arg in tail)
        return args, has_offset

    def texture_gather_compare_call(self, obj, obj_expr, method_args, is_main=False):
        compare_args, consumed = self.texture_gather_compare_base_arguments(
            obj_expr, method_args, is_main
        )
        tail = list(method_args[consumed:])
        if len(tail) > 1:
            return None
        if tail:
            offset = self.generate_expression(tail[0], is_main)
            return (
                "textureGatherCompareOffset("
                f"{obj}, {', '.join(compare_args + [offset])})"
            )
        return f"textureGatherCompare({obj}, {', '.join(compare_args)})"

    def texture_gather_compare_base_arguments(
        self, obj_expr, method_args, is_main=False
    ):
        if len(method_args) < 3:
            return [self.generate_expression(arg, is_main) for arg in method_args], len(
                method_args
            )

        mapped_type = self.resource_classification_type(
            self.expression_mapped_type(obj_expr)
        )
        if (
            mapped_type in {"sampler2DArrayShadow", "samplerCubeArrayShadow"}
            and len(method_args) >= 4
        ):
            constructor = self.sampled_array_coordinate_constructor_for_type(
                mapped_type
            )
            sampler = self.generate_expression(method_args[0], is_main)
            coord = self.generate_expression(method_args[1], is_main)
            layer = self.generate_expression(method_args[2], is_main)
            compare = self.generate_expression(method_args[3], is_main)
            return [sampler, f"{constructor}({coord}, {layer})", compare], 4

        return [self.generate_expression(arg, is_main) for arg in method_args[:3]], 3

    def texture_gather_argument(self, arg, is_main=False):
        component = self.texture_gather_component_index(arg)
        if component is not None:
            return component
        return self.generate_expression(arg, is_main)

    def texture_gather_component_index(self, arg):
        if not isinstance(arg, VariableNode):
            return None
        name = str(getattr(arg, "name", ""))
        parts = name.split("::")
        if len(parts) < 2 or parts[-2] != "component":
            return None
        return self.texture_gather_components.get(parts[-1])

    def is_multisample_resource_type(self, mapped_type):
        return bool(mapped_type and "MS" in str(mapped_type))

    def scalar_size_resource_type(self, mapped_type):
        return mapped_type in {
            "sampler1D",
            "isampler1D",
            "usampler1D",
            "samplerBuffer",
            "isamplerBuffer",
            "usamplerBuffer",
            "image1D",
            "iimage1D",
            "uimage1D",
        }

    def array_size_component(self, mapped_type):
        return "y" if mapped_type and "1DArray" in str(mapped_type) else "z"

    def resource_size_query_info(self, expr, is_main=False):
        if not isinstance(expr, MethodCallNode):
            return None
        method = expr.method
        if method not in self.texture_size_query_methods:
            return None

        obj = self.generate_expression(expr.object, is_main)
        mapped_type = self.expression_mapped_type(expr.object)
        query_function = (
            "imageSize"
            if self.is_storage_image_expression(expr.object)
            else "textureSize"
        )
        multisample = self.is_multisample_resource_type(mapped_type)

        lod = None
        if (
            query_function == "textureSize"
            and not multisample
            and method != "get_array_size"
        ):
            lod = self.generate_expression(expr.args[0], is_main) if expr.args else "0"

        components = {
            "get_width": "" if self.scalar_size_resource_type(mapped_type) else "x",
            "get_height": "y",
            "get_depth": "z",
            "get_array_size": self.array_size_component(mapped_type),
        }
        return {
            "object": obj,
            "function": query_function,
            "component": components[method],
            "lod": lod,
            "multisample": multisample,
        }

    def resource_size_query_call(self, info, lod=None):
        args = [info["object"]]
        if info["function"] == "textureSize" and not info["multisample"]:
            args.append(lod if lod is not None else info["lod"] or "0")
        return f"{info['function']}({', '.join(args)})"

    def resource_size_method_expression(self, expr, is_main=False):
        info = self.resource_size_query_info(expr, is_main)
        if info is None:
            return None
        size_call = self.resource_size_query_call(info)
        component = info["component"]
        return f"{size_call}.{component}" if component else size_call

    def texture_size_constructor_expression(self, expr, is_main=False):
        infos = [
            self.resource_size_query_info(arg, is_main)
            for arg in getattr(expr, "args", [])
        ]
        if not infos or any(info is None for info in infos):
            return None

        first = infos[0]
        if any(
            info["object"] != first["object"]
            or info["function"] != first["function"]
            or info["multisample"] != first["multisample"]
            for info in infos
        ):
            return None

        components = [info["component"] for info in infos]
        if components not in (["x", "y"], ["x", "y", "z"]):
            return None

        lods = {info["lod"] for info in infos if info["lod"] is not None}
        if len(lods) > 1:
            return None
        lod = next(iter(lods), None)
        return self.resource_size_query_call(first, lod)

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
        wide_vector_marker = "    // __crossgl_metal_wide_vector_support__\n"
        self.wide_vector_types = {}
        self.wide_vector_binary_helpers = set()
        self.wide_vector_compound_helpers = set()
        self.wide_vector_reserved_names = set()
        typedefs = getattr(ast, "typedefs", []) or []
        self.collect_callable_type_aliases(typedefs)
        self.validate_runtime_callable_alias_usage(ast)
        self.type_aliases = {
            alias.name: alias.alias_type
            for alias in typedefs
            if isinstance(alias, TypeAliasNode)
            and alias.name not in self.callable_type_aliases
        }
        self.type_alias_qualifiers = {
            alias.name: list(getattr(alias, "qualifiers", []) or [])
            for alias in typedefs
            if isinstance(alias, TypeAliasNode)
            and alias.name not in self.callable_type_aliases
        }
        # Body-local ``using`` and ``typedef`` aliases discovered while emitting
        # function bodies; these are inlined at their use sites rather than
        # emitted as typedefs.
        self.local_type_alias_names = set()
        self.local_struct_type_aliases = {}
        functions = getattr(ast, "functions", []) or []
        self.prepare_texture_usage(ast)
        self.wide_vector_reserved_names.update(
            self.collect_declared_identifier_names(ast)
        )
        self.user_function_names = {
            function.name
            for function in functions
            if isinstance(function, FunctionNode) and function.name
        }
        self.user_function_overloads_by_name = {}
        for function in functions:
            if isinstance(function, FunctionNode) and function.name:
                self.user_function_overloads_by_name.setdefault(
                    function.name, []
                ).append(function)
        self.prepare_value_template_materializations(ast, functions)
        self.prepare_metal_atomic_fence_transport_constants(ast)
        code = ""
        includes = getattr(ast, "includes", []) or []
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
        code += wide_vector_marker
        self.constant_struct_name = []

        # Get constants - support both 'constant' and 'constants' attributes
        constants = getattr(ast, "constant", []) or getattr(ast, "constants", []) or []
        for constant in constants:
            if isinstance(constant, ConstantBufferNode):
                self.process_constant_struct(ast)

        # Get structs - support both 'struct' and 'structs' attributes
        structs = getattr(ast, "structs", []) or getattr(ast, "struct", []) or []
        self.struct_name_map = self.build_struct_name_map(structs)
        self.struct_declarations = {
            struct_node.name: struct_node
            for struct_node in structs
            if isinstance(struct_node, StructNode)
            and getattr(struct_node, "name", None)
        }
        self.struct_member_types = self.collect_struct_member_types(structs)
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
        enums = getattr(ast, "enums", []) or []
        emitted_typedefs = [
            declaration
            for alias in typedefs
            if isinstance(alias, TypeAliasNode)
            for declaration in [self.format_type_alias_declaration(alias)]
            if declaration is not None
        ]
        if emitted_typedefs:
            code += "    // Typedefs\n"
            for declaration in emitted_typedefs:
                code += f"    {declaration}\n"
            code += "\n"

        if enums:
            code += "    // Enums\n"
            used_enum_names = {
                enum.name for enum in enums if isinstance(enum, EnumNode) and enum.name
            }
            anonymous_enum_index = 0
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
                    code += f"    enum {enum_name} {{\n"
                    for member_name, member_value in enum.members:
                        if member_value is not None:
                            value = self.generate_expression(member_value, False)
                            code += f"        {member_name} = {value},\n"
                        else:
                            code += f"        {member_name},\n"
                    code += "    };\n\n"
        for struct_node in structs:
            if isinstance(struct_node, StructNode):
                if getattr(struct_node, "aggregate_kind", None) == "union":
                    union_name = self.map_struct_name(struct_node.name or "anonymous")
                    code += (
                        f"    // Metal union {union_name} represented as "
                        "struct-like layout; overlapping storage is not modeled\n"
                    )
                if struct_node.name in self.constant_struct_name:
                    code += "    // cbuffers\n"
                    code += f"    cbuffer {self.map_struct_name(struct_node.name)} {{\n"
                else:
                    code += "    // Structs\n"
                    struct_alignas = ""
                    if hasattr(struct_node, "alignas") and struct_node.alignas:
                        parts = []
                        for item in struct_node.alignas:
                            if isinstance(item, tuple) and item[0] == "type":
                                parts.append(f"alignas({self.map_type(item[1])})")
                            else:
                                parts.append(
                                    f"alignas({self.generate_expression(item, False)})"
                                )
                        struct_alignas = " ".join(parts) + " "
                    code += (
                        f"    {self.format_generic_prefix(struct_node)}"
                        f"{struct_alignas}struct "
                        f"{self.map_struct_name(struct_node.name)} {{\n"
                    )
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
                    )
                    code += f"        {decl};\n"
                code += "    }\n\n"

        globals_list = [
            glob
            for glob in getattr(ast, "global_variables", []) or getattr(
                ast, "global_vars", []
            )
            if id(glob) not in self.metal_atomic_fence_transport_declaration_ids
        ]
        if globals_list:
            code += "    // Globals\n"
            for glob in globals_list:
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
                if isinstance(glob, AssignmentNode):
                    if isinstance(glob.left, VariableNode):
                        if self.is_sampler_variable(glob.left):
                            self.global_sampler_names.add(glob.left.name)
                            continue
                        self.global_variable_types[glob.left.name] = (
                            self.metal_declaration_expression_type(glob.left)
                        )
                    left = (
                        self.format_decl(glob.left, include_semantic=True)
                        if isinstance(glob.left, VariableNode)
                        else self.generate_expression(glob.left, False)
                    )
                    right = self.generate_initializer_value(
                        glob.right,
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
                    )
                    code += f"    {left} {glob.operator} {right};\n"
                elif isinstance(glob, VariableNode):
                    if self.is_sampler_variable(glob):
                        self.global_sampler_names.add(glob.name)
                        continue
                    self.global_variable_types[glob.name] = (
                        self.metal_declaration_expression_type(glob)
                    )
                    if id(glob) in self.storage_texture_declaration_ids:
                        self.global_storage_texture_names.add(glob.name)
                    if self.structured_buffer_pointer_type(glob):
                        self.global_structured_buffer_names.add(glob.name)
                    decl = self.format_global_decl(glob, include_semantic=True)
                    code += f"    {decl};\n"
            code += "\n"

        ordinary_function_code = []
        deferred_template_functions = []
        for f in functions:
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
                self.generate_top_level_function(
                    f,
                    template_value_bindings=default_bindings,
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
                    self.generate_top_level_function(function)
                )
        finally:
            self.preserve_unmaterialized_template_calls = False

        code += "".join(deferred_template_code)
        for key in self.ordered_value_template_specialization_keys(
            specialization_entries
        ):
            code += specialization_code[key]
        code += "".join(ordinary_function_code)

        code += "}\n"
        code = code.replace(
            wide_vector_marker,
            self.generate_wide_vector_support_code(indent=1),
            1,
        )
        return code

    def generate_top_level_function(
        self,
        function,
        *,
        template_value_bindings=None,
        template_type_bindings=None,
        output_name=None,
        specialization_key=None,
    ):
        qualifier = getattr(function, "qualifier", None)
        if qualifier == "vertex":
            return (
                "    // Vertex Shader\n"
                "    vertex {\n"
                + self.generate_function(
                    function,
                    stage_entry=function.name != "main",
                    template_value_bindings=template_value_bindings,
                    template_type_bindings=template_type_bindings,
                    output_name=output_name,
                    specialization_key=specialization_key,
                )
                + "    }\n\n"
            )
        if qualifier == "fragment":
            return (
                "    // Fragment Shader\n"
                "    fragment {\n"
                + self.generate_fragment_execution_layouts(function)
                + self.generate_function(
                    function,
                    stage_entry=function.name != "main",
                    template_value_bindings=template_value_bindings,
                    template_type_bindings=template_type_bindings,
                    output_name=output_name,
                    specialization_key=specialization_key,
                )
                + "    }\n\n"
            )
        if qualifier == "kernel":
            return (
                "    // Compute Shader\n"
                "    compute {\n"
                + self.generate_function(
                    function,
                    stage_entry=self.should_emit_kernel_stage_entry(function),
                    template_value_bindings=template_value_bindings,
                    template_type_bindings=template_type_bindings,
                    output_name=output_name,
                    specialization_key=specialization_key,
                )
                + "    }\n\n"
            )
        if qualifier in self.rt_qualifiers:
            return f"    // {qualifier} function\n" + self.generate_function(
                function,
                template_value_bindings=template_value_bindings,
                template_type_bindings=template_type_bindings,
                output_name=output_name,
                specialization_key=specialization_key,
            )
        return self.generate_function(
            function,
            template_value_bindings=template_value_bindings,
            template_type_bindings=template_type_bindings,
            output_name=output_name,
            specialization_key=specialization_key,
        )

    def process_constant_struct(self, node):
        constants = (
            getattr(node, "constant", []) or getattr(node, "constants", []) or []
        )
        structs = getattr(node, "structs", []) or getattr(node, "struct", []) or []
        for constant in constants:
            if isinstance(constant, ConstantBufferNode):
                self.constant_struct_name.extend(
                    struct.name for struct in structs if struct.name == constant.name
                )

    def collect_struct_member_types(self, structs):
        member_types = {}
        for struct_node in structs or []:
            struct_name = getattr(struct_node, "name", None)
            if not struct_name:
                continue
            members = {}
            for member in getattr(struct_node, "members", []) or []:
                member_name = getattr(member, "name", None)
                member_type = self.metal_declaration_expression_type(member)
                if member_name and member_type:
                    members[member_name] = member_type
            member_types[struct_name] = members
        return member_types

    def collect_struct_static_constants(self, structs):
        members = {}
        template_parameters = {}
        for struct_node in structs or []:
            struct_name = getattr(struct_node, "name", None)
            if not struct_name:
                continue
            mapped_name = self.map_struct_name(struct_name)
            template_parameters[mapped_name] = {
                name: kind
                for kind, name in (
                    getattr(struct_node, "template_parameters", None) or []
                )
                if name
            }
            for member in getattr(struct_node, "members", []) or []:
                if not self.is_compile_time_static_member(member):
                    continue
                member_name = getattr(member, "name", None)
                if member_name:
                    members[(mapped_name, member_name)] = member

        self.struct_static_constants = {}
        self.struct_static_constant_members = members
        self.struct_static_constant_resolution_stack = []
        self.struct_static_constexpr_member_keys = set()
        self.struct_template_parameters = template_parameters
        for key in members:
            self.resolve_struct_static_constant(key)

    def is_compile_time_static_member(self, member):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(member, "qualifiers", []) or []
        }
        return "constexpr" in qualifiers or (
            "static" in qualifiers
            and bool(qualifiers.intersection({"const", "constant"}))
        )

    def resolve_struct_static_constant(self, key):
        if key in self.struct_static_constants:
            return self.struct_static_constants[key]
        member = self.struct_static_constant_members.get(key)
        if member is None:
            return None
        default_value = getattr(member, "default_value", None)
        if default_value is None:
            return None
        if key in self.struct_static_constant_resolution_stack:
            cycle_start = self.struct_static_constant_resolution_stack.index(key)
            cycle = self.struct_static_constant_resolution_stack[cycle_start:] + [key]
            path = " -> ".join(f"{owner}::{name}" for owner, name in cycle)
            owner, member_name = key
            raise MetalStaticConstantResolutionError(
                owner,
                member_name,
                f"the initializer dependency chain is cyclic ({path})",
            )

        owner, _member_name = key
        self.struct_static_constant_resolution_stack.append(key)
        previous_owner = self.current_struct_static_constant_owner
        self.current_struct_static_constant_owner = owner
        try:
            local_integer_values = self.struct_static_integer_values(owner)
            integer_value = evaluate_literal_int_expression(
                default_value,
                local_integer_values,
            )
            if integer_value is not None:
                rendered = self.render_static_integral_value(member, integer_value)
            else:
                rendered = self.generate_expression(default_value, False)
                folded_value = self.evaluate_value_template_constant_expression(
                    rendered
                )
                if folded_value is not None:
                    rendered = self.render_static_integral_value(member, folded_value)
        except MetalTemplateArgumentResolutionError as error:
            raise self.contextualize_static_template_resolution_error(
                error,
                key,
                member,
            ) from error
        finally:
            self.current_struct_static_constant_owner = previous_owner
            self.struct_static_constant_resolution_stack.pop()
        if not rendered:
            return None
        self.struct_static_constants[key] = rendered
        return rendered

    def render_static_integral_value(self, member, value):
        member_type = self.normalized_metal_type(getattr(member, "vtype", None))
        if member_type == "bool":
            return "true" if value else "false"
        return str(value)

    def contextualize_static_template_resolution_error(self, error, key, member):
        owner, member_name = key
        return MetalTemplateArgumentResolutionError(
            error.function_name,
            error.parameter_name,
            error.argument_expression,
            error.selected_call,
            error.reason,
            error.argument_kind,
            getattr(member, "source_location", None) or error.source_location,
            owner=owner,
            member=member_name,
            requested_specialization=(
                getattr(error, "requested_specialization", None) or error.selected_call
            ),
        )

    def struct_static_integer_values(self, owner):
        values = {}
        for (
            candidate_owner,
            member_name,
        ), value in self.struct_static_constants.items():
            if candidate_owner != owner or not re.fullmatch(r"[-+]?\d+", value):
                continue
            values[member_name] = int(value)
            values[f"{owner}::{member_name}"] = int(value)
        return values

    def build_struct_name_map(self, structs):
        mapped_names = {}
        used_names = set()
        self.ambiguous_struct_names = set()
        for struct_node in structs or []:
            if not isinstance(struct_node, StructNode):
                continue
            raw_name = getattr(struct_node, "name", None)
            if not raw_name:
                continue
            if raw_name in mapped_names:
                self.ambiguous_struct_names.add(raw_name)
            base_name = self.sanitize_identifier(raw_name)
            candidate = base_name
            suffix = 2
            while candidate in used_names:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(candidate)
            mapped_names[raw_name] = candidate
        return mapped_names

    def format_generic_prefix(self, node, bound_parameter_names=None):
        defaults = getattr(node, "template_parameter_defaults", {}) or {}
        template_parameters = getattr(node, "template_parameters", None)
        bound_parameter_names = set(bound_parameter_names or ())
        if template_parameters:
            generics = []
            for kind, name in template_parameters:
                if not name or name in bound_parameter_names:
                    continue
                rendered = self.sanitize_identifier(name)
                default_type = defaults.get(name)
                if default_type and kind in {"typename", "class"}:
                    rendered = f"{rendered} = {self.map_type(default_type)}"
                generics.append(rendered)
        else:
            generics = [
                self.sanitize_identifier(name)
                for name in getattr(node, "generics", []) or []
                if name
            ]
        if not generics:
            return ""
        return f"generic<{', '.join(generics)}> "

    def map_struct_name(self, name):
        if not name:
            return name
        return self.struct_name_map.get(name, self.sanitize_identifier(name))

    def iter_ast_children(self, node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, dict):
            yield from node.values()
            return
        if isinstance(node, (list, tuple, set)):
            yield from node
            return
        yield from getattr(node, "__dict__", {}).values()

    def collect_callable_type_aliases(self, aliases):
        alias_nodes = {
            alias.name: alias
            for alias in aliases or []
            if isinstance(alias, TypeAliasNode) and getattr(alias, "name", None)
        }
        callable_aliases = {
            name: alias
            for name, alias in alias_nodes.items()
            if isinstance(alias, CallableTypeAliasNode)
            or getattr(alias, "is_function_type", False)
        }

        changed = True
        while changed:
            changed = False
            for name, alias in alias_nodes.items():
                if name in callable_aliases:
                    continue
                referenced_name = self.plain_type_alias_reference(alias.alias_type)
                if referenced_name not in callable_aliases:
                    continue
                callable_aliases[name] = callable_aliases[referenced_name]
                changed = True

        self.callable_type_aliases = callable_aliases
        self.callable_alias_declarations = {
            name: alias_nodes[name] for name in callable_aliases
        }

    def plain_type_alias_reference(self, type_name):
        text = str(type_name or "").strip()
        while text.endswith(("*", "&")):
            text = text[:-1].rstrip()
        qualifier_pattern = (
            r"^(?:(?:const|volatile|thread|threadgroup|device|constant|"
            r"restrict|__restrict|__restrict__)\s+)+"
        )
        text = re.sub(qualifier_pattern, "", text).strip()
        if self.crossgl_identifier_pattern.fullmatch(text):
            return text
        return None

    def validate_runtime_callable_alias_usage(self, ast):
        if not self.callable_type_aliases:
            return

        for node in self.iter_runtime_ast_nodes(ast):
            for attribute in ("vtype", "return_type", "target_type", "vector_type"):
                type_name = getattr(node, attribute, None)
                alias_name = self.callable_alias_reference(type_name)
                if alias_name is None:
                    continue
                alias = self.callable_type_aliases[alias_name]
                declaration = self.callable_alias_declarations[alias_name]
                raise MetalCallableAliasLoweringError(
                    alias_name,
                    self.format_callable_alias_signature(alias, alias_name),
                    self.callable_alias_usage(node, attribute),
                    "runtime callable values are not representable in CrossGL; "
                    "materialize the callback as a non-type template argument "
                    "or use a supported Metal function-table resource",
                    source_location=(
                        getattr(node, "source_location", None)
                        or getattr(declaration, "source_location", None)
                        or getattr(alias, "source_location", None)
                    ),
                )

    def iter_runtime_ast_nodes(self, ast):
        pending = [
            value
            for name, value in getattr(ast, "__dict__", {}).items()
            if name not in {"typedefs", "includes"}
        ]
        seen = set()
        while pending:
            node = pending.pop()
            if node is None or isinstance(node, (str, int, float, bool)):
                continue
            if isinstance(node, dict):
                pending.extend(node.values())
                continue
            if isinstance(node, (list, tuple, set)):
                pending.extend(node)
                continue
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            if isinstance(node, TypeAliasNode):
                continue
            yield node
            pending.extend(self.iter_ast_children(node))

    def callable_alias_reference(self, type_name):
        text = str(type_name or "").strip()
        if not text:
            return None

        carrier = text
        while carrier.endswith(("*", "&")):
            carrier = carrier[:-1].rstrip()
        base_name, generic_args = self.generic_type_parts(carrier)
        if base_name in {
            "visible_function_table",
            "intersection_function_table",
        } and any(
            self.plain_type_alias_reference(argument) in self.callable_type_aliases
            for argument in generic_args
        ):
            return None

        for alias_name in sorted(self.callable_type_aliases, key=len, reverse=True):
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(alias_name)}(?![A-Za-z0-9_])",
                text,
            ):
                return alias_name
        return None

    def callable_alias_usage(self, node, attribute):
        name = getattr(node, "name", None)
        node_name = type(node).__name__
        if name:
            return f"{node_name} '{name}' {attribute}"
        return f"{node_name} {attribute}"

    def format_callable_alias_signature(self, alias, alias_name):
        return_type = str(
            getattr(alias, "return_type", None) or alias.alias_type
        ).strip()
        qualifiers = [
            str(value).strip()
            for value in getattr(alias, "qualifiers", []) or []
            if str(value).strip()
        ]
        if qualifiers:
            return_type = " ".join(qualifiers + [return_type])

        parameters = getattr(alias, "parameters", None)
        if parameters is None:
            parameter_text = "..."
        else:
            rendered_parameters = []
            for parameter in parameters:
                parts = [
                    str(value).strip()
                    for value in getattr(parameter, "qualifiers", []) or []
                    if str(value).strip()
                ]
                parts.append(str(getattr(parameter, "vtype", "")).strip())
                rendered_parameters.append(" ".join(part for part in parts if part))
            parameter_text = ", ".join(rendered_parameters)

        indirection = str(getattr(alias, "indirection", "") or "")
        declarator = f"({indirection}{alias_name})" if indirection else alias_name
        return f"{return_type} {declarator}({parameter_text})"

    def value_template_parameter_names(self, function):
        return [
            entry[1]
            for entry in getattr(function, "template_parameters", None) or []
            if isinstance(entry, (tuple, list))
            and len(entry) >= 2
            and entry[0] == "value"
            and entry[1]
        ]

    def split_value_template_call_name(self, name, function_index=None):
        text = str(name).strip()
        if "<" not in text or not text.endswith(">"):
            return None
        angle = text.find("<")
        base = text[:angle].strip()
        if not base:
            return None
        arguments = self.split_generic_arguments(text[angle + 1 : -1])
        function_name = base.rsplit("::", 1)[-1]
        functions = (
            self.value_template_functions if function_index is None else function_index
        )
        if function_name not in functions:
            return None
        return function_name, arguments

    def value_template_overload_signature(self, function):
        parameters = ", ".join(self.metal_function_source_signature(function))
        template_parameters = ", ".join(
            name
            for _kind, name in getattr(function, "template_parameters", None) or []
            if name
        )
        return f"{function.name}<{template_parameters}>({parameters})"

    def resolve_value_template_function(
        self,
        function_name,
        call_arguments,
        explicit_template_arguments=None,
        function_index=None,
    ):
        functions = (
            self.value_template_functions if function_index is None else function_index
        )
        value_candidates = [
            function
            for function in functions.get(function_name, [])
            if len(getattr(function, "params", []) or []) == len(call_arguments)
        ]
        if not value_candidates:
            return None

        explicit_call = explicit_template_arguments is not None
        if explicit_call:
            explicit_count = len(explicit_template_arguments)
            value_candidates = [
                function
                for function in value_candidates
                if explicit_count
                <= len(getattr(function, "template_parameters", None) or [])
            ]
            overloads = value_candidates
        else:
            overloads = [
                function
                for function in self.user_function_overloads_by_name.get(
                    function_name, []
                )
                if len(getattr(function, "params", []) or []) == len(call_arguments)
            ]

        if not overloads:
            return None
        if len(overloads) == 1:
            return overloads[0] if overloads[0] in value_candidates else None

        argument_types = [
            self.expression_mapped_type(argument) for argument in call_arguments
        ]
        matching = []
        if all(argument_type is not None for argument_type in argument_types):
            matching = [
                function
                for function in overloads
                if self.metal_function_mapped_signature(function)
                == tuple(argument_types)
            ]
        if len(matching) == 1:
            return matching[0] if matching[0] in value_candidates else None

        # A non-template exact match wins over a same-signature function
        # template in C++. Leave that call on the ordinary overload.
        non_template_matches = [
            function
            for function in matching
            if not self.value_template_parameter_names(function)
        ]
        if len(non_template_matches) == 1:
            return None

        ambiguous = matching or overloads
        ambiguous_value_candidates = [
            function for function in ambiguous if function in value_candidates
        ]
        if len(ambiguous_value_candidates) <= 1 and not explicit_call:
            # Unknown argument types plus an unrelated overload are not enough
            # evidence to select the template. Preserve the call unchanged.
            return None
        signatures = ", ".join(
            self.value_template_overload_signature(function)
            for function in ambiguous_value_candidates
        )
        representative = ambiguous_value_candidates[0]
        raise MetalTemplateArgumentResolutionError(
            function_name,
            "<overload>",
            signatures,
            f"{function_name}(...)",
            "does not identify one unique value-template declaration",
            "overload",
            getattr(representative, "template_source_location", None)
            or getattr(representative, "source_location", None),
        )

    def prepare_value_template_materializations(self, ast, functions):
        self.value_template_functions = {}
        for function in functions:
            if (
                isinstance(function, FunctionNode)
                and getattr(function, "name", None)
                and self.value_template_parameter_names(function)
            ):
                self.value_template_functions.setdefault(function.name, []).append(
                    function
                )
        self.constexpr_value_template_functions = {
            name: [
                function
                for function in candidates
                if self.function_is_constexpr(function)
            ]
            for name, candidates in self.value_template_functions.items()
            if any(self.function_is_constexpr(function) for function in candidates)
        }
        self.constexpr_helper_values = {}
        self.constexpr_helper_resolution_stack = []
        self.default_value_template_bindings = {}
        self.value_template_specializations = {}
        self.pending_value_template_specializations = []
        self.value_template_specialization_dependencies = {}
        self.current_function_specialization_key = None
        self.max_template_specializations = getattr(
            ast,
            "max_template_specializations",
            DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT,
        )
        self.template_specialization_limit_source = getattr(
            ast,
            "template_specialization_limit_source",
            "max_template_specializations",
        )
        self.materialized_template_specialization_count = 0
        self.preserve_unmaterialized_template_calls = False
        self.suppressed_value_template_function_ids = set()
        self.existing_function_names = {
            function.name
            for function in functions
            if isinstance(function, FunctionNode) and getattr(function, "name", None)
        }
        if not self.value_template_functions:
            return

        omitted_calls = []
        seen = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if not isinstance(node, (dict, list, tuple, set)):
                node_id = id(node)
                if node_id in seen:
                    return
                seen.add(node_id)
            if isinstance(node, FunctionCallNode):
                call_name = str(getattr(node, "name", ""))
                unscoped_name = call_name.rsplit("::", 1)[-1]
                if unscoped_name in self.value_template_functions:
                    omitted_calls.append((unscoped_name, node.args))
            for child in self.iter_ast_children(node):
                visit(child)

        visit(ast)
        for function_name, call_arguments in omitted_calls:
            function = self.resolve_value_template_function(
                function_name, call_arguments
            )
            if function is None or id(function) in self.default_value_template_bindings:
                continue
            bindings = self.bind_value_template_arguments(
                function,
                [],
                selected_call=f"{function_name}(...)",
                require_defaults=False,
            )
            if bindings is None:
                continue
            self.default_value_template_bindings[id(function)] = bindings
            self.suppressed_value_template_function_ids.add(id(function))
            if all(
                kind == "value"
                for kind, name in getattr(function, "template_parameters", None) or []
                if name
            ):
                key = self.concrete_template_specialization_key(
                    function,
                    bindings,
                    {},
                )
                self.value_template_specializations[key] = function.name

    def function_is_constexpr(self, function):
        return "constexpr" in {
            str(qualifier).lower()
            for qualifier in getattr(function, "declaration_qualifiers", []) or []
        }

    def push_materialized_constexpr_expression_context(
        self,
        expression,
        *,
        required,
    ):
        if not self.current_function_materialization_bindings:
            return False
        self.materialized_constexpr_expression_contexts.append(
            {
                "function": self.current_function_name,
                "specialization": self.current_function_specialization,
                "required": required,
                "source_location": (
                    getattr(expression, "source_location", None)
                    or getattr(self.current_function, "source_location", None)
                ),
            }
        )
        return True

    def render_constexpr_helper_call(self, call, is_main=False):
        owner = self.current_struct_static_constant_owner
        resolving_static_member = bool(
            owner is not None and self.struct_static_constant_resolution_stack
        )
        materialization_context = (
            self.materialized_constexpr_expression_contexts[-1]
            if self.materialized_constexpr_expression_contexts
            else None
        )
        if not resolving_static_member and materialization_context is None:
            return None

        call_name = str(getattr(call, "name", ""))
        parsed = self.split_value_template_call_name(
            call_name,
            function_index=self.constexpr_value_template_functions,
        )
        if parsed is None:
            function_name = call_name.rsplit("::", 1)[-1]
            if function_name not in self.constexpr_value_template_functions:
                return None
            explicit_arguments = None
        else:
            function_name, explicit_arguments = parsed

        try:
            function = self.resolve_value_template_function(
                function_name,
                getattr(call, "args", []) or [],
                explicit_template_arguments=explicit_arguments,
                function_index=self.constexpr_value_template_functions,
            )
            if function is None:
                return None
            rendered = self.evaluate_struct_static_constexpr_helper(
                function,
                explicit_arguments,
                getattr(call, "args", []) or [],
                selected_call=call_name,
                is_main=is_main,
            )
        except MetalTemplateArgumentResolutionError as error:
            if (
                materialization_context is not None
                and not materialization_context["required"]
                and error.argument_kind in {"call", "constexpr_body"}
            ):
                return None
            if materialization_context is None or getattr(
                error, "enclosing_specialization", None
            ):
                raise
            raise self.contextualize_materialized_constexpr_resolution_error(
                error,
                call,
                materialization_context,
            ) from error
        if resolving_static_member:
            self.struct_static_constexpr_member_keys.add(
                self.struct_static_constant_resolution_stack[-1]
            )
        return rendered

    def contextualize_materialized_constexpr_resolution_error(
        self,
        error,
        call,
        context,
    ):
        return MetalTemplateArgumentResolutionError(
            error.function_name,
            error.parameter_name,
            error.argument_expression,
            error.selected_call,
            error.reason,
            error.argument_kind,
            getattr(call, "source_location", None)
            or error.source_location
            or context["source_location"],
            requested_specialization=(
                getattr(error, "requested_specialization", None) or error.selected_call
            ),
            enclosing_function=context["function"],
            enclosing_specialization=context["specialization"],
            nested_helper=str(getattr(call, "name", error.function_name)),
        )

    def evaluate_struct_static_constexpr_helper(
        self,
        function,
        explicit_arguments,
        call_arguments,
        *,
        selected_call,
        is_main,
    ):
        (
            template_bindings,
            value_bindings,
            type_bindings,
        ) = self.bind_struct_static_constexpr_template_arguments(
            function,
            explicit_arguments,
            selected_call=selected_call,
        )
        runtime_bindings = self.bind_struct_static_constexpr_call_arguments(
            function,
            call_arguments,
            selected_call=selected_call,
            is_main=is_main,
        )
        requested_specialization = self.constexpr_helper_specialization_name(
            function,
            template_bindings,
        )
        cache_key = (
            id(function),
            tuple(
                template_bindings.get(name)
                for _kind, name in getattr(function, "template_parameters", None) or []
                if name
            ),
            tuple(
                runtime_bindings.get(getattr(parameter, "name", None))
                for parameter in getattr(function, "params", []) or []
            ),
        )
        cached = self.constexpr_helper_values.get(cache_key)
        if cached is not None:
            return cached
        if cache_key in self.constexpr_helper_resolution_stack:
            raise MetalTemplateArgumentResolutionError(
                function.name,
                "<return>",
                requested_specialization,
                selected_call,
                "has a cyclic constexpr helper dependency",
                "constexpr_body",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
                requested_specialization=requested_specialization,
            )

        body = list(getattr(function, "body", None) or [])
        local_declarations = body[:-1]
        if (
            not body
            or not isinstance(body[-1], ReturnNode)
            or any(
                not self.is_supported_constexpr_local_declaration(statement)
                for statement in local_declarations
            )
        ):
            raise MetalTemplateArgumentResolutionError(
                function.name,
                "<return>",
                requested_specialization,
                selected_call,
                "contains statements outside the supported constexpr local "
                "declaration and return subset",
                "constexpr_body",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
                requested_specialization=requested_specialization,
            )

        return_type = self.substitute_template_value_text(
            getattr(function, "return_type", ""),
            bindings=type_bindings,
            honor_shadowing=False,
        )
        if not self.is_integral_constexpr_type(return_type):
            raise MetalTemplateArgumentResolutionError(
                function.name,
                "<return>",
                return_type,
                selected_call,
                "does not have an integral or boolean return type",
                "constexpr_body",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
                requested_specialization=requested_specialization,
            )

        previous_type_aliases = dict(self.type_aliases)
        previous_shadow_scopes = self.template_binding_shadow_scopes
        self.type_aliases.update(type_bindings)
        active_value_bindings = dict(value_bindings)
        active_value_bindings.update(runtime_bindings)
        self.template_value_bindings.append(active_value_bindings)
        self.template_binding_shadow_scopes = []
        self.constexpr_helper_resolution_stack.append(cache_key)
        try:
            for declaration in local_declarations:
                local_name = declaration.left.name
                local_expression = self.generate_expression(
                    declaration.right,
                    is_main,
                )
                local_value = self.normalize_struct_static_constexpr_expression(
                    local_expression,
                    function,
                    local_name,
                    selected_call,
                    requested_specialization,
                )
                active_value_bindings[local_name] = local_value
            rendered = self.generate_expression(body[-1].value, is_main)
        finally:
            self.constexpr_helper_resolution_stack.pop()
            self.template_binding_shadow_scopes = previous_shadow_scopes
            self.template_value_bindings.pop()
            self.type_aliases = previous_type_aliases

        result = self.normalize_struct_static_constexpr_expression(
            rendered,
            function,
            "<return>",
            selected_call,
            requested_specialization,
        )

        self.constexpr_helper_values[cache_key] = result
        return result

    def is_supported_constexpr_local_declaration(self, statement):
        if (
            not isinstance(statement, AssignmentNode)
            or getattr(statement, "operator", None) != "="
            or not isinstance(getattr(statement, "left", None), VariableNode)
            or not getattr(statement.left, "name", None)
        ):
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(statement.left, "qualifiers", []) or []
        }
        return "constexpr" in qualifiers and self.is_integral_constexpr_type(
            getattr(statement.left, "vtype", None)
        )

    def normalize_struct_static_constexpr_expression(
        self,
        rendered,
        function,
        expression_name,
        selected_call,
        requested_specialization,
    ):
        folded = self.evaluate_value_template_constant_expression(rendered)
        if folded is not None:
            return str(folded)

        dependencies = set(re.findall(r"\b[A-Za-z_]\w*\b", rendered))
        owner_parameters = self.current_struct_template_value_parameters()
        allowed = set(owner_parameters)
        allowed.update(self.constexpr_integral_type_names())
        allowed.update({"true", "false"})
        unresolved = sorted(dependencies - allowed)
        if unresolved or not dependencies.intersection(owner_parameters):
            reason = (
                f"remains dependent on {', '.join(unresolved)}"
                if unresolved
                else "cannot be evaluated as a supported integral constant expression"
            )
            raise MetalTemplateArgumentResolutionError(
                function.name,
                expression_name,
                rendered,
                selected_call,
                reason,
                "constexpr_body",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
                requested_specialization=requested_specialization,
            )
        return f"({rendered})"

    def bind_struct_static_constexpr_template_arguments(
        self,
        function,
        explicit_arguments,
        *,
        selected_call,
    ):
        parameters = getattr(function, "template_parameters", None) or []
        defaults = getattr(function, "template_parameter_defaults", {}) or {}
        explicit_arguments = list(explicit_arguments or [])
        template_bindings = {}
        value_bindings = {}
        type_bindings = {}
        argument_index = 0

        for kind, name in parameters:
            if not name:
                continue
            explicit = argument_index < len(explicit_arguments)
            if explicit:
                expression = explicit_arguments[argument_index]
                argument_index += 1
                argument_kind = "explicit" if kind == "value" else "explicit_type"
            else:
                expression = defaults.get(name)
                argument_kind = "default" if kind == "value" else "default_type"
                if expression is None:
                    raise MetalTemplateArgumentResolutionError(
                        function.name,
                        name,
                        None,
                        selected_call,
                        "was not supplied and has no declaration default",
                        "missing",
                        getattr(function, "template_source_location", None)
                        or getattr(function, "source_location", None),
                    )

            if kind == "value":
                resolved = self.resolve_struct_static_constexpr_value_argument(
                    expression,
                    template_bindings,
                    function,
                    name,
                    selected_call,
                    argument_kind,
                )
                value_bindings[name] = resolved
            else:
                resolved = self.resolve_struct_static_constexpr_type_argument(
                    expression,
                    template_bindings,
                    function,
                    name,
                    selected_call,
                    argument_kind,
                )
                type_bindings[name] = resolved
            template_bindings[name] = resolved

        if argument_index != len(explicit_arguments):
            return {}, {}, {}
        return template_bindings, value_bindings, type_bindings

    def resolve_struct_static_constexpr_value_argument(
        self,
        expression,
        earlier_bindings,
        function,
        parameter_name,
        selected_call,
        argument_kind,
    ):
        text = self.struct_static_constexpr_argument_text(
            expression,
            earlier_bindings,
        )
        normalized = re.sub(r"\s+", " ", text).strip()
        if normalized in {"true", "false"}:
            return normalized
        integer = re.fullmatch(r"([-+]?\d+)(?:[uUlL]+)?", normalized)
        if integer is not None:
            return integer.group(1)
        value = self.evaluate_value_template_constant_expression(normalized)
        if value is not None:
            return str(value)

        identifiers = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", normalized)))
        unresolved = sorted(
            set(identifiers) - self.current_struct_template_value_parameters()
        )
        if identifiers and not unresolved:
            return normalized
        reason = (
            f"remains dependent on {', '.join(unresolved or identifiers)}"
            if identifiers
            else "cannot be evaluated as a supported integral constant expression"
        )
        raise MetalTemplateArgumentResolutionError(
            function.name,
            parameter_name,
            str(expression),
            selected_call,
            reason,
            argument_kind,
            getattr(function, "template_source_location", None)
            or getattr(function, "source_location", None),
        )

    def resolve_struct_static_constexpr_type_argument(
        self,
        expression,
        earlier_bindings,
        function,
        parameter_name,
        selected_call,
        argument_kind,
    ):
        text = self.struct_static_constexpr_argument_text(
            expression,
            earlier_bindings,
        )
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
        helper_parameters = {
            name
            for _kind, name in getattr(function, "template_parameters", None) or []
            if name
        }
        owner_parameters = set(
            self.struct_template_parameters.get(
                self.current_struct_static_constant_owner,
                {},
            )
        )
        unresolved = sorted((identifiers & helper_parameters) - owner_parameters)
        if unresolved:
            raise MetalTemplateArgumentResolutionError(
                function.name,
                parameter_name,
                str(expression),
                selected_call,
                f"remains dependent on {', '.join(unresolved)}",
                argument_kind,
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
            )
        return text.strip()

    def struct_static_constexpr_argument_text(self, expression, earlier_bindings):
        if self.materialized_constexpr_expression_contexts:
            bindings = {
                name: value
                for scope in self.template_value_bindings
                for name, value in scope.items()
                if not self.template_value_binding_is_shadowed(name)
            }
        else:
            bindings = dict(self.struct_static_owner_member_bindings())
            for scope in self.template_value_bindings:
                bindings.update(scope)
        bindings.update(earlier_bindings)
        return self.substitute_template_value_text(
            str(expression),
            bindings=bindings,
            honor_shadowing=False,
        )

    def struct_static_owner_member_bindings(self):
        owner = self.current_struct_static_constant_owner
        return {
            member_name: value
            for (
                candidate_owner,
                member_name,
            ), value in self.struct_static_constants.items()
            if candidate_owner == owner
        }

    def current_struct_template_value_parameters(self):
        return {
            name
            for name, kind in self.struct_template_parameters.get(
                self.current_struct_static_constant_owner,
                {},
            ).items()
            if kind == "value"
        }

    def bind_struct_static_constexpr_call_arguments(
        self,
        function,
        call_arguments,
        *,
        selected_call,
        is_main,
    ):
        bindings = {}
        allowed = self.current_struct_template_value_parameters()
        for parameter, argument in zip(
            getattr(function, "params", []) or [],
            call_arguments,
        ):
            rendered = self.generate_expression(argument, is_main)
            folded = self.evaluate_value_template_constant_expression(rendered)
            if folded is not None:
                rendered = str(folded)
            else:
                identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", rendered))
                unresolved = sorted(identifiers - allowed)
                if unresolved or not identifiers:
                    reason = (
                        f"remains dependent on {', '.join(unresolved)}"
                        if unresolved
                        else "cannot be evaluated as a supported integral "
                        "constant expression"
                    )
                    raise MetalTemplateArgumentResolutionError(
                        function.name,
                        getattr(parameter, "name", None) or "<argument>",
                        rendered,
                        selected_call,
                        reason,
                        "call",
                        getattr(argument, "source_location", None)
                        or getattr(function, "source_location", None),
                    )
            bindings[getattr(parameter, "name", None)] = rendered
        return bindings

    def constexpr_helper_specialization_name(self, function, bindings):
        parameters = [
            name
            for _kind, name in getattr(function, "template_parameters", None) or []
            if name
        ]
        if not parameters:
            return function.name
        arguments = ",".join(str(bindings[name]) for name in parameters)
        return f"{function.name}<{arguments}>"

    def is_integral_constexpr_type(self, type_name):
        resolved = self.normalized_metal_type(self.resolve_type_alias(type_name))
        return resolved in self.constexpr_integral_type_names()

    @staticmethod
    def constexpr_integral_type_names():
        return {
            "bool",
            "char",
            "uchar",
            "short",
            "ushort",
            "int",
            "uint",
            "long",
            "ulong",
            "size_t",
            "ptrdiff_t",
            "int8_t",
            "uint8_t",
            "int16_t",
            "uint16_t",
            "int32_t",
            "uint32_t",
            "int64_t",
            "uint64_t",
        }

    def bind_value_template_arguments(
        self,
        function,
        explicit_arguments,
        *,
        selected_call,
        require_defaults=True,
    ):
        parameters = getattr(function, "template_parameters", None) or []
        defaults = getattr(function, "template_parameter_defaults", {}) or {}
        explicit_arguments = list(explicit_arguments or [])
        bindings = {}
        argument_index = 0
        for entry in parameters:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            kind, name = entry[0], entry[1]
            if not name:
                continue
            explicit = argument_index < len(explicit_arguments)
            if explicit:
                argument = explicit_arguments[argument_index]
                argument_index += 1
                if kind != "value":
                    raise MetalTemplateArgumentResolutionError(
                        function.name,
                        name,
                        argument,
                        selected_call,
                        (
                            "cannot be preserved by the value-template "
                            "specialization identity"
                        ),
                        "explicit_type",
                        getattr(function, "template_source_location", None)
                        or getattr(function, "source_location", None),
                    )
                bindings[name] = self.resolve_value_template_constant(
                    argument,
                    bindings,
                    function,
                    name,
                    selected_call,
                    is_default=False,
                )
                continue
            if kind != "value":
                continue
            default = defaults.get(name)
            if default is None:
                if require_defaults:
                    raise MetalTemplateArgumentResolutionError(
                        function.name,
                        name,
                        None,
                        selected_call,
                        "was not supplied and has no declaration default",
                        "missing",
                        getattr(function, "template_source_location", None)
                        or getattr(function, "source_location", None),
                    )
                return None
            bindings[name] = self.resolve_value_template_constant(
                default,
                bindings,
                function,
                name,
                selected_call,
                is_default=True,
            )

        if argument_index != len(explicit_arguments):
            return None
        value_names = self.value_template_parameter_names(function)
        if any(name not in bindings for name in value_names):
            return None
        return bindings

    def resolve_value_template_constant(
        self,
        expression,
        earlier_bindings,
        function,
        parameter_name,
        selected_call,
        *,
        is_default,
    ):
        text = self.substitute_template_value_text(
            str(expression), bindings=earlier_bindings, honor_shadowing=False
        ).strip()
        active_bindings = {
            name: value
            for scope in self.template_value_bindings
            for name, value in scope.items()
            if not self.template_value_binding_is_shadowed(name)
        }
        if active_bindings:
            text = self.substitute_template_value_text(
                text, bindings=active_bindings, honor_shadowing=False
            ).strip()
        normalized = re.sub(r"\s+", " ", text)
        if normalized in {"true", "false"}:
            return normalized
        integer = re.fullmatch(r"([-+]?\d+)(?:[uUlL]+)?", normalized)
        if integer is not None:
            return integer.group(1)

        value = self.evaluate_value_template_constant_expression(normalized)
        if value is not None:
            return str(value)

        identifiers = sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", normalized)))
        dependency = (
            f"remains dependent on {', '.join(identifiers)}"
            if identifiers
            else "cannot be evaluated as a supported integral constant expression"
        )
        raise MetalTemplateArgumentResolutionError(
            function.name,
            parameter_name,
            str(expression),
            selected_call,
            dependency,
            "default" if is_default else "explicit",
            getattr(function, "template_source_location", None)
            or getattr(function, "source_location", None),
        )

    def evaluate_value_template_constant_expression(self, expression):
        try:
            lexer = MetalLexer(expression, preprocess=False)
            parser = MetalParser(lexer.tokenize())
            node = parser.parse_expression()
            if parser.current_token[0] != "EOF":
                return None
        except (SyntaxError, ValueError, TypeError):
            return None
        return evaluate_literal_int_expression(node)

    def bind_concrete_function_template_arguments(
        self,
        function,
        explicit_arguments,
        *,
        selected_call,
    ):
        parameters = [
            entry
            for entry in getattr(function, "template_parameters", None) or []
            if isinstance(entry, (tuple, list)) and len(entry) >= 2 and entry[1]
        ]
        defaults = getattr(function, "template_parameter_defaults", {}) or {}
        explicit_arguments = list(explicit_arguments or [])
        if len(explicit_arguments) > len(parameters):
            raise MetalTemplateArgumentResolutionError(
                function.name,
                "<arguments>",
                ", ".join(str(argument) for argument in explicit_arguments),
                selected_call,
                "supplies more arguments than the selected template declares",
                "explicit",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
            )

        all_bindings = {}
        value_bindings = {}
        type_bindings = {}
        for index, (kind, name) in enumerate(parameters):
            explicit = index < len(explicit_arguments)
            expression = (
                explicit_arguments[index] if explicit else defaults.get(name)
            )
            if expression is None:
                raise MetalTemplateArgumentResolutionError(
                    function.name,
                    name,
                    None,
                    selected_call,
                    "was not supplied and has no declaration default",
                    "missing",
                    getattr(function, "template_source_location", None)
                    or getattr(function, "source_location", None),
                )
            if kind == "value":
                resolved = self.resolve_value_template_constant(
                    expression,
                    all_bindings,
                    function,
                    name,
                    selected_call,
                    is_default=not explicit,
                )
                value_bindings[name] = resolved
            else:
                resolved = self.resolve_concrete_template_type_argument(
                    expression,
                    all_bindings,
                    function,
                    name,
                    selected_call,
                    is_default=not explicit,
                )
                type_bindings[name] = resolved
            all_bindings[name] = resolved
        return value_bindings, type_bindings

    def resolve_concrete_template_type_argument(
        self,
        expression,
        earlier_bindings,
        function,
        parameter_name,
        selected_call,
        *,
        is_default,
    ):
        text = self.substitute_template_value_text(
            str(expression),
            bindings=earlier_bindings,
            honor_shadowing=False,
        ).strip()
        active_type_bindings = {
            name: value
            for scope in self.template_type_bindings
            for name, value in scope.items()
        }
        type_bindings = dict(active_type_bindings)
        type_bindings.update(earlier_bindings)
        for name in sorted(type_bindings, key=len, reverse=True):
            text = re.sub(
                rf"\b{re.escape(name)}\b",
                str(type_bindings[name]),
                text,
            )
        text = re.sub(r"^typename\s+", "", text).strip()
        text = self.resolve_local_type_aliases(text)
        if text in self.local_struct_type_aliases:
            text = self.local_struct_type_aliases[text]
        text = self.resolve_type_alias(text)

        dependent_names = {
            name
            for candidate in (self.current_function, function)
            for _kind, name in getattr(candidate, "template_parameters", None) or []
            if name
        }
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", str(text)))
        unresolved = sorted(identifiers.intersection(dependent_names))
        if not text or unresolved:
            reason = (
                f"remains dependent on {', '.join(unresolved)}"
                if unresolved
                else "does not identify a concrete type"
            )
            raise MetalTemplateArgumentResolutionError(
                function.name,
                parameter_name,
                str(expression),
                selected_call,
                reason,
                "default_type" if is_default else "explicit_type",
                getattr(function, "template_source_location", None)
                or getattr(function, "source_location", None),
            )
        return re.sub(r"\s+", " ", str(text)).strip()

    def concrete_template_specialization_key(
        self,
        function,
        value_bindings,
        type_bindings,
    ):
        return (
            id(function),
            tuple(
                (
                    kind,
                    (
                        value_bindings[name]
                        if kind == "value"
                        else type_bindings[name]
                    ),
                )
                for kind, name in getattr(function, "template_parameters", None) or []
                if name
            ),
        )

    def concrete_template_specialization_arguments(
        self,
        function,
        value_bindings,
        type_bindings,
    ):
        return [
            value_bindings[name] if kind == "value" else type_bindings[name]
            for kind, name in getattr(function, "template_parameters", None) or []
            if name
        ]

    def value_template_specialization_identifier(
        self,
        function,
        value_bindings,
        type_bindings,
    ):
        values = self.concrete_template_specialization_arguments(
            function,
            value_bindings,
            type_bindings,
        )
        parts = [function.name, *values]
        identifier = "_".join(
            part
            for part in (
                re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") for value in parts
            )
            if part
        )
        identifier = identifier or function.name
        if len(self.value_template_functions.get(function.name, [])) > 1:
            signature = "_".join(self.metal_function_source_signature(function))
            if signature:
                identifier = f"{identifier}_{signature}"
        return self.sanitize_identifier(identifier)

    def reserve_value_template_specialization_identifier(
        self,
        function,
        value_bindings,
        type_bindings,
    ):
        base = self.value_template_specialization_identifier(
            function,
            value_bindings,
            type_bindings,
        )
        candidate = base
        suffix = 2
        while candidate in self.existing_function_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.existing_function_names.add(candidate)
        return candidate

    def materialized_value_template_call_name(
        self,
        name,
        call_arguments,
        source_location=None,
    ):
        if self.preserve_unmaterialized_template_calls:
            return name
        parsed = self.split_value_template_call_name(name)
        if parsed is None:
            return name
        function_name, arguments = parsed
        try:
            function = self.resolve_value_template_function(
                function_name,
                call_arguments,
                explicit_template_arguments=arguments,
            )
            if function is None:
                return name
            value_bindings, type_bindings = (
                self.bind_concrete_function_template_arguments(
                    function,
                    arguments,
                    selected_call=str(name),
                )
            )
        except MetalTemplateArgumentResolutionError as error:
            raise self.contextualize_template_call_resolution_error(
                error,
                name,
                source_location,
            ) from error
        key = self.concrete_template_specialization_key(
            function,
            value_bindings,
            type_bindings,
        )
        self.record_value_template_specialization_dependency(key)
        existing = self.value_template_specializations.get(key)
        if existing is not None:
            return existing

        requested_arguments = tuple(
            str(argument)
            for argument in self.concrete_template_specialization_arguments(
                function,
                value_bindings,
                type_bindings,
            )
        )
        next_count = self.materialized_template_specialization_count + 1
        if next_count > self.max_template_specializations:
            requested_signature = (
                f"{function.name}<{', '.join(requested_arguments)}>"
            )
            raise MetalTemplateSpecializationError(
                "Metal template specialization limit exceeded while "
                f"materializing '{requested_signature}'; {next_count} unique "
                f"concrete signatures requested, limit "
                f"{self.max_template_specializations} from "
                f"{self.template_specialization_limit_source}.",
                limit=self.max_template_specializations,
                limit_source=self.template_specialization_limit_source,
                unique_specialization_count=next_count,
                requested_signature=requested_signature,
                source_location=source_location,
                caller_specialization=self.current_function_specialization,
                callee_template=function.name,
                requested_arguments=requested_arguments,
            )
        specialized_name = self.reserve_value_template_specialization_identifier(
            function,
            value_bindings,
            type_bindings,
        )
        self.materialized_template_specialization_count = next_count
        self.value_template_specializations[key] = specialized_name
        self.suppressed_value_template_function_ids.add(id(function))
        self.pending_value_template_specializations.append(
            (
                function,
                value_bindings,
                type_bindings,
                specialized_name,
                key,
            )
        )
        self.user_function_names.add(specialized_name)
        self.user_function_overloads_by_name.setdefault(specialized_name, []).append(
            function
        )
        return specialized_name

    def contextualize_template_call_resolution_error(
        self,
        error,
        call_name,
        source_location,
    ):
        if (
            getattr(error, "enclosing_specialization", None)
            or self.current_function_specialization_key is None
        ):
            return error
        return MetalTemplateArgumentResolutionError(
            error.function_name,
            error.parameter_name,
            error.argument_expression,
            error.selected_call,
            error.reason,
            error.argument_kind,
            source_location or error.source_location,
            requested_specialization=(
                getattr(error, "requested_specialization", None)
                or error.selected_call
            ),
            enclosing_function=self.current_function_name,
            enclosing_specialization=self.current_function_specialization,
            nested_helper=str(call_name),
            enclosing_context="template call",
        )

    def record_value_template_specialization_dependency(self, callee_key):
        caller_key = self.current_function_specialization_key
        if caller_key is None:
            return
        dependencies = self.value_template_specialization_dependencies.setdefault(
            caller_key,
            [],
        )
        if callee_key not in dependencies:
            dependencies.append(callee_key)

    def ordered_value_template_specialization_keys(self, entries):
        states = {}
        ordered = []
        stack = []

        def visit(key):
            state = states.get(key, 0)
            if state == 2:
                return
            if state == 1:
                cycle_start = stack.index(key)
                cycle = stack[cycle_start:] + [key]
                cycle_names = [
                    self.value_template_specializations.get(item, "<specialization>")
                    for item in cycle
                ]
                function, value_bindings, type_bindings, _name = entries[key]
                requested_arguments = tuple(
                    str(argument)
                    for argument in self.concrete_template_specialization_arguments(
                        function,
                        value_bindings,
                        type_bindings,
                    )
                )
                raise MetalTemplateSpecializationError(
                    "Metal template specialization graph is recursive: "
                    + " -> ".join(cycle_names),
                    requested_signature=(
                        f"{function.name}<{', '.join(requested_arguments)}>"
                    ),
                    caller_specialization=cycle_names[-2],
                    callee_template=function.name,
                    requested_arguments=requested_arguments,
                )

            states[key] = 1
            stack.append(key)
            for dependency in self.value_template_specialization_dependencies.get(
                key,
                [],
            ):
                if dependency in entries:
                    visit(dependency)
            stack.pop()
            states[key] = 2
            ordered.append(key)

        for key in entries:
            visit(key)
        return ordered

    def template_value_binding_is_shadowed(self, name):
        return any(
            name in scope for scope in reversed(self.template_binding_shadow_scopes)
        )

    def template_value_binding(self, name):
        if self.template_value_binding_is_shadowed(name):
            return None
        for bindings in reversed(self.template_value_bindings):
            if name in bindings:
                return bindings[name]
        return None

    def substitute_template_value_text(
        self, text, *, bindings=None, honor_shadowing=True
    ):
        text = str(text)
        if bindings is None:
            bindings = {
                name: value
                for scope in self.template_value_bindings
                for name, value in scope.items()
                if not honor_shadowing
                or not self.template_value_binding_is_shadowed(name)
            }
        if not bindings:
            return text
        return re.sub(
            r"\b[A-Za-z_]\w*\b",
            lambda match: str(bindings.get(match.group(0), match.group(0))),
            text,
        )

    def substitute_template_type_text(self, text, *, bindings=None):
        text = str(text)
        if bindings is None:
            bindings = {
                name: value
                for scope in self.template_type_bindings
                for name, value in scope.items()
            }
        if not bindings:
            return text
        return re.sub(
            r"\b[A-Za-z_]\w*\b",
            lambda match: str(bindings.get(match.group(0), match.group(0))),
            text,
        )

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
        self.global_storage_texture_names = set()
        self.current_storage_texture_names = set()
        self.global_structured_buffer_names = set()
        self.current_structured_buffer_names = set()
        self.current_stage_entry_resource_parameter_ids = set()
        self.global_sampler_names = set()
        self.user_function_names = set()
        self.user_function_overloads_by_name = {}
        self.identifier_maps = [{}]
        self.used_identifier_names = [set()]
        self.storage_texture_declaration_ids = (
            self.collect_storage_texture_declaration_ids(ast)
        )
        self.struct_member_types = {}
        self.struct_declarations = {}
        self.struct_name_map = {}
        self.ambiguous_struct_names = set()
        self.struct_static_constants = {}
        self.struct_static_constant_members = {}
        self.struct_static_constant_resolution_stack = []
        self.struct_static_constexpr_member_keys = set()
        self.current_struct_static_constant_owner = None
        self.struct_template_parameters = {}
        self.local_struct_type_aliases = {}
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

    def format_array_suffix(self, var, include_declarator_arrays=True):
        array_type = self.metal_array_type_parts(getattr(var, "vtype", None))
        suffix = f"[{self.format_array_extent(array_type[1])}]" if array_type else ""
        if not include_declarator_arrays:
            return suffix
        return suffix + self.format_declarator_array_suffix(var)

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
            self.metal_array_type_parts(getattr(var, "vtype", None))
        )

    def map_variable_type(self, var):
        raw_type = getattr(var, "vtype", None)
        constant_buffer_type = self.constant_buffer_pointer_type(var)
        if constant_buffer_type:
            return constant_buffer_type
        structured_buffer_type = self.structured_buffer_pointer_type(var)
        if structured_buffer_type:
            return structured_buffer_type
        array_type = self.metal_array_type_parts(raw_type)
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
        if resolved_type != type_to_map and self.is_metal_resource_type(resolved_type):
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
        mapped_type = self.map_variable_type(var)
        semantics = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(var, "attributes", []) or []
        }
        if semantics.intersection({"vertex_id", "instance_id", "primitive_id"}):
            return "int"
        return mapped_type

    def address_space_qualifier_prefix(self, var):
        if self.constant_buffer_pointer_type(
            var
        ) or self.structured_buffer_pointer_type(var):
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

    def effective_declaration_qualifiers(self, var):
        """Return direct qualifiers plus those carried by a resolved type alias."""
        qualifiers = [
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        ]
        metal_type = str(getattr(var, "vtype", "") or "").strip()
        while metal_type.endswith(("*", "&")):
            metal_type = metal_type[:-1].strip()

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
        resolved_type = self.resolve_type_alias(getattr(var, "vtype", None))
        return bool(
            self.pointer_element_type(resolved_type)
            or self.reference_element_type(resolved_type)
            or self.is_metal_resource_type(resolved_type)
        )

    def resource_memory_qualifier_prefix(self, var):
        qualifiers = self.resource_memory_qualifiers(var)
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
            type_str = f"{mapped_type}{type_array_suffix}"
        address_space = self.address_space_qualifier_prefix(var)
        qualifiers = set(self.effective_declaration_qualifiers(var))
        lowered_buffer_type = self.constant_buffer_pointer_type(
            var
        ) or self.structured_buffer_pointer_type(var)
        const_device_pointer = bool(
            "const" in qualifiers
            and "device" in qualifiers
            and self.pointer_element_type(getattr(var, "vtype", None)) is not None
        )
        const_str = (
            "const "
            if (getattr(var, "is_const", False) or const_device_pointer)
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
        semantic = " ".join(
            part for part in [address_space_annotations, semantic] if part
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

    def format_struct_member_decl(self, member, owner=None):
        """Render a struct field while preserving compile-time member constants."""
        declaration = self.format_decl(member, include_semantic=True)
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
            else:
                rendered_default = self.generate_expression(default_value, False)
            declaration += f" = {rendered_default}"
        return declaration

    def format_parameter_decl(self, var, index, semantic_context=None):
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
            return self.with_parameter_direction_qualifier(
                var, declaration, semantic_context=semantic_context
            )
        finally:
            var.name = original_name

    def with_parameter_direction_qualifier(
        self, var, declaration, semantic_context=None
    ):
        qualifiers = [
            str(qualifier).lower() for qualifier in getattr(var, "qualifiers", []) or []
        ]
        is_reference = self.reference_parameter(var)
        for qualifier in self.parameter_direction_qualifiers:
            if qualifier in qualifiers:
                if is_reference:
                    declaration = re.sub(r"(?<=\S)&(?=\s)", "", declaration, count=1)
                return f"{qualifier} {declaration}"
        if is_reference and not self.readonly_parameter(var, qualifiers):
            declaration = re.sub(r"(?<=\S)&(?=\s)", "", declaration, count=1)
            return f"inout {declaration}"
        if (
            is_reference
            and self.readonly_parameter(var, qualifiers)
            and getattr(var, "name", None) == "self"
            and "thread" in qualifiers
        ):
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
        code += "    " * indent
        implicit_buffer_bindings = (
            self.apply_implicit_stage_entry_buffer_bindings(func) if stage_entry else []
        )
        previous_variable_types = self.current_variable_types
        self.current_variable_types = dict(self.global_variable_types)
        previous_type_aliases = dict(self.type_aliases)
        previous_type_alias_qualifiers = dict(self.type_alias_qualifiers)
        previous_local_type_alias_names = set(self.local_type_alias_names)
        previous_local_struct_type_aliases = dict(self.local_struct_type_aliases)
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
        previous_function_specialization = self.current_function_specialization
        previous_function_materialization_bindings = (
            self.current_function_materialization_bindings
        )
        previous_function_specialization_key = self.current_function_specialization_key
        self.current_function_name = func.name
        self.current_function_return_type = func.return_type
        self.current_function = func
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
        try:
            for param in func.params:
                self.current_variable_types[param.name] = (
                    self.metal_declaration_expression_type(param)
                )
                if id(param) in self.storage_texture_declaration_ids:
                    self.current_storage_texture_names.add(param.name)
                if self.structured_buffer_pointer_type(param):
                    self.current_structured_buffer_names.add(param.name)
            semantic_context = {
                "kind": "parameter",
                "function_qualifier": getattr(func, "qualifier", None),
            }
            params = ", ".join(
                self.format_parameter_decl(p, index, semantic_context=semantic_context)
                for index, p in enumerate(func.params)
            )
            fn_semantic = self.map_semantic(self.function_semantic_attributes(func))
            suffix = f" {fn_semantic}" if fn_semantic else ""
            function_name = self.sanitize_identifier(
                output_name or self.function_output_name(func)
            )
            self.current_function_specialization = function_name
            self.current_function_materialization_bindings = {
                **active_type_bindings,
                **active_value_bindings,
            }
            self.current_function_specialization_key = specialization_key
            return_type = self.map_function_return_type(func.return_type)
            generic_prefix = (
                ""
                if getattr(func, "qualifier", None)
                else self.format_generic_prefix(
                    func,
                    bound_parameter_names={
                        *active_type_bindings,
                        *active_value_bindings,
                    },
                )
            )
            value_param_decls = (
                ""
                if generic_prefix
                else self.format_value_template_parameter_declarations(
                    func,
                    indent + 1,
                    body=func.body,
                    bound_value_names=active_value_bindings,
                )
            )
            code += (
                f"{generic_prefix}{return_type} {function_name}({params})"
                f"{suffix} {{\n"
            )
            code += value_param_decls
            code += self.generate_function_body(func.body, indent=indent + 1)
            code += "    }\n\n"
        finally:
            for param, attributes in implicit_buffer_bindings:
                param.attributes = attributes
            self.pop_identifier_scope()
            self.template_binding_shadow_scopes.pop()
            if active_type_bindings:
                self.template_type_bindings.pop()
            if active_value_bindings:
                self.template_value_bindings.pop()
            self.current_variable_types = previous_variable_types
            self.type_aliases = previous_type_aliases
            self.type_alias_qualifiers = previous_type_alias_qualifiers
            self.local_type_alias_names = previous_local_type_alias_names
            self.local_struct_type_aliases = previous_local_struct_type_aliases
            self.current_storage_texture_names = previous_storage_texture_names
            self.current_structured_buffer_names = previous_structured_buffer_names
            self.current_stage_entry_resource_parameter_ids = (
                previous_stage_entry_resource_parameter_ids
            )
            self.current_function_name = previous_function_name
            self.current_function_return_type = previous_function_return_type
            self.current_function = previous_function
            self.current_function_specialization = previous_function_specialization
            self.current_function_materialization_bindings = (
                previous_function_materialization_bindings
            )
            self.current_function_specialization_key = (
                previous_function_specialization_key
            )
        return code

    def format_value_template_parameter_declarations(
        self, func, indent, body=None, bound_value_names=None
    ):
        """Declare non-type (value) template parameters that a body consumes with
        a bitwise/shift operator.

        Entry-point kernels drop the ``generic<...>`` prefix, so value template
        parameters such as ``const int bits`` are referenced in the body with no
        declaration and default to ``float`` - which turns bitwise uses like
        ``bits & (bits - 1)`` into invalid float operations. An uninstantiated
        generic kernel is symbolic (not runnable) regardless, so emitting an
        integer placeholder simply keeps the module well-typed.

        The declaration is emitted ONLY for parameters that appear as an operand
        of a bitwise/shift operator, because those are the uses that require
        integer typing. Value parameters used in other roles - most importantly
        as array extents such as ``shared[tg_mem_size]`` - are left untouched so
        array sizing is not disturbed by an injected runtime local."""
        template_parameters = getattr(func, "template_parameters", None) or []
        bound_value_names = set(bound_value_names or ())
        value_names = []
        seen = set()
        for entry in template_parameters:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            kind, name = entry[0], entry[1]
            if kind != "value" or not name or name in seen or name in bound_value_names:
                continue
            seen.add(name)
            value_names.append(name)
        if not value_names:
            return ""
        if body is None:
            body = getattr(func, "body", []) or []
        bitwise_names = set()
        self.collect_bitwise_operand_identifiers(body, bitwise_names)
        names = [name for name in value_names if name in bitwise_names]
        if not names:
            return ""
        pad = "    " * indent
        return "".join(
            f"{pad}int {self.sanitize_identifier(name)} = 0;\n" for name in names
        )

    def collect_bitwise_operand_identifiers(self, node, found):
        """Record identifier names that appear anywhere inside the operands of a
        bitwise or shift operator within ``node``."""
        bitwise_ops = {"&", "|", "^", "<<", ">>"}
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, BinaryOpNode) and getattr(node, "op", None) in bitwise_ops:
            self.collect_identifier_references(node, found)
        for child in self.iter_ast_children(node):
            self.collect_bitwise_operand_identifiers(child, found)

    def collect_identifier_references(self, node, found):
        """Collect bare identifier names (undeclared VariableNode uses) in a
        subtree - i.e. references rather than typed declarations."""
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if (
            isinstance(node, VariableNode)
            and getattr(node, "name", None)
            and not getattr(node, "vtype", None)
        ):
            found.add(node.name)
        for child in self.iter_ast_children(node):
            self.collect_identifier_references(child, found)

    def map_function_return_type(self, return_type):
        pointer_buffer_type = self.map_pointer_return_buffer_type(return_type)
        if pointer_buffer_type:
            return pointer_buffer_type
        mapped_type = self.map_type(return_type)
        if str(mapped_type).rstrip().endswith("&"):
            return str(mapped_type).rstrip()[:-1].rstrip()
        return mapped_type

    def map_pointer_return_buffer_type(self, return_type):
        element_type = self.pointer_element_type(return_type)
        if not element_type:
            return None

        element_type = re.sub(
            r"^(?:(?:const|device|thread|threadgroup|constant|volatile|restrict)\s+)+",
            "",
            str(element_type).strip(),
        )
        element_type = self.resolve_type_alias(element_type)
        return f"RWStructuredBuffer<{self.map_type(element_type)}>"

    def function_output_name(self, func):
        host_name = self.function_host_name(func)
        if host_name and getattr(func, "qualifier", None) in {
            "vertex",
            "fragment",
            "kernel",
        }:
            return host_name
        return func.name

    def should_emit_kernel_stage_entry(self, func):
        if getattr(func, "name", None) == "main":
            return False
        return self.kernel_stage_entry_builtin_types_supported(func)

    def kernel_stage_entry_builtin_types_supported(self, func):
        # Keep imported kernels on the explicit stage-entry path only when the
        # current Metal generator can validate their builtin parameter shapes.
        # Metal lets the positional / dimension compute builtins be declared as a
        # scalar `uint`, a `uint2`, or a `uint3`; the driver fills the requested
        # component count. MLX's reduction kernels (softmax, rms_norm, layer_norm,
        # logsumexp, ...) use the scalar `uint` form for 1-D dispatches, so
        # restricting these to `uint3` wrongly denied them the explicit
        # stage-entry path and collapsed every materialized kernel to an unnamed
        # "main" entry point (invalid SPIR-V: duplicate entry-point names).
        positional_builtin_types = {"uint", "uint2", "uint3"}
        expected_types = {
            "thread_position_in_grid": positional_builtin_types,
            "thread_position_in_threadgroup": positional_builtin_types,
            "threadgroup_position_in_grid": positional_builtin_types,
            "thread_index_in_threadgroup": {"uint"},
            "threads_per_threadgroup": positional_builtin_types,
            "threadgroups_per_grid": positional_builtin_types,
            "thread_index_in_simdgroup": {"uint"},
            "threads_per_simdgroup": {"uint"},
        }
        for param in getattr(func, "params", []) or []:
            semantic = self.compute_builtin_attribute_name(param)
            if semantic is None:
                continue
            allowed_types = expected_types.get(semantic)
            if allowed_types is None:
                continue
            param_type = self.normalized_metal_type(getattr(param, "vtype", None))
            if param_type not in allowed_types:
                return False
        return True

    def compute_builtin_attribute_name(self, var):
        compute_builtins = {
            "thread_position_in_grid",
            "thread_position_in_threadgroup",
            "threadgroup_position_in_grid",
            "thread_index_in_threadgroup",
            "threads_per_threadgroup",
            "threadgroups_per_grid",
            "thread_index_in_simdgroup",
            "threads_per_simdgroup",
        }
        for attr in getattr(var, "attributes", []) or []:
            name = str(getattr(attr, "name", "")).lower()
            if name in compute_builtins:
                return name
        return None

    def function_host_name(self, func):
        for attr in getattr(func, "attributes", []) or []:
            if getattr(attr, "name", None) != "host_name":
                continue
            args = getattr(attr, "args", []) or []
            if not args:
                return None
            raw_name = str(args[0]).strip()
            if (
                len(raw_name) >= 2
                and raw_name[0] in {'"', "'"}
                and raw_name[-1] == raw_name[0]
            ):
                return raw_name[1:-1]
            return raw_name
        return None

    def function_semantic_attributes(self, func):
        return [
            attr
            for attr in getattr(func, "attributes", []) or []
            if getattr(attr, "name", None)
            not in (
                self.fragment_execution_attribute_names
                | self.function_metadata_attribute_names
            )
            and "::" not in str(getattr(attr, "name", ""))
        ]

    def generate_fragment_execution_layouts(self, func):
        layouts = []
        for attr in getattr(func, "attributes", []) or []:
            name = getattr(attr, "name", None)
            if name not in self.fragment_execution_attribute_names:
                continue
            if getattr(attr, "args", None):
                continue
            if name not in layouts:
                layouts.append(name)
        if not layouts:
            return ""
        return f"        layout({', '.join(layouts)}) in;\n"

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, TypeAliasNode):
                # Register a block-local ``using`` or ``typedef`` so later
                # declarations and expressions resolve it in lexical order. The
                # alias itself produces no CrossGL statement.
                self.register_local_type_alias(stmt)
                code = code[: len(code) - 4 * indent]
                continue
            if isinstance(stmt, VariableNode):
                self.current_variable_types[stmt.name] = (
                    self.metal_declaration_expression_type(stmt)
                )
                if id(stmt) in self.storage_texture_declaration_ids:
                    self.current_storage_texture_names.add(stmt.name)
                if self.structured_buffer_pointer_type(stmt):
                    self.current_structured_buffer_names.add(stmt.name)
                decl = self.format_decl(stmt, include_semantic=False)
                code += f"{decl};\n"
            elif isinstance(stmt, AssignmentNode):
                declaration = getattr(stmt, "left", None)
                if isinstance(declaration, VariableNode) and getattr(
                    declaration, "vtype", None
                ):
                    self.current_variable_types[declaration.name] = (
                        self.metal_declaration_expression_type(declaration)
                    )
                    if id(declaration) in self.storage_texture_declaration_ids:
                        self.current_storage_texture_names.add(declaration.name)
                    if self.structured_buffer_pointer_type(declaration):
                        self.current_structured_buffer_names.add(declaration.name)
                code += self.generate_assignment(stmt, is_main) + ";\n"
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    if stmt.value is None:
                        code += "return;\n"
                    else:
                        pushed_context = (
                            self.push_materialized_constexpr_expression_context(
                                stmt.value,
                                required=False,
                            )
                        )
                        try:
                            if (
                                self.wide_vector_type_info(
                                    self.current_function_return_type,
                                    getattr(stmt, "source_location", None),
                                )
                                is not None
                            ):
                                value = self.generate_initializer_value(
                                    stmt.value,
                                    is_main,
                                    self.current_function_return_type,
                                )
                            else:
                                value = self.generate_expression(stmt.value, is_main)
                        finally:
                            if pushed_context:
                                self.materialized_constexpr_expression_contexts.pop()
                        code += f"return {value};\n"
            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt.left, is_main)} {stmt.op} {self.generate_expression(stmt.right, is_main)};\n"
            elif (
                isinstance(stmt, CastNode) and self.map_type(stmt.target_type) == "void"
            ):
                code += f"{self.generate_expression(stmt.expression, is_main)};\n"
            elif isinstance(stmt, BlockNode):
                code += "{\n"
                code += self.generate_scoped_function_body(
                    stmt.statements, indent + 1, is_main
                )
                code += "    " * indent + "}\n"
            elif isinstance(stmt, RangeForNode):
                code += self.generate_range_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent, is_main)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent, is_main)
            elif isinstance(stmt, FunctionCallNode):
                lowered_callback = self.generate_callback_statement(
                    stmt, indent, is_main
                )
                if lowered_callback is not None:
                    code += lowered_callback
                else:
                    code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, (MethodCallNode, CallNode)):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, PostfixOpNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, ContinueNode):
                code += "continue;\n"
            elif isinstance(stmt, BreakNode):
                code += "break;\n"
            elif isinstance(stmt, DiscardNode):
                code += "discard;\n"
            elif isinstance(stmt, StaticAssertNode):
                cond = self.generate_expression(stmt.condition, is_main)
                if stmt.message is not None:
                    msg = (
                        stmt.message
                        if isinstance(stmt.message, str)
                        else self.generate_expression(stmt.message, is_main)
                    )
                    code += f"static_assert({cond}, {msg});\n"
                else:
                    code += f"static_assert({cond});\n"
            elif isinstance(stmt, EnumNode):
                code += self.generate_local_enum(stmt, indent, is_main)
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                expr = self.generate_expression(stmt, is_main)
                if expr:
                    code += f"{expr};\n"
                else:
                    code += f"// Unhandled statement type: {type(stmt).__name__}\n"
        return code

    def generate_scoped_function_body(self, body, indent=0, is_main=False):
        """Render a nested lexical block without leaking local type aliases."""
        previous_type_aliases = dict(self.type_aliases)
        previous_type_alias_qualifiers = dict(self.type_alias_qualifiers)
        previous_local_type_alias_names = set(self.local_type_alias_names)
        previous_local_struct_type_aliases = dict(self.local_struct_type_aliases)
        self.template_binding_shadow_scopes.append(set())
        try:
            return self.generate_function_body(body, indent, is_main)
        finally:
            self.template_binding_shadow_scopes.pop()
            self.type_aliases = previous_type_aliases
            self.type_alias_qualifiers = previous_type_alias_qualifiers
            self.local_type_alias_names = previous_local_type_alias_names
            self.local_struct_type_aliases = previous_local_struct_type_aliases

    def generate_callback_statement(self, call, indent, is_main=False):
        helper = str(call.name).rsplit("::", 1)[-1]
        if helper != "dispatch_bool":
            return None

        source_location = getattr(call, "source_location", None)
        if len(call.args) != 2:
            callback = next(
                (arg for arg in call.args if isinstance(arg, LambdaNode)), None
            )
            if callback is None:
                return None
            raise self.callback_lowering_error(
                helper,
                "expected a condition and one lambda callback",
                callback=callback,
                source_location=source_location,
            )
        if not isinstance(call.args[1], LambdaNode):
            return None

        callback = call.args[1]
        if callback.capture != "&":
            raise self.callback_lowering_error(
                helper,
                "callback must use reference-default capture",
                callback=callback,
                source_location=source_location,
            )
        if len(callback.params) != 1 or not callback.params[0].name:
            raise self.callback_lowering_error(
                helper,
                "callback must declare exactly one integral-constant parameter",
                callback=callback,
                source_location=source_location,
            )
        if callback.params[0].vtype != "auto":
            raise self.callback_lowering_error(
                helper,
                "callback parameter must use the deduced integral-constant type",
                callback=callback,
                source_location=source_location,
            )
        if getattr(callback, "return_type", None) not in (None, "void"):
            raise self.callback_lowering_error(
                helper,
                "callback return values cannot be represented by dispatch_bool",
                callback=callback,
                source_location=source_location,
            )
        if self.callback_contains_return(callback):
            raise self.callback_lowering_error(
                helper,
                "callback-local return statements cannot be inlined into the caller",
                callback=callback,
                source_location=source_location,
            )

        condition = self.generate_expression(call.args[0], is_main)
        parameter_name = callback.params[0].name
        true_body = self.generate_integral_constant_callback_body(
            callback.body, parameter_name, True, indent + 1, is_main
        )
        false_body = self.generate_integral_constant_callback_body(
            callback.body, parameter_name, False, indent + 1, is_main
        )
        branch_indent = "    " * indent
        return (
            f"if ({condition}) {{\n"
            f"{true_body}"
            f"{branch_indent}}} else {{\n"
            f"{false_body}"
            f"{branch_indent}}}\n"
        )

    def generate_integral_constant_callback_body(
        self, body, parameter_name, value, indent, is_main
    ):
        previous_variable_types = self.current_variable_types
        previous_type_aliases = dict(self.type_aliases)
        previous_type_alias_qualifiers = dict(self.type_alias_qualifiers)
        previous_local_type_alias_names = set(self.local_type_alias_names)
        previous_local_struct_type_aliases = dict(self.local_struct_type_aliases)
        previous_storage_texture_names = self.current_storage_texture_names
        previous_structured_buffer_names = self.current_structured_buffer_names
        self.current_variable_types = dict(previous_variable_types)
        self.current_storage_texture_names = set(previous_storage_texture_names)
        self.current_structured_buffer_names = set(previous_structured_buffer_names)
        self.integral_constant_bindings.append({parameter_name: value})
        self.push_identifier_scope()
        try:
            return self.generate_function_body(body, indent, is_main)
        finally:
            self.pop_identifier_scope()
            self.integral_constant_bindings.pop()
            self.current_variable_types = previous_variable_types
            self.type_aliases = previous_type_aliases
            self.type_alias_qualifiers = previous_type_alias_qualifiers
            self.local_type_alias_names = previous_local_type_alias_names
            self.local_struct_type_aliases = previous_local_struct_type_aliases
            self.current_storage_texture_names = previous_storage_texture_names
            self.current_structured_buffer_names = previous_structured_buffer_names

    def callback_contains_return(self, callback):
        def visit(node):
            if isinstance(node, ReturnNode):
                return True
            if isinstance(node, LambdaNode) and node is not callback:
                return False
            return any(visit(child) for child in self.iter_ast_children(node))

        return any(visit(statement) for statement in callback.body)

    def integral_constant_binding(self, name):
        for bindings in reversed(self.integral_constant_bindings):
            if name in bindings:
                return bindings[name]
        return None

    def render_integral_constant_binding(self, name):
        value = self.integral_constant_binding(name)
        if value is None:
            return None
        return "true" if value else "false"

    def register_local_type_alias(self, alias):
        """Register a function-body alias for subsequent lexical uses."""
        name = getattr(alias, "name", None)
        alias_type = getattr(alias, "alias_type", None)
        if not name or not alias_type:
            return
        alias_qualifiers = list(getattr(alias, "qualifiers", None) or [])
        # Struct aliases remain uninlined, but scoped static-member references
        # need their concrete owner to resolve constants and backing globals.
        self.local_struct_type_aliases[name] = alias_type
        if (
            self.wide_vector_type_info(
                alias_type, getattr(alias, "source_location", None)
            )
            is not None
        ):
            self.type_aliases[name] = alias_type
            self.type_alias_qualifiers[name] = alias_qualifiers
            self.local_type_alias_names.add(name)
            return
        if alias_qualifiers and (
            self.pointer_element_type(alias_type) is not None
            or self.reference_element_type(alias_type) is not None
            or self.is_metal_resource_type(alias_type)
        ):
            self.type_aliases[name] = alias_type
            self.type_alias_qualifiers[name] = alias_qualifiers
            self.local_type_alias_names.add(name)
            return
        # Inline body-local aliases that resolve to scalar/vector primitives or
        # to a concrete struct emitted in this module. Both forms would otherwise
        # become dangling CrossGL type names and default to float downstream.
        # Unresolved user-template aliases remain untouched until their template
        # arguments have been materialized by the Metal frontend.
        mapped_alias_type = self.map_type(alias_type)
        concrete_struct_alias = (
            alias_type in self.struct_name_map
            and mapped_alias_type in self.struct_name_map.values()
        )
        if (
            getattr(alias, "qualifiers", None)
            or getattr(alias, "array_sizes", None)
            or getattr(alias, "declarator_type_suffix", "")
            or (
                mapped_alias_type not in self.crossgl_typedef_source_types()
                and not concrete_struct_alias
            )
        ):
            return
        self.type_aliases[name] = alias_type
        self.type_alias_qualifiers[name] = alias_qualifiers
        self.local_type_alias_names.add(name)

    def resolve_local_type_aliases(self, metal_type):
        """Resolve concrete body-local aliases inside a Metal type expression."""
        if metal_type is None:
            return metal_type

        original = str(metal_type).strip()
        base = original
        suffix = ""
        while base.endswith("*") or base.endswith("&"):
            suffix = base[-1] + suffix
            base = base[:-1].strip()

        if base in self.local_type_alias_names:
            return f"{self.resolve_type_alias(base)}{suffix}"

        generic_base, generic_args = self.generic_type_parts(base)
        if not generic_base or not generic_args:
            return original
        resolved_args = [self.resolve_local_type_aliases(arg) for arg in generic_args]
        if resolved_args == generic_args:
            return original
        return f"{generic_base}<{', '.join(resolved_args)}>{suffix}"

    def generate_local_enum(self, enum, indent, is_main):
        enum_name = enum.name or "MetalAnonymousEnum"
        code = f"enum {enum_name} {{\n"
        for member_name, member_value in enum.members:
            code += "    " * (indent + 1) + member_name
            if member_value is not None:
                code += f" = {self.generate_expression(member_value, is_main)}"
            code += ",\n"
        code += "    " * indent + "};\n"
        return code

    def generate_for_loop(self, node, indent, is_main):
        previous_variable_types = self.current_variable_types
        self.current_variable_types = dict(previous_variable_types)
        self.template_binding_shadow_scopes.append(set())
        try:
            for declaration in self.for_initializer_declarations(node.init):
                self.current_variable_types[declaration.name] = declaration.vtype

            init = self.generate_for_clause(node.init, is_main)
            condition = self.generate_for_clause(node.condition, is_main)
            update = self.generate_for_clause(node.update, is_main)

            code = f"for ({init}; {condition}; {update}) {{\n"
            code += self.generate_scoped_function_body(node.body, indent + 1, is_main)
            code += "    " * indent + "}\n"
            return code
        finally:
            self.template_binding_shadow_scopes.pop()
            self.current_variable_types = previous_variable_types

    def for_initializer_declarations(self, initializer):
        if isinstance(initializer, (list, tuple)):
            return [
                declaration
                for item in initializer
                for declaration in self.for_initializer_declarations(item)
            ]
        declaration = (
            initializer.left if isinstance(initializer, AssignmentNode) else initializer
        )
        if isinstance(declaration, VariableNode) and declaration.vtype:
            return [declaration]
        return []

    def generate_for_clause(self, expr, is_main):
        if isinstance(expr, list):
            return ", ".join(self.generate_for_clause(item, is_main) for item in expr)
        return self.generate_expression(expr, is_main)

    def generate_range_for_loop(self, node, indent, is_main):
        iterable = self.generate_expression(node.iterable, is_main)
        self.template_binding_shadow_scopes.append(set())
        try:
            if any(node.name in bindings for bindings in self.template_value_bindings):
                self.template_binding_shadow_scopes[-1].add(node.name)
            code = f"for {node.name} in {iterable} {{\n"
            code += self.generate_scoped_function_body(node.body, indent + 1, is_main)
            code += "    " * indent + "}\n"
            return code
        finally:
            self.template_binding_shadow_scopes.pop()

    def generate_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)
        code = f"while ({condition}) {{\n"
        code += self.generate_scoped_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)
        code = "do {\n"
        code += self.generate_scoped_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + f"}} while ({condition});\n"
        return code

    def generate_if_statement(self, node, indent, is_main):
        code = ""
        if node.if_chain:
            for condition, body in node.if_chain:
                code += f"if ({self.generate_expression(condition, is_main)}) {{\n"
                code += self.generate_scoped_function_body(body, indent + 1, is_main)
                code += "    " * indent + "}"
        if node.else_if_chain:
            for condition, body in node.else_if_chain:
                code += (
                    f" else if ({self.generate_expression(condition, is_main)}) {{\n"
                )
                code += self.generate_scoped_function_body(body, indent + 1, is_main)
                code += "    " * indent + "}"

        if node.else_body:
            code += " else {\n"
            code += self.generate_scoped_function_body(
                node.else_body, indent + 1, is_main
            )
            code += "    " * indent + "}"

        code += "\n"
        return code

    def generate_assignment(self, node, is_main):
        if self.is_structured_buffer_element_access(node.left):
            structured_store = self.generate_structured_buffer_store(
                node.left, node.right, node.operator, is_main
            )
            if structured_store is not None:
                return structured_store
        lhs_info = self.wide_vector_expression_info(node.left)
        lhs = self.generate_expression(node.left, is_main)
        pushed_context = bool(
            self.is_supported_constexpr_local_declaration(node)
            and self.push_materialized_constexpr_expression_context(
                node.right,
                required=True,
            )
        )
        try:
            rhs = self.generate_initializer_value(
                node.right,
                is_main,
                (
                    getattr(node.left, "vtype", None)
                    if isinstance(node.left, VariableNode)
                    else None
                ),
                (
                    self.variable_has_array_initializer_shape(node.left)
                    if isinstance(node.left, VariableNode)
                    else False
                ),
            )
        finally:
            if pushed_context:
                self.materialized_constexpr_expression_contexts.pop()
        op = node.operator
        if lhs_info is not None and op != "=":
            binary_operator = op[:-1] if op.endswith("=") else None
            if self.wide_vector_binary_operation_name(binary_operator) is None:
                raise MetalWideVectorLoweringError(
                    lhs_info["source_type"],
                    "the compound operator has no semantics-preserving "
                    "aggregate lowering",
                    getattr(node, "source_location", None),
                    operation=op,
                )
            right_info = self.wide_vector_expression_info(node.right)
            if right_info is not None and right_info["key"] != lhs_info["key"]:
                raise MetalWideVectorLoweringError(
                    lhs_info["source_type"],
                    "compound-assignment operands have different element types "
                    "or widths",
                    getattr(node, "source_location", None),
                    operation=op,
                )
            if right_info is None and not (
                self.wide_vector_constructor_argument_is_scalar(node.right)
            ):
                raise MetalWideVectorLoweringError(
                    lhs_info["source_type"],
                    "the compound-assignment right operand is not scalar",
                    getattr(node, "source_location", None),
                    operation=op,
                )
            right_kind = "vector" if right_info is not None else "scalar"
            self.wide_vector_compound_helpers.add(
                (lhs_info["key"], binary_operator, right_kind)
            )
            helper_name = self.wide_vector_compound_helper_name(
                lhs_info, binary_operator, right_kind
            )
            return f"{helper_name}({lhs}, {rhs})"
        return f"{lhs} {op} {rhs}"

    def generate_initializer_value(
        self, expr, is_main=False, expected_type=None, expected_array=False
    ):
        if isinstance(expr, InitializerListNode):
            return self.generate_initializer_list(
                expr, is_main, expected_type, expected_array
            )
        return self.generate_expression(expr, is_main)

    def generate_initializer_list(
        self, node, is_main=False, expected_type=None, expected_array=False
    ):
        wide_vector = self.wide_vector_type_info(
            expected_type, getattr(node, "source_location", None)
        )
        if wide_vector is not None:
            return self.generate_wide_vector_constructor(
                expected_type,
                node.elements,
                is_main,
                getattr(node, "source_location", None),
                braced=True,
            )
        mapped_type = self.map_type(expected_type) if expected_type else None
        member_types = {}
        if expected_type:
            member_types = self.struct_member_types.get(
                self.normalized_metal_type(expected_type), {}
            )

        named_elements = []
        positional_elements = []
        for element in node.elements:
            if isinstance(element, DesignatedInitializerNode):
                named_elements.append(
                    self.generate_designated_initializer(element, is_main, member_types)
                )
            else:
                positional_elements.append(
                    self.generate_initializer_value(element, is_main)
                )

        elements = named_elements + positional_elements
        if expected_array:
            return "{" + ", ".join(elements) + "}"
        if mapped_type and mapped_type.startswith(("vec", "ivec", "uvec", "bvec")):
            return f"{mapped_type}({', '.join(elements)})"
        if mapped_type and named_elements:
            return f"{mapped_type}{{{', '.join(elements)}}}"
        if mapped_type and positional_elements and not named_elements:
            return f"{mapped_type}({', '.join(elements)})"
        return "{" + ", ".join(elements) + "}"

    def generate_designated_initializer(self, node, is_main=False, member_types=None):
        member_types = member_types or {}
        if len(node.designators) == 1 and node.designators[0][0] == "field":
            field_name = node.designators[0][1]
            value = self.generate_initializer_value(
                node.value, is_main, member_types.get(field_name)
            )
            return f"{field_name}: {value}"

        designators = []
        for designator in node.designators:
            kind = designator[0]
            if kind == "range":
                start = self.generate_expression(designator[1], is_main)
                end = self.generate_expression(designator[2], is_main)
                designators.append(f"[{start} ... {end}]")
                continue
            target = designator[1]
            if kind == "index":
                designators.append(f"[{self.generate_expression(target, is_main)}]")
            else:
                designators.append(f".{target}")
        value = self.generate_initializer_value(node.value, is_main)
        return f"{''.join(designators)} = {value}"

    def normalize_literal_string(self, value):
        if "'" in value and re.match(r"^(?:0[xX][0-9a-fA-F]|0[bB][01]|\d|\.\d)", value):
            value = value.replace("'", "")
        if re.fullmatch(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[hH]", value):
            return value[:-1]
        for pattern in (
            self.hex_integer_literal_pattern,
            self.binary_integer_literal_pattern,
            self.decimal_integer_literal_pattern,
        ):
            match = pattern.fullmatch(value)
            if match:
                suffix = match.group("suffix")
                if "u" in suffix.lower():
                    return f"{match.group('body')}u"
                return match.group("body")
        return value

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
                constant = self.render_integral_constant_binding(expr.name)
                if constant is not None:
                    return constant
                return self.render_identifier(expr.name)
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr, is_main)
        elif isinstance(expr, BinaryOpNode):
            cooperative_matrix_operation = {
                "*": "cooperative_matrix_multiply",
                "+": "cooperative_matrix_add",
                "-": "cooperative_matrix_subtract",
            }.get(expr.op)
            if cooperative_matrix_operation is not None and all(
                self.is_metal_cooperative_matrix_expression(operand)
                for operand in (expr.left, expr.right)
            ):
                left = self.generate_expression(expr.left, is_main)
                right = self.generate_expression(expr.right, is_main)
                return f"{cooperative_matrix_operation}({left}, {right})"
            wide_vector_binary = self.generate_wide_vector_binary_expression(
                expr, is_main
            )
            if wide_vector_binary is not None:
                return wide_vector_binary
            left = self.generate_binary_operand(expr.left, expr.op, False, is_main)
            right = self.generate_binary_operand(expr.right, expr.op, True, is_main)
            return f"{left} {expr.op} {right}"
        elif isinstance(expr, FunctionCallNode):
            lowered_method_call = self.generate_lowered_struct_method_call(
                expr, is_main
            )
            if lowered_method_call is not None:
                return lowered_method_call
            cooperative_matrix_call = self.generate_cooperative_matrix_function_call(
                expr, is_main
            )
            if cooperative_matrix_call is not None:
                return cooperative_matrix_call
            constructor_arguments = expr.args
            if (
                getattr(expr, "is_braced_constructor", False)
                and len(expr.args) == 1
                and isinstance(expr.args[0], InitializerListNode)
            ):
                constructor_arguments = expr.args[0].elements
            wide_vector_constructor = self.generate_wide_vector_constructor(
                expr.name,
                constructor_arguments,
                is_main,
                getattr(expr, "source_location", None),
                braced=getattr(expr, "is_braced_constructor", False),
            )
            if wide_vector_constructor is not None:
                return wide_vector_constructor
            if getattr(expr, "is_braced_constructor", False) and expr.args:
                initializer = expr.args[0]
                if isinstance(initializer, InitializerListNode):
                    return self.generate_initializer_list(
                        initializer, is_main, expr.name
                    )
            sizeof_value = self.render_metal_sizeof_expression(expr)
            if sizeof_value is not None:
                return sizeof_value
            numeric_limit = self.metal_numeric_limits_expression(expr.name, expr.args)
            if numeric_limit is not None:
                return numeric_limit
            uniform_value = self.metal_uniform_value_expression(
                expr.name, expr.args, is_main
            )
            if uniform_value is not None:
                return uniform_value
            sync_call = self.metal_synchronization_function_call(
                expr.name,
                expr.args,
                getattr(expr, "source_location", None),
            )
            if sync_call is not None:
                return sync_call
            atomic_call = self.metal_atomic_function_call(expr.name, expr.args, is_main)
            if atomic_call is not None:
                return atomic_call
            callback = next(
                (arg for arg in expr.args if isinstance(arg, LambdaNode)), None
            )
            if callback is not None:
                raise self.unsupported_callback_error(expr.name, callback)
            self.reject_unsupported_wide_vector_call(expr)
            constexpr_value = self.render_constexpr_helper_call(
                expr,
                is_main,
            )
            if constexpr_value is not None:
                return constexpr_value
            materialized_name = self.materialized_value_template_call_name(
                expr.name,
                expr.args,
                getattr(expr, "source_location", None),
            )
            function_name = self.map_function_call_name(materialized_name, expr.args)
            if function_name == "sampler":
                args = ", ".join(
                    self.generate_sampler_constructor_arg(arg, is_main)
                    for arg in expr.args
                )
            else:
                args = ", ".join(
                    self.generate_expression(arg, is_main) for arg in expr.args
                )
            return f"{function_name}({args})"
        elif isinstance(expr, LambdaNode):
            return self.generate_lambda_expression(expr, is_main)
        elif isinstance(expr, CallNode):
            callee = self.generate_expression(expr.callee, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{callee}({args})"
        elif isinstance(expr, MethodCallNode):
            wide_vector = self.wide_vector_expression_info(expr.object)
            if wide_vector is not None:
                raise MetalWideVectorLoweringError(
                    wide_vector["source_type"],
                    "the method has no semantics-preserving aggregate overload",
                    getattr(expr, "source_location", None),
                    operation=f"method {expr.method}",
                )
            obj = self.generate_expression(expr.object, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            method = expr.method
            resource_size_expr = self.resource_size_method_expression(expr, is_main)
            if resource_size_expr is not None:
                return resource_size_expr
            if method == "get_num_mip_levels":
                return f"textureQueryLevels({obj})"
            if method == "get_num_samples":
                samples_function = (
                    "imageSamples"
                    if self.is_storage_image_expression(expr.object)
                    else "textureSamples"
                )
                return f"{samples_function}({obj})"
            descriptor = self.resource_method_descriptor(method)
            if descriptor and descriptor["sampled_texture"]:
                diagnostic = self.unsupported_storage_texture_sampled_method(
                    expr.object, method, is_main
                )
                if diagnostic is not None:
                    return diagnostic
            if method == "sample_compare":
                option_call = self.texture_compare_option_method_call(
                    obj,
                    expr.object,
                    expr.args,
                    is_main,
                )
                if option_call is not None:
                    return option_call
            if descriptor and descriptor["storage_operation"] == "read":
                if self.is_storage_image_expression(expr.object):
                    coord = self.storage_image_coordinate_expression(
                        expr.object, expr.args
                    )
                    return f"imageLoad({obj}, {coord})" if coord else f"{obj}.read()"
                sampled_read = self.sampled_texture_read_expression(
                    expr.object, expr.args, is_main
                )
                if sampled_read is not None:
                    return sampled_read
                return (
                    f"{descriptor['function']}({obj}, {args})"
                    if args
                    else f"{obj}.read()"
                )
            if descriptor and descriptor["storage_operation"] == "write":
                if self.is_storage_image_expression(expr.object):
                    coord = self.storage_image_coordinate_expression(
                        expr.object, expr.args[1:]
                    )
                    value = (
                        self.generate_expression(expr.args[0], is_main)
                        if expr.args
                        else ""
                    )
                    if coord and value:
                        return f"imageStore({obj}, {coord}, {value})"
                    return f"{obj}.write({args})"
                return self.unsupported_sampled_texture_write(
                    expr.object, method, is_main
                )
            if descriptor and method == "gather":
                return self.texture_gather_call(obj, expr.object, expr.args, is_main)
            if descriptor and method == "gather_compare":
                gather_compare_call = self.texture_gather_compare_call(
                    obj, expr.object, expr.args, is_main
                )
                if gather_compare_call is not None:
                    return gather_compare_call
            if descriptor:
                return f"{descriptor['function']}({obj}, {args})"
            return f"{obj}.{method}({args})"
        elif isinstance(expr, MemberAccessNode):
            if expr.member == "value" and isinstance(expr.object, VariableNode):
                constant = self.render_integral_constant_binding(expr.object.name)
                if constant is not None:
                    return constant
            static_member = self.render_decltype_static_struct_member(expr)
            if static_member is not None:
                return static_member
            obj = self.generate_expression(expr.object, is_main)
            wide_vector = self.wide_vector_expression_info(expr.object)
            if wide_vector is not None:
                lane = self.wide_vector_lane_index(
                    str(expr.member), wide_vector["width"]
                )
                if lane is None:
                    raise MetalWideVectorLoweringError(
                        wide_vector["source_type"],
                        f"member selector '{expr.member}' is not a single lane",
                        getattr(expr, "source_location", None),
                        operation="member-access",
                    )
                return f"{obj}.lanes[{lane}]"
            return f"{obj}.{expr.member}"
        elif isinstance(expr, ArrayAccessNode):
            cooperative_matrix_element = self.metal_cooperative_matrix_element_access(
                expr
            )
            if cooperative_matrix_element is not None:
                matrix, element_index = cooperative_matrix_element
                matrix_text = self.generate_expression(matrix, is_main)
                index_text = self.generate_expression(element_index, is_main)
                return f"cooperative_matrix_element({matrix_text}, {index_text})"
            if (
                not self.suppress_structured_buffer_index_lowering
                and self.is_structured_buffer_element_access(expr)
            ):
                return self.generate_structured_buffer_load(expr, is_main)
            wide_vector = self.wide_vector_expression_info(expr.array)
            array = self.generate_expression(expr.array, is_main)
            index = self.generate_expression(expr.index, is_main)
            if wide_vector is not None:
                return f"{array}.lanes[{index}]"
            return f"{array}[{index}]"
        elif isinstance(expr, UnaryOpNode):
            if expr.op == "-" and self.is_metal_cooperative_matrix_expression(
                expr.operand
            ):
                operand = self.generate_expression(expr.operand, is_main)
                return f"cooperative_matrix_negate({operand})"
            wide_vector = self.wide_vector_expression_info(expr.operand)
            if wide_vector is not None and expr.op != "&":
                raise MetalWideVectorLoweringError(
                    wide_vector["source_type"],
                    "the unary operator has no semantics-preserving aggregate lowering",
                    getattr(expr, "source_location", None),
                    operation=expr.op,
                )
            if expr.op == "&" and self.is_structured_buffer_element_access(
                expr.operand
            ):
                operand = self.generate_without_structured_buffer_index_lowering(
                    expr.operand, is_main
                )
            else:
                operand = self.generate_expression(expr.operand, is_main)
            if expr.op == "post...":
                return operand
            if isinstance(expr.operand, (AssignmentNode, BinaryOpNode, TernaryOpNode)):
                operand = f"({operand})"
            return f"({expr.op}{operand})"
        elif isinstance(expr, PostfixOpNode):
            wide_vector = self.wide_vector_expression_info(expr.operand)
            if wide_vector is not None:
                raise MetalWideVectorLoweringError(
                    wide_vector["source_type"],
                    "the postfix operator has no semantics-preserving aggregate lowering",
                    getattr(expr, "source_location", None),
                    operation=expr.op,
                )
            operand = self.generate_expression(expr.operand, is_main)
            return f"{operand}{expr.op}"
        elif isinstance(expr, TernaryOpNode):
            wide_vector = self.wide_vector_expression_info(
                expr.true_expr
            ) or self.wide_vector_expression_info(expr.false_expr)
            if wide_vector is not None:
                raise MetalWideVectorLoweringError(
                    wide_vector["source_type"],
                    "conditional selection has no semantics-preserving aggregate lowering",
                    getattr(expr, "source_location", None),
                    operation="?:",
                )
            return f"{self.generate_expression(expr.condition, is_main)} ? {self.generate_expression(expr.true_expr, is_main)} : {self.generate_expression(expr.false_expr, is_main)}"
        elif isinstance(expr, CastNode):
            wide_vector_cast = self.generate_wide_vector_constructor(
                expr.target_type,
                [expr.expression],
                is_main,
                getattr(expr, "source_location", None),
            )
            if wide_vector_cast is not None:
                return wide_vector_cast
            mapped_type = self.map_type(expr.target_type)
            value = self.generate_expression(expr.expression, is_main)
            if not self.cast_uses_constructor_syntax(mapped_type):
                return (
                    f"({mapped_type})"
                    f"{self.generate_cast_operand(expr.expression, value)}"
                )
            return f"{self.sanitize_identifier(mapped_type)}({value})"
        elif isinstance(expr, VectorConstructorNode):
            size_query = self.texture_size_constructor_expression(expr, is_main)
            if size_query is not None:
                return size_query
            wide_vector_constructor = self.generate_wide_vector_constructor(
                expr.type_name,
                expr.args,
                is_main,
                getattr(expr, "source_location", None),
            )
            if wide_vector_constructor is not None:
                return wide_vector_constructor
            mapped_type = self.map_type(expr.type_name)
            if mapped_type == "sampler":
                args = ", ".join(
                    self.generate_sampler_constructor_arg(arg, is_main)
                    for arg in expr.args
                )
            else:
                args = ", ".join(
                    self.generate_expression(arg, is_main) for arg in expr.args
                )
            return f"{mapped_type}({args})"
        elif isinstance(expr, InitializerListNode):
            return self.generate_initializer_list(expr, is_main)
        elif isinstance(expr, DesignatedInitializerNode):
            return self.generate_designated_initializer(expr, is_main)
        elif isinstance(expr, TextureSampleNode):
            diagnostic = self.unsupported_storage_texture_sampled_method(
                expr.texture, "sample", is_main
            )
            if diagnostic is not None:
                return diagnostic
            texture = self.generate_expression(expr.texture, is_main)
            sampler_expr = expr.sampler
            coords_expr = expr.coordinates
            if coords_expr is None and self.is_samplerless_sample_expression(
                expr.texture
            ):
                coords_expr = sampler_expr
                sampler_expr = None
            sampler_name = self.expression_base_name(sampler_expr)
            sampler = (
                ""
                if sampler_name in self.global_sampler_names
                else self.generate_expression(sampler_expr, is_main)
            )
            sample_args = [texture]
            if sampler:
                sample_args.append(sampler)
            options = getattr(expr, "options", None)
            if options is None:
                options = [expr.lod] if getattr(expr, "lod", None) is not None else []
            coords, options = self.texture_sample_coordinate_and_options(
                expr.texture, coords_expr, options, is_main
            )
            sample_args.append(coords)

            if options:
                option_call = self.texture_sample_options_call(
                    options, sample_args, is_main
                )
                if option_call is not None:
                    return option_call

                lod_expr = self.unwrap_texture_option_argument(expr.lod, "level")
                lod = self.generate_expression(lod_expr, is_main)
                return f"textureLod({', '.join(sample_args + [lod])})"

            return f"texture({', '.join(sample_args)})"
        elif isinstance(expr, float) or isinstance(expr, int) or isinstance(expr, bool):
            return str(expr)
        else:
            return f"/* Unhandled expression: {type(expr).__name__} */"

    def generate_lambda_expression(self, expr, is_main=False):
        del is_main
        raise self.unsupported_callback_error("callback expression", expr)

    def unsupported_callback_error(self, helper, callback):
        return self.callback_lowering_error(
            helper,
            "the callback helper has no semantics-preserving CrossGL lowering",
            callback=callback,
        )

    def callback_lowering_error(
        self, helper, reason, callback=None, source_location=None
    ):
        return MetalCallableLoweringError(
            str(helper),
            reason,
            getattr(callback, "source_location", None) or source_location,
            capture=getattr(callback, "capture", None),
            enclosing_function=self.current_function_name,
            suggested_action=(
                "add a helper-specific lowering that preserves callback invocation "
                "count, compile-time arguments, and capture mutation semantics"
            ),
        )

    def is_metal_atomic_fence_call(self, name):
        function_name = str(name).lstrip(":")
        if function_name == "metal::atomic_thread_fence":
            return True
        return (
            function_name == "atomic_thread_fence"
            and function_name not in self.user_function_names
        )

    @staticmethod
    def metal_atomic_fence_operand_identifier(expr):
        if isinstance(expr, VariableNode):
            name = getattr(expr, "name", None)
        elif isinstance(expr, str):
            name = expr
        else:
            return None
        return str(name).lstrip(":").rsplit("::", 1)[-1]

    def metal_atomic_fence_operand_text(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            return (
                f"{self.metal_atomic_fence_operand_text(expr.left)} | "
                f"{self.metal_atomic_fence_operand_text(expr.right)}"
            )
        name = self.metal_atomic_fence_operand_identifier(expr)
        return name if name is not None else type(expr).__name__

    def collect_metal_atomic_fence_flags(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_metal_atomic_fence_flags(expr.left)
            right = self.collect_metal_atomic_fence_flags(expr.right)
            if left is None or right is None:
                return None
            return left + right
        name = self.metal_atomic_fence_operand_identifier(expr)
        if name in self.metal_atomic_fence_memory_flags:
            return (name,)
        return None

    def metal_atomic_thread_fence_call(self, args, source_location=None):
        if len(args) not in {2, 3}:
            raise MetalAtomicFenceLoweringError(
                "invalid-argument-count",
                memory_flags=(
                    self.metal_atomic_fence_operand_text(args[0]) if args else None
                ),
                memory_order=(
                    self.metal_atomic_fence_operand_text(args[1])
                    if len(args) > 1
                    else None
                ),
                thread_scope=(
                    self.metal_atomic_fence_operand_text(args[2])
                    if len(args) > 2
                    else None
                ),
                source_location=source_location,
            )

        flags = self.collect_metal_atomic_fence_flags(args[0])
        order = self.metal_atomic_fence_operand_identifier(args[1])
        scope = (
            self.metal_atomic_fence_operand_identifier(args[2])
            if len(args) == 3
            else "thread_scope_device"
        )
        contract = {
            "memory_flags": self.metal_atomic_fence_operand_text(args[0]),
            "memory_order": order,
            "thread_scope": scope,
            "source_location": source_location,
        }
        if flags is None:
            raise MetalAtomicFenceLoweringError("unsupported-memory-flags", **contract)
        if order not in self.metal_atomic_fence_memory_orders:
            raise MetalAtomicFenceLoweringError("unsupported-memory-order", **contract)
        if scope not in self.metal_atomic_fence_thread_scopes:
            raise MetalAtomicFenceLoweringError("unsupported-thread-scope", **contract)

        rendered_flags = " | ".join(flags)
        return f"atomicThreadFence({rendered_flags}, {order}, {scope})"

    def metal_synchronization_function_call(self, name, args, source_location=None):
        unscoped_name = str(name).split("::")[-1]

        if self.is_metal_atomic_fence_call(name):
            return self.metal_atomic_thread_fence_call(args, source_location)

        if unscoped_name in {"threadgroup_barrier", "simdgroup_barrier"}:
            flags = self.metal_mem_flag_names(args)
            if flags and len(flags) > 1:
                flags = flags - {"mem_none"}
            if unscoped_name == "threadgroup_barrier" and flags == {"mem_none"}:
                return "workgroupExecutionBarrier()"
            if flags == {"mem_threadgroup"}:
                return "workgroupBarrier()"
            if flags == {"mem_device"}:
                return "memoryBarrierBuffer()"
            if flags == {"mem_texture"}:
                return "memoryBarrierImage()"
            if flags == {"mem_device", "mem_threadgroup", "mem_texture"}:
                return "allMemoryBarrier()"
            return None

        return None

    def metal_uniform_value_expression(self, name, args, is_main):
        """Lower Metal's uniformity assertion while preserving its value."""
        function_name = str(name).lstrip(":")
        explicitly_metal = function_name.startswith("metal::")
        unscoped_name = function_name.split("::")[-1]
        if unscoped_name != "make_uniform":
            return None
        if "::" in function_name and not explicitly_metal:
            return None
        if not explicitly_metal and unscoped_name in self.user_function_names:
            return None
        if len(args) != 1:
            return None
        rendered = self.generate_expression(args[0], is_main)
        if isinstance(args[0], (AssignmentNode, BinaryOpNode, TernaryOpNode)):
            return f"({rendered})"
        return rendered

    def metal_atomic_function_call(self, name, args, is_main):
        unscoped_name = str(name).split("::")[-1]
        mapped = self.metal_atomic_intrinsics.get(unscoped_name)
        if mapped is None or unscoped_name in self.user_function_names:
            return None
        if len(args) < 2:
            return None
        # Drop Metal's trailing memory_order argument; keep the atomic target and
        # the operand value.
        target = self.generate_metal_atomic_target(args[0], is_main)
        value = self.generate_expression(args[1], is_main)
        return f"{mapped}({target}, {value})"

    def generate_metal_atomic_target(self, expr, is_main):
        # Metal addresses an atomic as a pointer (`&buffer[i]` or a pointer var);
        # the CrossGL atomic intrinsics and the GLSL/DirectX/SPIR-V backends
        # address the lvalue. Peel a leading address-of, and emit a structured
        # buffer element as a plain subscript `buffer[index]` (not buffer_load,
        # which the DirectX typed-buffer-atomic lowering does not recognise).
        if isinstance(expr, UnaryOpNode) and getattr(expr, "op", None) == "&":
            expr = expr.operand
        if self.is_structured_buffer_element_access(expr):
            buffer = self.generate_without_structured_buffer_index_lowering(
                expr.array, is_main
            )
            index = self.generate_expression(expr.index, is_main)
            return f"{buffer}[{index}]"
        return self.generate_expression(expr, is_main)

    def metal_mem_flag_names(self, args):
        if len(args) != 1:
            return None
        flags = self.collect_metal_mem_flags(args[0])
        return flags or None

    def collect_metal_mem_flags(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_metal_mem_flags(expr.left)
            right = self.collect_metal_mem_flags(expr.right)
            if left is None or right is None:
                return None
            return left | right
        if isinstance(expr, VariableNode):
            name = str(expr.name).split("::")[-1]
            if name.startswith("mem_"):
                return {name}
        return None

    def lowered_struct_method_context(self, function):
        """Return the concrete receiver contract for a lowered instance helper."""
        if not isinstance(function, FunctionNode):
            return None
        params = list(getattr(function, "params", []) or [])
        if not params:
            return None
        receiver = params[0]
        if getattr(receiver, "name", None) != "self" or not self.reference_parameter(
            receiver
        ):
            return None

        owner = self.normalized_metal_type(
            self.resolve_type_alias(getattr(receiver, "vtype", None))
        )
        if not owner or (
            owner not in self.struct_declarations
            and owner not in self.struct_name_map
            and owner not in self.struct_name_map.values()
        ):
            return None

        helper_name = str(getattr(function, "name", ""))
        prefixes = list(
            dict.fromkeys(
                (owner, self.map_struct_name(owner), self.sanitize_identifier(owner))
            )
        )
        helper_prefix = next(
            (prefix for prefix in prefixes if helper_name.startswith(f"{prefix}__")),
            None,
        )
        if helper_prefix is None:
            return None
        return {
            "owner": owner,
            "helper_prefix": helper_prefix,
            "receiver": receiver,
        }

    @staticmethod
    def lowered_method_transport_parameters(function):
        transported = []
        for param in list(getattr(function, "params", []) or [])[1:]:
            if not str(getattr(param, "name", "")).startswith("crosstl_ptr_"):
                break
            transported.append(param)
        return transported

    def lowered_method_parameter_contract(self, parameter):
        metal_type = self.resolve_type_alias(
            self.metal_declaration_expression_type(parameter)
        )
        qualifiers = tuple(self.effective_declaration_qualifiers(parameter))
        return re.sub(r"\s+", " ", str(metal_type).strip()), qualifiers

    def lowered_method_explicit_parameter_types(self, function, transport_count):
        params = list(getattr(function, "params", []) or [])
        explicit = params[1 + transport_count :]
        mapped = tuple(
            self.map_type(
                self.resolve_type_alias(self.metal_declaration_expression_type(param))
            )
            for param in explicit
        )
        source = tuple(
            self.normalized_metal_parameter_type(param) for param in explicit
        )
        return mapped, source

    def normalized_metal_parameter_type(self, parameter):
        return re.sub(
            r"\s+",
            " ",
            str(
                self.resolve_type_alias(
                    self.metal_declaration_expression_type(parameter)
                )
            ).strip(),
        )

    def lowered_method_candidate_signature(self, function, transport_count):
        _mapped, source = self.lowered_method_explicit_parameter_types(
            function, transport_count
        )
        return f"{function.name}({', '.join(source)})"

    def resolve_lowered_struct_method_call(self, expression):
        """Bind one retained implicit-this call to a concrete lowered helper."""
        context = self.lowered_struct_method_context(self.current_function)
        if context is None:
            return None

        method_name = str(getattr(expression, "name", ""))
        if not self.crossgl_identifier_pattern.fullmatch(method_name):
            return None
        # A value/callable or an ordinary free function with this name wins over
        # the structural lowered-helper inference.
        if (
            method_name in self.current_variable_types
            or method_name in self.global_variable_types
            or method_name in self.user_function_overloads_by_name
        ):
            return None

        helper_name = f"{context['helper_prefix']}__{method_name}"
        all_candidates = list(self.user_function_overloads_by_name.get(helper_name, []))
        if not all_candidates:
            return None

        current_transport = self.lowered_method_transport_parameters(
            self.current_function
        )
        current_transport_names = [param.name for param in current_transport]
        current_transport_contracts = {
            param.name: self.lowered_method_parameter_contract(param)
            for param in current_transport
        }
        current_receiver_readonly = self.readonly_parameter(
            context["receiver"],
            self.effective_declaration_qualifiers(context["receiver"]),
        )
        argument_types = tuple(
            self.expression_mapped_type(argument) for argument in expression.args
        )
        diagnostic_argument_types = tuple(
            argument_type or "<unknown>" for argument_type in argument_types
        )

        viable = []
        rejected_receiver = False
        for candidate in all_candidates:
            candidate_context = self.lowered_struct_method_context(candidate)
            params = list(getattr(candidate, "params", []) or [])
            if (
                candidate_context is None
                or candidate_context["owner"] != context["owner"]
                or len(params) != 1 + len(current_transport) + len(expression.args)
            ):
                continue
            candidate_transport = params[1 : 1 + len(current_transport)]
            candidate_transport_names = [param.name for param in candidate_transport]
            if candidate_transport_names != current_transport_names:
                continue
            if any(
                self.lowered_method_parameter_contract(param)
                != current_transport_contracts.get(param.name)
                for param in candidate_transport
            ):
                continue
            candidate_readonly = self.readonly_parameter(
                candidate_context["receiver"],
                self.effective_declaration_qualifiers(candidate_context["receiver"]),
            )
            if current_receiver_readonly and not candidate_readonly:
                rejected_receiver = True
                continue
            mapped_types, source_types = self.lowered_method_explicit_parameter_types(
                candidate, len(current_transport)
            )
            viable.append((candidate, mapped_types, source_types, candidate_readonly))

        candidate_signatures = tuple(
            self.lowered_method_candidate_signature(candidate, len(current_transport))
            for candidate in all_candidates
        )
        if not viable:
            reason = (
                "a read-only receiver cannot call a mutating sibling helper"
                if rejected_receiver
                else "no helper preserves the receiver and transported-resource contract"
            )
            raise MetalStructMethodCallResolutionError(
                context["owner"],
                method_name,
                diagnostic_argument_types,
                candidate_signatures,
                reason,
                getattr(expression, "source_location", None),
            )

        if any(argument_type is None for argument_type in argument_types):
            matches = viable
        else:
            matches = [
                candidate for candidate in viable if candidate[1] == argument_types
            ]
            if not matches:
                raise MetalStructMethodCallResolutionError(
                    context["owner"],
                    method_name,
                    diagnostic_argument_types,
                    tuple(
                        self.lowered_method_candidate_signature(
                            candidate, len(current_transport)
                        )
                        for candidate, _mapped, _source, _readonly in viable
                    ),
                    "no exact overload matches the available argument types",
                    getattr(expression, "source_location", None),
                )

        source_groups = {}
        for candidate in matches:
            source_groups.setdefault((candidate[2], candidate[3]), []).append(candidate)
        if len(source_groups) != 1:
            raise MetalStructMethodCallResolutionError(
                context["owner"],
                method_name,
                diagnostic_argument_types,
                tuple(
                    self.lowered_method_candidate_signature(
                        candidates[0][0], len(current_transport)
                    )
                    for candidates in source_groups.values()
                ),
                "multiple exact overloads remain after type matching",
                getattr(expression, "source_location", None),
            )

        declarations = next(iter(source_groups.values()))
        selected = next(
            (
                candidate
                for candidate, _mapped, _source, _readonly in declarations
                if getattr(candidate, "body", None)
            ),
            declarations[0][0],
        )
        if self.reference_element_type(getattr(selected, "return_type", None)):
            raise MetalStructMethodCallResolutionError(
                context["owner"],
                method_name,
                diagnostic_argument_types,
                (
                    self.lowered_method_candidate_signature(
                        selected, len(current_transport)
                    ),
                ),
                "reference-returning sibling helpers require lvalue-preserving lowering",
                getattr(expression, "source_location", None),
            )
        return selected, current_transport

    def generate_lowered_struct_method_call(self, expression, is_main=False):
        resolved = self.resolve_lowered_struct_method_call(expression)
        if resolved is None:
            return None
        selected, transported = resolved
        arguments = [self.render_identifier("self")]
        arguments.extend(self.render_identifier(param.name) for param in transported)
        arguments.extend(
            self.generate_expression(argument, is_main) for argument in expression.args
        )
        return f"{self.sanitize_identifier(selected.name)}({', '.join(arguments)})"

    def map_function_call_name(self, name, args=None):
        match = re.fullmatch(r"(?:metal::)?as_type<(.+)>", name)
        if not match:
            metal_type_constructor = self.map_metal_type_constructor_name(name)
            if metal_type_constructor is not None:
                return metal_type_constructor
            metal_bit_name = self.map_metal_bit_function_name(name)
            if metal_bit_name is not None:
                return metal_bit_name
            metal_math_name = self.map_metal_math_function_name(name)
            if metal_math_name is not None:
                return metal_math_name
            metal_wave_name = self.map_metal_wave_function_name(name, args)
            if metal_wave_name is not None:
                return metal_wave_name
            return self.sanitize_identifier(name)

        target_type = self.normalized_metal_type(
            self.resolve_type_alias(match.group(1))
        )
        mapped_type = self.map_type(target_type)
        alias_name = None
        if target_type.startswith("float") or mapped_type in {"float", "double"}:
            alias_name = "asfloat"
        elif target_type.startswith("uint") or mapped_type.startswith("uvec"):
            alias_name = "asuint"
        elif target_type.startswith("int") or mapped_type.startswith("ivec"):
            alias_name = "asint"

        if alias_name is not None:
            source_type = self.expression_mapped_type(args[0]) if args else None
            if source_type is None or self.crossgl_type_shape(
                source_type
            ) == self.crossgl_type_shape(mapped_type):
                return alias_name
        return f"as_type<{mapped_type}>" if alias_name is not None else name

    def metal_numeric_limits_expression(self, name, args=None):
        """Lower a concrete Metal numeric_limits call to an exact CrossGL value."""
        if args:
            return None
        match = re.fullmatch(
            r"(?:metal::)?numeric_limits<(.+)>::"
            r"(max|min|lowest|infinity|quiet_NaN|signaling_NaN|epsilon|denorm_min)",
            str(name),
        )
        if match is None:
            return None

        source_type, operation = match.groups()
        resolved_type = self.normalized_metal_type(
            self.resolve_type_alias(source_type.strip())
        )
        mapped_type = self.map_type(resolved_type)

        float_bits = {
            "max": "0x7f7fffffu",
            "min": "0x00800000u",
            "lowest": "0xff7fffffu",
            "infinity": "0x7f800000u",
            "quiet_NaN": "0x7fc00000u",
            "signaling_NaN": "0x7f800001u",
            "epsilon": "0x34000000u",
            "denorm_min": "0x00000001u",
        }
        if resolved_type in {"float", "float32_t"}:
            return f"asfloat({float_bits[operation]})"

        if resolved_type in {"half", "xhalf", "float16_t"}:
            half_values = {
                "max": "65504.0",
                "min": "0.00006103515625",
                "lowest": "-65504.0",
                "infinity": f"asfloat({float_bits['infinity']})",
                "quiet_NaN": f"asfloat({float_bits['quiet_NaN']})",
                "signaling_NaN": f"asfloat({float_bits['signaling_NaN']})",
                "epsilon": "0.0009765625",
                "denorm_min": "0.000000059604644775390625",
            }
            return f"{mapped_type}({half_values[operation]})"

        if resolved_type in {"bfloat", "bfloat16", "bfloat16_t"}:
            bfloat_bits = {
                "max": "0x7f7f0000u",
                "min": "0x00800000u",
                "lowest": "0xff7f0000u",
                "infinity": "0x7f800000u",
                "quiet_NaN": "0x7fc00000u",
                "signaling_NaN": "0x7f810000u",
                "epsilon": "0x3c000000u",
                "denorm_min": "0x00010000u",
            }
            return f"{mapped_type}(asfloat({bfloat_bits[operation]}))"

        if resolved_type in {"double", "float64_t"}:
            double_values = {
                "max": "1.7976931348623157e308",
                "min": "2.2250738585072014e-308",
                "lowest": "-1.7976931348623157e308",
                "infinity": f"double(asfloat({float_bits['infinity']}))",
                "quiet_NaN": f"double(asfloat({float_bits['quiet_NaN']}))",
                "signaling_NaN": f"double(asfloat({float_bits['signaling_NaN']}))",
                "epsilon": "2.2204460492503131e-16",
                "denorm_min": "4.9406564584124654e-324",
            }
            value = double_values[operation]
            return value if value.startswith("double(") else f"double({value})"

        integer_ranges = {
            "char": ("-128", "127"),
            "int8_t": ("-128", "127"),
            "short": ("-32768", "32767"),
            "int16_t": ("-32768", "32767"),
            "int": ("-2147483648", "2147483647"),
            "int32_t": ("-2147483648", "2147483647"),
            "long": ("-9223372036854775808", "9223372036854775807"),
            "int64_t": ("-9223372036854775808", "9223372036854775807"),
            "uchar": ("0", "255"),
            "uint8_t": ("0", "255"),
            "ushort": ("0", "65535"),
            "uint16_t": ("0", "65535"),
            "uint": ("0", "4294967295"),
            "uint32_t": ("0", "4294967295"),
            "ulong": ("0", "18446744073709551615"),
            "uint64_t": ("0", "18446744073709551615"),
            "size_t": ("0", "18446744073709551615"),
        }
        if resolved_type in integer_ranges and operation in {"min", "lowest", "max"}:
            minimum, maximum = integer_ranges[resolved_type]
            value = maximum if operation == "max" else minimum
            return f"{mapped_type}({value})"
        if resolved_type == "bool" and operation in {"min", "lowest", "max"}:
            return "true" if operation == "max" else "false"
        return None

    def crossgl_type_shape(self, type_name):
        vector_match = re.fullmatch(r"(?:[a-zA-Z0-9_]*vec)([234])", str(type_name))
        return int(vector_match.group(1)) if vector_match else 1

    def map_metal_bit_function_name(self, name):
        text = str(name)
        if text.startswith("metal::"):
            return self.metal_bit_intrinsics.get(text.split("::")[-1])
        if text in self.user_function_names:
            return None
        return self.metal_bit_intrinsics.get(text)

    def map_metal_wave_function_name(self, name, args=None):
        text = str(name)
        if text.startswith("metal::"):
            return self.metal_wave_intrinsics.get(text.split("::")[-1])
        mapped = self.metal_wave_intrinsics.get(text)
        if mapped is None:
            return None
        binding, _function = self.resolve_metal_user_function_overload(
            text,
            args or [],
            allow_wave_lane_conversion=True,
        )
        if binding in {"user", "unknown"}:
            return None
        return mapped

    def resolve_metal_user_function_overload(
        self,
        function_name,
        args,
        *,
        allow_wave_lane_conversion=False,
    ):
        candidates = [
            function
            for function in self.user_function_overloads_by_name.get(function_name, [])
            if len(getattr(function, "params", []) or []) == len(args)
        ]
        if not candidates:
            return "none", None

        argument_types = [self.expression_mapped_type(arg) for arg in args]
        if any(argument_type is None for argument_type in argument_types):
            return "unknown", None

        matching = [
            function
            for function in candidates
            if self.metal_function_mapped_signature(function) == tuple(argument_types)
        ]
        if not matching and allow_wave_lane_conversion:
            scored = [
                (score, function)
                for function in candidates
                if (
                    score := self.metal_wave_user_overload_match_score(
                        function, argument_types
                    )
                )
                is not None
            ]
            if scored:
                best_score = max(score for score, _function in scored)
                matching = [
                    function for score, function in scored if score == best_score
                ]
        if not matching:
            return "none", None

        source_groups = {}
        for function in matching:
            source_groups.setdefault(
                self.metal_function_source_signature(function), []
            ).append(function)
        if len(source_groups) > 1:
            candidate_names = [
                self.metal_function_candidate_signature(function)
                for functions in source_groups.values()
                for function in functions[:1]
            ]
            raise MetalBuiltinOverloadResolutionError(
                function_name,
                argument_types,
                candidate_names,
            )

        declarations = next(iter(source_groups.values()))
        function = next(
            (
                declaration
                for declaration in declarations
                if getattr(declaration, "body", None)
            ),
            declarations[0],
        )
        return "user", function

    def metal_wave_user_overload_match_score(self, function, argument_types):
        parameter_types = self.metal_function_mapped_signature(function)
        if not argument_types or len(parameter_types) != len(argument_types):
            return None
        if parameter_types[0] != argument_types[0]:
            return None

        score = 8
        for actual_type, parameter_type in zip(argument_types[1:], parameter_types[1:]):
            if actual_type == parameter_type:
                score += 8
                continue
            actual_info = self.metal_scalar_arithmetic_type_info(actual_type)
            parameter_info = self.metal_scalar_arithmetic_type_info(parameter_type)
            if (
                actual_info is None
                or parameter_info is None
                or actual_info[0] != "integer"
                or parameter_info[0] != "integer"
            ):
                return None
            score += 2
        return score

    def metal_function_mapped_signature(self, function):
        return tuple(
            self.map_type(self.resolve_type_alias(getattr(param, "vtype", None)))
            for param in getattr(function, "params", []) or []
        )

    def metal_function_source_signature(self, function):
        return tuple(
            self.normalized_metal_type(
                self.resolve_type_alias(getattr(param, "vtype", None))
            )
            for param in getattr(function, "params", []) or []
        )

    def metal_function_candidate_signature(self, function):
        parameters = ", ".join(self.metal_function_source_signature(function))
        return f"{function.name}({parameters})"

    def map_metal_type_constructor_name(self, name):
        text = str(name)
        is_local_alias = text in self.local_type_alias_names
        is_materialized_type = any(
            text in bindings for bindings in self.template_type_bindings
        )
        if (
            "::" not in text
            and text not in self.unscoped_metal_type_constructors
            and not is_local_alias
            and not is_materialized_type
        ):
            return None
        normalized = self.normalized_metal_type(
            self.substitute_template_type_text(
                self.resolve_local_type_aliases(text)
            )
        )
        mapped = self.map_type(normalized)
        if (
            normalized in self.type_map
            or normalized in self.struct_name_map
            or mapped != normalized
        ):
            return mapped
        return None

    def map_metal_math_function_name(self, name):
        for prefix in self.metal_math_namespace_prefixes:
            if not str(name).startswith(prefix):
                continue
            unscoped = str(name)[len(prefix) :]
            if unscoped in self.metal_math_intrinsics:
                return unscoped
        return None

    def generate_binary_operand(self, operand, parent_op, is_right, is_main=False):
        text = self.generate_expression(operand, is_main)
        if not isinstance(operand, BinaryOpNode):
            return text

        parent_precedence = self.binary_precedence.get(parent_op, 0)
        operand_precedence = self.binary_precedence.get(operand.op, 0)
        if operand_precedence < parent_precedence or (
            is_right
            and operand_precedence == parent_precedence
            and (
                parent_op not in {"+", "*", "&&", "||", "&", "|", "^"}
                or operand.op != parent_op
            )
        ):
            return f"({text})"
        return text

    def generate_cast_operand(self, operand, rendered_operand):
        if isinstance(operand, (AssignmentNode, BinaryOpNode, TernaryOpNode)):
            return f"({rendered_operand})"
        if isinstance(operand, str) and self.cast_literal_operand_pattern.fullmatch(
            rendered_operand
        ):
            return f"({rendered_operand})"
        return rendered_operand

    def cast_uses_constructor_syntax(self, mapped_type):
        mapped_text = str(mapped_type).strip()
        if "*" in mapped_text or "&" in mapped_text:
            return False
        if mapped_text in {"float", "double", "int", "uint", "bool"}:
            return True
        if (
            self.crossgl_identifier_pattern.fullmatch(mapped_text)
            and mapped_text[0].isupper()
        ):
            return True
        return bool(
            re.fullmatch(
                r"(?:f16vec[234]|[biu]?vec[234]|dvec[234]|"
                r"mat[234](?:x[234])?|dmat[234](?:x[234])?)",
                mapped_text,
            )
        )

    def map_type(self, metal_type):
        """Map a Metal type name to the closest CrossGL type name."""
        if not metal_type:
            return metal_type

        metal_type = self.substitute_template_type_text(metal_type)
        metal_type = self.substitute_template_value_text(metal_type)

        resolved_local_type = self.resolve_local_type_aliases(metal_type)
        if resolved_local_type != str(metal_type).strip():
            return self.map_type(resolved_local_type)

        alias_base = str(metal_type).strip()
        alias_suffix = ""
        while alias_base.endswith("*") or alias_base.endswith("&"):
            alias_suffix = alias_base[-1] + alias_suffix
            alias_base = alias_base[:-1].strip()
        resolved_alias = self.resolve_type_alias(alias_base)
        if resolved_alias != alias_base:
            wide_vector_alias = self.wide_vector_type_info(resolved_alias)
            if wide_vector_alias is not None:
                return f"{wide_vector_alias['type_name']}{alias_suffix}"

        array_type = self.metal_array_type_parts(metal_type)
        if array_type:
            element_type, size = array_type
            return f"{self.map_type(element_type)}[{self.format_array_extent(size)}]"

        base = metal_type.strip()
        if base.endswith("..."):
            base = base[:-3].strip()
        suffix = ""
        while base.endswith("*") or base.endswith("&"):
            suffix = base[-1] + suffix
            base = base[:-1].strip()
        for tag_prefix in ("struct ", "enum "):
            if base.startswith(tag_prefix):
                base = base[len(tag_prefix) :].strip()
                break

        uniform_payload_type = self.uniform_value_payload_type(base)
        if uniform_payload_type is not None:
            return f"{self.map_type(uniform_payload_type)}{suffix}"

        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        if base.startswith("raytracing::"):
            base = base.split("raytracing::", 1)[1]

        cooperative_matrix = self.map_metal_cooperative_matrix_type(base)
        if cooperative_matrix is not None:
            return f"{cooperative_matrix}{suffix}"

        conditional_type = self.resolve_conditional_type(base)
        if conditional_type is not None:
            return f"{conditional_type}{suffix}"

        atomic_element = self.atomic_element_type(base)
        if atomic_element is not None:
            return f"{self.map_type(atomic_element)}{suffix}"

        function_table_alias = self.function_table_type_alias(base)
        if function_table_alias:
            return f"{function_table_alias}{suffix}"

        pixel_payload_type = self.pixel_data_payload_type(base)
        if pixel_payload_type:
            return f"{self.map_type(pixel_payload_type)}{suffix}"

        vector_type = self.metal_vector_type_parts(base)
        if vector_type:
            element_type, size = vector_type
            return f"{self.map_generic_vector_type(element_type, size)}{suffix}"

        # Normalize Metal resource access qualifiers without dropping dimensions or
        # other non-resource generic arguments, e.g. matrix<bfloat, 4, 4>.
        if "<" in base and base.endswith(">"):
            base_name, inner = base.split("<", 1)
            inner = inner.rstrip(">")
            generic_args = self.split_generic_arguments(inner)
            if self.should_elide_resource_access_qualifier(base_name, generic_args):
                base = f"{base_name}<{generic_args[0].strip()}>"
            elif (
                not self.is_metal_resource_type_name(base_name)
                and "::" not in base_name
            ):
                mapped_base_name = self.sanitize_identifier(base_name.strip())
                mapped_args = ",".join(
                    self.map_generic_type_argument(arg) for arg in generic_args
                )
                return f"{mapped_base_name}<{mapped_args}>{suffix}"

        sampled_resource_type = self.map_sampled_texture_type(base)
        if sampled_resource_type:
            return f"{sampled_resource_type}{suffix}"

        mapped = self.type_map.get(base)
        if mapped is not None:
            return f"{mapped}{suffix}"
        mapped = self.struct_name_map.get(base)
        if mapped is not None:
            return f"{mapped}{suffix}"
        mapped = self.map_scoped_type_name(base)
        return f"{mapped}{suffix}"

    def metal_cooperative_matrix_type_parts(self, metal_type):
        candidate = self.normalized_metal_type(self.resolve_type_alias(metal_type))
        base_name, generic_args = self.generic_type_parts(candidate)
        if not base_name or base_name.rsplit("::", 1)[-1] != "simdgroup_matrix":
            return None
        return generic_args if len(generic_args) == 3 else None

    def map_metal_cooperative_matrix_type(self, metal_type):
        generic_args = self.metal_cooperative_matrix_type_parts(metal_type)
        if generic_args is None:
            return None
        element_type, rows, cols = generic_args
        return (
            f"CooperativeMatrix<{self.map_type(element_type)},"
            f"{self.map_generic_type_argument(rows)},"
            f"{self.map_generic_type_argument(cols)},"
            "subgroup,unspecified,unspecified>"
        )

    def is_metal_cooperative_matrix_expression(self, expression):
        return (
            self.metal_cooperative_matrix_type_parts(
                self.expression_metal_type(expression)
            )
            is not None
        )

    def metal_cooperative_matrix_element_access(self, expression):
        if not isinstance(expression, ArrayAccessNode):
            return None
        thread_elements = expression.array
        if not (
            isinstance(thread_elements, MethodCallNode)
            and thread_elements.method == "thread_elements"
            and not thread_elements.args
            and self.is_metal_cooperative_matrix_expression(thread_elements.object)
        ):
            return None
        return thread_elements.object, expression.index

    def generate_cooperative_matrix_function_call(self, expression, is_main):
        function_name = str(expression.name).rsplit("::", 1)[-1]
        operation = {
            "simdgroup_load": "cooperative_matrix_load",
            "simdgroup_store": "cooperative_matrix_store",
            "simdgroup_multiply_accumulate": "cooperative_matrix_multiply_accumulate",
        }.get(function_name)
        if operation is None:
            return None
        source_arguments = list(expression.args)
        if function_name == "simdgroup_store" and len(source_arguments) >= 2:
            source_arguments = [
                source_arguments[1],
                source_arguments[0],
                *source_arguments[2:],
            ]
        arguments = ", ".join(
            self.generate_expression(argument, is_main) for argument in source_arguments
        )
        return f"{operation}({arguments})"

    def resolve_conditional_type(self, base):
        """Resolve ``conditional_t<C, A, B>`` / ``conditional<C, A, B>::type`` to
        a concrete branch type.

        CrossGL has no dependent-type mechanism, so a generic (uninstantiated)
        emission cannot evaluate the compile-time condition ``C``. The quantized
        kernels alias integer pack types this way (e.g.
        ``conditional_t<bits == 5, uint64_t, uint32_t>``); leaving the alias
        unresolved makes the aliased variable default to ``float`` and turns the
        bitwise/shift packing math into invalid float operations. Resolving to
        the else-branch (``B``) keeps the whole expression tree integer-typed,
        matches the dominant instantiation of these kernels, and avoids selecting
        wider 64-bit branches that would otherwise demand the Int64 capability.
        """
        candidate = str(base).strip()
        type_accessor = False
        if candidate.endswith("::type"):
            candidate = candidate[: -len("::type")].strip()
            type_accessor = True
        if "<" not in candidate or not candidate.endswith(">"):
            return None
        name, inner = candidate.split("<", 1)
        name = name.strip()
        if name not in ("conditional_t", "conditional"):
            return None
        # ``conditional_t`` is the alias form (no ``::type``); the bare
        # ``conditional`` metafunction only yields a type through ``::type``.
        if name == "conditional" and not type_accessor:
            return None
        args = self.split_generic_arguments(inner[:-1])
        if len(args) != 3:
            return None
        else_branch = args[2].strip()
        if not else_branch:
            return None
        return self.map_type(else_branch)

    def map_generic_type_argument(self, argument):
        argument = str(argument).strip()
        if not argument:
            return argument
        if self.crossgl_identifier_pattern.fullmatch(argument):
            return self.sanitize_identifier(argument)
        base_name, generic_args = self.generic_type_parts(argument)
        if (
            base_name
            and generic_args
            and "::" not in base_name
            and argument.endswith(">")
        ):
            mapped_base_name = self.sanitize_identifier(base_name.strip())
            mapped_args = ",".join(
                self.map_generic_type_argument(arg) for arg in generic_args
            )
            return f"{mapped_base_name}<{mapped_args}>"
        return argument

    def format_type_alias_declaration(self, alias):
        if (
            alias.name in self.callable_type_aliases
            or getattr(alias, "is_function_type", False)
            or self.is_resource_type_alias(alias)
        ):
            return None
        if (
            self.wide_vector_type_info(
                alias.alias_type, getattr(alias, "source_location", None)
            )
            is not None
        ):
            return None

        mapped_type = self.crossgl_typedef_source_type(self.map_type_alias(alias))
        alias_name = self.sanitize_identifier(alias.name)
        if not mapped_type or not alias_name or mapped_type == alias_name:
            return None
        return f"typedef {mapped_type} {alias_name};"

    def map_type_alias(self, alias):
        storage_type = self.map_storage_texture_type(alias.alias_type)
        if storage_type:
            return storage_type
        if self.normalized_metal_type(alias.alias_type) == "half":
            return "f16"
        return self.map_type(alias.alias_type)

    def crossgl_typedef_source_type(self, mapped_type):
        mapped_type = str(mapped_type).strip()
        if not mapped_type or "::" in mapped_type:
            return None
        if mapped_type in {
            info["type_name"] for info in self.wide_vector_types.values()
        }:
            return mapped_type
        if any(char in mapped_type for char in "*&[]"):
            return None

        normalized_type = self.normalize_crossgl_typedef_source_type(mapped_type)
        if normalized_type is not None:
            return normalized_type
        if re.fullmatch(r"(?:matrix|vec|vector)<[^{};]+>", mapped_type):
            return mapped_type
        if mapped_type in self.crossgl_typedef_source_types():
            return mapped_type
        return None

    def normalize_crossgl_typedef_source_type(self, mapped_type):
        scalar_aliases = {
            "int8": "i8",
            "uint8": "u8",
            "int16": "i16",
            "uint16": "u16",
            "int64": "i64",
            "uint64": "u64",
            "float16": "f16",
            "float32": "f32",
            "float64": "f64",
        }
        if mapped_type in scalar_aliases:
            return scalar_aliases[mapped_type]

        vector_match = re.fullmatch(r"(f16|i8|u8|i16|u16)vec([2-4])", mapped_type)
        if vector_match:
            element_type, size = vector_match.groups()
            return f"vector<{element_type},{size}>"

        matrix_match = re.fullmatch(r"f16mat([2-4])(?:x([2-4]))?", mapped_type)
        if matrix_match:
            columns, rows = matrix_match.groups()
            rows = rows or columns
            return f"matrix<f16,{columns},{rows}>"
        return None

    def crossgl_typedef_source_types(self):
        return {
            "bool",
            "int",
            "uint",
            "float",
            "double",
            "f16",
            "bfloat",
            "bfloat16",
            "i8",
            "i16",
            "i32",
            "i64",
            "u8",
            "u16",
            "u32",
            "u64",
            "f32",
            "f64",
            "half",
            "char",
            "string",
            "void",
            "vec2",
            "vec3",
            "vec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "bvec2",
            "bvec3",
            "bvec4",
            "mat2",
            "mat3",
            "mat4",
            "mat2x2",
            "mat2x3",
            "mat2x4",
            "mat3x2",
            "mat3x3",
            "mat3x4",
            "mat4x2",
            "mat4x3",
            "mat4x4",
        }

    def map_scoped_type_name(self, type_name):
        if "::" not in str(type_name):
            text = str(type_name)
            if self.crossgl_identifier_pattern.match(text):
                return self.sanitize_identifier(text)
            return text

        base_name, _generic_args = self.generic_type_parts(type_name)
        if base_name and "::" in base_name:
            return type_name

        return self.sanitize_identifier(str(type_name).split("::")[-1])

    def is_resource_type_alias(self, alias):
        return self.is_metal_resource_type(alias.alias_type)

    def resolve_type_alias(self, metal_type):
        if not metal_type:
            return metal_type

        base = str(metal_type).strip()
        suffix = ""
        while base.endswith("*") or base.endswith("&"):
            suffix = base[-1] + suffix
            base = base[:-1].strip()

        seen = set()
        while base in self.type_aliases and base not in seen:
            seen.add(base)
            aliased = str(self.type_aliases[base]).strip()
            alias_suffix = ""
            while aliased.endswith("*") or aliased.endswith("&"):
                alias_suffix = aliased[-1] + alias_suffix
                aliased = aliased[:-1].strip()
            base = aliased
            suffix = alias_suffix + suffix
        return f"{base}{suffix}"

    def atomic_element_type(self, metal_type):
        # Metal atomics lower to their plain element type in CrossGL; atomicity is
        # carried by the atomic_* intrinsics, and the GLSL SSBO / DirectX UAV /
        # SPIR-V backends store atomics as the underlying scalar (an
        # atomic-wrapped element type is not a valid buffer element). Handles both
        # the generic `atomic<T>` form and Metal's `atomic_int`-style aliases.
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name == "atomic" and len(generic_args) == 1:
            return generic_args[0].strip()
        return {
            "atomic_int": "int",
            "atomic_uint": "uint",
            "atomic_bool": "bool",
            "atomic_ulong": "uint64_t",
            "atomic_float": "float",
        }.get(str(metal_type).strip())

    def uniform_value_payload_type(self, metal_type):
        """Return the value type carried by Metal's ``uniform<T>`` wrapper."""
        base_name, generic_args = self.generic_type_parts(metal_type)
        if len(generic_args) != 1:
            return None
        if base_name == "metal::uniform":
            return generic_args[0].strip()
        if base_name != "uniform" or base_name in self.struct_name_map:
            return None
        return generic_args[0].strip()

    def function_table_type_alias(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if (
            base_name in {"visible_function_table", "intersection_function_table"}
            and generic_args
        ):
            return base_name
        return None

    def pixel_data_payload_type(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name in self.pixel_data_type_wrappers and len(generic_args) == 1:
            return generic_args[0].strip()
        return None

    def map_sampled_texture_type(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if not base_name or not generic_args:
            return None

        resource_type = {
            "texture1d": "sampler1D",
            "texture1d_array": "sampler1DArray",
            "texture2d": "sampler2D",
            "texture2d_array": "sampler2DArray",
            "texture2d_ms": "sampler2DMS",
            "texture2d_ms_array": "sampler2DMSArray",
            "texture3d": "sampler3D",
            "texturecube": "samplerCube",
            "texturecube_array": "samplerCubeArray",
            "texture_buffer": "samplerBuffer",
            "depth2d": "sampler2DShadow",
            "depth2d_array": "sampler2DArrayShadow",
            "depthcube": "samplerCubeShadow",
            "depthcube_array": "samplerCubeArrayShadow",
            "depth2d_ms": "sampler2DMS",
            "depth2d_ms_array": "sampler2DMSArray",
        }.get(base_name)
        if resource_type is None:
            return None
        if base_name.startswith("depth"):
            return resource_type

        integer_prefix = self.integer_resource_prefix(generic_args[0])
        if integer_prefix is None:
            if self.is_float_resource_element(generic_args[0]):
                return resource_type
            return None
        return f"{integer_prefix}{resource_type}"

    def integer_resource_prefix(self, metal_type):
        mapped = self.map_type(str(metal_type).strip())
        if mapped in {"uint", "uint8", "uint16", "uint64"}:
            return "u"
        if mapped in {"int", "int8", "int16", "int64"}:
            return "i"
        return None

    def is_float_resource_element(self, metal_type):
        return self.map_type(str(metal_type).strip()) in {
            "float",
            "float16",
            "double",
        }

    def should_elide_resource_access_qualifier(self, base_name, generic_args):
        if len(generic_args) < 2 or not self.is_metal_resource_type_name(base_name):
            return False
        return self.normalized_access_qualifier(generic_args[1]).startswith("access::")

    def is_metal_resource_type_name(self, base_name):
        base = str(base_name).strip()
        while base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        return base in {
            "texture1d",
            "texture1d_array",
            "texture2d",
            "texture2d_array",
            "texture2d_ms",
            "texture2d_ms_array",
            "texture3d",
            "texturecube",
            "texturecube_array",
            "texture_buffer",
            "depth2d",
            "depth2d_array",
            "depth2d_ms",
            "depth2d_ms_array",
            "depthcube",
            "depthcube_array",
        }

    def is_metal_resource_type(self, metal_type):
        resolved_type = self.resolve_type_alias(metal_type)
        base_name, _generic_args = self.generic_type_parts(resolved_type)
        if not base_name:
            base_name = self.normalized_metal_type(resolved_type)
        return self.is_metal_resource_type_name(base_name)

    def generic_type_parts(self, metal_type):
        base = self.normalized_metal_type(metal_type)
        if "<" not in base or not base.endswith(">"):
            return None, []
        base_name, inner = base.split("<", 1)
        return base_name.strip(), self.split_generic_arguments(inner[:-1])

    def concrete_generic_vector_width(self, size):
        width_text = str(size).strip()
        while width_text.startswith("(") and width_text.endswith(")"):
            width_text = width_text[1:-1].strip()
        try:
            width = int(width_text, 0)
        except ValueError:
            return None
        return width if width > 0 else None

    def wide_vector_type_info(self, metal_type, source_location=None):
        if metal_type is None:
            return None

        resolved_type = self.resolve_local_type_aliases(metal_type)
        resolved_type = self.resolve_type_alias(resolved_type)
        candidate = str(resolved_type).strip()
        candidate = re.sub(r"^typename\s+", "", candidate)
        candidate = re.sub(
            r"^(?:(?:const|volatile|thread|threadgroup|device|constant)\s+)+",
            "",
            candidate,
        )
        while candidate.endswith("*") or candidate.endswith("&"):
            candidate = candidate[:-1].strip()

        vector_parts = self.metal_vector_type_parts(candidate)
        if vector_parts is None:
            return None
        element_type, size = vector_parts
        width = self.concrete_generic_vector_width(size)
        if width is None or width <= 4:
            return None

        mapped_element = self.map_type(self.resolve_type_alias(element_type))
        supported_elements = self.wide_vector_supported_scalar_types()
        if mapped_element not in supported_elements:
            raise MetalWideVectorLoweringError(
                candidate,
                f"element type '{element_type}' is not a supported scalar type",
                source_location,
            )

        key = (mapped_element, width)
        info = self.wide_vector_types.get(key)
        if info is not None:
            return info

        element_name = re.sub(r"[^A-Za-z0-9_]", "_", mapped_element).strip("_")
        base_name = f"CrossGLMetalVector_{element_name}_{width}"
        type_name = base_name
        suffix = 1
        while type_name in self.wide_vector_reserved_names or (
            self.wide_vector_generated_name_conflicts(type_name)
        ):
            type_name = f"{base_name}_{suffix}"
            suffix += 1
        self.wide_vector_reserved_names.add(type_name)
        info = {
            "key": key,
            "source_type": candidate,
            "element_type": mapped_element,
            "width": width,
            "type_name": type_name,
        }
        self.wide_vector_types[key] = info
        return info

    def wide_vector_supported_scalar_types(self):
        return {
            "bool",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int",
            "uint",
            "int64",
            "uint64",
            "float16",
            "bfloat16",
            "float",
            "double",
        }

    def wide_vector_generated_name_conflicts(self, type_name):
        generated_names = {
            type_name,
            f"{type_name}_splat",
            f"{type_name}_make",
        }
        operation_names = {
            operation_name
            for operator in ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>")
            for operation_name in [self.wide_vector_binary_operation_name(operator)]
        }
        generated_names.update(
            f"{type_name}_{operation_name}_{left_kind}_{right_kind}"
            for operation_name in operation_names
            for left_kind in ("scalar", "vector")
            for right_kind in ("scalar", "vector")
        )
        generated_names.update(
            f"{type_name}_{operation_name}_assign_{right_kind}"
            for operation_name in operation_names
            for right_kind in ("scalar", "vector")
        )
        return bool(
            generated_names.intersection(
                self.user_function_names | self.wide_vector_reserved_names
            )
        )

    def wide_vector_constructor_argument_is_scalar(self, argument):
        argument_type = self.expression_metal_type(argument)
        if argument_type is None:
            return True
        mapped_type = self.map_type(self.resolve_type_alias(argument_type))
        return mapped_type in self.wide_vector_supported_scalar_types()

    def wide_vector_type_info_from_parts(
        self, element_type, size, source_location=None
    ):
        width = self.concrete_generic_vector_width(size)
        if width is None or width <= 4:
            return None
        return self.wide_vector_type_info(
            f"metal::vec<{element_type},{width}>", source_location
        )

    def wide_vector_default_value(self, info):
        return "false" if info["element_type"] == "bool" else "0"

    def wide_vector_helper_name(self, info, operation):
        return f"{info['type_name']}_{operation}"

    def wide_vector_binary_operation_name(self, operator):
        return {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "%": "mod",
            "&": "bit_and",
            "|": "bit_or",
            "^": "bit_xor",
            "<<": "shift_left",
            ">>": "shift_right",
        }.get(operator)

    def wide_vector_binary_helper_name(self, info, operator, left_kind, right_kind):
        operation_name = self.wide_vector_binary_operation_name(operator)
        return self.wide_vector_helper_name(
            info, f"{operation_name}_{left_kind}_{right_kind}"
        )

    def wide_vector_compound_helper_name(self, info, operator, right_kind):
        operation_name = self.wide_vector_binary_operation_name(operator)
        return self.wide_vector_helper_name(
            info, f"{operation_name}_assign_{right_kind}"
        )

    def generate_wide_vector_support_code(self, indent=0):
        if not self.wide_vector_types:
            return ""

        pad = "    " * indent
        body_pad = "    " * (indent + 1)
        code = f"{pad}// Concrete Metal vectors wider than four lanes\n"
        for info in sorted(
            self.wide_vector_types.values(), key=lambda item: item["type_name"]
        ):
            type_name = info["type_name"]
            element_type = info["element_type"]
            width = info["width"]
            code += f"{pad}struct {type_name} {{\n"
            code += f"{body_pad}{element_type} lanes[{width}];\n"
            code += f"{pad}}};\n\n"

            splat_name = self.wide_vector_helper_name(info, "splat")
            code += f"{pad}{type_name} {splat_name}({element_type} value) {{\n"
            code += f"{body_pad}{type_name} result;\n"
            for lane in range(width):
                code += f"{body_pad}result.lanes[{lane}] = value;\n"
            code += f"{body_pad}return result;\n"
            code += f"{pad}}}\n\n"

            make_name = self.wide_vector_helper_name(info, "make")
            parameters = ", ".join(
                f"{element_type} value{lane}" for lane in range(width)
            )
            code += f"{pad}{type_name} {make_name}({parameters}) {{\n"
            code += f"{body_pad}{type_name} result;\n"
            for lane in range(width):
                code += f"{body_pad}result.lanes[{lane}] = value{lane};\n"
            code += f"{body_pad}return result;\n"
            code += f"{pad}}}\n\n"

        for key, operator, left_kind, right_kind in sorted(
            self.wide_vector_binary_helpers,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        ):
            info = self.wide_vector_types[key]
            type_name = info["type_name"]
            element_type = info["element_type"]
            width = info["width"]
            helper_name = self.wide_vector_binary_helper_name(
                info, operator, left_kind, right_kind
            )
            left_type = type_name if left_kind == "vector" else element_type
            right_type = type_name if right_kind == "vector" else element_type
            code += (
                f"{pad}{type_name} {helper_name}"
                f"({left_type} left, {right_type} right) {{\n"
            )
            code += f"{body_pad}{type_name} result;\n"
            for lane in range(width):
                left = f"left.lanes[{lane}]" if left_kind == "vector" else "left"
                right = f"right.lanes[{lane}]" if right_kind == "vector" else "right"
                code += (
                    f"{body_pad}result.lanes[{lane}] = " f"{left} {operator} {right};\n"
                )
            code += f"{body_pad}return result;\n"
            code += f"{pad}}}\n\n"

        for key, operator, right_kind in sorted(
            self.wide_vector_compound_helpers,
            key=lambda item: (item[0], item[1], item[2]),
        ):
            info = self.wide_vector_types[key]
            type_name = info["type_name"]
            element_type = info["element_type"]
            width = info["width"]
            helper_name = self.wide_vector_compound_helper_name(
                info, operator, right_kind
            )
            right_type = type_name if right_kind == "vector" else element_type
            code += (
                f"{pad}void {helper_name}"
                f"(inout {type_name} value, {right_type} right) {{\n"
            )
            for lane in range(width):
                right = f"right.lanes[{lane}]" if right_kind == "vector" else "right"
                code += (
                    f"{body_pad}value.lanes[{lane}] = value.lanes[{lane}] "
                    f"{operator} {right};\n"
                )
            code += f"{pad}}}\n\n"
        return code

    def generate_wide_vector_constructor(
        self,
        metal_type,
        arguments,
        is_main=False,
        source_location=None,
        braced=False,
    ):
        info = self.wide_vector_type_info(metal_type, source_location)
        if info is None:
            return None

        arguments = list(arguments or [])
        if braced:
            if len(arguments) > info["width"] or not all(
                self.wide_vector_constructor_argument_is_scalar(argument)
                for argument in arguments
            ):
                raise MetalWideVectorLoweringError(
                    info["source_type"],
                    "braced construction requires at most one scalar value per lane",
                    source_location,
                    operation="constructor",
                )
            arguments.extend(
                self.wide_vector_default_value(info)
                for _lane in range(info["width"] - len(arguments))
            )
            rendered = ", ".join(
                self.generate_expression(argument, is_main) for argument in arguments
            )
            return f"{self.wide_vector_helper_name(info, 'make')}({rendered})"

        if not arguments:
            arguments = [self.wide_vector_default_value(info)]

        if len(arguments) == 1:
            argument = arguments[0]
            argument_info = self.wide_vector_type_info(
                self.expression_metal_type(argument),
                getattr(argument, "source_location", None),
            )
            rendered = self.generate_expression(argument, is_main)
            if argument_info is not None:
                if argument_info["key"] != info["key"]:
                    raise MetalWideVectorLoweringError(
                        info["source_type"],
                        "copy construction requires the same element type and width",
                        source_location,
                    )
                return rendered
            if not self.wide_vector_constructor_argument_is_scalar(argument):
                raise MetalWideVectorLoweringError(
                    info["source_type"],
                    "single-argument construction requires a scalar or one "
                    "matching wide vector",
                    source_location,
                    operation="constructor",
                )
            return f"{self.wide_vector_helper_name(info, 'splat')}({rendered})"

        if len(arguments) == info["width"]:
            if not all(
                self.wide_vector_constructor_argument_is_scalar(argument)
                for argument in arguments
            ):
                raise MetalWideVectorLoweringError(
                    info["source_type"],
                    "full construction requires one scalar value per lane",
                    source_location,
                    operation="constructor",
                )
            rendered = ", ".join(
                self.generate_expression(argument, is_main) for argument in arguments
            )
            return f"{self.wide_vector_helper_name(info, 'make')}({rendered})"

        raise MetalWideVectorLoweringError(
            info["source_type"],
            "constructor arguments must be one scalar, one matching vector, "
            f"or exactly {info['width']} scalar lanes",
            source_location,
            operation="constructor",
        )

    def wide_vector_expression_info(self, expression):
        expression_type = self.expression_metal_type(expression)
        if self.metal_pointer_pointee_type_once(expression_type) is not None:
            return None
        return self.wide_vector_type_info(
            expression_type,
            getattr(expression, "source_location", None),
        )

    def metal_pointer_pointee_type_once(self, metal_type):
        if metal_type is None:
            return None
        resolved_type = str(self.resolve_type_alias(metal_type)).strip()
        while resolved_type.endswith("&"):
            resolved_type = resolved_type[:-1].strip()
        if not resolved_type.endswith("*"):
            return None
        return resolved_type[:-1].strip()

    def generate_wide_vector_binary_expression(self, expression, is_main=False):
        left_info = self.wide_vector_expression_info(expression.left)
        right_info = self.wide_vector_expression_info(expression.right)
        if left_info is None and right_info is None:
            return None

        info = left_info or right_info
        if left_info is not None and right_info is not None:
            if left_info["key"] != right_info["key"]:
                raise MetalWideVectorLoweringError(
                    info["source_type"],
                    "binary operands have different element types or widths",
                    getattr(expression, "source_location", None),
                    operation=expression.op,
                )
        if self.wide_vector_binary_operation_name(expression.op) is None:
            raise MetalWideVectorLoweringError(
                info["source_type"],
                "the operator has no semantics-preserving aggregate lowering",
                getattr(expression, "source_location", None),
                operation=expression.op,
            )
        scalar_operand = expression.right if left_info is not None else expression.left
        if (left_info is None or right_info is None) and not (
            self.wide_vector_constructor_argument_is_scalar(scalar_operand)
        ):
            raise MetalWideVectorLoweringError(
                info["source_type"],
                "the non-vector operand is not scalar",
                getattr(expression, "source_location", None),
                operation=expression.op,
            )

        left_kind = "vector" if left_info is not None else "scalar"
        right_kind = "vector" if right_info is not None else "scalar"
        self.wide_vector_binary_helpers.add(
            (info["key"], expression.op, left_kind, right_kind)
        )
        helper_name = self.wide_vector_binary_helper_name(
            info, expression.op, left_kind, right_kind
        )
        left = self.generate_expression(expression.left, is_main)
        right = self.generate_expression(expression.right, is_main)
        return f"{helper_name}({left}, {right})"

    def reject_unsupported_wide_vector_call(self, expression):
        wide_arguments = [
            info
            for argument in expression.args
            for info in [self.wide_vector_expression_info(argument)]
            if info is not None
        ]
        if not wide_arguments:
            return

        if str(expression.name) in self.user_function_names:
            return

        binding, _function = self.resolve_metal_user_function_overload(
            str(expression.name), expression.args
        )
        if binding == "user":
            return
        result_type = self.metal_constructor_result_type(expression.name)
        if result_type in self.struct_member_types:
            return

        raise MetalWideVectorLoweringError(
            wide_arguments[0]["source_type"],
            "the call has no semantics-preserving aggregate overload",
            getattr(expression, "source_location", None),
            operation=f"call {expression.name}",
        )

    def wide_vector_lane_index(self, member, width):
        direct_members = {
            "x": 0,
            "r": 0,
            "y": 1,
            "g": 1,
            "z": 2,
            "b": 2,
            "w": 3,
            "a": 3,
        }
        if member in direct_members:
            index = direct_members[member]
        else:
            selector = re.fullmatch(r"s([0-9a-fA-F])", str(member))
            if selector is None:
                return None
            index = int(selector.group(1), 16)
        return index if index < width else None

    def metal_declaration_expression_type(self, declaration):
        metal_type = getattr(declaration, "vtype", None)
        if metal_type is None:
            return None
        suffix = "".join(
            "[]" if size is None else f"[{self.format_array_extent(size)}]"
            for size in getattr(declaration, "array_sizes", []) or []
        )
        return f"{metal_type}{suffix}"

    def metal_type_contains_wide_vector(self, metal_type, seen_structs=None):
        if metal_type is None:
            return False
        resolved_type = str(self.resolve_type_alias(metal_type)).strip()
        while True:
            array_element = self.split_outer_metal_declarator_array_type(resolved_type)
            if array_element is None:
                break
            resolved_type = array_element
        while resolved_type.endswith("*") or resolved_type.endswith("&"):
            resolved_type = resolved_type[:-1].strip()
        if self.wide_vector_type_info(resolved_type) is not None:
            return True

        struct_name = self.normalized_metal_type(resolved_type)
        if struct_name not in self.struct_member_types:
            return False
        seen_structs = set(seen_structs or set())
        if struct_name in seen_structs:
            return False
        seen_structs.add(struct_name)
        return any(
            self.metal_type_contains_wide_vector(member_type, seen_structs)
            for member_type in self.struct_member_types[struct_name].values()
        )

    def reject_abi_visible_wide_vector_declaration(self, declaration):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(declaration, "qualifiers", []) or []
        }
        if not qualifiers.intersection({"device", "constant"}):
            return
        metal_type = self.metal_declaration_expression_type(declaration)
        if not self.metal_type_contains_wide_vector(metal_type):
            return
        raise MetalWideVectorLoweringError(
            self.resolve_type_alias(metal_type),
            "aggregate lowering does not preserve Metal ABI alignment in "
            "device- or constant-address-space storage",
            getattr(declaration, "source_location", None),
            operation="resource-layout",
        )

    def split_outer_metal_declarator_array_type(self, metal_type):
        if metal_type is None:
            return None
        match = re.fullmatch(r"(.+?)(\[[^\[\]]*\])((?:\[[^\[\]]*\])*)", str(metal_type))
        if match is None:
            return None
        base_type, _outer_extent, remaining_extents = match.groups()
        return f"{base_type}{remaining_extents}"

    def metal_array_type_parts(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if not self.is_metal_array_type_name(base_name) or len(generic_args) < 2:
            return None
        return generic_args[0].strip(), generic_args[1].strip()

    def is_metal_array_type_name(self, base_name):
        return base_name in {"array", "metal::array", "c10::metal::array"}

    def metal_vector_type_parts(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name not in {"vec", "vector"} or len(generic_args) < 2:
            return None
        return generic_args[0].strip(), generic_args[1].strip()

    def map_generic_vector_type(self, element_type, size):
        size = str(size).strip()
        mapped_element = self.map_type(element_type)
        prefixes = {
            "float": "vec",
            "float16": "f16vec",
            "bfloat16": "bfloat16vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "int16": "i16vec",
            "uint16": "u16vec",
            "int8": "i8vec",
            "uint8": "u8vec",
            "bool": "bvec",
        }
        prefix = prefixes.get(mapped_element)
        if prefix and size in {"2", "3", "4"}:
            return f"{prefix}{size}"
        wide_vector = self.wide_vector_type_info_from_parts(element_type, size)
        if wide_vector is not None:
            return wide_vector["type_name"]
        return f"vec<{mapped_element}, {size}>"

    def normalized_metal_type(self, metal_type):
        if not metal_type:
            return ""
        base = str(metal_type).strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        while base.endswith("*") or base.endswith("&"):
            base = base[:-1].strip()
        return base

    def access_qualified_texture_parts(self, metal_type):
        metal_type = self.resolve_type_alias(metal_type)
        array_type = self.metal_array_type_parts(metal_type)
        type_to_check = array_type[0] if array_type else metal_type
        base_name, generic_args = self.generic_type_parts(type_to_check)
        if len(generic_args) < 2:
            return None, generic_args
        access = self.normalized_access_qualifier(generic_args[1])
        if access not in self.storage_texture_accesses:
            return None, generic_args
        generic_args = [*generic_args]
        generic_args[1] = access
        return base_name, generic_args

    def normalized_access_qualifier(self, access):
        access = str(access).replace(" ", "")
        while access.startswith("metal::"):
            access = access.split("metal::", 1)[1]
        return access

    def is_access_qualified_storage_texture_type(self, metal_type):
        base_name, generic_args = self.access_qualified_texture_parts(metal_type)
        return bool(
            generic_args
            and base_name
            in {
                "texture1d",
                "texture1d_array",
                "texture2d",
                "texture2d_array",
                "texture_buffer",
                "texture3d",
            }
        )

    def map_storage_texture_type(self, metal_type):
        base_name, generic_args = self.access_qualified_texture_parts(metal_type)
        if not generic_args:
            return None
        return self.map_access_qualified_texture_type(base_name, generic_args)

    def storage_texture_access_attribute(self, var):
        if not (
            id(var) in self.storage_texture_declaration_ids
            or getattr(var, "name", None) in self.current_storage_texture_names
            or getattr(var, "name", None) in self.global_storage_texture_names
        ):
            return ""

        _, generic_args = self.access_qualified_texture_parts(
            getattr(var, "vtype", None)
        )
        if len(generic_args) < 2:
            return ""

        access = self.normalized_access_qualifier(generic_args[1])
        access_attributes = {
            "access::read": "@readonly",
            "access::write": "@writeonly",
            "access::read_write": "@readwrite",
        }
        return access_attributes.get(access, "")

    def storage_texture_format_attributes(self, var):
        if not (
            id(var) in self.storage_texture_declaration_ids
            or getattr(var, "name", None) in self.current_storage_texture_names
            or getattr(var, "name", None) in self.global_storage_texture_names
        ):
            return ""

        _, generic_args = self.access_qualified_texture_parts(
            getattr(var, "vtype", None)
        )
        if not generic_args:
            return ""

        element_type = self.normalized_metal_type(generic_args[0])
        if element_type == "half":
            return "@rgba16f @metal_texture_element_half"
        return ""

    def split_generic_arguments(self, inner):
        args = []
        current = []
        depth = 0
        for char in inner:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            if char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            args.append("".join(current).strip())
        return args

    def map_access_qualified_texture_type(self, base_name, generic_args):
        if len(generic_args) < 2:
            return None

        access = self.normalized_access_qualifier(generic_args[1])
        if access not in self.storage_texture_accesses:
            return None

        image_type = {
            "texture1d": "image1D",
            "texture1d_array": "image1DArray",
            "texture2d": "image2D",
            "texture2d_array": "image2DArray",
            "texture_buffer": "imageBuffer",
            "texture3d": "image3D",
        }.get(base_name)
        if image_type is None:
            return None

        element_type = generic_args[0].strip()
        integer_prefix = self.integer_resource_prefix(element_type)
        if integer_prefix == "u":
            return f"u{image_type}"
        if integer_prefix == "i":
            return f"i{image_type}"
        return image_type

    def expression_base_name(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            return self.expression_base_name(expr.array)
        if isinstance(expr, MemberAccessNode):
            return self.expression_base_name(expr.object)
        return None

    def metal_scalar_arithmetic_type_info(self, metal_type):
        type_name = self.normalized_metal_type(self.resolve_type_alias(metal_type))
        return self.metal_scalar_arithmetic_types.get(type_name)

    def metal_literal_string_type(self, value):
        text = str(value).replace("'", "")
        if text in {"true", "false"}:
            return "bool"
        if re.fullmatch(r"'(?:[^'\\]|\\.)'", str(value)):
            return "char"

        integer = re.fullmatch(
            r"(?:0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+)(?P<suffix>[uUlL]*)",
            text,
        )
        if integer is not None:
            suffix = integer.group("suffix").lower()
            if "l" in suffix:
                return "uint64_t" if "u" in suffix else "int64_t"
            return "uint" if "u" in suffix else "int"

        floating = re.fullmatch(
            r"(?:"
            r"0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)"
            r"[pP][+-]?[0-9]+|"
            r"(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+[eE][+-]?[0-9]+)"
            r"(?:[eE][+-]?[0-9]+)?"
            r")(?P<suffix>[fFhH]?)",
            text,
        )
        if floating is None:
            return None
        return "half" if floating.group("suffix").lower() == "h" else "float"

    def promoted_metal_integer_type(self, type_info):
        _family, signed, bits = type_info
        if bits < 32:
            return "int", True, 32
        if bits == 32:
            return ("int", True, 32) if signed else ("uint", False, 32)
        return ("int64_t", True, 64) if signed else ("uint64_t", False, 64)

    def metal_common_integer_type(self, left_info, right_info):
        left_name, left_signed, left_bits = self.promoted_metal_integer_type(left_info)
        right_name, right_signed, right_bits = self.promoted_metal_integer_type(
            right_info
        )
        if left_signed == right_signed:
            return left_name if left_bits >= right_bits else right_name
        signed_name, signed_bits = (
            (left_name, left_bits) if left_signed else (right_name, right_bits)
        )
        unsigned_name, unsigned_bits = (
            (right_name, right_bits) if left_signed else (left_name, left_bits)
        )
        if unsigned_bits >= signed_bits:
            return unsigned_name
        if signed_bits > unsigned_bits:
            return signed_name
        return "uint64_t" if signed_bits == 64 else "uint"

    def metal_binary_expression_type(self, expr):
        if expr.op in {"==", "!=", "<", "<=", ">", ">=", "&&", "||"}:
            return "bool"

        left_type = self.expression_metal_type(expr.left)
        right_type = self.expression_metal_type(expr.right)
        if left_type is None or right_type is None:
            return None
        left_wide = self.wide_vector_expression_info(expr.left)
        right_wide = self.wide_vector_expression_info(expr.right)
        if left_wide is not None or right_wide is not None:
            wide_type = left_wide or right_wide
            if (
                left_wide is not None
                and right_wide is not None
                and left_wide["key"] != right_wide["key"]
            ):
                return None
            if self.wide_vector_binary_operation_name(expr.op) is not None:
                return wide_type["source_type"]
            return None
        left_info = self.metal_scalar_arithmetic_type_info(left_type)
        right_info = self.metal_scalar_arithmetic_type_info(right_type)

        if expr.op in {"<<", ">>"}:
            if (
                left_info is None
                or right_info is None
                or left_info[0] != "integer"
                or right_info[0] != "integer"
            ):
                return None
            return self.promoted_metal_integer_type(left_info)[0]

        if left_info is None or right_info is None:
            left_name = self.normalized_metal_type(left_type)
            right_name = self.normalized_metal_type(right_type)
            if left_name == right_name and expr.op in {"+", "-", "*", "/", "%"}:
                return left_name
            return None

        if expr.op in {"&", "|", "^", "%"}:
            if left_info[0] != "integer" or right_info[0] != "integer":
                return None
            return self.metal_common_integer_type(left_info, right_info)
        if expr.op not in {"+", "-", "*", "/"}:
            return None
        if left_info[0] == "floating" or right_info[0] == "floating":
            floating_bits = max(
                info[2] for info in (left_info, right_info) if info[0] == "floating"
            )
            return {16: "half", 32: "float", 64: "double"}[floating_bits]
        return self.metal_common_integer_type(left_info, right_info)

    def expression_metal_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, bool):
            return "bool"
        if isinstance(expr, int):
            return "int"
        if isinstance(expr, float):
            return "float"
        if isinstance(expr, str):
            literal_type = self.metal_literal_string_type(expr)
            if literal_type is not None:
                return literal_type
            return self.current_variable_types.get(
                expr, self.global_variable_types.get(expr)
            )
        if isinstance(expr, VariableNode):
            name = getattr(expr, "name", None)
            if not name:
                return None
            if getattr(expr, "vtype", None):
                return self.metal_declaration_expression_type(expr)
            return self.current_variable_types.get(
                name, self.global_variable_types.get(name)
            )
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_metal_type(expr.array)
            declarator_element_type = self.split_outer_metal_declarator_array_type(
                array_type
            )
            if declarator_element_type is not None:
                return declarator_element_type
            pointer_element_type = self.metal_pointer_pointee_type_once(array_type)
            if pointer_element_type is not None:
                return pointer_element_type
            array_parts = self.metal_array_type_parts(array_type)
            if array_parts:
                return array_parts[0]
            wide_vector = self.wide_vector_type_info(
                array_type, getattr(expr, "source_location", None)
            )
            if wide_vector is not None:
                return wide_vector["element_type"]
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_metal_type(expr.object)
            if object_type is None:
                return None
            wide_vector = self.wide_vector_type_info(
                object_type, getattr(expr, "source_location", None)
            )
            if wide_vector is not None:
                lane = self.wide_vector_lane_index(
                    str(expr.member), wide_vector["width"]
                )
                return wide_vector["element_type"] if lane is not None else None
            object_type = self.normalized_metal_type(object_type)
            member_types = self.struct_member_types.get(object_type)
            if not member_types:
                return None
            return member_types.get(str(expr.member))
        if isinstance(expr, BinaryOpNode):
            return self.metal_binary_expression_type(expr)
        if isinstance(expr, CastNode):
            return self.resolve_type_alias(expr.target_type)
        if isinstance(expr, VectorConstructorNode):
            return self.resolve_type_alias(expr.type_name)
        if isinstance(expr, FunctionCallNode):
            if str(expr.name) in {"decltype", "metal::decltype"}:
                if len(expr.args) != 1:
                    return None
                return self.expression_metal_type(expr.args[0])
            target_match = re.fullmatch(r"(?:metal::)?as_type<(.+)>", expr.name)
            if target_match is not None:
                return self.resolve_type_alias(target_match.group(1).strip())
            constructor_type = self.metal_constructor_result_type(expr.name)
            if constructor_type is not None:
                return constructor_type
            lowered_method = self.resolve_lowered_struct_method_call(expr)
            if lowered_method is not None:
                function, _transported = lowered_method
                return getattr(function, "return_type", None)
            binding, function = self.resolve_metal_user_function_overload(
                str(expr.name), expr.args
            )
            if binding == "user":
                return getattr(function, "return_type", None)
            unscoped_name = str(expr.name).rsplit("::", 1)[-1]
            if (
                unscoped_name in self.metal_wave_intrinsics
                or unscoped_name == "simd_shuffle_and_fill_up"
            ) and expr.args:
                return self.expression_metal_type(expr.args[0])
        return None

    def metal_constructor_result_type(self, name):
        type_name = self.normalized_metal_type(self.resolve_type_alias(str(name)))
        if not type_name:
            return None
        if (
            type_name in self.type_map
            or self.metal_vector_type_parts(type_name) is not None
            or type_name in self.struct_member_types
        ):
            return type_name
        return None

    def expression_mapped_type(self, expr):
        metal_type = self.expression_metal_type(expr)
        storage_type = self.map_storage_texture_type(metal_type)
        if storage_type:
            return storage_type
        return (
            self.map_type(self.resolve_type_alias(metal_type)) if metal_type else None
        )

    def is_samplerless_sample_expression(self, expr):
        metal_type = self.resolve_type_alias(self.expression_metal_type(expr))
        return self.normalized_metal_type(metal_type) in {"SwiftUI::Layer"}

    def has_attribute(self, node, name):
        name = str(name).lower()
        return any(
            str(getattr(attr, "name", "")).lower() == name
            for attr in getattr(node, "attributes", []) or []
        )

    def buffer_attribute_binding(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "buffer":
                continue
            args = getattr(attr, "args", getattr(attr, "arguments", [])) or []
            if not args:
                return None
            text = self.format_metadata_argument(args[0]).strip()
            return int(text) if text.isdigit() else None
        return None

    def reference_element_type(self, metal_type):
        if not metal_type:
            return None
        base = str(metal_type).strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]

        reference_depth = 0
        while base.endswith("*") or base.endswith("&"):
            if base.endswith("&"):
                reference_depth += 1
            base = base[:-1].strip()
        return base if reference_depth else None

    def pointer_element_type(self, metal_type):
        if not metal_type:
            return None
        base = str(metal_type).strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]

        pointer_depth = 0
        while base.endswith("*") or base.endswith("&"):
            if base.endswith("*"):
                pointer_depth += 1
            base = base[:-1].strip()
        return base if pointer_depth else None

    def is_stage_entry_buffer_resource_parameter(self, var):
        if not getattr(var, "name", None):
            return False
        qualifiers = set(self.effective_declaration_qualifiers(var))
        if not qualifiers.intersection({"device", "constant"}):
            return False
        if self.stage_entry_array_resource_element_type(var) is not None:
            return True
        raw_type = self.resolve_type_alias(getattr(var, "vtype", None))
        if self.pointer_element_type(raw_type):
            return True
        return bool(self.reference_element_type(raw_type))

    def stage_entry_array_resource_element_type(self, var):
        array_dimensions = list(getattr(var, "array_sizes", None) or [])
        if not array_dimensions:
            return None
        qualifiers = set(self.effective_declaration_qualifiers(var))
        if not qualifiers.intersection({"device", "constant"}):
            return None
        element_type = str(getattr(var, "vtype", "") or "").strip()
        if not element_type or element_type.endswith(("*", "&")):
            return None
        if len(array_dimensions) != 1:
            raise MetalStageEntryArrayResourceError(
                getattr(var, "name", None),
                array_dimensions,
                "multidimensional-parameter-array",
                getattr(var, "source_location", None),
            )
        return element_type

    def apply_implicit_stage_entry_buffer_bindings(self, func):
        params = list(getattr(func, "params", []) or [])
        used_bindings = {
            binding
            for param in params
            for binding in [self.buffer_attribute_binding(param)]
            if binding is not None
        }
        snapshots = []
        next_binding = 0
        for param in params:
            if not self.is_stage_entry_buffer_resource_parameter(param):
                continue
            if self.buffer_attribute_binding(param) is not None:
                continue
            previous_attributes = list(getattr(param, "attributes", []) or [])
            if previous_attributes:
                continue
            while next_binding in used_bindings:
                next_binding += 1
            param.attributes = previous_attributes + [
                AttributeNode("buffer", [str(next_binding)])
            ]
            snapshots.append((param, previous_attributes))
            used_bindings.add(next_binding)
            next_binding += 1
        return snapshots

    def structured_buffer_pointer_type(self, var):
        if not (
            self.has_attribute(var, "buffer")
            or id(var) in self.current_stage_entry_resource_parameter_ids
        ):
            return None

        qualifiers = set(self.effective_declaration_qualifiers(var))
        if not qualifiers.intersection({"device", "constant"}):
            return None

        element_type = self.pointer_element_type(
            self.resolve_type_alias(getattr(var, "vtype", None))
        )
        array_element_type = self.stage_entry_array_resource_element_type(var)
        if element_type is None:
            element_type = array_element_type
        if not element_type:
            return None
        element_type = self.resolve_type_alias(element_type)
        if (
            array_element_type is None
            and "constant" in qualifiers
            and element_type in self.struct_member_types
        ):
            return None

        buffer_type = (
            "StructuredBuffer"
            if qualifiers.intersection({"constant", "const"})
            else "RWStructuredBuffer"
        )
        mapped_element_type = self.map_resource_pointer_element_type(var, element_type)
        return f"{buffer_type}<{mapped_element_type}>"

    def constant_buffer_pointer_type(self, var):
        if not self.has_attribute(var, "buffer"):
            return None

        qualifiers = set(self.effective_declaration_qualifiers(var))
        if "constant" not in qualifiers:
            return None

        element_type = self.pointer_element_type(
            self.resolve_type_alias(getattr(var, "vtype", None))
        )
        if not element_type:
            return None
        element_type = self.resolve_type_alias(element_type)
        if element_type not in self.struct_member_types:
            return None
        return f"ConstantBuffer<{self.map_type(element_type)}>"

    def is_structured_buffer_expression(self, expr):
        name = self.expression_base_name(expr)
        return bool(
            name
            and (
                name in self.current_structured_buffer_names
                or name in self.global_structured_buffer_names
            )
        )

    def is_structured_buffer_element_access(self, expr):
        return isinstance(
            expr, ArrayAccessNode
        ) and self.is_structured_buffer_expression(expr.array)

    def generate_without_structured_buffer_index_lowering(self, expr, is_main=False):
        previous = self.suppress_structured_buffer_index_lowering
        self.suppress_structured_buffer_index_lowering = True
        try:
            return self.generate_expression(expr, is_main)
        finally:
            self.suppress_structured_buffer_index_lowering = previous

    def generate_structured_buffer_load(self, access, is_main=False):
        buffer = self.generate_without_structured_buffer_index_lowering(
            access.array, is_main
        )
        index = self.generate_expression(access.index, is_main)
        return f"buffer_load({buffer}, {index})"

    def generate_structured_buffer_store(self, access, value, operator, is_main=False):
        buffer = self.generate_without_structured_buffer_index_lowering(
            access.array, is_main
        )
        index = self.generate_expression(access.index, is_main)
        rendered_value = self.generate_expression(value, is_main)
        if operator != "=":
            compound_ops = {
                "+=": "+",
                "-=": "-",
                "*=": "*",
                "/=": "/",
                "%=": "%",
                "&=": "&",
                "|=": "|",
                "^=": "^",
                "<<=": "<<",
                ">>=": ">>",
            }
            binary_op = compound_ops.get(operator)
            if binary_op is None:
                return None
            current_value = f"buffer_load({buffer}, {index})"
            rendered_value = f"{current_value} {binary_op} {rendered_value}"
        return f"buffer_store({buffer}, {index}, {rendered_value})"

    def is_storage_image_expression(self, expr):
        mapped_type = self.expression_mapped_type(expr)
        return bool(
            mapped_type and str(mapped_type).startswith(("image", "iimage", "uimage"))
        )

    def unsupported_storage_texture_sampled_method(self, texture_expr, method, is_main):
        if not self.is_storage_image_expression(texture_expr):
            return None
        texture = self.generate_expression(texture_expr, is_main)
        fallback = (
            "0.0"
            if method in {"sample_compare", "sample_compare_level"}
            else "vec4(0.0)"
        )
        return (
            f"{fallback} /* unsupported Metal storage texture sampled method: "
            f"{method} on {texture} */"
        )

    def unsupported_sampled_texture_write(self, texture_expr, method, is_main):
        texture = self.generate_expression(texture_expr, is_main)
        return f"/* unsupported Metal sampled texture write: {method} on {texture} */"

    def storage_image_coordinate_expression(self, image_expr, args):
        if not args:
            return ""
        rendered_args = [self.generate_expression(arg) for arg in args]
        image_type = self.expression_mapped_type(image_expr)
        if (
            image_type in {"image1DArray", "iimage1DArray", "uimage1DArray"}
            and len(rendered_args) >= 2
        ):
            return f"uvec2({rendered_args[0]}, {rendered_args[1]})"
        if (
            image_type in {"image2DArray", "iimage2DArray", "uimage2DArray"}
            and len(rendered_args) >= 2
        ):
            return f"uvec3({rendered_args[0]}, {rendered_args[1]})"
        return rendered_args[0]

    def sampled_texture_read_expression(self, texture_expr, args, is_main=False):
        mapped_type = self.expression_mapped_type(texture_expr)
        if not self.is_sampled_texture_type(mapped_type) or not args:
            return None

        texture = self.generate_expression(texture_expr, is_main)
        cube_read = self.unsupported_sampled_cube_texture_read_expression(
            texture, mapped_type
        )
        if cube_read is not None:
            return cube_read

        rendered_args = [self.generate_expression(arg, is_main) for arg in args]
        coord, tail = self.sampled_texture_read_coordinate_and_tail(
            mapped_type, rendered_args
        )
        if coord is None:
            return None

        if self.is_multisample_resource_type(mapped_type):
            if not tail:
                return None
            return f"texelFetch({texture}, {coord}, {tail[0]})"

        if self.is_sampled_buffer_texture_type(mapped_type):
            return f"texelFetch({texture}, {coord})"

        lod = tail[0] if tail else "0"
        return f"texelFetch({texture}, {coord}, {lod})"

    def unsupported_sampled_cube_texture_read_expression(self, texture, mapped_type):
        if not self.is_sampled_cube_texture_type(mapped_type):
            return None
        fallback = self.sampled_texture_read_zero_value(mapped_type)
        return (
            f"{fallback} /* unsupported Metal sampled cube texture read: "
            f"read on {texture} requires face-aware texel fetch */"
        )

    def is_sampled_cube_texture_type(self, mapped_type):
        mapped_type = self.resource_classification_type(mapped_type)
        return bool(mapped_type and "Cube" in str(mapped_type))

    def sampled_texture_read_zero_value(self, mapped_type):
        mapped_type = str(self.resource_classification_type(mapped_type) or "")
        if mapped_type.startswith("usampler"):
            return "uvec4(0u)"
        if mapped_type.startswith("isampler"):
            return "ivec4(0)"
        return "vec4(0.0)"

    def is_sampled_buffer_texture_type(self, mapped_type):
        mapped_type = self.resource_classification_type(mapped_type)
        return mapped_type in {"samplerBuffer", "isamplerBuffer", "usamplerBuffer"}

    def is_sampled_texture_type(self, mapped_type):
        if mapped_type == "sampler":
            return False
        return bool(
            mapped_type
            and str(mapped_type).startswith(("sampler", "isampler", "usampler"))
        )

    def sampled_texture_read_coordinate_and_tail(self, mapped_type, rendered_args):
        if not rendered_args:
            return None, []

        coord = rendered_args[0]
        tail = rendered_args[1:]
        constructor = self.sampled_texture_array_coordinate_constructor(mapped_type)
        if constructor is not None:
            if not tail:
                return None, []
            coord = f"{constructor}({coord}, {tail[0]})"
            tail = tail[1:]

        return coord, tail

    def sampled_texture_array_coordinate_constructor(self, mapped_type):
        mapped_type = str(mapped_type)
        if "1DArray" in mapped_type:
            return "uvec2"
        if "2DArray" in mapped_type or "2DMSArray" in mapped_type:
            return "uvec3"
        if "CubeArray" in mapped_type:
            return "vec4"
        return None

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
            else:
                out = self.map_semantics.get(key, self.map_semantics.get(name, None))
            if out is None:
                out = self.dynamic_fragment_output_semantic(name, args)
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

    def dynamic_fragment_output_semantic(self, name, args):
        if name == "color" and args:
            if re.fullmatch(r"\d+", args[0]):
                index = int(args[0])
                return "gl_FragColor" if index == 0 else f"gl_FragColor{index}"
            return "gl_FragColor"
        if name == "depth" and args and args[0].lower() in {"any", "less", "greater"}:
            return "gl_FragDepth"
        return None

    def generate_switch_statement(self, node, indent, is_main):
        previous_type_aliases = dict(self.type_aliases)
        previous_type_alias_qualifiers = dict(self.type_alias_qualifiers)
        previous_local_type_alias_names = set(self.local_type_alias_names)
        previous_local_struct_type_aliases = dict(self.local_struct_type_aliases)
        expression = self.generate_expression(node.expression, is_main)
        code = f"switch ({expression}) {{\n"
        try:
            for case in node.cases:
                case_value = self.generate_expression(case.value, is_main)
                code += "    " * (indent + 1) + f"case {case_value}:\n"
                code += self.generate_function_body(
                    case.statements, indent + 2, is_main
                )
                if not self.switch_case_has_explicit_terminator(case.statements):
                    code += "    " * (indent + 2) + "break;\n"

            if node.default:
                code += "    " * (indent + 1) + "default:\n"
                code += self.generate_function_body(node.default, indent + 2, is_main)
                if not self.switch_case_has_explicit_terminator(node.default):
                    code += "    " * (indent + 2) + "break;\n"

            code += "    " * indent + "}\n"
            return code
        finally:
            self.type_aliases = previous_type_aliases
            self.type_alias_qualifiers = previous_type_alias_qualifiers
            self.local_type_alias_names = previous_local_type_alias_names
            self.local_struct_type_aliases = previous_local_struct_type_aliases

    def switch_case_has_explicit_terminator(self, statements):
        if not statements:
            return False
        return isinstance(
            statements[-1], (BreakNode, ContinueNode, DiscardNode, ReturnNode)
        )
