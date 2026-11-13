"""Reverse code generator that emits CrossGL from Metal AST nodes."""

import re

from crosstl.translator.refusals import MetalSourceRefusal

from ...translator.cooperative_matrix import has_cooperative_matrix_fragment_mapping
from ...translator.host_workgroup_size import normalize_threadgroup_size_option
from ...translator.math_namespaces import (
    metal_fast_math_operation,
    metal_precise_math_operation,
)
from ...translator.precise_power import binary32_power_support
from ...translator.resource_identity import source_resource_attribute
from ..comma_expressions import crossgl_comma_expression
from ..import_refusals import ImportRefusalMixin, MetalImportRefused
from .aggregate_scalar_conversions import (
    refuse_aggregate_builtin_operation,
    render_explicit_aggregate_scalar_conversion,
)
from .alignment_specifiers import MetalAlignmentSpecifiers
from .buffer_layout_import import MetalBufferLayoutImport
from .builtin_operator_operands import operands_have_no_class_type
from .color_attachments import (
    color_attachment_semantic,
    enumerator_index_values,
    framebuffer_fetch_struct_names,
)
from .const_cast_buffer_writes import mark_const_cast_written_buffers
from .crossgl_codegen.alias_templates import MetalAliasTemplateMixin
from .crossgl_codegen.arithmetic_profiles import MetalArithmeticProfileMixin
from .crossgl_codegen.builtin_tables import MetalBuiltinTableMixin
from .crossgl_codegen.constexpr_evaluation import MetalConstexprMixin
from .crossgl_codegen.constructors import MetalConstructorMixin
from .crossgl_codegen.cooperative_matrices import MetalCooperativeMatrixMixin
from .crossgl_codegen.declaration_types import MetalDeclarationTypeMixin
from .crossgl_codegen.declarations import MetalDeclarationMixin
from .crossgl_codegen.expression_types import MetalExpressionTypeMixin
from .crossgl_codegen.functions import MetalFunctionMixin
from .crossgl_codegen.identifiers import MetalIdentifierMixin
from .crossgl_codegen.local_aggregates import MetalLocalAggregateMixin
from .crossgl_codegen.math_builtins import MetalMathBuiltinMixin
from .crossgl_codegen.operands import MetalOperandMixin
from .crossgl_codegen.overload_resolution import MetalOverloadResolutionMixin
from .crossgl_codegen.precise_math import MetalPreciseMathMixin
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
from .floating_point import contraction_directive_requires_precise
from .implicit_constructor_conversions import convert_user_call_arguments
from .instance_static_members import render_instance_static_struct_member
from .library_structs import declare_referenced_library_structs
from .local_enums import hoist_local_enums
from .matrix_columns import MetalMatrixColumnsError, matrix_columns_object
from .mesh_import import MetalMeshImport
from .MetalAst import *
from .MetalLexer import *
from .MetalParser import *
from .namespace_scope_variables import resolve_namespace_scope_variables
from .preprocessor import DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT
from .simd_aliases import SIMD_TYPE_ALIASES
from .sparse_color import (
    METAL_SPARSE_TEXTURE_METHODS,
    import_sparse_color_member,
    refuse_sparse_color_reference,
)
from .static_member_calls import unresolved_static_member_call_owner
from .unscoped_enums import unscoped_enumerator_owners
from .unused_generic_definitions import (
    raise_referenced_deferred_definitions,
    render_generic_definition,
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
        # The host picks a Metal kernel's threadgroup size at dispatch time; this option names it.
        self.threadgroup_size = normalize_threadgroup_size_option(threadgroup_size)
        if type(preserve_resource_origins) is not bool:
            raise MetalSourceRefusal("preserve_resource_origins must be a boolean")
        self.preserve_resource_origins = preserve_resource_origins
        if floating_contraction_profile not in (None, "rne-gradual", "rne-flush"):
            raise MetalSourceRefusal(
                "floating_contraction_profile must be 'rne-gradual', 'rne-flush', "
                "or None"
            )
        # Metal compiles with expression-level contraction even under
        # -fno-fast-math: a product feeding an addition in the same expression
        # becomes one fused operation. The profile selects how binary32
        # subnormals are treated by the portable single-rounding lowering.
        self.floating_contraction_profile = floating_contraction_profile
        if binary32_power_accuracy_profile not in (None, "portable-finite"):
            raise MetalSourceRefusal(
                "binary32_power_accuracy_profile must be 'portable-finite' or None"
            )
        self.binary32_power_accuracy_profile = binary32_power_accuracy_profile
        if binary32_power_operand_profile not in (None, "flush-subnormals"):
            raise MetalSourceRefusal(
                "binary32_power_operand_profile must be 'flush-subnormals' or None"
            )
        self.binary32_power_operand_profile = binary32_power_operand_profile
        if binary32_fma_profile not in (None, "rne-gradual", "rne-flush"):
            raise MetalSourceRefusal(
                "binary32_fma_profile must be 'rne-gradual', 'rne-flush', or None"
            )
        self.binary32_fma_profile = binary32_fma_profile
        if binary32_division_profile not in (None, "rne-gradual", "rne-flush"):
            raise MetalSourceRefusal(
                "binary32_division_profile must be 'rne-gradual', 'rne-flush', or None"
            )
        self.binary32_division_profile = binary32_division_profile
        if binary16_remainder_profile not in (None, "binary32-quotient"):
            raise MetalSourceRefusal(
                "binary16_remainder_profile must be 'binary32-quotient' or None"
            )
        self.binary16_remainder_profile = binary16_remainder_profile
        if binary16_widening_profile not in (None, "canonical-nan"):
            raise MetalSourceRefusal(
                "binary16_widening_profile must be 'canonical-nan' or None"
            )
        self.binary16_widening_profile = binary16_widening_profile
        if binary32_comparison_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_comparison_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_comparison_profile = binary32_comparison_profile
        if binary32_atan2_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_atan2_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_atan2_profile = binary32_atan2_profile
        if binary32_log_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_log_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_log_profile = binary32_log_profile
        if binary32_log2_operand_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_log2_operand_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        if binary32_log2_accuracy_profile not in (None, "portable-finite"):
            raise MetalSourceRefusal(
                "binary32_log2_accuracy_profile must be 'portable-finite' or None"
            )
        self.binary32_log2_operand_profile = binary32_log2_operand_profile
        self.binary32_log2_accuracy_profile = binary32_log2_accuracy_profile
        if binary32_sqrt_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_sqrt_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_sqrt_profile = binary32_sqrt_profile
        if binary32_rsqrt_profile not in (
            None,
            "preserve-subnormals",
            "flush-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_rsqrt_profile must be 'preserve-subnormals', "
                "'flush-subnormals', or None"
            )
        self.binary32_rsqrt_profile = binary32_rsqrt_profile
        if binary32_remainder_profile not in (
            None,
            "preserve-subnormals",
            "flush-arithmetic-subnormals",
        ):
            raise MetalSourceRefusal(
                "binary32_remainder_profile must be 'preserve-subnormals', "
                "'flush-arithmetic-subnormals', or None"
            )
        self.binary32_remainder_profile = binary32_remainder_profile
        if binary32_additive_profile not in (
            None,
            "rne-gradual",
            "rne-flush",
            "native-precise",
        ):
            raise MetalSourceRefusal(
                "binary32_additive_profile must be 'rne-gradual', 'rne-flush', "
                "'native-precise', or None"
            )
        self.binary32_additive_profile = binary32_additive_profile
        if binary32_multiplication_profile not in (None, "rne-gradual", "rne-flush"):
            raise MetalSourceRefusal(
                "binary32_multiplication_profile must be 'rne-gradual', 'rne-flush', or None"
            )
        self.binary32_multiplication_profile = binary32_multiplication_profile
        if not isinstance(preserve_pointer_pointee_const, bool):
            raise MetalSourceRefusal("preserve_pointer_pointee_const must be a boolean")
        self.preserve_pointer_pointee_const = preserve_pointer_pointee_const
        if not isinstance(resolve_standard_remove_cv_aliases, bool):
            raise MetalSourceRefusal(
                "resolve_standard_remove_cv_aliases must be a boolean"
            )
        self.resolve_standard_remove_cv_aliases = resolve_standard_remove_cv_aliases
        mapping_is_configured = cooperative_matrix_fragment_mapping is not None
        provenance_is_configured = (
            cooperative_matrix_fragment_mapping_provenance is not None
        )
        if mapping_is_configured != provenance_is_configured:
            raise MetalSourceRefusal(
                "cooperative_matrix_fragment_mapping and "
                "cooperative_matrix_fragment_mapping_provenance must be "
                "configured together"
            )
        if mapping_is_configured:
            if (
                not isinstance(cooperative_matrix_fragment_mapping, str)
                or not cooperative_matrix_fragment_mapping.strip()
            ):
                raise MetalSourceRefusal(
                    "cooperative_matrix_fragment_mapping must be a non-empty string"
                )
            if (
                not isinstance(cooperative_matrix_fragment_mapping_provenance, str)
                or not cooperative_matrix_fragment_mapping_provenance.strip()
            ):
                raise MetalSourceRefusal(
                    "cooperative_matrix_fragment_mapping_provenance must be a "
                    "non-empty string"
                )
            cooperative_matrix_fragment_mapping = (
                cooperative_matrix_fragment_mapping.strip()
            )
            cooperative_matrix_fragment_mapping_provenance = (
                cooperative_matrix_fragment_mapping_provenance.strip()
            )
            if not has_cooperative_matrix_fragment_mapping(
                cooperative_matrix_fragment_mapping
            ):
                raise MetalSourceRefusal(
                    "Unknown cooperative-matrix fragment mapping profile "
                    f"'{cooperative_matrix_fragment_mapping}'"
                )

        self.cooperative_matrix_fragment_mapping = cooperative_matrix_fragment_mapping
        self.cooperative_matrix_fragment_mapping_provenance = (
            cooperative_matrix_fragment_mapping_provenance
        )
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
            "texture1d": "texture1D",
            "texture1d<float>": "texture1D",
            "texture1d<half>": "texture1D",
            "texture1d<int>": "itexture1D",
            "texture1d<uint>": "utexture1D",
            "texture1d_array": "texture1DArray",
            "texture1d_array<float>": "texture1DArray",
            "texture1d_array<half>": "texture1DArray",
            "texture1d_array<int>": "itexture1DArray",
            "texture1d_array<uint>": "utexture1DArray",
            "texture2d": "texture2D",
            "texture2d<float>": "texture2D",
            "texture2d<half>": "texture2D",
            "texture2d<int>": "itexture2D",
            "texture2d<uint>": "utexture2D",
            "texture2d_ms": "texture2DMS",
            "texture2d_ms<float>": "texture2DMS",
            "texture2d_ms<half>": "texture2DMS",
            "texture2d_ms<int>": "itexture2DMS",
            "texture2d_ms<uint>": "utexture2DMS",
            "texture2d_ms_array": "texture2DMSArray",
            "texture2d_ms_array<float>": "texture2DMSArray",
            "texture2d_ms_array<half>": "texture2DMSArray",
            "texture2d_ms_array<int>": "itexture2DMSArray",
            "texture2d_ms_array<uint>": "utexture2DMSArray",
            "texture3d": "texture3D",
            "texture3d<float>": "texture3D",
            "texture3d<half>": "texture3D",
            "texture3d<int>": "itexture3D",
            "texture3d<uint>": "utexture3D",
            "texturecube": "textureCube",
            "texturecube<float>": "textureCube",
            "texturecube<half>": "textureCube",
            "texturecube<int>": "itextureCube",
            "texturecube<uint>": "utextureCube",
            "TextureCube": "textureCube",
            "texturecube_array": "textureCubeArray",
            "texturecube_array<float>": "textureCubeArray",
            "texturecube_array<half>": "textureCubeArray",
            "texturecube_array<int>": "itextureCubeArray",
            "texturecube_array<uint>": "utextureCubeArray",
            "texture2d_array": "texture2DArray",
            "texture2d_array<float>": "texture2DArray",
            "texture2d_array<half>": "texture2DArray",
            "texture2d_array<int>": "itexture2DArray",
            "texture2d_array<uint>": "utexture2DArray",
            "texture_buffer": "samplerBuffer",
            "texture_buffer<float>": "samplerBuffer",
            "texture_buffer<half>": "samplerBuffer",
            "texture_buffer<int>": "isamplerBuffer",
            "texture_buffer<uint>": "usamplerBuffer",
            "depth2d": "texture2DShadow",
            "depth2d<float>": "texture2DShadow",
            "depth2d_array": "texture2DArrayShadow",
            "depth2d_array<float>": "texture2DArrayShadow",
            "depthcube": "textureCubeShadow",
            "depthcube<float>": "textureCubeShadow",
            "depthcube_array": "textureCubeArrayShadow",
            "depthcube_array<float>": "textureCubeArrayShadow",
            "depth2d_ms": "texture2DMS",
            "depth2d_ms<float>": "texture2DMS",
            "depth2d_ms_array": "texture2DMSArray",
            "depth2d_ms_array<float>": "texture2DMSArray",
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
        # <simd/simd.h> aliases map like the Metal types they name.
        self.type_map.update(
            {
                alias: self.type_map[metal_type]
                for alias, metal_type in SIMD_TYPE_ALIASES.items()
                if metal_type in self.type_map
            }
        )
        self.type_aliases = {}
        self.type_alias_qualifiers = {}
        self.type_alias_pointee_qualifiers = {}
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
        self.struct_member_alias_resolution_stack = []
        self.current_type_resolution_context = None
        self.callable_type_aliases = {}
        self.metal_later_declarations_visible = False
        self.callable_alias_declarations = {}
        self.global_variable_types = {}
        self.current_variable_types = {}
        self.global_variable_type_qualifiers = {}
        self.current_variable_type_qualifiers = {}
        self.metal_source_overload_groups = {}
        self.metal_source_overload_output_names = {}
        self.out_of_line_call_operator_replacements = {}
        self.suppressed_out_of_line_call_operator_ids = set()
        self.storage_texture_declaration_ids = set()
        self.comparison_sampler_ids = set()
        self.global_storage_texture_names = set()
        self.current_storage_texture_names = set()
        self.global_structured_buffer_names = set()
        self.current_structured_buffer_names = set()
        self.current_stage_entry_resource_parameter_ids = set()
        self.global_sampler_names = set()
        self.suppress_structured_buffer_index_lowering = False
        self.structured_buffer_lvalue_roots = set()
        self.struct_member_types = {}
        self.struct_member_name_maps = {}
        self.resolved_struct_member_types = {}
        self.resolved_struct_member_qualifiers = {}
        self.struct_declarations = {}
        self.struct_name_map = {}
        self.ambiguous_struct_names = set()
        self.nested_struct_types_by_owner = {}
        self.struct_static_constants = {}
        self.struct_static_constant_members = {}
        self.struct_static_constant_owner_candidates = {}
        self.equivalent_struct_static_constants = {}
        self.struct_static_constant_resolution_stack = []
        self.struct_static_constexpr_member_keys = set()
        self.current_struct_static_constant_owner = None
        self.struct_template_parameters = {}
        self.local_type_alias_names = set()
        self.unresolved_local_type_aliases = {}
        self.local_struct_type_aliases = {}
        self.local_integral_constant_bindings = {}
        self.global_integral_constant_bindings = {}
        self.local_aggregate_declarations = []
        self.local_aggregate_declaration_keys = {}
        self.local_aggregate_reserved_names = set()
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
        self.constructor_contracts_by_owner = {}
        self.constructor_factory_names = {}
        self.pending_constructor_factories = []
        self.current_constructor_scope_index = None
        self.metal_enum_arithmetic_types = {}
        self.metal_enum_member_types = {}
        self.small_vector_index_types = {}
        self.small_vector_index_operations = set()
        self.small_vector_resource_index_operations = {}
        self.wide_vector_types = {}
        self.wide_vector_binary_helpers = set()
        self.wide_vector_compound_helpers = set()
        self.wide_vector_reserved_names = set()
        self.metal_precise_math_helper_names = {}
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
        self.identifier_maps = [{}]
        self.used_identifier_names = [set()]
        self.template_name_spellings = {}
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

        # [[attribute(n)]] is the vertex input at location n: see map_semantic.
        self.map_semantics = {
            "vertex_id": "gl_VertexID",
            "instance_id": "gl_InstanceIndex",
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
            "quadgroup_index_in_threadgroup": "gl_QuadGroupID",
            "thread_index_in_quadgroup": "gl_QuadGroupInvocationID",
            "thread_index_in_simdgroup": "gl_SubgroupInvocationID",
            "simdgroup_index_in_threadgroup": "gl_SubgroupID",
            "simdgroups_per_threadgroup": "gl_NumSubgroups",
            "threads_per_threadgroup": "gl_WorkGroupSize",
            "threads_per_simdgroup": "gl_SubgroupSize",
            "threadgroups_per_grid": "gl_NumWorkGroups",
            "stage_in": "",
        }

    def current_constructor_member_type(self, name):
        function = self.current_function
        if not getattr(function, "is_metal_constructor_factory", False):
            return None
        if name == "this":
            return getattr(function, "constructor_owner", None)
        member_types = getattr(function, "constructor_member_types", {}) or {}
        if not isinstance(name, str) or self.constructor_identifier_is_shadowed(name):
            return None
        if name in member_types:
            return member_types[name]
        # An unqualified static constant of the owner, as the constructor's
        # initializers and body see it; render_constructor_member_identifier
        # resolves its value the same way.
        owner = getattr(function, "constructor_owner", None)
        member = (
            self.struct_static_constant_members.get((self.map_struct_name(owner), name))
            if owner
            else None
        )
        if member is None or not getattr(member, "vtype", None):
            return None
        return self.normalized_metal_type(self.resolve_type_alias(member.vtype))

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
                for member_name, member_value in enum.members:
                    if member_value is not None:
                        value = self.generate_expression(member_value, False)
                        code += f"        {member_name} = {value},\n"
                    else:
                        code += f"        {member_name},\n"
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
                            if struct_node.name in getattr(
                                self, "metal_fetched_struct_names", ()
                            )
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

        code += local_aggregate_support_marker
        code += small_vector_index_support_marker
        code += cooperative_matrix_fragment_support_marker
        code += precise_math_support_marker

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
            + self.generate_metal_precise_math_support_code(indent=1),
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
            for param in func.params:
                self.current_variable_types[param.name] = (
                    self.metal_declaration_expression_type(param)
                )
                self.current_variable_type_qualifiers[param.name] = (
                    self.metal_declaration_type_qualifiers(param)
                )
                if id(param) in self.storage_texture_declaration_ids:
                    self.current_storage_texture_names.add(param.name)
                if self.structured_buffer_pointer_type(param):
                    self.current_structured_buffer_names.add(param.name)
            semantic_context = {
                "kind": "parameter",
                "function_qualifier": getattr(func, "qualifier", None),
            }
            parameter_declarations = [
                self.format_parameter_decl(p, index, semantic_context=semantic_context)
                for index, p in enumerate(func.params)
            ]
            parameter_names = [self.render_identifier(p.name) for p in func.params]
            for index, parameter in enumerate(func.params):
                layout = self.structured_buffer_parameter_layout(func, parameter)
                if layout is None and stage_entry:
                    layout = self.metal_constant_reference_layout(
                        parameter,
                        f"constant parameter {parameter.name} of "
                        f"{getattr(func, 'name', 'a function')}",
                    )
                if layout is not None:
                    parameter_declarations[index] = (
                        f"@buffer_layout({layout}) " + parameter_declarations[index]
                    )
            if stage_entry and self.preserve_resource_origins:
                for index, parameter in enumerate(func.params):
                    if self.is_stage_entry_buffer_resource_parameter(parameter):
                        parameter_declarations[index] = (
                            source_resource_attribute(
                                "metal",
                                output_name or self.function_output_name(func),
                                parameter.name,
                                index,
                            )
                            + parameter_declarations[index]
                        )
            if out_of_line_replacement is not None:
                for definition_name, helper_name in out_of_line_replacement[
                    "parameter_aliases"
                ]:
                    self.current_variable_types[definition_name] = (
                        self.current_variable_types.get(helper_name)
                    )
                    self.current_variable_type_qualifiers[definition_name] = (
                        self.current_variable_type_qualifiers.get(helper_name, ())
                    )
                    self.identifier_maps[-1][definition_name] = self.render_identifier(
                        helper_name
                    )
            fn_semantic = self.map_semantic(self.function_semantic_attributes(func))
            annotations = " ".join(
                item
                for item in [
                    fn_semantic,
                    *linkage_annotations,
                    self.metal_mesh_function_annotation(func),
                ]
                if item
            )
            suffix = f" {annotations}" if annotations else ""
            function_name = self.sanitize_identifier(
                output_name or self.function_output_name(func)
            )
            self.current_function_specialization = function_name
            self.current_function_materialization_bindings = {
                **active_type_bindings,
                **active_value_bindings,
            }
            self.current_function_specialization_key = specialization_key
            return_type = self.map_function_return_type(
                func.return_type,
                qualifiers=(
                    getattr(func, "return_qualifiers", None)
                    or getattr(func, "declaration_qualifiers", None)
                ),
            )
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
                    body=function_body,
                    bound_value_names=active_value_bindings,
                )
            )
            if function_body is None and getattr(func, "is_template_prototype", False):
                # A declaration-only primary that explicit specializations
                # complete; generators select them per deduced argument tuple.
                params = ", ".join(
                    declaration for declaration in parameter_declarations if declaration
                )
                return (
                    code
                    + f"{generic_prefix}{return_type} {function_name}({params});\n\n"
                )
            body = (
                value_param_decls
                + self.metal_mesh_body_prologue(func, indent + 1)
                + self.metal_object_body_prologue(func, indent + 1)
                + self.generate_function_body(
                    function_body,
                    indent=indent + 1 + self.metal_object_body_indent(func),
                )
                + self.metal_object_body_epilogue(func, indent + 1)
                + self.metal_mesh_body_epilogue(func, indent + 1)
            )
            for index, parameter in enumerate(func.params):
                if (
                    parameter.name in self.current_elided_parameter_uses
                    and "@maybe_unused" not in parameter_declarations[index]
                    and not re.search(rf"\b{re.escape(parameter_names[index])}\b", body)
                ):
                    parameter_declarations[index] += " @maybe_unused"
            params = ", ".join(
                declaration for declaration in parameter_declarations if declaration
            )
            code += (
                f"{generic_prefix}{return_type} {function_name}({params})"
                f"{suffix} {{\n"
            )
            code += body
            code += "    }\n\n"
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
            comma = crossgl_comma_expression(
                expr, lambda operand: self.generate_expression(operand, is_main)
            )
            if comma is not None:
                return comma
            free_operator_call = self.generate_metal_free_operator_call(
                expr.op,
                (expr.left, expr.right),
                is_main,
                getattr(expr, "source_location", None),
            )
            if free_operator_call is not None:
                return free_operator_call
            if self.binary32_comparison_profile is not None and expr.op in {
                "==",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
            }:
                profiled = self.generate_profiled_metal_comparison(expr, is_main)
                if profiled is not None:
                    return profiled
            if self.floating_contraction_profile is not None and expr.op in {"+", "-"}:
                contracted = self.generate_metal_contracted_arithmetic(expr, is_main)
                if contracted is not None:
                    return contracted
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
            if (
                expr.op in {"/", "+", "-", "*"}
                and self.metal_arithmetic_profile(expr.op)[0] is not None
            ):
                profiled = self.generate_profiled_metal_arithmetic(expr, is_main)
                if profiled is not None:
                    return profiled
            conversion_plan = self.metal_builtin_lowered_conversion_plan(
                expr.op,
                (expr.left, expr.right),
            )
            if conversion_plan is not None:
                helpers, _converted_types = conversion_plan
                operands = []
                for index, (operand, helper) in enumerate(
                    zip((expr.left, expr.right), helpers)
                ):
                    if helper is None:
                        operands.append(
                            self.generate_binary_operand(
                                operand,
                                expr.op,
                                index == 1,
                                is_main,
                            )
                        )
                        continue
                    helper_name = self.sanitize_identifier(
                        self.function_output_name(helper)
                    )
                    operands.append(
                        f"{helper_name}({self.generate_expression(operand, is_main)})"
                    )
                return f"{operands[0]} {expr.op} {operands[1]}"
            refuse_aggregate_builtin_operation(
                self, expr, getattr(expr, "source_location", None)
            )
            left = self.generate_binary_operand(expr.left, expr.op, False, is_main)
            right = self.generate_binary_operand(expr.right, expr.op, True, is_main)
            source_types = [
                self.expression_metal_type(operand)
                for operand in (expr.left, expr.right)
            ]
            if self.binary16_widening_profile is not None and expr.op in {
                "+",
                "-",
                "*",
                "/",
                "==",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
            }:
                computation = (
                    self.metal_binary_expression_type(expr)
                    if expr.op in {"+", "-", "*", "/"}
                    else self.metal_scalar_binary_result_type("+", *source_types)
                )
                left = self.render_metal_half_widening(
                    left, source_types[0], computation
                )
                right = self.render_metal_half_widening(
                    right, source_types[1], computation
                )
            bfloat_plan = self.metal_bfloat_arithmetic_plan(
                expr.op, *source_types, getattr(expr, "source_location", None)
            )
            if bfloat_plan is not None:
                operands = [left, right]
                for index, conversion_type in enumerate(bfloat_plan[1]):
                    if conversion_type is not None:
                        operands[index] = (
                            f"{self.map_type(conversion_type)}({operands[index]})"
                        )
                return f"{operands[0]} {expr.op} {operands[1]}"
            conversion = self.metal_vector_scalar_conversion(expr.op, *source_types)
            if conversion is not None:
                index, element_type = conversion
                operands = [left, right]
                operands[index] = self.metal_vector_scalar_operand(
                    operands[index], source_types[index], element_type
                )
                left, right = operands
            if expr.op in {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}:
                source_types = [
                    self.resolve_type_alias(source_type) for source_type in source_types
                ]
                operand_types = [
                    self.normalized_metal_type(vtype) for vtype in source_types
                ]
                if (
                    "bool" in operand_types
                    and all(
                        self.metal_pointer_pointee_type_once(vtype) is None
                        for vtype in source_types
                    )
                    and all(
                        self.metal_scalar_arithmetic_type_info(vtype) is not None
                        for vtype in operand_types
                    )
                ):
                    # C++ scalar bool arithmetic promotes to signed int before
                    # the operation, not back to bool after each subexpression.
                    if operand_types[0] == "bool":
                        left = f"int({left})"
                    if operand_types[1] == "bool":
                        right = f"int({right})"
            return f"{left} {expr.op} {right}"
        elif isinstance(expr, FunctionCallNode):
            filled_matrix = self.generate_metal_make_filled_matrix(expr, is_main)
            if filled_matrix is not None:
                return filled_matrix
            profiled_constructor = self.generate_metal_half_widening_constructor(
                expr.name, expr.args, is_main
            )
            if profiled_constructor is not None:
                return profiled_constructor
            lowered_static_call = self.generate_lowered_static_struct_method_call(
                expr, is_main
            )
            if lowered_static_call is not None:
                return lowered_static_call
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
            aggregate_conversion = render_explicit_aggregate_scalar_conversion(
                self,
                expr.name,
                constructor_arguments,
                is_main,
                getattr(expr, "source_location", None),
            )
            if aggregate_conversion is not None:
                return aggregate_conversion
            explicit_constructor = self.generate_explicit_constructor_call(
                expr.name,
                constructor_arguments,
                is_main,
                getattr(expr, "source_location", None),
            )
            if explicit_constructor is not None:
                return explicit_constructor
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
            bit_scan = self.metal_bit_scan_expression(expr.name, expr.args, is_main)
            if bit_scan is not None:
                return bit_scan
            sync_call = self.metal_synchronization_function_call(
                expr.name,
                expr.args,
                getattr(expr, "source_location", None),
            )
            if sync_call is not None:
                return sync_call
            atomic_call = self.metal_atomic_function_call(
                expr.name, expr.args, is_main, getattr(expr, "source_location", None)
            )
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
            self.validate_materialized_metal_stdlib_wrapper_call(expr)
            materialized_wave_call = (
                self.generate_materialized_bfloat_wave_wrapper_call(expr, is_main)
            )
            if materialized_wave_call is not None:
                return materialized_wave_call
            materialized_math_call = (
                self.generate_materialized_bfloat_math_wrapper_call(expr, is_main)
            )
            if materialized_math_call is not None:
                return materialized_math_call
            atan2_call = self.generate_metal_precise_atan2_call(expr, is_main)
            if atan2_call is not None:
                return atan2_call
            power_call = self.generate_metal_power_call(expr, is_main)
            if power_call is not None:
                return power_call
            log2_call = self.generate_metal_log2_call(expr, is_main)
            if log2_call is not None:
                return log2_call
            round_call = self.generate_metal_round_call(expr, is_main)
            if round_call is not None:
                return round_call
            fused_call = self.generate_metal_fma_call(expr, is_main)
            if fused_call is not None:
                return fused_call
            remainder_call = self.generate_metal_half_remainder_call(expr, is_main)
            if remainder_call is not None:
                return remainder_call
            remainder_call = self.generate_metal_remainder_call(expr, is_main)
            if remainder_call is not None:
                return remainder_call
            if self.resolve_metal_math_builtin_name(expr.name, expr.args) == "divide":
                result_type = self.metal_math_builtin_result_type(expr)
                left, right = (
                    self.generate_expression(arg, is_main) for arg in expr.args
                )
                if self.metal_math_builtin_namespace_mode(expr.name) != "fast":
                    if (
                        self.binary32_division_profile is not None
                        and self.current_function is None
                        and self.metal_profiled_arithmetic_type(result_type) is not None
                    ):
                        raise MetalDivisionProfileError(
                            self.binary32_division_profile,
                            result_type,
                            "global constant division cannot call a runtime arithmetic helper",
                            getattr(expr, "source_location", None),
                        )
                    profiled = self.render_profiled_metal_division(
                        left, right, result_type
                    )
                    if profiled is not None:
                        return profiled
                return f"(({left}) / ({right}))"
            if self.resolve_metal_math_builtin_name(expr.name, expr.args) == "copysign":
                self.metal_math_builtin_result_type(expr)
            wrapper_arguments = self.generate_storage_wrapper_call_arguments(
                materialized_name, expr, is_main
            )
            if self.binary16_widening_profile is not None:
                selected = self.selected_metal_callable(expr)
                if selected is None:
                    candidates = self.metal_user_function_overloads(
                        materialized_name, source_offset=self.alias_source_offset(expr)
                    )
                    signatures = {
                        self.metal_function_source_signature(candidate)
                        for candidate in candidates
                    }
                    if len(signatures) == 1 and len(candidates[0].params) == len(
                        expr.args
                    ):
                        candidate = candidates[0]
                        if all(
                            self.metal_source_overload_argument_match_rank(
                                argument,
                                self.expression_metal_type(argument),
                                parameter,
                            )
                            is not None
                            for argument, parameter in zip(expr.args, candidate.params)
                        ):
                            selected = candidate
                if selected is not None:
                    for index, parameter in enumerate(selected.params):
                        if index not in wrapper_arguments and index < len(expr.args):
                            wrapper_arguments[index] = (
                                self.generate_metal_value_argument(
                                    expr.args[index], parameter, is_main
                                )
                            )
            convert_user_call_arguments(
                self, materialized_name, expr, wrapper_arguments, is_main
            )
            materialized_name = self.transported_metal_source_overload_name(
                materialized_name,
                expr.args,
                getattr(expr, "source_location", None),
            )
            unbound_member = unresolved_static_member_call_owner(
                self, materialized_name
            )
            if unbound_member is not None:
                raise MetalStructMethodCallResolutionError(
                    unbound_member[0],
                    unbound_member[1],
                    (),
                    (),
                    "the struct declares no materialized static member function of "
                    "that name",
                    getattr(expr, "source_location", None),
                )
            function_name = self.map_function_call_name(
                materialized_name,
                expr.args,
                source_offset=self.alias_source_offset(expr),
            )
            if function_name == "sampler":
                args = ", ".join(
                    self.generate_sampler_constructor_arg(arg, is_main)
                    for arg in expr.args
                )
            else:
                args = ", ".join(
                    (
                        wrapper_arguments[index]
                        if index in wrapper_arguments
                        else self.generate_metal_wave_argument(
                            materialized_name, function_name, expr, index, is_main
                        )
                    )
                    for index in range(len(expr.args))
                )
            if metal_fast_math_operation(materialized_name) == function_name:
                # CrossGL keeps the namespace for the Metal backend; its parser
                # hands every other backend the plain builtin.
                function_name = f"fast::{function_name}"
            elif metal_precise_math_operation(materialized_name) == function_name:
                function_name = f"precise::{function_name}"
            elif function_name == "sizeof":
                # An operand this importer cannot size names a template parameter; CrossGL sizes it with MSL's
                # layout once the generators know the type.
                function_name = "metal_sizeof"
            return f"{function_name}({args})"
        elif isinstance(expr, LambdaNode):
            return self.generate_lambda_expression(expr, is_main)
        elif isinstance(expr, CallNode):
            if (
                isinstance(expr.callee, FunctionCallNode)
                and str(expr.callee.name) in {"decltype", "metal::decltype"}
                and len(expr.callee.args) == 1
            ):
                inferred_type = self.expression_metal_type(expr.callee)
                if inferred_type is not None:
                    constructor = self.map_type(inferred_type)
                    arguments = ", ".join(
                        self.generate_expression(arg, is_main) for arg in expr.args
                    )
                    return f"{constructor}({arguments})"
            callee = self.generate_postfix_operand(expr.callee, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{callee}({args})"
        elif isinstance(expr, MethodCallNode):
            sparse_member = import_sparse_color_member(self, expr, is_main)
            if sparse_member is not None:
                return sparse_member
            wide_vector = self.wide_vector_expression_info(expr.object)
            if wide_vector is not None:
                raise MetalWideVectorLoweringError(
                    wide_vector["source_type"],
                    "the method has no semantics-preserving aggregate overload",
                    getattr(expr, "source_location", None),
                    operation=f"method {expr.method}",
                )
            obj = self.generate_postfix_operand(expr.object, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            method = expr.method
            if method in METAL_SPARSE_TEXTURE_METHODS:
                # A sparse read returns sparse_color<T>: the texel and whether its tile is resident.
                return self.refuse_import(
                    f"sparse texture {method}: CrossGL has no texture read that reports tile residency "
                    "(the texel and residency pair MSL returns as sparse_color<T>)"
                )
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
                    return self.metal_depth_texel_component(expr.object, sampled_read)
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
                        value = self.storage_image_write_value(
                            expr.object, expr.args[0], value
                        )
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
            if static_member is None:
                # `t.value` for a static member names its constant (instance_static_members.py).
                static_member = render_instance_static_struct_member(self, expr)
            if static_member is not None:
                return static_member
            if matrix_columns_object(self, expr) is not None:
                raise MetalMatrixColumnsError(getattr(expr, "source_location", None))
            obj = self.generate_postfix_operand(expr.object, is_main)
            constructor_this = (
                isinstance(expr.object, VariableNode)
                and expr.object.name == "this"
                and self.render_constructor_member_identifier("this") is not None
            )
            if getattr(expr, "is_pointer", False) and not constructor_this:
                obj = f"(*{obj})"
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
            return f"{obj}.{self.render_struct_member_identifier(expr.object, expr.member)}"
        elif isinstance(expr, ArrayAccessNode):
            cooperative_matrix_element = self.metal_cooperative_matrix_element_access(
                expr
            )
            if cooperative_matrix_element is not None:
                matrix, element_index = cooperative_matrix_element
                self.register_cooperative_matrix_element_contract(matrix)
                matrix_text = self.generate_expression(matrix, is_main)
                index_text = self.generate_expression(element_index, is_main)
                return f"cooperative_matrix_element({matrix_text}, {index_text})"
            matrix = matrix_columns_object(self, expr.array)
            if matrix is not None:
                matrix_text = self.generate_postfix_operand(matrix, is_main)
                return f"{matrix_text}[{self.generate_expression(expr.index, is_main)}]"
            component_info = self.small_vector_index_info(expr)
            if component_info is not None:
                return self.generate_small_vector_component_read(
                    expr, component_info, is_main
                )
            if (
                not self.suppress_structured_buffer_index_lowering
                and id(expr) not in self.structured_buffer_lvalue_roots
                and self.is_structured_buffer_element_access(expr)
            ):
                return self.generate_structured_buffer_load(expr, is_main)
            wide_vector = self.wide_vector_expression_info(expr.array)
            array = self.generate_postfix_operand(expr.array, is_main)
            index = self.generate_expression(expr.index, is_main)
            if wide_vector is not None:
                return f"{array}.lanes[{index}]"
            return f"{array}[{index}]"
        elif isinstance(expr, UnaryOpNode):
            free_operator_call = self.generate_metal_free_operator_call(
                expr.op,
                (expr.operand,),
                is_main,
                getattr(expr, "source_location", None),
            )
            if free_operator_call is not None:
                return free_operator_call
            if expr.op in {"++", "--"} and isinstance(expr.operand, ArrayAccessNode):
                component_update = self.generate_small_vector_component_update(
                    expr.operand, expr.op, postfix=False, is_main=is_main
                )
                if component_update is not None:
                    return component_update
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
            elif expr.op == "&" and isinstance(expr.operand, ArrayAccessNode):
                component_info = self.small_vector_index_info(expr.operand)
                if component_info is not None:
                    array = self.generate_postfix_operand(expr.operand.array, is_main)
                    index = self.generate_expression(expr.operand.index, is_main)
                    operand = f"{array}[{index}]"
                else:
                    operand = self.generate_expression(expr.operand, is_main)
            elif expr.op in {"++", "--"}:
                operand = self.generate_with_structured_buffer_lvalue(
                    expr.operand,
                    lambda: self.generate_expression(expr.operand, is_main),
                )
            else:
                operand = self.generate_expression(expr.operand, is_main)
            if expr.op == "post...":
                return operand
            if isinstance(expr.operand, (AssignmentNode, BinaryOpNode, TernaryOpNode)):
                operand = f"({operand})"
            return f"({expr.op}{operand})"
        elif isinstance(expr, PostfixOpNode):
            if expr.op in {"++", "--"} and isinstance(expr.operand, ArrayAccessNode):
                component_update = self.generate_small_vector_component_update(
                    expr.operand, expr.op, postfix=True, is_main=is_main
                )
                if component_update is not None:
                    return component_update
            wide_vector = self.wide_vector_expression_info(expr.operand)
            if wide_vector is not None:
                raise MetalWideVectorLoweringError(
                    wide_vector["source_type"],
                    "the postfix operator has no semantics-preserving aggregate lowering",
                    getattr(expr, "source_location", None),
                    operation=expr.op,
                )
            operand = self.generate_with_structured_buffer_lvalue(
                expr.operand,
                lambda: self.generate_postfix_operand(expr.operand, is_main),
            )
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
            condition = self.generate_precedence_operand(
                expr.condition,
                self.conditional_precedence,
                is_main,
                parenthesize_equal=True,
            )
            true_expr = self.generate_expression(expr.true_expr, is_main)
            false_expr = self.generate_precedence_operand(
                expr.false_expr,
                self.conditional_precedence,
                is_main,
            )
            if self.binary16_widening_profile is not None:
                result_type = self.expression_metal_type(expr)
                true_expr = self.render_metal_half_widening(
                    true_expr, self.expression_metal_type(expr.true_expr), result_type
                )
                false_expr = self.render_metal_half_widening(
                    false_expr, self.expression_metal_type(expr.false_expr), result_type
                )
            return f"{condition} ? {true_expr} : {false_expr}"
        elif isinstance(expr, CastNode):
            compound_literal = self.generate_metal_compound_literal(expr, is_main)
            if compound_literal is not None:
                return compound_literal
            profiled_constructor = self.generate_metal_half_widening_constructor(
                expr.target_type, [expr.expression], is_main
            )
            if profiled_constructor is not None:
                return profiled_constructor
            cooperative_fragment = self.generate_cooperative_matrix_fragment_read(
                expr, is_main
            )
            if cooperative_fragment is not None:
                return cooperative_fragment
            aggregate_conversion = render_explicit_aggregate_scalar_conversion(
                self,
                expr.target_type,
                [expr.expression],
                is_main,
                getattr(expr, "source_location", None),
            )
            if aggregate_conversion is not None:
                return aggregate_conversion
            explicit_constructor = self.generate_explicit_constructor_call(
                expr.target_type,
                [expr.expression],
                is_main,
                getattr(expr, "source_location", None),
            )
            if explicit_constructor is not None:
                return explicit_constructor
            wide_vector_cast = self.generate_wide_vector_constructor(
                expr.target_type,
                [expr.expression],
                is_main,
                getattr(expr, "source_location", None),
            )
            if wide_vector_cast is not None:
                return wide_vector_cast
            mapped_type = self.map_type(expr.target_type)
            if (
                self.metal_pointer_pointee_type_once(
                    self.resolve_type_alias(expr.target_type)
                )
                is not None
            ):
                qualifier_set = set(getattr(expr, "qualifiers", ()) or ())
                pointer_qualifiers = [
                    qualifier
                    for qualifier in (
                        "const",
                        "threadgroup_imageblock",
                        "threadgroup",
                        "thread",
                        "device",
                        "constant",
                    )
                    if qualifier in qualifier_set
                ]
                if (
                    not self.preserve_pointer_pointee_const
                    and "threadgroup" in pointer_qualifiers
                ):
                    # Threadgroup pointer parameters and locals drop pointee
                    # const under this policy, so a cast to `const threadgroup
                    # T*` must too, or it cannot initialize those declarations.
                    pointer_qualifiers = [
                        qualifier
                        for qualifier in pointer_qualifiers
                        if qualifier != "const"
                    ]
                if pointer_qualifiers:
                    mapped_type = f"{' '.join(pointer_qualifiers)} {mapped_type}"
            value = self.generate_expression(expr.expression, is_main)
            if not self.cast_uses_constructor_syntax(mapped_type):
                return (
                    f"({mapped_type})"
                    f"{self.generate_cast_operand(expr.expression, value)}"
                )
            return f"{self.sanitize_identifier(mapped_type)}({value})"
        elif isinstance(expr, VectorConstructorNode):
            profiled_constructor = self.generate_metal_half_widening_constructor(
                expr.type_name, expr.args, is_main
            )
            if profiled_constructor is not None:
                return profiled_constructor
            size_query = self.texture_size_constructor_expression(expr, is_main)
            if size_query is not None:
                return size_query
            aggregate_conversion = render_explicit_aggregate_scalar_conversion(
                self,
                expr.type_name,
                list(expr.args),
                is_main,
                getattr(expr, "source_location", None),
            )
            if aggregate_conversion is not None:
                return aggregate_conversion
            explicit_constructor = self.generate_explicit_constructor_call(
                expr.type_name,
                list(expr.args),
                is_main,
                getattr(expr, "source_location", None),
            )
            if explicit_constructor is not None:
                return explicit_constructor
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
            return self.generate_initializer_list(
                expr,
                is_main,
                getattr(expr, "_metal_source_overload_expected_type", None),
            )
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
            sampler = self.generate_expression(sampler_expr, is_main)
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
                    return self.metal_depth_texel_component(expr.texture, option_call)

                lod_expr = self.unwrap_texture_option_argument(expr.lod, "level")
                lod = self.generate_expression(lod_expr, is_main)
                return self.metal_depth_texel_component(
                    expr.texture, f"textureLod({', '.join(sample_args + [lod])})"
                )

            return self.metal_depth_texel_component(
                expr.texture, f"texture({', '.join(sample_args)})"
            )
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
            return "gl_FragDepth"
        return None
