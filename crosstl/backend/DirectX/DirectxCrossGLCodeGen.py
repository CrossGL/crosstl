"""Reverse code generator that emits CrossGL from HLSL AST nodes."""

from crosstl.translator.arithmetic_conversions import source_integer_shape
from crosstl.translator.codegen.hlsl_buffer_layout import (
    BUFFER_LAYOUT_ATTRIBUTE,
    HLSL_CBUFFER_LAYOUT,
)
from crosstl.translator.lexer import KEYWORDS as CROSSGL_KEYWORDS
from crosstl.translator.refusals import DirectXSourceRefusal

from ...translator.unhandled import UNHANDLED
from ..comma_expressions import crossgl_comma_expression
from ..common_ast import (
    ArrayAccessNode,
    CastNode,
    InitializerListNode,
    TextureSampleNode,
)
from ..import_refusals import DirectXImportRefused, ImportRefusalMixin
from .crossgl_codegen.call_expressions import CallExpressionMixin
from .crossgl_codegen.control_flow import ControlFlowMixin
from .crossgl_codegen.function_statements import FunctionStatementMixin
from .crossgl_codegen.identifiers_and_attributes import IdentifierAttributeMixin
from .crossgl_codegen.program_declarations import ProgramDeclarationMixin
from .crossgl_codegen.resource_methods import ResourceMethodMixin
from .crossgl_codegen.storage_resources import StorageResourceMixin
from .crossgl_codegen.texture_method_descriptors import TextureMethodDescriptorMixin
from .crossgl_codegen.texture_usage import TextureUsageMixin
from .crossgl_codegen.type_mapping import TypeMappingMixin
from .crossgl_codegen.value_types import ValueTypeMixin
from .DirectxAst import *
from .DirectxLexer import *
from .DirectxParser import *
from .function_templates import (
    specialization_identifier_renames,
    template_call_result_raw_type,
)
from .hlsl_matrix_import import HLSLMatrixImportMixin, hlsl_matrix_type_map
from .implicit_vector_truncation import HLSLImplicitVectorTruncationMixin
from .import_tables import (
    HLSL_BITWISE_OPERATORS,
    HLSL_BUFFER_METHODS,
    HLSL_BYTE_ADDRESS_INTERLOCKED_METHODS,
    HLSL_FEEDBACK_METHODS,
    HLSL_FUNCTION_MAP,
    HLSL_FUNCTION_STATEMENT_SEQUENCES,
    HLSL_GENERIC_PASSTHROUGH_TYPES,
    HLSL_INTERLOCKED_FUNCTIONS,
    HLSL_LEGACY_TEXTURE_FUNCTION_SAMPLER_TYPES,
    HLSL_SEMANTICS,
    HLSL_SHADOW_TEXTURE_TYPES,
    HLSL_STRUCTURED_BUFFER_TYPES,
    HLSL_TEXTURE_GATHER_COMPARE_COMPONENTS,
    HLSL_TEXTURE_GATHER_COMPONENTS,
    HLSL_TEXTURE_METHODS,
)
from .namespaces import resolve_namespaces
from .precise_arithmetic import HLSLPreciseArithmeticMixin
from .push_constant_offsets import member_offsets as push_constant_member_offsets
from .raw_gather_textures import collect_raw_gather_texture_declaration_ids
from .shader_record_import import is_hlsl_shader_record, shader_record_layout
from .struct_method_calls import HLSLStructMethodCallMixin
from .struct_methods import analyze_receiver_mutations, collect_struct_methods
from .struct_templates import (
    name_explicit_struct_specializations,
    refuse_struct_template_references,
    struct_template_parameters,
    struct_template_prefix,
    struct_template_type_names,
    unsupported_struct_templates,
)
from .texel_access import HLSLTexelAccessMixin
from .texel_buffers import HLSLTexelBufferMixin
from .trigonometric_reduction import HLSLTrigonometricReductionMixin
from .unorm_store_rounding import HLSLUnormStoreRoundingMixin
from .value_shapes import HLSLValueShapeMixin

# Namespaced HLSL types that CrossGL keeps under the same scoped spelling.
_SCOPED_BUILTIN_TYPE_NAMES = frozenset({"dx::HitObject"})


def _renamed(renames, name):
    """``name`` as the program spells it: its new name when ``renames`` (which holds only the declarations this
    importer renamed, to avoid a CrossGL keyword, a scope separator or a clash) renames it, else unchanged.
    """
    if name in renames:
        return renames[name]
    return name


class HLSLIntegerConversionError(ValueError):
    """Reject an integer conversion whose source evaluation cannot be retained."""

    project_diagnostic_code = "project.translate.hlsl-integer-conversion-unsupported"
    missing_capabilities = ("hlsl.integer-arithmetic-conversion",)

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f"Cannot preserve HLSL integer arithmetic: {reason}")


class HLSLToCrossGLConverter(
    ResourceMethodMixin,
    IdentifierAttributeMixin,
    ValueTypeMixin,
    StorageResourceMixin,
    TextureUsageMixin,
    ProgramDeclarationMixin,
    FunctionStatementMixin,
    ControlFlowMixin,
    TypeMappingMixin,
    HLSLMatrixImportMixin,
    ImportRefusalMixin,
    HLSLPreciseArithmeticMixin,
    HLSLUnormStoreRoundingMixin,
    HLSLStructMethodCallMixin,
    HLSLTexelAccessMixin,
    HLSLTexelBufferMixin,
    HLSLTrigonometricReductionMixin,
    HLSLValueShapeMixin,
    HLSLImplicitVectorTruncationMixin,
    CallExpressionMixin,
    TextureMethodDescriptorMixin,
):
    """Serialize DirectX backend AST nodes back into CrossGL source."""

    IMPORT_REFUSAL_TYPE = DirectXImportRefused

    crossgl_reserved_identifiers = frozenset(CROSSGL_KEYWORDS)
    MATRIX_PACKINGS = ("column_major", "row_major")
    TRIGONOMETRIC_REDUCTIONS = ("exact", "revolutions", "warp")

    def __init__(
        self,
        matrix_packing=None,
        trigonometric_reduction=None,
        floating_point_contraction=None,
        unorm_store_rounding=None,
        zero_base_vertex=False,
    ):
        """Create a converter.

        ``matrix_packing`` mirrors the compiler's default matrix packing
        (DXC ``-Zpr`` is ``"row_major"``). With row-major packing a matrix
        register holds a row, so CrossGL holds each HLSL row as a column and
        ``mul(a, b)`` becomes ``b * a``.

        ``trigonometric_reduction`` selects how runtime ``sin``, ``cos`` and
        ``tan`` operands are reduced. ``"exact"`` (the default) keeps the
        target's own argument reduction. ``"revolutions"`` reproduces the
        reduction of Direct3D GPUs, which scale the operand by ``1 / (2 * pi)``
        in single precision and take the fractional revolution, so results of
        large operands (for example ``frac(sin(x) * 43758.5)`` hashes) match
        Direct3D instead of a correctly reduced operand. ``"warp"`` evaluates
        the intrinsics exactly as WARP, the Direct3D 12 software rasterizer,
        does: the same single-precision revolution reduction, a 32-entry
        table and short Taylor polynomials, with every operation rounded to
        single precision and floating-point contraction disabled. The results
        then match WARP bit for bit for normal operands; subnormal operands
        are not flushed to zero. Operands that DXC folds at compile time
        (literals and ``static const`` values) keep the exact intrinsic.

        ``floating_point_contraction`` controls fused multiply-adds.
        ``"allowed"`` (the default) lets the target contract arithmetic as
        Direct3D drivers may. ``"off"`` evaluates arithmetic as DXC lowers it
        and WARP executes it, rounding every product and sum: every function
        is ``precise``, and ``dot``, ``length``, ``distance``, ``normalize``,
        ``lerp``, ``smoothstep``, ``cross``, ``reflect``, ``faceforward`` and
        matrix products expand into precise helpers in DXC's order of
        operations, since target intrinsics fuse internally. Integer
        operands and operands DXC folds at compile time keep the intrinsic.

        ``unorm_store_rounding`` selects how values stored to storage
        textures with a ``unorm`` element type become integers. ``"target"``
        (the default) leaves the conversion to the target.
        ``"single_precision"`` converts as Direct3D does, scaling in single
        precision and rounding halfway cases to even, with channel scales the
        host binds per view (see ``unorm_store_rounding``).

        ``zero_base_vertex`` declares that every draw the shader runs in has
        a zero base vertex (``StartVertexLocation`` and ``BaseVertexLocation``
        0). ``SV_VertexID`` excludes the base vertex, which the vertex index
        of OpenGL, Vulkan and Metal includes, so by default it imports as
        ``gl_RelativeVertexID`` and other targets subtract their base vertex
        from their index. Under the declaration both agree, and it imports as
        ``gl_VertexID``, which targets without a base vertex (GLSL ES, WGSL)
        can also express.
        """
        if matrix_packing is not None and matrix_packing not in self.MATRIX_PACKINGS:
            raise DirectXSourceRefusal(
                "matrix_packing must be one of "
                f"{', '.join(self.MATRIX_PACKINGS)}, got {matrix_packing!r}"
            )
        if (
            trigonometric_reduction is not None
            and trigonometric_reduction not in self.TRIGONOMETRIC_REDUCTIONS
        ):
            raise DirectXSourceRefusal(
                "trigonometric_reduction must be one of "
                f"{', '.join(self.TRIGONOMETRIC_REDUCTIONS)}, "
                f"got {trigonometric_reduction!r}"
            )
        if (
            floating_point_contraction is not None
            and floating_point_contraction not in self.FLOATING_POINT_CONTRACTIONS
        ):
            raise DirectXSourceRefusal(
                "floating_point_contraction must be one of "
                f"{', '.join(self.FLOATING_POINT_CONTRACTIONS)}, "
                f"got {floating_point_contraction!r}"
            )
        self.matrix_packing = matrix_packing
        self.trigonometric_reduction = trigonometric_reduction or "exact"
        self.floating_point_contraction = floating_point_contraction or "allowed"
        if (
            unorm_store_rounding is not None
            and unorm_store_rounding not in self.UNORM_STORE_ROUNDINGS
        ):
            raise DirectXSourceRefusal(
                "unorm_store_rounding must be one of "
                f"{', '.join(self.UNORM_STORE_ROUNDINGS)}, "
                f"got {unorm_store_rounding!r}"
            )
        self.unorm_store_rounding = unorm_store_rounding or "target"
        self.reset_precise_arithmetic()
        self.revolution_reduction_types = set()
        self.warp_trigonometric_uses = set()
        self.hlsl_static_const_initializers = {}
        self.constant_initializer_depth = 0
        self.function_return_types = {}
        self.current_function_return_type = None
        self.function_overloads = {}
        self.source_function_names = set()
        self.source_functions = {}
        self.source_type_aliases = {}
        self.integer_compound_helpers = {}
        self.integer_helper_reserved_names = set()
        self.structured_buffer_types = set(HLSL_STRUCTURED_BUFFER_TYPES)
        self.generic_passthrough_types = set(HLSL_GENERIC_PASSTHROUGH_TYPES)
        self.type_map = {
            # Scalar Types
            "void": "void",
            "bool": "bool",
            "bool1": "bool",
            "int": "int",
            "int1": "int",
            "uint": "uint",
            "uint1": "uint",
            "UINT": "uint",
            "dword": "uint",
            "float": "float",
            "float1": "float",
            "float32_t": "float",
            "half": "float16",
            "half1": "float16",
            "fixed": "float",
            "fixed1": "float",
            "float16_t": "float16",
            "double": "double",
            "double1": "double",
            "min16float": "float16",
            "min16float1": "float16",
            "min10float": "float16",
            "min10float1": "float16",
            "min16int": "int16",
            "min16int1": "int16",
            "min12int": "int16",
            "min12int1": "int16",
            "int32_t": "int",
            "int16_t": "int16",
            "min16uint": "uint16",
            "min16uint1": "uint16",
            "uint32_t": "uint",
            "uint16_t": "uint16",
            "int64_t": "int64",
            "uint64_t": "uint64",
            "int8_t4_packed": "uint",
            "uint8_t4_packed": "uint",
            # Vector Types - float
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            # Vector Types - half
            "half2": "f16vec2",
            "half3": "f16vec3",
            "half4": "f16vec4",
            # Unity fixed precision aliases lower to regular float CrossGL types.
            "fixed2": "vec2",
            "fixed3": "vec3",
            "fixed4": "vec4",
            # Vector Types - double
            "double2": "dvec2",
            "double3": "dvec3",
            "double4": "dvec4",
            # Vector Types - int
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            # Vector Types - uint
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            # Vector Types - bool
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            # Matrix Types: HLSL floatRxC is CrossGL matCxR (ADR 0004)
            **hlsl_matrix_type_map(),
            # Texture Types
            # HLSL textures are separate from their samplers (SRVs): CrossGL's separate texture types.
            "Texture1D": "texture1D",
            "Texture1DArray": "texture1DArray",
            "Texture2D": "texture2D",
            "sampler2D_half": "sampler2D",
            "sampler2D_float": "sampler2D",
            "Texture3D": "texture3D",
            "TextureCube": "textureCube",
            "Texture2DArray": "texture2DArray",
            "TextureCubeArray": "textureCubeArray",
            "Texture2DMS": "texture2DMS",
            "Texture2DMSArray": "texture2DMSArray",
            # Vulkan input attachments ([[vk::input_attachment_index]]).
            "SubpassInput": "subpassInput",
            "SubpassInputMS": "subpassInputMS",
            "FeedbackTexture2D": "feedbackTexture2D",
            "FeedbackTexture2DArray": "feedbackTexture2DArray",
            # RW Texture Types (for compute shaders)
            "RWTexture1D": "image1D",
            "RWTexture1DArray": "image1DArray",
            "RWTexture2D": "image2D",
            "RWTexture2DArray": "image2DArray",
            "RWTexture2DMS": "image2DMS",
            "RWTexture2DMSArray": "image2DMSArray",
            "RWTexture3D": "image3D",
            "RWTextureCube": "imageCube",
            "RWTextureCubeArray": "imageCubeArray",
            "RasterizerOrderedTexture1D": "image1D",
            "RasterizerOrderedTexture1DArray": "image1DArray",
            "RasterizerOrderedTexture2D": "image2D",
            "RasterizerOrderedTexture2DArray": "image2DArray",
            "RasterizerOrderedTexture3D": "image3D",
            # Buffer Types
            "Buffer": "samplerBuffer",
            "RWBuffer": "imageBuffer",
            "StructuredBuffer": "buffer",
            "RWStructuredBuffer": "buffer",
            "ByteAddressBuffer": "ByteAddressBuffer",
            "RWByteAddressBuffer": "RWByteAddressBuffer",
            "RasterizerOrderedByteAddressBuffer": "RWByteAddressBuffer",
            "AppendStructuredBuffer": "buffer",
            "ConsumeStructuredBuffer": "buffer",
            "RaytracingAccelerationStructure": "accelerationStructureEXT",
            "RayQuery": "RayQuery<RAY_FLAG_NONE>",
            "InputPatch": "inputPatch",
            "OutputPatch": "outputPatch",
            "PointStream": "pointStream",
            "LineStream": "lineStream",
            "TriangleStream": "triangleStream",
            # Sampler Types
            "sampler": "sampler",
            "sampler_state": "sampler",
            "sampler1D": "sampler1D",
            "sampler2D": "sampler2D",
            "sampler3D": "sampler3D",
            "samplerCUBE": "samplerCube",
            "Sampler": "sampler",
            "SamplerState": "sampler",
            "SamplerComparisonState": "comparison_sampler",
        }
        self.type_map.update(self.minimum_precision_type_map())
        self.shadow_texture_type_map = dict(HLSL_SHADOW_TEXTURE_TYPES)
        self.function_map = dict(HLSL_FUNCTION_MAP)
        self.function_statement_sequences = dict(HLSL_FUNCTION_STATEMENT_SEQUENCES)
        self.interlocked_map = dict(HLSL_INTERLOCKED_FUNCTIONS)
        self.byte_address_interlocked_method_map = dict(
            HLSL_BYTE_ADDRESS_INTERLOCKED_METHODS
        )
        self.texture_method_map = dict(HLSL_TEXTURE_METHODS)
        self.feedback_method_map = dict(HLSL_FEEDBACK_METHODS)
        self.texture_gather_component_map = dict(HLSL_TEXTURE_GATHER_COMPONENTS)
        self.texture_gather_compare_component_map = dict(
            HLSL_TEXTURE_GATHER_COMPARE_COMPONENTS
        )
        self.legacy_texture_function_sampler_types = dict(
            HLSL_LEGACY_TEXTURE_FUNCTION_SAMPLER_TYPES
        )
        self.buffer_method_map = dict(HLSL_BUFFER_METHODS)
        self.semantic_map = dict(HLSL_SEMANTICS)
        if zero_base_vertex:
            for semantic in ("SV_VertexID", "VERTEX_ID"):
                self.semantic_map[semantic] = "gl_VertexID"
        self.semantic_map_upper = {
            str(key).upper(): value for key, value in self.semantic_map.items()
        }
        self.bitwise_op_map = dict(HLSL_BITWISE_OPERATORS)
        self.indentation = 0
        self.code = []
        self.shadow_texture_names = set()
        self.shadow_texture_declaration_ids = set()
        self.hlsl_program_constant_global_ids = set()
        self.global_variable_types = {}
        self.global_resource_array_dims = {}
        self.current_variable_types = {}
        self.current_resource_array_dims = {}
        self.current_identifier_renames = {}
        self.current_stage_struct_parameter_member_aliases = {}
        self.struct_member_types = {}
        self.struct_member_array_dims = {}
        self.structs_by_name = {}
        self.flattened_stage_struct_names = set()
        self.integer_constant_values = {}
        self.suppress_storage_image_index_lowering = False
        self.read_raw_texel_method_result = False
        self.storage_image_texel_temp_count = 0
        self.function_identifier_renames = {}
        self.global_identifier_renames = {}
        self.type_identifier_renames = {}
        self.legacy_sampler_bindings = {}
        self.legacy_sampler_declaration_ids = set()
        self.raw_gather_texture_declaration_ids = set()
        self.legacy_bound_texture_types = {}

    def texture_method_descriptor(self, member, arg_count=None, resource_type=None):
        def with_dropped_parameters(descriptor, parameters):
            if parameters:
                descriptor["drop_trailing_args"] = len(parameters)
                descriptor["dropped_parameters"] = list(parameters)
            return descriptor

        handled = self.texture_feedback_method_descriptor(member, resource_type)
        if handled is not UNHANDLED:
            return handled

        cube_family_resource = self.raw_type_base(resource_type) in {
            "TextureCube",
            "TextureCubeArray",
        }

        handled = self.texture_load_dimension_method_descriptor(
            member, arg_count, resource_type, with_dropped_parameters
        )
        if handled is not UNHANDLED:
            return handled
        handled = self.texture_query_method_descriptor(member, arg_count, resource_type)
        if handled is not UNHANDLED:
            return handled
        handled = self.texture_gather_method_descriptor(
            member,
            arg_count,
            resource_type,
            cube_family_resource,
            with_dropped_parameters,
        )
        if handled is not UNHANDLED:
            return handled
        handled = self.texture_sampling_method_descriptor(
            member, arg_count, cube_family_resource, with_dropped_parameters
        )
        if handled is not UNHANDLED:
            return handled
        return None

    def render_identifier(self, name):
        """Return the active CrossGL spelling for a user-declared identifier."""
        if not isinstance(name, str):
            return name
        if name in self.current_identifier_renames:
            return self.current_identifier_renames[name]
        rendered = _renamed(self.global_identifier_renames, name)
        if self.is_opaque_template_value_path(rendered):
            return self.sanitize_scoped_identifier_name(rendered)
        return rendered

    def collect_global_identifier_renames(self, ast):
        declared_names = {
            name
            for nodes in (
                getattr(ast, "structs", []) or [],
                getattr(ast, "global_variables", []) or [],
                getattr(ast, "functions", []) or [],
                getattr(ast, "typedefs", []) or [],
                getattr(ast, "enums", []) or [],
            )
            for node in nodes
            for name in [getattr(node, "name", None)]
            if isinstance(name, str) and name
        }
        used_names = {
            _renamed(self.function_identifier_renames, name) for name in declared_names
        }
        renames = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", None)
            if not isinstance(name, str) or "::" not in name:
                continue
            candidate = self.sanitize_scoped_identifier_name(name)
            if candidate in self.crossgl_reserved_identifiers:
                candidate = f"{candidate}_"
            while candidate in used_names or candidate in renames.values():
                candidate = f"{candidate}_"
            renames[name] = candidate
            used_names.add(candidate)
        return renames

    def render_function_identifier(self, name):
        if not isinstance(name, str):
            return name
        return _renamed(self.function_identifier_renames, name)

    def render_type_identifier(self, name):
        if not isinstance(name, str):
            return name
        return _renamed(self.type_identifier_renames, name)

    def format_subpass_input_attributes(self, attributes, indent):
        normalized_attributes = []
        for attr in attributes or []:
            attr_name = str(getattr(attr, "name", ""))
            args = getattr(attr, "args", getattr(attr, "arguments", []))
            if attr_name in {"vk::input_attachment_index", "vk::binding"}:
                attr_name = attr_name.rsplit("::", 1)[1]
                if attr_name == "binding" and len(args) >= 2:
                    normalized_attributes.append(AttributeNode("binding", [args[0]]))
                    normalized_attributes.append(AttributeNode("set", [args[1]]))
                    continue
            normalized_attributes.append(
                AttributeNode(
                    attr_name,
                    args,
                )
            )
        return self.format_attributes(normalized_attributes, indent)

    def function_call_value_raw_type(self, expr):
        if not isinstance(expr.name, MemberAccessNode):
            if isinstance(expr.name, str):
                deduced = template_call_result_raw_type(self, expr)
                if deduced is not None:
                    return deduced
                hit_object_type = self.hit_object_static_call_raw_type(expr)
                if hit_object_type is not None:
                    return hit_object_type
                raytracing_matrix = self.raytracing_matrix_call_raw_type(expr)
                if raytracing_matrix is not None:
                    return raytracing_matrix
                type_name = self.resolve_integer_alias(expr.name)
                if source_integer_shape(type_name) is not None:
                    return type_name
                if self.map_type(type_name) != self.sanitize_type_name(type_name):
                    return type_name
                candidates = self.source_functions.get(expr.name, [])
                candidates = [f for f in candidates if len(f.params) == len(expr.args)]
                argument_types = [
                    self.resolve_integer_alias(self.expression_value_raw_type(arg))
                    for arg in expr.args
                ]
                exact = [
                    f
                    for f in candidates
                    if [self.resolve_integer_alias(p.vtype) for p in f.params]
                    == argument_types
                ]
                returns = {
                    self.resolve_integer_alias(f.return_type)
                    for f in exact or candidates
                }
                if len(returns) == 1:
                    return returns.pop()
                return_type = self.function_return_types.get(expr.name)
                if return_type is None:
                    return_type = self.overload_return_raw_type(expr)
                if return_type is None and any(
                    (shape := source_integer_shape(t)) is not None and shape[1] == 64
                    for t in returns
                ):
                    raise HLSLIntegerConversionError("ambiguous-function-result")
                return return_type
            return None

        member = self.templated_method_base(expr.name.member)
        resource_type = self.expression_raw_type(expr.name.object)
        resource_base = self.raw_type_base(resource_type)
        method = self.struct_methods.get(resource_base, {}).get(member)
        if method is not None and method.definition is not None:
            return method.definition.return_type
        raytracing_matrix = self.raytracing_matrix_call_raw_type(expr)
        if raytracing_matrix is not None:
            return raytracing_matrix
        if member in {"Load", "Load2", "Load3", "Load4"}:
            if resource_base in {
                "ByteAddressBuffer",
                "RWByteAddressBuffer",
                "RasterizerOrderedByteAddressBuffer",
            }:
                return self.byte_address_load_raw_type(expr.name.member)
            return self.resource_element_raw_type(resource_type)
        if member == "Consume":
            return self.resource_element_raw_type(resource_type)
        return None

    def wide_integer_compound_expression(self, expr, is_main=False):
        operator = self.binary_operator_text(expr.operator)
        if operator not in {"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="}:
            return None
        left_type = self.expression_value_raw_type(expr.left)
        right_type = self.expression_value_raw_type(expr.right)
        common = self.wide_integer_common_type(left_type, right_type, operator[:-1])
        if common is None:
            return None
        left = self.generate_expression(expr.left, is_main)
        right = self.generate_expression(expr.right, is_main)
        shape = source_integer_shape(self.resolve_integer_alias(left_type))
        # A scalar destination can retain native compound assignment after the
        # RHS conversion. A vector destination needs computation before narrowing.
        common_shape = source_integer_shape(common)
        if (shape[2] == 1 and common_shape[2] == 1) or shape == common_shape:
            right = self.convert_integer_operand(right, right_type, common)
            return f"{left} {operator} {right}"
        if self.integer_expression_has_effects(expr.right):
            raise HLSLIntegerConversionError("compound-copy-in-alias")
        target = self.map_type(self.resolve_integer_alias(left_type))
        value = self.map_type(self.resolve_integer_alias(right_type))
        left_operand = self.convert_integer_operand("target", left_type, common)
        right_operand = self.convert_integer_operand("value", right_type, common)
        key = (target, value, operator[:-1], left_operand, right_operand)
        name = self.integer_compound_helpers.get(key)
        if name is None:
            name = "__crossgl_hlsl_integer_compound"
            while (
                name in self.integer_helper_reserved_names
                or name in self.integer_compound_helpers.values()
            ):
                name += "_"
            self.integer_compound_helpers[key] = name
        return f"{name}({left}, {right})"

    def visit(self, node):
        if isinstance(node, SwitchStatementNode):
            return self.visit_SwitchStatementNode(node)
        elif isinstance(node, SwitchCaseNode):
            return self.visit_SwitchCaseNode(node)
        elif isinstance(node, StructNode):
            return self.visit_StructNode(node)
        elif isinstance(node, BinaryOpNode):
            return self.visit_BinaryOpNode(node)
        elif isinstance(node, UnaryOpNode):
            return self.visit_UnaryOpNode(node)

        if hasattr(self, f"generate_{type(node).__name__}"):
            method = getattr(self, f"generate_{type(node).__name__}")
            return method(node)
        return self.generate_expression(node)

    def generate(self, ast):
        """Render ``ast`` as a CrossGL program.

        Raises :class:`DirectXImportRefused` listing every HLSL construct CrossGL cannot represent faithfully.
        """
        return self.refusals.run(self.render_program, ast)

    def render_program(self, ast):
        resolve_namespaces(ast, self.refuse_import)
        structs = [
            struct
            for struct in getattr(ast, "structs", []) or []
            if isinstance(struct, StructNode)
        ]
        self.unsupported_struct_templates = unsupported_struct_templates(structs)
        refuse_struct_template_references(self, ast, self.unsupported_struct_templates)
        self.struct_templates = {
            struct.name: struct_template_parameters(struct)
            for struct in structs
            if struct_template_parameters(struct)
            and struct.name not in self.unsupported_struct_templates
        }
        self.explicit_struct_specializations = name_explicit_struct_specializations(
            self, structs
        )
        self.struct_methods = collect_struct_methods(
            getattr(ast, "structs", []) or [], ast.functions
        )
        inline_methods = self.struct_method_functions(self.struct_methods)
        if inline_methods:
            ast.functions = list(ast.functions) + inline_methods
        self.current_struct_method = None
        self.current_struct_method_locals = set()
        self.source_function_names = {function.name for function in ast.functions}
        return_types = {}
        self.function_overloads = {}
        for function in ast.functions:
            return_types.setdefault(function.name, set()).add(
                str(getattr(function, "return_type", "") or "")
            )
            self.function_overloads.setdefault(function.name, []).append(
                (
                    [
                        getattr(param, "vtype", None)
                        for param in getattr(function, "params", []) or []
                    ],
                    str(getattr(function, "return_type", "") or ""),
                )
            )
        self.function_return_types = {
            name: next(iter(types))
            for name, types in return_types.items()
            if len(types) == 1 and next(iter(types))
        }
        self.source_functions = {}
        for function in ast.functions:
            self.source_functions.setdefault(function.name, []).append(function)
        self.source_type_aliases = {
            alias.name: getattr(alias, "alias_type", None) or alias.original_type
            for alias in getattr(ast, "typedefs", []) or []
        }
        self.integer_compound_helpers = {}
        self.integer_helper_reserved_names = set()
        self.reset_matrix_import_state()
        pending = [ast]
        while pending:
            node = pending.pop()
            name = getattr(node, "name", None)
            if isinstance(name, str):
                self.integer_helper_reserved_names.add(name)
            pending.extend(self.iter_ast_children(node))
        self.struct_member_types = self.collect_struct_member_types(ast.structs)
        self.structs_by_name = {
            struct.name: struct
            for struct in ast.structs
            if isinstance(struct, StructNode) and getattr(struct, "name", None)
        }
        analyze_receiver_mutations(
            self.struct_methods,
            {
                owner: set(members) - self.struct_static_member_names(owner)
                for owner, members in self.struct_member_types.items()
            },
            self.iter_ast_children,
            lambda call, index: self.call_writes_argument(call, index, ast.functions),
        )
        self.struct_member_array_dims = self.collect_struct_member_array_dims(
            ast.structs
        )
        (
            self.legacy_sampler_bindings,
            self.legacy_sampler_declaration_ids,
        ) = self.collect_legacy_sampler_bindings(ast)
        self.legacy_bound_texture_types = self.collect_legacy_bound_texture_types(ast)
        self.shadow_texture_names = self.collect_shadow_texture_names(ast)
        self.raw_gather_texture_declaration_ids = (
            collect_raw_gather_texture_declaration_ids(self, ast)
        )
        self.hlsl_program_constant_global_ids = (
            self.collect_hlsl_program_constant_global_ids(ast)
        )
        self.struct_type_names = {
            node.name
            for node in getattr(ast, "structs", []) or []
            if isinstance(node, StructNode)
            and not getattr(node, "is_forward_declaration", False)
            and getattr(node, "name", None)
        }
        self.structs_by_name = {
            struct.name: struct
            for struct in ast.structs
            if isinstance(struct, StructNode) and getattr(struct, "name", None)
        }
        self.global_variable_types = {}
        self.global_resource_array_dims = {}
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            for member in getattr(cbuffer, "members", []) or []:
                if isinstance(member, VariableNode) and isinstance(member.name, str):
                    self.record_variable_type(
                        member,
                        self.global_variable_types,
                        self.global_resource_array_dims,
                    )
        self.current_variable_types = {}
        self.current_resource_array_dims = {}
        self.integer_constant_values = self.collect_integer_constant_values(ast)
        self.revolution_reduction_types = set()
        self.warp_trigonometric_uses = set()
        self.reset_precise_arithmetic()
        self.collect_unorm_store_textures(ast)
        self.constant_initializer_depth = 0
        self.hlsl_static_const_initializers = self.collect_static_const_initializers(
            ast
        )
        self.function_identifier_renames = self.collect_function_identifier_renames(
            ast.functions
        )
        self.specialization_identifiers = specialization_identifier_renames(
            self,
            ast.functions,
            set(self.source_function_names)
            | set(self.function_identifier_renames.values()),
        )
        self.type_identifier_renames = self.collect_type_identifier_renames(ast)
        self.global_identifier_renames = self.collect_global_identifier_renames(ast)
        self.flattened_stage_struct_names = self.collect_flattened_stage_struct_names(
            ast
        )
        self.hlsl_member_offsets = push_constant_member_offsets(
            self, ast, self.structs_by_name
        )
        code = "shader main {\n"
        typedefs = getattr(ast, "typedefs", []) or []
        enums = getattr(ast, "enums", []) or []
        for alias in typedefs:
            declaration = self.generate_type_alias_declaration(alias)
            if declaration:
                code += f"    {declaration}\n"
        if enums:
            for enum in enums:
                if isinstance(enum, EnumNode):
                    code += self.generate_enum(enum)
        # Generate structs
        for node in ast.structs:
            if getattr(node, "is_forward_declaration", False):
                continue
            if getattr(node, "name", None) in self.flattened_stage_struct_names:
                continue
            if getattr(node, "name", None) in self.unsupported_struct_templates:
                continue
            if isinstance(node, StructNode):
                prefix = struct_template_prefix(self, node)
                code += (
                    f"    {prefix}struct {self.render_type_identifier(node.name)} {{\n"
                )
                previous_template_type_names = getattr(
                    self, "current_template_type_names", ()
                )
                self.current_template_type_names = struct_template_type_names(node)
                for member in node.members:
                    if isinstance(member, EnumNode):
                        code += self.generate_enum(member, 2)
                        continue
                    array_suffix = self.format_array_suffixes(member)
                    attributes = self.format_semantic_and_interpolation_attributes(
                        member, member.semantic
                    )
                    member_offset = self.hlsl_member_offsets.get(
                        (node.name, str(member.name))
                    )
                    if member_offset is not None:
                        attributes += f" @offset({member_offset})"
                    qualifier_prefix = self.format_precise_qualifier_prefix(member)
                    code += self.format_matrix_layout_attributes(member, 2)
                    code += (
                        f"        {qualifier_prefix}{self.map_variable_type(member)} "
                        f"{self.render_identifier(member.name)}"
                        f"{array_suffix}{attributes};\n"
                    )
                self.current_template_type_names = previous_template_type_names
                code += "    }\n"
            elif isinstance(node, PragmaNode):
                value = f" {node.value}" if node.value else ""
                code += f"    #pragma {node.directive}{value}\n"
            elif isinstance(node, IncludeNode):
                code += f"    #include {node.path}\n"
        for node in ast.global_variables:
            code += self.generate_global_variable_declaration(node)
        for node in self.collect_struct_variable_declarations(ast.structs):
            code += self.generate_global_variable_declaration(node)
        if ast.cbuffers:
            code += "    // Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        stage_map = {
            "vertex": "vertex",
            "fragment": "fragment",
            "compute": "compute",
            "geometry": "geometry",
            "tessellation_control": "tessellation_control",
            "tessellation_evaluation": "tessellation_evaluation",
            "mesh": "mesh",
            "task": "task",
            "ray_generation": "ray_generation",
            "ray_intersection": "ray_intersection",
            "ray_closest_hit": "ray_closest_hit",
            "ray_miss": "ray_miss",
            "ray_any_hit": "ray_any_hit",
            "ray_callable": "ray_callable",
        }
        functions_by_name = {
            func.name: func
            for func in ast.functions
            if not getattr(func, "is_prototype", False)
        }
        function_section_start = len(code)
        for func in ast.functions:
            if getattr(func, "is_prototype", False):
                # CrossGL functions are visible throughout the module; targets that
                # need declarations before use emit their own prototypes.
                continue
            stage_name = stage_map.get(func.qualifier)
            if stage_name:
                code += f"    // {stage_name} Shader\n"
                code += f"    {stage_name} {{\n"
                if stage_name == "tessellation_control":
                    code += self.generate_patch_constant_interface_outputs(
                        func, functions_by_name, self.structs_by_name
                    )
                entry_name = "main" if stage_name.startswith("ray_") else None
                code += self.generate_function(
                    func,
                    skip_attribute_names={"shader"},
                    entry_name=entry_name,
                    stage_entry=self.should_emit_stage_entry_attribute(
                        func, stage_name, entry_name
                    ),
                )
                code += "    }\n\n"
            else:
                code += self.generate_function(func)

        code += self.generate_integer_compound_helpers()
        code += self.generate_matrix_row_setters()
        reduction_helpers = (
            self.generate_trigonometric_helpers()
            + self.generate_precise_arithmetic_helpers()
            + self.generate_unorm_store_declarations()
        )
        if reduction_helpers:
            code = (
                code[:function_section_start]
                + reduction_helpers
                + code[function_section_start:]
            )
        code += "}\n"
        return code

    def generate_cbuffers(self, ast):
        code = ""
        used_names = self.collect_cbuffer_reserved_names(ast)
        for node in ast.cbuffers:
            if isinstance(node, StructNode):
                if is_hlsl_shader_record(node) is not None:
                    shader_record_layout(node, self.structs_by_name, is_cbuffer=True)
                cbuffer_name = self.unique_cbuffer_name(node.name, used_names)
                used_names.add(cbuffer_name)
                attributes = list(getattr(node, "attributes", []) or [])
                if getattr(node, "is_tbuffer", False):
                    attributes = [AttributeNode("tbuffer")] + attributes
                code += self.format_attributes(attributes, 1)
                code += self.format_binding_attributes(node, 1)
                code += f"    @ {BUFFER_LAYOUT_ATTRIBUTE}({HLSL_CBUFFER_LAYOUT})\n"
                code += f"    cbuffer {cbuffer_name} {{\n"
                for member in node.members:
                    self.record_variable_type(
                        member,
                        self.global_variable_types,
                        self.global_resource_array_dims,
                    )
                    code += self.format_matrix_layout_attributes(member, 2)
                    code += self.format_attributes(getattr(member, "attributes", []), 2)
                    code += self.format_binding_attributes(member, 2)
                    array_suffix = self.format_array_suffixes(member)
                    qualifier_prefix = self.format_precise_qualifier_prefix(member)
                    initializer = ""
                    if getattr(member, "value", None) is not None:
                        initializer_value = self.generate_expression(
                            member.value, expected_type=member.vtype
                        )
                        initializer = f" = {initializer_value}"
                    code += (
                        f"        {qualifier_prefix}{self.map_variable_type(member)} "
                        f"{self.render_identifier(member.name)}{array_suffix}"
                        f"{initializer};\n"
                    )
                code += "    }\n"
        return code

    def generate_expression(self, expr, is_main=False, expected_type=None):
        """Render a DirectX backend expression node as CrossGL syntax."""
        if expected_type is not None:
            splat = self.scalar_matrix_splat_expression(
                self.hlsl_matrix_shape(expected_type), expr, is_main
            )
            if splat is not None:
                return splat
        if isinstance(expr, str):
            receiver_member = self.struct_method_context_identifier(expr)
            if receiver_member is not None:
                return receiver_member
            return self.render_identifier(expr)
        elif isinstance(expr, VariableNode):
            return self.render_identifier(expr.name)
        elif isinstance(expr, BinaryOpNode):
            comma = crossgl_comma_expression(
                expr, lambda operand: self.generate_expression(operand, is_main)
            )
            if comma is not None:
                return comma
            flat_expression = self.generate_flat_long_binary_expression(expr, is_main)
            if flat_expression is not None:
                return flat_expression
            converted = self.wide_integer_binary_expression(expr, is_main)
            if converted is not None:
                return converted
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            left = self.maybe_parenthesize(expr.left, left)
            right = self.maybe_parenthesize(expr.right, right)
            matrix_product = self.matrix_product_expression(expr, left, right)
            if matrix_product is not None:
                return matrix_product
            return f"{left} {expr.op} {right}"

        elif isinstance(expr, AssignmentNode):
            converted = self.wide_integer_compound_expression(expr, is_main)
            if converted is not None:
                return converted
            matrix_assignment = self.matrix_assignment_expression(expr, is_main)
            if matrix_assignment is not None:
                return matrix_assignment
            left = self.generate_expression(expr.left, is_main)
            right = self.scalar_matrix_assignment_value(expr, is_main)
            if right is None:
                right = self.generate_expression(expr.right, is_main)
                right = self.truncated_assignment_value(expr, right)
            op = expr.operator
            return f"{left} {op} {right}"

        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            operand = self.maybe_parenthesize(expr.operand, operand)
            if getattr(expr, "is_postfix", False):
                return f"{operand}{expr.op}"
            if str(expr.op)[-1:] in {"+", "-"} and operand[:1] in {"+", "-"}:
                # '-(-x)' written as '--x' would read as a decrement, and '+(+x)' as an increment.
                return f"{expr.op}({operand})"
            return f"{expr.op}{operand}"
        elif isinstance(expr, FunctionCallNode):
            return self.generate_function_call_expression(expr, is_main)
        elif isinstance(expr, MemberAccessNode):
            return self.generate_member_access_expression(expr, is_main, expected_type)
        elif isinstance(expr, ArrayAccessNode):
            texel_access = self.generate_texel_access(expr, is_main)
            if texel_access is not None:
                return texel_access
            matrix_index = self.matrix_index_expression(expr, is_main)
            if matrix_index is not None:
                return matrix_index
            array = self.generate_expression(expr.array, is_main)
            if isinstance(
                expr.array, (BinaryOpNode, TernaryOpNode, UnaryOpNode, AssignmentNode)
            ):
                # A subscript binds tighter than any operator.
                array = f"({array})"
            index = self.generate_expression(expr.index, is_main)
            return f"{array}[{index}]"
        elif isinstance(expr, CastNode):
            matrix_shape = self.hlsl_matrix_shape(expr.target_type)
            if matrix_shape is not None:
                return self.matrix_conversion_expression(
                    matrix_shape, expr.expression, is_main
                )
            if (
                expr.expression == 0
                and not isinstance(expr.expression, bool)
                and self.raw_type_base(expr.target_type)
                in getattr(self, "structs_by_name", {})
            ):
                zero_value = self.zero_value_expression(expr.target_type)
                if zero_value is not None:
                    return zero_value
            target_type = self.map_type(expr.target_type)
            expression = self.generate_expression(expr.expression, is_main)
            return f"{target_type}({expression})"
        elif isinstance(expr, TextureSampleNode):
            # Sample returns the texture's element type; CrossGL returns four.
            narrowing = (
                ""
                if self.read_raw_texel_method_result
                else self.texel_narrowing_suffix(self.expression_raw_type(expr.texture))
            )
            self.read_raw_texel_method_result = False
            texture = self.generate_expression(expr.texture, is_main)
            sampler = self.generate_expression(expr.sampler, is_main)
            coords = self.generate_expression(expr.coordinates, is_main)
            if getattr(expr, "lod", None) is not None:
                lod = self.generate_expression(expr.lod, is_main)
                return f"textureLod({texture}, {sampler}, {coords}, {lod}){narrowing}"
            return f"texture({texture}, {sampler}, {coords}){narrowing}"

        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition, is_main)
            left = self.generate_expression(expr.true_expr, is_main)
            right = self.generate_expression(expr.false_expr, is_main)
            left_type = self.expression_value_raw_type(expr.true_expr)
            right_type = self.expression_value_raw_type(expr.false_expr)
            common = self.wide_integer_common_type(left_type, right_type, "?:")
            if common is not None:
                left = self.convert_integer_operand(left, left_type, common)
                right = self.convert_integer_operand(right, right_type, common)
            return f"{condition} ? {left} : {right}"

        elif isinstance(expr, VectorConstructorNode):
            matrix_constructor = self.matrix_constructor_expression(
                expr.type_name, expr.args, is_main
            )
            if matrix_constructor is not None:
                return matrix_constructor
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{self.map_type(expr.type_name)}({args})"
        elif isinstance(expr, InitializerListNode):
            matrix_initializer = self.matrix_initializer_expression(
                expected_type, expr, is_main
            )
            if matrix_initializer is not None:
                return matrix_initializer
            elements = ", ".join(
                self.generate_expression(element, is_main) for element in expr.elements
            )
            return f"{{{elements}}}"
        elif isinstance(expr, bool):
            return "true" if expr else "false"
        elif isinstance(expr, float):
            return self.format_float(expr)
        elif isinstance(expr, int):
            if isinstance(expr, IntegerLiteral):
                if "l" in expr.suffix:
                    return f"{self.map_type(expr.source_type)}({int(expr)})"
                return f"{int(expr)}u" if "u" in expr.suffix else str(int(expr))
            return str(expr)
        else:
            return str(expr)

    def sanitize_type_name(self, type_name):
        """Convert HLSL scoped type paths into CrossGL identifier-safe names."""
        raw = str(type_name)
        if raw in self.type_identifier_renames:
            return self.type_identifier_renames[raw]
        if raw in _SCOPED_BUILTIN_TYPE_NAMES:
            # CrossGL spells these Shader Model types with their namespace, and the
            # DirectX generator writes them back unchanged.
            return raw
        sanitized = raw.replace("::", "_")
        if sanitized in self.type_identifier_renames:
            return self.type_identifier_renames[sanitized]
        return sanitized

    def visit_StructNode(self, node):
        if getattr(node, "is_forward_declaration", False):
            return ""
        code = f"struct {self.render_type_identifier(node.name)} {{\n"
        self.indentation += 1

        for member in node.members:
            attributes = self.format_semantic_and_interpolation_attributes(
                member, member.semantic
            )

            array_suffix = self.format_array_suffixes(member)
            code += (
                self.get_indent() + f"{self.map_variable_type(member)} "
                f"{self.render_identifier(member.name)}"
                f"{array_suffix}{attributes};\n"
            )

        self.indentation -= 1
        code += self.get_indent() + "}\n"
        return code
