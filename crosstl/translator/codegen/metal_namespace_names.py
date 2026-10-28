"""Program-scope names that ``using namespace metal;`` makes ambiguous.

Metal output writes ``using namespace metal;``, which brings every name the Metal standard library declares in
namespace ``metal`` into the program's scope, and ``<metal_stdlib>`` also declares C and SIMD names at global scope.
Beside the library functions, which :mod:`.metal_library_functions` handles, these are the types, enumerations and
type aliases: the sampler options (``filter``, ``coord``, ``address``, ``mip_filter``), the texture sampling
arguments (``bias``, ``level``, ``gradient2d``, ``min_lod_clamp``), ``access``, ``quad``, ``matrix``, ``array``,
``vec``, the vector and packed types, ``int32_t`` and the type traits. GLSL and HLSL reserve none of them, so a
shader may name a function, a constant or a structure ``bias``, ``level``, ``filter`` or ``quad``. Its Metal
translation then fails to compile: ``reference to 'bias' is ambiguous``, or ``redefinition of 'quad' as different
kind of symbol``.

:func:`rename_metal_namespace_names` renames each such user function, with every call one of its overloads accepts,
each such program-scope constant with every reference to it, and each such structure with every use of its type and
every constructor call. A function parameter, a local variable or a structure member keeps its name: there it hides
the library name, which C++ allows. A structure the Metal library declares itself (:mod:`.metal_library_structs`) is
the library's and keeps its name.

``METAL_NAMESPACE_NAMES`` lists every name the ``<metal_stdlib>`` headers declare as a type, enumeration or alias
(not starting with ``_``) that fails to compile as the name of a user function under ``using namespace metal;``,
checked with the macOS Metal compiler (metalfe-32023.918).
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from ..semantics.function_name_collisions import rename_functions, user_function_names
from ..semantics.hub_types import call_name
from .metal_library_structs import is_metal_library_struct

__all__ = ["METAL_NAMESPACE_NAMES", "rename_metal_namespace_names"]

METAL_NAMESPACE_NAMES = frozenset(
    (
        "access add_const add_const_t add_pointer add_pointer_t address array array_ref atomic atomic_bool "
        "atomic_float atomic_int atomic_uint atomic_ulong bfloat16 bfloat2 bfloat3 bfloat4 bfloat8 bias "
        "block_color bool_constant bool16 bool2 bool3 bool4 bool8 border_color char16 char2 char3 char4 char8 "
        "command_buffer common_type common_type_t compare_func complex component compute_command "
        "compute_pipeline_state conditional conditional_t conjunction cooperative_tensor coord cull_mode "
        "decay depth_clip_mode depth_stencil_state depth2d depth2d_array depth2d_ms depth2d_ms_array "
        "depthcube depthcube_array dextents disjunction divides double16 double2 double3 double4 double8 "
        "enable_if enable_if_t extents false_type filter float_denorm_style float_round_style float16 float2 "
        "float2x2 float2x3 float2x4 float3 float3x2 float3x3 float3x4 float4 float4x2 float4x3 float4x4 "
        "float8 function_handle gradient2d gradient3d gradientcube half16 half2 half2x2 half2x3 half2x4 half3 "
        "half3x2 half3x3 half3x4 half4 half4x2 half4x3 half4x4 half8 has_tensor_offset imageblock "
        "imageblock_data_rate imageblock_layout imageblock_layout_explicit imageblock_layout_implicit "
        "imageblock_layout_t imageblock_slice imaginary index_sequence initializer_list int16 int16_t int2 "
        "int2b_format int3 int32_t int4 int4b_format int64_t int8 int8_t integer_sequence integral_constant "
        "interpolant intptr_t is_abstract is_arithmetic is_array is_class is_color_texture_atomic_channel "
        "is_color_texture_channel is_color_texture_imageblock_slice_storage_valid is_const is_constructible "
        "is_convertible is_cooperative_tensor is_default_constructible is_depth_texture_channel is_enum "
        "is_explicit_layout_imageblock_slice_storage is_explicit_layout_imageblock_struct is_floating_point "
        "is_function is_implicit_layout_imageblock_slice_storage is_implicit_layout_imageblock_struct "
        "is_integral is_lvalue_reference is_metal_buffer is_metal_buffer_pointee is_nothrow_constructible "
        "is_numeric_format is_pixel_data is_pointer is_same is_scalar is_signed is_simdgroup_matrix_element "
        "is_tensor_handle is_tensor_inline is_uniform is_unsigned is_vector is_volatile level llong llong16 "
        "llong2 llong3 llong4 llong8 lod_clamp log_type long16 long2 long3 long4 long8 mag_filter "
        "make_index_sequence make_integer_sequence make_scalar make_scalar_t make_signed make_signed_t "
        "make_unsigned make_unsigned_t matrix max_anisotropy mem_flags memory_coherence memory_order "
        "memory_scope mesh mesh_grid_properties metal_fp4_e2m1_format metal_fp8_e4m3_format "
        "metal_fp8_e5m2_format metal_fp8_ue8m0_format min_filter min_lod_clamp minus mip_filter "
        "MTLAccelerationStructureInstanceDescriptor MTLAccelerationStructureInstanceDescriptorType "
        "MTLAccelerationStructureInstanceOptions MTLAccelerationStructureMotionInstanceDescriptor "
        "MTLAccelerationStructureUserIDInstanceDescriptor MTLDispatchThreadgroupsIndirectArguments "
        "MTLDispatchThreadsIndirectArguments MTLDrawIndexedPrimitivesIndirectArguments "
        "MTLDrawPatchIndirectArguments MTLDrawPrimitivesIndirectArguments "
        "MTLIndirectAccelerationStructureInstanceDescriptor "
        "MTLIndirectAccelerationStructureMotionInstanceDescriptor MTLIndirectCommandBufferExecutionRange "
        "MTLIntersectionFunctionBufferArguments MTLMotionBorderMode MTLPackedFloat3 "
        "MTLQuadTessellationFactorsHalf MTLResourceID MTLTriangleTessellationFactorsHalf multiplies nullptr_t "
        "numeric_limits os_log packed_bfloat16 packed_bfloat2 packed_bfloat3 packed_bfloat4 packed_bfloat8 "
        "packed_bool16 packed_bool2 packed_bool3 packed_bool4 packed_bool8 packed_char16 packed_char2 "
        "packed_char3 packed_char4 packed_char8 packed_double16 packed_double2 packed_double3 packed_double4 "
        "packed_double8 packed_float16 packed_float2 packed_float3 packed_float4 packed_float8 packed_half16 "
        "packed_half2 packed_half3 packed_half4 packed_half8 packed_int16 packed_int2 packed_int2b "
        "packed_int3 packed_int4 packed_int4b packed_int8 packed_int8b packed_llong16 packed_llong2 "
        "packed_llong3 packed_llong4 packed_llong8 packed_long16 packed_long2 packed_long3 packed_long4 "
        "packed_long8 packed_metal_fp4_e2m1 packed_metal_fp8_e4m3 packed_metal_fp8_e5m2 "
        "packed_metal_fp8_ue8m0 packed_numeric_type packed_quad16 packed_quad2 packed_quad3 packed_quad4 "
        "packed_quad8 packed_short16 packed_short2 packed_short3 packed_short4 packed_short8 packed_uchar16 "
        "packed_uchar2 packed_uchar3 packed_uchar4 packed_uchar8 packed_uint16 packed_uint2 packed_uint2b "
        "packed_uint3 packed_uint4 packed_uint4b packed_uint8 packed_uint8b packed_ullong16 packed_ullong2 "
        "packed_ullong3 packed_ullong4 packed_ullong8 packed_ulong16 packed_ulong2 packed_ulong3 "
        "packed_ulong4 packed_ulong8 packed_ushort16 packed_ushort2 packed_ushort3 packed_ushort4 "
        "packed_ushort8 packed_vec patch_control_point plus primitive_type ptrdiff_t quad quad_vote quad16 "
        "quad2 quad3 quad4 quad8 r_address r16snorm r16unorm r8snorm r8unorm rasterization_rate_map_data "
        "rasterization_rate_map_decoder reduction remove_addrspace remove_addrspace_t remove_coherent "
        "remove_coherent_t remove_const remove_const_t remove_cv remove_cv_t remove_extent remove_extent_t "
        "remove_pointer remove_pointer_t remove_reference remove_reference_t remove_uniform remove_uniform_t "
        "remove_volatile remove_volatile_t render_command render_pipeline_state rg11b10f rg16snorm rg16unorm "
        "rg8snorm rg8unorm rgb10a2 rgb9e5 rgba16snorm rgba16unorm rgba8snorm rgba8unorm rounding_mode "
        "s_address sampler saturation_mode short16 short2 short3 short4 short8 simd_bfloat16 simd_bfloat2 "
        "simd_bfloat3 simd_bfloat4 simd_bfloat8 simd_bool16 simd_bool2 simd_bool3 simd_bool4 simd_bool8 "
        "simd_char16 simd_char2 simd_char3 simd_char4 simd_char8 simd_double16 simd_double2 simd_double3 "
        "simd_double4 simd_double8 simd_float16 simd_float2 simd_float3 simd_float4 simd_float8 simd_half16 "
        "simd_half2 simd_half3 simd_half4 simd_half8 simd_int16 simd_int2 simd_int3 simd_int4 simd_int8 "
        "simd_long16 simd_long2 simd_long3 simd_long4 simd_long8 simd_packed_bfloat16 simd_packed_bfloat2 "
        "simd_packed_bfloat3 simd_packed_bfloat4 simd_packed_bfloat8 simd_packed_bool16 simd_packed_bool2 "
        "simd_packed_bool3 simd_packed_bool4 simd_packed_bool8 simd_packed_char16 simd_packed_char2 "
        "simd_packed_char3 simd_packed_char4 simd_packed_char8 simd_packed_double16 simd_packed_double2 "
        "simd_packed_double3 simd_packed_double4 simd_packed_double8 simd_packed_float16 simd_packed_float2 "
        "simd_packed_float3 simd_packed_float4 simd_packed_float8 simd_packed_half16 simd_packed_half2 "
        "simd_packed_half3 simd_packed_half4 simd_packed_half8 simd_packed_int16 simd_packed_int2 "
        "simd_packed_int3 simd_packed_int4 simd_packed_int8 simd_packed_long16 simd_packed_long2 "
        "simd_packed_long3 simd_packed_long4 simd_packed_long8 simd_packed_quad16 simd_packed_quad2 "
        "simd_packed_quad3 simd_packed_quad4 simd_packed_quad8 simd_packed_short16 simd_packed_short2 "
        "simd_packed_short3 simd_packed_short4 simd_packed_short8 simd_packed_uchar16 simd_packed_uchar2 "
        "simd_packed_uchar3 simd_packed_uchar4 simd_packed_uchar8 simd_packed_uint16 simd_packed_uint2 "
        "simd_packed_uint3 simd_packed_uint4 simd_packed_uint8 simd_packed_ulong16 simd_packed_ulong2 "
        "simd_packed_ulong3 simd_packed_ulong4 simd_packed_ulong8 simd_packed_ushort16 simd_packed_ushort2 "
        "simd_packed_ushort3 simd_packed_ushort4 simd_packed_ushort8 simd_quad16 simd_quad2 simd_quad3 "
        "simd_quad4 simd_quad8 simd_short16 simd_short2 simd_short3 simd_short4 simd_short8 simd_uchar16 "
        "simd_uchar2 simd_uchar3 simd_uchar4 simd_uchar8 simd_uint16 simd_uint2 simd_uint3 simd_uint4 "
        "simd_uint8 simd_ulong16 simd_ulong2 simd_ulong3 simd_ulong4 simd_ulong8 simd_ushort16 simd_ushort2 "
        "simd_ushort3 simd_ushort4 simd_ushort8 simd_vote simdgroup_bfloat8x8 simdgroup_float8x8 "
        "simdgroup_half8x8 simdgroup_matrix size_t sparse_block_color sparse_color srgba8unorm t_address "
        "tensor tensor_blockwise tensor_handle tensor_inline tensor_offset tensor_plane_data "
        "tensor_plane_scales texture_buffer texture1d texture1d_array texture2d texture2d_array texture2d_ms "
        "texture2d_ms_array texture3d texturecube texturecube_array thread_scope topology triangle_fill_mode "
        "true_type uchar uchar16 uchar2 uchar3 uchar4 uchar8 uint16 uint16_t uint2 uint2b_format uint3 "
        "uint32_t uint4 uint4b_format uint64_t uint8 uint8_t uintptr_t ullong ullong16 ullong2 ullong3 "
        "ullong4 ullong8 ulong ulong16 ulong2 ulong3 ulong4 ulong8 underlying_type underlying_type_t uniform "
        "ushort ushort16 ushort2 ushort3 ushort4 ushort8 vec vec_elements vector_bool16 vector_bool2 "
        "vector_bool3 vector_bool4 vector_bool8 vector_char16 vector_char2 vector_char3 vector_char4 "
        "vector_char8 vector_double16 vector_double2 vector_double3 vector_double4 vector_double8 "
        "vector_float16 vector_float2 vector_float3 vector_float4 vector_float8 vector_half16 vector_half2 "
        "vector_half3 vector_half4 vector_half8 vector_int16 vector_int2 vector_int3 vector_int4 vector_int8 "
        "vector_long16 vector_long2 vector_long3 vector_long4 vector_long8 vector_quad16 vector_quad2 "
        "vector_quad3 vector_quad4 vector_quad8 vector_short16 vector_short2 vector_short3 vector_short4 "
        "vector_short8 vector_uchar16 vector_uchar2 vector_uchar3 vector_uchar4 vector_uchar8 vector_uint16 "
        "vector_uint2 vector_uint3 vector_uint4 vector_uint8 vector_ulong16 vector_ulong2 vector_ulong3 "
        "vector_ulong4 vector_ulong8 vector_ushort16 vector_ushort2 vector_ushort3 vector_ushort4 "
        "vector_ushort8 vertex_index vertex_value visible_function_table winding "
    ).split()
)
_RESOURCE_QUALIFIERS = frozenset(
    {"uniform", "buffer", "in", "out", "inout", "shared", "groupshared", "threadgroup"}
)


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


_BACK_LINKS = frozenset({"parent", "annotations", "source_location"})


def _children(node):
    """Every node ``node`` holds, types included, which :func:`~..semantics.rewrite.child_slots` skips."""
    for field, value in list(vars(node).items()):
        if field in _BACK_LINKS:
            continue
        if isinstance(value, A.ASTNode):
            yield value
        elif isinstance(value, dict):
            for item in list(value.values()):
                yield from (item if isinstance(item, list) else [item])
        elif isinstance(value, (list, tuple)):
            yield from value


def _walk(node, seen):
    if not isinstance(node, A.ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for child in list(_children(node)):
        yield from _walk(child, seen)


def _all_names(program):
    return {
        node.name
        for node in _walk(program, set())
        if isinstance(getattr(node, "name", None), str)
    }


def _qualifiers(variable):
    return {str(q).lower() for q in getattr(variable, "qualifiers", None) or []}


def _program_constants(program):
    """The program-scope values Metal declares at program scope: constants and qualifier-free globals."""
    values = list(getattr(program, "constants", None) or [])
    for variable in getattr(program, "global_variables", None) or []:
        if isinstance(variable, A.VariableNode) and not (
            _qualifiers(variable) & _RESOURCE_QUALIFIERS
        ):
            if getattr(
                variable, "initial_value", None
            ) is not None or "const" in _qualifiers(variable):
                values.append(variable)
    return values


def _functions(program):
    functions = list(getattr(program, "functions", None) or [])
    for stage in _stages(program):
        functions += list(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            functions.append(stage.entry_point)
    return functions


def _local_names(function):
    names = {
        getattr(p, "name", None) for p in getattr(function, "parameters", None) or []
    }
    for node in _walk(getattr(function, "body", None), set()):
        if isinstance(node, A.VariableNode):
            names.add(node.name)
    return names


def _fresh(name, taken):
    candidate = f"{name}_"
    while candidate in taken:
        candidate = f"{candidate}_"
    taken.add(candidate)
    return candidate


def _rename_values(program, renames):
    for value in _program_constants(program):
        if value.name in renames:
            value.name = renames[value.name]
    seen = set()

    def visit(node, hidden):
        if not isinstance(node, A.ASTNode) or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, A.FunctionNode):
            # A parameter or local of the same name hides the constant in its function.
            hidden = _local_names(node)
        if (
            isinstance(node, A.IdentifierNode)
            and node.name in renames
            and node.name not in hidden
        ):
            _set_identifier(node, renames[node.name])
        for child in list(_children(node)):
            visit(child, hidden)

    visit(program, frozenset())


def _rename_structs(program, renames):
    user_functions = user_function_names(program)
    for struct in getattr(program, "structs", None) or []:
        if struct.name in renames:
            struct.name = renames[struct.name]
    for node in _walk(program, set()):
        if isinstance(node, (A.NamedType, A.PrimitiveType)) and node.name in renames:
            node.name = renames[node.name]
        elif isinstance(node, A.FunctionCallNode):
            name = call_name(node)
            if name in renames and name not in user_functions:
                _rename_callee(node, name, renames[name])


def _rename_callee(node, old_name, new_name):
    for field in ("function", "name"):
        callee = getattr(node, field, None)
        if isinstance(callee, A.IdentifierNode) and callee.name == old_name:
            _set_identifier(callee, new_name)
        elif callee == old_name:
            setattr(node, field, new_name)


def _set_identifier(node, name):
    node.name = name
    node.identifier = name


def rename_metal_namespace_names(ast):
    """``ast``, or a copy whose functions, constants and structures named like ``metal`` names are renamed."""
    if not isinstance(ast, A.ShaderNode):
        return ast
    functions = user_function_names(ast) & METAL_NAMESPACE_NAMES
    values = {value.name for value in _program_constants(ast)} & METAL_NAMESPACE_NAMES
    structs = {
        struct.name
        for struct in getattr(ast, "structs", None) or []
        if struct.name in METAL_NAMESPACE_NAMES and not is_metal_library_struct(struct)
    }
    if not (functions or values or structs):
        return ast
    ast = deepcopy(ast)
    if functions:
        rename_functions(ast, functions, by_arity=True)
    taken = _all_names(ast)
    if values:
        _rename_values(ast, {name: _fresh(name, taken) for name in sorted(values)})
    if structs:
        _rename_structs(ast, {name: _fresh(name, taken) for name in sorted(structs)})
    return ast
