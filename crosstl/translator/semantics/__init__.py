"""Rule carrying: carry the source language's operation rules through CrossGL and discharge them per target.

``apply_rule_carrying(program, source, target)`` annotates the CrossGL AST with the rules ``source`` defines
(:mod:`.annotate`) and lowers them for ``target`` (:mod:`.lower`) as the rule table says (:mod:`.rule_table`). It
runs on every translation between parsing the CrossGL AST and target code generation. See
docs/source/adr/0001-rule-carrying.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .aggregate_equality import (
    AggregateEqualityOperandError,
    AggregateEqualityUnsupportedError,
    lower_aggregate_equality,
)
from .annotate import Annotator
from .array_length import fold_array_lengths
from .array_parameters import copy_written_array_parameters
from .array_values import ArrayConstructorUnsupportedError, hoist_array_constructors
from .atomic_forms import canonicalize_atomic_calls, hlsl_out_argument_atomics
from .binding_constants import BindingIndexUnresolvedError, fold_binding_constants
from .boolean_reductions import lower_boolean_reductions
from .buffer_strides import materialize_buffer_strides
from .builtin_input_reads import pass_builtin_inputs
from .builtin_values import BuiltinValueUnsupportedError, canonicalize_builtin_values
from .colour_attachment_inputs import refuse_colour_attachment_inputs
from .compute_entry_values import ComputeEntryValueNameError, lower_compute_entry_values
from .constant_buffer_size import (
    ConstantBufferTooLargeError,
    refuse_large_constant_buffers,
)
from .constant_extents import fold_constant_extents, fold_stage_layout_counts
from .constant_reference_layouts import (
    ConstantReferenceLayoutError,
    hoist_constant_reference_layouts,
)
from .debug_printf import DebugPrintfUnsupportedError, refuse_debug_printf
from .declaration_scopes import separate_initializer_scopes
from .declared_types import UndeclaredTypeError, refuse_undeclared_types
from .default_buffer_layouts import mark_default_buffer_layouts
from .descriptor_arrays import (
    DescriptorArrayUnsupportedError,
    flatten_descriptor_arrays,
)
from .device_references import (
    DeviceReferenceUnsupportedError,
    lower_device_struct_references,
)
from .dxr_ray_interfaces import (
    DxrRayInterfaceUnsupportedError,
    lower_dxr_ray_interfaces,
)
from .enum_values import lower_enum_values
from .feedback_textures import FeedbackTextureUnsupportedError, refuse_feedback_textures
from .fixed_samplers import bind_fixed_samplers
from .fragment_colour_outputs import (
    FragmentColourOutputUnsupportedError,
    name_fragment_colour_outputs,
)
from .function_name_collisions import rename_colliding_functions
from .geometry_streams import GeometryStreamUnsupportedError, lower_geometry_streams
from .glsl_mesh_stages import GlslMeshStageUnsupportedError, raise_glsl_mesh_stages
from .half import HalfFloatUnsupportedError, legalize_half_floats
from .hit_objects import HitObjectUnsupportedError, refuse_hit_objects
from .hlsl_geometry_streams import (
    HlslGeometryStreamUnsupportedError,
    raise_glsl_geometry_streams,
)
from .hlsl_ray_payloads import (
    HlslRayPayloadUnsupportedError,
    annotate_hlsl_ray_payloads,
)
from .hlsl_stage_interfaces import (
    HlslStageInterfaceUnsupportedError,
    legalize_hlsl_stage_interfaces,
)
from .hlsl_tess_factors import TessFactorUnsupportedError, fit_tess_factors_to_domain
from .hlsl_unsigned_system_values import deliver_unsigned_system_values
from .implicit_lod import ImplicitLodStageUnsupportedError, legalize_implicit_lod
from .interface_block_instances import (
    InterfaceBlockInstanceUnsupportedError,
    declare_interface_block_instances,
)
from .intrinsic_named_functions import (
    IntrinsicOverloadAmbiguousError,
    resolve_intrinsic_named_functions,
)
from .local_structs import LocalStructUnsupportedError, hoist_local_structs
from .lower import Lowerer, RuleInexpressibleError
from .matrix_spellings import (
    AmbiguousMatrixSpellingError,
    refuse_ambiguous_matrix_spellings,
)
from .metal_math_spellings import lower_metal_math_spellings
from .metal_ray_dispatch import (
    MetalRayDispatchUnsupportedError,
    lower_metal_ray_dispatch,
)
from .metal_runtime_types import (
    MetalRuntimeTypeUnsupportedError,
    refuse_metal_runtime_types,
)
from .module_stage_outputs import adopt_module_stage_outputs
from .narrow import NarrowIntegerUnsupportedError, legalize_narrow_integers
from .non_uniform_indices import strip_non_uniform_qualifiers
from .opaque_structs import OpaqueStructUnsupportedError, legalize_opaque_struct_members
from .packed_members import PackedMemberLayoutError, refuse_unplaced_packed_members
from .row_major_constant_members import store_row_major_constant_members_transposed
from .row_major_members import RowMajorMemberError, store_row_major_members_transposed
from .rule_table import FAMILIES, SOURCE_RULES, TARGET_ENTRIES
from .sampler_states import FixedSamplerStateUnsupportedError, legalize_sampler_states
from .scalar_swizzles import expand_scalar_swizzles
from .shader_execution_reordering import (
    ShaderExecutionReorderingUnsupportedError,
    refuse_shader_execution_reordering,
)
from .shader_records import (
    ShaderRecordBufferUnsupportedError,
    refuse_shader_record_buffers,
)
from .shadow_samples import ShadowSampleUnsupportedError, legalize_shadow_samples
from .sizeof_values import SizeofUnsupportedError, fold_sizeof_values
from .stage_constant_blocks import (
    StageConstantBlockConflictError,
    hoist_stage_constant_blocks,
)
from .stage_interface_structs import (
    StageInterfaceUnsupportedError,
    gather_stage_interface_variables,
)
from .stage_io_arrays import PerVertexInputUnsupportedError, refuse_per_vertex_inputs
from .stage_private_variables import hoist_stage_private_variables
from .static_locals import StaticLocalUnsupportedError, hoist_static_locals
from .storage_blocks import (
    SpirvArrayLengthUnsupportedError,
    normalize_spirv_storage_blocks,
)
from .struct_constructor_conversions import convert_struct_constructor_arguments
from .struct_scalar_splats import expand_struct_scalar_splats
from .structured_layouts import (
    StructuredLayoutUnsupportedError,
    legalize_structured_layouts,
)
from .subpass_inputs import SubpassInputUnsupportedError, lower_subpass_inputs
from .task_payload_storage import declare_task_payload_storage
from .tessellation_patches import (
    TessellationPatchUnsupportedError,
    lower_tessellation_patches,
)
from .texel_offsets import legalize_texel_offsets
from .texture_lod_clamp import refuse_texture_lod_clamp
from .texture_residency import declare_hlsl_residency_status, refuse_texture_residency
from .texture_swizzles import canonicalize_texture_swizzles
from .type_aliases import TypeAliasUnsupportedError, resolve_type_aliases
from .unsized_arrays import UnsizedArrayUnsupportedError, legalize_unsized_arrays
from .vector_logic import lower_vector_logic
from .vertex_input_positions import (
    VertexInputPositionUnsupportedError,
    attribute_vertex_input_positions,
)
from .warp import CudaWarpIntrinsicUnsupportedError, legalize_cuda_warp_intrinsics
from .written_stage_inputs import copy_written_stage_inputs

__all__ = [
    "AggregateEqualityOperandError",
    "AggregateEqualityUnsupportedError",
    "AmbiguousMatrixSpellingError",
    "ArrayConstructorUnsupportedError",
    "BindingIndexUnresolvedError",
    "BuiltinValueUnsupportedError",
    "ComputeEntryValueNameError",
    "ConstantReferenceLayoutError",
    "PackedMemberLayoutError",
    "StageInterfaceUnsupportedError",
    "TessFactorUnsupportedError",
    "ConstantBufferTooLargeError",
    "CudaWarpIntrinsicUnsupportedError",
    "DebugPrintfUnsupportedError",
    "DescriptorArrayUnsupportedError",
    "DeviceReferenceUnsupportedError",
    "DxrRayInterfaceUnsupportedError",
    "FAMILIES",
    "FeedbackTextureUnsupportedError",
    "FixedSamplerStateUnsupportedError",
    "FragmentColourOutputUnsupportedError",
    "GeometryStreamUnsupportedError",
    "HalfFloatUnsupportedError",
    "GlslMeshStageUnsupportedError",
    "HitObjectUnsupportedError",
    "HlslGeometryStreamUnsupportedError",
    "HlslRayPayloadUnsupportedError",
    "HlslStageInterfaceUnsupportedError",
    "ImplicitLodStageUnsupportedError",
    "InterfaceBlockInstanceUnsupportedError",
    "IntrinsicOverloadAmbiguousError",
    "LocalStructUnsupportedError",
    "MetalRayDispatchUnsupportedError",
    "MetalRuntimeTypeUnsupportedError",
    "NarrowIntegerUnsupportedError",
    "OpaqueStructUnsupportedError",
    "PerVertexInputUnsupportedError",
    "RuleInexpressibleError",
    "RowMajorMemberError",
    "RuleReport",
    "SOURCE_RULES",
    "ShaderExecutionReorderingUnsupportedError",
    "ShaderRecordBufferUnsupportedError",
    "ShadowSampleUnsupportedError",
    "SizeofUnsupportedError",
    "SpirvArrayLengthUnsupportedError",
    "StageConstantBlockConflictError",
    "StaticLocalUnsupportedError",
    "SubpassInputUnsupportedError",
    "VertexInputPositionUnsupportedError",
    "TARGET_ENTRIES",
    "TessellationPatchUnsupportedError",
    "TypeAliasUnsupportedError",
    "UndeclaredTypeError",
    "UnsizedArrayUnsupportedError",
    "apply_rule_carrying",
]


@dataclass
class RuleReport:
    """What the passes did: operations annotated per family, how they were discharged, and what stayed unknown."""

    source: str
    target: str
    annotated: dict = field(default_factory=dict)
    discharged: dict = field(default_factory=dict)
    unresolved: int = 0
    configurations: set = field(default_factory=set)

    def count(self, family):
        self.annotated[family] = self.annotated.get(family, 0) + 1

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "annotated": dict(sorted(self.annotated.items())),
            "discharged": dict(sorted(self.discharged.items())),
            "unresolved": self.unresolved,
            "configurations": sorted(self.configurations),
        }


def apply_rule_carrying(program, source, target) -> RuleReport:
    """Annotate ``program`` (a CrossGL ``ShaderNode``) with ``source``'s rules and lower them for ``target``.

    Structures declared in function bodies first move to module scope under fresh names (:mod:`.local_structs`).
    Non-square ``floatNxM`` matrix spellings, whose shape depends on the language, are refused with
    :class:`AmbiguousMatrixSpellingError` (:mod:`.matrix_spellings`).
    Index attributes that name enumerators or integer constants (``@buffer(BufferIndexVertices)``,
    ``@location(BASE + 1)``) first become integer literals, and one that names anything else is refused with
    :class:`BindingIndexUnresolvedError`; WGSL output keeps a name the program does not declare, for the host to
    declare (:mod:`.binding_constants`).
    User functions named like texture or image built-ins are then renamed, with the calls that resolve to them
    (:mod:`.intrinsic_named_functions`), so no later pass or generator reads such a call as the built-in. ``sizeof``
    then becomes its byte count where every target agrees on it, and ``metal_sizeof`` its byte count with MSL's
    layout; other operands are refused with :class:`SizeofUnsupportedError` (:mod:`.sizeof_values`). Every use of a
    type alias then names the aliased type, the MSL, HLSL and GLSL generators keeping program-scope aliases
    (:mod:`.type_aliases`). GLSL's
    ``stpq`` swizzles are spelled ``xyzw``, which every target accepts (:mod:`.texture_swizzles`), and swizzles of
    scalars become vector constructors (:mod:`.scalar_swizzles`). For targets without whole-value equality on
    structures and arrays, ``==`` and ``!=`` on them compare member by member (:mod:`.aggregate_equality`).
    ``&&``/``||`` on boolean vectors become ``and``/``or`` for HLSL and GLSL, and both forms ``&``/``|`` for WGSL (:mod:`.vector_logic`). For GLSL, WGSL and MSL, ``any``
    and ``all`` of a numeric vector reduce its conversion to a boolean vector, and of a scalar become the scalar
    converted to ``bool`` (:mod:`.boolean_reductions`). For OpenGL, SPIR-V,
    Metal, CUDA and HIP, HLSL-style atomic calls that return the previous value through an ``out`` argument become the
    value-returning forms, and for HLSL value-returning atomics on group-shared and other non-buffer destinations take
    HLSL's ``out`` argument (:mod:`.atomic_forms`). Fragment outputs with a ``COLORn`` semantic become colour
    attachment ``n`` (``@gl_FragData[n]``, :mod:`.fragment_colour_outputs`). A position built-in on a vertex input
    names the ``Position`` vertex attribute for targets other than HLSL (:mod:`.vertex_input_positions`). Built-in value
    attributes (``@builtin(...)``, ``@SV_*``, ``@gl_*``) take the spelling the target's generator reads, a built-in the
    target cannot express being refused with :class:`BuiltinValueUnsupportedError` (:mod:`.builtin_values`). For HLSL, MSL,
    SPIR-V and Slang, the plain-value parameters of ``@ compute fn`` entries move into one constant block per entry
    (:mod:`.compute_entry_values`). For MSL and
    HLSL, a vertex or fragment stage's stage-level interface variables become the entry point's input and output
    structures (:mod:`.stage_interface_structs`). A structure an
    HLSL or Slang program converts from one scalar, ``(S)0``, is written member by member (:mod:`.struct_scalar_splats`).
    A structure constructor argument whose type converts implicitly to its member's type is converted explicitly
    (:mod:`.struct_constructor_conversions`).
    For GLSL and SPIR-V, an HLSL-style geometry stage's input primitive and output stream become GLSL's per-vertex
    inputs, stage outputs and ``EmitVertex()`` (:mod:`.geometry_streams`); for HLSL a GLSL-style geometry stage
    becomes the stream form (:mod:`.hlsl_geometry_streams`), and for HLSL and MSL a GLSL-style mesh or task stage's
    output arrays and task payload become entry parameters (:mod:`.glsl_mesh_stages`), while for GLSL and SPIR-V a
    module-scope ``groupshared`` task payload is declared in task-payload storage (:mod:`.task_payload_storage`).
    HLSL-style hull and domain stages become GLSL tessellation stages that exchange per-vertex arrays, patch variables and tessellation levels, the
    hull's patch-constant function running in invocation 0 after a barrier (:mod:`.tessellation_patches`).
    SPIR-V storage block variables are then rewritten into the hub's buffer forms (:mod:`.storage_blocks`), and
    ``buffer T name[]`` declarations into structured buffers or buffer blocks, any other unsized array of values
    being refused with :class:`UnsizedArrayUnsupportedError` (:mod:`.unsized_arrays`). Buffers with a declared
    layout the target lays out differently become buffers of words (:mod:`.structured_layouts`), a
    ``buffer_dimensions`` stride output the target cannot query becomes the element stride as a constant
    (:mod:`.buffer_strides`), and constant
    references with a declared layout module constant buffers that keep it (:mod:`.constant_reference_layouts`);
    storage whose layout would move its packed vector members is refused (:mod:`.packed_members`). For targets that build arrays
    only in aggregate initializers, other array constructors become initializer lists or named locals
    (:mod:`.array_values`), and Metal callees that write an ``in`` array parameter get a copy of it
    (:mod:`.array_parameters`). For GLSL, an entry point's stage inputs and built-in values that its body writes are
    copied into locals, since GLSL inputs are read-only (:mod:`.written_stage_inputs`). ``a.length()`` of a sized
    array then becomes its element count for targets without that method (:mod:`.array_length`). Fixed sampler state
    (``sampler(...)`` constructors) becomes a bound sampler with a host record for Direct3D, Vulkan and OpenGL
    (:mod:`.fixed_samplers`), and is validated and refused for targets that cannot declare it (:mod:`.sampler_states`).
    Shader Model 6.9 hit objects (``dx::HitObject``) are refused for every target but Direct3D (:mod:`.hit_objects`).
    Shader record buffers are refused for targets without a shader binding table (:mod:`.shader_records`), debug
    prints for targets that cannot print from a shader (:mod:`.debug_printf`), and
    minimum-LOD clamped samples for targets that cannot attach the clamp to a sample (:mod:`.texture_lod_clamp`),
    and texture residency queries for targets that do not lower them (:mod:`.texture_residency`).
    Sampler feedback textures are refused for every target but Direct3D (:mod:`.feedback_textures`).
    GLSL-style mesh and ray stage interfaces are rewritten or refused for HLSL (:mod:`.hlsl_stage_interfaces`), and
    the structures used as ray payloads are declared with payload access qualifiers (:mod:`.hlsl_ray_payloads`).
    Shader Model 6.9 shader execution reordering (``dx::MaybeReorderThread``) is refused for every target but
    Direct3D, which writes the call back unchanged (:mod:`.shader_execution_reordering`).
    Slot attributes that name an integer constant or enumerator are folded to it (:mod:`.binding_constants`).
    User functions that share a name with a module-scope value are renamed (:mod:`.function_name_collisions`).
    Row-major struct members are stored as their column-major transpose, with every use rewritten, for targets
    whose structs cannot carry a matrix layout (:mod:`.row_major_members`), and so are the row-major members of
    Metal's unmarked constant blocks (:mod:`.row_major_constant_members`).
    Mutable stage-scope values become module-scope ``static`` variables for the targets that declare per-invocation
    storage at module scope (:mod:`.stage_private_variables`).
    For targets whose texel offsets are constant expressions, constant locals are forwarded into the offsets of
    texel lookups and constant-trip ``for`` loops computing them are unrolled (:mod:`.texel_offsets`).
    Vulkan subpass inputs become texel fetches at the fragment's pixel for targets without input attachments
    (:mod:`.subpass_inputs`).
    Fragment built-in inputs read by name are passed to the HLSL and MSL entry point and its helpers
    (:mod:`.builtin_input_reads`).
    Non-uniform resource index qualifiers are dropped for targets that have none (:mod:`.non_uniform_indices`).
    GLSL sampling functions on shadow samplers become CrossGL's explicit compare forms; for GLSL targets only the
    gathers and explicit-level samples do (:mod:`.shadow_samples`).
    Implicit-LOD samples reached from vertex, geometry, tessellation and ray stages take an explicit base level for
    targets that sample implicitly only in fragment shaders, and bias, derivative and LOD-query operations there are
    refused (:mod:`.implicit_lod`).
    Structs holding samplers,
    textures or images are flattened into separate resources for SPIR-V, or refused with
    :class:`OpaqueStructUnsupportedError` (:mod:`.opaque_structs`), and arrays of arrays of resources become
    one-dimensional descriptor arrays (:mod:`.descriptor_arrays`).
    CUDA warp shuffles are then emulated or refused for targets without CUDA warps (:mod:`.warp`). Binary16 values
    (:mod:`.half`) and narrow integers (:mod:`.narrow`) are then legalized for targets that compute them in 32-bit
    registers. The AST is rewritten in place. Raises :class:`RuleInexpressibleError` when the target
    cannot express a rule, :class:`HalfFloatUnsupportedError` for a binary16 construct the target cannot compute
    exactly, :class:`NarrowIntegerUnsupportedError` for a narrow-integer construct it cannot carry, and
    :class:`CudaWarpIntrinsicUnsupportedError` for a CUDA warp intrinsic it cannot compute with CUDA's lanes.
    The report's JSON form is also stored in ``program.annotations["rule_carrying"]``.
    """
    report = RuleReport(source=str(source or ""), target=str(target or ""))
    hoist_local_structs(program, report.target)
    hoist_stage_constant_blocks(program, report.target)
    refuse_undeclared_types(program, report.target)
    refuse_ambiguous_matrix_spellings(program, report.target)
    refuse_hit_objects(program, report.target)
    refuse_shader_execution_reordering(program, report.target)
    refuse_metal_runtime_types(program, report.target)
    fold_binding_constants(program, report.target)
    fold_constant_extents(program, report.target)
    fold_stage_layout_counts(program, report.target)
    resolve_intrinsic_named_functions(program)
    rename_colliding_functions(program)
    hoist_stage_private_variables(program, report.target)
    hoist_static_locals(program, report.target)
    store_row_major_members_transposed(program, report.target)
    store_row_major_constant_members_transposed(program, report.target)
    legalize_texel_offsets(program, report.target)
    fold_sizeof_values(program, report.target)
    resolve_type_aliases(program, report.target)
    lower_metal_math_spellings(program, report.target)
    lower_enum_values(program, report.target)
    lower_device_struct_references(program, report.target)
    separate_initializer_scopes(program, report.source, report.target)
    canonicalize_texture_swizzles(program)
    expand_scalar_swizzles(program)
    lower_aggregate_equality(program, report.target)
    lower_vector_logic(program, report.target)
    lower_boolean_reductions(program, report.target)
    lower_subpass_inputs(program, report.target)
    strip_non_uniform_qualifiers(program, report.target)
    canonicalize_atomic_calls(program, report.target)
    hlsl_out_argument_atomics(program, report.target)
    pass_builtin_inputs(program, report.target)
    name_fragment_colour_outputs(program, report.target)
    attribute_vertex_input_positions(program, report.target)
    canonicalize_builtin_values(program, report.target)
    deliver_unsigned_system_values(program, report.target)
    lower_compute_entry_values(program, report.target)
    adopt_module_stage_outputs(program, report.target)
    gather_stage_interface_variables(program, report.target)
    fit_tess_factors_to_domain(program, report.target)
    expand_struct_scalar_splats(program, report.source, report.target)
    convert_struct_constructor_arguments(program)
    lower_tessellation_patches(program, report.target)
    lower_geometry_streams(program, report.target)
    raise_glsl_geometry_streams(program, report.target)
    raise_glsl_mesh_stages(program, report.target)
    declare_task_payload_storage(program, report.target)
    lower_dxr_ray_interfaces(program, report.target)
    lower_metal_ray_dispatch(program, report.target)
    declare_interface_block_instances(program, report.target)
    normalize_spirv_storage_blocks(program, report.source, report.target)
    legalize_unsized_arrays(program, report.target)
    mark_default_buffer_layouts(program, report.target)
    legalize_structured_layouts(program, report.target)
    materialize_buffer_strides(program, report.target)
    hoist_constant_reference_layouts(program, report.target)
    refuse_unplaced_packed_members(program, report.target)
    hoist_array_constructors(program, report.target)
    copy_written_array_parameters(program, report.target)
    copy_written_stage_inputs(program, report.target)
    refuse_per_vertex_inputs(program, report.target)
    bind_fixed_samplers(program, report.target)
    refuse_colour_attachment_inputs(program, report.target)
    legalize_sampler_states(program, report.target)
    refuse_shader_record_buffers(program, report.target)
    refuse_debug_printf(program, report.target)
    refuse_texture_lod_clamp(program, report.target)
    refuse_texture_residency(program, report.target)
    refuse_feedback_textures(program, report.target)
    declare_hlsl_residency_status(program, report.target)
    fold_array_lengths(program, report.target)
    refuse_large_constant_buffers(program, report.target)
    legalize_hlsl_stage_interfaces(program, report.target)
    annotate_hlsl_ray_payloads(program, report.target)
    legalize_opaque_struct_members(program, report.target)
    flatten_descriptor_arrays(program, report.target)
    legalize_shadow_samples(program, report.target)
    legalize_implicit_lod(program, report.target)
    legalize_cuda_warp_intrinsics(program, report.target)
    legalize_half_floats(program, report.source, report.target)
    legalize_narrow_integers(program, report.source, report.target)
    Annotator(program, report.source, report).run()
    Lowerer(program, report.target, report).run()
    annotations = getattr(program, "annotations", None)
    if isinstance(annotations, dict):
        annotations["rule_carrying"] = report.to_json()
    return report
