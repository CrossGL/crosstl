"""Rule carrying: annotate and lower the source language's rules through CrossGL per target.

``apply_rule_carrying(program, source, target)`` annotates the CrossGL AST with the rules ``source`` defines
(:mod:`.annotate`) and lowers them for ``target`` (:mod:`.lower`) as the rule table says (:mod:`.rule_table`).
It runs on every translation between parsing the CrossGL AST and target code generation.  Each semantic pass is
documented in its own module (see the submodules of :mod:`crosstl.translator.semantics`).  The definitive
ordered list of passes and their sequencing constraints is :data:`_PASSES` in this module.
See also docs/source/adr/0001-rule-carrying.md.
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
from .bit_functions import legalize_bit_functions
from .block_matrix_orders import BlockMatrixOrderError, distribute_block_matrix_orders
from .boolean_reductions import lower_boolean_reductions
from .buffer_strides import materialize_buffer_strides
from .builtin_input_reads import pass_builtin_inputs
from .builtin_names import resolve_builtin_names
from .builtin_output_writes import (
    convert_signed_builtin_outputs,
    declare_builtin_outputs,
)
from .builtin_stage_identifiers import refuse_misplaced_builtin_identifiers
from .builtin_stages import refuse_misplaced_builtins
from .builtin_values import BuiltinValueUnsupportedError, canonicalize_builtin_values
from .colour_attachment_inputs import refuse_colour_attachment_inputs
from .compute_entry_values import ComputeEntryValueNameError, lower_compute_entry_values
from .constant_buffer_layouts import hoist_constant_buffer_layouts
from .constant_buffer_size import (
    ConstantBufferTooLargeError,
    refuse_large_constant_buffers,
)
from .constant_calls import fold_constant_calls
from .constant_extents import fold_constant_extents, fold_stage_layout_counts
from .constant_reference_layouts import (
    ConstantReferenceLayoutError,
    hoist_constant_reference_layouts,
)
from .debug_printf import DebugPrintfUnsupportedError, refuse_debug_printf
from .declaration_scopes import separate_initializer_scopes
from .declared_functions import UndeclaredFunctionError, refuse_undeclared_functions
from .declared_identifiers import (
    UndeclaredIdentifierError,
    generate_refusing_undeclared_identifiers,
    record_undeclared_identifiers,
)
from .declared_types import UndeclaredTypeError, refuse_undeclared_types
from .default_buffer_layouts import mark_default_buffer_layouts
from .descriptor_arrays import (
    DescriptorArrayUnsupportedError,
    flatten_descriptor_arrays,
)
from .device_private_globals import (
    DevicePrivateGlobalUnsupportedError,
    lower_device_private_globals,
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
from .float_relations import lower_float_relations
from .fragment_colour_outputs import (
    FragmentColourOutputUnsupportedError,
    name_fragment_colour_outputs,
)
from .fragment_rate_barycentrics import (
    MetalBarycentricsUnsupportedError,
    lower_shading_rate_flags,
    refuse_mixed_metal_barycentrics,
)
from .function_name_collisions import rename_colliding_functions
from .generic_struct_defaults import fill_generic_struct_defaults
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
from .member_placements import refuse_unplaced_member_offsets
from .metal_math_spellings import lower_metal_math_spellings
from .metal_ray_dispatch import (
    MetalRayDispatchUnsupportedError,
    lower_metal_ray_dispatch,
)
from .metal_runtime_types import (
    MetalRuntimeTypeUnsupportedError,
    refuse_metal_runtime_types,
)
from .metal_vector_scalar_operands import convert_metal_vector_scalar_operands
from .module_stage_outputs import adopt_module_stage_outputs
from .narrow import NarrowIntegerUnsupportedError, legalize_narrow_integers
from .narrow_arithmetic import wrap_narrow_arithmetic
from .non_uniform_indices import strip_non_uniform_qualifiers
from .opaque_structs import OpaqueStructUnsupportedError, legalize_opaque_struct_members
from .packed_members import PackedMemberLayoutError, refuse_unplaced_packed_members
from .push_constant_layouts import mark_push_constant_layouts
from .relative_vertex_ids import (
    RelativeVertexIdUnsupportedError,
    lower_relative_vertex_ids,
)
from .row_major_constant_members import store_row_major_constant_members_transposed
from .row_major_members import RowMajorMemberError, store_row_major_members_transposed
from .row_major_struct_members import (
    RowMajorStructMemberUnsupportedError,
    orient_row_major_buffer_struct_members,
    orient_row_major_struct_members,
)
from .rule_pragmas import take_declared_families
from .rule_table import FAMILIES, SOURCE_RULES, TARGET_ENTRIES
from .sampler_states import FixedSamplerStateUnsupportedError, legalize_sampler_states
from .scalar_brace_values import (
    ScalarBraceInitializerError,
    zero_empty_scalar_initializers,
)
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
from .slang_vertex_indices import spell_slang_vertex_indices
from .spirv_instructions import (
    SpirvOperationUnsupportedError,
    refuse_spirv_instructions,
)
from .stage_bound_functions import bind_stage_functions
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
from .static_assertions import (
    StaticAssertionFailedError,
    StaticAssertionUnresolvedError,
    resolve_static_assertions,
)
from .static_locals import StaticLocalUnsupportedError, hoist_static_locals
from .storage_blocks import (
    SpirvArrayLengthUnsupportedError,
    normalize_spirv_storage_blocks,
)
from .struct_constructor_conversions import convert_struct_constructor_arguments
from .struct_methods import StructMethodUnsupportedError, lower_struct_methods
from .struct_scalar_splats import expand_struct_scalar_splats
from .structured_layouts import (
    StructuredLayoutUnsupportedError,
    legalize_structured_layouts,
)
from .subpass_inputs import SubpassInputUnsupportedError, lower_subpass_inputs
from .task_payload_storage import declare_task_payload_storage
from .template_assertions import resolve_template_assertions
from .tessellation_patches import (
    TessellationPatchUnsupportedError,
    lower_tessellation_patches,
)
from .texel_offsets import legalize_texel_offsets
from .texture_lod_clamp import refuse_texture_lod_clamp
from .texture_residency import declare_hlsl_residency_status, refuse_texture_residency
from .texture_swizzles import canonicalize_texture_swizzles
from .type_aliases import TypeAliasUnsupportedError, resolve_type_aliases
from .unsized_arrays import (
    StorageBufferParameterUnsupportedError,
    UnsizedArrayUnsupportedError,
    legalize_unsized_arrays,
)
from .vector_logic import lower_vector_logic
from .vertex_input_positions import (
    VertexInputPositionUnsupportedError,
    attribute_vertex_input_positions,
)
from .void_casts import VoidCastError, lower_void_casts
from .void_values import VoidValueError, refuse_void_values
from .warp import CudaWarpIntrinsicUnsupportedError, legalize_cuda_warp_intrinsics
from .written_stage_inputs import copy_written_stage_inputs

# ---------------------------------------------------------------------------
# Ordered sequence of semantic passes run by apply_rule_carrying.
#
# Each entry is a ``(function, arg_keys)`` pair where ``arg_keys`` is a tuple
# of strings drawn from ``{"program", "source", "target"}``.  Using named
# pairs rather than lambdas means that tracebacks from a failing pass show
# the real function name instead of ``<lambda>``.
#
# To add a new pass: import it at the top of this file and insert a
# ``(function, arg_keys)`` pair at the right position.  No changes to
# ``apply_rule_carrying`` itself are needed.
#
# Ordering constraints (why each pass sits where it does):
#   - hoist_local_structs: local structures must move to module scope before
#     member functions are converted to free functions (lower_struct_methods).
#   - fill_generic_struct_defaults: generic defaults must be filled before
#     struct methods are lowered.
#   - bind_stage_functions: stage binding must precede struct-method lowering
#     so that stage-bound methods are correctly attributed.
#   - lower_struct_methods: member functions become free functions before any
#     pass that relies on a flat function namespace.
#   - lower_relative_vertex_ids: gl_RelativeVertexID must be resolved before
#     spell_slang_vertex_indices so that SV_VertexID is spelled correctly.
#   - legalize_half_floats / legalize_narrow_integers / wrap_narrow_arithmetic:
#     width-reducing passes run last, after all structural rewrites.
# ---------------------------------------------------------------------------
_PASSES = [
    (hoist_local_structs, ("program", "target")),
    (fill_generic_struct_defaults, ("program", "target")),
    (bind_stage_functions, ("program",)),
    (lower_struct_methods, ("program", "target")),
    (hoist_stage_constant_blocks, ("program", "target")),
    (distribute_block_matrix_orders, ("program", "target")),
    (resolve_template_assertions, ("program", "target")),
    (refuse_undeclared_types, ("program", "target")),
    (record_undeclared_identifiers, ("program", "target")),
    (refuse_undeclared_functions, ("program", "target")),
    (lower_void_casts, ("program", "target")),
    (refuse_void_values, ("program", "target")),
    (resolve_builtin_names, ("program", "target")),
    (refuse_misplaced_builtins, ("program", "target")),
    (refuse_misplaced_builtin_identifiers, ("program", "target")),
    (lower_relative_vertex_ids, ("program", "target")),
    (refuse_ambiguous_matrix_spellings, ("program", "target")),
    (refuse_hit_objects, ("program", "target")),
    (refuse_shader_execution_reordering, ("program", "target")),
    (refuse_metal_runtime_types, ("program", "target")),
    (fold_constant_calls, ("program", "target")),
    (fold_binding_constants, ("program", "target")),
    (fold_constant_extents, ("program", "target")),
    (fold_stage_layout_counts, ("program", "target")),
    (resolve_intrinsic_named_functions, ("program",)),
    (rename_colliding_functions, ("program",)),
    (hoist_stage_private_variables, ("program", "target")),
    (hoist_static_locals, ("program", "target")),
    (lower_device_private_globals, ("program", "target")),
    (orient_row_major_buffer_struct_members, ("program", "target")),
    (store_row_major_members_transposed, ("program", "target")),
    (store_row_major_constant_members_transposed, ("program", "target")),
    (legalize_texel_offsets, ("program", "target")),
    (fold_sizeof_values, ("program", "target")),
    (resolve_type_aliases, ("program", "target")),
    (orient_row_major_struct_members, ("program", "target")),
    (resolve_static_assertions, ("program", "target")),
    (zero_empty_scalar_initializers, ("program", "target")),
    (lower_metal_math_spellings, ("program", "target")),
    (lower_float_relations, ("program", "target")),
    (lower_enum_values, ("program", "target")),
    (lower_device_struct_references, ("program", "target")),
    (separate_initializer_scopes, ("program", "source", "target")),
    (canonicalize_texture_swizzles, ("program",)),
    (expand_scalar_swizzles, ("program",)),
    (lower_aggregate_equality, ("program", "target")),
    (lower_vector_logic, ("program", "target")),
    (lower_boolean_reductions, ("program", "target")),
    (lower_subpass_inputs, ("program", "target")),
    (strip_non_uniform_qualifiers, ("program", "target")),
    (canonicalize_atomic_calls, ("program", "target")),
    (hlsl_out_argument_atomics, ("program", "target")),
    (lower_shading_rate_flags, ("program", "target")),
    (refuse_mixed_metal_barycentrics, ("program", "target")),
    (declare_builtin_outputs, ("program", "target")),
    (spell_slang_vertex_indices, ("program", "target")),
    (pass_builtin_inputs, ("program", "target")),
    (name_fragment_colour_outputs, ("program", "target")),
    (attribute_vertex_input_positions, ("program", "target")),
    (canonicalize_builtin_values, ("program", "target")),
    (deliver_unsigned_system_values, ("program", "target")),
    (lower_compute_entry_values, ("program", "target")),
    (adopt_module_stage_outputs, ("program", "target")),
    (convert_signed_builtin_outputs, ("program", "target")),
    (gather_stage_interface_variables, ("program", "target")),
    (fit_tess_factors_to_domain, ("program", "target")),
    (expand_struct_scalar_splats, ("program", "source", "target")),
    (convert_metal_vector_scalar_operands, ("program", "source")),
    (convert_struct_constructor_arguments, ("program",)),
    (lower_tessellation_patches, ("program", "target")),
    (lower_geometry_streams, ("program", "target")),
    (raise_glsl_geometry_streams, ("program", "target")),
    (raise_glsl_mesh_stages, ("program", "target")),
    (declare_task_payload_storage, ("program", "target")),
    (lower_dxr_ray_interfaces, ("program", "target")),
    (lower_metal_ray_dispatch, ("program", "target")),
    (declare_interface_block_instances, ("program", "target")),
    (normalize_spirv_storage_blocks, ("program", "source", "target")),
    (legalize_unsized_arrays, ("program", "target")),
    (mark_default_buffer_layouts, ("program", "target")),
    (legalize_structured_layouts, ("program", "target")),
    (materialize_buffer_strides, ("program", "target")),
    (refuse_unplaced_member_offsets, ("program", "target")),
    (mark_push_constant_layouts, ("program", "target")),
    (hoist_constant_buffer_layouts, ("program", "target")),
    (hoist_constant_reference_layouts, ("program", "target")),
    (refuse_unplaced_packed_members, ("program", "target")),
    (hoist_array_constructors, ("program", "target")),
    (copy_written_array_parameters, ("program", "target")),
    (copy_written_stage_inputs, ("program", "target")),
    (refuse_per_vertex_inputs, ("program", "target")),
    (bind_fixed_samplers, ("program", "target")),
    (refuse_colour_attachment_inputs, ("program", "target")),
    (legalize_sampler_states, ("program", "target")),
    (refuse_shader_record_buffers, ("program", "target")),
    (refuse_debug_printf, ("program", "target")),
    (refuse_texture_lod_clamp, ("program", "target")),
    (refuse_texture_residency, ("program", "target")),
    (refuse_feedback_textures, ("program", "target")),
    (declare_hlsl_residency_status, ("program", "target")),
    (fold_array_lengths, ("program", "target")),
    (refuse_large_constant_buffers, ("program", "target")),
    (legalize_hlsl_stage_interfaces, ("program", "target")),
    (annotate_hlsl_ray_payloads, ("program", "target")),
    (legalize_opaque_struct_members, ("program", "target")),
    (flatten_descriptor_arrays, ("program", "target")),
    (legalize_shadow_samples, ("program", "target")),
    (legalize_implicit_lod, ("program", "target")),
    (legalize_cuda_warp_intrinsics, ("program", "target")),
    (legalize_half_floats, ("program", "source", "target")),
    (legalize_bit_functions, ("program", "source", "target")),
    (legalize_narrow_integers, ("program", "source", "target")),
    (wrap_narrow_arithmetic, ("program", "source", "target")),
    (refuse_spirv_instructions, ("program", "target")),
]

__all__ = [
    "AggregateEqualityOperandError",
    "AggregateEqualityUnsupportedError",
    "AmbiguousMatrixSpellingError",
    "ArrayConstructorUnsupportedError",
    "BindingIndexUnresolvedError",
    "BlockMatrixOrderError",
    "BuiltinValueUnsupportedError",
    "ComputeEntryValueNameError",
    "ConstantReferenceLayoutError",
    "DevicePrivateGlobalUnsupportedError",
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
    "RowMajorStructMemberUnsupportedError",
    "HlslRayPayloadUnsupportedError",
    "HlslStageInterfaceUnsupportedError",
    "ImplicitLodStageUnsupportedError",
    "InterfaceBlockInstanceUnsupportedError",
    "IntrinsicOverloadAmbiguousError",
    "LocalStructUnsupportedError",
    "MetalBarycentricsUnsupportedError",
    "MetalRayDispatchUnsupportedError",
    "MetalRuntimeTypeUnsupportedError",
    "NarrowIntegerUnsupportedError",
    "OpaqueStructUnsupportedError",
    "PerVertexInputUnsupportedError",
    "RelativeVertexIdUnsupportedError",
    "RuleInexpressibleError",
    "RowMajorMemberError",
    "RuleReport",
    "SOURCE_RULES",
    "ScalarBraceInitializerError",
    "ShaderExecutionReorderingUnsupportedError",
    "ShaderRecordBufferUnsupportedError",
    "ShadowSampleUnsupportedError",
    "SizeofUnsupportedError",
    "StaticAssertionFailedError",
    "StaticAssertionUnresolvedError",
    "SpirvArrayLengthUnsupportedError",
    "SpirvOperationUnsupportedError",
    "StageConstantBlockConflictError",
    "StaticLocalUnsupportedError",
    "StorageBufferParameterUnsupportedError",
    "StructMethodUnsupportedError",
    "SubpassInputUnsupportedError",
    "VertexInputPositionUnsupportedError",
    "TARGET_ENTRIES",
    "TessellationPatchUnsupportedError",
    "TypeAliasUnsupportedError",
    "UndeclaredFunctionError",
    "UndeclaredIdentifierError",
    "UndeclaredTypeError",
    "UnsizedArrayUnsupportedError",
    "VoidCastError",
    "VoidValueError",
    "apply_rule_carrying",
    "generate_refusing_undeclared_identifiers",
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

    Runs the ordered sequence of semantic passes in :data:`_PASSES` in turn, then the :class:`.annotate.Annotator`
    and :class:`.lower.Lowerer`.  Each pass is documented in its own submodule.  The AST is rewritten in place.

    Raises :class:`RuleInexpressibleError` when the target cannot express a rule,
    :class:`HalfFloatUnsupportedError` for a binary16 construct the target cannot compute exactly,
    :class:`NarrowIntegerUnsupportedError` for a narrow-integer construct it cannot carry, and
    :class:`CudaWarpIntrinsicUnsupportedError` for a CUDA warp intrinsic it cannot compute with CUDA's lanes.
    The report's JSON form is also stored in ``program.annotations["rule_carrying"]``.
    """
    report = RuleReport(source=str(source or ""), target=str(target or ""))
    declared_families = take_declared_families(program, report.target)
    _arg_map = {"program": program, "source": report.source, "target": report.target}
    for _pass in _PASSES:
        fn, arg_keys = _pass
        fn(*(_arg_map[k] for k in arg_keys))
    Annotator(program, report.source, report, declared_families).run()
    Lowerer(program, report.target, report).run()
    annotations = getattr(program, "annotations", None)
    if isinstance(annotations, dict):
        annotations["rule_carrying"] = report.to_json()
    return report
