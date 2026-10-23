"""Rule carrying: carry the source language's operation rules through CrossGL and discharge them per target.

``apply_rule_carrying(program, source, target)`` annotates the CrossGL AST with the rules ``source`` defines
(:mod:`.annotate`) and lowers them for ``target`` (:mod:`.lower`) as the rule table says (:mod:`.rule_table`). It
runs on every translation between parsing the CrossGL AST and target code generation. See
docs/source/adr/0001-rule-carrying.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .annotate import Annotator
from .array_length import fold_array_lengths
from .array_parameters import copy_written_array_parameters
from .array_values import ArrayConstructorUnsupportedError, hoist_array_constructors
from .atomic_forms import canonicalize_atomic_calls, hlsl_out_argument_atomics
from .buffer_strides import materialize_buffer_strides
from .builtin_values import BuiltinValueUnsupportedError, canonicalize_builtin_values
from .colour_attachment_inputs import refuse_colour_attachment_inputs
from .compute_entry_values import ComputeEntryValueNameError, lower_compute_entry_values
from .constant_reference_layouts import (
    ConstantReferenceLayoutError,
    hoist_constant_reference_layouts,
)
from .debug_printf import DebugPrintfUnsupportedError, refuse_debug_printf
from .declared_types import UndeclaredTypeError, refuse_undeclared_types
from .descriptor_arrays import (
    DescriptorArrayUnsupportedError,
    flatten_descriptor_arrays,
)
from .dxr_ray_interfaces import (
    DxrRayInterfaceUnsupportedError,
    lower_dxr_ray_interfaces,
)
from .function_name_collisions import rename_colliding_functions
from .geometry_streams import GeometryStreamUnsupportedError, lower_geometry_streams
from .half import HalfFloatUnsupportedError, legalize_half_floats
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
from .intrinsic_named_functions import (
    IntrinsicOverloadAmbiguousError,
    resolve_intrinsic_named_functions,
)
from .lower import Lowerer, RuleInexpressibleError
from .module_stage_outputs import adopt_module_stage_outputs
from .narrow import NarrowIntegerUnsupportedError, legalize_narrow_integers
from .non_uniform_indices import strip_non_uniform_qualifiers
from .opaque_structs import OpaqueStructUnsupportedError, legalize_opaque_struct_members
from .packed_members import PackedMemberLayoutError, refuse_unplaced_packed_members
from .rule_table import FAMILIES, SOURCE_RULES, TARGET_ENTRIES
from .sampler_states import FixedSamplerStateUnsupportedError, legalize_sampler_states
from .shader_records import (
    ShaderRecordBufferUnsupportedError,
    refuse_shader_record_buffers,
)
from .shadow_samples import ShadowSampleUnsupportedError, legalize_shadow_samples
from .stage_interface_structs import (
    StageInterfaceUnsupportedError,
    gather_stage_interface_variables,
)
from .stage_io_arrays import PerVertexInputUnsupportedError, refuse_per_vertex_inputs
from .storage_blocks import (
    SpirvArrayLengthUnsupportedError,
    normalize_spirv_storage_blocks,
)
from .struct_scalar_splats import expand_struct_scalar_splats
from .structured_layouts import (
    StructuredLayoutUnsupportedError,
    legalize_structured_layouts,
)
from .tessellation_patches import (
    TessellationPatchUnsupportedError,
    lower_tessellation_patches,
)
from .texture_lod_clamp import refuse_texture_lod_clamp
from .texture_swizzles import canonicalize_texture_swizzles
from .unsized_arrays import UnsizedArrayUnsupportedError, legalize_unsized_arrays
from .warp import CudaWarpIntrinsicUnsupportedError, legalize_cuda_warp_intrinsics

__all__ = [
    "ArrayConstructorUnsupportedError",
    "BuiltinValueUnsupportedError",
    "ComputeEntryValueNameError",
    "ConstantReferenceLayoutError",
    "PackedMemberLayoutError",
    "StageInterfaceUnsupportedError",
    "TessFactorUnsupportedError",
    "CudaWarpIntrinsicUnsupportedError",
    "DebugPrintfUnsupportedError",
    "DescriptorArrayUnsupportedError",
    "DxrRayInterfaceUnsupportedError",
    "FAMILIES",
    "FixedSamplerStateUnsupportedError",
    "GeometryStreamUnsupportedError",
    "HalfFloatUnsupportedError",
    "HlslGeometryStreamUnsupportedError",
    "HlslRayPayloadUnsupportedError",
    "HlslStageInterfaceUnsupportedError",
    "ImplicitLodStageUnsupportedError",
    "IntrinsicOverloadAmbiguousError",
    "NarrowIntegerUnsupportedError",
    "OpaqueStructUnsupportedError",
    "PerVertexInputUnsupportedError",
    "RuleInexpressibleError",
    "RuleReport",
    "SOURCE_RULES",
    "ShaderRecordBufferUnsupportedError",
    "ShadowSampleUnsupportedError",
    "SpirvArrayLengthUnsupportedError",
    "TARGET_ENTRIES",
    "TessellationPatchUnsupportedError",
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

    User functions named like texture or image built-ins are first renamed, with the calls that resolve to them
    (:mod:`.intrinsic_named_functions`), so no later pass or generator reads such a call as the built-in, and GLSL's
    ``stpq`` swizzles are spelled ``xyzw``, which every target accepts (:mod:`.texture_swizzles`). For OpenGL, SPIR-V
    and Metal, HLSL-style atomic calls that return the previous value through an ``out`` argument become the
    value-returning forms, and for HLSL value-returning atomics on group-shared and other non-buffer destinations take
    HLSL's ``out`` argument (:mod:`.atomic_forms`). Built-in value
    attributes (``@builtin(...)``, ``@SV_*``, ``@gl_*``) take the spelling the target's generator reads, a built-in the
    target cannot express being refused with :class:`BuiltinValueUnsupportedError` (:mod:`.builtin_values`). For HLSL, MSL,
    SPIR-V and Slang, the plain-value parameters of ``@ compute fn`` entries move into one constant block per entry
    (:mod:`.compute_entry_values`). For MSL and
    HLSL, a vertex or fragment stage's stage-level interface variables become the entry point's input and output
    structures (:mod:`.stage_interface_structs`). A structure an
    HLSL or Slang program converts from one scalar, ``(S)0``, is written member by member (:mod:`.struct_scalar_splats`).
    For GLSL and SPIR-V, an HLSL-style geometry stage's input primitive and output stream become GLSL's per-vertex
    inputs, stage outputs and ``EmitVertex()`` (:mod:`.geometry_streams`); for HLSL a GLSL-style geometry stage
    becomes the stream form (:mod:`.hlsl_geometry_streams`). HLSL-style hull and domain stages
    become GLSL tessellation stages that exchange per-vertex arrays, patch variables and tessellation levels, the
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
    (:mod:`.array_parameters`). ``a.length()`` of a sized
    array then becomes its element count for targets without that method (:mod:`.array_length`). Fixed sampler state
    (``sampler(...)`` constructors) is validated and refused for targets that cannot declare it (:mod:`.sampler_states`).
    Shader record buffers are refused for targets without a shader binding table (:mod:`.shader_records`), debug
    prints for targets that cannot print from a shader (:mod:`.debug_printf`), and
    minimum-LOD clamped samples for targets that cannot attach the clamp to a sample (:mod:`.texture_lod_clamp`).
    GLSL-style mesh and ray stage interfaces are rewritten or refused for HLSL (:mod:`.hlsl_stage_interfaces`), and
    the structures used as ray payloads are declared with payload access qualifiers (:mod:`.hlsl_ray_payloads`).
    User functions that share a name with a module-scope value are renamed (:mod:`.function_name_collisions`).
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
    refuse_undeclared_types(program, report.target)
    resolve_intrinsic_named_functions(program)
    rename_colliding_functions(program)
    canonicalize_texture_swizzles(program)
    strip_non_uniform_qualifiers(program, report.target)
    canonicalize_atomic_calls(program, report.target)
    hlsl_out_argument_atomics(program, report.target)
    canonicalize_builtin_values(program, report.target)
    deliver_unsigned_system_values(program, report.target)
    lower_compute_entry_values(program, report.target)
    adopt_module_stage_outputs(program, report.target)
    gather_stage_interface_variables(program, report.target)
    fit_tess_factors_to_domain(program, report.target)
    expand_struct_scalar_splats(program, report.source, report.target)
    lower_tessellation_patches(program, report.target)
    lower_geometry_streams(program, report.target)
    raise_glsl_geometry_streams(program, report.target)
    lower_dxr_ray_interfaces(program, report.target)
    normalize_spirv_storage_blocks(program, report.source, report.target)
    legalize_unsized_arrays(program, report.target)
    legalize_structured_layouts(program, report.target)
    materialize_buffer_strides(program, report.target)
    hoist_constant_reference_layouts(program, report.target)
    refuse_unplaced_packed_members(program, report.target)
    hoist_array_constructors(program, report.target)
    copy_written_array_parameters(program, report.target)
    refuse_per_vertex_inputs(program, report.target)
    refuse_colour_attachment_inputs(program, report.target)
    legalize_sampler_states(program, report.target)
    refuse_shader_record_buffers(program, report.target)
    refuse_debug_printf(program, report.target)
    refuse_texture_lod_clamp(program, report.target)
    fold_array_lengths(program, report.target)
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
