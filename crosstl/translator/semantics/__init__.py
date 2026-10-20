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
from .half import HalfFloatUnsupportedError, legalize_half_floats
from .hlsl_stage_interfaces import (
    HlslStageInterfaceUnsupportedError,
    legalize_hlsl_stage_interfaces,
)
from .implicit_lod import ImplicitLodStageUnsupportedError, legalize_implicit_lod
from .intrinsic_named_functions import (
    IntrinsicOverloadAmbiguousError,
    resolve_intrinsic_named_functions,
)
from .lower import Lowerer, RuleInexpressibleError
from .narrow import NarrowIntegerUnsupportedError, legalize_narrow_integers
from .opaque_structs import OpaqueStructUnsupportedError, legalize_opaque_struct_members
from .rule_table import FAMILIES, SOURCE_RULES, TARGET_ENTRIES
from .sampler_states import FixedSamplerStateUnsupportedError, legalize_sampler_states
from .shader_records import (
    ShaderRecordBufferUnsupportedError,
    refuse_shader_record_buffers,
)
from .shadow_samples import ShadowSampleUnsupportedError, legalize_shadow_samples
from .stage_io_arrays import PerVertexInputUnsupportedError, refuse_per_vertex_inputs
from .storage_blocks import (
    SpirvArrayLengthUnsupportedError,
    normalize_spirv_storage_blocks,
)
from .struct_scalar_splats import expand_struct_scalar_splats
from .texture_lod_clamp import refuse_texture_lod_clamp
from .texture_swizzles import canonicalize_texture_swizzles
from .unsized_arrays import UnsizedArrayUnsupportedError, legalize_unsized_arrays
from .warp import CudaWarpIntrinsicUnsupportedError, legalize_cuda_warp_intrinsics

__all__ = [
    "CudaWarpIntrinsicUnsupportedError",
    "FAMILIES",
    "FixedSamplerStateUnsupportedError",
    "HalfFloatUnsupportedError",
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
    ``stpq`` swizzles are spelled ``xyzw``, which every target accepts (:mod:`.texture_swizzles`). A structure an
    HLSL or Slang program converts from one scalar, ``(S)0``, is written member by member (:mod:`.struct_scalar_splats`).
    SPIR-V storage block variables are then rewritten into the hub's buffer forms (:mod:`.storage_blocks`), and
    ``buffer T name[]`` declarations into structured buffers or buffer blocks, any other unsized array of values
    being refused with :class:`UnsizedArrayUnsupportedError` (:mod:`.unsized_arrays`); ``a.length()`` of a sized
    array then becomes its element count for targets without that method (:mod:`.array_length`). Fixed sampler state
    (``sampler(...)`` constructors) is validated and refused for targets that cannot declare it (:mod:`.sampler_states`).
    Shader record buffers are refused for targets without a shader binding table (:mod:`.shader_records`), and
    minimum-LOD clamped samples for targets that cannot attach the clamp to a sample (:mod:`.texture_lod_clamp`).
    GLSL-style mesh and ray stage interfaces are rewritten or refused for HLSL (:mod:`.hlsl_stage_interfaces`).
    GLSL sampling functions on shadow samplers become CrossGL's explicit compare forms; for GLSL targets only the
    gathers and explicit-level samples do (:mod:`.shadow_samples`).
    Implicit-LOD samples reached from vertex, geometry, tessellation and ray stages take an explicit base level for
    targets that sample implicitly only in fragment shaders, and bias, derivative and LOD-query operations there are
    refused (:mod:`.implicit_lod`).
    Structs holding samplers,
    textures or images are flattened into separate resources for SPIR-V, or refused with
    :class:`OpaqueStructUnsupportedError` (:mod:`.opaque_structs`).
    CUDA warp shuffles are then emulated or refused for targets without CUDA warps (:mod:`.warp`). Binary16 values
    (:mod:`.half`) and narrow integers (:mod:`.narrow`) are then legalized for targets that compute them in 32-bit
    registers. The AST is rewritten in place. Raises :class:`RuleInexpressibleError` when the target
    cannot express a rule, :class:`HalfFloatUnsupportedError` for a binary16 construct the target cannot compute
    exactly, :class:`NarrowIntegerUnsupportedError` for a narrow-integer construct it cannot carry, and
    :class:`CudaWarpIntrinsicUnsupportedError` for a CUDA warp intrinsic it cannot compute with CUDA's lanes.
    The report's JSON form is also stored in ``program.annotations["rule_carrying"]``.
    """
    report = RuleReport(source=str(source or ""), target=str(target or ""))
    resolve_intrinsic_named_functions(program)
    canonicalize_texture_swizzles(program)
    expand_struct_scalar_splats(program, report.source, report.target)
    normalize_spirv_storage_blocks(program, report.source, report.target)
    legalize_unsized_arrays(program, report.target)
    refuse_per_vertex_inputs(program, report.target)
    legalize_sampler_states(program, report.target)
    refuse_shader_record_buffers(program, report.target)
    refuse_texture_lod_clamp(program, report.target)
    fold_array_lengths(program, report.target)
    legalize_hlsl_stage_interfaces(program, report.target)
    legalize_opaque_struct_members(program, report.target)
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
