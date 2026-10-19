"""Rule carrying: carry the source language's operation rules through CrossGL and discharge them per target.

``apply_rule_carrying(program, source, target)`` annotates the CrossGL AST with the rules ``source`` defines
(:mod:`.annotate`) and lowers them for ``target`` (:mod:`.lower`) as the rule table says (:mod:`.rule_table`). It
runs on every translation between parsing the CrossGL AST and target code generation. See
docs/source/adr/0001-rule-carrying.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .annotate import Annotator
from .half import HalfFloatUnsupportedError, legalize_half_floats
from .lower import Lowerer, RuleInexpressibleError
from .narrow import NarrowIntegerUnsupportedError, legalize_narrow_integers
from .opaque_structs import OpaqueStructUnsupportedError, legalize_opaque_struct_members
from .rule_table import FAMILIES, SOURCE_RULES, TARGET_ENTRIES
from .storage_blocks import (
    SpirvArrayLengthUnsupportedError,
    normalize_spirv_storage_blocks,
)
from .unsized_arrays import UnsizedArrayUnsupportedError, legalize_unsized_arrays
from .warp import CudaWarpIntrinsicUnsupportedError, legalize_cuda_warp_intrinsics

__all__ = [
    "CudaWarpIntrinsicUnsupportedError",
    "FAMILIES",
    "HalfFloatUnsupportedError",
    "NarrowIntegerUnsupportedError",
    "OpaqueStructUnsupportedError",
    "RuleInexpressibleError",
    "RuleReport",
    "SOURCE_RULES",
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

    SPIR-V storage block variables are first rewritten into the hub's buffer forms (:mod:`.storage_blocks`), and
    ``buffer T name[]`` declarations into structured buffers or buffer blocks, any other unsized array of values
    being refused with :class:`UnsizedArrayUnsupportedError` (:mod:`.unsized_arrays`). Structs holding samplers,
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
    normalize_spirv_storage_blocks(program, report.source, report.target)
    legalize_unsized_arrays(program, report.target)
    legalize_opaque_struct_members(program, report.target)
    legalize_cuda_warp_intrinsics(program, report.target)
    legalize_half_floats(program, report.source, report.target)
    legalize_narrow_integers(program, report.source, report.target)
    Annotator(program, report.source, report).run()
    Lowerer(program, report.target, report).run()
    annotations = getattr(program, "annotations", None)
    if isinstance(annotations, dict):
        annotations["rule_carrying"] = report.to_json()
    return report
