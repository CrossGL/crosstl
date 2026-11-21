"""OpenGL's and Direct3D's instance ID on SPIR-V's base-inclusive ``InstanceIndex``.

CrossGL's ``gl_InstanceID`` counts a draw's instances from zero, as OpenGL's ``gl_InstanceID`` does ("The value of
<baseinstance> does not affect the shader-visible value of gl_InstanceID", GL_ARB_base_instance; GLSL 4.60 section
7.1.1) and as Direct3D's ``SV_InstanceID``, which the HLSL and Slang importers read as ``gl_InstanceID``, does.
SPIR-V's ``BuiltIn InstanceIndex`` includes the draw's first instance: it is ``gl_InstanceIndex``, which
GL_KHR_vulkan_glsl defines as ``gl_InstanceID + gl_BaseInstance``. The SPIR-V generator decorated ``gl_InstanceID``
as ``InstanceIndex`` and subtracted nothing, so every draw with a non-zero first instance read shifted instance
numbers. slangc and DXC (``-fvk-support-nonzero-base-instance``) lower ``SV_InstanceID`` to
``InstanceIndex - BaseInstance``, with capability ``DrawParameters``.

:func:`lower_spirv_instance_ids` rewrites the vertex stages of a program bound for SPIR-V, in place, the way
:mod:`.relative_vertex_ids` lowers ``gl_RelativeVertexID``:

- a vertex entry point's parameter ``T v @ gl_InstanceID`` becomes a ``gl_InstanceIndex`` parameter under a fresh
  name, the entry point takes the base instance (reusing a parameter that already carries it) and declares
  ``T v = T(index) - T(base);`` first. A stage input structure member carrying the value carries
  ``gl_InstanceIndex`` instead, and the entry point copies the structure parameter into a local of the original
  name whose member it rebases first;
- a stage-level input variable carrying ``gl_InstanceID`` carries ``gl_InstanceIndex`` under a fresh name, the stage
  declares a ``gl_BaseInstance`` input (reusing one it already has), and a read of the variable becomes their
  difference;
- a read of ``gl_InstanceID`` by name becomes ``gl_InstanceIndex - gl_BaseInstance``.

``gl_InstanceIndex`` keeps a bare ``InstanceIndex``. Ray-tracing stages are not touched: there ``gl_InstanceID``
names the hit instance's ``InstanceId``. A helper outside every stage is only rewritten when the program has no
ray-tracing stage.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .relative_vertex_ids import (
    _carries,
    _fresh,
    _Lowering,
    _names,
    _rebased,
    _respell,
    _respell_reads,
    _stages,
)

__all__ = ["InstanceIdUnsupportedError", "lower_spirv_instance_ids"]

_INSTANCE_ID = "gl_InstanceID"
_INSTANCE_INDEX = "gl_InstanceIndex"
_BASE_INSTANCE = "gl_BaseInstance"
_TARGETS = frozenset({"vulkan", "spirv"})


class InstanceIdUnsupportedError(TranslationDiagnostic):
    """An instance ID in a position the base-instance rewrite cannot reach."""

    def __init__(self, detail, target):
        super().__init__(
            f"{_INSTANCE_ID}: {detail}",
            target=target,
            code=f"{CODE_PREFIX}instance-id-unsupported",
        )


def _stage_kind(stage):
    value = getattr(stage, "stage", None)
    return str(getattr(value, "value", value) or "").lower()


class _InstanceLowering(_Lowering):
    builtin = _INSTANCE_ID
    index = _INSTANCE_INDEX
    base = _BASE_INSTANCE
    base_stem = "cgl_base_instance"

    def refuse_missing_base(self, where):
        return None

    def unsupported(self, detail):
        return InstanceIdUnsupportedError(detail, self.target)

    def lower_stage_variables(self, stage, functions):
        """Rebase the reads of stage-level ``gl_InstanceID`` inputs in ``functions``."""
        variables = list(getattr(stage, "local_variables", None) or [])
        carriers = [v for v in variables if _carries(v, self.builtin)]
        if not carriers:
            return
        taken = {getattr(v, "name", None) for v in variables}
        for function in functions:
            taken |= _names(function)
        base = next((v.name for v in variables if _carries(v, self.base)), None)
        if base is None:
            base = _fresh(taken, self.base_stem)
            stage.local_variables.append(
                A.VariableNode(
                    base,
                    A.PrimitiveType("int"),
                    attributes=[A.AttributeNode("input"), A.AttributeNode(self.base)],
                )
            )
        for variable in carriers:
            name = variable.name
            stem = "cgl_instance_index" if name.startswith("gl_") else f"{name}_index"
            index = _fresh(taken, stem)
            variable.name = index
            _respell(variable, self.builtin, self.index)
            value_type = variable.var_type
            for function in functions:
                if name in _names(function):
                    continue
                _respell_reads(
                    function,
                    lambda: _rebased(
                        copy.deepcopy(value_type), A.IdentifierNode(index), base
                    ),
                    name,
                )

    def run(self):
        stages = _stages(self.program)
        shared = []
        if not any(_stage_kind(stage).startswith("ray_") for stage in stages):
            shared = list(getattr(self.program, "functions", None) or [])
        for stage in stages:
            if _stage_kind(stage) != "vertex":
                continue
            entry = getattr(stage, "entry_point", None)
            functions = list(getattr(stage, "local_functions", None) or [])
            if entry is not None:
                functions.append(entry)
                if isinstance(getattr(entry, "body", None), A.BlockNode):
                    for member in self.lower_entry(entry):
                        _respell(member, self.builtin, self.index)
            self.lower_stage_variables(stage, functions + shared)
            for function in functions + shared:
                if self.builtin in _names(function):
                    continue
                _respell_reads(
                    function, lambda: self.read_by_name(function), self.builtin
                )


def lower_spirv_instance_ids(program, target) -> None:
    """Spell ``program``'s vertex ``gl_InstanceID`` as SPIR-V's instance index minus its base instance, in place."""
    if str(target or "").lower() not in _TARGETS or not isinstance(
        program, A.ShaderNode
    ):
        return
    _InstanceLowering(program, target).run()
