"""Module-scope ``in`` and ``out`` declarations as the stage-level interface of the program's graphics stage.

A GLSL fragment shader with several outputs (``layout(location = 1) out vec4 outNormal;``, one per colour attachment
of a G-buffer pass) is imported as module-scope ``out`` declarations that its ``main`` and helpers write. A
shader that writes an output array, ``gl_FragDepth`` or ``gl_SampleMask`` is imported the same way. GLSL and SPIR-V
declare stage outputs that way too. HLSL and MSL have no module-scope outputs: the HLSL generator emitted them as
globals, which HLSL makes constant, and the Metal generator refused the stores.

CrossGL written the GLSL way declares stage inputs at module scope too (``in vec3 position @location(0);``). HLSL
and MSL read a module-scope variable as a constant: the HLSL generator declared the input as a global, which HLSL
makes a uniform, and the Metal generator as a ``constant`` initialized to zero, so the stage read 0 instead of its
input.

For those targets :func:`adopt_module_stage_outputs` moves each module-scope ``in`` and ``out`` declaration into the
program's one vertex or fragment stage as a stage-level ``@ input`` or ``@ output`` variable, keeping its location,
interpolation and built-in attributes.
A program with a geometry, tessellation, mesh or compute stage is left to the lowerings of those stages.
:mod:`.stage_interface_structs` then gathers the stage's inputs into the structure the entry point takes and its outputs into the
structure it returns, and passes them to the helpers that read or write them. A program with module-scope inputs or outputs and more than
one vertex or fragment stage does not say which stage they belong to, and is refused with
:class:`~.stage_interface_structs.StageInterfaceUnsupportedError`.
"""

from __future__ import annotations

from .. import ast as A
from .stage_interface_structs import StageInterfaceUnsupportedError

__all__ = ["adopt_module_stage_outputs"]

TARGETS = frozenset(
    {"metal", "directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "slang"}
)
_GRAPHICS = frozenset({"vertex", "fragment"})


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _stage_name(stage):
    value = getattr(stage, "stage", None)
    return str(getattr(value, "value", value) or "").lower()


_DIRECTIONS = {"in": "input", "out": "output"}
_NOT_INTERFACE = frozenset({"uniform", "buffer", "shared", "const", "static", "inout"})


def _direction(variable):
    """``in`` or ``out`` for a module-scope stage input or output, else None."""
    qualifiers = {
        str(qualifier).lower()
        for qualifier in getattr(variable, "qualifiers", None) or []
    }
    if qualifiers & _NOT_INTERFACE:
        return None
    directions = [direction for direction in _DIRECTIONS if direction in qualifiers]
    return directions[0] if len(directions) == 1 else None


def adopt_module_stage_outputs(program, target) -> int:
    """Move ``program``'s module-scope ``in`` and ``out`` declarations into its graphics stage; returns how many."""
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    outputs = [
        variable
        for variable in getattr(program, "global_variables", None) or []
        if isinstance(variable, A.VariableNode) and _direction(variable) is not None
    ]
    if not outputs:
        return 0
    all_stages = _stages(program)
    if any(_stage_name(stage) not in _GRAPHICS for stage in all_stages):
        # Geometry, tessellation and mesh stages write module-scope outputs through their own lowerings.
        return 0
    stages = all_stages
    if len(stages) != 1:
        raise StageInterfaceUnsupportedError(
            f"module-scope {_DIRECTIONS[_direction(outputs[0])]} '{outputs[0].name}' belongs to no single vertex "
            f"or fragment stage (the program has {len(stages)}); {target} entry points take their inputs and "
            "return their outputs as structures",
            target=target,
            reason="module-output-stage",
        )
    stage = stages[0]
    moved = {id(variable) for variable in outputs}
    program.global_variables = [
        variable for variable in program.global_variables if id(variable) not in moved
    ]
    for variable in outputs:
        direction = _direction(variable)
        variable.qualifiers = [
            qualifier
            for qualifier in variable.qualifiers or []
            if str(qualifier).lower() != direction
        ]
        variable.attributes = list(variable.attributes or []) + [
            A.AttributeNode(_DIRECTIONS[direction])
        ]
    stage.local_variables = (
        list(getattr(stage, "local_variables", None) or []) + outputs
    )
    return len(outputs)
