"""SPIR-V ``WorkgroupSize`` built from the specialization constants a compute stage names.

``layout(local_size_x_id = N) in;`` makes the x dimension the specialization constant with SpecId ``N``. SPIR-V has
no specialization of the ``LocalSize`` execution mode before 1.2's ``LocalSizeId``; the portable form, which glslang
writes too, is an ``OpSpecConstantComposite`` of the three dimensions decorated ``BuiltIn WorkgroupSize``. Its value
overrides ``LocalSize`` (SPIR-V 1.6 section 3.21, BuiltIn WorkgroupSize), so the module keeps ``LocalSize`` at the
defaults and always declares the composite, even when the shader never reads ``gl_WorkGroupSize``: without it, a
specialized size would not reach the dispatch. A dimension's default is ``local_size_x`` (or y, z), or 1.

Compute derivatives need even x and y dimensions, which a specialized dimension does not guarantee, so that
combination is refused.
"""

from __future__ import annotations

from .specialized_workgroup_size import (
    has_specialized_workgroup_size,
    workgroup_size_specialization_ids,
)

__all__ = [
    "declare_specialized_workgroup_size",
    "module_compute_stage_config",
    "stage_has_specialized_workgroup_size",
    "workgroup_size_components",
]

_TARGET = "SPIR-V"


def _stage_config(generator, stage):
    if stage is None:
        return generator.module_compute_entry_config or {}
    return getattr(stage, "execution_config", {}) or {}


def stage_has_specialized_workgroup_size(generator, stage) -> bool:
    """Whether the pipeline sets any workgroup dimension of ``stage``."""
    return has_specialized_workgroup_size(_stage_config(generator, stage))


def workgroup_size_components(generator, stage, uint_type):
    """The three ``uint`` dimensions of the workgroup, specialization constants where the stage names an id."""
    defaults = generator.compute_local_size(stage)
    ids = workgroup_size_specialization_ids(_stage_config(generator, stage), _TARGET)
    components = []
    for default, spec_id in zip(defaults, ids):
        if spec_id is None:
            components.append(generator.register_constant(default, uint_type))
        else:
            components.append(
                generator.register_specialization_constant(default, uint_type, spec_id)
            )
    return components, any(spec_id is not None for spec_id in ids)


def module_compute_stage_config(generator, ast):
    """The execution metadata of the shader's one compute stage, for built-ins read outside any stage.

    A helper function outside the stage that reads ``gl_WorkGroupSize`` is generated before the stage, so the
    composite it declares must already have the stage's size; ``WorkgroupSize`` overrides ``LocalSize``.
    """
    configs = [
        getattr(stage, "execution_config", None) or {}
        for stage_type, stage in (getattr(ast, "stages", None) or {}).items()
        if generator.stage_key(stage_type) == "compute"
    ]
    return dict(configs[0]) if len(configs) == 1 else None


def declare_specialized_workgroup_size(generator, stage) -> None:
    """Declare the ``WorkgroupSize`` composite of a stage whose size the pipeline specializes."""
    if not stage_has_specialized_workgroup_size(generator, stage):
        return
    if generator.requires_compute_derivatives:
        generator.refuse_lowering(
            "SPIR-V compute derivatives need even workgroup x and y dimensions, which a workgroup size "
            "the pipeline specializes (local_size_x_id, local_size_y_id) does not guarantee"
        )
        return
    # The built-in takes its size from the current stage; the entry point's stage owns these execution modes.
    previous_stage = generator.current_stage
    generator.current_stage = stage
    try:
        generator.ensure_compute_builtin("gl_WorkGroupSize")
    finally:
        generator.current_stage = previous_stage
