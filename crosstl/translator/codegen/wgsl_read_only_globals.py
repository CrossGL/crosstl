"""Refuse writes to module-scope values that WGSL cannot write.

WGSL ``const`` declarations and ``var<uniform>`` variables are read-only
(WGSL 7.2 and 7.3, "read" access mode of the uniform address space). A source
statement that assigns to one, or passes one to an ``inout``/``out``
parameter, is rejected by every source language as well, so the generator
raises ``ValueError`` instead of emitting an assignment that WGSL rejects.

Read-only storage buffers are refused by ``validate_storage_assignment_target``
through their access mode.
"""

from __future__ import annotations

from ..refusals import WGSLRefusal
from .wgsl_parameter_lowering import lvalue_root

READ_ONLY_QUALIFIERS = frozenset({"uniform", "const", "constant"})
WRITABLE_QUALIFIERS = frozenset({"buffer", "storage", "workgroup", "shared"})


def read_only_module_names(codegen, cbuffers, global_variables, stage_parameters):
    """Names of module-scope uniforms, constants and constant-buffer members."""
    names = set()
    for cbuffer in cbuffers:
        instance = codegen.cbuffer_instance_name(cbuffer)
        names.update(name for name in (getattr(cbuffer, "name", ""), instance) if name)
        for member in getattr(cbuffer, "members", None) or ():
            if getattr(member, "name", ""):
                names.add(member.name)
    for variable in global_variables:
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(variable, "qualifiers", None) or ()
        }
        if qualifiers & READ_ONLY_QUALIFIERS and not qualifiers & WRITABLE_QUALIFIERS:
            names.add(variable.name)
    for parameter in stage_parameters:
        if codegen.stage_uniform_parameter_type(parameter) is not None:
            names.add(parameter.name)
    return names


def validate_writable_module_root(codegen, target, action="write '{name}'"):
    """Raise when ``target`` designates part of a read-only module-scope value."""
    root = lvalue_root(target)
    if root is None:
        return
    name = root.name
    if name not in getattr(codegen, "_read_only_module_names", ()):
        return
    if codegen.is_local_identifier(name) or codegen.is_current_parameter_name(name):
        return
    raise WGSLRefusal(
        f"WGSL target cannot {action.format(name=name)}: it is a module-scope "
        "uniform or constant, which WGSL makes read-only"
    )
