"""Device address-space variables, refused for targets without program-scope device memory.

A CrossGL variable in the ``device`` address space, ``device T x;`` or ``var<device> x: T;`` at shader or stage scope,
is one object in device memory that every invocation of every launch reads and writes. CUDA and HIP declare it as a
``__device__`` variable (CUDA C++ Programming Guide, "Variable Memory Space Specifiers"), and CrossGL output keeps it
as written. No other target has such a variable:

* MSL requires a program-scope variable to be in the ``constant`` address space (MSL, "constant Address Space"), so
  ``device float x[64];`` at program scope does not compile;
* an HLSL or Slang global without ``static`` is a uniform that DXC and slangc place in the read-only ``$Globals``
  constant buffer, and a GLSL ``uniform`` is read-only too;
* a WGSL ``var<private>``, a SPIR-V ``Private`` variable and a GLSL global without a storage qualifier have one copy
  per invocation (WGSL, "Address Spaces"; SPIR-V, "Storage Class");
* Rust and Mojo output has no device-memory variable either.

The generators wrote such a variable as a uniform, as a per-invocation copy or not at all, so the output either failed
to compile or gave each invocation its own copy. For every target but CUDA, HIP and CrossGL,
:func:`refuse_device_variables` raises :class:`DeviceVariableUnsupportedError` for a ``device`` variable whose type is
a plain value: a scalar, vector, matrix, sized array or structure of plain values. A ``device`` pointer, a resource and
an unsized array are not refused here; the pointer and resource forms bind a buffer, and each generator handles or
refuses the others itself.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .stage_private_variables import _aliases, _is_value_type

__all__ = [
    "DEVICE_VARIABLE_TARGETS",
    "DeviceVariableUnsupportedError",
    "refuse_device_variables",
]

# Targets that declare a device address-space variable: CUDA and HIP as ``__device__``, CrossGL as written.
DEVICE_VARIABLE_TARGETS = frozenset({"", "cgl", "crossgl", "cuda", "hip"})


class DeviceVariableUnsupportedError(TranslationDiagnostic):
    """A ``device`` address-space variable that the target has no program-scope device memory for."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.variable_name = name
        self.missing_capabilities = (f"{target}.device-variable",)
        super().__init__(
            f"device variable '{name}' is one object shared by every invocation, and {target} output has no "
            "program-scope device memory to hold it; bind it as a buffer instead",
            target=target,
            code=f"{CODE_PREFIX}{target}-device-variable-unsupported",
        )


def _lowered(values) -> set[str]:
    return {str(getattr(value, "name", value)).lower() for value in values or []}


def _is_device(variable) -> bool:
    return "device" in _lowered(getattr(variable, "qualifiers", None)) or (
        "device" in _lowered(getattr(variable, "address_space_qualifiers", None))
    )


# CrossGL also spells scalars the WGSL way, as in ``var<device> x: array<f32, 64>``.
_WGSL_SCALARS = frozenset({"f16", "f32", "f64", "i32", "u32", "i64", "u64"})


def _is_plain_value(type_node, structs, aliases) -> bool:
    if isinstance(type_node, A.PrimitiveType) and type_node.name in _WGSL_SCALARS:
        return True
    if isinstance(type_node, (A.VectorType, A.MatrixType)):
        return _is_plain_value(type_node.element_type, structs, aliases)
    if isinstance(type_node, A.ArrayType) and type_node.size is not None:
        return _is_plain_value(type_node.element_type, structs, aliases)
    return _is_value_type(type_node, structs, aliases, set())


def _structs(declarations) -> dict:
    return {
        getattr(struct, "name", None): struct
        for struct in declarations or []
        if isinstance(struct, A.StructNode)
    }


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if hasattr(stages, "values") else list(stages)


def _scopes(program):
    """Each scope's variables with the structures and aliases its types resolve against."""
    globals_ = list(getattr(program, "global_variables", None) or [])
    program_structs = _structs(getattr(program, "structs", None))
    program_aliases = _aliases(globals_, {})
    yield globals_, program_structs, program_aliases
    for stage in _stages(program):
        variables = list(getattr(stage, "local_variables", None) or [])
        structs = {
            **program_structs,
            **_structs(getattr(stage, "local_structs", None)),
        }
        yield variables, structs, _aliases(variables, program_aliases)


def refuse_device_variables(program, target) -> None:
    """Raise :class:`DeviceVariableUnsupportedError` for a plain-value ``device`` variable ``target`` cannot hold."""
    target = str(target or "").lower()
    if target in DEVICE_VARIABLE_TARGETS or not isinstance(program, A.ShaderNode):
        return
    for variables, structs, aliases in _scopes(program):
        for variable in variables:
            if type(variable) is not A.VariableNode or not _is_device(variable):
                continue
            if getattr(variable, "is_type_alias", False):
                continue
            if _is_plain_value(getattr(variable, "var_type", None), structs, aliases):
                raise DeviceVariableUnsupportedError(
                    target,
                    variable.name,
                    source_location=getattr(variable, "source_location", None),
                )
