"""Fragment shading-rate flag constants as values, and the barycentric inputs one Metal fragment function can take.

GL_EXT_fragment_shading_rate declares four ``const int`` flags that a fragment shader tests ``gl_ShadingRateEXT``
against: ``gl_ShadingRateFlag2VerticalPixelsEXT`` (1), ``gl_ShadingRateFlag4VerticalPixelsEXT`` (2),
``gl_ShadingRateFlag2HorizontalPixelsEXT`` (4) and ``gl_ShadingRateFlag4HorizontalPixelsEXT`` (8). Only GLSL declares
them, and the generators of the other targets printed the names verbatim. SPIR-V's ``FragmentShadingRate`` flags
(``Vertical2Pixels`` = 1 ... ``Horizontal4Pixels`` = 8) and HLSL's ``SV_ShadingRate`` encoding (``D3D12_SHADING_RATE``,
``log2(width) << 2 | log2(height)``) give the same bits the same meaning, so :func:`lower_shading_rate_flags`
replaces each flag with its value for every target but GLSL.

MSL takes one ``[[barycentric_coord]]`` input per fragment function, with one interpolation (MSL 3.2 section
5.2.3.3), while GLSL reads ``gl_BaryCoordEXT`` and ``gl_BaryCoordNoPerspEXT`` side by side.
:func:`refuse_mixed_metal_barycentrics` refuses a Metal fragment stage that reads both with
:class:`MetalBarycentricsUnsupportedError`.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .builtin_values import glsl_builtin_name
from .rewrite import child_slots, get_child, set_child
from .stage_varyings import int_literal

__all__ = [
    "MetalBarycentricsUnsupportedError",
    "lower_shading_rate_flags",
    "refuse_mixed_metal_barycentrics",
]

_GLSL_TARGETS = frozenset({"opengl", "glsl"})
SHADING_RATE_FLAGS = {
    "gl_ShadingRateFlag2VerticalPixelsEXT": 1,
    "gl_ShadingRateFlag4VerticalPixelsEXT": 2,
    "gl_ShadingRateFlag2HorizontalPixelsEXT": 4,
    "gl_ShadingRateFlag4HorizontalPixelsEXT": 8,
}
_PERSPECTIVE = frozenset({"gl_BaryCoordEXT", "gl_BaryCoordNV"})
_NO_PERSPECTIVE = frozenset({"gl_BaryCoordNoPerspEXT", "gl_BaryCoordNoPerspNV"})


class MetalBarycentricsUnsupportedError(TranslationDiagnostic):
    """A fragment stage reading barycentric coordinates with two interpolations, which MSL cannot take."""

    def __init__(self, stage):
        self.missing_capabilities = ("metal.barycentric-coord-interpolations",)
        super().__init__(
            f"the {stage} stage reads both perspective-correct (gl_BaryCoordEXT) and linear "
            "(gl_BaryCoordNoPerspEXT) barycentric coordinates, but a Metal fragment function takes one "
            "[[barycentric_coord]] input with one interpolation (MSL 3.2 section 5.2.3.3)",
            target="metal",
            code=f"{CODE_PREFIX}metal-barycentrics-unsupported",
        )


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _stage_name(stage):
    value = getattr(stage, "stage", None)
    return str(getattr(value, "value", value) or "").lower()


def _shadowed(program):
    """Names the program declares itself, which keep their meaning."""
    names = {v.name for v in getattr(program, "global_variables", None) or []}
    names |= {c.name for c in getattr(program, "constants", None) or []}
    for stage in _stages(program):
        names |= {v.name for v in getattr(stage, "local_variables", None) or []}
    return names


def _replace(node, replacements):
    count = 0
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        if isinstance(child, A.IdentifierNode) and child.name in replacements:
            set_child(node, field, key, int_literal(replacements[child.name]))
            count += 1
        elif isinstance(child, A.ASTNode):
            count += _replace(child, replacements)
    return count


def lower_shading_rate_flags(program, target) -> int:
    """Replace the GLSL shading-rate flag constants ``program`` reads with their values; returns how many."""
    if str(target or "").lower() in _GLSL_TARGETS or not isinstance(
        program, A.ShaderNode
    ):
        return 0
    flags = {
        name: value
        for name, value in SHADING_RATE_FLAGS.items()
        if name not in _shadowed(program)
    }
    return _replace(program, flags) if flags else 0


def _walk(node, seen):
    if not isinstance(node, A.ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key), seen)


def _barycentrics(program, stage):
    """The barycentric built-ins a fragment stage reads by name or receives as parameters or members."""
    found = set()
    structs = {s.name: s for s in getattr(program, "structs", None) or []}
    functions = [stage.entry_point] if getattr(stage, "entry_point", None) else []
    functions += list(getattr(stage, "local_functions", None) or [])
    functions += list(getattr(program, "functions", None) or [])
    for function in functions:
        for parameter in getattr(function, "parameters", None) or []:
            found |= _carried(parameter)
            struct = structs.get(str(getattr(parameter.param_type, "name", "")))
            for member in getattr(struct, "members", None) or []:
                found |= _carried(member)
        for node in _walk(getattr(function, "body", None), set()):
            if isinstance(node, A.IdentifierNode):
                found.add(node.name)
    return found & (_PERSPECTIVE | _NO_PERSPECTIVE)


def _carried(node):
    names = set()
    for attribute in getattr(node, "attributes", None) or []:
        name = glsl_builtin_name(attribute)
        if name is not None:
            names.add(name)
    return names


def refuse_mixed_metal_barycentrics(program, target) -> None:
    """Refuse a Metal fragment stage that reads barycentric coordinates with and without perspective."""
    if str(target or "").lower() != "metal" or not isinstance(program, A.ShaderNode):
        return
    for stage in _stages(program):
        if _stage_name(stage) not in {"fragment", "pixel"}:
            continue
        found = _barycentrics(program, stage)
        if found & _PERSPECTIVE and found & _NO_PERSPECTIVE:
            raise MetalBarycentricsUnsupportedError(_stage_name(stage))
