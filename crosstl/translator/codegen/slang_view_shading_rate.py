"""Stage and direction rules for the multiview and shading-rate built-ins in Slang output.

CrossGL binds these as GLSL does (GL_EXT_multiview, GL_EXT_fragment_shading_rate), and the generator writes each as
the Slang system value slangc lowers to the matching SPIR-V built-in:

* ``gl_ViewIndex`` is ``SV_ViewID`` (``ViewIndex``), an input of the vertex, tessellation, geometry and fragment
  stages. Vulkan forbids ``ViewIndex`` in compute (VUID-ViewIndex-ViewIndex-04401).
* ``gl_ShadingRateEXT`` is the fragment input ``SV_ShadingRate`` (``ShadingRateKHR``, fragment stage only:
  VUID-ShadingRateKHR-ShadingRateKHR-04490).
* ``gl_PrimitiveShadingRateEXT`` is an ``SV_ShadingRate`` output (``PrimitiveShadingRateKHR``) of the vertex,
  geometry and mesh stages (VUID-PrimitiveShadingRateKHR-PrimitiveShadingRateKHR-04484).

Every one is a 32-bit integer scalar. ``SV_ShadingRate`` names the fragment input and the primitive output alike, so
an input bound to ``gl_PrimitiveShadingRateEXT`` would read the fragment rate instead: it is refused, as is any of
these in a stage or direction that lacks it. The output direction is checked with the other output semantics
(``SlangSemanticsMixin.validate_slang_output_semantic_stage``); this module checks entry inputs, both parameters and
the members of a structure an entry parameter takes.
"""

from __future__ import annotations

from ..refusals import SlangRefusal

__all__ = [
    "SLANG_PRIMITIVE_SHADING_RATE_STAGES",
    "validate_slang_view_shading_rate_inputs",
]

# The stages each input built-in exists in, by lower-cased CrossGL name.
_INPUT_STAGES = {
    "gl_viewindex": frozenset({"vertex", "hull", "domain", "geometry", "fragment"}),
    "gl_shadingrateext": frozenset({"fragment"}),
}
_OUTPUT_ONLY = frozenset({"gl_primitiveshadingrateext"})

# The stages that write gl_PrimitiveShadingRateEXT.
SLANG_PRIMITIVE_SHADING_RATE_STAGES = frozenset({"vertex", "geometry", "mesh"})

_INTEGER_SCALARS = frozenset({"int", "uint"})


def _bindings(codegen, parameter):
    """``(label, semantic, Slang type)`` for ``parameter`` and the members of the structure it takes."""
    semantic = codegen.semantic_from_node(parameter)
    type_name = codegen.slang_parameter_type_name(parameter)
    name = getattr(parameter, "name", "<parameter>")
    if semantic is not None:
        yield f"parameter '{name}'", semantic, codegen.convert_type(type_name)
        return
    base_type = codegen.type_name_string(type_name) or ""
    base_type = base_type.split("<", 1)[0].split("[", 1)[0].strip()
    struct = codegen.user_structs_by_name.get(base_type)
    for member in getattr(struct, "members", None) or []:
        member_semantic = codegen.semantic_from_node(member)
        if member_semantic is None:
            continue
        member_type = codegen.convert_type(
            codegen.slang_tess_factor_member_type_name(member)
        )
        yield f"struct member '{base_type}.{member.name}'", member_semantic, member_type


def validate_slang_view_shading_rate_inputs(codegen, shader_type, parameters):
    """Refuse an entry input bound to a multiview or shading-rate built-in its stage, direction or type lacks."""
    stage = codegen.slang_shader_stage_name(shader_type)
    for parameter in parameters or []:
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", None) or []
        }
        if "out" in qualifiers:
            continue
        for label, semantic, slang_type in _bindings(codegen, parameter):
            lower = str(semantic).lower()
            if lower in _OUTPUT_ONLY:
                raise SlangRefusal(
                    f"Unsupported {semantic} {label} for Slang {shader_type} stage; "
                    "output-only builtin semantics cannot be used as inputs"
                )
            stages = _INPUT_STAGES.get(lower)
            if stages is None:
                continue
            if stage not in stages:
                valid = ", ".join(sorted(stages))
                raise SlangRefusal(
                    f"Unsupported {semantic} {label} for Slang {shader_type} stage; "
                    f"valid stage is {valid}"
                )
            if str(slang_type) not in _INTEGER_SCALARS:
                raise SlangRefusal(
                    f"Unsupported {semantic} {label} for Slang codegen; "
                    "expected int or uint type"
                )
