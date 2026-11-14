"""The Metal output a vertex or fragment entry point's return value is written to.

A Metal vertex function returns ``float4`` (the position) or a structure that contains the position, and a fragment
function that returns a scalar or vector returns colour attachment 0 (MSL 3.2 sections 5.2.3.3 and 5.2.3.4). Any
other output is a member of a returned structure carrying the output's attribute.

The return semantic decides which output the value is. ``@ gl_FragData[N]`` (HLSL ``SV_Target<N>``) and, on a
fragment entry, ``@ location(N)`` (GLSL ``layout(location = N) out``) are colour attachment ``N``. A semantic that names no output the stage can return is refused: emitting the function without it
would write the value to the position or to attachment 0, which is not what the source means.
"""

from __future__ import annotations

import re

from .fragment_output_targets import frag_data_attachment_index

__all__ = [
    "metal_entry_return_attribute",
    "metal_entry_return_semantic_key",
    "refuse_unreturnable_metal_entry_semantic",
]

# The scalar and vector types a colour attachment holds (MSL 3.2 section 5.2.3.4).
_COLOR_VALUE_TYPE = re.compile(r"(?:float|half|int|uint|short|ushort|char|uchar)[234]?")
_FRAGMENT_OUTPUT_KEYS = frozenset({"sample_mask", "stencil"})
_FRAGMENT_OUTPUT_SEMANTICS = frozenset(
    {"gl_FragDepth", "gl_FragStencilRefEXT", "gl_SampleMask"}
)


def metal_entry_return_semantic_key(generator, func, semantic, stage_name=None):
    """The Metal attribute key of ``func``'s return value.

    ``@ gl_FragData[N]`` is ``color(N)``, and so is ``@ location(N)`` on a fragment entry (a GLSL
    ``layout(location = N) out`` variable). Any other semantic is classified by
    :meth:`MetalCodeGen.canonical_metal_semantic`. ``gl_FragData`` without a valid index, and ``location`` outside
    a fragment entry, keep their own name, which names no Metal output.
    """
    if semantic is not None and str(semantic) == "gl_FragData":
        index = frag_data_attachment_index(func)
        if index is not None:
            return f"color({index})"
    if (
        semantic is not None
        and stage_name == "fragment"
        and str(semantic).lower() == "location"
    ):
        index = generator.explicit_location_attribute_index(func)
        if index is not None:
            return f"color({index})"
    return generator.canonical_metal_semantic(semantic)


def metal_entry_return_attribute(generator, semantic, key):
    """The attribute text of a returned value's structure member: ``[[color(N)]]`` for an indexed colour output."""
    if semantic is not None and str(semantic) in {"gl_FragData", "location"} and key:
        if key.startswith("color("):
            return f" [[{key}]]"
    return generator.map_semantic(semantic)


def _is_fragment_output(semantic, key):
    if semantic is None:
        return True
    if str(semantic) in _FRAGMENT_OUTPUT_SEMANTICS or key in _FRAGMENT_OUTPUT_KEYS:
        return True
    return bool(key) and (key.startswith("color(") or key.startswith("depth("))


def refuse_unreturnable_metal_entry_semantic(
    generator, func, stage_name, semantic, key, return_type, *, in_structure=False
):
    """Refuse a scalar or vector entry return whose semantic or type no Metal ``stage_name`` function returns.

    ``key`` is :func:`metal_entry_return_semantic_key` of ``semantic`` and ``return_type`` the mapped Metal return
    type. ``in_structure`` says the value becomes a member of a stage output structure (HLSL ``out`` parameters),
    where a vertex value may be a user output. Structure and ``void`` returns are the caller's to exclude. Returns
    whether a refusal was recorded.
    """
    function_name = getattr(func, "name", "<anonymous>")
    if stage_name == "vertex":
        if semantic is not None and key != "position" and in_structure:
            return False
        if semantic is not None and key != "position":
            generator.refuse_lowering(
                f"vertex function '{function_name}' returns a value with semantic '{semantic}', which is no "
                "Metal vertex output it can return: a Metal vertex function returns its float4 position or a "
                "structure containing it (MSL 3.2 section 5.2.3.3); return a structure instead"
            )
            return True
        if return_type != "float4":
            generator.refuse_lowering(
                f"vertex function '{function_name}' returns {return_type}: a Metal vertex function returns its "
                "float4 position or a structure containing it (MSL 3.2 section 5.2.3.3)"
            )
            return True
        return False
    if stage_name != "fragment":
        return False
    if not _is_fragment_output(semantic, key):
        generator.refuse_lowering(
            f"fragment function '{function_name}' returns a value with semantic '{semantic}', which is no Metal "
            "fragment output: a fragment function returns colour attachments, depth, the sample mask or the "
            "stencil reference (MSL 3.2 section 5.2.3.4)"
        )
        return True
    if semantic is None and _COLOR_VALUE_TYPE.fullmatch(str(return_type)) is None:
        generator.refuse_lowering(
            f"fragment function '{function_name}' returns {return_type}, which colour attachment 0 cannot "
            "hold: a colour attachment is a scalar or vector (MSL 3.2 section 5.2.3.4)"
        )
        return True
    return False
