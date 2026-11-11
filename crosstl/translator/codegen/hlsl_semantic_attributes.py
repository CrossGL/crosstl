"""HLSL semantics for CrossGL semantic attributes, including those that carry an argument.

An HLSL semantic is an identifier whose trailing digits are its index (HLSL reference, "Semantics"); the ``SV_``
prefix is reserved for system values. CrossGL writes some interface semantics as attributes with an argument, and
each has one HLSL spelling:

- ``@location(N)`` on an entry parameter is user varying ``N``. It becomes ``TEXCOORDN``, the spelling stage
  structure members with ``@location(N)`` receive, so that both ends of a stage interface agree; a pixel-shader
  output is render target ``N``, ``SV_TargetN``.
- ``@user(name)`` (Metal ``[[user(name)]]``, matched between stages by name) becomes the user semantic ``name``.

Any other semantic attribute with an argument, and any text that is not an identifier, is refused rather than
written with its argument dropped.
"""

from __future__ import annotations

import re

from ..ast import ParameterNode
from .stage_utils import normalize_stage_name

__all__ = [
    "HLSL_SEMANTIC_NAME",
    "hlsl_argument_semantic_name",
    "hlsl_parameter_location_semantic",
    "hlsl_unmapped_semantic_name",
]

# An HLSL semantic: an identifier, whose trailing digits are the semantic index (HLSL "Semantics").
HLSL_SEMANTIC_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _attribute_arguments(attr):
    return list(getattr(attr, "arguments", None) or getattr(attr, "args", None) or [])


def _argument_text(argument):
    for field in ("name", "value"):
        value = getattr(argument, field, None)
        if isinstance(value, str):
            return value
    if isinstance(argument, str):
        return argument
    return None


def hlsl_unmapped_semantic_name(codegen, semantic_text):
    """The HLSL spelling of a semantic ``semantic_map`` does not rename: the text as written, if it is one.

    Such a semantic is a user semantic, an ``SV_`` system value (which the built-in value pass has checked against
    the kinds DXIL defines) or a CrossGL ``gl_`` built-in that HLSL has no system value for, which
    ``refuse_unmapped_hlsl_builtin`` refuses where it is emitted. Any other text is refused rather than written
    after a colon.
    """
    if HLSL_SEMANTIC_NAME.fullmatch(semantic_text):
        return semantic_text
    return codegen.refuse_lowering(
        f"'{semantic_text}' is not an HLSL semantic; a semantic is an identifier "
        "with an optional numeric index",
        semantic_text,
    )


def hlsl_user_semantic_name(codegen, attr):
    """The user semantic ``@user(name)`` names: ``name`` itself, when HLSL can spell it as a user semantic."""
    arguments = _attribute_arguments(attr)
    name = _argument_text(arguments[0]) if len(arguments) == 1 else None
    if name is None or not HLSL_SEMANTIC_NAME.fullmatch(name):
        return codegen.refuse_lowering(
            "'@user' needs one identifier argument to name an HLSL user semantic",
            "user",
        )
    if name.lower().startswith("sv_"):
        return codegen.refuse_lowering(
            f"'@user({name})' cannot be an HLSL user semantic: the SV_ prefix is "
            "reserved for system values",
            name,
        )
    return name


def hlsl_argument_semantic_name(codegen, attr):
    """The semantic text of a CrossGL semantic attribute other than ``gl_FragData``.

    ``@location(N)`` stays ``location``: stage structure members replace it with the varying slot it names, and
    entry parameters take :func:`hlsl_parameter_location_semantic` instead.
    """
    name = getattr(attr, "name", None)
    arguments = _attribute_arguments(attr)
    if not arguments or str(name).lower() == "location":
        return name
    if name == "user":
        return hlsl_user_semantic_name(codegen, attr)
    return codegen.refuse_lowering(
        f"semantic '@{name}' takes no argument in HLSL; its argument would be dropped",
        name,
    )


def hlsl_parameter_location_semantic(codegen, parameter, attr):
    """The HLSL semantic of an entry parameter's ``@location(N)``, or ``None`` when ``parameter`` is not one.

    A pixel-shader output is render target ``N`` (``SV_TargetN``); every other parameter is user varying ``N``
    (``TEXCOORDN``), which matches the members of a stage structure with the same location. A pixel-shader
    ``inout`` parameter would need ``SV_Target`` as an input, which HLSL does not define, so it is refused.
    """
    if not isinstance(parameter, ParameterNode):
        return None
    index = codegen.hlsl_location_attribute_index(attr)
    if index is None or index < 0:
        return codegen.refuse_lowering(
            f"'@location' on parameter '{getattr(parameter, 'name', '<anonymous>')}' "
            "needs a non-negative integer index",
            "location",
        )
    directions = codegen.hlsl_parameter_signature_directions(parameter)
    if normalize_stage_name(codegen.current_hlsl_semantic_stage) == "fragment" and (
        "out" in directions
    ):
        if "in" in directions:
            return codegen.refuse_lowering(
                f"fragment parameter '{getattr(parameter, 'name', '<anonymous>')}' is "
                "inout at a location; HLSL cannot read a render target through "
                "SV_Target",
                f"SV_Target{index}",
            )
        return f"SV_Target{index}"
    return f"TEXCOORD{index}"
