"""Colour attachment input structures of Metal fragment functions.

A Metal fragment function reads the colour attachments that earlier draws wrote to its pixel (programmable
blending) through an argument whose members carry ``[[color(N)]]``. CrossGL spells those members
``@gl_FragColor<N>``. Such an argument is not a stage input: it takes no attribute of its own, and a fragment
function that already has a ``[[stage_in]]`` argument cannot declare a second one (MSL specification section
5.2.3.4, "Fragment Function Input Attributes"). The emitter gave every struct argument without a semantic
``[[stage_in]]``, so a deferred-lighting fragment taking both its interpolated inputs and its G-buffer failed to
compile. :func:`is_metal_colour_attachment_input_parameter` recognises the colour-only argument so that it is
emitted without an attribute.
"""

from __future__ import annotations

import re

__all__ = ["is_metal_colour_attachment_input_parameter"]

_COLOUR_ATTACHMENT = re.compile(r"color\(\d+\)")


def is_metal_colour_attachment_input_parameter(codegen, parameter, shader_type):
    """Whether ``parameter`` of a fragment entry is a struct whose members all name colour attachments."""
    if shader_type != "fragment" or parameter is None:
        return False
    struct_name = codegen.metal_parameter_user_struct_type(parameter)
    if struct_name is None:
        return False
    members = getattr(codegen.structs_by_name.get(struct_name), "members", None) or []
    if not members:
        return False
    return all(_is_colour_attachment_member(codegen, member) for member in members)


def _is_colour_attachment_member(codegen, member):
    semantic = codegen.semantic_from_node(member)
    if semantic is None:
        return False
    canonical = codegen.canonical_metal_semantic(semantic)
    return (
        _COLOUR_ATTACHMENT.fullmatch(str(canonical or "").replace(" ", "")) is not None
    )
