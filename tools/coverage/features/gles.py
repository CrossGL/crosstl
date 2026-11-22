"""GLSL ES 3.20 built-in functions, from the same specification source as GLSL's (KhronosGroup/GLSL).

The source builds the GLSL 4.60 and the GLSL ES 3.20 specifications: ``ifdef::ESSL[]`` marks the text only GLSL ES
has, and ``ifdef::GLSL[]`` the text only desktop GLSL has. This list keeps the first and drops the second, so it has
every built-in function of GLSL ES 3.20, the shading language of OpenGL ES 3.2, in every stage it has. Subgroup
functions come from GL_KHR_shader_subgroup, which applies to GLSL ES 3.10 and later as it does to desktop GLSL.
"""

from __future__ import annotations

from ..model import Feature
from ..objects import UNIFORMS
from . import glsl, glsl_subgroup
from .common import pick

LANGUAGE = "gles"


def overloads(text: str) -> dict[str, tuple[str, list]]:
    """Every GLSL ES prototype, by function name, as :func:`.glsl.overloads` gives them.

    A function desktop GLSL lists too is kept only in the section the desktop text puts it in. The source opens the
    Shader Invocation Group Functions section, which GLSL ES does not have, inside ``ifdef::GLSL[]`` and leaves its
    subsections outside, so the GLSL ES text would list anyInvocation() under Subpass-Input Functions.
    """
    desktop = glsl.overloads(text)
    return {
        name: found
        for name, found in glsl.overloads(text, essl_text=True).items()
        if name not in desktop or desktop[name][0] == found[0]
    }


def parse(text: str, *subgroup_texts: str) -> list[Feature]:
    functions = [
        Feature(
            LANGUAGE,
            "function",
            name,
            pick(signatures, UNIFORMS[LANGUAGE]),
            group=group,
            crossgl=name,
            stage=glsl.stage(name, group),
            target=glsl.TARGETS.get(group, ""),
        )
        for name, (group, signatures) in sorted(overloads(text).items())
    ]
    return functions + glsl_subgroup.parse(*subgroup_texts, language=LANGUAGE)
