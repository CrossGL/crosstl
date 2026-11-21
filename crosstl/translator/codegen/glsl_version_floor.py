"""The ``#version`` line GLSL output starts from when the program has none, and the choices made against it.

GLSL output writes the lowest version that expresses the program. A program without a ``#version`` line of its
own starts from the lowest version of the output's profile, GLSL 3.30 (``#version 330 core``, the oldest desktop
version this generator writes), or GLSL ES 3.00 for GLSL ES output (:mod:`.glsl_source_directives`). Ray tracing
stages start from GLSL 4.60, which GL_EXT_ray_tracing requires. Once the program is rendered the generator raises
the line to the lowest version that has every construct it uses (:mod:`.glsl_version_requirements`) and renders it
again from there.

The emitter decides a few forms from the version while it renders: whether to write ``layout(binding)``, and
whether a half or bfloat16 conversion, a 64-bit bit cast or a floating remainder can be written at all. At the
floor it would otherwise choose the lesser form, or refuse, and the program would never leave the floor. While the
version may still rise (the program has no ``#version`` line and the profile does not fix the version, as WebGL 2
does), the emitter writes the full form and records the requirement, so resolution raises the line to a version
that has it. A program with its own ``#version`` line keeps today's choices against that line.
"""

from __future__ import annotations

from .glsl_version_requirements import GLSLRequirement, parse_version_line

__all__ = [
    "LOWEST_DESKTOP_VERSION_LINE",
    "RAY_TRACING_VERSION_LINE",
    "glsl_implicit_literal_conversion",
    "glsl_version_may_rise",
    "glsl_version_reaches",
]

LOWEST_DESKTOP_VERSION_LINE = "#version 330 core"
RAY_TRACING_VERSION_LINE = "#version 460 core"
# Desktop GLSL converts an int to a uint implicitly from GLSL 4.00, and to a float from GLSL 1.20 (GLSL 4.60
# section 4.1.10); GLSL ES converts nothing implicitly.
_IMPLICIT_INT_TO_UINT_VERSION = 400


def glsl_version_may_rise(codegen):
    """Whether the output's ``#version`` line is the generator's to choose: the program has none of its own."""
    return bool(
        getattr(codegen, "glsl_version_line_is_default", False)
    ) and not codegen.glsl_version_line_is_fixed()


def glsl_version_reaches(codegen, feature, desktop, es=None, desktop_extensions=()):
    """Whether the output can express ``feature``, first in GLSL ``desktop`` and GLSL ES ``es`` (``None``: never).

    True when the current version line or an enabled extension has it, and when the version may rise to one that
    has it, in which case the requirement is recorded for resolution.
    """
    number, profile = parse_version_line(
        getattr(codegen, "current_glsl_version_line", None)
    )
    if number is None:
        return True
    requirement = GLSLRequirement(feature, desktop, es, tuple(desktop_extensions))
    is_es = profile == "es"
    extensions = getattr(codegen, "current_glsl_extensions", ()) or ()
    if requirement.satisfied_by(number, is_es, extensions):
        return True
    if requirement.minimum(is_es) is None or not glsl_version_may_rise(codegen):
        return False
    recorded = getattr(codegen, "glsl_emission_requirements", None)
    if recorded is not None and requirement not in recorded:
        recorded.append(requirement)
    return True


def glsl_implicit_literal_conversion(codegen, actual_family, expected_family):
    """Whether the output converts a non-negative integer literal of ``actual_family`` implicitly.

    Only desktop GLSL converts implicitly: an int literal to a float at every version this generator writes, to a
    uint from GLSL 4.00. Below that version, and in GLSL ES, the literal is written in the expected type.
    """
    if actual_family == expected_family:
        return True
    if not codegen.GLSL_IMPLICIT_LITERAL_CONVERSIONS:
        return False
    if expected_family != "uint":
        return True
    version_line = getattr(codegen, "current_glsl_version_line", None)
    number, profile = parse_version_line(version_line)
    if number is None:
        return True
    return profile != "es" and number >= _IMPLICIT_INT_TO_UINT_VERSION
