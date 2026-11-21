"""The extension directives and the ``#version`` line a rendered GLSL program needs, added once it is complete.

Each step reads the finished program text, so it sees every construct the stages, helpers and preludes wrote.
The ES conservative-depth extension is required before the version requirements are resolved, so that the
version resolution sees it among the enabled extensions.
"""

from __future__ import annotations

from .glsl_builtin_extensions import require_builtin_extensions
from .glsl_fragment_depth_layout import require_conservative_depth_extension
from .glsl_shadow_lod import require_shadow_lod_extension

__all__ = ["finish_glsl_program_requirements"]


def finish_glsl_program_requirements(codegen, code: str) -> str:
    """``code`` with the extensions and the lowest ``#version`` that express what it uses."""
    code = require_conservative_depth_extension(
        code,
        codegen.glsl_version_is_es(getattr(codegen, "current_glsl_version_line", None)),
    )
    code = codegen.ensure_glsl_version_supports_generated_layouts(code)
    if getattr(codegen, "glsl_shadow_lod_extension_used", False):
        code = require_shadow_lod_extension(code)
    code = codegen.ensure_glsl_formatted_image_load_support(code)
    code = codegen.ensure_glsl_precise_support(code)
    code = require_builtin_extensions(
        code, getattr(codegen, "glsl_stage_builtin_extensions", ())
    )
    return codegen.ensure_glsl_int64_support(code)
