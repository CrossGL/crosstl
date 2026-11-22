"""The multiview view index ``gl_ViewIndex`` in OpenGL and WebGL output.

CrossGL reads the view index as ``gl_ViewIndex`` (HLSL ``SV_ViewID``, Metal ``amplification_id``), an ``int``.
GLSL has two spellings of it:

- GL_EXT_multiview declares ``highp int gl_ViewIndex`` in the vertex, tessellation, geometry and fragment languages
  and "can be applied to OpenGL GLSL versions 1.40 (#version 140) and higher". Vulkan GLSL and OpenGL SPIR-V
  (``GL_SPIRV``) use it; no OpenGL driver compiles it from source.
- GL_OVR_multiview2 declares ``uint gl_ViewID_OVR`` in the vertex and fragment languages of OpenGL and of OpenGL ES
  3.00, which is also a WebGL 2 extension. The view count, ``layout(num_views = N) in;``, is a vertex shader
  qualifier; "the behavior is as if it had been set to 1" in a vertex shader that does not declare it.

So a program whose view index is read only by fragment stages reads it through ``CROSSGL_VIEW_INDEX``, which the
prelude defines as ``gl_ViewIndex`` under GL_EXT_multiview for Vulkan and OpenGL SPIR-V and as
``int(gl_ViewID_OVR)`` under GL_OVR_multiview2 otherwise; WebGL output reads ``int(gl_ViewID_OVR)`` directly. A program whose geometry,
tessellation or mesh stages read it keeps ``gl_ViewIndex`` under GL_EXT_multiview, the only extension that declares
it there.

The view index is refused in a vertex stage, whose GL_OVR_multiview view count the source does not state, and in
any stage GLSL does not declare it for. A line outside every stage guard of a multi-stage program counts as read by
each of the program's stages.
"""

from __future__ import annotations

import re

from .stage_utils import normalize_stage_name

__all__ = ["VIEW_INDEX_MACRO", "lower_view_index"]

VIEW_INDEX = "gl_ViewIndex"
VIEW_INDEX_MACRO = "CROSSGL_VIEW_INDEX"
_OVR_VIEW_INDEX = "int(gl_ViewID_OVR)"
_EXT_MULTIVIEW = "#extension GL_EXT_multiview : require"
_OVR_MULTIVIEW = "#extension GL_OVR_multiview2 : require"

_FRAGMENT_PRELUDE = (
    "#if defined(GL_SPIRV) || defined(VULKAN)",
    _EXT_MULTIVIEW,
    f"#define {VIEW_INDEX_MACRO} {VIEW_INDEX}",
    "#else",
    _OVR_MULTIVIEW,
    f"#define {VIEW_INDEX_MACRO} {_OVR_VIEW_INDEX}",
    "#endif",
)

# The stages other than vertex whose GLSL language declares gl_ViewIndex (GL_EXT_multiview; GL_EXT_mesh_shader for
# mesh shaders, which glslang declares it in, unlike task shaders).
_EXT_STAGES = frozenset(
    {"fragment", "geometry", "tessellation_control", "tessellation_evaluation", "mesh"}
)
_VERTEX_REASON = (
    "a vertex shader reads the multiview view index, and GL_OVR_multiview needs its view count "
    "(layout(num_views)) there, which the source does not state"
)

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_VIEW_INDEX_WORD = re.compile(rf"\b{VIEW_INDEX}\b")
_DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b\s*(\w*)")


def _reading_stages(code, guards, program_stages):
    """The stages that read the view index in rendered ``code``, from the stage guards around each read."""
    guard_stages = {}
    for stage, macro in guards.items():
        # One macro guards task, amplification and object stages; the first names them.
        if macro:
            guard_stages.setdefault(macro, stage)
    stack = []
    stages = set()
    for line in _COMMENT.sub(" ", code).splitlines():
        directive = _DIRECTIVE.match(line)
        if directive:
            keyword, name = directive.groups()
            if keyword in {"if", "ifdef", "ifndef"}:
                stack.append(guard_stages.get(name) if keyword == "ifdef" else None)
            elif keyword in {"elif", "else"} and stack:
                stack[-1] = None
            elif keyword == "endif" and stack:
                stack.pop()
            continue
        if not _VIEW_INDEX_WORD.search(line):
            continue
        guarded = [stage for stage in stack if stage is not None]
        stages |= {guarded[-1]} if guarded else set(program_stages)
    return stages


def _with_prelude(code, lines):
    """``code`` with ``lines`` after its ``#version`` line (or first, when it has none)."""
    head, newline, rest = code.partition("\n")
    prelude = "\n".join(lines)
    if head.lstrip().startswith("#version"):
        return f"{head}\n{prelude}\n{rest}" if newline else f"{head}\n{prelude}\n"
    return f"{prelude}\n{code}"


def lower_view_index(codegen, ast, target_stage, code):
    """Spell the view index of rendered ``code`` for the target profile, or refuse the stages that cannot read it."""
    if not isinstance(code, str) or not _VIEW_INDEX_WORD.search(
        _COMMENT.sub(" ", code)
    ):
        return code
    program_stages = {
        normalize_stage_name(stage)
        for stage in codegen.glsl_stage_names(ast, target_stage)
    }
    guards = {
        stage: codegen.glsl_stage_guard_macro(stage)
        for stage in codegen.GLSL_STAGE_GUARD_MACROS
    }
    stages = _reading_stages(code, guards, program_stages)
    webgl = codegen.GLSL_TARGET_DISPLAY_NAME == "WebGL"
    allowed = {"fragment"} if webgl else _EXT_STAGES
    refused = False
    for stage in sorted(stages - allowed):
        refused = True
        reason = (
            _VERTEX_REASON
            if stage == "vertex"
            else f"GLSL declares the multiview view index in no {stage} shader"
        )
        codegen.refuse_lowering(f"'{VIEW_INDEX}': {reason}")
    if refused:
        return code
    if webgl:
        return _with_prelude(
            _VIEW_INDEX_WORD.sub(_OVR_VIEW_INDEX, code), (_OVR_MULTIVIEW,)
        )
    if stages <= {"fragment"}:
        return _with_prelude(
            _VIEW_INDEX_WORD.sub(VIEW_INDEX_MACRO, code), _FRAGMENT_PRELUDE
        )
    if _EXT_MULTIVIEW in code:
        return code
    return _with_prelude(code, (_EXT_MULTIVIEW,))
