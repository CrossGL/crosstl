"""Conservative depth layouts on the fragment depth output, however CrossGL declares it.

GLSL states a conservative depth promise only by redeclaring the built-in: ``layout(depth_less) out float
gl_FragDepth;`` (GLSL 4.60 section 4.4.2.3, GL_ARB_conservative_depth, and GL_EXT_conservative_depth for GLSL ES
3.00 and later). A depth layout on any other output is a compile error ("can only apply depth layout to
gl_FragDepth").

CrossGL puts the layout attribute (``@depth_any``, ``@depth_greater``, ``@depth_less``, ``@depth_unchanged``) on
whatever declares the depth output:

- a global ``out float gl_FragDepth @depth_less;``, the GLSL importer's form, which the global declaration path
  already writes as the redeclaration;
- a stage-level ``float depth @output @gl_FragDepth @depth_less;``, the SPIR-V importer's form;
- a member of the structure a fragment entry returns, ``float depth @gl_FragDepth @depth_greater;``, the HLSL
  importer's form, an entry's ``out`` parameter, or the entry's return value.

The generator writes each of the last three as the ``gl_FragDepth`` built-in. This module writes the redeclaration
that carries their layout, once per stage, and refuses a depth layout on anything that is not the fragment depth
output, two layouts on one output, and two outputs that promise different layouts.
"""

from __future__ import annotations

import re

from ..refusals import OpenGLRefusal

__all__ = [
    "CONSERVATIVE_DEPTH_EXTENSION",
    "DEPTH_LAYOUTS",
    "depth_layout",
    "entry_depth_redeclarations",
    "is_fragment_depth_output",
    "require_conservative_depth_extension",
    "stage_depth_redeclaration",
]

DEPTH_LAYOUTS = frozenset(
    {"depth_any", "depth_greater", "depth_less", "depth_unchanged"}
)
# GLSL ES has the depth layouts only through this extension (EXT_conservative_depth, written against GLSL ES 3.00);
# WebGL 2 exposes it as the EXT_conservative_depth WebGL extension.
CONSERVATIVE_DEPTH_EXTENSION = "GL_EXT_conservative_depth"
_FRAG_DEPTH = "gl_FragDepth"
_SPEC = "GLSL 4.60 section 4.4.2.3"

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_DEPTH_LAYOUT_USE = re.compile(
    r"\blayout\s*\([^)]*\bdepth_(?:any|greater|less|unchanged)\b[^)]*\)"
)
_EXTENSION_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*extension[ \t]+(\w+)", re.M)
_VERSION = re.compile(r"^[ \t]*#[ \t]*version[^\n]*\n", re.M)


def _normalized(attribute):
    name = str(getattr(attribute, "name", "") or "").lower().replace("-", "_")
    if name.startswith("glsl_"):
        name = name[len("glsl_") :]
    return name


def depth_layout(node, owner=None):
    """The conservative depth layout ``node`` carries, or ``None``; two different layouts are refused."""
    layouts = []
    for attribute in getattr(node, "attributes", None) or ():
        name = _normalized(attribute)
        if name in DEPTH_LAYOUTS and not getattr(attribute, "arguments", None):
            if name not in layouts:
                layouts.append(name)
    if len(layouts) > 1:
        owner = owner or getattr(node, "name", None) or "value"
        raise OpenGLRefusal(
            f"GLSL cannot give '{owner}' the conservative depth layouts "
            f"{', '.join(sorted(layouts))} at once: gl_FragDepth takes one depth layout ({_SPEC})"
        )
    return layouts[0] if layouts else None


def is_fragment_depth_output(codegen, node, semantic=None):
    """Whether ``node`` declares the fragment depth output: it is named ``gl_FragDepth`` or carries its semantic."""
    if getattr(node, "name", None) == _FRAG_DEPTH:
        return True
    if semantic is None:
        semantic = codegen.semantic_from_node(node)
    return bool(semantic) and codegen.map_semantic(semantic) == _FRAG_DEPTH


def _refuse_misplaced(owner, layout, stage_name=None):
    where = f" of the {stage_name} stage" if stage_name else ""
    raise OpenGLRefusal(
        f"GLSL cannot apply the conservative depth layout '{layout}' to '{owner}'{where}: only a fragment shader's "
        f"gl_FragDepth takes a depth layout ({_SPEC})"
    )


def _redeclaration(codegen, layout, stage_name):
    """``layout(<layout>) out float gl_FragDepth;`` the first time ``stage_name`` asks, else ``""``.

    The stage's interface table refuses a second output that promises a different layout.
    """
    declaration = f"layout({layout}) out float {_FRAG_DEPTH};"
    if codegen.reserve_stage_io_declaration(
        stage_name, "output", f"{_FRAG_DEPTH} depth layout", declaration
    ):
        return f"{declaration}\n"
    return ""


def stage_depth_redeclaration(codegen, node, stage_name):
    """The redeclaration a stage-level ``@output @gl_FragDepth @depth_*`` variable needs, or ``""`` without one."""
    owner = codegen.resource_node_name(node, "") or _FRAG_DEPTH
    layout = depth_layout(node, owner)
    if layout is None:
        return ""
    if stage_name not in (None, "fragment"):
        _refuse_misplaced(owner, layout, stage_name)
    return _redeclaration(codegen, layout, "fragment")


def _entry_depth_outputs(codegen, func):
    """``(node, owner, semantic)`` for each output of the entry ``func`` that can carry a depth layout."""
    outputs = []
    return_type = codegen.type_node_name(getattr(func, "return_type", None))
    struct = codegen.structs_by_name.get(return_type) if return_type else None
    if struct is not None:
        for member in getattr(struct, "members", None) or ():
            outputs.append(
                (
                    member,
                    f"{struct.name}.{member.name}",
                    codegen.stage_io_semantic_from_node(member),
                )
            )
    else:
        outputs.append(
            (func, f"{func.name} return value", codegen.function_return_semantic(func))
        )
    for param in getattr(func, "parameters", None) or ():
        qualifiers = {str(q).lower() for q in getattr(param, "qualifiers", None) or ()}
        if qualifiers & {"out", "inout"}:
            outputs.append(
                (
                    param,
                    f"{func.name} parameter {param.name}",
                    codegen.semantic_from_node(param),
                )
            )
    return outputs


def entry_depth_redeclarations(codegen, target_stage) -> str:
    """The ``gl_FragDepth`` redeclarations for the depth layouts of the stage entries being generated."""
    code = ""
    entries = getattr(codegen, "stage_entry_functions_by_stage", None) or {}
    for stage_name in sorted(entries):
        if target_stage is not None and stage_name != target_stage:
            continue
        lines = ""
        for node, owner, semantic in _entry_depth_outputs(codegen, entries[stage_name]):
            layout = depth_layout(node, owner)
            if layout is None:
                continue
            if stage_name != "fragment" or not is_fragment_depth_output(
                codegen, node, semantic
            ):
                _refuse_misplaced(owner, layout, stage_name)
            lines += _redeclaration(codegen, layout, "fragment")
        code += codegen.wrap_stage_guard(lines, stage_name)
    return code


def require_conservative_depth_extension(code, is_es):
    """GLSL ES ``code`` with ``#extension GL_EXT_conservative_depth : require`` when it uses a depth layout.

    Desktop GLSL has the layouts from GLSL 4.20, and the version requirements raise its ``#version`` line.
    """
    if not is_es or not isinstance(code, str):
        return code
    text = _COMMENT.sub(" ", code)
    if not _DEPTH_LAYOUT_USE.search(text):
        return code
    if CONSERVATIVE_DEPTH_EXTENSION in _EXTENSION_DIRECTIVE.findall(text):
        return code
    line = f"#extension {CONSERVATIVE_DEPTH_EXTENSION} : require\n"
    version = _VERSION.search(code)
    if version is None:
        return line + code
    return code[: version.end()] + line + code[version.end() :]
