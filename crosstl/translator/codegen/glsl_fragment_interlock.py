"""Fragment shader interlock in GLSL output.

A rasterizer-ordered resource (HLSL's ``RasterizerOrderedBuffer`` and ``RasterizerOrderedTexture2D``, CrossGL's
``@rasterizer_ordered``) orders the accesses that the fragments covering one pixel make to it by the order of their
primitives. GLSL expresses that with GL_ARB_fragment_shader_interlock: the fragment stage declares an interlock mode,
``layout(pixel_interlock_ordered) in;``, and brackets the accesses with ``beginInvocationInterlockARB()`` and
``endInvocationInterlockARB()``. DXC lowers rasterizer-ordered views to SPIR-V the same way, with one critical
section around the whole entry point, and so does this generator: ``beginInvocationInterlockARB()`` is the first
statement of the fragment stage's ``main()`` and ``endInvocationInterlockARB()`` comes before its final return.
Pixel ordering orders every pair of accesses that sample ordering orders, so it is correct whatever the sample rate.

The extension allows each call once, directly in ``main()``, outside any control flow and before any ``return``
(glslang: "cannot be placed after a return from main()"). A fragment ``main()`` that returns early cannot hold the
section, so it is refused. A program that calls the two functions itself keeps its own critical section, and an
interlock mode the stage declares (one of :data:`INTERLOCK_LAYOUTS`) is written as it stands. glslang accepts the
interlock layouts only from GLSL 4.50, and GLSL ES has no ARB interlock extension
(:data:`.glsl_version_requirements.BUILTIN_FUNCTION_REQUIREMENTS` records both).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..ast import FunctionCallNode
from .stage_utils import (
    collect_stage_entry_records,
    collect_stage_local_variables,
    normalize_stage_name,
)

__all__ = [
    "INTERLOCK_EXTENSION",
    "INTERLOCK_FUNCTIONS",
    "INTERLOCK_LAYOUTS",
    "FragmentInterlockPlan",
    "fragment_interlock_extension_lines",
    "fragment_interlock_layout",
    "plan_fragment_interlock",
    "wrap_fragment_interlock",
]

INTERLOCK_EXTENSION = "GL_ARB_fragment_shader_interlock"
INTERLOCK_FUNCTIONS = frozenset(
    {"beginInvocationInterlockARB", "endInvocationInterlockARB"}
)
INTERLOCK_LAYOUTS = frozenset(
    {
        "pixel_interlock_ordered",
        "pixel_interlock_unordered",
        "sample_interlock_ordered",
        "sample_interlock_unordered",
    }
)
ORDERED_LAYOUT = "pixel_interlock_ordered"
RASTERIZER_ORDERED = "rasterizer_ordered"

_BEGIN = "    beginInvocationInterlockARB();\n"
_END = "    endInvocationInterlockARB();\n"
_FINAL_RETURN = "    return;\n"
_RETURN = re.compile(r"^\s*return\b", re.MULTILINE)


@dataclass(frozen=True)
class FragmentInterlockPlan:
    """How a program's fragment stage orders its accesses.

    ``resources`` are the rasterizer-ordered resources the fragment stage can reach, ``wraps`` whether the
    generator opens and closes the critical section itself, and ``extension`` whether the output needs
    :data:`INTERLOCK_EXTENSION`.
    """

    resources: tuple[str, ...] = ()
    wraps: bool = False
    extension: bool = False


NO_INTERLOCK = FragmentInterlockPlan()


def _attribute_names(node):
    return {
        str(getattr(attribute, "name", attribute)).lower()
        for attribute in getattr(node, "attributes", ()) or ()
    }


def _rasterizer_ordered_resources(ast):
    declarations = list(getattr(ast, "global_variables", []) or [])
    declarations += collect_stage_local_variables(ast, "fragment")
    return tuple(
        sorted(
            {
                str(name)
                for node in declarations
                if RASTERIZER_ORDERED in _attribute_names(node)
                and (name := getattr(node, "name", None))
            }
        )
    )


def _calls_interlock(codegen, ast):
    return any(
        isinstance(node, FunctionCallNode)
        and codegen.function_call_name(node) in INTERLOCK_FUNCTIONS
        for node in codegen.walk_ast(ast)
    )


def _declared_layouts(ast):
    """The interlock modes the program's fragment stages and entry points declare."""
    layouts = set()
    for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
        if normalize_stage_name(stage_type) != "fragment":
            continue
        for qualifier in getattr(stage, "layout_qualifiers", []) or []:
            for entry in getattr(qualifier, "entries", []) or []:
                layouts.add(str(getattr(entry, "name", "")).lower())
        layouts |= _attribute_names(getattr(stage, "entry_point", None))
    for func in getattr(ast, "functions", []) or []:
        layouts |= _attribute_names(func)
    return layouts & INTERLOCK_LAYOUTS


def plan_fragment_interlock(codegen, ast, target_stage=None):
    """The :class:`FragmentInterlockPlan` of ``ast`` when its ``target_stage`` is generated."""
    fragment_entries = [
        func
        for _entry_id, stage_name, func in collect_stage_entry_records(
            ast, target_stage, codegen.combined_stage_entry_types()
        )
        if stage_name == "fragment"
    ]
    if not fragment_entries:
        return NO_INTERLOCK
    resources = _rasterizer_ordered_resources(ast)
    calls = _calls_interlock(codegen, ast)
    wraps = bool(resources) and not calls
    return FragmentInterlockPlan(
        resources=resources,
        wraps=wraps,
        extension=wraps or calls or bool(_declared_layouts(ast)),
    )


def fragment_interlock_extension_lines(codegen, ast):
    """The ``#extension`` line the interlock needs, unless the program enables the extension itself."""
    plan = getattr(codegen, "glsl_fragment_interlock_plan", NO_INTERLOCK)
    if (
        not plan.extension
        or INTERLOCK_EXTENSION in codegen.glsl_preprocessor_extensions(ast)
    ):
        return []
    return [f"#extension {INTERLOCK_EXTENSION} : require"]


def fragment_interlock_layout(codegen, func, stage_layout_qualifiers=None):
    """The fragment stage's ``layout(<mode>) in;`` line, or an empty string."""
    declared = codegen.glsl_stage_bare_attribute(
        func, INTERLOCK_LAYOUTS, stage_layout_qualifiers, "in"
    )
    plan = getattr(codegen, "glsl_fragment_interlock_plan", NO_INTERLOCK)
    layout = declared or (ORDERED_LAYOUT if plan.wraps else None)
    return f"layout({layout}) in;\n" if layout else ""


def wrap_fragment_interlock(codegen, shader_type, entry_name, code, body_start):
    """Bracket the body of a fragment ``main()`` that ``code[body_start:]`` holds with the critical section."""
    plan = getattr(codegen, "glsl_fragment_interlock_plan", NO_INTERLOCK)
    if shader_type != "fragment" or not plan.wraps:
        return code
    resources = ", ".join(f"'{name}'" for name in plan.resources)
    if (entry_name or "main") != "main":
        codegen.refuse_lowering(
            f"rasterizer-ordered access to {resources}: the fragment entry point is written as "
            f"'{entry_name}', and GL_ARB_fragment_shader_interlock orders accesses only from main()"
        )
        return code
    head, body = code[:body_start], code[body_start:]
    closes_with_return = body.endswith(_FINAL_RETURN)
    if closes_with_return:
        body = body[: -len(_FINAL_RETURN)]
    if _RETURN.search(body):
        codegen.refuse_lowering(
            f"rasterizer-ordered access to {resources}: the fragment entry point returns early, and "
            "GL_ARB_fragment_shader_interlock ends its critical section only in main() before any return"
        )
        return code
    return head + _BEGIN + body + _END + (_FINAL_RETURN if closes_with_return else "")
