"""SPIR-V execution modes for CrossGL's conservative depth layouts.

CrossGL spells GLSL's conservative depth layout (GL_ARB_conservative_depth, GLSL 4.20 section 4.4.2.3) as an
attribute next to the fragment depth output: ``out float gl_FragDepth @depth_greater;`` at stage level, or
``float depth @gl_FragDepth @depth_greater;`` in an output structure. SPIR-V states the promise with an execution
mode of the fragment entry point next to ``DepthReplacing`` (SPIR-V 1.6 section 3.6, as glslang emits it):

- ``depth_greater`` is ``DepthGreater``, ``depth_less`` is ``DepthLess`` and ``depth_unchanged`` is
  ``DepthUnchanged``;
- ``depth_any`` makes no promise, so it adds no mode.

The layout attributes never name the value, so :meth:`semantic_from_node` skips them. A layout on an output that
is not the ``FragDepth`` built-in has no SPIR-V meaning and is refused, as are two different layouts at once.
"""

from __future__ import annotations

__all__ = [
    "DEPTH_LAYOUT_ATTRIBUTES",
    "depth_layout_execution_modes",
    "record_depth_layout",
]

_DEPTH_EXECUTION_MODES = {
    "depth_greater": "DepthGreater",
    "depth_less": "DepthLess",
    "depth_unchanged": "DepthUnchanged",
}
DEPTH_LAYOUT_ATTRIBUTES = frozenset({"depth_any", *_DEPTH_EXECUTION_MODES})


def _depth_layouts(node):
    names = []
    for attribute in getattr(node, "attributes", None) or ():
        name = str(getattr(attribute, "name", "") or "").lower()
        if name in DEPTH_LAYOUT_ATTRIBUTES and name not in names:
            names.append(name)
    return names


def record_depth_layout(codegen, variable, node, storage_class) -> None:
    """Remember the conservative depth layout ``node`` gives the built-in output ``variable``.

    ``variable`` is the interface variable ``node`` lowered to, or ``None`` when it became no built-in. A layout on
    anything other than the ``FragDepth`` output is refused.
    """
    layouts = _depth_layouts(node)
    if not layouts:
        return
    name = str(getattr(node, "name", "") or "value")
    builtin = None
    if variable is not None:
        builtin = codegen.builtin_names_by_variable_id.get(variable.id)
    if storage_class != "Output" or builtin != "FragDepth":
        codegen.refuse_lowering(
            f"SPIR-V cannot apply the conservative depth layout '{layouts[0]}' to '{name}': only the "
            "fragment depth output (BuiltIn FragDepth) takes a depth execution mode"
        )
        return
    recorded = codegen.fragment_depth_layouts.setdefault(variable.id, layouts[0])
    if len(layouts) > 1 or recorded != layouts[0]:
        conflicting = sorted(set(layouts) | {recorded})
        codegen.refuse_lowering(
            f"SPIR-V cannot give '{name}' the conservative depth layouts {', '.join(conflicting)} at once: "
            "a fragment entry point takes one depth execution mode"
        )


def depth_layout_execution_modes(codegen, output_variables) -> list:
    """The depth execution modes, in a stable order, that the fragment outputs ``output_variables`` promise."""
    modes = []
    for variable in output_variables:
        layout = codegen.fragment_depth_layouts.get(variable.id)
        mode = _DEPTH_EXECUTION_MODES.get(layout)
        if mode is not None and mode not in modes:
            modes.append(mode)
    return modes
