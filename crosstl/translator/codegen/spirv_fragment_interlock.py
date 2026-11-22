"""Fragment shader interlock in SPIR-V output.

A rasterizer-ordered resource (HLSL's ``RasterizerOrderedBuffer`` and ``RasterizerOrderedTexture2D``, CrossGL's
``@rasterizer_ordered``) orders the accesses that the fragments covering one pixel make to it by the order of their
primitives. SPIR-V expresses that with SPV_EXT_fragment_shader_interlock: the fragment entry point declares an
interlock execution mode, and ``OpBeginInvocationInterlockEXT`` and ``OpEndInvocationInterlockEXT`` bracket the
accesses. DXC lowers rasterizer-ordered views with one critical section around the whole entry point, and so does
:mod:`.glsl_fragment_interlock` for GLSL; this generator does the same. ``OpBeginInvocationInterlockEXT`` comes
after the entry point copies its stage inputs and before the first statement of its body, and
``OpEndInvocationInterlockEXT`` comes before its ``OpReturn``, after the outputs are written. Pixel ordering orders every pair of accesses that sample ordering orders, so the default mode,
``PixelInterlockOrderedEXT``, is correct whatever the sample rate.

The extension requires each instruction to run exactly once in the entry point, the begin before the end, so a
fragment entry point that returns early cannot hold the section and is refused, as GLSL's rule refuses it. A program
that calls ``beginInvocationInterlockARB()`` and ``endInvocationInterlockARB()`` itself keeps its own critical
section, and the interlock layout its fragment stage declares (``layout(pixel_interlock_ordered) in;`` and the other
three of :data:`INTERLOCK_MODES`) chooses the execution mode and its capability.
"""

from __future__ import annotations

from ..ast import FunctionCallNode, ReturnNode
from .ast_walk import walk_ast_values
from .stage_utils import collect_stage_local_variables

__all__ = [
    "INTERLOCK_CALLS",
    "begin_entry_interlock",
    "end_entry_interlock",
    "fragment_interlock_execution_mode",
    "plan_fragment_interlock",
    "spirv_interlock_call",
]

INTERLOCK_EXTENSION = "SPV_EXT_fragment_shader_interlock"
INTERLOCK_CALLS = {
    "beginInvocationInterlockARB": "OpBeginInvocationInterlockEXT",
    "endInvocationInterlockARB": "OpEndInvocationInterlockEXT",
}
# GLSL's interlock layout -> (SPIR-V execution mode, the capability that mode requires).
INTERLOCK_MODES = {
    "pixel_interlock_ordered": (
        "PixelInterlockOrderedEXT",
        "FragmentShaderPixelInterlockEXT",
    ),
    "pixel_interlock_unordered": (
        "PixelInterlockUnorderedEXT",
        "FragmentShaderPixelInterlockEXT",
    ),
    "sample_interlock_ordered": (
        "SampleInterlockOrderedEXT",
        "FragmentShaderSampleInterlockEXT",
    ),
    "sample_interlock_unordered": (
        "SampleInterlockUnorderedEXT",
        "FragmentShaderSampleInterlockEXT",
    ),
}
_DEFAULT_LAYOUT = "pixel_interlock_ordered"
_RASTERIZER_ORDERED = "rasterizer_ordered"


def _attribute_names(node):
    return {
        str(getattr(attribute, "name", attribute)).lower()
        for attribute in getattr(node, "attributes", None) or ()
    }


def _call_name(node):
    function = getattr(node, "function", None)
    return function if isinstance(function, str) else getattr(function, "name", None)


def _declared_layouts(codegen, ast):
    """The interlock layouts the program's fragment stages and functions declare, in a stable order."""
    layouts = set()
    for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
        if codegen.stage_key(stage_type) != "fragment":
            continue
        for qualifier in getattr(stage, "layout_qualifiers", None) or ():
            for entry in getattr(qualifier, "entries", None) or ():
                layouts.add(str(getattr(entry, "name", "")).lower())
        layouts |= _attribute_names(getattr(stage, "entry_point", None))
    for function in getattr(ast, "functions", None) or ():
        layouts |= _attribute_names(function)
    return sorted(layouts & set(INTERLOCK_MODES))


def plan_fragment_interlock(codegen, ast) -> None:
    """Record which rasterizer-ordered resources ``ast`` declares and how its fragment stage orders them."""
    declarations = list(getattr(ast, "global_variables", None) or ())
    declarations += collect_stage_local_variables(ast, "fragment")
    codegen.fragment_interlock_resources = tuple(
        sorted(
            {
                str(node.name)
                for node in declarations
                if _RASTERIZER_ORDERED in _attribute_names(node)
                and getattr(node, "name", None)
            }
        )
    )
    codegen.fragment_interlock_calls = any(
        isinstance(node, FunctionCallNode) and _call_name(node) in INTERLOCK_CALLS
        for node in walk_ast_values(ast)
    )
    layouts = _declared_layouts(codegen, ast)
    if len(layouts) > 1:
        codegen.refuse_lowering(
            f"fragment shader interlock: the fragment stage declares the interlock layouts {', '.join(layouts)}, "
            "and a SPIR-V entry point takes one interlock execution mode"
        )
    codegen.fragment_interlock_layout = layouts[0] if layouts else _DEFAULT_LAYOUT
    codegen.fragment_interlock_function_ids = set()


def _require_interlock(codegen):
    _mode, capability = INTERLOCK_MODES[codegen.fragment_interlock_layout]
    codegen.require_capability(capability)
    codegen.require_extension(INTERLOCK_EXTENSION)


def _returns_early(body):
    statements = list(getattr(body, "statements", body) or ())
    final = (
        statements[-1:] if statements and isinstance(statements[-1], ReturnNode) else []
    )
    leading = statements[: len(statements) - len(final)]
    return any(isinstance(node, ReturnNode) for node in walk_ast_values(leading))


def begin_entry_interlock(codegen, function_node) -> None:
    """Open the critical section at the start of a fragment entry point when the program reaches a rasterizer-ordered
    resource and does not open the section itself."""
    resources = getattr(codegen, "fragment_interlock_resources", ())
    if (
        not resources
        or getattr(codegen, "fragment_interlock_calls", False)
        or codegen.current_execution_model != "Fragment"
    ):
        return
    if _returns_early(getattr(function_node, "body", None)):
        names = ", ".join(f"'{name}'" for name in resources)
        codegen.refuse_lowering(
            f"rasterizer-ordered access to {names}: the fragment entry point returns early, and "
            "SPV_EXT_fragment_shader_interlock requires its critical section to end exactly once before the return"
        )
        return
    _require_interlock(codegen)
    codegen.fragment_interlock_function_ids.add(codegen.current_function_id)
    codegen.emit("OpBeginInvocationInterlockEXT")


def end_entry_interlock(codegen) -> None:
    """Close the critical section :func:`begin_entry_interlock` opened, before the entry point's ``OpReturn``."""
    if codegen.current_function_id in getattr(
        codegen, "fragment_interlock_function_ids", ()
    ):
        codegen.emit("OpEndInvocationInterlockEXT")


def spirv_interlock_call(codegen, name, args):
    """Lower a ``beginInvocationInterlockARB()`` or ``endInvocationInterlockARB()`` call the program writes."""
    if args:
        codegen.refuse_lowering(f"'{name}' takes no arguments")
        return None
    _require_interlock(codegen)
    codegen.emit(INTERLOCK_CALLS[name])
    return None


def fragment_interlock_execution_mode(codegen, function_id):
    """The interlock execution mode of the fragment entry point ``function_id``, or ``None`` when it has none."""
    if not (
        getattr(codegen, "fragment_interlock_calls", False)
        or function_id in getattr(codegen, "fragment_interlock_function_ids", ())
    ):
        return None
    return INTERLOCK_MODES[codegen.fragment_interlock_layout][0]
