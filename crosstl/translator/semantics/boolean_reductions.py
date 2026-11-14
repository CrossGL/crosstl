"""CrossGL's ``any`` and ``all`` on numeric and scalar operands, spelled for targets that reduce only booleans.

CrossGL follows HLSL here: ``any(x)`` is true when some component of ``x`` is non-zero and ``all(x)`` when every
component is, for a scalar, vector or boolean operand of any component type (HLSL intrinsic reference, ``any`` and
``all``). The SPIR-V generator converts the operand itself (:mod:`..codegen.spirv_boolean_reductions`); HLSL and Slang
take the HLSL forms as they are.

Other targets reduce only booleans:

- GLSL and GLSL ES ``any``/``all`` take only ``bvec2``-``bvec4`` (GLSL 4.60 section 8.7; GLSL ES 3.00 section 8.7),
  and have no scalar form;
- WGSL ``any``/``all`` take ``bool`` or ``vecN<bool>`` (WGSL section 17.3, "Logical Built-in Functions");
- MSL ``any``/``all`` take ``bool`` or ``boolN`` (MSL 3.2 section 6.4, "Relational Functions"), and MSL converts no
  vector implicitly.

:func:`lower_boolean_reductions` converts a numeric vector operand to the boolean vector of its width (``bvecN(x)``,
non-zero components becoming ``true`` in all three languages: GLSL 4.60 section 5.4.1, WGSL section 16.1.2.2 "Value
Constructor Built-in Functions", MSL 3.2 section 2.13), and replaces the reduction of a scalar by the scalar itself
converted to ``bool``. An operand whose type the pass cannot infer, such as a matrix, is left to the generator.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name
from .rewrite import call, child_slots, get_child, set_child
from .scoped_walk import ScopedWalker

__all__ = ["lower_boolean_reductions"]

_TARGETS = frozenset({"opengl", "glsl", "webgl", "gles", "essl", "wgsl", "metal"})
_REDUCTIONS = frozenset({"any", "all"})


def lower_boolean_reductions(program, target) -> int:
    """Rewrite non-boolean ``any``/``all`` operands for ``target`` in place; returns how many were rewritten."""
    if target not in _TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    walker = _BooleanReductions(program)
    walker.run_program()
    return walker.rewritten


class _BooleanReductions(ScopedWalker):
    def __init__(self, program):
        super().__init__(program)
        self.rewritten = 0

    def _walk(self, node):
        if isinstance(node, A.VariableNode):
            # The walker declares a variable instead of visiting it; its initializer is rewritten first.
            for field, key in list(child_slots(node)):
                self._walk(get_child(node, field, key))
            self.visit(node)
            self.scope.declare(node.name, node.var_type)
            return
        super()._walk(node)

    def visit(self, node):
        # Children are visited first, so a nested reduction is rewritten before the expression that uses it.
        for field, key in list(child_slots(node)):
            replacement = self._replacement(get_child(node, field, key))
            if replacement is not None:
                set_child(node, field, key, replacement)
                self.rewritten += 1

    def _replacement(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return None
        name = call_name(node)
        arguments = node.arguments or []
        if (
            name not in _REDUCTIONS
            or len(arguments) != 1
            or name in self.scope.functions
        ):
            return None
        operand = arguments[0]
        value = self.scope.infer(operand)
        if value is None:
            return None
        if value.lanes == 1:
            # The reduction of one component is that component as a ``bool``.
            return operand if value.kind == "b" else call("bool", [operand])
        if value.kind == "b":
            return None
        # ``arguments`` and its ``args`` alias share one list.
        arguments[0] = call(f"bvec{value.lanes}", [operand])
        self.rewritten += 1
        return None
