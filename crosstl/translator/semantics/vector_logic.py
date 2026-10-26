"""Component-wise ``&&`` and ``||`` on boolean vectors, and CrossGL's ``and``/``or``, spelled for each target.

CrossGL follows HLSL before 2021 and MSL: ``x && y`` on ``bvec2`` operands combines them component by component and
evaluates both (MSL 2.4, "Logical Operators"; HLSL "Per-Component Math Operations"). A scalar operand applies to
every component. ``and(x, y)`` and ``or(x, y)`` (HLSL 2021) mean the same and are the explicit spelling.

Most targets accept only scalars:

- HLSL 2021, the language DXC compiles by default, rejects vector operands ("operands for short-circuiting logical
  binary operator must be scalar, for non-scalar types use 'and'");
- GLSL takes only scalar ``bool`` operands (GLSL 4.60 section 5.9), and the generator lowers ``and``/``or`` on
  vectors through unsigned lanes;
- WGSL ``&&``/``||`` take only ``bool``; ``&``/``|`` combine ``vecN<bool>`` operands and evaluate both (WGSL
  section 8.6, "Logical Expressions"). WGSL has no ``and``/``or`` built-in.

:func:`lower_vector_logic` rewrites ``&&``/``||`` with a vector operand into ``and``/``or`` for HLSL and GLSL
targets, whose generators convert and splat their operands (DXC accepts mixed and non-boolean ``and`` operands). For
WGSL both forms become ``&``/``|``, with scalar operands splatted and non-boolean ones converted (``bvecN(...)``),
because WGSL converts nothing implicitly. MSL and SPIR-V (``OpLogicalAnd``/``OpLogicalOr``) take vector operands but
convert no vector implicitly, so only a non-boolean operand is converted; their generators lower ``and``/``or``.
Scalar ``&&``/``||`` keep their short-circuit evaluation.
"""

from __future__ import annotations

from .. import ast as A
from .rewrite import call, child_slots, get_child, set_child
from .scoped_walk import ScopedWalker, program_stages

__all__ = ["lower_vector_logic"]

# How each target spells a vector logical operation: ``and``/``or`` calls, WGSL's ``&``/``|`` operators, or
# ``&&``/``||`` itself with non-boolean operands converted (MSL and SPIR-V take no implicit vector conversion).
_MODES = {
    "directx": "intrinsic",
    "hlsl": "intrinsic",
    "opengl": "intrinsic",
    "glsl": "intrinsic",
    "webgl": "intrinsic",
    "gles": "intrinsic",
    "essl": "intrinsic",
    "wgsl": "operator",
    "metal": "convert",
    "vulkan": "convert",
    "spirv": "convert",
}
_INTRINSICS = {"&&": "and", "||": "or"}
_WGSL_OPERATORS = {"&&": "&", "||": "|", "and": "&", "or": "|"}


def lower_vector_logic(program, target) -> int:
    """Rewrite vector logical operations for ``target`` in place; returns how many were rewritten."""
    mode = _MODES.get(target)
    if mode is None or not isinstance(program, A.ShaderNode):
        return 0
    walker = _VectorLogic(program, mode)
    walker.run_program()
    return walker.rewritten


def _user_function_names(program):
    names = set()
    for function in getattr(program, "functions", None) or []:
        names.add(getattr(function, "name", None))
    for stage in program_stages(program):
        for function in getattr(stage, "local_functions", None) or []:
            names.add(getattr(function, "name", None))
    return names


class _VectorLogic(ScopedWalker):
    def __init__(self, program, mode):
        super().__init__(program)
        self.mode = mode
        self.user_functions = _user_function_names(program)
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
        # Children are visited first, so nested operations are rewritten before the operation that uses them.
        for field, key in list(child_slots(node)):
            replacement = self._replacement(get_child(node, field, key))
            if replacement is not None:
                set_child(node, field, key, replacement)
                self.rewritten += 1

    def _replacement(self, node):
        operation, operands = self._logical_operation(node)
        if operation is None or (
            self.mode != "operator" and operation not in _INTRINSICS
        ):
            return None
        types = [self.scope.infer(operand) for operand in operands]
        lanes = max((value.lanes for value in types if value is not None), default=1)
        if operation in _INTRINSICS and lanes == 1:
            # Scalar ``&&``/``||`` keeps its short-circuit evaluation.
            return None
        if self.mode == "intrinsic":
            # The HLSL and GLSL generators convert and splat ``and``/``or`` operands themselves.
            return call(_INTRINSICS[operation], operands)
        if self.mode == "operator":
            operands = [
                _as_bool(operand, value, lanes)
                for operand, value in zip(operands, types)
            ]
            return A.BinaryOpNode(operands[0], _WGSL_OPERATORS[operation], operands[1])
        converted = [
            (
                _as_bool(operand, value, lanes)
                if value is not None and value.kind != "b"
                else operand
            )
            for operand, value in zip(operands, types)
        ]
        if any(new is not old for new, old in zip(converted, operands)):
            node.left, node.right = converted
            self.rewritten += 1
        return None

    def _logical_operation(self, node):
        if isinstance(node, A.BinaryOpNode) and node.op in _INTRINSICS:
            return node.op, [node.left, node.right]
        if isinstance(node, A.FunctionCallNode):
            name = getattr(node.function, "name", node.function)
            arguments = node.arguments or []
            if (
                name in ("and", "or")
                and len(arguments) == 2
                and name not in self.user_functions
            ):
                return name, list(arguments)
        return None, None


def _as_bool(operand, value, lanes):
    """``operand`` as a ``bool``/``bvec<lanes>``: kept when it is one, splatted from a scalar or converted otherwise."""
    if value is not None and value.kind == "b" and value.lanes == lanes:
        return operand
    if lanes == 1:
        return operand if value is None else call("bool", [operand])
    if value is not None and value.lanes == 1 and value.kind != "b":
        # WGSL's vector constructors splat or convert, not both.
        operand = call("bool", [operand])
    return call(f"bvec{lanes}", [operand])
