"""The stride output of ``buffer_dimensions(b, count, stride)`` on targets whose buffer queries report no stride.

HLSL's ``StructuredBuffer<T>::GetDimensions(out uint numStructs, out uint stride)`` reports the number of elements
and the element stride in bytes; CrossGL spells it ``buffer_dimensions(b, count, stride)``. GLSL's ``length()`` of a
runtime-sized array (GLSL 4.60 section 4.1.9), SPIR-V's ``OpArrayLength`` (SPIR-V 1.6 section 3.49.8, "Result Type
must be an integer type scalar") and the element count a Metal host passes report only the count. The OpenGL and
Metal generators assigned ``count`` and left ``stride`` unassigned, and the SPIR-V generator refused the query.

The stride is a constant of the element type and of the layout the target gives the buffer:

* a buffer marked ``@buffer_layout(hlsl_structured)`` or ``@buffer_layout(metal)`` keeps that layout on every
  target (:mod:`.structured_layouts`, and the generators' HLSL-layout structures), so its stride follows the HLSL
  structured-buffer rules (D3D11.3 section 5.3.7) or the MSL rules (MSL 3.2 section 2.2);
* an unmarked buffer has the target's own layout: std430 for OpenGL storage blocks and SPIR-V (GLSL 4.60 section
  7.6.2.2), and the MSL natural layout for Metal.

For those targets, :func:`lower_buffer_stride_queries` rewrites every ``buffer_dimensions(b, count, stride)``
statement on a structured buffer into ``buffer_dimensions(b, count)``, which each generator lowers to its element
count, followed by ``stride = N``. It runs after :mod:`.structured_layouts`, which answers the queries on the buffers
it turns into words itself. A three-operand query in an expression, on a value that is not a structured buffer, or on
a buffer whose element type has no layout here raises :class:`BufferStrideUnsupportedError` instead of leaving the
stride unassigned.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name
from .rewrite import child_slots, get_child, set_child
from .structured_layouts import (
    StructuredLayoutUnsupportedError,
    declared_buffer_layout,
    element_stride,
)

__all__ = ["BufferStrideUnsupportedError", "lower_buffer_stride_queries"]

# Targets whose buffer length query reports no stride, and the rules of their unmarked structured buffers.
_TARGET_RULES = {
    "opengl": "std430",
    "glsl": "std430",
    "vulkan": "std430",
    "spirv": "std430",
    "metal": "metal",
}
# ``@buffer_layout`` values and the rules they name.
_DECLARED_RULES = {"hlsl_structured": "hlsl", "metal": "metal", "std430": "std430"}
_STRUCTURED = frozenset(
    {
        "StructuredBuffer",
        "RWStructuredBuffer",
        "RasterizerOrderedStructuredBuffer",
        "AppendStructuredBuffer",
        "ConsumeStructuredBuffer",
    }
)


class BufferStrideUnsupportedError(TranslationDiagnostic):
    """A ``buffer_dimensions`` stride output the target cannot answer."""

    def __init__(self, target, reason):
        self.missing_capabilities = (f"{target}.buffer-stride",)
        super().__init__(
            reason,
            target=target,
            code=f"{CODE_PREFIX}{target}-buffer-stride-unsupported",
        )


def _is_stride_query(node):
    return (
        isinstance(node, A.FunctionCallNode)
        and call_name(node) == "buffer_dimensions"
        and len(node.arguments or []) == 3
    )


def _buffer_name(node):
    """The variable an operand names, through element indexing of an array of buffers."""
    while isinstance(node, A.ArrayAccessNode):
        node = node.array
    return node.name if isinstance(node, A.IdentifierNode) else None


def _structured_element(var_type, indexed):
    """The element type of a structured buffer type, unwrapping one array level when the operand indexes it."""
    if indexed and isinstance(var_type, A.ArrayType):
        var_type = var_type.element_type
    if (
        isinstance(var_type, A.NamedType)
        and var_type.name in _STRUCTURED
        and len(var_type.generic_args) == 1
    ):
        return var_type.generic_args[0]
    return None


class _Lowerer:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.count = 0

    def refuse(self, reason):
        raise BufferStrideUnsupportedError(self.target, reason)

    def stride(self, declaration, operand):
        name = declaration.name
        slot = "param_type" if isinstance(declaration, A.ParameterNode) else "var_type"
        element = _structured_element(
            getattr(declaration, slot, None), isinstance(operand, A.ArrayAccessNode)
        )
        if element is None:
            self.refuse(
                f"buffer_dimensions reports an element stride only for structured buffers; {name} is not one"
            )
        declared = declared_buffer_layout(declaration)
        rules = (
            _TARGET_RULES[self.target]
            if declared is None
            else _DECLARED_RULES.get(declared)
        )
        if rules is None:
            self.refuse(
                f"the structured buffer {name} declares the layout {declared!r}, which has no element stride here"
            )
        try:
            return element_stride(
                self.program, element, rules, f"structured buffer {name}"
            )
        except StructuredLayoutUnsupportedError as error:
            self.refuse(f"the element stride of {name} is unknown: {error}")

    def lowered(self, node, scope):
        buffer, count, stride_target = node.arguments
        name = _buffer_name(buffer)
        declaration = scope.get(name) if name is not None else None
        if declaration is None:
            self.refuse(
                "buffer_dimensions with a stride output needs a structured-buffer variable or parameter operand"
            )
        stride = self.stride(declaration, buffer)
        self.count += 1
        query = A.FunctionCallNode(
            A.IdentifierNode("buffer_dimensions"), [buffer, count]
        )
        assignment = A.AssignmentNode(
            stride_target, A.LiteralNode(stride, A.PrimitiveType("uint"))
        )
        return A.BlockNode(
            [A.ExpressionStatementNode(query), A.ExpressionStatementNode(assignment)]
        )

    def visit(self, node, scope):
        for slot_field, key in list(child_slots(node)):
            child = get_child(node, slot_field, key)
            if isinstance(child, A.ExpressionStatementNode) and _is_stride_query(
                child.expression
            ):
                set_child(node, slot_field, key, self.lowered(child.expression, scope))
            elif _is_stride_query(child):
                self.refuse(
                    "buffer_dimensions with a stride output is a statement, not a value"
                )
            elif isinstance(child, A.ASTNode):
                self.visit(child, scope)

    def function(self, function, outer):
        scope = dict(outer)
        for parameter in getattr(function, "parameters", None) or []:
            if isinstance(getattr(parameter, "name", None), str):
                scope[parameter.name] = parameter
        if getattr(function, "body", None) is not None:
            self.visit(function.body, scope)

    def run(self):
        globals_ = _declarations(getattr(self.program, "global_variables", None))
        for function in getattr(self.program, "functions", None) or []:
            self.function(function, globals_)
        stages = getattr(self.program, "stages", None) or {}
        for stage in stages.values() if isinstance(stages, dict) else stages:
            scope = dict(globals_)
            scope.update(_declarations(getattr(stage, "local_variables", None)))
            for function in getattr(stage, "local_functions", None) or []:
                self.function(function, scope)
            if getattr(stage, "entry_point", None) is not None:
                self.function(stage.entry_point, scope)
        return self.count


def _declarations(variables):
    return {
        variable.name: variable
        for variable in variables or []
        if isinstance(variable, A.VariableNode) and isinstance(variable.name, str)
    }


def lower_buffer_stride_queries(program, target) -> int:
    """Answer the stride output of every ``buffer_dimensions(b, count, stride)`` for ``target``; returns how many."""
    target = str(target or "")
    if target not in _TARGET_RULES or not isinstance(program, A.ShaderNode):
        return 0
    return _Lowerer(program, target).run()
