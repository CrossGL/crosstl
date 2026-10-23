"""Constant element strides for ``buffer_dimensions(buffer, count, stride)`` on targets without a stride query.

HLSL's ``GetDimensions(count, stride)`` on a structured buffer writes the element count and the element stride in
bytes. The stride is that of the declared element type under the buffer's layout rules, and DXC folds it to a
literal: for ``StructuredBuffer<S>`` with ``struct S { float4 a; float b; }`` ``cs_6_0`` stores ``i32 20``. HLSL and
Slang spell the three-output query, but GLSL's ``length()``, SPIR-V's ``OpArrayLength``, the element count a Metal
host passes beside a buffer and the CUDA, HIP, Mojo, Rust and WGSL buffers count elements only, so those generators
left ``stride`` unassigned or refused the call.

:func:`materialize_buffer_strides` rewrites, for those targets, each statement ``buffer_dimensions(b, count,
stride)`` into ``buffer_dimensions(b, count); stride = N;``, ``N`` being the element stride

* under the buffer's ``@buffer_layout`` (``hlsl_structured``, ``metal`` or ``std430``, with the rules of
  :mod:`.structured_layouts`);
* else under the target's own buffer rules, std430 for GLSL and SPIR-V (GLSL 4.60 section 7.6.2.2) and for WGSL
  (WGSL's host-shareable layout aligns ``vec3`` to 16 bytes like std430), and MSL's for Metal;
* for a buffer parameter, from the buffers its calls pass, whose strides must agree.

The pass runs after :func:`.structured_layouts.legalize_structured_layouts`, which already rewrote the query on the
buffers it turned into words. It raises :class:`.structured_layouts.StructuredLayoutUnsupportedError` when the
stride is not known: a buffer operand that is not a declared structured buffer or a buffer parameter, an unmarked
buffer on a target whose buffer rules the pass does not model, a parameter whose calls pass different strides, or
the query in a position that cannot hold two statements.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name, iter_functions
from .rewrite import SKIPPED_FIELDS, call, set_child
from .structured_layouts import StructuredLayoutUnsupportedError, element_stride

__all__ = ["TARGETS", "materialize_buffer_strides"]

_QUERY = "buffer_dimensions"
# Targets whose generators cannot spell the stride output, with the buffer rules of an unmarked buffer where they
# are modelled. HLSL and Slang spell it natively.
_TARGET_RULES = {
    "opengl": "std430",
    "glsl": "std430",
    "webgl": "std430",
    "vulkan": "std430",
    "spirv": "std430",
    "wgsl": "std430",
    "webgpu": "std430",
    "metal": "metal",
}
TARGETS = frozenset(_TARGET_RULES) | frozenset({"cuda", "hip", "mojo", "rust"})
_LAYOUT_RULES = {"hlsl_structured": "hlsl", "metal": "metal", "std430": "std430"}
_WRAPPERS = frozenset(
    {
        "StructuredBuffer",
        "RWStructuredBuffer",
        "AppendStructuredBuffer",
        "ConsumeStructuredBuffer",
        "RasterizerOrderedStructuredBuffer",
    }
)
# Fields that hold statements: a list is spliced, a single statement becomes a block.
_STATEMENT_FIELDS = frozenset(
    {
        "statements",
        "then_branch",
        "else_branch",
        "if_body",
        "else_body",
        "else_if_bodies",
        "body",
        "default_case",
    }
)


def _is_query(node, arity=3):
    return (
        isinstance(node, A.FunctionCallNode)
        and call_name(node) == _QUERY
        and len(node.arguments or []) == arity
    )


def _statement_query(node):
    """The three-output query a statement consists of, or ``None``."""
    if isinstance(node, A.ExpressionStatementNode) and _is_query(node.expression):
        return node.expression
    return node if _is_query(node) else None


def _walk(root):
    seen, stack = set(), [root]
    while stack:
        value = stack.pop()
        if isinstance(value, list):
            stack.extend(reversed(value))
            continue
        if not isinstance(value, A.ASTNode) or id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        stack.extend(
            reversed(
                [
                    child
                    for field, child in vars(value).items()
                    if field not in SKIPPED_FIELDS
                ]
            )
        )


def _declared_type(node):
    for field in ("param_type", "var_type", "vtype"):
        value = getattr(node, field, None)
        if value is not None:
            return value
    return None


def _element_type(type_node):
    """The element type of a structured buffer, or of an array of them, or ``None``."""
    while isinstance(type_node, A.ArrayType):
        type_node = type_node.element_type
    if (
        isinstance(type_node, A.NamedType)
        and type_node.name in _WRAPPERS
        and len(type_node.generic_args) == 1
    ):
        return type_node.generic_args[0]
    return None


def _declared_layout(node):
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() != "buffer_layout":
            continue
        for argument in getattr(attribute, "arguments", None) or []:
            value = getattr(argument, "name", getattr(argument, "value", argument))
            return str(value).strip().strip('"').lower()
    return None


def _literal(value):
    return A.LiteralNode(int(value), A.PrimitiveType("uint"))


class _Strides:
    """The program's buffers and functions, and the strides of the buffers its queries name."""

    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.shapes = {}
        self.globals = {}
        for node in getattr(program, "global_variables", None) or []:
            self.globals.setdefault(getattr(node, "name", None), node)
        self.functions = []  # (function, stage-local declarations by name)
        seen = set()
        for function in getattr(program, "functions", None) or []:
            self.functions.append((function, {}))
            seen.add(id(function))
        stages = getattr(program, "stages", None) or {}
        for stage in stages.values() if isinstance(stages, dict) else stages:
            scope = {}
            for node in getattr(stage, "local_variables", None) or []:
                scope.setdefault(getattr(node, "name", None), node)
            functions = list(getattr(stage, "local_functions", None) or [])
            if getattr(stage, "entry_point", None) is not None:
                functions.append(stage.entry_point)
            for function in functions:
                if id(function) not in seen:
                    self.functions.append((function, scope))
                    seen.add(id(function))
        for function in iter_functions(program):
            if id(function) not in seen:
                self.functions.append((function, {}))
                seen.add(id(function))
        self.scopes = {id(function): scope for function, scope in self.functions}
        self.parameter_strides = {}

    def refuse(self, reason):
        raise StructuredLayoutUnsupportedError(
            f"buffer_dimensions with a stride output: {reason}", target=self.target
        )

    # Strides ---------------------------------------------------------------

    def declaration_stride(self, declaration, context):
        element = _element_type(_declared_type(declaration))
        if element is None:
            self.refuse(f"{context} is not a structured buffer")
        layout = _declared_layout(declaration)
        if layout is None:
            rules = _TARGET_RULES.get(self.target)
            if rules is None:
                self.refuse(
                    f"{context} declares no buffer layout, and the {self.target} buffer rules are not modelled"
                )
        else:
            rules = _LAYOUT_RULES.get(layout)
            if rules is None:
                self.refuse(
                    f"{context} has the {layout} layout, whose stride is not modelled"
                )
        return element_stride(self.program, element, rules, context, self.shapes)

    def operand_stride(self, function, expression):
        """The element stride of the buffer ``expression`` names inside ``function``."""
        base = expression
        while isinstance(base, A.ArrayAccessNode):
            base = base.array_expr
        if not isinstance(base, A.IdentifierNode):
            self.refuse(f"the buffer operand {expression} is not a buffer name")
        name = base.name
        parameters = list(getattr(function, "parameters", None) or [])
        for index, parameter in enumerate(parameters):
            if getattr(parameter, "name", None) == name:
                return self.parameter_stride(function, index)
        for node in _walk(getattr(function, "body", None)):
            if isinstance(node, A.VariableNode) and node.name == name:
                self.refuse(f"{name} is a local variable, not a declared buffer")
        declaration = self.scopes.get(id(function), {}).get(name)
        if declaration is None:
            declaration = self.globals.get(name)
        if declaration is None:
            self.refuse(f"{name} is not a declared buffer")
        return self.declaration_stride(declaration, f"the buffer {name}")

    def parameter_stride(self, function, index):
        key = (id(function), index)
        if key in self.parameter_strides:
            stride = self.parameter_strides[key]
            if stride is None:
                self.refuse(f"{function.name} passes its buffer parameter to itself")
            return stride
        self.parameter_strides[key] = None
        parameter = function.parameters[index]
        context = f"the buffer parameter {parameter.name} of {function.name}"
        strides = set()
        for caller, call_node in self.calls_to(function):
            strides.add(self.operand_stride(caller, call_node.arguments[index]))
        if len(strides) > 1:
            self.refuse(
                f"the calls of {function.name} pass buffers with strides "
                f"{', '.join(str(value) for value in sorted(strides))} as {parameter.name}"
            )
        stride = (
            strides.pop() if strides else self.declaration_stride(parameter, context)
        )
        self.parameter_strides[key] = stride
        return stride

    def calls_to(self, function):
        """``(caller, call)`` for each call that may reach ``function``.

        A call reaches a function of its name and arity. When several overloads share both, a call is kept unless
        one of its arguments is a structured buffer where ``function`` takes something else, or the reverse. That
        over-approximates: a call two buffer overloads accept counts for both, so their strides must agree.
        """
        name = getattr(function, "name", None)
        arity = len(getattr(function, "parameters", None) or [])
        overloaded = (
            sum(
                1
                for candidate, _scope in self.functions
                if getattr(candidate, "name", None) == name
                and len(getattr(candidate, "parameters", None) or []) == arity
            )
            > 1
        )
        for caller, _scope in self.functions:
            for node in _walk(getattr(caller, "body", None)):
                if (
                    isinstance(node, A.FunctionCallNode)
                    and call_name(node) == name
                    and len(node.arguments or []) == arity
                    and (not overloaded or self.may_accept(function, caller, node))
                ):
                    yield caller, node

    def may_accept(self, function, caller, call_node):
        """Whether ``function`` may be the overload ``call_node`` in ``caller`` resolves to."""
        for parameter, argument in zip(function.parameters, call_node.arguments):
            argument_type = self.argument_type(caller, argument)
            if argument_type is None:
                continue
            takes_buffer = _element_type(_declared_type(parameter)) is not None
            if takes_buffer != (_element_type(argument_type) is not None):
                return False
        return True

    def argument_type(self, function, expression):
        """The declared type of the name ``expression`` indexes inside ``function``, or ``None`` when unknown."""
        base, depth = expression, 0
        while isinstance(base, A.ArrayAccessNode):
            base, depth = base.array_expr, depth + 1
        if not isinstance(base, A.IdentifierNode):
            return None
        name = base.name
        declaration = next(
            (
                parameter
                for parameter in getattr(function, "parameters", None) or []
                if getattr(parameter, "name", None) == name
            ),
            None,
        )
        if declaration is None:
            declaration = next(
                (
                    node
                    for node in _walk(getattr(function, "body", None))
                    if isinstance(node, A.VariableNode) and node.name == name
                ),
                None,
            )
        if declaration is None:
            declaration = self.scopes.get(id(function), {}).get(name)
        if declaration is None:
            declaration = self.globals.get(name)
        type_node = _declared_type(declaration) if declaration is not None else None
        for _ in range(depth):
            if not isinstance(type_node, A.ArrayType):
                return None
            type_node = type_node.element_type
        return type_node

    # Rewriting -------------------------------------------------------------

    def split(self, function, query):
        """``buffer_dimensions(b, count)`` and ``stride = N`` for a three-output query."""
        buffer, count, stride_target = query.arguments
        stride = self.operand_stride(function, buffer)
        return [
            A.ExpressionStatementNode(call(_QUERY, [buffer, count])),
            A.ExpressionStatementNode(
                A.AssignmentNode(stride_target, _literal(stride))
            ),
        ]

    def rewrite(self, function, node, seen):
        if id(node) in seen:
            return 0
        seen.add(id(node))
        count = 0
        for field, value in list(vars(node).items()):
            if field in SKIPPED_FIELDS:
                continue
            if isinstance(value, list):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                index = 0
                while index < len(value):
                    item = value[index]
                    query = (
                        _statement_query(item) if field in _STATEMENT_FIELDS else None
                    )
                    if query is not None:
                        value[index : index + 1] = self.split(function, query)
                        index += 2
                        count += 1
                        continue
                    if isinstance(item, A.ASTNode):
                        count += self.rewrite(function, item, seen)
                    index += 1
            elif isinstance(value, A.ASTNode):
                query = _statement_query(value) if field in _STATEMENT_FIELDS else None
                if query is not None:
                    set_child(
                        node, field, None, A.BlockNode(self.split(function, query))
                    )
                    count += 1
                else:
                    count += self.rewrite(function, value, seen)
        return count


def materialize_buffer_strides(program, target) -> int:
    """Rewrite each ``buffer_dimensions`` query with a stride output for ``target``; returns how many."""
    target = str(target or "")
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    functions = [
        function
        for function in iter_functions(program)
        if any(_is_query(node) for node in _walk(getattr(function, "body", None)))
    ]
    if not functions:
        return 0
    strides = _Strides(program, target)
    count = 0
    for function in functions:
        count += strides.rewrite(function, function, set())
        if any(_is_query(node) for node in _walk(function.body)):
            strides.refuse(
                f"the query in {function.name} is used in an expression, not as a statement"
            )
    return count
