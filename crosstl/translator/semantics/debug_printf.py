"""Debug printing: CrossGL ``printf`` refused for the targets that cannot print from a shader.

CrossGL ``printf(format, values...)`` is the shader debug print of HLSL compiled to SPIR-V, Slang and GLSL's
GL_EXT_debug_printf ``debugPrintfEXT``, with the Vulkan debug printf format. SPIR-V carries it as
``NonSemantic.DebugPrintf``, GLSL as ``debugPrintfEXT``, Metal 3.2 as ``os_log_default.log`` and HLSL as its
``printf`` intrinsic. HLSL's ``printf`` is a SPIR-V feature: DXC writes it as ``NonSemantic.DebugPrintf`` with
``-spirv`` and, since DXIL has no print operation, rejects it with "use of unsupported identifier 'printf'" when it
compiles to DXIL, so HLSL output keeps the print and DXIL compilation fails at it rather than dropping it.
:func:`refuse_debug_printf` refuses the targets that have no shader print: WebGL 2 (GLSL ES 3.00) and WGSL define
no shader print and no extension that adds one.

A debug print is a statement. GL_EXT_debug_printf's ``debugPrintfEXT``, SPIR-V's ``NonSemantic.DebugPrintf`` (an
``OpExtInst`` of result type ``OpTypeVoid``), HLSL's and Slang's ``printf``, Metal's ``os_log_default.log``, rust-gpu's
``debug_printf!`` and Mojo's ``print`` return nothing. C's ``printf`` returns an ``int`` whose meaning each language
defines apart: the number of arguments CUDA parsed, or 0 on success and -1 on failure in OpenCL C (OpenCL C 3.0
section 6.15.14.3). :func:`refuse_debug_printf_values` therefore refuses a print whose value the program reads, with
:class:`DebugPrintfValueUnsupportedError`, except when a CUDA, HIP or OpenCL C source is written back to its own
language, whose ``printf`` computes the value the source read. CrossGL output keeps it.

A program that declares its own ``printf`` function calls that function, which every target can.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions

__all__ = [
    "DEBUG_PRINTF_FUNCTIONS",
    "DebugPrintfUnsupportedError",
    "DebugPrintfValueUnsupportedError",
    "refuse_debug_printf",
    "refuse_debug_printf_values",
]

DEBUG_PRINTF_FUNCTIONS = frozenset({"printf", "debugPrintfEXT"})

_REASONS = {
    "webgl": "WebGL 2 (GLSL ES 3.00) has no shader print",
    "wgsl": "WGSL has no shader print",
    "webgpu": "WGSL has no shader print",
}

# The languages whose printf returns a value; each defines it apart, so only a source of that language reads it.
_VALUE_PRINT_LANGUAGES = frozenset({"cuda", "hip", "opencl"})
_CROSSGL_TARGETS = frozenset({"", "cgl", "crossgl"})
_SKIPPED_FIELDS = frozenset({"parent", "annotations", "source_location"})
_EXPRESSION_FIELD = frozenset({"expression"})
_FOR_EFFECT_FIELDS = frozenset({"init", "update"})
_STATEMENT_FIELD = frozenset({"statements"})


class DebugPrintfUnsupportedError(TranslationDiagnostic):
    """A debug print for a target that cannot print from a shader."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.debug-printf",)
        super().__init__(
            f"debug print '{name}' cannot be carried to {target}: {_REASONS[target]}",
            target=target,
            code=f"{CODE_PREFIX}{target}-debug-printf-unsupported",
        )


class DebugPrintfValueUnsupportedError(TranslationDiagnostic):
    """A debug print whose value the program reads, for a target whose print has no value or another one."""

    def __init__(self, source, target, name, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.debug-printf-value",)
        if target in _VALUE_PRINT_LANGUAGES:
            reason = f"{target}'s printf result is not the value {source or 'CrossGL'} defines"
        else:
            reason = f"{target}'s debug print is a statement with no value"
        super().__init__(
            f"the value of debug print '{name}' cannot be carried to {target}: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-debug-printf-value-unsupported",
        )


def _call_name(node):
    function = getattr(node, "function", None) or getattr(node, "name", None)
    return getattr(function, "name", function)


def _nodes(node, seen):
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _nodes(item, seen)
        return
    if isinstance(node, dict):
        for item in node.values():
            yield from _nodes(item, seen)
        return
    if not isinstance(node, A.ASTNode):
        return
    yield node
    for key, value in vars(node).items():
        if key not in {"parent", "source_location"}:
            yield from _nodes(value, seen)


def refuse_debug_printf(program, target) -> None:
    """Raise :class:`DebugPrintfUnsupportedError` for a debug print in ``program`` that ``target`` cannot carry."""
    target = str(target or "").lower()
    if target not in _REASONS or not isinstance(program, A.ShaderNode):
        return
    declared = {getattr(function, "name", None) for function in iter_functions(program)}
    for node in _nodes(program, set()):
        name = _call_name(node) if isinstance(node, A.FunctionCallNode) else None
        if name in DEBUG_PRINTF_FUNCTIONS and name not in declared:
            raise DebugPrintfUnsupportedError(
                target, name, source_location=getattr(node, "source_location", None)
            )


def _read_prints(program, declared):
    """The debug-print calls in ``program`` whose value is read, in source order.

    Type nodes are not walked: no language reads a print's value in a type.
    """
    found = []
    seen = set()
    # (node, discarded): ``discarded`` says the value of the expression ``node`` is not read.
    stack = [(program, False)]
    pop = stack.pop
    kinds = A.FIELD_KINDS
    node_class = A.ASTNode
    type_class = A.TypeNode
    while stack:
        node, discarded = pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        if not discarded and isinstance(node, A.FunctionCallNode):
            name = _call_name(node)
            if name in DEBUG_PRINTF_FUNCTIONS and name not in declared:
                found.append((node, name))
        discarding = _discarding_fields(node, discarded)
        children = []
        append = children.append
        for field, value in vars(node).items():
            if field in _SKIPPED_FIELDS:
                continue
            kind = kinds.get(type(value))
            if kind is None:
                kind = A.field_kind(value)
            if kind == A.FIELD_LEAF or kind == A.FIELD_SET:
                continue
            if kind == A.FIELD_NODE:
                items = (value,)
            elif kind == A.FIELD_LIST or kind == A.FIELD_TUPLE:
                items = value
            else:
                items = value.values()
            child_discarded = discarding is not None and discarding(field)
            for item in items:
                if isinstance(item, node_class) and not isinstance(item, type_class):
                    append((item, child_discarded))
        if isinstance(node, A.CommaExpressionNode):
            # Every operand but the last is evaluated for its effects only.
            last = len(children) - 1
            children = [
                (item, discarded or index < last)
                for index, (item, _) in enumerate(children)
            ]
        children.reverse()
        stack.extend(children)
    return found


def _discarding_fields(node, discarded):
    """A predicate on ``node``'s field names: the expression the field holds is evaluated for its effects only."""
    if isinstance(node, A.ExpressionStatementNode):
        if getattr(node, "is_tail_expression", False):
            return None
        return _EXPRESSION_FIELD.__contains__
    if isinstance(node, A.ForNode):
        return _FOR_EFFECT_FIELDS.__contains__
    # A statement list: a bare call an importer put there is a statement.
    if isinstance(node, (A.BlockNode, A.CaseNode)):
        return _STATEMENT_FIELD.__contains__
    return None


def refuse_debug_printf_values(program, source, target) -> None:
    """Raise :class:`DebugPrintfValueUnsupportedError` for a debug print whose value ``target`` cannot compute."""
    source = str(source or "").lower()
    target = str(target or "").lower()
    if target in _CROSSGL_TARGETS or not isinstance(program, A.ShaderNode):
        return
    if target in _VALUE_PRINT_LANGUAGES and source == target:
        return
    declared = {getattr(function, "name", None) for function in iter_functions(program)}
    for node, name in _read_prints(program, declared):
        raise DebugPrintfValueUnsupportedError(
            source, target, name, source_location=getattr(node, "source_location", None)
        )
