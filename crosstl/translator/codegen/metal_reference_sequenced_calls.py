"""Copied Metal reference arguments of a call whose enclosing call reaches the copied storage when it runs.

CrossGL follows GLSL for ``out`` and ``inout`` parameters (OpenGL Shading Language 4.60, section 6.1.1, "Function
Calling Conventions"): arguments are evaluated in order, left to right; an ``inout`` argument is copied in at its
turn, and an ``out`` or ``inout`` argument is copied back when the called function returns. A call that encloses
it runs after its arguments are evaluated, so it sees the copied-back value.

Metal lowers these parameters to ``thread T&``. An argument it cannot bind directly is copied into a thread
temporary before the statement and stored back after it
(``MetalHlslFunctionMixin.generate_statement_with_argument_writebacks``), or, when the rest of the statement reads
the storage afterwards, the call is evaluated first into a local (:mod:`.metal_reference_call_results`). Neither is
exact when an enclosing call reaches that storage itself while it runs:

* ``atomicAdd(tile[l], bump(tile[l]))``: the atomic operation reads and writes ``tile[l]`` after ``bump`` returns,
  but the copy-back after the statement overwrote the atomic's result;
* ``scale(tile[l], bump(tile[l]))``: ``scale``'s own copy of ``tile[l]`` is copied in before ``bump`` runs and
  stored back after ``scale`` returns, so the store of ``bump``'s copy must come between the two;
* a program function that reads or writes the storage as a module-scope variable, or through a pointer or array
  argument, after the call in its arguments returned;
* a call in an operand that is evaluated conditionally (the right operand of ``&&`` or ``||``, an arm of ``?:``,
  section 5.9): copies around the statement would run, and store back, whether or not the operand runs.

Such a call is evaluated in place as a comma expression, whose operands C++ sequences from left to right (C++14
[expr.comma]): the indices of each copied l-value that the statement could change are fixed, the copies are made
at their turn, the arguments before a side effect are evaluated into temporaries in order, the call runs, and each
copy is stored back before the comma expression yields the call's result. The enclosing call then evaluates it as
one of its own arguments. The temporaries are declared before the statement. A copied argument stored back through
a GLSL buffer block held as bytes has no store expression, and is refused.
"""

from __future__ import annotations

from ..ast import FunctionCallNode, IdentifierNode, UnaryOpNode
from .metal_reference_argument_aliasing import (
    metal_parameter_reaches_argument_storage,
)
from .metal_reference_argument_order import (
    _call_arguments,
    _copied_lvalue_indices,
    _hoisted_parameter_type,
    _index_type,
    _is_stable,
    _root_name,
    _statement_written_names,
    _storage_helpers_can_write,
)
from .metal_refusals import MetalLoweringRefused
from .wgsl_conditional_effects import is_effectful_builtin

__all__ = [
    "enclosing_call_reaches_copies",
    "generate_conditional_call_statement",
    "generate_sequenced_call_statement",
]


def _names(codegen, expr, roots):
    """Whether ``expr`` names one of the variables ``roots``."""
    return any(
        isinstance(node, IdentifierNode) and node.name in roots
        for node in codegen.iter_ast_nodes(expr)
    )


def _reaching_arguments(codegen, call, child, roots):
    """Positions of ``call``'s other arguments through which it reaches ``roots`` when it runs, or ``None``.

    ``None`` means the call reaches them otherwise: a program function that reads or writes one of them as a
    module-scope variable.
    """
    name = codegen.function_call_name(call)
    arguments = _call_arguments(call)
    others = [
        (index, argument)
        for index, argument in enumerate(arguments)
        if argument is not child
    ]
    if isinstance(name, str) and name in codegen.function_overloads_by_name:
        forwarded = set(
            codegen.required_function_thread_private_global_argument_names(name)
        )
        if forwarded & roots:
            return None
        return [
            index
            for index, argument in others
            if _root_name(codegen, _address_operand(argument)) in roots
            and metal_parameter_reaches_argument_storage(codegen, call, index)
        ]
    if not isinstance(name, str) or is_effectful_builtin(name):
        # A built-in that writes memory reaches what its other arguments name when it runs.
        return [index for index, argument in others if _names(codegen, argument, roots)]
    return []


def _address_operand(expr):
    if (
        isinstance(expr, UnaryOpNode)
        and str(getattr(expr, "op", getattr(expr, "operator", ""))) == "&"
        and not getattr(expr, "is_postfix", False)
    ):
        return expr.operand
    return expr


def enclosing_call_reaches_copies(codegen, path, roots):
    """How a call on ``path`` above its last node reaches one of ``roots`` when it runs.

    ``"in place"`` when it reaches them through an argument evaluated before the path's call: that argument's
    storage stays bound to the enclosing call, so the call is sequenced in place. ``"first"`` when it reaches them
    otherwise: evaluating the call first, with its copies stored back, gives the enclosing call the stored values.
    ``None`` when no enclosing call reaches them.
    """
    found = None
    for ancestor, child in zip(path, path[1:]):
        if not isinstance(ancestor, FunctionCallNode):
            continue
        reaching = _reaching_arguments(codegen, ancestor, child, roots)
        if reaching is None:
            found = "first"
            continue
        position = next(
            index
            for index, argument in enumerate(_call_arguments(ancestor))
            if argument is child
        )
        if any(index < position for index in reaching):
            return "in place"
        if reaching:
            found = "first"
    return found


class _Sequence:
    """The comma expression that evaluates one call with its copied reference arguments in order."""

    def __init__(self, codegen, indent):
        self.codegen = codegen
        self.indent_str = "    " * indent
        self.declarations = ""
        self.steps = []
        self.substitutions = {}
        self.previous = codegen.metal_argument_substitutions

    def temporary(self, prefix, type_text):
        self.codegen.metal_writeback_temporary_count += 1
        name = f"{prefix}{self.codegen.metal_writeback_temporary_count}"
        self.declarations += f"{self.indent_str}{type_text} {name};\n"
        return name

    def evaluate(self, prefix, type_text, expr):
        name = self.temporary(prefix, type_text)
        self.steps.append(f"{name} = {self.codegen.generate_expression(expr)}")
        self.substitute(expr, name)
        return name

    def substitute(self, expr, text):
        self.substitutions[id(expr)] = text
        self.codegen.metal_argument_substitutions = {
            **self.previous,
            **self.substitutions,
        }


def _copy_back(codegen, call, argument, source, name):
    if codegen.generate_glsl_buffer_block_store(argument, name, "=") is not None:
        raise MetalLoweringRefused(
            [
                f"reference argument '{source}' of '{codegen.function_call_name(call)}' copied back before the "
                "call that encloses it runs: it is stored through a buffer block held as bytes, which has no "
                "store expression; assign the call result to a local first"
            ]
        )
    target, value = source, name
    layout_store = codegen.metal_hlsl_layout_store_target(argument)
    if layout_store is not None:
        target, write_conversion = layout_store
        if write_conversion:
            value = f"{write_conversion}({name})"
    return f"{target} = {value}"


def _sequence_call(codegen, sequence, stmt, call, copied, discarded=False):
    """Fill ``sequence`` with the steps of ``call``; ``copied`` maps argument ids to ``(qualifier, parameter)``.

    The value of a ``discarded`` call is not used (a ``for`` update), so the comma expression ends with the last
    copy-back instead of the call's result.
    """
    written = _statement_written_names(codegen, stmt)
    shared_names = _storage_helpers_can_write(codegen)
    arguments = _call_arguments(call)
    ordered = any(
        id(argument) not in copied
        and not codegen.metal_expression_is_side_effect_free(argument)
        for argument in arguments
    )
    copy_backs = []
    for index, argument in enumerate(arguments):
        entry = copied.get(id(argument))
        if entry is None:
            if ordered and not _is_stable(codegen, argument, written, shared_names):
                raw_type = _hoisted_parameter_type(codegen, call, index)
                mapped = codegen.map_type(codegen.type_name_string(raw_type))
                sequence.evaluate("__crossgl_arg", mapped, argument)
            continue
        qualifier, parameter = entry
        for index_expr in _copied_lvalue_indices(argument):
            if not _is_stable(codegen, index_expr, written, shared_names):
                sequence.evaluate(
                    "__crossgl_index", _index_type(codegen, index_expr), index_expr
                )
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        name = sequence.temporary(
            "__crossgl_ref", codegen.map_type(codegen.type_name_string(raw_type))
        )
        source = codegen.generate_expression(argument)
        if qualifier == "inout":
            sequence.steps.append(f"{name} = {source}")
        copy_backs.append(_copy_back(codegen, call, argument, source, name))
        sequence.substitute(argument, name)
    call_code = codegen.generate_expression(call)
    result_type = codegen.type_name_string(codegen.expression_result_type(call))
    if not result_type:
        raise MetalLoweringRefused(
            [
                f"reference arguments of '{codegen.function_call_name(call)}' copied back before the call that "
                "encloses it runs: the call's result type is unknown"
            ]
        )
    result = None
    if result_type == "void" or discarded:
        sequence.steps.append(call_code)
    else:
        result = sequence.temporary("__crossgl_call", codegen.map_type(result_type))
        sequence.steps.append(f"{result} = {call_code}")
    sequence.steps.extend(copy_backs)
    if result is not None:
        sequence.steps.append(result)


def _nested_copies(codegen, call, writebacks):
    """Whether a copied argument of ``writebacks`` lies inside an argument of ``call`` without being one."""
    arguments = _call_arguments(call)
    return any(
        not any(argument is candidate for candidate in arguments)
        and any(
            node is argument
            for candidate in arguments
            for node in codegen.iter_ast_nodes(candidate)
        )
        for argument, _qualifier, _parameter in writebacks
    )


def generate_sequenced_call_statement(
    codegen, stmt, indent, call, writebacks, discarded=False
):
    """Emit ``stmt`` with ``call`` evaluated in place as a comma expression that stores its copies back."""
    arguments = _call_arguments(call)
    copied = {
        id(argument): (qualifier, parameter)
        for argument, qualifier, parameter in writebacks
        if any(argument is candidate for candidate in arguments)
    }
    if _nested_copies(codegen, call, writebacks):
        raise MetalLoweringRefused(
            [
                f"reference arguments of '{codegen.function_call_name(call)}' copied back before the call that "
                "encloses it runs: another argument copies a reference argument of its own; assign the inner "
                "call's result to a local first"
            ]
        )
    sequence = _Sequence(codegen, indent)
    try:
        _sequence_call(codegen, sequence, stmt, call, copied, discarded)
    finally:
        codegen.metal_argument_substitutions = sequence.previous
    expression = f"({', '.join(sequence.steps)})"
    sequenced = codegen.metal_sequenced_call_ids
    codegen.metal_argument_substitutions = {
        **sequence.previous,
        id(call): expression,
    }
    sequenced.add(id(call))
    try:
        # The rest of the statement is generated again; copies of the calls around this one are found there.
        body = codegen.generate_statement(stmt, indent)
    finally:
        sequenced.discard(id(call))
        codegen.metal_argument_substitutions = sequence.previous
    return sequence.declarations + body


def generate_conditional_call_statement(
    codegen, stmt, indent, calls, writebacks, discarded=()
):
    """Emit ``stmt`` when ``calls`` copy reference arguments in operands that are evaluated conditionally.

    Copies made before the statement and stored back after it would run whether or not the operand is evaluated.
    Each such call is evaluated in place instead, so its copies are made and stored back only when it runs. An
    inner call is sequenced before the call whose argument holds it; the statement is generated again for the rest.
    ``discarded`` holds the calls whose value is not used.
    """
    for call in calls:
        if not _nested_copies(codegen, call, writebacks):
            return generate_sequenced_call_statement(
                codegen,
                stmt,
                indent,
                call,
                writebacks,
                any(call is other for other in discarded),
            )
    raise MetalLoweringRefused(
        [
            f"reference arguments of '{codegen.function_call_name(calls[0])}' in a conditionally evaluated "
            "expression: each call that copies arguments holds another such call"
        ]
    )
