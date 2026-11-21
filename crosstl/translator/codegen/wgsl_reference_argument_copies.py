"""Copy-in/copy-out for by-reference arguments that WGSL cannot pass as pointers.

GLSL, HLSL and Metal pass ``inout``/``out`` arguments by copy-in/copy-out, and
the argument may be any l-value: ``bump(values[i])``, ``bump(s.count)``,
``bump(buffer.data[3])``, or a module-scope variable. All arguments are
evaluated once, left to right, when the call starts; an ``inout`` argument is
copied into the parameter then, and the l-value it designates receives the
parameter's value when the call returns (GLSL 4.60 section 6.1.1).

Core WGSL only takes a ``ptr<function, T>`` argument that is a whole
function-scope variable or a pointer parameter (WGSL 11.3.1). Those keep the
direct ``&v`` lowering of ``wgsl_parameter_lowering``. For every other
l-value, ``lower_reference_argument_copies`` rewrites the statement holding
the call into:

* a ``let`` for each array index of the l-value that a later step could
  change, so copy-in and copy-out address the same element;
* ``var _crosstl_ref_N: T = lvalue;`` (no initializer for ``out``), the
  copy-in, and ``&_crosstl_ref_N`` as the argument;
* the call itself, as a statement or as ``let _crosstl_call_N: R = f(...);``
  whose name replaces the call in the statement;
* ``lvalue = _crosstl_ref_N;``, the copy-out, for every copied argument in
  parameter order.

Earlier arguments of the call that contain a call are first evaluated into a
``let`` of the parameter type, so they still run before the copy-in. The
lowering moves the call ahead of the rest of its statement, so it raises
``ValueError`` when that would reorder it against another call with possible
side effects, or when the call sits where the source may skip it (the right
operand of ``&&``/``||``, a ternary branch or a ``for`` update). A ``while``
or ``for`` condition becomes a guard at the top of the loop body, so the copies
still run once per test. A ``do``-``while`` condition is the ``break if`` of a
``continuing`` block (``wgsl_loop_continuing``), which runs after every
iteration and every ``continue``; its copies precede the ``break if`` there.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ConstructorNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LambdaNode,
    LiteralNode,
    LoopNode,
    MemberAccessNode,
    ReturnNode,
    SwitchNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..refusals import WGSLRefusal
from .wgsl_loop_continuing import WGSLContinuingLoopNode
from .wgsl_must_use import WGSL_MUST_USE_BUILTINS
from .wgsl_parameter_lowering import lvalue_root, reference_parameter_qualifier
from .wgsl_read_only_globals import validate_writable_module_root
from .wgsl_value_side_effects import SHORT_CIRCUIT_OPERATORS, replaced

REFERENCE_COPY_PREFIX = "_crosstl_ref_"
CALL_RESULT_PREFIX = "_crosstl_call_"
CAPTURED_VALUE_PREFIX = "_crosstl_arg_"


class _Walk:
    """Evaluation-order state of one statement rewrite."""

    def __init__(self):
        self.prelude = []
        # A call that stays in the statement and is evaluated before a later
        # copied call would run after it once that call is hoisted.
        self.call_left_in_place = None


class WGSLReferenceArgumentCopiesMixin:
    """Statement lowering for copied by-reference arguments; see the module docstring."""

    # ------------------------------------------------------------------ detection

    def copied_reference_call(self, node):
        """``(function, copied argument indices)`` when ``node`` needs copies, else None."""
        if not isinstance(node, FunctionCallNode):
            return None
        if not isinstance(node.function, IdentifierNode):
            return None
        candidates = self.reference_argument_indices_by_name(node)
        if not any(
            index < len(node.arguments)
            and self.is_copyable_reference_argument(node.arguments[index])
            and self.direct_reference_argument(node.arguments[index]) is None
            for index in candidates
        ):
            return None
        function = self.resolve_function_overload(node.function.name, node.arguments)
        if function is None:
            return None
        copied = []
        for index, parameter in enumerate(getattr(function, "parameters", None) or ()):
            if index >= len(node.arguments):
                break
            if reference_parameter_qualifier(parameter) is None:
                continue
            argument = node.arguments[index]
            if not self.is_copyable_reference_argument(argument):
                continue
            if self.direct_reference_argument(argument) is None:
                copied.append(index)
        if not copied:
            return None
        return function, copied

    def is_copyable_reference_argument(self, argument):
        """An l-value whose root the caller may write: not a value parameter WGSL keeps immutable."""
        root = lvalue_root(argument)
        if root is None:
            return False
        if self.is_reference_parameter_name(root.name):
            return True
        plan = self.current_parameter_plan()
        if plan is not None and self.is_current_parameter_name(root.name):
            return root.name in plan["copies"]
        return True

    def contains_copied_reference_call(self, expr):
        if expr is None:
            return False
        for node in expr.walk():
            if self.copied_reference_call(node) is not None:
                return True
        return False

    def is_effect_free_call(self, node):
        """Whether a call keeps its place without reordering an observable effect."""
        if isinstance(node, ConstructorNode):
            return True
        if not isinstance(node, FunctionCallNode):
            return True
        if not isinstance(node.function, IdentifierNode):
            return False
        name = node.function.name
        if self.is_user_function_name(name):
            return False
        if name in getattr(self, "_structs_by_name", {}):
            return True
        return bool(self.is_type_constructor_name(name)) or name in (
            WGSL_MUST_USE_BUILTINS
        )

    def contains_effectful_call(self, expr):
        return any(
            isinstance(node, FunctionCallNode) and not self.is_effect_free_call(node)
            for node in expr.walk()
        )

    # ------------------------------------------------------------- statements

    def lower_reference_argument_copies(self, stmt):
        """The statements that replace ``stmt``, or None when it passes no copied argument."""
        if not getattr(self, "_function_overloads_by_name", None):
            return None
        if isinstance(stmt, (ExpressionStatementNode, AssignmentNode)):
            return self.lower_copied_statement_expression(stmt)
        if isinstance(stmt, VariableNode):
            return self.lower_copied_field(stmt, "initial_value")
        if isinstance(stmt, ReturnNode):
            return self.lower_copied_field(stmt, "value")
        if isinstance(stmt, SwitchNode):
            return self.lower_copied_field(stmt, "expression")
        if isinstance(stmt, IfNode):
            return self.lower_copied_if(stmt)
        if isinstance(stmt, WhileNode):
            return self.lower_copied_while(stmt)
        if isinstance(stmt, ForNode):
            return self.lower_copied_for(stmt)
        if isinstance(stmt, WGSLContinuingLoopNode):
            return self.lower_copied_break_if(stmt)
        if isinstance(stmt, DoWhileNode) and self.contains_copied_reference_call(
            stmt.condition
        ):
            # The WGSL emitter lowers every do-while to a continuing loop first.
            self.reject_conditional_copy(
                "a do-while condition",
                "the condition also runs after every continue, where no statement "
                "can hold the copies",
            )
        return None

    def lower_copied_break_if(self, stmt):
        """``stmt`` with the copies of its ``break if`` condition last in its ``continuing`` block."""
        if stmt.break_if is None or not self.contains_copied_reference_call(
            stmt.break_if
        ):
            return None
        walk = _Walk()
        condition = self.rewrite_copied_calls(stmt.break_if, walk)
        return [
            replaced(
                stmt,
                continuing=[*stmt.continuing, *walk.prelude],
                break_if=condition,
            )
        ]

    def lower_copied_statement_expression(self, stmt):
        expr = stmt.expression if isinstance(stmt, ExpressionStatementNode) else stmt
        if not self.contains_copied_reference_call(expr):
            return None
        walk = _Walk()
        if isinstance(expr, FunctionCallNode) and self.copied_reference_call(expr):
            # A call statement: run it in place, with no result variable.
            self.hoist_copied_call(expr, walk, statement=True)
            return walk.prelude
        if isinstance(expr, AssignmentNode):
            target = self.rewrite_copied_calls(expr.target, walk)
            value = self.rewrite_copied_calls(expr.value, walk)
            lowered = replaced(expr, target=target, value=value)
        else:
            lowered = self.rewrite_copied_calls(expr, walk)
        if isinstance(stmt, ExpressionStatementNode):
            lowered = replaced(stmt, expression=lowered)
        return [*walk.prelude, lowered]

    def lower_copied_field(self, stmt, field_name):
        value = getattr(stmt, field_name, None)
        if not self.contains_copied_reference_call(value):
            return None
        walk = _Walk()
        lowered = self.rewrite_copied_calls(value, walk)
        return [*walk.prelude, replaced(stmt, **{field_name: lowered})]

    def lower_copied_if(self, stmt):
        prelude = []
        condition = stmt.condition
        if self.contains_copied_reference_call(condition):
            walk = _Walk()
            condition = self.rewrite_copied_calls(condition, walk)
            prelude = walk.prelude
        else_branch = stmt.else_branch
        if isinstance(else_branch, IfNode) and self.contains_copied_reference_call(
            else_branch.condition
        ):
            # Generated as a statement of its own block, so its copies run
            # only when the earlier conditions are false.
            else_branch = BlockNode([else_branch])
        if condition is stmt.condition and else_branch is stmt.else_branch:
            return None
        return [*prelude, replaced(stmt, condition=condition, else_branch=else_branch)]

    def copied_loop_guard(self, condition):
        return IfNode(UnaryOpNode("!", condition), BlockNode([BreakNode()]))

    def lower_copied_while(self, stmt):
        if not self.contains_copied_reference_call(stmt.condition):
            return None
        guard = self.copied_loop_guard(stmt.condition)
        return [LoopNode(BlockNode([guard, stmt.body]))]

    def lower_copied_for(self, stmt):
        if self.contains_copied_reference_call(stmt.update):
            self.reject_conditional_copy(
                "a for-loop update",
                "the update also runs after every continue, where no statement "
                "can hold the copies",
            )
        prelude, init = [], stmt.init
        if isinstance(init, VariableNode) and self.contains_copied_reference_call(
            init.initial_value
        ):
            walk = _Walk()
            value = self.rewrite_copied_calls(init.initial_value, walk)
            prelude, init = walk.prelude, replaced(init, initial_value=value)
        elif init is not None and not isinstance(init, VariableNode):
            expression = getattr(init, "expression", init)
            if self.contains_copied_reference_call(expression):
                prelude = self.lower_copied_statement_expression(init)
                init = None
        condition, body = stmt.condition, stmt.body
        if self.contains_copied_reference_call(condition):
            body = BlockNode([self.copied_loop_guard(condition), body])
            condition = None
        if init is stmt.init and body is stmt.body:
            return None
        loop = replaced(stmt, init=init, condition=condition, body=body)
        if not prelude:
            return [loop]
        return [BlockNode([*prelude, loop])]

    # ------------------------------------------------------------ expressions

    def rewrite_copied_calls(self, expr, walk, conditional=False):
        """``expr`` with each copied call moved into ``walk.prelude``, in evaluation order."""
        if expr is None or isinstance(expr, LambdaNode):
            return expr
        if not self.contains_copied_reference_call(expr):
            self.note_calls_left_in_place(expr, walk)
            return expr
        rewrite = self.rewrite_copied_calls
        if isinstance(expr, FunctionCallNode) and self.copied_reference_call(expr):
            if conditional:
                self.reject_conditional_copy(
                    "the right operand of && or ||, or a ternary branch"
                )
            if walk.call_left_in_place is not None:
                raise WGSLRefusal(
                    "WGSL target cannot copy the by-reference arguments of "
                    f"{expr.function.name} in place: the copies move the call ahead "
                    "of an earlier call in the same statement; call it in a "
                    "statement of its own"
                )
            return self.hoist_copied_call(expr, walk)
        if isinstance(expr, BinaryOpNode):
            left = rewrite(expr.left, walk, conditional)
            right_conditional = conditional or expr.operator in SHORT_CIRCUIT_OPERATORS
            right = rewrite(expr.right, walk, right_conditional)
            return replaced(expr, left=left, right=right)
        if isinstance(expr, TernaryOpNode):
            return replaced(
                expr,
                condition=rewrite(expr.condition, walk, conditional),
                true_expr=rewrite(expr.true_expr, walk, True),
                false_expr=rewrite(expr.false_expr, walk, True),
            )
        if isinstance(expr, UnaryOpNode):
            return replaced(expr, operand=rewrite(expr.operand, walk, conditional))
        if isinstance(expr, CastNode):
            return replaced(
                expr, expression=rewrite(expr.expression, walk, conditional)
            )
        if isinstance(expr, MemberAccessNode):
            return replaced(
                expr, object_expr=rewrite(expr.object_expr, walk, conditional)
            )
        if isinstance(expr, SwizzleNode):
            return replaced(
                expr, vector_expr=rewrite(expr.vector_expr, walk, conditional)
            )
        if isinstance(expr, ArrayAccessNode):
            array_expr = rewrite(expr.array_expr, walk, conditional)
            index_expr = rewrite(expr.index_expr, walk, conditional)
            return replaced(expr, array_expr=array_expr, index_expr=index_expr)
        if isinstance(expr, (FunctionCallNode, ConstructorNode)):
            arguments = [
                rewrite(argument, walk, conditional) for argument in expr.arguments
            ]
            lowered = replaced(expr, arguments=arguments)
            if not self.is_effect_free_call(expr):
                walk.call_left_in_place = expr
            return lowered
        raise WGSLRefusal(
            "WGSL target cannot order the by-reference argument copies of a call "
            f"inside {type(expr).__name__}"
        )

    def note_calls_left_in_place(self, expr, walk):
        if walk.call_left_in_place is None and self.contains_effectful_call(expr):
            walk.call_left_in_place = expr

    def hoist_copied_call(self, call, walk, statement=False):
        """Emit copy-in, the call and copy-out; return what replaces the call."""
        function, copied = self.copied_reference_call(call)
        self.validate_reference_argument_aliasing(function, call.arguments)
        parameters = list(getattr(function, "parameters", None) or ())
        reference_roots = set()
        for index, argument in enumerate(call.arguments):
            parameter = parameters[index] if index < len(parameters) else None
            root = lvalue_root(argument)
            if root is not None and reference_parameter_qualifier(parameter):
                reference_roots.add(root.name)
        nested = [
            index
            for index, argument in enumerate(call.arguments)
            if index not in copied and self.contains_copied_reference_call(argument)
        ]
        last_nested = max(nested, default=-1)
        last_copied = max(copied)
        arguments, copy_outs = [], []
        for index, argument in enumerate(call.arguments):
            parameter = parameters[index] if index < len(parameters) else None
            if index in copied:
                argument = self.copy_in_reference_argument(
                    argument, parameter, walk, reference_roots, copy_outs
                )
            elif (
                index in nested
                or index < last_nested
                or (index < last_copied and self.contains_effectful_call(argument))
            ):
                # Evaluated now, before a later copy-in or a later hoisted call.
                argument = self.ordered_argument(
                    argument, parameter, walk, reference_roots, index < last_nested
                )
            arguments.append(argument)
        lowered = replaced(call, arguments=arguments)
        if statement:
            walk.prelude.append(ExpressionStatementNode(lowered))
            walk.prelude.extend(copy_outs)
            return None
        return_type = getattr(function, "return_type", None)
        if return_type is None or self.is_void_local_declaration_type(return_type):
            raise WGSLRefusal(
                f"WGSL target cannot use the void call of {call.function.name} as a value"
            )
        result = self.next_reference_copy_name(CALL_RESULT_PREFIX)
        walk.prelude.append(
            VariableNode(result, return_type, lowered, is_mutable=False)
        )
        walk.prelude.extend(copy_outs)
        return IdentifierNode(result)

    def ordered_argument(self, argument, parameter, walk, reference_roots, capture):
        """An argument evaluated ahead of a later copy, into a ``let`` when it could change.

        ``capture`` marks an argument that precedes a hoisted call, which may
        write anything the argument reads.
        """
        nested = _Walk()
        argument = self.rewrite_copied_calls(argument, nested)
        walk.prelude.extend(nested.prelude)
        if reference_parameter_qualifier(parameter) is not None:
            return argument
        if not capture and not self.contains_effectful_call(argument):
            return argument
        if self.is_stable_reference_index(argument, reference_roots):
            return argument
        if parameter is None or not self.is_plain_value_parameter(parameter):
            if isinstance(argument, IdentifierNode):
                return argument
            raise WGSLRefusal(
                "WGSL target cannot evaluate a resource argument ahead of the "
                "by-reference argument copies of the same call"
            )
        name = self.next_reference_copy_name(CAPTURED_VALUE_PREFIX)
        walk.prelude.append(
            VariableNode(name, parameter.param_type, argument, is_mutable=False)
        )
        return IdentifierNode(name)

    def copy_in_reference_argument(
        self, argument, parameter, walk, reference_roots, copy_outs
    ):
        qualifier = reference_parameter_qualifier(parameter)
        validate_writable_module_root(
            self, argument, f"pass '{{name}}' to {qualifier} parameter {parameter.name}"
        )
        lvalue = self.stable_reference_lvalue(argument, walk, reference_roots)
        name = self.next_reference_copy_name(REFERENCE_COPY_PREFIX)
        initial = lvalue if qualifier == "inout" else None
        walk.prelude.append(VariableNode(name, parameter.param_type, initial))
        copy_outs.append(AssignmentNode(lvalue, IdentifierNode(name)))
        return IdentifierNode(name)

    def stable_reference_lvalue(self, lvalue, walk, reference_roots):
        """``lvalue`` with each index that could change before copy-out bound to a ``let``."""
        if isinstance(lvalue, IdentifierNode):
            return lvalue
        if isinstance(lvalue, MemberAccessNode):
            inner = self.stable_reference_lvalue(
                lvalue.object_expr, walk, reference_roots
            )
            return replaced(lvalue, object_expr=inner)
        if isinstance(lvalue, SwizzleNode):
            components = str(lvalue.components)
            if len(set(components)) != len(components):
                raise WGSLRefusal(
                    f"WGSL target cannot copy out to swizzle .{components}: it repeats "
                    "a component, so it is not an l-value"
                )
            inner = self.stable_reference_lvalue(
                lvalue.vector_expr, walk, reference_roots
            )
            return replaced(lvalue, vector_expr=inner)
        if isinstance(lvalue, ArrayAccessNode):
            inner = self.stable_reference_lvalue(
                lvalue.array_expr, walk, reference_roots
            )
            index = lvalue.index_expr
            if not self.is_stable_reference_index(index, reference_roots):
                nested = _Walk()
                index = self.rewrite_copied_calls(index, nested)
                walk.prelude.extend(nested.prelude)
                name = self.next_reference_copy_name(CAPTURED_VALUE_PREFIX)
                walk.prelude.append(VariableNode(name, None, index, is_mutable=False))
                index = IdentifierNode(name)
            return replaced(lvalue, array_expr=inner, index_expr=index)
        raise WGSLRefusal(
            f"WGSL target cannot copy out a by-reference argument through {type(lvalue).__name__}"
        )

    def is_stable_reference_index(self, index, reference_roots):
        """Whether neither the callee nor another argument can change ``index``."""
        if isinstance(index, LiteralNode):
            return True
        if not isinstance(index, IdentifierNode):
            return False
        name = index.name
        return (
            self.is_local_identifier(name)
            and not self.is_reference_parameter_name(name)
            and name not in reference_roots
        )

    def reject_conditional_copy(self, where, reason=None):
        reason = reason or (
            "the copy-in and copy-out would run even when the source skips the call"
        )
        raise WGSLRefusal(
            "WGSL target cannot copy a by-reference argument that is not a whole "
            f"local variable in {where}: {reason}"
        )

    def next_reference_copy_name(self, prefix):
        index = getattr(self, "_reference_copy_count", 0)
        self._reference_copy_count = index + 1
        return f"{prefix}{index}"
