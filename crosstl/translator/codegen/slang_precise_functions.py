"""Slang emission of CrossGL ``precise`` functions.

A CrossGL function marked ``precise`` evaluates all of its floating-point arithmetic as written. Slang ignores
``precise`` on a function: ``precise void k()`` and ``precise float f()`` compile to SPIR-V without a single
``NoContraction`` decoration (checked with slangc 2026.20). A ``precise`` local does work, and Slang carries it
backward through data flow: every operation whose result reaches the local is decorated, including the operands of
an inlined call.

These members therefore declare every scalar, vector and matrix local of a precise function ``precise`` and
capture, in a precise local declared at function entry, each value the function stores elsewhere, returns or
branches on: ``o[i] = (crossglPreciseValue = a * b - c)``. The capture is evaluated where the value was, so the order
of evaluation and short-circuiting are kept. A value that has no precise Slang local, such as a structure built
from arithmetic, is refused with :class:`~crosstl.translator.refusals.SlangRefusal` rather than emitted without the
guarantee.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

_SCALAR = (
    r"(?:bool|int|uint|dword|half|float|double|min16float|min10float|min16int|min16uint|"
    r"float16_t|float32_t|float64_t|int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t)"
)

# Scalar, vector and matrix types whose values a precise Slang local can hold.
_PRECISE_VALUE_TYPE = re.compile(
    rf"^(?:{_SCALAR}(?:[1-4](?:x[1-4])?)?"
    rf"|vector\s*<\s*{_SCALAR}\s*,\s*[1-4]\s*>"
    rf"|matrix\s*<\s*{_SCALAR}\s*,\s*[1-4]\s*,\s*[1-4]\s*>)$"
)

# Locals with these qualifiers are shared or static storage, not values of one invocation of the function.
_NON_VALUE_LOCAL_QUALIFIERS = frozenset(
    {"groupshared", "shared", "threadgroup", "workgroup", "buffer", "static", "const"}
)


class SlangPreciseFunctionMixin:
    """``precise`` function members of ``SlangCodeGen``."""

    current_slang_precise_function = False
    slang_precise_captures = None
    current_slang_precise_locals = frozenset()
    current_slang_body_locals = frozenset()

    def enter_slang_precise_function(self, func, body_statements):
        """Start emitting ``func``'s body; return the state :meth:`leave_slang_precise_function` restores."""
        previous = (
            self.current_slang_precise_function,
            self.slang_precise_captures,
            self.current_slang_precise_locals,
            self.current_slang_body_locals,
        )
        precise = bool(self.slang_function_is_precise(func)) and bool(body_statements)
        self.current_slang_precise_function = precise
        self.slang_precise_captures = [] if precise else None
        if precise:
            locals_, value_locals = self.slang_precise_body_locals(body_statements)
            self.current_slang_body_locals = frozenset(locals_)
            self.current_slang_precise_locals = frozenset(value_locals)
        else:
            self.current_slang_body_locals = frozenset()
            self.current_slang_precise_locals = frozenset()
        return previous

    def leave_slang_precise_function(self, previous):
        (
            self.current_slang_precise_function,
            self.slang_precise_captures,
            self.current_slang_precise_locals,
            self.current_slang_body_locals,
        ) = previous

    @staticmethod
    def slang_precise_value_type(mapped_type):
        return isinstance(mapped_type, str) and bool(
            _PRECISE_VALUE_TYPE.match(mapped_type.strip())
        )

    def slang_precise_declared_type(self, node):
        declared = getattr(node, "var_type", None) or getattr(node, "vtype", None)
        if declared is None:
            return None
        try:
            return self.map_type(declared)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def slang_precise_local_qualifiers(node):
        return {str(value).lower() for value in getattr(node, "qualifiers", []) or []}

    def slang_precise_body_locals(self, body_statements):
        """The names ``body_statements`` declares, and those that only ever hold a precise value type."""
        declared, non_value = set(), set()
        for node in self.walk_ast(body_statements):
            if not isinstance(node, VariableNode) or not getattr(node, "name", None):
                continue
            declared.add(node.name)
            qualifiers = self.slang_precise_local_qualifiers(node)
            if qualifiers & _NON_VALUE_LOCAL_QUALIFIERS or not (
                self.slang_precise_value_type(self.slang_precise_declared_type(node))
            ):
                non_value.add(node.name)
        return declared, declared - non_value

    def slang_precise_local_prefix(self, node, declaration_prefix):
        """``precise `` for a local of the current precise function that the declaration does not mark already."""
        if not self.current_slang_precise_function:
            return ""
        if getattr(node, "name", None) not in self.current_slang_precise_locals:
            return ""
        if re.search(r"\b(?:precise|static|groupshared)\b", declaration_prefix):
            return ""
        return "precise "

    def slang_precise_local_initializer(self, node, initial_value):
        """Refuse an initializer whose arithmetic a local of a non-value type cannot keep exact."""
        if not self.current_slang_precise_function or initial_value is None:
            return
        if getattr(node, "name", None) in self.current_slang_precise_locals:
            return
        if self.slang_precise_local_qualifiers(node) & {"const", "static"}:
            return
        if self.slang_expression_may_round(initial_value):
            self.refuse_slang_precise(
                f"local '{node.name}' is initialized with floating-point arithmetic, "
                "and its type has no precise Slang local"
            )

    def slang_expression_may_round(self, expression):
        """Return whether evaluating ``expression`` may round a floating-point operation.

        Operations of unknown type count as floating point, so a value is captured rather than left to
        contraction.
        """
        for node in self.walk_ast(expression):
            if isinstance(
                node, (BinaryOpNode, UnaryOpNode, TernaryOpNode, FunctionCallNode)
            ):
                return True
        return False

    def slang_precise_root_name(self, expression):
        while isinstance(expression, (ArrayAccessNode, MemberAccessNode)):
            expression = (
                expression.array
                if isinstance(expression, ArrayAccessNode)
                else getattr(
                    expression, "object", getattr(expression, "object_expr", None)
                )
            )
        if isinstance(expression, IdentifierNode):
            return expression.name
        if isinstance(expression, str):
            return expression
        return getattr(expression, "name", None)

    def slang_precise_needs_capture(self, value):
        """Whether ``value`` carries arithmetic of this function that no precise local holds yet."""
        if value is None or isinstance(value, LiteralNode):
            return False
        if self.slang_expression_may_round(value):
            return True
        root = self.slang_precise_root_name(value)
        return (
            root in self.current_slang_body_locals
            and root not in self.current_slang_precise_locals
        )

    def slang_precise_capture(self, rendered, mapped_type, base_name):
        name = self.unique_expression_temp_name(base_name)
        self.slang_precise_captures.append((mapped_type, name))
        return f"({name} = {rendered})"

    def slang_precise_value_capture(
        self, value, rendered, value_type, what, base_name="crossglPreciseValue"
    ):
        """Capture ``rendered`` in a precise local of ``value_type``, or refuse when no local can hold it."""
        if not self.slang_precise_needs_capture(value):
            return rendered
        mapped = None
        for candidate in (value_type, self.slang_precise_expression_type(value)):
            if candidate is None:
                continue
            try:
                mapped = self.map_type(candidate)
            except (TypeError, ValueError):
                mapped = None
            if self.slang_precise_value_type(mapped):
                break
            mapped = None
        if mapped is None:
            if not self.slang_expression_may_round(value):
                # A copy of an aggregate local does no arithmetic of its own; each store into it was captured.
                return rendered
            self.refuse_slang_precise(what)
        return self.slang_precise_capture(rendered, mapped, base_name)

    def slang_precise_expression_type(self, value):
        try:
            return self.expression_result_type(value)
        except (TypeError, ValueError, AttributeError):
            return None

    def slang_precise_stored_value(self, target, value, rendered, op):
        """Capture a value the current precise function stores outside its precise locals."""
        if not self.current_slang_precise_function:
            return rendered
        if self.slang_precise_root_name(target) in self.current_slang_precise_locals:
            return rendered
        if op == "=" and not self.slang_precise_needs_capture(value):
            return rendered
        target_type = self.slang_precise_expression_type(target)
        if op != "=" and not self.slang_precise_needs_capture(value):
            # The operator itself rounds: keep its right operand exact as well.
            return self.slang_precise_capture_operand(value, rendered, target_type)
        return self.slang_precise_value_capture(
            value,
            rendered,
            target_type,
            "a floating-point value to a location whose type has no precise Slang local",
        )

    def slang_precise_capture_operand(self, value, rendered, value_type):
        if isinstance(value, LiteralNode):
            return rendered
        try:
            mapped = self.map_type(value_type) if value_type is not None else None
        except (TypeError, ValueError):
            mapped = None
        if not self.slang_precise_value_type(mapped):
            return rendered
        return self.slang_precise_capture(rendered, mapped, "crossglPreciseValue")

    def slang_precise_lowered_store(self, assignment):
        """Refuse a store that a lowering renders without a capture (buffer-block words, matrix columns)."""
        if not self.current_slang_precise_function:
            return
        if (
            self.slang_precise_root_name(assignment.left)
            in self.current_slang_precise_locals
        ):
            return
        if assignment.operator != "=" or self.slang_precise_needs_capture(
            assignment.right
        ):
            self.refuse_slang_precise(
                "a floating-point value through a lowered buffer or matrix-column store"
            )

    def slang_precise_lowered_condition(self, condition):
        """Refuse a component-wise selection whose condition a lowering renders without a capture."""
        if self.current_slang_precise_function and self.slang_precise_needs_capture(
            condition
        ):
            from ..refusals import SlangRefusal

            raise SlangRefusal(
                "A precise function selects on a computed vector condition, so its arithmetic cannot be kept "
                "as written in Slang"
            )

    def slang_precise_returned_value(self, value, rendered, return_type):
        """Capture a returned value: Slang does not keep the ``precise`` of a function result."""
        if not self.current_slang_precise_function or value is None:
            return rendered
        return self.slang_precise_value_capture(
            value,
            rendered,
            return_type,
            "a floating-point result whose type has no precise Slang local",
        )

    def slang_precise_condition(self, condition, rendered):
        """Capture a condition the current precise function branches or selects on."""
        if not self.current_slang_precise_function or condition is None:
            return rendered
        if isinstance(condition, str) or not self.slang_precise_needs_capture(
            condition
        ):
            return rendered
        return self.slang_precise_value_capture(
            condition,
            rendered,
            self.slang_precise_expression_type(condition) or "bool",
            "a condition whose type has no precise Slang local",
            "crossglPreciseCondition",
        )

    def slang_precise_capture_declarations(self, indent):
        """The declarations of the captures the current precise function has used, in first-use order."""
        if not self.slang_precise_captures:
            return ""
        return "".join(
            f"{indent}precise {mapped_type} {name};\n"
            for mapped_type, name in self.slang_precise_captures
        )

    def refuse_slang_precise(self, what):
        from ..refusals import SlangRefusal

        raise SlangRefusal(
            f"A precise function stores {what}, so its arithmetic cannot be kept as written in Slang"
        )
