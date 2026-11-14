"""HLSL emission of CrossGL ``precise`` functions.

A CrossGL function marked ``precise`` evaluates all of its floating-point arithmetic as written. HLSL attaches
``precise`` to variables and function results only, and DXC clears the ``fast`` flags of exactly the operations
whose results reach a precise value. A store of ``a * b - c`` to a buffer keeps ``fmul fast`` and ``fsub fast``
even when ``a``, ``b`` and ``c`` are precise locals, and so does a comparison of that value that only feeds a
branch (checked with DXC 1.9.2602).

These members therefore declare every local of a precise function ``precise`` and capture each value the function
stores elsewhere, and each condition it branches on, in a precise local declared at function entry:
``o[i] = (crossglPreciseValue = a * b - c)``. The capture is evaluated where the value was, so the order of
evaluation and short-circuiting are kept. Results already are precise (``current_hlsl_precise_return_type``).
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

# Scalar, vector and matrix types whose values a precise local can hold.
_PRECISE_VALUE_TYPE = re.compile(
    r"^(?:bool|int|uint|dword|half|float|double|min16float|min10float|min16int|min16uint|"
    r"float16_t|float32_t|float64_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t)"
    r"(?:[1-4](?:x[1-4])?)?$"
)


class HLSLPreciseFunctionMixin:
    """``precise`` function members of ``HLSLCodeGen``."""

    current_hlsl_precise_function = False
    hlsl_precise_captures = None
    current_hlsl_precise_locals = frozenset()

    # Locals with these qualifiers are shared or static storage, not values of one invocation of the function.
    _HLSL_NON_VALUE_LOCAL_QUALIFIERS = frozenset(
        {"groupshared", "shared", "threadgroup", "workgroup", "buffer", "static"}
    )

    def enter_hlsl_precise_function(self, func, body):
        """Start emitting ``func``'s body; return the state :meth:`leave_hlsl_precise_function` restores."""
        previous = (
            self.current_hlsl_precise_function,
            self.hlsl_precise_captures,
            self.current_hlsl_precise_locals,
        )
        precise = self.hlsl_has_precise_modifier(func)
        self.current_hlsl_precise_function = precise
        self.hlsl_precise_captures = [] if precise else None
        self.current_hlsl_precise_locals = (
            frozenset(self.hlsl_precise_local_names(body))
            if precise and body is not None
            else frozenset()
        )
        return previous

    def leave_hlsl_precise_function(self, previous):
        (
            self.current_hlsl_precise_function,
            self.hlsl_precise_captures,
            self.current_hlsl_precise_locals,
        ) = previous

    def hlsl_precise_value_type(self, mapped_type):
        return isinstance(mapped_type, str) and bool(
            _PRECISE_VALUE_TYPE.match(mapped_type.strip())
        )

    def hlsl_precise_function_local(self, node):
        """Return whether a local declared by the current precise function becomes precise."""
        if not self.current_hlsl_precise_function:
            return False
        qualifiers = {str(value).lower() for value in getattr(node, "qualifiers", [])}
        if qualifiers & self._HLSL_NON_VALUE_LOCAL_QUALIFIERS:
            return False
        declared = getattr(node, "var_type", None) or getattr(node, "vtype", None)
        try:
            mapped = self.map_type(declared)
        except (TypeError, ValueError):
            return False
        return self.hlsl_precise_value_type(mapped)

    def hlsl_precise_local_names(self, body):
        """Names of the locals of ``body`` that a precise function declares precise."""
        names = set()
        for node in self.walk_ast(body):
            if isinstance(node, VariableNode) and self.hlsl_precise_function_local(
                node
            ):
                names.add(node.name)
        return names

    def hlsl_expression_has_float_arithmetic(self, expression):
        """Return whether evaluating ``expression`` may round a floating-point operation.

        Operations of unknown type count as floating point, so a value is captured rather than left to
        contraction.
        """
        for node in self.walk_ast(expression):
            if isinstance(node, (BinaryOpNode, UnaryOpNode, TernaryOpNode)):
                return True
            if isinstance(node, FunctionCallNode):
                return True
        return False

    def hlsl_precise_capture(self, rendered, mapped_type, base_name):
        used = (
            set(self.current_identifier_reserved_names)
            | set(self.local_variable_types)
            | set(self.global_variable_types)
        )
        name = self.hlsl_unique_local_identifier(base_name, used)
        self.current_identifier_reserved_names.add(name)
        self.local_variable_types[name] = mapped_type
        self.hlsl_precise_captures.append((f"precise {mapped_type}", name))
        return f"({name} = {rendered})"

    def hlsl_precise_target_is_local(self, target):
        while isinstance(target, (ArrayAccessNode, MemberAccessNode)):
            target = (
                target.array
                if isinstance(target, ArrayAccessNode)
                else getattr(target, "object", getattr(target, "object_expr", None))
            )
        name = getattr(target, "name", target if isinstance(target, str) else None)
        return name in self.current_hlsl_precise_locals

    def hlsl_precise_stored_value(self, target, value, rendered, target_type, op):
        """Capture a value the current precise function stores to non-local storage."""
        if not self.current_hlsl_precise_function or self.hlsl_precise_captures is None:
            return rendered
        if self.hlsl_precise_target_is_local(target):
            return rendered
        if not self.hlsl_expression_has_float_arithmetic(value) and op == "=":
            return rendered
        try:
            mapped = self.map_type(target_type) if target_type is not None else None
        except (TypeError, ValueError):
            mapped = None
        if not self.hlsl_precise_value_type(mapped):
            mapped = self.hlsl_precise_expression_type(value)
        if mapped is None:
            self.raise_hlsl_precise_refusal(
                "a value of a type that has no precise HLSL local"
            )
        return self.hlsl_precise_capture(rendered, mapped, "crossglPreciseValue")

    def hlsl_precise_buffer_store_value(self, value, rendered_value):
        """Capture a value the current precise function writes through a buffer intrinsic.

        Returns the rendered capture, or ``rendered_value`` when no capture is needed. The capture holds the
        value in its own type; the intrinsic converts it to the element type as it did the uncaptured value.
        """
        if not self.current_hlsl_precise_function or self.hlsl_precise_captures is None:
            return rendered_value
        if rendered_value is not None or not self.hlsl_expression_has_float_arithmetic(
            value
        ):
            return rendered_value
        mapped = self.hlsl_precise_expression_type(value)
        if mapped is None:
            self.raise_hlsl_precise_refusal(
                "a buffer value of a type that has no precise HLSL local"
            )
        rendered = self.generate_expression_with_expected(
            value, self.expression_result_type(value)
        )
        return self.hlsl_precise_capture(rendered, mapped, "crossglPreciseValue")

    def hlsl_precise_expression_type(self, value):
        try:
            mapped = self.map_type(self.expression_result_type(value))
        except (TypeError, ValueError):
            return None
        return mapped if self.hlsl_precise_value_type(mapped) else None

    def hlsl_precise_condition(self, condition, rendered):
        """Capture a branch condition of the current precise function."""
        if not self.current_hlsl_precise_function or self.hlsl_precise_captures is None:
            return rendered
        if condition is None or isinstance(condition, str):
            return rendered
        if not self.hlsl_expression_has_float_arithmetic(condition):
            return rendered
        return self.hlsl_precise_capture(rendered, "bool", "crossglPreciseCondition")

    def raise_hlsl_precise_refusal(self, what):
        from ..refusals import DirectXRefusal

        raise DirectXRefusal(
            f"A precise function stores {what}, so its arithmetic cannot be kept as written in HLSL"
        )

    def hlsl_identifier_name(self, node):
        return node.name if isinstance(node, IdentifierNode) else None
