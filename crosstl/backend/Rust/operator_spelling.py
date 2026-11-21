"""Rust operators whose CrossGL spelling depends on the operand types.

- Rust spells bitwise complement and logical negation with one operator, ``!``: on ``bool`` and ``BVec`` it
  negates, on the integer primitives and glam's integer vectors it flips every bit. CrossGL keeps C's two
  operators, so ``!`` on an integer operand is written as ``~``.
- glam's ``==`` and ``!=`` on two vectors compare the whole vector (``PartialEq``) and give one ``bool``. CrossGL
  compares vectors component by component, so the importer writes ``all(a == b)`` and ``any(a != b)``.
- glam's ``v.is_nan()`` is true when any component is NaN and ``v.is_finite()`` when every component is finite;
  CrossGL's ``isnan`` and ``isfinite`` are componentwise, so they are reduced with ``any`` and ``all``. The
  ``_mask`` forms are the componentwise results themselves.
"""

from __future__ import annotations

import re

_INTEGER_TYPE_RE = re.compile(
    r"^(?:u?int(?:8|16|32|64)?(?:_t)?|[iu](?:8|16|32|64|size)|[iu]vec[234]|u?int(?:8|16|64)_t[234]?)$"
)
_VECTOR_TYPE_RE = re.compile(r"^(?:[dhiub]|i64|u64)?vec[234]$")
_VECTOR_REDUCTIONS = {"==": "all", "!=": "any"}
_FLOAT_VECTOR_TYPE_RE = re.compile(r"^(?:[dh]?vec[234]|D?Vec(?:[234]|3A))$")
# glam's float-vector predicates: (the reduction of the componentwise result, the CrossGL function).
_VECTOR_PREDICATES = {
    "is_nan": ("any", "isnan"),
    "is_finite": ("all", "isfinite"),
    "is_nan_mask": (None, "isnan"),
    "is_finite_mask": (None, "isfinite"),
}


class RustOperatorSpellingMixin:
    """Choose the CrossGL spelling of a Rust unary or binary operation from its operand types."""

    def crossgl_unary_operator(self, expression):
        op = getattr(expression, "op", None)
        if op != "!":
            return op
        operand_type = self.infer_value_type(getattr(expression, "operand", None))
        if self.is_integer_value_type(operand_type):
            return "~"
        return op

    def format_crossgl_binary(self, expression, left, right):
        """``(left op right)``, reduced with ``all``/``any`` when the operation compares two whole vectors."""
        op = getattr(expression, "op", None)
        code = f"({left} {op} {right})"
        reduction = _VECTOR_REDUCTIONS.get(op)
        if reduction is None:
            return code
        if self.is_vector_value_type(
            self.infer_value_type(expression.left)
        ) and self.is_vector_value_type(self.infer_value_type(expression.right)):
            return f"{reduction}{code}"
        return code

    def format_vector_predicate_call(self, method_name, obj, args, receiver_type):
        """``any(isnan(v))``, ``all(isfinite(v))`` or a ``_mask`` form for a glam float-vector predicate."""
        predicate = _VECTOR_PREDICATES.get(method_name)
        if predicate is None or args:
            return None
        if not self._value_type_matches(receiver_type, _FLOAT_VECTOR_TYPE_RE):
            return None
        reduction, function = predicate
        call = f"{function}({obj})"
        return f"{reduction}({call})" if reduction else call

    def is_integer_value_type(self, type_name):
        return self._value_type_matches(type_name, _INTEGER_TYPE_RE)

    def is_vector_value_type(self, type_name):
        return self._value_type_matches(type_name, _VECTOR_TYPE_RE)

    def _value_type_matches(self, type_name, pattern):
        if not isinstance(type_name, str) or not type_name:
            return False
        type_name = self.normalize_receiver_type(type_name)
        for candidate in (type_name, self.map_type(type_name)):
            if isinstance(candidate, str) and pattern.match(candidate.strip()):
                return True
        return False
