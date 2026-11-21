"""Integer promotion of 8- and 16-bit integer scalars in Slang output.

CrossGL, MSL, CUDA, HIP and OpenCL C convert an 8- or 16-bit integer scalar operand to ``int`` before an
arithmetic, shift, bitwise or relational operator applies (C17 6.3.1.1, C++ [conv.prom]), so ``int8_t v = 12``
gives ``v * v == 144``. Slang computes those operators at the operands' width (``int8_t * int8_t`` is an
``int8_t``, OpIMul on 8-bit integers in SPIR-V), so the same expression wraps to -112. When the translation's source
promotes (``semantics/narrow.py`` PROMOTING_SOURCES, recorded by rule carrying), the Slang emitter therefore:

- writes a narrow scalar operand of such an operator as ``int(operand)`` and types the result as ``int`` (or the
  wider operand's type), as C does. Vector operations do not promote in any source (MSL section 2.2, OpenCL C
  6.3) and are left alone, as is an operator whose other operand has a type the emitter does not know;
- converts a promoted result stored into a narrow scalar (initializer, assignment, return value, argument) with an
  explicit ``T(e)``, which is the C conversion of the store (C++ [conv.integral]) and spares slangc's implicit
  conversion warning;
- writes ``a /= b``, ``a %= b``, ``a <<= b`` and ``a >>= b`` on a narrow scalar ``a`` as ``a = T(int(a) op b)``:
  Slang would convert ``b`` to ``a``'s type and divide or shift at 8 or 16 bits, which differs from the promoted
  result even after the store reduces it. The other compound assignments are exact modulo 2^N and stay as they are.
  A target of one of those four that cannot be evaluated twice is refused.
"""

from __future__ import annotations

from ..ast import AssignmentNode, BinaryOpNode, UnaryOpNode
from ..semantics.narrow import PROMOTING_SOURCES
from ..semantics.rewrite import clone, is_pure

__all__ = ["SlangIntegerPromotionMixin", "source_promotes_narrow_scalars"]

SLANG_NARROW_SCALARS = frozenset({"int8_t", "uint8_t", "int16_t", "uint16_t"})
_RELATIONAL = frozenset({"<", "<=", ">", ">=", "==", "!="})
# Binary operators whose narrow scalar operands C promotes (C17 sections 6.5.5 to 6.5.12).
PROMOTING_BINARY = frozenset(
    {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"} | _RELATIONAL
)
PROMOTING_UNARY = frozenset({"-", "~", "+"})
# Compound assignments whose narrow-width result is not the promoted result reduced modulo 2^N.
_REWRITTEN_COMPOUND = frozenset({"/=", "%=", "<<=", ">>="})


def source_promotes_narrow_scalars(program) -> bool:
    """Whether the source of ``program`` promotes narrow scalar operands, from its rule-carrying annotation."""
    rules = (getattr(program, "annotations", None) or {}).get("rule_carrying") or {}
    return rules.get("source") in PROMOTING_SOURCES


class SlangIntegerPromotionMixin:
    """C integer promotion of narrow scalar operands for Slang output of promoting sources."""

    slang_narrow_scalars_promote = False

    def slang_narrow_scalar_type(self, type_name):
        """The Slang spelling of ``type_name`` when it is an 8- or 16-bit integer scalar, else ``None``."""
        name = self.type_name_string(type_name) if type_name is not None else None
        if not name:
            return None
        try:
            mapped = self.convert_type(name)
        except (TypeError, ValueError):
            return None
        return mapped if mapped in SLANG_NARROW_SCALARS else None

    def _slang_known_scalar(self, type_name):
        if self.slang_narrow_scalar_type(type_name) is not None:
            return True
        return bool(type_name) and self.is_scalar_value_type(type_name)

    def slang_promoted_operands(self, node):
        """``(left, right)`` flags of the operands of ``node`` written as ``int(...)``, or ``None``."""
        if not self.slang_narrow_scalars_promote or node.op not in PROMOTING_BINARY:
            return None
        left_type = self.expression_result_type(node.left)
        right_type = self.expression_result_type(node.right)
        flags = (
            self.slang_narrow_scalar_type(left_type) is not None,
            self.slang_narrow_scalar_type(right_type) is not None,
        )
        if not any(flags):
            return None
        if not (
            self._slang_known_scalar(left_type) and self._slang_known_scalar(right_type)
        ):
            return None
        return flags

    def slang_promotes_unary(self, node):
        return (
            self.slang_narrow_scalars_promote
            and node.op in PROMOTING_UNARY
            and not getattr(node, "is_postfix", False)
            and self.slang_narrow_scalar_type(self.expression_result_type(node.operand))
            is not None
        )

    def slang_promoted_operand_types(self, expr, left_type, right_type):
        """The operand types C computes ``expr`` in: a promoted narrow scalar is ``int``."""
        flags = self.slang_promoted_operands(expr)
        if flags is None:
            return left_type, right_type
        return ("int" if flags[0] else left_type), ("int" if flags[1] else right_type)

    def slang_promoted_store(self, expr, value, expected_type):
        """``value`` converted to the narrow scalar ``expected_type`` when it is a promoted ``int`` result."""
        if not self.slang_narrow_scalars_promote:
            return value
        narrow = self.slang_narrow_scalar_type(expected_type)
        if narrow is None:
            return value
        if isinstance(expr, BinaryOpNode):
            if expr.op not in PROMOTING_BINARY or expr.op in _RELATIONAL:
                return value
        elif not (
            isinstance(expr, UnaryOpNode)
            and expr.op in PROMOTING_UNARY
            and not getattr(expr, "is_postfix", False)
        ):
            return value
        result = self.expression_result_type(expr)
        if result is None or self.convert_type(self.type_name_string(result)) not in {
            "int",
            "uint",
        }:
            return value
        return f"{narrow}({value})"

    def slang_promoted_compound_assignment(self, node):
        """``a op= b`` on a narrow scalar as ``a = a op b`` when Slang's narrow-width result would differ."""
        if (
            not self.slang_narrow_scalars_promote
            or not isinstance(node, AssignmentNode)
            or node.operator not in _REWRITTEN_COMPOUND
            or self.slang_narrow_scalar_type(self.expression_result_type(node.left))
            is None
        ):
            return node
        if not is_pure(node.left):
            self.refuse_lowering(
                f"Slang compound assignment {node.operator} on an 8- or 16-bit integer: its target must be "
                "evaluated twice to compute the promoted result, and it has side effects"
            )
            return node
        operation = BinaryOpNode(clone(node.left), node.operator[:-1], node.right)
        operation.source_location = getattr(node, "source_location", None)
        rewritten = AssignmentNode(node.left, operation, "=")
        rewritten.source_location = getattr(node, "source_location", None)
        return rewritten
