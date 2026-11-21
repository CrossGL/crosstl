"""Mojo spellings of CrossGL pointer dereference (``*p``) and pointer member access (``p->m``).

Mojo has no prefix dereference operator, since a prefix ``*`` unpacks arguments, and no ``->``. An
``UnsafePointer`` is dereferenced with an empty subscript, so ``*p`` is written ``p[]`` and ``p->m`` is written
``p[].m``, which is ``(*p).m`` in C.
"""

from ..ast import MemberAccessNode, PointerAccessNode, UnaryOpNode
from ..refusals import MojoRefusal
from ..unhandled import UNHANDLED
from .c_family_operands import postfix_operand

__all__ = ["MojoPointerExpressionMixin"]


def _dereference(pointer_expr):
    """The CrossGL node ``*pointer_expr``."""
    return UnaryOpNode("*", pointer_expr)


def _pointee_member(expr):
    """The CrossGL node ``(*p).m`` for the pointer member access ``p->m``."""
    return MemberAccessNode(_dereference(expr.pointer_expr), expr.member)


class MojoPointerExpressionMixin:
    """Dereference and pointer member access members of ``MojoCodeGen.generate_expression``."""

    def generate_mojo_dereference_expression(self, expr):
        """``*p`` as ``p[]``, refusing an operand whose known type is not a pointer."""
        operand_type = self.expression_result_type(expr.operand)
        if operand_type is not None and self.pointer_element_type(operand_type) is None:
            raise MojoRefusal(
                f"Mojo dereference: operand of type {operand_type} is not a pointer, "
                "and Mojo dereferences only a pointer, with p[]"
            )
        return f"{postfix_operand(self.generate_expression(expr.operand))}[]"

    def generate_mojo_pointer_access_expression(self, expr):
        """``p->m`` as the member access ``p[].m`` on the dereferenced pointer."""
        return self.generate_mojo_member_access_expression(_pointee_member(expr))

    def mojo_pointer_expression_result_type(self, expr):
        """The pointee type of ``*p`` or the member type of ``p->m``, or ``UNHANDLED`` for any other node."""
        if isinstance(expr, PointerAccessNode):
            return self.expression_result_type(_pointee_member(expr))
        if isinstance(expr, UnaryOpNode) and self.map_operator(expr.op) == "*":
            return self.pointer_element_type(self.expression_result_type(expr.operand))
        return UNHANDLED
