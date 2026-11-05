"""Narrow-integer arithmetic of non-promoting sources on targets that compute it with integer promotion.

GLSL (``GL_EXT_shader_explicit_arithmetic_types``), HLSL with ``-enable-16bit-types``, SPIR-V and Slang compute an
operation on 8- and 16-bit integer scalars in the operands' type, so ``uint16_t`` ``a * b`` wraps at 16 bits. CrossGL
follows the C rules instead (``semantics/narrow.py``): a narrow scalar operand is promoted to ``int`` first (C++
[conv.prom]), and the Metal, DirectX, CUDA and HIP emitters compute it in 32 bits, which never wraps. For those
targets, an operation of a non-promoting source whose result keeps a narrow scalar type and can leave the operand
range is converted back to that type explicitly:

- ``+``, ``-``, ``*`` and ``<<`` compute on ``uint`` operands, ``T(uint(a) * uint(b))``, so the 32-bit
  intermediate wraps instead of overflowing a signed ``int`` (undefined in C, C++ and MSL), and the conversion to
  ``T`` reduces it modulo 2^N (C++ [conv.integral]);
- signed ``/`` (``-128 / -1``), unary ``-`` and unsigned ``~`` become ``T(e)``;
- ``&``, ``|``, ``^``, ``>>``, ``%``, unsigned ``/`` and signed ``~`` stay in range and keep their narrow type for
  the enclosing operation.

An operation stored straight into a register of its own type (assignment, initializer, return value, conversion)
is left alone: the store reduces it already. Vector operations do not promote in any source (MSL section 2.2,
OpenCL C 6.3). Metal and HLSL compute them natively at their width, except the DirectX 8-bit vectors, which HLSL
has no type for and the DirectX emitter keeps in ``int`` carriers; there, a range-changing 8-bit vector operation
of any source becomes ``T(e)`` too.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import HubType, call_name
from .narrow import PROMOTING_SOURCES, NarrowLegalizer, narrow_from_name, narrow_of
from .rewrite import call, set_child

__all__ = ["PROMOTING_TARGETS", "wrap_narrow_arithmetic"]

# Targets whose emitters compute narrow scalar arithmetic on promoted 32-bit operands.
PROMOTING_TARGETS = frozenset({"metal", "directx", "cuda", "hip"})
# Targets whose emitters keep 8-bit vectors in 32-bit carriers that do not wrap.
_CARRIED_BYTE_VECTOR_TARGETS = frozenset({"directx"})
# Operators computed on ``uint`` operands so the 32-bit intermediate wraps.
_UNSIGNED_INTERMEDIATE = frozenset({"+", "-", "*", "<<"})
_SHIFTS = frozenset({"<<", ">>"})
_GLSL_PREFIX = {"i": "i", "u": "u"}


def type_name(narrow: HubType) -> str:
    """The CrossGL spelling of an 8- or 16-bit integer scalar or vector type."""
    if narrow.lanes == 1:
        return f"{'' if narrow.kind == 'i' else 'u'}int{narrow.bits}_t"
    return f"{_GLSL_PREFIX[narrow.kind]}{narrow.bits}vec{narrow.lanes}"


def _leaves_range(operator, narrow: HubType, unary: bool) -> bool:
    """Whether ``operator`` on in-range operands of ``narrow`` can produce a value outside its range."""
    signed = narrow.kind == "i"
    if unary:
        return operator == "-" or (operator == "~" and not signed)
    if operator in _UNSIGNED_INTERMEDIATE:
        return True
    return operator == "/" and signed


class NarrowArithmeticWrapper(NarrowLegalizer):
    """Make the narrow result type of non-promoting arithmetic explicit; registers keep their narrow types."""

    def __init__(self, program, source, target):
        super().__init__(program, source, target)
        self.carried_byte_vectors = target in _CARRIED_BYTE_VECTOR_TARGETS

    # The base pass widens registers and normalizes every write; here the target has narrow registers.
    def retype(self):
        return None

    def verify(self):
        return None

    def statement(self, node):
        return self.visit(node)

    def _normalize_declaration(self, node, type_field, value_field):
        value = getattr(node, value_field, None)
        if value is not None:
            unwrapped = self._stored(value, narrow_of(getattr(node, type_field, None)))
            if unwrapped is not value:
                set_child(node, value_field, None, unwrapped)

    # Rewrites ---------------------------------------------------------------------------------------------------
    def rewrite(self, node):
        if isinstance(node, (A.BinaryOpNode, A.UnaryOpNode)):
            unary = isinstance(node, A.UnaryOpNode)
            return self.operation(node, self.arithmetic_narrow(node), unary=unary)
        if isinstance(node, A.AssignmentNode) and node.operator == "=":
            self._replace_stored(node, "value", self.narrow_lvalue(node.target))
        elif isinstance(node, A.ReturnNode) and node.value is not None:
            self._replace_stored(node, "value", self.return_type)
        elif isinstance(node, A.FunctionCallNode):
            arguments = list(node.arguments or [])
            narrow = narrow_from_name(call_name(node))
            if narrow is not None and len(arguments) == 1:
                unwrapped = self._stored(arguments[0], narrow)
                if unwrapped is not arguments[0]:
                    set_child(node, "arguments", 0, unwrapped)
        return node

    def _replace_stored(self, node, field, narrow):
        value = getattr(node, field, None)
        unwrapped = self._stored(value, narrow)
        if unwrapped is not value:
            set_child(node, field, None, unwrapped)

    @staticmethod
    def _stored(value, narrow):
        """``value`` without its explicit conversion when a store into ``narrow`` reduces it the same way."""
        if narrow is None or not isinstance(value, A.ASTNode):
            return value
        original = value.annotations.get("cgl_narrow_operation")
        if original is not None and value.annotations.get("cgl_narrow") == narrow:
            return original
        return value

    def operation(self, node, narrow, *, unary):
        if narrow is None:
            return node
        node.annotations["cgl_narrow"] = narrow
        if not _leaves_range(node.op, narrow, unary):
            return node
        if narrow.lanes > 1:
            if not (self.carried_byte_vectors and narrow.bits == 8):
                return node
            inner = node
        elif not unary and node.op in _UNSIGNED_INTERMEDIATE:
            left = call("uint", [node.left])
            right = node.right if node.op in _SHIFTS else call("uint", [node.right])
            inner = A.BinaryOpNode(left, node.op, right)
        else:
            inner = node
        wrapped = call(type_name(narrow), [inner])
        wrapped.source_location = getattr(node, "source_location", None)
        wrapped.annotations["cgl_narrow"] = narrow
        wrapped.annotations["cgl_narrow_operation"] = node
        return wrapped

    # Narrow typing ----------------------------------------------------------------------------------------------
    def narrow_value(self, expr):
        narrow = super().narrow_value(expr)
        if narrow is not None:
            return narrow
        if isinstance(expr, A.FunctionCallNode):
            return narrow_from_name(call_name(expr))
        if isinstance(expr, A.CastNode):
            return narrow_of(expr.target_type)
        if isinstance(expr, A.LiteralNode):
            return narrow_of(expr.literal_type)
        return None


def wrap_narrow_arithmetic(program, source, target) -> bool:
    """Make narrow arithmetic results explicit in ``program`` for ``target``; return whether the pass ran."""
    if target not in PROMOTING_TARGETS or not isinstance(program, A.ShaderNode):
        return False
    source = str(source or "")
    if source in PROMOTING_SOURCES and target not in _CARRIED_BYTE_VECTOR_TARGETS:
        return False
    NarrowArithmeticWrapper(program, source, target).run()
    return True
