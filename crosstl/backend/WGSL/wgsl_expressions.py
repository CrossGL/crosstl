"""Typing and CrossGL emission of WGSL expressions for the WGSL importer (W3C WGSL, sections 8 and 6.1).

Each expression is typed as WGSL types it (:meth:`ExpressionMixin.type_of`): literals without a suffix are abstract,
and an abstract operand takes the concrete type of the other operand, the parameter it is passed to or the
variable it initializes (section 6.1.5). :meth:`ExpressionMixin.emit` writes the expression as CrossGL, given the
concrete type the context wants; an abstract expression is folded and written in that type, so ``x * 2`` with an
``f32`` ``x`` writes ``x * 2.0``.

Pointers are written as the memory they point to. ``&e`` is ``e``, ``*p`` is what ``p`` points to, and a ``let``
bound to a pointer (:mod:`.WgslCrossGLCodeGen`) is a name for its memory, so every use of the pointer reads and
writes that memory as WGSL does.

Operations whose WGSL result is defined where CrossGL leaves it to the target are written so that every target
computes the WGSL value: a shift by a run-time amount masks the amount to the bit width (section 8.9), and integer
division and remainder by a run-time divisor go through helper functions that return WGSL's results for a zero
divisor and for the most negative ``i32`` divided by -1 (section 8.7).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import WgslAst as A
from .wgsl_module_context import literal_type, literal_value
from .wgsl_types import (
    ABSTRACT_FLOAT,
    ABSTRACT_INT,
    BOOL,
    I32,
    U32,
    Array,
    Matrix,
    Pointer,
    ResultStruct,
    Scalar,
    StructType,
    Vector,
    WgslImportError,
    concretize,
    crossgl_type,
    crossgl_type_expression,
    element_scalar,
    is_abstract,
    with_scalar,
)

__all__ = ["ExpressionMixin", "Symbol", "plain_expression_text"]

_SWIZZLE = frozenset("xyzw"), frozenset("rgba")
_SWIZZLE_INDEX = {c: i for s in ("xyzw", "rgba") for i, c in enumerate(s)}
_COMPARISONS = ("==", "!=", "<", ">", "<=", ">=")
_INT_MIN = -(2**31)


@dataclass
class Symbol:
    """A name in scope: ``kind`` is ``var``, ``let``, ``const``, ``param``, ``global``, ``override`` or ``alias``.

    ``type`` is the store type of a variable and the value type of a value; ``text`` is the CrossGL expression
    that reads it, the name itself unless the name is a pointer, whose text is the memory it points to. ``stable``
    says that the value cannot change while the name is in scope.
    """

    kind: str
    type: object
    text: str
    space: str | None = None
    stable: bool = False


def plain_expression_text(node) -> str:
    """A constant or override expression as CrossGL, without typing: integer literals and names only."""
    if isinstance(node, A.Literal):
        return str(literal_value(node)) if node.kind == "int" else node.text
    if isinstance(node, A.Ident):
        return node.name
    if isinstance(node, A.Unary):
        return f"({node.op}{plain_expression_text(node.operand)})"
    if isinstance(node, A.Binary):
        return f"({plain_expression_text(node.left)} {node.op} {plain_expression_text(node.right)})"
    raise WgslImportError(
        "an array count must be a constant or override expression",
        getattr(node, "line", None),
    )


def format_float(value: float, line=None) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        raise WgslImportError("a floating-point constant is not finite", line)
    text = repr(float(value))
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def scalar_literal(value, scalar: Scalar, line=None) -> str:
    """``value`` written as a CrossGL literal of ``scalar``."""
    name = concretize(scalar).name
    if name == "bool":
        return "true" if value else "false"
    if name in ("f32", "f16"):
        text = format_float(float(value), line)
        return f"half({text})" if name == "f16" else text
    if isinstance(value, float):
        if value != int(value):
            raise WgslImportError(f"{value} is not an integer", line)
        value = int(value)
    if name == "u32":
        if not 0 <= value <= 0xFFFFFFFF:
            raise WgslImportError(f"{value} does not fit in u32", line)
        return f"{value}u"
    if not _INT_MIN <= value <= 2**31 - 1:
        raise WgslImportError(f"{value} does not fit in i32", line)
    return "(-2147483647 - 1)" if value == _INT_MIN else str(value)


def _shape(type_):
    """``type_`` without its scalar, for comparing shapes."""
    return with_scalar(type_, None) if element_scalar(type_) is not None else type_


def unify(left, right):
    """The scalar both operands convert to (section 6.1.5), or ``None`` when neither is concrete."""
    a, b = element_scalar(left), element_scalar(right)
    if a is None or b is None:
        return a or b
    if not a.name.startswith("abstract"):
        return a
    if not b.name.startswith("abstract"):
        return b
    return ABSTRACT_FLOAT if ABSTRACT_FLOAT in (a, b) else ABSTRACT_INT


class ExpressionMixin:
    """Typing and emission of expressions; the converter provides ``ctx``, ``lookup``, ``struct_names``,
    ``member_name`` and ``require_helper``."""

    # Typing.

    def type_of(self, node):
        method = getattr(self, f"_type_{type(node).__name__}", None)
        if method is None:
            raise WgslImportError(
                f"unsupported expression {type(node).__name__}",
                getattr(node, "line", None),
            )
        return method(node)

    def _type_Literal(self, node):
        return literal_type(node)

    def _type_Ident(self, node):
        symbol = self.lookup(node.name)
        if symbol is None:
            raise WgslImportError(f"'{node.name}' is not declared", node.line)
        if symbol.kind == "alias":
            return symbol.type
        return symbol.type

    def _type_Unary(self, node):
        operand = self.type_of(node.operand)
        if node.op == "&":
            space = self.address_space(node.operand)
            return Pointer(space, self.deref(operand))
        if node.op == "*":
            if not isinstance(operand, Pointer):
                raise WgslImportError("'*' needs a pointer", node.line)
            return operand.elem
        return self.deref(operand)

    def _type_Binary(self, node):
        left, right = self.deref(self.type_of(node.left)), self.deref(
            self.type_of(node.right)
        )
        op = node.op
        if op in ("&&", "||"):
            return BOOL
        if op in ("<<", ">>"):
            # An abstract value shifted by a run-time amount is not a constant expression, so it is concrete.
            return (
                concretize(left)
                if is_abstract(left) and self.ctx.eval_const(node) is None
                else left
            )
        if op in _COMPARISONS:
            shape = left if not isinstance(left, Scalar) else right
            return Vector(shape.size, BOOL) if isinstance(shape, Vector) else BOOL
        scalar = unify(left, right)
        if op == "*" and isinstance(left, Matrix) and isinstance(right, Matrix):
            return Matrix(right.cols, left.rows, scalar)
        if op == "*" and isinstance(left, Matrix) and isinstance(right, Vector):
            return Vector(left.rows, scalar)
        if op == "*" and isinstance(left, Vector) and isinstance(right, Matrix):
            return Vector(right.cols, scalar)
        shape = right if isinstance(left, Scalar) else left
        return with_scalar(shape, scalar)

    def _type_Member(self, node):
        base = self.deref(self.type_of(node.base))
        if isinstance(base, Vector):
            if len(node.name) == 1:
                return base.elem
            return Vector(len(node.name), base.elem)
        if isinstance(base, StructType):
            for name, type_, _attributes, _line in self.ctx.struct_fields(base.name):
                if name == node.name:
                    return type_
        if isinstance(base, ResultStruct) and base.field(node.name) is not None:
            return base.field(node.name)
        raise WgslImportError(f"no member '{node.name}'", node.line)

    def _type_Index(self, node):
        base = self.deref(self.type_of(node.base))
        if isinstance(base, Array):
            return base.elem
        if isinstance(base, Vector):
            return base.elem
        if isinstance(base, Matrix):
            return base.column
        raise WgslImportError(
            "only arrays, vectors and matrices can be indexed", node.line
        )

    def _type_Call(self, node):
        from .wgsl_builtin_calls import call_type

        return call_type(self, node)

    @staticmethod
    def deref(type_):
        """The store type of a pointer's memory, the value of an atomic, and any other type itself."""
        if isinstance(type_, Pointer):
            type_ = type_.elem
        return type_

    def address_space(self, node):
        """The address space of the memory a reference expression names."""
        while isinstance(node, (A.Member, A.Index)):
            node = node.base
        if isinstance(node, A.Unary) and node.op == "*":
            pointer = self.type_of(node.operand)
            return pointer.space if isinstance(pointer, Pointer) else "function"
        if isinstance(node, A.Ident):
            symbol = self.lookup(node.name)
            if symbol is not None and isinstance(symbol.type, Pointer):
                return symbol.type.space
            if symbol is not None and symbol.space:
                return symbol.space
        return "function"

    # Emission.

    def emit(self, node, want=None) -> str:
        """``node`` as a CrossGL expression; ``want`` is the type an abstract ``node`` is converted to."""
        type_ = self.type_of(node)
        if is_abstract(type_):
            target = (
                want
                if want is not None and not is_abstract(want)
                else concretize(type_)
            )
            value = self.ctx.eval_const(node)
            if value is not None and isinstance(target, Scalar):
                return scalar_literal(value, target, getattr(node, "line", None))
            want = target
        method = getattr(self, f"_emit_{type(node).__name__}")
        return method(node, want)

    def _emit_Literal(self, node, want):
        target = want if isinstance(want, Scalar) else concretize(literal_type(node))
        return scalar_literal(literal_value(node), target, node.line)

    def _emit_Ident(self, node, want):
        symbol = self.lookup(node.name)
        if symbol is None:
            raise WgslImportError(f"'{node.name}' is not declared", node.line)
        return symbol.text

    def _emit_Unary(self, node, want):
        if node.op in ("&", "*"):
            return self.emit(node.operand, want)
        operand = self.emit(node.operand, self._operand_want(node.operand, want))
        operand_type = self.deref(self.type_of(node.operand))
        if node.op == "-" and element_scalar(operand_type) == U32:
            # WGSL negates a u32 modulo 2^32 (section 8.7); CrossGL's unary minus is for signed values.
            return (
                f"(0u - {operand})"
                if isinstance(operand_type, Scalar)
                else (f"({crossgl_type(operand_type)}(0u) - {operand})")
            )
        return f"({node.op}{operand})"

    def _operand_want(self, node, want):
        """The type an operand of an expression of type ``want`` converts to."""
        type_ = self.deref(self.type_of(node))
        if want is None or not is_abstract(type_):
            return None
        scalar = element_scalar(want)
        return with_scalar(type_, scalar) if scalar is not None else None

    def _binary_wants(self, node, want):
        left, right = self.deref(self.type_of(node.left)), self.deref(
            self.type_of(node.right)
        )
        if node.op in ("<<", ">>"):
            left_want = self._operand_want(node.left, want)
            return left_want, with_scalar(right, U32) if is_abstract(right) else None
        scalar = unify(left, right)
        if scalar is not None and scalar.name.startswith("abstract"):
            if node.op in _COMPARISONS or node.op in ("&&", "||"):
                scalar = concretize(scalar)
            else:
                scalar = (
                    element_scalar(want) if want is not None else concretize(scalar)
                )
        left_want = with_scalar(left, scalar) if is_abstract(left) and scalar else None
        right_want = (
            with_scalar(right, scalar) if is_abstract(right) and scalar else None
        )
        return left_want, right_want

    def _emit_Binary(self, node, want):
        left_want, right_want = self._binary_wants(node, want)
        left = self.emit(node.left, left_want)
        right = self.emit(node.right, right_want)
        left_type = concretize(
            self.deref(self.type_of(node.left)) if left_want is None else left_want
        )
        right_type = concretize(
            self.deref(self.type_of(node.right)) if right_want is None else right_want
        )
        op = node.op
        if op in ("<<", ">>"):
            right = self._masked_shift_amount(node.right, right, left_type, right_type)
        elif op in ("/", "%") and element_scalar(left_type) in (I32, U32):
            return self._integer_division(node, left, right, left_type, right_type)
        return f"({left} {op} {right})"

    def _masked_shift_amount(self, amount, text, value_type, amount_type):
        value = self.ctx.eval_const(amount)
        if isinstance(value, int) and 0 <= value < 32:
            return text
        if isinstance(amount_type, Vector):
            return f"({text} & {crossgl_type(amount_type)}(31u))"
        return f"({text} & 31u)"

    def _integer_division(self, node, left, right, left_type, right_type):
        divisor = self.ctx.eval_const(node.right)
        scalar = element_scalar(left_type)
        safe = (
            isinstance(divisor, int)
            and divisor != 0
            and (scalar == U32 or divisor != -1)
        )
        if safe and isinstance(right_type, Scalar):
            return f"({left} {node.op} {right})"
        result_type = left_type if not isinstance(left_type, Scalar) else right_type
        if isinstance(result_type, Scalar):
            result_type = left_type
        helper = self.require_helper("div" if node.op == "/" else "rem", result_type)
        if isinstance(result_type, Vector):
            left = (
                left
                if isinstance(left_type, Vector)
                else f"{crossgl_type(result_type)}({left})"
            )
            right = (
                right
                if isinstance(right_type, Vector)
                else f"{crossgl_type(result_type)}({right})"
            )
        return f"{helper}({left}, {right})"

    def _emit_Member(self, node, want):
        base_type = self.deref(self.type_of(node.base))
        base = self.emit(node.base)
        if isinstance(base_type, StructType):
            return f"{base}.{self.member_name(base_type.name, node.name)}"
        if isinstance(base_type, ResultStruct):
            return self.result_member(node, base_type)
        return f"{base}.{node.name}"

    def _emit_Index(self, node, want):
        base = self.emit(node.base)
        index_type = concretize(self.type_of(node.index))
        index = self.emit(node.index, index_type)
        return f"{base}[{index}]"

    def _emit_Call(self, node, want):
        from .wgsl_builtin_calls import emit_call

        return emit_call(self, node, want)

    # Helpers for callers.

    def emit_value(self, node, type_) -> str:
        """``node`` converted to the declared type ``type_`` (an initializer, argument or return value)."""
        return self.emit(node, type_ if is_abstract(self.type_of(node)) else None)

    def type_text(self, type_, line=None) -> str:
        return crossgl_type(type_, line, self.struct_names)

    def type_expression(self, type_, line=None) -> str:
        return crossgl_type_expression(type_, line, self.struct_names)

    def is_stable(self, node) -> bool:
        """Whether evaluating ``node`` again later gives the same value: a constant, a value name, or an
        operation on those."""
        if isinstance(node, A.Literal):
            return True
        if isinstance(node, A.Ident):
            symbol = self.lookup(node.name)
            return symbol is not None and symbol.stable
        if isinstance(node, A.Unary) and node.op not in ("*", "&"):
            return self.is_stable(node.operand)
        if isinstance(node, A.Binary):
            return self.is_stable(node.left) and self.is_stable(node.right)
        return False

    def has_call(self, node) -> bool:
        """Whether ``node`` calls a function of the module, which may have side effects."""
        if isinstance(node, A.Call) and node.callee.name in self.ctx.functions:
            return True
        if isinstance(node, list):
            return any(self.has_call(item) for item in node)
        if hasattr(node, "__dataclass_fields__"):
            return any(
                self.has_call(getattr(node, name)) for name in node.__dataclass_fields__
            )
        return False


def swizzle_components(name: str):
    if all(c in "xyzw" for c in name) or all(c in "rgba" for c in name):
        return [_SWIZZLE_INDEX[c] for c in name]
    return None
