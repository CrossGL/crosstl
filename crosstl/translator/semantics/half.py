"""Binary16 legalization: ``half`` values on targets whose emitters compute in binary32 (defect S7).

The SPIR-V emitter has no 16-bit floating-point type, and WGSL's ``f16`` needs the optional ``shader-f16`` feature
(WGSL section 4.1.1, ``enable f16``), so both emitters compute ``half`` in binary32 and never round to binary16
(``half h = 0.1h; h = h * 3.0h`` keeps binary32 precision). This pass, run before rule carrying for those targets,
gives every ``half`` register a ``float`` and keeps the invariant that it holds a binary16 value:

- a value written to a half register (initializer, assignment, compound assignment, increment, argument of a half
  parameter, return from a half function, store to a half buffer) and every conversion to ``half`` is rounded with
  ``cgl_half_round`` (ties to even), unless it is a binary16 value already. A ``double`` would be rounded twice and is
  refused;
- reads need nothing: binary32 represents every binary16 value;
- arithmetic whose result is ``half`` is rounded once per operation, as MSL section 2.1, HLSL with 16-bit types and
  ``GL_EXT_shader_explicit_arithmetic_types_float16`` compute it: ``+``, ``-`` and ``*`` round the binary32 result,
  ``/``, ``sqrt`` and ``fma`` call the exact helpers of :mod:`.half_helpers`, exact built-ins (``min``, ``abs``,
  ``floor`` and the like) need nothing, and the other math built-ins round their binary32 result;
- an operation of a ``half`` and a ``float`` is a ``float`` operation in every source. An integer operand converts to
  ``half``. A floating-point literal adopts ``half`` in HLSL and Slang (literal float) and is a ``float`` elsewhere;
  importers spell half literals (``1.5h``, ``1.5hf``) as conversions.

Registers, private and workgroup arrays and struct members become ``float``. Half pointees in device, constant or
unqualified memory, and buffer element types, are left to the emitters, which lay that storage out one 32-bit word
per element (a layout change for the host that the reflection metadata does not report yet). Constructs the pass
cannot keep exact are refused with :class:`HalfFloatUnsupportedError`: half matrices, ``%`` and ``fmod`` on half,
a half operation with an operand of unknown type, a half increment used as a value and a compound assignment to an
lvalue with side effects. ``min16float`` is a minimum-precision type that may compute in 32 bits and is not
legalized here.
"""

from __future__ import annotations

import copy
import math
import re
import struct
from dataclasses import dataclass

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from . import half_helpers as H
from .hub_types import HubType, TypeScope, call_name, element_type_node
from .rewrite import call, child_slots, clone, get_child, is_pure, set_child

__all__ = [
    "HALF_TARGETS",
    "HalfFloatUnsupportedError",
    "half_from_name",
    "legalize_half_floats",
    "round_to_half",
]

HALF_TARGETS = {
    "vulkan": "the SPIR-V emitter has no 16-bit floating-point type",
    "wgsl": "WGSL f16 needs the optional shader-f16 feature (WGSL section 4.1.1)",
}
# Sources whose floating-point literals adopt the type of a half operand (HLSL and Slang literal float).
FLOAT_LITERAL_ADOPTING = frozenset({"directx", "slang"})
HALF = HubType("f", 16)

_SCALARS = frozenset({"half", "float16", "float16_t", "f16"})
_VECTOR = re.compile(r"(?:f16vec|half|float16_t|float16)([234])")
_GENERIC_VECTOR = re.compile(r"vec([234])<(?:f16|half|float16_t|float16)>")
_MATRIX = re.compile(
    r"(?:f16mat[234](?:x[234])?|(?:half|float16_t|float16)[234]x[234]|mat[234]x[234]<(?:f16|half)>)"
)
_REGISTER_SPACES = frozenset(
    {"thread", "threadgroup", "workgroup", "private", "function", "shared"}
)
_STAGE_FIELDS = ("var_type", "param_type", "return_type", "member_type", "const_type")
_STEPS = {"++": "+", "--": "-"}
_COMPOUND = {"+=": "+", "-=": "-", "*=": "*", "/=": "/", "%=": "%"}
_STATEMENT_FIELDS = frozenset(
    {
        "statements",
        "body",
        "then_branch",
        "else_branch",
        "if_body",
        "else_body",
        "else_if_bodies",
    }
)
# Built-ins whose result on binary16 operands is a binary16 value: no rounding.
_EXACT = frozenset(
    {
        "abs",
        "fabs",
        "min",
        "max",
        "fmin",
        "fmax",
        "clamp",
        "saturate",
        "floor",
        "ceil",
        "trunc",
        "round",
        "rint",
        "roundEven",
        "sign",
        "select",
        "step",
        "copysign",
    }
)
# Built-ins computed in binary32 and rounded once; their accuracy bound is that of the binary32 function.
_ROUNDED = frozenset(
    {
        "exp",
        "exp2",
        "exp10",
        "log",
        "log2",
        "log10",
        "pow",
        "powr",
        "pown",
        "rootn",
        "ldexp",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "sinpi",
        "cospi",
        "tanpi",
        "rsqrt",
        "inversesqrt",
        "rcp",
        "fract",
        "mix",
        "lerp",
        "smoothstep",
        "dot",
        "length",
        "distance",
        "normalize",
        "cross",
        "reflect",
        "refract",
        "faceforward",
        "cbrt",
        "hypot",
        "degrees",
        "radians",
    }
)
_SCALAR_RESULT = frozenset({"dot", "length", "distance"})
_INTEGER_OPERAND = {"ldexp": 1, "pown": 1, "rootn": 1}
_EXACT_HELPERS = {"sqrt": "cgl_half_sqrt", "fma": "cgl_half_fma", "mad": "cgl_half_fma"}
_REFUSED_CALLS = frozenset({"fmod", "mod", "modf", "frexp", "remainder"})
_TO_BITS = re.compile(
    r"as_type<(?:ushort|short|uint16_t|int16_t|uint16|int16)>|asuint16|asint16"
    r"|float16BitsToUint16|float16BitsToInt16"
)
_FROM_BITS = re.compile(
    r"as_type<(?:half|float16|float16_t)>|asfloat16|uint16BitsToFloat16|int16BitsToFloat16"
)
# Other bit casts: of a half value (to a wider integer or a vector) and to half vectors.
_BIT_CAST = re.compile(
    r"as_type<.*>|bitcast<.*>|asuint|asint|asfloat|floatBitsToUint|floatBitsToInt"
    r"|float16BitsToUint16|float16BitsToInt16|asuint16|asint16"
)
_HALF_BIT_CAST = re.compile(
    r"(?:as_type|bitcast)<(?:(?:half|float16_t|float16|f16vec)[234]|vec[234]<(?:f16|half)>)>"
)


class HalfFloatUnsupportedError(TranslationDiagnostic):
    """A binary16 construct the target cannot compute exactly in binary32."""

    def __init__(self, target, what, *, reason_code, source_location=None):
        self.reason_code = reason_code
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.half-float-{reason_code}",)
        super().__init__(
            f"{what}: {HALF_TARGETS[target]}",
            target=target,
            code=f"{CODE_PREFIX}{target}-half-float-unsupported",
        )


# Types ----------------------------------------------------------------------------------------------------------
def half_from_name(name) -> HubType | None:
    name = str(name or "").replace(" ", "")
    if name in _SCALARS:
        return HALF
    for pattern in (_VECTOR, _GENERIC_VECTOR):
        match = pattern.fullmatch(name)
        if match:
            return HALF.with_lanes(int(match.group(1)))
    return None


def half_of(node) -> HubType | None:
    """The half value type a type node denotes (through references), or ``None``."""
    if isinstance(node, str):
        return half_from_name(node)
    if isinstance(node, A.PrimitiveType):
        return half_from_name(node.name)
    if isinstance(node, A.NamedType) and not node.generic_args:
        return half_from_name(node.name)
    if isinstance(node, A.VectorType):
        element = half_of(node.element_type)
        if element is not None and str(node.size).isdigit():
            return element.with_lanes(int(node.size))
        return None
    if isinstance(node, A.ReferenceType):
        return half_of(node.referenced_type)
    return None


def _is_half_matrix(node) -> bool:
    if isinstance(node, str):
        return bool(_MATRIX.fullmatch(node.replace(" ", "")))
    if isinstance(node, (A.PrimitiveType, A.NamedType)):
        return _is_half_matrix(str(node.name))
    if isinstance(node, A.MatrixType):
        return half_of(node.element_type) is not None
    for field in ("element_type", "pointee_type", "referenced_type"):
        inner = getattr(node, field, None)
        if inner is not None:
            return _is_half_matrix(inner)
    return False


def _float_type(lanes):
    scalar = A.PrimitiveType("float")
    return scalar if lanes == 1 else A.VectorType(scalar, lanes)


def _into_registers(node) -> bool:
    space = str(getattr(node, "address_space", None) or "").lower()
    return space in _REGISTER_SPACES or (
        isinstance(node, A.ReferenceType) and not space
    )


def widened(node):
    """``node`` with its half scalar, vector or element type replaced by ``float``; ``None`` if nothing is half.

    Arrays and pointers or references into register space are widened through; device, constant and unqualified
    pointees and buffer element types keep their half element, which the emitters lay out as 32-bit words.
    """
    if isinstance(node, str):
        half = half_from_name(node)
        return _float_type(half.lanes) if half is not None else None
    field = {
        A.ArrayType: "element_type",
        A.PointerType: "pointee_type",
        A.ReferenceType: "referenced_type",
    }.get(type(node))
    if field is not None:
        if field != "element_type" and not _into_registers(node):
            return None
        inner = widened(getattr(node, field))
        if inner is None:
            return None
        duplicate = copy.copy(node)
        setattr(duplicate, field, inner)
        return duplicate
    if isinstance(node, (A.PrimitiveType, A.NamedType, A.VectorType)):
        half = half_of(node)
        return _float_type(half.lanes) if half is not None else None
    return None


# Values ---------------------------------------------------------------------------------------------------------
def round_to_half(value: float) -> float | None:
    """``value`` rounded to binary32 and then to binary16, ties to even; ``None`` when that overflows or is NaN."""
    try:
        single = struct.unpack("<f", struct.pack("<f", value))[0]
        rounded = struct.unpack("<e", struct.pack("<e", single))[0]
    except (OverflowError, struct.error):
        return None
    return rounded if math.isfinite(rounded) else None


def _number(expr):
    """The numeric value of a literal (or a negated literal) and whether it is floating-point; ``None`` otherwise."""
    negate = False
    if isinstance(expr, A.UnaryOpNode) and expr.op in ("-", "+"):
        negate = expr.op == "-"
        expr = expr.operand
    if not isinstance(expr, A.LiteralNode):
        return None
    value = expr.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    literal_type = str(getattr(expr.literal_type, "name", expr.literal_type) or "")
    is_float = isinstance(value, float) or literal_type in ("float", "double", "half")
    return (-value if negate else value), is_float


def _half_literal(value: float):
    literal = A.LiteralNode(abs(value), A.PrimitiveType("float"))
    result = literal if math.copysign(1.0, value) > 0 else A.UnaryOpNode("-", literal)
    result.annotations["cgl_half"] = HALF
    return result


def _mark(node, lanes):
    node.annotations["cgl_half"] = HALF.with_lanes(lanes)
    return node


@dataclass(frozen=True)
class Kind:
    """An operand's category: ``half``, ``float`` (binary32), ``double``, ``int``, ``bool``, ``literal`` (a
    floating-point literal) or ``int_literal``."""

    name: str
    lanes: int = 1


# The pass -------------------------------------------------------------------------------------------------------
class HalfLegalizer:
    def __init__(self, program, source, target):
        self.program = program
        self.source = source
        self.target = target
        self.literals_adopt = source in FLOAT_LITERAL_ADOPTING
        self.scope = TypeScope(program)
        self.functions = {}
        for function in _functions(program):
            self.functions.setdefault(function.name, []).append(function)
        self.return_type = None
        self.required = set()

    def run(self):
        if not self._mentions_half():
            return
        self._refuse_matrices()
        for function in _functions(self.program):
            if function.name in _reserved_names() and not H.is_half_helper(function):
                self.refuse(
                    f"the program defines {function.name}, a name reserved for a binary16 helper",
                    "reserved-name",
                    function,
                )
        program, scope = self.program, self.scope
        for constant in getattr(program, "constants", None) or []:
            scope.declare(constant.name, constant.const_type)
            self._normalize_declaration(constant, "const_type", "value")
        self._declare_variables(getattr(program, "global_variables", None) or [])
        for cbuffer in getattr(program, "cbuffers", None) or []:
            scope.declare_cbuffer(cbuffer)
        for function in getattr(program, "functions", None) or []:
            if isinstance(function, A.FunctionNode):
                self.function(function)
        for stage in _stages(program):
            scope.push()
            self._declare_variables(getattr(stage, "local_variables", None) or [])
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                scope.declare_cbuffer(cbuffer)
            for function in list(getattr(stage, "local_functions", None) or []) + [
                getattr(stage, "entry_point", None)
            ]:
                if isinstance(function, A.FunctionNode):
                    self.function(function)
            scope.pop()
        self.retype()
        self.verify()
        self.inject()

    def _mentions_half(self) -> bool:
        for node in self.program.walk():
            for field in _STAGE_FIELDS:
                value = getattr(node, field, None)
                if widened(value) is not None or _is_half_matrix(value):
                    return True
            if isinstance(node, A.FunctionCallNode) and (
                half_from_name(call_name(node))
                or _is_half_matrix(call_name(node))
                or _FROM_BITS.fullmatch(call_name(node))
            ):
                return True
            if isinstance(node, A.CastNode) and half_of(node.target_type) is not None:
                return True
        return False

    def _refuse_matrices(self):
        for node in self.program.walk():
            types = [getattr(node, field, None) for field in _STAGE_FIELDS]
            if isinstance(node, A.FunctionCallNode):
                types.append(call_name(node))
            if isinstance(node, A.CastNode):
                types.append(node.target_type)
            if any(_is_half_matrix(value) for value in types if value is not None):
                self.refuse("a half matrix", "matrix", node)

    def _declare_variables(self, variables):
        for variable in variables:
            self.scope.declare(variable.name, variable.var_type)
            self.scope.declare_block_members(variable)
            self._normalize_declaration(variable, "var_type", "initial_value")

    def refuse(self, what, reason_code, node=None):
        raise HalfFloatUnsupportedError(
            self.target,
            what,
            reason_code=reason_code,
            source_location=getattr(node, "source_location", None),
        )

    # Traversal --------------------------------------------------------------------------------------------------
    def function(self, function):
        if function.body is None or H.is_half_helper(function):
            return
        self.scope.push()
        for parameter in function.parameters or []:
            self.scope.declare(parameter.name, _parameter_type(parameter))
        previous, self.return_type = self.return_type, half_of(function.return_type)
        if isinstance(function.body, list):
            function.body[:] = [self.statement(item) for item in function.body]
        else:
            self.visit(function.body)
        self.return_type = previous
        self.scope.pop()

    def visit(self, node):
        """Legalize ``node``'s subtree and return its replacement (``node`` itself if unchanged)."""
        if isinstance(node, (A.FunctionNode, A.StructNode, A.LambdaNode)):
            return node
        if isinstance(node, A.BlockNode):
            self.scope.push()
            node.statements = [self.statement(s) for s in node.statements or []]
            self.scope.pop()
            return node
        if isinstance(node, A.ForNode):
            self.scope.push()
            if node.init is not None:
                set_child(node, "init", None, self.statement(node.init))
            for field in ("condition", "body"):
                child = getattr(node, field)
                if child is not None:
                    set_child(node, field, None, self.visit(child))
            if node.update is not None:
                set_child(node, "update", None, self.statement(node.update))
            self.scope.pop()
            return node
        self.visit_children(node)
        if isinstance(node, A.VariableNode):
            self.scope.declare(node.name, node.var_type)
            self._normalize_declaration(node, "var_type", "initial_value")
            return node
        return self.rewrite(node)

    def visit_children(self, node):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            visit = self.statement if field in _STATEMENT_FIELDS else self.visit
            replacement = visit(child)
            if replacement is not child:
                set_child(node, field, key, replacement)

    def statement(self, node):
        """A statement: a half increment or decrement here becomes an assignment."""
        inner = node.expression if isinstance(node, A.ExpressionStatementNode) else node
        if isinstance(inner, A.UnaryOpNode) and inner.op in _STEPS:
            half = self.half_lvalue(inner.operand)
            if half is not None:
                self.visit_children(inner)
                self._require_pure(inner.operand, inner)
                step = A.BinaryOpNode(
                    clone(inner.operand),
                    _STEPS[inner.op],
                    A.LiteralNode(1, A.PrimitiveType("int")),
                )
                assignment = A.AssignmentNode(
                    inner.operand, self.coerce(self.rewrite_binary(step), half)
                )
                if isinstance(node, A.ExpressionStatementNode):
                    set_child(node, "expression", None, assignment)
                    return node
                return assignment
        return self.visit(node)

    # Rewrites ---------------------------------------------------------------------------------------------------
    def rewrite(self, node):
        if isinstance(node, A.AssignmentNode):
            return self.rewrite_assignment(node)
        if isinstance(node, A.UnaryOpNode):
            if node.op in _STEPS:
                if self.half_lvalue(node.operand) is not None:
                    self.refuse(
                        "a half increment or decrement used as a value",
                        "step-value",
                        node,
                    )
                return node
            half = self.half_value(node.operand)
            if node.op in ("-", "+") and half is not None:
                return _mark(node, half.lanes)
            return node
        if isinstance(node, A.FunctionCallNode):
            return self.rewrite_call(node)
        if isinstance(node, A.CastNode):
            half = half_of(node.target_type)
            if half is not None:
                return self.convert(node.expression, half, node)
            return node
        if isinstance(node, A.ReturnNode):
            if self.return_type is not None and node.value is not None:
                set_child(
                    node, "value", None, self.coerce(node.value, self.return_type)
                )
            return node
        if isinstance(node, A.BinaryOpNode):
            return self.rewrite_binary(node)
        if isinstance(node, A.TernaryOpNode):
            arms = {
                self.half_value(node.true_expr),
                self.half_value(node.false_expr),
            }
            if len(arms) == 1 and None not in arms:
                return _mark(node, arms.pop().lanes)
        return node

    def rewrite_binary(self, node):
        if node.op not in ("+", "-", "*", "/", "%"):
            return node
        left, right = self.kind(node.left), self.kind(node.right)
        if "half" not in (getattr(left, "name", None), getattr(right, "name", None)):
            return node
        if not self._half_operation((left, right), node, f"a half '{node.op}'"):
            return node
        for side, kind in (("left", left), ("right", right)):
            if kind.name != "half":
                set_child(node, side, None, self.coerce(getattr(node, side), HALF))
        lanes = max(left.lanes, right.lanes)
        if node.op == "%":
            self.refuse("'%' on half operands", "remainder", node)
        if node.op == "/":
            operands = [self._splat(node.left, lanes), self._splat(node.right, lanes)]
            return self.helper("cgl_half_div", lanes, operands, node)
        return self.helper("cgl_half_round", lanes, [node], node)

    def _half_operation(self, kinds, node, what) -> bool:
        """Whether operands with ``kinds`` (one of them half) make a half operation; refuse when that is unknown.

        A ``float`` or ``double`` operand makes a binary32 or binary64 operation, and so does a floating-point
        literal in sources where it does not adopt the half type. Integers convert to half.
        """
        for kind in kinds:
            if kind is None:
                self.refuse(f"{what} with an operand of unknown type", "unknown-operand", node)
            if kind.name == "bool":
                self.refuse(f"{what} with a bool operand", "bool-operand", node)
        return not any(
            kind.name in ("float", "double")
            or (kind.name == "literal" and not self.literals_adopt)
            for kind in kinds
        )

    def rewrite_assignment(self, node):
        half = self.half_lvalue(node.target)
        if half is None:
            return node
        if node.operator == "=":
            set_child(node, "value", None, self.coerce(node.value, half))
            return node
        operation = _COMPOUND.get(node.operator)
        if operation is None:
            self.refuse(
                f"compound assignment '{node.operator}' to a half value",
                "compound",
                node,
            )
        self._require_pure(node.target, node)
        combined = self.rewrite_binary(
            A.BinaryOpNode(clone(node.target), operation, node.value)
        )
        replacement = A.AssignmentNode(node.target, self.coerce(combined, half))
        replacement.source_location = node.source_location
        return replacement

    def rewrite_call(self, node):
        name = call_name(node)
        arguments = list(node.arguments or [])
        half = half_from_name(name)
        if half is not None:
            return self.construct(half, arguments, node)
        if (
            _TO_BITS.fullmatch(name)
            and len(arguments) == 1
            and self.half_value(arguments[0]) == HALF
        ):
            return self.helper("cgl_half_to_bits", 1, arguments, node, mark=False)
        if _BIT_CAST.fullmatch(name) and any(
            self.half_value(argument) is not None for argument in arguments
        ):
            self.refuse(f"the bit cast '{name}' of a half value", "bitcast", node)
        if _HALF_BIT_CAST.fullmatch(name):
            self.refuse(f"the bit cast '{name}' to a half vector", "bitcast", node)
        if _FROM_BITS.fullmatch(name):
            kind = self.kind(arguments[0]) if len(arguments) == 1 else None
            if kind is None or kind.name not in ("int", "int_literal") or kind.lanes != 1:
                self.refuse(
                    f"the bit cast '{name}' to half from a non-scalar 16-bit integer",
                    "bitcast",
                    node,
                )
            return self.helper("cgl_half_from_bits", 1, [call("uint", arguments)], node)
        callee = self._callee(name, len(arguments), node)
        if callee is not None:
            for index, parameter in enumerate(callee.parameters or []):
                if index >= len(arguments):
                    break
                declared = _parameter_type(parameter)
                if isinstance(declared, (A.ReferenceType, A.PointerType)):
                    continue
                if set(getattr(parameter, "qualifiers", None) or []) & {"out", "inout"}:
                    continue
                expected = half_of(declared)
                if expected is not None:
                    node.arguments[index] = self.coerce(arguments[index], expected)
            node.args = node.arguments
            return node
        if name == "buffer_store" and len(arguments) == 3:
            element = half_of(self._element_type(arguments[0]))
            if element is not None:
                node.arguments[2] = self.coerce(arguments[2], element)
                node.args = node.arguments
            return node
        if name == "buffer_load" and len(arguments) == 2:
            element = half_of(self._element_type(arguments[0]))
            return _mark(node, element.lanes) if element is not None else node
        if name in _EXACT or name in _ROUNDED or name in _EXACT_HELPERS:
            return self.rewrite_builtin(name, node)
        if name in _REFUSED_CALLS and any(
            self.half_value(argument) is not None for argument in arguments
        ):
            self.refuse(f"'{name}' on half operands", "remainder", node)
        return node

    def rewrite_builtin(self, name, node):
        arguments = list(node.arguments or [])
        kinds = [self.kind(argument) for argument in arguments]
        if not any(kind is not None and kind.name == "half" for kind in kinds):
            return node
        integer_index = _INTEGER_OPERAND.get(name)
        operands = [
            (index, kind)
            for index, kind in enumerate(kinds)
            if index != integer_index and not (kind is not None and kind.name == "bool")
        ]
        if not self._half_operation([k for _, k in operands], node, f"'{name}'"):
            return node
        lanes = 1
        for index, kind in operands:
            if kind.name != "half":
                node.arguments[index] = self.coerce(arguments[index], HALF)
            lanes = max(lanes, kind.lanes)
        node.args = node.arguments
        if name in _SCALAR_RESULT:
            lanes = 1
        if name in _EXACT_HELPERS:
            values = [
                argument if index == integer_index else self._splat(argument, lanes)
                for index, argument in enumerate(node.arguments)
            ]
            return self.helper(_EXACT_HELPERS[name], lanes, values, node)
        if name in _EXACT:
            return _mark(node, lanes)
        return self.helper("cgl_half_round", lanes, [node], node)

    # Conversions ------------------------------------------------------------------------------------------------
    def construct(self, half, arguments, node):
        """A half constructor: ``half(e)`` rounds, ``f16vec4(h, x, ...)`` rounds every component that is not half."""
        if len(arguments) == 1:
            return self.convert(arguments[0], half, node)
        parts = []
        exact = True
        for argument in arguments:
            kind = self.kind(argument)
            if kind is not None and kind.name == "half":
                parts.append(argument)
                continue
            folded = self._fold(argument)
            if folded is not None:
                parts.append(folded)
                continue
            if kind is not None and kind.name == "double":
                self.refuse(
                    "a double converted to half", "double-rounding", argument
                )
            exact = False
            parts.append(argument)
        vector = call(f"vec{half.lanes}", parts)
        vector.source_location = node.source_location
        if exact:
            return _mark(vector, half.lanes)
        return self.helper("cgl_half_round", half.lanes, [vector], node)

    def convert(self, expr, half, node):
        kind = self.kind(expr)
        if kind is not None and kind.name == "half" and kind.lanes in (1, half.lanes):
            if kind.lanes == half.lanes:
                return expr
            return _mark(call(f"vec{half.lanes}", [expr]), half.lanes)
        folded = self._fold(expr)
        if folded is not None:
            if half.lanes == 1:
                return folded
            return _mark(call(f"vec{half.lanes}", [folded]), half.lanes)
        if kind is not None and kind.name == "double":
            self.refuse("a double converted to half", "double-rounding", node)
        lanes = kind.lanes if kind is not None else half.lanes
        value = call("float" if lanes == 1 else f"vec{lanes}", [expr])
        rounded = self.helper("cgl_half_round", lanes, [value], node)
        if lanes != half.lanes:
            return _mark(call(f"vec{half.lanes}", [rounded]), half.lanes)
        return rounded

    def coerce(self, expr, half):
        """``expr`` as a value of ``half``: unchanged when it is a binary16 value already, rounded otherwise."""
        current = self.half_value(expr)
        if current is not None and current.lanes in (1, half.lanes):
            return expr
        return self.convert(expr, half, expr)

    def _fold(self, expr):
        number = _number(expr)
        if number is None:
            return None
        rounded = round_to_half(float(number[0]))
        return _half_literal(rounded) if rounded is not None else None

    def _splat(self, expr, lanes):
        value = self.half_value(expr)
        if lanes == 1 or (value is not None and value.lanes == lanes):
            return expr
        return _mark(call(f"vec{lanes}", [expr]), lanes)

    def helper(self, base, lanes, operands, node, *, mark=True):
        self.required.add((base, lanes))
        result = call(H.helper_name(base, lanes), operands)
        result.source_location = getattr(node, "source_location", None)
        return _mark(result, lanes) if mark else result

    def _callee(self, name, count, node):
        candidates = [
            function
            for function in self.functions.get(name, [])
            if len(function.parameters or []) == count
        ]
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        signatures = {
            tuple(half_of(_parameter_type(p)) for p in function.parameters or [])
            for function in candidates
        }
        if len(signatures) > 1 and any(any(s) for s in signatures):
            self.refuse(
                f"the call of overloaded '{name}' with half parameters",
                "overload",
                node,
            )
        return candidates[0]

    def _require_pure(self, target, node):
        if not is_pure(target):
            self.refuse(
                "a half update of an lvalue with side effects", "impure-lvalue", node
            )

    def _normalize_declaration(self, node, type_field, value_field):
        half = half_of(getattr(node, type_field, None))
        value = getattr(node, value_field, None)
        if value is None:
            return
        if half is not None:
            set_child(node, value_field, None, self.coerce(value, half))
        elif isinstance(value, A.ArrayLiteralNode):
            element = half_of(
                getattr(getattr(node, type_field, None), "element_type", None)
            )
            if element is not None:
                value.elements = [self.coerce(item, element) for item in value.elements]

    def _element_type(self, buffer):
        declared = self.scope.node_type(buffer)
        return element_type_node(declared) if declared is not None else None

    # Typing of source expressions -------------------------------------------------------------------------------
    def half_lvalue(self, expr):
        if isinstance(expr, A.SwizzleNode):
            base = self.half_lvalue(expr.vector_expr)
            if base is None:
                return None
            return base.with_lanes(len(str(expr.components)))
        if isinstance(expr, (A.IdentifierNode, A.ArrayAccessNode, A.MemberAccessNode)):
            return half_of(self.scope.node_type(expr))
        if isinstance(expr, A.UnaryOpNode) and expr.op == "*":
            pointer = self.scope.node_type(expr.operand)
            return half_of(getattr(pointer, "pointee_type", None))
        return None

    def half_value(self, expr):
        """The half type of a binary16-valued expression, or ``None``."""
        carried = (
            expr.annotations.get("cgl_half") if isinstance(expr, A.ASTNode) else None
        )
        if carried is not None:
            return carried
        lvalue = self.half_lvalue(expr)
        if lvalue is not None:
            return lvalue
        if isinstance(expr, (A.SwizzleNode, A.MemberAccessNode)):
            base = self.half_value(
                expr.vector_expr if isinstance(expr, A.SwizzleNode) else expr.object_expr
            )
            components = str(
                expr.components if isinstance(expr, A.SwizzleNode) else expr.member
            )
            if base is not None and base.lanes > 1 and re.fullmatch(
                r"[xyzw]{1,4}|[rgba]{1,4}", components
            ):
                return base.with_lanes(len(components))
            return None
        if isinstance(expr, A.ArrayAccessNode):
            base = self.half_value(expr.array_expr)
            return base.scalar if base is not None and base.lanes > 1 else None
        if isinstance(expr, A.FunctionCallNode):
            callee = self.functions.get(call_name(expr))
            if callee and len(callee) == 1:
                return half_of(callee[0].return_type)
        return None

    def kind(self, expr) -> Kind | None:
        half = self.half_value(expr)
        if half is not None:
            return Kind("half", half.lanes)
        number = _number(expr)
        if number is not None:
            return Kind("literal" if number[1] else "int_literal")
        inferred = self.scope.infer(expr)
        if inferred is None:
            return None
        if inferred.kind == "f":
            name = {16: "half", 32: "float", 64: "double"}.get(inferred.bits)
            return Kind(name, inferred.lanes) if name else None
        if inferred.kind in ("i", "u"):
            return Kind("int", inferred.lanes)
        return Kind("bool", inferred.lanes)

    # Retyping and helpers ---------------------------------------------------------------------------------------
    def retype(self):
        for node in self.program.walk():
            for field in _STAGE_FIELDS:
                old = getattr(node, field, None)
                new = widened(old)
                if new is None:
                    continue
                for alias in ("vtype",) if field in ("var_type", "param_type") else ():
                    if getattr(node, alias, None) is old:
                        setattr(node, alias, new)
                setattr(node, field, new)

    def verify(self):
        """No half register type or conversion is left; one that remains sits where the pass does not reach."""
        for node in self.program.walk():
            left = [
                field
                for field in _STAGE_FIELDS
                if widened(getattr(node, field, None)) is not None
            ]
            if (
                left
                or (
                    isinstance(node, A.FunctionCallNode)
                    and half_from_name(call_name(node))
                )
                or (isinstance(node, A.CastNode) and half_of(node.target_type) is not None)
            ):
                self.refuse(
                    f"a half type in {type(node).__name__} the pass cannot legalize",
                    "unlegalized",
                    node,
                )

    def inject(self):
        if not self.required:
            return
        functions = getattr(self.program, "functions", None)
        if not isinstance(functions, list):
            self.refuse(
                "the program has no function list to receive the binary16 helpers",
                "helpers",
            )
        functions[0:0] = [
            H.helper_function(base, lanes) for base, lanes in H.closure(self.required)
        ]


def _parameter_type(parameter):
    declared = getattr(parameter, "param_type", None)
    return declared if declared is not None else getattr(parameter, "var_type", None)


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _functions(program):
    yield from (
        f
        for f in getattr(program, "functions", None) or []
        if isinstance(f, A.FunctionNode)
    )
    for stage in _stages(program):
        yield from getattr(stage, "local_functions", None) or []
        if isinstance(getattr(stage, "entry_point", None), A.FunctionNode):
            yield stage.entry_point


def _reserved_names():
    return {
        H.helper_name(base, lanes)
        for base in H.SCALAR_HELPERS
        for lanes in (1, 2, 3, 4)
    }


def legalize_half_floats(program, source, target) -> bool:
    """Legalize binary16 values in ``program`` for ``target``; return whether the target needs the pass."""
    if target not in HALF_TARGETS or not isinstance(program, A.ShaderNode):
        return False
    HalfLegalizer(program, str(source or ""), target).run()
    return True
