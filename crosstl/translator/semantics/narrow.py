"""Narrow-integer legalization: 8- and 16-bit integer values on targets that have no 8- or 16-bit registers.

The SPIR-V emitter has no 8- or 16-bit integer types and WGSL has none at all (WGSL section 6.2.4), so a value of
``uchar``, ``short``, ``u8``, ``int16_t`` and the like would be computed in 32 bits and never wrap at its width
(defect S6: ``uchar v = 255; v = v + 1`` stores 256). This pass, run before rule carrying for those targets, gives
every narrow register a 32-bit ``int`` and keeps the invariant that it holds a value of its declared type:

- a value written to a narrow register (initializer, assignment, compound assignment, increment, argument of a
  narrow parameter, return from a narrow function) and every conversion to a narrow type is normalized: unsigned
  ``int(e) & mask``, signed ``((int(e) & mask) ^ sign) - sign``, which is the value modulo 2^N (C++ [conv.integral]
  and the explicit arithmetic types of GLSL, HLSL and SPIR-V);
- reads need nothing: ``int`` represents every 8- and 16-bit value, so C, C++, MSL, OpenCL C and CUDA integer
  promotion (C++ [conv.prom]) computes exactly the promoted result;
- arithmetic whose result keeps a narrow type is normalized as well: scalar operations of sources without integer
  promotion (GLSL, HLSL, SPIR-V, Slang, Rust, Mojo), and vector operations of every source (MSL section 2.2 and
  OpenCL C 6.3 do not promote vectors).

Registers, private and workgroup arrays and struct members become 32-bit; narrow pointees in device, constant or
unqualified memory are left to the emitters, which lay that storage out one 32-bit word per element as they do for
every target (a layout change for the host that the reflection metadata does not report yet). A narrow increment
used as a value and a compound assignment to an lvalue with side effects are refused with
:class:`NarrowIntegerUnsupportedError`. ``min16int`` and ``min16uint`` are
minimum-precision types that may compute in 32 bits, and are not narrow here.
"""

from __future__ import annotations

import copy
import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import HubType, TypeScope, call_name, iter_structs
from .rewrite import call, child_slots, clone, get_child, is_pure, set_child

__all__ = [
    "NARROW_TARGETS",
    "NarrowIntegerUnsupportedError",
    "legalize_narrow_integers",
]

NARROW_TARGETS = {
    "vulkan": "the SPIR-V emitter has no 8- or 16-bit integer types",
    "wgsl": "WGSL has no 8- or 16-bit integer types (WGSL section 6.2.4)",
}
# Sources whose scalar narrow operands are promoted to int before arithmetic (C++ [conv.prom], C99 6.3.1.1).
PROMOTING_SOURCES = frozenset({"metal", "cuda", "hip", "opencl", "cgl"})
# Sources where an int literal adopts the narrow type of the other operand (HLSL literal int, MSL vector operands).
_LITERAL_ADOPTING = frozenset({"directx", "metal", "opencl", "slang"})

_SCALARS = {
    "i8": HubType("i", 8),
    "int8": HubType("i", 8),
    "int8_t": HubType("i", 8),
    "char": HubType("i", 8),
    "signedchar": HubType("i", 8),
    "u8": HubType("u", 8),
    "uint8": HubType("u", 8),
    "uint8_t": HubType("u", 8),
    "uchar": HubType("u", 8),
    "unsignedchar": HubType("u", 8),
    "i16": HubType("i", 16),
    "int16": HubType("i", 16),
    "int16_t": HubType("i", 16),
    "short": HubType("i", 16),
    "signedshort": HubType("i", 16),
    "u16": HubType("u", 16),
    "uint16": HubType("u", 16),
    "uint16_t": HubType("u", 16),
    "ushort": HubType("u", 16),
    "unsignedshort": HubType("u", 16),
}
_GLSL_VECTOR = re.compile(r"(i8|u8|i16|u16)vec([234])")
_GENERIC_VECTOR = re.compile(r"vec([234])<(\w+)>")
_C_VECTOR = re.compile(r"([a-z0-9_]+?)([234])")
_REGISTER_QUALIFIERS = frozenset(
    {"shared", "groupshared", "workgroup", "threadgroup", "private", "static", "thread"}
)
_REGISTER_SPACES = frozenset(
    {"thread", "threadgroup", "workgroup", "private", "function", "shared"}
)
_STAGE_ATTRIBUTES = frozenset(
    {
        "compute",
        "vertex",
        "fragment",
        "kernel",
        "stage_entry",
        "mesh",
        "task",
        "amplification",
    }
)
_STEPS = {"++": "+", "--": "-"}
_COMPOUND = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "%=": "%",
    "<<=": "<<",
    ">>=": ">>",
    "&=": "&",
    "|=": "|",
    "^=": "^",
}
# Operations whose result may leave the range of in-range operands; ``&``, ``|``, ``^`` and ``>>`` stay in range.
_LEAVES_RANGE = frozenset({"+", "-", "*", "/", "%", "<<"})
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


class NarrowIntegerUnsupportedError(TranslationDiagnostic):
    """A narrow-integer construct the target cannot carry: host-visible narrow storage, or an unnormalizable write."""

    def __init__(self, target, what, *, reason_code, source_location=None):
        self.reason_code = reason_code
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.narrow-integer-{reason_code}",)
        super().__init__(
            f"{what}: {NARROW_TARGETS[target]}",
            target=target,
            code=f"{CODE_PREFIX}{target}-narrow-integer-unsupported",
        )


# Types ----------------------------------------------------------------------------------------------------------
def narrow_from_name(name) -> HubType | None:
    name = str(name or "").replace(" ", "")
    if name in _SCALARS:
        return _SCALARS[name]
    match = _GLSL_VECTOR.fullmatch(name)
    if match:
        return _SCALARS[match.group(1)].with_lanes(int(match.group(2)))
    for pattern, scalar, lanes in ((_GENERIC_VECTOR, 2, 1), (_C_VECTOR, 1, 2)):
        match = pattern.fullmatch(name)
        if match and match.group(scalar) in _SCALARS:
            return _SCALARS[match.group(scalar)].with_lanes(int(match.group(lanes)))
    return None


def narrow_of(node) -> HubType | None:
    """The narrow value type a type node denotes (through references), or ``None``."""
    if isinstance(node, str):
        return narrow_from_name(node)
    if isinstance(node, A.PrimitiveType):
        return narrow_from_name(node.name)
    if isinstance(node, A.NamedType) and not node.generic_args:
        return narrow_from_name(node.name)
    if isinstance(node, A.VectorType):
        element = narrow_of(node.element_type)
        if element is not None and str(node.size).isdigit():
            return element.with_lanes(int(node.size))
        return None
    if isinstance(node, A.ReferenceType):
        return narrow_of(node.referenced_type)
    return None


def _int_type(lanes):
    scalar = A.PrimitiveType("int")
    return scalar if lanes == 1 else A.VectorType(scalar, lanes)


def _into_registers(node) -> bool:
    """A pointer or reference into thread, private or workgroup memory, where every narrow value is an ``int``."""
    space = str(getattr(node, "address_space", None) or "").lower()
    return space in _REGISTER_SPACES or (
        isinstance(node, A.ReferenceType) and not space
    )


def widened(node):
    """``node`` with its narrow scalar, vector or element type replaced by ``int``; ``None`` if nothing is narrow.

    Arrays and pointers or references into register space are widened through. Pointers and references into
    device, constant or unqualified memory, and buffer element types, keep their narrow element: the emitters lay out
    that storage one 32-bit word per element, and the pointer-reinterpretation lowering reads the element width.
    """
    if isinstance(node, str):
        narrow = narrow_from_name(node)
        return _int_type(narrow.lanes) if narrow is not None else None
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
        narrow = narrow_of(node)
        return _int_type(narrow.lanes) if narrow is not None else None
    return None


def _parameter_type(parameter):
    declared = getattr(parameter, "param_type", None)
    return declared if declared is not None else getattr(parameter, "var_type", None)


# Normalization --------------------------------------------------------------------------------------------------
def _literal(value, lanes):
    literal = A.LiteralNode(value, A.PrimitiveType("int"))
    return literal if lanes == 1 else call(f"ivec{lanes}", [literal])


def _exempt(node):
    node.annotations["cgl_rule_exempt"] = True
    return node


def normalized(expr, narrow: HubType):
    """``expr`` converted to ``int`` (lane-wise) and reduced modulo 2^bits into ``narrow``'s range."""
    lanes = narrow.lanes
    if (
        lanes == 1
        and isinstance(expr, A.LiteralNode)
        and type(expr.value) is int  # noqa: E721 - bool literals are not folded
    ):
        reduced = expr.value & ((1 << narrow.bits) - 1)
        if narrow.kind == "i" and reduced >= 1 << (narrow.bits - 1):
            reduced -= 1 << narrow.bits
        folded = A.LiteralNode(abs(reduced), A.PrimitiveType("int"))
        result = folded if reduced >= 0 else A.UnaryOpNode("-", folded)
        result.annotations["cgl_narrow"] = narrow
        return result
    value = call("int" if lanes == 1 else f"ivec{lanes}", [expr])
    masked = A.BinaryOpNode(value, "&", _literal((1 << narrow.bits) - 1, lanes))
    if narrow.kind == "u":
        result = masked
    else:
        sign = 1 << (narrow.bits - 1)
        flipped = A.BinaryOpNode(masked, "^", _literal(sign, lanes))
        result = _exempt(A.BinaryOpNode(flipped, "-", _literal(sign, lanes)))
    result.annotations["cgl_narrow"] = narrow
    return result


# The pass -------------------------------------------------------------------------------------------------------
class NarrowLegalizer:
    def __init__(self, program, source, target):
        self.program = program
        self.source = source
        self.target = target
        self.promotes = source in PROMOTING_SOURCES
        self.literal_adopts = source in _LITERAL_ADOPTING
        self.scope = TypeScope(program)
        self.structs = {
            struct.name: struct
            for struct in iter_structs(program)
            if getattr(struct, "name", None)
        }
        self.functions = {}
        for function in self._functions():
            self.functions.setdefault(function.name, []).append(function)
        self.return_type = None

    # Program structure ------------------------------------------------------------------------------------------
    def _functions(self):
        yield from (
            f
            for f in getattr(self.program, "functions", None) or []
            if isinstance(f, A.FunctionNode)
        )
        for stage in self._stages():
            yield from getattr(stage, "local_functions", None) or []
            if isinstance(getattr(stage, "entry_point", None), A.FunctionNode):
                yield stage.entry_point

    def _stages(self):
        stages = getattr(self.program, "stages", None) or {}
        return list(stages.values()) if isinstance(stages, dict) else list(stages)

    def _entry_points(self):
        entries = [
            stage.entry_point
            for stage in self._stages()
            if isinstance(getattr(stage, "entry_point", None), A.FunctionNode)
        ]
        for function in getattr(self.program, "functions", None) or []:
            names = {
                getattr(a, "name", None) for a in getattr(function, "attributes", [])
            }
            if names & _STAGE_ATTRIBUTES or set(function.qualifiers or []) & {"kernel"}:
                entries.append(function)
        return entries

    def run(self):
        if not self._mentions_narrow():
            return
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
        for stage in self._stages():
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

    def _mentions_narrow(self) -> bool:
        for node in self.program.walk():
            for field in (
                "var_type",
                "param_type",
                "return_type",
                "member_type",
                "const_type",
            ):
                if widened(getattr(node, field, None)) is not None:
                    return True
            if isinstance(node, A.FunctionCallNode) and narrow_from_name(
                call_name(node)
            ):
                return True
            if isinstance(node, A.CastNode) and narrow_of(node.target_type) is not None:
                return True
        return False

    def _declare_variables(self, variables):
        for variable in variables:
            self.scope.declare(variable.name, variable.var_type)
            self.scope.declare_block_members(variable)
            self._normalize_declaration(variable, "var_type", "initial_value")

    def refuse(self, what, reason_code, node=None):
        raise NarrowIntegerUnsupportedError(
            self.target,
            what,
            reason_code=reason_code,
            source_location=getattr(node, "source_location", None),
        )

    # Functions --------------------------------------------------------------------------------------------------
    def function(self, function):
        if function.body is None:
            return
        self.scope.push()
        for parameter in function.parameters or []:
            self.scope.declare(parameter.name, _parameter_type(parameter))
        previous, self.return_type = self.return_type, narrow_of(function.return_type)
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
        """A statement: a narrow increment or decrement here becomes an assignment."""
        inner = node.expression if isinstance(node, A.ExpressionStatementNode) else node
        if isinstance(inner, A.UnaryOpNode) and inner.op in _STEPS:
            narrow = self.narrow_lvalue(inner.operand)
            if narrow is not None:
                self.visit_children(inner)
                step = A.BinaryOpNode(
                    clone(inner.operand),
                    _STEPS[inner.op],
                    A.LiteralNode(1, A.PrimitiveType("int")),
                )
                self._require_pure(inner.operand, inner)
                assignment = A.AssignmentNode(inner.operand, normalized(step, narrow))
                if isinstance(node, A.ExpressionStatementNode):
                    set_child(node, "expression", None, assignment)
                    return node
                return assignment
        return self.visit(node)

    # Rewrites ---------------------------------------------------------------------------------------------------
    def rewrite(self, node):
        if isinstance(node, A.AssignmentNode):
            return self.rewrite_assignment(node)
        if isinstance(node, A.UnaryOpNode) and node.op in _STEPS:
            if self.narrow_lvalue(node.operand) is not None:
                self.refuse(
                    "a narrow-integer increment or decrement used as a value",
                    "step-value",
                    node,
                )
            return node
        if isinstance(node, A.FunctionCallNode):
            return self.rewrite_call(node)
        if isinstance(node, A.CastNode):
            narrow = narrow_of(node.target_type)
            if narrow is not None:
                return normalized(node.expression, narrow)
            return node
        if isinstance(node, A.ReturnNode):
            if self.return_type is not None and node.value is not None:
                set_child(
                    node, "value", None, self.coerce(node.value, self.return_type)
                )
            return node
        if isinstance(node, (A.BinaryOpNode, A.UnaryOpNode)):
            narrow = self.arithmetic_narrow(node)
            if narrow is not None:
                return normalized(node, narrow)
        return node

    def rewrite_assignment(self, node):
        narrow = self.narrow_lvalue(node.target)
        if narrow is None:
            return node
        if node.operator == "=":
            set_child(node, "value", None, self.coerce(node.value, narrow))
            return node
        operation = _COMPOUND.get(node.operator)
        if operation is None:
            self.refuse(
                f"compound assignment '{node.operator}' to a narrow integer",
                "compound",
                node,
            )
        self._require_pure(node.target, node)
        combined = A.BinaryOpNode(clone(node.target), operation, node.value)
        replacement = A.AssignmentNode(node.target, normalized(combined, narrow))
        replacement.source_location = node.source_location
        return replacement

    def rewrite_call(self, node):
        name = call_name(node)
        arguments = list(node.arguments or [])
        narrow = narrow_from_name(name)
        if narrow is not None:
            if len(arguments) == 1:
                return normalized(arguments[0], narrow)
            vector = call(f"ivec{narrow.lanes}", arguments)
            return normalized(vector, narrow)
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
                expected = narrow_of(declared)
                if expected is not None:
                    node.arguments[index] = self.coerce(arguments[index], expected)
            node.args = node.arguments
            return node
        return node

    def _callee(self, name, count, node):
        candidates = [
            function
            for function in self.functions.get(name, [])
            if len(function.parameters or []) == count
        ]
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        signatures = {
            tuple(narrow_of(_parameter_type(p)) for p in function.parameters or [])
            for function in candidates
        }
        if len(signatures) > 1 and any(any(s) for s in signatures):
            self.refuse(
                f"the call of overloaded '{name}' with narrow-integer parameters",
                "overload",
                node,
            )
        return candidates[0]

    def _require_pure(self, target, node):
        if not is_pure(target):
            self.refuse(
                "a narrow-integer update of an lvalue with side effects",
                "impure-lvalue",
                node,
            )

    def _normalize_declaration(self, node, type_field, value_field):
        narrow = narrow_of(getattr(node, type_field, None))
        value = getattr(node, value_field, None)
        if value is None:
            return
        if narrow is not None:
            set_child(node, value_field, None, self.coerce(value, narrow))
        elif isinstance(value, A.ArrayLiteralNode):
            element = narrow_of(
                getattr(getattr(node, type_field, None), "element_type", None)
            )
            if element is not None:
                value.elements = [self.coerce(item, element) for item in value.elements]

    # Narrow typing of source expressions ------------------------------------------------------------------------
    def coerce(self, expr, narrow):
        """``expr`` as a value of ``narrow``: unchanged when it already is one, normalized otherwise."""
        return expr if self.narrow_value(expr) == narrow else normalized(expr, narrow)

    def narrow_lvalue(self, expr):
        if isinstance(expr, A.SwizzleNode):
            base = self.narrow_lvalue(expr.vector_expr)
            if base is None:
                return None
            return base.with_lanes(len(str(expr.components)))
        if isinstance(expr, (A.IdentifierNode, A.ArrayAccessNode, A.MemberAccessNode)):
            return narrow_of(self.scope.node_type(expr))
        if isinstance(expr, A.UnaryOpNode) and expr.op == "*":
            pointer = self.scope.node_type(expr.operand)
            return narrow_of(getattr(pointer, "pointee_type", None))
        return None

    def narrow_value(self, expr):
        """The narrow type of a source-level value, or ``None`` when it is promoted or not narrow."""
        carried = (
            expr.annotations.get("cgl_narrow") if isinstance(expr, A.ASTNode) else None
        )
        if carried is not None:
            return carried
        lvalue = self.narrow_lvalue(expr)
        if lvalue is not None:
            return lvalue
        if isinstance(expr, A.TernaryOpNode):
            arms = {
                self.narrow_value(expr.true_expr),
                self.narrow_value(expr.false_expr),
            }
            return arms.pop() if len(arms) == 1 else None
        if isinstance(expr, A.FunctionCallNode):
            callee = self.functions.get(call_name(expr))
            if callee and len(callee) == 1:
                return narrow_of(callee[0].return_type)
        return None

    def arithmetic_narrow(self, node):
        """The narrow type an arithmetic result keeps, when the source does not promote it."""
        if isinstance(node, A.UnaryOpNode):
            if node.op not in ("-", "~"):
                return None
            operand = self.narrow_value(node.operand)
            if operand is None or (self.promotes and operand.lanes == 1):
                return None
            return operand
        if node.op not in _LEAVES_RANGE:
            return None
        left, right = self.narrow_value(node.left), self.narrow_value(node.right)
        if left is None and right is None:
            return None
        if left is None or right is None:
            other = node.left if left is None else node.right
            known = left or right
            if not (
                isinstance(other, A.LiteralNode)
                and (self.literal_adopts or known.lanes > 1)
            ):
                return None
            left = right = known
        if left != right and not (
            left.scalar == right.scalar and 1 in (left.lanes, right.lanes)
        ):
            return None
        result = left if left.lanes >= right.lanes else right
        if self.promotes and result.lanes == 1:
            return None
        return result

    # Retyping ---------------------------------------------------------------------------------------------------
    def retype(self):
        for node in self.program.walk():
            for field in (
                "var_type",
                "param_type",
                "return_type",
                "member_type",
                "const_type",
            ):
                old = getattr(node, field, None)
                new = widened(old)
                if new is None:
                    continue
                for alias in ("vtype",) if field in ("var_type", "param_type") else ():
                    if getattr(node, alias, None) is old:
                        setattr(node, alias, new)
                setattr(node, field, new)

    def verify(self):
        """No narrow register type is left; one that remains sits where the pass does not reach, so it is refused."""
        for node in self.program.walk():
            left = [
                field
                for field in (
                    "var_type",
                    "param_type",
                    "return_type",
                    "member_type",
                    "const_type",
                )
                if widened(getattr(node, field, None)) is not None
            ]
            if (
                left
                or (
                    isinstance(node, A.FunctionCallNode)
                    and narrow_from_name(call_name(node))
                )
                or (
                    isinstance(node, A.CastNode)
                    and narrow_of(node.target_type) is not None
                )
            ):
                self.refuse(
                    f"a narrow-integer type in {type(node).__name__} the pass cannot legalize",
                    "unlegalized",
                    node,
                )


def legalize_narrow_integers(program, source, target) -> bool:
    """Legalize narrow integers in ``program`` for ``target``; return whether the target needs the pass."""
    if target not in NARROW_TARGETS or not isinstance(program, A.ShaderNode):
        return False
    NarrowLegalizer(program, str(source or ""), target).run()
    return True
