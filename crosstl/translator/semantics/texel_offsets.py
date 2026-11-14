"""Texel offsets that the target needs as constant expressions.

GLSL requires the offset of every texel lookup except ``textureGatherOffset`` to be a constant expression (GLSL 4.60
and GLSL ES 3.20 section 8.9.2, "The offset value must be a constant expression"), HLSL an immediate integer (the
``offset`` operand of ``Texture2D.Sample`` and its forms; DXC: "Offsets to texture access operations must be
immediate values"), WGSL a const-expression (WGSL section 17.7.6 ``textureSample``) and SPIR-V a ``ConstOffset``
operand, a non-constant ``Offset`` being valid on gathers only (VUID-RuntimeSpirv-Offset-10213 without
``maintenance8``). MSL accepts any ``int2`` offset, and Metal shaders keep one in a local (``int2 o = int2(1, 0);``)
or compute it in a loop (``for (int i = -1; i <= 1; i++) ... sample_compare(s, uv, ref, int2(i, j))``), which the
GLSL and HLSL generators refused and the SPIR-V generator emitted as an invalid ``Offset`` operand.

For the targets in :data:`TARGETS`, :func:`legalize_texel_offsets` makes such offsets constant where the program
already determines them:

- a local integer scalar or vector declared with a constant-expression initializer and never written afterwards
  (assigned, incremented, passed to a callee that may write it, or its address taken), or declared ``const``, holds
  that constant: its reads in a texel lookup's offset are replaced by the constant;
- a ``for`` loop whose induction variable reaches a non-constant offset of a lookup other than a gather is unrolled
  when the loop runs a constant number of times, at most :data:`MAX_TRIP_COUNT`: an integer induction variable with
  a constant start, compared with a constant bound (``<``, ``<=``, ``>``, ``>=``, ``!=``), stepped by ``++``, ``--``,
  ``+=`` or ``-=`` a constant, and neither written by the body nor left by a ``break`` or ``continue`` of the loop.
  Each iteration becomes a block declaring the induction variable ``const`` with that iteration's value, which is
  what the loop computes. At most :data:`MAX_UNROLLED_ITERATIONS` iterations are unrolled per function.

Offsets that stay non-constant are left to the generators, which refuse them.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..validation import (
    OFFSET_DIMENSION_INTRINSIC_NAMES,
    texture_offset_argument_indices,
)
from .hub_types import call_name, is_pure_callee, type_from_name, type_from_node
from .rewrite import child_slots, clone, get_child, set_child
from .scoped_walk import ScopedWalker, parameter_type, program_stages

__all__ = [
    "MAX_TRIP_COUNT",
    "MAX_UNROLLED_ITERATIONS",
    "TARGETS",
    "legalize_texel_offsets",
]

TARGETS = frozenset(
    {
        "opengl",
        "glsl",
        "webgl",
        "gles",
        "essl",
        "directx",
        "hlsl",
        "dx11",
        "dx12",
        "d3d11",
        "d3d12",
        "vulkan",
        "spirv",
        "wgsl",
        "slang",
    }
)
OFFSET_CALLS = frozenset(OFFSET_DIMENSION_INTRINSIC_NAMES)
# Gathers whose single offset may vary (GLSL 4.00 and GLSL ES 3.20; SPIR-V ``Offset`` with ImageGatherExtended).
VARIABLE_OFFSET_CALLS = frozenset({"textureGatherOffset", "textureGatherCompareOffset"})
MAX_TRIP_COUNT = 64
MAX_UNROLLED_ITERATIONS = 256

_LOOPS = (A.ForNode, A.ForInNode, A.WhileNode, A.DoWhileNode, A.LoopNode)
# Qualifiers of a local whose value the function alone determines.
_VALUE_QUALIFIERS = frozenset(
    {
        "const",
        "constexpr",
        "thread",
        "private",
        "function",
        "highp",
        "mediump",
        "lowp",
        "precise",
    }
)
_WRITING_UNARY = frozenset({"++", "--", "&"})
_ASSIGNING = frozenset(
    {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
)
_CONSTANT_UNARY = frozenset({"-", "+", "~", "!"})
_CONSTANT_BINARY = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "%",
        "<<",
        ">>",
        "&",
        "|",
        "^",
        "&&",
        "||",
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
    }
)
_COMPARISONS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "!=": lambda a, b: a != b,
}
_MIRRORED = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "!=": "!="}
_TYPE_CONSTRUCTOR = re.compile(
    r"(?:[biu]|i16|u16|i64|u64)?vec[234]|(?:u?int|short|ushort|long|ulong)[234]?"
    r"|(?:u?int(?:8|16|32|64)_t)[234]?|float|half|double|bool"
)


def legalize_texel_offsets(program, target) -> int:
    """Forward constant locals into texel offsets and unroll the loops that compute them; returns the rewrites."""
    if str(target or "").lower() not in TARGETS or not isinstance(
        program, A.ShaderNode
    ):
        return 0
    writers = _writing_parameters(program)
    rewrites = 0
    for function, stage in _functions(program):
        if getattr(function, "body", None) is None or not _has_offset_call(
            function.body
        ):
            continue
        budget = MAX_UNROLLED_ITERATIONS
        rejected = set()
        while True:
            finder = _OffsetWalker(program, writers, function, rejected)
            finder.run(function, stage)
            if finder.found is None:
                break
            entry, values = finder.found
            if max(len(values), 1) > budget:
                rejected.add(id(entry.loop))
                continue
            budget -= max(len(values), 1)
            _unroll(entry, values)
            rewrites += 1
        forwarder = _OffsetWalker(program, writers, function, rejected, forward=True)
        forwarder.run(function, stage)
        rewrites += forwarder.forwarded
    return rewrites


def _functions(program):
    for function in getattr(program, "functions", None) or []:
        if isinstance(function, A.FunctionNode):
            yield function, None
    for stage in program_stages(program):
        for function in getattr(stage, "local_functions", None) or []:
            if isinstance(function, A.FunctionNode):
                yield function, stage
        entry = getattr(stage, "entry_point", None)
        if isinstance(entry, A.FunctionNode):
            yield entry, stage


def _has_offset_call(node):
    if isinstance(node, list):
        return any(_has_offset_call(item) for item in node)
    if not isinstance(node, A.ASTNode):
        return False
    if isinstance(node, A.FunctionCallNode) and call_name(node) in OFFSET_CALLS:
        return True
    return any(
        _has_offset_call(get_child(node, field, key))
        for field, key in child_slots(node)
    )


def _writing_parameters(program):
    """Function name -> parameter positions some overload may write (``out``, ``inout``, references, pointers)."""
    writers = {}
    for function, _ in _functions(program):
        positions = writers.setdefault(function.name, set())
        for index, parameter in enumerate(function.parameters or []):
            qualifiers = {
                str(q).lower() for q in getattr(parameter, "qualifiers", None) or []
            }
            declared = parameter_type(parameter)
            if qualifiers & {"out", "inout"} or isinstance(
                declared, (A.ReferenceType, A.PointerType)
            ):
                positions.add(index)
    return writers


class _Local:
    """A local declaration in scope; ``value`` is the constant it holds, or ``None``."""

    __slots__ = ("declaration", "value")

    def __init__(self, declaration, value):
        self.declaration = declaration
        self.value = value


class _Loop:
    """An enclosing ``for`` loop, where it sits, and the iteration values when it can be unrolled."""

    __slots__ = ("loop", "parent", "field", "key", "values")

    def __init__(self, loop, parent, field, key, values):
        self.loop = loop
        self.parent = parent
        self.field = field
        self.key = key
        self.values = values


class _Found(Exception):
    pass


class _OffsetWalker(ScopedWalker):
    """Walks one function with its declarations in scope, to find a loop to unroll or to forward constants."""

    def __init__(self, program, writers, function, rejected, forward=False):
        super().__init__(program)
        self.writers = writers
        self.rejected = rejected
        self.forward = forward
        self.found = None
        self.forwarded = 0
        self.written = _written_names(function.body, writers)
        self.module = {}
        self.frames = [{}]
        self.loops = []
        self._module_constants(
            getattr(program, "constants", None) or [], "const_type", "value"
        )
        self._module_constants(
            getattr(program, "global_variables", None) or [],
            "var_type",
            "initial_value",
            True,
        )

    def _module_constants(self, declarations, type_field, value_field, qualified=False):
        for declaration in declarations:
            name = getattr(declaration, "name", None)
            if not name:
                continue
            if qualified:
                qualifiers = {
                    str(q).lower()
                    for q in getattr(declaration, "qualifiers", None) or []
                }
                if not qualifiers & {"const", "constexpr"}:
                    self.module.pop(name, None)
                    continue
            value = getattr(declaration, value_field, None)
            self.module[name] = (
                value if _constant(value, self.module.__contains__) else None
            )

    # Scopes ---------------------------------------------------------------------------------------------------
    def run(self, function, stage):
        saved = dict(self.module)
        self.scope.push()
        if stage is not None:
            for variable in getattr(stage, "local_variables", None) or []:
                if getattr(variable, "name", None) is None:
                    continue
                self.scope.declare(variable.name, variable.var_type)
                self.scope.declare_block_members(variable)
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                self.scope.declare_cbuffer(cbuffer)
            self._module_constants(
                getattr(stage, "local_variables", None) or [],
                "var_type",
                "initial_value",
                True,
            )
        self.frames.append({})
        for parameter in function.parameters or []:
            self.scope.declare(
                getattr(parameter, "name", None), parameter_type(parameter)
            )
            self._bind(getattr(parameter, "name", None), _Local(parameter, None))
        try:
            self._walk(function.body, function, "body", None)
        except _Found:
            pass
        self.frames.pop()
        self.scope.pop()
        self.scope.pop()
        self.module = saved

    def _push(self):
        self.scope.push()
        self.frames.append({})

    def _pop(self):
        self.frames.pop()
        self.scope.pop()

    def _bind(self, name, local):
        if name:
            self.frames[-1][name] = local

    def _local(self, name):
        for frame in reversed(self.frames):
            if name in frame:
                return frame[name]
        return None

    def _is_module_constant(self, name):
        return self._local(name) is None and self.module.get(name) is not None

    # Walk -----------------------------------------------------------------------------------------------------
    def _walk(self, node, parent=None, field=None, key=None):
        if isinstance(node, list):
            self._push()
            self._sequence(node, parent, field)
            self._pop()
            return
        if not isinstance(node, A.ASTNode) or isinstance(
            node, (A.FunctionNode, A.StructNode)
        ):
            return
        if isinstance(node, A.VariableNode):
            self._children(node)
            return
        if isinstance(node, A.BlockNode):
            self._push()
            self._sequence(node.statements, node, "statements")
            self._pop()
            return
        if isinstance(node, A.ForNode):
            self._for(node, parent, field, key)
            return
        if isinstance(node, A.SwitchNode):
            self._push()
            self._walk(node.expression, node, "expression")
            for case in list(node.cases or []) + _default_cases(node):
                self._walk(getattr(case, "value", None), case, "value")
                self._sequence(
                    getattr(case, "statements", None) or [], case, "statements"
                )
            self._pop()
            return
        self._children(node)
        if isinstance(node, A.FunctionCallNode):
            self._call(node)

    def _sequence(self, items, parent, field):
        for index, item in enumerate(list(items)):
            self._walk(item, parent, field, index)
            if isinstance(item, A.VariableNode):
                self._declare(item)

    def _children(self, node):
        lists = {}
        for field, key in list(child_slots(node)):
            if isinstance(getattr(node, field, None), list):
                lists.setdefault(field, None)
                continue
            self._walk(get_child(node, field, key), node, field, key)
        for field in lists:
            self._walk(getattr(node, field), node, field)

    def _for(self, node, parent, field, key):
        self._push()
        init = (
            node.init[0]
            if isinstance(node.init, list) and len(node.init) == 1
            else node.init
        )
        start = None
        if isinstance(init, A.VariableNode):
            self._children(init)
            start = self._int_value(init.initial_value)
            self._declare(init)
        elif isinstance(node.init, list):
            # ``for (int i = 0, j = 0; ...)``: each declaration is in scope for the rest of the statement.
            for item in node.init:
                self._walk(item, node, "init")
                if isinstance(item, A.VariableNode):
                    self._declare(item)
        else:
            self._walk(node.init, node, "init")
        values = self._trip_values(node, init, start)
        self._walk(node.condition, node, "condition")
        self._walk(node.update, node, "update")
        self.loops.append(_Loop(node, parent, field, key, values))
        self._walk(node.body, node, "body")
        self.loops.pop()
        self._pop()

    def _declare(self, node):
        self.scope.declare(node.name, node.var_type)
        value = None
        if self._holds_constant(node):
            resolved, _ = self._resolve(node.initial_value)
            if _constant(resolved, self._is_module_constant):
                value = resolved
        self._bind(node.name, _Local(node, value))

    def _holds_constant(self, node):
        if node.initial_value is None or isinstance(
            node.var_type, (A.ArrayType, A.PointerType, A.ReferenceType)
        ):
            return False
        value_type = type_from_node(node.var_type)
        if value_type is None or value_type.kind not in ("i", "u"):
            return False
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
        if not qualifiers <= _VALUE_QUALIFIERS:
            return False
        return (
            "const" in qualifiers
            or "constexpr" in qualifiers
            or node.name not in self.written
        )

    # Calls ----------------------------------------------------------------------------------------------------
    def _call(self, node):
        name = call_name(node)
        if name not in OFFSET_CALLS:
            return
        arguments = node.arguments or []
        for index in self._offset_indices(name, arguments):
            resolved, changed = self._resolve(arguments[index])
            constant = _constant(resolved, self._is_module_constant)
            if self.forward:
                if changed and constant:
                    folded = self._fold(resolved, self.scope.infer(arguments[index]))
                    set_child(node, "arguments", index, folded or resolved)
                    self.forwarded += 1
            elif not constant and name not in VARIABLE_OFFSET_CALLS:
                self._choose_loop(_identifier_names(resolved))

    def _offset_indices(self, name, arguments):
        count = len(arguments)
        if name == "texelFetchOffset":
            # ``texelFetchOffset(t, P, lod, offset)``, or ``(t, s, P, lod, offset)`` with a separate sampler.
            indices = {count - 1} if count in (4, 5) else set()
        else:
            indices = set(texture_offset_argument_indices(name, False, count))
            indices |= set(texture_offset_argument_indices(name, True, count))
        result = []
        for index in sorted(indices):
            value_type = self.scope.infer(arguments[index])
            if value_type is not None and value_type.kind in ("i", "u"):
                result.append(index)
        return result

    def _choose_loop(self, names):
        for entry in self.loops:
            if entry.values is None or id(entry.loop) in self.rejected:
                continue
            init = (
                entry.loop.init[0]
                if isinstance(entry.loop.init, list)
                else entry.loop.init
            )
            local = self._local(init.name)
            if init.name in names and local is not None and local.declaration is init:
                self.found = (entry, entry.values)
                raise _Found()

    # Values ---------------------------------------------------------------------------------------------------
    def _resolve(self, expr):
        """A copy of ``expr`` with the constant locals it reads replaced by their values, and whether any was."""
        if isinstance(expr, A.IdentifierNode):
            local = self._local(expr.name)
            if local is not None and local.value is not None:
                return clone(local.value), True
            return clone(expr), False
        if not isinstance(expr, A.ASTNode):
            return expr, False
        duplicate = clone(expr)
        return duplicate, self._substitute(duplicate)

    def _substitute(self, node):
        changed = False
        for field, key in list(child_slots(node)):
            if isinstance(node, A.FunctionCallNode) and field in ("function", "name"):
                continue
            child = get_child(node, field, key)
            if isinstance(child, A.IdentifierNode):
                local = self._local(child.name)
                if local is not None and local.value is not None:
                    set_child(node, field, key, clone(local.value))
                    changed = True
            elif isinstance(child, A.ASTNode):
                changed = self._substitute(child) or changed
        return changed

    def _fold(self, expr, value_type):
        """``expr`` as one ``ivecN``/``uvecN`` constructor of literals (or a literal), or ``None``.

        GLSL generators accept an offset spelled with literals only, so ``ivec2(2, -2) + ivec2(1, -1)`` is folded
        to ``ivec2(3, -3)`` with 32-bit wrapping arithmetic.
        """
        if (
            value_type is None
            or value_type.kind not in ("i", "u")
            or value_type.bits != 32
        ):
            return None
        lanes = _lanes(expr, self._module_expression, value_type.kind)
        if lanes is None or len(lanes) != value_type.lanes:
            return None
        scalar = A.PrimitiveType("int" if value_type.kind == "i" else "uint")
        literals = [_integer_literal(lane, scalar) for lane in lanes]
        if value_type.lanes == 1:
            return literals[0]
        prefix = "i" if value_type.kind == "i" else "u"
        return A.FunctionCallNode(
            A.IdentifierNode(f"{prefix}vec{value_type.lanes}"), literals
        )

    def _module_expression(self, name):
        return None if self._local(name) is not None else self.module.get(name)

    def _int_value(self, expr):
        if expr is None:
            return None
        resolved, _ = self._resolve(expr)
        return _evaluate(resolved, self._module_value)

    def _module_value(self, name):
        if self._local(name) is not None:
            return None
        value = self.module.get(name)
        return _evaluate(value, self._module_value) if value is not None else None

    def _trip_values(self, node, init, start):
        """The induction variable's value in each iteration, or ``None`` when the loop cannot be unrolled."""
        if not isinstance(init, A.VariableNode) or start is None:
            return None
        value_type = type_from_node(init.var_type)
        if (
            value_type is None
            or value_type.kind not in ("i", "u")
            or value_type.lanes != 1
        ):
            return None
        low, high = _range(value_type)
        comparison = self._bound(node.condition, init.name, value_type)
        step = _step(node.update, init.name, self._int_value)
        if comparison is None or not step or not low <= start < high:
            return None
        operator, bound = comparison
        if init.name in _written_names(node.body, self.writers) or _leaves(node.body):
            return None
        values, value = [], start
        while _COMPARISONS[operator](value, bound):
            values.append(value)
            value += step
            if len(values) > MAX_TRIP_COUNT or not low <= value < high:
                return None
        return values

    def _bound(self, condition, name, value_type):
        if (
            not isinstance(condition, A.BinaryOpNode)
            or condition.operator not in _COMPARISONS
        ):
            return None
        operator, other = condition.operator, condition.right
        if not _is_name(condition.left, name):
            if not _is_name(condition.right, name):
                return None
            operator, other = _MIRRORED[operator], condition.left
        if name in _identifier_names(other):
            return None
        bound = self._int_value(other)
        other_type = self.scope.infer(other)
        if (
            bound is None
            or other_type is None
            or (other_type.kind == "u") != (value_type.kind == "u")
        ):
            return None
        return operator, bound


def _default_cases(node):
    default = getattr(node, "default_case", None)
    if isinstance(default, list):
        return [A.CaseNode(None, default)] if default else []
    return [default] if isinstance(default, A.ASTNode) else []


def _range(value_type):
    if value_type.kind == "u":
        return 0, 1 << value_type.bits
    # The most negative value has no literal spelling in GLSL or HLSL.
    return -(1 << (value_type.bits - 1)) + 1, 1 << (value_type.bits - 1)


def _step(update, name, evaluate):
    if isinstance(update, list):
        update = update[0] if len(update) == 1 else None
    if isinstance(update, A.ExpressionStatementNode):
        update = update.expression
    if isinstance(update, A.UnaryOpNode) and _is_name(update.operand, name):
        return {"++": 1, "--": -1}.get(update.operator)
    if isinstance(update, A.AssignmentNode) and _is_name(update.target, name):
        if update.operator in ("+=", "-="):
            amount = evaluate(update.value)
            if amount is None:
                return None
            return amount if update.operator == "+=" else -amount
    return None


def _is_name(node, name):
    return isinstance(node, A.IdentifierNode) and node.name == name


def _unroll(entry, values):
    loop = entry.loop
    init = loop.init[0] if isinstance(loop.init, list) else loop.init
    iterations = []
    for value in values:
        declaration = A.VariableNode(
            init.name,
            init.var_type,
            _integer_literal(value, init.var_type),
            qualifiers=["const"],
            is_mutable=False,
        )
        body = clone(loop.body)
        if isinstance(body, A.BlockNode) and not any(
            isinstance(statement, A.VariableNode) and statement.name == init.name
            for statement in body.statements
        ):
            statements = [declaration, *body.statements]
        else:
            statements = [declaration, body]
        iterations.append(A.BlockNode(statements))
    container = getattr(entry.parent, entry.field)
    if isinstance(container, list) and isinstance(entry.key, int):
        container[entry.key : entry.key + 1] = iterations
    else:
        set_child(entry.parent, entry.field, entry.key, A.BlockNode(iterations))


def _integer_literal(value, type_node):
    scalar = type_from_node(type_node)
    literal_type = (
        type_node
        if isinstance(type_node, A.PrimitiveType)
        else A.PrimitiveType(
            "uint" if scalar is not None and scalar.kind == "u" else "int"
        )
    )
    literal = A.LiteralNode(abs(value), literal_type)
    return A.UnaryOpNode("-", literal) if value < 0 else literal


# Expressions ----------------------------------------------------------------------------------------------------
def _constant(node, is_constant_name):
    """Whether ``node`` is a constant expression whose names ``is_constant_name`` accepts."""
    if isinstance(node, A.LiteralNode):
        return isinstance(node.value, (bool, int, float))
    if isinstance(node, A.IdentifierNode):
        return is_constant_name(node.name)
    if isinstance(node, A.FunctionCallNode):
        name = call_name(node)
        if not (_TYPE_CONSTRUCTOR.fullmatch(name) or type_from_name(name) is not None):
            return False
        return all(
            _constant(argument, is_constant_name) for argument in node.arguments or []
        )
    if isinstance(node, A.ConstructorNode):
        return all(
            _constant(argument, is_constant_name) for argument in node.arguments or []
        )
    if isinstance(node, A.UnaryOpNode):
        return node.operator in _CONSTANT_UNARY and _constant(
            node.operand, is_constant_name
        )
    if isinstance(node, A.BinaryOpNode):
        return (
            node.operator in _CONSTANT_BINARY
            and _constant(node.left, is_constant_name)
            and _constant(node.right, is_constant_name)
        )
    if isinstance(node, A.TernaryOpNode):
        return all(
            _constant(part, is_constant_name)
            for part in (node.condition, node.true_expr, node.false_expr)
        )
    if isinstance(node, A.CastNode):
        return _constant(node.expression, is_constant_name)
    return False


def _evaluate(node, lookup):
    """The integer value of a constant integer expression, or ``None``; ``lookup`` gives a name's value."""
    if isinstance(node, A.LiteralNode):
        value = node.value
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(node, A.IdentifierNode):
        return lookup(node.name)
    if isinstance(node, A.UnaryOpNode) and node.operator in ("-", "+"):
        operand = _evaluate(node.operand, lookup)
        if operand is None:
            return None
        return -operand if node.operator == "-" else operand
    if isinstance(node, A.BinaryOpNode) and node.operator in ("+", "-", "*", "/", "%"):
        left, right = _evaluate(node.left, lookup), _evaluate(node.right, lookup)
        if left is None or right is None:
            return None
        if node.operator in ("/", "%"):
            if right == 0:
                return None
            # C division truncates toward zero.
            quotient = (
                abs(left) // abs(right) * (1 if (left < 0) == (right < 0) else -1)
            )
            return quotient if node.operator == "/" else left - quotient * right
        return {"+": left + right, "-": left - right, "*": left * right}[node.operator]
    return None


_VECTOR_CONSTRUCTOR = re.compile(r"(?:([iu])vec|(u?)int)([234])?")


def _wrap(value, kind):
    value %= 1 << 32
    return value - (1 << 32) if kind == "i" and value >= 1 << 31 else value


def _lanes(node, lookup, kind):
    """Lane values of a constant 32-bit integer scalar or vector expression, or ``None``."""
    if isinstance(node, A.LiteralNode):
        value = node.value
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value >= 1 << (31 if kind == "i" else 32)
        ):
            return None
        return [value]
    if isinstance(node, A.IdentifierNode):
        value = lookup(node.name)
        return _lanes(value, lookup, kind) if value is not None else None
    if isinstance(node, A.UnaryOpNode) and node.operator in ("-", "+", "~"):
        operand = _lanes(node.operand, lookup, kind)
        if operand is None:
            return None
        if node.operator == "-":
            return [_wrap(-lane, kind) for lane in operand]
        if node.operator == "~":
            return [_wrap(~lane, kind) for lane in operand]
        return operand
    if isinstance(node, A.BinaryOpNode) and node.operator in (
        "+",
        "-",
        "*",
        "/",
        "%",
        "&",
        "|",
        "^",
    ):
        left, right = _lanes(node.left, lookup, kind), _lanes(node.right, lookup, kind)
        if (
            left is None
            or right is None
            or (len(left) != len(right) and 1 not in (len(left), len(right)))
        ):
            return None
        count = max(len(left), len(right))
        left, right = left * (count // len(left)), right * (count // len(right))
        result = []
        for a, b in zip(left, right):
            if node.operator in ("/", "%"):
                if b == 0:
                    return None
                quotient = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
                result.append(
                    _wrap(quotient if node.operator == "/" else a - quotient * b, kind)
                )
            else:
                operation = {
                    "+": a + b,
                    "-": a - b,
                    "*": a * b,
                    "&": a & b,
                    "|": a | b,
                    "^": a ^ b,
                }
                result.append(_wrap(operation[node.operator], kind))
        return result
    if isinstance(node, A.FunctionCallNode):
        match = _VECTOR_CONSTRUCTOR.fullmatch(call_name(node))
        if match is None:
            return None
        lanes = []
        for argument in node.arguments or []:
            values = _lanes(argument, lookup, kind)
            if values is None:
                return None
            lanes.extend(values)
        count = int(match.group(3) or 1)
        if len(lanes) == 1 and count > 1:
            lanes = lanes * count
        return lanes if len(lanes) == count else None
    return None


def _identifier_names(node):
    names = set()

    def visit(current):
        if isinstance(current, A.IdentifierNode):
            names.add(current.name)
            return
        if not isinstance(current, A.ASTNode):
            return
        for field, key in child_slots(current):
            if isinstance(current, A.FunctionCallNode) and field in (
                "function",
                "name",
            ):
                continue
            visit(get_child(current, field, key))

    visit(node)
    return names


def _root_name(node):
    while True:
        if isinstance(node, A.IdentifierNode):
            return node.name
        if isinstance(node, A.MemberAccessNode):
            node = node.object_expr
        elif isinstance(node, A.ArrayAccessNode):
            node = node.array_expr
        elif isinstance(node, A.SwizzleNode):
            node = node.vector_expr
        else:
            return None


def _written_names(body, writers):
    """Names a statement may write: assignment and increment targets, addresses taken and arguments a callee writes.

    The analysis goes by name, so a write to any variable of a name counts for every declaration of that name.
    """
    written = set()

    def lvalue(target):
        root = _root_name(target)
        if root is not None:
            written.add(root)
        else:
            written.update(_identifier_names(target))

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, A.ASTNode):
            return
        if isinstance(node, A.AssignmentNode):
            lvalue(node.target)
        elif isinstance(node, A.BinaryOpNode) and node.operator in _ASSIGNING:
            lvalue(node.left)
        elif isinstance(node, A.UnaryOpNode) and node.operator in _WRITING_UNARY:
            lvalue(node.operand)
        elif isinstance(node, A.FunctionCallNode):
            name = call_name(node)
            reads_only = (
                is_pure_callee(name)
                or name.startswith(("texture", "texel"))
                or (name in writers and not writers[name])
            )
            if not reads_only:
                positions = writers.get(name)
                for index, argument in enumerate(node.arguments or []):
                    if positions is None or index in positions:
                        root = _root_name(argument)
                        if root is not None:
                            written.add(root)
        for field, key in child_slots(node):
            visit(get_child(node, field, key))

    visit(body)
    return written


def _leaves(body):
    """Whether ``body`` has a ``break`` or ``continue`` of the loop it is the body of."""

    def visit(node, in_switch):
        if isinstance(node, list):
            return any(visit(item, in_switch) for item in node)
        if not isinstance(node, A.ASTNode) or isinstance(node, _LOOPS):
            return False
        if isinstance(node, A.ContinueNode):
            return True
        if isinstance(node, A.BreakNode):
            return not in_switch
        nested = in_switch or isinstance(node, (A.SwitchNode, A.MatchNode))
        return any(
            visit(get_child(node, field, key), nested)
            for field, key in child_slots(node)
        )

    return visit(body, False)
