"""Array constructors used as values, for targets that build arrays only where they are declared.

GLSL writes an array value as a constructor, ``vec3[2](a, b)``, wherever an expression may stand (GLSL 4.60
section 5.4.4, "Array Constructors"). HLSL, MSL and the SPIR-V generator build an array from its elements only in
an aggregate initializer: the initializer list of a declaration, or a member of a structure or array being
initialized. A declaration's own array constructor is already its initializer list (:mod:`.unsized_arrays`);
anywhere else generators emitted the GLSL spelling or refused it.

:func:`hoist_array_constructors` rewrites the other array constructors for those targets:

- an array constructor that is a member of an aggregate (an argument of a structure constructor, or an element of
  an array constructor or initializer list) becomes an initializer list in place;
- any other one, a function argument, an assigned, returned or indexed value, gets a local of its own, declared
  with the constructor's elements as its initializer list just before the statement that uses it, and the
  constructor is replaced with the local.

The initializer of a shader-scope constant or variable, at program scope or in a stage block, gets the first
rewrite too, and a named structure constructor such as ``W{m: float[2](1.0, 2.0), s: 3.0}`` is an aggregate there.
An initializer has no statement to declare a local before, so a constructor indexed with an integer literal and
holding pure elements is replaced with the element it selects, ``float[2](1.0, 2.0)[1]`` with ``2.0``, and any other
array constructor in it is refused.

An unsized constructor, ``float[](1.0, 2.0)``, has as many elements as arguments (GLSL 4.60 section 5.4.4).

The local is evaluated where its statement starts, so it is introduced only when nothing the statement evaluates
before the constructor has effects, and only when the constructor's own elements are pure if the constructor sits
in an operand that may not be evaluated (a branch of ``?:``, the right side of ``&&`` or ``||``, a later ``else
if`` condition). A constructor in a loop condition or update is evaluated on every iteration and cannot be
hoisted. Such constructors are refused with :class:`ArrayConstructorUnsupportedError`.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, SourceSpan, TranslationDiagnostic
from .hub_types import call_name, iter_functions
from .rewrite import child_slots, get_child, is_pure, set_child
from .struct_scalar_splats import struct_member_types

__all__ = [
    "ArrayConstructorUnsupportedError",
    "array_constructor_shape",
    "hoist_array_constructors",
]

# ``element[size]``: the outermost size comes first, so ``float[2][3]`` has two ``float[3]`` elements (GLSL 4.60
# section 4.1.9). A size that names a constant, ``float[N]``, is the number of elements written.
_ARRAY_CONSTRUCTOR = re.compile(r"\s*(\w+)\s*\[\s*(\w*)\s*\]((?:\s*\[\s*\w*\s*\])*)\s*")
# Targets whose generators build arrays only in aggregate initializers.
_DECLARATION_ONLY_TARGETS = frozenset(
    {"metal", "directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "vulkan", "spirv"}
)
# C++ targets: no array constructor expression, and a C array cannot be assigned.
_CPP_TARGETS = frozenset({"cuda", "hip"})
_TEMPORARY_PREFIX = "cgl_array_value"
# Targets whose generators write a named structure constructor as an aggregate initializer.
_BRACED_CONSTRUCTOR_TARGETS = frozenset({"metal"})


class ArrayConstructorUnsupportedError(TranslationDiagnostic):
    """An array constructor the target cannot evaluate where and when the source does."""

    def __init__(self, message, *, target=None, reason, source_location=None):
        self.reason_code = reason
        self.source_location = source_location
        super().__init__(
            message,
            span=_span(source_location),
            target=target,
            code=f"{CODE_PREFIX}array-constructor-unsupported",
        )


def hoist_array_constructors(program, target) -> int:
    """Rewrite array constructors outside declarations in place; returns how many were rewritten."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target not in (
        _DECLARATION_ONLY_TARGETS | _CPP_TARGETS
    ):
        return 0
    hoister = _Hoister(program, target)
    for constant in getattr(program, "constants", None) or []:
        if isinstance(constant, A.ConstantNode):
            hoister.initializer(constant, "value", "constant")
    for variable in _shader_scope_variables(program):
        hoister.initializer(variable, "initial_value", "variable")
    for function in iter_functions(program):
        body = getattr(function, "body", None)
        if isinstance(body, A.BlockNode):
            hoister.block(body.statements)
        elif isinstance(body, list):
            hoister.block(body)
    return hoister.count


def array_constructor_shape(expr):
    """``(element spelling, size)`` of an array constructor call such as ``vec3[2](a, b)``, or ``None``.

    An unsized constructor, ``vec3[](a, b)``, has as many elements as arguments.
    """
    if not isinstance(expr, A.FunctionCallNode):
        return None
    match = _ARRAY_CONSTRUCTOR.fullmatch(
        call_name(expr) or _indexed_type(expr.function)
    )
    if match is None:
        return None
    size, inner = match.group(2), re.sub(r"\s+", "", match.group(3))
    count = int(size) if size.isdigit() else len(expr.arguments or [])
    return match.group(1) + inner, count


def _indexed_type(callee):
    """``float[2][3]`` for the callee the parser writes for that constructor, a built-in type name indexed with
    literal or named sizes; ``""`` for any other callee, so a call through an indexed value is not one.
    """
    sizes = []
    while isinstance(callee, A.ArrayAccessNode):
        index = callee.index_expr
        if isinstance(index, A.LiteralNode) and str(index.value).isdigit():
            sizes.append(str(index.value))
        elif isinstance(index, A.IdentifierNode) and index.name.isidentifier():
            sizes.append(index.name)
        else:
            return ""
        callee = callee.array_expr
    if not sizes or not isinstance(callee, A.IdentifierNode):
        return ""
    match = _ARRAY_CONSTRUCTOR.fullmatch(callee.name)
    base = match.group(1) if match else callee.name
    if not base.isidentifier() or isinstance(_element_type(base), A.NamedType):
        return ""
    return callee.name + "".join(f"[{size}]" for size in reversed(sizes))


def _shader_scope_variables(program):
    """The initialized variables declared at program scope or in a stage block, in declaration order."""
    variables = list(getattr(program, "global_variables", None) or [])
    stages = getattr(program, "stages", None) or {}
    for stage in list(stages.values()) if isinstance(stages, dict) else list(stages):
        variables.extend(getattr(stage, "local_variables", None) or [])
    return [
        variable
        for variable in variables
        if isinstance(variable, A.VariableNode)
        and isinstance(getattr(variable, "initial_value", None), A.ASTNode)
    ]


def _selected_element(node):
    """The element ``node`` selects when it indexes an array constructor of pure elements with an in-range integer
    literal, as ``float[2](1.0, 2.0)[1]`` selects ``2.0``; otherwise ``None``."""
    if not isinstance(node, A.ArrayAccessNode):
        return None
    constructor = node.array_expr
    shape = array_constructor_shape(constructor)
    index = getattr(node.index_expr, "value", None)
    if shape is None or not isinstance(node.index_expr, A.LiteralNode):
        return None
    if isinstance(index, str):
        index = index.rstrip("uU")
        index = int(index) if index.isdigit() else None
    elements = list(constructor.arguments or [])
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or len(elements) != shape[1]
        or not 0 <= index < len(elements)
        or not all(is_pure(element) for element in elements)
    ):
        return None
    return elements[index]


def _element_type(spelling):
    from ..lexer import Lexer
    from ..parser import Parser

    return Parser(Lexer(f"{spelling} x;").tokens).parse_type()


class _Slot:
    """How a statement evaluates one of its expressions: ``repeated`` per loop iteration, or ``conditional``."""

    def __init__(self, repeated=False, conditional=False):
        self.repeated = repeated
        self.conditional = conditional


class _Hoister:
    def __init__(self, program, target):
        self.target = target
        self.count = 0
        self._shader_scope = False
        self.structs = set(struct_member_types(program))
        self.names = _declared_names(program)

    # Statements ----------------------------------------------------------------------------------------------
    def block(self, statements):
        index = 0
        while index < len(statements):
            hoisted = self.statement(statements[index])
            statements[index:index] = hoisted
            index += len(hoisted) + 1

    def body(self, node, field):
        child = getattr(node, field, None)
        if isinstance(child, A.BlockNode):
            self.block(child.statements)
        elif isinstance(child, list):
            self.block(child)
        elif isinstance(child, A.ASTNode):
            hoisted = self.statement(child)
            if hoisted:
                set_child(node, field, None, A.BlockNode(hoisted + [child]))

    def statement(self, node):
        hoisted, effects = [], [False]
        once = _Slot()
        if isinstance(node, A.BlockNode):
            self.block(node.statements)
        elif isinstance(node, A.VariableNode):
            value = getattr(node, "initial_value", None)
            if array_constructor_shape(value) is not None:
                node.initial_value = A.ArrayLiteralNode(list(value.arguments or []))
            self.expression(node, "initial_value", hoisted, once, effects)
        elif isinstance(node, A.ExpressionStatementNode):
            self.expression(node, "expression", hoisted, once, effects)
        elif isinstance(node, A.ReturnNode):
            self.expression(node, "value", hoisted, once, effects)
        elif isinstance(node, A.IfNode):
            self.expression(node, "condition", hoisted, once, effects)
            conditions = getattr(node, "else_if_conditions", None) or []
            for index in range(len(conditions)):
                self.expression(
                    conditions, index, hoisted, _Slot(conditional=True), effects
                )
            self.body(node, "then_branch")
            for body in getattr(node, "else_if_bodies", None) or []:
                if isinstance(body, A.BlockNode):
                    self.block(body.statements)
            self.body(node, "else_branch")
        elif isinstance(node, A.ForNode):
            init = getattr(node, "init", None)
            if isinstance(init, A.ASTNode):
                hoisted.extend(self.statement(init))
            for field in ("condition", "update"):
                self.expression(node, field, hoisted, _Slot(repeated=True), effects)
            self.body(node, "body")
        elif isinstance(node, (A.WhileNode, A.DoWhileNode)):
            self.expression(node, "condition", hoisted, _Slot(repeated=True), effects)
            self.body(node, "body")
        elif isinstance(node, A.SwitchNode):
            self.expression(node, "expression", hoisted, once, effects)
            for case in getattr(node, "cases", None) or []:
                if isinstance(getattr(case, "statements", None), list):
                    self.block(case.statements)
            default = getattr(node, "default_case", None)
            if isinstance(default, list):
                self.block(default)
        elif isinstance(node, (A.ForInNode, A.LoopNode)):
            self.body(node, "body")
        elif isinstance(node, (A.AssignmentNode, A.ExpressionNode)):
            # A bare expression statement, such as a for-loop initializer.
            self._walk(node, hoisted, once, effects)
        return hoisted

    def initializer(self, node, field, kind):
        """Rewrite the array constructors in a shader-scope constant's or variable's initializer: members of its
        aggregate become lists, and an indexed constructor becomes the element it selects.
        """
        value = getattr(node, field, None)
        if array_constructor_shape(value) is not None:
            value = A.ArrayLiteralNode(list(value.arguments or []))
            setattr(node, field, value)
            self.count += 1
        value = self._select_elements(node, field)
        hoisted = []
        self._shader_scope = True
        try:
            self._walk(value, hoisted, _Slot(), [False])
        finally:
            self._shader_scope = False
        if hoisted:
            self._refuse(
                f"an array constructor in the initializer of shader-scope {kind} '{node.name}' is not a member "
                "of an aggregate",
                "array-constructor-at-shader-scope",
                getattr(node, "source_location", None),
            )

    def _select_elements(self, owner, field, key=None):
        """Replace each indexed constructor under ``owner.field`` with the element it selects, innermost first,
        so ``float[2](a, float[2](b, c)[0])[1]`` becomes ``b``; returns the new child.
        """
        node = get_child(owner, field, key)
        for child_field, child_key in list(child_slots(node)):
            self._select_elements(node, child_field, child_key)
        element = _selected_element(node)
        while element is not None:
            set_child(owner, field, key, element)
            self.count += 1
            node, element = element, _selected_element(element)
        return node

    # Expressions ---------------------------------------------------------------------------------------------
    def expression(self, owner, field, hoisted, slot, effects):
        value = owner[field] if isinstance(owner, list) else getattr(owner, field, None)
        if not isinstance(value, A.ASTNode):
            return
        shape = array_constructor_shape(value)
        before = effects[0]
        self._walk(value, hoisted, slot, effects)
        if shape is not None:
            name = self._hoist(value, shape, hoisted, slot, before)
            if isinstance(owner, list):
                owner[field] = A.IdentifierNode(name)
            else:
                set_child(owner, field, None, A.IdentifierNode(name))

    def _walk(self, node, hoisted, slot, effects):
        if (
            self.target in _CPP_TARGETS
            and isinstance(node, A.AssignmentNode)
            and array_constructor_shape(node.value) is not None
        ):
            element, size = array_constructor_shape(node.value)
            self._refuse(
                f"array constructor '{element}[{size}]' is assigned to an array",
                "array-constructor-assigned",
                getattr(node, "source_location", None),
            )
        aggregate = self._is_aggregate(node)
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            child_slot = slot
            if _is_conditional_operand(node, field):
                child_slot = _Slot(slot.repeated, True)
            # Effects of the constructor's own elements happen when the hoisted local is built, in order.
            before = effects[0]
            self._walk(child, hoisted, child_slot, effects)
            shape = array_constructor_shape(child)
            if shape is None:
                continue
            if aggregate:
                literal = A.ArrayLiteralNode(list(child.arguments or []))
                literal.source_location = getattr(child, "source_location", None)
                set_child(node, field, key, literal)
                self.count += 1
                continue
            name = self._hoist(child, shape, hoisted, child_slot, before)
            set_child(node, field, key, A.IdentifierNode(name))
        if _has_own_effects(node):
            effects[0] = True

    def _is_aggregate(self, node):
        if isinstance(node, A.ArrayLiteralNode):
            return True
        if array_constructor_shape(node) is not None:
            return True
        if isinstance(node, A.ConstructorNode):
            # Metal writes a named structure constructor, ``W{m: ..., s: ...}``, as the aggregate initializer
            # ``W{...}``, so in a shader-scope initializer its members are initialized in place. HLSL writes it as
            # a call, whose arguments cannot be lists, and in a function body array members get locals as other
            # call arguments do.
            return (
                self._shader_scope
                and self.target in _BRACED_CONSTRUCTOR_TARGETS
                and _type_spelling(node) in self.structs
            )
        return isinstance(node, A.FunctionCallNode) and call_name(node) in self.structs

    def _hoist(self, constructor, shape, hoisted, slot, effects_before):
        element, size = shape
        elements = list(constructor.arguments or [])
        location = getattr(constructor, "source_location", None)
        # The outermost size is written first: two ``float[3]`` elements are ``float[2][3]``.
        spelled = re.sub(
            r"\w+", lambda base: f"{base.group()}[{size}]", element, count=1
        )
        if slot.repeated:
            self._refuse(
                f"array constructor '{spelled}' is evaluated on every iteration of a loop condition or update",
                "array-constructor-in-loop-header",
                location,
            )
        if effects_before:
            self._refuse(
                f"array constructor '{spelled}' is evaluated after other effects of its statement",
                "array-constructor-after-effects",
                location,
            )
        if slot.conditional and not all(is_pure(element) for element in elements):
            self._refuse(
                f"array constructor '{spelled}' with effectful elements may not be evaluated",
                "array-constructor-conditional-effects",
                location,
            )
        if len(elements) != size:
            self._refuse(
                f"array constructor '{spelled}' has {len(elements)} elements",
                "array-constructor-element-count",
                location,
            )
        name = self._fresh_name()
        declared = A.ArrayType(
            _element_type(element), A.LiteralNode(size, A.PrimitiveType("int"))
        )
        temporary = A.VariableNode(
            name=name, var_type=declared, initial_value=A.ArrayLiteralNode(elements)
        )
        temporary.source_location = location
        hoisted.append(temporary)
        self.count += 1
        return name

    def _refuse(self, detail, reason, location):
        raise ArrayConstructorUnsupportedError(
            f"{detail}; the target builds arrays only in declarations and aggregate initializers, so the value "
            "cannot be built where the source builds it",
            target=self.target,
            reason=reason,
            source_location=location,
        )

    def _fresh_name(self):
        index = 0
        while f"{_TEMPORARY_PREFIX}_{index}" in self.names:
            index += 1
        name = f"{_TEMPORARY_PREFIX}_{index}"
        self.names.add(name)
        return name


def _type_spelling(node):
    constructed = getattr(node, "constructor_type", None)
    return getattr(constructed, "name", None)


def _is_conditional_operand(node, field):
    if isinstance(node, A.TernaryOpNode):
        return field in ("true_expr", "false_expr")
    if isinstance(node, A.BinaryOpNode) and node.op in ("&&", "||"):
        return field == "right"
    return False


def _has_own_effects(node):
    if isinstance(node, A.AssignmentNode):
        return True
    if isinstance(node, A.UnaryOpNode) and node.op in ("++", "--"):
        return True
    if isinstance(node, A.FunctionCallNode) and array_constructor_shape(node) is None:
        return not is_pure(A.FunctionCallNode(node.function, []))
    return False


def _declared_names(program):
    names = set()
    stack, seen = [program], set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        name = getattr(current, "name", None)
        if isinstance(name, str):
            names.add(name)
        stack.extend(
            get_child(current, field, key) for field, key in child_slots(current)
        )
    return names


def _span(location):
    if location is None:
        return None
    if isinstance(location, dict):
        line, column, file = (location.get(key) for key in ("line", "column", "file"))
    else:
        line, column, file = (
            getattr(location, key, None) for key in ("line", "column", "file")
        )
    if not isinstance(line, int):
        return None
    return SourceSpan(
        file=file, line=line, column=column if isinstance(column, int) else None
    )
