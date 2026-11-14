"""Atomic built-ins on storage buffers and workgroup variables, for WGSL.

GLSL, HLSL and the CrossGL that Metal sources import to apply atomic built-ins to ordinary ``int`` and ``uint``
memory: a buffer member, a structured-buffer element, or a ``shared``/``groupshared``/``threadgroup`` variable.
WGSL instead gives such memory the type ``atomic<i32>`` or ``atomic<u32>`` (WGSL section 6.2.8), allows atomic
types only in the ``storage`` address space with ``read_write`` access and in ``workgroup`` memory, passes a pointer
as the first argument of every atomic built-in (section 17.8), and permits no other access to atomic memory.

``plan_wgsl_buffer_atomics`` finds every location an atomic built-in addresses, a *slot*: a module-scope variable or
a structure member, after zero or more subscripts. A CrossGL ``atomic<T>`` declaration is a slot too. On a copy of
the AST it then rewrites every function:

- the target of an atomic call keeps its l-value form, and the generator takes its address;
- a read of a slot becomes ``atomicLoad``, ``slot = v`` becomes ``atomicStore``, and ``slot op= v`` or ``slot++``
  as a statement becomes ``atomicStore(slot, atomicLoad(slot) op v)``: the same non-atomic read-modify-write;
- ``atomicCompSwap`` and ``atomicCompareExchange`` are strong compare-exchanges, while WGSL has only
  ``atomicCompareExchangeWeak``, which may fail spuriously (section 17.8.5). The call runs before its statement as
  a loop that retries while the exchange failed although the old value equals the comparand. A loop condition
  that holds one is tested at the top of the loop body instead.

The plan runs after :func:`~crosstl.translator.codegen.wgsl_workgroup_locals.hoist_workgroup_locals` has moved the
entry points' workgroup locals to module scope, so their atomic slots are module slots.

The copy keeps each slot's scalar type, so expression typing is unchanged; the generator wraps the scalar in
``atomic<...>`` where it prints the slot's declaration.

``WGSLAtomicLoweringError`` refuses what WGSL cannot express or this lowering cannot keep exact: an atomic call on a
local, a parameter, a vector component or a type other than a 32-bit integer; a value read of an aggregate that holds
atomic memory, and a structure with atomic members as a local, parameter or return type; atomic memory passed to an
``out`` or ``inout`` parameter or used with ``&``; an assignment or increment of atomic memory used as a value; a
compound assignment whose target repeats a call; and a compare-exchange in a conditionally evaluated operand, a
``for`` update or a ``do``-``while`` condition.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CaseNode,
    CastNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LambdaNode,
    LiteralNode,
    LoopNode,
    MemberAccessNode,
    NamedType,
    PointerType,
    PrimitiveType,
    ReturnNode,
    StatementNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..refusals import TranslationRefusal

ATOMIC_MODIFY_NAMES = frozenset(
    {
        "atomicAdd",
        "atomicSub",
        "atomicMin",
        "atomicMax",
        "atomicAnd",
        "atomicOr",
        "atomicXor",
        "atomicExchange",
    }
)
ATOMIC_COMPARE_NAMES = frozenset({"atomicCompSwap", "atomicCompareExchange"})
ATOMIC_NAMES = (
    ATOMIC_MODIFY_NAMES | ATOMIC_COMPARE_NAMES | {"atomicLoad", "atomicStore"}
)
ATOMIC_ARITY = {
    **{name: 2 for name in ATOMIC_MODIFY_NAMES},
    **{name: 3 for name in ATOMIC_COMPARE_NAMES},
    "atomicLoad": 1,
    "atomicStore": 2,
}
SLOT_ANNOTATION = "wgsl_atomic_slot"
COMPARE_PREFIX = "_crosstl_cas_"
WRITE_QUALIFIERS = frozenset({"out", "inout"})
INCREMENT_OPERATORS = {"++": "+", "--": "-"}
SHORT_CIRCUIT_OPERATORS = frozenset({"&&", "||"})
STRUCTURED_BUFFER_NAMES = frozenset(
    {"rwstructuredbuffer", "structuredbuffer", "globallycoherentrwstructuredbuffer"}
)
EXPLICIT_ATOMIC_SCALARS = {"atomic_int": "int", "atomic_uint": "uint"}
# Fields that hold one child under two names (see ``ASTNode.child_nodes``).
FIELD_ALIASES = {
    "target": "left",
    "value": "right",
    "object_expr": "object",
    "array_expr": "array",
    "index_expr": "index",
    "condition": "if_condition",
    "then_branch": "if_body",
    "else_branch": "else_body",
    "var_type": "vtype",
}


class WGSLAtomicLoweringError(TranslationRefusal):
    """An atomic access the WGSL target cannot express exactly."""

    code = "project.translate.wgsl-atomic-unsupported"

    def __init__(self, reason):
        super().__init__(f"WGSL target cannot lower this atomic access: {reason}")


@dataclass(frozen=True)
class AtomicSlot:
    """Atomic memory: module variable ``name`` (``owner`` None) or member ``name`` of structure ``owner``."""

    owner: str | None
    name: str
    depth: int
    scalar: object = field(compare=False)

    @property
    def key(self):
        return (self.owner, self.name)

    @property
    def label(self):
        return f"{self.owner}.{self.name}" if self.owner else self.name


class WGSLCompareExchangeLoop(StatementNode):
    """A strong compare-exchange as a retry loop of ``atomicCompareExchangeWeak``; see the module docstring."""

    def __init__(self, target, compare, value, slot, name, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.compare = compare
        self.value = value
        self.slot = slot
        self.name = name


def _set(node, **fields):
    for name, value in fields.items():
        setattr(node, name, value)
        alias = FIELD_ALIASES.get(name)
        if alias is not None and alias in vars(node):
            setattr(node, alias, value)


def callee_name(call):
    function = getattr(call, "function", None)
    if isinstance(function, str):
        return function
    if isinstance(function, IdentifierNode):
        return function.name
    return None


def _qualifiers(node):
    return {str(q).lower() for q in getattr(node, "qualifiers", None) or []}


def _type_name(vtype):
    if isinstance(vtype, str):
        return vtype.strip()
    return getattr(vtype, "name", None)


def element_type(vtype):
    """The element of an indexable declaration type, or None."""
    if isinstance(vtype, ArrayType):
        return vtype.element_type
    if isinstance(vtype, PointerType):
        return vtype.pointee_type
    if (
        isinstance(vtype, NamedType)
        and str(vtype.name).lower() in STRUCTURED_BUFFER_NAMES
        and len(vtype.generic_args) == 1
    ):
        return vtype.generic_args[0]
    return None


def explicit_atomic_scalar(vtype):
    """``T`` for a CrossGL ``atomic<T>``, ``atomic_int`` or ``atomic_uint`` type, else None."""
    if (
        isinstance(vtype, NamedType)
        and vtype.name == "atomic"
        and len(vtype.generic_args) == 1
    ):
        return vtype.generic_args[0]
    scalar = EXPLICIT_ATOMIC_SCALARS.get(_type_name(vtype) or "")
    return PrimitiveType(scalar) if scalar else None


def _layers(vtype):
    """(subscript count, innermost type) of a declaration type."""
    depth = 0
    while element_type(vtype) is not None:
        vtype = element_type(vtype)
        depth += 1
    return depth, vtype


def with_innermost(vtype, inner):
    """A copy of ``vtype`` whose innermost element is ``inner``."""
    if element_type(vtype) is None:
        return inner
    result = copy.copy(vtype)
    nested = with_innermost(element_type(vtype), inner)
    if isinstance(vtype, ArrayType):
        result.element_type = nested
    elif isinstance(vtype, PointerType):
        result.pointee_type = nested
    else:
        result.generic_args = [nested]
    return result


def _body_statements(body):
    if body is None:
        return []
    if isinstance(body, BlockNode):
        return body.statements
    if isinstance(body, list):
        return body
    return [body]


def _walk(node):
    """``node`` and its descendants once each, without entering lambdas."""
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        if not isinstance(current, ASTNode) or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if not isinstance(current, LambdaNode):
            stack.extend(current.child_nodes())


def _repeatable(expr):
    """Whether evaluating ``expr`` twice is the same as once: no call, assignment or increment."""
    for node in _walk(expr):
        if isinstance(node, (FunctionCallNode, AssignmentNode)):
            return False
        if isinstance(node, UnaryOpNode) and node.operator in INCREMENT_OPERATORS:
            return False
    return True


class _Program:
    """The shader's declarations and the atomic slots."""

    def __init__(self, ast, scalar_name, is_stage_resource):
        self.ast = ast
        self.scalar_name = scalar_name
        self.structs = {}
        self.module_types = {}
        self.functions = []
        self.write_positions = {}
        self.slots = {}
        self.compare_count = 0
        self._holds = {}
        stages = list(getattr(ast, "stages", {}).values())
        for struct in [*(ast.structs or [])] + [
            s for stage in stages for s in getattr(stage, "local_structs", []) or []
        ]:
            if getattr(struct, "name", None):
                self.structs.setdefault(struct.name, struct)
        for variable in [*(ast.global_variables or [])] + [
            v for stage in stages for v in getattr(stage, "local_variables", []) or []
        ]:
            self.module_types.setdefault(variable.name, variable.var_type)
        owners = [(f, ast.global_variables) for f in ast.functions or []]
        for stage in stages:
            locals_ = stage.local_variables
            owners.extend((f, locals_) for f in stage.local_functions or [])
            if stage.entry_point is not None:
                owners.append((stage.entry_point, locals_))
        seen = set()
        for function, declarations in owners:
            if id(function) in seen or getattr(function, "body", None) is None:
                continue
            seen.add(id(function))
            resources = {
                id(p) for p in function.parameters or [] if is_stage_resource(p)
            }
            for parameter in function.parameters or []:
                if id(parameter) in resources:
                    self.module_types.setdefault(parameter.name, parameter.param_type)
            self.functions.append((function, resources, declarations))
            positions = self.write_positions.setdefault(function.name, set())
            for index, parameter in enumerate(function.parameters or []):
                if _qualifiers(parameter) & WRITE_QUALIFIERS:
                    positions.add(index)
        self.function_names = {function.name for function, _, _ in self.functions}

    def strip_explicit_atomics(self):
        """Record ``atomic<T>`` declarations as slots and give them their scalar type."""
        declarations = [*(self.ast.global_variables or [])]
        for stage in getattr(self.ast, "stages", {}).values():
            declarations.extend(stage.local_variables or [])
        for name, vtype in list(self.module_types.items()):
            depth, inner = _layers(vtype)
            scalar = explicit_atomic_scalar(inner)
            if scalar is None:
                continue
            self.add_slot(AtomicSlot(None, name, depth, scalar))
            stripped = with_innermost(vtype, scalar)
            self.module_types[name] = stripped
            for declaration in declarations:
                if declaration.name == name:
                    _set(declaration, var_type=stripped)
            for function, resources, _ in self.functions:
                for parameter in function.parameters or []:
                    if id(parameter) in resources and parameter.name == name:
                        parameter.param_type = stripped
        for struct in self.structs.values():
            for member in getattr(struct, "members", []) or []:
                depth, inner = _layers(member.member_type)
                scalar = explicit_atomic_scalar(inner)
                if scalar is not None:
                    self.add_slot(AtomicSlot(struct.name, member.name, depth, scalar))
                    member.member_type = with_innermost(member.member_type, scalar)

    def add_slot(self, slot):
        mapped = self.scalar_name(slot.scalar)
        if mapped not in ("i32", "u32"):
            raise WGSLAtomicLoweringError(
                f"{slot.label} is atomic {mapped or slot.scalar} memory; WGSL has atomic<i32> and "
                "atomic<u32> only (WGSL section 6.2.8)"
            )
        known = self.slots.get(slot.key)
        if known is not None and known.depth != slot.depth:
            raise WGSLAtomicLoweringError(
                f"{slot.label} is used as atomic memory at two subscript depths"
            )
        self.slots.setdefault(slot.key, slot)

    def struct_of(self, vtype):
        name = _type_name(vtype)
        if isinstance(vtype, NamedType) and vtype.generic_args:
            return None
        return self.structs.get(name) if name else None

    def member_type(self, struct, name):
        for member in getattr(struct, "members", []) or []:
            if member.name == name:
                return member.member_type
        return None

    def holds_atomics(self, vtype, visiting=None):
        """Whether a value of ``vtype`` contains atomic memory (a structure with atomic members, or arrays of one)."""
        _, inner = _layers(vtype)
        if explicit_atomic_scalar(inner) is not None:
            return True
        struct = self.struct_of(inner)
        if struct is None:
            return False
        if struct.name in self._holds:
            return self._holds[struct.name]
        visiting = visiting or set()
        if struct.name in visiting:
            return False
        visiting.add(struct.name)
        result = any(
            (struct.name, member.name) in self.slots
            or self.holds_atomics(member.member_type, visiting)
            for member in getattr(struct, "members", []) or []
        )
        self._holds[struct.name] = result
        return result

    def atomic_struct_names(self):
        return {
            name for name in sorted(self.structs) if self.holds_atomics(NamedType(name))
        }

    def resolve(self, expr, lookup):
        """(slot key, declared type, subscripts) of an l-value rooted at module memory, else None."""
        subscripts, node = 0, expr
        while isinstance(node, ArrayAccessNode):
            subscripts, node = subscripts + 1, node.array_expr
        if isinstance(node, IdentifierNode):
            name = self.module_name(node.name, lookup)
            if name is None:
                return None
            return (None, name), self.module_types[name], subscripts
        if isinstance(node, MemberAccessNode) and self.module_root(node, lookup):
            struct = self.struct_of(self.type_of(node.object_expr, lookup))
            member_type = self.member_type(struct, node.member) if struct else None
            if member_type is None:
                return None
            return (struct.name, node.member), member_type, subscripts
        return None

    def module_name(self, name, lookup):
        if lookup(name) is not None:
            return None
        return name if name in self.module_types else None

    def module_root(self, expr, lookup):
        while isinstance(expr, (ArrayAccessNode, MemberAccessNode)):
            expr = (
                expr.array_expr
                if isinstance(expr, ArrayAccessNode)
                else expr.object_expr
            )
        return (
            isinstance(expr, IdentifierNode)
            and self.module_name(expr.name, lookup) is not None
        )

    def type_of(self, expr, lookup):
        if isinstance(expr, IdentifierNode):
            entry = lookup(expr.name)
            if entry is not None and entry[0] == "local":
                return entry[1]
            name = self.module_name(expr.name, lookup)
            return self.module_types.get(name) if name else None
        if isinstance(expr, MemberAccessNode):
            struct = self.struct_of(self.type_of(expr.object_expr, lookup))
            return self.member_type(struct, expr.member) if struct else None
        if isinstance(expr, ArrayAccessNode):
            array_type = self.type_of(expr.array_expr, lookup)
            return element_type(array_type) if array_type is not None else None
        return None

    def scalar_slot(self, expr, lookup):
        """The slot an atomic call on ``expr`` addresses, or None when ``expr`` is not a module integer."""
        resolved = self.resolve(expr, lookup)
        if resolved is None:
            return None
        (owner, name), vtype, subscripts = resolved
        for _ in range(subscripts):
            vtype = element_type(vtype)
            if vtype is None:
                return None
        if not isinstance(vtype, PrimitiveType):
            return None
        return AtomicSlot(owner, name, subscripts, vtype)

    def collect_slots(self):
        def lookup(name):
            return None

        for function, _, _ in self.functions:
            for node in _walk(function.body):
                if (
                    isinstance(node, FunctionCallNode)
                    and callee_name(node) in ATOMIC_NAMES
                    and callee_name(node) not in self.function_names
                    and node.arguments
                ):
                    slot = self.scalar_slot(node.arguments[0], lookup)
                    if slot is not None:
                        self.add_slot(slot)


class _Rewriter:
    """Rewrites function bodies for the slots of a ``_Program``; see the module docstring."""

    def __init__(self, program):
        self.program = program
        self.scopes = []

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def declare(self, name, vtype):
        self.scopes[-1][name] = ("local", vtype)

    def refuse_local_type(self, vtype, what):
        if vtype is not None and self.program.holds_atomics(vtype):
            raise WGSLAtomicLoweringError(
                f"{what} has a type that holds atomic memory; WGSL allows atomic types only in "
                "storage and workgroup variables"
            )

    def function(self, function, resources):
        self.scopes = [{}]
        for parameter in function.parameters or []:
            if id(parameter) not in resources:
                self.refuse_local_type(
                    parameter.param_type, f"parameter {parameter.name}"
                )
                self.declare(parameter.name, parameter.param_type)
        self.refuse_local_type(function.return_type, f"the result of {function.name}")
        function.body = self.body(function.body)

    def body(self, body):
        if body is None:
            return None
        self.scopes.append({})
        try:
            if isinstance(body, BlockNode):
                body.statements = self.statements(body.statements)
                return body
            if isinstance(body, list):
                return self.statements(body)
            items = self.statement(body)
            return items[0] if len(items) == 1 else BlockNode(items)
        finally:
            self.scopes.pop()

    def statements(self, statements):
        result = []
        for statement in statements:
            result.extend(self.statement(statement))
        return result

    def statement(self, stmt):
        prelude = []
        result = self.statement_parts(stmt, prelude)
        return [*prelude, *result]

    def statement_parts(self, stmt, prelude):
        if isinstance(stmt, BlockNode):
            return [self.body(stmt)]
        if isinstance(stmt, VariableNode):
            return self.declaration(stmt, prelude)
        if isinstance(stmt, ExpressionStatementNode):
            stored = self.store(stmt.expression, prelude)
            stmt.expression = stored or self.expr(stmt.expression, prelude)
            if isinstance(
                stmt.expression, IdentifierNode
            ) and stmt.expression.name.startswith(COMPARE_PREFIX):
                return (
                    []
                )  # a compare-exchange whose result is unused: the loop is the statement
            return [stmt]
        if isinstance(stmt, AssignmentNode) or (
            isinstance(stmt, UnaryOpNode) and stmt.operator in INCREMENT_OPERATORS
        ):
            stored = self.store(stmt, prelude)
            return (
                [ExpressionStatementNode(stored)]
                if stored
                else [self.expr(stmt, prelude)]
            )
        if isinstance(stmt, ReturnNode):
            stmt.value = self.expr(stmt.value, prelude)
            return [stmt]
        if isinstance(stmt, IfNode):
            condition = self.expr(stmt.condition, prelude)
            _set(
                stmt,
                condition=condition,
                then_branch=self.body(stmt.then_branch),
                else_branch=self.body(stmt.else_branch),
            )
            return [stmt]
        if isinstance(stmt, WhileNode):
            guard = []
            condition = self.expr(stmt.condition, guard)
            body = self.body(stmt.body)
            if not guard:
                _set(stmt, condition=condition, body=body)
                return [stmt]
            return [
                LoopNode(
                    BlockNode(
                        [*self.loop_guard(guard, condition), *self.nonempty(body)]
                    )
                )
            ]
        if isinstance(stmt, ForNode):
            return [self.for_statement(stmt, prelude)]
        if isinstance(stmt, DoWhileNode):
            _set(
                stmt,
                body=self.body(stmt.body),
                condition=self.expr(stmt.condition, None),
            )
            return [stmt]
        if isinstance(stmt, LoopNode):
            stmt.body = self.body(stmt.body)
            return [stmt]
        if isinstance(stmt, SwitchNode):
            stmt.expression = self.expr(stmt.expression, prelude)
            default = stmt.default_case
            for case in [*(stmt.cases or []), default]:
                if isinstance(case, CaseNode):
                    self.scopes.append({})
                    case.statements = self.statements(case.statements or [])
                    self.scopes.pop()
            if default is not None and not isinstance(default, CaseNode):
                stmt.default_case = self.body(default)
            return [stmt]
        return [self.generic(stmt, lambda child: self.expr(child, None))]

    def declaration(self, stmt, prelude):
        # A local whose type holds atomic memory is refused where the generator prints it
        # (``refuse_atomic_local``), after any more specific initializer diagnostic.
        stmt.initial_value = self.expr(stmt.initial_value, prelude)
        self.declare(stmt.name, stmt.var_type)
        return [stmt]

    @staticmethod
    def nonempty(body):
        """``[body]``, or nothing for an empty block."""
        if isinstance(body, BlockNode) and not body.statements:
            return []
        return [body] if body is not None else []

    @staticmethod
    def loop_guard(guard, condition):
        return [*guard, IfNode(UnaryOpNode("!", condition), BlockNode([BreakNode()]))]

    def for_statement(self, stmt, prelude):
        self.scopes.append({})
        try:
            init = stmt.init
            if isinstance(init, list):
                init = [self.for_initializer(item, prelude) for item in init]
            elif init is not None:
                init = self.for_initializer(init, prelude)
            guard = []
            condition = self.expr(stmt.condition, guard)
            update = stmt.update
            if isinstance(update, list):
                update = [self.expr(item, None) for item in update]
            else:
                update = self.expr(update, None)
            body = self.body(stmt.body)
            if guard:
                body = BlockNode(
                    [*self.loop_guard(guard, condition), *self.nonempty(body)]
                )
                condition = None
            _set(stmt, init=init, condition=condition, update=update, body=body)
            return stmt
        finally:
            self.scopes.pop()

    def for_initializer(self, init, prelude):
        if isinstance(init, VariableNode):
            parts = self.declaration(init, prelude)
            if len(parts) != 1:
                raise WGSLAtomicLoweringError(
                    f"workgroup variable {init.name} is declared in a for-loop header"
                )
            return parts[0]
        if isinstance(init, ExpressionStatementNode):
            init.expression = self.expr(init.expression, prelude)
            return init
        return self.expr(init, prelude)

    def generic(self, node, visit):
        """Apply ``visit`` to the direct children of ``node``, keeping aliased fields together."""
        replaced = {}
        for name, value in list(vars(node).items()):
            if name in ("annotations", "parent", "source_location"):
                continue
            if isinstance(value, ASTNode):
                if id(value) not in replaced:
                    replaced[id(value)] = visit(value)
                setattr(node, name, replaced[id(value)])
            elif isinstance(value, list) and id(value) not in replaced:
                replaced[id(value)] = value
                value[:] = [
                    visit(item) if isinstance(item, ASTNode) else item for item in value
                ]
        return node

    def slot_at(self, expr):
        """The slot ``expr`` names at full depth; raises when it names an aggregate holding atomic memory."""
        resolved = self.program.resolve(expr, self.lookup)
        if resolved is not None:
            slot = self.program.slots.get(resolved[0])
            if slot is not None:
                if resolved[2] == slot.depth:
                    return slot
                if resolved[2] < slot.depth:
                    self.refuse_aggregate(slot.label)
        vtype = self.program.type_of(expr, self.lookup)
        if vtype is not None and self.program.module_root(expr, self.lookup):
            if self.program.holds_atomics(vtype):
                self.refuse_aggregate(_type_name(_layers(vtype)[1]))
        return None

    @staticmethod
    def refuse_aggregate(label):
        raise WGSLAtomicLoweringError(
            f"a value of {label} holds atomic memory; WGSL cannot load, store or copy an atomic "
            "aggregate as a whole"
        )

    def location(self, expr, prelude, conditional=False):
        """``expr`` as an l-value: only its subscripts are rewritten as values."""
        if isinstance(expr, ArrayAccessNode):
            _set(
                expr,
                array_expr=self.location(expr.array_expr, prelude, conditional),
                index_expr=self.expr(expr.index_expr, prelude, conditional),
            )
            return expr
        if isinstance(expr, MemberAccessNode):
            _set(
                expr, object_expr=self.location(expr.object_expr, prelude, conditional)
            )
            return expr
        if isinstance(expr, IdentifierNode):
            return expr
        return self.expr(expr, prelude, conditional)

    @staticmethod
    def call(name, arguments, slot):
        call = FunctionCallNode(IdentifierNode(name), arguments)
        call.add_annotation(SLOT_ANNOTATION, slot)
        return call

    def load(self, target, slot):
        return self.call("atomicLoad", [target], slot)

    def store(self, node, prelude):
        """``atomicStore`` for an assignment or increment statement whose target is a slot, else None."""
        if isinstance(node, AssignmentNode):
            target_expr, operator = node.target, node.operator
        elif isinstance(node, UnaryOpNode) and node.operator in INCREMENT_OPERATORS:
            target_expr, operator = (
                node.operand,
                INCREMENT_OPERATORS[node.operator] + "=",
            )
        else:
            return None
        slot = self.slot_at(target_expr)
        if slot is None:
            return None
        if operator != "=" and not _repeatable(target_expr):
            raise WGSLAtomicLoweringError(
                f"a compound assignment to atomic memory {slot.label} would evaluate its subscripts twice"
            )
        target = self.location(target_expr, prelude)
        if isinstance(node, AssignmentNode):
            value = self.expr(node.value, prelude)
        else:
            value = LiteralNode(1, copy.copy(slot.scalar))
        if operator != "=":
            value = BinaryOpNode(
                self.load(copy.deepcopy(target), slot), operator[:-1], value
            )
        return self.call("atomicStore", [target, value], slot)

    def expr(self, expr, prelude, conditional=False):
        if not isinstance(expr, ASTNode) or isinstance(expr, LambdaNode):
            return expr
        if isinstance(expr, FunctionCallNode):
            return self.call_expression(expr, prelude, conditional)
        if isinstance(expr, (IdentifierNode, MemberAccessNode, ArrayAccessNode)):
            slot = self.slot_at(expr)
            target = self.location(expr, prelude, conditional)
            return self.load(target, slot) if slot is not None else target
        if isinstance(expr, AssignmentNode):
            if self.slot_at(expr.target) is not None:
                raise WGSLAtomicLoweringError(
                    "an assignment to atomic memory is used as a value or in a for-loop header"
                )
            _set(
                expr,
                target=self.location(expr.target, prelude, conditional),
                value=self.expr(expr.value, prelude, conditional),
            )
            return expr
        if isinstance(expr, UnaryOpNode):
            if expr.operator in INCREMENT_OPERATORS or expr.operator == "&":
                if self.slot_at(expr.operand) is not None:
                    raise WGSLAtomicLoweringError(
                        f"atomic memory is used with {expr.operator} as a value; WGSL reaches atomic "
                        "memory only through atomic built-ins"
                    )
                expr.operand = self.location(expr.operand, prelude, conditional)
                return expr
            expr.operand = self.expr(expr.operand, prelude, conditional)
            return expr
        if isinstance(expr, BinaryOpNode) and expr.operator in SHORT_CIRCUIT_OPERATORS:
            _set(
                expr,
                left=self.expr(expr.left, prelude, conditional),
                right=self.expr(expr.right, prelude, True),
            )
            return expr
        if isinstance(expr, TernaryOpNode):
            _set(
                expr,
                condition=self.expr(expr.condition, prelude, conditional),
                true_expr=self.expr(expr.true_expr, prelude, True),
                false_expr=self.expr(expr.false_expr, prelude, True),
            )
            return expr
        if isinstance(expr, CastNode):
            expr.expression = self.expr(expr.expression, prelude, conditional)
            return expr
        return self.generic(expr, lambda child: self.expr(child, prelude, conditional))

    def call_expression(self, call, prelude, conditional):
        name = callee_name(call)
        arguments = call.arguments
        if name in ATOMIC_NAMES and name not in self.program.function_names:
            return self.atomic_call(call, name, prelude, conditional)
        helper = self.buffer_helper(call, name, prelude, conditional)
        if helper is not None:
            return helper
        if isinstance(call.function, MemberAccessNode):
            _set(
                call.function,
                object_expr=self.location(
                    call.function.object_expr, prelude, conditional
                ),
            )
        if name in ("buffer_dimensions", "arrayLength") and len(arguments) == 1:
            arguments[0] = self.location(arguments[0], prelude, conditional)
            return call
        writes = self.program.write_positions.get(name, set())
        for index, argument in enumerate(arguments):
            if index in writes:
                if self.slot_at(argument) is not None:
                    raise WGSLAtomicLoweringError(
                        f"atomic memory is passed to an out or inout parameter of {name}"
                    )
                arguments[index] = self.location(argument, prelude, conditional)
            else:
                arguments[index] = self.expr(argument, prelude, conditional)
        return call

    def aggregate_slot(self, expr):
        """The slot below ``expr`` when ``expr`` names its array without every subscript, else None."""
        resolved = self.program.resolve(expr, self.lookup)
        slot = self.program.slots.get(resolved[0]) if resolved else None
        return slot if slot is not None and resolved[2] < slot.depth else None

    def buffer_helper(self, call, name, prelude, conditional):
        """A structured-buffer load or store of atomic memory as ``atomicLoad`` or ``atomicStore``."""
        arguments = call.arguments
        method = None
        if isinstance(call.function, MemberAccessNode):
            method = str(call.function.member).lower()
            receiver = call.function.object_expr
            if self.aggregate_slot(receiver) is None:
                return None
            if method == "load" and len(arguments) == 1:
                arguments = [receiver, arguments[0]]
                name = "buffer_load"
            elif method in ("length", "getdimensions"):
                return None
            else:
                raise WGSLAtomicLoweringError(
                    f"method {call.function.member} of a buffer of atomic memory has no WGSL lowering"
                )
        if name not in ("buffer_load", "buffer_store") or len(arguments) < 2:
            return None
        element = ArrayAccessNode(arguments[0], arguments[1])
        slot = self.slot_at(element)
        if slot is None:
            return None
        if name == "buffer_load" and len(arguments) == 2:
            return self.load(self.location(element, prelude, conditional), slot)
        if name == "buffer_store" and len(arguments) == 3:
            target = self.location(element, prelude, conditional)
            value = self.expr(arguments[2], prelude, conditional)
            return self.call("atomicStore", [target, value], slot)
        raise WGSLAtomicLoweringError(
            f"{name} of atomic memory takes {2 if name == 'buffer_load' else 3} arguments"
        )

    def atomic_call(self, call, name, prelude, conditional):
        arguments = call.arguments
        if len(arguments) != ATOMIC_ARITY[name]:
            raise WGSLAtomicLoweringError(
                f"{name} takes {ATOMIC_ARITY[name]} arguments in WGSL; got {len(arguments)}"
            )
        slot = self.slot_at(arguments[0])
        if slot is None:
            raise WGSLAtomicLoweringError(
                f"{name} needs an integer in a storage buffer or workgroup variable as its first "
                "argument (WGSL section 6.2.8)"
            )
        arguments[0] = self.location(arguments[0], prelude, conditional)
        for index in range(1, len(arguments)):
            arguments[index] = self.expr(arguments[index], prelude, conditional)
        if name not in ATOMIC_COMPARE_NAMES:
            call.add_annotation(SLOT_ANNOTATION, slot)
            return call
        if prelude is None or conditional:
            raise WGSLAtomicLoweringError(
                f"{name} is a strong compare-exchange, which WGSL expresses as a retry loop before its "
                "statement; a conditionally evaluated operand, a for-loop update or a do-while "
                "condition has no such place"
            )
        program = self.program
        result = f"{COMPARE_PREFIX}{program.compare_count}"
        program.compare_count += 1
        prelude.append(WGSLCompareExchangeLoop(*arguments, slot, result))
        old_value = VariableNode(
            f"{result}_old",
            copy.copy(slot.scalar),
            MemberAccessNode(IdentifierNode(result), "old_value"),
            is_mutable=False,
        )
        prelude.append(old_value)
        self.declare(old_value.name, slot.scalar)
        return IdentifierNode(old_value.name)


def _needs_plan(ast):
    for node in _walk(ast):
        if isinstance(node, FunctionCallNode) and callee_name(node) in ATOMIC_NAMES:
            return True
        for vtype in (
            getattr(node, "var_type", None),
            getattr(node, "member_type", None),
        ):
            if (
                vtype is not None
                and explicit_atomic_scalar(_layers(vtype)[1]) is not None
            ):
                return True
    return False


def plan_wgsl_buffer_atomics(ast, scalar_name, is_stage_resource):
    """(AST, slots by key, names of structures holding atomics); the AST is a rewritten copy when needed."""
    if not _needs_plan(ast):
        return ast, {}, set()
    ast = copy.deepcopy(ast)
    program = _Program(ast, scalar_name, is_stage_resource)
    program.strip_explicit_atomics()
    program.collect_slots()
    rewriter = _Rewriter(program)
    for function, resources, _ in program.functions:
        rewriter.function(function, resources)
    return ast, dict(program.slots), program.atomic_struct_names()


class WGSLBufferAtomicsMixin:
    """WGSL generator side of the atomic lowering: declarations, calls and compare-exchange loops."""

    def plan_buffer_atomics(self, ast):
        ast, self._atomic_slots, self._atomic_struct_names = plan_wgsl_buffer_atomics(
            ast,
            lambda scalar: self.type_name_string(scalar),
            self.is_stage_resource_parameter,
        )
        return ast

    def atomic_type_name(self, vtype):
        """``atomic<i32>`` or ``atomic<u32>`` for a CrossGL ``atomic<T>`` type, else None."""
        scalar = explicit_atomic_scalar(vtype)
        if scalar is None:
            return None
        mapped = self.type_name_string(scalar)
        if mapped not in ("i32", "u32"):
            raise WGSLAtomicLoweringError(
                f"atomic<{mapped}> is not a WGSL type; WGSL has atomic<i32> and atomic<u32> only "
                "(WGSL section 6.2.8)"
            )
        return f"atomic<{mapped}>"

    def atomic_declaration_type(self, owner, name, vtype):
        """``vtype`` with its scalar wrapped in ``atomic<...>`` when it declares atomic memory."""
        slot = getattr(self, "_atomic_slots", {}).get((owner, name))
        if slot is None:
            return vtype
        return with_innermost(vtype, NamedType("atomic", [copy.copy(slot.scalar)]))

    def declares_atomic_memory(self, owner, name, vtype):
        if (owner, name) in getattr(self, "_atomic_slots", {}):
            return True
        struct_name = _type_name(_layers(vtype)[1])
        return struct_name in getattr(self, "_atomic_struct_names", ())

    def require_atomic_address_space(
        self, owner, name, vtype, address_space, access=""
    ):
        """Refuse atomic memory outside read-write storage and workgroup memory (WGSL section 6.2.8)."""
        if not self.declares_atomic_memory(owner, name, vtype):
            return
        access = access.strip(", ")
        if address_space == "workgroup" or (
            address_space == "storage" and access == "read_write"
        ):
            return
        place = f"{address_space} {access}".strip()
        label = f"{owner}.{name}" if owner else name
        if (owner, name) not in getattr(self, "_atomic_slots", {}):
            struct_name = _type_name(_layers(vtype)[1])
            label = f"{label}, of structure {struct_name} with atomic members,"
        raise WGSLAtomicLoweringError(
            f"{label} would hold atomic memory in the {place} address space; WGSL allows atomic "
            "types only in read_write storage and workgroup memory (WGSL section 6.2.8)"
        )

    def refuse_atomic_local(self, stmt):
        """Refuse a function variable whose type holds atomic memory (WGSL section 6.2.8)."""
        inner = _layers(stmt.var_type)[1]
        if explicit_atomic_scalar(inner) is None and _type_name(inner) not in getattr(
            self, "_atomic_struct_names", ()
        ):
            return
        raise WGSLAtomicLoweringError(
            f"local variable {stmt.name} has a type that holds atomic memory; WGSL allows atomic "
            "types only in storage and workgroup variables"
        )

    def buffer_atomic_call_type(self, node):
        slot = node.get_annotation(SLOT_ANNOTATION)
        if slot is None or callee_name(node) == "atomicStore":
            return None
        return slot.scalar

    def generate_buffer_atomic_call(self, node, function_name):
        slot = node.get_annotation(SLOT_ANNOTATION)
        if slot is None:
            raise WGSLAtomicLoweringError(
                f"{function_name} is in a position the WGSL atomic lowering does not reach"
            )
        target = self.generate_expression(node.arguments[0])
        values = [
            self.generate_expression_for_target(argument, slot.scalar)
            for argument in node.arguments[1:]
        ]
        return f"{function_name}({', '.join([f'&{target}', *values])})"

    def generate_compare_exchange_loop(self, stmt, indent):
        pad, inner = "    " * indent, "    " * (indent + 1)
        scalar = self.type_name_string(stmt.slot.scalar)
        name = stmt.name
        target = self.generate_expression(stmt.target)
        compare = self.generate_expression_for_target(stmt.compare, stmt.slot.scalar)
        value = self.generate_expression_for_target(stmt.value, stmt.slot.scalar)
        exchange = (
            f"atomicCompareExchangeWeak({name}_ptr, {name}_compare, {name}_value)"
        )
        return "\n".join(
            [
                f"{pad}let {name}_ptr = &{target};",
                f"{pad}let {name}_compare: {scalar} = {compare};",
                f"{pad}let {name}_value: {scalar} = {value};",
                f"{pad}var {name} = {exchange};",
                f"{pad}loop {{",
                f"{inner}if ({name}.exchanged || {name}.old_value != {name}_compare) {{",
                f"{inner}    break;",
                f"{inner}}}",
                f"{inner}{name} = {exchange};",
                f"{pad}}}",
            ]
        )
