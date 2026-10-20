"""Uniformity facts for locals that are reassigned outside loops.

Software subgroup lowering proves that loop bounds and helper arguments are
uniform across a subgroup. A lexical proof can only use locals that are never
written. Some kernels instead clamp a bound once, as in
``int limit = n; if (c) { int q = (group + 1) * 4; limit = min(n, q); }``.
Every write to such a local is a plain ``name = value;`` reached outside any
loop, so its fact at each point follows the writes before it: the value is
uniform after a write of a uniform value, and after a uniform branch when it
is uniform at the end of every branch.

``followed_locals`` selects these locals. ``ReassignmentFlow`` updates a set of
uniform names after each statement, given the backend's uniform-expression
test.

Immutable storage parameters (constant storage, or read-only storage named by
a checked immutable-storage contract) may also move their view in place, as
``mask += tid.z * strides[0];``. ``followed_offsets`` selects those whose only
writes are such offsets outside loops. The view stays uniform after an offset
by a uniform amount, and the flow follows it like a reassigned local.

A followed local may also be a record written member by member, as
``view.offset += stride;``. Its other members keep their values, so the
record stays uniform after a write of a uniform value while it was uniform.
"""

from ..ast import (
    AssignmentNode,
    BlockNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .subgroup_loop_carried import (
    carried_fixpoint,
    follow_carried_write,
    leaves_trip_early,
    loop_carried_names,
)

_LOOPS = (ForNode, WhileNode, DoWhileNode, LoopNode, ForInNode)
_STEPS = {"++", "--", "PRE_INCREMENT", "POST_INCREMENT", "PRE_DECREMENT"}
_STEPS |= {"POST_DECREMENT"}
_UNARY_WRITES = _STEPS | {"&", "ADDRESS_OF"}
_MEMBER_WRITES = frozenset(
    {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
)


def statement_list(body):
    if hasattr(body, "statements"):
        return list(body.statements or [])
    if isinstance(body, list):
        return list(body)
    if body is None:
        return []
    return [body]


def _expression(statement):
    if isinstance(statement, ExpressionStatementNode):
        return getattr(statement, "expression", None)
    return statement


_OFFSETS = {"+=", "-="}


def _offset_write(node, map_operator):
    """The parameter that ``name += value;`` or ``name -= value;`` moves."""
    target = getattr(node, "target", getattr(node, "left", None))
    operator = getattr(node, "operator", getattr(node, "op", "="))
    if isinstance(target, IdentifierNode) and (
        operator in _OFFSETS or map_operator(operator) in _OFFSETS
    ):
        return target.name
    return None


def _outside_loop_writes(body, accept):
    """Ids of top-level assignment statements outside loops that ``accept``."""
    accepted = set()

    def visit(statements, in_loop):
        for statement in statement_list(statements):
            node = _expression(statement)
            if isinstance(node, AssignmentNode):
                if not in_loop and accept(node):
                    accepted.add(id(node))
            elif isinstance(node, IfNode):
                visit(getattr(node, "then_branch", None), in_loop)
                for branch in getattr(node, "else_if_bodies", []) or []:
                    visit(branch, in_loop)
                visit(getattr(node, "else_branch", None), in_loop)
            elif isinstance(node, _LOOPS):
                visit(getattr(node, "body", None), True)
            elif isinstance(node, BlockNode):
                visit(node, in_loop)

    visit(body, False)
    return accepted


def followed_offsets(
    function,
    resources,
    call_mutations,
    *,
    walk,
    map_operator,
    identifier_names,
    write_roots,
):
    """Return the immutable storage parameters whose writes only offset them.

    ``resources`` names entry parameters whose storage nothing writes during
    the dispatch. Each selected parameter is written only by ``name += value;``
    or ``name -= value;`` statements outside loops. Any other write, a local of
    the same name, address-taking, a reference or pointer view and a call that
    may write it disqualify the parameter.
    """
    body = getattr(function, "body", None)
    declared = {
        node.name
        for node in walk(body)
        if isinstance(node, VariableNode) and getattr(node, "name", None)
    }
    candidates = set(resources) - declared
    if not candidates:
        return set()
    accepted = _outside_loop_writes(
        body, lambda node: _offset_write(node, map_operator) in candidates
    )
    for node in walk(body):
        if isinstance(node, AssignmentNode):
            if id(node) not in accepted:
                target = getattr(node, "target", getattr(node, "left", None))
                candidates.difference_update(write_roots(target))
        elif isinstance(node, UnaryOpNode):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator in _UNARY_WRITES or map_operator(operator) in _UNARY_WRITES:
                candidates.difference_update(identifier_names(node.operand))
        elif isinstance(node, FunctionCallNode):
            candidates.difference_update(call_mutations.get(id(node), set()))
        elif isinstance(node, VariableNode) and isinstance(
            getattr(node, "var_type", None), (PointerType, ReferenceType)
        ):
            candidates.difference_update(identifier_names(node.initial_value))
    return candidates


def follow_offset(node, offsets, names, uniform_value, map_operator):
    """Whether ``node`` keeps an offset parameter uniform; ``None`` if no offset.

    An offset by a uniform amount keeps a uniform view uniform.
    """
    name = (
        _offset_write(node, map_operator) if isinstance(node, AssignmentNode) else None
    )
    if name is None or name not in offsets:
        return None
    value = getattr(node, "value", getattr(node, "right", None))
    return name in names and uniform_value(value)


def member_write_root(node, map_operator):
    """The local that ``local.member = value;`` or ``local.member op= value;`` writes.

    Only ``.`` member chains rooted at an identifier qualify. Writes through
    a pointer, to an element or to a swizzle are not member writes.
    """
    if not isinstance(node, AssignmentNode):
        return None
    target = getattr(node, "target", getattr(node, "left", None))
    if not isinstance(target, MemberAccessNode):
        return None
    while isinstance(target, MemberAccessNode):
        target = target.object_expr
    operator = getattr(node, "operator", getattr(node, "op", "="))
    if isinstance(target, IdentifierNode) and (
        operator in _MEMBER_WRITES or map_operator(operator) in _MEMBER_WRITES
    ):
        return target.name
    return None


def follow_member_write(node, followed, names, uniform_value, map_operator):
    """Whether a member write keeps a followed record uniform; ``None`` if none.

    The record stays uniform when it was uniform and the value is uniform.
    """
    name = member_write_root(node, map_operator)
    if name is None or name not in followed:
        return None
    value = getattr(node, "value", getattr(node, "right", None))
    return name in names and uniform_value(value)


def _plain_write(node, map_operator):
    target = getattr(node, "target", getattr(node, "left", None))
    operator = getattr(node, "operator", getattr(node, "op", "="))
    if isinstance(target, IdentifierNode) and map_operator(operator) == "=":
        return target.name
    return None


def followed_locals(
    function,
    mutable_names,
    call_mutations,
    *,
    walk,
    map_operator,
    identifier_names,
    write_roots,
    value_type,
    volatile,
):
    """Return the written locals whose writes a flow analysis can follow.

    Each is a value declared once, and every write to it is a plain
    ``name = value;`` or member ``name.member op= value;`` statement outside
    loops. A local declared without an initializer is not uniform until a
    write makes it so. Increments, compound writes of the whole value, element
    writes, address-taking, pointer or reference views and calls that may
    write it disqualify a name. ``walk`` yields every node below a
    root, ``identifier_names`` the names an expression reads, ``write_roots``
    the names an assignment target writes, ``value_type``
    whether a declared type holds a plain value and ``volatile`` whether a
    declaration or type is volatile.
    """
    body = getattr(function, "body", None)
    parameters = {
        parameter.name
        for parameter in getattr(function, "parameters", []) or []
        if getattr(parameter, "name", None)
    }
    counts = {}
    declarations = {}
    for node in walk(body):
        if isinstance(node, VariableNode) and getattr(node, "name", None):
            counts[node.name] = counts.get(node.name, 0) + 1
            declarations[node.name] = node
    candidates = {
        name
        for name, declaration in declarations.items()
        if counts[name] == 1
        and name not in parameters
        and name in mutable_names
        and value_type(declaration.var_type)
        and not volatile(declaration)
        and not volatile(declaration.var_type)
    }
    if not candidates:
        return set()
    followed = set()

    def visit(statements, in_loop):
        for statement in statement_list(statements):
            node = _expression(statement)
            if isinstance(node, AssignmentNode):
                if not in_loop and (
                    _plain_write(node, map_operator)
                    or member_write_root(node, map_operator)
                ):
                    followed.add(id(node))
            elif isinstance(node, IfNode):
                visit(getattr(node, "then_branch", None), in_loop)
                for branch in getattr(node, "else_if_bodies", []) or []:
                    visit(branch, in_loop)
                visit(getattr(node, "else_branch", None), in_loop)
            elif isinstance(node, _LOOPS):
                visit(getattr(node, "body", None), True)
            elif isinstance(node, BlockNode):
                visit(node, in_loop)

    visit(body, False)
    for node in walk(body):
        if isinstance(node, AssignmentNode):
            if id(node) not in followed:
                target = getattr(node, "target", getattr(node, "left", None))
                candidates.difference_update(write_roots(target))
        elif isinstance(node, UnaryOpNode):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator in _UNARY_WRITES or map_operator(operator) in _UNARY_WRITES:
                candidates.difference_update(identifier_names(node.operand))
        elif isinstance(node, FunctionCallNode):
            candidates.difference_update(call_mutations.get(id(node), set()))
        elif isinstance(node, VariableNode) and isinstance(
            getattr(node, "var_type", None), (PointerType, ReferenceType)
        ):
            candidates.difference_update(identifier_names(node.initial_value))
    return candidates


class ReassignmentFlow:
    """Update uniform-name facts across the writes of ``followed`` locals.

    ``uniform_expression(expression, names, components)`` is the backend's
    test, ``declares_uniform(declaration, names, components)`` decides a
    local declared inside a followed branch and ``constant_condition``
    optionally folds a guard to ``True`` or ``False``.

    ``carried`` names locals that loops advance on every trip (see
    ``subgroup_loop_carried``). ``trip_uniform(loop, names, components)``
    proves that every invocation runs the same trips of ``loop``; without it
    a loop carries nothing. ``counter_loop(loop, names, components)`` names
    the counter of an initializer-free counter loop that every invocation
    runs alike, as ``for (; k <= end; k += 16)``; that counter stays uniform
    after the loop.
    """

    def __init__(
        self,
        followed,
        *,
        walk,
        map_operator,
        uniform_expression,
        declares_uniform,
        constant_condition=None,
        offsets=(),
        carried=(),
        trip_uniform=None,
        counter_loop=None,
    ):
        self.followed = set(followed)
        self.offsets = set(offsets)
        self.carried = set(carried)
        self.trip_uniform = trip_uniform
        self.counter_loop = counter_loop
        self.walk = walk
        self.map_operator = map_operator
        self.uniform_expression = uniform_expression
        self.declares_uniform = declares_uniform
        self.constant_condition = constant_condition

    def written(self, root):
        names = set()
        for node in self.walk(root):
            if isinstance(node, AssignmentNode):
                target = getattr(node, "target", getattr(node, "left", None))
                if isinstance(target, IdentifierNode):
                    names.add(target.name)
                root = member_write_root(node, self.map_operator)
                if root is not None:
                    names.add(root)
            elif isinstance(node, UnaryOpNode) and isinstance(
                getattr(node, "operand", None), IdentifierNode
            ):
                operator = getattr(node, "operator", getattr(node, "op", None))
                if operator in _STEPS or self.map_operator(operator) in _STEPS:
                    names.add(node.operand.name)
        return names

    def carried_trip(self, loop, names, components):
        """Return the facts at each trip's start and the carried names kept.

        A loop whose trips may differ between invocations, or that may end a
        trip early, keeps none of the locals it carries.
        """
        carried = loop_carried_names(loop, self.carried, self.walk, self.map_operator)
        if not carried:
            return set(names), frozenset()
        if (
            not isinstance(loop, ForNode)
            or self.trip_uniform is None
            or leaves_trip_early(loop, self.walk)
            or not self.trip_uniform(loop, names, components)
        ):
            return set(names) - carried, frozenset()

        def follow_trip(trial):
            self.follow_branch(getattr(loop, "body", None), trial, components)
            update = getattr(loop, "update", None)
            if update is not None:
                self.follow(update, trial, components)

        kept = carried_fixpoint(carried, names, follow_trip)
        return (set(names) - carried) | kept, kept

    def follow(self, statement, names, components):
        """Update ``names`` for the followed writes ``statement`` performs."""
        written = (self.followed | self.offsets | self.carried) & self.written(
            statement
        )
        if not written:
            return
        node = _expression(statement)
        if (
            follow_carried_write(
                node,
                self.carried,
                names,
                lambda value: self.uniform_expression(value, names, components),
                self.map_operator,
            )
            is not None
        ):
            return
        counter = (
            self.counter_loop(node, names, components)
            if isinstance(node, ForNode) and self.counter_loop is not None
            else None
        )
        if isinstance(node, ForNode) and written & self.carried:
            # Every invocation ran the same trips, each keeping ``kept``.
            _trip, kept = self.carried_trip(node, names, components)
            names.difference_update(written - kept - {counter})
            return
        if counter is not None and written == {counter}:
            # Every invocation ran the uniform loop's trips.
            return
        kept = follow_offset(
            node,
            self.offsets,
            names,
            lambda value: self.uniform_expression(value, names, components),
            self.map_operator,
        )
        if kept is None:
            kept = follow_member_write(
                node,
                self.followed,
                names,
                lambda value: self.uniform_expression(value, names, components),
                self.map_operator,
            )
        if kept is not None:
            if not kept:
                names.difference_update(written)
            return
        if isinstance(node, AssignmentNode):
            name = _plain_write(node, self.map_operator)
            value = getattr(node, "value", getattr(node, "right", None))
            if name and self.uniform_expression(value, names, components):
                names.add(name)
            else:
                names.difference_update(written)
            return
        if isinstance(node, (BlockNode, list)):
            self.follow_branch(node, names, components)
            return
        if isinstance(node, IfNode) and not getattr(node, "else_if_conditions", None):
            fixed = (
                self.constant_condition(node.condition)
                if self.constant_condition is not None
                else None
            )
            if fixed is not None:
                # A constant guard selects one branch for every invocation.
                taken = node.then_branch if fixed else node.else_branch
                self.follow_branch(taken, names, components)
                return
            if self.uniform_expression(node.condition, names, components):
                # Every invocation takes the same branch, so a fact holds
                # afterwards when it holds at the end of both branches.
                # Both branches start from the same facts, so a local either
                # branch makes uniform stays so only when both do.
                results = []
                for branch in (node.then_branch, node.else_branch):
                    branch_names = set(names)
                    self.follow_branch(branch, branch_names, components)
                    results.append(branch_names)
                names.clear()
                names.update(set.intersection(*results))
                return
        names.difference_update(written)

    def follow_branch(self, statements, names, components):
        """Follow a branch every invocation takes, scoping its declarations.

        A local the branch declares from uniform values makes the writes
        computed from it uniform. Its fact ends with the branch, so an outer
        name it shadows keeps its own fact.
        """
        components = dict(components)
        outer = {}
        for child in statement_list(statements):
            if isinstance(child, VariableNode) and getattr(child, "name", None):
                outer.setdefault(child.name, child.name in names)
                names.discard(child.name)
                components.pop(child.name, None)
                if self.declares_uniform(child, names, components):
                    names.add(child.name)
                continue
            self.follow(child, names, components)
        for name, was_uniform in outer.items():
            if was_uniform:
                names.add(name)
            else:
                names.discard(name)
