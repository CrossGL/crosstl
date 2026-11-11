"""Lane-symmetric values and the workgroup-uniform collectives they feed.

Software subgroups split a workgroup of ``k * width`` invocations into ``k``
subgroups of ``width`` lanes, where an invocation's lane is its flat index
modulo ``width``. Values then have one of three levels:

* *uniform*: every invocation that holds the value holds the same one;
* *lane-symmetric*: invocations with the same lane index hold the same value.
  The lane index is lane-symmetric, and so are uniform values, loads of
  immutable storage at lane-symmetric positions and pure arithmetic on
  lane-symmetric values;
* *varying*: anything else.

When every subgroup is full and enters a function with the same lanes, every
subgroup runs the same lane-symmetric computation. A reduction or vote of a
lane-symmetric value then gives one result to the whole workgroup, as does a
broadcast from a constant in-range lane or from a uniform unsigned local that
an enclosing condition bounds below the width. A prefix scan, a relative
shuffle or a broadcast from a lane-symmetric lane of one stays
lane-symmetric. A loop that leaves, or a helper that returns, on such a
result therefore branches the same way in every invocation.

The analysis is flow-insensitive and starts optimistic, then lowers levels
to a fixpoint. A local has a level when every write to it, its initializer
included, has that level and runs under control of that level. A local
declared without a value holds only what its writes store, since reading an
indeterminate value is undefined in the source. A local passed alone to a
private reference parameter receives the level of the helper's writes to
it. Control has a level when every enclosing condition, and every exit that
may precede it, has that level. Collective results require full subgroups,
a function every subgroup enters with the same lanes and lane-symmetric
control at the collective. Unknown statements, labelled exits, address
escapes, unresolved calls and recursion fail closed to varying.
"""

from dataclasses import dataclass
from typing import Any, Callable

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ConstructorNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    ReturnNode,
    SwitchNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)

VARYING = 0
SYMMETRIC = 1
UNIFORM = 2

_PURE_UNARY = frozenset({"+", "-", "!", "~"})
_STEPS = frozenset(
    {"++", "--", "PRE_INCREMENT", "POST_INCREMENT", "PRE_DECREMENT", "POST_DECREMENT"}
)
_SHORT_CIRCUIT = frozenset({"&&", "||"})
# Scalar integer conversions a constant lane index may pass through, with the
# range each one keeps unchanged.
_INTEGER_RANGES = {
    **dict.fromkeys(("char", "int8_t"), (-(1 << 7), (1 << 7) - 1)),
    **dict.fromkeys(("uchar", "uint8_t"), (0, (1 << 8) - 1)),
    **dict.fromkeys(("short", "int16_t"), (-(1 << 15), (1 << 15) - 1)),
    **dict.fromkeys(("ushort", "uint16_t"), (0, (1 << 16) - 1)),
    **dict.fromkeys(("int", "int32_t"), (-(1 << 31), (1 << 31) - 1)),
    **dict.fromkeys(("uint", "uint32_t"), (0, (1 << 32) - 1)),
    **dict.fromkeys(("long", "int64_t"), (-(1 << 63), (1 << 63) - 1)),
    **dict.fromkeys(("ulong", "uint64_t"), (0, (1 << 64) - 1)),
}
_FOLDED_BINARY = {
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}


@dataclass(frozen=True)
class SymmetryRules:
    """Target facts the analysis needs; every callback fails closed.

    ``collective(node)`` names the subgroup operation ``node`` performs, or
    returns None. ``call_target(call)`` is the resolved user function, and
    ``pure_call(call)`` says whether a builtin's result depends only on its
    arguments. ``call_dependencies(call)`` names the argument positions a
    helper's result is proven to depend on, an item with a ``parameter``
    position for a read of that argument's storage, or None without a proof.
    ``storage_load(node)`` returns ``(resource, index)`` for a builtin element
    load. ``call_writes(call)`` names the locals a call may write through its
    arguments. ``seeded_parameter(parameter)`` says whether a callee parameter
    receives its argument's value (or binding), so that the argument's level
    carries over. ``volatile(declaration)`` marks a local whose reads need not
    return its last write, and ``storage_parameter(parameter)`` a parameter
    whose writes reach storage rather than a value the function owns.
    ``private_reference(parameter)`` marks an ``out``/``inout`` or private
    reference parameter, whose writes a caller receives in its argument.
    """

    walk: Callable[[Any], Any]
    map_operator: Callable[[Any], Any]
    collective: Callable[[Any], Any]
    call_target: Callable[[Any], Any]
    pure_call: Callable[[Any], bool]
    call_dependencies: Callable[[Any], Any]
    storage_load: Callable[[Any], Any]
    constant_int: Callable[[Any], Any]
    call_writes: Callable[[Any], Any]
    seeded_parameter: Callable[[Any], bool]
    volatile: Callable[[Any], bool]
    storage_parameter: Callable[[Any], bool]
    private_reference: Callable[[Any], bool]
    width: int
    full_subgroups: bool
    reductions: frozenset
    broadcasts: frozenset
    lane_preserving: frozenset


def _may_fall_through(node):
    """Whether control may reach the end of ``node`` (conservatively True)."""
    if node is None:
        return True
    if isinstance(node, BlockNode):
        node = node.statements
    if isinstance(node, (list, tuple)):
        return not node or _may_fall_through(node[-1])
    if isinstance(node, ReturnNode):
        return False
    if isinstance(node, IfNode):
        branches = [
            node.then_branch,
            *(getattr(node, "else_if_bodies", None) or []),
            node.else_branch,
        ]
        return node.else_branch is None or any(
            _may_fall_through(branch) for branch in branches
        )
    return True


class _Site:
    """Where a node runs: enclosing conditions, loops and breakable statements."""

    __slots__ = ("guards", "loops", "breakables", "index")

    def __init__(self, guards, loops, breakables, index):
        self.guards = guards
        self.loops = loops
        self.breakables = breakables
        self.index = index

    def guarded(self, condition):
        return _Site(
            self.guards + (condition,), self.loops, self.breakables, self.index
        )


class _Unsupported(Exception):
    pass


class FunctionSymmetry:
    """Levels of one function body under fixed parameter levels."""

    def __init__(self, analysis, function, seeds, symmetric_entry):
        self.analysis = analysis
        self.rules = analysis.rules
        self.function = function
        self.seeds = dict(seeds)
        self.symmetric_entry = bool(symmetric_entry)
        self.sites = {}
        self.writes = []
        self.exits = []
        self.returns = []
        self.declared = set()
        self.declared_types = {}
        # Writes to storage, or to names the function neither declares nor
        # receives by value or by private reference.
        self.effects = False
        self.collective_names = set()
        self.falls_through = _may_fall_through(getattr(function, "body", None))
        self.unsupported = False
        self.levels = {}
        self.exit_levels = {}
        self.cache = {}
        self.index = 0
        try:
            self._statement(getattr(function, "body", None), _Site((), (), (), 0))
        except _Unsupported:
            self.unsupported = True
        if not self.unsupported:
            self._solve()

    # Collection -------------------------------------------------------------

    def _next_site(self, site):
        self.index += 1
        return _Site(site.guards, site.loops, site.breakables, self.index)

    def _statement(self, node, site):
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                self._statement(child, site)
            return
        if isinstance(node, BlockNode):
            self._statement(node.statements, site)
            return
        site = self._next_site(site)
        self.sites[id(node)] = site
        if isinstance(node, VariableNode):
            self._declaration(node, site)
        elif isinstance(node, ExpressionStatementNode):
            self._expression(node.expression, site)
        elif isinstance(node, IfNode):
            self._expression(node.condition, site)
            conditions = [node.condition]
            self._statement(node.then_branch, site.guarded(node.condition))
            for condition, body in zip(
                getattr(node, "else_if_conditions", None) or [],
                getattr(node, "else_if_bodies", None) or [],
            ):
                branch = site
                for previous in conditions:
                    branch = branch.guarded(previous)
                self._expression(condition, branch)
                conditions.append(condition)
                self._statement(body, branch.guarded(condition))
            if node.else_branch is not None:
                branch = site
                for previous in conditions:
                    branch = branch.guarded(previous)
                self._statement(node.else_branch, branch)
        elif isinstance(node, ForNode):
            if node.init is not None:
                self._statement(node.init, site)
            loop = _Site(
                site.guards,
                site.loops + (id(node),),
                site.breakables + (id(node),),
                site.index,
            )
            self._expression(node.condition, loop)
            body = loop.guarded(node.condition) if node.condition is not None else loop
            self._statement(node.body, body)
            if node.update is not None:
                self._expression(node.update, body)
        elif isinstance(node, (WhileNode, DoWhileNode)):
            loop = _Site(
                site.guards,
                site.loops + (id(node),),
                site.breakables + (id(node),),
                site.index,
            )
            self._expression(node.condition, loop)
            self._statement(node.body, loop.guarded(node.condition))
        elif isinstance(node, SwitchNode):
            self._expression(node.expression, site)
            inner = _Site(
                site.guards + (node.expression,),
                site.loops,
                site.breakables + (id(node),),
                site.index,
            )
            for case in [*(node.cases or []), node.default_case]:
                if case is None:
                    continue
                self._statement(getattr(case, "statements", case), inner)
        elif isinstance(node, ReturnNode):
            self._expression(node.value, site)
            self.exits.append((node, "return", site, None))
            self.returns.append(node)
        elif isinstance(node, (BreakNode, ContinueNode)):
            if getattr(node, "label", None) is not None:
                raise _Unsupported
            targets = site.loops if isinstance(node, ContinueNode) else site.breakables
            if not targets:
                raise _Unsupported
            self.exits.append((node, "leave", site, targets[-1]))
        elif isinstance(
            node,
            (
                AssignmentNode,
                FunctionCallNode,
                UnaryOpNode,
                BinaryOpNode,
                TernaryOpNode,
            ),
        ):
            self._expression(node, site)
        else:
            raise _Unsupported

    def _declaration(self, node, site):
        name = getattr(node, "name", None)
        if not isinstance(name, str):
            raise _Unsupported
        self.declared.add(name)
        self.declared_types[name] = getattr(node, "var_type", None)
        value = getattr(node, "initial_value", None)
        self._expression(value, site)
        if self.rules.volatile(node):
            self.writes.append((name, "varying", (), site))
            return
        if isinstance(getattr(node, "var_type", None), (PointerType, ReferenceType)):
            # An alias may write its referent, and its own value is an address.
            self.writes.append((name, "varying", (), site))
            for alias in self._identifier_names(value):
                self.writes.append((alias, "varying", (), site))
            return
        if value is None:
            # Reading an indeterminate value is undefined in the source, so
            # the local holds only what its later writes store.
            return
        self.writes.append((name, "set", (value,), site))

    def _expression(self, node, site):
        if not isinstance(node, ASTNode):
            return
        self.sites.setdefault(id(node), site)
        if isinstance(node, TernaryOpNode):
            self._expression(node.condition, site)
            self._expression(node.true_expr, site.guarded(node.condition))
            self._expression(node.false_expr, site.guarded(node.condition))
            return
        if (
            isinstance(node, BinaryOpNode)
            and self.rules.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            )
            in _SHORT_CIRCUIT
        ):
            self._expression(node.left, site)
            self._expression(node.right, site.guarded(node.left))
            return
        if isinstance(node, AssignmentNode):
            self._assignment(node, site)
        elif isinstance(node, UnaryOpNode):
            operator = self.rules.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            )
            if operator in _STEPS:
                root = self._root_name(node.operand)
                if root is None:
                    raise _Unsupported
                self.writes.append((root, "update", (), site))
            elif operator == "&":
                for name in self._identifier_names(node.operand):
                    self.writes.append((name, "varying", (), site))
        elif isinstance(node, FunctionCallNode):
            received = self._out_writes(node, site)
            for name in self.rules.call_writes(node) or ():
                if name not in received:
                    self.writes.append((name, "varying", (), site))
        for child in node.child_nodes():
            self._expression(child, site)

    def _out_writes(self, call, site):
        """Record the locals a helper writes through reference parameters.

        A local passed alone, as the whole argument, to a private reference
        receives the helper's writes to that parameter; any other mention of
        it among the arguments keeps the call's conservative writes.
        """
        target = self.rules.call_target(call)
        arguments = list(call.arguments or [])
        parameters = list(getattr(target, "parameters", None) or [])
        if target is None or len(parameters) != len(arguments):
            return set()
        mentions = {}
        for argument in arguments:
            for name in self._identifier_names(argument):
                mentions[name] = mentions.get(name, 0) + 1
        received = set()
        for index, (parameter, argument) in enumerate(zip(parameters, arguments)):
            if (
                self.rules.private_reference(parameter)
                and isinstance(argument, IdentifierNode)
                and mentions.get(argument.name) == 1
            ):
                self.writes.append((argument.name, "out", (call, index), site))
                received.add(argument.name)
        return received

    def _assignment(self, node, site):
        target = node.target
        operator = self.rules.map_operator(
            getattr(node, "operator", getattr(node, "op", "="))
        )
        root = self._root_name(target)
        if root is None:
            raise _Unsupported
        indices = []
        cursor = target
        while not isinstance(cursor, IdentifierNode):
            if isinstance(cursor, ArrayAccessNode):
                indices.append(cursor.index_expr)
                cursor = cursor.array_expr
            elif isinstance(cursor, MemberAccessNode):
                cursor = cursor.object_expr
            else:
                cursor = cursor.vector_expr
        if isinstance(target, IdentifierNode) and operator == "=":
            self.writes.append((root, "set", (node.value,), site))
        else:
            self.writes.append((root, "update", (node.value, *indices), site))

    def _root_name(self, node):
        while isinstance(node, (ArrayAccessNode, MemberAccessNode, SwizzleNode)):
            if isinstance(node, ArrayAccessNode):
                node = node.array_expr
            elif isinstance(node, MemberAccessNode):
                node = node.object_expr
            else:
                node = node.vector_expr
        return node.name if isinstance(node, IdentifierNode) else None

    def _identifier_names(self, node):
        if node is None:
            return set()
        return {
            child.name
            for child in self.rules.walk(node)
            if isinstance(child, IdentifierNode)
        }

    # Solution ---------------------------------------------------------------

    def _affects(self, exit_record, site):
        node, kind, exit_site, target = exit_record
        if kind == "return":
            return site.index > exit_site.index or bool(
                set(site.loops) & set(exit_site.loops)
            )
        return target in site.breakables

    def _guard_level(self, site):
        return min((self._level(guard) for guard in site.guards), default=UNIFORM)

    def _site_level(self, site):
        level = self._guard_level(site)
        for record in self.exits:
            if level == VARYING:
                break
            if self._affects(record, site):
                level = min(level, self.exit_levels[id(record[0])])
        return level

    def _solve_exits(self):
        self.exit_levels = {id(record[0]): UNIFORM for record in self.exits}
        changed = True
        while changed:
            changed = False
            for record in self.exits:
                node, _kind, site, _target = record
                level = self._guard_level(site)
                for other in self.exits:
                    if other is record or level == VARYING:
                        continue
                    if self._affects(other, site):
                        level = min(level, self.exit_levels[id(other[0])])
                if level < self.exit_levels[id(node)]:
                    self.exit_levels[id(node)] = level
                    changed = True

    def _owned_names(self):
        owned = set(self.declared)
        for parameter in getattr(self.function, "parameters", None) or []:
            name = getattr(parameter, "name", None)
            if isinstance(name, str) and not self.rules.storage_parameter(parameter):
                owned.add(name)
        return owned

    def _solve(self):
        written = {name for name, *_rest in self.writes}
        self.effects = bool(written - self._owned_names())
        self.levels = {name: UNIFORM for name in self.declared}
        for name, level in self.seeds.items():
            self.levels[name] = min(level, self.levels.get(name, UNIFORM))
        for name in written:
            self.levels.setdefault(name, VARYING)
        while True:
            self.cache = {}
            self._solve_exits()
            changed = False
            for name, mode, values, site in self.writes:
                current = self.levels[name]
                if current == VARYING:
                    continue
                if mode == "varying":
                    level = VARYING
                elif mode == "out":
                    level = min(self._site_level(site), self._out_level(*values))
                else:
                    level = min(
                        [self._site_level(site), *(self._level(v) for v in values)]
                    )
                    if mode == "update":
                        level = min(level, current)
                if level < current:
                    self.levels[name] = level
                    changed = True
            if not changed:
                break
        self.cache = {}
        changed = True
        while changed:
            changed = False
            for name, _mode, values, _site in self.writes:
                if name not in self.collective_names and any(
                    self.collective_derived(value)
                    for value in values
                    if isinstance(value, ASTNode)
                ):
                    self.collective_names.add(name)
                    changed = True

    def _level(self, node):
        if node is None or isinstance(node, (bool, int, float)):
            return UNIFORM
        key = id(node)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.cache[key] = VARYING
        level = self._compute(node)
        self.cache[key] = level
        return level

    def _compute(self, node):
        if isinstance(node, LiteralNode):
            return UNIFORM
        if isinstance(node, IdentifierNode):
            return self.levels.get(node.name, VARYING)
        operation = self.rules.collective(node)
        if operation is not None:
            return self._collective(node, operation)
        load = self.rules.storage_load(node)
        if load is not None:
            return min(self._level(load[0]), self._level(load[1]))
        if isinstance(node, ArrayAccessNode):
            return min(self._level(node.array_expr), self._level(node.index_expr))
        if isinstance(node, MemberAccessNode):
            return self._level(node.object_expr)
        if isinstance(node, SwizzleNode):
            return self._level(node.vector_expr)
        if isinstance(node, CastNode):
            return self._level(node.expression)
        if isinstance(node, ConstructorNode):
            return min(
                (
                    self._level(child)
                    for child in [
                        *(node.arguments or []),
                        *(node.named_arguments or {}).values(),
                    ]
                ),
                default=UNIFORM,
            )
        if isinstance(node, BinaryOpNode):
            return min(self._level(node.left), self._level(node.right))
        if isinstance(node, UnaryOpNode):
            operator = self.rules.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            )
            if operator not in _PURE_UNARY:
                return VARYING
            return self._level(node.operand)
        if isinstance(node, TernaryOpNode):
            return min(
                self._level(node.condition),
                self._level(node.true_expr),
                self._level(node.false_expr),
            )
        if isinstance(node, FunctionCallNode):
            return self._call(node)
        return VARYING

    def _collective(self, node, operation):
        site = self.sites.get(id(node))
        if (
            not self.rules.full_subgroups
            or not self.symmetric_entry
            or site is None
            or self._site_level(site) < SYMMETRIC
        ):
            return VARYING
        arguments = list(getattr(node, "arguments", None) or [])
        if not arguments:
            return VARYING
        levels = [self._level(argument) for argument in arguments]
        if min(levels) < SYMMETRIC:
            return VARYING
        if operation in self.rules.reductions:
            return UNIFORM
        if operation in self.rules.broadcasts and len(arguments) == 2:
            lane = self._constant_lane(arguments[1])
            if lane is not None and 0 <= lane < self.rules.width:
                return UNIFORM
            if levels[1] == UNIFORM and self._guarded_lane(arguments[1], site):
                return UNIFORM
            return SYMMETRIC
        if operation in self.rules.lane_preserving:
            return SYMMETRIC
        return VARYING

    def _guarded_lane(self, node, site):
        """Whether a uniform lane index is in range wherever ``site`` runs.

        The index must be an unsigned local written once, at its declaration,
        that an enclosing condition bounds below the subgroup width; integer
        conversions and masks that keep values below the width may wrap it.
        """
        width = self.rules.width
        name = self._lane_name(node)
        if name is None or name not in self.declared:
            return False
        type_name = str(
            getattr(
                self.declared_types.get(name), "name", self.declared_types.get(name)
            )
        )
        if not type_name.startswith("u") and type_name != "size_t":
            return False
        if [write for write in self.writes if write[0] == name] != [
            write for write in self.writes if write[0] == name and write[1] == "set"
        ] or sum(write[0] == name for write in self.writes) != 1:
            return False
        for guard in site.guards:
            if not isinstance(guard, BinaryOpNode):
                continue
            operator = self.rules.map_operator(
                getattr(guard, "op", getattr(guard, "operator", None))
            )
            left, right = guard.left, guard.right
            if operator in {">", ">="}:
                left, right = right, left
                operator = "<" if operator == ">" else "<="
            if operator not in {"<", "<="} or self._lane_name(left) != name:
                continue
            bound = self._constant_lane(right)
            if bound is not None and bound - (operator == "<") < width:
                return True
        return False

    def _lane_name(self, node):
        """The local a lane expression reads through value-keeping wrappers."""
        mask = self.rules.width - 1
        while True:
            if isinstance(node, IdentifierNode):
                return node.name
            if isinstance(node, CastNode):
                node = node.expression
                continue
            if (
                isinstance(node, FunctionCallNode)
                and str(getattr(node.name, "name", node.name)) in _INTEGER_RANGES
                and len(node.arguments or []) == 1
            ):
                node = node.arguments[0]
                continue
            if (
                isinstance(node, BinaryOpNode)
                and self.rules.map_operator(
                    getattr(node, "op", getattr(node, "operator", None))
                )
                == "&"
            ):
                value = self._constant_lane(node.right)
                if value is not None and value & mask == mask:
                    node = node.left
                    continue
            return None

    def _out_level(self, call, index):
        """The level of what a helper writes to its reference parameter ``index``."""
        arguments = list(call.arguments or [])
        if not all(self._effect_free(argument) for argument in arguments):
            return VARYING
        target = self.rules.call_target(call)
        parameters = list(getattr(target, "parameters", None) or [])
        seeds = {
            parameter.name: self._level(argument)
            for parameter, argument in zip(parameters, arguments)
            if self.rules.seeded_parameter(parameter)
            or self.rules.private_reference(parameter)
        }
        site = self.sites.get(id(call))
        entered = (
            self.symmetric_entry
            and site is not None
            and self._site_level(site) >= SYMMETRIC
        )
        return self.analysis.out_level(target, seeds, entered, parameters[index].name)

    def _effect_free(self, node):
        """Whether evaluating ``node`` writes nothing and calls only known helpers."""
        for child in self.rules.walk(node):
            if isinstance(child, AssignmentNode):
                return False
            if isinstance(child, UnaryOpNode) and self.rules.map_operator(
                getattr(child, "op", getattr(child, "operator", None))
            ) in _STEPS | {"&"}:
                return False
            if (
                isinstance(child, FunctionCallNode)
                and self.rules.collective(child) is None
                and not self.rules.pure_call(child)
                and self.rules.call_dependencies(child) is None
            ):
                return False
        return True

    def _constant_lane(self, node):
        """A lane index folded through value-preserving integer conversions.

        Conversions apply only to values they keep unchanged, so a folded
        index is the one every invocation passes; anything else is None.
        """
        value = self.rules.constant_int(node)
        if value is not None:
            return value
        if isinstance(node, (CastNode, FunctionCallNode)):
            if isinstance(node, CastNode):
                type_name, operands = node.target_type, [node.expression]
            else:
                type_name, operands = node.name, list(node.arguments or [])
            bounds = _INTEGER_RANGES.get(str(getattr(type_name, "name", type_name)))
            if bounds is None or len(operands) != 1:
                return None
            value = self._constant_lane(operands[0])
            if value is None or not bounds[0] <= value <= bounds[1]:
                return None
            return value
        if isinstance(node, BinaryOpNode):
            fold = _FOLDED_BINARY.get(
                self.rules.map_operator(
                    getattr(node, "op", getattr(node, "operator", None))
                )
            )
            left = self._constant_lane(node.left)
            right = self._constant_lane(node.right)
            if fold is None or left is None or right is None:
                return None
            return fold(left, right)
        return None

    def _call(self, call):
        arguments = list(call.arguments or [])
        if self.rules.pure_call(call):
            return min(
                (self._level(argument) for argument in arguments), default=UNIFORM
            )
        if not all(self._effect_free(argument) for argument in arguments):
            # An argument's own writes or unknown calls run at every evaluation.
            return VARYING
        dependencies = self.rules.call_dependencies(call)
        if dependencies is not None:
            # The result is a function of these arguments alone, a storage
            # read included through its resource argument's level.
            level = UNIFORM
            for dependency in dependencies:
                index = (
                    dependency
                    if isinstance(dependency, int)
                    else getattr(dependency, "parameter", None)
                )
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not 0 <= index < len(arguments)
                ):
                    return VARYING
                level = min(level, self._level(arguments[index]))
            return level
        target = self.rules.call_target(call)
        parameters = list(getattr(target, "parameters", None) or [])
        if target is None or len(parameters) != len(arguments):
            return VARYING
        seeds = {
            parameter.name: self._level(argument)
            for parameter, argument in zip(parameters, arguments)
            if self.rules.seeded_parameter(parameter)
        }
        site = self.sites.get(id(call))
        entered = (
            self.symmetric_entry
            and site is not None
            and self._site_level(site) >= SYMMETRIC
        )
        return self.analysis.return_level(target, seeds, entered)

    # Queries ----------------------------------------------------------------

    def level(self, node):
        """The level of ``node``; nodes outside this body are evaluated as written."""
        if self.unsupported:
            return VARYING
        return self._level(node)

    def name_level(self, name):
        if self.unsupported:
            return VARYING
        return self.levels.get(name, VARYING)

    def reach(self, node):
        """The level of the control under which ``node`` runs."""
        site = self.sites.get(id(node))
        if self.unsupported or site is None:
            return VARYING
        return self._site_level(site)

    def return_level(self):
        # A summary stands for a read-only value helper: one that writes no
        # storage and returns a value on every path.
        if self.unsupported or not self.returns or self.effects or self.falls_through:
            return VARYING
        return min(
            min(self._level(node.value), self.reach(node)) for node in self.returns
        )

    def collective_derived(self, node):
        """Whether ``node`` reads a subgroup collective's result.

        That is a collective in ``node`` itself, a local written from one, or
        a call to a helper that performs one.
        """
        for child in self.rules.walk(node) if node is not None else ():
            if self.rules.collective(child) is not None:
                return True
            if (
                isinstance(child, IdentifierNode)
                and child.name in self.collective_names
            ):
                return True
            if isinstance(
                child, FunctionCallNode
            ) and self.analysis.performs_collective(self.rules.call_target(child)):
                return True
        return False

    def entered_symmetrically(self, call):
        """Whether every subgroup reaches ``call`` with the same lanes."""
        return self.symmetric_entry and self.reach(call) >= SYMMETRIC


class LaneSymmetryAnalysis:
    """Memoized per-function facts and return summaries for one program."""

    def __init__(self, rules):
        self.rules = rules
        self.functions = {}
        self.visiting = set()
        self.collectives = {}

    def analyze(self, function, seeds, symmetric_entry):
        key = (
            id(function),
            tuple(sorted(seeds.items())),
            bool(symmetric_entry),
        )
        facts = self.functions.get(key)
        if facts is None:
            facts = FunctionSymmetry(self, function, seeds, symmetric_entry)
            self.functions[key] = facts
        return facts

    def performs_collective(self, function, _active=None):
        """Whether ``function``, or a helper it calls, performs a collective."""
        if function is None:
            return False
        key = id(function)
        if key in self.collectives:
            return self.collectives[key]
        active = set() if _active is None else _active
        if key in active:
            return False
        active.add(key)
        found = False
        for node in self.rules.walk(getattr(function, "body", None) or []):
            if self.rules.collective(node) is not None or (
                isinstance(node, FunctionCallNode)
                and self.performs_collective(self.rules.call_target(node), active)
            ):
                found = True
                break
        active.discard(key)
        self.collectives[key] = found
        return found

    def out_level(self, function, seeds, symmetric_entry, name):
        if id(function) in self.visiting:
            return VARYING
        self.visiting.add(id(function))
        try:
            facts = self.analyze(function, seeds, symmetric_entry)
            if facts.unsupported or facts.effects:
                return VARYING
            return facts.levels.get(name, VARYING)
        finally:
            self.visiting.discard(id(function))

    def return_level(self, function, seeds, symmetric_entry):
        if id(function) in self.visiting:
            return VARYING
        self.visiting.add(id(function))
        try:
            return self.analyze(function, seeds, symmetric_entry).return_level()
        finally:
            self.visiting.discard(id(function))
